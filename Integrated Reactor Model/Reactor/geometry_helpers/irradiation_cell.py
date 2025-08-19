import openmc
import os
import sys
import numpy as np
from .utils import generate_cell_id, get_irradiation_cell_name

# Add root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

# Add Reactor directory to path for materials
reactor_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(reactor_dir)

from utils.base_inputs import inputs

def parse_irradiation_type(lattice_position, inputs_dict):
    """Parse irradiation position type from lattice string.

    Parameters
    ----------
    lattice_position : str
        Position string from core lattice (e.g., 'I_1', 'I_2P', 'I_3B', 'I_4G')
    inputs_dict : dict
        Inputs dictionary containing irradiation_fill

    Returns
    -------
    str
        Irradiation type: 'PWR_loop', 'BWR_loop', 'Gas_capsule', 'vacuum', or 'fill'
    """
    if not lattice_position.startswith('I_'):
        raise ValueError(f"Invalid irradiation position: {lattice_position}")

    # Check for suffix
    if lattice_position.endswith('P'):
        return 'PWR_loop'
    elif lattice_position.endswith('B'):
        return 'BWR_loop'
    elif lattice_position.endswith('G'):
        return 'Gas_capsule'
    else:
        # No suffix, use exact irradiation_fill material name
        irradiation_fill = inputs_dict.get('irradiation_fill', 'Vacuum')
        return irradiation_fill

def generate_complex_cell_id(position, component_type, component_index=0, irradiation_type='HTWL'):
    """Generate cell ID for complex irradiation geometries.

    Base: 6XXYYZZ
    - 6 prefix for irradiation
    - XX = i position (00-99)
    - YY = j position (00-99)
    - ZZ = component code (00-99)

    For different irradiation types, we add an offset to avoid conflicts:
    - PWR_loop: base 6000000
    - BWR_loop: base 6200000
    - SIGMA: base 6100000
    """
    i, j = position
    if irradiation_type == 'SIGMA':
        type_offset = 100000
    elif irradiation_type == 'BWR_loop':
        type_offset = 200000
    else:  # PWR_loop or default
        type_offset = 0
    base = 6000000 + type_offset + i * 10000 + j * 100

    component_codes = {
        # Core components
        'spine': 0,
        'spine_below': 50,
        'spine_above': 51,
        # HTWL samples (1-4 at clock positions)
        'sample_1_ti': 1, 'sample_1_graphite': 2, 'sample_1_sic': 3,
        'sample_2_ti': 4, 'sample_2_graphite': 5, 'sample_2_sic': 6,
        'sample_3_ti': 7, 'sample_3_graphite': 8, 'sample_3_sic': 9,
        'sample_4_ti': 10, 'sample_4_graphite': 11, 'sample_4_sic': 12,
        # SIGMA layers
        'inner_he': 13, 'inner_graphite': 14, 'tungsten': 15,
        'outer_graphite': 16, 'outer_he': 17,
        # Structural
        'capsule_wall': 18,
        'autoclave_wall': 19,
        'co2_gap': 20,
        'thimble': 21,
        'thimble_below': 52,
        'thimble_auto_bot': 53,
        'thimble_above': 54,
        'co2_below': 55,
        'co2_annular': 56,
        'capsule_bottom_plate': 22,
        'capsule_top_plate': 23,
        'autoclave_bottom': 24,
        # Water regions
        'water_below': 30,
        'water_above': 31,
        'water_capsule': 32,
        'water_around': 33,
        'water_gap': 34,
        # Outer fill
        'outer_fill': 40
    }

    code = component_codes.get(component_type, 99)
    return base + code + component_index

def build_complex_htwl(mat_dict, position, inputs_dict, use_bwr_water=False, irradiation_type='PWR_loop'):
    """Build detailed HTWL geometry scaled to fit cell.

    Parameters
    ----------
    mat_dict : dict
        Dictionary of materials
    position : tuple
        (i, j) position in core lattice
    inputs_dict : dict
        Inputs dictionary
    use_bwr_water : bool
        If True, use BWR_fluid instead of HP_Borated_Water

    Returns
    -------
    openmc.Universe
        Universe containing the HTWL experiment
    """
    # Get cell dimensions
    if inputs_dict['assembly_type'] == 'Pin':
        cell_width = inputs_dict['pin_pitch'] * inputs_dict['n_side_pins'] * 100  # m to cm
    else:  # Plate assembly
        cell_width = (2 * inputs_dict['clad_structure_width'] + inputs_dict['fuel_plate_width']) * 100  # m to cm

    # Get scaling from inputs - use PWR or BWR diameter depending on water type
    if use_bwr_water:
        diameter_fraction = inputs_dict['BWR_loop_diameter']  # Get from inputs
    else:
        diameter_fraction = inputs_dict['PWR_loop_diameter']  # Get from inputs

    target_diameter = cell_width * diameter_fraction
    mcnp_outer_radius = 2.585  # Water gap outer radius in MCNP
    scale_factor = target_diameter / (2 * mcnp_outer_radius)

    # Scale all MCNP radii
    r_spine = 0.24 * scale_factor
    r_sample_ti = 0.1 * scale_factor
    r_sample_graphite = 0.3175 * scale_factor
    r_sample_sic = 0.45 * scale_factor
    r_sample_center = 1.03 * scale_factor  # Distance from center to sample centers
    r_capsule_inner = 1.698 * scale_factor
    r_capsule_outer = 1.778 * scale_factor
    r_autoclave_inner = 1.93675 * scale_factor
    r_autoclave_outer = 2.15265 * scale_factor
    # Thimble boundaries (CO2 goes up to thimble inner - they share the same surface)
    r_thimble_inner = 2.2987 * scale_factor  # This is ALSO the CO2 outer boundary
    r_thimble_outer = 2.54 * scale_factor
    r_water_gap_outer = 2.585 * scale_factor  # Water gap outer radius (from MCNP surface 107)

    # Z-planes (keep original dimensions, don't scale)
    fuel_height = inputs_dict['fuel_height'] * 100  # m to cm
    z_bottom = -fuel_height / 2
    z_top = fuel_height / 2

    # Fixed experimental positions (relative to center)
    z_autoclave_bottom_bot = -28.305
    z_autoclave_bottom_top = -27.305
    z_capsule_bottom_bot = -24.0
    z_capsule_bottom_top = -23.5
    z_capsule_top_bot = -0.5
    z_capsule_top_top = 0.0

    # Select water material
    water_mat = mat_dict['BWR_fluid'] if use_bwr_water else mat_dict['HP_Borated_Water']

    # Get coolant material for water gap (using same pattern as plate_fuel.py)
    coolant_type = inputs_dict.get('coolant_type', 'Light Water')
    coolant_mat = mat_dict[f'{coolant_type} Coolant']

    cells = []

    # Create surfaces
    # Radial surfaces (cylinders)
    cyl_spine = openmc.ZCylinder(r=r_spine)
    cyl_capsule_inner = openmc.ZCylinder(r=r_capsule_inner)
    cyl_capsule_outer = openmc.ZCylinder(r=r_capsule_outer)
    cyl_autoclave_inner = openmc.ZCylinder(r=r_autoclave_inner)
    cyl_autoclave_outer = openmc.ZCylinder(r=r_autoclave_outer)
    cyl_thimble_inner = openmc.ZCylinder(r=r_thimble_inner)  # This is ALSO the CO2 outer boundary
    cyl_thimble_outer = openmc.ZCylinder(r=r_thimble_outer)
    cyl_water_gap_outer = openmc.ZCylinder(r=r_water_gap_outer)

    # Sample cylinders (off-axis at 12, 3, 6, 9 o'clock)
    sample_positions = [
        (0, r_sample_center),      # Sample 1 at 12:00
        (r_sample_center, 0),      # Sample 2 at 3:00
        (0, -r_sample_center),     # Sample 3 at 6:00
        (-r_sample_center, 0),     # Sample 4 at 9:00
    ]

    sample_cylinders = []
    for x, y in sample_positions:
        sample_cylinders.append({
            'ti': openmc.ZCylinder(x0=x, y0=y, r=r_sample_ti),
            'graphite': openmc.ZCylinder(x0=x, y0=y, r=r_sample_graphite),
            'sic': openmc.ZCylinder(x0=x, y0=y, r=r_sample_sic),
        })

    # Z-planes
    z_bot = openmc.ZPlane(z_bottom)
    z_top = openmc.ZPlane(z_top)
    z_auto_bot_bot = openmc.ZPlane(z_autoclave_bottom_bot)
    z_auto_bot_top = openmc.ZPlane(z_autoclave_bottom_top)
    z_cap_bot_bot = openmc.ZPlane(z_capsule_bottom_bot)
    z_cap_bot_top = openmc.ZPlane(z_capsule_bottom_top)
    z_cap_top_bot = openmc.ZPlane(z_capsule_top_bot)
    z_cap_top_top = openmc.ZPlane(z_capsule_top_top)

    # Square boundaries for outer fill
    x_min = openmc.XPlane(-cell_width/2)
    x_max = openmc.XPlane(cell_width/2)
    y_min = openmc.YPlane(-cell_width/2)
    y_max = openmc.YPlane(cell_width/2)

    # Build cells from inside out, considering axial zones

    # ========== SPINE REGIONS ==========
    # MCNP cell 16801: -16908 16910 → spine starts at z=-27.305, not from bottom
    # Spine below samples (z: -27.305 to -23.5)
    spine_below_region = -cyl_spine & +z_auto_bot_top & -z_cap_bot_top
    spine_below_cell = openmc.Cell(name='spine_below')
    spine_below_cell.id = generate_complex_cell_id(position, 'spine_below', irradiation_type=irradiation_type)
    spine_below_cell.region = spine_below_region
    spine_below_cell.fill = mat_dict['Titanium']
    cells.append(spine_below_cell)

    # Spine in sample region (z: -23.5 to -0.5)
    spine_region = -cyl_spine & +z_cap_bot_top & -z_cap_top_bot
    spine_cell = openmc.Cell(name='spine')
    spine_cell.id = generate_complex_cell_id(position, 'spine', irradiation_type=irradiation_type)
    spine_cell.region = spine_region
    spine_cell.fill = mat_dict['Titanium']
    cells.append(spine_cell)

    # Spine above samples (z: -0.5 to z_top) - VARIABLE HEIGHT
    spine_above_region = -cyl_spine & +z_cap_top_bot & -z_top
    spine_above_cell = openmc.Cell(name='spine_above')
    spine_above_cell.id = generate_complex_cell_id(position, 'spine_above', irradiation_type=irradiation_type)
    spine_above_cell.region = spine_above_region
    spine_above_cell.fill = mat_dict['Titanium']
    cells.append(spine_above_cell)

    # ========== SAMPLE REGION (z: -23.5 to -0.5) ==========

    # Four samples
    for i, (x, y) in enumerate(sample_positions, 1):
        cyls = sample_cylinders[i-1]

        # Titanium core
        ti_region = -cyls['ti'] & +z_cap_bot_top & -z_cap_top_bot
        ti_cell = openmc.Cell(name=f'sample_{i}_ti')
        ti_cell.id = generate_complex_cell_id(position, f'sample_{i}_ti', irradiation_type=irradiation_type)
        ti_cell.region = ti_region
        ti_cell.fill = mat_dict['Titanium']
        cells.append(ti_cell)

        # Graphite holder
        graphite_region = +cyls['ti'] & -cyls['graphite'] & +z_cap_bot_top & -z_cap_top_bot
        graphite_cell = openmc.Cell(name=f'sample_{i}_graphite')
        graphite_cell.id = generate_complex_cell_id(position, f'sample_{i}_graphite', irradiation_type=irradiation_type)
        graphite_cell.region = graphite_region
        graphite_cell.fill = mat_dict['graphite']
        cells.append(graphite_cell)

        # SiC cladding
        sic_region = +cyls['graphite'] & -cyls['sic'] & +z_cap_bot_top & -z_cap_top_bot
        sic_cell = openmc.Cell(name=f'sample_{i}_sic')
        sic_cell.id = generate_complex_cell_id(position, f'sample_{i}_sic', irradiation_type=irradiation_type)
        sic_cell.region = sic_region
        sic_cell.fill = mat_dict['SiC']
        cells.append(sic_cell)

    # Water inside capsule (around samples and spine)
    # This is the complex region: inside capsule, outside spine, outside all 4 samples
    water_capsule_region = -cyl_capsule_inner & +cyl_spine & +z_cap_bot_top & -z_cap_top_bot
    for cyls in sample_cylinders:
        water_capsule_region = water_capsule_region & +cyls['sic']

    water_capsule_cell = openmc.Cell(name='water_capsule')
    water_capsule_cell.id = generate_complex_cell_id(position, 'water_capsule', irradiation_type=irradiation_type)
    water_capsule_cell.region = water_capsule_region
    water_capsule_cell.fill = water_mat
    cells.append(water_capsule_cell)

    # ========== CAPSULE PLATES ==========
    # Capsule bottom plate
    cap_bot_region = -cyl_capsule_outer & +z_cap_bot_bot & -z_cap_bot_top
    cap_bot_cell = openmc.Cell(name='capsule_bottom_plate')
    cap_bot_cell.id = generate_complex_cell_id(position, 'capsule_bottom_plate', irradiation_type=irradiation_type)
    cap_bot_cell.region = cap_bot_region
    cap_bot_cell.fill = mat_dict['Titanium']
    cells.append(cap_bot_cell)

    # Capsule top plate
    cap_top_region = -cyl_capsule_outer & +z_cap_top_bot & -z_cap_top_top
    cap_top_cell = openmc.Cell(name='capsule_top_plate')
    cap_top_cell.id = generate_complex_cell_id(position, 'capsule_top_plate', irradiation_type=irradiation_type)
    cap_top_cell.region = cap_top_region
    cap_top_cell.fill = mat_dict['Titanium']
    cells.append(cap_top_cell)

    # ========== CAPSULE WALLS ==========
    # Capsule cylindrical walls (only in sample region)
    cap_wall_region = +cyl_capsule_inner & -cyl_capsule_outer & +z_cap_bot_top & -z_cap_top_bot
    cap_wall_cell = openmc.Cell(name='capsule_wall')
    cap_wall_cell.id = generate_complex_cell_id(position, 'capsule_wall', irradiation_type=irradiation_type)
    cap_wall_cell.region = cap_wall_region
    cap_wall_cell.fill = mat_dict['Titanium']
    cells.append(cap_wall_cell)

    # ========== WATER REGIONS ==========
    # Water below capsule (between capsule bottom and autoclave bottom) - EXCLUDE SPINE
    water_below_region = -cyl_autoclave_inner & +cyl_spine & +z_auto_bot_top & -z_cap_bot_bot
    water_below_cell = openmc.Cell(name='water_below')
    water_below_cell.id = generate_complex_cell_id(position, 'water_below', irradiation_type=irradiation_type)
    water_below_cell.region = water_below_region
    water_below_cell.fill = water_mat
    cells.append(water_below_cell)

    # Water above capsule (from capsule top to top of fuel) - VARIABLE HEIGHT - EXCLUDE SPINE
    water_above_region = -cyl_autoclave_inner & +cyl_spine & +z_cap_top_top & -z_top
    water_above_cell = openmc.Cell(name='water_above')
    water_above_cell.id = generate_complex_cell_id(position, 'water_above', irradiation_type=irradiation_type)
    water_above_cell.region = water_above_region
    water_above_cell.fill = water_mat
    cells.append(water_above_cell)

    # Water around capsule (between capsule and autoclave in sample region + plates)
    water_around_region = +cyl_capsule_outer & -cyl_autoclave_inner & +z_cap_bot_bot & -z_cap_top_top
    water_around_cell = openmc.Cell(name='water_around')
    water_around_cell.id = generate_complex_cell_id(position, 'water_around', irradiation_type=irradiation_type)
    water_around_cell.region = water_around_region
    water_around_cell.fill = water_mat
    cells.append(water_around_cell)

    # ========== AUTOCLAVE ==========
    # Autoclave bottom plate (SOLID DISK from center to outer radius)
    # MCNP cell 16821: -16902 means r < 2.15265 (solid disk)
    auto_bot_region = -cyl_autoclave_outer & +z_auto_bot_bot & -z_auto_bot_top
    auto_bot_cell = openmc.Cell(name='autoclave_bottom')
    auto_bot_cell.id = generate_complex_cell_id(position, 'autoclave_bottom', irradiation_type=irradiation_type)
    auto_bot_cell.region = auto_bot_region
    auto_bot_cell.fill = mat_dict['Titanium']
    cells.append(auto_bot_cell)

    # Autoclave cylindrical walls (full height except bottom plate)
    auto_wall_region = +cyl_autoclave_inner & -cyl_autoclave_outer & +z_auto_bot_top & -z_top
    auto_wall_cell = openmc.Cell(name='autoclave_wall')
    auto_wall_cell.id = generate_complex_cell_id(position, 'autoclave_wall', irradiation_type=irradiation_type)
    auto_wall_cell.region = auto_wall_region
    auto_wall_cell.fill = mat_dict['Titanium']
    cells.append(auto_wall_cell)

    # ========== CO2 GAP (SIMPLIFIED TO 2 REGIONS) ==========
    # 1. CO2 FULL CYLINDER below autoclave (z < -28.305)
    co2_below_region = -cyl_thimble_inner & +z_bot & -z_auto_bot_bot
    co2_below_cell = openmc.Cell(name='co2_below')
    co2_below_cell.id = generate_complex_cell_id(position, 'co2_below', irradiation_type=irradiation_type)
    co2_below_cell.region = co2_below_region
    co2_below_cell.fill = mat_dict['CO2']
    cells.append(co2_below_cell)

    # 2. CO2 ANNULAR SLEEVE above autoclave (z > -28.305)
    # Between autoclave outer and thimble inner for full height
    co2_annular_region = +cyl_autoclave_outer & -cyl_thimble_inner & +z_auto_bot_bot & -z_top
    co2_annular_cell = openmc.Cell(name='co2_annular')
    co2_annular_cell.id = generate_complex_cell_id(position, 'co2_annular', irradiation_type=irradiation_type)
    co2_annular_cell.region = co2_annular_region
    co2_annular_cell.fill = mat_dict['CO2']
    cells.append(co2_annular_cell)

    # ========== THIMBLE (AXIALLY DISCRETIZED TO AVOID OVERLAPS) ==========
    # Thimble below autoclave - matches CO2 below axial bounds
    thimble_below_region = +cyl_thimble_inner & -cyl_thimble_outer & +z_bot & -z_auto_bot_bot
    thimble_below_cell = openmc.Cell(name='thimble_below')
    thimble_below_cell.id = generate_complex_cell_id(position, 'thimble_below', irradiation_type=irradiation_type)
    thimble_below_cell.region = thimble_below_region
    thimble_below_cell.fill = mat_dict['Al6061']
    cells.append(thimble_below_cell)

    # Thimble in autoclave bottom plate region (z: -28.305 to -27.305)
    thimble_auto_bot_region = +cyl_thimble_inner & -cyl_thimble_outer & +z_auto_bot_bot & -z_auto_bot_top
    thimble_auto_bot_cell = openmc.Cell(name='thimble_auto_bot')
    thimble_auto_bot_cell.id = generate_complex_cell_id(position, 'thimble_auto_bot', irradiation_type=irradiation_type)
    thimble_auto_bot_cell.region = thimble_auto_bot_region
    thimble_auto_bot_cell.fill = mat_dict['Al6061']
    cells.append(thimble_auto_bot_cell)

    # Thimble above autoclave - matches CO2 around axial bounds
    thimble_above_region = +cyl_thimble_inner & -cyl_thimble_outer & +z_auto_bot_top & -z_top
    thimble_above_cell = openmc.Cell(name='thimble_above')
    thimble_above_cell.id = generate_complex_cell_id(position, 'thimble_above', irradiation_type=irradiation_type)
    thimble_above_cell.region = thimble_above_region
    thimble_above_cell.fill = mat_dict['Al6061']
    cells.append(thimble_above_cell)

    # ========== WATER GAP ==========
    # Water gap between thimble and outer Al (from MCNP surface 107)
    water_gap_region = +cyl_thimble_outer & -cyl_water_gap_outer & +z_bot & -z_top
    water_gap_cell = openmc.Cell(name='water_gap')
    water_gap_cell.id = generate_complex_cell_id(position, 'water_gap', irradiation_type=irradiation_type)
    water_gap_cell.region = water_gap_region
    water_gap_cell.fill = coolant_mat
    cells.append(water_gap_cell)

    # ========== OUTER FILL (Al6061) ==========
    # Fill from water gap to square boundary
    outer_fill_region = +cyl_water_gap_outer & +x_min & -x_max & +y_min & -y_max & +z_bot & -z_top
    outer_fill_cell = openmc.Cell(name='outer_fill')
    outer_fill_cell.id = generate_complex_cell_id(position, 'outer_fill', irradiation_type=irradiation_type)
    outer_fill_cell.region = outer_fill_region
    outer_fill_cell.fill = mat_dict['Al6061']
    cells.append(outer_fill_cell)

    # Create universe
    i, j = position
    universe_id = 6000000 + i * 1000 + j * 10
    universe = openmc.Universe(universe_id=universe_id, name=f'htwl_{i}_{j}', cells=cells)

    return universe

def build_complex_sigma(mat_dict, position, inputs_dict):
    """Build detailed SIGMA geometry scaled to fit cell.

    Parameters
    ----------
    mat_dict : dict
        Dictionary of materials
    position : tuple
        (i, j) position in core lattice
    inputs_dict : dict
        Inputs dictionary

    Returns
    -------
    openmc.Universe
        Universe containing the SIGMA experiment
    """
    # Get cell dimensions
    if inputs_dict['assembly_type'] == 'Pin':
        cell_width = inputs_dict['pin_pitch'] * inputs_dict['n_side_pins'] * 100  # m to cm
    else:  # Plate assembly
        cell_width = (2 * inputs_dict['clad_structure_width'] + inputs_dict['fuel_plate_width']) * 100  # m to cm

    # Get scaling from inputs - SIGMA uses Gas_capsule diameter
    diameter_fraction = inputs_dict['Gas_capsule_diameter']  # Get from inputs
    target_diameter = cell_width * diameter_fraction
    mcnp_outer_radius = 2.618  # Water gap outer radius in MCNP
    scale_factor = target_diameter / (2 * mcnp_outer_radius)

    # Scale all MCNP radii
    r_spine = 0.25 * scale_factor
    r_inner_he = 0.5 * scale_factor
    r_inner_graphite = 1.3 * scale_factor
    r_tungsten = 1.8 * scale_factor
    r_outer_graphite = 2.35 * scale_factor
    r_outer_he = 2.4511 * scale_factor
    r_thimble_inner = 2.4511 * scale_factor
    r_thimble_outer = 2.54 * scale_factor
    r_water_gap = 2.618 * scale_factor

    # Z boundaries (full fuel height)
    fuel_height = inputs_dict['fuel_height'] * 100  # m to cm
    z_bottom = -fuel_height / 2
    z_top = fuel_height / 2

    cells = []

    # Create surfaces
    # Radial surfaces (all concentric cylinders)
    cyl_spine = openmc.ZCylinder(r=r_spine)
    cyl_inner_he = openmc.ZCylinder(r=r_inner_he)
    cyl_inner_graphite = openmc.ZCylinder(r=r_inner_graphite)
    cyl_tungsten = openmc.ZCylinder(r=r_tungsten)
    cyl_outer_graphite = openmc.ZCylinder(r=r_outer_graphite)
    cyl_outer_he = openmc.ZCylinder(r=r_outer_he)
    cyl_thimble_outer = openmc.ZCylinder(r=r_thimble_outer)
    cyl_water_gap = openmc.ZCylinder(r=r_water_gap)

    # Z-planes
    z_bot = openmc.ZPlane(z_bottom)
    z_top = openmc.ZPlane(z_top)

    # Square boundaries for outer fill
    x_min = openmc.XPlane(-cell_width/2)
    x_max = openmc.XPlane(cell_width/2)
    y_min = openmc.YPlane(-cell_width/2)
    y_max = openmc.YPlane(cell_width/2)

    # Get coolant material
    coolant_type = inputs_dict.get('coolant_type', 'Light Water')
    coolant_mat = mat_dict[f'{coolant_type} Coolant']

    # Build cells from inside out (all extend full height)

    # Titanium spine
    spine_region = -cyl_spine & +z_bot & -z_top
    spine_cell = openmc.Cell(name='spine')
    spine_cell.id = generate_complex_cell_id(position, 'spine', irradiation_type='SIGMA')
    spine_cell.region = spine_region
    spine_cell.fill = mat_dict['Titanium']
    cells.append(spine_cell)

    # Inner helium gap
    inner_he_region = +cyl_spine & -cyl_inner_he & +z_bot & -z_top
    inner_he_cell = openmc.Cell(name='inner_he')
    inner_he_cell.id = generate_complex_cell_id(position, 'inner_he', irradiation_type='SIGMA')
    inner_he_cell.region = inner_he_region
    inner_he_cell.fill = mat_dict['HT_Helium']
    cells.append(inner_he_cell)

    # Inner graphite holder
    inner_graphite_region = +cyl_inner_he & -cyl_inner_graphite & +z_bot & -z_top
    inner_graphite_cell = openmc.Cell(name='inner_graphite')
    inner_graphite_cell.id = generate_complex_cell_id(position, 'inner_graphite', irradiation_type='SIGMA')
    inner_graphite_cell.region = inner_graphite_region
    inner_graphite_cell.fill = mat_dict['graphite']
    cells.append(inner_graphite_cell)

    # Tungsten sample
    tungsten_region = +cyl_inner_graphite & -cyl_tungsten & +z_bot & -z_top
    tungsten_cell = openmc.Cell(name='tungsten')
    tungsten_cell.id = generate_complex_cell_id(position, 'tungsten', irradiation_type='SIGMA')
    tungsten_cell.region = tungsten_region
    tungsten_cell.fill = mat_dict['Tungsten']
    cells.append(tungsten_cell)

    # Outer graphite holder
    outer_graphite_region = +cyl_tungsten & -cyl_outer_graphite & +z_bot & -z_top
    outer_graphite_cell = openmc.Cell(name='outer_graphite')
    outer_graphite_cell.id = generate_complex_cell_id(position, 'outer_graphite', irradiation_type='SIGMA')
    outer_graphite_cell.region = outer_graphite_region
    outer_graphite_cell.fill = mat_dict['graphite']
    cells.append(outer_graphite_cell)

    # Outer helium gap
    outer_he_region = +cyl_outer_graphite & -cyl_outer_he & +z_bot & -z_top
    outer_he_cell = openmc.Cell(name='outer_he')
    outer_he_cell.id = generate_complex_cell_id(position, 'outer_he', irradiation_type='SIGMA')
    outer_he_cell.region = outer_he_region
    outer_he_cell.fill = mat_dict['HT_Helium']
    cells.append(outer_he_cell)

    # Thimble walls
    thimble_region = +cyl_outer_he & -cyl_thimble_outer & +z_bot & -z_top
    thimble_cell = openmc.Cell(name='thimble')
    thimble_cell.id = generate_complex_cell_id(position, 'thimble', irradiation_type='SIGMA')
    thimble_cell.region = thimble_region
    thimble_cell.fill = mat_dict['Titanium']
    cells.append(thimble_cell)

    # Water gap
    water_gap_region = +cyl_thimble_outer & -cyl_water_gap & +z_bot & -z_top
    water_gap_cell = openmc.Cell(name='water_gap')
    water_gap_cell.id = generate_complex_cell_id(position, 'water_gap', irradiation_type='SIGMA')
    water_gap_cell.region = water_gap_region
    water_gap_cell.fill = coolant_mat
    cells.append(water_gap_cell)

    # Outer fill (Al6061 from water gap to square boundary)
    outer_fill_region = +cyl_water_gap & +x_min & -x_max & +y_min & -y_max & +z_bot & -z_top
    outer_fill_cell = openmc.Cell(name='outer_fill')
    outer_fill_cell.id = generate_complex_cell_id(position, 'outer_fill', irradiation_type='SIGMA')
    outer_fill_cell.region = outer_fill_region
    outer_fill_cell.fill = mat_dict['Al6061']
    cells.append(outer_fill_cell)

    # Create universe with different base for SIGMA
    i, j = position
    universe_id = 6100000 + i * 1000 + j * 10  # Different base from HTWL
    universe = openmc.Universe(universe_id=universe_id, name=f'sigma_{i}_{j}', cells=cells)

    return universe

def build_irradiation_cell_uni(mat_dict, position=None, inputs_dict=None):
    """Build an irradiation cell universe.

    Parameters
    ----------
    mat_dict : dict
        Dictionary of materials
    position : tuple, optional
        (i, j) position in core lattice. If provided, assigns unique ID.
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    openmc.Universe
        Universe containing the irradiation cell
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    # Get irradiation type based on lattice position
    if position is not None:
        i, j = position
        core_lattice = inputs_dict['core_lattice']
        lattice_position = core_lattice[i][j]
        irradiation_type = parse_irradiation_type(lattice_position, inputs_dict)
    else:
        # Fallback to default when no position provided
        irradiation_type = inputs_dict.get('irradiation_fill', 'Vacuum')

    # Check if we should use complex geometry
    if inputs_dict.get('irradiation_cell_complexity', 'Simple') == 'Complex':
        if irradiation_type == 'PWR_loop':
            return build_complex_htwl(mat_dict, position, inputs_dict, use_bwr_water=False, irradiation_type='PWR_loop')
        elif irradiation_type == 'BWR_loop':
            return build_complex_htwl(mat_dict, position, inputs_dict, use_bwr_water=True, irradiation_type='BWR_loop')
        elif irradiation_type == 'Gas_capsule':
            return build_complex_sigma(mat_dict, position, inputs_dict)
        # For other types, fall through to simple geometry

    # ========== SIMPLE GEOMETRY (original implementation) ==========

    # Calculate cell dimensions based on assembly type
    if inputs_dict['assembly_type'] == 'Pin':
        cell_width = inputs_dict['pin_pitch'] * inputs_dict['n_side_pins'] * 100  # m to cm
    else:  # Plate assembly
        cell_width = (2 * inputs_dict['clad_structure_width'] + inputs_dict['fuel_plate_width']) * 100  # m to cm

    # Define the dimensions
    x0 = -cell_width/2
    x3 = cell_width/2
    y0 = -cell_width/2
    y3 = cell_width/2

    if irradiation_type in ['PWR_loop','BWR_loop', 'Gas_capsule']:
        if irradiation_type == 'PWR_loop':
            circle_radius = inputs_dict['PWR_loop_diameter']/2 * cell_width
        elif irradiation_type == 'BWR_loop':
            circle_radius = inputs_dict['BWR_loop_diameter']/2 * cell_width
        elif irradiation_type == 'Gas_capsule':
            circle_radius = inputs_dict['Gas_capsule_diameter']/2 * cell_width

        # Create circular surface
        inner_circle = openmc.ZCylinder(r=circle_radius)

        if inputs_dict['irradiation_clad']:
            # Calculate inner dimensions with cladding
            clad_thickness = inputs_dict['irradiation_clad_thickness'] * 100  # m to cm
            x1 = x0 + clad_thickness
            x2 = x3 - clad_thickness
            y1 = y0 + clad_thickness
            y2 = y3 - clad_thickness

            # Create planes
            x0p = openmc.XPlane(x0)
            x1p = openmc.XPlane(x1)
            x2p = openmc.XPlane(x2)
            x3p = openmc.XPlane(x3)
            y0p = openmc.YPlane(y0)
            y1p = openmc.YPlane(y1)
            y2p = openmc.YPlane(y2)
            y3p = openmc.YPlane(y3)

            # Map clad type to material
            clad_material_map = {
                'Zirc2': 'Zircaloy',
                'Zirc4': 'Zircaloy',
                'Al6061': 'Al6061'
            }
            clad_material = clad_material_map[inputs_dict['clad_type']]

            # Define regions for PWR loop with cladding
            # Inner circle (PWR loop material) - bounded by cladding box
            inner_region = -inner_circle & +x1p & -x2p & +y1p & -y2p

            # Outer annular region (Al6061) - inside cladding box but outside circle
            outer_region = +inner_circle & +x1p & -x2p & +y1p & -y2p

            # Cladding regions (same as original)
            bottom_clad_region = +x0p & -x3p & +y0p & -y1p
            top_clad_region = +x0p & -x3p & +y2p & -y3p
            left_clad_region = +x0p & -x1p & +y1p & -y2p
            right_clad_region = +x2p & -x3p & +y1p & -y2p

            # Create cells
            inner_cell = openmc.Cell(name=f'{irradiation_type}_center')
            inner_cell.region = inner_region
            if position is not None:
                inner_cell.id = generate_cell_id('irradiation', position)
                inner_cell.name = get_irradiation_cell_name(position, inputs_dict['core_lattice']) + f'_{irradiation_type}'
            inner_cell.fill = mat_dict[irradiation_type]

            outer_cell = openmc.Cell(name=f'{irradiation_type}_outer')
            outer_cell.region = outer_region
            outer_cell.fill = mat_dict['Al6061']

            # Create cladding cells
            bottom_clad = openmc.Cell(name='irradiation_bottom_clad')
            bottom_clad.region = bottom_clad_region
            bottom_clad.fill = mat_dict[clad_material]

            top_clad = openmc.Cell(name='irradiation_top_clad')
            top_clad.region = top_clad_region
            top_clad.fill = mat_dict[clad_material]

            left_clad = openmc.Cell(name='irradiation_left_clad')
            left_clad.region = left_clad_region
            left_clad.fill = mat_dict[clad_material]

            right_clad = openmc.Cell(name='irradiation_right_clad')
            right_clad.region = right_clad_region
            right_clad.fill = mat_dict[clad_material]

            # Assign IDs to cladding cells if position is provided
            if position is not None:
                bottom_clad.id = generate_cell_id('irradiation', position, clad_part='bottom')
                top_clad.id = generate_cell_id('irradiation', position, clad_part='top')
                left_clad.id = generate_cell_id('irradiation', position, clad_part='left')
                right_clad.id = generate_cell_id('irradiation', position, clad_part='right')

                pos_name = get_irradiation_cell_name(position, inputs_dict['core_lattice'])
                bottom_clad.name = f"{pos_name}_bottom_clad"
                top_clad.name = f"{pos_name}_top_clad"
                left_clad.name = f"{pos_name}_left_clad"
                right_clad.name = f"{pos_name}_right_clad"

            # Create universe with all cells
            cells = [
                inner_cell,
                outer_cell,
                bottom_clad,
                top_clad,
                left_clad,
                right_clad
            ]

        else:
            # Without cladding
            x0p = openmc.XPlane(x0)
            x3p = openmc.XPlane(x3)
            y0p = openmc.YPlane(y0)
            y3p = openmc.YPlane(y3)

            # Inner circle (loop material)
            inner_region = -inner_circle & +x0p & -x3p & +y0p & -y3p

            # Outer annular region (Al6061)
            outer_region = +inner_circle & +x0p & -x3p & +y0p & -y3p

            # Create cells
            inner_cell = openmc.Cell(name=f'{irradiation_type}_center', region=inner_region)
            if position is not None:
                inner_cell.id = generate_cell_id('irradiation', position)
                inner_cell.name = get_irradiation_cell_name(position, inputs_dict['core_lattice']) + f'_{irradiation_type}'
            inner_cell.fill = mat_dict[irradiation_type]

            outer_cell = openmc.Cell(name=f'{irradiation_type}_outer', region=outer_region)
            outer_cell.fill = mat_dict['Al6061']

            cells = [inner_cell, outer_cell]

    else:
        # STANDARD SQUARE GEOMETRY (any material)
        # Use the exact material name from irradiation_fill
        fill_material = mat_dict[irradiation_type]

        if inputs_dict['irradiation_clad']:
            # Calculate inner dimensions with cladding
            clad_thickness = inputs_dict['irradiation_clad_thickness'] * 100  # m to cm
            x1 = x0 + clad_thickness
            x2 = x3 - clad_thickness
            y1 = y0 + clad_thickness
            y2 = y3 - clad_thickness

            # Create planes
            x0p = openmc.XPlane(x0)
            x1p = openmc.XPlane(x1)
            x2p = openmc.XPlane(x2)
            x3p = openmc.XPlane(x3)
            y0p = openmc.YPlane(y0)
            y1p = openmc.YPlane(y1)
            y2p = openmc.YPlane(y2)
            y3p = openmc.YPlane(y3)

            # Map clad type to material
            clad_material_map = {
                'Zirc2': 'Zircaloy',
                'Zirc4': 'Zircaloy',
                'Al6061': 'Al6061'
            }
            clad_material = clad_material_map[inputs_dict['clad_type']]

            # Define regions (same as original)
            center_region = +x1p & -x2p & +y1p & -y2p
            bottom_clad_region = +x0p & -x3p & +y0p & -y1p
            top_clad_region = +x0p & -x3p & +y2p & -y3p
            left_clad_region = +x0p & -x1p & +y1p & -y2p
            right_clad_region = +x2p & -x3p & +y1p & -y2p

            # Create cells
            center_cell = openmc.Cell(name='irradiation_center')
            center_cell.region = center_region
            if position is not None:
                center_cell.id = generate_cell_id('irradiation', position)
                center_cell.name = get_irradiation_cell_name(position, inputs_dict['core_lattice'])
            center_cell.fill = fill_material  # Use selected material

            # Create cladding cells (same as original)
            bottom_clad = openmc.Cell(name='irradiation_bottom_clad')
            bottom_clad.region = bottom_clad_region
            bottom_clad.fill = mat_dict[clad_material]

            top_clad = openmc.Cell(name='irradiation_top_clad')
            top_clad.region = top_clad_region
            top_clad.fill = mat_dict[clad_material]

            left_clad = openmc.Cell(name='irradiation_left_clad')
            left_clad.region = left_clad_region
            left_clad.fill = mat_dict[clad_material]

            right_clad = openmc.Cell(name='irradiation_right_clad')
            right_clad.region = right_clad_region
            right_clad.fill = mat_dict[clad_material]

            # Assign IDs if position provided
            if position is not None:
                bottom_clad.id = generate_cell_id('irradiation', position, clad_part='bottom')
                top_clad.id = generate_cell_id('irradiation', position, clad_part='top')
                left_clad.id = generate_cell_id('irradiation', position, clad_part='left')
                right_clad.id = generate_cell_id('irradiation', position, clad_part='right')

                pos_name = get_irradiation_cell_name(position, inputs_dict['core_lattice'])
                bottom_clad.name = f"{pos_name}_bottom_clad"
                top_clad.name = f"{pos_name}_top_clad"
                left_clad.name = f"{pos_name}_left_clad"
                right_clad.name = f"{pos_name}_right_clad"

            cells = [
                center_cell,
                bottom_clad,
                top_clad,
                left_clad,
                right_clad
            ]

        else:
            # Create single cell without cladding
            x0p = openmc.XPlane(x0)
            x3p = openmc.XPlane(x3)
            y0p = openmc.YPlane(y0)
            y3p = openmc.YPlane(y3)

            main_region = +x0p & -x3p & +y0p & -y3p
            main_cell = openmc.Cell(name='irradiation_cell', region=main_region)
            if position is not None:
                cell_id = generate_cell_id('irradiation', position)
                main_cell.id = cell_id
                main_cell.name = get_irradiation_cell_name(position, inputs_dict['core_lattice'])

            main_cell.fill = fill_material  # Use selected material
            cells = [main_cell]

    # Create universe with explicit ID from the start
    if position is not None:
        i, j = position
        # Generate a unique universe ID based on position
        universe_base = 6000000  # Irradiation universes
        universe_id = universe_base + i * 1000 + j * 10
        irradiation_universe = openmc.Universe(universe_id=universe_id, name='irradiation_universe', cells=cells)
        irradiation_universe.name = f"irradiation_universe_{i}_{j}"
    else:
        irradiation_universe = openmc.Universe(name='irradiation_universe', cells=cells)

    return irradiation_universe
