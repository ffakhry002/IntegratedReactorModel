import openmc
import numpy as np
import os
import sys

# Add root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

from inputs import inputs
from .pin_fuel import build_fuel_assembly_uni as build_pin_assembly
from .plate_fuel import build_fuel_assembly_uni as build_plate_assembly
from .irradiation_cell import build_irradiation_cell_uni

def build_core_uni(mat_dict, inputs_dict=None):
    """Build the full core universe with proper cell definitions"""
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    # Convert all dimensions to cm for OpenMC
    tank_r = inputs_dict['tank_radius'] * 100
    refl_thickness = inputs_dict['reflector_thickness'] * 100
    bioshield_thickness = inputs_dict['bioshield_thickness'] * 100

    # Calculate radii for cylinders
    r1 = openmc.ZCylinder(r=tank_r)  # Tank boundary
    r2 = openmc.ZCylinder(r=tank_r + refl_thickness)  # Reflector boundary
    r3 = openmc.ZCylinder(r=tank_r + refl_thickness + bioshield_thickness, boundary_type='vacuum')  # Bioshield boundary

    # Calculate heights for each section (all converted to cm)
    fuel = inputs_dict['fuel_height'] * 100  # Core height
    half_fuel = fuel/2  # Half of fuel height for centering
    plenum = inputs_dict['plenum_height'] * 100  # Plenum height
    top_refl = inputs_dict['top_reflector_thickness'] * 100  # Top reflector
    top_bio = inputs_dict['top_bioshield_thickness'] * 100  # Top bioshield
    feed = inputs_dict['feed_thickness'] * 100  # Feed section
    bottom_refl = inputs_dict['bottom_reflector_thickness'] * 100  # Bottom reflector
    bottom_bio = inputs_dict['bottom_bioshield_thickness'] * 100  # Bottom bioshield

    # Define axial planes centered around z=0
    z_bottom_bio = openmc.ZPlane(z0=-half_fuel-feed-bottom_refl-bottom_bio, boundary_type='vacuum')
    z_bottom_refl = openmc.ZPlane(z0=-half_fuel-feed-bottom_refl)
    z_feed = openmc.ZPlane(z0=-half_fuel-feed)
    z_bottom_fuel = openmc.ZPlane(z0=-half_fuel)
    z_top_fuel = openmc.ZPlane(z0=half_fuel)
    z_plenum = openmc.ZPlane(z0=half_fuel+plenum)
    z_top_refl = openmc.ZPlane(z0=half_fuel+plenum+top_refl)
    z_top_bio = openmc.ZPlane(z0=half_fuel+plenum+top_refl+top_bio, boundary_type='vacuum')

    # Create coolant universe
    coolant_cell = openmc.Cell(fill=mat_dict[f"{inputs_dict['coolant_type']} Outer"])
    coolant_universe = openmc.Universe(cells=[coolant_cell])

    # Create core lattice
    lattice_array = np.array(inputs_dict['core_lattice'])
    n_rows, n_cols = lattice_array.shape

    # Calculate assembly pitch
    if inputs_dict['assembly_type'] == 'Pin':
        assembly_pitch = inputs_dict['pin_pitch'] * inputs_dict['n_side_pins'] * 100
    else:
        assembly_pitch = (inputs_dict['plates_per_assembly'] * inputs_dict['fuel_plate_pitch'] +
                        2 * inputs_dict['clad_structure_width']) * 100

    # Define core lattice
    core_lattice = openmc.RectLattice()

    # Calculate lower left position to ensure the center of the lattice is at (0,0)
    # For odd-sized grids, the center point is the center of the middle assembly
    # For even-sized grids, the center point is at the corner where the central assemblies meet
    ll_x = -assembly_pitch * n_cols / 2
    ll_y = -assembly_pitch * n_rows / 2

    core_lattice.lower_left = (ll_x, ll_y)
    core_lattice.pitch = (assembly_pitch, assembly_pitch)

    # Create universe array with proper bounds
    universe_array = np.empty(lattice_array.shape, dtype=openmc.Universe)
    first_irr_universe = None

    # Fill universe array
    for i in range(n_rows):
        for j in range(n_cols):
            position = (i, j)
            if lattice_array[i,j] == 'F':
                universe_array[i,j] = build_pin_assembly(mat_dict, position=position, inputs_dict=inputs_dict) if inputs_dict['assembly_type'] == 'Pin' \
                                    else build_plate_assembly(mat_dict, position=position, inputs_dict=inputs_dict)
            elif lattice_array[i,j] == 'E':
                universe_array[i,j] = build_pin_assembly(mat_dict, position=position, is_enhanced=True, inputs_dict=inputs_dict) if inputs_dict['assembly_type'] == 'Pin' \
                                    else build_plate_assembly(mat_dict, position=position, is_enhanced=True, inputs_dict=inputs_dict)
            elif lattice_array[i,j].startswith('I_'):
                universe_array[i,j] = build_irradiation_cell_uni(mat_dict, position=position, inputs_dict=inputs_dict)
                if first_irr_universe is None:
                    first_irr_universe = universe_array[i,j]
            else:  # 'C' for coolant
                universe_array[i,j] = coolant_universe

    core_lattice.universes = universe_array
    core_lattice.outer = coolant_universe

    # Create cells for each region with explicit bounds
    cells = []

    # Active core region (centered around z=0)
    active_core_cell = openmc.Cell(fill=core_lattice,
                                  region=-r1 & +z_bottom_fuel & -z_top_fuel)
    cells.append(active_core_cell)

    # Feed region (below core)
    feed_cell = openmc.Cell(fill=mat_dict[f"{inputs_dict['coolant_type']} Feed"],
                           region=(-r1 & +z_feed & -z_bottom_fuel) )
    cells.append(feed_cell)

    # Plenum region (above core)
    plenum_cell = openmc.Cell(fill=mat_dict[f"{inputs_dict['coolant_type']} Plenum"],
                             region=-r1 & +z_top_fuel & -z_plenum)
    cells.append(plenum_cell)

    # Outer radial reflector (with complete axial extent)
    reflector_cell = openmc.Cell(fill=mat_dict[inputs_dict['reflector_material']],
                                region=+r1 & -r2 & +z_bottom_refl & -z_top_refl)
    cells.append(reflector_cell)

    # Bottom reflector in tank region
    bottom_reflector_cell = openmc.Cell(fill=mat_dict[inputs_dict['reflector_material']],
                                      region=-r1 & +z_bottom_refl & -z_feed)
    cells.append(bottom_reflector_cell)

    # Top reflector in tank region
    top_reflector_cell = openmc.Cell(fill=mat_dict[inputs_dict['reflector_material']],
                                   region=-r1 & +z_plenum & -z_top_refl)
    cells.append(top_reflector_cell)

    # Bioshield regions with explicit bounds for each section
    bioshield_radial = openmc.Cell(fill=mat_dict[inputs_dict['bioshield_material']],
                                  region=+r2 & -r3 & +z_bottom_bio & -z_top_bio)
    cells.append(bioshield_radial)

    bioshield_bottom = openmc.Cell(fill=mat_dict[inputs_dict['bioshield_material']],
                                  region=-r2 & +z_bottom_bio & -z_bottom_refl)
    cells.append(bioshield_bottom)

    bioshield_top = openmc.Cell(fill=mat_dict[inputs_dict['bioshield_material']],
                               region=-r2 & +z_top_refl & -z_top_bio)
    cells.append(bioshield_top)

    # Create the core universe with all bounded cells
    core_universe = openmc.Universe(cells=cells)

    return core_universe, first_irr_universe
