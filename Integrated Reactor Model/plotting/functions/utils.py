"""
Helper functions for plotting.
"""

import numpy as np
import openmc
from inputs import inputs
import sys
import os

# Add path to geometry helpers
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from Reactor.geometry_helpers.irradiation_experiments import get_experiment_config, get_scaled_radii

def _get_cell_dimensions(inputs_dict):
    """Get cell width and height from inputs.

    Parameters
    ----------
    inputs_dict : dict
        Inputs dictionary

    Returns
    -------
    tuple
        (cell_width, height) in cm
    """
    if inputs_dict['assembly_type'] == 'Pin':
        cell_width = inputs_dict['pin_pitch'] * inputs_dict['n_side_pins'] * 100  # Convert m to cm
    else:
        cell_width = (inputs_dict['fuel_plate_width'] + 2 * inputs_dict['clad_structure_width']) * 100  # Convert m to cm

    height = inputs_dict['fuel_height'] * 100  # Convert m to cm
    return cell_width, height

def _calculate_sigma_scaling_and_radii(cell_width, inputs_dict):
    """Calculate SIGMA scaling factor and scaled radii.

    Parameters
    ----------
    cell_width : float
        Cell width in cm
    inputs_dict : dict
        Inputs dictionary

    Returns
    -------
    tuple
        (r_sample_inner, r_sample_outer) in cm
    """
    diameter_fraction = inputs_dict['Gas_capsule_diameter']
    target_diameter = cell_width * diameter_fraction

    # Get scaled radii from centralized configuration
    scaled_radii, scale_factor = get_scaled_radii('Gas_capsule', target_diameter)
    r_sample_inner = scaled_radii['sample_inner']  # Inner radius of sample annulus
    r_sample_outer = scaled_radii['sample_outer']  # Outer radius of sample annulus

    return r_sample_inner, r_sample_outer

def _calculate_annular_volume(r_inner, r_outer, height):
    """Calculate annular volume.

    Parameters
    ----------
    r_inner : float
        Inner radius in cm
    r_outer : float
        Outer radius in cm
    height : float
        Height in cm

    Returns
    -------
    float
        Annular volume in cm³
    """
    return np.pi * (r_outer**2 - r_inner**2) * height

def _calculate_cylindrical_volume(diameter_key, cell_width, height, inputs_dict):
    """Calculate cylindrical volume for loop-type irradiation.

    Parameters
    ----------
    diameter_key : str
        Key for diameter in inputs (e.g., 'PWR_loop_diameter')
    cell_width : float
        Cell width in cm
    height : float
        Height in cm
    inputs_dict : dict
        Inputs dictionary

    Returns
    -------
    float
        Cylindrical volume in cm³
    """
    circle_radius = inputs_dict[diameter_key]/2 * cell_width
    return np.pi * circle_radius**2 * height

def _calculate_square_volume_with_cladding(cell_width, height, inputs_dict):
    """Calculate square volume with optional cladding adjustment.

    Parameters
    ----------
    cell_width : float
        Cell width in cm
    height : float
        Height in cm
    inputs_dict : dict
        Inputs dictionary

    Returns
    -------
    float
        Square volume in cm³
    """
    width = cell_width
    if inputs_dict.get('irradiation_clad', False):
        clad_thickness = inputs_dict['irradiation_clad_thickness'] * 100  # Convert m to cm
        width = width - (2 * clad_thickness)  # Subtract cladding from both sides
    return width * width * height

def _calculate_htwl_scaling_and_radii(cell_width, inputs_dict, use_bwr_water=False):
    """Calculate HTWL scaling factor and scaled radii for sample regions.

    Parameters
    ----------
    cell_width : float
        Cell width in cm
    inputs_dict : dict
        Inputs dictionary
    use_bwr_water : bool
        Whether this is BWR (True) or PWR (False)

    Returns
    -------
    tuple
        (r_sample_inner, r_sample_outer, sample_height) in cm
    """
    # Use BWR or PWR diameter depending on water type
    if use_bwr_water:
        diameter_fraction = inputs_dict['BWR_loop_diameter']
        irradiation_type = 'BWR_loop'
    else:
        diameter_fraction = inputs_dict['PWR_loop_diameter']
        irradiation_type = 'PWR_loop'

    target_diameter = cell_width * diameter_fraction

    # Get scaled radii from centralized configuration
    scaled_radii, scale_factor = get_scaled_radii(irradiation_type, target_diameter)
    r_sample_inner = scaled_radii['sample_inner']  # Inner radius of sample annulus
    r_sample_outer = scaled_radii['sample_outer']   # Outer radius of sample annulus

    # HTWL sample height from centralized configuration
    config = get_experiment_config(irradiation_type)
    sample_height = config['sample_height']  # cm

    return r_sample_inner, r_sample_outer, sample_height

def get_cell_volume(cell_id, sp, is_irradiation=False, inputs_dict=None):
    """Get the volume of a cell using ONLY deterministic calculations.

    This function NEVER uses cell.volume from OpenMC to avoid any potential
    Monte Carlo statistical noise in volume calculations. Always calculates
    volumes deterministically from exact geometry.

    Parameters
    ----------
    cell_id : int
        ID of the cell
    sp : openmc.StatePoint
        StatePoint file containing the geometry information
    is_irradiation : bool, optional
        Whether this is an irradiation position cell (default: False)
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    float
        Volume of the cell in cm³
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    # ALWAYS calculate deterministically - NEVER use OpenMC's cell.volume
    # This ensures perfect precision and eliminates any Monte Carlo noise

    cell_width, height = _get_cell_dimensions(inputs_dict)

    # Check for Complex SIGMA sample cells first (higher priority)
    if cell_id >= 6100000 and cell_id < 6200000:  # SIGMA base range
        # Check if this is a sample cell (component code 15)
        component_code = cell_id % 100
        if component_code == 15:  # sample component (tungsten in geometry but could be any material)
            # Calculate annular SIGMA sample volume using same scaling as geometry
            r_sample_inner, r_sample_outer = _calculate_sigma_scaling_and_radii(cell_width, inputs_dict)
            return _calculate_annular_volume(r_sample_inner, r_sample_outer, height)

        # HTWL sample cells (component codes 3, 6, 9, 12 for the four sample outer rings)
        elif component_code in [3, 6, 9, 12]:
            # Determine if this is PWR or BWR
            use_bwr_water = 6200000 <= cell_id < 6300000  # BWR range

            # Calculate single HTWL sample ring volume with correct height
            r_sample_inner, r_sample_outer, sample_height = _calculate_htwl_scaling_and_radii(cell_width, inputs_dict, use_bwr_water)
            single_sample_volume = _calculate_annular_volume(r_sample_inner, r_sample_outer, sample_height)

            # Note: This returns volume of ONE sample ring.
            return single_sample_volume

    # For irradiation positions, calculate volume based on EXACT geometry
    if is_irradiation:
        # Reverse-engineer position from cell ID
        # Irradiation cell IDs: 3000000 + i * 100000 + j * 1000 + part_num
        if cell_id >= 3000000:
            position_code = cell_id - 3000000
            i = position_code // 100000
            j = (position_code % 100000) // 1000

            # Get the position string from core lattice to determine irradiation type
            try:
                core_lattice = inputs_dict['core_lattice']
                if 0 <= i < len(core_lattice) and 0 <= j < len(core_lattice[i]):
                    lattice_position = core_lattice[i][j]

                    # Determine irradiation type and calculate EXACT deterministic volume
                    if lattice_position.endswith('P'):  # PWR_loop
                        return _calculate_cylindrical_volume('PWR_loop_diameter', cell_width, height, inputs_dict)
                    elif lattice_position.endswith('B'):  # BWR_loop
                        return _calculate_cylindrical_volume('BWR_loop_diameter', cell_width, height, inputs_dict)
                    elif lattice_position.endswith('G'):  # Gas_capsule
                        # Simple mode - full capsule volume
                        # Note: Complex mode should never reach here because it uses specific sample cells
                        return _calculate_cylindrical_volume('Gas_capsule_diameter', cell_width, height, inputs_dict)
                    else:
                        # Standard square geometry
                        return _calculate_square_volume_with_cladding(cell_width, height, inputs_dict)

            except (IndexError, KeyError):
                pass  # Fall back to old calculation if something goes wrong

        # Fallback to old square calculation if we can't determine geometry
        return _calculate_square_volume_with_cladding(cell_width, height, inputs_dict)
    else:
        # Non-irradiation cells use square geometry
        return cell_width * cell_width * height

def calculate_deterministic_irradiation_volume(tally_name, inputs_dict=None):
    """Calculate deterministic volume for any irradiation position from tally name.

    This unified function eliminates discrepancies between cell and mesh tallies
    by always using the same deterministic calculation regardless of tally type.

    Parameters
    ----------
    tally_name : str
        Name of the irradiation tally (e.g., 'I_1B', 'I_2', 'I_3P_axial', 'I_4G')
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    float
        Volume in cm³ for the entire irradiation region
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    # Extract base position name (remove _axial suffix if present)
    base_name = tally_name.replace('_axial', '')

    cell_width, height = _get_cell_dimensions(inputs_dict)

    # Determine geometry type from position name
    if base_name.endswith('P'):  # PWR_loop
        # Check if this is Complex mode - if so, calculate HTWL sample volume
        if inputs_dict.get('irradiation_cell_complexity', 'Simple') == 'Complex':
            # Calculate single HTWL sample volume (12 o'clock position only)
            r_sample_inner, r_sample_outer, sample_height = _calculate_htwl_scaling_and_radii(cell_width, inputs_dict, use_bwr_water=False)
            single_sample_volume = _calculate_annular_volume(r_sample_inner, r_sample_outer, sample_height)
            return single_sample_volume  # Just one sample ring, not 4×
        else:
            return _calculate_cylindrical_volume('PWR_loop_diameter', cell_width, height, inputs_dict)
    elif base_name.endswith('B'):  # BWR_loop
        # Check if this is Complex mode - if so, calculate HTWL sample volume
        if inputs_dict.get('irradiation_cell_complexity', 'Simple') == 'Complex':
            # Calculate single HTWL sample volume (12 o'clock position only)
            r_sample_inner, r_sample_outer, sample_height = _calculate_htwl_scaling_and_radii(cell_width, inputs_dict, use_bwr_water=True)
            single_sample_volume = _calculate_annular_volume(r_sample_inner, r_sample_outer, sample_height)
            return single_sample_volume  # Just one sample ring, not 4×
        else:
            return _calculate_cylindrical_volume('BWR_loop_diameter', cell_width, height, inputs_dict)
    elif base_name.endswith('G'):  # Gas_capsule
        # Check if this is Complex mode - if so, calculate sample annular volume
        if inputs_dict.get('irradiation_cell_complexity', 'Simple') == 'Complex':
            # Calculate annular sample volume using same scaling as geometry
            r_sample_inner, r_sample_outer = _calculate_sigma_scaling_and_radii(cell_width, inputs_dict)
            return _calculate_annular_volume(r_sample_inner, r_sample_outer, height)
        else:
            # Simple mode - full capsule volume
            return _calculate_cylindrical_volume('Gas_capsule_diameter', cell_width, height, inputs_dict)
    else:
        # Standard square geometry
        return _calculate_square_volume_with_cladding(cell_width, height, inputs_dict)

def get_mesh_volume(mesh):
    """Calculate the volume of a mesh element.

    Parameters
    ----------
    mesh : openmc.RegularMesh or openmc.CylindricalMesh
        The mesh to calculate volume for

    Returns
    -------
    float
        Volume of a single mesh element in cm³
    """
    if isinstance(mesh, openmc.CylindricalMesh):
        # Cylindrical volume = pi * r² * height
        # For single radial and azimuthal cell covering full cylinder
        r_outer = mesh.r_grid[-1]  # Outer radius
        r_inner = mesh.r_grid[0]   # Inner radius
        height = mesh.z_grid[-1] - mesh.z_grid[0]

        # Total cylindrical volume
        total_volume = np.pi * (r_outer**2 - r_inner**2) * height

        # Number of mesh cells (r × phi × z)
        n_r = len(mesh.r_grid) - 1
        n_phi = len(mesh.phi_grid) - 1
        n_z = len(mesh.z_grid) - 1
        total_cells = n_r * n_phi * n_z

        return total_volume / total_cells
    else:
        # Existing RegularMesh calculation
        return np.prod(np.array(mesh.upper_right) - np.array(mesh.lower_left))/np.prod(mesh.dimension)

def get_tally_volume(tally, sp, inputs_dict=None):
    """Get the volume associated with a tally using UNIFIED deterministic calculations.

    For irradiation positions, this function now uses a unified deterministic
    calculation regardless of whether the tally uses CellFilter or MeshFilter.
    This eliminates the 0.7% discrepancy between different tally types.

    Parameters
    ----------
    tally : openmc.Tally
        The tally to get volume for
    sp : openmc.StatePoint
        StatePoint file containing geometry information
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    float
        Volume in cm³
    """
    # UNIFIED APPROACH: For irradiation positions, always use deterministic calculation
    if tally.name and tally.name.startswith('I_'):
        # For mesh tallies, we need volume per segment
        if any(isinstance(f, openmc.MeshFilter) for f in tally.filters):
            total_volume = calculate_deterministic_irradiation_volume(tally.name, inputs_dict)
            # For axial mesh tallies, get number of axial segments
            n_segments = len(tally.mean.ravel())  # Total number of mesh elements
            return total_volume / n_segments
        else:
            # For cell tallies, return total volume
            return calculate_deterministic_irradiation_volume(tally.name, inputs_dict)

    # For non-irradiation tallies, use original logic
    for filter in tally.filters:
        if isinstance(filter, openmc.MeshFilter):
            return get_mesh_volume(sp.meshes[filter._mesh.id])
        elif isinstance(filter, openmc.CellFilter):
            # Check if this is an irradiation position tally
            is_irradiation = tally.name and tally.name.startswith('I_')
            return get_cell_volume(filter.bins[0], sp, is_irradiation, inputs_dict)

    raise ValueError(f"No volume information found for tally {tally.name}")
