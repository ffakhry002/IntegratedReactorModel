"""
Helper functions for plotting.
"""

import numpy as np
import openmc
from inputs import inputs

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

    # Calculate dimensions deterministically
    if inputs_dict['assembly_type'] == 'Pin':
        cell_width = inputs_dict['pin_pitch'] * inputs_dict['n_side_pins'] * 100  # Convert m to cm
    else:
        cell_width = (inputs_dict['fuel_plate_width'] + 2 * inputs_dict['clad_structure_width']) * 100  # Convert m to cm

    height = inputs_dict['fuel_height'] * 100  # Convert m to cm

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
                        circle_radius = inputs_dict['PWR_loop_diameter']/2 * cell_width
                        volume = np.pi * circle_radius**2 * height
                    elif lattice_position.endswith('B'):  # BWR_loop
                        circle_radius = inputs_dict['BWR_loop_diameter']/2 * cell_width
                        volume = np.pi * circle_radius**2 * height
                    elif lattice_position.endswith('G'):  # Gas_capsule
                        circle_radius = inputs_dict['Gas_capsule_diameter']/2 * cell_width
                        volume = np.pi * circle_radius**2 * height
                    else:
                        # Standard square geometry
                        width = cell_width
                        if inputs_dict.get('irradiation_clad', False):
                            clad_thickness = inputs_dict['irradiation_clad_thickness'] * 100  # Convert m to cm
                            width = width - (2 * clad_thickness)  # Subtract cladding from both sides
                        volume = width * width * height

                    return volume

            except (IndexError, KeyError):
                pass  # Fall back to old calculation if something goes wrong

        # Fallback to old square calculation if we can't determine geometry
        width = cell_width
        if inputs_dict.get('irradiation_clad', False):
            clad_thickness = inputs_dict['irradiation_clad_thickness'] * 100  # Convert m to cm
            width = width - (2 * clad_thickness)  # Subtract cladding from both sides
        return width * width * height
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
        Name of the irradiation tally (e.g., 'I_1B', 'I_2', 'I_3P_axial')
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

    # Calculate cell dimensions deterministically
    if inputs_dict['assembly_type'] == 'Pin':
        cell_width = inputs_dict['pin_pitch'] * inputs_dict['n_side_pins'] * 100  # Convert m to cm
    else:
        cell_width = (inputs_dict['fuel_plate_width'] + 2 * inputs_dict['clad_structure_width']) * 100  # Convert m to cm

    height = inputs_dict['fuel_height'] * 100  # Convert m to cm

    # Determine geometry type from position name
    if base_name.endswith('P'):  # PWR_loop
        circle_radius = inputs_dict['PWR_loop_diameter']/2 * cell_width
        total_volume = np.pi * circle_radius**2 * height
    elif base_name.endswith('B'):  # BWR_loop
        circle_radius = inputs_dict['BWR_loop_diameter']/2 * cell_width
        total_volume = np.pi * circle_radius**2 * height
    elif base_name.endswith('G'):  # Gas_capsule
        circle_radius = inputs_dict['Gas_capsule_diameter']/2 * cell_width
        total_volume = np.pi * circle_radius**2 * height
    else:
        # Standard square geometry
        width = cell_width
        if inputs_dict.get('irradiation_clad', False):
            clad_thickness = inputs_dict['irradiation_clad_thickness'] * 100  # Convert m to cm
            width = width - (2 * clad_thickness)  # Subtract cladding from both sides
        total_volume = width * width * height

    return total_volume

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
        # Cylindrical volume = π * r² * height
        # For single radial and azimuthal cell covering full cylinder
        r_outer = mesh.r_grid[-1]  # Outer radius
        r_inner = mesh.r_grid[0]   # Inner radius (should be 0)
        height = mesh.z_grid[-1] - mesh.z_grid[0]

        # Total cylindrical volume
        total_volume = np.pi * (r_outer**2 - r_inner**2) * height

        # Number of mesh cells (r × φ × z)
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
