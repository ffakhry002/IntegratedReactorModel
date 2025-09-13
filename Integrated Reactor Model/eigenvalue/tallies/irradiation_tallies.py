"""
Functions for creating tallies in irradiation positions.

This module provides functions for:
- Energy-dependent flux tallies in irradiation positions
- Axial flux distribution tallies in irradiation positions
"""

import openmc
import numpy as np
from inputs import inputs
from Reactor.geometry_helpers.utils import generate_cell_id, generate_filter_id
from Reactor.geometry_helpers.irradiation_cell import parse_irradiation_type, generate_complex_cell_id
from Reactor.geometry_helpers.irradiation_experiments import get_experiment_config, get_scaled_radii, get_reference_axial_bounds
from eigenvalue.tallies.energy_groups import get_energy_bins

def create_irradiation_tallies(inputs_dict=None):
    """Create tallies for irradiation positions.

    Parameters
    ----------
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    openmc.Tallies
        Collection of energy-dependent flux tallies for each irradiation position
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    tallies = openmc.Tallies()

    # Get energy group structure
    energy_bins = get_energy_bins(inputs_dict)
    energy_filter = openmc.EnergyFilter(energy_bins)
    energy_filter.id = generate_filter_id('energy', component='irradiation')
    print(f"Created irradiation energy filter - ID: {energy_filter.id}, Groups: {len(energy_bins)-1}")

    # Create tallies for each irradiation position in the core
    core_layout = inputs_dict['core_lattice']
    for i, row in enumerate(core_layout):
        for j, pos in enumerate(row):
            if pos.startswith('I_'):  # This is an irradiation position
                # Parse irradiation type to determine if we need complex targeting
                irradiation_type = parse_irradiation_type(pos, inputs_dict)

                # Determine which cell to target based on complexity and type
                if inputs_dict.get('irradiation_cell_complexity', 'Simple') == 'Complex':
                    if irradiation_type == 'Gas_capsule':
                        # For Complex Gas_capsule (SIGMA), target the single sample annular region
                        cell_id = generate_complex_cell_id((i, j), 'sample', irradiation_type='SIGMA')
                        cell_filter = openmc.CellFilter([cell_id])
                        cell_filter.id = generate_filter_id('cell', position=(i, j), component='irradiation', index=1)
                        print(f"Created Complex Gas_capsule tally {pos} targeting SIGMA sample cell ID: {cell_id}, filter ID: {cell_filter.id}")
                    elif irradiation_type in ['PWR_loop', 'BWR_loop']:
                        # For Complex HTWL, target only the 12 o'clock sample (sample_1)
                        irrad_type = 'PWR_loop' if irradiation_type == 'PWR_loop' else 'BWR_loop'
                        sample_cell_id = generate_complex_cell_id((i, j), 'sample_1_sample', irradiation_type=irrad_type)
                        cell_filter = openmc.CellFilter([sample_cell_id])
                        cell_filter.id = generate_filter_id('cell', position=(i, j), component='irradiation', index=2)
                        print(f"Created Complex {irradiation_type} tally {pos} targeting HTWL 12 o'clock sample cell ID: {sample_cell_id}, filter ID: {cell_filter.id}")
                    else:
                        # For other complex types, fall back to standard
                        cell_id = generate_cell_id('irradiation', (i, j))
                        cell_filter = openmc.CellFilter([cell_id])
                        cell_filter.id = generate_filter_id('cell', position=(i, j), component='irradiation', index=3)
                else:
                    # For all Simple cases, use standard irradiation cell
                    cell_id = generate_cell_id('irradiation', (i, j))
                    cell_filter = openmc.CellFilter([cell_id])
                    cell_filter.id = generate_filter_id('cell', position=(i, j), component='irradiation', index=0)

                # Create tally for this position
                tally = openmc.Tally(name=pos)
                tally.filters = [cell_filter, energy_filter]
                tally.scores = ['flux']
                tallies.append(tally)

    return tallies

def create_irradiation_axial_tallies(inputs_dict=None):
    """Create axial flux tallies for irradiation positions.

    This creates mesh tallies for each irradiation position:
    - Cylindrical meshes for PWR/BWR/Gas positions (P, B, G suffixes)
    - Regular (square) meshes for other positions (no suffix)
    - For Complex Gas_capsule: annular cylindrical mesh for tungsten region

    All meshes divide the position into axial segments and use single energy group.

    Parameters
    ----------
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    openmc.Tallies
        Collection of axial flux tallies for each irradiation position
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    # Get number of axial segments from inputs
    n_axial_segments = inputs_dict['irradiation_axial_segments']

    tallies = openmc.Tallies()

    # Get core dimensions from inputs (in cm)
    half_height = inputs_dict['fuel_height'] * 50  # Convert to cm
    if inputs_dict['assembly_type'] == 'Pin':
        cell_width = inputs_dict['pin_pitch'] * inputs_dict['n_side_pins'] * 100  # Convert to cm
    else:
        cell_width = (2 * inputs_dict['clad_structure_width'] + inputs_dict['fuel_plate_width']) * 100  # Convert to cm

    # Create single energy group filter (0 to 20 MeV) ONCE - REUSE for all positions
    energy_filter = openmc.EnergyFilter([0.0, 20.0e6])  # Single group for total flux
    energy_filter.id = generate_filter_id('energy', component='axial')
    print(f"Created axial energy filter - ID: {energy_filter.id}, reused for all irradiation positions")


    core_layout = inputs_dict['core_lattice']
    for i, row in enumerate(core_layout):
        for j, pos in enumerate(row):
            if pos.startswith('I_'):
                # Parse irradiation type to determine mesh type
                irradiation_type = parse_irradiation_type(pos, inputs_dict)

                # Calculate position in core (in cm) - center of assembly
                x_pos = (j - len(row)/2 + 0.5) * cell_width
                y_pos = (i - len(core_layout)/2 + 0.5) * cell_width

                # Create mesh based on irradiation type and complexity
                if irradiation_type in ['PWR_loop', 'BWR_loop', 'Gas_capsule']:
                    # Check if this is Complex Gas_capsule (needs special annular mesh)
                    if (inputs_dict.get('irradiation_cell_complexity', 'Simple') == 'Complex' and
                        irradiation_type == 'Gas_capsule'):
                        # Create annular cylindrical mesh for sample region
                        mesh = _create_sigma_sample_mesh(
                            pos, x_pos, y_pos, cell_width, half_height, n_axial_segments, inputs_dict
                        )
                    elif (inputs_dict.get('irradiation_cell_complexity', 'Simple') == 'Complex' and
                          irradiation_type in ['PWR_loop', 'BWR_loop']):
                        # Create annular cylindrical mesh for HTWL 12 o'clock sample
                        mesh = _create_htwl_sample_mesh(
                            pos, irradiation_type, x_pos, y_pos, cell_width,
                            half_height, n_axial_segments, inputs_dict
                        )
                    else:
                        # Create regular cylindrical mesh for loop-type positions
                        mesh = _create_cylindrical_irradiation_mesh(
                            pos, irradiation_type, x_pos, y_pos,
                            cell_width, half_height, n_axial_segments, inputs_dict
                        )
                else:
                    # Create regular (square) mesh for other positions
                    mesh = _create_regular_irradiation_mesh(
                        pos, x_pos, y_pos, cell_width, half_height,
                        n_axial_segments, inputs_dict
                    )

                # Create mesh filter and tally
                mesh_filter = openmc.MeshFilter(mesh)
                mesh_filter.id = generate_filter_id('mesh', position=(i, j), component='axial')

                # REUSE the single energy filter created above (no longer inside loop)
                tally = openmc.Tally(name=f"{pos}_axial")
                tally.filters = [mesh_filter, energy_filter]
                tally.scores = ['flux']
                tallies.append(tally)

    return tallies

def _create_sigma_sample_mesh(pos, x_pos, y_pos, cell_width, half_height, n_axial_segments, inputs_dict):
    """Create annular cylindrical mesh for SIGMA sample region.

    This creates a mesh that covers only the sample annular region (3rd annular circle)
    between the inner graphite and outer graphite regions.

    Parameters
    ----------
    pos : str
        Position name (e.g., 'I_4G')
    x_pos, y_pos : float
        Center position in cm
    cell_width : float
        Assembly cell width in cm
    half_height : float
        Half height of fuel region in cm
    n_axial_segments : int
        Number of axial segments
    inputs_dict : dict
        Inputs dictionary

    Returns
    -------
    openmc.CylindricalMesh
        Annular cylindrical mesh for the tungsten region
    """
    # Use same scaling as SIGMA geometry in build_complex_sigma
    diameter_fraction = inputs_dict['Gas_capsule_diameter']
    target_diameter = cell_width * diameter_fraction

    # Get scaled radii from centralized configuration
    scaled_radii, scale_factor = get_scaled_radii('Gas_capsule', target_diameter)
    r_sample_inner = scaled_radii['sample_inner']  # Inner radius of sample annulus
    r_sample_outer = scaled_radii['sample_outer']  # Outer radius of sample annulus

    # Create annular cylindrical mesh
    # r_grid: from inner to outer sample radius
    r_grid = [r_sample_inner, r_sample_outer]

        # z_grid: axial segments
    if inputs_dict.get('match_GS_height', False):
        # Height matching enabled - use same proportional scaling as HTWL
        z_ref_bottom, z_ref_top, ref_height = get_reference_axial_bounds(inputs_dict)

        # Use same proportional scaling logic as HTWL
        fuel_height = inputs_dict['fuel_height'] * 100  # Convert to cm
        axial_resolution = fuel_height / n_axial_segments  # cm per segment (same as HTWL)
        n_sample_segments = int(ref_height / axial_resolution)  # Proportional segments

        z_grid = [z_ref_bottom + i * ref_height/n_sample_segments for i in range(n_sample_segments + 1)]
        print(f"Height matching enabled: using z-range {z_ref_bottom:.1f} to {z_ref_top:.1f} cm")
        print(f"Gas sample mesh: fuel_height={fuel_height}cm with {n_axial_segments} segments "
              f"→ sample_height={ref_height}cm with {n_sample_segments} segments "
              f"(resolution={axial_resolution:.2f}cm/segment)")
    else:
        # Standard full height
        z_grid = [-half_height + i * (2*half_height)/n_axial_segments for i in range(n_axial_segments + 1)]

    # phi_grid: single azimuthal bin (full 360°)
    phi_grid = [0.0, 2*np.pi]  # 0 to 2π

    # Create mesh with origin at the assembly center
    mesh = openmc.CylindricalMesh(
        r_grid=r_grid,
        z_grid=z_grid,
        phi_grid=phi_grid,
        origin=[x_pos, y_pos, 0.0]  # Center at assembly position
    )

    print(f"Created SIGMA sample annular mesh for {pos}: "
          f"r_inner={r_sample_inner:.2f}cm, r_outer={r_sample_outer:.2f}cm, "
          f"origin=({x_pos:.1f}, {y_pos:.1f}, 0.0)")

    return mesh

def _create_htwl_sample_mesh(pos, irradiation_type, x_pos, y_pos, cell_width,
                             half_height, n_axial_segments, inputs_dict):
    """Create annular cylindrical mesh for HTWL 12 o'clock sample region.

    This creates a mesh that covers only the sample annular region of the 12 o'clock
    position, with proper vertical dimensions and adjusted axial resolution.

    Parameters
    ----------
    pos : str
        Position name (e.g., 'I_1P', 'I_2B')
    irradiation_type : str
        Type of irradiation ('PWR_loop' or 'BWR_loop')
    x_pos, y_pos : float
        Center position of the assembly in cm
    cell_width : float
        Assembly cell width in cm
    half_height : float
        Half height of fuel region in cm (not used for HTWL)
    n_axial_segments : int
        Number of axial segments for standard fuel height
    inputs_dict : dict
        Inputs dictionary

    Returns
    -------
    openmc.CylindricalMesh
        Annular cylindrical mesh for the HTWL sample region
    """
    # Use same scaling as HTWL geometry
    if irradiation_type == 'BWR_loop':
        diameter_fraction = inputs_dict['BWR_loop_diameter']
    else:  # PWR_loop
        diameter_fraction = inputs_dict['PWR_loop_diameter']

    target_diameter = cell_width * diameter_fraction

    # Get scaled radii from centralized configuration
    scaled_radii, scale_factor = get_scaled_radii(irradiation_type, target_diameter)
    r_sample_inner = scaled_radii['sample_inner']  # Inner radius of sample annulus
    r_sample_outer = scaled_radii['sample_outer']   # Outer radius of sample annulus
    r_sample_center = scaled_radii['sample_center'] # Distance from center to sample center

    # HTWL sample vertical dimensions (from centralized configuration)
    config = get_experiment_config(irradiation_type)
    z_sample_bottom = config['z_planes']['capsule_bottom_top']  # cm
    z_sample_top = config['z_planes']['capsule_top_bot']        # cm
    sample_height = z_sample_top - z_sample_bottom              # cm

    # Calculate adjusted number of axial segments to maintain resolution
    # If fuel height has n_axial_segments, and HTWL sample is shorter,
    # we want proportionally fewer segments to maintain same resolution
    fuel_height = inputs_dict['fuel_height'] * 100  # Convert to cm
    axial_resolution = fuel_height / n_axial_segments  # cm per segment
    n_sample_segments = int(sample_height / axial_resolution)

    print(f"HTWL sample mesh: fuel_height={fuel_height}cm with {n_axial_segments} segments "
          f"→ sample_height={sample_height}cm with {n_sample_segments} segments "
          f"(resolution={axial_resolution:.2f}cm/segment)")

    # Create annular cylindrical mesh
    # r_grid: annular region from inner to outer sample radius
    r_grid = [r_sample_inner, r_sample_outer]

    # z_grid: axial segments from sample bottom to top
    z_grid = [z_sample_bottom + i * sample_height/n_sample_segments
              for i in range(n_sample_segments + 1)]

    # phi_grid: single azimuthal bin (full 360°)
    phi_grid = [0.0, 2*np.pi]  # 0 to 2π

    # Create mesh with origin offset to the 12 o'clock sample position
    # 12 o'clock sample is at (0, r_sample_center) relative to assembly center
    mesh = openmc.CylindricalMesh(
        r_grid=r_grid,
        z_grid=z_grid,
        phi_grid=phi_grid,
        origin=[x_pos, y_pos + r_sample_center, 0.0]  # Offset to 12 o'clock position
    )

    print(f"Created HTWL sample annular mesh for {pos} ({irradiation_type}): "
          f"r_inner={r_sample_inner:.3f}cm, r_outer={r_sample_outer:.3f}cm, "
          f"z_range=[{z_sample_bottom}, {z_sample_top}]cm, "
          f"origin=({x_pos:.1f}, {y_pos + r_sample_center:.1f}, 0.0)")

    return mesh

def _create_cylindrical_irradiation_mesh(pos, irradiation_type, x_pos, y_pos,
                                        cell_width, half_height, n_axial_segments, inputs_dict):
    """Create a cylindrical mesh for loop-type irradiation positions.

    Parameters
    ----------
    pos : str
        Position name (e.g., 'I_1P', 'I_2B', 'I_3G')
    irradiation_type : str
        Type of irradiation ('PWR_loop', 'BWR_loop', 'Gas_capsule')
    x_pos, y_pos : float
        Center position in cm
    cell_width : float
        Assembly cell width in cm
    half_height : float
        Half height of fuel region in cm
    n_axial_segments : int
        Number of axial segments
    inputs_dict : dict
        Inputs dictionary

    Returns
    -------
    openmc.CylindricalMesh
        Cylindrical mesh for the irradiation position
    """
    # Get radius from diameter parameter (same calculation as in geometry)
    if irradiation_type == 'PWR_loop':
        circle_radius = inputs_dict['PWR_loop_diameter']/2 * cell_width
    elif irradiation_type == 'BWR_loop':
        circle_radius = inputs_dict['BWR_loop_diameter']/2 * cell_width
    elif irradiation_type == 'Gas_capsule':
        circle_radius = inputs_dict['Gas_capsule_diameter']/2 * cell_width
    else:
        raise ValueError(f"Unknown irradiation type: {irradiation_type}")

    # NOTE: Cladding does NOT affect cylinder radius - only square boundary planes
    # The cylinder radius stays the same regardless of cladding setting
    # (cladding only affects the square constraints in the actual geometry)

    # Create cylindrical mesh
    # r_grid: from center (0) to cavity radius
    r_grid = [0.0, circle_radius]

    # z_grid: axial segments from -half_height to +half_height
    z_grid = [-half_height + i * (2*half_height)/n_axial_segments for i in range(n_axial_segments + 1)]

    # phi_grid: single azimuthal bin (full 360°)
    phi_grid = [0.0, 2*np.pi]  # 0 to 2π

    # Create mesh with origin at the assembly center
    mesh = openmc.CylindricalMesh(
        r_grid=r_grid,
        z_grid=z_grid,
        phi_grid=phi_grid,
        origin=[x_pos, y_pos, 0.0]  # Center at assembly position
    )

    print(f"Created cylindrical mesh for {pos} ({irradiation_type}): "
          f"radius={circle_radius:.2f}cm, origin=({x_pos:.1f}, {y_pos:.1f}, 0.0)")

    return mesh

def _create_regular_irradiation_mesh(pos, x_pos, y_pos, cell_width, half_height,
                                   n_axial_segments, inputs_dict):
    """Create a regular (square) mesh for non-loop irradiation positions.

    Parameters
    ----------
    pos : str
        Position name (e.g., 'I_1', 'I_2')
    x_pos, y_pos : float
        Center position in cm
    cell_width : float
        Assembly cell width in cm
    half_height : float
        Half height of fuel region in cm
    n_axial_segments : int
        Number of axial segments
    inputs_dict : dict
        Inputs dictionary

    Returns
    -------
    openmc.RegularMesh
        Regular mesh for the irradiation position
    """
    # Calculate mesh width (subtract cladding if present)
    width = cell_width
    if inputs_dict.get('irradiation_clad', False):
        clad_thickness = inputs_dict['irradiation_clad_thickness'] * 100  # Convert m to cm
        width = width - (2 * clad_thickness)  # Subtract cladding from both sides

    # Create regular mesh
    mesh = openmc.RegularMesh()
    mesh.dimension = [1, 1, n_axial_segments]  # Single radial cell, multiple axial segments

    # Set mesh boundaries
    half_width = width/2
    mesh.lower_left = [x_pos - half_width, y_pos - half_width, -half_height]
    mesh.upper_right = [x_pos + half_width, y_pos + half_width, half_height]

    print(f"Created regular mesh for {pos}: "
          f"width={width:.2f}cm, center=({x_pos:.1f}, {y_pos:.1f})")

    return mesh
