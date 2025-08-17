"""
Functions for creating tallies in irradiation positions.

This module provides functions for:
- Energy-dependent flux tallies in irradiation positions
- Axial flux distribution tallies in irradiation positions
"""

import openmc
import numpy as np
from inputs import inputs
from Reactor.geometry_helpers.utils import generate_cell_id
from Reactor.geometry_helpers.irradiation_cell import parse_irradiation_type
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
    print(f"Created irradiation energy filter - ID: {energy_filter.id}, Groups: {len(energy_bins)-1}")

    # Create tallies for each irradiation position in the core
    core_layout = inputs_dict['core_lattice']
    for i, row in enumerate(core_layout):
        for j, pos in enumerate(row):
            if pos.startswith('I_'):  # This is an irradiation position
                # Create cell filter for this position
                cell_id = generate_cell_id('irradiation', (i, j))
                cell_filter = openmc.CellFilter([cell_id])
                # Cell filter created for each irradiation position

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

    # Create tallies for each irradiation position
    core_layout = inputs_dict['core_lattice']
    for i, row in enumerate(core_layout):
        for j, pos in enumerate(row):
            if pos.startswith('I_'):
                # Parse irradiation type to determine mesh type
                irradiation_type = parse_irradiation_type(pos, inputs_dict)

                # Calculate position in core (in cm) - center of assembly
                x_pos = (j - len(row)/2 + 0.5) * cell_width
                y_pos = (i - len(core_layout)/2 + 0.5) * cell_width

                # Create mesh based on irradiation type
                if irradiation_type in ['PWR_loop', 'BWR_loop', 'Gas_capsule']:
                    # Create cylindrical mesh for loop-type positions
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

                # Create single energy group filter (0 to 20 MeV)
                energy_filter = openmc.EnergyFilter([0.0, 20.0e6])  # Single group for total flux

                tally = openmc.Tally(name=f"{pos}_axial")
                tally.filters = [mesh_filter, energy_filter]
                tally.scores = ['flux']
                tallies.append(tally)

    return tallies

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

    # Subtract cladding thickness if present
    if inputs_dict.get('irradiation_clad', False):
        clad_thickness = inputs_dict['irradiation_clad_thickness'] * 100  # Convert m to cm
        # Reduce the available radius by cladding thickness impact
        available_cell_width = cell_width - (2 * clad_thickness)
        circle_radius = circle_radius * (available_cell_width / cell_width)

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
