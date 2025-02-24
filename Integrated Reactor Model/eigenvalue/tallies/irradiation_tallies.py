"""
Functions for creating tallies in irradiation positions.

This module provides functions for:
- Energy-dependent flux tallies in irradiation positions
- Axial flux distribution tallies in irradiation positions
"""

import openmc
from inputs import inputs
from Reactor.geometry_helpers.utils import generate_cell_id
from eigenvalue.tallies.energy_groups import get_energy_bins

def create_irradiation_tallies():
    """Create tallies for irradiation positions."""
    tallies = openmc.Tallies()

    # Get energy group structure
    energy_bins = get_energy_bins()
    energy_filter = openmc.EnergyFilter(energy_bins)

    # Create tallies for each irradiation position in the core
    core_layout = inputs['core_lattice']
    for i, row in enumerate(core_layout):
        for j, pos in enumerate(row):
            if pos.startswith('I_'):  # This is an irradiation position
                # Create cell filter for this position
                cell_id = generate_cell_id('irradiation', (i, j))
                cell_filter = openmc.CellFilter([cell_id])

                # Create tally for this position
                tally = openmc.Tally(name=pos)
                tally.filters = [cell_filter, energy_filter]
                tally.scores = ['flux']
                tallies.append(tally)

    return tallies

def create_irradiation_axial_tallies(n_axial_segments=50):
    """Create axial flux tallies for irradiation positions.

    This creates a mesh tally for each irradiation position that divides
    the position into axial segments to measure flux variation with height.
    Uses a single energy group for total flux.

    Parameters
    ----------
    n_axial_segments : int, optional
        Number of axial segments to divide each position into. Default is 50.

    Returns
    -------
    openmc.Tallies
        Collection of axial flux tallies for each irradiation position
    """
    tallies = openmc.Tallies()

    # Get core dimensions from inputs (in cm)
    half_height = inputs['fuel_height'] * 50  # Convert to cm
    if inputs['assembly_type'] == 'Pin':
        width = inputs['pin_pitch'] * inputs['n_side_pins'] * 100  # Convert to cm
    else:
        width = (inputs['fuel_plate_width'] + 2 * inputs['clad_structure_width']) * 100  # Convert to cm

    # Subtract cladding thickness if present
    if inputs.get('irradiation_clad', False):
        clad_thickness = inputs['irradiation_clad_thickness'] * 100  # Convert to cm
        width = width - (2 * clad_thickness)  # Subtract cladding from both sides

    # Create tallies for each irradiation position
    core_layout = inputs['core_lattice']
    for i, row in enumerate(core_layout):
        for j, pos in enumerate(row):
            if pos.startswith('I_'):
                # Create a mesh for this position
                mesh = openmc.RegularMesh()
                mesh.dimension = [1, 1, n_axial_segments]  # Single radial cell, multiple axial segments

                # Calculate position in core (in cm)
                x_pos = (j - len(row)/2 + 0.5) * width
                y_pos = (i - len(core_layout)/2 + 0.5) * width

                # Set mesh boundaries
                half_width = width/2
                mesh.lower_left = [x_pos - half_width, y_pos - half_width, -half_height]
                mesh.upper_right = [x_pos + half_width, y_pos + half_width, half_height]

                # Create mesh filter and tally
                mesh_filter = openmc.MeshFilter(mesh)

                # Create single energy group filter (0 to 20 MeV)
                energy_filter = openmc.EnergyFilter([0.0, 20.0e6])  # Single group for total flux

                tally = openmc.Tally(name=f"{pos}_axial")
                tally.filters = [mesh_filter, energy_filter]
                tally.scores = ['flux']
                tallies.append(tally)

    return tallies
