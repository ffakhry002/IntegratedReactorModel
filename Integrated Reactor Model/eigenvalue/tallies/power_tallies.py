"""
Functions for creating power tallies.

This module provides functions for:
- Total core power tally
- Assembly-wise power tallies with axial distributions
"""

import openmc
from inputs import inputs
from Reactor.geometry_helpers.utils import generate_cell_id

def create_power_tallies(n_axial_segments=200):
    """Create power tallies for the entire core and individual assemblies.

    Parameters
    ----------
    n_axial_segments : int, optional
        Number of axial segments for assembly power distributions. Default is 50.

    Returns
    -------
    openmc.Tallies
        Collection of power tallies
    """
    tallies = openmc.Tallies()

    # Get core dimensions from inputs (in cm)
    half_height = inputs['fuel_height'] * 50  # Convert to cm
    if inputs['assembly_type'] == 'Pin':
        width = inputs['pin_pitch'] * inputs['n_side_pins'] * 100  # Convert to cm
    else:
        width = (inputs['fuel_plate_width'] + 2 * inputs['clad_structure_width']) * 100  # Convert to cm

    # Create total core power tally
    total_power_tally = openmc.Tally(name='total_power')
    total_power_tally.scores = ['kappa-fission']
    tallies.append(total_power_tally)

    # Create assembly-wise power tallies
    core_layout = inputs['core_lattice']
    for i, row in enumerate(core_layout):
        for j, pos in enumerate(row):
            if pos in ['F', 'E']:  # This is a fuel assembly position
                # Create a mesh for this assembly
                mesh = openmc.RegularMesh()
                mesh.dimension = [1, 1, n_axial_segments]  # Single radial cell, multiple axial segments

                # Calculate position in core (in cm)
                x_pos = (j - len(row)/2 + 0.5) * width
                y_pos = (i - len(core_layout)/2 + 0.5) * width

                # Set mesh boundaries
                half_width = width/2
                mesh.lower_left = [x_pos - half_width, y_pos - half_width, -half_height]
                mesh.upper_right = [x_pos + half_width, y_pos + half_width, half_height]

                # Create mesh filter
                mesh_filter = openmc.MeshFilter(mesh)

                # Create tally for this assembly
                tally = openmc.Tally(name=f"assembly_power_{i}_{j}")
                tally.filters = [mesh_filter]
                tally.scores = ['kappa-fission']
                tallies.append(tally)

    return tallies
