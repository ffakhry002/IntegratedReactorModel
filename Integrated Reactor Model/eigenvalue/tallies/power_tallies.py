"""
Functions for creating power tallies.

This module provides functions for:
- Total core power tally
- Assembly-wise power tallies with axial distributions
- Fuel element-wise power tallies with axial distributions
"""

import openmc
import numpy as np
from inputs import inputs
from Reactor.geometry_helpers.utils import generate_cell_id

def create_power_tallies(n_axial_segments=200):
    """Create power tallies for the entire core and individual assemblies or fuel elements.

    Parameters
    ----------
    n_axial_segments : int, optional
        Number of axial segments for power distributions. Default is 200.

    Returns
    -------
    openmc.Tallies
        Collection of power tallies
    """
    tallies = openmc.Tallies()

    # Get core dimensions from inputs (in cm)
    half_height = inputs['fuel_height'] * 50  # Convert to cm

    # Calculate assembly width
    if inputs['assembly_type'] == 'Pin':
        assembly_width = inputs['pin_pitch'] * inputs['n_side_pins'] * 100  # Convert to cm
    else:
        assembly_width = (inputs['fuel_plate_width'] + 2 * inputs['clad_structure_width']) * 100  # Convert to cm

    # Create total core power tally
    total_power_tally = openmc.Tally(name='total_power')
    total_power_tally.scores = ['kappa-fission']
    tallies.append(total_power_tally)

    # Check if we're doing element-level or assembly-level tallying
    if inputs.get('element_level_power_tallies', False):
        # Element-level tallying
        if inputs['assembly_type'] == 'Pin':
            # Pin-type fuel elements
            create_pin_element_tallies(tallies, n_axial_segments, half_height, assembly_width)
        else:
            # Plate-type fuel elements
            create_plate_element_tallies(tallies, n_axial_segments, half_height, assembly_width)
    else:
        # Assembly-level tallying (original implementation)
        create_assembly_tallies(tallies, n_axial_segments, half_height, assembly_width)

    return tallies

def create_assembly_tallies(tallies, n_axial_segments, half_height, assembly_width):
    """Create assembly-level power tallies.

    Parameters
    ----------
    tallies : openmc.Tallies
        Collection of tallies to add to
    n_axial_segments : int
        Number of axial segments
    half_height : float
        Half-height of the core in cm
    assembly_width : float
        Width of an assembly in cm
    """
    # Create assembly-wise power tallies
    core_layout = inputs['core_lattice']
    for i, row in enumerate(core_layout):
        for j, pos in enumerate(row):
            if pos in ['F', 'E']:  # This is a fuel assembly position
                # Create a mesh for this assembly
                mesh = openmc.RegularMesh()
                mesh.dimension = [1, 1, n_axial_segments]  # Single radial cell, multiple axial segments

                # Calculate position in core (in cm)
                x_pos = (j - len(row)/2 + 0.5) * assembly_width
                y_pos = (i - len(core_layout)/2 + 0.5) * assembly_width

                # Set mesh boundaries
                half_width = assembly_width/2
                mesh.lower_left = [x_pos - half_width, y_pos - half_width, -half_height]
                mesh.upper_right = [x_pos + half_width, y_pos + half_width, half_height]

                # Create mesh filter
                mesh_filter = openmc.MeshFilter(mesh)

                # Create tally for this assembly
                tally = openmc.Tally(name=f"assembly_power_{i}_{j}")
                tally.filters = [mesh_filter]
                tally.scores = ['kappa-fission']
                tallies.append(tally)

def create_pin_element_tallies(tallies, n_axial_segments, half_height, assembly_width):
    """Create pin-level power tallies.

    Parameters
    ----------
    tallies : openmc.Tallies
        Collection of tallies to add to
    n_axial_segments : int
        Number of axial segments
    half_height : float
        Half-height of the core in cm
    assembly_width : float
        Width of an assembly in cm
    """
    # Get pin dimensions
    pin_pitch = inputs['pin_pitch'] * 100  # Convert to cm
    n_side_pins = inputs['n_side_pins']
    guide_tube_positions = inputs['guide_tube_positions']

    # Create pin-wise power tallies
    core_layout = inputs['core_lattice']
    for i, row in enumerate(core_layout):
        for j, pos in enumerate(row):
            if pos in ['F', 'E']:  # This is a fuel assembly position
                # Calculate assembly position in core (in cm)
                assembly_x = (j - len(row)/2 + 0.5) * assembly_width
                assembly_y = (i - len(core_layout)/2 + 0.5) * assembly_width

                # Calculate assembly boundaries
                assembly_half_width = assembly_width/2
                assembly_left = assembly_x - assembly_half_width
                assembly_bottom = assembly_y - assembly_half_width

                # Loop through each pin position in the assembly
                for pin_i in range(n_side_pins):
                    for pin_j in range(n_side_pins):
                        # Skip guide tube positions
                        if (pin_i, pin_j) in guide_tube_positions:
                            continue

                        # Calculate pin position within assembly (in cm)
                        pin_x = assembly_left + (pin_j + 0.5) * pin_pitch
                        pin_y = assembly_bottom + (pin_i + 0.5) * pin_pitch

                        # Create a mesh for this pin
                        mesh = openmc.RegularMesh()
                        mesh.dimension = [1, 1, n_axial_segments]  # Single radial cell, multiple axial segments

                        # Set mesh boundaries
                        half_pin_pitch = pin_pitch/2
                        mesh.lower_left = [pin_x - half_pin_pitch, pin_y - half_pin_pitch, -half_height]
                        mesh.upper_right = [pin_x + half_pin_pitch, pin_y + half_pin_pitch, half_height]

                        # Create mesh filter
                        mesh_filter = openmc.MeshFilter(mesh)

                        # Create tally for this pin
                        tally = openmc.Tally(name=f"pin_power_{i}_{j}_{pin_i}_{pin_j}")
                        tally.filters = [mesh_filter]
                        tally.scores = ['kappa-fission']
                        tallies.append(tally)

def create_plate_element_tallies(tallies, n_axial_segments, half_height, assembly_width):
    """Create plate-level power tallies.

    Parameters
    ----------
    tallies : openmc.Tallies
        Collection of tallies to add to
    n_axial_segments : int
        Number of axial segments
    half_height : float
        Half-height of the core in cm
    assembly_width : float
        Width of an assembly in cm
    """
    # Get plate dimensions
    plates_per_assembly = inputs['plates_per_assembly']
    fuel_plate_pitch = inputs['fuel_plate_pitch'] * 100  # Convert to cm
    fuel_plate_width = inputs['fuel_plate_width'] * 100  # Convert to cm
    clad_structure_width = inputs['clad_structure_width'] * 100  # Convert to cm

    # Create plate-wise power tallies
    core_layout = inputs['core_lattice']
    for i, row in enumerate(core_layout):
        for j, pos in enumerate(row):
            if pos in ['F', 'E']:  # This is a fuel assembly position
                # Calculate assembly position in core (in cm)
                assembly_x = (j - len(row)/2 + 0.5) * assembly_width
                assembly_y = (i - len(core_layout)/2 + 0.5) * assembly_width

                # Calculate assembly boundaries
                assembly_half_width = assembly_width/2
                assembly_left = assembly_x - assembly_half_width
                assembly_bottom = assembly_y - assembly_half_width

                # Calculate the starting position for plates (accounting for structure width)
                plate_region_start = assembly_bottom + clad_structure_width

                # Loop through each plate in the assembly
                for plate_k in range(plates_per_assembly):
                    # Calculate plate position within assembly (in cm)
                    # Plates are stacked in the y-direction
                    plate_y = plate_region_start + (plate_k + 0.5) * fuel_plate_pitch

                    # Create a mesh for this plate
                    mesh = openmc.RegularMesh()
                    mesh.dimension = [1, 1, n_axial_segments]  # Single radial cell, multiple axial segments

                    # Set mesh boundaries
                    half_plate_thickness = fuel_plate_pitch/2
                    mesh.lower_left = [assembly_left, plate_y - half_plate_thickness, -half_height]
                    mesh.upper_right = [assembly_left + fuel_plate_width, plate_y + half_plate_thickness, half_height]

                    # Create mesh filter
                    mesh_filter = openmc.MeshFilter(mesh)

                    # Create tally for this plate
                    tally = openmc.Tally(name=f"plate_power_{i}_{j}_{plate_k}")
                    tally.filters = [mesh_filter]
                    tally.scores = ['kappa-fission']
                    tallies.append(tally)
