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

    # Get irradiation type
    irradiation_type = inputs_dict.get('irradiation_type')

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
            inner_cell = openmc.Cell(name='pwr_loop_center')
            inner_cell.region = inner_region
            if position is not None:
                inner_cell.id = generate_cell_id('irradiation', position)
                inner_cell.name = get_irradiation_cell_name(position, inputs_dict['core_lattice']) + '_pwr'
            inner_cell.fill = mat_dict['PWR_loop']

            outer_cell = openmc.Cell(name='pwr_loop_outer')
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

            # Inner circle (PWR loop material)
            inner_region = -inner_circle & +x0p & -x3p & +y0p & -y3p

            # Outer annular region (Al6061)
            outer_region = +inner_circle & +x0p & -x3p & +y0p & -y3p

            # Create cells
            inner_cell = openmc.Cell(name='pwr_loop_center', region=inner_region)
            if position is not None:
                inner_cell.id = generate_cell_id('irradiation', position)
                inner_cell.name = get_irradiation_cell_name(position, inputs_dict['core_lattice']) + '_pwr'
            inner_cell.fill = mat_dict['PWR_loop']

            outer_cell = openmc.Cell(name='pwr_loop_outer', region=outer_region)
            outer_cell.fill = mat_dict['Al6061']

            cells = [inner_cell, outer_cell]

    else:
        # STANDARD SQUARE GEOMETRY (vacuum or fill)
        # Select material based on irradiation_type
        if irradiation_type == 'vacuum':
            fill_material = mat_dict['Vacuum']
        elif irradiation_type == 'fill':
            fill_material = mat_dict['Test pos']
        else:
            # Default fallback
            fill_material = mat_dict['Vacuum']

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

    irradiation_universe = openmc.Universe(name='irradiation_universe', cells=cells)
    return irradiation_universe
