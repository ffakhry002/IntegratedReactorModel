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

from inputs import inputs

def build_irradiation_cell_uni(mat_dict, position=None):
    """Build an irradiation cell universe.

    Parameters
    ----------
    mat_dict : dict
        Dictionary of materials
    position : tuple, optional
        (i, j) position in core lattice. If provided, assigns unique ID.

    Returns
    -------
    openmc.Universe
        Universe containing the irradiation cell
    """

    # Calculate cell dimensions based on assembly type
    if inputs['assembly_type'] == 'Pin':
        cell_width = inputs['pin_pitch'] * inputs['n_side_pins'] * 100  # m to cm
    else:  # Plate assembly
        cell_width = (2 * inputs['clad_structure_width'] + inputs['fuel_plate_width']) * 100  # m to cm

    # Define the dimensions
    x0 = -cell_width/2
    x3 = cell_width/2
    y0 = -cell_width/2
    y3 = cell_width/2

    if inputs['irradiation_clad']:
        # Calculate inner dimensions with cladding
        clad_thickness = inputs['irradiation_clad_thickness'] * 100  # m to cm
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
        clad_material = clad_material_map[inputs['clad_type']]

        # Create cells with cladding - ensuring no overlaps
        # Define regions for each part of the irradiation cell
        # Each region is explicitly defined to avoid any possible overlap

        # Center region (between all cladding)
        center_region = +x1p & -x2p & +y1p & -y2p

        # Bottom cladding (full width at bottom)
        bottom_clad_region = +x0p & -x3p & +y0p & -y1p

        # Top cladding (full width at top)
        top_clad_region = +x0p & -x3p & +y2p & -y3p

        # Left cladding (between top and bottom)
        left_clad_region = +x0p & -x1p & +y1p & -y2p

        # Right cladding (between top and bottom)
        right_clad_region = +x2p & -x3p & +y1p & -y2p

        # Create cells in order from inside to outside
        # Center cell first
        center_cell = openmc.Cell(name='irradiation_center')
        center_cell.region = center_region
        if position is not None:
            center_cell.id = generate_cell_id('irradiation', position)  # Center cell (type 3)
            center_cell.name = get_irradiation_cell_name(position, inputs['core_lattice'])
        center_cell.fill = mat_dict['Test pos'] if inputs['irradiation_cell_fill'] == 'fill' \
                          else mat_dict['Vacuum']

        # Create the cladding cells
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
            # Each clad part gets its own type number (4-7)
            bottom_clad.id = generate_cell_id('irradiation', position, clad_part='bottom')
            top_clad.id = generate_cell_id('irradiation', position, clad_part='top')
            left_clad.id = generate_cell_id('irradiation', position, clad_part='left')
            right_clad.id = generate_cell_id('irradiation', position, clad_part='right')

            # Also assign names
            pos_name = get_irradiation_cell_name(position, inputs['core_lattice'])
            bottom_clad.name = f"{pos_name}_bottom_clad"
            top_clad.name = f"{pos_name}_top_clad"
            left_clad.name = f"{pos_name}_left_clad"
            right_clad.name = f"{pos_name}_right_clad"

        # Create universe with all cells - order matters for proper nesting
        # Put center cell first, then cladding cells
        cells = [
            center_cell,
            bottom_clad,
            top_clad,
            left_clad,
            right_clad
        ]
        irradiation_universe = openmc.Universe(name='irradiation_universe', cells=cells)

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
            main_cell.name = get_irradiation_cell_name(position, inputs['core_lattice'])

        main_cell.fill = mat_dict['Test pos'] if inputs['irradiation_cell_fill'] == 'fill' \
                        else mat_dict['Vacuum']

        irradiation_universe = openmc.Universe(name='irradiation_universe', cells=[main_cell])

    return irradiation_universe
