import openmc
import numpy as np
import os
import sys

# Add root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from inputs import inputs
from .utils import generate_cell_id

def build_plate_cell_fuel_uni(mat_dict, is_enhanced=False):
    """Build a single fuel plate universe with proper bounds"""
    # Calculate total thickness from inputs
    fuel_thickness = inputs['fuel_meat_thickness']*100
    clad_thickness = inputs['clad_thickness']*100
    total_thickness = fuel_thickness + 2*clad_thickness
    coolant_channel = inputs['fuel_plate_pitch']*100 - total_thickness  # Add this

    # Define plate dimensions
    fuel_plate_width = inputs['fuel_plate_width']*100
    fuel_meat_width = inputs['fuel_meat_width']*100

    # Map clad type to material name
    clad_material_map = {
        'Zirc2': 'Zircaloy',
        'Zirc4': 'Zircaloy',
        'Al6061': 'Al6061'
    }
    base_clad_material = clad_material_map[inputs['clad_type']]
    clad_material = f"{base_clad_material}-Enhanced" if is_enhanced else base_clad_material
    fuel_name = f"{inputs['fuel_type']}-Enhanced" if is_enhanced else inputs['fuel_type']

    # Define all boundaries explicitly
    x0 = -fuel_plate_width/2
    x1 = -fuel_meat_width/2
    x2 = fuel_meat_width/2
    x3 = fuel_plate_width/2
    y0 = -total_thickness/2
    y1 = y0 + clad_thickness
    y2 = y1 + fuel_thickness
    y3 = y2 + clad_thickness

    # Add coolant channel bounds
    y_top = y3 + coolant_channel/2
    y_bottom = y0 - coolant_channel/2

    # Create all the planes with explicit bounds
    x0p = openmc.XPlane(x0)
    x1p = openmc.XPlane(x1)
    x2p = openmc.XPlane(x2)
    x3p = openmc.XPlane(x3)
    y0p = openmc.YPlane(y0)
    y1p = openmc.YPlane(y1)
    y2p = openmc.YPlane(y2)
    y3p = openmc.YPlane(y3)
    y_top_p = openmc.YPlane(y_top)
    y_bottom_p = openmc.YPlane(y_bottom)

    # Define all cells with complete bounds
    cells = []

    # Left clad cell
    left_clad = openmc.Cell(fill=mat_dict[clad_material],
                           region=+x0p & -x1p & +y0p & -y3p)
    cells.append(left_clad)

    # Bottom clad cell
    bottom_clad = openmc.Cell(fill=mat_dict[clad_material],
                             region=+x1p & -x2p & +y0p & -y1p)
    cells.append(bottom_clad)

    # Fuel cell
    fuel = openmc.Cell(fill=mat_dict[fuel_name],
                      region=+x1p & -x2p & +y1p & -y2p)
    cells.append(fuel)

    # Top clad cell
    top_clad = openmc.Cell(fill=mat_dict[clad_material],
                          region=+x1p & -x2p & +y2p & -y3p)
    cells.append(top_clad)

    # Right clad cell
    right_clad = openmc.Cell(fill=mat_dict[clad_material],
                            region=+x2p & -x3p & +y0p & -y3p)
    cells.append(right_clad)

    # Top coolant channel with explicit bounds
    top_moderator = openmc.Cell(fill=mat_dict[f"{inputs['coolant_type']} Coolant"],
                               region=+x0p & -x3p & +y3p & -y_top_p)
    cells.append(top_moderator)

    # Bottom coolant channel with explicit bounds
    bottom_moderator = openmc.Cell(fill=mat_dict[f"{inputs['coolant_type']} Coolant"],
                                  region=+x0p & -x3p & +y_bottom_p & -y0p)
    cells.append(bottom_moderator)

    # Create universe with all bounded cells
    fuel_plate_universe = openmc.Universe(cells=cells)
    return fuel_plate_universe

def build_fuel_assembly_uni(mat_dict, position=None, is_enhanced=False):
    """Build a fuel plate assembly universe with proper cell definitions"""
    # Get dimensions from inputs
    fuel_plate_pitch = inputs['fuel_plate_pitch']*100
    plates_per_assembly = inputs['plates_per_assembly']
    fuel_plate_width = inputs['fuel_plate_width']*100
    assembly_side_width = inputs['clad_structure_width']*100

    # Map clad type to material
    clad_material_map = {
        'Zirc2': 'Zircaloy',
        'Zirc4': 'Zircaloy',
        'Al6061': 'Al6061'
    }
    clad_material = clad_material_map[inputs['clad_type']]

    # Calculate assembly dimensions
    fuel_region_width = plates_per_assembly * fuel_plate_pitch
    assembly_width = fuel_region_width + 2 * assembly_side_width

    # Define assembly boundaries
    x0 = -assembly_width/2
    x1 = x0 + assembly_side_width
    x3 = assembly_width/2
    x2 = x3 - assembly_side_width
    y0 = -assembly_width/2
    y1 = y0 + assembly_side_width
    y3 = assembly_width/2
    y2 = y3 - assembly_side_width

    # Create boundary planes
    x0p = openmc.XPlane(x0)
    x1p = openmc.XPlane(x1)
    x2p = openmc.XPlane(x2)
    x3p = openmc.XPlane(x3)
    y0p = openmc.YPlane(y0)
    y1p = openmc.YPlane(y1)
    y2p = openmc.YPlane(y2)
    y3p = openmc.YPlane(y3)

    # Create assembly cells with explicit bounds
    cells = []

    # Side structure plates
    left_structure = openmc.Cell(fill=mat_dict[clad_material],
                                region=+x0p & -x1p & +y0p & -y3p)
    cells.append(left_structure)

    right_structure = openmc.Cell(fill=mat_dict[clad_material],
                                 region=+x2p & -x3p & +y0p & -y3p)
    cells.append(right_structure)

    top_structure = openmc.Cell(fill=mat_dict[clad_material],
                               region=+x1p & -x2p & +y2p & -y3p)
    cells.append(top_structure)

    bottom_structure = openmc.Cell(fill=mat_dict[clad_material],
                                  region=+x1p & -x2p & +y0p & -y1p)
    cells.append(bottom_structure)

    # Build fuel plate universe
    fuel = build_plate_cell_fuel_uni(mat_dict, is_enhanced)

    # Create fuel plate lattice
    yoffset = (assembly_width - fuel_plate_pitch * plates_per_assembly) / 2
    fuel_plates = openmc.RectLattice()
    fuel_plates.lower_left = [-fuel_plate_width/2.0, -assembly_width/2.0 + yoffset]
    fuel_plates.pitch = [fuel_plate_width, fuel_plate_pitch]

    # Create coolant universe for outer region
    coolant_cell = openmc.Cell(fill=mat_dict[f"{inputs['coolant_type']} Coolant"])
    coolant_universe = openmc.Universe(cells=[coolant_cell])
    fuel_plates.outer = coolant_universe

    # Set up fuel plate array
    fuel_plates.universes = np.tile(fuel, (plates_per_assembly, 1))

    # Create fuel region cell with proper bounds
    fuel_cell = openmc.Cell(name='fuel_plate_cell')
    if position is not None:
        fuel_cell.id = generate_cell_id('fuel', position, is_enhanced)
        fuel_cell.name = f"{'enhanced_' if is_enhanced else ''}fuel_plate_cell_{position[0]}_{position[1]}"
    fuel_cell.fill = fuel_plates
    # Add explicit bounds for the fuel region
    fuel_cell.region = (+x1p & -x2p & +y1p & -y2p) & ~(+x0p & -x1p & +y0p & -y3p) & ~(+x2p & -x3p & +y0p & -y3p)
    cells.append(fuel_cell)

    # Create assembly universe with all cells
    assembly_universe = openmc.Universe(cells=cells)
    return assembly_universe
