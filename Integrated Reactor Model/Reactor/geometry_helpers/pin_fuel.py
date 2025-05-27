import openmc
import numpy as np
import os
import sys

# Add root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from utils.base_inputs import inputs
from .utils import generate_cell_id, get_irradiation_cell_name

def build_pin_cell_fuel_uni(mat_dict, is_enhanced=False, inputs_dict=None):
    """Build a fuel pin cell universe.

    Args:
        mat_dict: Dictionary of OpenMC materials
        is_enhanced: Whether this is an enhanced fuel position (with higher enrichment)
        inputs_dict: Custom inputs dictionary. If None, uses the global inputs.

    Returns:
        openmc.Universe: Universe containing the fuel pin cell
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    # Create cylinders for fuel pin regions - convert m to cm
    r1 = openmc.ZCylinder(r=inputs_dict['r_fuel'] * 100)  # m to cm
    r2 = openmc.ZCylinder(r=inputs_dict['r_clad_inner'] * 100)  # m to cm
    r3 = openmc.ZCylinder(r=inputs_dict['r_clad_outer'] * 100)  # m to cm

    # Map clad type to material name
    clad_material_map = {
        'Zirc2': 'Zircaloy',
        'Zirc4': 'Zircaloy',
        'Al6061': 'Al6061'
    }
    base_clad_material = clad_material_map[inputs_dict['clad_type']]

    # Add -Enhanced suffix for enhanced fuel positions
    clad_material = f"{base_clad_material}-Enhanced" if is_enhanced else base_clad_material

    # Get the appropriate fuel material name based on whether this is enhanced fuel
    fuel_name = f"{inputs_dict['fuel_type']}-Enhanced" if is_enhanced else inputs_dict['fuel_type']

    # Create cells
    fuel_cell = openmc.Cell(fill=mat_dict[fuel_name], region=-r1)
    he_cell = openmc.Cell(fill=mat_dict['Helium'], region=+r1 & -r2)
    zr_cell = openmc.Cell(fill=mat_dict[clad_material], region=+r2 & -r3)
    coolant_cell = openmc.Cell(fill=mat_dict[f"{inputs_dict['coolant_type']} Coolant"], region=+r3)

    # Create universe
    pin_universe = openmc.Universe(name='fuel', cells=[fuel_cell, he_cell, zr_cell, coolant_cell])
    return pin_universe

def build_guide_tube_uni(mat_dict, inputs_dict=None):
    """Build a guide tube universe.

    Args:
        mat_dict: Dictionary of OpenMC materials
        inputs_dict: Custom inputs dictionary. If None, uses the global inputs.

    Returns:
        openmc.Universe: Universe containing the guide tube
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    # Create cylinders for guide tube regions - convert m to cm
    r1 = openmc.ZCylinder(r=inputs_dict['r_clad_inner'] * 100)  # m to cm
    r2 = openmc.ZCylinder(r=inputs_dict['r_clad_outer'] * 100)  # m to cm

    # Map clad type to material name
    clad_material_map = {
        'Zirc2': 'Zircaloy',
        'Zirc4': 'Zircaloy',
        'Al6061': 'Al6061'
    }
    clad_material = clad_material_map[inputs_dict['clad_type']]

    # Create cells
    inner_coolant_cell = openmc.Cell(fill=mat_dict[f"{inputs_dict['coolant_type']} Feed"], region=-r1)
    zr_cell = openmc.Cell(fill=mat_dict[clad_material], region=+r1 & -r2)
    outer_coolant_cell = openmc.Cell(fill=mat_dict[f"{inputs_dict['coolant_type']} Coolant"], region=+r2)

    # Create universe
    guide_tube = openmc.Universe(name='guide_tube', cells=[inner_coolant_cell, zr_cell, outer_coolant_cell])
    return guide_tube

def build_fuel_assembly_uni(mat_dict, position=None, is_enhanced=False, inputs_dict=None):
    """Build a fuel pin assembly universe.

    Parameters
    ----------
    mat_dict : dict
        Dictionary of materials
    position : tuple, optional
        (i, j) position in core lattice. If provided, assigns unique ID.
    is_enhanced : bool, optional
        Whether to use enhanced enrichment fuel
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    # Get parameters from inputs - convert m to cm
    pin_pitch = inputs_dict['pin_pitch'] * 100  # m to cm
    n_side_pins = inputs_dict['n_side_pins']
    guide_tube_positions = inputs_dict['guide_tube_positions']

    # Build universes for fuel pins and guide tubes
    guide_tube = build_guide_tube_uni(mat_dict, inputs_dict)
    fuel = build_pin_cell_fuel_uni(mat_dict, is_enhanced, inputs_dict)

    # Create lattice
    ll = -pin_pitch * n_side_pins / 2
    pin_lattice = openmc.RectLattice(name='Fuel assembly')
    pin_lattice.lower_left = (ll, ll)
    pin_lattice.pitch = (pin_pitch, pin_pitch)

    # Fill lattice with fuel pins
    pin_lattice.universes = np.tile(fuel, (n_side_pins, n_side_pins))

    # Place guide tubes
    for i, j in guide_tube_positions:
        pin_lattice.universes[i, j] = guide_tube

    # Set outer universe to coolant
    coolant_uni = openmc.Universe(cells=[openmc.Cell(fill=mat_dict[f"{inputs_dict['coolant_type']} Coolant"])])
    pin_lattice.outer = coolant_uni

    # Create assembly cell and universe
    assembly_cell = openmc.Cell(name='fuel_pin_cell')
    if position is not None:
        assembly_cell.id = generate_cell_id('fuel', position, is_enhanced)
        assembly_cell.name = f"{'enhanced_' if is_enhanced else ''}fuel_pin_cell_{position[0]}_{position[1]}"
    assembly_cell.fill = pin_lattice
    assembly_uni = openmc.Universe(name='Fuel assembly', cells=[assembly_cell])

    return assembly_uni
