"""
Main script for running pin cell depletion calculations.
"""

import openmc
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from inputs import inputs
from materials import make_materials

def build_pin_cell_fuel_uni(mat_dict):
    """Build a fuel pin cell universe.

    Args:
        mat_dict (dict): Dictionary of materials

    Returns:
        openmc.Universe: Universe containing the fuel pin cell
    """
    # Create cylinders for fuel pin regions - convert m to cm
    r1 = openmc.ZCylinder(r=inputs['r_fuel'] * 100)  # m to cm
    r2 = openmc.ZCylinder(r=inputs['r_clad_inner'] * 100)  # m to cm
    r3 = openmc.ZCylinder(r=inputs['r_clad_outer'] * 100)  # m to cm

    # Create cells
    fuel_cell = openmc.Cell(fill=mat_dict[inputs['fuel_type']], region=-r1)
    he_cell = openmc.Cell(fill=mat_dict['Helium'], region=+r1 & -r2)
    clad_cell = openmc.Cell(fill=mat_dict[inputs['clad_type']], region=+r2 & -r3)
    water_cell = openmc.Cell(fill=mat_dict['Light Water'], region=+r3)

    # Create universe
    pin_universe = openmc.Universe(name='fuel', cells=[fuel_cell, he_cell, clad_cell, water_cell])

    return pin_universe

if __name__ == "__main__":
    # Get materials
    mat_dict, _ = make_materials()

    # Material colors for plotting
    mat_colors = {
        # Fuels
        mat_dict['UO2']: 'orange',
        mat_dict['U3Si2']: 'darkgoldenrod',
        mat_dict['U10Mo']: 'goldenrod',
        # Cladding
        mat_dict['Zircaloy']: 'dimgray',
        mat_dict['Al6061']: 'silver',
        # Gap
        mat_dict['Helium']: 'white',
        # Coolant
        mat_dict['Light Water']: 'lightblue'
    }

    # Build pin
    pin_universe = build_pin_cell_fuel_uni(mat_dict)

    # Plot parameters
    pin_params = {
        'color_by': 'material',
        'colors': mat_colors,
        'pixels': [400, 400],
        'width': (inputs['r_clad_outer'] * 250, inputs['r_clad_outer'] * 250)  # Make plot slightly larger than pin
    }

    # Create Pin XY plot
    plot_xy = pin_universe.plot(basis='xy', **pin_params)
    plot_xy.figure.set_size_inches(6, 6)
    plot_xy.figure.savefig('pin_xy.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nPin plot has been saved as: pin_xy.png")
