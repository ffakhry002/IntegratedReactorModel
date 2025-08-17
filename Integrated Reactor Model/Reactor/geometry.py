import openmc
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import shutil
import importlib

# Add root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

# Add Reactor directory to path for materials
reactor_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(reactor_dir)

from Reactor.geometry_helpers.pin_fuel import build_fuel_assembly_uni as build_pin_assembly, build_pin_cell_fuel_uni
from Reactor.geometry_helpers.plate_fuel import build_fuel_assembly_uni as build_plate_assembly, build_plate_cell_fuel_uni
from Reactor.geometry_helpers.irradiation_cell import build_irradiation_cell_uni
from Reactor.geometry_helpers.core import build_core_uni
from Reactor.materials import make_materials
from utils.base_inputs import inputs

def plot_geometry(output_dir=None, inputs_dict=None):
    """Create and plot the reactor geometry.

    Parameters
    ----------
    output_dir : str, optional
        Directory to save outputs. If None, saves in local directory.
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    None
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs
    # Determine output directory
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output')
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Create materials with None for mat_list to get all materials
    mat_dict, mat_obj = make_materials(mat_list=None, inputs_dict=inputs_dict)

    # Save materials to output file
    materials_output = os.path.join(output_dir, 'materials.txt')
    with open(materials_output, 'w') as f:
        for mat in mat_obj:
            f.write(str(mat) + '\n\n')
    print(f"Materials output saved to: {materials_output}")

    # Define material colors for plotting - each specific material gets its own color
    mat_colors = {
        # Fuels
        mat_dict['UO2']: 'orange',
        mat_dict['U3Si2']: 'darkgoldenrod',
        mat_dict['U10Mo']: 'goldenrod',
        mat_dict['DU']: 'peru',  # Depleted uranium
        mat_dict.get('UO2-Enhanced', mat_dict['UO2']): 'orange',  # Enhanced fuels use same color as base
        mat_dict.get('U3Si2-Enhanced', mat_dict['U3Si2']): 'darkgoldenrod',
        mat_dict.get('U10Mo-Enhanced', mat_dict['U10Mo']): 'goldenrod',

        # Cladding materials
        mat_dict['Zircaloy']: 'dimgray',
        mat_dict['Al6061']: 'silver',
        mat_dict['HT9']: 'darkgray',
        mat_dict['SS316']: 'gray',

        # Enhanced cladding materials (pink)
        mat_dict.get('Zircaloy-Enhanced', mat_dict['Zircaloy']): 'pink',
        mat_dict.get('Al6061-Enhanced', mat_dict['Al6061']): 'pink',
        mat_dict.get('HT9-Enhanced', mat_dict['HT9']): 'pink',

        # Coolant materials
        mat_dict[f"{inputs_dict['coolant_type']} Coolant"]: 'lightblue',
        mat_dict[f"{inputs_dict['coolant_type']} Feed"]: 'deepskyblue',
        mat_dict[f"{inputs_dict['coolant_type']} Outer"]: 'skyblue',
        mat_dict[f"{inputs_dict['coolant_type']} Plenum"]: 'cornflowerblue',
        # Gap material
        mat_dict['Helium']: 'white',

        # Irradiation cell materials
        mat_dict['Test pos']: 'slategray',  # Fill material (Al-water mixture)
        mat_dict['Vacuum']: 'black',
        mat_dict['PWR_loop']: 'mediumpurple',  # PWR loop
        mat_dict['BWR_loop']: 'orange',     # BWR loop
        mat_dict['Gas_capsule']: 'limegreen',     # Gas capsule

        # Core structure materials
        mat_dict['Steel']: 'lightgray',  # Light gray
        mat_dict['SS316']: 'darkgray',  # Medium gray
        mat_dict['Concrete']: 'tan',
        mat_dict['beryllium']: 'lightgreen',
        mat_dict['beryllium oxide']: 'mediumseagreen',
        mat_dict['mgo']: 'purple'
    }

    # Common plot parameters
    plot_params = {
        'pixels': inputs_dict['pixels'],
        'colors': mat_colors,
        'color_by': 'material'
    }

    # Build and plot core
    core_universe, first_irr_universe = build_core_uni(mat_dict, inputs_dict=inputs_dict)  # Get both core and first irradiation cell

    # Calculate total core dimensions
    total_radius = (inputs_dict['tank_radius'] + inputs_dict['reflector_thickness'] +
                   inputs_dict['bioshield_thickness']) * 100  # m to cm
    total_height = (inputs_dict['bottom_bioshield_thickness'] + inputs_dict['bottom_reflector_thickness'] +
                   inputs_dict['feed_thickness'] + inputs_dict['fuel_height'] + inputs_dict['plenum_height'] +
                   inputs_dict['top_reflector_thickness'] + inputs_dict['top_bioshield_thickness']) * 100  # m to cm

    # Core plots
    core_params = plot_params.copy()

    # XY plot at core midplane
    core_params['width'] = (total_radius * 2.2, total_radius * 2.2)  # 10% margin
    core_params['basis'] = 'xy'
    core_params['origin'] = (0, 0, inputs_dict['fuel_height'] * 50)  # Plot at core midplane
    plot_core_xy = core_universe.plot(**core_params)
    plot_core_xy.figure.set_size_inches(8, 8)
    plot_core_xy.figure.savefig(os.path.join(output_dir, 'core_xy.png'),
                               dpi=300, bbox_inches='tight')
    plt.close()

    # YZ plot through center
    core_params['width'] = (total_radius * 2.2, total_height * 1.1)
    core_params['basis'] = 'yz'
    core_params['origin'] = (0, 0, 0)
    plot_core_yz = core_universe.plot(**core_params)
    plot_core_yz.figure.set_size_inches(8, 12)
    plot_core_yz.figure.savefig(os.path.join(output_dir, 'core_yz.png'),
                               dpi=300, bbox_inches='tight')
    plt.close()

    # XZ plot through center
    core_params['basis'] = 'xz'
    plot_core_xz = core_universe.plot(**core_params)
    plot_core_xz.figure.set_size_inches(8, 12)
    plot_core_xz.figure.savefig(os.path.join(output_dir, 'core_xz.png'),
                               dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate irradiation cell dimensions
    if inputs_dict['assembly_type'] == 'Pin':
        cell_width = inputs_dict['pin_pitch'] * inputs_dict['n_side_pins'] * 100  # m to cm
    else:
        cell_width = (2 * inputs_dict['clad_structure_width'] + inputs_dict['fuel_plate_width']) * 100  # m to cm

    # Irradiation cell plots using the stored universe
    irradiation_params = plot_params.copy()
    irradiation_params['width'] = (cell_width * 1.1, cell_width * 1.1)  # 10% margin

    has_irradiation = any('I' in cell for row in inputs_dict['core_lattice'] for cell in row)
    if has_irradiation:
        try:
            print("\nGenerating flux trap plots...")
            # Create Irradiation Cell XY plot
            plot_irr_xy = first_irr_universe.plot(basis='xy', **irradiation_params)
            plot_irr_xy.figure.set_size_inches(6, 6)
            plot_irr_xy.figure.savefig(os.path.join(output_dir, 'irradiation_cell_xy.png'),
                                    dpi=300, bbox_inches='tight')
            plt.close()

            # Create Irradiation Cell YZ plot
            plot_irr_yz = first_irr_universe.plot(basis='yz', **irradiation_params)
            plot_irr_yz.figure.set_size_inches(6, 6)
            plot_irr_yz.figure.savefig(os.path.join(output_dir, 'irradiation_cell_yz.png'),
                                    dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating flux trap plots: {str(e)}")
    else:
        print("\nSkipping flux trap plots (no irradiation positions in core)")


    if inputs_dict['assembly_type'] == 'Pin':
        # Calculate pin dimensions in cm
        pin_pitch = inputs_dict['pin_pitch'] * 100  # m to cm
        n_side_pins = inputs_dict['n_side_pins']
        assembly_width = pin_pitch * n_side_pins
        pin_radius = inputs_dict['r_clad_outer'] * 100  # m to cm

        # Build universes
        assembly_universe = build_pin_assembly(mat_dict, inputs_dict=inputs_dict)
        pin_universe = build_pin_cell_fuel_uni(mat_dict, inputs_dict=inputs_dict)

        # Assembly plots
        assembly_params = plot_params.copy()
        assembly_params['width'] = (assembly_width * 1.1, assembly_width * 1.1)  # 10% margin

        # Create Pin Assembly XY plot
        plot_xy = assembly_universe.plot(basis='xy', **assembly_params)
        plot_xy.figure.set_size_inches(6, 6)
        plot_xy.figure.savefig(os.path.join(output_dir, 'pin_assembly_xy.png'),
                              dpi=300, bbox_inches='tight')
        plt.close()

        # Create Pin Assembly YZ plot
        plot_yz = assembly_universe.plot(basis='yz', **assembly_params)
        plot_yz.figure.set_size_inches(6, 6)
        plot_yz.figure.savefig(os.path.join(output_dir, 'pin_assembly_yz.png'),
                              dpi=300, bbox_inches='tight')
        plt.close()

        # Pin cell plots
        pin_params = plot_params.copy()
        pin_params['width'] = (pin_pitch * 1.1, pin_pitch * 1.1)  # 10% margin

        # Create Single Pin XY plot
        plot_pin_xy = pin_universe.plot(basis='xy', **pin_params)
        plot_pin_xy.figure.set_size_inches(6, 6)
        plot_pin_xy.figure.savefig(os.path.join(output_dir, 'single_pin_xy.png'),
                                  dpi=300, bbox_inches='tight')
        plt.close()

        # Create Single Pin YZ plot
        plot_pin_yz = pin_universe.plot(basis='yz', **pin_params)
        plot_pin_yz.figure.set_size_inches(6, 6)
        plot_pin_yz.figure.savefig(os.path.join(output_dir, 'single_pin_yz.png'),
                                  dpi=300, bbox_inches='tight')
        plt.close()

    else:  # Plate assembly
        # Calculate plate dimensions in cm
        plate_pitch = inputs_dict['fuel_plate_pitch'] * 100  # m to cm
        assembly_width = inputs_dict['plates_per_assembly'] * plate_pitch
        plate_width = inputs_dict['fuel_plate_width'] * 100  # m to cm
        total_assembly_width = assembly_width + 2 * (inputs_dict['clad_structure_width'] * 100)  # Add side plates

        # Build universes with dummy positions to avoid ID conflicts with main simulation
        dummy_position = (99, 99)  # Use position that won't conflict with real core positions
        assembly_universe = build_plate_assembly(mat_dict, position=dummy_position, inputs_dict=inputs_dict)
        plate_universe = build_plate_cell_fuel_uni(mat_dict, inputs_dict=inputs_dict, universe_id=9999999)  # Use high ID to avoid conflicts

        # Assembly plots
        assembly_params = plot_params.copy()
        assembly_params['width'] = (total_assembly_width * 1.1, total_assembly_width * 1.1)  # 10% margin

        # Create Plate Assembly XY plot
        plot_xy = assembly_universe.plot(basis='xy', **assembly_params)
        plot_xy.figure.set_size_inches(6, 6)
        plot_xy.figure.savefig(os.path.join(output_dir, 'plate_assembly_xy.png'),
                              dpi=300, bbox_inches='tight')
        plt.close()

        # Create Plate Assembly YZ plot
        plot_yz = assembly_universe.plot(basis='yz', **assembly_params)
        plot_yz.figure.set_size_inches(6, 6)
        plot_yz.figure.savefig(os.path.join(output_dir, 'plate_assembly_yz.png'),
                              dpi=300, bbox_inches='tight')
        plt.close()

        # Single plate plots
        plate_params = plot_params.copy()
        plate_params['width'] = (plate_width * 1.1, plate_width * 1.1)  # 10% margin

        # Create Single Plate XY plot
        plot_plate_xy = plate_universe.plot(basis='xy', **plate_params)
        plot_plate_xy.figure.set_size_inches(6, 6)
        plot_plate_xy.figure.savefig(os.path.join(output_dir, 'single_plate_xy.png'),
                                    dpi=300, bbox_inches='tight')
        plt.close()

        # Create Single Plate YZ plot
        plot_plate_yz = plate_universe.plot(basis='yz', **plate_params)
        plot_plate_yz.figure.set_size_inches(6, 6)
        plot_plate_yz.figure.savefig(os.path.join(output_dir, 'single_plate_yz.png'),
                                    dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Geometry plots saved to: {output_dir}")

if __name__ == '__main__':
    plot_geometry()
