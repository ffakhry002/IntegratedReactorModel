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

        # Create organized subfolders for different plot types
    subfolders = {
        'core': os.path.join(output_dir, 'core_images'),
        'fuel': os.path.join(output_dir, 'fuel_images'),
    }

    # Check which irradiation types exist in the core lattice
    has_P = any(cell.endswith('P') for row in inputs_dict['core_lattice'] for cell in row if cell.startswith('I_'))
    has_B = any(cell.endswith('B') for row in inputs_dict['core_lattice'] for cell in row if cell.startswith('I_'))
    has_G = any(cell.endswith('G') for row in inputs_dict['core_lattice'] for cell in row if cell.startswith('I_'))
    has_blank = any(cell.startswith('I_') and not cell.endswith(('P', 'B', 'G')) for row in inputs_dict['core_lattice'] for cell in row)

    # Only create irradiation folders for types that exist
    if has_P:
        subfolders['irradiation_P'] = os.path.join(output_dir, 'irradiation_P')
    if has_B:
        subfolders['irradiation_B'] = os.path.join(output_dir, 'irradiation_B')
    if has_G:
        subfolders['irradiation_G'] = os.path.join(output_dir, 'irradiation_G')
    if has_blank:
        subfolders['default_irradiation'] = os.path.join(output_dir, 'default_irradiation')

    # Create all the subfolders
    for folder_path in subfolders.values():
        os.makedirs(folder_path, exist_ok=True)

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
        # Fuel materials
        mat_dict['UO2']: 'orange',
        mat_dict['U3Si2']: 'darkgoldenrod',
        mat_dict['U10Mo']: 'goldenrod',
        mat_dict['DU']: 'peru',
        mat_dict.get('UO2-Enhanced', mat_dict['UO2']): 'orange',
        mat_dict.get('U3Si2-Enhanced', mat_dict['U3Si2']): 'darkgoldenrod',
        mat_dict.get('U10Mo-Enhanced', mat_dict['U10Mo']): 'goldenrod',

        # Cladding materials
        mat_dict['Zircaloy']: 'dimgray',
        mat_dict['Al6061']: 'silver',
        mat_dict['HT9']: 'darkgray',
        mat_dict['SS316']: 'gray',

        # Enhanced cladding materials
        mat_dict.get('Zircaloy-Enhanced', mat_dict['Zircaloy']): 'pink',
        mat_dict.get('Al6061-Enhanced', mat_dict['Al6061']): 'pink',
        mat_dict.get('HT9-Enhanced', mat_dict['HT9']): 'pink',

        # Coolant materials
        mat_dict[f"{inputs_dict['coolant_type']} Coolant"]: 'lightblue',
        mat_dict[f"{inputs_dict['coolant_type']} Feed"]: 'deepskyblue',
        mat_dict[f"{inputs_dict['coolant_type']} Outer"]: 'skyblue',
        mat_dict[f"{inputs_dict['coolant_type']} Plenum"]: 'cornflowerblue',

        # ====== IRRADIATION EXPERIMENT MATERIALS ======
        # Core experiment materials - DISTINCT COLORS
        mat_dict['Titanium']: 'darkred',           # Dark red for Ti (was steelblue)
        mat_dict['graphite']: 'dimgray',           # Dark gray for graphite
        mat_dict['SiC']: 'darkgreen',              # Dark green for SiC
        mat_dict['Tungsten']: 'darkviolet',        # Dark purple for tungsten
        mat_dict['CO2']: 'black',            # Light yellow for CO2 gas
        mat_dict['Helium']: 'lightgray',               # White for regular helium
        mat_dict['HT_Helium']: 'lavender',         # Light purple for high-temp helium

        # Water materials - DISTINCT BLUES/CYANS
        mat_dict['HP_Borated_Water']: 'cyan',      # Bright cyan for borated water
        mat_dict['BWR_fluid']: 'turquoise',        # Turquoise for BWR water

        # Structural materials for experiments
        mat_dict['Al6061']: 'orange',              # Orange for aluminum (shows clearly)

        # Loop/capsule materials (smeared)
        mat_dict['Test pos']: 'slategray',
        mat_dict['Vacuum']: 'black',
        mat_dict['PWR_loop']: 'mediumpurple',      # For simple/smeared PWR
        mat_dict['BWR_loop']: 'coral',             # For simple/smeared BWR
        mat_dict['Gas_capsule']: 'limegreen',      # For simple/smeared gas

        # Other structural materials
        mat_dict['Steel']: 'lightgray',
        mat_dict['Concrete']: 'tan',
        mat_dict['beryllium']: 'lightgreen',
        mat_dict['beryllium oxide']: 'mediumseagreen',
        mat_dict['mgo']: 'purple',
    }

    # Common plot parameters
    plot_params = {
        'pixels': inputs_dict['pixels'],
        'colors': mat_colors,
        'color_by': 'material'
    }

    # Build and plot core
    core_universe, irradiation_universes = build_core_uni(mat_dict, inputs_dict=inputs_dict)  # Get both core and irradiation universes

    # Calculate total core dimensions
    total_radius = (inputs_dict['tank_radius'] + inputs_dict['reflector_thickness'] +
                   inputs_dict['bioshield_thickness']) * 100  # m to cm
    total_height = (inputs_dict['bottom_bioshield_thickness'] + inputs_dict['bottom_reflector_thickness'] +
                   inputs_dict['feed_thickness'] + inputs_dict['fuel_height'] + inputs_dict['plenum_height'] +
                   inputs_dict['top_reflector_thickness'] + inputs_dict['top_bioshield_thickness']) * 100  # m to cm

    # Core plots
    core_params = plot_params.copy()

    # Determine optimal viewing height for core based on irradiation types present
    has_htwl = any(cell.endswith('P') or cell.endswith('B') for row in inputs_dict['core_lattice'] for cell in row if cell.startswith('I_'))
    has_sigma = any(cell.endswith('G') for row in inputs_dict['core_lattice'] for cell in row if cell.startswith('I_'))

    if has_htwl:
        # If HTWL experiments present, view at sample region to show them
        optimal_core_z = -12.0  # Middle of HTWL sample region
        print(f"Core contains HTWL experiments - using z = {optimal_core_z} cm to show samples")
    elif has_sigma:
        # If only SIGMA experiments, view at center
        optimal_core_z = 0.0
        print(f"Core contains SIGMA experiments - using z = {optimal_core_z} cm")
    else:
        # No complex irradiation, view at fuel midplane
        optimal_core_z = 0.0  # Fuel midplane
        print(f"No complex irradiation - using z = {optimal_core_z} cm (fuel midplane)")

    # XY plot at optimal viewing height
    core_params['width'] = (total_radius * 2.2, total_radius * 2.2)  # 10% margin
    core_params['basis'] = 'xy'
    core_params['origin'] = (0, 0, optimal_core_z)  # Plot at optimal height
    print("plotting core xy")
    plot_core_xy = core_universe.plot(**core_params)
    plot_core_xy.figure.set_size_inches(10, 10)
    plot_core_xy.figure.savefig(os.path.join(subfolders['core'], 'core_xy.png'),
                               dpi=3000, bbox_inches='tight')
    plt.close()

    # YZ plot through center
    print("plotting core yz")
    core_params['width'] = (total_radius * 2.2, total_height * 1.1)
    core_params['basis'] = 'yz'
    core_params['origin'] = (0, 0, 0)
    plot_core_yz = core_universe.plot(**core_params)
    plot_core_yz.figure.set_size_inches(8, 12)
    plot_core_yz.figure.savefig(os.path.join(subfolders['core'], 'core_yz.png'),
                               dpi=3000, bbox_inches='tight')
    plt.close()

    # XZ plot through center
    print("plotting core xz")
    core_params['basis'] = 'xz'
    plot_core_xz = core_universe.plot(**core_params)
    plot_core_xz.figure.set_size_inches(8, 12)
    plot_core_xz.figure.savefig(os.path.join(subfolders['core'], 'core_xz.png'),
                               dpi=3000, bbox_inches='tight')
    plt.close()

    # Calculate irradiation cell dimensions
    if inputs_dict['assembly_type'] == 'Pin':
        cell_width = inputs_dict['pin_pitch'] * inputs_dict['n_side_pins'] * 100  # m to cm
    else:
        cell_width = (2 * inputs_dict['clad_structure_width'] + inputs_dict['fuel_plate_width']) * 100  # m to cm

    # Irradiation cell plots using the stored universe
    irradiation_params = plot_params.copy()
    fuel_height_cm = inputs_dict['fuel_height'] * 100  # Convert m to cm
    irradiation_params['width'] = (cell_width * 1.1, cell_width * 1.1)  # 10% margin for XY plots

    has_irradiation = any('I' in cell for row in inputs_dict['core_lattice'] for cell in row)
    print(f"has_irradiation: {has_irradiation}")
    if has_irradiation and irradiation_universes:
        print("plotting irradiation cells")
        try:
            print(f"\nGenerating flux trap plots for {len(irradiation_universes)} unique irradiation types...")

            # Plot each unique irradiation type
            for irr_type, irr_data in irradiation_universes.items():
                universe = irr_data['universe']
                position_name = irr_data['position_name']

                # Create type-specific filenames and determine subfolder
                type_suffix = f"_{irr_type}" if irr_type != 'blank' else "_blank"

                # Determine which subfolder to use based on irradiation type
                if irr_type == 'P' and 'irradiation_P' in subfolders:
                    irr_folder = subfolders['irradiation_P']
                elif irr_type == 'B' and 'irradiation_B' in subfolders:
                    irr_folder = subfolders['irradiation_B']
                elif irr_type == 'G' and 'irradiation_G' in subfolders:
                    irr_folder = subfolders['irradiation_G']
                elif 'default_irradiation' in subfolders:
                    irr_folder = subfolders['default_irradiation']
                else:
                    # Fallback to main output directory if no appropriate subfolder exists
                    irr_folder = output_dir

                print(f"  - Plotting irradiation type '{irr_type}' (representative: {position_name})")

                # Set optimal viewing height based on irradiation type
                if irr_type in ['P', 'B']:  # PWR_loop or BWR_loop (HTWL geometry)
                    # Samples exist from z = -23.5 to -0.5 cm, so view at middle
                    optimal_z = -12.0  # Middle of sample region to show all 4 samples
                    print(f"  Using z = {optimal_z} cm to view HTWL samples")
                else:  # Gas_capsule (SIGMA) or other geometries
                    # Uniform geometry, view at center
                    optimal_z = 0.0
                    print(f"  Using z = {optimal_z} cm for uniform geometry")

                # Create Irradiation Cell XY plot with optimal viewing height
                print("plotting irradiation cell xy")
                irradiation_params_xy = irradiation_params.copy()
                irradiation_params_xy['origin'] = (0, 0, optimal_z)
                plot_irr_xy = universe.plot(basis='xy', **irradiation_params_xy)
                plot_irr_xy.figure.set_size_inches(6, 6)
                plot_irr_xy.figure.savefig(os.path.join(irr_folder, f'irradiation_cell{type_suffix}_xy.png'),
                                        dpi=3000, bbox_inches='tight')
                plt.close()

                # Create Irradiation Cell YZ plot with full fuel height
                print("plotting irradiation cell yz")
                irradiation_params_yz = irradiation_params.copy()
                irradiation_params_yz['width'] = (cell_width * 1.1, fuel_height_cm * 1.1)  # Full height for YZ
                plot_irr_yz = universe.plot(basis='yz', **irradiation_params_yz)
                plot_irr_yz.figure.set_size_inches(6, 8)  # Taller figure for axial view
                plot_irr_yz.figure.savefig(os.path.join(irr_folder, f'irradiation_cell{type_suffix}_yz.png'),
                                        dpi=3000, bbox_inches='tight')
                plt.close()

                # Create Irradiation Cell XZ plot with full fuel height
                print("plotting irradiation cell xz")
                irradiation_params_xz = irradiation_params.copy()
                irradiation_params_xz['width'] = (cell_width * 1.1, fuel_height_cm * 1.1)  # Full height for XZ
                plot_irr_xz = universe.plot(basis='xz', **irradiation_params_xz)
                plot_irr_xz.figure.set_size_inches(6, 8)  # Taller figure for axial view
                plot_irr_xz.figure.savefig(os.path.join(irr_folder, f'irradiation_cell{type_suffix}_xz.png'),
                                        dpi=3000, bbox_inches='tight')
                plt.close()

                # Additional YZ and XZ plots with 6cm axial slice (±3cm from optimal height)
                print(f"plotting irradiation cell yz with 6cm axial slice around optimal height (z = {optimal_z} ± 3cm)")
                irradiation_params_yz_slice = irradiation_params.copy()
                irradiation_params_yz_slice['width'] = (cell_width * 1.1, 6.0)  # 6cm axial height
                irradiation_params_yz_slice['origin'] = (0, 0, optimal_z)  # Center at optimal height
                plot_irr_yz_slice = universe.plot(basis='yz', **irradiation_params_yz_slice)
                plot_irr_yz_slice.figure.set_size_inches(6, 6)  # Square figure for 6cm slice
                plot_irr_yz_slice.figure.savefig(os.path.join(irr_folder, f'irradiation_cell{type_suffix}_yz_6cm_slice.png'),
                                                dpi=3000, bbox_inches='tight')
                plt.close()

                print(f"plotting irradiation cell xz with 6cm axial slice around optimal height (z = {optimal_z} ± 3cm)")
                irradiation_params_xz_slice = irradiation_params.copy()
                irradiation_params_xz_slice['width'] = (cell_width * 1.1, 6.0)  # 6cm axial height
                irradiation_params_xz_slice['origin'] = (0, 0, optimal_z)  # Center at optimal height
                plot_irr_xz_slice = universe.plot(basis='xz', **irradiation_params_xz_slice)
                plot_irr_xz_slice.figure.set_size_inches(6, 6)  # Square figure for 6cm slice
                plot_irr_xz_slice.figure.savefig(os.path.join(irr_folder, f'irradiation_cell{type_suffix}_xz_6cm_slice.png'),
                                                dpi=3000, bbox_inches='tight')
                plt.close()

        except Exception as e:
            print(f"Error generating flux trap plots: {str(e)}")
    else:
        print("\nSkipping flux trap plots (no irradiation positions in core)")


    if inputs_dict['assembly_type'] == 'Pin':
        print("plotting pin assembly")
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
        print("plotting pin assembly xy")
        plot_xy = assembly_universe.plot(basis='xy', **assembly_params)
        plot_xy.figure.set_size_inches(6, 6)
        plot_xy.figure.savefig(os.path.join(subfolders['fuel'], 'pin_assembly_xy.png'),
                              dpi=3000, bbox_inches='tight')
        plt.close()

        # Create Pin Assembly YZ plot
        print("plotting pin assembly yz")
        plot_yz = assembly_universe.plot(basis='yz', **assembly_params)
        plot_yz.figure.set_size_inches(6, 6)
        plot_yz.figure.savefig(os.path.join(subfolders['fuel'], 'pin_assembly_yz.png'),
                              dpi=3000, bbox_inches='tight')
        plt.close()

        # Pin cell plots
        pin_params = plot_params.copy()
        pin_params['width'] = (pin_pitch * 1.1, pin_pitch * 1.1)  # 10% margin

        # Create Single Pin XY plot
        print("plotting single pin xy")
        plot_pin_xy = pin_universe.plot(basis='xy', **pin_params)
        plot_pin_xy.figure.set_size_inches(6, 6)
        plot_pin_xy.figure.savefig(os.path.join(subfolders['fuel'], 'single_pin_xy.png'),
                                  dpi=3000, bbox_inches='tight')
        plt.close()
        print("plotting single pin yz")
        # Create Single Pin YZ plot
        plot_pin_yz = pin_universe.plot(basis='yz', **pin_params)
        plot_pin_yz.figure.set_size_inches(6, 6)
        plot_pin_yz.figure.savefig(os.path.join(subfolders['fuel'], 'single_pin_yz.png'),
                                  dpi=3000, bbox_inches='tight')
        plt.close()
        print("plotting single pin xz")
    else:  # Plate assembly
        print("plotting plate assembly")
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
        print("plotting plate assembly xy")
        plot_xy = assembly_universe.plot(basis='xy', **assembly_params)
        plot_xy.figure.set_size_inches(6, 6)
        plot_xy.figure.savefig(os.path.join(subfolders['fuel'], 'plate_assembly_xy.png'),
                              dpi=3000, bbox_inches='tight')
        plt.close()

        # Create Plate Assembly YZ plot
        print("plotting plate assembly yz")
        plot_yz = assembly_universe.plot(basis='yz', **assembly_params)
        plot_yz.figure.set_size_inches(6, 6)
        plot_yz.figure.savefig(os.path.join(subfolders['fuel'], 'plate_assembly_yz.png'),
                              dpi=3000, bbox_inches='tight')
        plt.close()

        # Single plate plots
        plate_params = plot_params.copy()
        plate_params['width'] = (plate_width * 1.1, plate_width * 1.1)  # 10% margin

        # Create Single Plate XY plot
        print("plotting single plate xy")
        plot_plate_xy = plate_universe.plot(basis='xy', **plate_params)
        plot_plate_xy.figure.set_size_inches(6, 6)
        plot_plate_xy.figure.savefig(os.path.join(subfolders['fuel'], 'single_plate_xy.png'),
                                    dpi=3000, bbox_inches='tight')
        plt.close()

        # Create Single Plate YZ plot
        print("plotting single plate yz")
        plot_plate_yz = plate_universe.plot(basis='yz', **plate_params)
        plot_plate_yz.figure.set_size_inches(6, 6)
        plot_plate_yz.figure.savefig(os.path.join(subfolders['fuel'], 'single_plate_yz.png'),
                                    dpi=3000, bbox_inches='tight')
        plt.close()

    # Create color legend as the final step
    print("creating color legend")
    create_color_legend(mat_colors, output_dir)

    print(f"Geometry plots saved to: {output_dir}")

def create_color_legend(mat_colors, output_dir):
    """Create a color legend showing all materials and their colors."""
    import matplotlib.patches as patches

    # Filter out None entries and get material names
    valid_materials = {}
    for material, color in mat_colors.items():
        if material is not None and color is not None:
            # Get material name, handling both string names and Material objects
            if hasattr(material, 'name'):
                mat_name = material.name
            else:
                mat_name = str(material)
            valid_materials[mat_name] = color

    # Sort materials alphabetically for consistent legend
    sorted_materials = sorted(valid_materials.items())

        # Calculate figure dimensions based on number of materials
    n_materials = len(sorted_materials)
    n_cols = 8  # More columns for better layout
    n_rows = (n_materials + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with more width and better spacing
    fig_width = n_cols * 3.5  # More space per column
    fig_height = max(8, n_rows * 1.2)  # More space per row
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Set limits with padding
    ax.set_xlim(0, n_cols * 3.5)
    ax.set_ylim(0, n_rows * 1.2)

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add title
    plt.suptitle('Material Color Legend', fontsize=18, fontweight='bold', y=0.97)

    # Add color patches and labels
    for i, (mat_name, color) in enumerate(sorted_materials):
        row = n_rows - 1 - (i // n_cols)  # Start from top
        col = i % n_cols

        # Calculate positions with better spacing
        x_pos = col * 3.5 + 0.2
        y_pos = row * 1.2 + 0.4

        # Create larger color patch
        rect = patches.Rectangle((x_pos, y_pos),
                               0.8, 0.6,  # Bigger blocks
                               linewidth=1.5,
                               edgecolor='black',
                               facecolor=color)
        ax.add_patch(rect)

                # Add material name with more space
        ax.text(x_pos + 1.0, y_pos + 0.3,
               mat_name,
               fontsize=14,
               verticalalignment='center',
               fontweight='normal',
               ha='left')  # Left align text

    # Save legend
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'material_color_legend.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Color legend saved with {n_materials} materials")

if __name__ == '__main__':
    plot_geometry()
