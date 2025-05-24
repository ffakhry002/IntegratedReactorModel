# Set matplotlib backend for thread safety
import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.patches import Rectangle, Circle, Patch

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

from inputs import inputs

def initialize_globals(inputs_dict=None):
    """Initialize global variables for geometry plotting from inputs.

    Args:
        inputs_dict (dict, optional): Custom inputs dictionary. If None, uses the global inputs.

    Sets up global variables for pin pitch, fuel and clad dimensions, output folder,
    and fuel height from the inputs configuration.
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    global pin_pitch, r_fuel, r_clad_inner, r_clad_outer, output_folder, fuel_height
    global fuel_meat_width, fuel_plate_width, fuel_meat_thickness, clad_thickness, fuel_plate_pitch
    global coolant_type, clad_type, fuel_type, plates_per_assembly, fuel_plate_thickness, coolant_thickness
    global clad_structure_width, n_side_pins, n_guide_tubes

    # Pin Fuel Geometry (convert from meters to centimeters)
    pin_pitch = inputs_dict["pin_pitch"] * 100  # Convert to cm
    r_fuel = inputs_dict["r_fuel"] * 100        # Convert to cm
    r_clad_inner = inputs_dict["r_clad_inner"] * 100  # Convert to cm
    r_clad_outer = inputs_dict["r_clad_outer"] * 100  # Convert to cm

    # Plate Fuel Geometry (convert from meters to centimeters)
    fuel_meat_width = inputs_dict["fuel_meat_width"] * 100  # Convert to cm
    fuel_plate_width = inputs_dict["fuel_plate_width"] * 100  # Convert to cm
    fuel_meat_thickness = inputs_dict["fuel_meat_thickness"] * 100  # Convert to cm
    clad_thickness = inputs_dict["clad_thickness"] * 100  # Convert to cm
    fuel_plate_pitch = inputs_dict["fuel_plate_pitch"] * 100  # Convert to cm
    fuel_plate_thickness = fuel_meat_thickness + 2 * clad_thickness  # Calculate total plate thickness
    coolant_thickness = (fuel_plate_pitch - fuel_plate_thickness) / 2  # Coolant thickness above and below
    fuel_height = inputs_dict["fuel_height"] * 100  # Convert to cm
    plates_per_assembly = inputs_dict["plates_per_assembly"]  # Number of plates per assembly

    # Material and geometry properties
    clad_structure_width = inputs_dict["clad_structure_width"] * 100  # Convert to cm

    # Pin assembly geometry
    n_side_pins = inputs_dict["n_side_pins"]
    n_guide_tubes = inputs_dict["n_guide_tubes"]

    # Material types
    coolant_type = inputs_dict["coolant_type"]
    clad_type = inputs_dict["clad_type"]
    fuel_type = inputs_dict["fuel_type"]

    output_folder = inputs_dict["outputs_folder"]

def get_material_color(material):
    color_map = {
        "Light Water": "turquoise",
        "Heavy Water": "darkblue",
        "U3Si2": "limegreen",
        "U10Mo": "yellow",
        "UO2": "red",
        "Al6061": "lightgray",
        "Zirc4": "rosybrown",
        "Zirc2": "sienna"
    }
    return color_map.get(material, "gray")

def plot_pin(output_dir, file_name='single_channel_pin.png', inputs_dict=None):
    """Plot a single fuel pin channel cross section.

    Args:
        output_dir (str): Directory to save the plot
        file_name (str, optional): Name of the output file. Defaults to 'single_channel_pin.png'.
        inputs_dict (dict, optional): Custom inputs dictionary. If None, uses the global inputs.
    """
    initialize_globals(inputs_dict)
    # Create figure and axis for the pin plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set plot limits slightly larger than pin pitch
    plot_limit = pin_pitch * 0.6
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)

    # Set background to white (everything outside the coolant region)
    ax.set_facecolor('white')

    # Plot coolant region as a square (Rectangle)
    coolant_color = get_material_color(coolant_type)
    coolant = Rectangle((-pin_pitch/2, -pin_pitch/2), pin_pitch, pin_pitch, color=coolant_color)
    ax.add_patch(coolant)

    # Plot cladding region
    clad_color = get_material_color(clad_type)
    clad = Circle((0, 0), r_clad_outer, color=clad_color)
    ax.add_patch(clad)

    # Plot gap region (as white to show the space between cladding and fuel)
    gap = Circle((0, 0), r_clad_inner, color='white')
    ax.add_patch(gap)

    # Plot fuel region
    fuel_color = get_material_color(fuel_type)
    fuel = Circle((0, 0), r_fuel, color=fuel_color)
    ax.add_patch(fuel)

    ax.set_aspect('equal')
    ax.set_title('Fuel Pin Cross Section (Single Channel)')
    ax.set_xlabel('Distance (cm)')
    ax.set_ylabel('Distance (cm)')

    # Add legend
    fuel_patch = Patch(color=fuel_color, label=f'Fuel ({fuel_type})')
    clad_patch = Patch(color=clad_color, label=f'Cladding ({clad_type})')
    gap_patch = Patch(color='white', label=f'Gap')
    coolant_patch = Patch(color=coolant_color, label=f'Coolant ({coolant_type})')
    ax.legend(handles=[fuel_patch, clad_patch, gap_patch, coolant_patch], loc='upper right')

    # Save the pin plot
    plot_filename = os.path.join(output_dir, file_name)
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print("Plotted: Pin single channel")

def plot_plate(output_dir, file_name='single_channel_plate.png', inputs_dict=None):
    """Plot a single fuel plate channel cross section.

    Args:
        output_dir (str): Directory to save the plot
        file_name (str, optional): Name of the output file. Defaults to 'single_channel_plate.png'.
        inputs_dict (dict, optional): Custom inputs dictionary. If None, uses the global inputs.
    """
    initialize_globals(inputs_dict)

    # Create figure and axis for the plate plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set plot limits slightly larger than the plate width and thickness
    plot_limit_x = fuel_plate_width * 0.6
    plot_limit_y = fuel_plate_pitch * 0.6
    ax.set_xlim(-plot_limit_x, plot_limit_x)
    ax.set_ylim(-plot_limit_y, plot_limit_y)

    # Set background to white
    ax.set_facecolor('white')

    # Calculate the dimensions of the fuel plate components
    x0 = -fuel_plate_width / 2
    x1 = -fuel_meat_width / 2
    x2 = fuel_meat_width / 2
    x3 = fuel_plate_width / 2

    y0 = -fuel_plate_thickness / 2
    y1 = y0 + clad_thickness
    y2 = y1 + fuel_meat_thickness
    y3 = y2 + clad_thickness

    # Plot the coolant regions (above and below)
    coolant_color = get_material_color(coolant_type)
    coolant_top = Rectangle((x0, y3), fuel_plate_width, coolant_thickness, color=coolant_color)
    coolant_bottom = Rectangle((x0, y0 - coolant_thickness), fuel_plate_width, coolant_thickness, color=coolant_color)
    ax.add_patch(coolant_top)
    ax.add_patch(coolant_bottom)

    # Plot the cladding regions
    clad_color = get_material_color(clad_type)
    fuel_color = get_material_color(fuel_type)

    # Plot left cladding
    left_clad = Rectangle((x0, y0), x1 - x0, y3 - y0, color=clad_color)
    ax.add_patch(left_clad)

    # Plot right cladding
    right_clad = Rectangle((x2, y0), x3 - x2, y3 - y0, color=clad_color)
    ax.add_patch(right_clad)

    # Plot top cladding
    top_clad = Rectangle((x1, y2), x2 - x1, y3 - y2, color=clad_color)
    ax.add_patch(top_clad)

    # Plot bottom cladding
    bottom_clad = Rectangle((x1, y0), x2 - x1, y1 - y0, color=clad_color)
    ax.add_patch(bottom_clad)

    # Plot fuel meat region
    fuel_meat = Rectangle((x1, y1), x2 - x1, y2 - y1, color=fuel_color)
    ax.add_patch(fuel_meat)

    ax.set_aspect('equal')
    ax.set_title('Fuel Plate Cross Section (Single Channel)')
    ax.set_xlabel('Distance (cm)')
    ax.set_ylabel('Distance (cm)')

    # Add legend
    fuel_patch = Patch(color=fuel_color, label=f'Fuel ({fuel_type})')
    clad_patch = Patch(color=clad_color, label=f'Cladding ({clad_type})')
    coolant_patch = Patch(color=coolant_color, label=f'Coolant ({coolant_type})')
    ax.legend(handles=[fuel_patch, clad_patch, coolant_patch], loc='upper right')

    # Save the plate plot
    plot_filename = os.path.join(output_dir, file_name)
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print("Plotted: Plate single channel")

def plot_plate_assembly(output_dir, file_name='assembly_plot_plate.png', inputs_dict=None):
    """Plot an entire plate assembly cross section.

    Args:
        output_dir (str): Directory to save the plot
        file_name (str, optional): Name of the output file. Defaults to 'assembly_plot_plate.png'.
        inputs_dict (dict, optional): Custom inputs dictionary. If None, uses the global inputs.
    """
    initialize_globals(inputs_dict)

    # Create figure and axis for the assembly plot
    fig, ax = plt.subplots(figsize=(10, plates_per_assembly * 5))

    # Calculate the total height of the assembly
    total_height = 2 * clad_structure_width + plates_per_assembly * fuel_plate_pitch

    # Set plot limits for assembly with clad border
    plot_limit_x = (fuel_plate_width + 2 * clad_structure_width) * 0.6
    plot_limit_y = total_height * 1.1  # Add 10% margin at the top
    ax.set_xlim(-plot_limit_x, plot_limit_x)
    ax.set_ylim(0, plot_limit_y)  # Start from y=0

    # Set background to white
    ax.set_facecolor('white')

    # Fill in the clad structure area
    clad_color = get_material_color(clad_type)
    assembly_width = fuel_plate_width + 2 * clad_structure_width
    clad_fill = Rectangle((-assembly_width / 2, 0), assembly_width, total_height, color=clad_color)
    ax.add_patch(clad_fill)

    # Repeat the plate and coolant structure for each plate in the assembly
    for i in range(plates_per_assembly):
        y_offset = clad_structure_width*2 + i * fuel_plate_pitch  # Start above the bottom clad structure

        # Plot coolant above and below each plate
        coolant_top = Rectangle((-fuel_plate_width / 2, y_offset + fuel_plate_thickness / 2),
                                fuel_plate_width, coolant_thickness, color=get_material_color(coolant_type))
        coolant_bottom = Rectangle((-fuel_plate_width / 2, y_offset - fuel_plate_thickness / 2 - coolant_thickness),
                                   fuel_plate_width, coolant_thickness, color=get_material_color(coolant_type))
        ax.add_patch(coolant_top)
        ax.add_patch(coolant_bottom)

        # Plot the plate components
        plot_plate_at_position(ax, y_offset, inputs_dict)

    ax.set_aspect('equal')
    ax.set_title('Fuel Plate Assembly Cross Section')
    ax.set_xlabel('Distance (cm)')
    ax.set_ylabel('Height (cm)')

    # Add legend
    fuel_patch = Patch(color=get_material_color(fuel_type), label=f'Fuel ({fuel_type})')
    clad_patch = Patch(color=get_material_color(clad_type), label=f'Cladding ({clad_type})')
    coolant_patch = Patch(color=get_material_color(coolant_type), label=f'Coolant ({coolant_type})')
    ax.legend(handles=[fuel_patch, clad_patch, coolant_patch], loc='upper right')

    # Save the assembly plot
    plot_filename = os.path.join(output_dir, file_name)
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print("Plotted: Plate assembly")

def plot_plate_at_position(ax, y_offset, inputs_dict=None):
    """Plot a single fuel plate at a specified vertical position.

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on
        y_offset (float): Vertical offset position for the plate
        inputs_dict (dict, optional): Custom inputs dictionary. If None, uses the global inputs.

    Adds the fuel plate components (fuel meat, cladding) to the specified
    matplotlib axes at the given vertical position.
    """
    initialize_globals(inputs_dict)

    # Calculate the dimensions of the fuel plate components
    x0 = -fuel_plate_width / 2
    x1 = -fuel_meat_width / 2
    x2 = fuel_meat_width / 2
    x3 = fuel_plate_width / 2

    y0 = y_offset - fuel_plate_thickness / 2
    y1 = y0 + clad_thickness
    y2 = y1 + fuel_meat_thickness
    y3 = y2 + clad_thickness

    # Plot the cladding regions
    clad_color = get_material_color(clad_type)
    fuel_color = get_material_color(fuel_type)

    # Plot left cladding
    left_clad = Rectangle((x0, y0), x1 - x0, y3 - y0, color=clad_color)
    ax.add_patch(left_clad)

    # Plot right cladding
    right_clad = Rectangle((x2, y0), x3 - x2, y3 - y0, color=clad_color)
    ax.add_patch(right_clad)

    # Plot top cladding
    top_clad = Rectangle((x1, y2), x2 - x1, y3 - y2, color=clad_color)
    ax.add_patch(top_clad)

    # Plot bottom cladding
    bottom_clad = Rectangle((x1, y0), x2 - x1, y1 - y0, color=clad_color)
    ax.add_patch(bottom_clad)

    # Plot fuel meat region
    fuel_meat = Rectangle((x1, y1), x2 - x1, y2 - y1, color=fuel_color)
    ax.add_patch(fuel_meat)

def plot_pin_assembly(output_dir, file_name='assembly_plot_pin.png', inputs_dict=None):
    """Plot an entire pin assembly cross section.

    Args:
        output_dir (str): Directory to save the plot
        file_name (str, optional): Name of the output file. Defaults to 'assembly_plot_pin.png'.
        inputs_dict (dict, optional): Custom inputs dictionary. If None, uses the global inputs.
    """
    initialize_globals(inputs_dict)

    # Create figure and axis for pin assembly plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set plot limits based on pin pitch and number of pins
    assembly_size = n_side_pins * pin_pitch
    plot_limit = assembly_size / 2 * 1.1  # Extended plot limits
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)

    # Plot background
    ax.set_facecolor('white')

    # Plot the entire assembly area as coolant first
    coolant_area = Rectangle((-assembly_size/2, -assembly_size/2), assembly_size, assembly_size,
                             color=get_material_color(coolant_type))
    ax.add_patch(coolant_area)

    # Plot all pins
    for i in range(n_side_pins):
        for j in range(n_side_pins):
            x = (i - n_side_pins / 2 + 0.5) * pin_pitch
            y = (j - n_side_pins / 2 + 0.5) * pin_pitch

            # Plot fuel pin
            clad = Circle((x, y), r_clad_outer, color=get_material_color(clad_type))
            gap = Circle((x, y), r_clad_inner, color='white')
            fuel = Circle((x, y), r_fuel, color=get_material_color(fuel_type))
            ax.add_patch(clad)
            ax.add_patch(gap)
            ax.add_patch(fuel)

    ax.set_aspect('equal')
    ax.set_title('Pin Assembly')
    ax.set_xlabel('Distance (cm)')
    ax.set_ylabel('Distance (cm)')

    # Add legend
    fuel_patch = Patch(color=get_material_color(fuel_type), label=f'Fuel ({fuel_type})')
    clad_patch = Patch(color=get_material_color(clad_type), label=f'Cladding ({clad_type})')
    gap_patch = Patch(color='white', label=f'Gap')
    coolant_patch = Patch(color=get_material_color(coolant_type), label=f'Coolant ({coolant_type})')
    ax.legend(handles=[fuel_patch, clad_patch, gap_patch, coolant_patch], loc='upper right')

    # Save the pin assembly plot
    plot_filename = os.path.join(output_dir, file_name)
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print("Plotted: Pin assembly")

def main():
    """Main function for testing geometry plots."""
    # Use global inputs for standalone execution
    from inputs import inputs
    initialize_globals(inputs)
    # Create output directory
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    output_dir = os.path.join(root_dir, 'ThermalHydraulics', output_folder, 'geometry_plots')
    os.makedirs(output_dir, exist_ok=True)

    if inputs.get("assembly_type") == "Pin":
        plot_pin(output_dir, file_name='single_channel_pin.png', inputs_dict=inputs)
        plot_pin_assembly(output_dir, file_name='assembly_plot_pin.png', inputs_dict=inputs)
    elif inputs.get("assembly_type") == "Plate":
        plot_plate(output_dir, file_name='single_channel_plate.png', inputs_dict=inputs)
        plot_plate_assembly(output_dir, file_name='assembly_plot_plate.png', inputs_dict=inputs)

if __name__ == "__main__":
    main()
