"""Functions for plotting power distributions."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from inputs import inputs

def get_radial_profiles(element_data, core_layout, is_element_level=False):
    """Get radial power profiles in X and Y directions.

    Parameters
    ----------
    element_data : dict
        Dictionary containing power data (either assembly or element level)
    core_layout : list
        Core layout from inputs
    is_element_level : bool, optional
        Whether the data is at element level (True) or assembly level (False)

    Returns
    -------
    tuple
        Contains x_positions, y_positions and their corresponding power values
        in format (x_pos, x_powers, x_linear, x_linear_mid, y_pos, y_powers, y_linear, y_linear_mid)
    """
    # Convert core layout to numpy array for easier manipulation
    layout = np.array(core_layout)
    center_i = len(layout) // 2
    center_j = len(layout[0]) // 2

    # Get assembly pitch in meters
    if inputs['assembly_type'] == 'Pin':
        assembly_pitch = inputs['pin_pitch'] * inputs['n_side_pins']
    else:
        assembly_pitch = inputs['fuel_plate_width'] + 2*inputs['clad_structure_width']

    # Initialize lists for each direction
    x_pos, x_powers, x_linear, x_linear_mid = [], [], [], []
    y_pos, y_powers, y_linear, y_linear_mid = [], [], [], []

    # Find center z-index for midplane
    center_z = None
    for data in element_data.values():
        if 'axial_distribution' in data:
            center_z = len(data['axial_distribution']) // 2
            break

    if not is_element_level:
        # Assembly-level data processing (original implementation)
        for (i, j), data in element_data.items():
            # Calculate positions in meters from core center
            x = (j - center_j) * assembly_pitch
            y = (i - center_i) * assembly_pitch

            # X direction (y ≈ 0)
            if i == center_i:
                x_pos.append(x)
                x_powers.append(data['total_power'])
                x_linear.append(np.mean(data['axial_distribution']))
                if center_z is not None:
                    x_linear_mid.append(data['axial_distribution'][center_z])

            # Y direction (x ≈ 0)
            if j == center_j:
                y_pos.append(y)
                y_powers.append(data['total_power'])
                y_linear.append(np.mean(data['axial_distribution']))
                if center_z is not None:
                    y_linear_mid.append(data['axial_distribution'][center_z])
    else:
        # Element-level data processing
        if inputs['assembly_type'] == 'Pin':
            # For pin-type elements
            pin_pitch = inputs['pin_pitch']
            n_side_pins = inputs['n_side_pins']

            # Group elements by their position along central axes
            x_elements = {}  # Elements along x-axis (center row)
            y_elements = {}  # Elements along y-axis (center column)

            for (i, j, pin_i, pin_j), data in element_data.items():
                # Calculate global position in meters from core center
                assembly_x = (j - center_j) * assembly_pitch
                assembly_y = (i - center_i) * assembly_pitch

                # Calculate pin position within assembly
                pin_x = (pin_j - n_side_pins/2 + 0.5) * pin_pitch
                pin_y = (pin_i - n_side_pins/2 + 0.5) * pin_pitch

                # Global pin position
                global_x = assembly_x + pin_x
                global_y = assembly_y + pin_y

                # X direction (pins along center row)
                if abs(global_y) < pin_pitch/2:  # Close to y=0
                    x_elements[global_x] = data

                # Y direction (pins along center column)
                if abs(global_x) < pin_pitch/2:  # Close to x=0
                    y_elements[global_y] = data

            # Convert to lists and sort by position
            for x_pos_m, data in sorted(x_elements.items()):
                x_pos.append(x_pos_m)
                x_powers.append(data['total_power'])
                x_linear.append(np.mean(data['axial_distribution']))
                if center_z is not None:
                    x_linear_mid.append(data['axial_distribution'][center_z])

            for y_pos_m, data in sorted(y_elements.items()):
                y_pos.append(y_pos_m)
                y_powers.append(data['total_power'])
                y_linear.append(np.mean(data['axial_distribution']))
                if center_z is not None:
                    y_linear_mid.append(data['axial_distribution'][center_z])

        else:
            # For plate-type elements
            plate_pitch = inputs['fuel_plate_pitch']
            plates_per_assembly = inputs['plates_per_assembly']

            # Group elements by their position along central axes
            x_elements = {}  # Elements along x-axis
            y_elements = {}  # Elements along y-axis

            for (i, j, plate_k), data in element_data.items():
                # Calculate global position in meters from core center
                assembly_x = (j - center_j) * assembly_pitch
                assembly_y = (i - center_i) * assembly_pitch

                # Calculate plate position within assembly (plates are stacked in y-direction)
                plate_y = (plate_k - plates_per_assembly/2 + 0.5) * plate_pitch

                # Global plate position
                global_x = assembly_x
                global_y = assembly_y + plate_y

                # X direction (plates in center row of assemblies)
                if i == center_i:
                    # For each assembly along x-axis, take the center plate
                    center_plate_k = plates_per_assembly // 2
                    if plate_k == center_plate_k:
                        x_elements[global_x] = data

                # Y direction (all plates in center column of assemblies)
                if j == center_j:
                    y_elements[global_y] = data

            # Convert to lists and sort by position
            for x_pos_m, data in sorted(x_elements.items()):
                x_pos.append(x_pos_m)
                x_powers.append(data['total_power'])
                x_linear.append(np.mean(data['axial_distribution']))
                if center_z is not None:
                    x_linear_mid.append(data['axial_distribution'][center_z])

            for y_pos_m, data in sorted(y_elements.items()):
                y_pos.append(y_pos_m)
                y_powers.append(data['total_power'])
                y_linear.append(np.mean(data['axial_distribution']))
                if center_z is not None:
                    y_linear_mid.append(data['axial_distribution'][center_z])

    # Convert lists to tuples for return
    if len(x_pos) > 0:
        # Sort everything together based on x_pos
        sort_indices = np.argsort(x_pos)
        x_pos = [x_pos[i] for i in sort_indices]
        x_powers = [x_powers[i] for i in sort_indices]
        x_linear = [x_linear[i] for i in sort_indices]
        if x_linear_mid:
            x_linear_mid = [x_linear_mid[i] for i in sort_indices]

    if len(y_pos) > 0:
        # Sort everything together based on y_pos
        sort_indices = np.argsort(y_pos)
        y_pos = [y_pos[i] for i in sort_indices]
        y_powers = [y_powers[i] for i in sort_indices]
        y_linear = [y_linear[i] for i in sort_indices]
        if y_linear_mid:
            y_linear_mid = [y_linear_mid[i] for i in sort_indices]

    return (x_pos, x_powers, x_linear, x_linear_mid,
            y_pos, y_powers, y_linear, y_linear_mid)

def plot_power_distributions(sp, plot_dir):
    """Plot power distributions and save data to CSV.

    Parameters
    ----------
    sp : openmc.StatePoint
        StatePoint file containing the tally results
    plot_dir : str
        Directory to save the plots and CSV files
    """
    print(f"\nStarting power distribution plotting...")

    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Get core dimensions
    half_height = inputs['fuel_height'] * 50  # Convert to cm

    # Get total core power in MW
    power_mw = inputs.get('core_power', 1.0)
    print(f"Total core power: {power_mw:.2f} MW")

    # Check if we're using element-level tallies
    is_element_level = inputs.get('element_level_power_tallies', False)

    # Determine tally name pattern based on tallying mode
    if is_element_level:
        if inputs['assembly_type'] == 'Pin':
            tally_pattern = "pin_power"
            element_type = "pin"
        else:
            tally_pattern = "plate_power"
            element_type = "plate"
    else:
        tally_pattern = "assembly_power"
        element_type = "assembly"

    # Dictionary to store power data
    element_data = {}
    total_raw_power = 0  # Sum of all raw tally values

    # First pass: collect all raw tally data and sum
    core_layout = inputs['core_lattice']

    # Get all tallies that match our pattern
    matching_tallies = [tally for tally in sp.tallies.values()
                        if tally.name and tally_pattern in tally.name]

    for tally in matching_tallies:
        # Extract indices from tally name
        parts = tally.name.split('_')

        if is_element_level:
            if inputs['assembly_type'] == 'Pin':
                # Format: pin_power_i_j_pin_i_pin_j
                i, j, pin_i, pin_j = map(int, parts[2:6])
                key = (i, j, pin_i, pin_j)
            else:
                # Format: plate_power_i_j_plate_k
                i, j, plate_k = map(int, parts[2:5])
                key = (i, j, plate_k)
        else:
            # Format: assembly_power_i_j
            i, j = map(int, parts[2:4])
            key = (i, j)

        # Get raw tally data
        axial_power = tally.mean.flatten()

        if 'n_segments' not in locals():
            n_segments = len(axial_power)
            z = np.linspace(-half_height, half_height, n_segments)
            segment_height = (2 * half_height) / n_segments  # Height in cm

        element_data[key] = {
            'position': core_layout[i][j],
            'raw_distribution': axial_power,
            'raw_total': np.sum(axial_power)
        }
        total_raw_power += np.sum(axial_power)

    # Calculate scaling factor to convert raw values to MW
    power_scale = power_mw / total_raw_power
    print(f"Power scaling factor: {power_scale:.2e}")

    # Second pass: scale powers and calculate linear powers
    for pos, data in element_data.items():
        # Step 1: Scale raw values to get actual power in MW for each segment
        power_mw_dist = data['raw_distribution'] * power_scale  # [MW]

        # Step 2: Convert segment power to linear power
        # First get power per unit length in MW/cm
        power_per_cm = power_mw_dist / segment_height  # [MW/cm]

        # For element-level tallies, each tally is already for a single element
        # For assembly-level tallies, we need to divide by number of elements per assembly
        if not is_element_level:
            # Get number of fuel elements per assembly
            if inputs['assembly_type'] == 'Plate':
                n_elements = inputs['plates_per_assembly']
            else:  # Pin type
                n_elements = inputs['n_side_pins']**2 - inputs['n_guide_tubes']

            # Divide by number of elements to get power per element
            power_per_cm = power_per_cm / n_elements

        # Convert MW/cm to kW/m:
        # 1) Convert MW to kW: multiply by 1000 [kW/cm]
        # 2) Convert per cm to per m: multiply by 100 [kW/m]
        linear_power = power_per_cm * 1000 * 100  # [kW/m per element]

        # Store results
        element_data[pos].update({
            'axial_distribution': linear_power,  # [kW/m per element]
            'total_power': np.sum(power_mw_dist)  # [MW]
        })

    # Find element with maximum power
    max_power_element = max(element_data.items(), key=lambda x: x[1]['total_power'])
    max_power_pos = max_power_element[0]
    max_power_data = max_power_element[1]

    # Calculate core total and average element power
    core_total = np.zeros(n_segments)
    avg_element_power = np.zeros(n_segments)
    for data in element_data.values():
        core_total += data['axial_distribution']
        avg_element_power += data['axial_distribution']
    avg_element_power /= len(element_data)

    # Create DataFrame for CSV
    # First create the z-position column
    df_dict = {'z_position_cm': z}

    # Add columns for each element
    for pos, data in element_data.items():
        if is_element_level:
            if inputs['assembly_type'] == 'Pin':
                i, j, pin_i, pin_j = pos
                col_name = f"Pin_{i}_{j}_{pin_i}_{pin_j}_{data['position']}"
            else:
                i, j, plate_k = pos
                col_name = f"Plate_{i}_{j}_{plate_k}_{data['position']}"
        else:
            i, j = pos
            col_name = f"Assembly_{i}_{j}_{data['position']}"

        df_dict[col_name] = data['axial_distribution']

    # Add summary columns
    df_dict['Core_Total'] = core_total
    df_dict[f'Average_{element_type.capitalize()}'] = avg_element_power

    # Add hot element column with appropriate name
    if is_element_level:
        if inputs['assembly_type'] == 'Pin':
            i, j, pin_i, pin_j = max_power_pos
            hot_col_name = f'Hot_Pin_{i}_{j}_{pin_i}_{pin_j}'
        else:
            i, j, plate_k = max_power_pos
            hot_col_name = f'Hot_Plate_{i}_{j}_{plate_k}'
    else:
        i, j = max_power_pos
        hot_col_name = f'Hot_Assembly_{i}_{j}'

    df_dict[hot_col_name] = max_power_data['axial_distribution']

    # Create DataFrame
    df = pd.DataFrame(df_dict)

    # Add total powers as a new row
    totals = {'z_position_cm': 'Total Power (MW)'}
    for pos, data in element_data.items():
        if is_element_level:
            if inputs['assembly_type'] == 'Pin':
                i, j, pin_i, pin_j = pos
                col_name = f"Pin_{i}_{j}_{pin_i}_{pin_j}_{data['position']}"
            else:
                i, j, plate_k = pos
                col_name = f"Plate_{i}_{j}_{plate_k}_{data['position']}"
        else:
            i, j = pos
            col_name = f"Assembly_{i}_{j}_{data['position']}"

        totals[col_name] = data['total_power']

    totals['Core_Total'] = sum(data['total_power'] for data in element_data.values())
    totals[f'Average_{element_type.capitalize()}'] = totals['Core_Total'] / len(element_data)
    totals[hot_col_name] = max_power_data['total_power']

    # Append totals row
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    # Save to CSV
    csv_path = os.path.join(plot_dir, f'detailed_{element_type}_power_distribution.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed power distribution to: {csv_path}")

    # Get radial profiles
    (x_pos, x_powers, x_linear, x_linear_mid,
     y_pos, y_powers, y_linear, y_linear_mid) = get_radial_profiles(element_data, inputs['core_lattice'], is_element_level)

    # Create figure with three subplots
    if n_segments > 20 and n_segments % 20 == 0:
        fig = plt.figure(figsize=(20, 12))
        gs = plt.GridSpec(2, 2, figure=fig)
        ax_radial = fig.add_subplot(gs[0, 0])
        ax_linear = fig.add_subplot(gs[0, 1])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
    else:
        fig = plt.figure(figsize=(20, 8))
        gs = plt.GridSpec(1, 3, figure=fig)
        ax_radial = fig.add_subplot(gs[0, 0])
        ax_linear = fig.add_subplot(gs[0, 1])
        ax1 = fig.add_subplot(gs[0, 2])

    # Plot 1: Element powers (X and Y directions)
    if len(x_pos) > 0:
        ax_radial.plot(x_pos, x_powers, 'bo-', label='X Direction')
    if len(y_pos) > 0:
        ax_radial.plot(y_pos, y_powers, 'ro-', label='Y Direction')
    ax_radial.set_xlabel('Distance from Core Center [m]')
    ax_radial.set_ylabel(f'{element_type.capitalize()} Power [MW]')
    ax_radial.set_title(f'Radial Power Distribution ({element_type.capitalize()} Level)')
    ax_radial.grid(True)
    ax_radial.legend()

    # Plot 2: Linear powers comparison
    if len(x_pos) > 0:
        ax_linear.plot(x_pos, x_linear, 'b-', label='X Direction (Avg)')
    if len(y_pos) > 0:
        ax_linear.plot(y_pos, y_linear, 'r-', label='Y Direction (Avg)')

    # Add midplane values if available
    if len(x_pos) > 0 and len(x_linear_mid) > 0:
        ax_linear.plot(x_pos, x_linear_mid, 'b--', label='X Direction (Midplane)')
    if len(y_pos) > 0 and len(y_linear_mid) > 0:
        ax_linear.plot(y_pos, y_linear_mid, 'r--', label='Y Direction (Midplane)')

    ax_linear.set_xlabel('Distance from Core Center [m]')
    ax_linear.set_ylabel('Linear Power per Element [kW/m]')
    ax_linear.set_title(f'Radial Linear Power Distribution ({element_type.capitalize()} Level)')
    ax_linear.grid(True)
    ax_linear.legend()

    # Get hot element peak for normalization
    hot_element_peak = np.max(max_power_data['axial_distribution'])

    # Plot power distributions in kW/m on left axis of first plot
    line1 = ax1.plot(z/100, avg_element_power, 'b-', label=f'Average {element_type.capitalize()}')

    # Add hot element label with appropriate position info
    if is_element_level:
        if inputs['assembly_type'] == 'Pin':
            i, j, pin_i, pin_j = max_power_pos
            hot_label = f'Hot Pin (Assembly {i},{j}, Pin {pin_i},{pin_j})'
        else:
            i, j, plate_k = max_power_pos
            hot_label = f'Hot Plate (Assembly {i},{j}, Plate {plate_k})'
    else:
        i, j = max_power_pos
        hot_label = f'Hot Assembly (Row {i}, Col {j})'

    line2 = ax1.plot(z/100, max_power_data['axial_distribution'], 'r-', label=hot_label)
    ax1.set_xlabel('Height from Core Midplane [m]')
    ax1.set_ylabel('Linear Power per Element [kW/m]')
    ax1.grid(True)

    # Create second y-axis for normalized values
    ax1_norm = ax1.twinx()
    ax1_norm.plot(z/100, avg_element_power/hot_element_peak, 'b--', alpha=0.5)
    ax1_norm.plot(z/100, max_power_data['axial_distribution']/hot_element_peak, 'r--', alpha=0.5)
    ax1_norm.set_ylabel('Normalized to Peak')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right')

    ax1.set_title(f'Axial Power Distribution (per {element_type.capitalize()})')

    if n_segments > 20 and n_segments % 20 == 0:
        # Calculate number of segments to combine
        combine_factor = n_segments // 20

        # Create new z positions for coarser mesh
        z_coarse = np.linspace(-half_height, half_height, 20)

        # Function to combine segments
        def combine_segments(data):
            return np.array([np.mean(data[i:i+combine_factor])
                           for i in range(0, len(data), combine_factor)])

        # Combine data for average and hot element
        avg_element_power_coarse = combine_segments(avg_element_power)
        hot_element_power_coarse = combine_segments(max_power_data['axial_distribution'])

        # Plot coarse data
        line1 = ax2.plot(z_coarse/100, avg_element_power_coarse, 'b-', label=f'Average {element_type.capitalize()}')
        line2 = ax2.plot(z_coarse/100, hot_element_power_coarse, 'r-', label=hot_label)
        ax2.set_xlabel('Height from Core Midplane [m]')
        ax2.set_ylabel('Linear Power per Element [kW/m]')
        ax2.grid(True)

        # Set y-axis limits for power plot
        min_power = min(np.min(avg_element_power_coarse), np.min(hot_element_power_coarse))
        max_power = max(np.max(avg_element_power_coarse), np.max(hot_element_power_coarse))
        power_range = max_power - min_power
        ax2.set_ylim([min_power - 0.2*power_range, max_power + 0.2*power_range])

        # Create second y-axis for hot element / core average ratio only
        ax2_norm = ax2.twinx()
        ratio = hot_element_power_coarse / avg_element_power_coarse
        ratio_line = ax2_norm.plot(z_coarse/100, ratio, 'g-', label=f'Hot {element_type.capitalize()} / Core Average', linewidth=2)
        ax2_norm.set_ylabel(f'Hot {element_type.capitalize()} / Core Average Ratio')

        # Set y-axis limits for ratio plot with fixed minimum of 0.9 and max of 2.0 or higher if needed
        ratio_max = max(2.0, np.max(ratio))
        ax2_norm.set_ylim([0.9, ratio_max])

        # Add legends
        lines2, labels2 = ax2.get_legend_handles_labels()
        ratio_lines, ratio_labels = ax2_norm.get_legend_handles_labels()
        ax2.legend(lines2 + ratio_lines, labels2 + ratio_labels, loc='upper right')

        ax2.set_title('Coarse (20 segments) Power Distribution')

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f'{element_type}_power_distributions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved power distribution plot to: {plot_path}")


def plot_2d_power_map(sp, plot_dir):
    """Create a 2D power distribution heat map visualization of the reactor core.

    This function uses the CSV data created by plot_power_distributions to ensure
    consistent power values between all plots.
    """
    print("\nCreating power distribution heat map...")
    os.makedirs(plot_dir, exist_ok=True)

    # Get basic parameters
    power_mw = inputs.get('core_power', 1.0)
    is_element_level = inputs.get('element_level_power_tallies', False)

    if is_element_level:
        if inputs['assembly_type'] == 'Pin':
            element_type = "pin"
        else:
            element_type = "plate"
    else:
        element_type = "assembly"

    # Read power values from CSV file that was generated by plot_power_distributions
    csv_path = os.path.join(plot_dir, f'detailed_{element_type}_power_distribution.csv')

    try:
        # Check if CSV file exists
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            print("Please run plot_power_distributions first.")
            return {}

        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Get the total power values from the last row
        power_row = df.iloc[-1]  # Last row contains total powers

        # Extract element positions and powers
        element_data = {}
        i_indices = []
        j_indices = []
        pin_indices = []
        powers = []

        # Process the data for different element types
        for col in df.columns:
            # Skip non-element columns
            if col == 'z_position_cm' or col == 'Core_Total' or col.startswith('Average_') or col.startswith('Hot_'):
                continue

            # Extract position info from column name
            if is_element_level:
                if inputs['assembly_type'] == 'Pin':
                    # Format: Pin_i_j_pin_i_pin_j_position
                    parts = col.split('_')
                    i, j, pin_i, pin_j = map(int, parts[1:5])
                    position = parts[5]
                    key = (i, j, pin_i, pin_j)

                    element_data[key] = {
                        'i': i, 'j': j,
                        'pin_i': pin_i, 'pin_j': pin_j,
                        'position': position,
                        'total_power': power_row[col]
                    }

                    i_indices.append(i)
                    j_indices.append(j)
                    pin_indices.append((pin_i, pin_j))
                    powers.append(power_row[col])
                else:
                    # Format: Plate_i_j_plate_k_position
                    parts = col.split('_')
                    i, j, plate_k = map(int, parts[1:4])
                    position = parts[4]
                    key = (i, j, plate_k)

                    element_data[key] = {
                        'i': i, 'j': j,
                        'plate_k': plate_k,
                        'position': position,
                        'total_power': power_row[col]
                    }

                    i_indices.append(i)
                    j_indices.append(j)
                    pin_indices.append(plate_k)
                    powers.append(power_row[col])
            else:
                # Format: Assembly_i_j_position
                parts = col.split('_')
                i, j = map(int, parts[1:3])
                position = parts[3]
                key = (i, j)

                element_data[key] = {
                    'i': i, 'j': j,
                    'position': position,
                    'total_power': power_row[col]
                }

                i_indices.append(i)
                j_indices.append(j)
                powers.append(power_row[col])

        # Find and report the hot element for debugging
        if powers:
            max_power_idx = np.argmax(powers)
            max_power = powers[max_power_idx]
            max_i = i_indices[max_power_idx]
            max_j = j_indices[max_power_idx]

            if is_element_level:
                if inputs['assembly_type'] == 'Pin':
                    max_pin = pin_indices[max_power_idx]
                    print(f"\nHot Pin: Assembly {max_i},{max_j}, Pin {max_pin[0]},{max_pin[1]}")
                else:
                    max_plate = pin_indices[max_power_idx]
                    print(f"\nHot Plate: Assembly {max_i},{max_j}, Plate {max_plate}")
            else:
                print(f"\nHot Assembly: {max_i},{max_j}")
        # Create the heat map visualization
        try:
            # Setup plot parameters
            min_power, max_power = min(powers), max(powers)
            if max_power <= 0:
                print("Maximum power is zero or negative. Skipping heat map.")
                return element_data

            norm = plt.Normalize(min_power, max_power)
            cmap = plt.cm.viridis
            fig = plt.figure(figsize=(12, 10), facecolor='white')
            ax = fig.add_subplot(111)

            # Plot specific heatmap based on element type
            if is_element_level and inputs['assembly_type'] == 'Pin':
                # Pin-type elements
                n_side_pins = inputs['n_side_pins']
                max_i, max_j = max(i_indices) + 1, max(j_indices) + 1

                heatmap_data = np.full((max_i*n_side_pins, max_j*n_side_pins), np.nan)
                for idx, (i, j, pin, power) in enumerate(zip(i_indices, j_indices, pin_indices, powers)):
                    pin_i, pin_j = pin
                    heatmap_data[i*n_side_pins + pin_i, j*n_side_pins + pin_j] = power

                masked_heatmap = np.ma.masked_invalid(heatmap_data)
                heat_map = ax.pcolormesh(
                    np.arange(0, max_j*n_side_pins + 1),
                    np.arange(0, max_i*n_side_pins + 1),
                    masked_heatmap, cmap=cmap, norm=norm,
                    edgecolors='white', linewidth=0.1
                )

                # Mark the hot pin
                max_idx = np.argmax(powers)
                max_i, max_j = i_indices[max_idx], j_indices[max_idx]
                max_pin_i, max_pin_j = pin_indices[max_idx]
                global_i = max_i * n_side_pins + max_pin_i
                global_j = max_j * n_side_pins + max_pin_j
                ax.plot(global_j + 0.5, global_i + 0.5, 'r*', markersize=12,
                       markeredgecolor='black', markeredgewidth=1)

            elif is_element_level:
                # Plate-type elements
                plates_per_assembly = inputs['plates_per_assembly']
                fuel_plate_width = inputs['fuel_plate_width'] * 100
                fuel_plate_pitch = inputs['fuel_plate_pitch'] * 100

                assembly_positions = set((data['i'], data['j']) for data in element_data.values())
                max_i = max(pos[0] for pos in assembly_positions) + 1
                max_j = max(pos[1] for pos in assembly_positions) + 1

                # Create position mapping for hot plate marking
                plate_positions = {}

                for i, j in assembly_positions:
                    assembly_plates = [k for k in element_data.keys()
                                     if isinstance(k, tuple) and len(k) > 2 and k[0] == i and k[1] == j]

                    for plate_key in assembly_plates:
                        plate_k = plate_key[2]
                        power = element_data[plate_key]['total_power']  # Using total_power from CSV

                        rel_x = j
                        rel_y = i + (plate_k - plates_per_assembly/2 + 0.5) * fuel_plate_pitch / (fuel_plate_width * 1.3)

                        # Store position for potential hot plate marking
                        plate_positions[plate_key] = (rel_x, rel_y)

                        rect = plt.Rectangle(
                            (rel_x - 0.45, rel_y - 0.4 * fuel_plate_pitch / (fuel_plate_width * 1.3)),
                            0.9, 0.8 * fuel_plate_pitch / (fuel_plate_width * 1.3),
                            facecolor=cmap(norm(power)),
                            edgecolor='white',
                            linewidth=0.5
                        )
                        ax.add_patch(rect)

                # Mark the hot plate with a star
                max_idx = np.argmax(powers)
                key_list = list(element_data.keys())
                hot_plate_key = key_list[max_idx]
                if hot_plate_key in plate_positions:
                    x_pos, y_pos = plate_positions[hot_plate_key]
                    ax.plot(x_pos, y_pos, 'r*', markersize=12,
                           markeredgecolor='black', markeredgewidth=1)

                ax.set_xlim(-0.5, max_j - 0.5)
                ax.set_ylim(-0.5, max_i + 0.5)

            else:
                # Assembly-level
                max_i, max_j = max(i_indices) + 1, max(j_indices) + 1

                heatmap_data = np.full((max_i, max_j), np.nan)
                for i, j, power in zip(i_indices, j_indices, powers):
                    heatmap_data[i, j] = power

                masked_heatmap = np.ma.masked_invalid(heatmap_data)
                heat_map = ax.pcolormesh(
                    np.arange(0, max_j + 1),
                    np.arange(0, max_i + 1),
                    masked_heatmap, cmap=cmap, norm=norm,
                    edgecolors='black', linewidth=0.5
                )

                # Mark the hot assembly
                max_idx = np.argmax(powers)
                max_i, max_j = i_indices[max_idx], j_indices[max_idx]
                ax.plot(max_j + 0.5, max_i + 0.5, 'r*', markersize=12,
                       markeredgecolor='black', markeredgewidth=1)

            # Add colorbar and finalize plot
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Power [MW]', fontsize=12)

            # Add the hot element power value to the title
            hot_power = max(powers)
            ax.set_title(f'Power Distribution Heat Map ({element_type.capitalize()} Level)\nHot {element_type} power: {hot_power:.6f} MW', fontsize=14)
            ax.set_xlabel('Core Position', fontsize=12)
            ax.set_ylabel('Core Position', fontsize=12)
            ax.grid(False)
            ax.invert_yaxis()
            ax.set_aspect('equal')

            # Save the plot
            heatmap_path = os.path.join(plot_dir, f'power_heatmap_{element_type}.png')
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved power distribution heat map to: {heatmap_path}")

        except Exception as e:
            print(f"Error creating heat map visualization: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"Error processing CSV file: {e}")
        import traceback
        traceback.print_exc()
        return {}

    return element_data
