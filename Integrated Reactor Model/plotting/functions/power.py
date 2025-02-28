"""Functions for plotting power distributions."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        in format (x_pos, x_powers, x_linear, y_pos, y_powers, y_linear)
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
    x_pos, x_powers, x_linear = [], [], []
    y_pos, y_powers, y_linear = [], [], []

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

            # Y direction (x ≈ 0)
            if j == center_j:
                y_pos.append(y)
                y_powers.append(data['total_power'])
                y_linear.append(np.mean(data['axial_distribution']))
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

            for y_pos_m, data in sorted(y_elements.items()):
                y_pos.append(y_pos_m)
                y_powers.append(data['total_power'])
                y_linear.append(np.mean(data['axial_distribution']))

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

            for y_pos_m, data in sorted(y_elements.items()):
                y_pos.append(y_pos_m)
                y_powers.append(data['total_power'])
                y_linear.append(np.mean(data['axial_distribution']))

    # Convert lists to tuples for return
    if len(x_pos) > 0:
        x_pos, x_powers, x_linear = zip(*sorted(zip(x_pos, x_powers, x_linear)))
    if len(y_pos) > 0:
        y_pos, y_powers, y_linear = zip(*sorted(zip(y_pos, y_powers, y_linear)))

    return (x_pos, x_powers, x_linear,
            y_pos, y_powers, y_linear)

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
    (x_pos, x_powers, x_linear,
     y_pos, y_powers, y_linear) = get_radial_profiles(element_data, inputs['core_lattice'], is_element_level)

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
        ax_linear.plot(x_pos, x_linear, 'b--', label='X Direction (Avg)')
    if len(y_pos) > 0:
        ax_linear.plot(y_pos, y_linear, 'r--', label='Y Direction (Avg)')

    # Add midplane values if we have enough elements
    if len(element_data) > 0:
        center_z = n_segments // 2

        if is_element_level:
            # For element-level data, we need to filter differently
            if inputs['assembly_type'] == 'Pin':
                # Get pins along center row
                x_linear_mid = []
                for x_p in x_pos:
                    for (i, j, pin_i, pin_j), data in element_data.items():
                        # Calculate global position
                        assembly_x = (j - len(core_layout[0])/2 + 0.5) * (inputs['pin_pitch'] * inputs['n_side_pins'])
                        pin_x = (pin_j - inputs['n_side_pins']/2 + 0.5) * inputs['pin_pitch']
                        global_x = assembly_x + pin_x

                        if abs(global_x - x_p) < 0.0001:  # Close enough to be the same point
                            x_linear_mid.append(data['axial_distribution'][center_z])
                            break

                # Get pins along center column
                y_linear_mid = []
                for y_p in y_pos:
                    for (i, j, pin_i, pin_j), data in element_data.items():
                        # Calculate global position
                        assembly_y = (i - len(core_layout)/2 + 0.5) * (inputs['pin_pitch'] * inputs['n_side_pins'])
                        pin_y = (pin_i - inputs['n_side_pins']/2 + 0.5) * inputs['pin_pitch']
                        global_y = assembly_y + pin_y

                        if abs(global_y - y_p) < 0.0001:  # Close enough to be the same point
                            y_linear_mid.append(data['axial_distribution'][center_z])
                            break
            else:
                # For plates, which are stacked in y-direction
                # Get center plates along x-axis
                x_linear_mid = []
                for x_p in x_pos:
                    for (i, j, plate_k), data in element_data.items():
                        # Calculate global position
                        assembly_x = (j - len(core_layout[0])/2 + 0.5) * (inputs['fuel_plate_width'] + 2*inputs['clad_structure_width'])

                        if abs(assembly_x - x_p) < 0.0001:  # Close enough to be the same point
                            x_linear_mid.append(data['axial_distribution'][center_z])
                            break

                # Get all plates along y-axis
                y_linear_mid = []
                for y_p in y_pos:
                    for (i, j, plate_k), data in element_data.items():
                        # Calculate global position
                        assembly_y = (i - len(core_layout)/2 + 0.5) * (inputs['fuel_plate_width'] + 2*inputs['clad_structure_width'])
                        plate_y = (plate_k - inputs['plates_per_assembly']/2 + 0.5) * inputs['fuel_plate_pitch']
                        global_y = assembly_y + plate_y

                        if abs(global_y - y_p) < 0.0001:  # Close enough to be the same point
                            y_linear_mid.append(data['axial_distribution'][center_z])
                            break
        else:
            # Original assembly-level implementation
            x_linear_mid = [data['axial_distribution'][center_z]
                           for (i, j), data in element_data.items()
                           if i == len(core_layout) // 2]
            y_linear_mid = [data['axial_distribution'][center_z]
                           for (i, j), data in element_data.items()
                           if j == len(core_layout[0]) // 2]

        if len(x_pos) > 0 and len(x_linear_mid) > 0:
            ax_linear.plot(x_pos, x_linear_mid, 'b-', label='X Direction (Midplane)')
        if len(y_pos) > 0 and len(y_linear_mid) > 0:
            ax_linear.plot(y_pos, y_linear_mid, 'r-', label='Y Direction (Midplane)')

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

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Core Total Power: {sum(data['total_power'] for data in element_data.values()):.2f} MW")
    print(f"Average {element_type.capitalize()} Power: {np.mean([data['total_power'] for data in element_data.values()]):.2f} MW")
    print(f"Hot {element_type.capitalize()} Power: {max_power_data['total_power']:.2f} MW")
    print(f"Peak/Average Ratio: {max_power_data['total_power']/np.mean([data['total_power'] for data in element_data.values()]):.3f}")
    print(f"Peak Linear Power (per element): {np.max(max_power_data['axial_distribution']):.2f} kW/m")
    print(f"Segment height: {segment_height:.2f} cm")

    # Calculate average linear power across all elements and axial positions
    all_linear_powers = np.array([data['axial_distribution'] for data in element_data.values()])
    avg_linear_power = np.mean(all_linear_powers)
    max_linear_power = np.max(all_linear_powers)

    # Print debug values for first element and overall statistics
    first_pos = list(element_data.keys())[0]
    first_data = element_data[first_pos]
    print("\nDebug - First Element:")
    print(f"Raw total: {first_data['raw_total']:.2e}")
    print(f"Power in MW: {first_data['total_power']:.2e} MW")
    print(f"Peak linear power (per element): {np.max(first_data['axial_distribution']):.2f} kW/m")
    print("\nOverall Linear Power Statistics:")
    print(f"Average linear power per element: {avg_linear_power:.2f} kW/m")
    print(f"Maximum linear power per element: {max_linear_power:.2f} kW/m")
    print(f"Peak/Average linear power ratio: {max_linear_power/avg_linear_power:.3f}")
