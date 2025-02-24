"""Functions for plotting power distributions."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from inputs import inputs

def get_radial_profiles(assembly_data, core_layout):
    """Get radial power profiles in X and Y directions.

    Parameters
    ----------
    assembly_data : dict
        Dictionary containing assembly power data
    core_layout : list
        Core layout from inputs

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

    # Collect powers along each direction
    for (i, j), data in assembly_data.items():
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

    # Sort all lists by position
    x_pos, x_powers, x_linear = zip(*sorted(zip(x_pos, x_powers, x_linear)))
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

    # Dictionary to store assembly powers
    assembly_data = {}
    total_raw_power = 0  # Sum of all raw tally values

    # First pass: collect all raw tally data and sum
    core_layout = inputs['core_lattice']
    for i, row in enumerate(core_layout):
        for j, pos in enumerate(row):
            if pos in ['F', 'E']:
                tally_name = f"assembly_power_{i}_{j}"
                print(f"Processing assembly tally: {tally_name}")
                tally = sp.get_tally(name=tally_name)

                # Get raw tally data
                axial_power = tally.mean.flatten()

                if 'n_segments' not in locals():
                    n_segments = len(axial_power)
                    z = np.linspace(-half_height, half_height, n_segments)
                    segment_height = (2 * half_height) / n_segments  # Height in cm

                assembly_data[(i, j)] = {
                    'position': pos,
                    'raw_distribution': axial_power,
                    'raw_total': np.sum(axial_power)
                }
                total_raw_power += np.sum(axial_power)

    # Calculate scaling factor to convert raw values to MW
    power_scale = power_mw / total_raw_power
    print(f"Power scaling factor: {power_scale:.2e}")

    # Second pass: scale powers and calculate linear powers
    for pos, data in assembly_data.items():
        # Step 1: Scale raw values to get actual power in MW for each segment
        power_mw_dist = data['raw_distribution'] * power_scale  # [MW]

        # Step 2: Convert segment power to linear power
        # First get power per unit length in MW/cm
        power_per_cm = power_mw_dist / segment_height  # [MW/cm]

        # Get number of fuel elements per assembly
        if inputs['assembly_type'] == 'Plate':
            n_elements = inputs['plates_per_assembly']
        else:  # Pin type
            n_elements = inputs['n_side_pins']**2 - inputs['n_guide_tubes']

        print(f"\nNumber of fuel elements per assembly: {n_elements}")

        # Convert MW/cm to kW/m:
        # 1) Convert MW to kW: multiply by 1000 [kW/cm]
        # 2) Convert per cm to per m: multiply by 100 [kW/m]
        # 3) Divide by number of fuel elements to get power per element
        linear_power = power_per_cm * 1000 * 100 / n_elements  # [kW/m per element]

        # Store results
        assembly_data[pos].update({
            'axial_distribution': linear_power,  # [kW/m per element]
            'total_power': np.sum(power_mw_dist)  # [MW]
        })

    # Find assembly with maximum power
    max_power_assembly = max(assembly_data.items(), key=lambda x: x[1]['total_power'])
    max_power_pos = max_power_assembly[0]
    max_power_data = max_power_assembly[1]

    # Calculate core total and average assembly power
    core_total = np.zeros(n_segments)
    avg_assembly_power = np.zeros(n_segments)
    for data in assembly_data.values():
        core_total += data['axial_distribution']
        avg_assembly_power += data['axial_distribution']
    avg_assembly_power /= len(assembly_data)

    # Create DataFrame for CSV
    # First create the z-position column
    df_dict = {'z_position_cm': z}

    # Add columns for each assembly
    for (i, j), data in assembly_data.items():
        col_name = f"Assembly_{i}_{j}_{data['position']}"
        df_dict[col_name] = data['axial_distribution']

    # Add summary columns
    df_dict['Core_Total'] = core_total
    df_dict['Assembly_Average'] = avg_assembly_power
    df_dict[f'Hot_Assembly_{max_power_pos[0]}_{max_power_pos[1]}'] = max_power_data['axial_distribution']

    # Create DataFrame
    df = pd.DataFrame(df_dict)

    # Add total powers as a new row
    totals = {'z_position_cm': 'Total Power (MW)'}
    for (i, j), data in assembly_data.items():
        totals[f"Assembly_{i}_{j}_{data['position']}"] = data['total_power']
    totals['Core_Total'] = sum(data['total_power'] for data in assembly_data.values())
    totals['Assembly_Average'] = totals['Core_Total'] / len(assembly_data)
    totals[f'Hot_Assembly_{max_power_pos[0]}_{max_power_pos[1]}'] = max_power_data['total_power']

    # Append totals row
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    # Save to CSV
    csv_path = os.path.join(plot_dir, 'detailed_power_distribution.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed power distribution to: {csv_path}")

    # Get radial profiles
    (x_pos, x_powers, x_linear,
     y_pos, y_powers, y_linear) = get_radial_profiles(assembly_data, inputs['core_lattice'])

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

    # Plot 1: Assembly powers (X and Y directions)
    ax_radial.plot(x_pos, x_powers, 'bo-', label='X Direction')
    ax_radial.plot(y_pos, y_powers, 'ro-', label='Y Direction')
    ax_radial.set_xlabel('Distance from Core Center [m]')
    ax_radial.set_ylabel('Assembly Power [MW]')
    ax_radial.set_title('Radial Power Distribution')
    ax_radial.grid(True)
    ax_radial.legend()

    # Plot 2: Linear powers comparison
    ax_linear.plot(x_pos, x_linear, 'b--', label='X Direction (Avg)')
    ax_linear.plot(y_pos, y_linear, 'r--', label='Y Direction (Avg)')

    # Add midplane values
    center_z = n_segments // 2
    x_linear_mid = [data['axial_distribution'][center_z]
                   for (i, j), data in assembly_data.items()
                   if i == len(core_layout) // 2]
    y_linear_mid = [data['axial_distribution'][center_z]
                   for (i, j), data in assembly_data.items()
                   if j == len(core_layout[0]) // 2]

    ax_linear.plot(x_pos, x_linear_mid, 'b-', label='X Direction (Midplane)')
    ax_linear.plot(y_pos, y_linear_mid, 'r-', label='Y Direction (Midplane)')
    ax_linear.set_xlabel('Distance from Core Center [m]')
    ax_linear.set_ylabel('Linear Power per Element [kW/m]')
    ax_linear.set_title('Radial Linear Power Distribution')
    ax_linear.grid(True)
    ax_linear.legend()

    # Get hot assembly peak for normalization
    hot_assembly_peak = np.max(max_power_data['axial_distribution'])

    # Plot power distributions in kW/m on left axis of first plot
    line1 = ax1.plot(z/100, avg_assembly_power, 'b-', label='Average Assembly')
    line2 = ax1.plot(z/100, max_power_data['axial_distribution'], 'r-',
                     label=f'Hot Assembly (Row {max_power_pos[0]}, Col {max_power_pos[1]})')
    ax1.set_xlabel('Height from Core Midplane [m]')
    ax1.set_ylabel('Linear Power per Element [kW/m]')
    ax1.grid(True)

    # Create second y-axis for normalized values
    ax1_norm = ax1.twinx()
    ax1_norm.plot(z/100, avg_assembly_power/hot_assembly_peak, 'b--', alpha=0.5)
    ax1_norm.plot(z/100, max_power_data['axial_distribution']/hot_assembly_peak, 'r--', alpha=0.5)
    ax1_norm.set_ylabel('Normalized to Peak')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right')

    ax1.set_title('Axial Power Distribution (per Fuel Element)')

    if n_segments > 20 and n_segments % 20 == 0:
        # Calculate number of segments to combine
        combine_factor = n_segments // 20

        # Create new z positions for coarser mesh
        z_coarse = np.linspace(-half_height, half_height, 20)

        # Function to combine segments
        def combine_segments(data):
            return np.array([np.mean(data[i:i+combine_factor])
                           for i in range(0, len(data), combine_factor)])

        # Combine data for average and hot assembly
        avg_assembly_power_coarse = combine_segments(avg_assembly_power)
        hot_assembly_power_coarse = combine_segments(max_power_data['axial_distribution'])

        # Plot coarse data
        line1 = ax2.plot(z_coarse/100, avg_assembly_power_coarse, 'b-', label='Average Assembly')
        line2 = ax2.plot(z_coarse/100, hot_assembly_power_coarse, 'r-',
                        label=f'Hot Assembly (Row {max_power_pos[0]}, Col {max_power_pos[1]})')
        ax2.set_xlabel('Height from Core Midplane [m]')
        ax2.set_ylabel('Linear Power per Element [kW/m]')
        ax2.grid(True)

        # Set y-axis limits for power plot
        min_power = min(np.min(avg_assembly_power_coarse), np.min(hot_assembly_power_coarse))
        max_power = max(np.max(avg_assembly_power_coarse), np.max(hot_assembly_power_coarse))
        power_range = max_power - min_power
        ax2.set_ylim([min_power - 0.2*power_range, max_power + 0.2*power_range])

        # Create second y-axis for hot assembly / core average ratio only
        ax2_norm = ax2.twinx()
        ratio = hot_assembly_power_coarse / avg_assembly_power_coarse
        ratio_line = ax2_norm.plot(z_coarse/100, ratio, 'g-', label='Hot Assembly / Core Average', linewidth=2)
        ax2_norm.set_ylabel('Hot Assembly / Core Average Ratio')

        # Set y-axis limits for ratio plot with fixed minimum of 0.9 and max of 2.0 or higher if needed
        ratio_max = max(2.0, np.max(ratio))
        ax2_norm.set_ylim([0.9, ratio_max])

        # Add legends
        lines2, labels2 = ax2.get_legend_handles_labels()
        ratio_lines, ratio_labels = ax2_norm.get_legend_handles_labels()
        ax2.legend(lines2 + ratio_lines, labels2 + ratio_labels, loc='upper right')

        ax2.set_title('Coarse (20 segments) Power Distribution')

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, 'power_distributions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved power distribution plot to: {plot_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Core Total Power: {totals['Core_Total']:.2f} MW")
    print(f"Average Assembly Power: {totals['Assembly_Average']:.2f} MW")
    print(f"Hot Assembly Power: {max_power_data['total_power']:.2f} MW")
    print(f"Peak/Average Ratio: {max_power_data['total_power']/totals['Assembly_Average']:.3f}")
    print(f"Peak Linear Power (per element): {np.max(max_power_data['axial_distribution']):.2f} kW/m")
    print(f"Segment height: {segment_height:.2f} cm")

    # Calculate average linear power across all assemblies and axial positions
    all_linear_powers = np.array([data['axial_distribution'] for data in assembly_data.values()])
    avg_linear_power = np.mean(all_linear_powers)
    max_linear_power = np.max(all_linear_powers)

    # Print debug values for first assembly and overall statistics
    first_pos = list(assembly_data.keys())[0]
    first_data = assembly_data[first_pos]
    print("\nDebug - First Assembly:")
    print(f"Raw total: {first_data['raw_total']:.2e}")
    print(f"Power in MW: {first_data['total_power']:.2e} MW")
    print(f"Peak linear power (per element): {np.max(first_data['axial_distribution']):.2f} kW/m")
    print("\nOverall Linear Power Statistics:")
    print(f"Average linear power per element: {avg_linear_power:.2f} kW/m")
    print(f"Maximum linear power per element: {max_linear_power:.2f} kW/m")
    print(f"Peak/Average linear power ratio: {max_linear_power/avg_linear_power:.3f}")
