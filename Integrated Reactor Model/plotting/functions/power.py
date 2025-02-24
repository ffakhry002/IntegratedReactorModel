"""Functions for plotting power distributions."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from inputs import inputs

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

    # Create power distribution plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Get hot assembly peak for normalization
    hot_assembly_peak = np.max(max_power_data['axial_distribution'])

    # Plot power distributions in kW/m on left axis
    line1 = ax1.plot(z/100, avg_assembly_power, 'b-', label='Average Assembly')
    line2 = ax1.plot(z/100, max_power_data['axial_distribution'], 'r-',
                     label=f'Hot Assembly (Row {max_power_pos[0]}, Col {max_power_pos[1]})')
    ax1.set_xlabel('Height from Core Midplane [m]')
    ax1.set_ylabel('Linear Power per Element [kW/m]')
    ax1.grid(True)

    # Create second y-axis for normalized values
    ax2 = ax1.twinx()
    ax2.plot(z/100, avg_assembly_power/hot_assembly_peak, 'b--', alpha=0.5)
    ax2.plot(z/100, max_power_data['axial_distribution']/hot_assembly_peak, 'r--', alpha=0.5)
    ax2.set_ylabel('Normalized to Peak')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right')

    plt.title('Axial Power Distribution (per Fuel Element)')
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
