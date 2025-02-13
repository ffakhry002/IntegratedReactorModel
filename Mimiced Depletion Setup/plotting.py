"""Functions for plotting depletion calculation results."""

import os
import numpy as np
import matplotlib.pyplot as plt
import openmc.deplete
import h5py
from inputs import inputs

def plot_depletion_results(results_file, plot_dir):
    """Plot k-effective vs time/burnup for pin cell depletion.

    Parameters
    ----------
    results_file : str
        Path to the depletion results HDF5 file
    plot_dir : str
        Directory to save the plots
    """
    if not os.path.exists(results_file):
        print(f"No results file found at: {results_file}")
        return False

    try:
        # Load results
        results = openmc.deplete.Results(results_file)

        # Get k-effective data
        time_seconds, keff = results.get_keff()
        time_days = time_seconds / (24 * 60 * 60)  # Convert seconds to days

        # Get heavy metal mass from log file
        log_file = os.path.join(plot_dir, "depletion_log.txt")
        with open(log_file, 'r') as f:
            for line in f:
                if "Heavy Metal Mass:" in line:
                    heavy_metal_mass_g = float(line.split(":")[1].split()[0])
                    heavy_metal_mass_kg = heavy_metal_mass_g / 1000
                    break

        # Get power density from inputs
        power_density = inputs['power_density']  # W/gHM

        # Calculate burnup points
        total_power = power_density * heavy_metal_mass_g  # W
        burnup = time_days * total_power/1e6 / heavy_metal_mass_kg  # MWd/kgHM

        # Create figure with two x-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot k-effective with error bars
        ax1.errorbar(time_days, keff[:, 0], yerr=2*keff[:, 1],
                    label='k-effective ± 2σ', color='blue',
                    marker='o', markersize=4, capsize=3)

        # Configure time axis
        ax1.set_xlabel('Time [days]')
        ax1.set_ylabel('k-effective')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Add burnup axis on top
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xlabel('Burnup [MWd/kgHM]')

        # Calculate nice tick positions
        time_min, time_max = ax1.get_xlim()
        n_ticks = min(len(time_days), 6)  # Limit number of ticks
        time_ticks = np.linspace(time_min, time_max, n_ticks)
        burnup_ticks = time_ticks * total_power/1e6 / heavy_metal_mass_kg

        ax2.set_xticks(time_ticks)
        ax2.set_xticklabels([f'{b:.1f}' for b in burnup_ticks])
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='left')

        # Add title and legend
        plt.title(f'Pin Cell Depletion Results\n{inputs["fuel_type"]} Fuel, {inputs["n%"]}% Enrichment')
        ax1.legend(loc='best')

        # Adjust layout and save
        plt.tight_layout()
        plot_file = os.path.join(plot_dir, 'keff_vs_burnup.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nPlot saved as: {plot_file}")
        return True

    except Exception as e:
        print(f"Error plotting results: {str(e)}")
        return False

if __name__ == "__main__":
    # When run directly, look for results in outputs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'outputs')
    results_file = os.path.join(output_dir, 'depletion_results.h5')

    if os.path.exists(results_file):
        plot_depletion_results(results_file, output_dir)
    else:
        print(f"No results file found at: {results_file}")
