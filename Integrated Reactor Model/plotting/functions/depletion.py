"""Functions for plotting depletion calculation results."""

import os
import numpy as np
import matplotlib.pyplot as plt
import openmc.deplete
from inputs import inputs
import h5py
from Reactor.materials import make_materials

def plot_depletion_results(plot_dir):
    """Plot results from depletion calculation."""

    # Function to plot a single depletion type with burnup axis
    def plot_depletion_type(depletion_type, ax1, color, label):
        # Handle core vs k-infinity calculations
        if depletion_type == 'core':
            results_file = os.path.join('depletion', 'outputs', 'core_keff', 'depletion_results.h5')
            input_flag = 'deplete_core'
        else:
            results_file = os.path.join('depletion', 'outputs', f"{depletion_type}_k∞", 'depletion_results.h5')
            input_flag = f"deplete_{depletion_type}"

        # Only try to plot if the depletion type is enabled in inputs
        if not inputs.get(input_flag, False):
            return None

        # Check if results file exists
        if not os.path.exists(results_file):
            print(f"No results file found for {depletion_type} at: {results_file}")
            return None

        try:
            # Get heavy metal mass from file
            with h5py.File(results_file, 'r') as f:
                if 'heavy_metal' in f:
                    heavy_metal_mass_g = f['heavy_metal'][()]
                    print(f"Heavy metal mass for {depletion_type}: {heavy_metal_mass_g/1000:.2f} kg")
                    heavy_metal_mass_kg = heavy_metal_mass_g / 1000
                else:
                    print(f"Warning: No heavy metal mass found for {depletion_type}")
                    return None

            # Load results and get time points
            results = openmc.deplete.Results(results_file)
            time_seconds, keff = results.get_keff()
            time_days = time_seconds / (24 * 60 * 60)  # Convert seconds to days

            # Get power density from simulation parameters file
            params_file = os.path.join(os.path.dirname(results_file), "simulation_parameters.txt")
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    for line in f:
                        if "Power density:" in line:
                            power_density = float(line.split(":")[1].split()[0])  # W/gHM
                            print(f"Power density for {depletion_type}: {power_density:.2f} W/gHM")
                            break
                    else:
                        print(f"Warning: Could not find power density in {params_file}")
                        return None
            else:
                print(f"Warning: No simulation parameters file found at {params_file}")
                return None

            # Calculate total power (W)
            total_power = power_density * heavy_metal_mass_g  # W/gHM * g = W
            print(f"Total power for {depletion_type}: {total_power/1e6:.3f} MW")

            # Calculate burnup points
            burnup = time_days * total_power/1e6 / heavy_metal_mass_kg  # MWd/kgHM

            # Plot k-effective with error bars
            line = ax1.errorbar(time_days, keff[:, 0], yerr=keff[:, 1],
                             label=label, color=color, marker='o', markersize=4,
                             capsize=3)

            # Add burnup axis on top if not already present
            if len(ax1.get_shared_x_axes().get_siblings(ax1)) == 1:  # Only ax1 is in the group
                ax2 = ax1.twiny()
                ax2.set_xlim(ax1.get_xlim())

                # Calculate evenly spaced time points
                time_min, time_max = ax1.get_xlim()
                n_ticks = min(len(time_days), 8)  # Limit number of ticks for readability
                time_ticks = np.linspace(time_min, time_max, n_ticks)

                # Calculate corresponding burnup values
                burnup_ticks = time_ticks * total_power/1e6 / heavy_metal_mass_kg

                ax2.set_xticks(time_ticks)
                ax2.set_xticklabels([f'{b:.2f}' for b in burnup_ticks])
                ax2.set_xlabel('Burnup [MWd/kgHM]')
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='left')

            ax1.set_xlabel('Time [days]')
            ax1.grid(True, alpha=0.3, linestyle='--')

            return line, burnup
        except Exception as e:
            print(f"Error plotting {depletion_type}: {str(e)}")
            return None

    # Create list to store which plots have data
    plots_with_data = []

    # Check which plots have data
    if inputs.get('deplete_core', False):
        plots_with_data.append('core')
    if inputs.get('deplete_assembly', False) or inputs.get('deplete_assembly_enhanced', False):
        plots_with_data.append('assembly')
    if inputs.get('deplete_element', False) or inputs.get('deplete_element_enhanced', False):
        plots_with_data.append('element')

    num_plots = len(plots_with_data)
    if num_plots == 0:
        print("No depletion calculations enabled in inputs")
        return False

    # Create figure with appropriate number of subplots
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 8*num_plots), squeeze=False)
    axes = axes.flatten()  # Convert to 1D array for easier indexing

    plot_idx = 0

    # Plot core results if enabled
    if 'core' in plots_with_data:
        ax = axes[plot_idx]
        lines = []

        # Plot k-effective and add fuel type to title
        core_keff = plot_depletion_type('core', ax, 'red', 'k-effective')
        if core_keff:
            lines.append(core_keff[0])
            burnup = core_keff[1]  # Get burnup values from the plot_depletion_type return

            ax.set_xlabel('Time [days]')
            ax.set_ylabel('Multiplication Factor')
            ax.set_title(f'Core Multiplication Factors ({inputs["fuel_type"]} Fuel)')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best')
        plot_idx += 1

    # Plot assembly results if enabled
    if 'assembly' in plots_with_data:
        ax = axes[plot_idx]
        lines = []
        assembly_line = plot_depletion_type('assembly', ax, 'red', 'Standard Assembly')
        if assembly_line:
            lines.append(assembly_line[0])
        assembly_enhanced_line = plot_depletion_type('assembly_enhanced', ax, 'blue', 'Enhanced Assembly')
        if assembly_enhanced_line:
            lines.append(assembly_enhanced_line[0])
        if lines:
            ax.set_xlabel('Time [days]')
            ax.set_ylabel('Multiplication Factor')
            ax.set_title(f'Assembly Multiplication Factors ({inputs["fuel_type"]} Fuel)')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best')
        plot_idx += 1

    # Plot element results if enabled
    if 'element' in plots_with_data:
        ax = axes[plot_idx]
        lines = []
        element_type = inputs['assembly_type']
        element_line = plot_depletion_type('element', ax, 'red', f'Standard {element_type}')
        if element_line:
            lines.append(element_line[0])
        element_enhanced_line = plot_depletion_type('element_enhanced', ax, 'blue', f'Enhanced {element_type}')
        if element_enhanced_line:
            lines.append(element_enhanced_line[0])
        if lines:
            ax.set_xlabel('Time [days]')
            ax.set_ylabel('Multiplication Factor')
            ax.set_title(f'{element_type} Multiplication Factors ({inputs["fuel_type"]} Fuel)')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best')
        plot_idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'depletion_results.png'),
               dpi=300, bbox_inches='tight')
    plt.close()

    # Plot nuclide concentrations if specified in inputs
    if 'depletion_nuclides' in inputs:
        # Create nuclide_depletion directory
        nuclide_plot_dir = os.path.join(plot_dir, 'nuclide_depletion')
        os.makedirs(nuclide_plot_dir, exist_ok=True)
        plot_nuclide_evolution(nuclide_plot_dir)

    return True

def plot_nuclide_evolution(plot_dir):
    """Plot nuclide evolution for each depletion type."""
    # Get active depletion types
    active_depletions = [k for k, v in inputs.items()
                        if k.startswith('deplete_') and v]
    if not active_depletions:
        return

    # Plot each depletion type separately
    for depletion_flag in active_depletions:
        depletion_type = depletion_flag.replace('deplete_', '')

        # Set up paths and labels
        if depletion_type == 'core':
            results_file = os.path.join('depletion', 'outputs', 'core_keff', 'depletion_results.h5')
            plot_title = 'Full Core'
        else:
            results_file = os.path.join('depletion', 'outputs', f"{depletion_type}_k∞", "depletion_results.h5")
            if depletion_type.startswith('assembly'):
                plot_title = 'Standard Assembly' if depletion_type == 'assembly' else 'Enhanced Assembly'
            else:  # element
                element_type = inputs['assembly_type']
                plot_title = f"{'Enhanced ' if 'enhanced' in depletion_type else 'Standard '}{element_type}"

        if not os.path.exists(results_file):
            print(f"No results file found for {depletion_type} at: {results_file}")
            continue

        try:
            results = openmc.deplete.Results(results_file)

            # Get first material
            with h5py.File(results_file, 'r') as f:
                if 'materials' not in f:
                    continue
                mat_ids = list(f['materials'].keys())
                if not mat_ids:
                    continue
                mat_id = mat_ids[0]

                # Get heavy metal mass and power for burnup calculation
                if 'heavy_metal' in f:
                    heavy_metal_mass_g = f['heavy_metal'][()]
                    heavy_metal_mass_kg = heavy_metal_mass_g / 1000
                else:
                    continue

            # Get power density from simulation parameters file
            params_file = os.path.join(os.path.dirname(results_file), "simulation_parameters.txt")
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    for line in f:
                        if "Power density:" in line:
                            power_density = float(line.split(":")[1].split()[0])  # W/gHM
                            break
                    else:
                        continue
            else:
                continue

            # Calculate total power (W)
            total_power = power_density * heavy_metal_mass_g  # W/gHM * g = W

            # Create figure with stacked subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[1, 1])

            # Plot each nuclide
            for nuc in inputs['depletion_nuclides']:
                try:
                    time_seconds, atoms = results.get_atoms(mat_id, nuc)
                    time_days = time_seconds / (24 * 60 * 60)
                    burnup = time_days * (total_power/1e6) / heavy_metal_mass_kg

                    # Linear plot
                    ax1.plot(time_days, atoms, 'o-', label=nuc, markersize=4)
                    # Log plot
                    ax2.plot(time_days, atoms, 'o-', label=nuc, markersize=4)

                except Exception as e:
                    print(f"Error plotting nuclide {nuc} for {depletion_type}: {str(e)}")
                    continue

            # Configure plots
            for ax, scale in [(ax1, 'linear'), (ax2, 'log')]:
                ax.set_xlabel('Time [days]', fontsize=10)
                ax.set_ylabel('Atomic density [atoms/b-cm]', fontsize=10)
                if scale == 'log':
                    ax.set_yscale('log')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax.tick_params(labelsize=9)

                # Add burnup axis on top
                burnup_ax = ax.twiny()
                burnup_ax.set_xlim(ax.get_xlim())

                # Calculate evenly spaced time points
                time_min, time_max = ax.get_xlim()
                n_ticks = min(len(time_days), 8)  # Limit number of ticks for readability
                time_ticks = np.linspace(time_min, time_max, n_ticks)

                # Calculate corresponding burnup values
                burnup_ticks = time_ticks * total_power/1e6 / heavy_metal_mass_kg

                burnup_ax.set_xticks(time_ticks)
                burnup_ax.set_xticklabels([f'{b:.2f}' for b in burnup_ticks], fontsize=9)
                if ax == ax1:
                    burnup_ax.set_xlabel('Burnup [MWd/kgHM]', fontsize=10)
                plt.setp(burnup_ax.get_xticklabels(), rotation=45, ha='left')
                burnup_ax.tick_params(labelsize=9)

            # Set title for entire figure with adjusted position and size
            fig.suptitle(f'{plot_title} Nuclide Evolution', y=0.98, fontsize=12)

            # Adjust layout to prevent overlapping
            plt.subplots_adjust(top=0.9, bottom=0.1, right=0.85, hspace=0.3)

            plt.savefig(os.path.join(plot_dir, f'{depletion_type}_nuclide_evolution.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error processing results for {depletion_type}: {str(e)}")
            continue

    return True
