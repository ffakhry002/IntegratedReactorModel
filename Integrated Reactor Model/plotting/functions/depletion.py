"""Functions for plotting depletion calculation results."""

import os
import numpy as np
import matplotlib.pyplot as plt
import openmc.deplete
from inputs import inputs
import h5py
from Reactor.materials import make_materials

def calculate_burnup_from_time(time_days, total_power_w, heavy_metal_mass_kg):
    return time_days * (total_power_w/1e6) / heavy_metal_mass_kg

def get_nice_tick_interval(data_min, data_max, target_ticks=10, min_ticks=8, max_ticks=15):
    data_range = data_max - data_min
    magnitude = 10 ** np.floor(np.log10(data_range/target_ticks))

    best_interval = None
    best_num_ticks = 0

    # Try current magnitude
    for multiplier in [1, 2, 2.5, 5]:
        interval = magnitude * multiplier
        num_ticks = data_range / interval
        if min_ticks <= num_ticks <= max_ticks:
            if best_interval is None or abs(num_ticks - target_ticks) < abs(best_num_ticks - target_ticks):
                best_interval = interval
                best_num_ticks = num_ticks

    # Try next smaller magnitude if needed
    if best_interval is None:
        magnitude = magnitude / 10
        for multiplier in [10, 5, 2.5, 2, 1]:
            interval = magnitude * multiplier
            num_ticks = data_range / interval
            if min_ticks <= num_ticks <= max_ticks:
                if best_interval is None or abs(num_ticks - target_ticks) < abs(best_num_ticks - target_ticks):
                    best_interval = interval
                    best_num_ticks = num_ticks

    return best_interval or (data_range / target_ticks)

def generate_tick_positions(data_min, data_max, interval):
    first_tick = np.ceil(data_min / interval) * interval
    last_tick = np.floor(data_max / interval) * interval
    num_ticks = int((last_tick - first_tick) / interval) + 1
    return np.linspace(first_tick, last_tick, num_ticks)

def setup_burnup_axis(ax, time_days, total_power_w, heavy_metal_mass_kg, fontsize=9):
    burnup_ax = ax.twiny()
    burnup_ax.set_xlim(ax.get_xlim())

    # Calculate burnup range
    time_min, time_max = ax.get_xlim()
    burnup_min = calculate_burnup_from_time(time_min, total_power_w, heavy_metal_mass_kg)
    burnup_max = calculate_burnup_from_time(time_max, total_power_w, heavy_metal_mass_kg)

    # Get nice tick interval and positions
    interval = get_nice_tick_interval(burnup_min, burnup_max)
    burnup_ticks = generate_tick_positions(burnup_min, burnup_max, interval)

    # Convert burnup ticks back to time points
    time_ticks = burnup_ticks * heavy_metal_mass_kg / (total_power_w/1e6)

    # Set ticks and labels
    burnup_ax.set_xticks(time_ticks)
    burnup_ax.set_xticklabels([f'{b:.1f}' for b in burnup_ticks], fontsize=fontsize)
    burnup_ax.set_xlabel('Burnup [MWd/kgHM]', fontsize=fontsize+1)
    plt.setp(burnup_ax.get_xticklabels(), rotation=45, ha='left')
    burnup_ax.tick_params(labelsize=fontsize)

    return burnup_ax

def load_depletion_data(results_file):
    """Load depletion calculation data from results file.

    Parameters
    ----------
    results_file : str
        Path to the depletion results file

    Returns
    -------
    tuple
        (results, heavy_metal_mass_g, power_density) or (None, None, None) if loading fails
    """
    if not os.path.exists(results_file):
        print(f"No results file found at: {results_file}")
        return None, None, None

    try:
        # Load results object
        results = openmc.deplete.Results(results_file)

        # Get heavy metal mass
        with h5py.File(results_file, 'r') as f:
            if 'heavy_metal' not in f:
                print(f"No heavy metal mass data found in {results_file}")
                return None, None, None
            heavy_metal_mass_g = f['heavy_metal'][()]

        # Get power density from parameters file
        params_file = os.path.join(os.path.dirname(results_file), "simulation_outputs.txt")
        if not os.path.exists(params_file):
            print(f"No simulation parameters file found at {params_file}")
            return None, None, None

        with open(params_file, 'r') as f:
            for line in f:
                if "Power density:" in line:
                    power_density = float(line.split(":")[1].split()[0])  # W/gHM
                    break
            else:
                print(f"No power density found in {params_file}")
                return None, None, None

        return results, heavy_metal_mass_g, power_density

    except Exception as e:
        print(f"Error loading depletion data: {str(e)}")
        return None, None, None

def plot_depletion_results(plot_dir, root_dir=None):
    """Plot results from depletion calculation.

    Parameters
    ----------
    plot_dir : str
        Directory to save plots to
    root_dir : str, optional
        Root directory of the project. If not provided, will attempt to find it.
    """
    # Get root directory if not provided
    if root_dir is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Function to plot a single depletion type with burnup axis
    def plot_depletion_type(depletion_type, ax1, color, label):
        # Handle core vs k-infinity calculations
        if depletion_type == 'core':
            results_file = os.path.join(root_dir, 'depletion', 'outputs', 'core_keff', 'depletion_results.h5')
            input_flag = 'deplete_core'
        else:
            results_file = os.path.join(root_dir, 'depletion', 'outputs', f"{depletion_type}_k∞", 'depletion_results.h5')
            input_flag = f"deplete_{depletion_type}"

        # Only try to plot if the depletion type is enabled in inputs
        if not inputs.get(input_flag, False):
            return None

        # Load depletion data
        results, heavy_metal_mass_g, power_density = load_depletion_data(results_file)
        if results is None:
            return None

        try:
            # Calculate derived quantities
            heavy_metal_mass_kg = heavy_metal_mass_g / 1000
            total_power = power_density * heavy_metal_mass_g

            # Get time points and k-effective
            time_seconds, keff = results.get_keff()
            time_days = time_seconds / (24 * 60 * 60)

            # Calculate burnup points
            burnup = calculate_burnup_from_time(time_days, total_power, heavy_metal_mass_kg)

            # Plot k-effective with error bars
            line = ax1.errorbar(time_days, keff[:, 0], yerr=keff[:, 1],
                             label=label, color=color, marker='o', markersize=4,
                             capsize=3)

            # Check if k crosses from above 1 to below 1
            k_values = keff[:, 0]
            k_above_1 = k_values > 1
            if any(k_above_1) and any(~k_above_1):  # Has points both above and below 1
                for i in range(len(k_values)-1):
                    if k_values[i] > 1 and k_values[i+1] < 1:
                        # Linear interpolation to find exact crossing point
                        frac = (1 - k_values[i]) / (k_values[i+1] - k_values[i])
                        time_at_k1 = time_days[i] + frac * (time_days[i+1] - time_days[i])
                        burnup_at_k1 = burnup[i] + frac * (burnup[i+1] - burnup[i])

                        # Add vertical red dashed line at crossing
                        ax1.axvline(x=time_at_k1, color='red', linestyle='--', alpha=0.5)

                        # Add horizontal line at k=1
                        ax1.axhline(y=1, color='black', linestyle='--', alpha=0.3)

                        # Add text label - moved to left side
                        ax1.text(time_at_k1, 1.01,
                                f'Burnup: {burnup_at_k1:.1f} MWd/kgU\nTime: {time_at_k1:.1f} days',
                                color='red', ha='left', va='bottom')
                        break

            # Add burnup axis on top if not already present
            if len(ax1.get_shared_x_axes().get_siblings(ax1)) == 1:
                setup_burnup_axis(ax1, time_days, total_power, heavy_metal_mass_kg)

            ax1.set_xlabel('Time [days]')
            ax1.grid(True, alpha=0.3, linestyle='--')

            return line, burnup
        except Exception as e:
            print(f"Error plotting {depletion_type}: {str(e)}")
            return None

    # Create list of plots to generate
    plots_with_data = []
    if inputs.get('deplete_core', False):
        plots_with_data.append('core')
    if inputs.get('deplete_assembly', False) or inputs.get('deplete_assembly_enhanced', False):
        plots_with_data.append('assembly')
    if inputs.get('deplete_element', False) or inputs.get('deplete_element_enhanced', False):
        plots_with_data.append('element')

    if not plots_with_data:
        print("No depletion calculations enabled in inputs")
        return False

    # Create figure with appropriate number of subplots
    fig, axes = plt.subplots(len(plots_with_data), 1, figsize=(12, 8*len(plots_with_data)), squeeze=False)
    axes = axes.flatten()

    # Generate each plot
    for plot_idx, plot_type in enumerate(plots_with_data):
        ax = axes[plot_idx]
        lines = []

        if plot_type == 'core':
            # Plot core results
            core_keff = plot_depletion_type('core', ax, 'red', 'k-effective')
            if core_keff:
                lines.append(core_keff)
                ax.set_title(f'Core Multiplication Factors ({inputs["fuel_type"]} Fuel)')

        elif plot_type == 'assembly':
            # Plot assembly results
            assembly_line = plot_depletion_type('assembly', ax, 'red', 'Standard Assembly')
            if assembly_line:
                lines.append(assembly_line)
            assembly_enhanced_line = plot_depletion_type('assembly_enhanced', ax, 'blue', 'Enhanced Assembly')
            if assembly_enhanced_line:
                lines.append(assembly_enhanced_line)
            ax.set_title(f'Assembly Multiplication Factors ({inputs["fuel_type"]} Fuel)')

        else:  # element
            # Plot element results
            element_type = inputs['assembly_type']
            element_line = plot_depletion_type('element', ax, 'red', f'Standard {element_type}')
            if element_line:
                lines.append(element_line)
            element_enhanced_line = plot_depletion_type('element_enhanced', ax, 'blue', f'Enhanced {element_type}')
            if element_enhanced_line:
                lines.append(element_enhanced_line)
            ax.set_title(f'{element_type} Multiplication Factors ({inputs["fuel_type"]} Fuel)')

        if lines:
            ax.set_ylabel('Multiplication Factor')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'depletion_results.png'),
               dpi=300, bbox_inches='tight')
    plt.close()

    # Plot nuclide concentrations if specified
    if 'depletion_nuclides' in inputs:
        nuclide_plot_dir = os.path.join(plot_dir, 'nuclide_depletion')
        os.makedirs(nuclide_plot_dir, exist_ok=True)
        plot_nuclide_evolution(nuclide_plot_dir, root_dir=root_dir)

    return True

def plot_nuclide_evolution(plot_dir, root_dir=None):
    """Plot nuclide evolution for each depletion type.

    Parameters
    ----------
    plot_dir : str
        Directory to save plots to
    root_dir : str, optional
        Root directory of the project. If not provided, will attempt to find it.
    """
    if root_dir is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
            results_file = os.path.join(root_dir, 'depletion', 'outputs', 'core_keff', 'depletion_results.h5')
            plot_title = 'Full Core'
        else:
            results_file = os.path.join(root_dir, 'depletion', 'outputs', f"{depletion_type}_k∞", "depletion_results.h5")
            if depletion_type.startswith('assembly'):
                plot_title = 'Standard Assembly' if depletion_type == 'assembly' else 'Enhanced Assembly'
            else:  # element
                element_type = inputs['assembly_type']
                plot_title = f"{'Enhanced ' if 'enhanced' in depletion_type else 'Standard '}{element_type}"

        # Load depletion data
        results, heavy_metal_mass_g, power_density = load_depletion_data(results_file)
        if results is None:
            continue

        try:
            # Get first material
            with h5py.File(results_file, 'r') as f:
                if 'materials' not in f:
                    continue
                mat_ids = list(f['materials'].keys())
                if not mat_ids:
                    continue
                mat_id = mat_ids[0]

            # Calculate total power
            heavy_metal_mass_kg = heavy_metal_mass_g / 1000
            total_power = power_density * heavy_metal_mass_g

            # Create figure with stacked subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[1, 1])

            # Plot each nuclide
            for nuc in inputs['depletion_nuclides']:
                try:
                    time_seconds, atoms = results.get_atoms(mat_id, nuc)
                    time_days = time_seconds / (24 * 60 * 60)

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

                # Add burnup axis
                setup_burnup_axis(ax, time_days, total_power, heavy_metal_mass_kg)

            # Set title and adjust layout
            fig.suptitle(f'{plot_title} Nuclide Evolution', y=0.98, fontsize=12)
            plt.subplots_adjust(top=0.9, bottom=0.1, right=0.85, hspace=0.3)

            # Save and close
            plt.savefig(os.path.join(plot_dir, f'{depletion_type}_nuclide_evolution.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error processing results for {depletion_type}: {str(e)}")
            continue

    return True
