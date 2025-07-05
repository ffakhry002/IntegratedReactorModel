"""
Functions for plotting axial flux distributions with energy breakdown at irradiation positions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import openmc

# Add root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)  # Insert at beginning to avoid conflicts

# Import inputs after adding to path
from inputs import inputs

# Now we can safely import from eigenvalue.tallies
from eigenvalue.tallies.normalization import calc_norm_factor
from eigenvalue.tallies.energy_groups import get_energy_bins
from plotting.functions.utils import get_tally_volume
from eigenvalue.outputs import collapse_to_three_groups


def plot_axial_flux_energy_breakdown(sp, plot_dir, inputs_dict=None):
    """Plot axial flux distributions with energy breakdown for irradiation positions.

    Creates two main plots:
    1. Multi-line plots: Separate lines for thermal, epithermal, fast, and total flux vs height
    2. Heatmap plots: 2D plots with height vs energy bins, showing flux intensity

    Each irradiation position gets its own subplot in both visualization types.

    Parameters
    ----------
    sp : openmc.StatePoint
        StatePoint file containing the tally results
    plot_dir : str
        Directory to save plots
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    # Calculate normalization factor
    power_mw = inputs_dict.get('core_power', 1.0)
    norm_factor = calc_norm_factor(power_mw, sp)

    # Find all irradiation position tallies
    irradiation_tallies = []
    for tally in sp.tallies.values():
        if tally.name and tally.name.startswith('I_') and not tally.name.endswith('_axial'):
            irradiation_tallies.append(tally)

    if not irradiation_tallies:
        print("No irradiation position tallies found")
        return

    print(f"Found {len(irradiation_tallies)} irradiation positions for axial energy breakdown")

    # Get energy bins from the first tally
    first_tally = irradiation_tallies[0]
    energy_filter = first_tally.find_filter(openmc.EnergyFilter)
    energy_bins = energy_filter.bins
    n_energy_groups = len(energy_bins) - 1  # Number of groups is bins - 1

    # Calculate energy midpoints in MeV
    energy_mids_mev = 0.5 * (energy_bins[:, 0] + energy_bins[:, 1]) / 1e6

    # Get core dimensions from inputs (in cm)
    half_height = inputs_dict['fuel_height'] * 50  # Convert to cm

    # Get the axial flux tallies to determine z positions
    axial_tallies = []
    for tally in sp.tallies.values():
        if tally.name and tally.name.endswith('_axial'):
            axial_tallies.append(tally)

    if not axial_tallies:
        print("No axial tallies found")
        return

    # Get z positions from the first axial tally
    first_axial = axial_tallies[0]
    n_axial_segments = first_axial.mean.shape[0]
    z = np.linspace(-half_height, half_height, n_axial_segments)

    # Process data for each irradiation position
    irradiation_data = {}

    for tally in irradiation_tallies:
        # Get volume for normalization
        volume = get_tally_volume(tally, sp, inputs_dict)

        # Get the flux data and normalize
        mean = tally.mean.ravel() * norm_factor / volume
        std_dev = tally.std_dev.ravel() * norm_factor / volume

        # Calculate three-group data
        means_3group, std_devs_3group = collapse_to_three_groups(mean, std_dev, inputs_dict)

        # Find corresponding axial tally
        axial_tally_name = f"{tally.name}_axial"
        axial_tally = None
        for ax_tally in axial_tallies:
            if ax_tally.name == axial_tally_name:
                axial_tally = ax_tally
                break

        if axial_tally:
            # Get axial distribution (total flux only from axial tally)
            axial_volume = get_tally_volume(axial_tally, sp, inputs_dict)
            axial_total = axial_tally.mean.reshape(-1) * norm_factor / axial_volume

            # For energy-resolved axial distribution, we need to create it from the full 3D mesh data
            # Since we don't have energy-resolved axial tallies, we'll use a different approach
            # We'll create synthetic axial distributions based on the three-group data

            # Store the data
            irradiation_data[tally.name] = {
                'fine_group_flux': mean,  # Fine energy group flux
                'three_group_flux': means_3group,  # Three-group flux
                'three_group_std': std_devs_3group,
                'axial_total': axial_total,
                'energy_mids': energy_mids_mev,
                'z_positions': z
            }

    # Create multi-line plots
    print("Creating multi-line energy breakdown plots...")
    _create_multiline_plots(irradiation_data, inputs_dict, plot_dir, half_height)

    # Create heatmap plots
    print("Creating heatmap energy breakdown plots...")
    _create_heatmap_plots(irradiation_data, inputs_dict, plot_dir, half_height)

    print(f"Axial flux energy breakdown plots saved to: {plot_dir}")
    print(f"  - axial_flux_energy_multiline.png")
    print(f"  - axial_flux_energy_heatmap.png")


def _create_multiline_plots(irradiation_data, inputs_dict, plot_dir, half_height):
    """Create multi-line plots for each irradiation position."""
    # Sort the irradiation positions by their number (I_1, I_2, etc.)
    sorted_positions = sorted(irradiation_data.items(),
                            key=lambda x: int(x[0].split('_')[1]))
    n_positions = len(sorted_positions)

    # Create figure with one plot per row (matching flux_traps.py style)
    fig, axes = plt.subplots(n_positions, 1, figsize=(12, 5*n_positions))
    if n_positions == 1:
        axes = [axes]

    # Define colors matching flux_traps.py
    colors = ['blue', 'green', 'red', 'black']

    for idx, (pos_name, data) in enumerate(sorted_positions):
        ax = axes[idx]

        z = data['z_positions']
        axial_total = data['axial_total']

        # For the three-group plots, we need to create axial distributions
        # We'll use the ratio of each group to total to scale the axial distribution
        total_flux = np.sum(data['three_group_flux'])
        if total_flux > 0:
            thermal_fraction = data['three_group_flux'][0] / total_flux
            epithermal_fraction = data['three_group_flux'][1] / total_flux
            fast_fraction = data['three_group_flux'][2] / total_flux

            # Create axial distributions for each group
            axial_thermal = axial_total * thermal_fraction
            axial_epithermal = axial_total * epithermal_fraction
            axial_fast = axial_total * fast_fraction
        else:
            axial_thermal = np.zeros_like(axial_total)
            axial_epithermal = np.zeros_like(axial_total)
            axial_fast = np.zeros_like(axial_total)

        # Plot each energy group - matching the style of flux_traps.py plot 4
        ax.plot(z, axial_thermal, 'b-', label='Thermal', linewidth=1)
        ax.plot(z, axial_epithermal, 'g-', label='Epithermal', linewidth=1)
        ax.plot(z, axial_fast, 'r-', label='Fast', linewidth=1)
        ax.plot(z, axial_total, 'k-', label='Total', linewidth=1)

        # Configure axial flux plot - matching flux_traps.py style
        ax.grid(True, which="major", ls="-", alpha=0.2)
        ax.set_ylabel('Flux [n/cm²/s]')
        ax.set_xlabel('Height [cm]')
        ax.set_title(f'{pos_name} - Axial Flux by Energy Group')
        ax.legend()
        ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        # Add fuel boundary lines
        ax.axvline(x=half_height, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=-half_height, color='gray', linestyle='--', alpha=0.5)

        # Add minor grid
        ax.grid(True, which='minor', linestyle=':', alpha=0.1)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'axial_flux_energy_multiline.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def _create_heatmap_plots(irradiation_data, inputs_dict, plot_dir, half_height):
    """Create heatmap plots for each irradiation position."""
    # Sort the irradiation positions by their number (I_1, I_2, etc.)
    sorted_positions = sorted(irradiation_data.items(),
                            key=lambda x: int(x[0].split('_')[1]))
    n_positions = len(sorted_positions)

    # Calculate subplot arrangement - 2 columns
    n_cols = 2
    n_rows = (n_positions + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))

    # Handle different cases for axes array
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = axes.flatten()

    for idx, (pos_name, data) in enumerate(sorted_positions):
        ax = axes_flat[idx]

        z = data['z_positions']
        n_z = len(z)
        energy_mids = data['energy_mids']
        n_energy = len(energy_mids)

        # Create a 2D array for the heatmap
        # We need to distribute the fine-group flux across axial positions
        # Since we only have total axial distribution, we'll use it to scale
        axial_total = data['axial_total']
        fine_flux = data['fine_group_flux']

        # Normalize axial distribution
        if np.sum(axial_total) > 0:
            axial_normalized = axial_total / np.sum(axial_total)
        else:
            axial_normalized = np.ones(n_z) / n_z

        # Create 2D flux array: [axial_positions, energy_groups] - transposed
        flux_2d = np.outer(axial_normalized, fine_flux)

        # Use logarithmic scale for better visualization
        if np.any(flux_2d > 0):
            vmin = np.min(flux_2d[flux_2d > 0])
            vmax = np.max(flux_2d)

            # Ensure reasonable bounds for log scale
            if vmin <= 0 or vmax <= vmin:
                vmin = vmax * 1e-6
                if vmax <= 0:
                    vmax = 1e-10
                    vmin = 1e-16

            # Create the heatmap with height on y-axis and energy on x-axis (flipped)
            im = ax.pcolormesh(energy_mids, z, flux_2d,
                              norm=LogNorm(vmin=vmin, vmax=vmax),
                              cmap='viridis', shading='auto')
        else:
            # Fallback to linear scale if no positive values
            im = ax.pcolormesh(energy_mids, z, flux_2d,
                              cmap='viridis', shading='auto')

        # Set scales
        ax.set_xscale('log')
        ax.set_xlim(1e-9, 20.0)  # Set x-axis limits from 1e-9 to 20 MeV

        # Set labels
        ax.set_ylabel('Height [cm]')
        ax.set_xlabel('Energy [MeV]')
        ax.set_title(f'{pos_name} - Energy vs Height Heatmap', fontsize=10)

        # Add fuel boundary lines
        ax.axhline(y=half_height, color='white', linestyle='--', alpha=0.8, linewidth=1)
        ax.axhline(y=-half_height, color='white', linestyle='--', alpha=0.8, linewidth=1)

        # Add energy group boundaries from inputs
        thermal_cutoff_mev = inputs_dict['thermal_cutoff'] * 1e-6  # Convert eV to MeV
        fast_cutoff_mev = inputs_dict['fast_cutoff'] * 1e-6  # Convert eV to MeV
        ax.axvline(x=thermal_cutoff_mev, color='white', linestyle=':', alpha=0.5, linewidth=1)
        ax.axvline(x=fast_cutoff_mev, color='white', linestyle=':', alpha=0.5, linewidth=1)

        # Add colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Flux [n/cm²/s]', fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        # Grid
        ax.grid(True, which='major', linestyle='-', alpha=0.2)
        ax.grid(True, which='minor', linestyle=':', alpha=0.1)

    # Hide unused subplots
    for idx in range(n_positions, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'axial_flux_energy_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function for testing the axial flux energy breakdown plots."""
    # Set up paths
    sim_data_dir = os.path.join(root_dir, 'simulation_data')
    statepoint_path = os.path.join(sim_data_dir, 'transport_data', 'statepoint.eigenvalue.h5')
    flux_plot_dir = os.path.join(sim_data_dir, 'flux_plots')

    # Check if statepoint file exists
    if not os.path.exists(statepoint_path):
        print(f"Error: Statepoint file not found at {statepoint_path}")
        print("Please run the main simulation first to generate the statepoint file.")
        return

    print(f"Loading statepoint file: {statepoint_path}")
    print(f"Output directory: {flux_plot_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(flux_plot_dir, exist_ok=True)

    try:
        # Load statepoint
        sp = openmc.StatePoint(statepoint_path)

        # Run the plotting function
        print("\nGenerating axial flux energy breakdown plots...")
        plot_axial_flux_energy_breakdown(sp, flux_plot_dir, inputs)

        print("\nPlots generated successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
