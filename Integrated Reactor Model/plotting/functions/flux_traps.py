"""
Functions for plotting neutron flux distributions in irradiation positions (flux traps).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import openmc
from inputs import inputs

from execution.tallies.normalization import calc_norm_factor
from plotting.functions.utils import get_tally_volume
from execution.outputs import collapse_to_three_groups
from execution.tallies.energy_groups import get_energy_bins

def plot_flux_trap_distributions(sp, power_mw, plot_dir):
    """Plot flux distributions for all irradiation positions."""
    # Calculate normalization factor
    norm_factor = calc_norm_factor(power_mw, sp)

    # Create figure with four subplots
    fig, (ax1, ax3, ax2, ax4) = plt.subplots(4, 1, figsize=(12, 20), height_ratios=[1, 1, 1, 1])
    chamber_count = 0

    # Store data for reuse in linear plot
    plot_data = []  # Store (energy_mids, mean, label) for each chamber

    # Store three-group data for second plot
    three_group_data = []
    # Get energy boundaries from inputs
    three_group_energies = [
        1e-11,  # Lower bound
        inputs['thermal_cutoff'] * 1e-6,  # Thermal/epithermal boundary (eV -> MeV)
        inputs['fast_cutoff'] * 1e-6,     # Epithermal/fast boundary (eV -> MeV)
        20.0   # Upper bound (20 MeV)
    ]

    # Get the first irradiation tally to determine energy bins
    n_bins = None
    for tally in sp.tallies.values():
        if tally.name.startswith('I_') and not tally.name.endswith('_axial'):
            energy_filter = tally.find_filter(openmc.EnergyFilter)
            energy_bins = energy_filter.bins
            n_bins = len(energy_bins)
            break

    if n_bins is None:
        raise ValueError("No irradiation position tallies found")

    # Plot each irradiation position
    for tally in sp.tallies.values():
        if tally.name.startswith('I_') and not tally.name.endswith('_axial'):
            # Get volume for normalization
            volume = get_tally_volume(tally, sp)

            # Get the flux data and normalize
            mean = tally.mean.ravel() * norm_factor / volume
            std_dev = tally.std_dev.ravel() * norm_factor / volume

            # Get energy bins from the tally itself
            energy_filter = tally.find_filter(openmc.EnergyFilter)
            energy_bins = energy_filter.bins

            # Calculate midpoints correctly
            energy_mids_mev = 0.5 * (energy_bins[:, 0] + energy_bins[:, 1]) / 1e6

            # Plot fine-group fluxes (log-log)
            ax1.plot(energy_mids_mev, mean, '-', label=tally.name, linewidth=1)

            # Store data for linear plot
            plot_data.append((energy_mids_mev, mean, tally.name))

            # Calculate three-group data
            means, std_devs = collapse_to_three_groups(mean, std_dev)
            three_group_data.append((means, std_devs, tally.name))

    # Configure fine-group log-log plot
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, which="major", ls="-", alpha=0.2)
    ax1.set_xlabel('Energy [MeV]')
    ax1.set_ylabel('Flux [n/cm²/s]')
    ax1.set_title(f'Fine-group Fluxes (Log-Log) ({n_bins} Energy Bins)')
    ax1.legend()

    # Plot fine-group fluxes (linear)
    for energy_mids, mean, label in plot_data:
        ax3.plot(energy_mids, mean, '-', label=label, linewidth=1)

    # Configure fine-group linear plot
    ax3.set_xscale('log')  # Keep x-axis logarithmic for energy
    ax3.grid(True, which="major", ls="-", alpha=0.2)
    ax3.set_xlabel('Energy [MeV]')
    ax3.set_ylabel('Flux [n/cm²/s]')
    ax3.set_title(f'Fine-group Fluxes (Linear Y) ({n_bins} Energy Bins)')
    ax3.legend()
    ax3.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Plot three-group fluxes
    colors = ['b', 'orange', 'g', 'r']

    for (means, std_devs, tally_name), color in zip(three_group_data, colors):
        total_flux = np.sum(means)
        fractions = means / total_flux

        label = (f"{tally_name}: "
                 f"[T: {fractions[0]:.3f} "
                 f"E: {fractions[1]:.3f} "
                 f"F: {fractions[2]:.3f}]")

        # Plot the horizontal lines for the three groups
        ax2.hlines(y=means[0], xmin=three_group_energies[0], xmax=three_group_energies[1], color=color, linewidth=2)
        ax2.hlines(y=means[1], xmin=three_group_energies[1], xmax=three_group_energies[2], color=color, linewidth=2)
        ax2.hlines(y=means[2], xmin=three_group_energies[2], xmax=three_group_energies[3], color=color, linewidth=2)

        ax2.vlines(x=three_group_energies[1], ymin=means[0], ymax=means[1], color=color, linewidth=2)
        ax2.vlines(x=three_group_energies[2], ymin=means[1], ymax=means[2], color=color, linewidth=2)

        # Add label using a small horizontal line for the legend
        ax2.plot([], [], color=color, label=label, linewidth=2)

    # Configure three-group plot
    ax2.set_xscale('log')
    ax2.set_yscale('log')  # Changed back to log scale for absolute values
    ax2.grid(True, which="major", ls="-", alpha=0.2)
    ax2.grid(True, which="minor", ls=":", alpha=0.1)
    ax2.set_xlabel('Energy [MeV]')
    ax2.set_ylabel('Flux [n/cm²/s]')  # Changed back to flux units
    ax2.set_title('Three-group Flux Distribution')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Set x-axis limits explicitly to show full range
    ax2.set_xlim(1e-11, 20.0)

    # Plot axial flux distributions
    colors = plt.cm.viridis(np.linspace(0, 1, len(three_group_data)))
    half_height = inputs['fuel_height'] * 50  # Convert to cm
    z = np.linspace(-half_height, half_height, 50)  # Match n_axial_segments default

    for tally in sp.tallies.values():
        if tally.name.endswith('_axial'):
            base_name = tally.name[:-6]  # Remove '_axial' suffix
            # Get mesh volume for normalization
            volume = get_tally_volume(tally, sp)

            # Get the flux data and normalize
            # For axial tallies, we need to reshape to get the axial profile
            mean = tally.mean.reshape(-1) * norm_factor / volume  # Simplified reshape for single energy group
            std_dev = tally.std_dev.reshape(-1) * norm_factor / volume

            # Plot axial profile - switched coordinates
            ax4.plot(z, mean, '-', label=base_name, linewidth=1)

    # Configure axial flux plot - swapped x/y labels and settings
    ax4.grid(True, which="major", ls="-", alpha=0.2)
    ax4.set_ylabel('Total Flux [n/cm²/s]')  # Switched to ylabel
    ax4.set_xlabel('Height [cm]')           # Switched to xlabel
    ax4.set_title('Axial Flux Distribution')
    ax4.legend()
    ax4.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))  # Changed to yaxis
    ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))         # Changed to y axis

    # Rest of the code remains the same
    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True, which='major', linestyle='-', alpha=0.2)
        ax.grid(True, which='minor', linestyle=':', alpha=0.1)

    plt.subplots_adjust(right=0.85, hspace=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'irradiation_fluxes.png'), dpi=300, bbox_inches='tight')
    plt.close()
