"""
Functions for plotting Shannon entropy convergence and k-effective.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from inputs import inputs

def plot_entropy(sp, plot_dir):
    """Plot k-effective and Shannon entropy convergence over batches.

    Parameters
    ----------
    sp : openmc.StatePoint
        StatePoint file containing the simulation results
    plot_dir : str
        Directory to save the plot
    """
    # Get entropy data and batch numbers
    entropy = sp.entropy
    n_batches = len(entropy)
    batches = np.arange(1, n_batches + 1)

    # Get k-effective data
    k_data = sp.k_generation

    # Get number of inactive batches from inputs
    n_inactive = inputs['inactive']

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1])

    # Plot k-effective vs batch number
    ax1.plot(batches, k_data, 'b-', linewidth=1, label='k-effective')
    ax1.axvline(x=n_inactive, color='red', linestyle='--',
                label=f'Active Batches Start (n={n_inactive})')
    ax1.set_ylabel('k')
    ax1.set_title('k-effective Convergence')
    ax1.grid(True, which='major', linestyle='-', alpha=0.2)
    ax1.grid(True, which='minor', linestyle=':', alpha=0.1)
    ax1.legend()

    # Plot entropy vs batch number
    ax2.plot(batches, entropy, 'b-', linewidth=1, label='Shannon Entropy')
    ax2.axvline(x=n_inactive, color='red', linestyle='--',
                label=f'Active Batches Start (n={n_inactive})')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Shannon Entropy')
    ax2.set_title('Shannon Entropy Convergence')
    ax2.grid(True, which='major', linestyle='-', alpha=0.2)
    ax2.grid(True, which='minor', linestyle=':', alpha=0.1)
    ax2.legend()

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.3)

    # Save plot
    plt.savefig(os.path.join(plot_dir, 'entropy.png'), dpi=300, bbox_inches='tight')
    plt.close()
