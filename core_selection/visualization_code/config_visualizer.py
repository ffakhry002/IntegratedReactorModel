"""
Configuration visualization functions.
Handles core configuration grid visualizations and irradiation pattern analysis.
UPDATED: Support for 6x6 mode detection in titles
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os
from typing import Dict, List, Tuple, Any
from .data_loader import detect_6x6_mode


def visualize_configuration(ax, config, position_number=None):
    """Visualize a single configuration without index labels."""
    # Create color map
    colors = {
        'C': 'lightblue',    # Coolant
        'F': 'lightgreen',   # Fuel
        'I': 'red'          # Irradiation
    }

    # Draw grid
    for i in range(8):
        for j in range(8):
            cell_type = config[i, j]
            color = colors[cell_type]

            # Create rectangle
            rect = patches.Rectangle((j, 7-i), 1, 1,
                                   linewidth=1,
                                   edgecolor='black',
                                   facecolor=color)
            ax.add_patch(rect)

            # Add text label (just the cell type)
            ax.text(j+0.5, 7-i+0.5, cell_type,
                   ha='center', va='center',
                   fontsize=8, weight='bold')

    # Set limits and aspect
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')

    # Add position number if provided
    if position_number is not None:
        ax.set_title(f"#{position_number}", fontsize=10, pad=5)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add grid lines
    for i in range(9):
        ax.axhline(i, color='black', linewidth=0.5)
        ax.axvline(i, color='black', linewidth=0.5)


def create_method_visualization(method_name, configurations, selected_indices, output_dir):
    """Create visualization for a single sampling method showing all sampled configurations."""
    # Detect 6x6 mode
    is_6x6 = detect_6x6_mode()
    mode_str = " (6x6 Central Square)" if is_6x6 else ""

    n_samples = len(selected_indices)

    # Calculate grid dimensions
    cols = math.ceil(math.sqrt(n_samples))
    rows = math.ceil(n_samples / cols)

    # Create figure with appropriate size
    fig_width = min(3 * cols, 20)  # Limit max width
    fig_height = min(3 * rows, 20)  # Limit max height

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create title with method name and sample count
    method_title = {
        'lhs': 'Latin Hypercube Sampling (Physics Space)',
        'lhs_lattice': 'LHS Lattice (Configuration Space)',
        'sobol': 'Sobol Sequence (Physics Space)',
        'sobol_lattice': 'Sobol Lattice (Configuration Space)',
        'halton': 'Halton Sequence',
        'jaccard_geometric': 'Jaccard Distance (Geometric)',
        'jaccard_lattice': 'Jaccard Distance (Lattice)',
        'euclidean_geometric': 'Euclidean Distance (Geometric)',
        'manhattan_geometric': 'Manhattan Distance (Geometric)'
    }.get(method_name, method_name.upper())

    fig.suptitle(f'{method_title} - {n_samples} Samples{mode_str}', fontsize=16)

    # Plot each configuration
    for plot_idx, config_idx in enumerate(selected_indices):
        ax = plt.subplot(rows, cols, plot_idx + 1)
        config = configurations[config_idx]

        # Just show position number, not configuration index
        visualize_configuration(ax, config, position_number=plot_idx + 1)

    # Hide empty subplots
    for i in range(n_samples, rows * cols):
        ax = plt.subplot(rows, cols, i + 1)
        ax.axis('off')

    plt.tight_layout()

    # Save figure
    filename = f'{output_dir}/{method_name}_all_samples.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    return filename


def create_core_grid_visualization(configurations, selected_indices, method_name, output_file, color):
    """Create a grid visualization of core configurations."""
    # Detect 6x6 mode
    is_6x6 = detect_6x6_mode()
    mode_str = " (6x6 Central Square)" if is_6x6 else ""

    n_samples = len(selected_indices)

    # Calculate grid dimensions
    cols = min(5, n_samples)  # Max 5 columns
    rows = (n_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    fig.suptitle(f'{method_name} - Core Configurations{mode_str}', fontsize=16, fontweight='bold')

    # Flatten axes for easier indexing
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, config_idx in enumerate(selected_indices):
        if i >= len(axes):
            break

        ax = axes[i]
        config = configurations[config_idx]

        # Create color map
        config_colors = np.zeros((8, 8, 3))
        for row in range(8):
            for col in range(8):
                if config[row, col] == 'C':
                    config_colors[row, col] = [0.7, 0.9, 1.0]  # Light blue for coolant
                elif config[row, col] == 'F':
                    config_colors[row, col] = [0.7, 1.0, 0.7]  # Light green for fuel
                elif config[row, col] == 'I':
                    config_colors[row, col] = [1.0, 0.4, 0.4]  # Red for irradiation

        ax.imshow(config_colors, aspect='equal')
        ax.set_title(f'Config {config_idx}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add grid lines
        for x in range(9):
            ax.axvline(x-0.5, color='black', linewidth=0.5)
        for y in range(9):
            ax.axhline(y-0.5, color='black', linewidth=0.5)

    # Hide unused subplots
    for i in range(len(selected_indices), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def create_irradiation_analysis(irradiation_sets, selected_indices, method_name, output_file, color):
    """Create irradiation pattern analysis plots."""
    # Detect 6x6 mode
    is_6x6 = detect_6x6_mode()
    mode_str = " (6x6 Central Square)" if is_6x6 else ""

    selected_irrad = [irradiation_sets[i] for i in selected_indices]

    # Create single plot for position frequency heatmap
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle(f'{method_name} - Irradiation Position Frequency{mode_str}', fontsize=16, fontweight='bold')

    # Position frequency heatmap
    position_counts = np.zeros((8, 8))
    for irrad_set in selected_irrad:
        for i, j in irrad_set:
            position_counts[i, j] += 1

    im = ax.imshow(position_counts, cmap='Reds', aspect='equal', origin='upper')
    ax.set_title('Irradiation Position Frequency', fontsize=14)
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Frequency')
    cbar.ax.tick_params(labelsize=10)

    # Add grid
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(np.arange(8))
    ax.set_yticklabels(np.arange(8))

    # Add grid lines
    ax.set_xticks(np.arange(8.5), minor=True)
    ax.set_yticks(np.arange(8.5), minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def create_all_cores_visualization(n_total_configs, selected_indices, method_name, output_file, color):
    """Create a visualization showing all cores with selected ones highlighted."""
    # Detect 6x6 mode
    is_6x6 = detect_6x6_mode()
    mode_str = " (6x6 Central Square)" if is_6x6 else ""

    # Calculate grid dimensions for all configs
    grid_size = int(np.ceil(np.sqrt(n_total_configs)))

    # Create figure
    fig_size = min(20, grid_size * 0.1)  # Scale figure size, max 20 inches
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))

    # Create grid showing all configurations
    grid = np.zeros((grid_size, grid_size))

    # Map indices to grid positions
    for idx in range(n_total_configs):
        row = idx // grid_size
        col = idx % grid_size
        if row < grid_size and col < grid_size:
            grid[row, col] = 0.3  # Default gray color

    # Highlight selected indices
    for idx in selected_indices:
        row = idx // grid_size
        col = idx % grid_size
        if row < grid_size and col < grid_size:
            grid[row, col] = 1.0  # Highlight color

    # Create custom colormap
    from matplotlib.colors import ListedColormap
    colors_list = ['white', 'lightgray', color]  # white=empty, gray=unselected, color=selected
    cmap = ListedColormap(colors_list)

    # Plot
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect='equal')

    # Add title
    ax.set_title(f'{method_name} - All Configurations{mode_str}\n({len(selected_indices)} selected out of {n_total_configs})',
                fontsize=14, fontweight='bold')

    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a border
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file
