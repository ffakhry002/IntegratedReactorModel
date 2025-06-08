# spatial_error_heatmaps.py - COMPLETE REWRITE with actual position tracking

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from utils.txt_to_data import parse_reactor_data

def create_spatial_error_heatmaps(df, output_dir, models, encodings, optimizations):
    """Create spatial error heatmaps showing error for each cell position"""

    # Load test configurations to get actual positions
    test_lattices, test_flux, test_keff, test_descriptions = parse_reactor_data("ML/data/test.txt")

    # Create position map for each test configuration
    position_maps = []
    for config_id, lattice in enumerate(test_lattices):
        pos_map = {}
        for i in range(8):
            for j in range(8):
                if lattice[i, j].startswith('I_'):
                    pos_map[lattice[i, j]] = (i, j)
        position_maps.append(pos_map)

    # Create heatmaps
    for optimization in optimizations:
        create_cell_position_heatmap_with_positions(
            df, output_dir, models, encodings, optimization, position_maps
        )

def create_cell_position_heatmap_with_positions(df, output_dir, models, encodings,
                                               optimization, position_maps):
    """Create heatmap showing mean error at actual cell positions"""

    # Filter data for this optimization
    opt_df = df[df['optimization_method'] == optimization]

    # Create figure
    n_models = len(models)
    n_encodings = len(encodings)

    fig, axes = plt.subplots(n_models, n_encodings,
                            figsize=(4*n_encodings, 4*n_models))

    # Ensure axes is 2D
    if n_models == 1 and n_encodings == 1:
        axes = np.array([[axes]])
    elif n_models == 1:
        axes = axes.reshape(1, -1)
    elif n_encodings == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f'Spatial Error Heatmaps - Mean Error by Cell Position\nOptimization: {optimization}',
                 fontsize=16, fontweight='bold')

    # Process each model-encoding combination
    for i, model in enumerate(models):
        for j, encoding in enumerate(encodings):
            ax = axes[i, j]

            # Get data for this combination
            mask = (opt_df['model_class'] == model) & (opt_df['encoding'] == encoding)
            subset = opt_df[mask]

            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Calculate errors for each cell position
            position_errors = {}
            position_counts = {}

            # Initialize 8x8 grids
            for row in range(8):
                for col in range(8):
                    position_errors[(row, col)] = []

            # Collect errors by actual position
            for _, test_row in subset.iterrows():
                config_id = int(test_row['config_id'])

                # Get position map for this configuration
                if config_id < len(position_maps):
                    pos_map = position_maps[config_id]

                    # For each irradiation position
                    for label, (row, col) in pos_map.items():
                        # Extract the number from label (e.g., 'I_1' -> 1)
                        irr_num = int(label.split('_')[1])

                        # Get the error for this position
                        error_col = f'I_{irr_num}_rel_error'
                        if error_col in test_row and pd.notna(test_row[error_col]):
                            position_errors[(row, col)].append(test_row[error_col])

            # Calculate mean error for each position
            error_grid = np.zeros((8, 8))
            for row in range(8):
                for col in range(8):
                    errors = position_errors[(row, col)]
                    if errors:
                        error_grid[row, col] = np.mean(errors)
                    else:
                        error_grid[row, col] = np.nan

            # Create heatmap
            im = ax.imshow(error_grid, cmap='RdYlGn_r', vmin=0, vmax=15,
                          interpolation='nearest', aspect='equal')

            # Add text annotations for non-NaN values
            for row in range(8):
                for col in range(8):
                    if not np.isnan(error_grid[row, col]) and error_grid[row, col] > 0:
                        text = ax.text(col, row, f'{error_grid[row, col]:.1f}',
                                     ha='center', va='center',
                                     color='white' if error_grid[row, col] > 7.5 else 'black',
                                     fontsize=8)

            # Styling
            if i == 0:
                ax.set_title(encoding, fontsize=10)
            if j == 0:
                ax.set_ylabel(model, fontsize=10)

            ax.set_xticks(range(8))
            ax.set_yticks(range(8))
            ax.set_xticklabels(range(8), fontsize=8)
            ax.set_yticklabels(range(8), fontsize=8)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.invert_yaxis()

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r',
                               norm=plt.Normalize(vmin=0, vmax=15))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Mean Relative Error (%)', fontsize=12)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, f'spatial_cell_errors_{optimization}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
