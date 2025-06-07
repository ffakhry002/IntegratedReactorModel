"""
Spatial error heatmaps showing MAPE for each reactor position
FIXED: Array comparison issue with better error handling
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os

def create_spatial_error_heatmaps(df, output_dir, models, encodings, optimizations):
    """Create spatial error heatmaps for each optimization method"""

    try:
        # Convert to lists if numpy arrays
        if isinstance(models, np.ndarray):
            models = models.tolist()
        if isinstance(encodings, np.ndarray):
            encodings = encodings.tolist()
        if isinstance(optimizations, np.ndarray):
            optimizations = optimizations.tolist()

        # Ensure they are lists
        models = list(models) if hasattr(models, '__iter__') else []
        encodings = list(encodings) if hasattr(encodings, '__iter__') else []
        optimizations = list(optimizations) if hasattr(optimizations, '__iter__') else []

        if len(models) == 0 or len(encodings) == 0 or len(optimizations) == 0:
            print("  Warning: No models, encodings, or optimizations found. Skipping spatial heatmaps.")
            return

        # Create one file per optimization method
        for optimization in optimizations:
            try:
                # Check if we have data for this optimization
                opt_data = df[df['optimization_method'] == optimization]
                if len(opt_data) == 0:
                    print(f"  Warning: No data found for optimization '{optimization}'. Skipping.")
                    continue

                create_optimization_spatial_heatmap(df, output_dir, models, encodings, optimization)
            except Exception as e:
                print(f"  Error creating spatial heatmap for {optimization}: {e}")
                continue

    except Exception as e:
        print(f"  Critical error in spatial error heatmaps: {e}")
        raise

def create_optimization_spatial_heatmap(df, output_dir, models, encodings, optimization):
    """Create spatial heatmap grid for a single optimization method"""

    # Filter data for this optimization
    opt_df = df[df['optimization_method'] == optimization]

    # Create figure with grid of subplots
    n_models = len(models)
    n_encodings = len(encodings)

    fig, axes = plt.subplots(n_models, n_encodings,
                            figsize=(4*n_encodings, 4*n_models))

    # Ensure axes is always 2D array
    if n_models == 1 and n_encodings == 1:
        axes = np.array([[axes]])
    elif n_models == 1:
        axes = axes.reshape(1, -1)
    elif n_encodings == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f'Spatial Error Heatmaps - MAPE by Position\nOptimization: {optimization}',
                 fontsize=16, fontweight='bold')

    # Define reactor layout (8x8 grid)
    reactor_layout = get_standard_reactor_layout()

    # Color scale limits
    vmin = 0
    vmax = 15  # Maximum MAPE %

    for i, model in enumerate(models):
        for j, encoding in enumerate(encodings):
            ax = axes[i, j]

            try:
                # Get data for this model-encoding combination
                mask = (opt_df['model_class'] == model) & (opt_df['encoding'] == encoding)
                subset = opt_df[mask]

                # Skip if no data
                if len(subset) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                # Calculate average MAPE for each position
                position_errors = calculate_position_errors(subset)

                # Create reactor visualization
                create_reactor_viz(ax, reactor_layout, position_errors, vmin, vmax)

            except Exception as e:
                print(f"    Error for {model}-{encoding}: {e}")
                ax.text(0.5, 0.5, 'Error', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)

            # Set title
            if i == 0:
                ax.set_title(encoding, fontsize=10)

            # Set y-label
            if j == 0:
                ax.set_ylabel(model, fontsize=10)

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r',
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('MAPE (%)', fontsize=12)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, f'spatial_error_{optimization}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

def get_standard_reactor_layout():
    """Get the standard 8x8 reactor layout"""
    # This should match your actual reactor configurations
    # Using a typical pattern with control rods (C), fuel (F), and irradiation positions (I_x)
    layout = [
        ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['F', 'F', 'I_1', 'F', 'F', 'I_2', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'I_3', 'F', 'F', 'I_4', 'F', 'F'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C']
    ]
    return np.array(layout)

def calculate_position_errors(subset_df):
    """Calculate average MAPE for each irradiation position"""
    position_errors = {}

    # Calculate errors for each position
    for i in range(1, 5):  # I_1 through I_4
        real_col = f'I_{i}_real'
        pred_col = f'I_{i}_predicted'

        if real_col in subset_df.columns and pred_col in subset_df.columns:
            # Get non-null values
            mask = subset_df[real_col].notna() & subset_df[pred_col].notna()
            real_values = subset_df.loc[mask, real_col]
            pred_values = subset_df.loc[mask, pred_col]

            if len(real_values) > 0:
                # Convert to numpy arrays to avoid ambiguous truth value error
                real_arr = real_values.values.astype(float)
                pred_arr = pred_values.values.astype(float)

                # Calculate MAPE only for non-zero values
                nonzero_mask = real_arr != 0
                if np.sum(nonzero_mask) > 0:
                    mape = np.mean(np.abs((pred_arr[nonzero_mask] - real_arr[nonzero_mask]) / real_arr[nonzero_mask]) * 100)
                    position_errors[f'I_{i}'] = float(mape)

    return position_errors

def create_reactor_viz(ax, layout, position_errors, vmin, vmax):
    """Create a single reactor visualization"""

    # Color map
    cmap = plt.cm.RdYlGn_r  # Red (high error) to Green (low error)

    # Position mapping for the 4 irradiation positions
    position_map = {
        (2, 2): 'I_1',
        (2, 5): 'I_2',
        (5, 2): 'I_3',
        (5, 5): 'I_4'
    }

    for i in range(8):
        for j in range(8):
            cell_type = str(layout[i, j])  # Ensure it's a string

            # Determine cell color
            if cell_type == 'C':
                color = '#2c3e50'  # Dark gray for control rods
                edge_color = 'black'
                edge_width = 2
            elif cell_type == 'F':
                color = '#7f8c8d'  # Medium gray for fuel
                edge_color = 'black'
                edge_width = 1
            elif cell_type.startswith('I_'):
                # Get position from layout
                pos_key = cell_type
                if pos_key in position_errors:
                    error = position_errors[pos_key]
                    norm_error = (error - vmin) / (vmax - vmin)
                    norm_error = np.clip(norm_error, 0, 1)
                    color = cmap(norm_error)
                else:
                    color = 'white'
                edge_color = 'black'
                edge_width = 2
            else:
                color = 'white'
                edge_color = 'gray'
                edge_width = 1

            # Draw cell
            rect = mpatches.Rectangle((j, 7-i), 1, 1,
                                    facecolor=color,
                                    edgecolor=edge_color,
                                    linewidth=edge_width)
            ax.add_patch(rect)

            # Add text for irradiation positions
            if cell_type.startswith('I_') and cell_type in position_errors:
                error = position_errors[cell_type]
                ax.text(j+0.5, 7-i+0.5, f'{error:.1f}%',
                       ha='center', va='center',
                       fontsize=8, fontweight='bold',
                       color='white' if error > (vmax-vmin)/2 else 'black')

    # Set axis limits
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
