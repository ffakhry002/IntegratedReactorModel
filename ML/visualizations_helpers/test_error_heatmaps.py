"""
Test error heat maps for flux visualizations
Creates heat maps for each test core showing error values at each irradiation position
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os
from .spatial_error_heatmaps import create_default_position_maps


def create_test_error_heatmaps(df, output_dir, models, encodings, optimizations, energy_group=None):
    """
    Create test error heat maps for each encoding and optimization combination.

    Args:
        df: DataFrame with test results
        output_dir: Directory to save heat maps
        models: List of model names
        encodings: List of encoding methods
        optimizations: List of optimization methods
        energy_group: Optional energy group ('thermal', 'epithermal', 'fast', 'total', or None)
    """

    # Create test error heat map directory
    heatmap_dir = os.path.join(output_dir, 'test_error_heatmap')
    os.makedirs(heatmap_dir, exist_ok=True)

    print(f"\nCreating test error heat maps...")

    # Try to load test configurations to get actual positions
    position_maps = _load_test_position_maps(df)

    # Process each model, encoding, and optimization combination
    for model in models:
        # Create model-specific subdirectory
        model_dir = os.path.join(heatmap_dir, model)
        os.makedirs(model_dir, exist_ok=True)

        for encoding in encodings:
            for optimization in optimizations:
                # Filter data for this combination
                mask = (
                    (df['model_class'] == model) &
                    (df['encoding'] == encoding) &
                    (df['optimization_method'] == optimization)
                )
                subset = df[mask]

                if subset.empty:
                    continue

                # Create proper naming: model_encoding_optimization_flux_type
                if energy_group:
                    combination_name = f"{model}_{encoding}_{optimization}_{energy_group}_flux"
                else:
                    combination_name = f"{model}_{encoding}_{optimization}_total_flux"

                print(f"  Processing {combination_name}...")

                # Create grid of all test cores for this combination
                _create_test_cores_grid(
                    subset, model_dir, combination_name,
                    position_maps, energy_group
                )


def _load_test_position_maps(df):
    """Load test configuration position maps"""
    try:
        from utils.txt_to_data import parse_reactor_data
    except ImportError:
        print("  Warning: Could not import parse_reactor_data. Using default position mapping.")
        return create_default_position_maps(df)

    # Try different possible paths for test.txt
    test_file_paths = [
        "ML/data/test.txt",
        "../ML/data/test.txt",
        "data/test.txt",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test.txt"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ML", "data", "test.txt")
    ]

    for test_path in test_file_paths:
        if os.path.exists(test_path):
            try:
                parse_result = parse_reactor_data(test_path)

                if isinstance(parse_result, tuple) and len(parse_result) >= 3:
                    test_lattices = parse_result[0]
                    # Create position map for each test configuration
                    position_maps = []
                    for config_id, lattice in enumerate(test_lattices):
                        pos_map = {}
                        for i in range(min(8, lattice.shape[0])):
                            for j in range(min(8, lattice.shape[1])):
                                if lattice[i, j].startswith('I_'):
                                    pos_map[lattice[i, j]] = (i, j)
                        position_maps.append(pos_map)
                    return position_maps
            except Exception as e:
                print(f"  Warning: Failed to parse {test_path}: {e}")
                continue

    print("  Warning: Could not load test configurations. Using default position mapping.")
    return create_default_position_maps(df)


def _create_test_cores_grid(subset, heatmap_dir, combination_name, position_maps, energy_group):
    """Create grid layout showing all test cores for a specific model-encoding-optimization combination"""

    # Get unique test configurations
    test_configs = sorted(subset['config_id'].unique())
    n_configs = len(test_configs)

    if n_configs == 0:
        return

    # Calculate grid dimensions (aim for roughly square layout)
    cols = int(np.ceil(np.sqrt(n_configs)))
    rows = int(np.ceil(n_configs / cols))

        # Create figure (matching spatial_error_heatmaps proportions)
    fig_width = max(16, 3 * cols)
    fig_height = max(12, 3 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Title
    energy_str = f'{energy_group.capitalize()} ' if energy_group else ''
    title = f'Test Error Heat Maps - {energy_str}Flux\n{combination_name.replace("_", " ").title()}'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Flatten axes for easier indexing
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Plot each test configuration
    for i, config_id in enumerate(test_configs):
        if i >= len(axes):
            break

        ax = axes[i]
        config_subset = subset[subset['config_id'] == config_id]

        if config_subset.empty:
            ax.set_visible(False)
            continue

        # Create heat map for this test core
        _create_single_core_heatmap(
            config_subset.iloc[0], config_id, ax,
            position_maps, energy_group
        )

    # Hide unused subplots
    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)

    # Add colorbar (positioned to not overlap, matching spatial_error_heatmaps style)
    cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])
    # Use standard RdYlGn_r colormap for colorbar (only showing error range 0-15)
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=0, vmax=15))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Relative Error (%)', fontsize=12)

    plt.tight_layout()

    # Save the plot
    filename = f'{combination_name}_test_cores_error_grid.png'
    filepath = os.path.join(heatmap_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {filename}")


def _create_single_core_heatmap(test_row, config_id, ax, position_maps, energy_group):
    """Create a single heat map for one test core"""

    # Initialize 8x8 grid - use NaN for all positions initially
    error_grid = np.full((8, 8), np.nan)

    # Define coolant positions (will stay white/NaN)
    coolant_positions = [
        (0, 0), (0, 1), (0, 6), (0, 7),  # Top row corners and adjacent
        (1, 0), (1, 7),                   # Second row corners
        (6, 0), (6, 7),                   # Seventh row corners
        (7, 0), (7, 1), (7, 6), (7, 7)   # Bottom row corners and adjacent
    ]

    # Fill fuel region with grey background (use negative value to distinguish from error values)
    for row in range(8):
        for col in range(8):
            if (row, col) not in coolant_positions:
                error_grid[row, col] = -1  # Grey background marker

    # Get position map for this configuration
    config_id_int = int(config_id)
    if config_id_int < len(position_maps):
        pos_map = position_maps[config_id_int]
    else:
        # Use default pattern if config_id exceeds available maps
        pos_map = {'I_1': (2, 2), 'I_2': (2, 5), 'I_3': (5, 2), 'I_4': (5, 5)}

    # Fill in error values for each irradiation position
    for label, (row, col) in pos_map.items():
        try:
            irr_num = int(label.split('_')[1])
        except:
            continue

        # Determine which error column to use based on energy group
        if energy_group:
            error_col = f'I_{irr_num}_{energy_group}_rel_error'
        else:
            error_col = f'I_{irr_num}_rel_error'

        if error_col in test_row and pd.notna(test_row[error_col]):
            error_value = test_row[error_col]  # Use absolute values like spatial_error_heatmaps
            if isinstance(error_value, (int, float)):
                error_grid[row, col] = abs(error_value)

            # Create a custom colormap with grey for background
    # Get the RdYlGn_r colormap
    base_cmap = cm.get_cmap('RdYlGn_r')

    # Create colors: grey for -1, then RdYlGn_r for 0-15
    colors = ['lightgrey'] + [base_cmap(i/15) for i in range(16)]
    custom_cmap = mcolors.ListedColormap(colors)

    # Create bounds: -1.5 to -0.5 (grey), then 0 to 15 (RdYlGn_r)
    bounds = [-1.5, -0.5] + list(range(16))
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

    # Create heat map using custom colormap
    im = ax.imshow(error_grid, cmap=custom_cmap, norm=norm,
                   interpolation='nearest', aspect='equal')

    # Add grid lines (matching spatial_error_heatmaps styling)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(range(8), fontsize=8)
    ax.set_yticklabels(range(8), fontsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.invert_yaxis()

    # Add text annotations for error values (matching spatial_error_heatmaps style)
    for row in range(8):
        for col in range(8):
            if not np.isnan(error_grid[row, col]) and error_grid[row, col] >= 0:  # Only show actual error values
                error_val = error_grid[row, col]
                # Choose text color based on background (matching spatial_error_heatmaps)
                text_color = 'white' if error_val > 7.5 else 'black'
                ax.text(col, row, f'{error_val:.1f}',
                       ha='center', va='center', color=text_color,
                       fontsize=8)

    # Set title with config ID (matching spatial_error_heatmaps font size)
    ax.set_title(f'Config {config_id}', fontsize=10)
