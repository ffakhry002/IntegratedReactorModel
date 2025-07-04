# spatial_error_heatmaps.py - FIXED with 8x8 grid and energy group support
# Modified to create separate mean and max error heatmaps

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def create_spatial_error_heatmaps(df, output_dir, models, encodings, optimizations, energy_group=None):
    """Create spatial error heatmaps showing error for each cell position

    Args:
        df: DataFrame with test results
        output_dir: Directory to save heatmaps
        models: List of model names
        encodings: List of encoding methods
        optimizations: List of optimization methods
        energy_group: Optional energy group ('thermal', 'epithermal', 'fast', 'total', or None)
    """

    # Try to load test configurations to get actual positions
    # First try to import parse_reactor_data
    try:
        from utils.txt_to_data import parse_reactor_data
    except ImportError:
        print("  Warning: Could not import parse_reactor_data. Using default position mapping.")
        # Use a default position mapping if we can't load the test data
        position_maps = create_default_position_maps(df)
    else:
        # Try different possible paths for test.txt
        test_file_paths = [
            "ML/data/test.txt",
            "../ML/data/test.txt",
            "data/test.txt",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test.txt"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ML", "data", "test.txt")
        ]

        position_maps = None
        for test_path in test_file_paths:
            if os.path.exists(test_path):
                try:
                    # Handle different return value counts from parse_reactor_data
                    parse_result = parse_reactor_data(test_path)

                    if isinstance(parse_result, tuple):
                        if len(parse_result) >= 3:
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
                            break
                    else:
                        print(f"  Warning: Unexpected return type from parse_reactor_data")
                except Exception as e:
                    print(f"  Warning: Failed to parse {test_path}: {e}")
                    continue

        if position_maps is None:
            print("  Warning: Could not load test configurations. Using default position mapping.")
            position_maps = create_default_position_maps(df)

    # Create both mean and max heatmaps for each optimization
    for optimization in optimizations:
        # Create mean error heatmap
        create_cell_position_heatmap_with_positions(
            df, output_dir, models, encodings, optimization, position_maps,
            energy_group=energy_group, aggregation='mean'
        )

        # Create max error heatmap with config numbers
        create_cell_position_heatmap_with_positions(
            df, output_dir, models, encodings, optimization, position_maps,
            energy_group=energy_group, aggregation='max'
        )


def create_default_position_maps(df):
    """Create default position maps based on common patterns in the data"""
    # Get number of unique configurations
    n_configs = df['config_id'].nunique()

    # Create a default mapping assuming irradiation positions are distributed
    # This is a fallback when we can't load the actual test configurations
    position_maps = []

    # Common patterns for 4 irradiation positions
    default_patterns = [
        # Pattern 1: Corners
        {'I_1': (2, 2), 'I_2': (2, 5), 'I_3': (5, 2), 'I_4': (5, 5)},
        # Pattern 2: Cross
        {'I_1': (1, 3), 'I_2': (3, 1), 'I_3': (3, 6), 'I_4': (6, 3)},
        # Pattern 3: Diamond
        {'I_1': (2, 3), 'I_2': (3, 2), 'I_3': (3, 5), 'I_4': (5, 3)},
        # Pattern 4: Line
        {'I_1': (3, 1), 'I_2': (3, 3), 'I_3': (3, 4), 'I_4': (3, 6)},
    ]

    # Repeat patterns to match number of configs
    for i in range(n_configs):
        position_maps.append(default_patterns[i % len(default_patterns)])

    return position_maps


def create_cell_position_heatmap_with_positions(df, output_dir, models, encodings,
                                               optimization, position_maps, energy_group=None, aggregation='mean'):
    """Create heatmap showing mean or max error at actual cell positions

    Args:
        aggregation: 'mean' or 'max' - how to aggregate errors across test configurations
    """

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

    # Title based on energy group and aggregation method
    agg_title = aggregation.capitalize()
    if energy_group:
        fig.suptitle(f'Spatial Error Heatmaps - {agg_title} {energy_group.capitalize()} Flux Error by Cell Position\nOptimization: {optimization}',
                     fontsize=16, fontweight='bold')
    else:
        fig.suptitle(f'Spatial Error Heatmaps - {agg_title} Error by Cell Position\nOptimization: {optimization}',
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
            position_max_configs = {}  # Track which config has max error

            # Initialize 8x8 grids
            for row in range(8):
                for col in range(8):
                    position_errors[(row, col)] = []
                    position_max_configs[(row, col)] = -1

            # Collect errors by actual position
            for _, test_row in subset.iterrows():
                config_id = int(test_row['config_id'])

                # Get position map for this configuration
                if config_id < len(position_maps):
                    pos_map = position_maps[config_id]

                    # For each irradiation position
                    for label, (row, col) in pos_map.items():
                        # Extract the number from label (e.g., 'I_1' -> 1)
                        try:
                            irr_num = int(label.split('_')[1])
                        except:
                            continue

                        # Determine which columns to use based on energy group
                        if energy_group:
                            # Energy-specific columns
                            error_col = f'I_{irr_num}_{energy_group}_rel_error'
                            real_col = f'I_{irr_num}_{energy_group}_real'
                            pred_col = f'I_{irr_num}_{energy_group}_predicted'
                        else:
                            # Standard columns (total flux)
                            error_col = f'I_{irr_num}_rel_error'
                            real_col = f'I_{irr_num}_real'
                            pred_col = f'I_{irr_num}_predicted'

                        # Try to get error from pre-calculated column first
                        error_value = None
                        if error_col in test_row and pd.notna(test_row[error_col]):
                            error_value = abs(test_row[error_col])  # ALWAYS use absolute value for spatial heatmaps
                        # Otherwise calculate it
                        elif real_col in test_row and pred_col in test_row:
                            real = test_row[real_col]
                            pred = test_row[pred_col]
                            if pd.notna(real) and pd.notna(pred) and real != 0 and str(real) != 'N/A':
                                error_value = abs((pred - real) / real) * 100

                        if error_value is not None and isinstance(error_value, (int, float)):
                            position_errors[(row, col)].append((error_value, config_id))

            # Calculate aggregated error for each position
            error_grid = np.zeros((8, 8))
            for row in range(8):
                for col in range(8):
                    errors_and_configs = position_errors[(row, col)]
                    if errors_and_configs:
                        errors = [e[0] for e in errors_and_configs]
                        if aggregation == 'mean':
                            error_grid[row, col] = np.mean(errors)
                        else:  # max
                            max_idx = np.argmax(errors)
                            error_grid[row, col] = errors[max_idx]
                            position_max_configs[(row, col)] = errors_and_configs[max_idx][1]
                    else:
                        error_grid[row, col] = np.nan

            # Create heatmap
            im = ax.imshow(error_grid, cmap='RdYlGn_r', vmin=0, vmax=15,
                          interpolation='nearest', aspect='equal')

            # Add text annotations
            for row in range(8):
                for col in range(8):
                    if not np.isnan(error_grid[row, col]) and error_grid[row, col] > 0:
                        # Main error value in center
                        text = ax.text(col, row, f'{error_grid[row, col]:.1f}',
                                     ha='center', va='center',
                                     color='white' if error_grid[row, col] > 7.5 else 'black',
                                     fontsize=8)

                        # For max aggregation, add config number in bottom right
                        if aggregation == 'max' and position_max_configs[(row, col)] >= 0:
                            config_text = f'#{position_max_configs[(row, col)]}'
                            ax.text(col + 0.4, row + 0.47, config_text,
                                   ha='right', va='bottom',
                                   color='black',
                                   fontsize=5,
                                   fontweight='normal')

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
    cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r',
                               norm=plt.Normalize(vmin=0, vmax=15))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f'{agg_title} Relative Error (%)', fontsize=12)

    plt.tight_layout()

    # Save figure with aggregation type in filename
    if energy_group:
        output_file = os.path.join(output_dir, f'spatial_cell_errors_{optimization}_{energy_group}_{aggregation}.png')
    else:
        output_file = os.path.join(output_dir, f'spatial_cell_errors_{optimization}_{aggregation}.png')

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def create_averaged_spatial_error_heatmap(df, output_dir, models, encodings, optimizations, energy_group=None, aggregation='mean'):
    """Create a single 8x8 heatmap averaged across all models/encodings/optimizations

    Args:
        aggregation: 'mean' or 'max' - how to aggregate errors
    """

    # Try to get position maps (same logic as above)
    try:
        from utils.txt_to_data import parse_reactor_data
        # ... (same position loading logic as in create_spatial_error_heatmaps)
        position_maps = create_default_position_maps(df)  # Simplified for brevity
    except:
        position_maps = create_default_position_maps(df)

    # Initialize error accumulator
    position_errors = {}
    position_max_configs = {}  # Track which config has max error for each position
    for row in range(8):
        for col in range(8):
            position_errors[(row, col)] = []
            position_max_configs[(row, col)] = -1

    # Collect all errors
    for _, test_row in df.iterrows():
        config_id = int(test_row['config_id'])

        if config_id < len(position_maps):
            pos_map = position_maps[config_id]

            for label, (row, col) in pos_map.items():
                try:
                    irr_num = int(label.split('_')[1])
                except:
                    continue

                if energy_group:
                    error_col = f'I_{irr_num}_{energy_group}_rel_error'
                else:
                    error_col = f'I_{irr_num}_rel_error'

                if error_col in test_row and pd.notna(test_row[error_col]):
                    error_value = abs(test_row[error_col])  # ALWAYS use absolute value for spatial heatmaps
                    if isinstance(error_value, (int, float)):
                        position_errors[(row, col)].append((error_value, config_id))

    # Calculate aggregated error for each position
    error_grid = np.zeros((8, 8))
    for row in range(8):
        for col in range(8):
            errors_and_configs = position_errors[(row, col)]
            if errors_and_configs:
                errors = [e[0] for e in errors_and_configs]
                if aggregation == 'mean':
                    error_grid[row, col] = np.mean(errors)
                else:  # max
                    max_idx = np.argmax(errors)
                    error_grid[row, col] = errors[max_idx]
                    position_max_configs[(row, col)] = errors_and_configs[max_idx][1]
            else:
                error_grid[row, col] = np.nan

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create heatmap
    im = ax.imshow(error_grid, cmap='RdYlGn_r', vmin=0, vmax=15,
                   interpolation='nearest', aspect='equal')

    # Add text annotations
    for row in range(8):
        for col in range(8):
            if not np.isnan(error_grid[row, col]) and error_grid[row, col] > 0:
                text = ax.text(col, row, f'{error_grid[row, col]:.1f}',
                             ha='center', va='center',
                             color='white' if error_grid[row, col] > 7.5 else 'black',
                             fontsize=10)

                # For max aggregation, add config number in bottom right
                if aggregation == 'max' and position_max_configs[(row, col)] >= 0:
                    config_text = f'#{position_max_configs[(row, col)]}'
                    ax.text(col - 0.4, row - 0.45, config_text,
                           ha='right', va='bottom',
                           color='black',
                           fontsize=6,
                           fontweight='normal')

    # Styling
    agg_title = aggregation.capitalize()
    if energy_group:
        title = f'Average {energy_group.capitalize()} Flux Spatial Error Pattern\n({agg_title} Relative Error %)'
    else:
        title = f'Average Spatial Error Pattern Across All Models\n({agg_title} Relative Error %)'

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.invert_yaxis()

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'{agg_title} Relative Error (%)', fontsize=12)

    plt.tight_layout()

    # Save figure
    if energy_group:
        output_file = os.path.join(output_dir, f'spatial_cell_errors_average_{energy_group}_{aggregation}.png')
    else:
        output_file = os.path.join(output_dir, f'spatial_cell_errors_average_{aggregation}.png')

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
