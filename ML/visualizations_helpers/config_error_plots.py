"""
Configuration error plots showing error trends across test configurations
UPDATED: Multiple optimizations as subplots, all encodings shown, separate table images
UPDATED: Added support for energy-discretized results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.patches import Rectangle

def create_config_error_plots(df, output_dir, energy_group=None, target_type=None):
    """
    Create config vs error plots for max flux, mean flux, and k-eff

    Args:
        df: DataFrame with test results
        output_dir: Directory to save plots
        energy_group: Energy group to analyze ('thermal', 'epithermal', 'fast', 'total')
        target_type: 'flux' or 'keff' - if specified, only create that type
    """

    # Get unique optimization methods
    optimizations = df['optimization_method'].unique()

    if target_type == 'keff':
        # Only create k-eff plot
        create_keff_error_plot(df, output_dir, optimizations)
    elif target_type:
        # Create flux plots for specific energy group
        create_max_flux_error_plot(df, output_dir, optimizations, energy_group)
        create_mean_flux_error_plot(df, output_dir, optimizations, energy_group)
    else:
        # Create all three plot types
        create_max_flux_error_plot(df, output_dir, optimizations, energy_group)
        create_mean_flux_error_plot(df, output_dir, optimizations, energy_group)
        if not energy_group:  # K-eff doesn't have energy groups
            create_keff_error_plot(df, output_dir, optimizations)

def create_max_flux_error_plot(df, output_dir, optimizations, energy_group=None):
    """Create plot showing maximum flux error for each configuration"""

    # Prepare data
    plot_data = prepare_flux_error_data(df, 'max', energy_group)

    # Create plots with subplots for each optimization
    filename_base = f'max_{energy_group}_flux_error_by_config' if energy_group else 'max_flux_error_by_config'
    title_base = f'Maximum {energy_group.capitalize()} Flux Relative Error by Configuration' if energy_group else 'Maximum Flux Relative Error by Configuration'

    create_multi_optimization_plot(plot_data, optimizations, output_dir,
                                   filename_base, title_base, 'max')

def create_mean_flux_error_plot(df, output_dir, optimizations, energy_group=None):
    """Create plot showing mean flux error for each configuration"""

    # Prepare data
    plot_data = prepare_flux_error_data(df, 'mean', energy_group)

    # Create plots with subplots for each optimization
    filename_base = f'mean_{energy_group}_flux_error_by_config' if energy_group else 'mean_flux_error_by_config'
    title_base = f'Mean {energy_group.capitalize()} Flux Relative Error by Configuration' if energy_group else 'Mean Flux Relative Error by Configuration'

    create_multi_optimization_plot(plot_data, optimizations, output_dir,
                                   filename_base, title_base, 'mean')

def create_keff_error_plot(df, output_dir, optimizations):
    """Create plot showing k-eff error for each configuration"""

    # Prepare data
    plot_data = prepare_keff_error_data(df)

    # Create plots with subplots for each optimization
    create_multi_optimization_plot(plot_data, optimizations, output_dir,
                                   'keff_error_by_config',
                                   'K-effective Relative Error by Configuration',
                                   'keff')

def create_multi_optimization_plot(plot_data, optimizations, output_dir, filename_base, title_base, error_type):
    """Create plot with subplots for each optimization method"""

    if plot_data.empty:
        return

    # Define colors and get unique values
    model_colors = {
        'xgboost': '#1f77b4',
        'random_forest': '#ff7f0e',
        'svm': '#2ca02c',
        'neural_net': '#d62728'
    }

    encoding_order = ['one_hot', 'categorical', 'physics', 'spatial', 'graph']
    model_order = ['xgboost', 'random_forest', 'svm', 'neural_net']

    # Create figure with subplots
    n_opts = len(optimizations)
    fig, axes = plt.subplots(n_opts, 1, figsize=(14, 6*n_opts), squeeze=False)

    # Process each optimization
    for opt_idx, optimization in enumerate(optimizations):
        ax = axes[opt_idx, 0]

        # Filter data for this optimization
        opt_data = plot_data[plot_data['optimization'] == optimization]

        if opt_data.empty:
            continue

        # Create box plot data
        box_data = []
        positions = []
        colors = []
        labels = []
        model_positions = {}  # Track where each model's boxes are

        pos = 0
        for model_idx, model in enumerate(model_order):
            model_data = opt_data[opt_data['model'] == model]

            if not model_data.empty:
                model_start_pos = pos

                # Add separator line before each model (except first)
                if model_idx > 0:
                    ax.axvline(x=pos-0.5, color='gray', linestyle='--', alpha=0.5)

                for encoding in encoding_order:
                    enc_data = model_data[model_data['encoding'] == encoding]

                    if not enc_data.empty:
                        box_data.append(enc_data['error'].values)
                        positions.append(pos)
                        colors.append(model_colors[model])
                        labels.append(f'{encoding}')
                        pos += 1

                # Store model position info
                model_end_pos = pos - 1
                model_positions[model] = (model_start_pos, model_end_pos)

        # Create box plot
        bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                       showmeans=True, meanline=True)

        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Customize mean lines
        for line in bp['means']:
            line.set_color('red')
            line.set_linewidth(2)

        # Set labels
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')

        # Add model labels - FIXED POSITIONING
        y_min, y_max = ax.get_ylim()
        label_y = y_max + (y_max - y_min) * 0.02  # Just above the plot area

        for model, (start_pos, end_pos) in model_positions.items():
            mid_pos = (start_pos + end_pos) / 2
            model_display = {
                'xgboost': 'XGBOOST',
                'random_forest': 'RANDOM FOREST',
                'svm': 'SVM',
                'neural_net': 'NEURAL NET'
            }
            ax.text(mid_pos, label_y, model_display[model],
                   ha='center', va='bottom', fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=model_colors[model], linewidth=2))

        # Customize plot
        ax.set_ylabel('Relative Error (%)', fontsize=12)
        ax.set_title(f'{title_base}\nOptimization: {optimization}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Set x-axis limits with padding
        if positions:
            ax.set_xlim(positions[0] - 1, positions[-1] + 1)

        # Adjust y-axis to make room for model labels
        ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.15)

    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(output_dir, f'{filename_base}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {plot_file}")

    # Create separate table image
    create_error_table_image(plot_data, optimizations, model_order, encoding_order,
                           output_dir, filename_base, error_type)

def create_error_table_image(plot_data, optimizations, model_order, encoding_order,
                            output_dir, filename_base, error_type):
    """Create a separate image with just the statistics table"""

    # Calculate statistics
    stats_data = []

    for optimization in optimizations:
        opt_data = plot_data[plot_data['optimization'] == optimization]

        for model in model_order:
            model_data = opt_data[opt_data['model'] == model]

            if not model_data.empty:
                # Track if this is the first row for this model
                first_row = True
                model_rows = 0

                # Count how many encodings have data for this model
                for encoding in encoding_order:
                    enc_data = model_data[model_data['encoding'] == encoding]
                    if not enc_data.empty:
                        model_rows += 1

                # Now create the rows
                for encoding in encoding_order:
                    enc_data = model_data[model_data['encoding'] == encoding]

                    if not enc_data.empty:
                        error_vals = enc_data['error']

                        stats_data.append({
                            'Model': model if first_row else '',
                            'Model_span': model_rows if first_row else 0,
                            'Optimization': optimization,
                            'Encoding': encoding,
                            'Mean': f'{error_vals.mean():.3f}%',
                            'Std Dev': f'{error_vals.std():.3f}%',
                            'Min': f'{error_vals.min():.3f}%',
                            'Max': f'{error_vals.max():.3f}%',
                            'P25': f'{error_vals.quantile(0.25):.3f}%',
                            'P50': f'{error_vals.quantile(0.50):.3f}%',
                            'P75': f'{error_vals.quantile(0.75):.3f}%'
                        })

                        first_row = False

    # Create figure for table
    fig, ax = plt.subplots(figsize=(14, len(stats_data) * 0.3 + 2))
    ax.axis('off')

    # Create table data
    col_labels = ['Model', 'Optimization', 'Encoding', 'Mean', 'Std Dev', 'Min', 'Max', 'P25', 'P50', 'P75']
    table_data = []

    for row in stats_data:
        table_data.append([
            row['Model'], row['Optimization'], row['Encoding'],
            row['Mean'], row['Std Dev'], row['Min'], row['Max'],
            row['P25'], row['P50'], row['P75']
        ])

    # Create table
    table = ax.table(cellText=table_data, colLabels=col_labels,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Header styling
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # Row coloring and span handling
    current_row = 1
    model_colors = {
        'xgboost': '#E6F3FF',
        'random_forest': '#FFF0E6',
        'svm': '#E6FFE6',
        'neural_net': '#FFE6E6'
    }

    i = 0
    while i < len(stats_data):
        row_data = stats_data[i]

        # Apply model-specific coloring
        model_name = None
        for model in model_order:
            if row_data['Model'] == model or (i > 0 and stats_data[i-1]['Model'] == model):
                model_name = model
                break

        # Merge cells for model name if span > 1
        if row_data['Model_span'] > 1:
            # Apply color to all cells in this model's rows
            for j in range(row_data['Model_span']):
                for col in range(len(col_labels)):
                    if current_row + j <= len(table_data):
                        cell = table[(current_row + j, col)]
                        if model_name in model_colors:
                            cell.set_facecolor(model_colors[model_name])
        else:
            # Single row coloring
            for col in range(len(col_labels)):
                cell = table[(current_row, col)]
                if model_name in model_colors:
                    cell.set_facecolor(model_colors[model_name])

        current_row += 1
        i += 1

    # Title
    if error_type == 'max':
        title = 'Maximum Flux Error Statistics by Model and Encoding'
    elif error_type == 'mean':
        title = 'Mean Flux Error Statistics by Model and Encoding'
    else:
        title = 'K-effective Error Statistics by Model and Encoding'

    plt.title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save table
    table_file = os.path.join(output_dir, f'{filename_base}_table.png')
    plt.savefig(table_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {table_file}")

def prepare_flux_error_data(df, error_type, energy_group=None):
    """Prepare flux error data for plotting"""
    plot_data = []

    for _, row in df.iterrows():
        errors = []
        for i in range(1, 5):
            if energy_group:
                # Energy-specific columns
                real_col = f'I_{i}_{energy_group}_real'
                pred_col = f'I_{i}_{energy_group}_predicted'
            else:
                # Standard columns
                real_col = f'I_{i}_real'
                pred_col = f'I_{i}_predicted'

            if real_col in row and pred_col in row:
                real_val = row[real_col]
                pred_val = row[pred_col]

                if pd.notna(real_val) and pd.notna(pred_val) and real_val != 0 and real_val != 'N/A':
                    error = abs((pred_val - real_val) / real_val) * 100
                    errors.append(error)

        if errors:
            if error_type == 'max':
                error_value = max(errors)
            else:  # mean
                error_value = np.mean(errors)

            plot_data.append({
                'config_id': row['config_id'],
                'model': row['model_class'],
                'encoding': row['encoding'],
                'optimization': row['optimization_method'],
                'error': error_value
            })

    return pd.DataFrame(plot_data)

def prepare_keff_error_data(df):
    """Prepare k-eff error data for plotting"""
    plot_data = []

    for _, row in df.iterrows():
        if 'keff_real' in row and 'keff_predicted' in row:
            real_val = row['keff_real']
            pred_val = row['keff_predicted']

            if pd.notna(real_val) and pd.notna(pred_val) and real_val != 0:
                error = abs((pred_val - real_val) / real_val) * 100

                plot_data.append({
                    'config_id': row['config_id'],
                    'model': row['model_class'],
                    'encoding': row['encoding'],
                    'optimization': row['optimization_method'],
                    'error': error
                })

    return pd.DataFrame(plot_data)

# For backward compatibility - simple single optimization plot
def create_simplified_error_plot(plot_data, title):
    """Create a simplified error plot showing only model performance"""

    if plot_data.empty:
        return

    # Define model colors
    model_colors = {
        'xgboost': '#1f77b4',
        'random_forest': '#ff7f0e',
        'svm': '#2ca02c',
        'neural_net': '#d62728'
    }

    # Create figure with space for table
    fig = plt.figure(figsize=(12, 10))

    # Main plot
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)

    # Ensure all models are represented
    all_models = ['xgboost', 'random_forest', 'svm', 'neural_net']
    model_groups = plot_data.groupby('model')

    # Create box plots for each model
    errors = []
    colors = []

    for model in all_models:
        if model in model_groups.groups:
            group = model_groups.get_group(model)
            errors.append(group['error'].values)
            colors.append(model_colors.get(model, 'gray'))
        else:
            # Add empty data for missing models
            errors.append([np.nan])  # Use NaN for missing data
            colors.append(model_colors.get(model, 'gray'))

    # Create box plot
    bp = ax1.boxplot(errors, labels=all_models, patch_artist=True,
                     showmeans=True, meanline=True)

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Customize mean lines
    for line in bp['means']:
        line.set_color('red')
        line.set_linewidth(2)

    # Add grid
    ax1.grid(True, alpha=0.3, axis='y')

    # Labels and title
    ax1.set_xlabel('Model Type', fontsize=12)
    ax1.set_ylabel('Relative Error (%)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')

    # Create statistics table
    ax2 = plt.subplot2grid((4, 1), (3, 0))
    ax2.axis('off')

    # Calculate statistics for table
    stats_data = []
    for model in all_models:
        if model in model_groups.groups:
            group = model_groups.get_group(model)
            error_data = group['error']
            stats_data.append([
                model,
                f'{error_data.mean():.2f}%',
                f'{error_data.std():.2f}%',
                f'{error_data.min():.2f}%',
                f'{error_data.max():.2f}%'
            ])
        else:
            stats_data.append([
                model,
                'N/A',
                'N/A',
                'N/A',
                'N/A'
            ])

    # Create table
    if stats_data:
        table = ax2.table(cellText=stats_data,
                         colLabels=['Model', 'Mean Error', 'Std Dev', 'Min Error', 'Max Error'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0, 0.8, 1])

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Header styling
        for i in range(5):
            cell = table[(0, i)]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')

        # Row coloring
        for i in range(1, len(stats_data) + 1):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E9EDF5')

    plt.tight_layout()
