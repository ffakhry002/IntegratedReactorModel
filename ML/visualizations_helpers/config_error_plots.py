"""
Configuration error plots showing error trends across test configurations
FIXED: Simplified to avoid overcrowding
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def create_config_error_plots(df, output_dir):
    """Create config vs error plots for max flux, mean flux, and k-eff"""

    # Create three plots
    create_max_flux_error_plot(df, output_dir)
    create_mean_flux_error_plot(df, output_dir)
    create_keff_error_plot(df, output_dir)

def create_max_flux_error_plot(df, output_dir):
    """Create plot showing maximum flux error for each configuration"""

    plt.figure(figsize=(12, 8))

    # Calculate max error for each configuration
    plot_data = prepare_flux_error_data(df, 'max')

    # Create the plot
    create_simplified_error_plot(plot_data, 'Maximum Flux Relative Error by Configuration')

    # Save figure
    output_file = os.path.join(output_dir, 'max_flux_error_by_config.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

def create_mean_flux_error_plot(df, output_dir):
    """Create plot showing mean flux error for each configuration"""

    plt.figure(figsize=(12, 8))

    # Calculate mean error for each configuration
    plot_data = prepare_flux_error_data(df, 'mean')

    # Create the plot
    create_simplified_error_plot(plot_data, 'Mean Flux Relative Error by Configuration')

    # Save figure
    output_file = os.path.join(output_dir, 'mean_flux_error_by_config.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

def create_keff_error_plot(df, output_dir):
    """Create plot showing k-eff error for each configuration"""

    plt.figure(figsize=(12, 8))

    # Prepare k-eff error data
    plot_data = prepare_keff_error_data(df)

    # Create the plot
    create_simplified_error_plot(plot_data, 'K-effective Relative Error by Configuration')

    # Save figure
    output_file = os.path.join(output_dir, 'keff_error_by_config.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

def prepare_flux_error_data(df, error_type):
    """Prepare flux error data for plotting"""
    plot_data = []

    # Get unique combinations
    for _, row in df.iterrows():
        # Calculate flux errors
        errors = []
        for i in range(1, 5):
            real_col = f'I_{i}_real'
            pred_col = f'I_{i}_predicted'

            if real_col in row and pred_col in row:
                real_val = row[real_col]
                pred_val = row[pred_col]

                if pd.notna(real_val) and pd.notna(pred_val) and real_val != 0:
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

    # Group by model and calculate aggregated errors
    model_groups = plot_data.groupby('model')

    # Create box plots for each model
    models = []
    errors = []
    colors = []

    for model, group in model_groups:
        models.append(model)
        errors.append(group['error'].values)
        colors.append(model_colors.get(model, 'gray'))

    # Create box plot
    bp = ax1.boxplot(errors, labels=models, patch_artist=True,
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
    for model, group in model_groups:
        mean_err = group['error'].mean()
        std_err = group['error'].std()
        min_err = group['error'].min()
        max_err = group['error'].max()
        stats_data.append([
            model,
            f'{mean_err:.2f}%',
            f'{std_err:.2f}%',
            f'{min_err:.2f}%',
            f'{max_err:.2f}%'
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
