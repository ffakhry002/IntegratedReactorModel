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
    bp = plt.boxplot(errors, labels=models, patch_artist=True,
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
    plt.grid(True, alpha=0.3, axis='y')

    # Labels and title
    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('Relative Error (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')

    # Add statistics below the plot
    stats_text = "Statistics by Model:\n"
    for model, group in model_groups:
        mean_err = group['error'].mean()
        std_err = group['error'].std()
        min_err = group['error'].min()
        max_err = group['error'].max()
        stats_text += f"{model}: Mean={mean_err:.2f}%, Std={std_err:.2f}%, Min={min_err:.2f}%, Max={max_err:.2f}%\n"

    plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
