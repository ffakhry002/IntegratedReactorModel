"""
Relative error tracker plots organized by encoding method
FIXED: Statistics moved to table below plot
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def create_rel_error_tracker_plots(df, output_dir, encodings):
    """Create relative error plots organized by encoding method"""

    # Create plots for each error type and encoding
    for encoding in encodings:
        # Max flux error
        create_encoding_error_plot(df, output_dir, encoding, 'max_flux',
                                 os.path.join(output_dir, 'max_rel_error'))

        # Mean flux error
        create_encoding_error_plot(df, output_dir, encoding, 'mean_flux',
                                 os.path.join(output_dir, 'mean_rel_error'))

        # K-eff error
        create_encoding_error_plot(df, output_dir, encoding, 'keff',
                                 os.path.join(output_dir, 'keff_rel_error'))

# rel_error_trackers.py - UPDATED create_encoding_error_plot function

def create_encoding_error_plot(df, output_dir, encoding, error_type, subfolder):
    """Create error plot for a specific encoding method with statistics table"""

    # Filter data for this encoding
    enc_df = df[df['encoding'] == encoding]

    # Prepare data based on error type
    if error_type == 'max_flux':
        plot_data = prepare_max_flux_data(enc_df)
        title = f'Maximum Flux Relative Error - {encoding.upper()} Encoding'
        ylabel = 'Max Relative Error (%)'
    elif error_type == 'mean_flux':
        plot_data = prepare_mean_flux_data(enc_df)
        title = f'Mean Flux Relative Error - {encoding.upper()} Encoding'
        ylabel = 'Mean Relative Error (%)'
    else:  # keff
        plot_data = prepare_keff_data(enc_df)
        title = f'K-effective Relative Error - {encoding.upper()} Encoding'
        ylabel = 'Relative Error (%)'

    if plot_data.empty:
        return

    # Create figure with subplot for table
    fig = plt.figure(figsize=(12, 10))

    # Main plot
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)

    # Define styling
    model_colors = {
        'xgboost': '#1f77b4',
        'random_forest': '#ff7f0e',
        'svm': '#2ca02c',
        'neural_net': '#d62728'
    }

    optimization_markers = {
        'optuna': 'o',
        'three_stage': 's',
        'none': '^'
    }

    # Line styles for optimization methods
    optimization_lines = {
        'optuna': '-',        # solid
        'three_stage': '--',  # dashed
        'none': ':'          # dotted
    }

    # Get unique values
    models = plot_data['model'].unique()
    optimizations = plot_data['optimization'].unique()

    # Sort by config_id
    plot_data = plot_data.sort_values('config_id')

    # Plot each model-optimization combination
    for model in models:
        for optimization in optimizations:
            subset = plot_data[
                (plot_data['model'] == model) &
                (plot_data['optimization'] == optimization)
            ]

            if not subset.empty:
                ax1.plot(subset['config_id'], subset['error'],
                        color=model_colors.get(model, 'black'),
                        marker=optimization_markers.get(optimization, 'o'),
                        linestyle=optimization_lines.get(optimization, '-'),  # Use line style mapping
                        markersize=8,
                        linewidth=2,
                        alpha=0.8,
                        label=f'{model} ({optimization})')

    # Customize plot
    ax1.set_xlabel('Configuration ID', fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)

    # Create statistics table
    ax2 = plt.subplot2grid((4, 1), (3, 0))
    ax2.axis('off')

    # Calculate statistics for table
    stats_data = []
    for model in models:
        for optimization in optimizations:
            subset = plot_data[
                (plot_data['model'] == model) &
                (plot_data['optimization'] == optimization)
            ]
            if len(subset) > 0:
                error_data = subset['error']
                stats_data.append([
                    model,
                    optimization,
                    f'{error_data.mean():.3f}%',  # Changed to 3 decimal places
                    f'{error_data.max():.3f}%',
                    f'{error_data.min():.3f}%',
                    f'{error_data.std():.3f}%'
                ])

    # Sort stats_data for consistent ordering
    stats_data.sort(key=lambda x: (x[0], x[1]))

    # Create table
    if stats_data:
        table = ax2.table(cellText=stats_data,
                         colLabels=['Model', 'Optimization', 'Mean', 'Max', 'Min', 'Std Dev'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.05, 0, 0.9, 1])

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Header styling
        for i in range(6):
            cell = table[(0, i)]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')

        # Row coloring
        for i in range(1, len(stats_data) + 1):
            for j in range(6):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E9EDF5')

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(subfolder, f'{encoding}_error.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

def prepare_max_flux_data(df):
    """Prepare maximum flux error data"""
    plot_data = []

    for _, row in df.iterrows():
        errors = []
        for i in range(1, 5):
            if f'I_{i}_rel_error' in row:
                error = row[f'I_{i}_rel_error']
                if pd.notna(error):
                    errors.append(error)

        if errors:
            plot_data.append({
                'config_id': row['config_id'],
                'model': row['model_class'],
                'optimization': row['optimization_method'],
                'error': max(errors)
            })

    return pd.DataFrame(plot_data)

def prepare_mean_flux_data(df):
    """Prepare mean flux error data"""
    plot_data = []

    for _, row in df.iterrows():
        errors = []
        for i in range(1, 5):
            if f'I_{i}_rel_error' in row:
                error = row[f'I_{i}_rel_error']
                if pd.notna(error):
                    errors.append(error)

        if errors:
            plot_data.append({
                'config_id': row['config_id'],
                'model': row['model_class'],
                'optimization': row['optimization_method'],
                'error': np.mean(errors)
            })

    return pd.DataFrame(plot_data)

def prepare_keff_data(df):
    """Prepare k-eff error data"""
    plot_data = []

    for _, row in df.iterrows():
        if 'keff_rel_error' in row:
            error = row['keff_rel_error']
            if pd.notna(error):
                plot_data.append({
                    'config_id': row['config_id'],
                    'model': row['model_class'],
                    'optimization': row['optimization_method'],
                    'error': error
                })

    return pd.DataFrame(plot_data)
