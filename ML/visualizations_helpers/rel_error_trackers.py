"""
Relative error tracker plots organized by encoding method
FIXED: Connected graphs sharing y-axis, merged model cells, more padding
UPDATED: Added support for energy-discretized results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def create_rel_error_tracker_plots(df, output_dir, encodings, energy_group=None, target_type=None):
    """
    Create relative error plots organized by encoding method

    Args:
        df: DataFrame with test results
        output_dir: Directory to save plots
        encodings: List of encoding methods
        energy_group: Energy group to analyze ('thermal', 'epithermal', 'fast', 'total')
        target_type: 'flux' or 'keff' - if specified, only create that type
    """

    if target_type == 'keff':
        # Only create k-eff plots
        for encoding in encodings:
            create_encoding_error_plot(df, output_dir, encoding, 'keff',
                                     output_dir, energy_group=None)
    elif target_type:
        # Create flux plots for specific energy group
        # Create subdirectories
        os.makedirs(os.path.join(output_dir, 'max_rel_error'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'mean_rel_error'), exist_ok=True)

        for encoding in encodings:
            # Max flux error
            create_encoding_error_plot(df, output_dir, encoding, 'max_flux',
                                     os.path.join(output_dir, 'max_rel_error'),
                                     energy_group=energy_group)

            # Mean flux error
            create_encoding_error_plot(df, output_dir, encoding, 'mean_flux',
                                     os.path.join(output_dir, 'mean_rel_error'),
                                     energy_group=energy_group)
    else:
        # Create all plots (default behavior)
        # Create subdirectories
        os.makedirs(os.path.join(output_dir, 'max_rel_error'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'mean_rel_error'), exist_ok=True)
        if not energy_group:
            os.makedirs(os.path.join(output_dir, 'keff_rel_error'), exist_ok=True)

        for encoding in encodings:
            # Max flux error
            create_encoding_error_plot(df, output_dir, encoding, 'max_flux',
                                     os.path.join(output_dir, 'max_rel_error'),
                                     energy_group=energy_group)

            # Mean flux error
            create_encoding_error_plot(df, output_dir, encoding, 'mean_flux',
                                     os.path.join(output_dir, 'mean_rel_error'),
                                     energy_group=energy_group)

            # K-eff error (only if not energy-specific)
            if not energy_group:
                create_encoding_error_plot(df, output_dir, encoding, 'keff',
                                         os.path.join(output_dir, 'keff_rel_error'))

def create_encoding_error_plot(df, output_dir, encoding, error_type, subfolder, energy_group=None):
    """Create error plot for a specific encoding method with connected graphs sharing y-axis"""

    # Filter data for this encoding
    enc_df = df[df['encoding'] == encoding]

    # Prepare data based on error type
    if error_type == 'max_flux':
        plot_data = prepare_max_flux_data(enc_df, energy_group)
        if energy_group:
            title_base = f'Maximum {energy_group.capitalize()} Flux Relative Error - {encoding.upper()} Encoding'
        else:
            title_base = f'Maximum Flux Relative Error - {encoding.upper()} Encoding'
        ylabel = 'Relative Error (%)'
    elif error_type == 'mean_flux':
        plot_data = prepare_mean_flux_data(enc_df, energy_group)
        if energy_group:
            title_base = f'Mean {energy_group.capitalize()} Flux Relative Error - {encoding.upper()} Encoding'
        else:
            title_base = f'Mean Flux Relative Error - {encoding.upper()} Encoding'
        ylabel = 'Relative Error (%)'
    else:  # keff
        plot_data = prepare_keff_data(enc_df)
        title_base = f'K-effective Relative Error - {encoding.upper()} Encoding'
        ylabel = 'Relative Error (%)'

    if plot_data.empty:
        return

    # Define styling
    model_colors = {
        'xgboost': '#1f77b4',
        'random_forest': '#ff7f0e',
        'svm': '#2ca02c',
        'neural_net': '#d62728'
    }

    # Get unique values
    models = sorted(plot_data['model'].unique())
    optimizations = sorted(plot_data['optimization'].unique())

    # Debug output
    print(f"    Found models: {models}")
    print(f"    Found optimizations: {optimizations}")

    # Sort by config_id
    plot_data = plot_data.sort_values('config_id')

    # Calculate number of optimization methods
    n_opts = len(optimizations)

    # Create figure with more vertical space for padding
    fig = plt.figure(figsize=(18, 10))  # Fixed width, slightly reduced height

    # Create grid with less space between graphs and table
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 0.15, 1.5], hspace=0.25)  # Reduced spacing

    # Create connected subplots sharing y-axis
    axes = []

    # Create subplots that touch each other
    graph_gs = gs[0:3, 0].subgridspec(1, n_opts, wspace=0, hspace=0)  # No space between graphs

    for opt_idx, optimization in enumerate(optimizations):
        if opt_idx == 0:
            ax = fig.add_subplot(graph_gs[0, opt_idx])
            first_ax = ax
        else:
            ax = fig.add_subplot(graph_gs[0, opt_idx], sharey=first_ax)
        axes.append(ax)

        # Filter data for this optimization
        opt_data = plot_data[plot_data['optimization'] == optimization]

        # Check if there's any data for this optimization
        if len(opt_data) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, alpha=0.5)
        else:
            # Plot each model
            for model in models:
                model_data = opt_data[opt_data['model'] == model]

                if not model_data.empty:
                    # Sort by config_id to ensure proper line connection
                    model_data = model_data.sort_values('config_id')

                    # Determine markers for each point based on prediction direction
                    config_ids = []
                    errors = []
                    markers = []

                    for idx, row in model_data.iterrows():
                        signed_errors = row['signed_errors']
                        # Determine if configuration has more overpredictions or underpredictions
                        positive_count = sum(1 for e in signed_errors if e >= 0)
                        negative_count = sum(1 for e in signed_errors if e < 0)

                        if positive_count > negative_count:
                            marker = 'o'  # Circle for overprediction
                        elif negative_count > positive_count:
                            marker = 's'  # Square for underprediction
                        else:
                            # Tie: use mean to determine
                            mean_signed = np.mean(signed_errors)
                            if mean_signed >= 0:
                                marker = 'o'  # Circle for overprediction
                            else:
                                marker = 's'  # Square for underprediction

                        config_ids.append(row['config_id'])
                        errors.append(row['error'])  # error is already absolute
                        markers.append(marker)

                    # Plot ONE SINGLE LINE connecting all points
                    ax.plot(config_ids, errors,
                           color=model_colors.get(model, 'black'),
                           linestyle='-',
                           linewidth=2,
                           alpha=0.8,
                           label=f'{model}')

                    # Plot markers individually with different shapes for over/under prediction
                    for i, (config_id, error, marker) in enumerate(zip(config_ids, errors, markers)):
                        ax.plot(config_id, error,
                               color=model_colors.get(model, 'black'),
                               marker=marker,
                               markersize=6,
                               alpha=0.8)

        # Customize subplot
        ax.set_xlabel('Configuration ID', fontsize=11)

        # Only show y-label and ticks on leftmost plot
        if opt_idx == 0:
            ax.set_ylabel(ylabel, fontsize=11)
        else:
            ax.tick_params(axis='y', labelleft=False)

        ax.set_title(f'{optimization.upper()}', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        # Set x-axis limits to be consistent
        ax.set_xlim(plot_data['config_id'].min() - 1, plot_data['config_id'].max() + 1)

        # Add vertical separator line (except for last plot)
        if opt_idx < n_opts - 1:
            ax.axvline(x=ax.get_xlim()[1], color='black', linewidth=1.5, alpha=0.8)

    # Main title
    fig.suptitle(title_base, fontsize=14, fontweight='bold', y=0.98)

    # Create statistics table in the bottom grid space
    table_ax = fig.add_subplot(gs[4, :])  # Skip row 3 for padding
    table_ax.axis('off')

    # Calculate statistics for table
    stats_data = []
    for model in models:
        has_data_for_model = False
        for optimization in optimizations:
            subset = plot_data[
                (plot_data['model'] == model) &
                (plot_data['optimization'] == optimization)
            ]
            if len(subset) > 0:
                error_data = subset['error']  # These are already absolute values

                # Calculate over/under prediction percentages using signed_errors
                overpred_count = 0
                underpred_count = 0
                total_count = 0

                for _, row in subset.iterrows():
                    signed_errors = row['signed_errors']
                    for signed_error in signed_errors:
                        total_count += 1
                        if signed_error >= 0:
                            overpred_count += 1
                        else:
                            underpred_count += 1

                overpred_percent = (overpred_count / total_count * 100) if total_count > 0 else 0
                underpred_percent = (underpred_count / total_count * 100) if total_count > 0 else 0

                stats_data.append([
                    model,
                    optimization,
                    f'{error_data.mean():.3f}%',  # Already absolute values
                    f'{error_data.max():.3f}%',   # Already absolute values
                    f'{error_data.min():.3f}%',   # Already absolute values
                    f'{error_data.std():.3f}%',   # Already absolute values
                    f'{overpred_percent:.1f}%',
                    f'{underpred_percent:.1f}%'
                ])
                has_data_for_model = True

        # If no data for any optimization, skip this model
        if not has_data_for_model:
            print(f"  Warning: No data found for model '{model}'")

    # Create table
    if stats_data:
        table = table_ax.table(cellText=stats_data,
                             colLabels=['Model', 'Optimization', 'Mean', 'Max', 'Min', 'Std Dev', '% Over', '% Under'],
                             cellLoc='center',
                             loc='center',
                             bbox=[0.05, 0.1, 0.9, 0.8])  # Wider table for additional columns

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2.0)

        # Header styling
        for i in range(8):  # Updated to include new columns
            cell = table[(0, i)]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')

        # Row styling
        current_model = None
        first_row_of_model = {}

        for i in range(1, len(stats_data) + 1):
            model_name = stats_data[i-1][0]

            # Track first row of each model
            if model_name != current_model:
                current_model = model_name
                first_row_of_model[model_name] = i

            # Style model cell
            model_cell = table[(i, 0)]
            model_cell.set_facecolor(model_colors.get(model_name, 'gray'))
            model_cell.set_text_props(weight='bold', color='white')

            # Clear model name for non-first rows of same model
            if i != first_row_of_model[model_name]:
                model_cell.get_text().set_text('')

            # Style other cells
            for j in range(1, 8):  # Updated to include new columns
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E9EDF5')
                else:
                    table[(i, j)].set_facecolor('#FFFFFF')

    plt.tight_layout()

    # Save figure with energy group in filename
    if energy_group:
        filename = f'{encoding}_{energy_group}_{error_type}_error.png'
    else:
        filename = f'{encoding}_error.png'

    output_file = os.path.join(subfolder, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

def prepare_max_flux_data(df, energy_group=None):
    """Prepare maximum flux error data"""
    plot_data = []

    for _, row in df.iterrows():
        errors = []
        for i in range(1, 5):
            if energy_group:
                # Energy-specific column
                error_col = f'I_{i}_{energy_group}_rel_error'
            else:
                # Standard column
                error_col = f'I_{i}_rel_error'

            if error_col in row:
                error = row[error_col]
                if pd.notna(error) and error != 'N/A':
                    errors.append(error)  # Keep signed values for marker differentiation

        if errors:
            # Store both signed errors (for marker determination) and max absolute error (for table stats)
            signed_errors = errors
            max_abs_error = max([abs(e) for e in errors])

            plot_data.append({
                'config_id': row['config_id'],
                'model': row['model_class'],
                'optimization': row['optimization_method'],
                'error': max_abs_error,  # Table uses absolute max
                'signed_errors': signed_errors  # Store for marker determination
            })

    return pd.DataFrame(plot_data)

def prepare_mean_flux_data(df, energy_group=None):
    """Prepare mean flux error data"""
    plot_data = []

    for _, row in df.iterrows():
        errors = []
        for i in range(1, 5):
            if energy_group:
                # Energy-specific column
                error_col = f'I_{i}_{energy_group}_rel_error'
            else:
                # Standard column
                error_col = f'I_{i}_rel_error'

            if error_col in row:
                error = row[error_col]
                if pd.notna(error) and error != 'N/A':
                    errors.append(error)  # Keep signed values for marker differentiation

        if errors:
            # Store both signed errors (for marker determination) and mean absolute error (for table stats)
            signed_errors = errors
            mean_abs_error = np.mean([abs(e) for e in errors])

            plot_data.append({
                'config_id': row['config_id'],
                'model': row['model_class'],
                'optimization': row['optimization_method'],
                'error': mean_abs_error,  # Table uses absolute mean
                'signed_errors': signed_errors  # Store for marker determination
            })

    return pd.DataFrame(plot_data)

def prepare_keff_data(df):
    """Prepare k-eff error data"""
    plot_data = []

    for _, row in df.iterrows():
        if 'keff_rel_error' in row:
            error = row['keff_rel_error']
            if pd.notna(error) and error != 'N/A':
                # For k-eff: store both signed error (for markers) and absolute error (for table stats)
                plot_data.append({
                    'config_id': row['config_id'],
                    'model': row['model_class'],
                    'optimization': row['optimization_method'],
                    'error': abs(error),  # Table uses absolute error
                    'signed_errors': [error]  # Store signed error for marker determination
                })

    return pd.DataFrame(plot_data)
