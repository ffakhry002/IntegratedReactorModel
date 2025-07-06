"""
Energy breakdown visualization for multi-energy group flux predictions
Creates stacked bar charts comparing real vs predicted flux distributions
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import os

def create_energy_breakdown_plots(df, output_dir):
    """
    Create comprehensive energy breakdown visualizations including:
    1. Stacked bar charts showing energy distribution
    2. Flux vs configuration line plots
    3. Maximum error configuration plots
    4. Summary energy distribution plots

    Args:
        df: DataFrame with test results containing energy-discretized flux
        output_dir: Directory to save plots
    """
    # Set style to white background
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'

    # Define energy groups and colors
    energy_groups = ['thermal', 'epithermal', 'fast']
    colors = {
        'thermal': '#FF6B6B',      # Red
        'epithermal': '#4ECDC4',   # Teal
        'fast': '#45B7D1'          # Blue
    }

    # Get unique models
    models = df['model_class'].unique()
    encodings = df['encoding'].unique()
    optimizations = df['optimization_method'].unique()

    # Create plots for each model/encoding/optimization combination
    for model in models:
        # Create model-specific subdirectory
        model_dir = os.path.join(output_dir, model)
        os.makedirs(model_dir, exist_ok=True)

        for encoding in encodings:
            for optimization in optimizations:
                # Filter data
                mask = (
                    (df['model_class'] == model) &
                    (df['encoding'] == encoding) &
                    (df['optimization_method'] == optimization)
                )
                subset = df[mask]

                if len(subset) == 0:
                    continue

                # Prepare data for plotting
                config_ids = sorted(subset['config_id'].unique())
                n_configs = len(config_ids)

                # Calculate grid dimensions - 5 configs per row
                n_cols = 5
                n_rows = int(np.ceil(n_configs / n_cols))

                # Create figure with subplots
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))

                # Flatten axes for easier indexing
                if n_rows == 1 and n_cols == 1:
                    axes = np.array([[axes]])
                elif n_rows == 1 or n_cols == 1:
                    axes = axes.reshape(n_rows, n_cols)

                # Process each configuration
                for config_idx, config_id in enumerate(config_ids):
                    row = config_idx // n_cols
                    col = config_idx % n_cols
                    ax = axes[row, col]

                    config_data = subset[subset['config_id'] == config_id].iloc[0]

                    # Initialize data for 4 positions
                    positions = ['I_1', 'I_2', 'I_3', 'I_4']
                    bar_data = []
                    bar_labels = []

                    # Collect data for each position (real then predicted)
                    for pos_idx, pos in enumerate(positions):
                        pos_num = pos_idx + 1

                        # Real values - stacked for this position
                        real_bottom = 0
                        real_values = []
                        for energy in energy_groups:
                            real_col = f'I_{pos_num}_{energy}_real'
                            if real_col in config_data:
                                val = config_data[real_col]
                                if pd.notna(val) and val != 'N/A':
                                    real_values.append((energy, val, real_bottom))
                                    real_bottom += val

                        # Predicted values - stacked for this position
                        pred_bottom = 0
                        pred_values = []
                        for energy in energy_groups:
                            pred_col = f'I_{pos_num}_{energy}_predicted'
                            if pred_col in config_data:
                                val = config_data[pred_col]
                                if pd.notna(val) and val != 'N/A':
                                    pred_values.append((energy, val, pred_bottom))
                                    pred_bottom += val

                        bar_data.append(('real', pos_num, real_values))
                        bar_data.append(('pred', pos_num, pred_values))
                        bar_labels.extend([f'{pos}\nReal', f'{pos}\nPred'])

                    # Create bars
                    bar_width = 0.8
                    x_positions = np.arange(len(bar_labels))

                    # Plot stacked bars
                    for i, (bar_type, pos_num, values) in enumerate(bar_data):
                        for energy, value, bottom in values:
                            alpha = 0.8 if bar_type == 'real' else 0.5
                            hatch = None if bar_type == 'real' else '//'

                            ax.bar(i, value, bar_width,
                                  bottom=bottom,
                                  color=colors[energy],
                                  alpha=alpha,
                                  edgecolor='black',
                                  linewidth=1,
                                  hatch=hatch)

                    # Customize subplot
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(bar_labels, fontsize=8)
                    ax.set_ylabel('Flux', fontsize=9)
                    ax.set_title(f'Config {config_id}', fontsize=10, fontweight='bold')
                    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
                    ax.grid(True, axis='y', alpha=0.3)
                    ax.set_xlim(-0.5, len(x_positions) - 0.5)

                # Hide empty subplots
                for i in range(n_configs, n_rows * n_cols):
                    row = i // n_cols
                    col = i % n_cols
                    axes[row, col].set_visible(False)

                # Create legend
                legend_elements = []
                for energy in energy_groups:
                    legend_elements.append(
                        Rectangle((0, 0), 1, 1, fc=colors[energy], alpha=0.8,
                                 edgecolor='black', linewidth=1, label=f'{energy.capitalize()} (Real)')
                    )
                    legend_elements.append(
                        Rectangle((0, 0), 1, 1, fc=colors[energy], alpha=0.5,
                                 edgecolor='black', linewidth=1, hatch='//', label=f'{energy.capitalize()} (Pred)')
                    )

                # Place legend at the bottom
                fig.legend(handles=legend_elements, loc='lower center',
                          bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize=10)

                # Main title
                fig.suptitle(f'Energy Distribution by Configuration: {model.upper()} - {encoding} - {optimization}',
                            fontsize=14, fontweight='bold')

                # Adjust layout
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.1)

                # Save plot
                filename = f'energy_breakdown_{model}_{encoding}_{optimization}.png'
                filepath = os.path.join(model_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"  ✓ Created energy breakdown plot: {model}/{filename}")

    # Create relative error vs configuration plots (6 panels: 4 positions + mean + max)
    print("\n📈 Creating relative error vs configuration plots...")
    create_flux_vs_config_plots(df, output_dir)

def create_flux_vs_config_plots(df, output_dir):
    """
    Create plots showing relative errors vs configuration with 6 panels:
    - 4 panels for individual positions (I_1, I_2, I_3, I_4)
    - 1 panel for mean error across all positions
    - 1 panel for max error across all positions
    Shows thermal, epithermal, fast, and total error lines using rel_error_tracker style

    Args:
        df: DataFrame with test results containing energy-discretized flux
        output_dir: Directory to save plots
    """
    # Set style to match rel_error_trackers
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'

    # Define colors for energy groups (matching rel_error_trackers style)
    colors = {
        'thermal': '#FF6B6B',    # Red
        'epithermal': '#4ECDC4', # Teal
        'fast': '#45B7D1',       # Blue
        'total': '#2D3748'       # Dark gray
    }

    # Get unique models for separate plots
    models = df['model_class'].unique()
    encodings = df['encoding'].unique()
    optimizations = df['optimization_method'].unique()

    for model in models:
        # Create model-specific subdirectory
        model_dir = os.path.join(output_dir, model)
        os.makedirs(model_dir, exist_ok=True)

        for encoding in encodings:
            for optimization in optimizations:
                # Filter data
                mask = (
                    (df['model_class'] == model) &
                    (df['encoding'] == encoding) &
                    (df['optimization_method'] == optimization)
                )
                subset = df[mask]

                if len(subset) == 0:
                    continue

                # Sort by config_id for proper line plotting
                subset = subset.sort_values('config_id')
                config_ids = subset['config_id'].values

                # Create 3x2 subplot (6 panels: 4 positions + mean + max)
                fig, axes = plt.subplots(3, 2, figsize=(16, 18))

                positions = ['I_1', 'I_2', 'I_3', 'I_4']

                # Plot individual position errors (panels 0-3)
                for pos_idx, pos in enumerate(positions):
                    row = pos_idx // 2
                    col = pos_idx % 2
                    ax = axes[row, col]
                    pos_num = pos_idx + 1

                    # Plot each energy group + total relative errors
                    energy_groups = ['thermal', 'epithermal', 'fast', 'total']

                    for energy in energy_groups:
                        # Get relative error values
                        error_col = f'I_{pos_num}_{energy}_rel_error'

                        if error_col in subset.columns:
                            error_values = subset[error_col].values

                            # Filter out N/A and NaN values
                            valid_mask = pd.notna(error_values) & (error_values != 'N/A')

                            valid_configs = config_ids[valid_mask]
                            valid_errors = error_values[valid_mask]

                            if len(valid_errors) > 0:
                                # Convert to absolute values and ensure numeric
                                valid_errors = [abs(float(err)) for err in valid_errors]

                                # Plot using rel_error_tracker style
                                ax.plot(valid_configs, valid_errors,
                                       color=colors[energy],
                                       marker='o',
                                       linestyle='-',
                                       markersize=6,
                                       linewidth=2,
                                       alpha=0.8,
                                       label=f'{energy.capitalize()}')

                    # Customize subplot
                    ax.set_xlabel('Configuration ID', fontsize=11)
                    ax.set_ylabel('Relative Error (%)', fontsize=11)
                    ax.set_title(f'{pos} Position', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='best', fontsize=9)

                # Calculate mean errors across all positions (panel 4)
                ax_mean = axes[2, 0]
                energy_groups = ['thermal', 'epithermal', 'fast', 'total']

                for energy in energy_groups:
                    config_mean_errors = []
                    valid_config_ids = []

                    for _, row in subset.iterrows():
                        position_errors = []
                        for pos_num in range(1, 5):
                            error_col = f'I_{pos_num}_{energy}_rel_error'
                            if error_col in row and pd.notna(row[error_col]) and row[error_col] != 'N/A':
                                position_errors.append(abs(float(row[error_col])))

                        if position_errors:
                            config_mean_errors.append(np.mean(position_errors))
                            valid_config_ids.append(row['config_id'])

                    if config_mean_errors:
                        ax_mean.plot(valid_config_ids, config_mean_errors,
                                   color=colors[energy],
                                   marker='o',
                                   linestyle='-',
                                   markersize=6,
                                   linewidth=2,
                                   alpha=0.8,
                                   label=f'{energy.capitalize()}')

                ax_mean.set_xlabel('Configuration ID', fontsize=11)
                ax_mean.set_ylabel('Mean Relative Error (%)', fontsize=11)
                ax_mean.set_title('Mean Error Across All Positions', fontsize=12, fontweight='bold')
                ax_mean.grid(True, alpha=0.3)
                ax_mean.legend(loc='best', fontsize=9)

                # Calculate max errors across all positions (panel 5)
                ax_max = axes[2, 1]

                for energy in energy_groups:
                    config_max_errors = []
                    valid_config_ids = []

                    for _, row in subset.iterrows():
                        position_errors = []
                        for pos_num in range(1, 5):
                            error_col = f'I_{pos_num}_{energy}_rel_error'
                            if error_col in row and pd.notna(row[error_col]) and row[error_col] != 'N/A':
                                position_errors.append(abs(float(row[error_col])))

                        if position_errors:
                            config_max_errors.append(max(position_errors))
                            valid_config_ids.append(row['config_id'])

                    if config_max_errors:
                        ax_max.plot(valid_config_ids, config_max_errors,
                                  color=colors[energy],
                                  marker='o',
                                  linestyle='-',
                                  markersize=6,
                                  linewidth=2,
                                  alpha=0.8,
                                  label=f'{energy.capitalize()}')

                ax_max.set_xlabel('Configuration ID', fontsize=11)
                ax_max.set_ylabel('Max Relative Error (%)', fontsize=11)
                ax_max.set_title('Max Error Across All Positions', fontsize=12, fontweight='bold')
                ax_max.grid(True, alpha=0.3)
                ax_max.legend(loc='best', fontsize=9)

                # Main title
                fig.suptitle(f'Relative Error vs Configuration: {model.upper()} - {encoding} - {optimization}',
                            fontsize=16, fontweight='bold')

                plt.tight_layout()

                # Save plot
                filename = f'error_vs_config_{model}_{encoding}_{optimization}.png'
                filepath = os.path.join(model_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"  ✓ Created error vs config plot: {model}/{filename}")
