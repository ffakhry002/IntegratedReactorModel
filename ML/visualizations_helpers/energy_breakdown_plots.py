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
    Create stacked bar charts showing energy distribution for real vs predicted flux

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
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"  ✓ Created energy breakdown plot: {filename}")

    # Create summary plot showing average distribution across all models
    create_summary_energy_distribution(df, output_dir)

def create_summary_energy_distribution(df, output_dir):
    """Create a summary plot showing average energy distribution across all models"""

    # Set style to white background
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'

    energy_groups = ['thermal', 'epithermal', 'fast']

    # Calculate average percentages for each energy group
    models = df['model_class'].unique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left plot: Average distribution by model (sum of all positions)
    model_data = {model: {energy: [] for energy in energy_groups} for model in models}

    for model in models:
        model_subset = df[df['model_class'] == model]

        for _, row in model_subset.iterrows():
            total_real = 0
            energy_real = {energy: 0 for energy in energy_groups}

            # Sum across ALL positions (treating them equally)
            for pos in range(1, 5):
                for energy in energy_groups:
                    real_col = f'I_{pos}_{energy}_real'
                    if real_col in row and pd.notna(row[real_col]) and row[real_col] != 'N/A':
                        energy_real[energy] += row[real_col]
                        total_real += row[real_col]

            # Calculate percentages
            if total_real > 0:
                for energy in energy_groups:
                    model_data[model][energy].append(energy_real[energy] / total_real * 100)

    # Plot average percentages for each model
    x = np.arange(len(models))
    width = 0.25

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, energy in enumerate(energy_groups):
        means = []
        stds = []

        for model in models:
            if model_data[model][energy]:
                means.append(np.mean(model_data[model][energy]))
                stds.append(np.std(model_data[model][energy]))
            else:
                means.append(0)
                stds.append(0)

        ax1.bar(x + i*width, means, width, yerr=stds, label=energy.capitalize(),
               color=colors[i], alpha=0.8, capsize=5)

    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Average Percentage (%)', fontsize=12)
    ax1.set_title('Average Energy Distribution by Model\n(Sum of All Positions)',
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    # Right plot: Prediction accuracy by energy group (sum of all positions)
    energy_errors = {energy: [] for energy in energy_groups}

    for _, row in df.iterrows():
        for energy in energy_groups:
            # Sum real and predicted values across all positions
            total_real = 0
            total_pred = 0

            for pos in range(1, 5):
                real_col = f'I_{pos}_{energy}_real'
                pred_col = f'I_{pos}_{energy}_predicted'

                if (real_col in row and pred_col in row and
                    pd.notna(row[real_col]) and pd.notna(row[pred_col]) and
                    row[real_col] != 'N/A' and row[pred_col] != 'N/A'):

                    total_real += row[real_col]
                    total_pred += row[pred_col]

            # Calculate relative error for the sum
            if total_real > 0:
                rel_error = abs((total_pred - total_real) / total_real) * 100
                energy_errors[energy].append(rel_error)

    # Create box plot
    data_to_plot = [energy_errors[energy] for energy in energy_groups]
    positions = range(len(energy_groups))

    bp = ax2.boxplot(data_to_plot, positions=positions, patch_artist=True,
                    showmeans=True, meanline=True)

    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_xticklabels([e.capitalize() for e in energy_groups])
    ax2.set_ylabel('Relative Error (%)', fontsize=12)
    ax2.set_title('Prediction Error Distribution by Energy Group\n(Sum of All Positions)',
                 fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)

    # Add mean values as text
    for i, energy in enumerate(energy_groups):
        if energy_errors[energy]:
            mean_val = np.mean(energy_errors[energy])
            ax2.text(i, ax2.get_ylim()[1] * 0.95, f'Mean: {mean_val:.1f}%',
                    ha='center', va='top', fontsize=10)

    plt.tight_layout()

    # Save plot
    filepath = os.path.join(output_dir, 'energy_distribution_summary.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Created energy distribution summary plot")
