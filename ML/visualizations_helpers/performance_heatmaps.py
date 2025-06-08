"""
Performance heatmaps showing R² scores for all model combinations
FIXED: Clarified difference between per-optimization and averaged heatmaps
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from .data_loader import get_model_aggregated_metrics

def create_performance_heatmaps(df, output_dir):
    """Create R² heatmaps for k-eff and flux predictions"""

    # Get aggregated metrics
    aggregated_df = get_model_aggregated_metrics(df)

    # Create separate heatmaps for flux and k-eff
    create_r2_heatmap(aggregated_df, 'flux', output_dir)
    create_r2_heatmap(aggregated_df, 'keff', output_dir)

def create_r2_heatmap(aggregated_df, target_type, output_dir):
    """Create R² heatmaps showing both individual optimization results and averages"""

    # Filter for target type
    metric_col = f'r2_{target_type}'

    # Check if we have data for this target type
    if metric_col not in aggregated_df.columns:
        print(f"  Warning: No R² data found for {target_type}. Skipping heatmap.")
        return

    # Get unique values
    models = sorted(aggregated_df['model_class'].unique())
    encodings = sorted(aggregated_df['encoding'].unique())
    optimizations = sorted(aggregated_df['optimization_method'].unique())

    # Convert to lists if needed
    if hasattr(models, '__len__'):
        models = list(models)
    if hasattr(encodings, '__len__'):
        encodings = list(encodings)
    if hasattr(optimizations, '__len__'):
        optimizations = list(optimizations)

    if len(models) == 0 or len(encodings) == 0 or len(optimizations) == 0:
        print(f"  Warning: Insufficient data for {target_type} heatmap. Skipping.")
        return

    # For single optimization, only create one heatmap
    if len(optimizations) == 1:
        create_single_optimization_heatmap(aggregated_df, target_type, output_dir,
                                         models, encodings, optimizations[0])
    else:
        # Create per-optimization heatmaps
        create_per_optimization_heatmaps(aggregated_df, target_type, output_dir,
                                       models, encodings, optimizations)

        # Create averaged heatmap
        create_averaged_r2_heatmap(aggregated_df, target_type, output_dir,
                                 models, encodings)

def create_single_optimization_heatmap(aggregated_df, target_type, output_dir,
                                     models, encodings, optimization):
    """Create a single heatmap when there's only one optimization method"""

    metric_col = f'r2_{target_type}'

    # Create figure
    plt.figure(figsize=(10, 8))

    # Create matrix
    matrix = np.zeros((len(encodings), len(models)))

    for i, encoding in enumerate(encodings):
        for j, model in enumerate(models):
            # Find the corresponding R² value
            mask = (aggregated_df['model_class'] == model) & \
                   (aggregated_df['encoding'] == encoding) & \
                   (aggregated_df['optimization_method'] == optimization)

            if mask.any():
                value = aggregated_df.loc[mask, metric_col].values[0]
                if pd.notna(value):
                    matrix[i, j] = value
                else:
                    matrix[i, j] = np.nan
            else:
                matrix[i, j] = np.nan

    # Create heatmap
    sns.heatmap(matrix,
                xticklabels=models,
                yticklabels=encodings,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                vmin=0.7,
                vmax=1.0,
                cbar_kws={'label': 'R² Score'})

    plt.title(f'Model Performance - R² Scores for {target_type.upper()} Prediction\n' +
              f'Optimization Method: {optimization}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Encoding Method', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, f'r2_heatmap_{target_type}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

def create_per_optimization_heatmaps(aggregated_df, target_type, output_dir,
                                   models, encodings, optimizations):
    """Create heatmaps for each optimization method side by side"""

    metric_col = f'r2_{target_type}'

    # Create figure with subplots for each optimization
    fig, axes = plt.subplots(1, len(optimizations), figsize=(6*len(optimizations), 8))
    if len(optimizations) == 1:
        axes = [axes]

    fig.suptitle(f'Model Performance by Optimization - R² Scores for {target_type.upper()} Prediction',
                 fontsize=16, fontweight='bold')

    vmin = 0.7  # Minimum R² for color scale
    vmax = 1.0  # Maximum R² for color scale

    for idx, optimization in enumerate(optimizations):
        ax = axes[idx]

        # Create matrix for this optimization
        matrix = np.zeros((len(encodings), len(models)))

        for i, encoding in enumerate(encodings):
            for j, model in enumerate(models):
                # Find the corresponding R² value
                mask = (aggregated_df['model_class'] == model) & \
                       (aggregated_df['encoding'] == encoding) & \
                       (aggregated_df['optimization_method'] == optimization)

                if mask.any():
                    value = aggregated_df.loc[mask, metric_col].values[0]
                    if pd.notna(value):
                        matrix[i, j] = value
                    else:
                        matrix[i, j] = np.nan
                else:
                    matrix[i, j] = np.nan

        # Create heatmap
        sns.heatmap(matrix,
                    xticklabels=models,
                    yticklabels=encodings,
                    annot=True,
                    fmt='.3f',
                    cmap='RdYlGn',
                    vmin=vmin,
                    vmax=vmax,
                    cbar_kws={'label': 'R² Score'},
                    ax=ax)

        ax.set_title(f'Optimization: {optimization}', fontsize=12)
        ax.set_xlabel('Model', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Encoding Method', fontsize=10)
        else:
            ax.set_ylabel('')

        # Rotate x labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, f'r2_heatmap_{target_type}_by_optimization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

def create_averaged_r2_heatmap(aggregated_df, target_type, output_dir, models, encodings):
    """Create a heatmap showing R² scores averaged across all optimization methods"""

    metric_col = f'r2_{target_type}'

    # Create matrix averaging across optimizations
    matrix = np.zeros((len(encodings), len(models)))
    count_matrix = np.zeros((len(encodings), len(models)))

    for i, encoding in enumerate(encodings):
        for j, model in enumerate(models):
            # Get all R² values for this model-encoding combination
            mask = (aggregated_df['model_class'] == model) & \
                   (aggregated_df['encoding'] == encoding)

            if mask.any():
                values = aggregated_df.loc[mask, metric_col].dropna()
                if len(values) > 0:
                    matrix[i, j] = values.mean()
                    count_matrix[i, j] = len(values)
                else:
                    matrix[i, j] = np.nan
            else:
                matrix[i, j] = np.nan

    # Create figure
    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(matrix,
                xticklabels=models,
                yticklabels=encodings,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                vmin=0.7,
                vmax=1.0,
                cbar_kws={'label': 'R² Score'})

    plt.title(f'Average Model Performance Across All Optimizations\n' +
              f'R² Scores for {target_type.upper()} Prediction',
              fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Encoding Method', fontsize=12)

    # Rotate x labels
    plt.xticks(rotation=45, ha='right')

    # Add note about averaging
    plt.figtext(0.5, 0.01,
                f'Note: Values shown are averages across {int(count_matrix[~np.isnan(count_matrix)].max())} optimization methods',
                ha='center', fontsize=10, style='italic')

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, f'r2_heatmap_{target_type}_average.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
