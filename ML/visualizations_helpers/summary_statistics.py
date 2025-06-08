"""
Summary statistics visualizations to identify best model combinations
FIXED: Removed overall performance, pareto frontier, and ranking heatmap
       Added support for multiple optimizations in error distribution
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def create_summary_statistics_plots(df, output_dir):
    """Create comprehensive summary statistics visualizations"""

    print("  Creating summary statistics visualizations...")

    # Create only the requested summary views
    try:
        create_best_model_summary(df, output_dir)
        print("    ✓ Best model summary created")
    except Exception as e:
        print(f"    ✗ Error in best model summary: {e}")

    try:
        create_error_distribution_comparison(df, output_dir)
        print("    ✓ Error distribution comparison created")
    except Exception as e:
        print(f"    ✗ Error in error distribution: {e}")

# summary_statistics.py - UPDATED create_best_model_summary function

def create_best_model_summary(df, output_dir):
    """Create summary showing best model for each metric"""

    # Calculate aggregated metrics
    summary_data = []

    # Group by model configuration
    for model in df['model_class'].unique():
        for encoding in df['encoding'].unique():
            for optimization in df['optimization_method'].unique():
                subset = df[
                    (df['model_class'] == model) &
                    (df['encoding'] == encoding) &
                    (df['optimization_method'] == optimization)
                ]

                if not subset.empty:
                    result = {
                        'model': model,
                        'encoding': encoding,
                        'optimization': optimization,
                        'combination': f'{model}-{encoding}-{optimization}'
                    }

                    # Calculate mean flux error from individual position errors
                    # This matches how your rel_error graphs calculate it
                    all_rel_errors = []
                    for _, row in subset.iterrows():
                        for i in range(1, 5):
                            if f'I_{i}_rel_error' in row:
                                error = row[f'I_{i}_rel_error']
                                if pd.notna(error):
                                    all_rel_errors.append(error)

                    if all_rel_errors:
                        result['mean_flux_error'] = np.mean(all_rel_errors)

                    # Mean of maximum flux errors
                    max_errors = []
                    for _, row in subset.iterrows():
                        errors = []
                        for i in range(1, 5):
                            if f'I_{i}_rel_error' in row:
                                error = row[f'I_{i}_rel_error']
                                if pd.notna(error):
                                    errors.append(error)
                        if errors:
                            max_errors.append(max(errors))

                    if max_errors:
                        result['mean_max_flux_error'] = np.mean(max_errors)

                    # K-eff metrics
                    if 'keff_rel_error' in subset.columns:
                        result['mean_keff_error'] = subset['keff_rel_error'].mean()

                    summary_data.append(result)

    summary_df = pd.DataFrame(summary_data)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('Model Performance Summary - Best Combinations', fontsize=16, fontweight='bold')

    # Plot 1: Best by mean flux error
    ax = axes[0]
    if 'mean_flux_error' in summary_df.columns:
        top_10 = summary_df.nsmallest(10, 'mean_flux_error')
        y_pos = np.arange(len(top_10))
        ax.barh(y_pos, top_10['mean_flux_error'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_10['combination'], fontsize=8)
        ax.set_xlabel('Mean Flux Error (%)')
        ax.set_title('Top 10 Models by Mean Flux Error')
        ax.invert_yaxis()

        # Add value labels
        for i, v in enumerate(top_10['mean_flux_error']):
            ax.text(v + 0.05, i, f'{v:.3f}%', va='center', fontsize=8)

    # Plot 2: Best by mean of max flux errors
    ax = axes[1]
    if 'mean_max_flux_error' in summary_df.columns:
        top_10 = summary_df.nsmallest(10, 'mean_max_flux_error')
        y_pos = np.arange(len(top_10))
        ax.barh(y_pos, top_10['mean_max_flux_error'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_10['combination'], fontsize=8)
        ax.set_xlabel('Mean of Maximum Flux Errors (%)')
        ax.set_title('Top 10 Models by Mean Maximum Flux Error')
        ax.invert_yaxis()

        for i, v in enumerate(top_10['mean_max_flux_error']):
            ax.text(v + 0.05, i, f'{v:.3f}%', va='center', fontsize=8)

    # Plot 3: Best by mean k-eff error
    ax = axes[2]
    if 'mean_keff_error' in summary_df.columns:
        top_10 = summary_df.nsmallest(10, 'mean_keff_error')
        y_pos = np.arange(len(top_10))
        ax.barh(y_pos, top_10['mean_keff_error'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_10['combination'], fontsize=8)
        ax.set_xlabel('Mean K-eff Error (%)')
        ax.set_title('Top 10 Models by Mean K-eff Error')
        ax.invert_yaxis()

        for i, v in enumerate(top_10['mean_keff_error']):
            ax.text(v + 0.001, i, f'{v:.3f}%', va='center', fontsize=8)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, 'best_model_summary.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

def create_error_distribution_comparison(df, output_dir):
    """Create violin plots comparing error distributions with support for multiple optimizations"""

    # Get unique optimization methods
    optimizations = df['optimization_method'].unique()
    n_opts = len(optimizations)

    # Adjust figure size based on number of optimizations
    fig_height = 6
    if n_opts > 1:
        fig_height = 4 * n_opts  # More height for multiple rows

    # Create subplots - one row per optimization method
    fig, axes = plt.subplots(n_opts, 3, figsize=(18, fig_height))

    # Handle single optimization case
    if n_opts == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Error Distribution Comparison by Model Components', fontsize=16, fontweight='bold')

    for opt_idx, optimization in enumerate(optimizations):
        # Filter data for this optimization
        opt_df = df[df['optimization_method'] == optimization]

        # Plot 1: By model type
        ax = axes[opt_idx, 0]
        if 'avg_flux_rel_error' in opt_df.columns:
            sns.violinplot(data=opt_df, x='model_class', y='avg_flux_rel_error', ax=ax)
            ax.set_xlabel('Model Type')
            ax.set_ylabel('Flux Relative Error (%)')
            ax.set_title(f'Error by Model ({optimization})')
            ax.tick_params(axis='x', rotation=45)

        # Plot 2: By encoding
        ax = axes[opt_idx, 1]
        if 'avg_flux_rel_error' in opt_df.columns:
            sns.violinplot(data=opt_df, x='encoding', y='avg_flux_rel_error', ax=ax)
            ax.set_xlabel('Encoding Method')
            ax.set_ylabel('Flux Relative Error (%)')
            ax.set_title(f'Error by Encoding ({optimization})')
            ax.tick_params(axis='x', rotation=45)

        # Plot 3: Combined comparison
        ax = axes[opt_idx, 2]
        if 'avg_flux_rel_error' in opt_df.columns:
            # Create a combined label for model+encoding
            opt_df['model_encoding'] = opt_df['model_class'].str[:3] + '-' + opt_df['encoding'].str[:3]

            # Get top 6 combinations by performance
            mean_errors = opt_df.groupby('model_encoding')['avg_flux_rel_error'].mean().sort_values()
            top_combinations = mean_errors.head(6).index

            # Filter to show only top combinations
            filtered_df = opt_df[opt_df['model_encoding'].isin(top_combinations)]

            sns.boxplot(data=filtered_df, x='model_encoding', y='avg_flux_rel_error', ax=ax)
            ax.set_xlabel('Model-Encoding Combination')
            ax.set_ylabel('Flux Relative Error (%)')
            ax.set_title(f'Top 6 Combinations ({optimization})')
            ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, 'error_distribution_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
