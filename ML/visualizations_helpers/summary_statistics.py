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
    """Create violin plots comparing error distributions organized by optimization and error type"""

    # Calculate mean and max flux errors from individual positions
    mean_flux_errors = []
    max_flux_errors = []

    for _, row in df.iterrows():
        # Collect individual position errors
        position_errors = []
        for i in range(1, 5):
            if f'I_{i}_rel_error' in row:
                error = row[f'I_{i}_rel_error']
                if pd.notna(error):
                    position_errors.append(error)

        if position_errors:
            # Mean of individual errors
            mean_flux_errors.append(np.mean(position_errors))
            # Maximum error among positions
            max_flux_errors.append(max(position_errors))
        else:
            mean_flux_errors.append(np.nan)
            max_flux_errors.append(np.nan)

    # Add these to dataframe
    df = df.copy()  # Don't modify original
    df['mean_flux_error'] = mean_flux_errors
    df['max_flux_error'] = max_flux_errors

    # Get unique values
    optimizations = sorted(df['optimization_method'].unique())
    models = sorted(df['model_class'].unique())
    n_opts = len(optimizations)
    n_models = len(models)

    # Create figure with rows for each optimization×error_type, columns for each model + top6
    n_rows = n_opts * 2  # 2 error types (mean, max)
    n_cols = n_models + 1  # models + top 6 combinations

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 4*n_rows))

    # Ensure axes is 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle('Error Distribution Comparison - All Model/Encoding Combinations',
                 fontsize=16, fontweight='bold')

    # Process each optimization method
    for opt_idx, optimization in enumerate(optimizations):
        # Filter data for this optimization
        opt_df = df[df['optimization_method'] == optimization]

        # Row for mean errors
        mean_row_idx = opt_idx * 2

        # Plot each model type for mean errors
        for model_idx, model in enumerate(models):
            ax = axes[mean_row_idx, model_idx]

            # Filter for this model
            model_df = opt_df[opt_df['model_class'] == model]

            if not model_df.empty:
                sns.violinplot(data=model_df, x='encoding', y='mean_flux_error', ax=ax)
                ax.set_xlabel('Encoding Method')
                ax.set_ylabel('Mean Flux Error (%)')
                ax.set_title(f'{model}\n{optimization} - Mean Error', fontsize=10)
                ax.tick_params(axis='x', rotation=45)

                # Add grid for readability
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

        # Top 6 combinations for mean error
        ax = axes[mean_row_idx, -1]
        opt_df['model_encoding'] = opt_df['model_class'].str[:3] + '-' + opt_df['encoding'].str[:3]
        mean_errors_by_combo = opt_df.groupby('model_encoding')['mean_flux_error'].mean().sort_values()
        top_combinations = mean_errors_by_combo.head(6).index
        filtered_df = opt_df[opt_df['model_encoding'].isin(top_combinations)]

        if not filtered_df.empty:
            sns.boxplot(data=filtered_df, x='model_encoding', y='mean_flux_error', ax=ax)
            ax.set_xlabel('Model-Encoding')
            ax.set_ylabel('Mean Flux Error (%)')
            ax.set_title(f'Top 6 Combinations\n{optimization} - Mean Error', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

        # Row for max errors
        max_row_idx = opt_idx * 2 + 1

        # Plot each model type for max errors
        for model_idx, model in enumerate(models):
            ax = axes[max_row_idx, model_idx]

            # Filter for this model
            model_df = opt_df[opt_df['model_class'] == model]

            if not model_df.empty:
                sns.violinplot(data=model_df, x='encoding', y='max_flux_error', ax=ax)
                ax.set_xlabel('Encoding Method')
                ax.set_ylabel('Max Flux Error (%)')
                ax.set_title(f'{model}\n{optimization} - Max Error', fontsize=10)
                ax.tick_params(axis='x', rotation=45)

                # Add grid for readability
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

        # Top 6 combinations for max error
        ax = axes[max_row_idx, -1]
        max_errors_by_combo = opt_df.groupby('model_encoding')['max_flux_error'].mean().sort_values()
        top_combinations_max = max_errors_by_combo.head(6).index
        filtered_df_max = opt_df[opt_df['model_encoding'].isin(top_combinations_max)]

        if not filtered_df_max.empty:
            sns.boxplot(data=filtered_df_max, x='model_encoding', y='max_flux_error', ax=ax)
            ax.set_xlabel('Model-Encoding')
            ax.set_ylabel('Max Flux Error (%)')
            ax.set_title(f'Top 6 Combinations\n{optimization} - Max Error', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

    # Add row labels on the left
    for opt_idx, optimization in enumerate(optimizations):
        # Mean error row
        mean_row_idx = opt_idx * 2
        axes[mean_row_idx, 0].annotate(f'{optimization}\nMean Error',
                                       xy=(-0.3, 0.5), xycoords='axes fraction',
                                       fontsize=12, fontweight='bold',
                                       ha='right', va='center', rotation=90)

        # Max error row
        max_row_idx = opt_idx * 2 + 1
        axes[max_row_idx, 0].annotate(f'{optimization}\nMax Error',
                                      xy=(-0.3, 0.5), xycoords='axes fraction',
                                      fontsize=12, fontweight='bold',
                                      ha='right', va='center', rotation=90)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, 'error_distribution_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
