"""
Summary statistics visualizations showing best performing models
FIXED: Removed flux predictions from k-eff models, improved alignment, proper optimization detection
UPDATED: Added support for energy-discretized results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def create_summary_statistics_plots(df, output_dir, has_energy_discretization=False):
    """
    Create summary visualizations for model performance

    Args:
        df: DataFrame with test results
        output_dir: Directory to save plots
        has_energy_discretization: Whether the data has energy groups
    """

    if has_energy_discretization:
        # Create energy-specific summary plots
        create_energy_summary_comparison(df, output_dir)
        create_best_models_by_energy_group(df, output_dir)
        # Also create standard best models summary with energy data
        create_best_models_summary(df, output_dir, has_energy_discretization=True)
    else:
        # Create standard summary plots
        create_best_models_summary(df, output_dir)
        create_optimization_comparison(df, output_dir)

def create_energy_summary_comparison(df, output_dir):
    """Create comparison of model performance across energy groups"""

    # Set style to white background
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'

    # Define energy groups
    energy_groups = ['thermal', 'epithermal', 'fast', 'total']

    # Calculate mean errors for each model/encoding/optimization combination
    summary_data = []

    for energy in energy_groups:
        # Group by model configuration
        grouped = df.groupby(['model_class', 'encoding', 'optimization_method'])

        for name, group in grouped:
            model_class, encoding, optimization = name

            # Calculate mean error for this energy group
            errors = []
            max_errors = []
            for _, row in group.iterrows():
                position_errors = []
                for i in range(1, 5):
                    error_col = f'I_{i}_{energy}_rel_error'
                    if error_col in row:
                        error = row[error_col]
                        if pd.notna(error) and error != 'N/A':
                            position_errors.append(abs(error))

                if position_errors:
                    errors.extend(position_errors)
                    max_errors.append(max(position_errors))

            if errors:
                summary_data.append({
                    'model': model_class,
                    'encoding': encoding,
                    'optimization': optimization,
                    'energy_group': energy,
                    'mean_error': np.mean(errors),
                    'mean_max_error': np.mean(max_errors) if max_errors else np.nan,
                    'std_error': np.std(errors),
                    'max_error': max(errors)
                })

    summary_df = pd.DataFrame(summary_data)

    if summary_df.empty:
        return

    # Create figure with subplots - 3 horizontal bar charts like in the images
    fig = plt.figure(figsize=(20, 8))

    # Define colors for models
    model_colors = {
        'xgboost': '#1f77b4',
        'random_forest': '#ff7f0e',
        'svm': '#2ca02c',
        'neural_net': '#d62728'
    }

    # Plot 1: Top 10 Models by Mean Flux Error
    ax1 = plt.subplot(1, 3, 1)

    # Get top 10 models by mean flux error (average across all energy groups)
    model_mean_errors = []
    grouped = summary_df.groupby(['model', 'encoding', 'optimization'])
    for name, group in grouped:
        model, encoding, optimization = name
        mean_error = group['mean_error'].mean()
        model_mean_errors.append({
            'label': f'{model}-{encoding}-{optimization}',
            'model': model,
            'error': mean_error
        })

    top_models = sorted(model_mean_errors, key=lambda x: x['error'])[:10]

    y_pos = np.arange(len(top_models))
    errors = [m['error'] for m in top_models]
    colors = [model_colors.get(m['model'], 'gray') for m in top_models]
    labels = [m['label'] for m in top_models]

    bars = ax1.barh(y_pos, errors, color=colors, alpha=0.8)

    # Add value labels
    for i, (bar, error) in enumerate(zip(bars, errors)):
        ax1.text(error + 0.02, i, f'{error:.3f}%', va='center', fontsize=9)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel('Mean Flux Error (%)', fontsize=11)
    ax1.set_title('Top 10 Models by Mean Flux Error', fontsize=12, fontweight='bold')
    ax1.grid(True, axis='x', alpha=0.3)
    ax1.invert_yaxis()

    # Plot 2: Top 10 Models by Mean Maximum Flux Error
    ax2 = plt.subplot(1, 3, 2)

    # Get top 10 models by mean max error
    model_max_errors = []
    for name, group in grouped:
        model, encoding, optimization = name
        mean_max_error = group['mean_max_error'].mean()
        if not np.isnan(mean_max_error):
            model_max_errors.append({
                'label': f'{model}-{encoding}-{optimization}',
                'model': model,
                'error': mean_max_error
            })

    top_max_models = sorted(model_max_errors, key=lambda x: x['error'])[:10]

    y_pos = np.arange(len(top_max_models))
    errors = [m['error'] for m in top_max_models]
    colors = [model_colors.get(m['model'], 'gray') for m in top_max_models]
    labels = [m['label'] for m in top_max_models]

    bars = ax2.barh(y_pos, errors, color=colors, alpha=0.8)

    # Add value labels
    for i, (bar, error) in enumerate(zip(bars, errors)):
        ax2.text(error + 0.02, i, f'{error:.3f}%', va='center', fontsize=9)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel('Mean of Maximum Flux Errors (%)', fontsize=11)
    ax2.set_title('Top 10 Models by Mean Maximum Flux Error', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3)
    ax2.invert_yaxis()

    # Plot 3: Top 10 Models by Mean K-eff Error
    ax3 = plt.subplot(1, 3, 3)

    # Calculate k-eff errors
    keff_errors = []
    grouped_orig = df.groupby(['model_class', 'encoding', 'optimization_method'])

    for name, group in grouped_orig:
        model, encoding, optimization = name

        keff_rel_errors = []
        for _, row in group.iterrows():
            if 'keff_real' in row and 'keff_predicted' in row:
                real = row['keff_real']
                pred = row['keff_predicted']
                if pd.notna(real) and pd.notna(pred) and real != 0:
                    rel_error = abs((pred - real) / real) * 100
                    keff_rel_errors.append(rel_error)

        if keff_rel_errors:
            keff_errors.append({
                'label': f'{model}-{encoding}-{optimization}',
                'model': model,
                'error': np.mean(keff_rel_errors)
            })

    if keff_errors:
        top_keff_models = sorted(keff_errors, key=lambda x: x['error'])[:10]

        y_pos = np.arange(len(top_keff_models))
        errors = [m['error'] for m in top_keff_models]
        colors = [model_colors.get(m['model'], 'gray') for m in top_keff_models]
        labels = [m['label'] for m in top_keff_models]

        bars = ax3.barh(y_pos, errors, color=colors, alpha=0.8)

        # Add value labels
        for i, (bar, error) in enumerate(zip(bars, errors)):
            ax3.text(error + 0.0001, i, f'{error:.3f}%', va='center', fontsize=9)

        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels, fontsize=9)
        ax3.set_xlabel('Mean K-eff Error (%)', fontsize=11)
        ax3.set_title('Top 10 Models by Mean K-eff Error', fontsize=12, fontweight='bold')
        ax3.grid(True, axis='x', alpha=0.3)
        ax3.invert_yaxis()
    else:
        ax3.text(0.5, 0.5, 'No K-eff Data Available', ha='center', va='center',
                transform=ax3.transAxes, fontsize=14, alpha=0.5)
        ax3.set_xticks([])
        ax3.set_yticks([])

    plt.suptitle('Model Performance Summary - Best Combinations', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, 'model_performance_summary.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

    # Create error distribution comparison plot
    create_error_distribution_comparison(df, output_dir)

def create_best_models_by_energy_group(df, output_dir):
    """Create detailed best models table for each energy group"""

    energy_groups = ['thermal', 'epithermal', 'fast', 'total']

    # Prepare data for table
    table_data = []

    for energy in energy_groups:
        # Calculate metrics for each model configuration
        grouped = df.groupby(['model_class', 'encoding', 'optimization_method'])

        energy_configs = []
        for name, group in grouped:
            model_class, encoding, optimization = name

            # Calculate metrics for this energy group
            errors = []
            for _, row in group.iterrows():
                for i in range(1, 5):
                    error_col = f'I_{i}_{energy}_rel_error'
                    if error_col in row:
                        error = row[error_col]
                        if pd.notna(error) and error != 'N/A':
                            errors.append(abs(error))

            if errors:
                energy_configs.append({
                    'model': model_class,
                    'encoding': encoding,
                    'optimization': optimization,
                    'mean_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'max_error': max(errors),
                    'min_error': min(errors)
                })

        # Sort by mean error and take top 3
        if energy_configs:
            energy_df = pd.DataFrame(energy_configs)
            energy_df = energy_df.sort_values('mean_error').head(3)

            for rank, (_, row) in enumerate(energy_df.iterrows(), 1):
                table_data.append({
                    'Energy Group': energy.capitalize(),
                    'Rank': rank,
                    'Model': row['model'].upper(),
                    'Encoding': row['encoding'],
                    'Optimization': row['optimization'],
                    'Mean Error': f"{row['mean_error']:.3f}%",
                    'Std Dev': f"{row['std_error']:.3f}%",
                    'Max Error': f"{row['max_error']:.3f}%"
                })

    if not table_data:
        return

    # Create figure for table
    fig, ax = plt.subplots(figsize=(14, len(table_data) * 0.4 + 2))
    ax.axis('off')

    # Convert to list format for table
    col_labels = ['Energy Group', 'Rank', 'Model', 'Encoding', 'Optimization',
                  'Mean Error', 'Std Dev', 'Max Error']

    table_rows = []
    for row in table_data:
        table_rows.append([row[col] for col in col_labels])

    # Create table
    table = ax.table(cellText=table_rows, colLabels=col_labels,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Header styling
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # Row coloring by energy group
    energy_colors = {
        'Thermal': '#FFE6E6',
        'Epithermal': '#E6F3FF',
        'Fast': '#E6FFE6',
        'Total': '#FFF0E6'
    }

    for i in range(1, len(table_rows) + 1):
        energy_group = table_rows[i-1][0]
        color = energy_colors.get(energy_group, '#FFFFFF')

        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(color)

        # Bold the rank 1 entries
        if table_rows[i-1][1] == 1:
            for j in range(len(col_labels)):
                table[(i, j)].set_text_props(weight='bold')

    plt.title('Top 3 Model Configurations by Energy Group',
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save table
    output_file = os.path.join(output_dir, 'best_models_by_energy_table.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

# Keep the original functions for non-energy-discretized data
def create_best_models_summary(df, output_dir, has_energy_discretization=False):
    """Create summary showing best performing model combinations"""

    # Calculate mean errors for each model/encoding/optimization combination
    summary_data = []

    # Group by model configuration
    grouped = df.groupby(['model_class', 'encoding', 'optimization_method'])

    for name, group in grouped:
        model_class, encoding, optimization = name

        # Calculate metrics based on model type
        if has_energy_discretization:
            # For energy-discretized data, calculate mean across all energy groups
            flux_errors = []
            for energy_group in ['thermal', 'epithermal', 'fast', 'total']:
                for _, row in group.iterrows():
                    for i in range(1, 5):
                        error_col = f'I_{i}_{energy_group}_rel_error'
                        if error_col in row:
                            error = row[error_col]
                            if pd.notna(error) and error != 'N/A':
                                flux_errors.append(abs(error))

            if flux_errors:
                flux_mape = np.mean(flux_errors)
                max_flux_error = max(flux_errors)
                model_type = 'flux'
            else:
                flux_mape = None
                max_flux_error = None
                model_type = 'keff'
        else:
            # Original logic for non-energy data
            if 'mape_flux' in group.columns and group['mape_flux'].notna().any():
                # This is a flux model
                flux_mape = group['mape_flux'].mean()
                max_flux_error = group['mape_flux'].max()

                # Skip k-eff predictions for flux models
                keff_mape = None
                model_type = 'flux'

            else:
                # This might be a k-eff only model or have no flux data
                flux_mape = None
                max_flux_error = None
                model_type = 'keff'

        # K-eff metrics (for both flux and k-eff models)
        if 'keff_real' in group.columns and 'keff_predicted' in group.columns:
            # Calculate k-eff MAPE
            keff_errors = []
            for _, row in group.iterrows():
                if pd.notna(row['keff_real']) and pd.notna(row['keff_predicted']) and row['keff_real'] != 0:
                    error = abs((row['keff_predicted'] - row['keff_real']) / row['keff_real']) * 100
                    keff_errors.append(error)

            if keff_errors:
                keff_mape = np.mean(keff_errors)
            else:
                keff_mape = None
        else:
            keff_mape = None

        # Determine final model type
        if flux_mape is not None and keff_mape is not None:
            model_type = 'both'
        elif flux_mape is not None:
            model_type = 'flux'
        elif keff_mape is not None:
            model_type = 'keff'
        else:
            continue  # Skip if no valid metrics

        summary_data.append({
            'model': model_class,
            'encoding': encoding,
            'optimization': optimization,
            'flux_mape': flux_mape,
            'keff_mape': keff_mape,
            'max_flux_error': max_flux_error,
            'model_type': model_type,
            'n_configs': len(group)
        })

    summary_df = pd.DataFrame(summary_data)

    if summary_df.empty:
        print("  Warning: No summary data available")
        return

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))

    # Main title
    title = 'Model Performance Summary - Best Configurations'
    if has_energy_discretization:
        title += '\n(Averaged Across All Energy Groups)'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Define colors
    model_colors = {
        'xgboost': '#1f77b4',
        'random_forest': '#ff7f0e',
        'svm': '#2ca02c',
        'neural_net': '#d62728'
    }

    # 1. Top flux models (top-left) - FIXED to exclude k-eff only models
    ax1 = plt.subplot(2, 2, 1)
    flux_models = summary_df[summary_df['model_type'].isin(['flux', 'both'])].copy()

    if not flux_models.empty:
        flux_models = flux_models.sort_values('flux_mape').head(10)

        y_pos = np.arange(len(flux_models))
        bars = ax1.barh(y_pos, flux_models['flux_mape'])

        # Color bars by model type
        for i, (idx, row) in enumerate(flux_models.iterrows()):
            bars[i].set_color(model_colors.get(row['model'], 'gray'))

        # Labels
        labels = [f"{row['model']}-{row['encoding']}-{row['optimization']}"
                 for _, row in flux_models.iterrows()]
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=9)
        ax1.set_xlabel('Mean MAPE (%)', fontsize=10)
        ax1.set_title('Top 10 Flux Prediction Models', fontsize=12, fontweight='bold')
        ax1.grid(True, axis='x', alpha=0.3)
        ax1.invert_yaxis()

        # Add value labels
        for i, (idx, row) in enumerate(flux_models.iterrows()):
            ax1.text(row['flux_mape'] + 0.05, i, f"{row['flux_mape']:.2f}%",
                    va='center', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No Flux Models Found', ha='center', va='center',
                transform=ax1.transAxes, fontsize=12, alpha=0.5)

    # 2. Top k-eff models (top-right)
    ax2 = plt.subplot(2, 2, 2)
    keff_models = summary_df[summary_df['keff_mape'].notna()].copy()

    if not keff_models.empty:
        keff_models = keff_models.sort_values('keff_mape').head(10)

        y_pos = np.arange(len(keff_models))
        bars = ax2.barh(y_pos, keff_models['keff_mape'])

        # Color bars by model type
        for i, (idx, row) in enumerate(keff_models.iterrows()):
            bars[i].set_color(model_colors.get(row['model'], 'gray'))

        # Labels
        labels = [f"{row['model']}-{row['encoding']}-{row['optimization']}"
                 for _, row in keff_models.iterrows()]
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=9)
        ax2.set_xlabel('Mean MAPE (%)', fontsize=10)
        ax2.set_title('Top 10 K-eff Prediction Models', fontsize=12, fontweight='bold')
        ax2.grid(True, axis='x', alpha=0.3)
        ax2.invert_yaxis()

        # Add value labels
        for i, (idx, row) in enumerate(keff_models.iterrows()):
            ax2.text(row['keff_mape'] + 0.0005, i, f"{row['keff_mape']:.3f}%",
                    va='center', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No K-eff Models Found', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12, alpha=0.5)

    # 3. Encoding comparison (bottom-left)
    ax3 = plt.subplot(2, 2, 3)
    encoding_stats = []

    for encoding in summary_df['encoding'].unique():
        enc_data = summary_df[summary_df['encoding'] == encoding]

        # Only include flux models for flux statistics
        flux_enc_data = enc_data[enc_data['model_type'].isin(['flux', 'both'])]

        if not flux_enc_data.empty:
            encoding_stats.append({
                'encoding': encoding,
                'mean_flux_mape': flux_enc_data['flux_mape'].mean() if not flux_enc_data['flux_mape'].isna().all() else None,
                'mean_keff_mape': enc_data['keff_mape'].mean() if 'keff_mape' in enc_data and not enc_data['keff_mape'].isna().all() else None,
                'count': len(enc_data)
            })

    if encoding_stats:
        enc_df = pd.DataFrame(encoding_stats)
        enc_df = enc_df.sort_values('mean_flux_mape', na_position='last')

        x = np.arange(len(enc_df))
        width = 0.35

        # Plot bars - handle NaN values
        flux_values = enc_df['mean_flux_mape'].fillna(0).values
        keff_values = enc_df['mean_keff_mape'].fillna(0).values

        bars1 = ax3.bar(x - width/2, flux_values, width, label='Flux MAPE', color='skyblue')
        bars2 = ax3.bar(x + width/2, keff_values, width, label='K-eff MAPE', color='lightcoral')

        ax3.set_xlabel('Encoding Method', fontsize=10)
        ax3.set_ylabel('Mean MAPE (%)', fontsize=10)
        ax3.set_title('Average Performance by Encoding', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(enc_df['encoding'], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, axis='y', alpha=0.3)

    # 4. Optimization comparison (bottom-right)
    ax4 = plt.subplot(2, 2, 4)
    opt_stats = []

    for optimization in summary_df['optimization_method'].unique():
        opt_data = summary_df[summary_df['optimization_method'] == optimization]

        # Only include flux models for flux statistics
        flux_opt_data = opt_data[opt_data['model_type'].isin(['flux', 'both'])]

        if not flux_opt_data.empty:
            opt_stats.append({
                'optimization': optimization,
                'mean_flux_mape': flux_opt_data['flux_mape'].mean() if not flux_opt_data['flux_mape'].isna().all() else None,
                'mean_keff_mape': opt_data['keff_mape'].mean() if 'keff_mape' in opt_data and not opt_data['keff_mape'].isna().all() else None,
                'count': len(opt_data)
            })

    if opt_stats:
        opt_df = pd.DataFrame(opt_stats)
        opt_df = opt_df.sort_values('mean_flux_mape', na_position='last')

        x = np.arange(len(opt_df))
        width = 0.35

        # Plot bars - handle NaN values
        flux_values = opt_df['mean_flux_mape'].fillna(0).values
        keff_values = opt_df['mean_keff_mape'].fillna(0).values

        bars1 = ax4.bar(x - width/2, flux_values, width, label='Flux MAPE', color='skyblue')
        bars2 = ax4.bar(x + width/2, keff_values, width, label='K-eff MAPE', color='lightcoral')

        ax4.set_xlabel('Optimization Method', fontsize=10)
        ax4.set_ylabel('Mean MAPE (%)', fontsize=10)
        ax4.set_title('Average Performance by Optimization', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(opt_df['optimization'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, 'best_models_summary.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

    # Create detailed statistics table
    create_detailed_stats_table(summary_df, output_dir)

def create_optimization_comparison(df, output_dir):
    """Create comparison of optimization methods"""

    # Get optimization methods
    optimizations = df['optimization_method'].unique()

    # Skip if only one optimization method
    if len(optimizations) <= 1:
        print("  Skipping optimization comparison (only one method found)")
        return

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Flux comparison
    flux_data = []
    for opt in optimizations:
        opt_df = df[df['optimization_method'] == opt]
        if 'mape_flux' in opt_df.columns:
            errors = opt_df['mape_flux'].dropna()
            if len(errors) > 0:
                flux_data.append(errors.values)
            else:
                flux_data.append([])
        else:
            flux_data.append([])

    # Create box plot for flux
    if any(len(d) > 0 for d in flux_data):
        bp1 = ax1.boxplot(flux_data, labels=optimizations, patch_artist=True,
                         showmeans=True, meanline=True)

        # Color boxes
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for patch, color in zip(bp1['boxes'], colors[:len(bp1['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax1.set_xlabel('Optimization Method', fontsize=12)
        ax1.set_ylabel('Flux MAPE (%)', fontsize=12)
        ax1.set_title('Flux Prediction Error by Optimization Method', fontsize=14, fontweight='bold')
        ax1.grid(True, axis='y', alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No Flux Data Available', ha='center', va='center',
                transform=ax1.transAxes, fontsize=12, alpha=0.5)

    # K-eff comparison
    keff_data = []
    for opt in optimizations:
        opt_df = df[df['optimization_method'] == opt]
        if 'keff_real' in opt_df.columns and 'keff_predicted' in opt_df.columns:
            # Calculate k-eff errors
            errors = []
            for _, row in opt_df.iterrows():
                if pd.notna(row['keff_real']) and pd.notna(row['keff_predicted']) and row['keff_real'] != 0:
                    error = abs((row['keff_predicted'] - row['keff_real']) / row['keff_real']) * 100
                    errors.append(error)
            keff_data.append(errors)
        else:
            keff_data.append([])

    # Create box plot for k-eff
    if any(len(d) > 0 for d in keff_data):
        bp2 = ax2.boxplot(keff_data, labels=optimizations, patch_artist=True,
                         showmeans=True, meanline=True)

        # Color boxes
        for patch, color in zip(bp2['boxes'], colors[:len(bp2['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_xlabel('Optimization Method', fontsize=12)
        ax2.set_ylabel('K-eff MAPE (%)', fontsize=12)
        ax2.set_title('K-eff Prediction Error by Optimization Method', fontsize=14, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No K-eff Data Available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12, alpha=0.5)

    plt.suptitle('Optimization Method Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, 'optimization_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

def create_detailed_stats_table(summary_df, output_dir):
    """Create a detailed statistics table as an image"""

    # Sort by flux MAPE
    flux_models = summary_df[summary_df['model_type'].isin(['flux', 'both'])].copy()

    if flux_models.empty:
        return

    flux_models = flux_models.sort_values('flux_mape').head(20)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, len(flux_models) * 0.4 + 2))
    ax.axis('off')

    # Prepare table data
    table_data = []
    for _, row in flux_models.iterrows():
        flux_str = f"{row['flux_mape']:.3f}%" if pd.notna(row['flux_mape']) else "N/A"
        keff_str = f"{row['keff_mape']:.3f}%" if pd.notna(row['keff_mape']) else "N/A"
        max_flux_str = f"{row['max_flux_error']:.3f}%" if pd.notna(row['max_flux_error']) else "N/A"

        table_data.append([
            row['model'].upper(),
            row['encoding'],
            row['optimization'],
            flux_str,
            keff_str,
            max_flux_str,
            str(row['n_configs'])
        ])

    # Create table
    col_labels = ['Model', 'Encoding', 'Optimization', 'Flux MAPE', 'K-eff MAPE', 'Max Flux Error', 'N Configs']

    table = ax.table(cellText=table_data,
                    colLabels=col_labels,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Header styling
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # Row coloring
    model_colors = {
        'XGBOOST': '#E6F3FF',
        'RANDOM_FOREST': '#FFF0E6',
        'SVM': '#E6FFE6',
        'NEURAL_NET': '#FFE6E6'
    }

    for i in range(1, len(table_data) + 1):
        model_name = table_data[i-1][0]
        color = model_colors.get(model_name, '#FFFFFF')

        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(color)

    plt.title('Top 20 Model Configurations - Detailed Statistics',
             fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save table
    output_file = os.path.join(output_dir, 'detailed_stats_table.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

def create_error_distribution_comparison(df, output_dir):
    """Create violin plots comparing error distributions organized by optimization and error type"""

    # Set style to white background
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'

    # Check if this is energy-discretized data
    has_energy = any(col.endswith('_thermal_real') for col in df.columns)

    if has_energy:
        # Create plots for each energy group
        energy_groups = ['thermal', 'epithermal', 'fast', 'total']
        for energy in energy_groups:
            create_error_distribution_for_energy(df, output_dir, energy)
    else:
        # Create plot for regular total flux
        create_error_distribution_for_total(df, output_dir)

def create_error_distribution_for_energy(df, output_dir, energy_group):
    """Create error distribution plots for a specific energy group"""

    # Calculate mean and max flux errors from individual positions
    mean_flux_errors = []
    max_flux_errors = []

    for _, row in df.iterrows():
        # Collect individual position errors for this energy group
        position_errors = []
        for i in range(1, 5):
            error_col = f'I_{i}_{energy_group}_rel_error'
            if error_col in row:
                error = row[error_col]
                if pd.notna(error) and error != 'N/A':
                    position_errors.append(abs(error))

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

    fig.suptitle(f'Error Distribution Comparison - {energy_group.capitalize()} Flux - All Model/Encoding Combinations',
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

    # Save figure - in the specific energy folder if provided
    output_file = os.path.join(output_dir, f'error_distribution_comparison_{energy_group}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

def create_error_distribution_for_total(df, output_dir):
    """Create error distribution plots for regular total flux models"""

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
