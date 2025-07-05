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
        # Removed optimization comparison as requested

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

    # Create figure with subplots - 3 horizontal bar charts for Mean MAPE, Mean of Max MAPE, and Max MAPE
    fig = plt.figure(figsize=(20, 8))

    # Define colors for models
    model_colors = {
        'xgboost': '#1f77b4',
        'random_forest': '#ff7f0e',
        'svm': '#2ca02c',
        'neural_net': '#d62728'
    }

    # Plot 1: Top 10 Models by Mean MAPE
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
    ax1.set_xlabel('Mean MAPE (%)', fontsize=11)
    ax1.set_title('Top 10 Models by Mean MAPE', fontsize=12, fontweight='bold')
    ax1.grid(True, axis='x', alpha=0.3)
    ax1.invert_yaxis()

    # Plot 2: Top 10 Models by Mean of Max MAPE
    ax2 = plt.subplot(1, 3, 2)

    # Get top 10 models by mean of max error
    model_mean_of_max_errors = []
    for name, group in grouped:
        model, encoding, optimization = name
        mean_of_max_error = group['mean_max_error'].mean()
        if not np.isnan(mean_of_max_error):
            model_mean_of_max_errors.append({
                'label': f'{model}-{encoding}-{optimization}',
                'model': model,
                'error': mean_of_max_error
            })

    top_mean_of_max_models = sorted(model_mean_of_max_errors, key=lambda x: x['error'])[:10]

    y_pos = np.arange(len(top_mean_of_max_models))
    errors = [m['error'] for m in top_mean_of_max_models]
    colors = [model_colors.get(m['model'], 'gray') for m in top_mean_of_max_models]
    labels = [m['label'] for m in top_mean_of_max_models]

    bars = ax2.barh(y_pos, errors, color=colors, alpha=0.8)

    # Add value labels
    for i, (bar, error) in enumerate(zip(bars, errors)):
        ax2.text(error + 0.02, i, f'{error:.3f}%', va='center', fontsize=9)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel('Mean of Max MAPE (%)', fontsize=11)
    ax2.set_title('Top 10 Models by Mean of Max MAPE', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3)
    ax2.invert_yaxis()

    # Plot 3: Top 10 Models by Max MAPE
    ax3 = plt.subplot(1, 3, 3)

    # Get top 10 models by max flux error (highest single error across all energy groups)
    model_max_errors = []
    for name, group in grouped:
        model, encoding, optimization = name
        max_error = group['max_error'].max()  # Get the highest max error
        if not np.isnan(max_error):
            model_max_errors.append({
                'label': f'{model}-{encoding}-{optimization}',
                'model': model,
                'error': max_error
            })

    top_max_models = sorted(model_max_errors, key=lambda x: x['error'])[:10]

    y_pos = np.arange(len(top_max_models))
    errors = [m['error'] for m in top_max_models]
    colors = [model_colors.get(m['model'], 'gray') for m in top_max_models]
    labels = [m['label'] for m in top_max_models]

    bars = ax3.barh(y_pos, errors, color=colors, alpha=0.8)

    # Add value labels
    for i, (bar, error) in enumerate(zip(bars, errors)):
        ax3.text(error + 0.02, i, f'{error:.3f}%', va='center', fontsize=9)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=9)
    ax3.set_xlabel('Max MAPE (%)', fontsize=11)
    ax3.set_title('Top 10 Models by Max MAPE', fontsize=12, fontweight='bold')
    ax3.grid(True, axis='x', alpha=0.3)
    ax3.invert_yaxis()

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

def create_best_models_summary(df, output_dir, has_energy_discretization=False, target_context='auto'):
    """Create summary showing best performing model combinations - THREE PANELS (Mean MAPE, Mean of Max MAPE, Max MAPE)"""

    # Auto-detect context from output directory if not specified
    if target_context == 'auto':
        output_path = output_dir.lower()
        if 'keff' in output_path:
            target_context = 'keff'
        elif 'thermal' in output_path:
            target_context = 'thermal'
        elif 'epithermal' in output_path:
            target_context = 'epithermal'
        elif 'fast' in output_path:
            target_context = 'fast'
        else:
            target_context = 'flux'  # Default to flux

    # Calculate mean errors for each model/encoding/optimization combination
    summary_data = []

    # Group by model configuration
    grouped = df.groupby(['model_class', 'encoding', 'optimization_method'])

    for name, group in grouped:
        model_class, encoding, optimization = name

        # Calculate metrics based on context
        mean_error = None
        mean_of_max_error = None
        max_error = None

        if target_context == 'keff':
            # K-eff metrics
            if 'keff_real' in group.columns and 'keff_predicted' in group.columns:
                keff_errors = []
                config_max_errors = []
                for _, row in group.iterrows():
                    if pd.notna(row['keff_real']) and pd.notna(row['keff_predicted']) and row['keff_real'] != 0:
                        error = abs((row['keff_predicted'] - row['keff_real']) / row['keff_real']) * 100
                        keff_errors.append(error)
                        config_max_errors.append(error)  # For k-eff, each config has only one error

                if keff_errors:
                    mean_error = np.mean(keff_errors)
                    mean_of_max_error = np.mean(config_max_errors)
                    max_error = max(keff_errors)

        elif target_context in ['thermal', 'epithermal', 'fast']:
            # Specific energy group flux metrics
            flux_errors = []
            config_max_errors = []
            for _, row in group.iterrows():
                position_errors = []
                for i in range(1, 5):
                    error_col = f'I_{i}_{target_context}_rel_error'
                    if error_col in row:
                        error = row[error_col]
                        if pd.notna(error) and error != 'N/A':
                            position_errors.append(abs(error))

                if position_errors:
                    flux_errors.extend(position_errors)
                    config_max_errors.append(max(position_errors))

            if flux_errors:
                mean_error = np.mean(flux_errors)
                mean_of_max_error = np.mean(config_max_errors) if config_max_errors else None
                max_error = max(flux_errors)

        else:
            # Total flux metrics (default)
            flux_errors = []
            config_max_errors = []

            # Try to find flux error columns in the data
            for _, row in group.iterrows():
                # Look for individual position flux errors (I_1_rel_error, I_2_rel_error, etc.)
                position_errors = []
                for i in range(1, 5):
                    error_col = f'I_{i}_rel_error'
                    if error_col in row and pd.notna(row[error_col]) and row[error_col] != 'N/A':
                        position_errors.append(abs(row[error_col]))

                # If we found position errors, add their mean and max to our list
                if position_errors:
                    flux_errors.extend(position_errors)
                    config_max_errors.append(max(position_errors))

            # If no position errors found, try looking for flux-related columns more broadly
            if not flux_errors:
                for _, row in group.iterrows():
                    # Look for any flux-related error columns
                    for col in row.index:
                        if any(term in col.lower() for term in ['flux', 'mape']) and 'rel_error' in col.lower():
                            if pd.notna(row[col]) and row[col] != 'N/A':
                                flux_errors.append(abs(row[col]))

            # If still no flux errors, try the MAPE column directly
            if not flux_errors:
                if 'MAPE' in group.columns and group['MAPE'].notna().any():
                    flux_errors = abs(group['MAPE']).tolist()
                elif 'mape' in group.columns and group['mape'].notna().any():
                    flux_errors = abs(group['mape']).tolist()

            if flux_errors:
                mean_error = np.mean(flux_errors)
                mean_of_max_error = np.mean(config_max_errors) if config_max_errors else mean_error
                max_error = max(flux_errors)

        # Only add if we have valid metrics
        if mean_error is not None and max_error is not None:
            summary_data.append({
                'model': model_class,
                'encoding': encoding,
                'optimization': optimization,
                'mean_error': mean_error,
                'mean_of_max_error': mean_of_max_error if mean_of_max_error is not None else mean_error,
                'max_error': max_error,
                'n_configs': len(group)
            })

    summary_df = pd.DataFrame(summary_data)

    if summary_df.empty:
        print("  Warning: No summary data available")
        return

    # Create figure with THREE PANELS for flux, TWO PANELS for k-eff
    n_panels = 2 if target_context == 'keff' else 3
    fig = plt.figure(figsize=(24 if n_panels == 3 else 16, 8))

    # Main title based on context
    title_map = {
        'keff': 'K-eff Model Performance Summary - Best Configurations',
        'thermal': 'Thermal Flux Model Performance Summary - Best Configurations',
        'epithermal': 'Epithermal Flux Model Performance Summary - Best Configurations',
        'fast': 'Fast Flux Model Performance Summary - Best Configurations',
        'flux': 'Total Flux Model Performance Summary - Best Configurations'
    }

    title = title_map.get(target_context, 'Model Performance Summary - Best Configurations')
    if has_energy_discretization and target_context == 'flux':
        title += '\n(Averaged Across All Energy Groups)'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Define colors
    model_colors = {
        'xgboost': '#1f77b4',
        'random_forest': '#ff7f0e',
        'svm': '#2ca02c',
        'neural_net': '#d62728'
    }

    # PANEL 1: Mean MAPE
    ax1 = plt.subplot(1, n_panels, 1)

    # Sort by mean error and take top 10
    top_models_mean = summary_df.sort_values('mean_error').head(10)

    if not top_models_mean.empty:
        y_pos = np.arange(len(top_models_mean))
        bars = ax1.barh(y_pos, top_models_mean['mean_error'])

        # Color bars by model type
        for i, (idx, row) in enumerate(top_models_mean.iterrows()):
            bars[i].set_color(model_colors.get(row['model'], 'gray'))

        # Labels
        labels = [f"{row['model']}-{row['encoding']}-{row['optimization']}"
                 for _, row in top_models_mean.iterrows()]
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=9)
        ax1.set_xlabel('Mean MAPE (%)', fontsize=10)

        ylabel_map = {
            'keff': 'Top 10 K-eff Models',
            'thermal': 'Top 10 Thermal Flux Models',
            'epithermal': 'Top 10 Epithermal Flux Models',
            'fast': 'Top 10 Fast Flux Models',
            'flux': 'Top 10 Flux Models'
        }
        ax1.set_title(ylabel_map.get(target_context, 'Top 10 Models'), fontsize=12, fontweight='bold')
        ax1.grid(True, axis='x', alpha=0.3)
        ax1.invert_yaxis()

        # Add value labels
        for i, (idx, row) in enumerate(top_models_mean.iterrows()):
            precision = 3 if target_context == 'keff' else 2
            ax1.text(row['mean_error'] + max(top_models_mean['mean_error']) * 0.02, i,
                    f"{row['mean_error']:.{precision}f}%", va='center', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No Models Found', ha='center', va='center',
                transform=ax1.transAxes, fontsize=12, alpha=0.5)

    # PANEL 2: Mean of Max MAPE (only for flux, not k-eff)
    if target_context != 'keff':
        ax2 = plt.subplot(1, n_panels, 2)

        # Sort by mean of max error and take top 10
        top_models_mean_of_max = summary_df.sort_values('mean_of_max_error').head(10)

        if not top_models_mean_of_max.empty:
            y_pos = np.arange(len(top_models_mean_of_max))
            bars = ax2.barh(y_pos, top_models_mean_of_max['mean_of_max_error'])

            # Color bars by model type
            for i, (idx, row) in enumerate(top_models_mean_of_max.iterrows()):
                bars[i].set_color(model_colors.get(row['model'], 'gray'))

            # Labels
            labels = [f"{row['model']}-{row['encoding']}-{row['optimization']}"
                     for _, row in top_models_mean_of_max.iterrows()]
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels, fontsize=9)
            ax2.set_xlabel('Mean of Max MAPE (%)', fontsize=10)

            ylabel_map_mean_of_max = {
                'thermal': 'Top 10 Thermal Flux Models (Mean of Max Error)',
                'epithermal': 'Top 10 Epithermal Flux Models (Mean of Max Error)',
                'fast': 'Top 10 Fast Flux Models (Mean of Max Error)',
                'flux': 'Top 10 Flux Models (Mean of Max Error)'
            }
            ax2.set_title(ylabel_map_mean_of_max.get(target_context, 'Top 10 Models (Mean of Max Error)'), fontsize=12, fontweight='bold')
            ax2.grid(True, axis='x', alpha=0.3)
            ax2.invert_yaxis()

            # Add value labels
            for i, (idx, row) in enumerate(top_models_mean_of_max.iterrows()):
                ax2.text(row['mean_of_max_error'] + max(top_models_mean_of_max['mean_of_max_error']) * 0.02, i,
                        f"{row['mean_of_max_error']:.2f}%", va='center', fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'No Models Found', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12, alpha=0.5)

    # PANEL 3 (or 2 for k-eff): Max MAPE
    ax3 = plt.subplot(1, n_panels, n_panels)

    # Sort by max error and take top 10
    top_models_max = summary_df.sort_values('max_error').head(10)

    if not top_models_max.empty:
        y_pos = np.arange(len(top_models_max))
        bars = ax3.barh(y_pos, top_models_max['max_error'])

        # Color bars by model type
        for i, (idx, row) in enumerate(top_models_max.iterrows()):
            bars[i].set_color(model_colors.get(row['model'], 'gray'))

        # Labels
        labels = [f"{row['model']}-{row['encoding']}-{row['optimization']}"
                 for _, row in top_models_max.iterrows()]
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels, fontsize=9)
        ax3.set_xlabel('Max MAPE (%)', fontsize=10)

        ylabel_map_max = {
            'keff': 'Top 10 K-eff Models (Max Error)',
            'thermal': 'Top 10 Thermal Flux Models (Max Error)',
            'epithermal': 'Top 10 Epithermal Flux Models (Max Error)',
            'fast': 'Top 10 Fast Flux Models (Max Error)',
            'flux': 'Top 10 Flux Models (Max Error)'
        }
        ax3.set_title(ylabel_map_max.get(target_context, 'Top 10 Models (Max Error)'), fontsize=12, fontweight='bold')
        ax3.grid(True, axis='x', alpha=0.3)
        ax3.invert_yaxis()

        # Add value labels
        for i, (idx, row) in enumerate(top_models_max.iterrows()):
            precision = 3 if target_context == 'keff' else 2
            ax3.text(row['max_error'] + max(top_models_max['max_error']) * 0.02, i,
                    f"{row['max_error']:.{precision}f}%", va='center', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No Models Found', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12, alpha=0.5)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, 'best_models_summary.png')
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
    if 'optimization_method' in df.columns:
        optimizations = sorted(df['optimization_method'].unique())
    else:
        # If no optimization_method column, create a default one
        df = df.copy()
        df['optimization_method'] = 'default'
        optimizations = ['default']

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
    if 'optimization_method' in df.columns:
        optimizations = sorted(df['optimization_method'].unique())
    else:
        # If no optimization_method column, create a default one
        df = df.copy()
        df['optimization_method'] = 'default'
        optimizations = ['default']

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
