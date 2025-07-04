"""
Optuna optimization visualization helper functions.

This module provides comprehensive visualization tools for analyzing Optuna optimization results,
including parameter importance, optimization history, parameter relationships, and more.
"""

import os
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import warnings
import joblib
warnings.filterwarnings('ignore')

# Import Optuna visualization modules
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_contour,
    plot_edf,
    plot_intermediate_values,
    plot_parallel_coordinate,
    plot_pareto_front,
    plot_slice,
    plot_timeline
)

# For matplotlib-based plots
import optuna.visualization.matplotlib as optuna_plt


def create_optuna_visualization_directory(base_path: str, model_name: str, target: str, encoding: str = None) -> str:
    """
    Create directory structure for Optuna visualizations.

    Args:
        base_path: Base directory for visualizations
        model_name: Name of the ML model (e.g., 'xgboost', 'random_forest')
        target: Target variable (e.g., 'flux', 'keff')
        encoding: Encoding method (e.g., 'categorical', 'physics')

    Returns:
        Path to the created directory
    """
    # Create folder name that includes target type for clarity
    # Examples: svm_physics_total_flux, svm_physics_keff, etc.
    if encoding:
        # Clean up target name for folder
        if target.startswith('flux_'):
            # flux_total -> total_flux
            target_clean = target.replace('flux_', '') + '_flux'
        else:
            # keff -> keff
            target_clean = target
        model_folder_name = f"{model_name}_{encoding}_{target_clean}"
    else:
        model_folder_name = f"{model_name}_{target}"

    optuna_dir = os.path.join(base_path, 'optuna_analysis', model_folder_name)
    os.makedirs(optuna_dir, exist_ok=True)
    return optuna_dir


def save_optimization_history(study: optuna.Study, save_dir: str) -> None:
    """Save optimization history plot showing how the objective value improves over trials."""
    # Get completed trials
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]

    if not complete_trials:
        print("  ⚠ No completed trials to plot")
        return

    # Extract trial numbers and values
    trial_numbers = [t.number for t in complete_trials]
    trial_values = [t.value for t in complete_trials]

    # Calculate running best
    best_values = []
    current_best = float('inf')
    for value in trial_values:
        current_best = min(current_best, value)
        best_values.append(current_best)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # First subplot: Linear scale
    ax1.scatter(trial_numbers, trial_values, alpha=0.3, s=20, label='Trial Values')
    ax1.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best Value')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Objective Value')
    ax1.set_title(f'Optimization History - {study.study_name if study.study_name else "Study"}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Second subplot: Improvement ratio (LINEAR scale)
    initial_value = trial_values[0]
    improvement_ratio = [initial_value / v for v in trial_values]
    best_improvement = [initial_value / v for v in best_values]

    ax2.scatter(trial_numbers, improvement_ratio, alpha=0.3, s=20, label='Trial Improvement')
    ax2.plot(trial_numbers, best_improvement, 'r-', linewidth=2, label='Best Improvement')

    # Add reference lines
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='No Improvement')
    max_improvement = max(best_improvement)
    ax2.axhline(y=max_improvement, color='green', linestyle=':', alpha=0.5,
               label=f'Max Improvement: {max_improvement:.3f}')

    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Improvement Ratio (Initial / Current)')
    ax2.set_title('Optimization Improvement Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'optimization_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved optimization history")

def save_param_importances(study: optuna.Study, save_dir: str) -> None:
    """Save parameter importance plot using fANOVA."""
    try:
        fig = optuna_plt.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'parameter_importances.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved parameter importances")
    except Exception as e:
        print(f"  ⚠ Could not generate parameter importance plot: {str(e)}")


def save_param_relationships(study: optuna.Study, save_dir: str, n_params: Optional[int] = None) -> None:
    """Save contour plots showing relationships between parameters."""
    try:
        # Get parameter names
        params = list(study.best_params.keys())

        # Limit parameters if specified
        if n_params and len(params) > n_params:
            # Get most important parameters if importance analysis is available
            try:
                importance = optuna.importance.get_param_importances(study)
                sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                params = [p[0] for p in sorted_params[:n_params]]
            except:
                params = params[:n_params]

        # Create contour plot for top parameters
        if len(params) >= 2:
            fig = optuna_plt.plot_contour(study, params=params[:2])
            plt.suptitle(f'Parameter Relationship: {str(params[0])} vs {str(params[1])}')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'param_contour_top2.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved parameter contour plot")

            # If we have more parameters, create additional plots
            if len(params) >= 4:
                fig = optuna_plt.plot_contour(study, params=params[2:4])
                plt.suptitle(f'Parameter Relationship: {str(params[2])} vs {str(params[3])}')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'param_contour_next2.png'), dpi=300, bbox_inches='tight')
                plt.close()

    except Exception as e:
        print(f"  ⚠ Could not generate contour plots: {str(e)}")


def save_slice_plots(study: optuna.Study, save_dir: str) -> None:
    """Save slice plots showing how each parameter affects the objective."""
    try:
        fig = optuna_plt.plot_slice(study)
        plt.suptitle('Parameter Slice Plots')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'parameter_slices.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved parameter slice plots")
    except Exception as e:
        print(f"  ⚠ Could not generate slice plots: {str(e)}")


def save_parallel_coordinate_plot(study: optuna.Study, save_dir: str, top_n: int = 20) -> None:
    """Save parallel coordinate plot for top N trials."""
    try:
        # Get top N trials
        trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))[:top_n]

        fig = optuna_plt.plot_parallel_coordinate(study, target_name="Objective Value")
        plt.title(f'Parallel Coordinates - Top {len(trials)} Trials')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'parallel_coordinates.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved parallel coordinate plot")
    except Exception as e:
        print(f"  ⚠ Could not generate parallel coordinate plot: {str(e)}")


def save_edf_plot(study: optuna.Study, save_dir: str) -> None:
    """Save empirical distribution function plot."""
    try:
        fig = optuna_plt.plot_edf(study)
        plt.title('Empirical Distribution Function')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'edf_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved EDF plot")
    except Exception as e:
        print(f"  ⚠ Could not generate EDF plot: {str(e)}")


def save_timeline_plot(study: optuna.Study, save_dir: str) -> None:
    """Save timeline plot showing when each trial was completed."""
    try:
        fig = optuna_plt.plot_timeline(study)
        plt.title('Optimization Timeline')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved timeline plot")
    except Exception as e:
        print(f"  ⚠ Could not generate timeline plot: {str(e)}")


def save_hyperparameter_history(study: optuna.Study, save_dir: str) -> None:
    """Save individual hyperparameter value history over trials."""
    # Ensure all parameter names are strings
    params = [str(p) for p in study.best_params.keys()]
    n_params = len(params)

    if n_params == 0:
        print("  ⚠ No parameters to plot")
        return

    # Create subplots
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 4*n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for idx, param in enumerate(params):
        values = []
        trial_numbers = []

        # Convert param back to original type for lookup
        original_param = list(study.best_params.keys())[idx]

        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                if original_param in trial.params:
                    values.append(trial.params[original_param])
                    trial_numbers.append(trial.number)

        if values:
            # Check if parameter is numeric or categorical
            try:
                # Try to convert first few values to float to check if numeric
                test_values = values[:min(5, len(values))]
                numeric_test = []
                for v in test_values:
                    if isinstance(v, (int, float)):
                        numeric_test.append(float(v))
                    elif isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit():
                        numeric_test.append(float(v))
                    else:
                        raise ValueError("Not numeric")
                is_numeric = True
            except (ValueError, TypeError):
                is_numeric = False

            if is_numeric:
                # Handle numeric parameters normally
                try:
                    numeric_values = []
                    for v in values:
                        if isinstance(v, (int, float)):
                            numeric_values.append(float(v))
                        else:
                            numeric_values.append(float(v))

                    axes[idx].scatter(trial_numbers, numeric_values, alpha=0.6)
                    axes[idx].set_ylabel(str(param))
                    axes[idx].grid(True, alpha=0.3)

                    # Add best value line
                    best_value = study.best_params[original_param]
                    try:
                        best_numeric = float(best_value)
                        axes[idx].axhline(y=best_numeric, color='red', linestyle='--',
                                        label=f'Best: {best_numeric:.4g}')
                        axes[idx].legend()
                    except (ValueError, TypeError):
                        # If best_value can't be converted to float, just show as string
                        axes[idx].text(0.02, 0.98, f'Best: {best_value}',
                                     transform=axes[idx].transAxes, verticalalignment='top',
                                     bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
                except (ValueError, TypeError):
                    # Fall back to categorical handling if conversion fails
                    is_numeric = False

            if not is_numeric:
                # Handle categorical parameters
                # Convert categorical values to numeric indices for plotting
                unique_values = list(set(values))
                value_to_index = {val: idx for idx, val in enumerate(unique_values)}
                numeric_indices = [value_to_index[val] for val in values]

                axes[idx].scatter(trial_numbers, numeric_indices, alpha=0.6)
                axes[idx].set_ylabel(str(param))
                axes[idx].set_yticks(range(len(unique_values)))
                axes[idx].set_yticklabels([str(val) for val in unique_values])
                axes[idx].grid(True, alpha=0.3)

                # Add best value line
                best_value = study.best_params[original_param]
                if best_value in value_to_index:
                    best_index = value_to_index[best_value]
                    axes[idx].axhline(y=best_index, color='red', linestyle='--',
                                    label=f'Best: {best_value}')
                    axes[idx].legend()

    axes[-1].set_xlabel('Trial Number')
    plt.suptitle('Hyperparameter Values Over Trials')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hyperparameter_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved hyperparameter history")


def save_objective_statistics(study: optuna.Study, save_dir: str) -> None:
    """Save detailed statistics about the optimization."""
    stats_file = os.path.join(save_dir, 'optimization_statistics.txt')

    with open(stats_file, 'w') as f:
        f.write(f"Optuna Optimization Statistics\n")
        f.write(f"{'='*50}\n\n")

        # Basic statistics
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        f.write(f"Total trials: {len(study.trials)}\n")
        f.write(f"Completed trials: {len(complete_trials)}\n")
        f.write(f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}\n")
        f.write(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n\n")

        # Best trial info
        f.write(f"Best trial:\n")
        f.write(f"  Number: {study.best_trial.number}\n")
        f.write(f"  Value: {study.best_value:.6f}\n")
        f.write(f"  Parameters:\n")
        for param, value in study.best_params.items():
            f.write(f"    {str(param)}: {value}\n")

        # Objective value statistics
        if complete_trials:
            values = [t.value for t in complete_trials if t.value is not None]
            if values:
                f.write(f"\nObjective value statistics:\n")
                f.write(f"  Mean: {np.mean(values):.6f}\n")
                f.write(f"  Std: {np.std(values):.6f}\n")
                f.write(f"  Min: {np.min(values):.6f}\n")
                f.write(f"  Max: {np.max(values):.6f}\n")
                f.write(f"  Median: {np.median(values):.6f}\n")

        # Parameter ranges actually explored
        f.write(f"\nParameter ranges explored:\n")
        for param in study.best_params.keys():
            param_values = [t.params[param] for t in complete_trials if param in t.params]
            if param_values:
                f.write(f"  {str(param)}:\n")

                # Check if parameter is numeric or categorical
                try:
                    # Try to convert to float to check if numeric
                    numeric_values = [float(v) for v in param_values]
                    is_numeric = True
                except (ValueError, TypeError):
                    is_numeric = False

                if is_numeric:
                    # Handle numeric parameters
                    f.write(f"    Min: {np.min(numeric_values):.6f}\n")
                    f.write(f"    Max: {np.max(numeric_values):.6f}\n")
                    f.write(f"    Mean: {np.mean(numeric_values):.6f}\n")
                    f.write(f"    Std: {np.std(numeric_values):.6f}\n")
                    f.write(f"    Unique values: {len(np.unique(numeric_values))}\n")
                else:
                    # Handle categorical parameters
                    unique_values = list(set(param_values))
                    f.write(f"    Possible values: {unique_values}\n")
                    f.write(f"    Unique values: {len(unique_values)}\n")
                    # Count frequency of each value
                    value_counts = {}
                    for val in param_values:
                        value_counts[val] = value_counts.get(val, 0) + 1
                    f.write(f"    Value frequencies:\n")
                    for val, count in sorted(value_counts.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"      {val}: {count} times ({100*count/len(param_values):.1f}%)\n")

    print(f"  ✓ Saved optimization statistics")


def save_convergence_plot(study: optuna.Study, save_dir: str) -> None:
    """Plot convergence of best value over trials."""
    values = []
    best_values = []
    current_best = float('inf')

    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
            values.append(trial.value)
            current_best = min(current_best, trial.value)
            best_values.append(current_best)

    if best_values:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # First subplot: Linear scale
        ax1.plot(best_values, label='Best Value', linewidth=2)
        ax1.scatter(range(len(values)), values, alpha=0.3, s=20, label='Trial Values')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Optimization Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Second subplot: Improvement ratio (LINEAR scale)
        initial_value = values[0]
        improvement_ratio = [initial_value / v for v in values]
        best_improvement = [initial_value / v for v in best_values]

        ax2.plot(best_improvement, label='Best Value Improvement', linewidth=2, color='red')
        ax2.scatter(range(len(improvement_ratio)), improvement_ratio,
                   alpha=0.3, s=20, label='Trial Improvement')

        # Add horizontal line at y=1 (no improvement)
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='No Improvement')

        # Add horizontal line at best improvement for reference
        max_improvement = max(best_improvement)
        ax2.axhline(y=max_improvement, color='green', linestyle=':', alpha=0.5,
                   label=f'Max Improvement: {max_improvement:.3f}')

        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Improvement Ratio (Initial / Current)')
        ax2.set_title('Convergence - Improvement Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Set y-axis to start at 0 for better visualization
        ax2.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'convergence_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved convergence plot")

def save_parameter_correlation_matrix(study: optuna.Study, save_dir: str) -> None:
    """Save correlation matrix between parameters for completed trials."""
    try:
        # Get trials dataframe
        trials_df = study.trials_dataframe()
        complete_trials = trials_df[trials_df['state'] == 'COMPLETE']

        # Extract parameter columns
        param_columns = [col for col in complete_trials.columns if col.startswith('params_')]

        if len(param_columns) > 1:
            # Filter out non-numeric parameters
            numeric_param_columns = []
            for col in param_columns:
                try:
                    # Try to convert column to numeric
                    pd.to_numeric(complete_trials[col], errors='raise')
                    numeric_param_columns.append(col)
                except (ValueError, TypeError):
                    # Skip categorical parameters
                    param_name = col.replace('params_', '')
                    print(f"    Skipping categorical parameter '{param_name}' from correlation matrix")
                    continue

            if len(numeric_param_columns) > 1:
                # Calculate correlation matrix for numeric parameters only
                numeric_data = complete_trials[numeric_param_columns].apply(pd.to_numeric, errors='coerce')
                corr_matrix = numeric_data.corr()

                # Clean up column names
                clean_names = [col.replace('params_', '') for col in numeric_param_columns]
                corr_matrix.index = clean_names
                corr_matrix.columns = clean_names

                # Plot
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
                plt.title('Parameter Correlation Matrix (Numeric Parameters Only)')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'parameter_correlations.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved parameter correlation matrix")
            else:
                print(f"  ⚠ Not enough numeric parameters for correlation matrix (found {len(numeric_param_columns)})")
        else:
            print(f"  ⚠ Not enough parameters for correlation matrix (found {len(param_columns)})")

    except Exception as e:
        print(f"  ⚠ Could not generate correlation matrix: {str(e)}")


def save_objective_distribution(study: optuna.Study, save_dir: str) -> None:
    """Save distribution plot of objective values."""
    try:
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        values = [t.value for t in complete_trials if t.value is not None]

        if values:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Histogram
            ax1.hist(values, bins=100, alpha=0.7, edgecolor='black')
            ax1.axvline(study.best_value, color='red', linestyle='--',
                       label=f'Best: {study.best_value:.4f}')
            ax1.set_xlabel('Objective Value')
            ax1.set_ylabel('Count')
            ax1.set_title('Distribution of Objective Values')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Box plot
            ax2.boxplot(values, vert=True)
            ax2.set_ylabel('Objective Value')
            ax2.set_title('Box Plot of Objective Values')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'objective_distribution.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved objective distribution plots")

    except Exception as e:
        print(f"  ⚠ Could not generate distribution plots: {str(e)}")


def generate_all_optuna_visualizations(
    study: optuna.Study,
    save_base_dir: str,
    model_name: str,
    target: str,
    encoding: str = None,
    include_all: bool = True
) -> None:
    """
    Generate all available Optuna visualizations for a study.

    Args:
        study: Completed Optuna study
        save_base_dir: Base directory for saving visualizations
        model_name: Name of the ML model
        target: Target variable name
        encoding: Encoding method name
        include_all: Whether to include all visualization types
    """
    if encoding:
        print(f"\nGenerating Optuna visualizations for {model_name}_{encoding} - {target}...")
    else:
        print(f"\nGenerating Optuna visualizations for {model_name} - {target}...")

    # Create directory
    save_dir = create_optuna_visualization_directory(save_base_dir, model_name, target, encoding)

    # Use original study directly
    filtered_study = study

    # List of visualization functions with their names for error reporting
    core_visualizations = [
        (save_optimization_history, "optimization history"),
        (save_param_importances, "parameter importances"),
        (save_slice_plots, "slice plots"),
        (save_hyperparameter_history, "hyperparameter history"),
        (save_objective_statistics, "objective statistics"),
        (save_convergence_plot, "convergence plot")
    ]

    additional_visualizations = [
        (save_param_relationships, "parameter relationships"),
        (save_edf_plot, "EDF plot"),
        (save_timeline_plot, "timeline plot"),
        (save_parameter_correlation_matrix, "parameter correlation matrix"),
        (save_objective_distribution, "objective distribution")
    ]

    # Generate core visualizations with error handling using filtered study
    for viz_func, viz_name in core_visualizations:
        try:
            viz_func(filtered_study, save_dir)
        except Exception as e:
            print(f"  ⚠ ERROR generating {viz_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    if include_all:
        # Generate additional visualizations with error handling using filtered study
        for viz_func, viz_name in additional_visualizations:
            try:
                viz_func(filtered_study, save_dir)
            except Exception as e:
                print(f"  ⚠ ERROR generating {viz_name}: {str(e)}")

    # Save study for later analysis
    try:
        study_file = os.path.join(save_dir, 'study.pkl')
        joblib.dump(study, study_file)
        print(f"  ✓ Saved study object to {study_file}")
    except Exception as e:
        print(f"  ⚠ ERROR saving study object: {str(e)}")

    print(f"\nOptuna visualizations saved to: {save_dir}")


# Additional custom visualizations

def plot_learning_curves_by_param(study: optuna.Study, param_name: str, save_dir: str) -> None:
    """Plot how objective changes with a specific parameter."""
    try:
        trials_df = study.trials_dataframe()
        complete_trials = trials_df[trials_df['state'] == 'COMPLETE'].copy()

        if f'params_{param_name}' in complete_trials.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(complete_trials[f'params_{param_name}'],
                       complete_trials['value'],
                       alpha=0.6)

            # Add trend line
            z = np.polyfit(complete_trials[f'params_{param_name}'],
                          complete_trials['value'], 2)
            p = np.poly1d(z)
            x_line = np.linspace(complete_trials[f'params_{param_name}'].min(),
                               complete_trials[f'params_{param_name}'].max(), 100)
            plt.plot(x_line, p(x_line), "r--", alpha=0.8, label='Trend')

            plt.xlabel(param_name)
            plt.ylabel('Objective Value')
            plt.title(f'Objective vs {param_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'objective_vs_{param_name}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

    except Exception as e:
        print(f"  ⚠ Could not plot learning curve for {param_name}: {str(e)}")
