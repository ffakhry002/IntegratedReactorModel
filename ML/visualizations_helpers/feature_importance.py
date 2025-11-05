"""
Feature importance plots for physics-based encoding
Shows both individual features and aggregated local features
Supports variable feature counts (vacuum/fill modes with different NCI options)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.inspection import permutation_importance
from .feature_importance_utils import (
    detect_physics_mode,
    reorder_importances_for_plot,
    get_aggregated_features,
    get_feature_colors
)

def create_feature_importance_plots(df, output_dir, models, target_type='flux', energy_group=None):
    """Create feature importance plots for each model type for a specific target.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with test results
    output_dir : str
        Output directory (should already be target-specific, e.g., total_flux/feature_importance)
    models : list
        List of model types to process
    target_type : str, optional
        Either 'flux' or 'keff', by default 'flux'
    energy_group : str, optional
        Optional energy group ('thermal', 'epithermal', 'fast') for energy-specific models

    Returns
    -------
    None
    """

    # Get unique model-encoding-optimization combinations
    physics_df = df[df['encoding'] == 'physics']

    if physics_df.empty:
        print("  No physics encoding results found for feature importance")
        return

    # Process each model type for the specified target
    for model_type in models:
        model_df = physics_df[physics_df['model_class'] == model_type]

        if not model_df.empty:
            # Create plots for the specified target only
            create_model_feature_importance(model_df, model_type, target_type, output_dir, energy_group)

def create_model_feature_importance(model_df, model_type, target, output_dir, energy_group=None):
    """Create feature importance plots for a specific model and target.

    Parameters
    ----------
    model_df : pandas.DataFrame
        DataFrame filtered for specific model
    model_type : str
        Type of machine learning model
    target : str
        Target variable ('flux' or 'keff')
    output_dir : str
        Directory to save plots
    energy_group : str, optional
        Optional energy group specification

    Returns
    -------
    None
    """

    # Try to load a saved model to get feature importances
    model_path = find_model_file(model_type, target, energy_group)

    if model_path and os.path.exists(model_path):
        try:
            # Load the model
            model_data = joblib.load(model_path)

            # Extract feature importances
            importances = extract_feature_importances(model_data, model_type, target)

            if importances is not None:
                # Create two plots
                create_full_feature_plot(importances, model_type, target, output_dir)
                create_aggregated_feature_plot(importances, model_type, target, output_dir)
            else:
                print(f"  Could not extract importances for {model_type} {target}")

        except Exception as e:
            print(f"  Error loading model for {model_type} {target}: {e}")
    else:
        print(f"  No model file found for {model_type} {target}")

def find_model_file(model_type, target, energy_group=None):
    """Find the saved model file for a given model type and target.

    Parameters
    ----------
    model_type : str
        Type of machine learning model
    target : str
        Target variable ('flux' or 'keff')
    energy_group : str, optional
        Optional energy group specification

    Returns
    -------
    str or None
        Path to the model file if found, None otherwise
    """

    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Look in standard model output directory
    model_dir = os.path.join(script_dir, "outputs", "models")

    # Pattern to match: {model_type}_{target}_*physics*.pkl
    # This handles both old naming (physics_*) and new naming (*_physics_*)
    import glob

    # Adjust target for energy-specific models
    if energy_group and target == 'flux':
        # For energy groups, look for models trained on that specific energy
        # e.g., flux_thermal_only, flux_epithermal_only, flux_fast_only
        search_target = f"flux_{energy_group}_only"
    else:
        search_target = target

    # Try multiple patterns to be backward compatible
    patterns = [
        os.path.join(model_dir, f"{model_type}_{search_target}_*physics*.pkl"),
        os.path.join(model_dir, f"{model_type}_{search_target}_physics_*.pkl"),
        os.path.join(model_dir, f"{model_type}_{search_target}_physics*.pkl")
    ]

    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))

    # Remove duplicates
    matches = list(set(matches))

    # Return the most recent file if multiple matches
    if matches:
        return max(matches, key=os.path.getctime)

    return None

def extract_feature_importances(model_data, model_type, target):
    """Extract feature importances from a loaded model.

    Parameters
    ----------
    model_data : dict
        Loaded model data from joblib
    model_type : str
        Type of machine learning model
    target : str
        Target variable ('flux' or 'keff')

    Returns
    -------
    numpy.ndarray or None
        Feature importances if available, None otherwise
    """

    # Skip SVM and neural_net models
    if model_type in ['svm', 'neural_net']:
        return None

    # Get the actual model
    if 'model' in model_data:
        model = model_data['model']
    else:
        return None

    # Handle position-independent wrapper if present
    if hasattr(model, 'model'):
        model = model.model

    # Extract importances based on model type
    if model_type == 'xgboost':
        # XGBoost has feature_importances_
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        # For MultiOutputRegressor
        elif hasattr(model, 'estimators_'):
            # Average importances across outputs
            importances = []
            for estimator in model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
            if importances:
                return np.mean(importances, axis=0)

    elif model_type == 'random_forest':
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_

    return None

def create_full_feature_plot(importances, model_type, target, output_dir):
    """Create a plot showing all individual features (supports 22, 29, or 37 features).

    Parameters
    ----------
    importances : numpy.ndarray
        Feature importance values
    model_type : str
        Type of machine learning model
    target : str
        Target variable ('flux' or 'keff')
    output_dir : str
        Directory to save the plot

    Returns
    -------
    None
    """
    # Detect mode from feature count
    n_features = len(importances)
    irradiation_mode, nci_mode, is_valid = detect_physics_mode(n_features)

    if not is_valid:
        print(f"  Warning: Unexpected feature count {n_features} for physics encoding")
        print(f"  Expected: 22 (vacuum), 29 (fill+single NCI), or 37 (fill+separate NCI)")
        return

    # Reorder importances for better visualization
    reordered_importances, feature_names = reorder_importances_for_plot(
        importances, irradiation_mode, nci_mode)

    # Get colors
    colors, _ = get_feature_colors(irradiation_mode, nci_mode)

    # Create figure (adjust height based on feature count)
    fig_height = 10 + (n_features - 22) * 0.2  # Scale height with feature count
    plt.figure(figsize=(12, fig_height))

    # Create bar plot
    y_pos = np.arange(len(feature_names))
    bars = plt.barh(y_pos, reordered_importances, color=colors, alpha=0.8)

    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, reordered_importances)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', va='center', fontsize=9)

    # Customize plot
    plt.yticks(y_pos, feature_names)
    plt.xlabel('Feature Importance', fontsize=12)

    mode_str = f"{irradiation_mode.capitalize()}"
    if irradiation_mode == 'fill':
        mode_str += f" + {nci_mode.capitalize()} NCI"

    plt.title(f'{model_type.upper()} Feature Importance - {target.upper()} ({mode_str}, {n_features} features)',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', label='Global Features')]

    if irradiation_mode == 'fill':
        legend_elements.extend([
            Patch(facecolor='#ff7f0e', label='Local: Fuel Density'),
            Patch(facecolor='#2ca02c', label='Local: Coolant Contact'),
            Patch(facecolor='#d62728', label='Local: Edge Distance'),
            Patch(facecolor='#9467bd', label='Local: Center Distance'),
            Patch(facecolor='#e377c2', label='Local: Vehicle Type')
        ])
        if nci_mode == 'separate':
            legend_elements.extend([
                Patch(facecolor='#17becf', label='NCI from P vehicles'),
                Patch(facecolor='#bcbd22', label='NCI from B vehicles'),
                Patch(facecolor='#7f7f7f', label='NCI from G vehicles')
            ])
        else:
            legend_elements.append(Patch(facecolor='#8c564b', label='NCI'))
    else:
        legend_elements.extend([
            Patch(facecolor='#ff7f0e', label='Local: Fuel Density'),
            Patch(facecolor='#2ca02c', label='Local: Coolant Contact'),
            Patch(facecolor='#d62728', label='Local: Edge Distance'),
            Patch(facecolor='#9467bd', label='Local: Center Distance'),
            Patch(facecolor='#8c564b', label='NCI')
        ])

    plt.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()

    # Save
    output_file = os.path.join(output_dir, f'{model_type}_all_features.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

def create_aggregated_feature_plot(importances, model_type, target, output_dir):
    """Create a plot showing global features and averaged local features"""

    # Detect mode from feature count
    n_features = len(importances)
    irradiation_mode, nci_mode, is_valid = detect_physics_mode(n_features)

    if not is_valid:
        print(f"  Warning: Unexpected feature count {n_features} for aggregated plot")
        return

    # Get aggregated features
    aggregated_importances, feature_names = get_aggregated_features(
        importances, irradiation_mode, nci_mode)

    # Get colors
    _, agg_colors = get_feature_colors(irradiation_mode, nci_mode)

    # Create figure
    fig_height = 6 + len(feature_names) * 0.1
    plt.figure(figsize=(10, fig_height))

    # Create bar plot
    y_pos = np.arange(len(feature_names))
    bars = plt.barh(y_pos, aggregated_importances, color=agg_colors, alpha=0.8)

    # Add value labels
    for bar, importance in zip(bars, aggregated_importances):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', va='center', fontsize=10)

    # Customize plot
    plt.yticks(y_pos, feature_names)
    plt.xlabel('Feature Importance', fontsize=12)

    mode_str = f"{irradiation_mode.capitalize()}"
    if irradiation_mode == 'fill':
        mode_str += f" + {nci_mode.capitalize()} NCI"

    plt.title(f'{model_type.upper()} Feature Importance - {target.upper()} (Aggregated, {mode_str})',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Global Features'),
        Patch(facecolor='#ff7f0e', label='Local Features (averaged)'),
        Patch(facecolor='#8c564b', label='NCI (averaged)')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # Add note about averaging
    plt.figtext(0.5, 0.02, 'Note: Local and NCI features are averaged across all 4 positions',
                ha='center', fontsize=9, style='italic')

    plt.tight_layout()

    # Save
    output_file = os.path.join(output_dir, f'{model_type}_aggregated_features.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
