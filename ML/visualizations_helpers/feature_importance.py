"""
Feature importance plots for physics-based encoding
Shows both individual features and aggregated local features
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.inspection import permutation_importance

def create_feature_importance_plots(df, output_dir, models, target_type='flux', energy_group=None):
    """Create feature importance plots for each model type for a specific target

    Args:
        df: DataFrame with test results
        output_dir: Output directory (should already be target-specific, e.g., total_flux/feature_importance)
        models: List of model types to process
        target_type: Either 'flux' or 'keff'
        energy_group: Optional energy group ('thermal', 'epithermal', 'fast') for energy-specific models
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
    """Create feature importance plots for a specific model and target"""

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
    """Find the saved model file for a given model type and target"""

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

# feature_importance.py - UPDATED extract_feature_importances function

def extract_feature_importances(model_data, model_type, target):
    """Extract feature importances from a loaded model"""

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
    """Create a plot showing all 22 individual features"""

    # Correct structure: Global(2), Local(16), NCI(4) = 22 features
    # Local features: 4 per position (fuel density, coolant contact, edge distance, center distance)

    # Extract importances in logical order for visualization
    reordered_importances = [
        # Global features (0-1)
        importances[0],  # Global: Avg Distance
        importances[1],  # Global: Symmetry
        # Fuel Density for all positions (2,6,10,14)
        importances[2],   # Pos1: Fuel Density
        importances[6],   # Pos2: Fuel Density
        importances[10],  # Pos3: Fuel Density
        importances[14],  # Pos4: Fuel Density
        # Coolant Contact for all positions (3,7,11,15)
        importances[3],   # Pos1: Coolant Contact
        importances[7],   # Pos2: Coolant Contact
        importances[11],  # Pos3: Coolant Contact
        importances[15],  # Pos4: Coolant Contact
        # Edge Distance for all positions (4,8,12,16)
        importances[4],   # Pos1: Edge Distance
        importances[8],   # Pos2: Edge Distance
        importances[12],  # Pos3: Edge Distance
        importances[16],  # Pos4: Edge Distance
        # Center Distance for all positions (5,9,13,17)
        importances[5],   # Pos1: Center Distance
        importances[9],   # Pos2: Center Distance
        importances[13],  # Pos3: Center Distance
        importances[17],  # Pos4: Center Distance
        # NCI for all positions (18,19,20,21)
        importances[18],  # Pos1: NCI
        importances[19],  # Pos2: NCI
        importances[20],  # Pos3: NCI
        importances[21],  # Pos4: NCI
    ]

    # Define feature names in new order
    feature_names = [
        'Global: Avg Distance',
        'Global: Symmetry',
        'Pos1: Fuel Density',
        'Pos2: Fuel Density',
        'Pos3: Fuel Density',
        'Pos4: Fuel Density',
        'Pos1: Coolant Contact',
        'Pos2: Coolant Contact',
        'Pos3: Coolant Contact',
        'Pos4: Coolant Contact',
        'Pos1: Edge Distance',
        'Pos2: Edge Distance',
        'Pos3: Edge Distance',
        'Pos4: Edge Distance',
        'Pos1: Center Distance',
        'Pos2: Center Distance',
        'Pos3: Center Distance',
        'Pos4: Center Distance',
        'Pos1: NCI',
        'Pos2: NCI',
        'Pos3: NCI',
        'Pos4: NCI'
    ]

    # Create figure
    plt.figure(figsize=(12, 10))

    # Create bar plot
    y_pos = np.arange(len(feature_names))

    # Color scheme: Blue for global, then different shades for each local feature type
    colors = (
        ['#1f77b4'] * 2 +      # Blue for global features
        ['#ff7f0e'] * 4 +      # Orange for fuel density
        ['#2ca02c'] * 4 +      # Green for coolant contact
        ['#d62728'] * 4 +      # Red for edge distance
        ['#9467bd'] * 4 +      # Purple for center distance
        ['#8c564b'] * 4        # Brown for NCI
    )

    bars = plt.barh(y_pos, reordered_importances, color=colors, alpha=0.8)

    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, reordered_importances)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', va='center', fontsize=9)

    # Customize plot
    plt.yticks(y_pos, feature_names)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'{model_type.upper()} Feature Importance - {target.upper()} (Grouped by Feature Type)',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # Add legend with new color scheme
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Global Features'),
        Patch(facecolor='#ff7f0e', label='Local: Fuel Density'),
        Patch(facecolor='#2ca02c', label='Local: Coolant Contact'),
        Patch(facecolor='#d62728', label='Local: Edge Distance'),
        Patch(facecolor='#9467bd', label='Local: Center Distance'),
        Patch(facecolor='#8c564b', label='Neutron Competition Index (NCI)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    # Add vertical separators between feature groups
    plt.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5)  # After global
    plt.axhline(y=5.5, color='gray', linestyle='--', alpha=0.5)  # After fuel density
    plt.axhline(y=9.5, color='gray', linestyle='--', alpha=0.5)  # After coolant contact
    plt.axhline(y=13.5, color='gray', linestyle='--', alpha=0.5) # After edge distance
    plt.axhline(y=17.5, color='gray', linestyle='--', alpha=0.5) # After center distance

    plt.tight_layout()

    # Save
    output_file = os.path.join(output_dir, f'{model_type}_all_features.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

def create_aggregated_feature_plot(importances, model_type, target, output_dir):
    """Create a plot showing global features and averaged local features"""

    # Extract global features
    global_importances = [importances[0], importances[1]]  # Avg distance, symmetry

    # Calculate averaged local features (4 features × 4 positions)
    local_fuel_density = np.mean([importances[2], importances[6], importances[10], importances[14]])
    local_coolant_contact = np.mean([importances[3], importances[7], importances[11], importances[15]])
    local_edge_distance = np.mean([importances[4], importances[8], importances[12], importances[16]])
    local_center_distance = np.mean([importances[5], importances[9], importances[13], importances[17]])

    # Calculate averaged NCI features
    local_nci = np.mean([importances[18], importances[19], importances[20], importances[21]])

    # Combine for plotting
    aggregated_importances = np.concatenate([
        global_importances,
        [local_fuel_density, local_coolant_contact, local_edge_distance, local_center_distance, local_nci]
    ])

    feature_names = [
        'Global: Avg Distance',
        'Global: Symmetry',
        'Local: Fuel Density (avg)',
        'Local: Coolant Contact (avg)',
        'Local: Edge Distance (avg)',
        'Local: Center Distance (avg)',
        'Local: NCI (avg)'
    ]

    # Create figure
    plt.figure(figsize=(10, 6))

    # Create bar plot
    y_pos = np.arange(len(feature_names))
    colors = ['#1f77b4'] * 2 + ['#ff7f0e'] * 4 + ['#8c564b'] * 1  # Blue for global, orange for local, brown for NCI

    bars = plt.barh(y_pos, aggregated_importances, color=colors, alpha=0.8)

    # Add value labels
    for bar, importance in zip(bars, aggregated_importances):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', va='center', fontsize=10)

    # Customize plot
    plt.yticks(y_pos, feature_names)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'{model_type.upper()} Feature Importance - {target.upper()} (Aggregated)',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Global Features'),
        Patch(facecolor='#ff7f0e', label='Local Features (averaged)'),
        Patch(facecolor='#8c564b', label='NCI (averaged)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    # Add note about averaging
    plt.figtext(0.5, 0.02, 'Note: Local and NCI features are averaged across all 4 positions',
                ha='center', fontsize=9, style='italic')

    plt.tight_layout()

    # Save
    output_file = os.path.join(output_dir, f'{model_type}_aggregated_features.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
