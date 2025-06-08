#!/usr/bin/env python3
"""
Main visualization runner for Nuclear Reactor ML results
Generates comprehensive visualization suite for model performance analysis
"""

import os
import sys
import pandas as pd
import json
from datetime import datetime

# Add visualization helpers to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import visualization modules
from visualizations_helpers.performance_heatmaps import create_performance_heatmaps
from visualizations_helpers.spatial_error_heatmaps import create_spatial_error_heatmaps
from visualizations_helpers.feature_importance import create_feature_importance_plots
from visualizations_helpers.config_error_plots import create_config_error_plots
from visualizations_helpers.rel_error_trackers import create_rel_error_tracker_plots
from visualizations_helpers.summary_statistics import create_summary_statistics_plots
from visualizations_helpers.data_loader import load_test_results

def ensure_directories(base_output_dir):
    """Create all necessary output directories"""
    directories = [
        base_output_dir,
        os.path.join(base_output_dir, 'performance_heatmaps'),
        os.path.join(base_output_dir, 'spatial_error_heatmaps'),
        os.path.join(base_output_dir, 'feature_importance'),
        os.path.join(base_output_dir, 'feature_importance', 'flux'),
        os.path.join(base_output_dir, 'feature_importance', 'keff'),
        os.path.join(base_output_dir, 'config_error_plots'),
        os.path.join(base_output_dir, 'rel_error_trackers'),
        os.path.join(base_output_dir, 'rel_error_trackers', 'max_rel_error'),
        os.path.join(base_output_dir, 'rel_error_trackers', 'mean_rel_error'),
        os.path.join(base_output_dir, 'rel_error_trackers', 'keff_rel_error'),
        os.path.join(base_output_dir, 'summary_statistics')
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created/verified directory: {directory}")

def main():
    """Main visualization pipeline"""
    print("\n" + "="*80)
    print("NUCLEAR REACTOR ML VISUALIZATION PIPELINE")
    print("="*80 + "\n")

    # Get script directory and set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_base_dir = os.path.join(script_dir, 'outputs', 'visualizations')

    # Create output directories
    ensure_directories(output_base_dir)

    # Get Excel file path from user
    print("\nPlease provide the path to your test results Excel file.")
    print("(This should be in ML/outputs/excel_reports/ or similar location)")

    # First, show available files if we can find the directory
    possible_dirs = [
        os.path.join(script_dir, '..', 'ML', 'outputs', 'excel_reports'),
        os.path.join(script_dir, 'outputs', 'excel_reports'),
        os.path.join(script_dir, '..', 'outputs', 'excel_reports'),
        'ML/outputs/excel_reports'
    ]

    found_files = []
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.startswith('test_results_') and f.endswith('.xlsx')]
            for f in files:
                found_files.append(os.path.join(dir_path, f))

    if found_files:
        print("\nFound these test result files:")
        for i, f in enumerate(found_files, 1):
            print(f"  {i}. {f}")
        print(f"  {len(found_files)+1}. Enter custom path")

        choice = input(f"\nSelect file (1-{len(found_files)+1}): ").strip()

        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(found_files):
                excel_file = found_files[choice_idx]
            else:
                excel_file = input("Enter full path to Excel file: ").strip()
        except:
            excel_file = input("Enter full path to Excel file: ").strip()
    else:
        excel_file = input("Enter full path to Excel file: ").strip()

    # Verify file exists
    if not os.path.exists(excel_file):
        print(f"ERROR: File not found: {excel_file}")
        return

    print(f"\nLoading test results from: {excel_file}")

    # Load test results
    try:
        test_results_df = load_test_results(excel_file)
        print(f"Loaded {len(test_results_df)} test results")

        # Extract unique values for iteration
        models = test_results_df['model_class'].unique()
        encodings = test_results_df['encoding'].unique()
        optimizations = test_results_df['optimization_method'].unique()

        print(f"\nFound configurations:")
        print(f"  Models: {', '.join(models)}")
        print(f"  Encodings: {', '.join(encodings)}")
        print(f"  Optimizations: {', '.join(optimizations)}")

        # Check for missing expected values
        expected_models = ['xgboost', 'random_forest', 'svm', 'neural_net']
        expected_encodings = ['one_hot', 'categorical', 'physics', 'spatial', 'graph']
        expected_optimizations = ['optuna', 'three_stage', 'none']

        missing_models = [m for m in expected_models if m not in models]
        missing_encodings = [e for e in expected_encodings if e not in encodings]
        missing_optimizations = [o for o in expected_optimizations if o not in optimizations]

        if missing_models:
            print(f"\n⚠️  Missing models: {', '.join(missing_models)}")
        if missing_encodings:
            print(f"⚠️  Missing encodings: {', '.join(missing_encodings)}")
        if missing_optimizations:
            print(f"⚠️  Missing optimizations: {', '.join(missing_optimizations)}")

        # Check data availability
        print(f"\n📊 Data availability check:")

        # Check for flux data
        flux_cols = [col for col in test_results_df.columns if 'flux' in col.lower() or col.startswith('I_')]
        if flux_cols:
            print(f"  ✓ Flux data found ({len(flux_cols)} columns)")
        else:
            print(f"  ✗ No flux data found")

        # Check for k-eff data
        keff_cols = [col for col in test_results_df.columns if 'keff' in col.lower()]
        if keff_cols:
            print(f"  ✓ K-eff data found ({len(keff_cols)} columns)")
        else:
            print(f"  ✗ No k-eff data found")

        # Ask user if they want to continue
        if missing_models or missing_encodings or missing_optimizations:
            response = input("\n⚠️  Some expected data is missing. Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("Visualization cancelled.")
                return

    except Exception as e:
        print(f"ERROR loading test results: {e}")
        return

    # Generate visualizations
    print("\n" + "-"*60)
    print("Generating visualizations...")
    print("-"*60)

    try:
        # 1. Performance Heatmaps (R²)
        print("\n1. Creating performance heatmaps...")
        try:
            create_performance_heatmaps(
                test_results_df,
                os.path.join(output_base_dir, 'performance_heatmaps')
            )
        except Exception as e:
            print(f"  ERROR in performance heatmaps: {e}")
            print("  Continuing with other visualizations...")

        # 2. Spatial Error Heatmaps
        print("\n2. Creating spatial error heatmaps...")
        try:
            create_spatial_error_heatmaps(
                test_results_df,
                os.path.join(output_base_dir, 'spatial_error_heatmaps'),
                models, encodings, optimizations
            )
        except Exception as e:
            print(f"  ERROR in spatial error heatmaps: {e}")
            print("  Continuing with other visualizations...")

        # 3. Feature Importance Plots
        print("\n3. Creating feature importance plots...")
        try:
            create_feature_importance_plots(
                test_results_df,
                os.path.join(output_base_dir, 'feature_importance'),
                models
            )
        except Exception as e:
            print(f"  ERROR in feature importance plots: {e}")
            print("  Continuing with other visualizations...")

        # 4. Config Error Plots
        print("\n4. Creating configuration error plots...")
        try:
            create_config_error_plots(
                test_results_df,
                os.path.join(output_base_dir, 'config_error_plots')
            )
        except Exception as e:
            print(f"  ERROR in config error plots: {e}")
            print("  Continuing with other visualizations...")

        # 5. Relative Error Tracker Plots
        print("\n5. Creating relative error tracker plots...")
        try:
            create_rel_error_tracker_plots(
                test_results_df,
                os.path.join(output_base_dir, 'rel_error_trackers'),
                encodings
            )
        except Exception as e:
            print(f"  ERROR in relative error tracker plots: {e}")
            print("  Continuing with other visualizations...")

        # 6. Summary Statistics
        print("\n6. Creating summary statistics visualizations...")
        try:
            create_summary_statistics_plots(
                test_results_df,
                os.path.join(output_base_dir, 'summary_statistics')
            )
        except Exception as e:
            print(f"  ERROR in summary statistics: {e}")
            print("  Continuing with other visualizations...")

    except Exception as e:
        print(f"\nCRITICAL ERROR during visualization generation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create summary report
    summary_file = os.path.join(output_base_dir, 'visualization_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("NUCLEAR REACTOR ML VISUALIZATION SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source data: {excel_file}\n")
        f.write(f"Output directory: {output_base_dir}\n")
        f.write("\nVisualization Categories:\n")
        f.write("1. Performance Heatmaps - R² scores for all model combinations\n")
        f.write("2. Spatial Error Heatmaps - MAPE by reactor position\n")
        f.write("3. Feature Importance - Physics-based encoding analysis\n")
        f.write("4. Config Error Plots - Error trends across configurations\n")
        f.write("5. Relative Error Trackers - Detailed error analysis by encoding\n")
        f.write("6. Summary Statistics - Best model combinations\n")

    print("\n" + "="*80)
    print("VISUALIZATION PIPELINE COMPLETE!")
    print("="*80)
    print(f"All visualizations saved to: {output_base_dir}")
    print(f"Summary report: {summary_file}")
    print("\nVisualization categories generated:")
    print("  ✓ Performance heatmaps (R² scores)")
    print("  ✓ Spatial error heatmaps (MAPE by position)")
    print("  ✓ Feature importance plots")
    print("  ✓ Configuration error plots")
    print("  ✓ Relative error trackers")
    print("  ✓ Summary statistics")

if __name__ == "__main__":
    main()
