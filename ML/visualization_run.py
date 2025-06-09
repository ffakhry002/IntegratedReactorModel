#!/usr/bin/env python3
"""
Main visualization runner for Nuclear Reactor ML results
Generates comprehensive visualization suite for model performance analysis
UPDATED: Handles energy-discretized results with appropriate folder structure
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
from visualizations_helpers.summary_statistics import create_summary_statistics_plots, create_error_distribution_for_energy, create_error_distribution_for_total
from visualizations_helpers.energy_breakdown_plots import create_energy_breakdown_plots
from visualizations_helpers.data_loader import load_test_results

def detect_energy_discretization(df):
    """Detect if the Excel file contains energy-discretized results"""
    # Check for energy-specific columns
    energy_columns = []
    for col in df.columns:
        if any(energy in col for energy in ['_thermal_', '_epithermal_', '_fast_']):
            energy_columns.append(col)

    has_energy = len(energy_columns) > 0

    if has_energy:
        print(f"\n✓ Detected energy-discretized results with {len(energy_columns)} energy-specific columns")
    else:
        print("\n✓ Detected standard flux results (no energy discretization)")

    return has_energy

def ensure_directories(base_output_dir, has_energy_discretization=False):
    """Create all necessary output directories"""
    directories = [base_output_dir]

    if has_energy_discretization:
        # Create energy-specific directories
        energy_dirs = ['thermal', 'epithermal', 'fast', 'total', 'keff']
        for energy_dir in energy_dirs:
            energy_path = os.path.join(base_output_dir, energy_dir)
            directories.append(energy_path)

            # Add subdirectories for each energy type
            if energy_dir != 'keff':  # keff doesn't need these subdirs
                directories.extend([
                    os.path.join(energy_path, 'performance_heatmaps'),
                    os.path.join(energy_path, 'spatial_error_heatmaps'),
                    os.path.join(energy_path, 'config_error_plots'),
                    os.path.join(energy_path, 'rel_error_trackers'),
                ])
            else:
                # K-eff specific directories
                directories.extend([
                    os.path.join(energy_path, 'performance_heatmaps'),
                    os.path.join(energy_path, 'config_error_plots'),
                    os.path.join(energy_path, 'rel_error_trackers'),
                ])

        # Summary statistics stays in main directory
        directories.append(os.path.join(base_output_dir, 'summary_statistics'))

        # Energy breakdown plots in main directory
        directories.append(os.path.join(base_output_dir, 'energy_breakdown'))
    else:
        # Standard directory structure
        directories.extend([
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
        ])

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
    seen_absolute_paths = set()  # Track absolute paths to avoid duplicates

    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            # Get absolute path of the directory to avoid duplicates
            abs_dir_path = os.path.abspath(dir_path)

            files = [f for f in os.listdir(dir_path) if f.endswith('.xlsx')]
            for f in files:
                full_path = os.path.join(dir_path, f)
                abs_path = os.path.abspath(full_path)

                # Only add if we haven't seen this absolute path before
                if abs_path not in seen_absolute_paths:
                    seen_absolute_paths.add(abs_path)
                    found_files.append(full_path)

    if found_files:
        print("\nFound these Excel files:")
        for i, f in enumerate(found_files, 1):
            print(f"  {i}. {os.path.basename(f)}")
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

        # Detect if we have energy-discretized results
        has_energy_discretization = detect_energy_discretization(test_results_df)

        # Get output folder name from user
        print("\n" + "-"*60)
        default_name = 'energy_visualizations' if has_energy_discretization else 'visualizations'
        output_folder_name = input(f"Output folder name (default: {default_name}): ").strip()
        if not output_folder_name:
            output_folder_name = default_name

        output_base_dir = os.path.join(script_dir, 'outputs', output_folder_name)

        # Create output directories
        ensure_directories(output_base_dir, has_energy_discretization)

        # Extract unique values for iteration
        models = test_results_df['model_class'].unique()
        encodings = test_results_df['encoding'].unique()
        optimizations = test_results_df['optimization_method'].unique()

        print(f"\nFound configurations:")
        print(f"  Models: {', '.join(models)}")
        print(f"  Encodings: {', '.join(encodings)}")
        print(f"  Optimizations: {', '.join(optimizations)}")

        # Check data availability
        print(f"\n📊 Data availability check:")

        # Check for flux data
        if has_energy_discretization:
            thermal_cols = [col for col in test_results_df.columns if '_thermal_' in col]
            epithermal_cols = [col for col in test_results_df.columns if '_epithermal_' in col]
            fast_cols = [col for col in test_results_df.columns if '_fast_' in col]
            total_cols = [col for col in test_results_df.columns if '_total_' in col]

            print(f"  ✓ Thermal flux data found ({len(thermal_cols)} columns)")
            print(f"  ✓ Epithermal flux data found ({len(epithermal_cols)} columns)")
            print(f"  ✓ Fast flux data found ({len(fast_cols)} columns)")
            print(f"  ✓ Total flux data found ({len(total_cols)} columns)")
        else:
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

    except Exception as e:
        print(f"ERROR loading test results: {e}")
        return

    # Generate visualizations
    print("\n" + "-"*60)
    print("Generating visualizations...")
    print("-"*60)

    if has_energy_discretization:
        # Generate energy-specific visualizations
        print("\n🔋 Generating energy-discretized visualizations...")

        energy_groups = ['thermal', 'epithermal', 'fast', 'total']

        for energy_group in energy_groups:
            print(f"\n{'='*50}")
            print(f"Processing {energy_group.upper()} energy group")
            print(f"{'='*50}")

            energy_output_dir = os.path.join(output_base_dir, energy_group)

            try:
                # Performance heatmaps for this energy group
                print(f"\n1. Creating {energy_group} performance heatmaps...")
                create_performance_heatmaps(
                    test_results_df,
                    os.path.join(energy_output_dir, 'performance_heatmaps'),
                    energy_group=energy_group
                )

                # Spatial error heatmaps
                print(f"\n2. Creating {energy_group} spatial error heatmaps...")
                create_spatial_error_heatmaps(
                    test_results_df,
                    os.path.join(energy_output_dir, 'spatial_error_heatmaps'),
                    models, encodings, optimizations,
                    energy_group=energy_group
                )

                # Config error plots
                print(f"\n3. Creating {energy_group} configuration error plots...")
                create_config_error_plots(
                    test_results_df,
                    os.path.join(energy_output_dir, 'config_error_plots'),
                    energy_group=energy_group
                )

                # Relative error trackers
                print(f"\n4. Creating {energy_group} relative error tracker plots...")
                create_rel_error_tracker_plots(
                    test_results_df,
                    os.path.join(energy_output_dir, 'rel_error_trackers'),
                    encodings,
                    energy_group=energy_group
                )

                # Error distribution comparison for this energy group
                print(f"\n5. Creating {energy_group} error distribution comparison...")
                create_error_distribution_for_energy(
                    test_results_df,
                    energy_output_dir,  # Save directly in energy folder
                    energy_group
                )

            except Exception as e:
                print(f"  ERROR processing {energy_group}: {e}")
                print("  Continuing with other energy groups...")

        # K-eff visualizations
        if any('keff' in col for col in test_results_df.columns):
            print(f"\n{'='*50}")
            print(f"Processing K-EFF results")
            print(f"{'='*50}")

            keff_output_dir = os.path.join(output_base_dir, 'keff')

            try:
                # K-eff specific visualizations
                print("\n1. Creating k-eff performance heatmaps...")
                create_performance_heatmaps(
                    test_results_df,
                    os.path.join(keff_output_dir, 'performance_heatmaps'),
                    target_type='keff'
                )

                print("\n2. Creating k-eff configuration error plots...")
                create_config_error_plots(
                    test_results_df,
                    os.path.join(keff_output_dir, 'config_error_plots'),
                    target_type='keff'
                )

                print("\n3. Creating k-eff relative error tracker plots...")
                create_rel_error_tracker_plots(
                    test_results_df,
                    os.path.join(keff_output_dir, 'rel_error_trackers'),
                    encodings,
                    target_type='keff'
                )
            except Exception as e:
                print(f"  ERROR processing k-eff: {e}")

        # Summary statistics (in main directory)
        print("\n6. Creating summary statistics visualizations...")
        try:
            create_summary_statistics_plots(
                test_results_df,
                os.path.join(output_base_dir, 'summary_statistics'),
                has_energy_discretization=True
            )
        except Exception as e:
            print(f"  ERROR in summary statistics: {e}")

        # Energy breakdown plots (in main directory)
        print("\n7. Creating energy breakdown plots...")
        try:
            create_energy_breakdown_plots(
                test_results_df,
                os.path.join(output_base_dir, 'energy_breakdown')
            )
        except Exception as e:
            print(f"  ERROR in energy breakdown plots: {e}")

    else:
        # Standard visualizations (existing behavior)
        try:
            # Continue with existing visualization generation code...
            # [Rest of the original visualization generation code remains the same]

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

            # 7. Error Distribution Comparison
            print("\n7. Creating error distribution comparison...")
            try:
                create_error_distribution_for_total(
                    test_results_df,
                    output_base_dir  # Save in main output directory
                )
            except Exception as e:
                print(f"  ERROR in error distribution comparison: {e}")
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
        f.write(f"Energy-discretized: {'Yes' if has_energy_discretization else 'No'}\n")

        if has_energy_discretization:
            f.write("\nEnergy Group Visualizations:\n")
            f.write("  - thermal/    : Thermal neutron flux visualizations\n")
            f.write("  - epithermal/ : Epithermal neutron flux visualizations\n")
            f.write("  - fast/       : Fast neutron flux visualizations\n")
            f.write("  - total/      : Total flux visualizations\n")
            f.write("  - keff/       : K-effective visualizations\n")
            f.write("\nSummary Visualizations (main directory):\n")
            f.write("  - summary_statistics/ : Best model combinations across all energy groups\n")
            f.write("  - energy_breakdown/   : Stacked bar charts showing energy distribution\n")
        else:
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

    if has_energy_discretization:
        print("\nEnergy-specific visualizations generated in:")
        print("  ✓ thermal/    - Thermal neutron flux analysis")
        print("  ✓ epithermal/ - Epithermal neutron flux analysis")
        print("  ✓ fast/       - Fast neutron flux analysis")
        print("  ✓ total/      - Total flux analysis")
        print("  ✓ keff/       - K-effective analysis")
        print("\nSummary visualizations in main directory:")
        print("  ✓ summary_statistics/ - Overall performance comparison")
        print("  ✓ energy_breakdown/   - Energy distribution analysis")
    else:
        print("\nVisualization categories generated:")
        print("  ✓ Performance heatmaps (R² scores)")
        print("  ✓ Spatial error heatmaps (MAPE by position)")
        print("  ✓ Feature importance plots")
        print("  ✓ Configuration error plots")
        print("  ✓ Relative error trackers")
        print("  ✓ Summary statistics")

if __name__ == "__main__":
    main()
