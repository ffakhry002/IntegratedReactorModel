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
import joblib
from datetime import datetime

# Add visualization helpers to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import visualization modules
from visualizations_helpers.performance_heatmaps import create_performance_heatmaps
from visualizations_helpers.spatial_error_heatmaps import create_spatial_error_heatmaps
from visualizations_helpers.test_error_heatmaps import create_test_error_heatmaps
from visualizations_helpers.feature_importance import create_feature_importance_plots
from visualizations_helpers.config_error_plots import create_config_error_plots
from visualizations_helpers.rel_error_trackers import create_rel_error_tracker_plots
from visualizations_helpers.summary_statistics import create_summary_statistics_plots, create_error_distribution_for_energy, create_error_distribution_for_total, create_best_models_summary
from visualizations_helpers.energy_breakdown_plots import create_energy_breakdown_plots
from visualizations_helpers.data_loader import load_test_results
from visualizations_helpers.optuna_visualizations import generate_all_optuna_visualizations
from visualizations_helpers.core_config_visualizations import generate_core_config_visualizations

def detect_energy_discretization(df):
    """Detect if the Excel file contains energy-discretized results"""
    # Returns (has_energy_discretization, single_energy_mode)
    # single_energy_mode will be 'thermal_only', 'epithermal_only', 'fast_only', or None

    # Check if we have flux_mode column to determine the type of results
    if 'flux_mode' in df.columns:
        flux_modes = df['flux_mode'].unique()
        print(f"\nDetected flux modes: {list(flux_modes)}")

        # Energy discretized means we have all three energy groups
        # This happens with 'energy' or 'bin' modes
        has_energy_discretization = any(mode in ['energy', 'bin'] for mode in flux_modes)

        if has_energy_discretization:
            print(f"✓ Detected energy-discretized results (all three energy groups)")
            return True, None

        # Check for single energy group modes
        single_modes = [mode for mode in flux_modes if mode in ['thermal_only', 'epithermal_only', 'fast_only']]
        if single_modes:
            single_mode = single_modes[0]  # Take the first one if multiple
            print(f"✓ Detected single energy group results: {single_mode}")
            return False, single_mode

        # Check for total mode
        if 'total' in flux_modes:
            print(f"✓ Detected total flux results (no energy discretization)")
            return False, None

    # Fallback: Check for energy-specific columns for backward compatibility
    # Only consider it energy discretized if we have ALL THREE energy groups
    thermal_cols = [col for col in df.columns if '_thermal_' in col]
    epithermal_cols = [col for col in df.columns if '_epithermal_' in col]
    fast_cols = [col for col in df.columns if '_fast_' in col]

    has_all_three = len(thermal_cols) > 0 and len(epithermal_cols) > 0 and len(fast_cols) > 0

    if has_all_three:
        total_energy_cols = len(thermal_cols) + len(epithermal_cols) + len(fast_cols)
        print(f"\n✓ Detected energy-discretized results with all three energy groups")
        print(f"  Thermal: {len(thermal_cols)} columns, Epithermal: {len(epithermal_cols)} columns, Fast: {len(fast_cols)} columns")
        return True, None
    elif len(thermal_cols) > 0 or len(epithermal_cols) > 0 or len(fast_cols) > 0:
        # Single energy group detected
        energy_type = 'thermal' if thermal_cols else ('epithermal' if epithermal_cols else 'fast')
        print(f"\n✓ Detected single energy group results: {energy_type} flux")
        return False, f"{energy_type}_only"
    else:
        print("\n✓ Detected standard flux results (no energy discretization)")
        return False, None

def find_optuna_studies():
    """Find saved Optuna study files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Look for studies in the outputs folder
    possible_dirs = [
        os.path.join(script_dir, 'outputs', 'optuna_studies'),
        os.path.join(script_dir, '..', 'outputs', 'optuna_studies'),
        os.path.join(script_dir, '..', 'ML', 'outputs', 'optuna_studies'),
        'ML/outputs/optuna_studies',
        # Also check old locations for backward compatibility
        os.path.join(script_dir, 'hyperparameter_tuning', 'saved_studies'),
        os.path.join(script_dir, '..', 'hyperparameter_tuning', 'saved_studies'),
    ]

    study_files = {}

    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('_study.pkl')]
            for f in files:
                full_path = os.path.join(dir_path, f)
                # Keep the full key for better matching
                # Expected formats (old and new):
                # OLD: svm_flux_total_study.pkl, svm_keff_study.pkl
                # NEW: svm_flux_total_categorical_study.pkl, svm_keff_physics_study.pkl

                key = f.replace('_study.pkl', '')

                # Keep the full key with encoding information
                study_files[key] = full_path

            if files:
                break  # Found studies, no need to check other directories

    return study_files

def ensure_directories(base_output_dir, has_energy_discretization=False, single_energy_mode=None):
    """Create all necessary output directories"""
    directories = [base_output_dir]

    if has_energy_discretization:
        # For energy discretized mode, we'll create directories dynamically based on actual data
        # Just create the base directories here
        directories.extend([
            os.path.join(base_output_dir, 'energy_breakdown')
            # Removed summary_statistics and optuna_analysis - they're now created within target folders
        ])
    elif single_energy_mode:
        # Single energy mode directory structure (fast_only, thermal_only, epithermal_only)
        energy_name = single_energy_mode.replace('_only', '')
        directories.extend([
            os.path.join(base_output_dir, f'{energy_name}_flux'),
            os.path.join(base_output_dir, f'{energy_name}_flux', 'performance_heatmaps'),
            os.path.join(base_output_dir, f'{energy_name}_flux', 'spatial_error_heatmaps'),
            os.path.join(base_output_dir, f'{energy_name}_flux', 'config_error_plots'),
            os.path.join(base_output_dir, f'{energy_name}_flux', 'rel_error_trackers')
            # Removed summary_statistics and optuna_analysis - they're now created within target folders
        ])
    else:
        # Standard directory structure - check what type of data we have
        # This will be determined later based on actual data, but set up base structure
        directories.extend([
            # Removed summary_statistics and optuna_analysis - they're now created within target folders
        ])

        # We'll add specific directories later based on actual data content

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
    #TODO: add a check to see if the file is in the correct directory
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
        has_energy_discretization, single_energy_mode = detect_energy_discretization(test_results_df)

        # Get output folder name from user
        print("\n" + "-"*60)
        if has_energy_discretization:
            default_name = 'energy_visualizations'
        elif single_energy_mode:
            energy_name = single_energy_mode.replace('_only', '')
            default_name = f'{energy_name}_flux_visualizations'
        else:
            default_name = 'visualizations'
        output_folder_name = input(f"Output folder name (default: {default_name}): ").strip()
        if not output_folder_name:
            output_folder_name = default_name

        output_base_dir = os.path.join(script_dir, 'outputs', output_folder_name)

        # Create output directories
        ensure_directories(output_base_dir, has_energy_discretization, single_energy_mode)

        # Generate core configuration visualizations first
        print("\n" + "-"*60)
        print("Generating Core Configuration Visualizations...")
        print("-"*60)

        try:
            generate_core_config_visualizations(output_base_dir)
        except Exception as e:
            print(f"ERROR generating core configuration visualizations: {e}")
            print("Continuing with other visualizations...")

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
        elif single_energy_mode:
            energy_name = single_energy_mode.replace('_only', '')
            energy_cols = [col for col in test_results_df.columns if f'_{energy_name}_' in col]
            print(f"  ✓ {energy_name.title()} flux data found ({len(energy_cols)} columns)")
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

        # Determine which energy groups we actually have data for
        energy_groups = []
        if any('_thermal_' in col for col in test_results_df.columns):
            energy_groups.append('thermal')
        if any('_epithermal_' in col for col in test_results_df.columns):
            energy_groups.append('epithermal')
        if any('_fast_' in col for col in test_results_df.columns):
            energy_groups.append('fast')
        # Don't add 'total' for multi-energy unless explicitly present
        if any('_total_' in col for col in test_results_df.columns):
            energy_groups.append('total')

        for energy_group in energy_groups:
            print(f"\n{'='*50}")
            print(f"Processing {energy_group.upper()} energy group")
            print(f"{'='*50}")

            energy_output_dir = os.path.join(output_base_dir, energy_group)

            # Create directories for this energy group
            os.makedirs(energy_output_dir, exist_ok=True)
            os.makedirs(os.path.join(energy_output_dir, 'performance_heatmaps'), exist_ok=True)
            os.makedirs(os.path.join(energy_output_dir, 'spatial_error_heatmaps'), exist_ok=True)
            os.makedirs(os.path.join(energy_output_dir, 'config_error_plots'), exist_ok=True)
            os.makedirs(os.path.join(energy_output_dir, 'rel_error_trackers'), exist_ok=True)
            os.makedirs(os.path.join(energy_output_dir, 'feature_importance'), exist_ok=True)
            os.makedirs(os.path.join(energy_output_dir, 'summary_statistics'), exist_ok=True)

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

                # Test error heatmaps
                print(f"\n3. Creating {energy_group} test error heatmaps...")
                try:
                    create_test_error_heatmaps(
                        test_results_df,
                        energy_output_dir,
                        models, encodings, optimizations,
                        energy_group=energy_group
                    )
                except Exception as e:
                    print(f"  ERROR in test error heatmaps: {e}")
                    print("  Continuing with other visualizations...")

                # Config error plots
                print(f"\n4. Creating {energy_group} configuration error plots...")
                create_config_error_plots(
                    test_results_df,
                    os.path.join(energy_output_dir, 'config_error_plots'),
                    energy_group=energy_group
                )

                # Relative error trackers
                print(f"\n5. Creating {energy_group} relative error tracker plots...")
                create_rel_error_tracker_plots(
                    test_results_df,
                    os.path.join(energy_output_dir, 'rel_error_trackers'),
                    encodings,
                    energy_group=energy_group
                )

                # 6. Feature importance plots for this energy group
                print(f"\n6. Creating {energy_group} feature importance plots...")
                try:
                    create_feature_importance_plots(
                        test_results_df,
                        os.path.join(energy_output_dir, 'feature_importance'),
                        models,
                        target_type='flux'  # Since we're dealing with energy groups, it's flux
                    )
                except Exception as e:
                    print(f"  ERROR in {energy_group} feature importance: {e}")
                    print("  Continuing with other visualizations...")

                # 7. Summary Statistics - Only best models summary and error distribution for individual energy groups
                print(f"\n7. Creating {energy_group} summary statistics...")
                try:
                    # Only create the best models summary for this specific energy group
                    create_best_models_summary(
                        test_results_df,
                        os.path.join(energy_output_dir, 'summary_statistics'),
                        has_energy_discretization=True,
                        target_context=energy_group
                    )
                except Exception as e:
                    print(f"  ERROR in {energy_group} best models summary: {e}")
                    print("  Continuing with other visualizations...")

                # 8. Error distribution comparison (within summary statistics folder)
                print(f"\n8. Creating {energy_group} error distribution...")
                try:
                    create_error_distribution_for_energy(
                        test_results_df,
                        os.path.join(energy_output_dir, 'summary_statistics'),  # Put in summary_statistics folder
                        energy_group
                    )
                except Exception as e:
                    print(f"  ERROR in {energy_group} error distribution: {e}")
                    print("  Continuing with other visualizations...")

            except Exception as e:
                print(f"  ERROR processing {energy_group}: {e}")
                print("  Continuing with other energy groups...")

        # K-eff visualizations - only if there's actual keff data
        has_keff_data = False
        keff_cols = [col for col in test_results_df.columns if 'keff' in col.lower()]
        if keff_cols:
            # Check if any keff columns have non-null, non-N/A values
            for col in keff_cols:
                if not test_results_df[col].isna().all() and (test_results_df[col] != 'N/A').any():
                    has_keff_data = True
                    break

        if has_keff_data:
            print(f"\n{'='*50}")
            print(f"Processing K-EFF results")
            print(f"{'='*50}")

            keff_output_dir = os.path.join(output_base_dir, 'keff')

            # Create k-eff directories
            os.makedirs(keff_output_dir, exist_ok=True)
            os.makedirs(os.path.join(keff_output_dir, 'performance_heatmaps'), exist_ok=True)
            os.makedirs(os.path.join(keff_output_dir, 'config_error_plots'), exist_ok=True)
            os.makedirs(os.path.join(keff_output_dir, 'rel_error_trackers'), exist_ok=True)
            os.makedirs(os.path.join(keff_output_dir, 'summary_statistics'), exist_ok=True)

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

                # 4. Summary Statistics (within k-eff folder)
                print("\n4. Creating k-eff summary statistics...")
                try:
                    create_summary_statistics_plots(
                        test_results_df,
                        os.path.join(keff_output_dir, 'summary_statistics'),
                        has_energy_discretization=True
                    )
                except Exception as e:
                    print(f"  ERROR in k-eff summary statistics: {e}")
                    print("  Continuing with other visualizations...")
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

    elif single_energy_mode:
        # Single energy group visualizations
        energy_name = single_energy_mode.replace('_only', '')
        print(f"\n🔋 Generating {energy_name} flux visualizations...")

        energy_output_dir = os.path.join(output_base_dir, f'{energy_name}_flux')

        try:
            # Performance heatmaps for this energy group
            print(f"\n1. Creating {energy_name} flux performance heatmaps...")
            create_performance_heatmaps(
                test_results_df,
                os.path.join(energy_output_dir, 'performance_heatmaps'),
                energy_group=energy_name
            )

            # Spatial error heatmaps
            print(f"\n2. Creating {energy_name} flux spatial error heatmaps...")
            create_spatial_error_heatmaps(
                test_results_df,
                os.path.join(energy_output_dir, 'spatial_error_heatmaps'),
                models, encodings, optimizations,
                energy_group=energy_name
            )

            # Test error heatmaps
            print(f"\n3. Creating {energy_name} flux test error heatmaps...")
            try:
                create_test_error_heatmaps(
                    test_results_df,
                    energy_output_dir,
                    models, encodings, optimizations,
                    energy_group=energy_name
                )
            except Exception as e:
                print(f"  ERROR in test error heatmaps: {e}")
                print("  Continuing with other visualizations...")

            # Config error plots
            print(f"\n4. Creating {energy_name} flux configuration error plots...")
            create_config_error_plots(
                test_results_df,
                os.path.join(energy_output_dir, 'config_error_plots'),
                energy_group=energy_name
            )

            # Relative error trackers
            print(f"\n5. Creating {energy_name} flux relative error tracker plots...")
            create_rel_error_tracker_plots(
                test_results_df,
                os.path.join(energy_output_dir, 'rel_error_trackers'),
                encodings,
                energy_group=energy_name
            )

            # Summary statistics (in main directory)
            print(f"\n6. Creating {energy_name} flux summary statistics...")
            create_summary_statistics_plots(
                test_results_df,
                os.path.join(output_base_dir, 'summary_statistics'),
                has_energy_discretization=False,
                single_energy_mode=single_energy_mode
            )

        except Exception as e:
            print(f"  ERROR processing {energy_name} flux: {e}")

    else:
        # Standard visualizations - need to check what data types we have
        print("\n🔍 Detecting data types in standard mode...")

        # Check if we have total flux data
        has_total_flux = any(col.startswith('I_') and col.endswith('_predicted') and
                            not any(energy in col for energy in ['_thermal_', '_epithermal_', '_fast_'])
                            for col in test_results_df.columns)

        # Check if we have k-eff data with actual values (not just columns)
        has_keff = False
        keff_cols = [col for col in test_results_df.columns if 'keff' in col.lower()]
        if keff_cols:
            # Check if any keff columns have non-null, non-N/A values
            for col in keff_cols:
                if not test_results_df[col].isna().all() and (test_results_df[col] != 'N/A').any():
                    has_keff = True
                    break

        print(f"  Total flux data: {'Yes' if has_total_flux else 'No'}")
        print(f"  K-eff data: {'Yes' if has_keff else 'No'}")

        # Create subdirectories based on data types - only create what's needed

        try:
            # Process total flux visualizations
            if has_total_flux:
                print(f"\n{'='*50}")
                print(f"Processing TOTAL FLUX results")
                print(f"{'='*50}")

                flux_output_dir = os.path.join(output_base_dir, 'total_flux')
                # Create directories as needed
                os.makedirs(flux_output_dir, exist_ok=True)
                os.makedirs(os.path.join(flux_output_dir, 'performance_heatmaps'), exist_ok=True)
                os.makedirs(os.path.join(flux_output_dir, 'spatial_error_heatmaps'), exist_ok=True)
                os.makedirs(os.path.join(flux_output_dir, 'config_error_plots'), exist_ok=True)
                os.makedirs(os.path.join(flux_output_dir, 'rel_error_trackers'), exist_ok=True)
                os.makedirs(os.path.join(flux_output_dir, 'feature_importance'), exist_ok=True)
                os.makedirs(os.path.join(flux_output_dir, 'summary_statistics'), exist_ok=True)

                # 1. Performance Heatmaps (R²)
                print("\n1. Creating total flux performance heatmaps...")
                try:
                    create_performance_heatmaps(
                        test_results_df,
                        os.path.join(flux_output_dir, 'performance_heatmaps'),
                        target_type='flux'
                    )
                except Exception as e:
                    print(f"  ERROR in performance heatmaps: {e}")
                    print("  Continuing with other visualizations...")

                # 2. Spatial Error Heatmaps
                print("\n2. Creating total flux spatial error heatmaps...")
                try:
                    create_spatial_error_heatmaps(
                        test_results_df,
                        os.path.join(flux_output_dir, 'spatial_error_heatmaps'),
                        models, encodings, optimizations
                    )
                except Exception as e:
                    print(f"  ERROR in spatial error heatmaps: {e}")
                    print("  Continuing with other visualizations...")

                # 3. Test Error Heatmaps
                print("\n3. Creating total flux test error heatmaps...")
                try:
                    create_test_error_heatmaps(
                        test_results_df,
                        flux_output_dir,
                        models, encodings, optimizations
                    )
                except Exception as e:
                    print(f"  ERROR in test error heatmaps: {e}")
                    print("  Continuing with other visualizations...")

                # 4. Feature Importance Plots
                print("\n4. Creating total flux feature importance plots...")
                try:
                    create_feature_importance_plots(
                        test_results_df,
                        os.path.join(flux_output_dir, 'feature_importance'),
                        models,
                        target_type='flux'
                    )
                except Exception as e:
                    print(f"  ERROR in feature importance plots: {e}")
                    print("  Continuing with other visualizations...")

                # 5. Config Error Plots
                print("\n5. Creating total flux configuration error plots...")
                try:
                    create_config_error_plots(
                        test_results_df,
                        os.path.join(flux_output_dir, 'config_error_plots'),
                        target_type='flux'
                    )
                except Exception as e:
                    print(f"  ERROR in config error plots: {e}")
                    print("  Continuing with other visualizations...")

                # 6. Relative Error Tracker Plots
                print("\n6. Creating total flux relative error tracker plots...")
                try:
                    create_rel_error_tracker_plots(
                        test_results_df,
                        os.path.join(flux_output_dir, 'rel_error_trackers'),
                        encodings,
                        target_type='flux'
                    )
                except Exception as e:
                    print(f"  ERROR in relative error tracker plots: {e}")
                    print("  Continuing with other visualizations...")

                # 7. Summary Statistics (within total flux folder)
                print("\n7. Creating total flux summary statistics...")
                try:
                    create_summary_statistics_plots(
                        test_results_df,
                        os.path.join(flux_output_dir, 'summary_statistics')
                    )
                except Exception as e:
                    print(f"  ERROR in flux summary statistics: {e}")
                    print("  Continuing with other visualizations...")

                # 8. Error Distribution (within summary statistics folder)
                print("\n8. Creating total flux error distribution...")
                try:
                    create_error_distribution_for_total(
                        test_results_df,
                        os.path.join(flux_output_dir, 'summary_statistics')  # Put in summary_statistics folder
                    )
                except Exception as e:
                    print(f"  ERROR in flux error distribution: {e}")
                    print("  Continuing with other visualizations...")

            # Process k-eff visualizations
            if has_keff:
                print(f"\n{'='*50}")
                print(f"Processing K-EFF results")
                print(f"{'='*50}")

                keff_output_dir = os.path.join(output_base_dir, 'keff')
                # Create directories as needed
                os.makedirs(keff_output_dir, exist_ok=True)
                os.makedirs(os.path.join(keff_output_dir, 'performance_heatmaps'), exist_ok=True)
                os.makedirs(os.path.join(keff_output_dir, 'config_error_plots'), exist_ok=True)
                os.makedirs(os.path.join(keff_output_dir, 'rel_error_trackers'), exist_ok=True)
                os.makedirs(os.path.join(keff_output_dir, 'feature_importance'), exist_ok=True)
                os.makedirs(os.path.join(keff_output_dir, 'summary_statistics'), exist_ok=True)

                # K-eff specific visualizations
                print("\n1. Creating k-eff performance heatmaps...")
                try:
                    create_performance_heatmaps(
                        test_results_df,
                        os.path.join(keff_output_dir, 'performance_heatmaps'),
                        target_type='keff'
                    )
                except Exception as e:
                    print(f"  ERROR in k-eff performance heatmaps: {e}")

                print("\n2. Creating k-eff feature importance plots...")
                try:
                    create_feature_importance_plots(
                        test_results_df,
                        os.path.join(keff_output_dir, 'feature_importance'),
                        models,
                        target_type='keff'
                    )
                except Exception as e:
                    print(f"  ERROR in k-eff feature importance plots: {e}")

                print("\n3. Creating k-eff configuration error plots...")
                try:
                    create_config_error_plots(
                        test_results_df,
                        os.path.join(keff_output_dir, 'config_error_plots'),
                        target_type='keff'
                    )
                except Exception as e:
                    print(f"  ERROR in k-eff config error plots: {e}")

                print("\n4. Creating k-eff relative error tracker plots...")
                try:
                    create_rel_error_tracker_plots(
                        test_results_df,
                        os.path.join(keff_output_dir, 'rel_error_trackers'),
                        encodings,
                        target_type='keff'
                    )
                except Exception as e:
                    print(f"  ERROR in k-eff relative error trackers: {e}")

                # 5. Summary Statistics (within k-eff folder)
                print("\n5. Creating k-eff summary statistics...")
                try:
                    create_summary_statistics_plots(
                        test_results_df,
                        os.path.join(keff_output_dir, 'summary_statistics')
                    )
                except Exception as e:
                    print(f"  ERROR in k-eff summary statistics: {e}")
                    print("  Continuing with other visualizations...")

            # Summary statistics and error distributions are now created within their respective target folders

        except Exception as e:
            print(f"\nCRITICAL ERROR during visualization generation: {e}")
            import traceback
            traceback.print_exc()
            return

    # Generate Optuna visualizations if studies exist
    print("\n" + "-"*60)
    print("Checking for Optuna optimization studies...")
    print("-"*60)

    study_files = find_optuna_studies()
    relevant_studies = {}  # Initialize for later use

    if study_files:
        print(f"\nFound {len(study_files)} Optuna study files:")

        # Determine which models are in the Excel file
        models_in_excel = set(test_results_df['model_class'].unique())

        # Determine which flux modes are present
        flux_modes_in_excel = set()
        if 'flux_mode' in test_results_df.columns:
            flux_modes_in_excel = set(test_results_df['flux_mode'].unique())

            # CRITICAL FIX: For merged multi-energy data, also include individual energy modes
            # If we have 'energy' mode and individual energy columns, include individual modes
            if 'energy' in flux_modes_in_excel or 'bin' in flux_modes_in_excel:
                if any('_thermal_' in col for col in test_results_df.columns):
                    flux_modes_in_excel.add('thermal_only')
                if any('_epithermal_' in col for col in test_results_df.columns):
                    flux_modes_in_excel.add('epithermal_only')
                if any('_fast_' in col for col in test_results_df.columns):
                    flux_modes_in_excel.add('fast_only')
                print(f"  🔧 Multi-energy data detected - added individual energy modes to search")
        else:
            # Infer from columns
            if has_energy_discretization:
                # For energy discretized data, include both multi-energy modes AND individual energy modes
                flux_modes_in_excel = {'energy', 'bin'}
                # ALSO include individual energy modes if we have their data
                if any('_thermal_' in col for col in test_results_df.columns):
                    flux_modes_in_excel.add('thermal_only')
                if any('_epithermal_' in col for col in test_results_df.columns):
                    flux_modes_in_excel.add('epithermal_only')
                if any('_fast_' in col for col in test_results_df.columns):
                    flux_modes_in_excel.add('fast_only')
            elif single_energy_mode:
                flux_modes_in_excel = {single_energy_mode}
            elif any(col.startswith('I_') and col.endswith('_predicted') and
                    not any(energy in col for energy in ['_thermal_', '_epithermal_', '_fast_'])
                    for col in test_results_df.columns):
                flux_modes_in_excel = {'total'}

        # Check if k-eff is present - need to check for actual k-eff data, not just column names
        has_keff_data = False
        keff_cols = [col for col in test_results_df.columns if 'keff' in col.lower()]
        if keff_cols:
            # Check if any keff columns have non-null, non-N/A values
            for col in keff_cols:
                if not test_results_df[col].isna().all() and (test_results_df[col] != 'N/A').any():
                    has_keff_data = True
                    break

        print(f"\nModels in Excel: {models_in_excel}")
        print(f"Flux modes in Excel: {flux_modes_in_excel}")
        print(f"Has k-eff data: {has_keff_data}")

        # Filter study files to only include those present in Excel
        relevant_studies = {}
        for key, path in study_files.items():
            # Parse the full key to check relevance
            # Keys are like: svm_flux_total_categorical, svm_keff_physics, random_forest_flux_thermal_only_physics

            # Handle model names that contain underscores (like random_forest)
            known_models = ['svm', 'xgboost', 'random_forest']
            model_name = None
            target_type = None

            for known_model in known_models:
                if key.startswith(known_model + '_'):
                    model_name = known_model
                    remainder = key[len(known_model) + 1:]  # Remove model name and underscore
                    parts = remainder.split('_')
                    if len(parts) >= 1:
                        target_type = parts[0]
                    break

            if model_name is None or target_type is None:
                print(f"  Warning: Could not parse study key: {key}")
                continue

            # Check if this model is in the Excel
            if model_name not in models_in_excel:
                continue

            # Check if this target type is relevant
            if target_type == 'keff':
                # Only include k-eff studies if we actually have k-eff data
                if has_keff_data:
                    relevant_studies[key] = path
                else:
                    print(f"  Skipping {key} - no k-eff data in Excel file")
            elif target_type == 'flux':
                # Determine the flux mode from the remainder after removing model name
                remainder = key[len(model_name) + 1:]  # Remove model name and underscore
                parts = remainder.split('_')

                if len(parts) >= 3 and parts[-1] in ['categorical', 'physics', 'one_hot', 'spatial', 'graph']:
                    # Has encoding at the end: flux_total_categorical, flux_thermal_only_physics
                    if len(parts) == 3:
                        # flux_total_encoding
                        flux_mode = parts[1]
                    elif len(parts) == 4 and parts[2] == 'only':
                        # flux_thermal_only_encoding
                        flux_mode = f"{parts[1]}_only"
                    else:
                        flux_mode = 'total'  # Default
                elif len(parts) == 2:
                    # Old format: flux_total, flux_energy, flux_bin
                    flux_mode = parts[1]
                elif len(parts) == 3 and parts[2] == 'only':
                    # Old format: flux_thermal_only
                    flux_mode = f"{parts[1]}_only"
                else:
                    flux_mode = 'total'  # Default

                # Check if this flux mode is in the Excel
                if flux_mode in flux_modes_in_excel:
                    relevant_studies[key] = path

        if relevant_studies:
            print(f"\nRelevant Optuna studies for this Excel: {len(relevant_studies)}")
            for key, path in relevant_studies.items():
                print(f"  - {key}: {os.path.basename(path)}")

            print(f"\nGenerating Optuna visualizations...")

            for key, study_path in relevant_studies.items():
                try:
                    print(f"\n📊 Processing {key}...")

                    # Load the study
                    study = joblib.load(study_path)

                    # Use the key to determine proper target naming and directory
                    # Keys are like: svm_flux_total, svm_flux_thermal_only, svm_keff, random_forest_flux_thermal_only

                    # Parse model name and target using the same logic as filtering
                    known_models = ['svm', 'xgboost', 'random_forest']
                    model_name = None
                    target_remainder = None

                    for known_model in known_models:
                        if key.startswith(known_model + '_'):
                            model_name = known_model
                            target_remainder = key[len(known_model) + 1:]  # Remove model name and underscore
                            break

                    if model_name is None or target_remainder is None:
                        print(f"  Warning: Could not parse study key for visualization: {key}")
                        continue

                    # Parse target from remainder and determine encoding
                    parts = target_remainder.split('_')
                    encoding = None

                    if parts[0] == 'keff':
                        target = 'keff'
                        # For keff: keff_encoding
                        if len(parts) >= 2:
                            encoding = parts[1]
                        # Put optuna analysis in keff directory
                        target_base_dir = os.path.join(output_base_dir, 'keff')
                    elif parts[0] == 'flux':
                        if len(parts) == 3:
                            # flux_total_encoding, flux_energy_encoding, flux_bin_encoding
                            flux_mode = parts[1]
                            encoding = parts[2]
                            target = f"flux_{flux_mode}"

                            # Determine target directory based on flux mode
                            if flux_mode == 'total':
                                target_base_dir = os.path.join(output_base_dir, 'total_flux')
                            elif flux_mode in ['thermal', 'epithermal', 'fast']:
                                target_base_dir = os.path.join(output_base_dir, flux_mode)
                            elif flux_mode in ['energy', 'bin']:
                                # For energy/bin modes, put in total_flux directory
                                target_base_dir = os.path.join(output_base_dir, 'total_flux')
                            else:
                                target_base_dir = os.path.join(output_base_dir, 'total_flux')
                        elif len(parts) == 4 and parts[2] == 'only':
                            # flux_thermal_only_encoding -> flux_thermal
                            energy_name = parts[1]
                            encoding = parts[3]
                            target = f"flux_{energy_name}"
                            target_base_dir = os.path.join(output_base_dir, energy_name)
                        elif len(parts) == 2:
                            # flux_total, flux_energy, flux_bin (no encoding)
                            flux_mode = parts[1]
                            target = f"flux_{flux_mode}"

                            # Determine target directory based on flux mode
                            if flux_mode == 'total':
                                target_base_dir = os.path.join(output_base_dir, 'total_flux')
                            elif flux_mode in ['thermal', 'epithermal', 'fast']:
                                target_base_dir = os.path.join(output_base_dir, flux_mode)
                            elif flux_mode in ['energy', 'bin']:
                                target_base_dir = os.path.join(output_base_dir, 'total_flux')
                            else:
                                target_base_dir = os.path.join(output_base_dir, 'total_flux')
                        elif len(parts) == 3 and parts[2] == 'only':
                            # flux_thermal_only -> flux_thermal (no encoding)
                            energy_name = parts[1]
                            target = f"flux_{energy_name}"
                            target_base_dir = os.path.join(output_base_dir, energy_name)
                        else:
                            target = 'flux'
                            target_base_dir = os.path.join(output_base_dir, 'total_flux')
                    else:
                        target = 'unknown'
                        target_base_dir = os.path.join(output_base_dir, 'total_flux')

                    # Ensure the target directory exists
                    os.makedirs(target_base_dir, exist_ok=True)

                    # Generate visualizations with target-specific base directory
                    generate_all_optuna_visualizations(
                        study=study,
                        save_base_dir=target_base_dir,
                        model_name=model_name,
                        target=target,
                        encoding=encoding,
                        include_all=True
                    )

                    print(f"  ✓ Completed visualizations for {key}")

                except Exception as e:
                    print(f"  ✗ ERROR processing {key}: {e}")
                    import traceback
                    print("    Traceback:")
                    traceback.print_exc()
                    print("    Continuing with other studies...")
        else:
            print("\nNo relevant Optuna studies found for the models in this Excel file.")
    else:
        print("\nNo Optuna study files found.")
        print("To generate Optuna visualizations, ensure your optimization studies are saved")
        print("in ML/outputs/optuna_studies/")

    print("="*80)
    print(f"All visualizations saved to: {output_base_dir}")

if __name__ == "__main__":
    main()
