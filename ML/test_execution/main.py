"""
Main execution logic for model testing
UPDATED: Show flux modes, allow model selection, test in order (total->energy->bin)
"""

import os
from datetime import datetime
from .model_tester import ReactorModelTester
from .excel_reporter import ExcelReporter
import pandas as pd


def run_testing(outputs_dir=None):
    """Main function to run the testing pipeline"""
    print("\n" + "="*60)
    print("NUCLEAR REACTOR MODEL TESTING SYSTEM")
    print("="*60)

    # Set default outputs directory
    if outputs_dir is None:
        outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")

    models_dir = os.path.join(outputs_dir, "models")
    excel_reports_dir = os.path.join(outputs_dir, "excel_reports")

    # Initialize tester with models directory
    tester = ReactorModelTester(models_dir=models_dir)

    # Find available models
    print("\nSearching for trained models...")
    models = tester.find_available_models()

    if not models:
        print(f"No trained models found in '{models_dir}' directory.")
        print("Please train models first using main.py")
        return

    # Group models by type for better display
    flux_models = [m for m in models if m['model_type'] == 'flux']
    keff_models = [m for m in models if m['model_type'] == 'keff']

    print(f"\nFound {len(models)} trained models:")

    # Display flux models by flux mode
    if flux_models:
        print("\nFLUX MODELS:")
        total_flux = [m for m in flux_models if m.get('flux_mode', 'total') == 'total']
        energy_flux = [m for m in flux_models if m.get('flux_mode') == 'energy']
        bin_flux = [m for m in flux_models if m.get('flux_mode') == 'bin']

        model_index = 1
        model_map = {}

        if total_flux:
            print("  Total Flux Models:")
            for model in total_flux:
                print(f"    {model_index}. {model['model_class']} - total flux "
                      f"({model['encoding']}, {model['optimization_method']})")
                model_map[model_index] = model
                model_index += 1

        if energy_flux:
            print("  Energy Flux Models:")
            for model in energy_flux:
                print(f"    {model_index}. {model['model_class']} - energy flux "
                      f"({model['encoding']}, {model['optimization_method']})")
                model_map[model_index] = model
                model_index += 1

        if bin_flux:
            print("  Bin Flux Models:")
            for model in bin_flux:
                print(f"    {model_index}. {model['model_class']} - bin flux "
                      f"({model['encoding']}, {model['optimization_method']})")
                model_map[model_index] = model
                model_index += 1

    if keff_models:
        print("\nK-EFF MODELS:")
        if 'model_index' not in locals():
            model_index = 1
            model_map = {}
        for model in keff_models:
            print(f"    {model_index}. {model['model_class']} - k-eff "
                  f"({model['encoding']}, {model['optimization_method']})")
            model_map[model_index] = model
            model_index += 1

    # Get user selection
    print("\n" + "-"*40)
    selection = input("Select models to test (e.g., 1,2,3 or 'all', default: all): ").strip()

    if selection.lower() == 'all' or selection == '':
        selected_models = models
    else:
        try:
            indices = [int(x.strip()) for x in selection.split(',')]
            selected_models = [model_map[i] for i in indices if i in model_map]
            if not selected_models:
                print("No valid models selected.")
                return
        except:
            print("Invalid selection format. Please use comma-separated numbers.")
            return

    # Group selected models by type
    selected_total_flux = [m for m in selected_models if m['model_type'] == 'flux' and m.get('flux_mode', 'total') == 'total']
    selected_energy_flux = [m for m in selected_models if m['model_type'] == 'flux' and m.get('flux_mode') == 'energy']
    selected_bin_flux = [m for m in selected_models if m['model_type'] == 'flux' and m.get('flux_mode') == 'bin']
    selected_keff = [m for m in selected_models if m['model_type'] == 'keff']

    print(f"\nSelected {len(selected_models)} models for testing:")
    if selected_total_flux:
        print(f"  - {len(selected_total_flux)} total flux model(s)")
    if selected_energy_flux:
        print(f"  - {len(selected_energy_flux)} energy flux model(s)")
    if selected_bin_flux:
        print(f"  - {len(selected_bin_flux)} bin flux model(s)")
    if selected_keff:
        print(f"  - {len(selected_keff)} k-eff model(s)")

    # Get test file
    print("\n" + "-"*40)
    test_file = input("Enter path to test file (default: ML/data/test.txt): ").strip()
    if not test_file:
        test_file = "ML/data/test.txt"

    if not os.path.exists(test_file):
        print(f"Error: Test file '{test_file}' not found.")
        return

    # Get training file for comparison
    training_file = input("Enter path to training file (default: ML/data/train.txt): ").strip()
    if not training_file:
        training_file = "ML/data/train.txt"

    # Ask if user wants detailed matching information
    show_details = input("\nShow detailed matching information? (y/n, default: n): ").strip().lower()
    show_match_details = show_details == 'y'

    # Initialize reporter
    reporter = ExcelReporter()

    # Store all result files for summary at the end
    result_files = []

    # Function to get output filename
    def get_output_filename(model_type_desc):
        print("\n" + "-"*40)
        print(f"Results for: {model_type_desc}")
        output_file = input("Output Excel filename (press Enter for timestamp): ").strip()
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{model_type_desc.lower().replace(' ', '_')}_{timestamp}.xlsx"
        elif not output_file.endswith('.xlsx'):
            output_file += '.xlsx'
        return os.path.join(excel_reports_dir, output_file)

    # Test and save results for each model type
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60)

    # Test total flux models
    if selected_total_flux:
        print(f"\nTesting {len(selected_total_flux)} total flux model(s)...")
        tester.available_models = selected_total_flux

        # Use a modified test_file that doesn't save intermediate results yet
        total_flux_results = tester.test_file(test_file, training_file, show_match_details, None)

        # Get output filename and save
        output_path = get_output_filename("Total Flux Models")
        reporter.create_report(total_flux_results, output_path)
        result_files.append(output_path)

        print(f"Saved total flux results to: {os.path.basename(output_path)}")

    # Test energy flux models
    if selected_energy_flux:
        print(f"\nTesting {len(selected_energy_flux)} energy flux model(s)...")
        tester.available_models = selected_energy_flux

        energy_flux_results = tester.test_file(test_file, training_file, show_match_details, None)

        # Get output filename and save
        output_path = get_output_filename("Energy Flux Models")
        reporter.create_report(energy_flux_results, output_path)
        result_files.append(output_path)

        print(f"Saved energy flux results to: {os.path.basename(output_path)}")

    # Test bin flux models
    if selected_bin_flux:
        print(f"\nTesting {len(selected_bin_flux)} bin flux model(s)...")

        tester.available_models = selected_bin_flux
        bin_flux_results = tester.test_file(test_file, training_file, show_match_details, None)

        # Get output filename and save
        output_path = get_output_filename("Bin Flux Models")
        reporter.create_report(bin_flux_results, output_path)
        result_files.append(output_path)

        print(f"Saved bin flux results to: {os.path.basename(output_path)}")

    # Test k-eff models
    if selected_keff:
        print(f"\nTesting {len(selected_keff)} k-eff model(s)...")
        tester.available_models = selected_keff

        keff_results = tester.test_file(test_file, training_file, show_match_details, None)

        # Get output filename and save
        output_path = get_output_filename("K-eff Models")
        reporter.create_report(keff_results, output_path)
        result_files.append(output_path)

        print(f"Saved k-eff results to: {os.path.basename(output_path)}")

    # Optional: Create combined results if multiple types were tested
    if len(result_files) > 1:
        print("\n" + "-"*40)
        combine = input("Create combined results file? (y/n, default: n): ").strip().lower()
        if combine == 'y':
            # Reload all results and combine
            all_results = []
            for file in result_files:
                df = pd.read_excel(file, sheet_name='Test Results')
                # Convert DataFrame back to list of dicts
                results = df.to_dict('records')
                all_results.extend(results)

            # Get output filename for combined results
            output_path = get_output_filename("Combined Results")
            reporter.create_report(all_results, output_path)
            print(f"Saved combined results to: {os.path.basename(output_path)}")

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print("\nResult files created:")
    for file in result_files:
        print(f"  - {os.path.basename(file)}")
