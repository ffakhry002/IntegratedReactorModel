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
        thermal_only = [m for m in flux_models if m.get('flux_mode') == 'thermal_only']
        epithermal_only = [m for m in flux_models if m.get('flux_mode') == 'epithermal_only']
        fast_only = [m for m in flux_models if m.get('flux_mode') == 'fast_only']

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

        if thermal_only:
            print("  Thermal Only Models:")
            for model in thermal_only:
                print(f"    {model_index}. {model['model_class']} - thermal flux only "
                      f"({model['encoding']}, {model['optimization_method']})")
                model_map[model_index] = model
                model_index += 1

        if epithermal_only:
            print("  Epithermal Only Models:")
            for model in epithermal_only:
                print(f"    {model_index}. {model['model_class']} - epithermal flux only "
                      f"({model['encoding']}, {model['optimization_method']})")
                model_map[model_index] = model
                model_index += 1

        if fast_only:
            print("  Fast Only Models:")
            for model in fast_only:
                print(f"    {model_index}. {model['model_class']} - fast flux only "
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
    selected_thermal_only = [m for m in selected_models if m['model_type'] == 'flux' and m.get('flux_mode') == 'thermal_only']
    selected_epithermal_only = [m for m in selected_models if m['model_type'] == 'flux' and m.get('flux_mode') == 'epithermal_only']
    selected_fast_only = [m for m in selected_models if m['model_type'] == 'flux' and m.get('flux_mode') == 'fast_only']
    selected_keff = [m for m in selected_models if m['model_type'] == 'keff']

    print(f"\nSelected {len(selected_models)} models for testing:")
    if selected_total_flux:
        print(f"  - {len(selected_total_flux)} total flux model(s)")
    if selected_energy_flux:
        print(f"  - {len(selected_energy_flux)} energy flux model(s)")
    if selected_bin_flux:
        print(f"  - {len(selected_bin_flux)} bin flux model(s)")
    if selected_thermal_only:
        print(f"  - {len(selected_thermal_only)} thermal only model(s)")
    if selected_epithermal_only:
        print(f"  - {len(selected_epithermal_only)} epithermal only model(s)")
    if selected_fast_only:
        print(f"  - {len(selected_fast_only)} fast only model(s)")
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

    # Run all tests at once (the test_file method handles different model types internally)
    print(f"\nTesting all {len(selected_models)} selected models...")
    tester.available_models = selected_models

    # Run comprehensive testing
    all_results = tester.test_file(test_file, training_file, show_match_details, None)

    if not all_results:
        print("No results generated. Check for errors in model loading or data files.")
        return

    # Group results by model type for separate Excel files
    results_by_type = {
        'total_flux': [],
        'energy_flux': [],
        'bin_flux': [],
        'thermal_only': [],
        'epithermal_only': [],
        'fast_only': [],
        'keff': []
    }

    for result in all_results:
        model_type = result.get('model_type', 'unknown')
        flux_mode = result.get('flux_mode', 'total')

        if model_type == 'flux':
            if flux_mode == 'total':
                results_by_type['total_flux'].append(result)
            elif flux_mode == 'energy':
                results_by_type['energy_flux'].append(result)
            elif flux_mode == 'bin':
                results_by_type['bin_flux'].append(result)
            elif flux_mode == 'thermal_only':
                results_by_type['thermal_only'].append(result)
            elif flux_mode == 'epithermal_only':
                results_by_type['epithermal_only'].append(result)
            elif flux_mode == 'fast_only':
                results_by_type['fast_only'].append(result)
        elif model_type == 'keff':
            results_by_type['keff'].append(result)

    # Create separate Excel files for each model type that has results
    if results_by_type['total_flux']:
        output_path = get_output_filename("Total Flux Models")
        reporter.create_report(results_by_type['total_flux'], output_path)
        result_files.append(output_path)
        print(f"Saved total flux results to: {os.path.basename(output_path)}")

    if results_by_type['energy_flux']:
        output_path = get_output_filename("Energy Flux Models")
        reporter.create_report(results_by_type['energy_flux'], output_path)
        result_files.append(output_path)
        print(f"Saved energy flux results to: {os.path.basename(output_path)}")

    if results_by_type['bin_flux']:
        output_path = get_output_filename("Bin Flux Models")
        reporter.create_report(results_by_type['bin_flux'], output_path)
        result_files.append(output_path)
        print(f"Saved bin flux results to: {os.path.basename(output_path)}")

    if results_by_type['thermal_only']:
        output_path = get_output_filename("Thermal Only Models")
        reporter.create_report(results_by_type['thermal_only'], output_path)
        result_files.append(output_path)
        print(f"Saved thermal only results to: {os.path.basename(output_path)}")

    if results_by_type['epithermal_only']:
        output_path = get_output_filename("Epithermal Only Models")
        reporter.create_report(results_by_type['epithermal_only'], output_path)
        result_files.append(output_path)
        print(f"Saved epithermal only results to: {os.path.basename(output_path)}")

    if results_by_type['fast_only']:
        output_path = get_output_filename("Fast Only Models")
        reporter.create_report(results_by_type['fast_only'], output_path)
        result_files.append(output_path)
        print(f"Saved fast only results to: {os.path.basename(output_path)}")

    if results_by_type['keff']:
        output_path = get_output_filename("K-eff Models")
        reporter.create_report(results_by_type['keff'], output_path)
        result_files.append(output_path)
        print(f"Saved k-eff results to: {os.path.basename(output_path)}")

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print("\nResult files created:")
    for file in result_files:
        print(f"  - {os.path.basename(file)}")
