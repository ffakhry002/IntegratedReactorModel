"""
Main execution logic for model testing
"""

import os
from datetime import datetime
from .model_tester import ReactorModelTester
from .excel_reporter import ExcelReporter


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

    print(f"\nFound {len(models)} trained models:")
    for i, model in enumerate(models):
        print(f"  {i+1}. {model['model_class']} - {model['model_type']} "
              f"({model['encoding']}, {model['optimization_method']})")

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

    # Run tests
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60)

    results = tester.test_file(test_file, training_file, show_match_details)

    # Create Excel report
    print("\n" + "-"*40)
    output_file = input("Output Excel filename (press Enter for timestamp): ").strip()
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"test_results_{timestamp}.xlsx"
    elif not output_file.endswith('.xlsx'):
        output_file += '.xlsx'

    # Full path for Excel file
    output_path = os.path.join(excel_reports_dir, output_file)

    # Generate report
    reporter = ExcelReporter()
    reporter.create_report(results, output_path)

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
