"""
Nuclear Reactor ML Training System
Main entry point for interactive model training
"""

import sys
import os
import logging
from datetime import datetime
import traceback
from utils.txt_to_data import create_reactor_data_excel
from utils.log_structure import setup_logging, close_logging

# Add the ML directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from execution.interactive_menu import InteractiveTrainer

def main():
    """Main entry point for the Nuclear Reactor ML Training System.

    Sets up logging, creates directory structure, processes data files,
    and launches the interactive training interface.
    """
    # Get the directory where main.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create outputs directory structure in the same folder as main.py
    outputs_dir = os.path.join(script_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "excel_reports"), exist_ok=True)

    # Setup logging
    log_filepath, logger = setup_logging(outputs_dir)

    try:
        print("\n" + "="*60)
        print("NUCLEAR REACTOR ML TRAINING SYSTEM")
        print("="*60)
        print("Version: 1.0")
        print("Parallel Computing: Enabled")
        print(f"Python Version: {sys.version}")
        print(f"Working Directory: {os.getcwd()}")
        print("="*60 + "\n")

        # Log system information
        try:
            import platform
            print("System Information:")
            print(f"  OS: {platform.system()} {platform.release()}")
            print(f"  Machine: {platform.machine()}")
            print(f"  Processor: {platform.processor()}")

            # Log installed package versions
            print("\nKey Package Versions:")
            packages = ['numpy', 'pandas', 'sklearn', 'xgboost', 'optuna', 'joblib']
            for pkg in packages:
                try:
                    module = __import__(pkg)
                    version = getattr(module, '__version__', 'unknown')
                    print(f"  {pkg}: {version}")
                except:
                    print(f"  {pkg}: not installed")
            print()
        except Exception as e:
            print(f"Could not log system info: {e}")

        # Process test data if exists
        if os.path.exists(os.path.join(script_dir, "data", "test.txt")):
            test_file_path = os.path.join(script_dir, "data", "test.txt")
            test_excel_path = create_reactor_data_excel(
                data_file_path=test_file_path,
                output_prefix="test_data_",
                output_dir=os.path.join(outputs_dir, "excel_reports")
            )
            print(f"Test data Excel: {test_excel_path}")

        # Process training data if exists
        if os.path.exists(os.path.join(script_dir, "data", "train.txt")):
            train_file_path = os.path.join(script_dir, "data", "train.txt")
            train_excel_path = create_reactor_data_excel(
                data_file_path=train_file_path,
                output_prefix="train_data_",
                output_dir=os.path.join(outputs_dir, "excel_reports")
            )
            print(f"Train data Excel: {train_excel_path}")

        # Create and run the interactive trainer with outputs directory
        trainer = InteractiveTrainer(outputs_dir=outputs_dir)
        trainer.run()

        # Log successful completion
        print(f"\n{'='*60}")
        print(f"ML Training Session Completed Successfully")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log saved to: {log_filepath}")
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n\nTraining cancelled by user.")
        print(f"Partial results may be available in: {outputs_dir}")
        print(f"Log saved to: {log_filepath}")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print(f"\nLog saved to: {log_filepath}")
        sys.exit(1)
    finally:
        # Close logging
        close_logging(logger)

if __name__ == "__main__":
    main()
