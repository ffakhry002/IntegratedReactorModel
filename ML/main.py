"""
Nuclear Reactor ML Training System
Main entry point for interactive model training
"""

import sys
import os
from utils.txt_to_data import create_reactor_data_excel

# Add the ML directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from execution.interactive_menu import InteractiveTrainer

def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("NUCLEAR REACTOR ML TRAINING SYSTEM")
    print("="*60)
    print("Version: 1.0")
    print("Parallel Computing: Enabled")
    print("="*60 + "\n")

    # Get the directory where main.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create outputs directory structure in the same folder as main.py
    outputs_dir = os.path.join(script_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "excel_reports"), exist_ok=True)

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

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
