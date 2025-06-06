import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_execution.main import run_testing

if __name__ == "__main__":
    try:
        # Get ML directory path
        ml_dir = os.path.dirname(os.path.abspath(__file__))
        outputs_dir = os.path.join(ml_dir, "outputs")

        # Ensure excel_reports directory exists
        excel_reports_dir = os.path.join(outputs_dir, "excel_reports")
        os.makedirs(excel_reports_dir, exist_ok=True)

        # Run testing with outputs directory
        run_testing(outputs_dir=outputs_dir)
    except KeyboardInterrupt:
        print("\n\nTesting cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
