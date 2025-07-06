#!/usr/bin/env python3
"""
Comprehensive test script to validate that models learn physics, not labels
Tests with actual train.txt and test.txt data
"""

import numpy as np
import pandas as pd
import joblib
import os
import glob
import sys
from datetime import datetime

# Add ML directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.txt_to_data import parse_reactor_data
from ML_models.encodings.encoding_methods import ReactorEncodings
from test_execution.model_tester import ReactorModelTester


class PhysicsLearningTester:
    """Test if models truly learn position->flux physics.

    Parameters
    ----------
    models_dir : str, optional
        Directory containing trained models, by default "ML/outputs/models"
    outputs_dir : str, optional
        Directory for output files, by default "ML/outputs"

    Returns
    -------
    None
    """

    def __init__(self, models_dir="ML/outputs/models", outputs_dir="ML/outputs"):
        """Initialize PhysicsLearningTester.

        Parameters
        ----------
        models_dir : str, optional
            Directory containing trained models, by default "ML/outputs/models"
        outputs_dir : str, optional
            Directory for output files, by default "ML/outputs"

        Returns
        -------
        None
        """
        self.models_dir = models_dir
        self.outputs_dir = outputs_dir
        self.encodings = ReactorEncodings()
        self.results = []

    def load_available_models(self):
        """Load all trained models.

        Parameters
        ----------
        None

        Returns
        -------
        list
            List of dictionaries containing model information and loaded models
        """
        print("\nSearching for trained models...")
        model_files = glob.glob(os.path.join(self.models_dir, "*.pkl"))

        models = []
        for filepath in model_files:
            try:
                data = joblib.load(filepath)
                model_info = {
                    'filepath': filepath,
                    'filename': os.path.basename(filepath),
                    'model_class': data.get('model_class', 'unknown'),
                    'model_type': data.get('model_type', 'unknown'),
                    'encoding': data.get('encoding', 'unknown'),
                    'optimization': data.get('optimization_method', 'unknown'),
                    'model': data.get('model'),
                    'use_log_flux': data.get('use_log_flux', False),
                    'flux_scale': data.get('flux_scale', 1e14)
                }

                # Only include flux models for this test
                if model_info['model_type'] == 'flux':
                    models.append(model_info)

            except Exception as e:
                print(f"  Warning: Could not load {filepath}: {e}")

        print(f"Found {len(models)} flux prediction models")
        return models

    def create_label_swapped_configs(self, lattices, flux_data):
        """Create configurations with swapped labels to test label independence.

        Parameters
        ----------
        lattices : list
            List of lattice configurations
        flux_data : list
            List of flux data corresponding to lattices

        Returns
        -------
        list
            List of dictionaries containing original and swapped configurations
        """
        swapped_configs = []

        for idx, (lattice, flux) in enumerate(zip(lattices, flux_data)):
            # Create a copy with swapped labels
            swapped_lattice = lattice.copy()
            swapped_flux = {}

            # Find all irradiation positions
            positions = {}
            for i in range(8):
                for j in range(8):
                    if lattice[i,j].startswith('I_'):
                        positions[lattice[i,j]] = (i, j)

            # Swap I_1 ↔ I_4 and I_2 ↔ I_3
            swaps = {'I_1': 'I_4', 'I_4': 'I_1', 'I_2': 'I_3', 'I_3': 'I_2'}

            for old_label, pos in positions.items():
                new_label = swaps.get(old_label, old_label)
                swapped_lattice[pos] = new_label
                # The flux at this position stays the same!
                swapped_flux[new_label] = flux[old_label]

            swapped_configs.append({
                'original_idx': idx,
                'original_lattice': lattice,
                'swapped_lattice': swapped_lattice,
                'original_flux': flux,
                'swapped_flux': swapped_flux,
                'positions': positions
            })

        return swapped_configs

    def test_position_vs_label_learning(self, model_info, test_configs):
        """Test if model predictions depend on position or label.

        Parameters
        ----------
        model_info : dict
            Dictionary containing model information and loaded model
        test_configs : list
            List of test configurations with swapped labels

        Returns
        -------
        dict
            Dictionary containing test results and scores
        """
        print(f"\n  Testing {model_info['model_class']} ({model_info['encoding']})...")

        model = model_info['model']
        encoding_method = model_info['encoding']

        position_based_score = 0
        label_based_score = 0
        total_tests = 0

        for config in test_configs[:5]:  # Test first 5 configurations
            # Encode original configuration
            if encoding_method == 'one_hot':
                features_orig, _, pos_order = self.encodings.one_hot_encoding(config['original_lattice'])
            elif encoding_method == 'categorical':
                features_orig, _, pos_order = self.encodings.categorical_encoding(config['original_lattice'])
            elif encoding_method == 'physics':
                features_orig, _, pos_order = self.encodings.physics_based_encoding(config['original_lattice'])
            elif encoding_method == 'spatial':
                features_orig, _, pos_order = self.encodings.spatial_convolution_encoding(config['original_lattice'])
            elif encoding_method == 'graph':
                features_orig, _, pos_order = self.encodings.graph_based_encoding(config['original_lattice'])

            # Encode swapped configuration
            if encoding_method == 'one_hot':
                features_swap, _, pos_order_swap = self.encodings.one_hot_encoding(config['swapped_lattice'])
            elif encoding_method == 'categorical':
                features_swap, _, pos_order_swap = self.encodings.categorical_encoding(config['swapped_lattice'])
            elif encoding_method == 'physics':
                features_swap, _, pos_order_swap = self.encodings.physics_based_encoding(config['swapped_lattice'])
            elif encoding_method == 'spatial':
                features_swap, _, pos_order_swap = self.encodings.spatial_convolution_encoding(config['swapped_lattice'])
            elif encoding_method == 'graph':
                features_swap, _, pos_order_swap = self.encodings.graph_based_encoding(config['swapped_lattice'])

            # Check if features are identical (they should be for position-based learning)
            features_identical = np.allclose(features_orig, features_swap, rtol=1e-5)

            # Make predictions
            pred_orig = model.predict(features_orig.reshape(1, -1))[0]
            pred_swap = model.predict(features_swap.reshape(1, -1))[0]

            # Transform predictions back from log scale if needed
            if model_info['use_log_flux']:
                pred_orig = 10 ** pred_orig
                pred_swap = 10 ** pred_swap
            else:
                pred_orig = pred_orig * model_info['flux_scale']
                pred_swap = pred_swap * model_info['flux_scale']

            # Check if predictions are position-based or label-based
            # For position-based: predictions should be identical
            # For label-based: predictions would follow the labels

            predictions_identical = np.allclose(pred_orig, pred_swap, rtol=0.01)

            if predictions_identical:
                position_based_score += 1
            else:
                # Check if predictions followed the labels
                # This is harder to verify without the exact mapping
                label_based_score += 1

            total_tests += 1

            # Detailed output for first config
            if config['original_idx'] == 0:
                print(f"    Config {config['original_idx']+1} test:")
                print(f"      Features identical: {features_identical}")
                print(f"      Predictions identical: {predictions_identical}")
                if not predictions_identical:
                    print(f"      Max prediction difference: {np.max(np.abs(pred_orig - pred_swap)):.2e}")

        # Return scores
        position_ratio = position_based_score / total_tests if total_tests > 0 else 0
        return {
            'model': f"{model_info['model_class']}_{model_info['encoding']}",
            'position_based_score': position_based_score,
            'label_based_score': label_based_score,
            'total_tests': total_tests,
            'position_ratio': position_ratio,
            'is_position_based': position_ratio > 0.8  # 80% threshold
        }

    def test_edge_vs_center_learning(self, model_info, lattices):
        """Test if model correctly learns edge=low flux, center=high flux.

        Parameters
        ----------
        model_info : dict
            Dictionary containing model information and loaded model
        lattices : list
            List of lattice configurations to test

        Returns
        -------
        dict or None
            Dictionary containing test results, or None if insufficient data
        """
        print(f"\n  Testing edge vs center learning...")

        model = model_info['model']
        encoding_method = model_info['encoding']

        edge_fluxes = []
        center_fluxes = []

        for lattice in lattices[:10]:  # Test first 10 configurations
            # Find irradiation positions and classify as edge or center
            edge_positions = []
            center_positions = []

            for i in range(8):
                for j in range(8):
                    if lattice[i,j].startswith('I_'):
                        # Classify position
                        if i in [0, 1, 6, 7] or j in [0, 1, 6, 7]:
                            edge_positions.append((i, j))
                        elif 2 <= i <= 5 and 2 <= j <= 5:
                            center_positions.append((i, j))

            if not edge_positions or not center_positions:
                continue  # Skip if no clear edge/center positions

            # Encode and predict
            if encoding_method == 'one_hot':
                features, _, pos_order = self.encodings.one_hot_encoding(lattice)
            elif encoding_method == 'categorical':
                features, _, pos_order = self.encodings.categorical_encoding(lattice)
            elif encoding_method == 'physics':
                features, _, pos_order = self.encodings.physics_based_encoding(lattice)
            elif encoding_method == 'spatial':
                features, _, pos_order = self.encodings.spatial_convolution_encoding(lattice)
            elif encoding_method == 'graph':
                features, _, pos_order = self.encodings.graph_based_encoding(lattice)

            predictions = model.predict(features.reshape(1, -1))[0]

            # Transform predictions back
            if model_info['use_log_flux']:
                predictions = 10 ** predictions
            else:
                predictions = predictions * model_info['flux_scale']

            # Map predictions to positions
            for idx, pos in enumerate(pos_order):
                if pos in edge_positions:
                    edge_fluxes.append(predictions[idx])
                elif pos in center_positions:
                    center_fluxes.append(predictions[idx])

        # Calculate statistics
        if edge_fluxes and center_fluxes:
            avg_edge = np.mean(edge_fluxes)
            avg_center = np.mean(center_fluxes)

            # Center should have higher flux than edge
            learns_physics = avg_center > avg_edge
            ratio = avg_center / avg_edge if avg_edge > 0 else float('inf')

            return {
                'avg_edge_flux': avg_edge,
                'avg_center_flux': avg_center,
                'center_to_edge_ratio': ratio,
                'learns_physics': learns_physics
            }
        else:
            return None

    def generate_report(self, results):
        """Generate comprehensive test report.

        Parameters
        ----------
        results : list
            List of test results for all models

        Returns
        -------
        str
            Path to the generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.outputs_dir, f"physics_learning_test_{timestamp}.txt")

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PHYSICS LEARNING TEST REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models Tested: {len(results)}\n\n")

            # Summary table
            f.write("SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Model':<30} {'Position-Based?':<15} {'Edge<Center?':<15} {'Physics Score':<15}\n")
            f.write("-"*80 + "\n")

            for r in results:
                physics_score = "GOOD" if r['position_test']['is_position_based'] and r['edge_center_test']['learns_physics'] else "POOR"
                f.write(f"{r['model']:<30} "
                       f"{'YES' if r['position_test']['is_position_based'] else 'NO':<15} "
                       f"{'YES' if r['edge_center_test']['learns_physics'] else 'NO':<15} "
                       f"{physics_score:<15}\n")

            # Detailed results
            f.write("\n\nDETAILED RESULTS\n")
            f.write("="*80 + "\n")

            for r in results:
                f.write(f"\nModel: {r['model']}\n")
                f.write("-"*40 + "\n")

                # Position vs label test
                pt = r['position_test']
                f.write(f"Position Independence Test:\n")
                f.write(f"  Position-based predictions: {pt['position_based_score']}/{pt['total_tests']}\n")
                f.write(f"  Label-based predictions: {pt['label_based_score']}/{pt['total_tests']}\n")
                f.write(f"  Position ratio: {pt['position_ratio']:.2%}\n")
                f.write(f"  Verdict: {'POSITION-BASED' if pt['is_position_based'] else 'LABEL-BASED'}\n")

                # Edge vs center test
                ect = r['edge_center_test']
                f.write(f"\nEdge vs Center Physics Test:\n")
                f.write(f"  Average edge flux: {ect['avg_edge_flux']:.2e}\n")
                f.write(f"  Average center flux: {ect['avg_center_flux']:.2e}\n")
                f.write(f"  Center/Edge ratio: {ect['center_to_edge_ratio']:.2f}\n")
                f.write(f"  Verdict: {'CORRECT PHYSICS' if ect['learns_physics'] else 'INCORRECT PHYSICS'}\n")

                # Overall assessment
                f.write(f"\nOverall Assessment:\n")
                if pt['is_position_based'] and ect['learns_physics']:
                    f.write("  ✓ Model correctly learns position-based physics\n")
                elif not pt['is_position_based']:
                    f.write("  ✗ Model is learning label-based patterns (not physics)\n")
                elif not ect['learns_physics']:
                    f.write("  ✗ Model does not correctly learn edge<center physics\n")

            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

        print(f"\nReport saved to: {report_path}")
        return report_path

    def run_comprehensive_test(self, train_file="ML/data/train.txt", test_file="ML/data/test.txt"):
        """Run all tests on available models.

        Parameters
        ----------
        train_file : str, optional
            Path to training data file, by default "ML/data/train.txt"
        test_file : str, optional
            Path to test data file, by default "ML/data/test.txt"

        Returns
        -------
        None
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE PHYSICS LEARNING TEST")
        print("="*80)

        # Load models
        models = self.load_available_models()
        if not models:
            print("No models found to test!")
            return

        # Load test data
        print(f"\nLoading test data from {test_file}...")
        test_data = parse_reactor_data(test_file)
        if len(test_data) == 4:
            test_lattices, test_flux, test_keff, _ = test_data
        else:
            test_lattices, test_flux, test_keff = test_data

        # Create label-swapped configurations
        print("\nCreating label-swapped test configurations...")
        swapped_configs = self.create_label_swapped_configs(test_lattices, test_flux)

        # Test each model
        results = []
        for model_info in models:
            print(f"\n{'='*60}")
            print(f"Testing: {model_info['model_class']} with {model_info['encoding']} encoding")
            print(f"{'='*60}")

            # Test 1: Position vs Label learning
            position_test = self.test_position_vs_label_learning(model_info, swapped_configs)

            # Test 2: Edge vs Center physics
            edge_center_test = self.test_edge_vs_center_learning(model_info, test_lattices)

            results.append({
                'model': f"{model_info['model_class']}_{model_info['encoding']}_{model_info['optimization']}",
                'position_test': position_test,
                'edge_center_test': edge_center_test
            })

            # Print quick summary
            print(f"\n  Summary:")
            print(f"    Position-based learning: {'YES' if position_test['is_position_based'] else 'NO'}")
            if edge_center_test:
                print(f"    Correct physics (edge<center): {'YES' if edge_center_test['learns_physics'] else 'NO'}")

        # Generate report
        self.generate_report(results)

        # Print final summary
        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)

        good_models = sum(1 for r in results
                         if r['position_test']['is_position_based']
                         and r['edge_center_test']['learns_physics'])

        print(f"\nModels with correct physics learning: {good_models}/{len(results)}")

        if good_models == 0:
            print("\nWARNING: No models are learning position-based physics!")
            print("   This suggests the encoding fix has not been applied.")
            print("   Please ensure you're using the updated encoding_methods.py")
        elif good_models == len(results):
            print("\nSUCCESS: All models are learning position-based physics!")
            print("   The encoding fix is working correctly.")
        else:
            print("\nPARTIAL SUCCESS: Some models are learning physics correctly.")
            print("   Check which models were trained with the old encoding.")


def main():
    """Run the comprehensive test.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Determine paths
    if os.path.exists("ML/outputs/models"):
        models_dir = "ML/outputs/models"
        outputs_dir = "ML/outputs"
        train_file = "ML/data/train.txt"
        test_file = "ML/data/test.txt"
    elif os.path.exists("outputs/models"):
        models_dir = "outputs/models"
        outputs_dir = "outputs"
        train_file = "data/train.txt"
        test_file = "data/test.txt"
    else:
        print("Error: Could not find models directory")
        print("Please run from the project root directory")
        return

    # Create tester and run
    tester = PhysicsLearningTester(models_dir, outputs_dir)
    tester.run_comprehensive_test(train_file, test_file)


if __name__ == "__main__":
    main()
