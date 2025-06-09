#!/usr/bin/env python3
"""
Interactive parameter selection for geometric/physics-based sampling methods.
Allows users to choose which parameters to include in the sampling process.
"""

import numpy as np
import json
from typing import List, Dict, Set, Tuple
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()


class InteractiveParameterSelector:
    """Handle interactive selection of geometric parameters for sampling."""

    # Default parameter descriptions
    PARAMETER_INFO = {
        'avg_distance_from_core_center': {
            'name': 'Average Distance from Core Center',
            'description': 'How far irradiation positions are from the reactor center on average',
            'default': True
        },
        'min_inter_position_distance': {
            'name': 'Minimum Inter-Position Distance',
            'description': 'Smallest distance between any two irradiation positions',
            'default': True
        },
        'clustering_coefficient': {
            'name': 'Clustering Coefficient',
            'description': 'Radius of smallest circle containing all positions (how grouped they are)',
            'default': True
        },
        'symmetry_balance': {
            'name': 'Symmetry Balance',
            'description': 'Distance between center of mass of positions and reactor center',
            'default': True
        },
        'local_fuel_density': {
            'name': 'Local Fuel Density',
            'description': 'Average number of fuel positions adjacent to irradiation positions',
            'default': True
        },
        'avg_distance_to_edge': {
            'name': 'Average Distance to Edge',
            'description': 'How far irradiation positions are from reactor edges/corners on average',
            'default': True
        }
    }

    def __init__(self):
        self.selected_parameters = []
        self.available_parameters = []

    def check_available_parameters(self, restrict_6x6=False) -> List[str]:
        """Check which parameters are available in the physics parameters file."""
        suffix = "_6x6" if restrict_6x6 else ""
        params_file = SCRIPT_DIR / f'output/data/physics_parameters{suffix}.pkl'

        if not params_file.exists():
            print(f"Physics parameters file not found: {params_file}")
            return []

        import pickle
        with open(params_file, 'rb') as f:
            data = pickle.load(f)

        if data['parameters']:
            first_param = data['parameters'][0]
            self.available_parameters = [key for key in self.PARAMETER_INFO.keys() if key in first_param]
        else:
            self.available_parameters = []

        return self.available_parameters

    def interactive_selection(self) -> List[str]:
        """Interactively select which parameters to use."""
        print("\n" + "="*60)
        print("GEOMETRIC PARAMETER SELECTION")
        print("="*60)
        print("\nAvailable parameters for sampling:")
        print("-"*60)

        # Display available parameters with descriptions
        for i, param_key in enumerate(self.available_parameters, 1):
            info = self.PARAMETER_INFO[param_key]
            print(f"\n{i}. {info['name']}")
            print(f"   {info['description']}")
            print(f"   Default: {'Yes' if info['default'] else 'No'}")

        print("\n" + "-"*60)
        print("\nOptions:")
        print("  1. Use all available parameters (recommended)")
        print("  2. Use default parameters only")
        print("  3. Custom selection")
        print("  4. Load previous selection")

        while True:
            choice = input("\nSelect option (1-4): ").strip()

            if choice == '1':
                self.selected_parameters = self.available_parameters.copy()
                break
            elif choice == '2':
                self.selected_parameters = [p for p in self.available_parameters
                                          if self.PARAMETER_INFO[p]['default']]
                break
            elif choice == '3':
                self.selected_parameters = self._custom_selection()
                break
            elif choice == '4':
                loaded = self._load_previous_selection()
                if loaded:
                    self.selected_parameters = loaded
                    break
                else:
                    print("No previous selection found or invalid selection.")
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")

        # Save selection for future use
        self._save_selection()

        print("\n" + "-"*60)
        print("Selected parameters:")
        for param in self.selected_parameters:
            print(f"  ✓ {self.PARAMETER_INFO[param]['name']}")
        print("-"*60)

        return self.selected_parameters

    def _custom_selection(self) -> List[str]:
        """Allow custom selection of parameters."""
        selected = []

        print("\n" + "-"*60)
        print("CUSTOM PARAMETER SELECTION")
        print("For each parameter, enter 'y' to include or 'n' to exclude:")
        print("-"*60)

        for param_key in self.available_parameters:
            info = self.PARAMETER_INFO[param_key]
            while True:
                response = input(f"\nInclude '{info['name']}'? (y/n): ").strip().lower()
                if response in ['y', 'yes']:
                    selected.append(param_key)
                    break
                elif response in ['n', 'no']:
                    break
                else:
                    print("Please enter 'y' or 'n'")

        if not selected:
            print("\nWarning: No parameters selected! Using all parameters.")
            return self.available_parameters.copy()

        return selected

    def _save_selection(self):
        """Save the current selection to a file."""
        selection_file = SCRIPT_DIR / 'output/parameter_selection.json'
        selection_file.parent.mkdir(parents=True, exist_ok=True)

        with open(selection_file, 'w') as f:
            json.dump({
                'selected_parameters': self.selected_parameters,
                'available_parameters': self.available_parameters
            }, f, indent=2)

    def _load_previous_selection(self) -> List[str]:
        """Load a previous parameter selection."""
        selection_file = SCRIPT_DIR / 'output/parameter_selection.json'

        if not selection_file.exists():
            return []

        try:
            with open(selection_file, 'r') as f:
                data = json.load(f)

            # Validate that selected parameters are still available
            selected = data.get('selected_parameters', [])
            valid_selected = [p for p in selected if p in self.available_parameters]

            if valid_selected:
                print(f"\nLoaded previous selection with {len(valid_selected)} parameters")
                return valid_selected
            else:
                return []

        except Exception as e:
            print(f"Error loading previous selection: {e}")
            return []

    def filter_physics_parameters(self, physics_params: List[Dict], selected_params: List[str]) -> np.ndarray:
        """Filter physics parameters to only include selected ones."""
        n_configs = len(physics_params)
        n_features = len(selected_params)

        feature_matrix = np.zeros((n_configs, n_features))

        for i in range(n_configs):
            for j, param_key in enumerate(selected_params):
                feature_matrix[i, j] = physics_params[i][param_key]

        return feature_matrix


def get_parameter_selection(interactive=True, restrict_6x6=False) -> Tuple[List[str], bool]:
    """
    Get parameter selection either interactively or from saved selection.

    Returns:
        Tuple of (selected_parameters, was_interactive)
    """
    selector = InteractiveParameterSelector()
    available = selector.check_available_parameters(restrict_6x6)

    if not available:
        print("No parameters available!")
        return [], False

    if interactive:
        selected = selector.interactive_selection()
        return selected, True
    else:
        # Try to load previous selection
        loaded = selector._load_previous_selection()
        if loaded:
            return loaded, False
        else:
            # Use all available parameters as default
            return available, False


if __name__ == "__main__":
    # Test the interactive selection
    import argparse
    parser = argparse.ArgumentParser(description='Test interactive parameter selection')
    parser.add_argument('--restrict-6x6', action='store_true',
                       help='Use 6x6 restricted configurations')
    args = parser.parse_args()

    selected, was_interactive = get_parameter_selection(interactive=True,
                                                       restrict_6x6=args.restrict_6x6)

    print(f"\n{'Interactively' if was_interactive else 'Automatically'} selected {len(selected)} parameters")
