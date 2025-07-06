"""
Core configuration generator with console progress output.
Generates all possible configurations and saves them organized in folders.
"""

import numpy as np
import itertools
import pickle
import os
import time
from typing import List, Tuple, Set, FrozenSet
import argparse
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()


class CoreConfigGenerator:
    def __init__(self, use_6x6_restriction=False):
        self.configurations = []
        self.irradiation_sets = []
        self.all_configurations_before_symmetry = []
        self.all_irradiation_sets_before_symmetry = []
        self.use_6x6_restriction = use_6x6_restriction

        # Create directories
        self.create_directories()

    def create_directories(self):
        """Create required directory structure.

        Creates all necessary output directories for core configuration data,
        sampling results, and method scripts.
        """
        dirs = [
            'output',
            'output/data',
            'output/core_configs',
            'output/samples_picked',
            'output/samples_picked/pkl',
            'output/samples_picked/txt',
            'output/samples_picked/results',
            'sampling_methods'
        ]
        for dir_path in dirs:
            full_path = SCRIPT_DIR / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")

    def create_base_lattice(self):
        """Create the base 8x8 lattice with coolant in corners and adjacent positions.

        Returns
        -------
        tuple
            (lattice, coolant_positions) where lattice is 8x8 numpy array and
            coolant_positions is list of (row, col) tuples
        """
        lattice = np.array([['F' for _ in range(8)] for _ in range(8)])

        coolant_positions = [
            (0, 0), (0, 7), (7, 0), (7, 7),  # Corners
            (0, 1), (1, 0),                   # Top-left adjacents
            (0, 6), (1, 7),                   # Top-right adjacents
            (6, 0), (7, 1),                   # Bottom-left adjacents
            (6, 7), (7, 6),                   # Bottom-right adjacents
        ]

        for i, j in coolant_positions:
            lattice[i, j] = 'C'

        return lattice, coolant_positions

    def get_fuel_positions(self, lattice):
        """Get all fuel positions from the lattice.

        Parameters
        ----------
        lattice : numpy.ndarray
            8x8 lattice array with 'F' for fuel, 'C' for coolant

        Returns
        -------
        list
            List of (row, col) tuples representing fuel positions
        """
        fuel_positions = [(i, j) for i in range(8) for j in range(8) if lattice[i, j] == 'F']

        if self.use_6x6_restriction:
            # Filter to only positions in the central 6x6 square (rows/cols 1-6)
            fuel_positions = [(i, j) for i, j in fuel_positions if 1 <= i <= 6 and 1 <= j <= 6]

        return fuel_positions

    def get_canonical_form(self, positions: List[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
        """Get the canonical form of a set of positions under D4 symmetry.

        Parameters
        ----------
        positions : List[Tuple[int, int]]
            List of (row, col) position tuples

        Returns
        -------
        FrozenSet[Tuple[int, int]]
            Lexicographically smallest representation under D4 symmetry transformations
        """
        transformations = []
        pos_array = np.array(positions)

        # Rotations (0°, 90°, 180°, 270°)
        for k in range(4):
            if k == 0:
                transformed = pos_array
            elif k == 1:  # 90° rotation: (i,j) -> (j, 7-i)
                transformed = np.array([(j, 7-i) for i, j in positions])
            elif k == 2:  # 180° rotation: (i,j) -> (7-i, 7-j)
                transformed = np.array([(7-i, 7-j) for i, j in positions])
            else:  # 270° rotation: (i,j) -> (7-j, i)
                transformed = np.array([(7-j, i) for i, j in positions])
            transformations.append(frozenset(map(tuple, transformed)))

        # Reflections
        transformations.append(frozenset((7-i, j) for i, j in positions))  # Horizontal
        transformations.append(frozenset((i, 7-j) for i, j in positions))  # Vertical
        transformations.append(frozenset((j, i) for i, j in positions))    # Main diagonal
        transformations.append(frozenset((7-j, 7-i) for i, j in positions))  # Anti-diagonal

        return min(transformations, key=lambda x: sorted(x))

    def generate_configurations(self):
        """Generate configurations with console progress output.

        Generates all possible 4-irradiation position combinations and applies
        D4 symmetry reduction to eliminate duplicates. Shows real-time progress.
        """
        start_time = time.time()

        # Create base lattice
        base_lattice, coolant_positions = self.create_base_lattice()
        fuel_positions = self.get_fuel_positions(base_lattice)

        total_fuel = len(fuel_positions)
        print(f"\nTotal fuel positions: {total_fuel}")

        # Calculate total combinations
        total_combinations = len(list(itertools.combinations(range(total_fuel), 4)))
        print(f"Total possible combinations: {total_combinations:,}")
        print("\nGenerating configurations...")
        print("-" * 60)

        # Use a set to store canonical forms for O(1) duplicate checking
        seen_canonical = set()
        self.configurations = []
        self.irradiation_sets = []

        # Process combinations
        processed = 0
        last_percent = 0

        for irrad_indices in itertools.combinations(range(total_fuel), 4):
            # Get actual positions
            irrad_positions = [fuel_positions[i] for i in irrad_indices]

            # Create configuration for "before symmetry" list
            config = base_lattice.copy()
            for i, j in irrad_positions:
                config[i, j] = 'I'

            # Save all configurations before symmetry
            self.all_configurations_before_symmetry.append(config.copy())
            self.all_irradiation_sets_before_symmetry.append(irrad_positions)

            # Get canonical form
            canonical = self.get_canonical_form(irrad_positions)

            # Check if we've seen this canonical form
            if canonical not in seen_canonical:
                seen_canonical.add(canonical)
                self.configurations.append(config)
                self.irradiation_sets.append(irrad_positions)

            # Update progress
            processed += 1
            percent = int((processed / total_combinations) * 100)

            if percent > last_percent:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                eta = (total_combinations - processed) / rate

                print(f"\rProgress: {percent}% | "
                      f"Processed: {processed:,}/{total_combinations:,} | "
                      f"Unique: {len(self.configurations):,} | "
                      f"Rate: {rate:.0f} configs/sec | "
                      f"ETA: {eta:.0f}s", end='', flush=True)

                last_percent = percent

        # Final update
        elapsed = time.time() - start_time
        print(f"\n\nGeneration complete!")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Total configurations before symmetry: {len(self.all_configurations_before_symmetry):,}")
        print(f"Unique configurations after symmetry: {len(self.configurations):,}")
        print(f"Reduction factor: {len(self.all_configurations_before_symmetry)/len(self.configurations):.1f}x")

    def save_configurations(self):
        """Save configurations to organized folders.

        Saves configurations in multiple formats:
        - Pickle files for programmatic access
        - Text files for human review
        - Summary files with statistics
        """
        print("\nSaving configurations...")

        # Determine filename suffix based on restriction
        suffix = "_6x6" if self.use_6x6_restriction else ""

        # Save main pickle file to data folder
        data = {
            'configurations': self.configurations,
            'irradiation_sets': self.irradiation_sets
        }
        pkl_path = SCRIPT_DIR / f'output/data/core_configurations_optimized{suffix}.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Saved optimized configurations to {pkl_path.relative_to(SCRIPT_DIR)}")

        # Save all configurations before symmetry
        all_data = {
            'configurations': self.all_configurations_before_symmetry,
            'irradiation_sets': self.all_irradiation_sets_before_symmetry
        }
        all_pkl_path = SCRIPT_DIR / f'output/data/all_configurations_before_symmetry{suffix}.pkl'
        with open(all_pkl_path, 'wb') as f:
            pickle.dump(all_data, f)
        print(f"✓ Saved all configurations to {all_pkl_path.relative_to(SCRIPT_DIR)}")

        # Save text file with ALL configurations (before symmetry)
        all_txt_path = SCRIPT_DIR / f'output/core_configs/all_configurations_before_symmetry{suffix}.txt'
        with open(all_txt_path, 'w') as f:
            restriction_info = " (6x6 Central Square)" if self.use_6x6_restriction else ""
            f.write(f"ALL CORE CONFIGURATIONS (Before Symmetry Reduction){restriction_info}\n")
            f.write("="*60 + "\n")
            f.write(f"Total configurations: {len(self.all_configurations_before_symmetry):,}\n")
            if self.use_6x6_restriction:
                f.write("Restriction: Only positions in central 6x6 square (rows/cols 1-6)\n")
            f.write("\n")

            # Write all configurations
            for i, (config, irrad_pos) in enumerate(zip(
                self.all_configurations_before_symmetry[:len(self.all_configurations_before_symmetry)],
                self.all_irradiation_sets_before_symmetry[:len(self.all_irradiation_sets_before_symmetry)]
            )):
                f.write(f"Configuration {i+1}:\n")
                f.write("Irradiation positions: " + str(irrad_pos) + "\n")
                for row in config:
                    f.write(" ".join(row) + "\n")
                f.write("\n")

        print(f"✓ Saved all configurations text to {all_txt_path.relative_to(SCRIPT_DIR)}")

        # Save text file with symmetry-reduced configurations
        reduced_txt_path = SCRIPT_DIR / f'output/core_configs/configurations_after_symmetry{suffix}.txt'
        with open(reduced_txt_path, 'w') as f:
            restriction_info = " (6x6 Central Square)" if self.use_6x6_restriction else ""
            f.write(f"UNIQUE CORE CONFIGURATIONS (After D4 Symmetry Reduction){restriction_info}\n")
            f.write("="*60 + "\n")
            f.write(f"Total unique configurations: {len(self.configurations):,}\n")
            f.write(f"Reduction from: {len(self.all_configurations_before_symmetry):,} original configurations\n")
            f.write(f"Reduction factor: {len(self.all_configurations_before_symmetry)/len(self.configurations):.1f}x\n")
            if self.use_6x6_restriction:
                f.write("Restriction: Only positions in central 6x6 square (rows/cols 1-6)\n")
            f.write("\n")

            # Write all unique configurations
            for i, (config, irrad_pos) in enumerate(zip(
                self.configurations[:len(self.configurations)],
                self.irradiation_sets[:len(self.irradiation_sets)]
            )):
                f.write(f"Configuration {i+1}:\n")
                f.write("Irradiation positions: " + str(irrad_pos) + "\n")
                for row in config:
                    f.write(" ".join(row) + "\n")
                f.write("\n")

        print(f"✓ Saved symmetry-reduced text to {reduced_txt_path.relative_to(SCRIPT_DIR)}")

        # Create a summary file
        summary_path = SCRIPT_DIR / f'output/core_configs/generation_summary{suffix}.txt'
        with open(summary_path, 'w') as f:
            restriction_info = " (6x6 Central Square)" if self.use_6x6_restriction else ""
            f.write(f"CORE CONFIGURATION GENERATION SUMMARY{restriction_info}\n")
            f.write("="*60 + "\n\n")
            f.write("Grid size: 8x8\n")
            if self.use_6x6_restriction:
                f.write("Restriction: Only positions in central 6x6 square (rows/cols 1-6)\n")
                fuel_count = len(self.get_fuel_positions(self.create_base_lattice()[0]))
                f.write(f"Available fuel positions (6x6): {fuel_count}\n")
            else:
                f.write(f"Total fuel positions: {52}\n")
            f.write("Irradiation positions per configuration: 4\n")
            f.write(f"Total coolant positions: {12}\n\n")
            f.write(f"Total possible combinations: {len(self.all_configurations_before_symmetry):,}\n")
            f.write(f"Unique configurations (D4 symmetry): {len(self.configurations):,}\n")
            f.write(f"Reduction factor: {len(self.all_configurations_before_symmetry)/len(self.configurations):.1f}x\n\n")
            f.write("D4 Symmetry Group Operations:\n")
            f.write("- Identity (no transformation)\n")
            f.write("- 90° rotation\n")
            f.write("- 180° rotation\n")
            f.write("- 270° rotation\n")
            f.write("- Horizontal reflection\n")
            f.write("- Vertical reflection\n")
            f.write("- Main diagonal reflection\n")
            f.write("- Anti-diagonal reflection\n")

        print(f"✓ Saved generation summary to {summary_path.relative_to(SCRIPT_DIR)}")

        print("Results saved to:")
        print(f"  {SCRIPT_DIR}/output/data/        - Pickle files")
        print(f"  {SCRIPT_DIR}/output/core_configs/   - Text files with configurations")

        print("\nGenerated files:")
        print(f"  {SCRIPT_DIR}/output/data/core_configurations_optimized{suffix}.pkl")
        print(f"  {SCRIPT_DIR}/output/data/all_configurations_before_symmetry{suffix}.pkl")
        print(f"  {SCRIPT_DIR}/output/core_configs/all_configurations_before_symmetry{suffix}.txt")
        print(f"  {SCRIPT_DIR}/output/core_configs/configurations_after_symmetry{suffix}.txt")
        print(f"  {SCRIPT_DIR}/output/core_configs/generation_summary{suffix}.txt")


def visualize_configuration(config, title="Core Configuration"):
    """Create a simple text visualization of a core configuration.

    Parameters
    ----------
    config : numpy.ndarray
        8x8 configuration array with 'C', 'F', 'I' elements
    title : str, optional
        Title for the visualization (default: "Core Configuration")
    """
    print(f"\n{title}")
    print("-" * 33)
    for row in config:
        print("| " + " | ".join(row) + " |")
    print("-" * 33)
    print("C: Coolant, F: Fuel, I: Irradiation")


def main():
    """Main function to generate core configurations with optional 6x6 restriction.

    Parses command line arguments and runs the complete configuration generation
    workflow including directory setup, generation, and saving.
    """
    parser = argparse.ArgumentParser(description='Generate core configurations')
    parser.add_argument('--restrict-6x6', action='store_true',
                        help='Restrict configurations to central 6x6 square')
    args = parser.parse_args()

    restriction_info = " (6x6 Central Square)" if args.restrict_6x6 else ""
    print(f"CORE CONFIGURATION GENERATOR{restriction_info}")
    print("="*60)
    print("Generating all possible 8x8 core configurations")
    print("with 4 irradiation positions and applying D4 symmetry reduction.")
    if args.restrict_6x6:
        print("\nRESTRICTION: Only using positions in central 6x6 square (rows/cols 1-6)")
    print("\nDirectory structure will be created:")
    print("  output/data/           - PKL files")
    print("  output/core_configs/   - Text files with configurations")
    print("  output/samples_picked/ - Sampling results")
    print("  sampling_methods/ - Individual sampling method scripts")
    print("="*60)

    # Create and run the generator
    generator = CoreConfigGenerator(use_6x6_restriction=args.restrict_6x6)
    generator.generate_configurations()
    generator.save_configurations()

    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
