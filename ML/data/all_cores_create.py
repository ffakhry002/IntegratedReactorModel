"""
All Cores Test Generator
Generates all possible reactor core configurations with 4 irradiation positions.
Uses D4 symmetry reduction to eliminate duplicate configurations.
"""

import numpy as np
import itertools
import time
from typing import List, Tuple, Set, FrozenSet
import os


class AllCoresGenerator:
    def __init__(self):
        """Initialize the all cores generator."""
        self.configurations = []
        self.irradiation_sets = []
        self.all_configurations_before_symmetry = []
        self.all_irradiation_sets_before_symmetry = []

    def create_base_lattice(self):
        """Create the base 8x8 lattice with coolant in corners and adjacent positions.

        The corners (0,0), (1,0), (0,1) and three mirrored corners are coolants.

        Returns
        -------
        tuple
            (lattice, coolant_positions) where lattice is 8x8 numpy array and
            coolant_positions is list of (row, col) tuples
        """
        lattice = np.array([['F' for _ in range(8)] for _ in range(8)], dtype='<U4')

        # Set coolant positions: corners (0,0), (1,0), (0,1) and three mirrored corners
        coolant_positions = [
            (0, 0), (0, 7), (7, 0), (7, 7),  # Four corners
            (0, 1), (1, 0),                   # Adjacent to (0,0)
            (0, 6), (1, 7),                   # Adjacent to (0,7) - mirrored
            (6, 0), (7, 1),                   # Adjacent to (7,0) - mirrored
            (6, 7), (7, 6),                   # Adjacent to (7,7) - mirrored
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
        """Generate all possible configurations with 4 irradiation positions.

        Applies D4 symmetry reduction to eliminate duplicates.
        """
        start_time = time.time()

        # Create base lattice
        base_lattice, coolant_positions = self.create_base_lattice()
        fuel_positions = self.get_fuel_positions(base_lattice)

        total_fuel = len(fuel_positions)
        print(f"Total fuel positions: {total_fuel}")
        print(f"This should be 52 (64 total - 12 coolant positions)")

        # Calculate total combinations (52 choose 4)
        total_combinations = len(list(itertools.combinations(range(total_fuel), 4)))
        print(f"Total possible combinations (52 choose 4): {total_combinations:,}")
        print("\nGenerating configurations with symmetry reduction...")
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
            for idx, (i, j) in enumerate(irrad_positions):
                config[i, j] = 'I'

            # Save all configurations before symmetry
            self.all_configurations_before_symmetry.append(config.copy())
            self.all_irradiation_sets_before_symmetry.append(irrad_positions)

            # Get canonical form
            canonical = self.get_canonical_form(irrad_positions)

            # Check if we've seen this canonical form
            if canonical not in seen_canonical:
                seen_canonical.add(canonical)
                # Create canonical configuration with sorted irradiation positions
                canonical_config = base_lattice.copy()
                sorted_positions = sorted(list(canonical))
                for idx, (i, j) in enumerate(sorted_positions):
                    canonical_config[i, j] = f'I_{idx+1}'
                self.configurations.append(canonical_config)
                self.irradiation_sets.append(sorted_positions)

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

    def format_configuration_for_output(self, config):
        """Format a configuration as a nested list for output.

        Parameters
        ----------
        config : numpy.ndarray
            8x8 configuration array

        Returns
        -------
        list
            Nested list representation of the configuration
        """
        return config.tolist()

    def save_configurations_to_file(self, filename="all_reactor_configurations.txt"):
        """Save all unique configurations to a text file in the specified format.

        Parameters
        ----------
        filename : str
            Name of the output file
        """
        output_path = os.path.join(os.path.dirname(__file__), filename)

        with open(output_path, 'w') as f:
            for i, config in enumerate(self.configurations):
                f.write(f"RUN {i+1}:\n")
                f.write("----------------------------------------\n")
                f.write(f"Description: random_geometric_{i+1}\n")
                f.write(f"  core_lattice: {self.format_configuration_for_output(config)}\n")
                f.write("=" * 80 + "\n\n")
                f.write("=" * 80 + "\n\n")

        print(f"✓ Saved {len(self.configurations)} unique configurations to {output_path}")
        return output_path


def main():
    """Main function to generate all reactor core configurations."""
    print("ALL REACTOR CORES TEST GENERATOR")
    print("="*60)
    print("Generating all possible 8x8 reactor core configurations")
    print("with 4 irradiation positions and D4 symmetry reduction.")
    print("="*60)

    # Create and run the generator
    generator = AllCoresGenerator()
    generator.generate_configurations()

    # Save configurations
    output_file = generator.save_configurations_to_file()

    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print(f"Output saved to: {output_file}")
    print("="*60)


if __name__ == "__main__":
    main()
