#!/usr/bin/env python3
"""
Comprehensive checker for finding ALL test configurations in generated file,
accounting for D4 symmetry group transformations.
"""

import sys
import re
from typing import List, Tuple, Set, Dict

def parse_lattice(lattice_str):
    """Parse a lattice string into a 2D list."""
    if 'core_lattice:' in lattice_str:
        lattice_str = lattice_str.split('core_lattice:')[1].strip()
    try:
        return eval(lattice_str)
    except:
        return None

def get_instrument_positions(lattice):
    """Extract only the positions (not labels) of instruments."""
    positions = []
    for i in range(len(lattice)):
        for j in range(len(lattice[i])):
            if isinstance(lattice[i][j], str) and lattice[i][j].startswith('I_'):
                positions.append((i, j))
    return sorted(positions)

def apply_d4_transformations(positions):
    """Apply all 8 D4 symmetry transformations to a set of positions."""
    transformations = []

    # Original
    transformations.append(sorted(positions))

    # 90° rotation: (i,j) -> (j, 7-i)
    transformations.append(sorted([(j, 7-i) for i, j in positions]))

    # 180° rotation: (i,j) -> (7-i, 7-j)
    transformations.append(sorted([(7-i, 7-j) for i, j in positions]))

    # 270° rotation: (i,j) -> (7-j, i)
    transformations.append(sorted([(7-j, i) for i, j in positions]))

    # Horizontal flip: (i,j) -> (7-i, j)
    transformations.append(sorted([(7-i, j) for i, j in positions]))

    # Vertical flip: (i,j) -> (i, 7-j)
    transformations.append(sorted([(i, 7-j) for i, j in positions]))

    # Transpose: (i,j) -> (j, i)
    transformations.append(sorted([(j, i) for i, j in positions]))

    # Anti-diagonal: (i,j) -> (7-j, 7-i)
    transformations.append(sorted([(7-j, 7-i) for i, j in positions]))

    return transformations

def parse_test_file(filename):
    """Parse the test file and extract all configurations."""
    configurations = {}

    with open(filename, 'r') as f:
        content = f.read()

    # Split by RUN entries
    runs = re.split(r'RUN \d+:', content)

    for run in runs[1:]:  # Skip the first empty split
        lines = run.strip().split('\n')

        # Extract RUN number and description
        run_match = re.search(r'Description: ([\w_]+)', run)
        if not run_match:
            continue

        description = run_match.group(1)

        # Find the core_lattice line
        for line in lines:
            if 'core_lattice:' in line:
                lattice = parse_lattice(line)
                if lattice:
                    # Extract RUN number from the content before this run
                    run_num_match = re.search(r'RUN (\d+):', content[content.find(run)-50:content.find(run)])
                    if run_num_match:
                        run_num = run_num_match.group(1)
                        configurations[f'RUN {run_num}: {description}'] = lattice
                break

    return configurations

def find_all_configurations(test_configs, generated_file='all_reactor_configurations.txt'):
    """Find all test configurations in the generated file."""
    # Load generated configurations
    print("Loading generated configurations...")
    generated_positions = {}

    with open(generated_file, 'r') as f:
        lines = f.readlines()

    current_run = None
    for i, line in enumerate(lines):
        if line.startswith('RUN'):
            current_run = line.strip()
        elif 'core_lattice:' in line:
            gen_config = parse_lattice(line)
            if gen_config:
                gen_positions = get_instrument_positions(gen_config)
                generated_positions[current_run] = gen_positions

    print(f"Loaded {len(generated_positions)} generated configurations\n")

    # Check each test configuration
    results = {
        'found': [],
        'not_found': []
    }

    for test_name, test_config in test_configs.items():
        test_positions = get_instrument_positions(test_config)
        test_symmetries = apply_d4_transformations(test_positions)

        found = False
        for gen_run, gen_positions in generated_positions.items():
            for sym_idx, sym_positions in enumerate(test_symmetries):
                if gen_positions == sym_positions:
                    transformation_names = [
                        "Original", "90° rotation", "180° rotation", "270° rotation",
                        "Horizontal flip", "Vertical flip", "Transpose", "Anti-diagonal"
                    ]
                    results['found'].append({
                        'test': test_name,
                        'generated': gen_run,
                        'transformation': transformation_names[sym_idx],
                        'positions': test_positions
                    })
                    found = True
                    break
            if found:
                break

        if not found:
            results['not_found'].append({
                'test': test_name,
                'positions': test_positions
            })

    return results

def main():
    print("="*80)
    print("COMPREHENSIVE CONFIGURATION MATCHER")
    print("="*80)

    # Parse test file
    print("\nParsing test file...")
    test_file = input("Enter the path to your test file (or press Enter for 'test.txt'): ").strip()
    if not test_file:
        test_file = 'test.txt'

    try:
        test_configs = parse_test_file(test_file)
        print(f"Found {len(test_configs)} test configurations")
    except FileNotFoundError:
        print(f"Error: Could not find file '{test_file}'")
        return
    except Exception as e:
        print(f"Error parsing test file: {e}")
        return

    # Find matches
    print("\nSearching for matches...")
    results = find_all_configurations(test_configs)

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\n✓ FOUND: {len(results['found'])} configurations")
    print("-"*40)
    for match in sorted(results['found'], key=lambda x: x['test']):
        print(f"{match['test']}")
        print(f"  → {match['generated']} (via {match['transformation']})")
        print(f"  Positions: {match['positions']}")
        print()

    print(f"\n✗ NOT FOUND: {len(results['not_found'])} configurations")
    print("-"*40)
    for missing in sorted(results['not_found'], key=lambda x: x['test']):
        print(f"{missing['test']}")
        print(f"  Positions: {missing['positions']}")
        print()

    # Summary statistics
    total = len(test_configs)
    found = len(results['found'])
    not_found = len(results['not_found'])

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total test configurations: {total}")
    print(f"Found: {found} ({found/total*100:.1f}%)")
    print(f"Not found: {not_found} ({not_found/total*100:.1f}%)")

    # Optional: Save results to file
    save = input("\nSave results to file? (y/n): ").strip().lower()
    if save == 'y':
        output_file = 'configuration_match_results.txt'
        with open(output_file, 'w') as f:
            f.write("CONFIGURATION MATCH RESULTS\n")
            f.write("="*80 + "\n\n")

            f.write(f"FOUND ({len(results['found'])} configurations):\n")
            f.write("-"*40 + "\n")
            for match in sorted(results['found'], key=lambda x: x['test']):
                f.write(f"{match['test']} → {match['generated']} (via {match['transformation']})\n")

            f.write(f"\n\nNOT FOUND ({len(results['not_found'])} configurations):\n")
            f.write("-"*40 + "\n")
            for missing in sorted(results['not_found'], key=lambda x: x['test']):
                f.write(f"{missing['test']}\n")

        print(f"\nResults saved to '{output_file}'")

if __name__ == "__main__":
    main()
