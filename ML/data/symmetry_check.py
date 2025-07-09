#!/usr/bin/env python3
"""
D4 Symmetry Log File Analyzer

This script analyzes a parametric study log file to find which core lattices
are D4 rotations/reflections of each other. It removes the numbering from
irradiation positions (I_1 -> I) before comparison.
"""

import re
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict


class D4LogAnalyzer:
    """Analyze log files for D4 symmetric core lattices"""
    
    def __init__(self):
        self.runs = {}
        self.symmetry_groups = defaultdict(list)
        
    def parse_log_file(self, filepath):
        """Parse the log file and extract core lattices"""
        print(f"Parsing log file: {filepath}")
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Split by RUN entries
        run_pattern = r'RUN (\d+):(.*?)(?=RUN \d+:|$)'
        matches = re.findall(run_pattern, content, re.DOTALL)
        
        for run_num, run_content in matches:
            # Extract description
            desc_match = re.search(r'Description: ([\w_]+)', run_content)
            if not desc_match:
                continue
            description = desc_match.group(1)
            
            # Extract core_lattice
            lattice_match = re.search(r'core_lattice:\s*(\[.*?\])\s*(?:Success|Modified)', 
                                    run_content, re.DOTALL)
            if lattice_match:
                lattice_str = lattice_match.group(1)
                # Clean up the string
                lattice_str = re.sub(r'\s+', ' ', lattice_str)
                try:
                    lattice_list = eval(lattice_str)
                    lattice_array = np.array(lattice_list, dtype='<U10')
                    
                    self.runs[run_num] = {
                        'description': description,
                        'lattice_original': lattice_array.copy(),
                        'lattice_normalized': self.normalize_lattice(lattice_array)
                    }
                except:
                    print(f"Failed to parse lattice for RUN {run_num}")
        
        print(f"Successfully parsed {len(self.runs)} runs")
    
    def normalize_lattice(self, lattice):
        """Replace I_1, I_2, etc. with just 'I'"""
        normalized = lattice.copy()
        for i in range(normalized.shape[0]):
            for j in range(normalized.shape[1]):
                if normalized[i, j].startswith('I_'):
                    normalized[i, j] = 'I'
        return normalized
    
    def apply_d4_transformation(self, lattice, transform_type):
        """Apply a D4 transformation to a lattice"""
        n = lattice.shape[0]
        
        if transform_type == 'identity':
            return lattice
        elif transform_type == 'rot90':
            return np.rot90(lattice, -1)  # Clockwise
        elif transform_type == 'rot180':
            return np.rot90(lattice, 2)
        elif transform_type == 'rot270':
            return np.rot90(lattice, -3)  # 270 clockwise = 90 counter-clockwise
        elif transform_type == 'flip_h':
            return np.flipud(lattice)
        elif transform_type == 'flip_v':
            return np.fliplr(lattice)
        elif transform_type == 'transpose':
            return lattice.T
        elif transform_type == 'anti_diag':
            # Flip along anti-diagonal
            return np.flipud(np.fliplr(lattice)).T
    
    def get_canonical_form(self, lattice):
        """Get canonical form of a lattice under D4 transformations"""
        transformations = [
            'identity', 'rot90', 'rot180', 'rot270',
            'flip_h', 'flip_v', 'transpose', 'anti_diag'
        ]
        
        candidates = []
        
        for transform in transformations:
            transformed = self.apply_d4_transformation(lattice, transform)
            # Convert to tuple for comparison
            lattice_tuple = tuple(tuple(row) for row in transformed)
            candidates.append((lattice_tuple, transform))
        
        # Select lexicographically smallest as canonical
        canonical = min(candidates, key=lambda x: x[0])
        return canonical[0], canonical[1]
    
    def find_symmetry_groups(self):
        """Group runs by D4 symmetry equivalence"""
        print("\nFinding symmetry groups...")
        
        for run_num, run_data in self.runs.items():
            normalized_lattice = run_data['lattice_normalized']
            canonical_form, transform = self.get_canonical_form(normalized_lattice)
            
            # Store the transformation that takes this lattice to canonical
            run_data['canonical_form'] = canonical_form
            run_data['transform_to_canonical'] = transform
            
            # Add to symmetry group
            self.symmetry_groups[canonical_form].append(run_num)
        
        print(f"Found {len(self.symmetry_groups)} unique configurations")
    
    def display_lattice(self, lattice, indent=""):
        """Pretty print a lattice"""
        for row in lattice:
            print(indent + " ".join(f"{cell:^4}" for cell in row))
    
    def print_results(self):
        """Print the analysis results"""
        print("\n" + "="*80)
        print("D4 SYMMETRY ANALYSIS RESULTS")
        print("="*80)
        
        # Sort groups by size (largest first) and then by first run number
        sorted_groups = sorted(self.symmetry_groups.items(), 
                             key=lambda x: (-len(x[1]), min(x[1])))
        
        group_num = 1
        for canonical_form, run_nums in sorted_groups:
            if len(run_nums) > 1:  # Only show groups with multiple members
                print(f"\nSymmetry Group {group_num}: {len(run_nums)} configurations")
                print("-" * 60)
                
                # Show canonical form
                canonical_array = np.array(canonical_form)
                print("Canonical form:")
                self.display_lattice(canonical_array, "  ")
                
                print("\nRuns in this group:")
                for run_num in sorted(run_nums):
                    run_data = self.runs[run_num]
                    transform = run_data['transform_to_canonical']
                    print(f"  - RUN {run_num}: {run_data['description']} "
                          f"(transform: {transform})")
                
                # Show how the first non-canonical member maps
                if len(run_nums) > 1:
                    example_run = sorted(run_nums)[1]  # Second member
                    example_data = self.runs[example_run]
                    print(f"\n  Example: RUN {example_run} original lattice:")
                    self.display_lattice(example_data['lattice_original'], "    ")
                    print(f"\n  After normalization (I_n → I):")
                    self.display_lattice(example_data['lattice_normalized'], "    ")
                    print(f"\n  After {example_data['transform_to_canonical']} → canonical form")
                
                group_num += 1
        
        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"Total runs analyzed: {len(self.runs)}")
        print(f"Unique configurations: {len(self.symmetry_groups)}")
        
        # Group size distribution
        group_sizes = [len(runs) for runs in self.symmetry_groups.values()]
        size_counts = {}
        for size in group_sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
        
        print("\nGroup size distribution:")
        for size in sorted(size_counts.keys()):
            print(f"  {size} run(s): {size_counts[size]} groups")
        
        # Singletons (unique configurations)
        singletons = [runs[0] for runs in self.symmetry_groups.values() if len(runs) == 1]
        if singletons:
            print(f"\nUnique configurations (no symmetry matches): {len(singletons)} runs")
            print("  Runs:", ", ".join(f"RUN {r}" for r in sorted(singletons)[:10]), end="")
            if len(singletons) > 10:
                print(f"... and {len(singletons) - 10} more")
            else:
                print()


def main():
    """Main execution function"""
    print("="*80)
    print("D4 SYMMETRY LOG FILE ANALYZER")
    print("="*80)
    
    # Get input file
    filepath = input("\nEnter path to parametric study log file: ").strip()
    if not filepath:
        print("No file specified!")
        return
    
    # Create analyzer and process
    analyzer = D4LogAnalyzer()
    
    try:
        analyzer.parse_log_file(filepath)
        analyzer.find_symmetry_groups()
        analyzer.print_results()
        
        # Optional: save results to file
        save_results = input("\nSave results to file? (y/n): ").strip().lower()
        if save_results == 'y':
            output_file = input("Enter output filename (or press Enter for 'symmetry_analysis.txt'): ").strip()
            if not output_file:
                output_file = "symmetry_analysis.txt"
            
            import sys
            from io import StringIO
            
            # Capture print output
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            analyzer.print_results()
            
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            with open(output_file, 'w') as f:
                f.write(output)
            
            print(f"\nResults saved to: {output_file}")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
