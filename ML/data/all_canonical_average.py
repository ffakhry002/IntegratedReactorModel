#!/usr/bin/env python3
"""
D4 Symmetry Averaging for Nuclear Reactor Predictions

This script:
1. Loads predictions from an Excel file (~270k configurations)
2. Loads configuration definitions to understand lattice structures
3. Groups configurations by D4 symmetry equivalence
4. Averages predictions within each group (accounting for position mapping)
5. Outputs a reduced Excel file with averaged results

The ML model should give identical predictions for symmetric configurations,
but in practice they vary slightly. Averaging gives more accurate results.
"""

import os
import sys
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Set, FrozenSet
from datetime import datetime
import time
from collections import defaultdict
from tqdm import tqdm


class D4SymmetryAverager:
    """Class to handle D4 symmetry grouping and averaging of reactor predictions"""

    def __init__(self):
        self.predictions_df = None
        self.configurations = {}
        self.symmetry_groups = defaultdict(list)
        self.canonical_configs = {}

    def parse_lattice(self, lattice_str):
        """Parse a lattice string into a 2D numpy array"""
        if 'core_lattice:' in lattice_str:
            lattice_str = lattice_str.split('core_lattice:')[1].strip()
        try:
            lattice_list = eval(lattice_str)
            return np.array(lattice_list, dtype='<U4')
        except:
            return None

    def load_predictions(self, excel_path):
        """Load predictions from Excel file"""
        print(f"\nLoading predictions from {excel_path}...")
        self.predictions_df = pd.read_excel(excel_path)
        print(f"Loaded {len(self.predictions_df)} predictions")

        # Extract unique column types
        flux_cols = [col for col in self.predictions_df.columns if '_flux' in col and 'average' not in col]
        print(f"Found flux columns: {len(flux_cols)}")

        return self.predictions_df

    def load_configurations(self, config_path):
        """Load reactor configurations from text file"""
        print(f"\nLoading configurations from {config_path}...")

        with open(config_path, 'r') as f:
            content = f.read()

        # Split by RUN entries
        runs = re.split(r'RUN \d+:', content)

        for run in runs[1:]:  # Skip empty first split
            lines = run.strip().split('\n')

            # Extract description
            desc_match = re.search(r'Description: ([\w_]+)', run)
            if not desc_match:
                continue

            description = desc_match.group(1)

            # Find core_lattice
            for line in lines:
                if 'core_lattice:' in line:
                    lattice = self.parse_lattice(line)
                    if lattice is not None:
                        self.configurations[description] = lattice
                    break

        print(f"Loaded {len(self.configurations)} configurations")
        return self.configurations

    def get_irradiation_positions(self, lattice):
        """Extract positions of irradiation points (I_1, I_2, etc.)"""
        positions = {}
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                if lattice[i, j].startswith('I_'):
                    label = lattice[i, j]
                    positions[label] = (i, j)
        return positions

    def apply_d4_transformation(self, pos, transform_type):
        """Apply a D4 transformation to a position"""
        i, j = pos

        if transform_type == 'identity':
            return (i, j)
        elif transform_type == 'rot90':
            return (j, 7-i)
        elif transform_type == 'rot180':
            return (7-i, 7-j)
        elif transform_type == 'rot270':
            return (7-j, i)
        elif transform_type == 'flip_h':
            return (7-i, j)
        elif transform_type == 'flip_v':
            return (i, 7-j)
        elif transform_type == 'transpose':
            return (j, i)
        elif transform_type == 'anti_diag':
            return (7-j, 7-i)

    def get_canonical_form_and_mapping(self, irr_positions):
        """
        Get canonical form of irradiation positions and the mapping
        Returns: (canonical_positions, transform_type, position_mapping)
        """
        transformations = [
            'identity', 'rot90', 'rot180', 'rot270',
            'flip_h', 'flip_v', 'transpose', 'anti_diag'
        ]

        candidates = []

        for transform in transformations:
            # Apply transformation to all positions
            transformed = {}
            for label, pos in irr_positions.items():
                new_pos = self.apply_d4_transformation(pos, transform)
                transformed[new_pos] = label

            # Sort positions to create canonical representation
            sorted_positions = sorted(transformed.keys())

            # Create tuple for comparison
            candidate = (tuple(sorted_positions), transform, transformed)
            candidates.append(candidate)

        # Select lexicographically smallest as canonical
        canonical = min(candidates, key=lambda x: x[0])

        return canonical

    def group_by_symmetry(self):
        """Group configurations by D4 symmetry equivalence"""
        print("\nGrouping configurations by D4 symmetry...")

        # Process each configuration
        for desc, lattice in tqdm(self.configurations.items(), desc="Processing configs"):
            # Get irradiation positions
            irr_positions = self.get_irradiation_positions(lattice)

            # Get canonical form and mapping
            canonical_positions, transform, position_mapping = self.get_canonical_form_and_mapping(irr_positions)

            # Use canonical positions as group key
            group_key = canonical_positions

            # Store configuration info
            config_info = {
                'description': desc,
                'lattice': lattice,
                'irr_positions': irr_positions,
                'transform': transform,
                'position_mapping': position_mapping
            }

            self.symmetry_groups[group_key].append(config_info)

            # Store first config of each group as canonical
            if group_key not in self.canonical_configs:
                self.canonical_configs[group_key] = config_info

        print(f"\nFound {len(self.symmetry_groups)} unique configurations")
        print(f"Average group size: {np.mean([len(g) for g in self.symmetry_groups.values()]):.2f}")

        # Show distribution
        group_sizes = [len(g) for g in self.symmetry_groups.values()]
        size_counts = pd.Series(group_sizes).value_counts().sort_index()
        print("\nGroup size distribution:")
        for size, count in size_counts.items():
            print(f"  {size} configs: {count} groups")

    def average_group_predictions(self, group_configs):
        """Average predictions for a group of symmetric configurations

        Key insight: We average by POSITION in canonical form, not by label.
        I_1, I_2, etc. are just labels - what matters is the physical position.
        """
        # Get canonical config (first in group)
        canonical = group_configs[0]
        canonical_positions = canonical['irr_positions']

        # Create mapping from canonical positions to their labels
        canonical_pos_to_label = {pos: label for label, pos in canonical_positions.items()}

        # Initialize aggregators by canonical POSITION (not label!)
        position_flux_data = defaultdict(lambda: defaultdict(list))
        keff_values = []
        average_flux_values = defaultdict(list)

        # Process each configuration in the group
        for config in group_configs:
            desc = config['description']

            # Find matching row in predictions
            pred_row = self.predictions_df[self.predictions_df['description'] == desc]
            if pred_row.empty:
                continue
            pred_row = pred_row.iloc[0]

            # Add k-effective
            keff_values.append(pred_row['keff'])

            # Get the transformation that takes this config to canonical form
            transform = config['transform']

            # For each irradiation position in this config
            for label, original_pos in config['irr_positions'].items():
                # Transform to canonical position
                canonical_pos = self.apply_d4_transformation(original_pos, transform)

                # Get the position number from THIS config's label
                pos_num = int(label.split('_')[1])

                # Collect flux values from THIS position in THIS config
                # and store them under the CANONICAL position
                for flux_type in ['thermal', 'epithermal', 'fast', 'total_flux']:
                    col_name = f'I_{pos_num}_{flux_type}'
                    if col_name in pred_row:
                        position_flux_data[canonical_pos][flux_type].append(pred_row[col_name])

                # Collect percentage values
                for flux_type in ['thermal', 'epithermal', 'fast']:
                    col_name = f'I_{pos_num}_{flux_type}_percent'
                    if col_name in pred_row:
                        position_flux_data[canonical_pos][f'{flux_type}_percent'].append(pred_row[col_name])

            # Collect average flux values (position-independent)
            for flux_type in ['thermal', 'epithermal', 'fast', 'total']:
                col_name = f'average_{flux_type}_flux'
                if col_name in pred_row:
                    average_flux_values[col_name].append(pred_row[col_name])

            # Min/max total flux
            for stat in ['min', 'max']:
                col_name = f'{stat}_total_flux'
                if col_name in pred_row:
                    average_flux_values[col_name].append(pred_row[col_name])

        # Build the averaged result
        averaged_data = {}
        averaged_data['config_id'] = f"Avg_{canonical['description']}"
        averaged_data['description'] = canonical['description']
        averaged_data['group_size'] = len(group_configs)

        # Average k-effective
        if keff_values:
            averaged_data['keff'] = np.mean(keff_values)

        # Average flux values by position, then assign to canonical labels
        for canonical_pos, flux_data in position_flux_data.items():
            # Find what label this position has in the canonical config
            if canonical_pos in canonical_pos_to_label:
                canonical_label = canonical_pos_to_label[canonical_pos]
                canonical_num = int(canonical_label.split('_')[1])

                # Average all flux types for this position
                for flux_type in ['thermal', 'epithermal', 'fast', 'total_flux']:
                    if flux_type in flux_data and flux_data[flux_type]:
                        col_name = f'I_{canonical_num}_{flux_type}'
                        averaged_data[col_name] = np.mean(flux_data[flux_type])

                # Average percentage values
                for flux_type in ['thermal', 'epithermal', 'fast']:
                    key = f'{flux_type}_percent'
                    if key in flux_data and flux_data[key]:
                        col_name = f'I_{canonical_num}_{flux_type}_percent'
                        averaged_data[col_name] = np.mean(flux_data[key])

        # Average position-independent values
        for col_name, values in average_flux_values.items():
            if values:
                averaged_data[col_name] = np.mean(values)

        return averaged_data


    def debug_group_averaging(self, group_configs):
        """Debug helper to show how positions map in a group"""
        print(f"\nDebug: Group with {len(group_configs)} configs")
        canonical = group_configs[0]

        # Show canonical positions
        print(f"Canonical config: {canonical['description']}")
        for label, pos in sorted(canonical['irr_positions'].items()):
            print(f"  {label} at {pos}")

        # Show how each config maps
        for i, config in enumerate(group_configs[1:], 1):
            print(f"\nConfig {i}: {config['description']} (transform: {config['transform']})")
            transform = config['transform']

            for label, original_pos in sorted(config['irr_positions'].items()):
                canonical_pos = self.apply_d4_transformation(original_pos, transform)
                # Find what's at this canonical position in the canonical config
                canonical_label = None
                for c_label, c_pos in canonical['irr_positions'].items():
                    if c_pos == canonical_pos:
                        canonical_label = c_label
                        break
                print(f"  {label} at {original_pos} → canonical pos {canonical_pos} (canonical {canonical_label})")

    def create_averaged_dataframe(self):
        """Create dataframe with averaged predictions"""
        print("\nAveraging predictions within symmetry groups...")

        averaged_rows = []

        for group_key, group_configs in tqdm(self.symmetry_groups.items(), desc="Averaging groups"):
            averaged_data = self.average_group_predictions(group_configs)

            # Find the config with the lowest number in this group
            lowest_num = float('inf')
            lowest_config_desc = group_configs[0]['description']

            for cfg in group_configs:
                desc = cfg['description']
                import re
                match = re.search(r'(\d+)', desc)
                if match:
                    num = int(match.group(1))
                    if num < lowest_num:
                        lowest_num = num
                        lowest_config_desc = desc

            # Update the averaged data with the lowest numbered description
            averaged_data['config_id'] = f"Avg_{lowest_config_desc}"
            averaged_data['description'] = lowest_config_desc

            averaged_rows.append(averaged_data)

        # Create dataframe
        averaged_df = pd.DataFrame(averaged_rows)

        # Reorder columns to match original format
        column_order = ['config_id', 'description', 'group_size', 'keff']

        # Add flux columns in order
        for i in range(1, 5):
            for flux_type in ['thermal', 'epithermal', 'fast']:
                col = f'I_{i}_{flux_type}'
                if col in averaged_df.columns:
                    column_order.append(col)

        for i in range(1, 5):
            col = f'I_{i}_total_flux'
            if col in averaged_df.columns:
                column_order.append(col)

        # Add percentage columns
        for i in range(1, 5):
            for flux_type in ['thermal', 'epithermal', 'fast']:
                col = f'I_{i}_{flux_type}_percent'
                if col in averaged_df.columns:
                    column_order.append(col)

        # Add average columns
        for flux_type in ['thermal', 'epithermal', 'fast', 'total']:
            col = f'average_{flux_type}_flux'
            if col in averaged_df.columns:
                column_order.append(col)

        # Add min/max
        for col in ['min_total_flux', 'max_total_flux']:
            if col in averaged_df.columns:
                column_order.append(col)

        # Add any remaining columns
        remaining_cols = [col for col in averaged_df.columns if col not in column_order]
        column_order.extend(remaining_cols)

        # Reorder
        averaged_df = averaged_df[[col for col in column_order if col in averaged_df.columns]]

        return averaged_df

    def save_canonical_configurations(self, output_base_path):
        """Save canonical configurations to text file in original format"""
        text_filename = output_base_path.replace('.xlsx', '_canonical_configs.txt')

        print(f"\nSaving canonical configurations to {text_filename}...")

        with open(text_filename, 'w') as f:
            run_num = 1

            # Sort groups by the lowest config number in each group
            def get_lowest_config_number(group_configs):
                """Extract the lowest configuration number from a group"""
                numbers = []
                for cfg in group_configs:
                    desc = cfg['description']
                    # Try to extract number from descriptions like 'core_config_123' or 'config_123'
                    import re
                    match = re.search(r'(\d+)', desc)
                    if match:
                        numbers.append(int(match.group(1)))
                return min(numbers) if numbers else float('inf')

            # Sort groups by their lowest configuration number
            sorted_groups = sorted(self.symmetry_groups.items(),
                                key=lambda x: get_lowest_config_number(x[1]))

            for group_key, group_configs in sorted_groups:
                canonical = self.canonical_configs[group_key]

                # Find the config with the lowest number in this group
                lowest_num = float('inf')
                lowest_config_desc = canonical['description']

                for cfg in group_configs:
                    desc = cfg['description']
                    match = re.search(r'(\d+)', desc)
                    if match:
                        num = int(match.group(1))
                        if num < lowest_num:
                            lowest_num = num
                            lowest_config_desc = desc

                # Write in original format
                f.write(f"RUN {run_num}:\n")
                f.write("-" * 40 + "\n")
                # Use the lowest numbered config's description
                f.write(f"Description: {lowest_config_desc}\n")

                # Convert lattice to list format for output
                lattice_list = canonical['lattice'].tolist()
                f.write(f"  core_lattice: {lattice_list}\n")

                # Add symmetry group information
                f.write(f"  // Symmetry group size: {len(group_configs)}\n")
                # Show ALL group members, no truncation
                f.write(f"  // Group members: {', '.join([cfg['description'] for cfg in group_configs])}\n")

                # Show irradiation positions clearly
                f.write(f"  // Irradiation positions:\n")
                for label, pos in sorted(canonical['irr_positions'].items()):
                    f.write(f"  //   {label} at position {pos}\n")

                f.write("=" * 80 + "\n\n")
                f.write("=" * 80 + "\n\n")

                run_num += 1

        print(f"✓ Saved {len(self.symmetry_groups)} canonical configurations")
        return text_filename

    def save_results(self, averaged_df, output_path):
        """Save averaged results to Excel file"""
        print(f"\nSaving results to {output_path}...")

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            averaged_df.to_excel(writer, sheet_name='Averaged_Predictions', index=False)

            # Format the worksheet
            ws = writer.sheets['Averaged_Predictions']

            # Auto-adjust column widths
            for column in ws.columns:
                max_length = 0
                column_letter = column[0].column_letter

                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass

                adjusted_width = min(max_length + 2, 25)
                ws.column_dimensions[column_letter].width = adjusted_width

        print(f"✓ Saved {len(averaged_df)} averaged configurations")

        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"  Original configurations: {len(self.predictions_df)}")
        print(f"  Unique configurations: {len(averaged_df)}")
        print(f"  Reduction factor: {len(self.predictions_df)/len(averaged_df):.1f}x")
        print(f"  Average group size: {averaged_df['group_size'].mean():.2f}")
        print(f"  Max group size: {averaged_df['group_size'].max()}")
        print(f"  Min group size: {averaged_df['group_size'].min()}")

def main():
    """Main execution function"""
    print("="*80)
    print("D4 SYMMETRY AVERAGING FOR REACTOR PREDICTIONS")
    print("="*80)

    # Get input files
    excel_path = input("\nEnter path to predictions Excel file (270k configs): ").strip()
    if not excel_path:
        print("No Excel file specified!")
        return

    config_path = input("Enter path to configurations text file: ").strip()
    if not config_path:
        print("No configuration file specified!")
        return

    # Validate files exist
    if not os.path.exists(excel_path):
        print(f"Error: Excel file '{excel_path}' not found!")
        return

    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found!")
        return

    # Get output filename
    output_path = input("\nEnter output Excel filename (or press Enter for timestamp): ").strip()
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"averaged_predictions_{timestamp}.xlsx"
    elif not output_path.endswith('.xlsx'):
        output_path += '.xlsx'

    # Create averager and process
    averager = D4SymmetryAverager()

    try:
        # Load data
        averager.load_predictions(excel_path)
        averager.load_configurations(config_path)

        # Group by symmetry
        averager.group_by_symmetry()

        # Average predictions
        averaged_df = averager.create_averaged_dataframe()

        # Save results
        averager.save_results(averaged_df, output_path)

        # Save canonical configurations text file
        canonical_text_path = averager.save_canonical_configurations(output_path)

        print("\n" + "="*80)
        print("AVERAGING COMPLETE!")
        print("="*80)
        print(f"\nOutput files:")
        print(f"  - Excel with averaged predictions: {output_path}")
        print(f"  - Canonical configurations: {canonical_text_path}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
