#!/usr/bin/env python3
"""
ML Model Symmetry Deviation Analysis - FULLY FIXED VERSION
Now properly maps ALL 270k configurations through D4 transformations
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better performance
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import defaultdict
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

class SymmetryDeviationAnalyzer:
    def __init__(self):
        self.full_predictions_df = None
        self.averaged_df = None
        self.group_memberships = {}
        self.all_lattices = {}  # Store ALL 270k lattices
        self.canonical_lattices = {}  # Store canonical lattice for each group
        self.transformation_map = {}  # Store transformation for each config to its canonical
        self.deviation_stats = {}
        self.output_dir = None
        self.full_predictions_dict = {}

    def setup_output_directory(self):
        """Create output directory for visualizations"""
        ml_data_dir = Path("ML/data")
        self.output_dir = ml_data_dir / "deviation_visualizations"
        self.output_dir.mkdir(exist_ok=True)
        print(f"✓ Output directory: {self.output_dir}")

    def parse_lattice_string(self, lattice_str):
        """Parse a lattice string into a numpy array"""
        try:
            # Clean up the string
            lattice_str = re.sub(r'\s+', ' ', lattice_str)
            lattice_list = eval(lattice_str)
            return np.array(lattice_list, dtype='<U10')
        except:
            return None

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

    def get_irradiation_positions(self, lattice):
        """Extract positions of irradiation points (I_1, I_2, etc.)"""
        positions = {}
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                if lattice[i, j].startswith('I_'):
                    label = lattice[i, j]
                    positions[label] = (i, j)
        return positions

    def find_transformation_to_canonical(self, config_positions, canonical_positions):
        """Find which D4 transformation maps this config to canonical"""
        transformations = [
            'identity', 'rot90', 'rot180', 'rot270',
            'flip_h', 'flip_v', 'transpose', 'anti_diag'
        ]

        # Get just the position sets (ignore labels)
        config_pos_set = set(config_positions.values())
        canonical_pos_set = set(canonical_positions.values())

        for transform in transformations:
            # Apply transformation to config positions
            transformed_positions = set()
            for pos in config_pos_set:
                new_pos = self.apply_d4_transformation(pos, transform)
                transformed_positions.add(new_pos)

            # Check if transformed positions match canonical positions
            if transformed_positions == canonical_pos_set:
                return transform

        return None

    def create_position_mapping(self, config_positions, canonical_positions, transform):
        """Create mapping of which config position corresponds to which canonical position"""
        mapping = {}

        for config_label, config_pos in config_positions.items():
            # Transform the config position to canonical space
            transformed_pos = self.apply_d4_transformation(config_pos, transform)

            # Find which canonical label is at this position
            for canon_label, canon_pos in canonical_positions.items():
                if canon_pos == transformed_pos:
                    mapping[config_label] = canon_label
                    break

        return mapping

    def load_all_configurations(self, full_config_path):
        """Load ALL 270k configurations from the original text file"""
        print(f"\nLoading ALL configurations from: {full_config_path}")

        with open(full_config_path, 'r') as f:
            content = f.read()

        # Split by RUN entries - your exact format
        runs = re.split(r'RUN \d+:', content)

        print(f"Found {len(runs)-1} RUN entries in file")

        failed_parses = 0
        successful_parses = 0

        for run_idx, run in enumerate(tqdm(runs[1:], desc="Loading all lattices"), 1):
            if not run.strip():
                continue

            try:
                # Extract description - your exact format
                desc_match = re.search(r'Description:\s*([\w_]+)', run)
                if not desc_match:
                    failed_parses += 1
                    if failed_parses <= 5:
                        print(f"  Failed to find description in RUN {run_idx}")
                    continue

                description = desc_match.group(1).strip()

                # Extract lattice - handle the indentation and equals signs
                # The lattice is between "core_lattice:" and the first "===" line
                lattice_match = re.search(r'core_lattice:\s*(\[.*?\])\s*\n=', run, re.DOTALL)

                if not lattice_match:
                    failed_parses += 1
                    if failed_parses <= 5:
                        print(f"  Failed to find lattice in RUN {run_idx} (desc: {description})")
                    continue

                lattice_str = lattice_match.group(1)

                # Parse the lattice string
                lattice = self.parse_lattice_string(lattice_str)

                if lattice is not None:
                    self.all_lattices[description] = lattice
                    successful_parses += 1
                else:
                    failed_parses += 1
                    if failed_parses <= 5:
                        print(f"  Failed to parse lattice string for {description}")

            except Exception as e:
                failed_parses += 1
                if failed_parses <= 5:
                    print(f"  Error processing RUN {run_idx}: {str(e)}")

        print(f"✓ Successfully loaded {len(self.all_lattices)} lattice configurations")
        print(f"  Successful parses: {successful_parses}")
        print(f"  Failed parses: {failed_parses}")

        if len(self.all_lattices) == 0:
            print("\n❌ ERROR: No lattices were loaded!")
            print("Debugging: Trying to parse first RUN manually...")

            # Try to debug the first RUN
            if len(runs) > 1:
                first_run = runs[1]
                print(f"First RUN content (first 500 chars):\n{first_run[:500]}")

                # Try to extract description
                desc_match = re.search(r'Description:\s*([\w_]+)', first_run)
                if desc_match:
                    print(f"  Found description: {desc_match.group(1)}")
                else:
                    print("  Could not find description")

                # Try to extract lattice
                if 'core_lattice:' in first_run:
                    print("  Found 'core_lattice:' keyword")
                    lattice_match = re.search(r'core_lattice:\s*(\[.*?\])', first_run, re.DOTALL)
                    if lattice_match:
                        print(f"  Found lattice match (first 100 chars): {lattice_match.group(1)[:100]}")
                    else:
                        print("  Could not match lattice pattern")
                else:
                    print("  'core_lattice:' keyword not found")
        else:
            # Show some sample loaded configurations
            print("\nSample loaded configurations:")
            for name in list(self.all_lattices.keys())[:5]:
                print(f"  - {name}")

    def parse_canonical_groups(self, canonical_txt_path):
        """Parse the canonical text file to extract group memberships AND canonical lattices"""
        print(f"\nParsing canonical groups from: {canonical_txt_path}")

        with open(canonical_txt_path, 'r') as f:
            content = f.read()

        # Split by RUN entries - same as the full file format
        runs = re.split(r'RUN \d+:', content)

        print(f"Processing {len(runs)-1} canonical groups...")

        successful_lattices = 0
        failed_lattices = 0

        for run in tqdm(runs[1:], desc="Parsing canonical groups"):
            if not run.strip():
                continue

            # Extract main description
            desc_match = re.search(r'Description:\s*([\w_]+)', run)
            if not desc_match:
                continue
            canonical_desc = desc_match.group(1).strip()

            # Extract lattice - SAME FORMAT AS FULL FILE
            lattice_match = re.search(r'core_lattice:\s*(\[.*?\])\s*(?:\n|//)', run, re.DOTALL)
            if not lattice_match:
                # Try with === separator
                lattice_match = re.search(r'core_lattice:\s*(\[.*?\])\s*\n=', run, re.DOTALL)

            if lattice_match:
                lattice_str = lattice_match.group(1)
                lattice = self.parse_lattice_string(lattice_str)
                if lattice is not None:
                    self.canonical_lattices[canonical_desc] = lattice
                    successful_lattices += 1
                else:
                    failed_lattices += 1
                    if failed_lattices <= 3:
                        print(f"  Failed to parse lattice for {canonical_desc}")
            else:
                failed_lattices += 1
                if failed_lattices <= 3:
                    print(f"  No lattice found for {canonical_desc}")
                    # Debug: show what we're trying to parse
                    if failed_lattices == 1:
                        print(f"  Sample run content (first 500 chars):\n{run[:500]}")

            # Extract group members - handle comment format
            members_match = re.search(r'//\s*Group members:\s*([^\n]+)', run)
            if members_match:
                members_str = members_match.group(1)
                members = [m.strip() for m in members_str.split(',')]
                self.group_memberships[canonical_desc] = members
            else:
                # If no group members comment, this is a singleton group
                self.group_memberships[canonical_desc] = [canonical_desc]

        print(f"✓ Found {len(self.group_memberships)} canonical groups")
        print(f"✓ Loaded {successful_lattices} canonical lattices")
        if failed_lattices > 0:
            print(f"  WARNING: Failed to load {failed_lattices} lattices")

        # Show statistics
        group_sizes = [len(members) for members in self.group_memberships.values()]
        print(f"  Average group size: {np.mean(group_sizes):.1f}")
        print(f"  Max group size: {max(group_sizes)}")
        print(f"  Min group size: {min(group_sizes)}")

        # Debug: Show sample canonical lattices
        if self.canonical_lattices:
            print(f"\nSample canonical lattices loaded:")
            for name in list(self.canonical_lattices.keys())[:5]:
                print(f"  - {name}")
        else:
            print("\n❌ ERROR: No canonical lattices were loaded!")
            print("This will prevent position mapping from working.")

    def compute_all_transformations(self):
        """Compute transformation for each config to its canonical form"""
        print("\nComputing transformations for all configurations...")

        # Debug: Check what we have
        print(f"Debug: We have {len(self.canonical_lattices)} canonical lattices")
        print(f"Debug: We have {len(self.all_lattices)} total lattices")
        print(f"Debug: We have {len(self.group_memberships)} groups")

        # Show sample names to debug matching issues
        if self.canonical_lattices:
            print(f"Sample canonical names: {list(self.canonical_lattices.keys())[:5]}")
        if self.all_lattices:
            print(f"Sample lattice names: {list(self.all_lattices.keys())[:5]}")
        if self.group_memberships:
            sample_group = list(self.group_memberships.items())[0]
            print(f"Sample group: {sample_group[0]} has members: {sample_group[1][:5]}...")

        matched_configs = 0
        unmatched_canonicals = 0
        unmatched_members = 0

        for canonical_desc, member_descs in tqdm(self.group_memberships.items(),
                                                 desc="Computing transformations"):
            if canonical_desc not in self.canonical_lattices:
                unmatched_canonicals += 1
                if unmatched_canonicals <= 3:
                    print(f"  Warning: Canonical '{canonical_desc}' not in canonical_lattices")
                continue

            canonical_lattice = self.canonical_lattices[canonical_desc]
            canonical_positions = self.get_irradiation_positions(canonical_lattice)

            for member_desc in member_descs:
                if member_desc not in self.all_lattices:
                    unmatched_members += 1
                    if unmatched_members <= 3:
                        print(f"  Warning: Member '{member_desc}' not in all_lattices")
                    continue

                member_lattice = self.all_lattices[member_desc]
                member_positions = self.get_irradiation_positions(member_lattice)

                # Find transformation
                transform = self.find_transformation_to_canonical(member_positions, canonical_positions)

                if transform:
                    # Create position mapping
                    position_mapping = self.create_position_mapping(
                        member_positions, canonical_positions, transform
                    )

                    self.transformation_map[member_desc] = {
                        'canonical': canonical_desc,
                        'transform': transform,
                        'position_mapping': position_mapping
                    }
                    matched_configs += 1
                else:
                    if matched_configs == 0 and unmatched_members < 10:
                        print(f"  Warning: No valid transformation found for '{member_desc}'")

        print(f"✓ Computed transformations for {len(self.transformation_map)} configurations")
        print(f"  Matched configs: {matched_configs}")
        print(f"  Unmatched canonicals: {unmatched_canonicals}")
        print(f"  Unmatched members: {unmatched_members}")

        if len(self.transformation_map) == 0:
            print("\n❌ No transformations computed - likely a naming mismatch!")
            print("Common causes:")
            print("  1. The canonical file refers to configs not in the full file")
            print("  2. Different naming conventions between files")
            print("  3. The canonical groups file might be from a different dataset")

            # Try to diagnose the issue
            if self.group_memberships:
                # Get all member names from groups
                all_member_names = set()
                for members in self.group_memberships.values():
                    all_member_names.update(members)

                # Check overlap with loaded lattices
                lattice_names = set(self.all_lattices.keys())
                overlap = all_member_names & lattice_names

                print(f"\nDiagnostic:")
                print(f"  Total unique members in groups: {len(all_member_names)}")
                print(f"  Total lattices loaded: {len(lattice_names)}")
                print(f"  Overlap: {len(overlap)} configs")

                if len(overlap) == 0:
                    print("  ⚠️ ZERO overlap - the files are incompatible!")
                    print("\n  Sample group members (first 10):")
                    for name in list(all_member_names)[:10]:
                        print(f"    {name}")
                    print("\n  Sample lattice names (first 10):")
                    for name in list(lattice_names)[:10]:
                        print(f"    {name}")

    def load_predictions(self, full_excel_path, averaged_excel_path):
        """Load both prediction Excel files"""
        print(f"\nLoading full predictions from: {full_excel_path}")
        start_time = time.time()
        self.full_predictions_df = pd.read_excel(full_excel_path)
        print(f"✓ Loaded {len(self.full_predictions_df)} full predictions in {time.time()-start_time:.1f}s")

        # Create dictionary for faster lookups
        print("Creating lookup dictionary for faster processing...")
        self.full_predictions_dict = {
            row['description']: row.to_dict()
            for _, row in tqdm(self.full_predictions_df.iterrows(),
                              total=len(self.full_predictions_df),
                              desc="Building lookup dict")
        }

        print(f"\nLoading averaged predictions from: {averaged_excel_path}")
        start_time = time.time()
        self.averaged_df = pd.read_excel(averaged_excel_path)
        print(f"✓ Loaded {len(self.averaged_df)} averaged predictions in {time.time()-start_time:.1f}s")

    def calculate_deviations_with_proper_mapping(self):
        """Calculate deviation statistics with PROPER position mapping for ALL configs"""
        print("\nCalculating deviation statistics with full position mapping...")

        # Identify which columns we have
        flux_types = []
        for col in ['thermal', 'epithermal', 'fast']:
            if f'I_1_{col}' in self.full_predictions_df.columns:
                flux_types.append(col)

        has_keff = 'keff' in self.full_predictions_df.columns

        print(f"  Found flux types: {flux_types}")
        print(f"  Has k-eff: {has_keff}")

        # Debug: Check data alignment
        print(f"\nDebug Info:")
        print(f"  Total predictions in Excel: {len(self.full_predictions_dict)}")
        print(f"  Total lattices loaded: {len(self.all_lattices)}")
        print(f"  Total transformations computed: {len(self.transformation_map)}")
        print(f"  Total canonical groups: {len(self.group_memberships)}")

        # Check for naming mismatches
        excel_names = set(self.full_predictions_dict.keys())
        lattice_names = set(self.all_lattices.keys())
        overlap = excel_names & lattice_names
        print(f"  Configs in both Excel and lattices: {len(overlap)}")

        if len(overlap) < len(excel_names):
            print(f"  WARNING: {len(excel_names - lattice_names)} configs in Excel but not in lattices")
            # Show a few examples
            examples = list(excel_names - lattice_names)[:5]
            if examples:
                print(f"    Examples: {examples}")

        # Create averaged lookup dict
        averaged_dict = {
            row['description']: row.to_dict()
            for _, row in self.averaged_df.iterrows()
        }

        # Process each canonical group
        skipped_groups = 0
        processed_groups = 0
        no_lattice_groups = 0
        no_transform_members = 0

        for canonical_desc, member_descs in tqdm(self.group_memberships.items(),
                                                 desc="Processing groups with full mapping"):

            if canonical_desc not in self.canonical_lattices:
                no_lattice_groups += 1
                continue

            # Get canonical lattice and its irradiation positions
            canonical_lattice = self.canonical_lattices[canonical_desc]
            canonical_positions = self.get_irradiation_positions(canonical_lattice)

            # Get averaged values for this group
            if canonical_desc not in averaged_dict:
                skipped_groups += 1
                continue

            avg_data = averaged_dict[canonical_desc]

            # Collect PROPERLY MAPPED values for each canonical position
            canonical_position_values = defaultdict(lambda: defaultdict(list))
            keff_values = []
            members_with_transform = 0

            # Process each member with proper mapping
            for member_desc in member_descs:
                if member_desc not in self.full_predictions_dict:
                    continue

                member_data = self.full_predictions_dict[member_desc]

                # Add k-eff (position-independent)
                if has_keff and 'keff' in member_data:
                    keff_values.append(member_data['keff'])

                # Get position mapping for this member
                if member_desc not in self.transformation_map:
                    no_transform_members += 1
                    continue

                members_with_transform += 1
                position_mapping = self.transformation_map[member_desc]['position_mapping']

                # Map flux values through the transformation
                for member_label, canonical_label in position_mapping.items():
                    member_label_num = int(member_label.split('_')[1])
                    canonical_label_num = int(canonical_label.split('_')[1])

                    for flux_type in flux_types:
                        member_col = f'I_{member_label_num}_{flux_type}'

                        if member_col in member_data and pd.notna(member_data[member_col]):
                            # Store under the CANONICAL position
                            canonical_position_values[canonical_label][flux_type].append(
                                member_data[member_col]
                            )

            # Only process if we have some members with transformations
            if members_with_transform == 0:
                skipped_groups += 1
                continue

            # Calculate statistics for properly mapped values
            group_stats = {}

            # K-effective statistics
            if keff_values:
                keff_avg = avg_data.get('keff', np.mean(keff_values))
                if keff_avg > 0:
                    keff_deviations = np.abs(np.array(keff_values) - keff_avg) / keff_avg * 100
                    group_stats['keff'] = {
                        'values': np.array(keff_values),
                        'average': keff_avg,
                        'deviations_pct': keff_deviations,
                        'mean_dev': np.mean(keff_deviations),
                        'max_dev': np.max(keff_deviations),
                        'min_dev': np.min(keff_deviations),
                        'std_dev': np.std(keff_deviations),
                        'cv': np.std(keff_values) / keff_avg
                    }

            # Flux statistics by CANONICAL position
            for canonical_label in canonical_positions.keys():
                canonical_num = int(canonical_label.split('_')[1])

                for flux_type in flux_types:
                    if canonical_label in canonical_position_values:
                        if flux_type in canonical_position_values[canonical_label]:
                            flux_values = np.array(canonical_position_values[canonical_label][flux_type])

                            col_name = f'I_{canonical_num}_{flux_type}'

                            if col_name in avg_data:
                                flux_avg = avg_data[col_name]
                            else:
                                flux_avg = np.mean(flux_values) if len(flux_values) > 0 else 0

                            if flux_avg > 0 and len(flux_values) > 0:
                                flux_deviations = np.abs(flux_values - flux_avg) / flux_avg * 100
                                group_stats[col_name] = {
                                    'values': flux_values,
                                    'average': flux_avg,
                                    'deviations_pct': flux_deviations,
                                    'mean_dev': np.mean(flux_deviations),
                                    'max_dev': np.max(flux_deviations),
                                    'min_dev': np.min(flux_deviations),
                                    'std_dev': np.std(flux_deviations),
                                    'cv': np.std(flux_values) / flux_avg,
                                    'n_values': len(flux_values)  # Track how many values we have
                                }

            if group_stats:
                self.deviation_stats[canonical_desc] = group_stats
                processed_groups += 1

        print(f"\n✓ Calculated deviations for {processed_groups} groups")
        print(f"  Groups skipped (no lattice): {no_lattice_groups}")
        print(f"  Groups skipped (no avg data): {skipped_groups}")
        print(f"  Total members without transformations: {no_transform_members}")

        if processed_groups == 0:
            print("\n❌ ERROR: No groups were successfully processed!")
            print("Possible causes:")
            print("  1. Configuration names don't match between files")
            print("  2. Lattice parsing failed")
            print("  3. Transformation computation failed")
            print("\nTrying to diagnose naming issues...")

            # Show sample names from each source
            print("\nSample names from Excel predictions:")
            for name in list(self.full_predictions_dict.keys())[:5]:
                print(f"    {name}")

            print("\nSample names from lattice file:")
            for name in list(self.all_lattices.keys())[:5]:
                print(f"    {name}")

            print("\nSample names from canonical groups:")
            for name in list(self.group_memberships.keys())[:5]:
                print(f"    {name}")

        # Print diagnostic info
        print("\nDiagnostic Summary:")
        total_mapped = sum(1 for d in self.deviation_stats.values()
                          for k, v in d.items()
                          if k != 'keff' and 'n_values' in v)
        print(f"  Total position-flux combinations with proper mapping: {total_mapped}")

    def create_deviation_summary_plots(self):
        """Create summary plots instead of bar charts with 41k bars"""
        print("\nCreating deviation summary plots...")

        if not self.deviation_stats:
            print("  WARNING: No deviation statistics available, skipping summary plots")
            return

        quantities = ['keff', 'thermal', 'epithermal', 'fast']

        # Collect summary statistics for each quantity
        summary_stats = {q: {'avg_devs': [], 'max_devs': []} for q in quantities}

        for group_name, group_stats in self.deviation_stats.items():
            if 'keff' in group_stats:
                summary_stats['keff']['avg_devs'].append(group_stats['keff']['mean_dev'])
                summary_stats['keff']['max_devs'].append(group_stats['keff']['max_dev'])

            for flux_type in ['thermal', 'epithermal', 'fast']:
                position_avg_devs = []
                position_max_devs = []
                for i in range(1, 5):
                    col_name = f'I_{i}_{flux_type}'
                    if col_name in group_stats:
                        position_avg_devs.append(group_stats[col_name]['mean_dev'])
                        position_max_devs.append(group_stats[col_name]['max_dev'])

                if position_avg_devs:
                    summary_stats[flux_type]['avg_devs'].append(np.mean(position_avg_devs))
                    summary_stats[flux_type]['max_devs'].append(np.mean(position_max_devs))

        # Create violin plots instead of bar charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for idx, quantity in enumerate(quantities):
            ax = axes[idx]

            if summary_stats[quantity]['avg_devs']:
                data_to_plot = [
                    summary_stats[quantity]['avg_devs'],
                    summary_stats[quantity]['max_devs']
                ]

                bp = ax.boxplot(data_to_plot, labels=['Avg Dev', 'Max Dev'],
                               patch_artist=True, showmeans=True)

                # Color the boxes
                colors = ['lightblue', 'lightcoral']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)

                # Add violin plot overlay
                parts = ax.violinplot(data_to_plot, positions=[1, 2],
                                     widths=0.7, showmeans=False, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor('gray')
                    pc.set_alpha(0.3)

                ax.set_title(f'{quantity.capitalize()} - Deviation Distribution',
                           fontsize=12, fontweight='bold')
                ax.set_ylabel('Deviation (%)')
                ax.grid(True, alpha=0.3, axis='y')

                # Add statistics text
                avg_mean = np.mean(summary_stats[quantity]['avg_devs'])
                max_mean = np.mean(summary_stats[quantity]['max_devs'])
                ax.text(0.95, 0.95, f'Mean of avg: {avg_mean:.2f}%\nMean of max: {max_mean:.2f}%',
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, f'No data for {quantity}',
                       transform=ax.transAxes, ha='center', va='center')

        plt.suptitle('Deviation Summary Across All Groups (With Proper Position Mapping)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'deviation_summary.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("✓ Saved deviation summary plots")

    def create_deviation_histograms(self):
        """Create histograms of deviation distributions"""
        if not self.deviation_stats:
            print("  WARNING: No deviation statistics available, skipping histograms")
            return

        quantities = ['keff', 'thermal', 'epithermal', 'fast']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for idx, quantity in enumerate(quantities):
            ax = axes[idx]

            # Collect all deviations for this quantity
            all_deviations = []

            for group_stats in self.deviation_stats.values():
                if quantity == 'keff':
                    if 'keff' in group_stats:
                        all_deviations.extend(group_stats['keff']['deviations_pct'])
                else:
                    for i in range(1, 5):
                        col_name = f'I_{i}_{quantity}'
                        if col_name in group_stats:
                            all_deviations.extend(group_stats[col_name]['deviations_pct'])

            if all_deviations:
                ax.hist(all_deviations, bins=100, alpha=0.7, edgecolor='black')
                ax.axvline(x=1, color='r', linestyle='--', alpha=0.7, label='1%')
                ax.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='5%')

                mean_dev = np.mean(all_deviations)
                median_dev = np.median(all_deviations)
                p95 = np.percentile(all_deviations, 95)

                ax.set_title(f'{quantity.capitalize()} Deviation Distribution',
                           fontsize=12, fontweight='bold')
                ax.set_xlabel('Deviation from Average (%)')
                ax.set_ylabel('Frequency')
                ax.legend()

                stats_text = f'Mean: {mean_dev:.2f}%\nMedian: {median_dev:.2f}%\n95th %: {p95:.2f}%'
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       verticalalignment='top', horizontalalignment='right')

        plt.suptitle('Deviation Distributions (With Proper Position Mapping)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'deviation_histograms.png', dpi=150, bbox_inches='tight')
        plt.close()

    def create_cv_analysis(self):
        """Create coefficient of variation analysis"""
        if not self.deviation_stats:
            print("  WARNING: No deviation statistics available, skipping CV analysis")
            return

        quantities = ['keff', 'thermal', 'epithermal', 'fast']
        cv_data = {q: [] for q in quantities}

        for group_stats in self.deviation_stats.values():
            if 'keff' in group_stats:
                cv_data['keff'].append(group_stats['keff']['cv'])

            for flux_type in ['thermal', 'epithermal', 'fast']:
                cvs = []
                for i in range(1, 5):
                    col_name = f'I_{i}_{flux_type}'
                    if col_name in group_stats:
                        cvs.append(group_stats[col_name]['cv'])
                if cvs:
                    cv_data[flux_type].append(np.mean(cvs))

        fig, ax = plt.subplots(figsize=(10, 6))

        box_data = [cv_data[q] for q in quantities if cv_data[q]]
        labels = [q.capitalize() for q in quantities if cv_data[q]]

        if box_data:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)

            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)

        ax.set_title('Coefficient of Variation Distribution by Quantity',
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Coefficient of Variation')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='1% CV threshold')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'cv_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    def create_correlation_matrix(self):
        """Create correlation matrix of deviations"""
        if not self.deviation_stats:
            print("  WARNING: No deviation statistics available, skipping correlation matrix")
            return

        deviation_df = []

        for group_name, group_stats in self.deviation_stats.items():
            row = {'group': group_name}

            if 'keff' in group_stats:
                row['keff_dev'] = group_stats['keff']['mean_dev']

            for flux_type in ['thermal', 'epithermal', 'fast']:
                devs = []
                for i in range(1, 5):
                    col_name = f'I_{i}_{flux_type}'
                    if col_name in group_stats:
                        devs.append(group_stats[col_name]['mean_dev'])
                if devs:
                    row[f'{flux_type}_dev'] = np.mean(devs)

            if len(row) > 1:
                deviation_df.append(row)

        if deviation_df:
            df = pd.DataFrame(deviation_df)

            corr_cols = [col for col in df.columns if col != 'group']
            if len(corr_cols) > 1:
                corr_matrix = df[corr_cols].corr()

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                           center=0, square=True, linewidths=1,
                           cbar_kws={"shrink": 0.8})

                ax.set_title('Correlation Matrix of Deviations',
                            fontsize=14, fontweight='bold')

                labels = [l.replace('_dev', '') for l in corr_matrix.columns]
                ax.set_xticklabels([l.capitalize() for l in labels])
                ax.set_yticklabels([l.capitalize() for l in labels])

                plt.tight_layout()
                plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
                plt.close()

    def create_cumulative_distribution(self):
        """Create cumulative distribution functions"""
        if not self.deviation_stats:
            print("  WARNING: No deviation statistics available, skipping cumulative distribution")

            # Create placeholder plot
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No deviation data available for cumulative distribution',
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_xlabel('Maximum Deviation (%)', fontsize=12)
            ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
            ax.set_title('Cumulative Distribution - No Data Available', fontsize=14, fontweight='bold')
            plt.savefig(self.output_dir / 'cumulative_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            return

        quantities = ['keff', 'thermal', 'epithermal', 'fast']

        fig, ax = plt.subplots(figsize=(12, 8))

        for quantity in quantities:
            max_deviations = []

            for group_stats in self.deviation_stats.values():
                if quantity == 'keff':
                    if 'keff' in group_stats:
                        max_deviations.append(group_stats['keff']['max_dev'])
                else:
                    for i in range(1, 5):
                        col_name = f'I_{i}_{quantity}'
                        if col_name in group_stats:
                            max_deviations.append(group_stats[col_name]['max_dev'])

            if max_deviations:
                sorted_devs = np.sort(max_deviations)
                cumulative = np.arange(1, len(sorted_devs) + 1) / len(sorted_devs) * 100
                ax.plot(sorted_devs, cumulative, label=quantity.capitalize(), linewidth=2)

        ax.set_xlabel('Maximum Deviation (%)', fontsize=12)
        ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
        ax.set_title('Cumulative Distribution of Maximum Deviations',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax.axvline(x=1, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=5, color='orange', linestyle='--', alpha=0.5)
        ax.axhline(y=95, color='gray', linestyle=':', alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'cumulative_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

    def create_summary_table(self):
        """Create and save summary statistics table"""
        quantities = ['keff', 'thermal', 'epithermal', 'fast']
        summary_data = []

        for quantity in quantities:
            all_deviations = []
            all_max_devs = []

            for group_stats in self.deviation_stats.values():
                if quantity == 'keff':
                    if 'keff' in group_stats:
                        all_deviations.extend(group_stats['keff']['deviations_pct'])
                        all_max_devs.append(group_stats['keff']['max_dev'])
                else:
                    for i in range(1, 5):
                        col_name = f'I_{i}_{quantity}'
                        if col_name in group_stats:
                            all_deviations.extend(group_stats[col_name]['deviations_pct'])
                            all_max_devs.append(group_stats[col_name]['max_dev'])

            if all_deviations:
                summary_data.append({
                    'Quantity': quantity.capitalize(),
                    'Mean Dev (%)': np.mean(all_deviations),
                    'Median Dev (%)': np.median(all_deviations),
                    '95th % (%)': np.percentile(all_deviations, 95),
                    '99th % (%)': np.percentile(all_deviations, 99),
                    'Max Dev (%)': np.max(all_deviations),
                    '% < 1%': np.sum(np.array(all_deviations) < 1) / len(all_deviations) * 100,
                    '% < 5%': np.sum(np.array(all_deviations) < 5) / len(all_deviations) * 100
                })

        summary_df = pd.DataFrame(summary_data)

        # Save to Excel
        summary_df.to_excel(self.output_dir / 'summary_statistics.xlsx', index=False)

        # Create visual table
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')

        display_df = summary_df.round(2)

        table = ax.table(cellText=display_df.values,
                        colLabels=display_df.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.12] * len(display_df.columns))

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        for i in range(len(display_df.columns)):
            table[(0, i)].set_facecolor('#366092')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.title('Summary Statistics of Symmetry Deviations (Properly Mapped)',
                 fontsize=14, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / 'summary_table.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("\n" + "="*60)
        print("SUMMARY TABLE:")
        print("="*60)
        print(summary_df.to_string(index=False))

    def run_analysis(self, full_excel, averaged_excel, canonical_txt, full_config_txt):
        """Run complete analysis pipeline with full position mapping"""
        print("\n" + "="*60)
        print("SYMMETRY DEVIATION ANALYSIS - FULLY POSITION-AWARE VERSION")
        print("="*60)

        start_time = time.time()

        # Setup
        self.setup_output_directory()

        # Load ALL configurations
        self.load_all_configurations(full_config_txt)

        # Check if we loaded any configurations
        if len(self.all_lattices) == 0:
            print("\n❌ FATAL ERROR: No configurations were loaded from the file!")
            print("Cannot proceed with analysis.")
            print("\nPlease check:")
            print("  1. The file path is correct")
            print("  2. The file contains configurations in the expected format")
            print("  3. The file is not empty or corrupted")
            return

        # Load canonical groups
        self.parse_canonical_groups(canonical_txt)

        # Compute all transformations
        self.compute_all_transformations()

        # Check if we computed any transformations
        if len(self.transformation_map) == 0:
            print("\n⚠️ WARNING: No transformations were computed!")
            print("This means configurations couldn't be mapped to their canonical forms.")
            print("Proceeding with limited analysis...")

        # Load predictions
        self.load_predictions(full_excel, averaged_excel)

        # Calculate deviations with PROPER mapping
        self.calculate_deviations_with_proper_mapping()

        # Create visualizations
        print("\nGenerating visualizations...")
        self.create_deviation_summary_plots()
        self.create_deviation_histograms()
        self.create_cv_analysis()
        self.create_correlation_matrix()
        self.create_cumulative_distribution()
        self.create_summary_table()

        total_time = time.time() - start_time

        print("\n" + "="*60)
        print(f"✓ Analysis complete in {total_time:.1f} seconds!")
        print(f"  All visualizations saved to: {self.output_dir}")
        print("="*60)

        # Print explanation of the complete fix
        if len(self.deviation_stats) > 0:
            pass
        else:
            print("\n⚠️ No deviation statistics were calculated.")
            print("Check the debug output above to see what went wrong.")

        print("="*60)


def main():
    """Main function to run the analysis"""
    analyzer = SymmetryDeviationAnalyzer()

    # Get file paths from user
    print("\n" + "="*60)
    print("ML MODEL SYMMETRY DEVIATION ANALYZER - FULL VERSION")
    print("="*60)

    full_excel = input("\nEnter path to full predictions Excel (270k rows): ").strip()
    averaged_excel = input("Enter path to averaged predictions Excel (~41k rows): ").strip()
    canonical_txt = input("Enter path to canonical configurations text file (~41k): ").strip()
    full_config_txt = input("Enter path to FULL configurations text file (270k): ").strip()

    # Validate files exist
    for filepath, name in [(full_excel, "Full predictions"),
                           (averaged_excel, "Averaged predictions"),
                           (canonical_txt, "Canonical configs"),
                           (full_config_txt, "Full configs")]:
        if not os.path.exists(filepath):
            print(f"Error: {name} file not found: {filepath}")
            return

    # Run analysis
    try:
        analyzer.run_analysis(full_excel, averaged_excel, canonical_txt, full_config_txt)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
