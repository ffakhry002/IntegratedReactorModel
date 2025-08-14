#!/usr/bin/env python3
"""
Symmetry Group Comparison Diagnostic Tool with Physical Position Mapping
Compare flux predictions for a specific configuration with all members of its symmetry group
using proper D4 transformation mapping
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class SymmetryGroupDiagnostic:
    def __init__(self):
        self.full_predictions_df = None
        self.averaged_df = None
        self.group_memberships = {}
        self.all_lattices = {}
        self.canonical_lattices = {}
        self.transformation_map = {}
        self.output_dir = Path("ML/data/group_diagnostics")
        self.full_predictions_dict = {}

    def setup_output_directory(self):
        """Create output directory for diagnostic plots"""
        self.output_dir.mkdir(exist_ok=True)

    def parse_lattice_string(self, lattice_str):
        """Parse a lattice string into a numpy array"""
        try:
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

        config_pos_set = set(config_positions.values())
        canonical_pos_set = set(canonical_positions.values())

        for transform in transformations:
            transformed_positions = set()
            for pos in config_pos_set:
                new_pos = self.apply_d4_transformation(pos, transform)
                transformed_positions.add(new_pos)

            if transformed_positions == canonical_pos_set:
                return transform

        return None

    def create_position_mapping(self, config_positions, canonical_positions, transform):
        """Create mapping of which config position corresponds to which canonical position"""
        mapping = {}

        for config_label, config_pos in config_positions.items():
            transformed_pos = self.apply_d4_transformation(config_pos, transform)

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

        runs = re.split(r'RUN \d+:', content)

        for run_idx, run in enumerate(tqdm(runs[1:], desc="Loading lattices"), 1):
            if not run.strip():
                continue

            try:
                desc_match = re.search(r'Description:\s*([\w_]+)', run)
                if not desc_match:
                    continue

                description = desc_match.group(1).strip()
                lattice_match = re.search(r'core_lattice:\s*(\[.*?\])\s*\n=', run, re.DOTALL)

                if not lattice_match:
                    continue

                lattice_str = lattice_match.group(1)
                lattice = self.parse_lattice_string(lattice_str)

                if lattice is not None:
                    self.all_lattices[description] = lattice

            except Exception as e:
                continue

        print(f"✓ Loaded {len(self.all_lattices)} lattice configurations")

    def parse_canonical_groups(self, canonical_txt_path):
        """Parse the canonical text file to extract group memberships AND canonical lattices"""
        print(f"\nParsing canonical groups from: {canonical_txt_path}")

        with open(canonical_txt_path, 'r') as f:
            content = f.read()

        runs = re.split(r'RUN \d+:', content)

        for run in tqdm(runs[1:], desc="Parsing canonical groups"):
            if not run.strip():
                continue

            desc_match = re.search(r'Description:\s*([\w_]+)', run)
            if not desc_match:
                continue
            canonical_desc = desc_match.group(1).strip()

            # Extract lattice
            lattice_match = re.search(r'core_lattice:\s*(\[.*?\])\s*(?:\n|//)', run, re.DOTALL)
            if not lattice_match:
                lattice_match = re.search(r'core_lattice:\s*(\[.*?\])\s*\n=', run, re.DOTALL)

            if lattice_match:
                lattice_str = lattice_match.group(1)
                lattice = self.parse_lattice_string(lattice_str)
                if lattice is not None:
                    self.canonical_lattices[canonical_desc] = lattice

            # Extract group members
            members_match = re.search(r'//\s*Group members:\s*([^\n]+)', run)
            if members_match:
                members_str = members_match.group(1)
                members = [m.strip() for m in members_str.split(',')]
                self.group_memberships[canonical_desc] = members
            else:
                self.group_memberships[canonical_desc] = [canonical_desc]

        print(f"✓ Found {len(self.group_memberships)} canonical groups")
        print(f"✓ Loaded {len(self.canonical_lattices)} canonical lattices")

        # Create reverse mapping
        self.member_to_canonical = {}
        for canonical, members in self.group_memberships.items():
            for member in members:
                self.member_to_canonical[member] = canonical

    def compute_all_transformations(self):
        """Compute transformation for each config to its canonical form"""
        print("\nComputing transformations for all configurations...")

        for canonical_desc, member_descs in tqdm(self.group_memberships.items(),
                                                 desc="Computing transformations"):
            if canonical_desc not in self.canonical_lattices:
                continue

            canonical_lattice = self.canonical_lattices[canonical_desc]
            canonical_positions = self.get_irradiation_positions(canonical_lattice)

            for member_desc in member_descs:
                if member_desc not in self.all_lattices:
                    continue

                member_lattice = self.all_lattices[member_desc]
                member_positions = self.get_irradiation_positions(member_lattice)

                transform = self.find_transformation_to_canonical(member_positions, canonical_positions)

                if transform:
                    position_mapping = self.create_position_mapping(
                        member_positions, canonical_positions, transform
                    )

                    self.transformation_map[member_desc] = {
                        'canonical': canonical_desc,
                        'transform': transform,
                        'position_mapping': position_mapping
                    }

        print(f"✓ Computed transformations for {len(self.transformation_map)} configurations")

    def load_predictions(self, full_excel_path, averaged_excel_path):
        """Load prediction Excel files"""
        print(f"\nLoading predictions...")
        self.full_predictions_df = pd.read_excel(full_excel_path)
        print(f"✓ Loaded {len(self.full_predictions_df)} full predictions")

        # Create dictionary for faster lookups
        self.full_predictions_dict = {
            row['description']: row.to_dict()
            for _, row in self.full_predictions_df.iterrows()
        }

        self.averaged_df = pd.read_excel(averaged_excel_path)
        print(f"✓ Loaded {len(self.averaged_df)} averaged predictions")

    def create_comparison_plots(self, config_name):
        """Create comparison plots for a specific configuration with proper position mapping"""
        print(f"\n" + "="*60)
        print(f"Analyzing configuration: {config_name}")
        print("="*60)

        # Find the canonical group
        if config_name not in self.member_to_canonical:
            print(f"❌ Configuration '{config_name}' not found in any symmetry group!")
            return

        canonical_name = self.member_to_canonical[config_name]
        print(f"✓ Found in canonical group: {canonical_name}")

        # Get all members of this group
        group_members = self.group_memberships[canonical_name]
        print(f"✓ Group has {len(group_members)} members")

        # Check canonical lattice exists
        if canonical_name not in self.canonical_lattices:
            print(f"❌ No canonical lattice found for {canonical_name}")
            return

        canonical_lattice = self.canonical_lattices[canonical_name]
        canonical_positions = self.get_irradiation_positions(canonical_lattice)

        # Get averaged prediction
        avg_prediction = self.averaged_df[
            self.averaged_df['description'] == canonical_name
        ]

        if len(avg_prediction) == 0:
            print("⚠️ No averaged prediction found")
            avg_data = None
        else:
            avg_data = avg_prediction.iloc[0].to_dict()

        # Identify available flux types
        flux_types = []
        for col in ['thermal', 'epithermal', 'fast']:
            if f'I_1_{col}' in self.full_predictions_df.columns:
                flux_types.append(col)

        has_keff = 'keff' in self.full_predictions_df.columns

        # Collect properly mapped data for each canonical position
        canonical_position_data = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        keff_data = {}

        for member_desc in group_members:
            if member_desc not in self.full_predictions_dict:
                continue

            member_data = self.full_predictions_dict[member_desc]

            # Store k-eff (position-independent)
            if has_keff and 'keff' in member_data:
                keff_data[member_desc] = member_data['keff']

            # Get position mapping for this member
            if member_desc not in self.transformation_map:
                print(f"  Warning: No transformation found for {member_desc}")
                continue

            position_mapping = self.transformation_map[member_desc]['position_mapping']

            # Map flux values through the transformation
            for member_label, canonical_label in position_mapping.items():
                member_label_num = int(member_label.split('_')[1])
                canonical_label_num = int(canonical_label.split('_')[1])

                for flux_type in flux_types:
                    member_col = f'I_{member_label_num}_{flux_type}'

                    if member_col in member_data and pd.notna(member_data[member_col]):
                        # Store under the CANONICAL position and member name
                        canonical_position_data[canonical_label_num][flux_type][member_desc] = member_data[member_col]

        # Create figure with subplots for each position + k-eff + boxplot
        fig = plt.figure(figsize=(20, 12))

        # Color scheme
        colors = {'thermal': '#FF6B6B', 'epithermal': '#FFD93D', 'fast': '#4DABF7'}

        # Create member labels for x-axis (shortened)
        member_labels = []
        for m in group_members:
            if m in self.transformation_map:  # Only include members with valid transformations
                if m.startswith('config_') or m.startswith('core_config_'):
                    member_labels.append(m.split('_')[-1])
                else:
                    member_labels.append(m[-6:] if len(m) > 6 else m)

        # Plot for each irradiation position (I_1 to I_4)
        for pos_idx in range(1, 5):
            ax = plt.subplot(2, 4, pos_idx)

            if pos_idx not in canonical_position_data:
                ax.text(0.5, 0.5, f'No data for I_{pos_idx}',
                       transform=ax.transAxes, ha='center', va='center')
                continue

            position_data = canonical_position_data[pos_idx]

            # Prepare data for grouped bar chart
            bar_groups = []
            bar_labels = []
            bar_colors = []
            deviation_values = []
            deviation_positions = []  # Track x positions for non-average points

            # Store averages for deviation calculation
            flux_averages = {}
            for flux_type in flux_types:
                if avg_data and f'I_{pos_idx}_{flux_type}' in avg_data:
                    flux_averages[flux_type] = avg_data[f'I_{pos_idx}_{flux_type}']

            # Calculate total average
            if avg_data:
                total_average = sum(avg_data.get(f'I_{pos_idx}_{ft}', 0) for ft in flux_types)
            else:
                total_average = 0

            x_counter = 0

            # For each flux type, add all member values then the average
            for flux_type in flux_types:
                avg_val = flux_averages.get(flux_type, 0)

                # Add member values
                for member_desc in group_members:
                    if member_desc in self.transformation_map and member_desc in position_data[flux_type]:
                        value = position_data[flux_type][member_desc]
                        bar_groups.append(value)
                        bar_labels.append(f"{flux_type[0].upper()}-{member_desc.split('_')[-1][-4:]}")
                        bar_colors.append(colors[flux_type])

                        # Calculate absolute deviation
                        if avg_val > 0:
                            deviation = abs((value - avg_val) / avg_val) * 100
                        else:
                            deviation = 0
                        deviation_values.append(deviation)
                        deviation_positions.append(x_counter)
                        x_counter += 1

                # Add average value (but don't add to deviation plot)
                if avg_data and f'I_{pos_idx}_{flux_type}' in avg_data:
                    bar_groups.append(avg_val)
                    bar_labels.append(f"{flux_type[0].upper()}-AVG")
                    bar_colors.append(colors[flux_type])
                    x_counter += 1  # Skip this position for deviations

            # Add total flux (sum of all types)
            for member_desc in group_members:
                if member_desc in self.transformation_map:
                    total = sum(position_data[ft].get(member_desc, 0) for ft in flux_types)
                    if total > 0:
                        bar_groups.append(total)
                        bar_labels.append(f"TOT-{member_desc.split('_')[-1][-4:]}")
                        bar_colors.append('#2ECC71')

                        # Calculate absolute deviation for total
                        if total_average > 0:
                            deviation = abs((total - total_average) / total_average) * 100
                        else:
                            deviation = 0
                        deviation_values.append(deviation)
                        deviation_positions.append(x_counter)
                        x_counter += 1

            # Add average total (but don't add to deviation plot)
            if avg_data and total_average > 0:
                bar_groups.append(total_average)
                bar_labels.append("TOT-AVG")
                bar_colors.append('#27AE60')
                x_counter += 1  # Skip this position

            # Create bar chart
            x_positions = np.arange(len(bar_groups))
            bars = ax.bar(x_positions, bar_groups, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=0.5)

            # Create secondary y-axis for deviations
            ax2 = ax.twinx()

            # Plot deviations in segments (disconnected at averages)
            if deviation_values:
                # Group consecutive positions for connected lines
                segments = []
                current_segment_pos = []
                current_segment_val = []

                for i, pos in enumerate(deviation_positions):
                    if i == 0 or pos == deviation_positions[i-1] + 1:
                        current_segment_pos.append(pos)
                        current_segment_val.append(deviation_values[i])
                    else:
                        if current_segment_pos:
                            segments.append((current_segment_pos, current_segment_val))
                        current_segment_pos = [pos]
                        current_segment_val = [deviation_values[i]]

                if current_segment_pos:
                    segments.append((current_segment_pos, current_segment_val))

                # Plot each segment
                for seg_pos, seg_val in segments:
                    ax2.plot(seg_pos, seg_val, 'ko-', markersize=4, linewidth=1, alpha=0.8)

            ax2.set_ylabel('Absolute Deviation (%)', fontsize=9, color='black')
            ax2.tick_params(axis='y', labelcolor='black')

            # Set y-limits for deviation axis to ensure visibility
            if deviation_values:
                max_dev = max(deviation_values)
                ax2.set_ylim(0, max_dev*1.2)

            ax.set_xlabel('Type-Config', fontsize=8)
            ax.set_ylabel('Flux', fontsize=10)
            ax.set_title(f'Irradiation Position I_{pos_idx}', fontsize=11, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(bar_labels, rotation=90, ha='right', fontsize=6)
            ax.grid(True, alpha=0.3, axis='y')
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        # K-eff subplot
        if has_keff and keff_data:
            ax_keff = plt.subplot(2, 4, 5)

            keff_members = list(keff_data.keys())
            keff_values = list(keff_data.values())
            keff_labels = [m.split('_')[-1][-6:] for m in keff_members]

            # Calculate absolute deviations for k-eff
            keff_deviations = []
            keff_deviation_positions = []
            avg_keff = avg_data['keff'] if avg_data and 'keff' in avg_data else np.mean(keff_values)

            for i, value in enumerate(keff_values):
                if avg_keff > 0:
                    deviation = abs((value - avg_keff) / avg_keff) * 100
                else:
                    deviation = 0
                keff_deviations.append(deviation)
                keff_deviation_positions.append(i)

            # Add average k-eff (but don't add to deviation plot)
            if avg_data and 'keff' in avg_data:
                keff_values.append(avg_data['keff'])
                keff_labels.append('AVG')
                # Don't add deviation position for average

            x_pos = np.arange(len(keff_values))
            colors_keff = ['#95A5A6'] * (len(keff_values) - 1) + ['#E74C3C'] if avg_data and 'keff' in avg_data else ['#95A5A6'] * len(keff_values)

            ax_keff.bar(x_pos, keff_values, color=colors_keff, alpha=0.7, edgecolor='black')
            ax_keff.set_xlabel('Configuration', fontsize=10)
            ax_keff.set_ylabel('k-effective', fontsize=10)
            ax_keff.set_title('k-effective Values', fontsize=11, fontweight='bold')
            ax_keff.set_xticks(x_pos)
            ax_keff.set_xticklabels(keff_labels, rotation=45, ha='right', fontsize=8)
            ax_keff.grid(True, alpha=0.3, axis='y')

            # Create secondary y-axis for deviations
            ax_keff2 = ax_keff.twinx()
            if keff_deviations:
                ax_keff2.plot(keff_deviation_positions, keff_deviations, 'ko-', markersize=4, linewidth=1, alpha=0.8)
            ax_keff2.set_ylabel('Absolute Deviation (%)', fontsize=9, color='black')
            ax_keff2.tick_params(axis='y', labelcolor='black')

            # Set y-limits for deviation axis
            if keff_deviations:
                max_dev = max(keff_deviations)
                ax_keff2.set_ylim(0, max_dev*1.2)

            # Add mean line on primary axis - REMOVED
            # ax_keff.axhline(y=mean_keff, color='red', linestyle='--', alpha=0.5, label=f'Mean: {mean_keff:.6f}')
            # ax_keff.legend(loc='upper left', fontsize=8)

        # Deviation boxplot
        ax_dev = plt.subplot(2, 4, 6)

        if avg_data:
            deviations = defaultdict(list)

            # Calculate deviations for properly mapped positions
            for pos_idx in range(1, 5):
                if pos_idx not in canonical_position_data:
                    continue

                for flux_type in flux_types:
                    avg_col = f'I_{pos_idx}_{flux_type}'
                    if avg_col in avg_data and avg_data[avg_col] > 0:
                        for member_desc, value in canonical_position_data[pos_idx][flux_type].items():
                            deviation = abs(value - avg_data[avg_col]) / avg_data[avg_col] * 100
                            deviations[flux_type].append(deviation)

            # Add k-eff deviations
            if has_keff and 'keff' in avg_data and avg_data['keff'] > 0:
                for member_desc, value in keff_data.items():
                    deviation = abs(value - avg_data['keff']) / avg_data['keff'] * 100
                    deviations['keff'].append(deviation)

            # Create boxplot
            if deviations:
                box_data = []
                box_labels = []
                box_colors_list = []

                for key in ['thermal', 'epithermal', 'fast', 'keff']:
                    if key in deviations:
                        box_data.append(deviations[key])
                        box_labels.append(key.capitalize())
                        if key == 'keff':
                            box_colors_list.append('#95A5A6')
                        else:
                            box_colors_list.append(colors.get(key, '#95A5A6'))

                bp = ax_dev.boxplot(box_data, labels=box_labels, patch_artist=True)

                for patch, color in zip(bp['boxes'], box_colors_list):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.5)

                ax_dev.set_ylabel('Absolute Deviation (%)', fontsize=10)
                ax_dev.set_title('Absolute Deviations from Average', fontsize=11, fontweight='bold')
                ax_dev.grid(True, alpha=0.3, axis='y')
                # Removed 1% line
        else:
            ax_dev.text(0.5, 0.5, 'No average data available',
                       transform=ax_dev.transAxes, ha='center', va='center')

        # Core configuration visualization
        ax_core = plt.subplot(2, 4, 7)

        # Get the lattice for the input configuration
        if config_name in self.all_lattices:
            config_lattice = self.all_lattices[config_name]

            # Get irradiation positions
            irrad_positions = self.get_irradiation_positions(config_lattice)

            # Create color map
            colors_map = {
                'C': '#B3E5FC',    # Light blue for coolant
                'F': '#C8E6C9',    # Light green for fuel
                'I': '#FFCDD2'     # Light red for irradiation
            }

            # Draw the 8x8 grid
            for i in range(8):
                for j in range(8):
                    cell_value = config_lattice[i, j]

                    # Determine cell type (first character)
                    if cell_value.startswith('I'):
                        cell_type = 'I'
                        color = '#FF5252'  # Darker red for irradiation
                    elif cell_value == 'C':
                        cell_type = 'C'
                        color = colors_map['C']
                    elif cell_value == 'F':
                        cell_type = 'F'
                        color = colors_map['F']
                    else:
                        cell_type = cell_value[0] if cell_value else ''
                        color = 'white'

                    # Create rectangle
                    rect = plt.Rectangle((j, 7-i), 1, 1,
                                        linewidth=1,
                                        edgecolor='black',
                                        facecolor=color)
                    ax_core.add_patch(rect)

                    # Add text label
                    if cell_value.startswith('I'):
                        # Show the full I_n label
                        ax_core.text(j+0.5, 7-i+0.5, cell_value,
                                   ha='center', va='center',
                                   fontsize=10, weight='bold', color='black')
                    else:
                        # Show just the cell type
                        ax_core.text(j+0.5, 7-i+0.5, cell_type,
                                   ha='center', va='center',
                                   fontsize=9, weight='bold', color='gray')

            # Set limits and aspect
            ax_core.set_xlim(0, 8)
            ax_core.set_ylim(0, 8)
            ax_core.set_aspect('equal')
            ax_core.set_title(f'Core Configuration: {config_name}', fontsize=11, fontweight='bold')

            # Remove ticks
            ax_core.set_xticks([])
            ax_core.set_yticks([])

            # Add grid lines
            for i in range(9):
                ax_core.axhline(i, color='black', linewidth=0.5)
                ax_core.axvline(i, color='black', linewidth=0.5)

            # Add legend
            legend_elements = [
                plt.Rectangle((0,0),1,1, facecolor=colors_map['C'], edgecolor='black', label='Coolant'),
                plt.Rectangle((0,0),1,1, facecolor=colors_map['F'], edgecolor='black', label='Fuel'),
                plt.Rectangle((0,0),1,1, facecolor='#FF5252', edgecolor='black', label='Irradiation')
            ]
            ax_core.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.15),
                         ncol=3, fontsize=8, frameon=False)
        else:
            ax_core.text(0.5, 0.5, f'Configuration {config_name}\nnot found in lattice data',
                       transform=ax_core.transAxes, ha='center', va='center')
            ax_core.set_xlim(0, 1)
            ax_core.set_ylim(0, 1)
            ax_core.axis('off')

        # Overall title with all group member numbers
        group_numbers = []
        for member in group_members:
            # Extract just the number from the config name
            if 'config_' in member:
                num = member.split('config_')[-1]
            elif 'core_config_' in member:
                num = member.split('core_config_')[-1]
            else:
                num = member.split('_')[-1] if '_' in member else member
            group_numbers.append(num)

        # Format title with main config and others
        main_num = config_name.split('config_')[-1] if 'config_' in config_name else config_name.split('_')[-1]
        other_nums = [n for n in group_numbers if n != main_num]

        if other_nums:
            title_text = f'Symmetry Group Analysis: config_{main_num} [{", ".join(other_nums[:10])}'
            if len(other_nums) > 10:
                title_text += f', ... ({len(other_nums)-10} more)'
            title_text += ']'
        else:
            title_text = f'Symmetry Group Analysis: config_{main_num}'

        fig.suptitle(title_text, fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save figure
        self.setup_output_directory()
        output_file = self.output_dir / f'group_comparison_{config_name.replace("/", "_")}_mapped.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved comparison plot to: {output_file}")

        plt.show()

        print("\n" + "="*60)
        print("Analysis complete with proper physical position mapping!")
        print("="*60)

    def run_analysis(self, full_excel, averaged_excel, canonical_txt, full_config_txt):
        """Run the analysis with all necessary data loading"""
        # Load all configurations
        self.load_all_configurations(full_config_txt)

        # Parse canonical groups
        self.parse_canonical_groups(canonical_txt)

        # Compute transformations
        self.compute_all_transformations()

        # Load predictions
        self.load_predictions(full_excel, averaged_excel)

        print(f"\n✓ Ready for analysis!")
        print(f"  Total configs with transformations: {len(self.transformation_map)}")
        print(f"  Total canonical groups: {len(self.group_memberships)}")


def main():
    """Main function to run diagnostic analysis"""
    print("\n" + "="*60)
    print("SYMMETRY GROUP DIAGNOSTIC WITH PHYSICAL POSITION MAPPING")
    print("="*60)

    diagnostic = SymmetryGroupDiagnostic()

    # Get file paths
    full_excel = input("\nEnter path to full predictions Excel (270k rows): ").strip()
    averaged_excel = input("Enter path to averaged predictions Excel (~41k rows): ").strip()
    canonical_txt = input("Enter path to canonical configurations text file: ").strip()
    full_config_txt = input("Enter path to FULL configurations text file (270k): ").strip()

    # Validate files
    for filepath, name in [(full_excel, "Full predictions"),
                           (averaged_excel, "Averaged predictions"),
                           (canonical_txt, "Canonical configs"),
                           (full_config_txt, "Full configs")]:
        if not os.path.exists(filepath):
            print(f"Error: {name} file not found: {filepath}")
            return

    # Run analysis
    diagnostic.run_analysis(full_excel, averaged_excel, canonical_txt, full_config_txt)

    # Interactive mode
    while True:
        print("\n" + "-"*60)
        config_name = input("Enter configuration name to analyze (or 'quit' to exit): ").strip()

        if config_name.lower() in ['quit', 'exit', 'q']:
            break

        if not config_name:
            continue

        try:
            diagnostic.create_comparison_plots(config_name)
        except Exception as e:
            print(f"Error analyzing {config_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDiagnostic analysis complete!")


if __name__ == "__main__":
    main()
