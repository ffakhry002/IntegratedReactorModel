#!/usr/bin/env python3
"""
Histogram Generator for Nuclear Reactor Model Predictions

This script processes Excel files created by predict.py and generates:
1. Histograms for keff, thermal, epithermal, fast, and total flux data
2. Box plots for flux data showing outliers and quartiles
3. PNG output with all plots
4. Excel file with histogram data (separate sheets)

Author: Generated for nuclear reactor prediction analysis
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import warnings
warnings.filterwarnings('ignore')

# Try to import ternary for ternary plots
try:
    import ternary
    HAS_TERNARY = True
except ImportError:
    HAS_TERNARY = False
    print("Note: Install 'python-ternary' for ternary plots: pip install python-ternary")

class HistogramGenerator:
    """Class to generate histograms and box plots from prediction Excel files"""

    def __init__(self):
        self.data = None
        self.available_columns = {}
        self.output_dir = None
        self.setup_directories()

    def setup_directories(self):
        """Setup output directories"""
        ml_dir = os.path.dirname(os.path.abspath(__file__))
        outputs_dir = os.path.join(ml_dir, "outputs")
        self.output_dir = os.path.join(outputs_dir, "excel_reports")

        # Ensure directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def select_input_file(self):
        """Let user select the Excel file to process"""
        print("\n" + "="*60)
        print("HISTOGRAM GENERATOR FOR REACTOR PREDICTIONS")
        print("="*60)

        # First check for Excel files in the excel_reports directory
        if self.output_dir and os.path.exists(self.output_dir):
            excel_files = [f for f in os.listdir(self.output_dir) if f.endswith('.xlsx')]
        else:
            excel_files = []

        if excel_files:
            print(f"\nFound {len(excel_files)} Excel files in {self.output_dir}:")
            for i, filename in enumerate(sorted(excel_files), 1):
                file_path = os.path.join(self.output_dir, filename)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"    {i}. {filename} ({file_size:.1f} KB)")

            print(f"    {len(excel_files) + 1}. Browse for a different file")

            try:
                choice = input(f"\nSelect file number (1-{len(excel_files) + 1}): ").strip()
                choice_num = int(choice)

                if 1 <= choice_num <= len(excel_files):
                    selected_file = os.path.join(self.output_dir, sorted(excel_files)[choice_num - 1])
                    print(f"Selected: {os.path.basename(selected_file)}")
                    return selected_file
                elif choice_num == len(excel_files) + 1:
                    # Browse for file
                    pass
                else:
                    print("Invalid selection!")
                    return None

            except ValueError:
                print("Invalid input!")
                return None

        # Browse for file using file dialog
        print("\nOpening file browser...")
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        file_path = filedialog.askopenfilename(
            title="Select Excel file from predict.py",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialdir=self.output_dir
        )

        root.destroy()

        if file_path:
            print(f"Selected: {os.path.basename(file_path)}")
            return file_path
        else:
            print("No file selected.")
            return None

    def load_and_analyze_data(self, file_path):
        """Load Excel file and analyze available columns"""
        try:
            # Read the Excel file
            self.data = pd.read_excel(file_path, sheet_name='Predictions')
            print(f"\nLoaded data with {len(self.data)} rows and {len(self.data.columns)} columns")

            # Analyze available columns with FIXED detection logic
            self.available_columns = {
                'keff': [],
                'thermal': [],
                'epithermal': [],
                'fast': [],
                'total': [],
                'thermal_percent': [],
                'epithermal_percent': [],
                'fast_percent': []
            }

            for col in self.data.columns:
                if col == 'keff':
                    self.available_columns['keff'].append(col)
                # FIXED: More precise column detection
                elif col.endswith('_thermal') and 'I_' in col:  # I_1_thermal, I_2_thermal, etc.
                    self.available_columns['thermal'].append(col)
                elif col.endswith('_epithermal') and 'I_' in col:  # I_1_epithermal, I_2_epithermal, etc.
                    self.available_columns['epithermal'].append(col)
                elif col.endswith('_fast') and 'I_' in col:  # I_1_fast, I_2_fast, etc.
                    self.available_columns['fast'].append(col)
                elif 'total_flux' in col and 'I_' in col:  # I_1_total_flux, I_2_total_flux, etc.
                    self.available_columns['total'].append(col)
                elif col.endswith('_thermal_percent'):  # I_1_thermal_percent, etc.
                    self.available_columns['thermal_percent'].append(col)
                elif col.endswith('_epithermal_percent'):  # I_1_epithermal_percent, etc.
                    self.available_columns['epithermal_percent'].append(col)
                elif col.endswith('_fast_percent'):  # I_1_fast_percent, etc.
                    self.available_columns['fast_percent'].append(col)

            # Display what was found
            print("\nAvailable data types:")
            for data_type, columns in self.available_columns.items():
                if columns:
                    print(f"  {data_type.capitalize()}: {len(columns)} irradiation positions")

            return True

        except Exception as e:
            print(f"Error loading file: {e}")
            return False

    def extract_values(self, data_type):
        """Extract all values for a given data type across all irradiation positions"""
        columns = self.available_columns[data_type]
        if not columns:
            return np.array([])

        all_values = []
        for col in columns:
            # Get non-null, non-empty values from this column
            values = self.data[col].dropna()
            # Filter out any non-numeric values
            numeric_values = pd.to_numeric(values, errors='coerce').dropna()
            all_values.extend(numeric_values.tolist())

        extracted_array = np.array(all_values)

        # Add verification output for data extraction
        if len(extracted_array) > 0:
            print(f"  {data_type.capitalize()}: {len(extracted_array)} values, range [{np.min(extracted_array):.2e}, {np.max(extracted_array):.2e}]")

        return extracted_array

    def has_percentage_data(self):
        """Check if we have all three percentage columns for ternary plots"""
        return (len(self.available_columns['thermal_percent']) > 0 and
                len(self.available_columns['epithermal_percent']) > 0 and
                len(self.available_columns['fast_percent']) > 0)

    def create_energy_distribution_clear(self):
        """Create clear visualization for thermal/epithermal/fast percentages distribution"""
        # Extract percentage data for each irradiation position
        thermal_cols = self.available_columns['thermal_percent']
        epithermal_cols = self.available_columns['epithermal_percent']
        fast_cols = self.available_columns['fast_percent']

        if not (thermal_cols and epithermal_cols and fast_cols):
            print("Missing percentage data for energy distribution plot")
            return None

        # Combine data from all irradiation positions and configurations
        all_data = []

        for i in range(len(self.data)):
            row = self.data.iloc[i]

            # Get percentages for all irradiation positions in this configuration
            for t_col, e_col, f_col in zip(thermal_cols, epithermal_cols, fast_cols):
                thermal_pct = row[t_col]
                epithermal_pct = row[e_col]
                fast_pct = row[f_col]

                # Only include if all values are valid
                if pd.notna(thermal_pct) and pd.notna(epithermal_pct) and pd.notna(fast_pct):
                    # Check if percentages sum to approximately 100
                    total = thermal_pct + epithermal_pct + fast_pct
                    if abs(total - 100) < 5:  # Allow small rounding errors
                        all_data.append((thermal_pct, epithermal_pct, fast_pct))

        if not all_data:
            print("No valid percentage data found for energy distribution plot")
            return None

        print(f"Creating CLEAR energy distribution plots with {len(all_data)} data points...")

        # Extract individual arrays
        thermal_data = np.array([point[0] for point in all_data])
        epithermal_data = np.array([point[1] for point in all_data])
        fast_data = np.array([point[2] for point in all_data])

        # Create figure with 2x2 subplots for comprehensive analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. HEXBIN PLOT: Thermal vs Epithermal (INCREASED GRANULARITY)
        hb1 = ax1.hexbin(thermal_data, epithermal_data, gridsize=80, cmap='plasma', mincnt=1)
        ax1.set_xlabel('Thermal %', fontsize=12)
        ax1.set_ylabel('Epithermal %', fontsize=12)
        ax1.set_title('High-Resolution Density: Thermal vs Epithermal\n(80x80 hexagonal bins)',
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(hb1, ax=ax1, label='Point Density')

        # 2. HEXBIN PLOT: Thermal vs Fast (INCREASED GRANULARITY)
        hb2 = ax2.hexbin(thermal_data, fast_data, gridsize=80, cmap='viridis', mincnt=1)
        ax2.set_xlabel('Thermal %', fontsize=12)
        ax2.set_ylabel('Fast %', fontsize=12)
        ax2.set_title('High-Resolution Density: Thermal vs Fast\n(80x80 hexagonal bins)',
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(hb2, ax=ax2, label='Point Density')

        # 3. HEXBIN PLOT: Epithermal vs Fast (INCREASED GRANULARITY)
        hb3 = ax3.hexbin(epithermal_data, fast_data, gridsize=80, cmap='coolwarm', mincnt=1)
        ax3.set_xlabel('Epithermal %', fontsize=12)
        ax3.set_ylabel('Fast %', fontsize=12)
        ax3.set_title('High-Resolution Density: Epithermal vs Fast\n(80x80 hexagonal bins)',
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(hb3, ax=ax3, label='Point Density')

        # 4. ALL POINTS SCATTER (NO SAMPLING)
        print(f"Creating scatter plot with ALL {len(all_data)} points...")
        scatter = ax4.scatter(thermal_data, epithermal_data,
                            c=fast_data, cmap='rainbow', alpha=0.3, s=1)
        ax4.set_xlabel('Thermal %', fontsize=12)
        ax4.set_ylabel('Epithermal %', fontsize=12)
        ax4.set_title(f'ALL Data Points: Thermal vs Epithermal\n({len(all_data):,} points, color = Fast %)',
                     fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Fast %', rotation=270, labelpad=15)

        # Add comprehensive statistics
        stats_text = f"""Dataset Statistics:
Points: {len(all_data):,}

Thermal %:
  Mean: {np.mean(thermal_data):.1f}%
  Std:  {np.std(thermal_data):.1f}%
  Range: {np.min(thermal_data):.1f}% - {np.max(thermal_data):.1f}%

Epithermal %:
  Mean: {np.mean(epithermal_data):.1f}%
  Std:  {np.std(epithermal_data):.1f}%
  Range: {np.min(epithermal_data):.1f}% - {np.max(epithermal_data):.1f}%

Fast %:
  Mean: {np.mean(fast_data):.1f}%
  Std:  {np.std(fast_data):.1f}%
  Range: {np.min(fast_data):.1f}% - {np.max(fast_data):.1f}%"""

        ax4.text(1.02, 1.0, stats_text, transform=ax4.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"energy_distribution_clear.png"
        if self.output_dir:
            plot_path = os.path.join(self.output_dir, plot_filename)
        else:
            plot_path = plot_filename

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"CLEAR energy distribution analysis saved to: {plot_filename}")

        plt.show()

        return plot_path

    def create_histograms_and_boxplots(self):
        """Create histograms and box plots for reactor data types"""
        # Define the specific data types the user wants (in order)
        desired_types = ['keff', 'total', 'thermal', 'epithermal', 'fast']

        # Filter to only available types
        available_types = [dt for dt in desired_types if self.available_columns[dt]]

        if not available_types:
            print("No data available for plotting!")
            return None

        print(f"\nGenerating plots for: {', '.join(available_types)}")

        # Calculate rows: one for each histogram + one for box plot
        flux_types = [dt for dt in available_types if dt != 'keff']
        n_hist_rows = len(available_types)
        n_total_rows = n_hist_rows + (1 if flux_types else 0)  # +1 for box plots if flux data exists

        # Create figure with subplots
        fig, axes = plt.subplots(n_total_rows, 1, figsize=(12, 4 * n_total_rows))

        # Handle case where we only have one subplot
        if n_total_rows == 1:
            axes = [axes]
        elif axes is None:
            print("Error creating subplots")
            return None

        # Store histogram data for Excel export
        histogram_data = {}
        num_bins = 25
        # Create histograms for each data type
        for i, data_type in enumerate(available_types):
            values = self.extract_values(data_type)

            if len(values) == 0:
                print(f"No valid data for {data_type}")
                continue

            # Calculate histogram with 1000 bins
            hist, bin_edges = np.histogram(values, bins=num_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Store histogram data
            histogram_data[data_type] = {
                'bin_centers': bin_centers,
                'counts': hist,
                'bin_edges': bin_edges
            }

            # Plot histogram
            ax = axes[i]
            ax.hist(values, bins=num_bins, alpha=0.7, edgecolor='black', linewidth=0.5)

            # Set title and labels with proper names
            data_name = {
                'keff': 'K-effective',
                'total': 'Total Flux',
                'thermal': 'Thermal Flux',
                'epithermal': 'Epithermal Flux',
                'fast': 'Fast Flux'
            }.get(data_type, data_type.capitalize())

            title = f'{data_name} Distribution ({num_bins} discretised bins)\n'
            title += f'Min: {np.min(values):.2e}, Max: {np.max(values):.2e}, N: {len(values)}'
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency')

            # Set x-axis with 10 evenly spaced ticks
            x_min, x_max = np.min(values), np.max(values)
            x_ticks = np.linspace(x_min, x_max, 10)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f'{x:.2e}' for x in x_ticks], rotation=45)

            # Format axis
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

        # Create box plots for flux data if available
        if flux_types:
            box_ax = axes[n_hist_rows]

            # Prepare data for box plots (in specific order)
            box_data = []
            box_labels = []

            for flux_type in ['total', 'thermal', 'epithermal', 'fast']:
                if flux_type in flux_types:
                    values = self.extract_values(flux_type)
                    if len(values) > 0:
                        box_data.append(values)
                        # Use proper names
                        label_name = {
                            'total': 'Total Flux',
                            'thermal': 'Thermal Flux',
                            'epithermal': 'Epithermal Flux',
                            'fast': 'Fast Flux'
                        }.get(flux_type, flux_type.capitalize())
                        box_labels.append(label_name)

            if box_data:
                # Create box plot
                bp = box_ax.boxplot(box_data, labels=box_labels, patch_artist=True)

                # Color the boxes
                colors = ['gold', 'lightblue', 'lightgreen', 'lightcoral']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)

                box_ax.set_title('Box Plots for Flux Data\n(Shows quartiles, median, and outliers)',
                                fontsize=12, fontweight='bold')
                box_ax.set_ylabel('Flux Values')
                box_ax.grid(True, alpha=0.3)
                box_ax.set_axisbelow(True)

                # IMPROVED: Set y-axis limits based on actual data range to prevent squashing
                # Calculate the overall min and max across all flux types
                all_flux_values = np.concatenate(box_data)
                data_min = np.min(all_flux_values)
                data_max = np.max(all_flux_values)
                data_range = data_max - data_min

                # Add 10% padding above and below for better visualization
                padding = data_range * 0.1
                y_min = data_min - padding
                y_max = data_max + padding

                # Set the limits
                box_ax.set_ylim(y_min, y_max)

                print(f"  Box plot y-axis: [{y_min:.2e}, {y_max:.2e}] (data range: {data_range:.2e})")

                # Use scientific notation for y-axis
                box_ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"histogram_analysis_{num_bins}.png"
        if self.output_dir:
            plot_path = os.path.join(self.output_dir, plot_filename)
        else:
            plot_path = plot_filename

        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_filename}")

        plt.show()

        # Save histogram data to Excel
        self.save_histogram_data_to_excel(histogram_data, num_bins)

        return plot_path

    def save_histogram_data_to_excel(self, histogram_data, num_bins):
        """Save histogram data to Excel file with separate sheets"""
        excel_filename = f"histogram_data_{num_bins}.xlsx"
        if self.output_dir:
            excel_path = os.path.join(self.output_dir, excel_filename)
        else:
            excel_path = excel_filename

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

            for data_type, hist_info in histogram_data.items():
                # Create DataFrame for this histogram
                df = pd.DataFrame({
                    'Bin_Centers': hist_info['bin_centers'],
                    'Counts': hist_info['counts'],
                    'Bin_Lower_Edge': hist_info['bin_edges'][:-1],
                    'Bin_Upper_Edge': hist_info['bin_edges'][1:]
                })

                # Calculate additional statistics
                original_values = self.extract_values(data_type)
                stats_df = pd.DataFrame({
                    'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                    'Value': [
                        len(original_values),
                        np.mean(original_values),
                        np.std(original_values),
                        np.min(original_values),
                        np.percentile(original_values, 25),
                        np.percentile(original_values, 50),
                        np.percentile(original_values, 75),
                        np.max(original_values)
                    ]
                })

                # Write histogram data to sheet
                df.to_excel(writer, sheet_name=f'{data_type.capitalize()}_Histogram', index=False)

                # Write statistics to the same sheet (starting from column F)
                stats_df.to_excel(writer, sheet_name=f'{data_type.capitalize()}_Histogram',
                                index=False, startcol=5)

                # Format the sheet
                worksheet = writer.sheets[f'{data_type.capitalize()}_Histogram']

                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter

                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass

                    adjusted_width = min(max_length + 2, 25)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

        print(f"Histogram data saved to: {excel_filename}")
        return excel_path

    def run(self):
        """Main execution method"""
        try:
            # Select input file
            input_file = self.select_input_file()
            if not input_file:
                print("No input file selected. Exiting.")
                return

            # Load and analyze data
            if not self.load_and_analyze_data(input_file):
                print("Failed to load data. Exiting.")
                return

            # Create standard plots (keff, total flux, thermal flux, epithermal flux, fast flux, box plot)
            plot_path = self.create_histograms_and_boxplots()

            # Automatically create energy distribution plots if percentage data is available
            if self.has_percentage_data():
                print(f"\nDetected percentage data! Creating energy distribution plots...")
                try:
                    self.create_energy_distribution_clear()
                except Exception as e:
                    print(f"Error creating energy distribution plots: {e}")

            if plot_path:
                print("\n" + "="*60)
                print("ANALYSIS COMPLETE")
                print("="*60)
                print(f"Files saved to: {self.output_dir}")
            else:
                print("No plots were generated.")

        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    generator = HistogramGenerator()
    generator.run()


if __name__ == "__main__":
    main()
