"""
Excel report generation for test results
"""

import pandas as pd
import numpy as np
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


class ExcelReporter:
    """Class to handle Excel report generation"""

    def create_report(self, results, output_filename):
        """Create comprehensive Excel report with test results"""
        # Convert results to DataFrame
        df = pd.DataFrame(results)

        # Determine which types of results we have
        has_keff = any('keff_real' in r for r in results if r)
        has_flux = any('I_1_real' in r for r in results if r)

        # Create Excel writer
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            # Create different sheets based on model type
            self._create_results_sheet(writer, df, has_keff, has_flux)
            self._create_summary_sheet(writer, df, has_keff, has_flux)
            self._create_best_models_sheet(writer, df, has_keff, has_flux)

        print(f"\nExcel report saved to: {output_filename}")
        print(f"Sheets created:")
        print(f"  - Test Results: Detailed results for all configurations")
        print(f"  - Summary: Statistics grouped by model, encoding, and optimization")
        print(f"  - Best Models by Encoding: Best performing model for each encoding method")

        return output_filename

    def _create_results_sheet(self, writer, df, has_keff, has_flux):
        """Create the main results sheet with merged keff and flux rows"""

        # Define the key columns for matching rows (excluding in_training)
        key_columns = ['config_id', 'description', 'model_class', 'encoding', 'optimization_method']

        # If we have both keff and flux, merge them
        if has_keff and has_flux:
            # Separate keff and flux rows
            keff_df = df[df['model_type'] == 'keff'].copy()
            flux_df = df[df['model_type'] == 'flux'].copy()

            # Drop model_type from both as it's no longer needed after merge
            keff_df = keff_df.drop('model_type', axis=1, errors='ignore')
            flux_df = flux_df.drop('model_type', axis=1, errors='ignore')

            # Identify keff and flux specific columns
            keff_specific_cols = ['keff_real', 'keff_predicted', 'keff_rel_error']
            if 'mape_keff' in keff_df.columns:
                keff_specific_cols.append('mape_keff')

            flux_specific_cols = []
            for i in range(1, 5):
                flux_specific_cols.extend([f'I_{i}_real', f'I_{i}_predicted', f'I_{i}_rel_error'])
            flux_specific_cols.extend(['avg_flux_real', 'avg_flux_predicted', 'avg_flux_rel_error'])
            if 'mape_flux' in flux_df.columns:
                flux_specific_cols.append('mape_flux')

            # Keep only relevant columns for each dataframe
            # Include in_training in the columns to keep
            keff_cols_to_keep = ['in_training'] + key_columns + [col for col in keff_specific_cols if col in keff_df.columns]
            flux_cols_to_keep = ['in_training'] + key_columns + [col for col in flux_specific_cols if col in flux_df.columns]

            keff_df = keff_df[keff_cols_to_keep]
            flux_df = flux_df[flux_cols_to_keep]

            # Merge on key columns AND in_training to ensure they match
            merge_columns = ['in_training'] + key_columns
            merged_df = pd.merge(keff_df, flux_df, on=merge_columns, how='outer')

            # Define column order with in_training first
            base_columns = ['in_training'] + key_columns
            keff_columns = [col for col in keff_specific_cols if col in merged_df.columns]
            flux_columns = [col for col in flux_specific_cols if col in merged_df.columns]

            column_order = base_columns + keff_columns + flux_columns
            df_ordered = merged_df[column_order]

        else:
            # If we only have one type, use the original logic
            # Put in_training first, then other base columns
            base_columns = ['in_training', 'config_id', 'description', 'model_class', 'model_type', 'encoding', 'optimization_method']

            keff_columns = []
            if has_keff:
                keff_columns = ['keff_real', 'keff_predicted', 'keff_rel_error']
                if 'mape_keff' in df.columns:
                    keff_columns.append('mape_keff')

            flux_columns = []
            if has_flux:
                for i in range(1, 5):
                    flux_columns.extend([f'I_{i}_real', f'I_{i}_predicted', f'I_{i}_rel_error'])
                flux_columns.extend(['avg_flux_real', 'avg_flux_predicted', 'avg_flux_rel_error'])
                if 'mape_flux' in df.columns:
                    flux_columns.append('mape_flux')

            column_order = base_columns + keff_columns + flux_columns
            column_order = [col for col in column_order if col in df.columns]
            df_ordered = df[column_order]

        # Write to Excel
        df_ordered.to_excel(writer, sheet_name='Test Results', index=False)

        # Format the sheet
        ws = writer.sheets['Test Results']
        self._format_worksheet(ws, df_ordered, has_keff, has_flux)

    def _create_summary_sheet(self, writer, df, has_keff, has_flux):
        """Create summary statistics sheet"""
        summary_data = []

        # Group by model characteristics
        grouping_cols = ['model_class', 'model_type', 'encoding', 'optimization_method']

        for name, group in df.groupby(grouping_cols):
            model_class, model_type, encoding, opt_method = name

            summary_row = {
                'Model': model_class,
                'Type': model_type,
                'Encoding': encoding,
                'Optimization': opt_method,
                'Configurations Tested': len(group)
            }

            if has_keff and model_type == 'keff' and 'keff_rel_error' in group:
                errors = group['keff_rel_error'].dropna()
                if len(errors) > 0:
                    summary_row['Avg K-eff Error (%)'] = errors.mean()
                    summary_row['Max K-eff Error (%)'] = errors.max()
                    summary_row['Min K-eff Error (%)'] = errors.min()
                    summary_row['Std K-eff Error (%)'] = errors.std()

            if has_flux and model_type == 'flux' and 'avg_flux_rel_error' in group:
                errors = group['avg_flux_rel_error'].dropna()
                if len(errors) > 0:
                    summary_row['Avg Flux Error (%)'] = errors.mean()
                    summary_row['Max Flux Error (%)'] = errors.max()
                    summary_row['Min Flux Error (%)'] = errors.min()
                    summary_row['Std Flux Error (%)'] = errors.std()

            summary_data.append(summary_row)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Format summary sheet
            ws_summary = writer.sheets['Summary']
            self._format_summary_sheet(ws_summary, summary_df)

    def _create_best_models_sheet(self, writer, df, has_keff, has_flux):
        """Create sheet showing best models per encoding"""
        best_models_data = []

        for encoding in df['encoding'].unique():
            encoding_subset = df[df['encoding'] == encoding]

            # Best k-eff model for this encoding
            if has_keff and len(encoding_subset[encoding_subset['model_type'] == 'keff']) > 0:
                keff_subset = encoding_subset[encoding_subset['model_type'] == 'keff']
                if 'mape_keff' in keff_subset:
                    # Group by model and optimization, then find minimum
                    grouped = keff_subset.groupby(['model_class', 'optimization_method'])['mape_keff'].mean()
                    if len(grouped) > 0:
                        best_idx = grouped.idxmin()
                        best_models_data.append({
                            'Encoding': encoding,
                            'Target': 'k-eff',
                            'Best Model': best_idx[0],
                            'Optimization': best_idx[1],
                            'Avg Error (%)': grouped.min()
                        })

            # Best flux model for this encoding
            if has_flux and len(encoding_subset[encoding_subset['model_type'] == 'flux']) > 0:
                flux_subset = encoding_subset[encoding_subset['model_type'] == 'flux']
                if 'mape_flux' in flux_subset:
                    grouped = flux_subset.groupby(['model_class', 'optimization_method'])['mape_flux'].mean()
                    if len(grouped) > 0:
                        best_idx = grouped.idxmin()
                        best_models_data.append({
                            'Encoding': encoding,
                            'Target': 'flux',
                            'Best Model': best_idx[0],
                            'Optimization': best_idx[1],
                            'Avg Error (%)': grouped.min()
                        })

        if best_models_data:
            best_models_df = pd.DataFrame(best_models_data)
            best_models_df.to_excel(writer, sheet_name='Best Models by Encoding', index=False)

            # Format best models sheet
            ws_best = writer.sheets['Best Models by Encoding']
            self._format_best_models_sheet(ws_best, best_models_df)

    def _format_worksheet(self, ws, df, has_keff, has_flux):
        """Apply formatting to the main results worksheet"""
        # Format headers
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment

        # Format columns
        for col_idx, col in enumerate(df.columns, 1):
            col_letter = get_column_letter(col_idx)

            # Format in_training column - center align T/F values
            if col == 'in_training':
                for row in range(2, len(df) + 2):
                    cell = ws[f"{col_letter}{row}"]
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    # Optionally add color coding
                    if cell.value == 'T':
                        cell.fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")  # Light green
                    else:
                        cell.fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")  # Light red

            # Format percentage/error columns
            elif 'rel_error' in col or 'Error' in col:
                for row in range(2, len(df) + 2):
                    cell = ws[f"{col_letter}{row}"]
                    if cell.value is not None and isinstance(cell.value, (int, float)):
                        cell.number_format = '0.00'

            # Format scientific notation for flux columns
            elif ('flux' in col or col.startswith('I_')) and 'error' not in col:
                for row in range(2, len(df) + 2):
                    cell = ws[f"{col_letter}{row}"]
                    if cell.value is not None and isinstance(cell.value, (int, float)):
                        cell.number_format = '0.00E+00'

            # Format k-eff values
            elif 'keff' in col and 'error' not in col:
                for row in range(2, len(df) + 2):
                    cell = ws[f"{col_letter}{row}"]
                    if cell.value is not None and isinstance(cell.value, (int, float)):
                        cell.number_format = '0.000000'

        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = get_column_letter(column[0].column)

            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass

            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width

    def _format_summary_sheet(self, ws, df):
        """Apply formatting to summary sheet"""
        # Format headers
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_alignment = Alignment(horizontal="center", vertical="center")

        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment

        # Format percentage columns
        for col_idx, col in enumerate(df.columns, 1):
            if 'Error' in col:
                col_letter = get_column_letter(col_idx)
                for row in range(2, len(df) + 2):
                    cell = ws[f"{col_letter}{row}"]
                    if cell.value is not None:
                        cell.number_format = '0.00'

        # Auto-adjust columns
        for column in ws.columns:
            max_length = 0
            column_letter = get_column_letter(column[0].column)
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width

    def _format_best_models_sheet(self, ws, df):
        """Apply formatting to best models sheet"""
        # Format headers
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_alignment = Alignment(horizontal="center", vertical="center")

        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment

        # Format error column
        if 'Avg Error (%)' in df.columns:
            col_idx = df.columns.get_loc('Avg Error (%)') + 1
            col_letter = get_column_letter(col_idx)
            for row in range(2, len(df) + 2):
                cell = ws[f"{col_letter}{row}"]
                if cell.value is not None:
                    cell.number_format = '0.00'

        # Auto-adjust columns
        for column in ws.columns:
            max_length = 0
            column_letter = get_column_letter(column[0].column)
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
