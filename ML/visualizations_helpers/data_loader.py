"""
Data loader module for test results
UPDATED: Added support for energy-discretized results
"""

import pandas as pd
import numpy as np

def load_test_results(excel_file_path):
    """Load and prepare test results from Excel file"""
    # Read the main test results sheet
    try:
        df = pd.read_excel(excel_file_path, sheet_name='Test Results')
    except Exception as e:
        print(f"ERROR: Could not read Excel file: {e}")
        raise

    # Check for required columns
    required_cols = ['model_class', 'encoding', 'optimization_method', 'config_id']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"WARNING: Missing required columns: {missing_cols}")
        print("Available columns:", list(df.columns))

    # Detect if we have energy-discretized data
    has_energy_data = any('_thermal_' in col or '_epithermal_' in col or '_fast_' in col
                         for col in df.columns)

    if has_energy_data:
        # Calculate MAPE for each energy group if not present
        for energy_group in ['thermal', 'epithermal', 'fast', 'total']:
            mape_col = f'mape_{energy_group}_flux'
            if mape_col not in df.columns:
                flux_errors = []
                for idx, row in df.iterrows():
                    errors = []
                    for i in range(1, 5):
                        real_col = f'I_{i}_{energy_group}_real'
                        pred_col = f'I_{i}_{energy_group}_predicted'
                        if real_col in row and pred_col in row:
                            real = row[real_col]
                            pred = row[pred_col]
                            if pd.notna(real) and pd.notna(pred) and real != 0 and real != 'N/A':
                                errors.append(abs((pred - real) / real) * 100)
                    if errors:
                        flux_errors.append(np.mean(errors))
                    else:
                        flux_errors.append(np.nan)
                df[mape_col] = flux_errors

        # Calculate R² for each energy group
        for energy_group in ['thermal', 'epithermal', 'fast', 'total']:
            r2_col = f'r2_{energy_group}_flux'
            if r2_col not in df.columns:
                df[r2_col] = calculate_r2_scores(df, 'flux', energy_group=energy_group)
    else:
        # Standard processing for non-energy data
        # Calculate additional metrics if not present
        if 'I_1_real' in df.columns and 'mape_flux' not in df.columns:
            # Calculate MAPE for flux if not already present
            flux_errors = []
            for idx, row in df.iterrows():
                errors = []
                for i in range(1, 5):
                    if f'I_{i}_real' in row and f'I_{i}_predicted' in row:
                        real = row[f'I_{i}_real']
                        pred = row[f'I_{i}_predicted']
                        if pd.notna(real) and pd.notna(pred) and real != 0:
                            errors.append(abs((pred - real) / real) * 100)
                if errors:
                    flux_errors.append(np.mean(errors))
                else:
                    flux_errors.append(np.nan)
            df['mape_flux'] = flux_errors

        # Calculate R² for flux models if not present
        if 'r2_flux' not in df.columns:
            df['r2_flux'] = calculate_r2_scores(df, 'flux')

    # Calculate R² for keff models if not present (same for both energy and non-energy)
    if 'r2_keff' not in df.columns:
        df['r2_keff'] = calculate_r2_scores(df, 'keff')

    return df

def calculate_r2_scores(df, target_type, energy_group=None):
    """Calculate R² scores for grouped model configurations"""
    r2_scores = []

    for idx, row in df.iterrows():
        if target_type == 'flux':
            # For flux, calculate R² across all positions
            actual_values = []
            predicted_values = []

            if energy_group:
                # Energy-specific calculation
                for i in range(1, 5):
                    real_col = f'I_{i}_{energy_group}_real'
                    pred_col = f'I_{i}_{energy_group}_predicted'
                    if real_col in row and pred_col in row:
                        real = row[real_col]
                        pred = row[pred_col]
                        if pd.notna(real) and pd.notna(pred) and real != 'N/A' and pred != 'N/A':
                            actual_values.append(real)
                            predicted_values.append(pred)
            else:
                # Standard calculation
                for i in range(1, 5):
                    if f'I_{i}_real' in row and f'I_{i}_predicted' in row:
                        real = row[f'I_{i}_real']
                        pred = row[f'I_{i}_predicted']
                        if pd.notna(real) and pd.notna(pred):
                            actual_values.append(real)
                            predicted_values.append(pred)

            if len(actual_values) >= 2:
                r2 = calculate_r2(actual_values, predicted_values)
                r2_scores.append(r2)
            else:
                r2_scores.append(np.nan)

        else:  # keff
            if 'keff_real' in row and 'keff_predicted' in row:
                # For single keff value, we need to aggregate by model
                r2_scores.append(np.nan)  # Will be calculated separately
            else:
                r2_scores.append(np.nan)

    return r2_scores

def calculate_r2(actual, predicted):
    """Calculate R² score"""
    actual = np.array(actual)
    predicted = np.array(predicted)

    ss_tot = np.sum((actual - np.mean(actual))**2)
    ss_res = np.sum((actual - predicted)**2)

    if ss_tot == 0:
        return 0

    return 1 - (ss_res / ss_tot)

def get_model_aggregated_metrics(df, energy_group=None):
    """
    Calculate aggregated metrics for each model configuration

    Args:
        df: DataFrame with test results
        energy_group: Energy group to analyze ('thermal', 'epithermal', 'fast', 'total')
    """
    # Group by model configuration
    grouped = df.groupby(['model_class', 'encoding', 'optimization_method'])

    aggregated = []
    for name, group in grouped:
        model_class, encoding, optimization = name

        # Calculate aggregated metrics
        result = {
            'model_class': model_class,
            'encoding': encoding,
            'optimization_method': optimization,
            'n_configs': len(group)
        }

        if energy_group:
            # Energy-specific metrics
            mape_col = f'mape_{energy_group}_flux'
            if mape_col in group.columns:
                result['mean_mape_flux'] = group[mape_col].mean()
                result['max_mape_flux'] = group[mape_col].max()
                result['std_mape_flux'] = group[mape_col].std()

            # Calculate overall R² for this energy group
            all_actual = []
            all_predicted = []
            for _, row in group.iterrows():
                for i in range(1, 5):
                    real_col = f'I_{i}_{energy_group}_real'
                    pred_col = f'I_{i}_{energy_group}_predicted'
                    if real_col in row and pred_col in row:
                        real = row[real_col]
                        pred = row[pred_col]
                        if pd.notna(real) and pd.notna(pred) and real != 'N/A' and pred != 'N/A':
                            all_actual.append(real)
                            all_predicted.append(pred)

            if len(all_actual) > 0:
                result['r2_flux'] = calculate_r2(all_actual, all_predicted)
        else:
            # Standard metrics
            # For flux models
            if 'mape_flux' in group.columns:
                result['mean_mape_flux'] = group['mape_flux'].mean()
                result['max_mape_flux'] = group['mape_flux'].max()
                result['std_mape_flux'] = group['mape_flux'].std()

                # Calculate overall R² for flux
                all_actual = []
                all_predicted = []
                for _, row in group.iterrows():
                    for i in range(1, 5):
                        if f'I_{i}_real' in row and f'I_{i}_predicted' in row:
                            real = row[f'I_{i}_real']
                            pred = row[f'I_{i}_predicted']
                            if pd.notna(real) and pd.notna(pred):
                                all_actual.append(real)
                                all_predicted.append(pred)

                if len(all_actual) > 0:
                    result['r2_flux'] = calculate_r2(all_actual, all_predicted)

        # For keff models (same for both energy and non-energy)
        if 'keff_real' in group.columns and 'keff_predicted' in group.columns:
            actual_keff = group['keff_real'].dropna()
            predicted_keff = group['keff_predicted'].dropna()

            if len(actual_keff) > 0:
                result['mean_mape_keff'] = np.mean(np.abs((predicted_keff - actual_keff) / actual_keff) * 100)
                result['max_mape_keff'] = np.max(np.abs((predicted_keff - actual_keff) / actual_keff) * 100)
                result['r2_keff'] = calculate_r2(actual_keff, predicted_keff)

        aggregated.append(result)

    return pd.DataFrame(aggregated)
