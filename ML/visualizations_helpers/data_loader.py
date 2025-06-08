"""
Data loader module for test results
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

    # Calculate R² for keff models if not present
    if 'r2_keff' not in df.columns:
        df['r2_keff'] = calculate_r2_scores(df, 'keff')

    return df

def calculate_r2_scores(df, target_type):
    """Calculate R² scores for grouped model configurations"""
    r2_scores = []

    for idx, row in df.iterrows():
        if target_type == 'flux':
            # For flux, calculate R² across all positions
            actual_values = []
            predicted_values = []
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

def get_model_aggregated_metrics(df):
    """Calculate aggregated metrics for each model configuration"""
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

        # For keff models
        if 'keff_real' in group.columns and 'keff_predicted' in group.columns:
            actual_keff = group['keff_real'].dropna()
            predicted_keff = group['keff_predicted'].dropna()

            if len(actual_keff) > 0:
                result['mean_mape_keff'] = np.mean(np.abs((predicted_keff - actual_keff) / actual_keff) * 100)
                result['max_mape_keff'] = np.max(np.abs((predicted_keff - actual_keff) / actual_keff) * 100)
                result['r2_keff'] = calculate_r2(actual_keff, predicted_keff)

        aggregated.append(result)

    return pd.DataFrame(aggregated)
