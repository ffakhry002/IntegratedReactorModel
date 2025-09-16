import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
base_path = Path("/Users/dima/Downloads/Thesis data/Machine Learning")

print("Reading data files...")

# Read the combined energy model data
combined_df = pd.read_excel("multi_energy.xlsx")
eigenvalue_df = pd.read_excel("../../eigenvalue/excel_reports/eigenvalue.xlsx")

print(f"Combined data shape: {combined_df.shape}")
print(f"Eigenvalue data shape: {eigenvalue_df.shape}")

# CONFIGURABLE PARAMETERS - Change these to filter different models
MODEL_CLASS = "svm"  # Options: "xgboost", "random_forest", "svm"
ENCODING = "physics"     # Options: "physics", "categorical", "one_hot", "graph"
OPTIMIZATION = "three_stage"  # Options: "three_stage", "optuna", "none"

print(f"\n" + "="*80)
print(f"FILTERING DATA FOR:")
print(f"Model Class: {MODEL_CLASS}")
print(f"Encoding: {ENCODING}")
print(f"Optimization: {OPTIMIZATION}")
print("="*80)

# Function to filter and extract individual I values
def filter_and_extract_individual_values(df, model_class, encoding, optimization, data_type="combined"):
    """Filter data by model parameters and extract individual I_1 to I_4 values"""

    # Filter the data
    filtered_df = df[
        (df['model_class'] == model_class) &
        (df['encoding'] == encoding) &
        (df['optimization_method'] == optimization)
    ].copy()

    print(f"\n{data_type.upper()} DATA:")
    print(f"Original rows: {len(df)}")
    print(f"After filtering: {len(filtered_df)}")

    if len(filtered_df) == 0:
        print(f"WARNING: No data found for {model_class} + {encoding} + {optimization}")
        return pd.DataFrame()

    results = []

    for idx, row in filtered_df.iterrows():
        base_result = {
            'config_id': row['config_id'],
            'description': row.get('description', ''),
        }

        if data_type == "combined":
            # Extract all individual I_1 to I_4 values for thermal, epithermal, fast
            for i in range(1, 5):
                result = base_result.copy()
                result['I_position'] = f'I_{i}'
                result['thermal_error'] = row[f'I_{i}_thermal_rel_error']
                result['epithermal_error'] = row[f'I_{i}_epithermal_rel_error']
                result['fast_error'] = row[f'I_{i}_fast_rel_error']
                results.append(result)

        elif data_type == "eigenvalue":
            # For eigenvalue, we don't have I_1 to I_4, just keff_rel_error
            # We'll create 4 identical entries to match the structure
            for i in range(1, 5):
                result = base_result.copy()
                result['I_position'] = f'I_{i}'
                result['eigenvalue_error'] = row['keff_rel_error']
                results.append(result)

    return pd.DataFrame(results)

# Filter and process both datasets
combined_filtered = filter_and_extract_individual_values(combined_df, MODEL_CLASS, ENCODING, OPTIMIZATION, "combined")
eigenvalue_filtered = filter_and_extract_individual_values(eigenvalue_df, MODEL_CLASS, ENCODING, OPTIMIZATION, "eigenvalue")

# Merge on config_id and I_position
if not combined_filtered.empty and not eigenvalue_filtered.empty:
    print(f"\nMERGING DATA:")
    merged_data = pd.merge(combined_filtered, eigenvalue_filtered[['config_id', 'I_position', 'eigenvalue_error']],
                          on=['config_id', 'I_position'], how='inner')
    print(f"Merged data shape: {merged_data.shape}")
    print(f"Unique configurations: {len(merged_data['config_id'].unique())}")
    print(f"Total data points (configs × I positions): {len(merged_data)}")

    if not merged_data.empty:
        # Show sample data
        print(f"\nSample merged data:")
        print(merged_data[['config_id', 'I_position', 'thermal_error', 'epithermal_error', 'fast_error', 'eigenvalue_error']].head(8))

        # Create three different versions of the pair plot data

        # VERSION 1: Original errors (no transformation)
        original_data = merged_data[['thermal_error', 'epithermal_error', 'fast_error', 'eigenvalue_error']].copy()
        original_data.columns = ['Thermal', 'Epithermal', 'Fast', 'Eigenvalue']

        # VERSION 2: Log-transformed errors (add constant to handle negative values)
        # Add constant to make all values positive before log transform
        min_val = min(original_data.min().min(), -100)  # Ensure we handle negative errors
        offset = abs(min_val) + 1  # Add 1 to avoid log(0)

        log_data = original_data.copy()
        for col in log_data.columns:
            log_data[f'{col}_Log'] = np.log10(log_data[col] + offset)

        log_transformed = log_data[['Thermal_Log', 'Epithermal_Log', 'Fast_Log', 'Eigenvalue_Log']].copy()
        log_transformed.columns = ['Log(Thermal)', 'Log(Epithermal)', 'Log(Fast)', 'Log(Eigenvalue)']

        # VERSION 3: Log thermal only
        log_thermal_only = original_data.copy()
        log_thermal_only['Thermal'] = np.log10(log_thermal_only['Thermal'] + offset)
        log_thermal_only.columns = ['Log(Thermal)', 'Epithermal', 'Fast', 'Eigenvalue']

        print(f"\n" + "="*80)
        print("CREATING THREE VERSIONS OF PAIR PLOTS:")
        print("="*80)

        # Function to create and save pair plot
        def create_pair_plot(data, title_suffix, file_suffix, data_description):
            print(f"\nCreating {data_description}...")

            plt.figure(figsize=(14, 12))

            # Create pair plot with seaborn
            g = sns.pairplot(data, diag_kind='hist', plot_kws={'alpha': 0.4, 's': 20})

            # Customize the plot
            g.fig.suptitle(f'Pair Plot: Individual I₁-I₄ Relative Errors {title_suffix}\n' +
                          f'Model: {MODEL_CLASS} | Encoding: {ENCODING} | Optimization: {OPTIMIZATION}',
                          fontsize=16, y=1.02)

            # Add correlation values to the upper triangle
            for i in range(len(data.columns)):
                for j in range(i+1, len(data.columns)):
                    corr = data.iloc[:, i].corr(data.iloc[:, j])
                    g.axes[i, j].text(0.5, 0.5, f'r = {corr:.3f}',
                                    transform=g.axes[i, j].transAxes,
                                    ha='center', va='center', fontsize=12,
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            # Add trend lines to lower triangle
            for i in range(len(data.columns)):
                for j in range(i):
                    sns.regplot(data=data, x=data.columns[j], y=data.columns[i],
                              ax=g.axes[i, j], scatter=False, color='red', line_kws={'alpha': 0.8})

            plt.tight_layout()

            # Save the plot
            output_file = f"individual_I_pairplot_{MODEL_CLASS}_{ENCODING}_{OPTIMIZATION}_{file_suffix}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Pair plot saved to: {output_file}")

            plt.show()

            # Print correlation matrix
            correlation_matrix = data.corr()
            print(f"\nCorrelation Matrix ({data_description}):")
            print(correlation_matrix.round(3))

            return correlation_matrix

        # Create all three versions
        print(f"\n1. ORIGINAL ERRORS (no transformation)")
        corr_original = create_pair_plot(original_data, "(Original Scale)", "original", "Original Errors")

        print(f"\n2. LOG-TRANSFORMED ALL ERRORS")
        corr_log_all = create_pair_plot(log_transformed, "(Log All Errors)", "log_all", "Log All Errors")

        print(f"\n3. LOG-TRANSFORMED THERMAL ONLY")
        corr_log_thermal = create_pair_plot(log_thermal_only, "(Log Thermal Only)", "log_thermal", "Log Thermal Only")

        # Compare correlations
        print(f"\n" + "="*80)
        print("CORRELATION COMPARISON:")
        print("="*80)

        # Compare Epithermal-Fast correlations
        epithermal_fast_orig = corr_original.loc['Epithermal', 'Fast']
        epithermal_fast_log_all = corr_log_all.loc['Log(Epithermal)', 'Log(Fast)']
        epithermal_fast_log_thermal = corr_log_thermal.loc['Epithermal', 'Fast']

        print(f"\nEpithermal-Fast Correlation:")
        print(f"  Original:           r = {epithermal_fast_orig:.3f}")
        print(f"  Log All Errors:     r = {epithermal_fast_log_all:.3f}")
        print(f"  Log Thermal Only:   r = {epithermal_fast_log_thermal:.3f}")

        # Compare Thermal-Fast correlations
        thermal_fast_orig = corr_original.loc['Thermal', 'Fast']
        thermal_fast_log_all = corr_log_all.loc['Log(Thermal)', 'Log(Fast)']
        thermal_fast_log_thermal = corr_log_thermal.loc['Log(Thermal)', 'Fast']

        print(f"\nThermal-Fast Correlation:")
        print(f"  Original:           r = {thermal_fast_orig:.3f}")
        print(f"  Log All Errors:     r = {thermal_fast_log_all:.3f}")
        print(f"  Log Thermal Only:   r = {thermal_fast_log_thermal:.3f}")

        # Compare Thermal-Epithermal correlations
        thermal_epithermal_orig = corr_original.loc['Thermal', 'Epithermal']
        thermal_epithermal_log_all = corr_log_all.loc['Log(Thermal)', 'Log(Epithermal)']
        thermal_epithermal_log_thermal = corr_log_thermal.loc['Log(Thermal)', 'Epithermal']

        print(f"\nThermal-Epithermal Correlation:")
        print(f"  Original:           r = {thermal_epithermal_orig:.3f}")
        print(f"  Log All Errors:     r = {thermal_epithermal_log_all:.3f}")
        print(f"  Log Thermal Only:   r = {thermal_epithermal_log_thermal:.3f}")

        print(f"\n" + "="*80)
        print("ANALYSIS:")
        print("="*80)
        print("• Original: Shows raw error correlations")
        print("• Log All: May reduce correlations if errors span wide ranges")
        print("• Log Thermal: May change thermal correlations while preserving others")
        print("• Check which transformation gives most meaningful patterns")

    else:
        print("No matching configurations found after merge!")

else:
    print("No data available after filtering!")
