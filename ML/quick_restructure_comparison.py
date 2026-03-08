"""
Quick comparison: Current structure (4 models) vs Restructured (1 model)
Tests whether pooling positions improves SVR performance for vacuum mode TOTAL FLUX.

Current:     4 SVR models (one per position)
Restructured: 1 SVR model (all positions pooled → 4× training data)

Run from ML directory:
    python quick_restructure_comparison.py
"""

import numpy as np
import sys
import os
import time
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set to vacuum mode
os.environ['NCI_DISTANCE_CUTOFF'] = '1'

print("=" * 70)
print("QUICK COMPARISON: Current (4 models) vs Restructured (1 model)")
print("Testing with TOTAL FLUX (vacuum mode)")
print("=" * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n1. Loading data...")

from execution.data_handler import DataHandler
from ML_models.encodings import encoding_methods

# Force vacuum mode
encoding_methods.IRRADIATION_MODE = 'vacuum'
encoding_methods.NCI_MODE = 'single'

handler = DataHandler()
X, y_flux, y_keff, groups, lattices = handler.load_and_prepare_data(
    '/Users/dima/Downloads/Thesis data/Machine Learning/train_test_data/train.txt',
    encoding_method='physics',
    flux_mode='total'  # 4 outputs: total flux at each of 4 positions
)

print(f"\nData loaded:")
print(f"  X shape: {X.shape}")
print(f"  y_flux shape: {y_flux.shape}")
print(f"  Unique configs: {len(np.unique(groups))}")
print(f"  Total samples (with augmentation): {X.shape[0]}")

# Verify feature structure
n_features = X.shape[0]
print(f"\n  Vacuum mode physics encoding: {X.shape[1]} features")
print(f"    - 2 global (avg_dist, symmetry)")
print(f"    - 16 local (4 per position × 4 positions)")
print(f"    - 4 NCI (1 per position)")

# =============================================================================
# RESTRUCTURE DATA FOR POSITION POOLING
# =============================================================================
print("\n2. Restructuring data for position pooling...")

def restructure_for_position_pooling(X, y_flux, groups):
    """
    Restructure so each row is one (config, position) pair.

    Vacuum mode physics encoding (22 features total):
    - 2 global features
    - 4 local features per position × 4 positions = 16
    - 1 NCI per position × 4 positions = 4

    After restructuring (7 features per row):
    - 2 global + 4 local + 1 NCI = 7 features per position

    Targets:
    - 1 total flux value per position
    """
    n_configs = X.shape[0]
    n_positions = 4

    # Vacuum mode structure
    n_global = 2
    n_local_per_pos = 4
    n_nci_per_pos = 1

    X_new, y_new, groups_new = [], [], []

    for idx in range(n_configs):
        # Global features (shared across all positions)
        global_feats = X[idx, :n_global]

        for pos in range(n_positions):
            # Local features for THIS position
            local_start = n_global + pos * n_local_per_pos
            local_feats = X[idx, local_start:local_start + n_local_per_pos]

            # NCI for THIS position
            nci_start = n_global + n_positions * n_local_per_pos + pos * n_nci_per_pos
            nci_feats = X[idx, nci_start:nci_start + n_nci_per_pos]

            # Combine into position-specific feature vector
            pos_features = np.concatenate([global_feats, local_feats, nci_feats])
            X_new.append(pos_features)

            # Target: total flux for THIS position (single value)
            y_new.append(y_flux[idx, pos])

            # Same group for all positions of same config
            groups_new.append(groups[idx])

    return np.array(X_new), np.array(y_new), np.array(groups_new)

X_pooled, y_pooled, groups_pooled = restructure_for_position_pooling(X, y_flux, groups)

print(f"\nRestructured data:")
print(f"  X_pooled shape: {X_pooled.shape}  (was {X.shape})")
print(f"  y_pooled shape: {y_pooled.shape}  (was {y_flux.shape})")
print(f"  Effective training samples: {X_pooled.shape[0]} (4× original!)")

# =============================================================================
# SVR PARAMETERS
# Using FIXED gamma for fair apples-to-apples comparison
# =============================================================================
svr_params = {
    'kernel': 'rbf',
    'C': 100.0,
    'gamma': 0.01,  # Fixed for all - fair comparison
    'epsilon': 0.01,
    'cache_size': 2000,
    'max_iter': 20000,  # Reduced for speed
    'verbose': False,
}

print(f"\nUsing FIXED hyperparameters for fair comparison:")
print(f"  C={svr_params['C']}, gamma={svr_params['gamma']}, epsilon={svr_params['epsilon']}")

n_folds = 5

# =============================================================================
# METHOD 1: CURRENT APPROACH (4 separate SVR models)
# =============================================================================
print(f"\n3. Testing CURRENT approach (4 separate SVR models)...")
print(f"   Each SVR sees all 22 features, predicts 1 position's flux")

# Create model - MultiOutputRegressor trains 4 separate SVRs
model_current = MultiOutputRegressor(
    Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(**svr_params))
    ]),
    n_jobs=4
)

# Cross-validation with GroupKFold
cv = GroupKFold(n_splits=n_folds)
start_time = time.time()

scores_current = {'mape': [], 'rmse': [], 'r2': []}
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_flux, groups)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_flux[train_idx], y_flux[val_idx]

    model_current.fit(X_train, y_train)
    y_pred = model_current.predict(X_val)

    # Convert from log scale for MAPE calculation
    y_val_orig = 10 ** y_val
    y_pred_orig = 10 ** y_pred

    # MAPE (on original scale)
    mape = np.mean(np.abs((y_val_orig - y_pred_orig) / (y_val_orig + 1e-10))) * 100
    # RMSE (on log scale - what we trained on)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    # R² (on log scale)
    r2 = r2_score(y_val.flatten(), y_pred.flatten())

    scores_current['mape'].append(mape)
    scores_current['rmse'].append(rmse)
    scores_current['r2'].append(r2)
    print(f"   Fold {fold+1}/{n_folds}: MAPE={mape:.2f}%, RMSE={rmse:.4f}, R²={r2:.4f}")

time_current = time.time() - start_time
mean_mape_current = np.mean(scores_current['mape'])
std_mape_current = np.std(scores_current['mape'])
mean_rmse_current = np.mean(scores_current['rmse'])
mean_r2_current = np.mean(scores_current['r2'])

print(f"\n   CURRENT APPROACH RESULTS:")
print(f"   Mean MAPE: {mean_mape_current:.2f}% ± {std_mape_current:.2f}%")
print(f"   Mean RMSE: {mean_rmse_current:.4f} (log scale)")
print(f"   Mean R²:   {mean_r2_current:.4f}")
print(f"   Time: {time_current:.1f}s")
print(f"   Models trained: 4 (one per position)")

# =============================================================================
# METHOD 2: RESTRUCTURED APPROACH (1 model, positions pooled)
# =============================================================================
print(f"\n4. Testing RESTRUCTURED approach (1 SVR model, positions pooled)...")
print(f"   Single SVR sees 7 position-specific features")
print(f"   Training data: {X_pooled.shape[0]} samples (4× more!)")

# Create single SVR model (not MultiOutputRegressor!)
model_pooled = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(**svr_params))
])

# Cross-validation
cv_pooled = GroupKFold(n_splits=n_folds)
start_time = time.time()

scores_pooled = {'mape': [], 'rmse': [], 'r2': []}
for fold, (train_idx, val_idx) in enumerate(cv_pooled.split(X_pooled, y_pooled, groups_pooled)):
    X_train, X_val = X_pooled[train_idx], X_pooled[val_idx]
    y_train, y_val = y_pooled[train_idx], y_pooled[val_idx]

    model_pooled.fit(X_train, y_train)
    y_pred = model_pooled.predict(X_val)

    # Convert from log scale for MAPE calculation
    y_val_orig = 10 ** y_val
    y_pred_orig = 10 ** y_pred

    # MAPE (on original scale)
    mape = np.mean(np.abs((y_val_orig - y_pred_orig) / (y_val_orig + 1e-10))) * 100
    # RMSE (on log scale)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    # R² (on log scale)
    r2 = r2_score(y_val, y_pred)

    scores_pooled['mape'].append(mape)
    scores_pooled['rmse'].append(rmse)
    scores_pooled['r2'].append(r2)
    print(f"   Fold {fold+1}/{n_folds}: MAPE={mape:.2f}%, RMSE={rmse:.4f}, R²={r2:.4f}")

time_pooled = time.time() - start_time
mean_mape_pooled = np.mean(scores_pooled['mape'])
std_mape_pooled = np.std(scores_pooled['mape'])
mean_rmse_pooled = np.mean(scores_pooled['rmse'])
mean_r2_pooled = np.mean(scores_pooled['r2'])

print(f"\n   RESTRUCTURED APPROACH RESULTS:")
print(f"   Mean MAPE: {mean_mape_pooled:.2f}% ± {std_mape_pooled:.2f}%")
print(f"   Mean RMSE: {mean_rmse_pooled:.4f} (log scale)")
print(f"   Mean R²:   {mean_r2_pooled:.4f}")
print(f"   Time: {time_pooled:.1f}s")
print(f"   Models trained: 1 (pooled all positions)")

# =============================================================================
# METHOD 3: HYBRID - 1 model, all 22 features, but reordered per position
# =============================================================================
print(f"\n5. Testing HYBRID approach (1 model, all 22 features, reordered)...")
print(f"   'My' position's features always come first, others follow")
print(f"   Model sees full context but learns position-invariant patterns")

def restructure_hybrid(X, y_flux, groups):
    """
    Restructure with ALL features, but reorder so 'my' position is always first.

    For each (config, position) pair:
    - Features: [global, MY_local, MY_nci, OTHER_local, OTHER_nci]
    - Target: MY flux

    This way the model learns:
    - Indices 0-1: global features
    - Indices 2-5: MY local environment (always in same place!)
    - Index 6: MY NCI
    - Indices 7-18: OTHER positions' local features (context)
    - Indices 19-21: OTHER positions' NCI
    """
    n_configs = X.shape[0]
    n_positions = 4

    # Vacuum mode structure
    n_global = 2
    n_local_per_pos = 4
    n_nci_per_pos = 1

    X_new, y_new, groups_new = [], [], []

    for idx in range(n_configs):
        global_feats = X[idx, :n_global]

        # Extract all local features
        all_local = []
        for pos in range(n_positions):
            start = n_global + pos * n_local_per_pos
            all_local.append(X[idx, start:start + n_local_per_pos])

        # Extract all NCI features
        nci_start = n_global + n_positions * n_local_per_pos
        all_nci = X[idx, nci_start:nci_start + n_positions]

        for my_pos in range(n_positions):
            # MY features first
            my_local = all_local[my_pos]
            my_nci = all_nci[my_pos:my_pos+1]

            # OTHER features after
            other_local = np.concatenate([all_local[p] for p in range(n_positions) if p != my_pos])
            other_nci = np.array([all_nci[p] for p in range(n_positions) if p != my_pos])

            # Combine: [global, MY_local, MY_nci, OTHER_local, OTHER_nci]
            pos_features = np.concatenate([global_feats, my_local, my_nci, other_local, other_nci])
            X_new.append(pos_features)

            # Target: MY flux
            y_new.append(y_flux[idx, my_pos])
            groups_new.append(groups[idx])

    return np.array(X_new), np.array(y_new), np.array(groups_new)

X_hybrid, y_hybrid, groups_hybrid = restructure_hybrid(X, y_flux, groups)

print(f"\n   Hybrid data:")
print(f"   X_hybrid shape: {X_hybrid.shape}")
print(f"   Feature layout: [global₂, MY_local₄, MY_nci₁, OTHER_local₁₂, OTHER_nci₃]")
print(f"   Training samples: {X_hybrid.shape[0]} (4× original)")

# Create model
model_hybrid = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(**svr_params))
])

# Cross-validation
cv_hybrid = GroupKFold(n_splits=n_folds)
start_time = time.time()

scores_hybrid = {'mape': [], 'rmse': [], 'r2': []}
for fold, (train_idx, val_idx) in enumerate(cv_hybrid.split(X_hybrid, y_hybrid, groups_hybrid)):
    X_train, X_val = X_hybrid[train_idx], X_hybrid[val_idx]
    y_train, y_val = y_hybrid[train_idx], y_hybrid[val_idx]

    model_hybrid.fit(X_train, y_train)
    y_pred = model_hybrid.predict(X_val)

    # Convert from log scale for MAPE calculation
    y_val_orig = 10 ** y_val
    y_pred_orig = 10 ** y_pred

    # MAPE (on original scale)
    mape = np.mean(np.abs((y_val_orig - y_pred_orig) / (y_val_orig + 1e-10))) * 100
    # RMSE (on log scale)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    # R² (on log scale)
    r2 = r2_score(y_val, y_pred)

    scores_hybrid['mape'].append(mape)
    scores_hybrid['rmse'].append(rmse)
    scores_hybrid['r2'].append(r2)
    print(f"   Fold {fold+1}/{n_folds}: MAPE={mape:.2f}%, RMSE={rmse:.4f}, R²={r2:.4f}")

time_hybrid = time.time() - start_time
mean_mape_hybrid = np.mean(scores_hybrid['mape'])
std_mape_hybrid = np.std(scores_hybrid['mape'])
mean_rmse_hybrid = np.mean(scores_hybrid['rmse'])
mean_r2_hybrid = np.mean(scores_hybrid['r2'])

print(f"\n   HYBRID APPROACH RESULTS:")
print(f"   Mean MAPE: {mean_mape_hybrid:.2f}% ± {std_mape_hybrid:.2f}%")
print(f"   Mean RMSE: {mean_rmse_hybrid:.4f} (log scale)")
print(f"   Mean R²:   {mean_r2_hybrid:.4f}")
print(f"   Time: {time_hybrid:.1f}s")
print(f"   Models trained: 1 (with full context)")

# =============================================================================
# COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

print(f"\n{'Approach':<38} {'Feat':<6} {'MAPE':<18} {'RMSE':<10} {'R²':<8} {'Time':<8} {'Models'}")
print("-" * 105)
print(f"{'1. Current (4 separate models)':<38} {X.shape[1]:<6} {mean_mape_current:.2f}% ± {std_mape_current:.2f}%  {mean_rmse_current:.4f}    {mean_r2_current:.4f}   {time_current:.1f}s     4")
print(f"{'2. Restructured (local only)':<38} {X_pooled.shape[1]:<6} {mean_mape_pooled:.2f}% ± {std_mape_pooled:.2f}%  {mean_rmse_pooled:.4f}    {mean_r2_pooled:.4f}   {time_pooled:.1f}s     1")
print(f"{'3. Hybrid (full context, reordered)':<38} {X_hybrid.shape[1]:<6} {mean_mape_hybrid:.2f}% ± {std_mape_hybrid:.2f}%  {mean_rmse_hybrid:.4f}    {mean_r2_hybrid:.4f}   {time_hybrid:.1f}s     1")
print("-" * 105)

# Find best approach
results = [
    ("Current (4 models)", mean_mape_current, "4 separate SVRs, each sees all features"),
    ("Restructured (7 feat)", mean_mape_pooled, "1 SVR, only position-local features"),
    ("Hybrid (22 feat)", mean_mape_hybrid, "1 SVR, all features but 'my' position first"),
]
best = min(results, key=lambda x: x[1])

print(f"\n📊 Analysis:")
print(f"  • Current:      Each of 4 models sees 22 features (including irrelevant ones)")
print(f"  • Restructured: 1 model sees only 7 relevant features per position")
print(f"  • Hybrid:       1 model sees 22 features, but 'my' features always first")
print(f"")
print(f"  Training samples per model:")
print(f"    Current:      {X.shape[0]} samples × 4 models")
print(f"    Restructured: {X_pooled.shape[0]} samples × 1 model (4× more!)")
print(f"    Hybrid:       {X_hybrid.shape[0]} samples × 1 model (4× more!)")

print(f"\n🏆 BEST: {best[0]} with {best[1]:.2f}% MAPE")
print(f"   {best[2]}")

# Interpretation
if mean_mape_hybrid < mean_mape_pooled and mean_mape_hybrid < mean_mape_current:
    print(f"\n💡 Insight: Hybrid wins! Other positions' context DOES help prediction.")
    print(f"   The model benefits from knowing the full configuration.")
elif mean_mape_pooled < mean_mape_current:
    print(f"\n💡 Insight: Restructured wins! Pooling data + clean features is better.")
    print(f"   Other positions' local features were just noise.")
else:
    print(f"\n💡 Insight: Current approach works well.")
    print(f"   Position-specific models may capture unique patterns.")

print("\n" + "=" * 70)
