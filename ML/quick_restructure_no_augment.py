"""
Quick comparison WITHOUT augmentation:
Tests whether augmentation actually helps in the hybrid approach.

Compares:
1. Current (4 separate models) - NO augmentation
2. Restructured (local only) - NO augmentation
3. Hybrid (full context) - NO augmentation

Run from ML directory:
    python quick_restructure_no_augment.py
"""

import numpy as np
import sys
import os
import time
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold  # Regular KFold since no augmentation groups
from sklearn.metrics import mean_squared_error, r2_score

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set to vacuum mode
os.environ['NCI_DISTANCE_CUTOFF'] = '1'

print("=" * 70)
print("COMPARISON WITHOUT AUGMENTATION")
print("Testing: Does augmentation actually help with hybrid restructuring?")
print("=" * 70)

# =============================================================================
# LOAD DATA (with augmentation, then filter to remove it)
# =============================================================================
print("\n1. Loading data and removing augmentation...")

from execution.data_handler import DataHandler
from ML_models.encodings import encoding_methods

# Force vacuum mode
encoding_methods.IRRADIATION_MODE = 'vacuum'
encoding_methods.NCI_MODE = 'single'

handler = DataHandler()
X_aug, y_flux_aug, y_keff_aug, groups_aug, lattices_aug = handler.load_and_prepare_data(
    '/Users/dima/Downloads/Thesis data/Machine Learning/train_test_data/train.txt',
    encoding_method='physics',
    flux_mode='total'
)[:5]

print(f"\n  With augmentation:")
print(f"    X shape: {X_aug.shape}")
print(f"    Unique configs (groups): {len(np.unique(groups_aug))}")

# Remove augmentation: keep only FIRST sample from each group
# Each group has 8 augmented copies, we keep index 0 of each
unique_groups = np.unique(groups_aug)
first_indices = []
for g in unique_groups:
    group_indices = np.where(groups_aug == g)[0]
    first_indices.append(group_indices[0])  # Keep only first of each group

first_indices = np.array(first_indices)
X = X_aug[first_indices]
y_flux = y_flux_aug[first_indices]
y_keff = y_keff_aug[first_indices]

print(f"\n  WITHOUT augmentation (first of each group):")
print(f"    X shape: {X.shape}")
print(f"    Samples: {X.shape[0]} (was {X_aug.shape[0]})")

# =============================================================================
# RESTRUCTURE FUNCTIONS (same as before)
# =============================================================================

def restructure_for_position_pooling(X, y_flux):
    """Restructure so each row is one position (7 features)."""
    n_configs = X.shape[0]
    n_positions = 4
    n_global = 2
    n_local_per_pos = 4
    n_nci_per_pos = 1

    X_new, y_new = [], []

    for idx in range(n_configs):
        global_feats = X[idx, :n_global]

        for pos in range(n_positions):
            local_start = n_global + pos * n_local_per_pos
            local_feats = X[idx, local_start:local_start + n_local_per_pos]

            nci_start = n_global + n_positions * n_local_per_pos + pos * n_nci_per_pos
            nci_feats = X[idx, nci_start:nci_start + n_nci_per_pos]

            pos_features = np.concatenate([global_feats, local_feats, nci_feats])
            X_new.append(pos_features)
            y_new.append(y_flux[idx, pos])

    return np.array(X_new), np.array(y_new)


def restructure_hybrid(X, y_flux):
    """Restructure with ALL features, MY position first (22 features)."""
    n_configs = X.shape[0]
    n_positions = 4
    n_global = 2
    n_local_per_pos = 4
    n_nci_per_pos = 1

    X_new, y_new = [], []

    for idx in range(n_configs):
        global_feats = X[idx, :n_global]

        all_local = []
        for pos in range(n_positions):
            start = n_global + pos * n_local_per_pos
            all_local.append(X[idx, start:start + n_local_per_pos])

        nci_start = n_global + n_positions * n_local_per_pos
        all_nci = X[idx, nci_start:nci_start + n_positions]

        for my_pos in range(n_positions):
            my_local = all_local[my_pos]
            my_nci = all_nci[my_pos:my_pos+1]

            other_local = np.concatenate([all_local[p] for p in range(n_positions) if p != my_pos])
            other_nci = np.array([all_nci[p] for p in range(n_positions) if p != my_pos])

            pos_features = np.concatenate([global_feats, my_local, my_nci, other_local, other_nci])
            X_new.append(pos_features)
            y_new.append(y_flux[idx, my_pos])

    return np.array(X_new), np.array(y_new)


# =============================================================================
# PREPARE DATA
# =============================================================================
print("\n2. Preparing restructured data...")

X_pooled, y_pooled = restructure_for_position_pooling(X, y_flux)
X_hybrid, y_hybrid = restructure_hybrid(X, y_flux)

print(f"   Current:      {X.shape[0]} samples, {X.shape[1]} features, 4 outputs")
print(f"   Restructured: {X_pooled.shape[0]} samples, {X_pooled.shape[1]} features, 1 output")
print(f"   Hybrid:       {X_hybrid.shape[0]} samples, {X_hybrid.shape[1]} features, 1 output")

# =============================================================================
# SVR PARAMETERS
# =============================================================================
svr_params = {
    'kernel': 'rbf',
    'C': 100.0,
    'gamma': 0.01,
    'epsilon': 0.01,
    'cache_size': 2000,
    'max_iter': 20000,
    'verbose': False,
}

n_folds = 5
print(f"\nUsing {n_folds}-fold CV (regular KFold, no groups needed)")

# =============================================================================
# METHOD 1: CURRENT (4 separate models)
# =============================================================================
print(f"\n3. Testing CURRENT approach (4 separate SVR models)...")

model_current = MultiOutputRegressor(
    Pipeline([('scaler', StandardScaler()), ('svr', SVR(**svr_params))]),
    n_jobs=4
)

cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
start_time = time.time()

scores_current = {'mape': [], 'rmse': [], 'r2': []}
for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_flux[train_idx], y_flux[val_idx]

    model_current.fit(X_train, y_train)
    y_pred = model_current.predict(X_val)

    y_val_orig = 10 ** y_val
    y_pred_orig = 10 ** y_pred

    mape = np.mean(np.abs((y_val_orig - y_pred_orig) / (y_val_orig + 1e-10))) * 100
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val.flatten(), y_pred.flatten())

    scores_current['mape'].append(mape)
    scores_current['rmse'].append(rmse)
    scores_current['r2'].append(r2)
    print(f"   Fold {fold+1}: MAPE={mape:.2f}%, RMSE={rmse:.4f}, R²={r2:.4f}")

time_current = time.time() - start_time
mean_current = {k: np.mean(v) for k, v in scores_current.items()}
std_current = np.std(scores_current['mape'])

# =============================================================================
# METHOD 2: RESTRUCTURED (7 features)
# =============================================================================
print(f"\n4. Testing RESTRUCTURED approach (1 SVR, 7 features)...")

model_pooled = Pipeline([('scaler', StandardScaler()), ('svr', SVR(**svr_params))])

cv_pooled = KFold(n_splits=n_folds, shuffle=True, random_state=42)
start_time = time.time()

scores_pooled = {'mape': [], 'rmse': [], 'r2': []}
for fold, (train_idx, val_idx) in enumerate(cv_pooled.split(X_pooled)):
    X_train, X_val = X_pooled[train_idx], X_pooled[val_idx]
    y_train, y_val = y_pooled[train_idx], y_pooled[val_idx]

    model_pooled.fit(X_train, y_train)
    y_pred = model_pooled.predict(X_val)

    y_val_orig = 10 ** y_val
    y_pred_orig = 10 ** y_pred

    mape = np.mean(np.abs((y_val_orig - y_pred_orig) / (y_val_orig + 1e-10))) * 100
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    scores_pooled['mape'].append(mape)
    scores_pooled['rmse'].append(rmse)
    scores_pooled['r2'].append(r2)
    print(f"   Fold {fold+1}: MAPE={mape:.2f}%, RMSE={rmse:.4f}, R²={r2:.4f}")

time_pooled = time.time() - start_time
mean_pooled = {k: np.mean(v) for k, v in scores_pooled.items()}
std_pooled = np.std(scores_pooled['mape'])

# =============================================================================
# METHOD 3: HYBRID (22 features, MY first)
# =============================================================================
print(f"\n5. Testing HYBRID approach (1 SVR, 22 features, MY first)...")

model_hybrid = Pipeline([('scaler', StandardScaler()), ('svr', SVR(**svr_params))])

cv_hybrid = KFold(n_splits=n_folds, shuffle=True, random_state=42)
start_time = time.time()

scores_hybrid = {'mape': [], 'rmse': [], 'r2': []}
for fold, (train_idx, val_idx) in enumerate(cv_hybrid.split(X_hybrid)):
    X_train, X_val = X_hybrid[train_idx], X_hybrid[val_idx]
    y_train, y_val = y_hybrid[train_idx], y_hybrid[val_idx]

    model_hybrid.fit(X_train, y_train)
    y_pred = model_hybrid.predict(X_val)

    y_val_orig = 10 ** y_val
    y_pred_orig = 10 ** y_pred

    mape = np.mean(np.abs((y_val_orig - y_pred_orig) / (y_val_orig + 1e-10))) * 100
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    scores_hybrid['mape'].append(mape)
    scores_hybrid['rmse'].append(rmse)
    scores_hybrid['r2'].append(r2)
    print(f"   Fold {fold+1}: MAPE={mape:.2f}%, RMSE={rmse:.4f}, R²={r2:.4f}")

time_hybrid = time.time() - start_time
mean_hybrid = {k: np.mean(v) for k, v in scores_hybrid.items()}
std_hybrid = np.std(scores_hybrid['mape'])

# =============================================================================
# COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON SUMMARY (NO AUGMENTATION)")
print("=" * 70)

print(f"\n{'Approach':<38} {'Feat':<6} {'Samples':<8} {'MAPE':<18} {'RMSE':<10} {'R²':<8}")
print("-" * 100)
print(f"{'1. Current (4 models)':<38} {X.shape[1]:<6} {X.shape[0]:<8} {mean_current['mape']:.2f}% ± {std_current:.2f}%  {mean_current['rmse']:.4f}    {mean_current['r2']:.4f}")
print(f"{'2. Restructured (7 feat)':<38} {X_pooled.shape[1]:<6} {X_pooled.shape[0]:<8} {mean_pooled['mape']:.2f}% ± {std_pooled:.2f}%  {mean_pooled['rmse']:.4f}    {mean_pooled['r2']:.4f}")
print(f"{'3. Hybrid (22 feat, MY first)':<38} {X_hybrid.shape[1]:<6} {X_hybrid.shape[0]:<8} {mean_hybrid['mape']:.2f}% ± {std_hybrid:.2f}%  {mean_hybrid['rmse']:.4f}    {mean_hybrid['r2']:.4f}")
print("-" * 100)

print(f"\n📊 Key comparison:")
print(f"   Base configs: {X.shape[0]}")
print(f"   With 4× position pooling: {X_hybrid.shape[0]} samples")
print(f"   (This is the TRUE 4× gain - no augmentation redundancy)")

# Find best
results = [
    ("Current", mean_current['mape']),
    ("Restructured", mean_pooled['mape']),
    ("Hybrid", mean_hybrid['mape']),
]
best = min(results, key=lambda x: x[1])
print(f"\n🏆 BEST: {best[0]} with {best[1]:.2f}% MAPE")

print("\n" + "=" * 70)
print("COMPARE WITH AUGMENTED RESULTS:")
print("  Run quick_restructure_comparison.py to see WITH augmentation")
print("  Then compare: does augmentation help or just add redundancy?")
print("=" * 70)
