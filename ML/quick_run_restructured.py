"""
Quick end-to-end run: restructured XGBoost with real data.

Fill mode + separate NCI + cutoff 1 + lambda optimization + 37 features
→ restructured 32N position pooling → Optuna (15 trials) → final model → test MAPE

Run from ML directory:
    python quick_run_restructured.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set fill + separate + cutoff 1
os.environ['NCI_DISTANCE_CUTOFF'] = '1'
from ML_models.encodings import encoding_methods
encoding_methods.IRRADIATION_MODE = 'fill'
encoding_methods.NCI_MODE = 'separate'

from execution.data_handler import DataHandler
from execution.model_trainer import ModelTrainer
from execution.config import TrainingConfig

TRAIN_FILE = 'data/train.txt'
TEST_FILE = 'data/test.txt'
N_TRIALS = 15

print("=" * 70)
print("QUICK RUN: Restructured XGBoost (fill + separate NCI + lambda opt)")
print("=" * 70)

start = time.time()

# --- Load and encode training data ---
print("\n1. Loading training data...")
handler = DataHandler()
X, y_flux, y_keff, groups, lattices = handler.load_and_prepare_data(
    TRAIN_FILE, encoding_method='physics', flux_mode='total')

print(f"\n   X: {X.shape} (should be 37 features)")
print(f"   y_flux: {y_flux.shape}")
print(f"   groups: {len(set(groups.tolist()))} unique spatial patterns")

# --- Split (use all for training, CV handles validation) ---
data_splits = handler.split_data(X, y_flux, y_keff, groups, test_size=0.0, lattices=lattices)

# --- Also load test data for final evaluation ---
print("\n2. Loading test data for evaluation...")
handler_test = DataHandler()
X_test_all, y_flux_test_all, y_keff_test_all, _, lattices_test_all = handler_test.load_and_prepare_data(
    TEST_FILE, encoding_method='physics', flux_mode='total')
print(f"   Test configs: {X_test_all.shape[0]}")

# Put a subset of test data in splits for evaluation
data_splits['X_test'] = X_test_all[:200]  # first 200 for speed
data_splits['y_flux_test'] = y_flux_test_all[:200]
data_splits['y_keff_test'] = y_keff_test_all[:200]
data_splits['lattices_test'] = lattices_test_all[:200]

# --- Configure ---
config = TrainingConfig()
config.optimization = 'optuna'
config.n_trials = N_TRIALS
config.n_jobs = 1
config.n_jobs_xgb = 1
config.cores_per_xgb_trial = 1
config.use_restructured = True
config.flux_mode = 'total'
config.test_size = 0.0

# --- Train ---
print(f"\n3. Training with {N_TRIALS} Optuna trials (restructured + lambda opt)...")
trainer = ModelTrainer(data_handler=handler)

model, metrics, best_params = trainer.train_model(
    model_type='xgboost',
    target='flux',
    data_splits=data_splits,
    config=config,
    encoding='physics'
)

# --- Report ---
elapsed = time.time() - start
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"  Training configs: {len(set(groups.tolist()))} spatial patterns, {X.shape[0]} augmented samples")
print(f"  Restructured to: {X.shape[0] * 4} position-pooled samples")
print(f"  Features: {X.shape[1]} (fill + separate NCI)")
print(f"  Optuna trials: {N_TRIALS}")
print(f"  Test MAPE: {metrics.get('mape', 'N/A')}")
print(f"  Test R²: {metrics.get('r2', 'N/A')}")
print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"  Best params: {best_params}")

# Lambda values if present
for k in ['lambda_P', 'lambda_B', 'lambda_G']:
    if k in best_params:
        print(f"  {k}: {best_params[k]:.3f}")
