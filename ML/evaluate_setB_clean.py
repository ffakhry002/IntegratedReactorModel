#!/usr/bin/env python3
"""
Re-evaluate trained XGBoost models on a cleaned Test Set B.

Removes leaked geometries (Self-adjusted 4 and 13) from Set B,
retrains XGBoost with the best hyperparameters extracted from the
original training .out log, and reports metrics in the same format
as the training pipeline (so collect_results.sh can parse them).

Usage:
    python evaluate_setB_clean.py \
        --log-dir ML_logs/learning_curve_random/N013_keff \
        --train-file data/train_lc_set_13.txt \
        --target keff

    python evaluate_setB_clean.py \
        --log-dir ML_logs/learning_curve/N052_total \
        --train-file data/train.txt --max-runs 1716 \
        --target flux --flux-mode total
"""

import os
import sys
import re
import ast
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('NCI_DISTANCE_CUTOFF', '0')

from execution.data_handler import DataHandler
from ML_models.xgboost_train import XGBoostReactorModel
from utils.lambda_feature_regenerator import LambdaFeatureRegenerator
from utils.txt_to_data import parse_reactor_data
from ML_models.encodings.encoding_methods import IRRADIATION_MODE, NCI_MODE, NCI_DISTANCE_CUTOFF
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

LEAKED_GEOMETRY_NAMES = {'Self adjusted 4', 'Self-adjusted 13'}
TEST_SPLIT_RUN = 4224
AUGMENT_FACTOR = 8


def geometry_name_from_description(desc):
    """Extract geometry name from description (part before '---')."""
    if '---' in desc:
        return desc.split('---')[0].strip()
    if ' -- ' in desc:
        return desc.split(' -- ')[0].strip()
    return desc.strip()


def find_out_file(log_dir):
    """Find the .out file in a log directory."""
    candidates = [f for f in os.listdir(log_dir) if f.endswith('.out')]
    if not candidates:
        raise FileNotFoundError(f"No .out file found in {log_dir}")
    if len(candidates) > 1:
        candidates.sort(key=lambda f: os.path.getmtime(os.path.join(log_dir, f)),
                        reverse=True)
    return os.path.join(log_dir, candidates[0])


def parse_best_params(out_file):
    """Extract best hyperparameters dict from .out log file."""
    with open(out_file) as f:
        content = f.read()
    match = re.search(r"Best parameters found: (\{[^}]+\})", content)
    if not match:
        raise ValueError(f"Could not find 'Best parameters found' in {out_file}")
    return ast.literal_eval(match.group(1))


def compute_metrics(y_true, y_pred, target, use_log_flux=False):
    """Compute MSE, RMSE, MAE, R², MAPE matching model_trainer conventions.

    For flux (2D arrays): MSE/R² are per-column averages, MAPE is on raw
    (back-transformed) values when use_log_flux=True.
    """
    if len(y_true) == 0:
        return {'mse': None, 'rmse': None, 'mae': None,
                'r2': None, 'mape': None,
                'mse_log': None, 'r2_log': None,
                'mse_raw': None, 'r2_raw': None,
                'n_samples': 0}

    if target == 'keff' or (len(y_true.shape) == 1):
        y_t = y_true.ravel()
        y_p = y_pred.ravel()
        mse = float(mean_squared_error(y_t, y_p))
        mae = float(mean_absolute_error(y_t, y_p))
        r2 = float(r2_score(y_t, y_p))
        mape = float(np.mean(np.abs((y_t - y_p) / (y_t + 1e-10))) * 100)
        mse_log = mse
        r2_log = r2
        mse_raw = mse
        r2_raw = r2
        n_samples = len(y_t)
    else:
        n_outputs = y_true.shape[1]
        mse = float(np.mean([mean_squared_error(y_true[:, i], y_pred[:, i])
                             for i in range(n_outputs)]))
        mae = float(np.mean([mean_absolute_error(y_true[:, i], y_pred[:, i])
                             for i in range(n_outputs)]))
        r2 = float(np.mean([r2_score(y_true[:, i], y_pred[:, i])
                            for i in range(n_outputs)]))
        if use_log_flux:
            mse_log = mse
            r2_log = r2
            y_orig = 10 ** y_true
            p_orig = 10 ** y_pred
            mse_raw = float(np.mean([mean_squared_error(y_orig[:, i], p_orig[:, i])
                                     for i in range(n_outputs)]))
            r2_raw = float(np.mean([r2_score(y_orig[:, i], p_orig[:, i])
                                    for i in range(n_outputs)]))
            mape = float(np.mean(np.abs((y_orig - p_orig) / y_orig)) * 100)
        else:
            mse_log = mse
            r2_log = r2
            mse_raw = mse
            r2_raw = r2
            mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100)
        n_samples = len(y_true)

    rmse = float(np.sqrt(mse))

    return {'mse': mse, 'rmse': rmse, 'mae': mae,
            'r2': r2, 'mape': mape,
            'mse_log': mse_log, 'r2_log': r2_log,
            'mse_raw': mse_raw, 'r2_raw': r2_raw,
            'n_samples': n_samples}


def build_clean_setB_mask(descriptions, n_augmented):
    """Build a boolean mask over augmented samples: True = clean Set B sample.

    Returns (clean_setB_mask, leaked_run_count, clean_run_count).
    """
    split_idx = TEST_SPLIT_RUN * AUGMENT_FACTOR
    mask = np.zeros(n_augmented, dtype=bool)

    leaked_runs = 0
    clean_runs = 0
    for run_idx in range(TEST_SPLIT_RUN, len(descriptions)):
        geo = geometry_name_from_description(descriptions[run_idx])
        start = run_idx * AUGMENT_FACTOR
        end = start + AUGMENT_FACTOR
        if end > n_augmented:
            break
        if geo in LEAKED_GEOMETRY_NAMES:
            leaked_runs += 1
        else:
            mask[start:end] = True
            clean_runs += 1

    return mask, leaked_runs, clean_runs


def evaluate_subset(model, X, y, target, fc, label, use_log_flux=False):
    """Evaluate model on a subset and print formatted metrics."""
    if target == 'flux':
        preds = model.predict_flux(X)
    else:
        preds = model.predict_keff(X)

    overall = compute_metrics(y, preds, target, use_log_flux=use_log_flux)

    print(f"\n  --- {label} ({overall['n_samples']} samples) ---")
    print(f"  MSE: {overall['mse']:.10f}  R²: {overall['r2']:.6f}  MAPE: {overall['mape']:.8f}%")
    if target == 'flux' and use_log_flux:
        print(f"  [dual-space] log_mse={overall['mse_log']:.10f}  log_r2={overall['r2_log']:.6f}  "
              f"raw_mse={overall['mse_raw']:.10f}  raw_r2={overall['r2_raw']:.6f}")

    fill_types = ['4G', '3G1P', '3G1B', '2G2P', '2G2B', '2G1P1B']
    ft_metrics = {}
    if fc is not None:
        print(f"\n  Per-fill-type breakdown for {label}:")
        for ft in fill_types:
            ft_mask = (fc == ft)
            if np.any(ft_mask):
                m = compute_metrics(y[ft_mask], preds[ft_mask], target, use_log_flux=use_log_flux)
                ft_metrics[ft] = m
                msg = (f"    {ft:>7s}: MSE={m['mse']:.10f}  "
                       f"R²={m['r2']:.6f}  "
                       f"MAPE={m['mape']:.8f}%  "
                       f"(n={m['n_samples']})")
                if target == 'flux' and use_log_flux:
                    msg += (f"  [raw: raw_mse={m['mse_raw']:.10f} "
                            f"raw_r2={m['r2_raw']:.6f}]")
                print(msg)
            else:
                ft_metrics[ft] = {'mse': None, 'r2': None, 'mape': None, 'n_samples': 0}

    return overall, ft_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Re-evaluate XGBoost on cleaned Test Set B (leaked geometries removed).')
    parser.add_argument('--log-dir', required=True,
                        help='Directory containing the original .out log file')
    parser.add_argument('--train-file', required=True,
                        help='Training data file (e.g. data/train_lc_set_13.txt or data/train.txt)')
    parser.add_argument('--max-runs', type=int, default=None,
                        help='Max training runs to load (for selected experiment with train.txt)')
    parser.add_argument('--target', required=True, choices=['keff', 'flux'])
    parser.add_argument('--flux-mode', default='total',
                        choices=['total', 'thermal_only', 'epithermal_only', 'fast_only'])
    parser.add_argument('--test-file', default='data/test.txt')
    parser.add_argument('--test-c-file', default=None,
                        help='Test Set C file (default: data/test_c.txt if it exists)')
    parser.add_argument('--use-restructured', action='store_true', default=True,
                        help='Use restructured position pooling for flux (default: True)')
    args = parser.parse_args()

    if args.test_c_file is None:
        args.test_c_file = os.environ.get('TEST_C_FILE', 'data/test_c.txt')

    print("=" * 70)
    print("  Set B Clean Evaluation")
    print("=" * 70)
    print(f"  Log dir:      {args.log_dir}")
    print(f"  Train file:   {args.train_file}")
    if args.max_runs:
        print(f"  Max runs:     {args.max_runs}")
    print(f"  Target:       {args.target}")
    if args.target == 'flux':
        print(f"  Flux mode:    {args.flux_mode}")
    print(f"  Test file:    {args.test_file}")
    print(f"  Leaked geos:  {LEAKED_GEOMETRY_NAMES}")
    print(f"  NCI cutoff:   {NCI_DISTANCE_CUTOFF}")
    print("=" * 70)

    # ── 1. Parse best params from .out file ──────────────────────────
    out_file = find_out_file(args.log_dir)
    print(f"\nParsing params from: {os.path.basename(out_file)}")
    all_params = parse_best_params(out_file)

    lambda_names = ['lambda_P', 'lambda_B', 'lambda_G', 'lambda_decay']
    lambdas = {k: all_params[k] for k in lambda_names if k in all_params}
    model_params = {k: v for k, v in all_params.items() if k not in lambda_names}

    print(f"  Model params: {list(model_params.keys())}")
    if lambdas:
        print(f"  Lambda params: { {k: f'{v:.4f}' for k, v in lambdas.items()} }")

    # ── 2. Load training data ────────────────────────────────────────
    print(f"\nLoading training data from {args.train_file}...")
    train_handler = DataHandler()
    flux_mode = args.flux_mode if args.target == 'flux' else 'total'
    train_result = train_handler.load_and_prepare_data(
        args.train_file, 'physics', flux_mode=flux_mode,
        max_runs=args.max_runs)

    X_train, y_flux_train, y_keff_train, groups_train, lattices_train, fc_train = train_result
    y_train = y_flux_train if args.target == 'flux' else y_keff_train
    print(f"  Training samples: {X_train.shape[0]}  features: {X_train.shape[1]}")

    # ── 3. Load test data ────────────────────────────────────────────
    print(f"\nLoading test data from {args.test_file}...")
    test_handler = DataHandler()
    test_result = test_handler.load_and_prepare_data(
        args.test_file, 'physics', flux_mode=flux_mode)

    X_test, y_flux_test, y_keff_test, _, lattices_test, fc_test = test_result
    y_test = y_flux_test if args.target == 'flux' else y_keff_test
    print(f"  Test samples: {X_test.shape[0]}  features: {X_test.shape[1]}")

    # ── 4. Get descriptions for Set B filtering ──────────────────────
    _, _, _, descriptions, _ = parse_reactor_data(args.test_file)

    # ── 5. Apply lambda values ───────────────────────────────────────
    irradiation_mode = IRRADIATION_MODE
    nci_mode = NCI_MODE

    if lambdas:
        print(f"\nApplying optimized lambda values to features:")
        for k, v in lambdas.items():
            print(f"    {k}: {v:.3f}")

        regenerator = LambdaFeatureRegenerator('physics')

        fd_train = regenerator.separate_features(
            X_train, lattices_train, irradiation_mode, nci_mode)
        X_train = regenerator.regenerate_features(fd_train, **lambdas)
        print(f"  Regenerated training features with optimized lambdas")

        if lattices_test is not None:
            fd_test = regenerator.separate_features(
                X_test, lattices_test, irradiation_mode, nci_mode)
            X_test = regenerator.regenerate_features(fd_test, **lambdas)
            print(f"  Regenerated test features with optimized lambdas")

    # ── 6. Restructure training data if flux + restructured ──────────
    use_restructured = (args.use_restructured and args.target == 'flux')
    nci_disabled = (NCI_DISTANCE_CUTOFF == 2)

    if use_restructured:
        from utils.data_restructure import restructure_for_position_pooling
        X_train_final, y_train_final, _ = restructure_for_position_pooling(
            X_train, y_train, groups_train, irradiation_mode, nci_mode,
            nci_disabled=nci_disabled)
        print(f"  Restructured training data: {X_train.shape} -> {X_train_final.shape}")
    else:
        X_train_final, y_train_final = X_train, y_train

    # ── 7. Train XGBoost with extracted params ───────────────────────
    print(f"\nTraining XGBoost with extracted params...")
    train_start = time.time()

    model = XGBoostReactorModel(
        use_restructured=use_restructured,
        irradiation_mode=irradiation_mode,
        nci_mode=nci_mode,
        nci_disabled=nci_disabled,
        **model_params)

    if args.target == 'flux':
        model.fit_flux(X_train_final, y_train_final)
    else:
        model.fit_keff(X_train_final, y_train_final)

    train_time = time.time() - train_start
    print(f"  Training complete! ({train_time:.1f} seconds)")

    # ── 8. Build masks ───────────────────────────────────────────────
    split_idx = TEST_SPLIT_RUN * AUGMENT_FACTOR
    clean_mask, n_leaked, n_clean = build_clean_setB_mask(descriptions, len(X_test))

    print(f"\n  Set B: {n_leaked + n_clean} runs total "
          f"({(n_leaked + n_clean) // 33} geometries × 33 fills)")
    print(f"  Leaked (removed): {n_leaked} runs ({n_leaked // 33} geometries) "
          f"→ {n_leaked * AUGMENT_FACTOR} augmented rows")
    print(f"  Clean (kept):     {n_clean} runs ({n_clean // 33} geometries) "
          f"→ {n_clean * AUGMENT_FACTOR} augmented rows")

    # ── 9. Evaluate ──────────────────────────────────────────────────
    # Match model_trainer: for flux, use_log_flux=True (back-transform for MAPE,
    # per-column averaging for R²/MSE)
    use_log_flux = (args.target == 'flux')

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'=' * 60}")

    # Overall test
    if args.target == 'flux':
        preds_all = model.predict_flux(X_test)
    else:
        preds_all = model.predict_keff(X_test)

    overall = compute_metrics(y_test, preds_all, args.target, use_log_flux=use_log_flux)
    print(f"\n  Test MSE: {overall['mse']:.10f}")
    print(f"  Test RMSE: {overall['rmse']:.10f}")
    print(f"  Test MAE: {overall['mae']:.10f}")
    print(f"  Test R²: {overall['r2']:.6f}")
    print(f"  Test MAPE: {overall['mape']:.8f}%")
    if args.target == 'flux' and use_log_flux:
        print(f"  [dual-space] log_mse={overall['mse_log']:.10f}  log_r2={overall['r2_log']:.6f}  "
              f"raw_mse={overall['mse_raw']:.10f}  raw_r2={overall['r2_raw']:.6f}")

    # Set A
    setA_X = X_test[:split_idx]
    setA_y = y_test[:split_idx]
    setA_fc = fc_test[:split_idx] if fc_test is not None else None

    setA_overall, setA_ft = evaluate_subset(
        model, setA_X, setA_y, args.target, setA_fc,
        "Test Set A (random lattice)", use_log_flux=use_log_flux)

    # Set B (CLEAN)
    setB_X = X_test[clean_mask]
    setB_y = y_test[clean_mask]
    setB_fc = fc_test[clean_mask] if fc_test is not None else None

    setB_overall, setB_ft = evaluate_subset(
        model, setB_X, setB_y, args.target, setB_fc,
        "Test Set B (self-adjusted, CLEANED)", use_log_flux=use_log_flux)

    # Test Set C
    if args.test_c_file and os.path.exists(args.test_c_file):
        print(f"\nLoading Test Set C from {args.test_c_file}...")
        handler_c = DataHandler()
        result_c = handler_c.load_and_prepare_data(
            args.test_c_file, 'physics', flux_mode=flux_mode)

        if len(result_c) == 6:
            X_c, yf_c, yk_c, _, lat_c, fc_c = result_c
        elif len(result_c) == 5:
            X_c, yf_c, yk_c, _, lat_c = result_c
            fc_c = None
        else:
            X_c, yf_c, yk_c = result_c[:3]
            lat_c, fc_c = None, None

        y_c = yf_c if args.target == 'flux' else yk_c

        if lambdas and lat_c is not None:
            fd_c = regenerator.separate_features(X_c, lat_c, irradiation_mode, nci_mode)
            X_c = regenerator.regenerate_features(fd_c, **lambdas)

        setC_overall, setC_ft = evaluate_subset(
            model, X_c, y_c, args.target, fc_c,
            "Test Set C (random cores)", use_log_flux=use_log_flux)
    else:
        print(f"\n  Test Set C file not found, skipping.")

    print(f"\n{'=' * 60}")
    print(f"  Evaluation complete!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
