#!/usr/bin/env python
"""Assemble submodels saved by single-HPO-per-job runs into a composite model and evaluate.

Usage
-----
python assemble_composite.py \
    --nn_config 2 \
    --model_dir execution/models \
    --test_file data/test.txt \
    --test_c_file data/test_c.txt \
    --encoding physics

Environment variables read (same ones used during training):
    NCI_DISTANCE_CUTOFF   (0, 1, or 2)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ML_models.neural_net_train import NeuralNetReactorModel
from neural_net_configs.composite import NeuralNetCompositeThesis
from execution.data_handler import DataHandler


def _expected_paths(nn_config: int, model_dir: str) -> list[str]:
    """Return the ordered list of expected submodel file paths."""
    if nn_config in (2, 5):
        return [
            os.path.join(model_dir, f'nn_config{nn_config}_group{g}_submodel.pkl')
            for g in range(4)
        ]
    if nn_config == 3:
        paths = []
        for g in range(4):
            for pos in range(4):
                paths.append(
                    os.path.join(model_dir, f'nn_config3_group{g}_pos{pos}_submodel.pkl')
                )
        return paths
    raise ValueError(f"Assemble not needed for nn_config={nn_config} (single-model config)")


def _load_submodel(path: str) -> NeuralNetReactorModel:
    """Load a single NeuralNetReactorModel from *path*."""
    model, _meta = NeuralNetReactorModel.load_model(path)
    return model


def _build_composite(
    nn_config: int,
    models: list[NeuralNetReactorModel],
    encoding: str,
    irradiation_mode: str,
    nci_mode: str,
    nci_disabled: bool,
    optimize_lambda: bool,
) -> NeuralNetCompositeThesis:
    composite = NeuralNetCompositeThesis(
        nn_config=nn_config,
        models=models,
        encoding=encoding,
        irradiation_mode=irradiation_mode,
        nci_mode=nci_mode,
        nci_disabled=nci_disabled,
        optimize_lambda=optimize_lambda,
    )
    composite.set_flux_mode('energy_sixteen')
    return composite


def _evaluate(composite, X, y, lattices, label, use_log_flux):
    """Compute and print evaluation metrics for one test set."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    composite._lattices_eval = lattices
    preds = composite.predict_flux(X)

    n_outputs = y.shape[1]
    mse = float(np.mean([
        mean_squared_error(y[:, i], preds[:, i]) for i in range(n_outputs)
    ]))
    mae = float(np.mean([
        mean_absolute_error(y[:, i], preds[:, i]) for i in range(n_outputs)
    ]))
    r2 = float(np.mean([
        r2_score(y[:, i], preds[:, i]) for i in range(n_outputs)
    ])) if len(y) > 1 else 0.0

    if use_log_flux:
        y_orig = 10 ** y
        p_orig = 10 ** preds
        mape = float(np.mean(np.abs((y_orig - p_orig) / y_orig)) * 100)
        mse_log = float(mean_squared_error(y, preds))
    else:
        mape = float(np.mean(np.abs((y - preds) / (y + 1e-10))) * 100)
        mse_log = None

    metrics = {
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'n_samples': int(len(y)),
    }
    if mse_log is not None:
        metrics['mse_log'] = mse_log

    print(f"\n  --- {label} ({metrics['n_samples']} samples) ---")
    print(f"  MSE:  {mse:.10f}")
    if mse_log is not None:
        print(f"  MSE (log): {mse_log:.10f}")
    print(f"  RMSE: {metrics['rmse']:.10f}")
    print(f"  MAE:  {mae:.10f}")
    print(f"  R2:   {r2:.6f}")
    print(f"  MAPE: {mape:.8f}%")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Assemble thesis NN submodels into a composite and evaluate.')
    parser.add_argument('--nn_config', type=int, required=True,
                        choices=[2, 3, 5],
                        help='Thesis NN config (2, 3, or 5)')
    parser.add_argument('--model_dir', type=str, default='execution/models',
                        help='Directory containing submodel .pkl files')
    parser.add_argument('--test_file', type=str, default='data/test.txt',
                        help='Path to test data file')
    parser.add_argument('--test_c_file', type=str, default=None,
                        help='Path to test-set-C data file')
    parser.add_argument('--encoding', type=str, default='physics',
                        help='Encoding method used during training')
    parser.add_argument('--output', type=str, default=None,
                        help='Save path for the assembled composite .pkl')
    parser.add_argument('--no_lambda', action='store_true',
                        help='Disable per-submodel lambda regeneration during inference')
    args = parser.parse_args()

    nn_config = args.nn_config
    model_dir = args.model_dir
    encoding = args.encoding

    # ── check all submodel files exist ──────────────────────────────
    paths = _expected_paths(nn_config, model_dir)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print("ERROR: Missing submodel files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    print(f"Assembling nn_config={nn_config} from {len(paths)} submodel(s) in {model_dir}")

    # ── load submodels ──────────────────────────────────────────────
    models = [_load_submodel(p) for p in paths]
    print(f"Loaded {len(models)} submodel(s)")

    # Infer settings from the first submodel
    m0 = models[0]
    irradiation_mode = getattr(m0, 'irradiation_mode', 'fill')
    nci_mode = getattr(m0, 'nci_mode', 'separate')
    nci_disabled = getattr(m0, 'nci_disabled', False)
    optimize_lambda = not args.no_lambda and any(
        getattr(m, 'regen_lambdas_', None) for m in models
    )

    # ── build composite ─────────────────────────────────────────────
    composite = _build_composite(
        nn_config, models, encoding,
        irradiation_mode, nci_mode, nci_disabled, optimize_lambda,
    )
    print(f"Composite built: nn_config={nn_config}, optimize_lambda={optimize_lambda}")

    # ── load test data ──────────────────────────────────────────────
    dh = DataHandler()
    flux_mode = 'energy_sixteen'
    all_metrics = {}

    if args.test_file and os.path.exists(args.test_file):
        result = dh.load_and_prepare_data(args.test_file, encoding, flux_mode=flux_mode)
        if len(result) == 6:
            X_test, y_flux_test, _, _, lattices_test, _ = result
        elif len(result) == 5:
            X_test, y_flux_test, _, _, lattices_test = result
        else:
            X_test, y_flux_test = result[0], result[1]
            lattices_test = None

        all_metrics['test'] = _evaluate(
            composite, X_test, y_flux_test, lattices_test,
            'Test Set', dh.use_log_flux,
        )
    else:
        print(f"Skipping test evaluation (file not found: {args.test_file})")

    if args.test_c_file and os.path.exists(args.test_c_file):
        result_c = dh.load_and_prepare_data(args.test_c_file, encoding, flux_mode=flux_mode)
        if len(result_c) == 6:
            X_tc, y_flux_tc, _, _, lattices_tc, _ = result_c
        elif len(result_c) == 5:
            X_tc, y_flux_tc, _, _, lattices_tc = result_c
        else:
            X_tc, y_flux_tc = result_c[0], result_c[1]
            lattices_tc = None

        all_metrics['test_c'] = _evaluate(
            composite, X_tc, y_flux_tc, lattices_tc,
            'Test Set C', dh.use_log_flux,
        )
    else:
        if args.test_c_file:
            print(f"Skipping Set C evaluation (file not found: {args.test_c_file})")

    # ── save composite model ────────────────────────────────────────
    out_path = args.output or os.path.join(
        model_dir,
        f'neural_net_composite_flux_energy_sixteen_{encoding}_raytune_config{nn_config}.pkl',
    )
    composite.save_model(
        filepath=out_path,
        model_type='flux',
        encoding=encoding,
        optimization_method='raytune',
        flux_scale=1.0,
        use_log_flux=True,
        flux_mode='energy_sixteen',
        irradiation_mode=irradiation_mode,
        nci_mode=nci_mode,
    )

    # ── save metrics JSON ───────────────────────────────────────────
    metrics_path = os.path.splitext(out_path)[0] + '_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()
