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


def _compute_subset_metrics(y, preds, use_log_flux):
    """Compute MSE, RMSE, MAE, R², MAPE for an (N, C) true/pred pair."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    if len(y) == 0:
        return {'mse': None, 'rmse': None, 'mae': None, 'r2': None,
                'mape': None, 'mse_log': None, 'n_samples': 0}

    n_out = y.shape[1] if y.ndim > 1 else 1
    if n_out > 1:
        mse = float(np.mean([mean_squared_error(y[:, i], preds[:, i])
                             for i in range(n_out)]))
        mae = float(np.mean([mean_absolute_error(y[:, i], preds[:, i])
                             for i in range(n_out)]))
        r2 = float(np.mean([r2_score(y[:, i], preds[:, i])
                            for i in range(n_out)])) if len(y) > 1 else 0.0
    else:
        mse = float(mean_squared_error(y, preds))
        mae = float(mean_absolute_error(y, preds))
        r2 = float(r2_score(y, preds)) if len(y) > 1 else 0.0

    if use_log_flux:
        y_orig = 10 ** y
        p_orig = 10 ** preds
        mape = float(np.mean(np.abs((y_orig - p_orig) / y_orig)) * 100)
        mse_log = float(mean_squared_error(y.ravel(), preds.ravel()))
    else:
        mape = float(np.mean(np.abs((y - preds) / (y + 1e-10))) * 100)
        mse_log = None

    out = {'mse': mse, 'rmse': float(np.sqrt(mse)), 'mae': mae,
           'r2': r2, 'mape': mape, 'n_samples': int(len(y))}
    if mse_log is not None:
        out['mse_log'] = mse_log
    return out


def _per_flux_type_breakdown(y, preds, use_log_flux, indent="  "):
    """Print and return per-flux-type (total/thermal/epi/fast) metrics.

    Column layout: col = pos*4 + group  (group 0=total 1=thermal 2=epi 3=fast).
    """
    if y.ndim < 2 or y.shape[1] != 16:
        return {}

    flux_names = ['Total', 'Thermal', 'Epithermal', 'Fast']
    result = {}
    print(f"{indent}Per flux type:")
    for g, name in enumerate(flux_names):
        cols = [p * 4 + g for p in range(4)]
        m = _compute_subset_metrics(y[:, cols], preds[:, cols], use_log_flux)
        result[name.lower()] = m
        print(f"{indent}  {name:>11s}: MSE={m['mse']:.10f}  "
              f"R\u00b2={m['r2']:.6f}  MAPE={m['mape']:.8f}%")
    return result


def _evaluate(composite, X, y, lattices, label, use_log_flux,
              fill_categories=None, test_split_run=None):
    """Compute and print evaluation metrics — overall, per-flux-type,
    Set A/B split, and per-fill-type breakdowns.
    """
    composite._lattices_eval = lattices
    preds = composite.predict_flux(X)

    overall = _compute_subset_metrics(y, preds, use_log_flux)

    print(f"\n  --- {label} ({overall['n_samples']} samples) ---")
    print(f"  MSE:  {overall['mse']:.10f}")
    if overall.get('mse_log') is not None:
        print(f"  MSE (log): {overall['mse_log']:.10f}")
    print(f"  RMSE: {overall['rmse']:.10f}")
    print(f"  MAE:  {overall['mae']:.10f}")
    print(f"  R\u00b2:   {overall['r2']:.6f}")
    print(f"  MAPE: {overall['mape']:.8f}%")

    ft_overall = _per_flux_type_breakdown(y, preds, use_log_flux)
    if ft_overall:
        overall['per_flux_type'] = ft_overall

    metrics = {'overall': overall}

    if test_split_run is not None:
        split_idx = test_split_run * 8
        if split_idx < len(y):
            y_a, p_a = y[:split_idx], preds[:split_idx]
            y_b, p_b = y[split_idx:], preds[split_idx:]
            fc_a = fill_categories[:split_idx] if fill_categories is not None else None
            fc_b = fill_categories[split_idx:] if fill_categories is not None else None

            m_a = _compute_subset_metrics(y_a, p_a, use_log_flux)
            m_b = _compute_subset_metrics(y_b, p_b, use_log_flux)

            print(f"\n  --- Test Set A (random lattice, {m_a['n_samples']} samples) ---")
            print(f"  MSE: {m_a['mse']:.10f}  R\u00b2: {m_a['r2']:.6f}  MAPE: {m_a['mape']:.8f}%")
            ft_a = _per_flux_type_breakdown(y_a, p_a, use_log_flux)
            if ft_a:
                m_a['per_flux_type'] = ft_a

            print(f"\n  --- Test Set B (self-adjusted, {m_b['n_samples']} samples) ---")
            print(f"  MSE: {m_b['mse']:.10f}  R\u00b2: {m_b['r2']:.6f}  MAPE: {m_b['mape']:.8f}%")
            ft_b = _per_flux_type_breakdown(y_b, p_b, use_log_flux)
            if ft_b:
                m_b['per_flux_type'] = ft_b

            fill_types = ['4G', '3G1P', '3G1B', '2G2P', '2G2B', '2G1P1B']
            for set_label, y_s, p_s, fc_s in [
                ('test_set_a', y_a, p_a, fc_a),
                ('test_set_b', y_b, p_b, fc_b),
            ]:
                set_m = _compute_subset_metrics(y_s, p_s, use_log_flux)
                set_dict = {'overall': set_m}
                if fc_s is not None:
                    set_name = "Set A (random)" if set_label == 'test_set_a' else "Set B (self-adj)"
                    print(f"\n  Per-fill-type breakdown for {set_name}:")
                    for ft in fill_types:
                        mask = fc_s == ft
                        if np.any(mask):
                            fm = _compute_subset_metrics(y_s[mask], p_s[mask], use_log_flux)
                            ft_fill = _per_flux_type_breakdown(
                                y_s[mask], p_s[mask], use_log_flux, indent="           ")
                            if ft_fill:
                                fm['per_flux_type'] = ft_fill
                            set_dict[ft] = fm
                            print(f"    {ft:>7s}: MSE={fm['mse']:.10f}  "
                                  f"R\u00b2={fm['r2']:.6f}  "
                                  f"MAPE={fm['mape']:.8f}%  "
                                  f"(n={fm['n_samples']})")
                metrics[set_label] = set_dict

    elif fill_categories is not None:
        fill_types = ['4G', '3G1P', '3G1B', '2G2P', '2G2B', '2G1P1B']
        print(f"\n  Per-fill-type breakdown:")
        for ft in fill_types:
            mask = fill_categories == ft
            if np.any(mask):
                fm = _compute_subset_metrics(y[mask], preds[mask], use_log_flux)
                ft_fill = _per_flux_type_breakdown(
                    y[mask], preds[mask], use_log_flux, indent="           ")
                if ft_fill:
                    fm['per_flux_type'] = ft_fill
                metrics[ft] = fm
                print(f"    {ft:>7s}: MSE={fm['mse']:.10f}  "
                      f"R\u00b2={fm['r2']:.6f}  "
                      f"MAPE={fm['mape']:.8f}%  "
                      f"(n={fm['n_samples']})")

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
    parser.add_argument('--test_split_run', type=int, default=None,
                        help='RUN boundary for Set A / Set B split (e.g. 528)')
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
        fill_categories = None
        if len(result) == 6:
            X_test, y_flux_test, _, _, lattices_test, fill_categories = result
        elif len(result) == 5:
            X_test, y_flux_test, _, _, lattices_test = result
        else:
            X_test, y_flux_test = result[0], result[1]
            lattices_test = None

        test_split_run = args.test_split_run

        all_metrics['test'] = _evaluate(
            composite, X_test, y_flux_test, lattices_test,
            'Test Set', dh.use_log_flux,
            fill_categories=fill_categories,
            test_split_run=test_split_run,
        )
    else:
        print(f"Skipping test evaluation (file not found: {args.test_file})")

    if args.test_c_file and os.path.exists(args.test_c_file):
        result_c = dh.load_and_prepare_data(args.test_c_file, encoding, flux_mode=flux_mode)
        fill_categories_c = None
        if len(result_c) == 6:
            X_tc, y_flux_tc, _, _, lattices_tc, fill_categories_c = result_c
        elif len(result_c) == 5:
            X_tc, y_flux_tc, _, _, lattices_tc = result_c
        else:
            X_tc, y_flux_tc = result_c[0], result_c[1]
            lattices_tc = None

        all_metrics['test_c'] = _evaluate(
            composite, X_tc, y_flux_tc, lattices_tc,
            'Test Set C', dh.use_log_flux,
            fill_categories=fill_categories_c,
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
