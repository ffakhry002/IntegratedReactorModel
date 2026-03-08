"""
Smoke test: verify optimize_flux_model accepts use_restructured and runs
a single trial with restructured data without errors.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperparameter_tuning.optuna_optimization import optimize_flux_model


def test_restructured_single_trial():
    """Run 1 Optuna trial with restructured=True, xgboost, fill+separate."""
    np.random.seed(42)
    N = 80  # need >= 10 groups for GroupKFold(n_splits=10)
    X = np.random.rand(N, 37)
    y_flux = np.random.rand(N, 4)
    groups = np.repeat(np.arange(N // 8), 8)  # 10 groups × 8 augmentations

    best_params, study = optimize_flux_model(
        X_train=X,
        y_flux_train=y_flux,
        model_type='xgboost',
        n_trials=1,
        n_jobs=1,
        cores_per_trial=1,
        use_log_flux=True,
        groups=groups,
        flux_mode='total',
        encoding='physics',
        lattices_train=None,  # no lambda optimization for this test
        irradiation_mode='fill',
        nci_mode='separate',
        flux_loss='mse',
        use_restructured=True,
    )

    assert len(study.trials) == 1, f"Expected 1 trial, got {len(study.trials)}"
    assert study.best_value < float('inf'), "Trial should have produced a finite score"
    print(f"Best MSE: {study.best_value:.6f}")
    print(f"Best params: {best_params}")
    print("PASS: test_restructured_single_trial")


def test_non_restructured_still_works():
    """Verify default (non-restructured) path is unaffected."""
    np.random.seed(42)
    N = 80
    X = np.random.rand(N, 37)
    y_flux = np.random.rand(N, 4)
    groups = np.repeat(np.arange(N // 8), 8)

    best_params, study = optimize_flux_model(
        X_train=X,
        y_flux_train=y_flux,
        model_type='xgboost',
        n_trials=1,
        n_jobs=1,
        cores_per_trial=1,
        use_log_flux=True,
        groups=groups,
        flux_mode='total',
        encoding='physics',
        lattices_train=None,
        irradiation_mode='fill',
        nci_mode='separate',
        flux_loss='mape',
        use_restructured=False,
    )

    assert len(study.trials) == 1, f"Expected 1 trial, got {len(study.trials)}"
    assert study.best_value < float('inf'), "Trial should have produced a finite score"
    print(f"Best MAPE: {study.best_value:.2f}%")
    print("PASS: test_non_restructured_still_works")


if __name__ == '__main__':
    test_restructured_single_trial()
    print()
    test_non_restructured_still_works()
    print("\nAll Optuna restructured tests passed!")
