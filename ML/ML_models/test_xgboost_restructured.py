"""Unit tests for XGBoostReactorModel with use_restructured=True."""

import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML_models.xgboost_train import XGBoostReactorModel
from utils.data_restructure import restructure_for_position_pooling


def test_fit_restructured():
    """fit_flux with restructured 1D targets should work."""
    N = 50
    X = np.random.rand(N, 37)
    y_flux = np.random.rand(N, 4)
    groups = np.arange(N)

    X_pool, y_pool, _ = restructure_for_position_pooling(X, y_flux, groups, 'fill', 'separate')

    model = XGBoostReactorModel(
        use_restructured=True, irradiation_mode='fill', nci_mode='separate',
        n_estimators=10, max_depth=3)
    model.fit_flux(X_pool, y_pool)

    assert model.flux_model is not None
    assert not hasattr(model.flux_model, 'estimators_'), \
        "Should be single XGBRegressor, not MultiOutputRegressor"
    print("PASS: test_fit_restructured")


def test_predict_restructured():
    """predict_flux should accept original-layout input and return (n, 4)."""
    N = 50
    X = np.random.rand(N, 37)
    y_flux = np.random.rand(N, 4)
    groups = np.arange(N)

    X_pool, y_pool, _ = restructure_for_position_pooling(X, y_flux, groups, 'fill', 'separate')

    model = XGBoostReactorModel(
        use_restructured=True, irradiation_mode='fill', nci_mode='separate',
        n_estimators=10, max_depth=3)
    model.set_flux_mode('total')
    model.fit_flux(X_pool, y_pool)

    # Predict with ORIGINAL layout (not restructured) -- model restructures internally
    X_test = np.random.rand(10, 37)
    predictions = model.predict_flux(X_test)

    assert predictions.shape == (10, 4), f"Expected (10, 4), got {predictions.shape}"
    print("PASS: test_predict_restructured")


def test_non_restructured_unchanged():
    """Default (non-restructured) path should work exactly as before."""
    N = 50
    X = np.random.rand(N, 37)
    y_flux = np.random.rand(N, 4)

    model = XGBoostReactorModel(n_estimators=10, max_depth=3)
    model.set_flux_mode('total')
    model.fit_flux(X, y_flux)

    X_test = np.random.rand(10, 37)
    predictions = model.predict_flux(X_test)

    assert predictions.shape == (10, 4), f"Expected (10, 4), got {predictions.shape}"
    assert hasattr(model.flux_model, 'estimators_'), \
        "Should be MultiOutputRegressor"
    print("PASS: test_non_restructured_unchanged")


def test_metadata_save_load():
    """Metadata round-trip should preserve restructured settings."""
    model = XGBoostReactorModel(
        use_restructured=True, irradiation_mode='fill', nci_mode='separate',
        n_estimators=10, max_depth=3)

    meta = model._get_model_specific_metadata()
    assert meta['use_restructured'] is True
    assert meta['restructured_irradiation_mode'] == 'fill'
    assert meta['restructured_nci_mode'] == 'separate'

    # Simulate loading into a fresh model
    fresh = XGBoostReactorModel(n_estimators=10)
    fresh._restore_model_specific_attributes(meta)
    assert fresh.use_restructured is True
    assert fresh.irradiation_mode == 'fill'
    assert fresh.nci_mode == 'separate'
    print("PASS: test_metadata_save_load")


def test_metadata_backward_compat():
    """Loading old models (no restructured keys) should default to False."""
    old_data = {'use_multioutput': True}

    model = XGBoostReactorModel(n_estimators=10)
    model._restore_model_specific_attributes(old_data)
    assert model.use_restructured is False
    assert model.use_multioutput is True
    print("PASS: test_metadata_backward_compat")


if __name__ == '__main__':
    np.random.seed(42)
    test_fit_restructured()
    test_predict_restructured()
    test_non_restructured_unchanged()
    test_metadata_save_load()
    test_metadata_backward_compat()
    print("\nAll XGBoost restructured tests passed!")
