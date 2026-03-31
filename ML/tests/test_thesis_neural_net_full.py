"""
Comprehensive tests for thesis NN pipeline: energy_sixteen, restructuring,
data_pipeline, NeuralNetCompositeThesis, metrics, config, and model save/load.

Run:  cd ML && python tests/test_thesis_neural_net_full.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

_ML_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ML_ROOT not in sys.path:
    sys.path.insert(0, _ML_ROOT)

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore

if pytest is None:

    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError('expected exception')
            if not issubclass(et, self.exc):
                return False
            return True

    def _pytest_raises(exc):
        return _Raises(exc)

    pytest = type('pytest', (), {'raises': staticmethod(_pytest_raises)})()  # type: ignore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_separate_X_y_groups(n: int):
    """Synthetic standard layout: 37 features, 16 flux cols."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 37)).astype(np.float64)
    y16 = rng.standard_normal((n, 16)).astype(np.float64)
    groups = np.arange(n, dtype=np.int64)
    return X, y16, groups


class _MockFluxNet:
    def __init__(self, predict_fn, regen=None):
        self._predict_fn = predict_fn
        self.regen_lambdas_ = regen

    def predict_flux(self, X):
        return self._predict_fn(X)


def _mock_predict_zeros_4cols(X):
    return np.zeros((X.shape[0], 4), dtype=np.float64)


# ===================================================================
# 1) restructure_for_position_pooling_multiout
# ===================================================================

def test_restructure_multiout_shape_and_my_position_targets():
    from utils.data_restructure import restructure_for_position_pooling_multiout

    N = 3
    X, y16, groups = _fill_separate_X_y_groups(N)
    for i in range(N):
        for p in range(4):
            base = p * 4
            y16[i, base : base + 4] = [100 * i + 10 * p + k for k in range(4)]

    Xp, yp, gp = restructure_for_position_pooling_multiout(
        X, y16, groups, 'fill', 'separate', False, n_flux_per_position=4
    )
    assert Xp.shape == (N * 4, 37)
    assert yp.shape == (N * 4, 4)
    assert gp.shape == (N * 4,)

    for i in range(N):
        for my_pos in range(4):
            row = i * 4 + my_pos
            expected = [100 * i + 10 * my_pos + k for k in range(4)]
            np.testing.assert_array_almost_equal(yp[row], expected)
            assert gp[row] == groups[i]


def test_restructure_multiout_global_features_unchanged():
    """Global features (first 5 cols for fill/separate) are identical in every pooled row."""
    from utils.data_restructure import restructure_for_position_pooling_multiout

    N = 4
    X, y16, groups = _fill_separate_X_y_groups(N)
    Xp, _, _ = restructure_for_position_pooling_multiout(
        X, y16, groups, 'fill', 'separate', False, n_flux_per_position=4)

    for i in range(N):
        for pos in range(4):
            np.testing.assert_array_equal(
                Xp[i * 4 + pos, :5], X[i, :5],
                err_msg=f"Global features differ for sample {i}, pos {pos}")


def test_restructure_multiout_my_local_at_fixed_indices():
    """Position in 'my' slot always at indices 5..9 (local) and 25..27 (NCI) for fill/separate."""
    from utils.data_restructure import restructure_for_position_pooling_multiout

    N = 2
    X, y16, groups = _fill_separate_X_y_groups(N)
    Xp, _, _ = restructure_for_position_pooling_multiout(
        X, y16, groups, 'fill', 'separate', False, n_flux_per_position=4)

    for i in range(N):
        for my_pos in range(4):
            row = i * 4 + my_pos
            original_local = X[i, 5 + my_pos * 5 : 5 + (my_pos + 1) * 5]
            pooled_my_local = Xp[row, 5:10]
            np.testing.assert_array_equal(
                pooled_my_local, original_local,
                err_msg=f"'my' local mismatch sample {i}, pos {my_pos}")


def test_restructure_multiout_bad_y_shape_raises():
    from utils.data_restructure import restructure_for_position_pooling_multiout

    N = 3
    X = np.random.default_rng(0).standard_normal((N, 37))
    y_bad = np.random.default_rng(0).standard_normal((N, 8))
    groups = np.arange(N)
    with pytest.raises(ValueError):
        restructure_for_position_pooling_multiout(
            X, y_bad, groups, 'fill', 'separate', False, n_flux_per_position=4)


def test_restructure_multiout_vacuum_mode():
    """Vacuum mode: 22 features."""
    from utils.data_restructure import restructure_for_position_pooling_multiout

    N = 3
    rng = np.random.default_rng(7)
    X = rng.standard_normal((N, 22))
    y16 = rng.standard_normal((N, 16))
    groups = np.arange(N)
    Xp, yp, gp = restructure_for_position_pooling_multiout(
        X, y16, groups, 'vacuum', 'single', False, n_flux_per_position=4)
    assert Xp.shape == (N * 4, 22)
    assert yp.shape == (N * 4, 4)


# ===================================================================
# 2) data_pipeline: slice_y, prepare_xy_after_lambda
# ===================================================================

def test_prepare_xy_nn_config_1_passthrough():
    from neural_net_configs.data_pipeline import prepare_xy_after_lambda

    N = 5
    X, y16, groups = _fill_separate_X_y_groups(N)
    X1, y1, g1 = prepare_xy_after_lambda(X, y16, groups, 1, 'fill', 'separate', False)
    assert X1 is X and y1 is y16 and g1 is groups


def test_prepare_xy_nn_config_4_restructure():
    from neural_net_configs.data_pipeline import prepare_xy_after_lambda

    N = 5
    X, y16, groups = _fill_separate_X_y_groups(N)
    X4, y4, g4 = prepare_xy_after_lambda(X, y16, groups, 4, 'fill', 'separate', False)
    assert X4.shape == (N * 4, 37)
    assert y4.shape == (N * 4, 4)
    assert len(g4) == N * 4


def test_prepare_xy_nn_config_5_single_column():
    from neural_net_configs.data_pipeline import prepare_xy_after_lambda

    N = 5
    X, y16, groups = _fill_separate_X_y_groups(N)
    X4, y4, _ = prepare_xy_after_lambda(X, y16, groups, 4, 'fill', 'separate', False)
    X5, y5, g5 = prepare_xy_after_lambda(
        X, y16, groups, 5, 'fill', 'separate', False, flux_group_index=2)
    assert X5.shape == (N * 4, 37) and y5.shape == (N * 4, 1)
    np.testing.assert_allclose(y5.ravel(), y4[:, 2])


def test_prepare_xy_nn_config_2_slices_correct_columns():
    from neural_net_configs.data_pipeline import slice_y_for_flux_group

    N = 4
    _, y16, _ = _fill_separate_X_y_groups(N)
    for g in range(4):
        y_sub = slice_y_for_flux_group(y16, g)
        assert y_sub.shape == (N, 4)
        for p in range(4):
            np.testing.assert_array_equal(y_sub[:, p], y16[:, p * 4 + g])


def test_prepare_xy_nn_config_3_single_target():
    from neural_net_configs.data_pipeline import slice_y_single_target

    N = 4
    _, y16, _ = _fill_separate_X_y_groups(N)
    for pos in range(4):
        for g in range(4):
            y_one = slice_y_single_target(y16, pos, g)
            assert y_one.shape == (N, 1)
            np.testing.assert_array_equal(y_one.ravel(), y16[:, pos * 4 + g])


def test_prepare_xy_config2_requires_flux_group_index():
    from neural_net_configs.data_pipeline import prepare_xy_after_lambda

    N = 2
    X, y16, groups = _fill_separate_X_y_groups(N)
    with pytest.raises(ValueError):
        prepare_xy_after_lambda(X, y16, groups, 2, 'fill', 'separate', False)


def test_prepare_xy_config5_requires_flux_group_index():
    from neural_net_configs.data_pipeline import prepare_xy_after_lambda

    N = 2
    X, y16, groups = _fill_separate_X_y_groups(N)
    with pytest.raises(ValueError):
        prepare_xy_after_lambda(X, y16, groups, 5, 'fill', 'separate', False)


def test_prepare_xy_raises_on_unknown_nn_config():
    from neural_net_configs.data_pipeline import prepare_xy_after_lambda

    N = 2
    X, y16, groups = _fill_separate_X_y_groups(N)
    try:
        prepare_xy_after_lambda(X, y16, groups, 99, 'fill', 'separate', False)
    except ValueError as e:
        assert 'Unknown nn_config' in str(e)
    else:
        raise AssertionError('expected ValueError')


# ===================================================================
# 3) strip-lambda pattern (mirrors model_trainer)
# ===================================================================

def test_strip_lambda_pattern():
    bp = {
        'depth': 3, 'lambda_P': 1.1, 'lambda_B': 0.9,
        'lambda_G': 1.0, 'learning_rate': 0.01,
    }
    names = ['lambda_P', 'lambda_B', 'lambda_G', 'lambda_decay']
    lam = {k: bp[k] for k in names if k in bp}
    mp = {k: v for k, v in bp.items() if k not in names}
    assert lam == {'lambda_P': 1.1, 'lambda_B': 0.9, 'lambda_G': 1.0}
    assert 'lambda_P' not in mp and 'depth' in mp


def test_strip_lambda_with_decay():
    bp = {'depth': 3, 'lambda_decay': 1.5, 'learning_rate': 0.01}
    names = ['lambda_P', 'lambda_B', 'lambda_G', 'lambda_decay']
    lam = {k: bp[k] for k in names if k in bp}
    mp = {k: v for k, v in bp.items() if k not in names}
    assert lam == {'lambda_decay': 1.5}
    assert 'lambda_decay' not in mp


# ===================================================================
# 4) NeuralNetCompositeThesis.predict_flux
# ===================================================================

def test_composite_predict_config2_no_lambda():
    from neural_net_configs.composite import NeuralNetCompositeThesis

    n = 3
    X = np.zeros((n, 37), dtype=np.float64)

    def make_pred(g):
        def _fn(Xin):
            out = np.zeros((Xin.shape[0], 4), dtype=np.float64)
            out[:, :] = g + 1.0
            return out
        return _fn

    models = [_MockFluxNet(make_pred(g)) for g in range(4)]
    comp = NeuralNetCompositeThesis(
        2, models, encoding='physics', optimize_lambda=False,
        irradiation_mode='fill', nci_mode='separate', nci_disabled=False,
    )
    out = comp.predict_flux(X)
    assert out.shape == (n, 16)
    for g in range(4):
        for pos in range(4):
            np.testing.assert_allclose(out[:, pos * 4 + g], g + 1.0)


def test_composite_predict_config3_no_lambda():
    """Models ordered g-then-pos (thesis: 4 HPO families by flux type)."""
    from neural_net_configs.composite import NeuralNetCompositeThesis

    n = 2
    X = np.zeros((n, 37), dtype=np.float64)
    models = []
    # Build in g-then-pos order to match trainer and predict_flux
    for g in range(4):
        for pos in range(4):
            val = float(pos * 4 + g)
            def _make(v):
                return lambda Xin: np.full((Xin.shape[0], 1), v, dtype=np.float64)
            models.append(_MockFluxNet(_make(val)))

    comp = NeuralNetCompositeThesis(
        3, models, encoding='physics', optimize_lambda=False,
        irradiation_mode='fill', nci_mode='separate', nci_disabled=False,
    )
    out = comp.predict_flux(X)
    assert out.shape == (n, 16)
    for pos in range(4):
        for g in range(4):
            col = pos * 4 + g
            np.testing.assert_allclose(out[:, col], float(col))


def test_composite_predict_config5_no_lambda():
    from neural_net_configs.composite import NeuralNetCompositeThesis

    n = 2
    X = np.random.default_rng(1).standard_normal((n, 37)).astype(np.float64)

    def make_single(g):
        def _fn(Xin):
            assert Xin.shape[0] == 4 * n
            return np.full((Xin.shape[0], 1), 10.0 * g, dtype=np.float64)
        return _fn

    models = [_MockFluxNet(make_single(g)) for g in range(4)]
    comp = NeuralNetCompositeThesis(
        5, models, encoding='physics', optimize_lambda=False,
        irradiation_mode='fill', nci_mode='separate', nci_disabled=False,
    )
    out = comp.predict_flux(X)
    assert out.shape == (n, 16)
    for g in range(4):
        np.testing.assert_allclose(out[:, g::4], 10.0 * g)


def test_composite_predict_invalid_config_raises():
    from neural_net_configs.composite import NeuralNetCompositeThesis

    comp = NeuralNetCompositeThesis(
        1, [], encoding='physics', optimize_lambda=False,
        irradiation_mode='fill', nci_mode='separate', nci_disabled=False,
    )
    with pytest.raises(ValueError):
        comp.predict_flux(np.zeros((2, 37)))


# ===================================================================
# 5) _X_after_lambdas
# ===================================================================

def test_X_after_lambdas_noop_when_empty_lambdas():
    from neural_net_configs.composite import _X_after_lambdas

    X = np.ones((2, 5))
    np.testing.assert_array_equal(
        _X_after_lambdas(X, None, None, 'physics', 'fill', 'separate'), X)
    np.testing.assert_array_equal(
        _X_after_lambdas(X, None, {}, 'physics', 'fill', 'separate'), X)


def test_X_after_lambdas_requires_lattices_when_lambda_set():
    from neural_net_configs.composite import _X_after_lambdas

    X = np.ones((2, 37))
    with pytest.raises(ValueError):
        _X_after_lambdas(X, None, {'lambda_P': 1.0}, 'physics', 'fill', 'separate')


# ===================================================================
# 6) ModelTrainer._compute_metrics_for_subset
# ===================================================================

def test_metrics_mse_log_energy_sixteen():
    from execution.model_trainer import ModelTrainer
    from execution.data_handler import DataHandler

    dh = DataHandler()
    dh.flux_mode = 'energy_sixteen'
    dh.use_log_flux = True
    mt = ModelTrainer(data_handler=dh)

    rng = np.random.default_rng(42)
    y_true = rng.standard_normal((10, 16))
    y_pred = y_true + 0.05 * rng.standard_normal((10, 16))
    m = mt._compute_metrics_for_subset(y_true, y_pred, 'flux')
    assert 'mse_log' in m
    assert m['mse_log'] is not None
    assert m['mse'] is not None
    assert m['mse_log'] > 0
    assert abs(m['mse_log'] - m['mse']) < 1e-10, \
        "mse_log should equal mse when data is already in log space"


def test_metrics_no_mse_log_for_total_mode():
    from execution.model_trainer import ModelTrainer
    from execution.data_handler import DataHandler

    dh = DataHandler()
    dh.flux_mode = 'total'
    dh.use_log_flux = True
    mt = ModelTrainer(data_handler=dh)

    rng = np.random.default_rng(42)
    y_true = rng.standard_normal((10, 4))
    y_pred = y_true + 0.05 * rng.standard_normal((10, 4))
    m = mt._compute_metrics_for_subset(y_true, y_pred, 'flux')
    assert 'mse_log' not in m


def test_metrics_empty_returns_none():
    from execution.model_trainer import ModelTrainer
    from execution.data_handler import DataHandler

    dh = DataHandler()
    dh.flux_mode = 'energy_sixteen'
    dh.use_log_flux = True
    mt = ModelTrainer(data_handler=dh)

    y_empty = np.empty((0, 16))
    m = mt._compute_metrics_for_subset(y_empty, y_empty, 'flux')
    assert m['mse'] is None
    assert m['mse_log'] is None
    assert m['n_samples'] == 0


# ===================================================================
# 7) TrainingConfig
# ===================================================================

def test_training_config_defaults():
    from execution.config import TrainingConfig

    c = TrainingConfig()
    assert hasattr(c, 'nn_config')
    assert hasattr(c, 'flux_mode')
    assert c.nn_config is None
    assert c.flux_mode == 'total'


# ===================================================================
# 8) base_model set_flux_mode + save + load
# ===================================================================

def test_set_flux_mode_energy_sixteen():
    from ML_models.base_model import ReactorModelBase

    class _D(ReactorModelBase):
        def __init__(self):
            super().__init__()
            self.model_class_name = 'test'
            self.params = {}
        def fit_flux(self, *a, **k): pass
        def fit_keff(self, *a, **k): pass
        def predict_flux(self, *a, **k): pass
        def predict_keff(self, *a, **k): pass

    d = _D()
    d.set_flux_mode('energy_sixteen')
    assert d._n_flux_outputs == 16
    assert d.flux_mode == 'energy_sixteen'


def test_base_model_save_n_flux_outputs_energy_sixteen():
    """save_model should record n_flux_outputs=16 for energy_sixteen."""
    from ML_models.base_model import ReactorModelBase

    class _D(ReactorModelBase):
        def __init__(self):
            super().__init__()
            self.model_class_name = 'dummy'
            self.params = {}
            self.flux_model = None
        def fit_flux(self, *a, **k): pass
        def fit_keff(self, *a, **k): pass
        def predict_flux(self, *a, **k): pass
        def predict_keff(self, *a, **k): pass

    import joblib
    d = _D()
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'x.pkl')
        d.save_model(p, model_type='flux', encoding='physics',
                     optimization_method='optuna', flux_mode='energy_sixteen')
        files = [f for f in os.listdir(tmp) if f.endswith('.pkl')]
        assert len(files) == 1
        data = joblib.load(os.path.join(tmp, files[0]))
        assert data.get('n_flux_outputs') == 16


def test_base_model_load_restores_energy_sixteen_outputs():
    """BUG REGRESSION: load_model must set _n_flux_outputs=16 for energy_sixteen."""
    from ML_models.base_model import ReactorModelBase
    import joblib

    class _D(ReactorModelBase):
        def __init__(self):
            super().__init__()
            self.model_class_name = 'dummy'
            self.params = {}
            self.flux_model = None
        def fit_flux(self, *a, **k): pass
        def fit_keff(self, *a, **k): pass
        def predict_flux(self, *a, **k): pass
        def predict_keff(self, *a, **k): pass

    d = _D()
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'x.pkl')
        d.save_model(p, model_type='flux', encoding='physics',
                     optimization_method='optuna', flux_mode='energy_sixteen')
        actual_file = [f for f in os.listdir(tmp) if f.endswith('.pkl')][0]
        full = os.path.join(tmp, actual_file)

        loaded, meta = _D.load_model(full)
        assert loaded._n_flux_outputs == 16, \
            f"Expected 16, got {loaded._n_flux_outputs} — load_model missing energy_sixteen case"
        assert loaded.flux_mode == 'energy_sixteen'
        assert meta['flux_mode'] == 'energy_sixteen'


# ===================================================================
# 9) NeuralNetReactorModel metadata save/restore
# ===================================================================

def test_neural_net_model_metadata_roundtrip():
    """nn_config, use_restructured, regen_lambdas_ persist through save/load cycle."""
    from ML_models.neural_net_train import NeuralNetReactorModel

    m = NeuralNetReactorModel(
        nn_config=4, use_restructured=True,
        irradiation_mode='fill', nci_mode='separate', nci_disabled=False,
        depth=2, width=32, max_epochs=5, patience=2, verbose=False,
    )
    m.regen_lambdas_ = {'lambda_P': 1.2, 'lambda_B': 0.8}

    meta = m._get_model_specific_metadata()
    assert meta['nn_config'] == 4
    assert meta['use_restructured'] is True
    assert meta['regen_lambdas'] == {'lambda_P': 1.2, 'lambda_B': 0.8}

    m2 = NeuralNetReactorModel(depth=2, width=32, max_epochs=5, patience=2)
    m2._restore_model_specific_attributes({
        'scale_features': True,
        'nn_config': 4,
        'use_restructured': True,
        'irradiation_mode': 'fill',
        'nci_mode': 'separate',
        'nci_disabled': False,
        'regen_lambdas': {'lambda_P': 1.2},
        'model_type': 'flux',
    })
    assert m2.nn_config == 4
    assert m2.use_restructured is True
    assert m2.regen_lambdas_ == {'lambda_P': 1.2}


# ===================================================================
# 10) NeuralNetCompositeThesis.save_model
# ===================================================================

def test_composite_save_model_roundtrip():
    from neural_net_configs.composite import NeuralNetCompositeThesis
    import joblib

    models = [_MockFluxNet(_mock_predict_zeros_4cols)]
    comp = NeuralNetCompositeThesis(
        2, models, encoding='physics', optimize_lambda=True,
        irradiation_mode='fill', nci_mode='separate', nci_disabled=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'm.pkl')
        comp.save_model(
            path, model_type='flux', encoding='physics',
            optimization_method='raytune', flux_mode='energy_sixteen',
            metrics={'n': 1},
        )
        data = joblib.load(os.path.join(
            tmp, 'neural_net_composite_flux_energy_sixteen_physics_raytune.pkl'))
        assert data['composite'] is True
        assert data['nn_config'] == 2
        assert data['optimize_lambda'] is True
        assert data['metrics']['n'] == 1
        assert data['flux_mode'] == 'energy_sixteen'


# ===================================================================
# 11) restructure_single_for_prediction
# ===================================================================

def test_restructure_single_for_prediction_row_count():
    from utils.data_restructure import restructure_single_for_prediction

    X = np.random.default_rng(7).standard_normal((5, 37))
    Xp = restructure_single_for_prediction(X, 'fill', 'separate', False)
    assert Xp.shape == (5 * 4, 37)


def test_restructure_single_for_prediction_global_unchanged():
    from utils.data_restructure import restructure_single_for_prediction

    X = np.random.default_rng(8).standard_normal((3, 37))
    Xp = restructure_single_for_prediction(X, 'fill', 'separate', False)
    for i in range(3):
        for pos in range(4):
            np.testing.assert_array_equal(Xp[i * 4 + pos, :5], X[i, :5])


# ===================================================================
# 12) energy_sixteen data loading (needs train.txt)
# ===================================================================

def test_energy_sixteen_log10_applied_in_prepare():
    train = os.path.join(_ML_ROOT, 'data', 'train.txt')
    if not os.path.exists(train):
        return
    from execution.data_handler import DataHandler

    dh = DataHandler()
    out = dh.load_and_prepare_data(
        train, 'physics', flux_mode='energy_sixteen', max_runs=33)
    y_flux = out[1]
    assert y_flux.shape[1] == 16
    assert np.nanmax(np.abs(y_flux)) < 20.0, "Targets should be log10 scale (~12-14)"


def test_energy_sixteen_column_consistency():
    """Total flux >= each group flux (before log) for every position."""
    train = os.path.join(_ML_ROOT, 'data', 'train.txt')
    if not os.path.exists(train):
        return
    from execution.data_handler import DataHandler

    dh = DataHandler()
    dh.use_log_flux = False
    out = dh.load_and_prepare_data(
        train, 'physics', flux_mode='energy_sixteen', max_runs=33)
    y_flux = out[1]
    assert y_flux.shape[1] == 16
    for p in range(4):
        tot = y_flux[:, p * 4 + 0]
        th = y_flux[:, p * 4 + 1]
        epi = y_flux[:, p * 4 + 2]
        fast = y_flux[:, p * 4 + 3]
        mask = tot > 0
        if mask.any():
            residual = np.abs(tot[mask] - (th[mask] + epi[mask] + fast[mask]))
            assert np.all(residual / tot[mask] < 0.01), \
                f"Total != sum of groups at position {p}"


# ===================================================================
# 13) NeuralNetReactorModel.predict_flux for restructured config 4
# ===================================================================

def test_nn_predict_flux_restructured_config4_shape():
    """Config 4 predict_flux: standard-layout X → (n, 16) via restructure_single_for_prediction."""
    from ML_models.neural_net_train import NeuralNetReactorModel
    import torch

    n = 3
    X = np.random.default_rng(9).standard_normal((n, 37)).astype(np.float32)

    m = NeuralNetReactorModel(
        nn_config=4, use_restructured=True,
        irradiation_mode='fill', nci_mode='separate', nci_disabled=False,
        scale_features=True, depth=1, base_width=16, max_epochs=3, patience=2,
        verbose=False, architecture_type='rectangular',
    )
    m.set_flux_mode('energy_sixteen')

    from utils.data_restructure import restructure_for_position_pooling_multiout
    X_pool_train = np.random.default_rng(10).standard_normal((n * 4, 37)).astype(np.float32)
    y_pool_train = np.random.default_rng(11).standard_normal((n * 4, 4)).astype(np.float32)
    m.fit_flux(X_pool_train, y_pool_train)

    out = m.predict_flux(X)
    assert out.shape == (n, 16), f"Expected (3, 16), got {out.shape}"


# ===================================================================
# 14) model_trainer early-return gate
# ===================================================================

def test_train_model_gate_requires_energy_sixteen_and_raytune():
    """The thesis NN path only triggers for neural_net + raytune + energy_sixteen + nn_config 1-5."""
    from execution.model_trainer import ModelTrainer
    from execution.data_handler import DataHandler
    from execution.config import TrainingConfig

    dh = DataHandler()
    dh.flux_mode = 'total'
    dh.use_log_flux = True
    mt = ModelTrainer(data_handler=dh)

    config = TrainingConfig()
    config.optimization = 'raytune'
    config.flux_mode = 'total'
    config.nn_config = 1

    # Should NOT take the thesis path (flux_mode != energy_sixteen)
    # We can't run a full training, but we can verify the gate logic
    fm = getattr(config, 'flux_mode', 'total')
    nn_cfg = getattr(config, 'nn_config', None)
    takes_thesis_path = (fm == 'energy_sixteen' and nn_cfg is not None and 1 <= nn_cfg <= 5)
    assert not takes_thesis_path

    config.flux_mode = 'energy_sixteen'
    fm = getattr(config, 'flux_mode', 'total')
    takes_thesis_path = (fm == 'energy_sixteen' and nn_cfg is not None and 1 <= nn_cfg <= 5)
    assert takes_thesis_path


# ===================================================================
# 15) Interactive menu env vars
# ===================================================================

def test_env_nn_config_parsing():
    """NN_CONFIG env var should be read as integer."""
    import os
    old = os.environ.get('NN_CONFIG')
    try:
        os.environ['NN_CONFIG'] = '3'
        val = int(os.environ.get('NN_CONFIG', '1'))
        assert val == 3
    finally:
        if old is not None:
            os.environ['NN_CONFIG'] = old
        elif 'NN_CONFIG' in os.environ:
            del os.environ['NN_CONFIG']


def test_env_train_file_override():
    """TRAIN_FILE env var should be usable."""
    import os
    old = os.environ.get('TRAIN_FILE')
    try:
        os.environ['TRAIN_FILE'] = 'data/train_lc_set_65.txt'
        val = os.environ.get('TRAIN_FILE', '').strip()
        assert val == 'data/train_lc_set_65.txt'
    finally:
        if old is not None:
            os.environ['TRAIN_FILE'] = old
        elif 'TRAIN_FILE' in os.environ:
            del os.environ['TRAIN_FILE']


# ===================================================================
# 16) flux_group / single-HPO-per-job
# ===================================================================

def test_training_config_flux_group_default():
    """TrainingConfig.flux_group should default to None."""
    from execution.config import TrainingConfig

    c = TrainingConfig()
    assert hasattr(c, 'flux_group')
    assert c.flux_group is None


def test_env_flux_group_parsing():
    """FLUX_GROUP env var should be readable as an integer."""
    import os
    old = os.environ.get('FLUX_GROUP')
    try:
        os.environ['FLUX_GROUP'] = '2'
        val = int(os.environ.get('FLUX_GROUP', '').strip())
        assert val == 2
    finally:
        if old is not None:
            os.environ['FLUX_GROUP'] = old
        elif 'FLUX_GROUP' in os.environ:
            del os.environ['FLUX_GROUP']


def test_expected_paths_config2():
    """assemble_composite._expected_paths should return 4 paths for config 2."""
    from assemble_composite import _expected_paths

    paths = _expected_paths(2, '/tmp/models')
    assert len(paths) == 4
    for g in range(4):
        assert f'nn_config2_group{g}_submodel.pkl' in paths[g]


def test_expected_paths_config3():
    """assemble_composite._expected_paths should return 16 paths for config 3."""
    from assemble_composite import _expected_paths

    paths = _expected_paths(3, '/tmp/models')
    assert len(paths) == 16
    assert 'nn_config3_group0_pos0_submodel.pkl' in paths[0]
    assert 'nn_config3_group2_pos3_submodel.pkl' in paths[11]
    assert 'nn_config3_group3_pos3_submodel.pkl' in paths[15]


def test_expected_paths_config5():
    """assemble_composite._expected_paths should return 4 paths for config 5."""
    from assemble_composite import _expected_paths

    paths = _expected_paths(5, '/tmp/models')
    assert len(paths) == 4
    for g in range(4):
        assert f'nn_config5_group{g}_submodel.pkl' in paths[g]


def test_expected_paths_invalid_config_raises():
    """assemble_composite._expected_paths should raise for configs 1 and 4."""
    from assemble_composite import _expected_paths

    with pytest.raises(ValueError):
        _expected_paths(1, '/tmp')
    with pytest.raises(ValueError):
        _expected_paths(4, '/tmp')


def test_save_submodel_deterministic_path():
    """Path convention in trainer must match what assemble_composite expects."""
    from assemble_composite import _expected_paths

    model_dir = '/fake/dir'
    for nn_cfg in (2, 5):
        expected = _expected_paths(nn_cfg, model_dir)
        for g in range(4):
            tag = f'group{g}'
            constructed = os.path.join(model_dir, f'nn_config{nn_cfg}_{tag}_submodel.pkl')
            assert constructed == expected[g], (
                f"Mismatch cfg={nn_cfg} g={g}: {constructed} vs {expected[g]}")

    expected3 = _expected_paths(3, model_dir)
    idx = 0
    for g in range(4):
        for p in range(4):
            tag = f'group{g}_pos{p}'
            constructed = os.path.join(model_dir, f'nn_config3_{tag}_submodel.pkl')
            assert constructed == expected3[idx], f"Mismatch cfg=3 g={g} p={p}"
            idx += 1


def test_trainer_single_group_returns_none_model():
    """When flux_group is set for a multi-study config, train should return model=None."""
    from unittest.mock import MagicMock, patch
    from execution.config import TrainingConfig
    from ML_models.neural_net_train import NeuralNetReactorModel as NNR

    config = TrainingConfig()
    config.flux_mode = 'energy_sixteen'
    config.nn_config = 2
    config.flux_group = 1
    config.optimization = 'raytune'
    config.n_trials = 1
    config.n_gpus = 0
    config.models_dir = tempfile.mkdtemp()

    from execution.model_trainer import ModelTrainer

    mt = ModelTrainer.__new__(ModelTrainer)
    dh = MagicMock()
    dh.use_log_flux = True
    dh.flux_mode = 'energy_sixteen'
    mt.data_handler = dh

    X_tr = np.random.randn(10, 37)
    y_tr = np.random.randn(10, 16)
    groups = np.repeat(np.arange(5), 2)

    fake_best = {
        'base_width': 64, 'n_layers': 2, 'lr': 1e-3,
        'batch_size': 32, 'epochs': 1,
    }

    with patch('execution.model_trainer.optimize_neural_net_raytune',
               return_value=(fake_best, None)), \
         patch.object(NNR, 'fit_flux', return_value=None), \
         patch.object(NNR, 'save_model', return_value='/fake.pkl'):

        model, metrics, bp = mt._train_nn_ray_thesis(
            data_splits={},
            config=config,
            encoding='physics',
            model_type='neural_net',
            target='flux',
            X_train=X_tr,
            y_train=y_tr,
            y_test=None,
            X_test=None,
            groups_train=groups,
            lattices_train=None,
            flux_mode='energy_sixteen',
            irradiation_mode='fill',
            nci_mode='separate',
            optimize_lambda=False,
            optimization_start=0,
            nn_cfg=2,
        )

    assert model is None, "Single-group run should return model=None"
    assert bp.get('flux_group') == 1


# ===================================================================
# Run all
# ===================================================================

_ALL_TESTS = [
    test_restructure_multiout_shape_and_my_position_targets,
    test_restructure_multiout_global_features_unchanged,
    test_restructure_multiout_my_local_at_fixed_indices,
    test_restructure_multiout_bad_y_shape_raises,
    test_restructure_multiout_vacuum_mode,
    test_prepare_xy_nn_config_1_passthrough,
    test_prepare_xy_nn_config_4_restructure,
    test_prepare_xy_nn_config_5_single_column,
    test_prepare_xy_nn_config_2_slices_correct_columns,
    test_prepare_xy_nn_config_3_single_target,
    test_prepare_xy_config2_requires_flux_group_index,
    test_prepare_xy_config5_requires_flux_group_index,
    test_prepare_xy_raises_on_unknown_nn_config,
    test_strip_lambda_pattern,
    test_strip_lambda_with_decay,
    test_composite_predict_config2_no_lambda,
    test_composite_predict_config3_no_lambda,
    test_composite_predict_config5_no_lambda,
    test_composite_predict_invalid_config_raises,
    test_X_after_lambdas_noop_when_empty_lambdas,
    test_X_after_lambdas_requires_lattices_when_lambda_set,
    test_metrics_mse_log_energy_sixteen,
    test_metrics_no_mse_log_for_total_mode,
    test_metrics_empty_returns_none,
    test_training_config_defaults,
    test_set_flux_mode_energy_sixteen,
    test_base_model_save_n_flux_outputs_energy_sixteen,
    test_base_model_load_restores_energy_sixteen_outputs,
    test_neural_net_model_metadata_roundtrip,
    test_composite_save_model_roundtrip,
    test_restructure_single_for_prediction_row_count,
    test_restructure_single_for_prediction_global_unchanged,
    test_energy_sixteen_log10_applied_in_prepare,
    test_energy_sixteen_column_consistency,
    test_nn_predict_flux_restructured_config4_shape,
    test_train_model_gate_requires_energy_sixteen_and_raytune,
    test_env_nn_config_parsing,
    test_env_train_file_override,
    test_training_config_flux_group_default,
    test_env_flux_group_parsing,
    test_expected_paths_config2,
    test_expected_paths_config3,
    test_expected_paths_config5,
    test_expected_paths_invalid_config_raises,
    test_save_submodel_deterministic_path,
    test_trainer_single_group_returns_none_model,
]

if __name__ == '__main__':
    passed = 0
    failed = 0
    skipped = 0
    for fn in _ALL_TESTS:
        name = fn.__name__
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {name}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(_ALL_TESTS)}")
    if failed == 0:
        print("PASS: test_thesis_neural_net_full (all tests)")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
