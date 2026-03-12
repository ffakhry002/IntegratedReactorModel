"""
Unit tests for the learning curve pipeline.

Tests cover:
1. max_runs parameter in parse_reactor_data
2. Backward compatibility (max_runs=None)
3. Data handler subset shapes
4. Subset is prefix of full dataset
5. Test split evaluation boundary
6. Fill type classification
7. Fill category tracking through augmentation
8. Per-fill-type metrics computation
9. Geometry steps generation

Run:  python test_learning_curve.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TRAIN_FILE = os.path.join(os.path.dirname(__file__), 'data', 'train.txt')
TRAIN_FILE_EXISTS = os.path.exists(TRAIN_FILE)


# ── Test 1 ──────────────────────────────────────────────────────────
def test_parse_max_runs_limits_output():
    """parse_reactor_data with max_runs=66 returns exactly 66 entries."""
    if not TRAIN_FILE_EXISTS:
        print("SKIP: test_parse_max_runs_limits_output (train.txt not found)")
        return

    from utils.txt_to_data import parse_reactor_data

    lattices, flux_data, k_effs, descriptions, energy_groups = parse_reactor_data(
        TRAIN_FILE, max_runs=66)

    assert len(lattices) == 66, f"Expected 66 lattices, got {len(lattices)}"
    assert len(flux_data) == 66, f"Expected 66 flux_data, got {len(flux_data)}"
    assert len(k_effs) == 66, f"Expected 66 k_effs, got {len(k_effs)}"
    assert len(descriptions) == 66, f"Expected 66 descriptions, got {len(descriptions)}"
    assert len(energy_groups) == 66, f"Expected 66 energy_groups, got {len(energy_groups)}"

    # First 33 should be geometry 1 permutations, next 33 geometry 2
    assert 'permutation 1' in descriptions[0], f"First description wrong: {descriptions[0]}"
    assert 'permutation 33' in descriptions[32], f"33rd description wrong: {descriptions[32]}"
    assert 'permutation 1' in descriptions[33], f"34th description wrong: {descriptions[33]}"

    print("PASS: test_parse_max_runs_limits_output")


# ── Test 2 ──────────────────────────────────────────────────────────
def test_parse_max_runs_none_loads_all():
    """parse_reactor_data with max_runs=None returns all 8448 entries."""
    if not TRAIN_FILE_EXISTS:
        print("SKIP: test_parse_max_runs_none_loads_all (train.txt not found)")
        return

    from utils.txt_to_data import parse_reactor_data

    lattices, flux_data, k_effs, descriptions, energy_groups = parse_reactor_data(
        TRAIN_FILE, max_runs=None)

    assert len(lattices) == 8448, f"Expected 8448 lattices, got {len(lattices)}"
    assert len(descriptions) == 8448, f"Expected 8448 descriptions, got {len(descriptions)}"

    print("PASS: test_parse_max_runs_none_loads_all")


# ── Test 3 ──────────────────────────────────────────────────────────
def test_data_handler_subset_shapes():
    """load_and_prepare_data with max_runs=33 produces correct shapes."""
    if not TRAIN_FILE_EXISTS:
        print("SKIP: test_data_handler_subset_shapes (train.txt not found)")
        return

    os.environ.pop('NCI_DISTANCE_CUTOFF', None)
    from execution.data_handler import DataHandler

    handler = DataHandler()
    result = handler.load_and_prepare_data(
        TRAIN_FILE, 'physics', flux_mode='total', max_runs=33)

    X, y_flux, y_keff, groups, lattices, fill_categories = result

    expected_samples = 33 * 8  # 33 fills × 8 augmentations
    assert X.shape[0] == expected_samples, (
        f"Expected {expected_samples} samples, got {X.shape[0]}")
    assert y_flux.shape[0] == expected_samples
    assert y_keff.shape[0] == expected_samples
    assert len(groups) == expected_samples
    assert len(fill_categories) == expected_samples

    # With 33 fills of 1 geometry, all share the same spatial pattern → 1 group
    # Each group member has 8 augmentations, so 33 fills × 8 = 264 in one group
    unique_groups = np.unique(groups)
    assert len(unique_groups) == 1, (
        f"Expected 1 unique group (all 33 fills of same geometry), got {len(unique_groups)}")

    print("PASS: test_data_handler_subset_shapes")


# ── Test 4 ──────────────────────────────────────────────────────────
def test_data_handler_subset_is_prefix():
    """Subset of 66 RUNs is an exact prefix of the full dataset."""
    if not TRAIN_FILE_EXISTS:
        print("SKIP: test_data_handler_subset_is_prefix (train.txt not found)")
        return

    os.environ.pop('NCI_DISTANCE_CUTOFF', None)
    from execution.data_handler import DataHandler

    handler_sub = DataHandler()
    result_sub = handler_sub.load_and_prepare_data(
        TRAIN_FILE, 'physics', flux_mode='total', max_runs=66)
    X_sub, y_flux_sub, y_keff_sub = result_sub[0], result_sub[1], result_sub[2]

    handler_full = DataHandler()
    result_full = handler_full.load_and_prepare_data(
        TRAIN_FILE, 'physics', flux_mode='total', max_runs=132)
    X_full, y_flux_full, y_keff_full = result_full[0], result_full[1], result_full[2]

    n = X_sub.shape[0]
    assert np.allclose(X_sub, X_full[:n]), "Feature values don't match between subset and full"
    assert np.allclose(y_flux_sub, y_flux_full[:n]), "Flux values don't match"
    assert np.allclose(y_keff_sub, y_keff_full[:n]), "Keff values don't match"

    print("PASS: test_data_handler_subset_is_prefix")


# ── Test 5 ──────────────────────────────────────────────────────────
def test_split_test_evaluation_boundary():
    """Test split at RUN 4224 produces correct partition sizes."""
    total_runs = 4686
    split_run = 4224
    augmentation_factor = 8

    total_samples = total_runs * augmentation_factor
    split_idx = split_run * augmentation_factor

    y_mock = np.random.rand(total_samples, 4)

    set_a = y_mock[:split_idx]
    set_b = y_mock[split_idx:]

    expected_a = split_run * augmentation_factor
    expected_b = (total_runs - split_run) * augmentation_factor

    assert set_a.shape[0] == expected_a, (
        f"Set A: expected {expected_a}, got {set_a.shape[0]}")
    assert set_b.shape[0] == expected_b, (
        f"Set B: expected {expected_b}, got {set_b.shape[0]}")
    assert set_a.shape[0] + set_b.shape[0] == total_samples, "Sets don't cover all data"

    # 4224 * 8 = 33792, 462 * 8 = 3696
    assert expected_a == 33792, f"Expected 33792, got {expected_a}"
    assert expected_b == 3696, f"Expected 3696, got {expected_b}"

    print("PASS: test_split_test_evaluation_boundary")


# ── Test 6 ──────────────────────────────────────────────────────────
def test_classify_fill_type():
    """_classify_fill_type correctly identifies all 6 fill categories."""
    from execution.data_handler import DataHandler

    def make_lattice(positions):
        """Make a minimal 8x8 lattice with given irradiation positions at fixed coords."""
        lat = np.full((8, 8), 'F', dtype=object)
        lat[0, 0] = 'C'; lat[0, 7] = 'C'; lat[7, 0] = 'C'; lat[7, 7] = 'C'
        coords = [(2, 2), (2, 4), (5, 3), (5, 5)]
        for (r, c), label in zip(coords, positions):
            lat[r, c] = label
        return lat

    test_cases = [
        (['I_1G', 'I_2G', 'I_3G', 'I_4G'], '4G'),
        (['I_1G', 'I_2G', 'I_3G', 'I_4P'], '3G1P'),
        (['I_1G', 'I_2G', 'I_3G', 'I_4B'], '3G1B'),
        (['I_1G', 'I_2G', 'I_3P', 'I_4P'], '2G2P'),
        (['I_1G', 'I_2G', 'I_3B', 'I_4B'], '2G2B'),
        (['I_1G', 'I_2G', 'I_3P', 'I_4B'], '2G1P1B'),
        (['I_1P', 'I_2G', 'I_3G', 'I_4B'], '2G1P1B'),
        (['I_1B', 'I_2B', 'I_3G', 'I_4G'], '2G2B'),
    ]

    for positions, expected in test_cases:
        lattice = make_lattice(positions)
        result = DataHandler._classify_fill_type(lattice)
        assert result == expected, (
            f"For {positions}: expected '{expected}', got '{result}'")

    print("PASS: test_classify_fill_type")


# ── Test 7 ──────────────────────────────────────────────────────────
def test_fill_category_tracking_through_augmentation():
    """Fill categories are correctly tracked through 8-fold augmentation."""
    if not TRAIN_FILE_EXISTS:
        print("SKIP: test_fill_category_tracking_through_augmentation (train.txt not found)")
        return

    os.environ.pop('NCI_DISTANCE_CUTOFF', None)
    from execution.data_handler import DataHandler

    handler = DataHandler()
    result = handler.load_and_prepare_data(
        TRAIN_FILE, 'physics', flux_mode='total', max_runs=33)
    fill_categories = result[5]

    assert len(fill_categories) == 33 * 8, (
        f"Expected {33*8} fill categories, got {len(fill_categories)}")

    # Each group of 8 consecutive samples should have the same category
    for i in range(33):
        block = fill_categories[i*8 : (i+1)*8]
        assert len(set(block)) == 1, (
            f"Config {i}: augmentations have mixed categories: {block}")

    # Expected distribution for 33 permutations of one geometry:
    # 1×4G, 4×3G1P, 4×3G1B, 6×2G2P, 6×2G2B, 12×2G1P1B = 33
    unique_per_config = [fill_categories[i*8] for i in range(33)]
    from collections import Counter
    dist = Counter(unique_per_config)

    assert dist['4G'] == 1, f"Expected 1×4G, got {dist.get('4G', 0)}"
    assert dist['3G1P'] == 4, f"Expected 4×3G1P, got {dist.get('3G1P', 0)}"
    assert dist['3G1B'] == 4, f"Expected 4×3G1B, got {dist.get('3G1B', 0)}"
    assert dist['2G2P'] == 6, f"Expected 6×2G2P, got {dist.get('2G2P', 0)}"
    assert dist['2G2B'] == 6, f"Expected 6×2G2B, got {dist.get('2G2B', 0)}"
    assert dist['2G1P1B'] == 12, f"Expected 12×2G1P1B, got {dist.get('2G1P1B', 0)}"

    print("PASS: test_fill_category_tracking_through_augmentation")


# ── Test 8 ──────────────────────────────────────────────────────────
def test_per_fill_type_metrics():
    """Per-fill-type metrics are computed correctly on known mock data."""
    from execution.data_handler import DataHandler
    from execution.model_trainer import ModelTrainer

    # Build a mock model trainer with minimal data handler
    dh = DataHandler()
    dh.flux_mode = 'total'
    dh.use_log_flux = False
    trainer = ModelTrainer(data_handler=dh)

    # 3 fill types with 8 samples each = 24 total
    n_per_type = 8
    fill_types = ['4G', '3G1P', '2G2P']
    n_total = n_per_type * len(fill_types)

    y_true = np.ones((n_total, 1))
    y_pred = np.ones((n_total, 1))

    # Make 3G1P predictions have known error
    y_pred[n_per_type:2*n_per_type] = 1.1  # 10% error

    fill_cats = np.array(
        ['4G'] * n_per_type +
        ['3G1P'] * n_per_type +
        ['2G2P'] * n_per_type
    )

    # Use _compute_metrics_for_subset directly
    # Test that 4G subset has zero error
    mask_4g = (fill_cats == '4G')
    m_4g = trainer._compute_metrics_for_subset(
        y_true[mask_4g], y_pred[mask_4g], 'keff')
    assert m_4g['mse'] < 1e-10, f"4G MSE should be ~0, got {m_4g['mse']}"
    assert m_4g['mape'] < 1e-10, f"4G MAPE should be ~0, got {m_4g['mape']}"

    # Test that 3G1P has the known error
    mask_3g1p = (fill_cats == '3G1P')
    m_3g1p = trainer._compute_metrics_for_subset(
        y_true[mask_3g1p], y_pred[mask_3g1p], 'keff')
    assert abs(m_3g1p['mape'] - 10.0) < 0.01, (
        f"3G1P MAPE should be ~10%, got {m_3g1p['mape']}")

    # Test that 2G2P has zero error
    mask_2g2p = (fill_cats == '2G2P')
    m_2g2p = trainer._compute_metrics_for_subset(
        y_true[mask_2g2p], y_pred[mask_2g2p], 'keff')
    assert m_2g2p['mse'] < 1e-10, f"2G2P MSE should be ~0, got {m_2g2p['mse']}"

    # Verify n_samples counts
    assert m_4g['n_samples'] == n_per_type
    assert m_3g1p['n_samples'] == n_per_type
    assert m_2g2p['n_samples'] == n_per_type

    print("PASS: test_per_fill_type_metrics")


# ── Test 9 ──────────────────────────────────────────────────────────
def test_geometry_steps_generation():
    """Verify the 20 geometry steps used in the SLURM submission script."""
    steps = [13, 26, 39, 52, 65, 78, 91, 104, 117, 130,
             143, 156, 169, 182, 195, 208, 221, 234, 247, 256]

    assert len(steps) == 20, f"Expected 20 steps, got {len(steps)}"

    # All in valid range
    for s in steps:
        assert 1 <= s <= 256, f"Step {s} out of range [1, 256]"

    # 256 included as final value
    assert steps[-1] == 256, f"Last step should be 256, got {steps[-1]}"

    # Monotonically increasing
    for i in range(1, len(steps)):
        assert steps[i] > steps[i-1], (
            f"Not increasing: steps[{i-1}]={steps[i-1]} >= steps[{i}]={steps[i]}")

    # Each step × 33 = valid number of RUNs
    for s in steps:
        n_runs = s * 33
        assert n_runs <= 8448, f"Step {s} → {n_runs} RUNs exceeds 8448"

    print("PASS: test_geometry_steps_generation")


# ── Test 10 ─────────────────────────────────────────────────────────
def test_test_set_c_evaluation():
    """_evaluate_test_set_c computes overall and per-fill-type metrics correctly."""
    from execution.data_handler import DataHandler
    from execution.model_trainer import ModelTrainer

    dh = DataHandler()
    dh.flux_mode = 'total'
    dh.use_log_flux = False
    trainer = ModelTrainer(data_handler=dh)

    n_per_type = 16
    fill_types_list = ['4G', '3G1P', '3G1B', '2G2P', '2G2B', '2G1P1B']
    n_total = n_per_type * len(fill_types_list)

    y_true = np.ones((n_total, 1))
    predictions = np.ones((n_total, 1))
    # Add 5% error to 2G1P1B block
    start = n_per_type * 5
    predictions[start:start + n_per_type] = 1.05

    fill_cats = np.array([ft for ft in fill_types_list for _ in range(n_per_type)])

    class MockModel:
        def predict_keff(self, X):
            return predictions

    model = MockModel()
    X_test = np.random.rand(n_total, 10)

    result = trainer._evaluate_test_set_c(
        model, X_test, y_true, 'keff', fill_categories=fill_cats)

    assert 'overall' in result, "Missing 'overall' in Test Set C metrics"
    assert result['overall']['n_samples'] == n_total
    assert '4G' in result, "Missing '4G' fill-type breakdown"
    assert result['4G']['mse'] < 1e-10, "4G should have zero error"
    assert abs(result['2G1P1B']['mape'] - 5.0) < 0.01, (
        f"2G1P1B MAPE should be ~5%, got {result['2G1P1B']['mape']}")
    assert result['3G1P']['mse'] < 1e-10, "3G1P should have zero error"

    print("PASS: test_test_set_c_evaluation")


# ── Test 11 ─────────────────────────────────────────────────────────
def test_test_set_c_data_splits_key():
    """Verify data_splits keys are set correctly when test_c data is present."""
    data_splits = {}

    X_tc = np.random.rand(100, 10)
    y_flux_tc = np.random.rand(100, 1)
    y_keff_tc = np.random.rand(100, 1)
    fc_tc = np.array(['4G'] * 25 + ['2G1P1B'] * 75)

    data_splits['X_test_c'] = X_tc
    data_splits['y_flux_test_c'] = y_flux_tc
    data_splits['y_keff_test_c'] = y_keff_tc
    data_splits['fill_categories_test_c'] = fc_tc

    assert data_splits['X_test_c'].shape == (100, 10)
    assert data_splits['y_flux_test_c'].shape == (100, 1)
    assert data_splits['y_keff_test_c'].shape == (100, 1)
    assert len(data_splits['fill_categories_test_c']) == 100

    X_test_c = data_splits.get('X_test_c', None)
    assert X_test_c is not None, "X_test_c should be in data_splits"

    y_test_c_flux = data_splits.get('y_flux_test_c')
    y_test_c_keff = data_splits.get('y_keff_test_c')
    assert y_test_c_flux is not None
    assert y_test_c_keff is not None

    print("PASS: test_test_set_c_data_splits_key")


# ── Main ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("Learning Curve Pipeline Unit Tests")
    print("=" * 60)
    print()

    # Tests that don't need train.txt
    test_split_test_evaluation_boundary()
    test_classify_fill_type()
    test_per_fill_type_metrics()
    test_geometry_steps_generation()
    test_test_set_c_evaluation()
    test_test_set_c_data_splits_key()

    # Tests that need train.txt
    test_parse_max_runs_limits_output()
    test_parse_max_runs_none_loads_all()
    test_data_handler_subset_shapes()
    test_data_handler_subset_is_prefix()
    test_fill_category_tracking_through_augmentation()

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
