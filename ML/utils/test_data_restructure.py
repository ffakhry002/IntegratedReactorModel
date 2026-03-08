"""Unit tests for data_restructure.py"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_restructure import (
    _get_feature_layout,
    restructure_for_position_pooling,
    restructure_single_for_prediction,
)


def test_feature_layout():
    assert _get_feature_layout('vacuum', 'single') == (2, 4, 1, 4)
    assert _get_feature_layout('fill', 'single') == (5, 5, 1, 4)
    assert _get_feature_layout('fill', 'separate') == (5, 5, 3, 4)
    print("PASS: test_feature_layout")


def test_output_shapes_fill_separate():
    """Fill + separate NCI: 37 features."""
    N = 10
    X = np.random.rand(N, 37)
    y = np.random.rand(N, 4)
    groups = np.arange(N)

    Xp, yp, gp = restructure_for_position_pooling(X, y, groups, 'fill', 'separate')
    assert Xp.shape == (N * 4, 37), f"Expected ({N*4}, 37), got {Xp.shape}"
    assert yp.shape == (N * 4,), f"Expected ({N*4},), got {yp.shape}"
    assert gp.shape == (N * 4,), f"Expected ({N*4},), got {gp.shape}"
    print("PASS: test_output_shapes_fill_separate")


def test_output_shapes_vacuum():
    """Vacuum: 22 features."""
    N = 10
    X = np.random.rand(N, 22)
    y = np.random.rand(N, 4)
    groups = np.arange(N)

    Xp, yp, gp = restructure_for_position_pooling(X, y, groups, 'vacuum', 'single')
    assert Xp.shape == (N * 4, 22), f"Expected ({N*4}, 22), got {Xp.shape}"
    assert yp.shape == (N * 4,), f"Expected ({N*4},), got {yp.shape}"
    print("PASS: test_output_shapes_vacuum")


def test_output_shapes_fill_single():
    """Fill + single NCI: 29 features."""
    N = 10
    X = np.random.rand(N, 29)
    y = np.random.rand(N, 4)
    groups = np.arange(N)

    Xp, yp, gp = restructure_for_position_pooling(X, y, groups, 'fill', 'single')
    assert Xp.shape == (N * 4, 29), f"Expected ({N*4}, 29), got {Xp.shape}"
    print("PASS: test_output_shapes_fill_single")


def test_global_features_unchanged():
    """Global features should be identical across all 4 rows from same sample."""
    N = 5
    X = np.random.rand(N, 37)
    y = np.random.rand(N, 4)
    groups = np.arange(N)

    Xp, _, _ = restructure_for_position_pooling(X, y, groups, 'fill', 'separate')

    for i in range(N):
        original_global = X[i, :5]
        for pos in range(4):
            pooled_global = Xp[i * 4 + pos, :5]
            np.testing.assert_array_equal(
                pooled_global, original_global,
                err_msg=f"Global features differ for sample {i}, pos {pos}"
            )
    print("PASS: test_global_features_unchanged")


def test_my_features_correct():
    """'My' local and NCI features should match the correct position from original."""
    N = 3
    X = np.random.rand(N, 37)
    y = np.random.rand(N, 4)
    groups = np.arange(N)

    Xp, yp, _ = restructure_for_position_pooling(X, y, groups, 'fill', 'separate')

    for i in range(N):
        for my_pos in range(4):
            row = Xp[i * 4 + my_pos]

            # MY local: indices 5:10 in restructured should match position my_pos in original
            original_local_start = 5 + my_pos * 5
            expected_local = X[i, original_local_start:original_local_start + 5]
            np.testing.assert_array_equal(
                row[5:10], expected_local,
                err_msg=f"MY local wrong for sample {i}, pos {my_pos}"
            )

            # MY NCI: indices 10:13 in restructured should match position my_pos in original
            original_nci_start = 25 + my_pos * 3
            expected_nci = X[i, original_nci_start:original_nci_start + 3]
            np.testing.assert_array_equal(
                row[10:13], expected_nci,
                err_msg=f"MY NCI wrong for sample {i}, pos {my_pos}"
            )
    print("PASS: test_my_features_correct")


def test_target_values_correct():
    """Each restructured row's target should match the original position's flux."""
    N = 5
    X = np.random.rand(N, 37)
    y = np.arange(N * 4).reshape(N, 4).astype(float)  # Distinctive values
    groups = np.arange(N)

    _, yp, _ = restructure_for_position_pooling(X, y, groups, 'fill', 'separate')

    for i in range(N):
        for pos in range(4):
            assert yp[i * 4 + pos] == y[i, pos], \
                f"Target mismatch: sample {i}, pos {pos}"
    print("PASS: test_target_values_correct")


def test_groups_preserved():
    """All 4 rows from one sample should have the same group ID."""
    N = 8
    X = np.random.rand(N, 37)
    y = np.random.rand(N, 4)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # 8-fold augmentation pattern

    _, _, gp = restructure_for_position_pooling(X, y, groups, 'fill', 'separate')

    for i in range(N):
        for pos in range(4):
            assert gp[i * 4 + pos] == groups[i], \
                f"Group mismatch: sample {i}, pos {pos}"
    print("PASS: test_groups_preserved")


def test_prediction_restructure_matches_training():
    """restructure_single_for_prediction should produce same features as training restructure."""
    N = 5
    X = np.random.rand(N, 37)
    y = np.random.rand(N, 4)
    groups = np.arange(N)

    Xp_train, _, _ = restructure_for_position_pooling(X, y, groups, 'fill', 'separate')
    Xp_pred = restructure_single_for_prediction(X, 'fill', 'separate')

    np.testing.assert_array_equal(
        Xp_train, Xp_pred,
        err_msg="Prediction restructure doesn't match training restructure"
    )
    print("PASS: test_prediction_restructure_matches_training")


def test_other_features_are_remaining_positions():
    """'Other' features should contain exactly the 3 non-target positions, in order."""
    X = np.random.rand(1, 37)
    y = np.random.rand(1, 4)
    groups = np.array([0])

    Xp, _, _ = restructure_for_position_pooling(X, y, groups, 'fill', 'separate')

    for my_pos in range(4):
        row = Xp[my_pos]
        other_positions = [p for p in range(4) if p != my_pos]

        # Other local: indices 13:28 (3 positions × 5 features)
        for j, other_pos in enumerate(other_positions):
            expected_local = X[0, 5 + other_pos * 5: 5 + other_pos * 5 + 5]
            actual_local = row[13 + j * 5: 13 + j * 5 + 5]
            np.testing.assert_array_equal(
                actual_local, expected_local,
                err_msg=f"Other local wrong for my_pos={my_pos}, other={other_pos}"
            )

        # Other NCI: indices 28:37 (3 positions × 3 features)
        for j, other_pos in enumerate(other_positions):
            expected_nci = X[0, 25 + other_pos * 3: 25 + other_pos * 3 + 3]
            actual_nci = row[28 + j * 3: 28 + j * 3 + 3]
            np.testing.assert_array_equal(
                actual_nci, expected_nci,
                err_msg=f"Other NCI wrong for my_pos={my_pos}, other={other_pos}"
            )
    print("PASS: test_other_features_are_remaining_positions")


def test_wrong_feature_count_raises():
    """Should raise ValueError if feature count doesn't match mode."""
    X = np.random.rand(5, 30)  # Wrong size for any mode
    y = np.random.rand(5, 4)
    groups = np.arange(5)

    try:
        restructure_for_position_pooling(X, y, groups, 'fill', 'separate')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        restructure_single_for_prediction(X, 'fill', 'separate')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS: test_wrong_feature_count_raises")


if __name__ == '__main__':
    test_feature_layout()
    test_output_shapes_fill_separate()
    test_output_shapes_vacuum()
    test_output_shapes_fill_single()
    test_global_features_unchanged()
    test_my_features_correct()
    test_target_values_correct()
    test_groups_preserved()
    test_prediction_restructure_matches_training()
    test_other_features_are_remaining_positions()
    test_wrong_feature_count_raises()
    print("\nAll tests passed!")
