"""
Verify there is NO data leakage in the restructured pipeline.

Tests:
1. All 4 position rows from the same config are in the same CV fold
2. No augmentation of the same spatial pattern leaks across folds
3. The Optuna objective's restructuring + GroupKFold is leak-free
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import GroupKFold
from utils.data_restructure import restructure_for_position_pooling


def test_positions_never_split_across_folds():
    """All 4 rows from one original sample must land in the same fold."""
    N = 80
    X = np.random.rand(N, 37)
    y = np.random.rand(N, 4)
    # 10 spatial patterns × 8 augmentations each
    groups = np.repeat(np.arange(10), 8)

    Xp, yp, gp = restructure_for_position_pooling(X, y, groups, 'fill', 'separate')

    cv = GroupKFold(n_splits=10)
    for fold, (train_idx, val_idx) in enumerate(cv.split(Xp, yp, gp)):
        train_set = set(train_idx.tolist())
        val_set = set(val_idx.tolist())

        # For every original sample, check all 4 of its rows
        for orig_idx in range(N):
            row_indices = [orig_idx * 4 + pos for pos in range(4)]

            in_train = [r in train_set for r in row_indices]
            in_val = [r in val_set for r in row_indices]

            # All 4 must be in the same set
            assert all(in_train) or all(in_val), (
                f"LEAKAGE! Fold {fold}, original sample {orig_idx}: "
                f"rows {row_indices} split across train/val. "
                f"in_train={in_train}, in_val={in_val}"
            )

    print("PASS: All 4 position rows always in same fold (no position leakage)")


def test_augmentations_never_split():
    """All augmentations of the same spatial pattern must be in same fold."""
    N = 80
    X = np.random.rand(N, 37)
    y = np.random.rand(N, 4)
    # 10 spatial patterns × 8 augmentations = groups 0..9
    groups = np.repeat(np.arange(10), 8)

    Xp, yp, gp = restructure_for_position_pooling(X, y, groups, 'fill', 'separate')

    cv = GroupKFold(n_splits=10)
    for fold, (train_idx, val_idx) in enumerate(cv.split(Xp, yp, gp)):
        train_groups = set(gp[train_idx].tolist())
        val_groups = set(gp[val_idx].tolist())

        overlap = train_groups & val_groups
        assert len(overlap) == 0, (
            f"LEAKAGE! Fold {fold}: groups {overlap} appear in both train and val"
        )

    print("PASS: No spatial pattern appears in both train and val (no augmentation leakage)")


def test_group_counts_after_restructure():
    """Restructuring should multiply rows but NOT create new groups."""
    N = 80
    X = np.random.rand(N, 37)
    y = np.random.rand(N, 4)
    groups = np.repeat(np.arange(10), 8)

    _, _, gp = restructure_for_position_pooling(X, y, groups, 'fill', 'separate')

    original_unique = set(groups.tolist())
    pooled_unique = set(gp.tolist())

    assert original_unique == pooled_unique, (
        f"Group sets differ! Original: {original_unique}, Pooled: {pooled_unique}"
    )

    # Should be 4x more rows but same number of unique groups
    assert len(gp) == 4 * len(groups)
    assert len(pooled_unique) == len(original_unique)

    print(f"PASS: {len(gp)} rows, {len(pooled_unique)} groups (same as original)")


def test_fold_sizes_reasonable():
    """Each fold should have roughly equal samples, and restructured should be 4x."""
    N = 80
    X = np.random.rand(N, 37)
    y = np.random.rand(N, 4)
    groups = np.repeat(np.arange(10), 8)

    Xp, yp, gp = restructure_for_position_pooling(X, y, groups, 'fill', 'separate')

    cv = GroupKFold(n_splits=10)
    for fold, (train_idx, val_idx) in enumerate(cv.split(Xp, yp, gp)):
        n_train = len(train_idx)
        n_val = len(val_idx)
        assert n_train + n_val == len(Xp), "Train + val should equal total"

        # Each fold holds out 1 group = 8 augmentations × 4 positions = 32 rows
        assert n_val == 32, (
            f"Fold {fold}: expected 32 val rows (1 group × 8 aug × 4 pos), got {n_val}"
        )

    print("PASS: Each fold holds out exactly 1 group (32 restructured rows)")


def test_target_position_alignment():
    """After restructuring, row i*4+p should predict position p's flux from sample i."""
    N = 20
    X = np.random.rand(N, 37)
    # Make targets distinctive so we can trace them
    y = np.zeros((N, 4))
    for i in range(N):
        for p in range(4):
            y[i, p] = i * 100 + p  # e.g., sample 5 pos 2 = 502

    groups = np.arange(N)

    _, yp, _ = restructure_for_position_pooling(X, y, groups, 'fill', 'separate')

    for i in range(N):
        for p in range(4):
            expected = i * 100 + p
            actual = yp[i * 4 + p]
            assert actual == expected, (
                f"Target mismatch: sample {i} pos {p}, "
                f"expected {expected}, got {actual}"
            )

    print("PASS: Every restructured target maps to the correct (sample, position) pair")


if __name__ == '__main__':
    np.random.seed(42)
    test_positions_never_split_across_folds()
    test_augmentations_never_split()
    test_group_counts_after_restructure()
    test_fold_sizes_reasonable()
    test_target_position_alignment()
    print("\nAll leakage tests passed -- no data leakage detected.")
