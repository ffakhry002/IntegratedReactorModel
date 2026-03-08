"""
Data restructuring utilities for position-pooling (Option B).

Converts multi-output reactor data (1 row per config, 4 flux targets) into
single-output data (4 rows per config, 1 flux target each) by reordering
features so "my" position is always at fixed indices.

Feature layout after restructuring:
  [global] [MY_local] [MY_nci] [OTHER_local] [OTHER_nci]

The model sees "my" features at consistent indices regardless of which
physical position is being predicted, enabling position-invariant learning.
"""

import numpy as np
from typing import Tuple


def _get_feature_layout(irradiation_mode: str, nci_mode: str, nci_disabled: bool = False):
    """Return (n_global, n_local_per_pos, n_nci_per_pos, n_positions) for a given mode.

    Vacuum:              2 global, 4 local/pos, 1 NCI/pos  -> 22 total
    Vacuum (no NCI):     2 global, 4 local/pos, 0 NCI/pos  -> 18 total
    Fill + single:       5 global, 5 local/pos, 1 NCI/pos  -> 29 total
    Fill + separate:     5 global, 5 local/pos, 3 NCI/pos  -> 37 total
    Fill (no NCI):       5 global, 5 local/pos, 0 NCI/pos  -> 25 total
    """
    n_positions = 4

    if irradiation_mode == 'vacuum':
        n_nci = 0 if nci_disabled else 1
        return 2, 4, n_nci, n_positions
    elif nci_disabled:
        return 5, 5, 0, n_positions
    elif nci_mode == 'separate':
        return 5, 5, 3, n_positions
    else:
        return 5, 5, 1, n_positions


def restructure_for_position_pooling(
    X: np.ndarray,
    y_flux: np.ndarray,
    groups: np.ndarray,
    irradiation_mode: str = 'fill',
    nci_mode: str = 'separate',
    nci_disabled: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restructure data for position pooling (training time).

    Each (config) row with 4 flux targets becomes 4 (config, position) rows
    with 1 flux target each. Features are reordered so the target position's
    local/NCI features always occupy the same indices.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
        Feature matrix in original layout.
    y_flux : np.ndarray, shape (N, 4)
        Flux targets, one column per position.
    groups : np.ndarray, shape (N,)
        Group IDs for cross-validation.
    irradiation_mode : str
        'vacuum' or 'fill'
    nci_mode : str
        'single' or 'separate'

    Returns
    -------
    X_pooled : np.ndarray, shape (4N, F)
        Restructured features.
    y_pooled : np.ndarray, shape (4N,)
        Single flux target per row.
    groups_pooled : np.ndarray, shape (4N,)
        Group IDs (same group for all 4 rows from one original sample).
    """
    n_global, n_local, n_nci, n_pos = _get_feature_layout(irradiation_mode, nci_mode, nci_disabled)
    n_samples = X.shape[0]

    expected_features = n_global + n_local * n_pos + n_nci * n_pos
    if X.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features for {irradiation_mode}/{nci_mode}, "
            f"got {X.shape[1]}"
        )
    if y_flux.shape != (n_samples, n_pos):
        raise ValueError(
            f"Expected y_flux shape ({n_samples}, {n_pos}), got {y_flux.shape}"
        )

    n_out = n_samples * n_pos
    X_pooled = np.empty((n_out, expected_features), dtype=X.dtype)
    y_pooled = np.empty(n_out, dtype=y_flux.dtype)
    groups_pooled = np.empty(n_out, dtype=groups.dtype)

    local_block_start = n_global
    nci_block_start = n_global + n_local * n_pos

    for idx in range(n_samples):
        row = X[idx]
        global_feats = row[:n_global]

        all_local = []
        for p in range(n_pos):
            s = local_block_start + p * n_local
            all_local.append(row[s:s + n_local])

        all_nci = []
        for p in range(n_pos):
            s = nci_block_start + p * n_nci
            all_nci.append(row[s:s + n_nci])

        for my_pos in range(n_pos):
            out_idx = idx * n_pos + my_pos

            other_local = np.concatenate([all_local[p] for p in range(n_pos) if p != my_pos])
            other_nci = np.concatenate([all_nci[p] for p in range(n_pos) if p != my_pos])

            X_pooled[out_idx] = np.concatenate([
                global_feats,
                all_local[my_pos],
                all_nci[my_pos],
                other_local,
                other_nci,
            ])

            y_pooled[out_idx] = y_flux[idx, my_pos]
            groups_pooled[out_idx] = groups[idx]

    return X_pooled, y_pooled, groups_pooled


def restructure_single_for_prediction(
    X: np.ndarray,
    irradiation_mode: str = 'fill',
    nci_mode: str = 'separate',
    nci_disabled: bool = False,
) -> np.ndarray:
    """Restructure feature matrix for prediction (inference time).

    Converts each original-layout row into 4 restructured rows (one per
    position), suitable for feeding into a model trained on pooled data.
    The 4 predictions can then be stacked back into (n, 4).

    Parameters
    ----------
    X : np.ndarray, shape (n, F)
        Feature matrix in original layout.
    irradiation_mode : str
        'vacuum' or 'fill'
    nci_mode : str
        'single' or 'separate'

    Returns
    -------
    X_pooled : np.ndarray, shape (4n, F)
        Restructured features. Rows are ordered as:
        [sample0_pos0, sample0_pos1, sample0_pos2, sample0_pos3,
         sample1_pos0, ...]
        so reshaping predictions to (n, 4) recovers per-position outputs.
    """
    n_global, n_local, n_nci, n_pos = _get_feature_layout(irradiation_mode, nci_mode, nci_disabled)
    n_samples = X.shape[0]
    n_features = n_global + n_local * n_pos + n_nci * n_pos

    if X.shape[1] != n_features:
        raise ValueError(
            f"Expected {n_features} features for {irradiation_mode}/{nci_mode}, "
            f"got {X.shape[1]}"
        )

    X_pooled = np.empty((n_samples * n_pos, n_features), dtype=X.dtype)

    local_block_start = n_global
    nci_block_start = n_global + n_local * n_pos

    for idx in range(n_samples):
        row = X[idx]
        global_feats = row[:n_global]

        all_local = []
        for p in range(n_pos):
            s = local_block_start + p * n_local
            all_local.append(row[s:s + n_local])

        all_nci = []
        for p in range(n_pos):
            s = nci_block_start + p * n_nci
            all_nci.append(row[s:s + n_nci])

        for my_pos in range(n_pos):
            out_idx = idx * n_pos + my_pos

            other_local = np.concatenate([all_local[p] for p in range(n_pos) if p != my_pos])
            other_nci = np.concatenate([all_nci[p] for p in range(n_pos) if p != my_pos])

            X_pooled[out_idx] = np.concatenate([
                global_feats,
                all_local[my_pos],
                all_nci[my_pos],
                other_local,
                other_nci,
            ])

    return X_pooled
