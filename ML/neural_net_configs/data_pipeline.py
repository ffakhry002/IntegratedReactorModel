"""
Prepare (X, y, groups) for neural-net training after lambda regeneration.

energy_sixteen column layout (per row): for position p in 0..3,
  cols [4*p + 0] = total, [4*p+1] = thermal, [4*p+2] = epithermal, [4*p+3] = fast.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from utils.data_restructure import (
    restructure_for_position_pooling_multiout,
)


def slice_y_for_flux_group(y_16: np.ndarray, group_index: int) -> np.ndarray:
    """Config 2: one network per flux *type* across 4 positions — 4 outputs.

    group_index: 0=total, 1=thermal, 2=epithermal, 3=fast.
    Returns shape (N, 4): columns are that group at positions 0..3.
    """
    if y_16.shape[1] != 16:
        raise ValueError(f"Expected y shape (*, 16), got {y_16.shape}")
    cols = [p * 4 + group_index for p in range(4)]
    return y_16[:, cols]


def slice_y_single_target(y_16: np.ndarray, position_index: int, group_index: int) -> np.ndarray:
    """Config 3: single (position, group) target — shape (N, 1)."""
    j = position_index * 4 + group_index
    return y_16[:, j : j + 1]


def prepare_xy_after_lambda(
    X_regen: np.ndarray,
    y_flux: np.ndarray,
    groups: np.ndarray,
    nn_config: int,
    irradiation_mode: str,
    nci_mode: str,
    nci_disabled: bool,
    flux_group_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply NN layout to lambda-regenerated features (standard 8N rows, y columns vary).

    Parameters
    ----------
    flux_group_index : int, optional
        For config 5: which of {0,1,2,3} (tot,th,epi,fast) for single-output net.
        For config 2: which group slice (same encoding).
    """
    n_pos = 4
    if nn_config == 1:
        return X_regen, y_flux, groups
    if nn_config == 2:
        if flux_group_index is None:
            raise ValueError("nn_config=2 requires flux_group_index 0..3")
        y_sub = slice_y_for_flux_group(y_flux, flux_group_index)
        return X_regen, y_sub, groups
    if nn_config == 3:
        # Caller passes already-sliced y (N,1) for one (position, group)
        return X_regen, y_flux, groups
    if nn_config == 4:
        if y_flux.shape[1] != 16:
            raise ValueError("nn_config=4 requires energy_sixteen y with 16 columns")
        return restructure_for_position_pooling_multiout(
            X_regen, y_flux, groups, irradiation_mode, nci_mode, nci_disabled,
            n_flux_per_position=4,
        )
    if nn_config == 5:
        if y_flux.shape[1] != 16:
            raise ValueError("nn_config=5 requires energy_sixteen y with 16 columns")
        Xp, y4, gp = restructure_for_position_pooling_multiout(
            X_regen, y_flux, groups, irradiation_mode, nci_mode, nci_disabled,
            n_flux_per_position=4,
        )
        if flux_group_index is None:
            raise ValueError("nn_config=5 requires flux_group_index 0..3")
        y1 = y4[:, flux_group_index : flux_group_index + 1]
        return Xp, y1, gp
    raise ValueError(f"Unknown nn_config: {nn_config}")
