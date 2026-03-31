"""Multi-network thesis layouts: assemble 16-channel predictions from submodels."""
from __future__ import annotations

import os
from datetime import datetime

import joblib
import numpy as np
from typing import List, Optional

from ML_models.neural_net_train import NeuralNetReactorModel


def _X_after_lambdas(
    X: np.ndarray,
    lattices: Optional[np.ndarray],
    lambdas: Optional[dict],
    encoding: str,
    irradiation_mode: str,
    nci_mode: str,
) -> np.ndarray:
    """Regenerate feature matrix when joint λ was used (physics + lattices)."""
    if not lambdas:
        return X
    if lattices is None:
        raise ValueError(
            "Per-network λ regeneration requires lattices (set composite._lattices_eval)."
        )
    from utils.lambda_feature_regenerator import LambdaFeatureRegenerator

    fr = LambdaFeatureRegenerator(encoding)
    fd = fr.separate_features(X, lattices, irradiation_mode, nci_mode)
    return fr.regenerate_features(fd, **lambdas)


class NeuralNetCompositeThesis:
    """Holds multiple ``NeuralNetReactorModel`` instances for nn_config 2, 3, or 5."""

    def __init__(
        self,
        nn_config: int,
        models: List[NeuralNetReactorModel],
        encoding: str = 'physics',
        irradiation_mode: str = 'fill',
        nci_mode: str = 'separate',
        nci_disabled: bool = False,
        optimize_lambda: bool = True,
    ):
        self.nn_config = nn_config
        self.models = models
        self.encoding = encoding
        self.irradiation_mode = irradiation_mode
        self.nci_mode = nci_mode
        self.nci_disabled = nci_disabled
        self.optimize_lambda = optimize_lambda
        self.model_class_name = 'neural_net_composite'
        self.flux_mode = 'energy_sixteen'
        self._n_flux_outputs = 16
        # Set before predict_flux when λ differs per submodel (main test vs Set C)
        self._lattices_eval: Optional[np.ndarray] = None

    def predict_flux(self, X: np.ndarray) -> np.ndarray:
        """Return (n, 16) predictions in log space."""
        n = X.shape[0]
        out = np.zeros((n, 16), dtype=np.float64)
        lattices = self._lattices_eval

        if self.nn_config == 2:
            for g, m in enumerate(self.models):
                lam = getattr(m, 'regen_lambdas_', None) or {}
                Xi = _X_after_lambdas(
                    X, lattices, lam if self.optimize_lambda else None,
                    self.encoding, self.irradiation_mode, self.nci_mode,
                )
                pred = m.predict_flux(Xi)
                for pos in range(4):
                    out[:, pos * 4 + g] = pred[:, pos]

        elif self.nn_config == 3:
            k = 0
            for g in range(4):
                for pos in range(4):
                    col = pos * 4 + g
                    m = self.models[k]
                    lam = getattr(m, 'regen_lambdas_', None) or {}
                    Xi = _X_after_lambdas(
                        X, lattices, lam if self.optimize_lambda else None,
                        self.encoding, self.irradiation_mode, self.nci_mode,
                    )
                    out[:, col] = m.predict_flux(Xi).ravel()
                    k += 1

        elif self.nn_config == 5:
            from utils.data_restructure import restructure_single_for_prediction

            for g, m in enumerate(self.models):
                lam = getattr(m, 'regen_lambdas_', None) or {}
                Xi = _X_after_lambdas(
                    X, lattices, lam if self.optimize_lambda else None,
                    self.encoding, self.irradiation_mode, self.nci_mode,
                )
                X_pool = restructure_single_for_prediction(
                    Xi, self.irradiation_mode, self.nci_mode, self.nci_disabled)
                pred = m.predict_flux(X_pool).ravel()
                for i in range(n):
                    for pos in range(4):
                        idx = i * 4 + pos
                        out[i, pos * 4 + g] = pred[idx]
        else:
            raise ValueError(f"Composite predict not used for nn_config={self.nn_config}")

        return out

    def set_flux_mode(self, flux_mode: str):
        self.flux_mode = flux_mode

    def save_model(
        self,
        filepath,
        model_type='flux',
        encoding='physics',
        optimization_method='raytune',
        flux_scale=1e14,
        use_log_flux=True,
        flux_mode='energy_sixteen',
        irradiation_mode='fill',
        nci_mode='separate',
        **extra_metadata,
    ):
        """Persist composite and submodels (same kwargs contract as ``ReactorModelBase``)."""
        dir_name = os.path.dirname(filepath) or '.'
        os.makedirs(dir_name, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        new_filename = (
            f"{self.model_class_name}_{model_type}_{flux_mode}_{encoding}_"
            f"{optimization_method}.pkl"
        )
        full_path = os.path.join(dir_name, new_filename)
        save_dict = {
            'model_class': self.model_class_name,
            'composite': True,
            'nn_config': self.nn_config,
            'models': self.models,
            'encoding': self.encoding,
            'optimize_lambda': self.optimize_lambda,
            'irradiation_mode': self.irradiation_mode,
            'nci_mode': self.nci_mode,
            'nci_disabled': self.nci_disabled,
            'model_type': model_type,
            'flux_scale': flux_scale,
            'use_log_flux': use_log_flux,
            'flux_mode': flux_mode,
            'saved_at': datetime.now().isoformat(),
            **extra_metadata,
        }
        joblib.dump(save_dict, full_path)
        print(f"Composite model saved to: {full_path}")
        return full_path
