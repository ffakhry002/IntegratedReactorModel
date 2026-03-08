"""
Smoke test: ModelTrainer with use_restructured=True.
Runs 1 Optuna trial + final training + prediction shape check.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force fill mode before importing anything that reads the module-level toggle
from ML_models.encodings import encoding_methods
encoding_methods.IRRADIATION_MODE = 'fill'
encoding_methods.NCI_MODE = 'separate'

from execution.data_handler import DataHandler
from execution.model_trainer import ModelTrainer
from execution.config import TrainingConfig


def test_trainer_restructured_flux():
    """Full train_model flow with restructured=True."""
    np.random.seed(42)
    N = 80  # 10 groups × 8 augmentations
    X = np.random.rand(N, 37)
    y_flux = np.random.rand(N, 4)
    y_keff = np.random.rand(N)
    groups = np.repeat(np.arange(10), 8)

    data_splits = {
        'X_train': X,
        'X_test': np.random.rand(16, 37),
        'y_flux_train': y_flux,
        'y_flux_test': np.random.rand(16, 4),
        'y_keff_train': y_keff,
        'y_keff_test': np.random.rand(16),
        'groups_train': groups,
        'groups_test': np.repeat(np.arange(2), 8),
        'lattices_train': None,
        'lattices_test': None,
    }

    config = TrainingConfig()
    config.optimization = 'optuna'
    config.n_trials = 1
    config.n_jobs = 1
    config.n_jobs_xgb = 1
    config.cores_per_xgb_trial = 1
    config.use_restructured = True
    config.flux_mode = 'total'

    handler = DataHandler()
    handler.flux_mode = 'total'
    handler.use_log_flux = True
    trainer = ModelTrainer(data_handler=handler)

    model, metrics, best_params = trainer.train_model(
        model_type='xgboost',
        target='flux',
        data_splits=data_splits,
        config=config,
        encoding='physics'
    )

    # Model should be restructured
    assert model.use_restructured is True, "Model should have use_restructured=True"

    # predict_flux should accept original layout and return (n, 4)
    preds = model.predict_flux(data_splits['X_test'])
    assert preds.shape == (16, 4), f"Expected (16, 4), got {preds.shape}"

    print(f"Metrics: {metrics}")
    print("PASS: test_trainer_restructured_flux")


if __name__ == '__main__':
    test_trainer_restructured_flux()
    print("\nTrainer restructured test passed!")
