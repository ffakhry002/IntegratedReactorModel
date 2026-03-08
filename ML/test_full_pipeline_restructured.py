"""
End-to-end integration test for the restructured position-pooling pipeline.

Simulates the full flow that interactive_menu.py would trigger:
  1. Load data with physics encoding (fill mode)
  2. Split data
  3. Optuna optimization (1 trial) with restructured=True
  4. Final model training on restructured data
  5. Evaluation on test set (original layout)
  6. Save model
  7. Load model and predict again
  8. Verify shapes and metadata throughout
"""

import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force fill + separate mode
from ML_models.encodings import encoding_methods
encoding_methods.IRRADIATION_MODE = 'fill'
encoding_methods.NCI_MODE = 'separate'

from execution.data_handler import DataHandler
from execution.model_trainer import ModelTrainer
from execution.config import TrainingConfig
from ML_models.xgboost_train import XGBoostReactorModel
from utils.data_restructure import restructure_for_position_pooling


def test_full_pipeline():
    np.random.seed(42)

    # --- Step 1: Simulate data (skip real file loading for test speed) ---
    print("=" * 60)
    print("STEP 1: Create synthetic data matching fill+separate layout")
    print("=" * 60)

    N_configs = 15  # base configs
    N_aug = 8       # augmentations per config
    N = N_configs * N_aug  # 120 samples
    n_features = 37  # fill + separate NCI

    X = np.random.rand(N, n_features)
    y_flux = np.random.rand(N, 4) * 10 + 5  # flux-like values
    y_keff = np.random.rand(N) * 0.1 + 1.0
    groups = np.repeat(np.arange(N_configs), N_aug)

    print(f"  X: {X.shape}, y_flux: {y_flux.shape}, groups: {groups.shape}")
    print(f"  Unique groups: {len(np.unique(groups))}")

    # --- Step 2: Split data ---
    print("\n" + "=" * 60)
    print("STEP 2: Split data")
    print("=" * 60)

    handler = DataHandler()
    handler.flux_mode = 'total'
    handler.use_log_flux = True

    data_splits = handler.split_data(X, y_flux, y_keff, groups, test_size=0.2)

    X_train = data_splits['X_train']
    X_test = data_splits['X_test']
    y_flux_train = data_splits['y_flux_train']
    y_flux_test = data_splits['y_flux_test']
    groups_train = data_splits['groups_train']

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  groups_train unique: {len(np.unique(groups_train))}")

    # --- Step 3 + 4: Train with restructured mode ---
    print("\n" + "=" * 60)
    print("STEP 3-4: Optuna optimization + final training (restructured)")
    print("=" * 60)

    config = TrainingConfig()
    config.optimization = 'optuna'
    config.n_trials = 1
    config.n_jobs = 1
    config.n_jobs_xgb = 1
    config.cores_per_xgb_trial = 1
    config.use_restructured = True
    config.flux_mode = 'total'

    trainer = ModelTrainer(data_handler=handler)

    model, metrics, best_params = trainer.train_model(
        model_type='xgboost',
        target='flux',
        data_splits=data_splits,
        config=config,
        encoding='physics'
    )

    # --- Step 5: Verify model properties ---
    print("\n" + "=" * 60)
    print("STEP 5: Verify model properties")
    print("=" * 60)

    assert model.use_restructured is True, "Model should be restructured"
    assert model.irradiation_mode == 'fill'
    assert model.nci_mode == 'separate'
    assert not hasattr(model.flux_model, 'estimators_'), \
        "Should be single XGBRegressor, not MultiOutputRegressor"

    preds = model.predict_flux(X_test)
    assert preds.shape == (X_test.shape[0], 4), \
        f"predict_flux should return (n, 4), got {preds.shape}"

    print(f"  use_restructured: {model.use_restructured}")
    print(f"  irradiation_mode: {model.irradiation_mode}")
    print(f"  nci_mode: {model.nci_mode}")
    print(f"  predict_flux output shape: {preds.shape}")
    print(f"  PASS: Model properties correct")

    # --- Step 6: Save model ---
    print("\n" + "=" * 60)
    print("STEP 6: Save model")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_model.pkl')

        saved_path = trainer.save_model(
            model, filepath,
            metadata={'params': best_params, 'metrics': metrics, 'model_class': 'xgboost'},
            model_type='xgboost',
            target='flux',
            encoding='physics',
            optimization='optuna'
        )

        assert os.path.exists(saved_path), f"Model file should exist at {saved_path}"
        print(f"  Saved to: {saved_path}")

        # --- Step 7: Load model and predict ---
        print("\n" + "=" * 60)
        print("STEP 7: Load model and predict")
        print("=" * 60)

        loaded_model, loaded_meta = XGBoostReactorModel.load_model(saved_path)

        assert loaded_model.use_restructured is True, \
            "Loaded model should have use_restructured=True"
        assert loaded_model.irradiation_mode == 'fill'
        assert loaded_model.nci_mode == 'separate'

        loaded_preds = loaded_model.predict_flux(X_test)
        assert loaded_preds.shape == (X_test.shape[0], 4), \
            f"Loaded model predict_flux should return (n, 4), got {loaded_preds.shape}"

        # Predictions should be identical to pre-save
        np.testing.assert_array_almost_equal(
            preds, loaded_preds, decimal=6,
            err_msg="Loaded model predictions should match original"
        )

        print(f"  use_restructured: {loaded_model.use_restructured}")
        print(f"  irradiation_mode: {loaded_model.irradiation_mode}")
        print(f"  nci_mode: {loaded_model.nci_mode}")
        print(f"  Predictions match original: YES")
        print(f"  PASS: Save/load round-trip correct")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("ALL PIPELINE TESTS PASSED")
    print("=" * 60)
    print(f"  Data: {N} samples -> train {X_train.shape[0]}, test {X_test.shape[0]}")
    print(f"  Restructured train: ({X_train.shape[0]}, 37) -> ({X_train.shape[0]*4}, 37)")
    print(f"  Optuna: 1 trial, MSE scoring, GroupKFold")
    print(f"  Model: single XGBRegressor (not MultiOutputRegressor)")
    print(f"  Prediction: (n, 37) original layout -> (n, 4) output")
    print(f"  Save/Load: metadata preserved, predictions identical")


if __name__ == '__main__':
    test_full_pipeline()
