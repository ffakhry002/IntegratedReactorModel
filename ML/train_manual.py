#!/usr/bin/env python3
"""
Manual Model Training from Optimized Parameters
Use this to train models directly with known hyperparameters (e.g., from crashed runs)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from execution.data_handler import DataHandler
from ML_models.xgboost_train import XGBoostReactorModel
from ML_models.svm_train import SVMReactorModel
from ML_models.neural_net_train import NeuralNetReactorModel
from ML_models.random_forest_train import RandomForestReactorModel

# =============================================================================
# CONFIGURATION - EDIT THESE
# =============================================================================

# Model type: 'xgboost', 'svm', 'random_forest', 'neural_net'
MODEL_TYPE = 'xgboost'

# Target: 'flux' or 'keff'
TARGET = 'flux'

# Encoding: 'physics', 'one_hot', 'categorical', 'spatial', 'graph'
ENCODING = 'physics'

# Optimization method (for metadata only): 'optuna', 'three_stage'
OPTIMIZATION_METHOD = 'optuna'

# Data file
DATA_FILE = 'data/train.txt'

# Flux mode (for flux only): 'total', 'energy', 'bin', 'thermal_only', etc.
FLUX_MODE = 'total'

# =============================================================================
# FIXED PARAMETERS (FROM YOUR CONFIGURATION)
# These don't change across optimizations
# =============================================================================

FIXED_PARAMS = {
    'xgboost': {
        'three_stage': {
            'colsample_bylevel': 1.0,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'gamma': 0.0,
            'min_child_weight': 1,
            'random_state': 42,
            'n_jobs': 8,
            'tree_method': 'hist'
        },
        'optuna': {
            'random_state': 42,
            'n_jobs': 8,
            'tree_method': 'hist'
        }
    },
    'svm': {
        'three_stage': {
            'tol': 1e-4,        # Final model tolerance
            'max_iter': 250000,
            'shrinking': False,
            'cache_size': 50000
        },
        'optuna': {
            'tol': 1e-4,        # Final model tolerance
            'max_iter': 250000,
            'shrinking': False,
            'cache_size': 50000
        }
    },
    'random_forest': {
        'three_stage': {
            'random_state': 42,
            'n_jobs': -1
        },
        'optuna': {
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'neural_net': {
        'three_stage': {
            'max_epochs': 1000,
            'patience': 20,
            'validation_fraction': 0.1,
            'verbose': False,
            'random_state': 42
        },
        'optuna': {
            'max_epochs': 1000,
            'patience': 20,
            'validation_fraction': 0.1,
            'verbose': False,
            'random_state': 42
        }
    }
}

# =============================================================================
# OPTIMIZED PARAMETERS - PASTE YOUR PARAMETERS HERE
# Remove any 'estimator__', 'svr__', or 'estimator__svr__' prefixes
# =============================================================================

OPTIMIZED_PARAMS = {
    # EXAMPLE - XGBoost from your crashed run:
    'colsample_bytree': 0.646094213291807,
    'learning_rate': 0.029721821248305497,
    'max_depth': 6,
    'n_estimators': 7380,
    'subsample': 0.9213929279308097

    # EXAMPLE - SVM:
    # 'C': 10.0,
    # 'gamma': 0.005,
    # 'epsilon': 0.01,
    # 'kernel': 'rbf'

    # EXAMPLE - Neural Net:
    # 'architecture_type': 'rectangular',
    # 'base_width': 200,
    # 'depth': 3,
    # 'activations': 'relu',
    # 'learning_rate': 0.001,
    # 'weight_decay': 0.01,
    # 'optimizer': 'adam',
    # 'batch_size': 128,
    # 'dropout_rate': 0.2,
    # 'use_batch_norm': True
}

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_model():
    """Train model with specified parameters"""

    print("="*60)
    print("MANUAL MODEL TRAINING FROM PARAMETERS")
    print("="*60)
    print(f"Model: {MODEL_TYPE}")
    print(f"Target: {TARGET}")
    print(f"Encoding: {ENCODING}")
    print(f"Optimization: {OPTIMIZATION_METHOD}")
    print(f"Data: {DATA_FILE}")
    print("="*60)

    # Combine optimized and fixed parameters
    fixed = FIXED_PARAMS.get(MODEL_TYPE, {}).get(OPTIMIZATION_METHOD, {})
    all_params = {**fixed, **OPTIMIZED_PARAMS}

    print("\nCombined Parameters:")
    for k, v in sorted(all_params.items()):
        print(f"  {k}: {v}")

    # Load data
    print("\n" + "-"*60)
    print("LOADING DATA")
    print("-"*60)
    data_handler = DataHandler()

    result = data_handler.load_and_prepare_data(
        DATA_FILE,
        ENCODING,
        flux_mode=FLUX_MODE if TARGET == 'flux' else 'total'
    )

    X, y_flux, y_keff, groups = result

    # Use all data for training (no test split)
    data_splits = data_handler.split_data(X, y_flux, y_keff, groups, test_size=0.0)
    X_train = data_splits['X_train']

    if TARGET == 'flux':
        y_train = data_splits['y_flux_train']
    else:
        y_train = data_splits['y_keff_train']

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Target shape: {y_train.shape}")

    # Create model
    print("\n" + "-"*60)
    print("CREATING MODEL")
    print("-"*60)

    if MODEL_TYPE == 'xgboost':
        model = XGBoostReactorModel(**all_params)
    elif MODEL_TYPE == 'svm':
        model = SVMReactorModel(**all_params)
    elif MODEL_TYPE == 'random_forest':
        model = RandomForestReactorModel(**all_params)
    elif MODEL_TYPE == 'neural_net':
        model = NeuralNetReactorModel(**all_params)
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

    print(f"Model created: {model.__class__.__name__}")

    # Train model
    print("\n" + "-"*60)
    print("TRAINING MODEL")
    print("-"*60)

    if TARGET == 'flux':
        print("Training flux model...")
        model.fit_flux(X_train, y_train)
    else:
        print("Training keff model...")
        model.fit_keff(X_train, y_train)

    print("Training complete!")

    # Save model
    print("\n" + "-"*60)
    print("SAVING MODEL")
    print("-"*60)

    os.makedirs('outputs/models', exist_ok=True)

    filepath = f'outputs/models/{MODEL_TYPE}_{TARGET}_{ENCODING}_{OPTIMIZATION_METHOD}_manual.pkl'

    # Get encoding metadata
    from ML_models.encodings.encoding_methods import IRRADIATION_MODE, NCI_MODE

    saved_path = model.save_model(
        filepath=filepath,
        model_type=TARGET,
        encoding=ENCODING,
        optimization_method=OPTIMIZATION_METHOD,
        flux_scale=1.0,
        use_log_flux=data_handler.use_log_flux if TARGET == 'flux' else False,
        flux_mode=FLUX_MODE if TARGET == 'flux' else 'total',
        irradiation_mode=IRRADIATION_MODE,
        nci_mode=NCI_MODE,
        params=all_params,
        metrics={'note': 'Trained manually from recovered parameters'}
    )

    print(f"✓ Model saved to: {saved_path}")
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    train_model()
