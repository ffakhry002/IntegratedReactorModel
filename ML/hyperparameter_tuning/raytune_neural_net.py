"""
Ray Tune Hyperparameter Optimization for PyTorch Neural Networks
Clean, efficient multi-GPU support with proper GroupKFold integration
"""

import os
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import make_scorer
import torch

def optimize_neural_net_raytune(X_train, y_train, groups=None, n_trials=100,
                                n_gpus=2, target_type='flux', use_log_flux=True):
    """
    Optimize neural network hyperparameters using Ray Tune

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    groups : np.ndarray, optional
        Group labels for GroupKFold (prevents augmentation leakage)
    n_trials : int
        Number of hyperparameter combinations to try
    n_gpus : int
        Number of GPUs to use
    target_type : str
        'flux' or 'keff'
    use_log_flux : bool
        Whether flux data is log-transformed

    Returns
    -------
    best_params : dict
        Best hyperparameters found
    analysis : ray.tune.ExperimentAnalysis
        Full Ray Tune results
    """

    print(f"\n{'='*60}")
    print(f"RAY TUNE OPTIMIZATION FOR NEURAL NETWORK")
    print(f"{'='*60}")
    print(f"Target: {target_type.upper()}")
    print(f"Trials: {n_trials}")
    print(f"GPUs: {n_gpus}")
    if groups is not None:
        print(f"Using GroupKFold (preventing augmentation leakage)")
        print(f"Unique configs: {len(np.unique(groups))}")
    print(f"{'='*60}\n")

    # Define search space
    config_space = {
        "depth": tune.randint(1, 6),  # 1-5 hidden layers
        "width": tune.randint(50, 401),  # 50-400 neurons
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 0.1),
        "activation": tune.choice(['relu', 'elu']),
        "optimizer": tune.choice(['adam', 'adamw', 'rmsprop']),
        "batch_size": tune.choice([64, 128, 256]),
        "dropout_rate": tune.uniform(0.0, 0.5),  # NEW: Dropout for regularization
        "use_batch_norm": tune.choice([True, False]),  # NEW: Batch normalization
        # Fixed params
        "max_epochs": 1500,
        "patience": 50,
        "validation_fraction": 0.1,
        "verbose": False,
        "random_state": 42
    }

    # Define training function
    def train_neural_net(config, X=X_train, y=y_train, groups=groups):
        """Train function called by Ray Tune"""
        from ML_models.neural_net_train import PyTorchRegressorWrapper

        # Use CUDA - Ray Tune handles GPU assignment via CUDA_VISIBLE_DEVICES
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create model with Ray-assigned device
        model = PyTorchRegressorWrapper(
            depth=config["depth"],
            width=config["width"],
            activation=config["activation"],
            optimizer=config["optimizer"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            batch_size=config["batch_size"],
            dropout_rate=config["dropout_rate"],
            use_batch_norm=config["use_batch_norm"],
            max_epochs=config["max_epochs"],
            patience=config["patience"],
            validation_fraction=config["validation_fraction"],
            device=device,  # Ray-assigned GPU
            verbose=config["verbose"],
            random_state=config["random_state"]
        )

        # Cross-validation with GroupKFold
        if groups is not None:
            cv = GroupKFold(n_splits=5)
            # Create custom CV that passes groups to fit()
            cv_scores = []
            for train_idx, test_idx in cv.split(X, y, groups):
                X_train_fold, X_test_fold = X[train_idx], X[test_idx]
                y_train_fold, y_test_fold = y[train_idx], y[test_idx]
                groups_train_fold = groups[train_idx]

                # Fit with groups for internal validation (zero leakage!)
                model.fit(X_train_fold, y_train_fold, groups=groups_train_fold)
                predictions = model.predict(X_test_fold)

                # Score
                if target_type == 'flux':
                    if use_log_flux:
                        y_true_orig = 10 ** y_test_fold
                        y_pred_orig = 10 ** predictions
                        mape = np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-10))) * 100
                    else:
                        mape = np.mean(np.abs((y_test_fold - predictions) / (y_test_fold + 1e-10))) * 100
                    cv_scores.append(mape)
                else:
                    from sklearn.metrics import mean_squared_error
                    mse = mean_squared_error(y_test_fold, predictions)
                    cv_scores.append(mse)

            scores = np.array(cv_scores)
        else:
            # No groups - use standard CV
            cv = 5
            if target_type == 'flux':
                def mape_scorer(y_true, y_pred):
                    if use_log_flux:
                        y_true_orig = 10 ** y_true
                        y_pred_orig = 10 ** y_pred
                        mape = np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-10))) * 100
                    else:
                        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
                    return -mape
                scorer = make_scorer(mape_scorer, greater_is_better=True)
            else:
                scorer = 'neg_mean_squared_error'

            scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=1)

        # Report to Ray Tune (positive values - scheduler minimizes)
        mean_score = float(np.mean(scores))
        tune.report(score=mean_score)  # MAPE or MSE (minimize both)

    # ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="score",
        mode="min",
        max_t=n_trials,
        grace_period=10,
        reduction_factor=2
    )

    # Progress reporter
    reporter = CLIReporter(
        metric_columns=["score", "training_iteration"],
        max_report_frequency=30
    )

    # Run optimization
    print("Starting Ray Tune optimization...")
    analysis = tune.run(
        train_neural_net,
        config=config_space,
        num_samples=n_trials,
        scheduler=scheduler,
        progress_reporter=reporter,
        resources_per_trial={"cpu": 1, "gpu": 1/n_gpus},  # Share GPUs across trials
        raise_on_failed_trial=False,
        verbose=1
    )

    # Get best result
    best_trial = analysis.best_trial
    best_params = best_trial.config
    best_score = best_trial.last_result["score"]

    print(f"\n{'='*60}")
    print(f"RAY TUNE OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best score: {best_score:.4f}")
    print(f"Best hyperparameters:")
    for param, value in best_params.items():
        if param not in ['max_epochs', 'patience', 'validation_fraction', 'verbose', 'random_state']:
            print(f"  {param}: {value}")
    print(f"{'='*60}\n")

    # Return only the hyperparameters (not fixed params)
    return_params = {
        'depth': best_params['depth'],
        'width': best_params['width'],
        'learning_rate': best_params['learning_rate'],
        'weight_decay': best_params['weight_decay'],
        'activation': best_params['activation'],
        'optimizer': best_params['optimizer'],
        'batch_size': best_params['batch_size'],
        'dropout_rate': best_params['dropout_rate'],
        'use_batch_norm': best_params['use_batch_norm'],
        'max_epochs': best_params['max_epochs'],
        'patience': best_params['patience'],
        'device': None  # Will auto-detect during final training
    }

    return return_params, analysis
