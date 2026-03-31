"""
Ray Tune Hyperparameter Optimization for PyTorch Neural Networks
Clean, efficient multi-GPU support with proper GroupKFold integration
"""

import os
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.execution.placement_groups import PlacementGroupFactory
from optuna.samplers import TPESampler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import torch
import pickle
import ray
import json

def optimize_neural_net_raytune(
    X_train,
    y_train,
    groups=None,
    n_trials=10,
    n_gpus=2,
    target_type='flux',
    use_log_flux=True,
    encoding='physics',
    lattices_train=None,
    irradiation_mode='fill',
    nci_mode='separate',
    nci_disabled=False,
    nn_config=1,
    flux_mode='total',
    optimize_lambda=None,
    flux_group_index=None,
    score_metric='mse_log',
):
    """
    Optimize neural network hyperparameters using Ray Tune.

    When ``lattices_train`` is set and ``optimize_lambda`` is True (default for
    physics encoding), NCI lambda parameters are searched **jointly** with NN
    architecture (same pattern as Optuna flux): each trial regenerates features
    before GroupKFold CV.

    Parameters
    ----------
    nn_config : int
        Thesis layout 1–5 (see ``neural_net_configs.data_pipeline``).
    flux_mode : str
        e.g. ``energy_sixteen`` for 16-channel flux.
    flux_group_index : int, optional
        For nn_config 2 or 5: which flux group 0=tot, 1=th, 2=epi, 3=fast.
    score_metric : str
        ``mse_log`` (default for thesis NN) or ``mape``.
    """
    from utils.lambda_feature_regenerator import (
        LambdaFeatureRegenerator,
        get_lambda_params_for_encoding,
    )
    from neural_net_configs.data_pipeline import prepare_xy_after_lambda

    if optimize_lambda is None:
        nci_cut = int(os.environ.get('NCI_DISTANCE_CUTOFF', '0'))
        optimize_lambda = (
            encoding == 'physics'
            and lattices_train is not None
            and nci_cut != 2
        )

    feature_data_base = None
    lambda_param_names = []
    if optimize_lambda and lattices_train is not None:
        fr = LambdaFeatureRegenerator(encoding)
        feature_data_base = fr.separate_features(
            X_train, lattices_train, irradiation_mode, nci_mode
        )
        lambda_param_names = get_lambda_params_for_encoding(irradiation_mode, nci_mode)
        print(f"Joint lambda optimization: {lambda_param_names}")

    print(f"\n{'='*60}")
    print(f"RAY TUNE OPTIMIZATION FOR NEURAL NETWORK")
    print(f"{'='*60}")
    print(f"Target: {target_type.upper()}")
    print(f"nn_config: {nn_config}, flux_mode: {flux_mode}, score: {score_metric}")
    print(f"optimize_lambda: {optimize_lambda}")
    print(f"Trials: {n_trials}")
    print(f"GPUs: {n_gpus}")
    if groups is not None:
        print(f"Using GroupKFold (preventing augmentation leakage)")
        print(f"Unique configs: {len(np.unique(groups))}")
    print(f"{'='*60}\n")

    # Define search space - Simplified for faster convergence
    config_space = {
        # Architecture parameters
        "architecture_type": tune.choice([
            'rectangular', 'pyramidal', 'funnel',
            'hourglass',  # 'expanding', 'bottleneck'  # Commented out
        ]),
        "depth": tune.randint(2, 8),  # 2-7 hidden layers
        "base_width": tune.randint(50, 601),  # 50-600 neurons (base for calculating layer widths)

        # Activation - SIMPLIFIED: Just uniform activations
        "activation": tune.choice(['relu', 'elu', 'gelu']),  # Just uniform activations
        # "activation_strategy": tune.choice([  # COMMENTED OUT - using uniform only
        #     'uniform',        # Same activation for all layers
        #     'mixed',          # Alternating activations
        #     'progressive',    # Gradual transition
        #     'deep_relu'       # ReLU early, ELU deep
        # ]),
        # "primary_activation": tune.choice(['relu', 'elu', 'gelu', 'selu']),  # COMMENTED OUT

        # Training parameters
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 0.1),
        "optimizer": tune.choice(['adam', 'adamw', 'rmsprop']),
        "batch_size": tune.choice([128, 256, 512]),
        "dropout_rate": tune.uniform(0.0, 0.5),
        "use_batch_norm": True,  # FIXED at True (not optimized)

        # Fixed params
        "max_epochs": 1500,
        "patience": 50,
        "validation_fraction": 0.1,
        "verbose": False,
        "random_state": 42
    }

    if optimize_lambda and lambda_param_names:
        if 'lambda_decay' in lambda_param_names:
            config_space['lambda_decay'] = tune.uniform(0.5, 2.0)
        if 'lambda_P' in lambda_param_names:
            config_space['lambda_P'] = tune.uniform(0.5, 2.0)
            config_space['lambda_B'] = tune.uniform(0.5, 2.0)
            config_space['lambda_G'] = tune.uniform(0.5, 2.0)

    fr_ref = None
    if optimize_lambda and feature_data_base is not None:
        fr_ref = LambdaFeatureRegenerator(encoding)

    # ── Put large data into Ray object store so the trial closure stays small ──
    # (Deferred until after ray.init; see below.)
    _X_ref = _y_ref = _groups_ref = _fdb_ref = None

    def _ensure_object_refs():
        """Lazily put large arrays into the Ray object store (called after ray.init)."""
        nonlocal _X_ref, _y_ref, _groups_ref, _fdb_ref
        if _X_ref is not None:
            return
        _X_ref = ray.put(X_train)
        _y_ref = ray.put(y_train)
        _groups_ref = ray.put(groups) if groups is not None else None
        _fdb_ref = ray.put(feature_data_base) if feature_data_base is not None else None

    # Define training function — capture only lightweight refs, not raw arrays
    def train_neural_net(config):
        """Train function called by Ray Tune"""
        from ML_models.neural_net_train import PyTorchFlexibleRegressorWrapper
        from neural_net_configs.data_pipeline import prepare_xy_after_lambda as _prepare
        import torch
        import gc

        X_local = ray.get(_X_ref)
        y_local = ray.get(_y_ref)
        groups_local = ray.get(_groups_ref) if _groups_ref is not None else None
        fdb_local = ray.get(_fdb_ref) if _fdb_ref is not None else None

        activations = config['activation']

        print(f"\n{'='*60}")
        print(f"NEW TRIAL STARTING")
        print(f"{'='*60}")

        # Regenerate features with trial lambdas
        if optimize_lambda and fdb_local is not None and fr_ref is not None:
            if 'lambda_decay' in lambda_param_names:
                X_regen = fr_ref.regenerate_features(
                    fdb_local, lambda_decay=config['lambda_decay']
                )
            else:
                X_regen = fr_ref.regenerate_features(
                    fdb_local,
                    lambda_P=config['lambda_P'],
                    lambda_B=config['lambda_B'],
                    lambda_G=config['lambda_G'],
                )
        else:
            X_regen = X_local

        try:
            X_cv, y_cv, groups_cv = _prepare(
                X_regen,
                y_local,
                groups_local,
                nn_config,
                irradiation_mode,
                nci_mode,
                nci_disabled,
                flux_group_index=flux_group_index,
            )
        except Exception as e:
            print(f"  prepare_xy failed: {e}")
            tune.report({"score": float('inf'), "training_iteration": 0})
            return

        print(f"Architecture: {config['architecture_type']}, depth={config['depth']}, nn_config={nn_config}")
        print(f"CV data: X={X_cv.shape}, y={y_cv.shape}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Cross-validation with GroupKFold - CREATE NEW MODEL FOR EACH FOLD
        if groups_cv is not None:
            cv = GroupKFold(n_splits=5)
            cv_scores = []

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_cv, y_cv, groups_cv)):
                print(f"  Fold {fold_idx + 1}/5...")

                X_train_fold, X_test_fold = X_cv[train_idx], X_cv[test_idx]
                y_train_fold, y_test_fold = y_cv[train_idx], y_cv[test_idx]
                groups_train_fold = groups_cv[train_idx]

                model = PyTorchFlexibleRegressorWrapper(
                    architecture_type=config["architecture_type"],
                    base_width=config["base_width"],
                    depth=config["depth"],
                    activations=activations,
                    optimizer=config["optimizer"],
                    learning_rate=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                    batch_size=config["batch_size"],
                    dropout_rate=config["dropout_rate"],
                    use_batch_norm=True,
                    max_epochs=config["max_epochs"],
                    patience=config["patience"],
                    validation_fraction=config["validation_fraction"],
                    device=device,
                    verbose=config["verbose"],
                    random_state=config["random_state"]
                )

                model.fit(X_train_fold, y_train_fold, groups=groups_train_fold)
                predictions = model.predict(X_test_fold)

                if target_type == 'flux':
                    if score_metric == 'mse_log' and use_log_flux:
                        fold_score = mean_squared_error(y_test_fold, predictions)
                    elif score_metric == 'mape' or score_metric != 'mse_log':
                        if use_log_flux:
                            y_true_orig = 10 ** y_test_fold
                            y_pred_orig = 10 ** predictions
                            fold_score = np.mean(
                                np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-10))
                            ) * 100
                        else:
                            fold_score = np.mean(
                                np.abs((y_test_fold - predictions) / (y_test_fold + 1e-10))
                            ) * 100
                    else:
                        fold_score = mean_squared_error(y_test_fold, predictions)
                    cv_scores.append(fold_score)
                else:
                    fold_score = mean_squared_error(y_test_fold, predictions.ravel())
                    cv_scores.append(fold_score)

                current_mean = float(np.mean(cv_scores))
                print(f"    ✓ Fold {fold_idx + 1} score: {cv_scores[-1]:.6f} | Running avg: {current_mean:.6f}")

                del model, X_train_fold, X_test_fold, y_train_fold, y_test_fold, predictions, groups_train_fold
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                tune.report({"score": current_mean, "training_iteration": fold_idx + 1})

            print(f"  Trial complete! Final score: {current_mean:.6f}")
        else:
            from sklearn.model_selection import KFold
            print("  WARNING: No groups — using KFold (possible augmentation leakage)")
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_cv)):
                X_train_fold, X_test_fold = X_cv[train_idx], X_cv[test_idx]
                y_train_fold, y_test_fold = y_cv[train_idx], y_cv[test_idx]
                model = PyTorchFlexibleRegressorWrapper(
                    architecture_type=config["architecture_type"],
                    base_width=config["base_width"],
                    depth=config["depth"],
                    activations=activations,
                    optimizer=config["optimizer"],
                    learning_rate=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                    batch_size=config["batch_size"],
                    dropout_rate=config["dropout_rate"],
                    use_batch_norm=True,
                    max_epochs=config["max_epochs"],
                    patience=config["patience"],
                    validation_fraction=config["validation_fraction"],
                    device=device,
                    verbose=config["verbose"],
                    random_state=config["random_state"]
                )
                model.fit(X_train_fold, y_train_fold, groups=None)
                predictions = model.predict(X_test_fold)
                if target_type == 'flux' and score_metric == 'mse_log' and use_log_flux:
                    fold_score = mean_squared_error(y_test_fold, predictions)
                elif target_type == 'flux':
                    if use_log_flux:
                        y_true_orig = 10 ** y_test_fold
                        y_pred_orig = 10 ** predictions
                        fold_score = np.mean(
                            np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-10))) * 100
                    else:
                        fold_score = np.mean(
                            np.abs((y_test_fold - predictions) / (y_test_fold + 1e-10))) * 100
                else:
                    fold_score = mean_squared_error(y_test_fold, predictions.ravel())
                cv_scores.append(fold_score)
                current_mean = float(np.mean(cv_scores))
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                tune.report({"score": current_mean, "training_iteration": fold_idx + 1})

    # ASHA scheduler - RELAXED for maximum GPU flooding
    # More lenient settings to keep 192 trials running
    scheduler = ASHAScheduler(
        max_t=5,           # 5 CV folds per trial
        grace_period=2,    # Must complete 3 folds before stopping (was 1)
        reduction_factor=2 # Top 75% proceed (was 2 = 50%)
    )

    # Intelligent search algorithm (Optuna's TPE)
    # Don't pass metric/mode here - will pass to tune.run() instead
    n_startup = n_trials // 5  # 50 random trials or 1/3 of total
    search_alg = OptunaSearch(
        sampler=TPESampler(
            n_startup_trials=n_startup,  # Random exploration first
            n_ei_candidates=150,           # Candidates for intelligent selection
            seed=42
        )
    )

    # Progress reporter
    reporter = CLIReporter(
        metric_columns=["score", "training_iteration"],
        max_report_frequency=30
    )

    # Run optimization
    print("Starting Ray Tune optimization with Optuna TPE search...")
    print(f"Resources: {n_gpus} GPUs available")
    print(f"Search strategy: {n_startup} random trials, then intelligent TPE")

    # Detect available CPUs (SLURM-aware)
    total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK',
                     os.environ.get('SLURM_CPUS_ON_NODE', os.cpu_count() or 32)))
    max_concurrent = total_cpus  # 1 CPU per trial → concurrency = #CPUs
    gpu_fraction = n_gpus / max_concurrent  # spread GPUs evenly across trials

    # Initialize Ray explicitly to ensure proper resource allocation
    if not ray.is_initialized():
        ray.init(
            num_cpus=total_cpus,
            num_gpus=n_gpus,
            ignore_reinit_error=True,
            object_store_memory=50*1024*1024*1024,
            _memory=500*1024*1024*1024,
            _system_config={
                "automatic_object_spilling_enabled": True,
                "object_spilling_config": json.dumps({
                    "type": "filesystem",
                    "params": {"directory_path": "/tmp/ray_spill"}
                })
            }
        )

    _ensure_object_refs()

    resources_per_trial = PlacementGroupFactory([
        {"CPU": 1, "GPU": gpu_fraction}
    ])

    print(f"\nResource Allocation (PlacementGroupFactory):")
    print(f"  CPUs available: {total_cpus}")
    print(f"  CPU per trial: 1.0")
    print(f"  GPU per trial: {gpu_fraction:.5f} ({n_gpus}/{max_concurrent})")
    print(f"  Trials per GPU: ~{int(max_concurrent / n_gpus)}")
    print(f"  Max concurrent trials: {max_concurrent}")
    print(f"  Memory management: Enabled (GC + CUDA cache clearing after each fold)")
    print(f"  Object store spilling: Enabled → /tmp/ray_spill\n")

    analysis = tune.run(
        train_neural_net,
        config=config_space,
        num_samples=n_trials,
        scheduler=scheduler,
        search_alg=search_alg,
        progress_reporter=reporter,
        resources_per_trial=resources_per_trial,
        max_concurrent_trials=max_concurrent,
        metric="score",      # Needed for analysis.best_trial
        mode="min",          # Needed for analysis.best_trial
        raise_on_failed_trial=False,
        verbose=1
    )

    # Get best result (use explicit method since scheduler already has metric/mode)
    best_trial = analysis.get_best_trial(metric="score", mode="min")
    if best_trial is None:
        print("WARNING: No completed trials; returning default hyperparameters.")
        return {
            'architecture_type': 'rectangular',
            'base_width': 100,
            'depth': 2,
            'activations': 'relu',
            'learning_rate': 0.001,
            'weight_decay': 0.001,
            'optimizer': 'adam',
            'batch_size': 128,
            'dropout_rate': 0.0,
            'use_batch_norm': True,
            'max_epochs': 1500,
            'patience': 50,
            'device': None,
        }, analysis
    best_params = best_trial.config
    best_score = best_trial.last_result.get("score", float("inf"))

    print(f"\n{'='*60}")
    print(f"RAY TUNE OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best score: {best_score:.4f}")
    print(f"Best hyperparameters:")
    for param, value in best_params.items():
        if param not in ['max_epochs', 'patience', 'validation_fraction', 'verbose', 'random_state']:
            print(f"  {param}: {value}")
    print(f"{'='*60}\n")

    # Save results and create visualizations
    _save_raytune_results(analysis, target_type, n_gpus, encoding)

    # SIMPLIFIED: Use uniform activation (no complex strategies)
    best_activations = best_params['activation']

    # Return NN hyperparameters + joint lambda values for ModelTrainer
    return_params = {
        'architecture_type': best_params['architecture_type'],
        'base_width': best_params['base_width'],
        'depth': best_params['depth'],
        'activations': best_activations,  # Uniform activation for all layers
        'learning_rate': best_params['learning_rate'],
        'weight_decay': best_params['weight_decay'],
        'optimizer': best_params['optimizer'],
        'batch_size': best_params['batch_size'],
        'dropout_rate': best_params['dropout_rate'],
        'use_batch_norm': True,  # FIXED at True
        'max_epochs': best_params['max_epochs'],
        'patience': best_params['patience'],
        'device': None,  # Will auto-detect during final training
    }
    for lk in ('lambda_decay', 'lambda_P', 'lambda_B', 'lambda_G'):
        if lk in best_params:
            return_params[lk] = best_params[lk]

    return return_params, analysis


def _save_raytune_results(analysis, target_type, n_gpus, encoding='physics'):
    """Save Ray Tune analysis pickle (no visualizations)."""

    base_dir = os.path.dirname(os.path.dirname(__file__))
    outputs_dir = os.path.join(base_dir, 'outputs', 'raytune_results', target_type)
    os.makedirs(outputs_dir, exist_ok=True)

    try:
        results_file = f"raytune_{target_type}_ngpus{n_gpus}_analysis.pkl"
        results_path = os.path.join(outputs_dir, results_file)
        with open(results_path, 'wb') as f:
            pickle.dump(analysis, f)
        print(f"Results saved to: {results_path}")
    except Exception as e:
        print(f"Could not save results: {e}")

    print(f"All Ray Tune results saved to: {outputs_dir}\n")
