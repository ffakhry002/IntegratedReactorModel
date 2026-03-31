"""
Ray Tune Hyperparameter Optimization for PyTorch Neural Networks
Clean, efficient multi-GPU support with proper GroupKFold integration
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.execution.placement_groups import PlacementGroupFactory
from optuna.samplers import TPESampler
import optuna
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
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

    # Define training function
    def train_neural_net(config, X=X_train, y=y_train, groups=groups):
        """Train function called by Ray Tune"""
        from ML_models.neural_net_train import PyTorchFlexibleRegressorWrapper
        import torch
        import gc  # For memory management

        activations = config['activation']

        print(f"\n{'='*60}")
        print(f"NEW TRIAL STARTING")
        print(f"{'='*60}")

        # Regenerate features with trial lambdas
        if optimize_lambda and feature_data_base is not None and fr_ref is not None:
            if 'lambda_decay' in lambda_param_names:
                X_regen = fr_ref.regenerate_features(
                    feature_data_base, lambda_decay=config['lambda_decay']
                )
            else:
                X_regen = fr_ref.regenerate_features(
                    feature_data_base,
                    lambda_P=config['lambda_P'],
                    lambda_B=config['lambda_B'],
                    lambda_G=config['lambda_G'],
                )
        else:
            X_regen = X_train

        try:
            X_cv, y_cv, groups_cv = prepare_xy_after_lambda(
                X_regen,
                y_train,
                groups,
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

    # Initialize Ray explicitly to ensure proper resource allocation
    if not ray.is_initialized():
        ray.init(
            num_cpus=32,
            num_gpus=n_gpus,
            ignore_reinit_error=True,
            object_store_memory=50*1024*1024*1024,  # 50 GB object store
            _memory=500*1024*1024*1024,  # Reserve 500 GB for workers
            _system_config={
                "automatic_object_spilling_enabled": True,  # Spill to disk if RAM full
                "object_spilling_config": json.dumps({
                    "type": "filesystem",
                    "params": {"directory_path": "/tmp/ray_spill"}
                })
            }
        )

    # Solution 3: Use PlacementGroupFactory for fractional resource allocation
    # This allows Ray Tune to properly pack multiple trials on same GPU
    resources_per_trial = PlacementGroupFactory([
        {"CPU": 1, "GPU": 3/32}  # 1 CPU, 3/32 GPU per trial
    ])

    print(f"\nResource Allocation (PlacementGroupFactory):")
    print(f"  CPU per trial: 1.0")
    print(f"  GPU per trial: {3/32:.5f} (3/32)")
    print(f"  Expected trials per GPU: ~{int(32/3)} trials")
    print(f"  Max concurrent trials: 32")
    print(f"  Memory management: Enabled (GC + CUDA cache clearing after each fold)")
    print(f"  Object store spilling: Enabled → /tmp/ray_spill\n")

    analysis = tune.run(
        train_neural_net,
        config=config_space,
        num_samples=n_trials,
        scheduler=scheduler,
        search_alg=search_alg,
        progress_reporter=reporter,
        resources_per_trial=resources_per_trial,  # Use PlacementGroupFactory
        max_concurrent_trials=32,
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
    """Save Ray Tune results and create visualizations (similar to Optuna)"""

    # Create separate directory for each target type
    base_dir = os.path.dirname(os.path.dirname(__file__))
    outputs_dir = os.path.join(base_dir, 'outputs', 'raytune_results', target_type)
    plots_dir = os.path.join(outputs_dir, 'plots')
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Save analysis object
    try:
        results_file = f"raytune_{target_type}_ngpus{n_gpus}_analysis.pkl"
        results_path = os.path.join(outputs_dir, results_file)
        with open(results_path, 'wb') as f:
            pickle.dump(analysis, f)
        print(f"Results saved to: {results_path}")
    except Exception as e:
        print(f"Could not save results: {e}")

    # Try to get underlying Optuna study for native Optuna visualizations
    try:
        if hasattr(analysis, 'search_alg') and hasattr(analysis.search_alg, '_ot_study'):
            optuna_study = analysis.search_alg._ot_study

            # Save Optuna study to CONSISTENT location (same as other models)
            optuna_studies_dir = os.path.join(base_dir, 'outputs', 'optuna_studies')
            os.makedirs(optuna_studies_dir, exist_ok=True)

            # Use consistent naming: neural_net_raytune_{target}_{encoding}_study.pkl
            study_filename = f"neural_net_raytune_{target_type}_{encoding}_study.pkl"
            study_path = os.path.join(optuna_studies_dir, study_filename)

            import joblib
            joblib.dump(optuna_study, study_path)
            print(f"\nOptuna study saved to: {study_path}")
            print(f"You can load it later for visualization using:")
            print(f"  study = joblib.load('{study_path}')")

            # Create Optuna-style plots
            _create_optuna_plots(optuna_study, plots_dir, target_type)
        else:
            print("  Optuna study not accessible (using non-Optuna search)")
    except Exception as e:
        print(f"  Could not save Optuna study or create Optuna plots: {e}")

    # Convert to DataFrame for visualization
    df = analysis.dataframe()

    # 1. Convergence Plot
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(df['score'].values, marker='o', alpha=0.6, markersize=4)
        plt.xlabel('Trial Number', fontsize=12)
        plt.ylabel(f'Score ({"MAPE %" if target_type == "flux" else "MSE"})', fontsize=12)
        plt.title(f'Ray Tune Optimization Convergence - {target_type.upper()}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'convergence_{target_type}.png'), dpi=150)
        plt.close()
        print(f"  Saved: convergence_{target_type}.png")
    except Exception as e:
        print(f"  Could not create convergence plot: {e}")

    # 2. Best Score Progress
    try:
        best_so_far = []
        current_best = float('inf')
        for score in df['score'].values:
            if score < current_best:
                current_best = score
            best_so_far.append(current_best)

        plt.figure(figsize=(10, 6))
        plt.plot(best_so_far, linewidth=2, color='#2E86AB')
        plt.xlabel('Trial Number', fontsize=12)
        plt.ylabel(f'Best Score So Far', fontsize=12)
        plt.title(f'Best Score Progress - {target_type.upper()}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'best_progress_{target_type}.png'), dpi=150)
        plt.close()
        print(f"  Saved: best_progress_{target_type}.png")
    except Exception as e:
        print(f"  Could not create progress plot: {e}")

    # 3. Parameter Importance (scatter plots)
    try:
        params_to_plot = ['depth', 'width', 'learning_rate', 'weight_decay', 'dropout_rate']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, param in enumerate(params_to_plot):
            if param in df.columns:
                axes[i].scatter(df[param], df['score'], alpha=0.5, s=20)
                axes[i].set_xlabel(param, fontsize=10)
                axes[i].set_ylabel('Score', fontsize=10)
                axes[i].set_title(f'{param} vs Score', fontsize=11)
                axes[i].grid(True, alpha=0.3)

        # Hide unused subplot
        axes[5].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'parameter_importance_{target_type}.png'), dpi=150)
        plt.close()
        print(f"  Saved: parameter_importance_{target_type}.png")
    except Exception as e:
        print(f"  Could not create parameter importance plot: {e}")

    # 4. Categorical Parameter Performance
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Activation
        if 'activation' in df.columns:
            activation_scores = df.groupby('activation')['score'].mean().sort_values()
            axes[0].bar(range(len(activation_scores)), activation_scores.values)
            axes[0].set_xticks(range(len(activation_scores)))
            axes[0].set_xticklabels(activation_scores.index, rotation=45)
            axes[0].set_title('Activation Function', fontsize=12)
            axes[0].set_ylabel('Mean Score', fontsize=10)
            axes[0].grid(True, alpha=0.3, axis='y')

        # Optimizer
        if 'optimizer' in df.columns:
            optimizer_scores = df.groupby('optimizer')['score'].mean().sort_values()
            axes[1].bar(range(len(optimizer_scores)), optimizer_scores.values)
            axes[1].set_xticks(range(len(optimizer_scores)))
            axes[1].set_xticklabels(optimizer_scores.index, rotation=45)
            axes[1].set_title('Optimizer', fontsize=12)
            axes[1].set_ylabel('Mean Score', fontsize=10)
            axes[1].grid(True, alpha=0.3, axis='y')

        # Batch Normalization
        if 'use_batch_norm' in df.columns:
            batchnorm_scores = df.groupby('use_batch_norm')['score'].mean()
            labels = ['False', 'True']
            axes[2].bar(range(len(batchnorm_scores)), batchnorm_scores.values)
            axes[2].set_xticks(range(len(batchnorm_scores)))
            axes[2].set_xticklabels(labels)
            axes[2].set_title('Batch Normalization', fontsize=12)
            axes[2].set_ylabel('Mean Score', fontsize=10)
            axes[2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'categorical_params_{target_type}.png'), dpi=150)
        plt.close()
        print(f"  Saved: categorical_params_{target_type}.png")
    except Exception as e:
        print(f"  Could not create categorical params plot: {e}")

    # 5. Save top 10 trials to text file
    try:
        top_10 = df.nsmallest(10, 'score')
        summary_file = os.path.join(outputs_dir, f'top10_{target_type}.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Top 10 Hyperparameter Combinations - {target_type.upper()}\n")
            f.write("="*80 + "\n\n")
            for rank, (idx, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"Rank {rank}:\n")
                f.write(f"  Score: {row['score']:.4f}\n")
                f.write(f"  depth={row['depth']}, width={row['width']}\n")
                f.write(f"  learning_rate={row['learning_rate']:.6f}, weight_decay={row['weight_decay']:.6f}\n")
                f.write(f"  activation={row['activation']}, optimizer={row['optimizer']}\n")
                f.write(f"  batch_size={row['batch_size']}, dropout={row['dropout_rate']:.3f}\n")
                f.write(f"  batch_norm={row['use_batch_norm']}\n\n")
        print(f"  Saved: top10_{target_type}.txt")
    except Exception as e:
        print(f"  Could not save top 10 summary: {e}")

    print(f"\nAll Ray Tune results saved to: {outputs_dir}\n")


def _create_optuna_plots(study, plots_dir, target_type):
    """Create Optuna's native visualization plots"""

    print("\nCreating Optuna-style visualizations...")

    # 1. Optimization History
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(os.path.join(plots_dir, f'optuna_history_{target_type}.png'))
        print(f"  Saved: optuna_history_{target_type}.png")
    except Exception as e:
        print(f"  Could not create optimization history: {e}")

    # 2. Parameter Importances
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(os.path.join(plots_dir, f'optuna_importances_{target_type}.png'))
        print(f"  Saved: optuna_importances_{target_type}.png")
    except Exception as e:
        print(f"  Could not create parameter importances: {e}")

    # 3. Parallel Coordinate Plot
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(os.path.join(plots_dir, f'optuna_parallel_{target_type}.png'))
        print(f"  Saved: optuna_parallel_{target_type}.png")
    except Exception as e:
        print(f"  Could not create parallel coordinate: {e}")

    # 4. Slice Plot
    try:
        fig = optuna.visualization.plot_slice(study)
        fig.write_image(os.path.join(plots_dir, f'optuna_slice_{target_type}.png'))
        print(f"  Saved: optuna_slice_{target_type}.png")
    except Exception as e:
        print(f"  Could not create slice plot: {e}")

    # 5. Contour Plot (for key parameter pairs)
    try:
        fig = optuna.visualization.plot_contour(study, params=['depth', 'width'])
        fig.write_image(os.path.join(plots_dir, f'optuna_contour_depth_width_{target_type}.png'))
        print(f"  Saved: optuna_contour_depth_width_{target_type}.png")
    except Exception as e:
        print(f"  Could not create contour plot: {e}")

    print("  Optuna visualizations complete!")
