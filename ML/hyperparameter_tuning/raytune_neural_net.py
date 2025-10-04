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
from optuna.samplers import TPESampler
import optuna
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import make_scorer
import torch
import pickle

def optimize_neural_net_raytune(X_train, y_train, groups=None, n_trials=250,
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

            # Enumerate to enable ASHA early stopping
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
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

                # Report after each fold for ASHA early stopping
                current_mean = float(np.mean(cv_scores))
                tune.report({"score": current_mean, "training_iteration": fold_idx + 1})

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
            # cross_val_score returns negative (higher is better convention)
            # Convert back to positive for consistency
            scores = -scores  # Now positive MAPE/MSE
            mean_score = float(np.mean(scores))
            tune.report({"score": mean_score})  # Dictionary format for API compatibility

    # ASHA scheduler for early stopping (kills bad trials after 1-2 folds!)
    scheduler = ASHAScheduler(
        metric="score",
        mode="min",
        max_t=5,           # 5 CV folds per trial
        grace_period=1,    # Can stop after just 1 fold if clearly bad
        reduction_factor=2 # Top 50% proceed to next fold
    )

    # Intelligent search algorithm (Optuna's TPE)
    n_startup = min(50, n_trials // 3)  # 50 random trials or 1/3 of total
    search_alg = OptunaSearch(
        sampler=TPESampler(
            n_startup_trials=n_startup,  # Random exploration first
            n_ei_candidates=50,           # Candidates for intelligent selection
            seed=42
        ),
        metric="score",
        mode="min"
    )

    # Progress reporter
    reporter = CLIReporter(
        metric_columns=["score", "training_iteration"],
        max_report_frequency=30
    )

    # Run optimization
    print("Starting Ray Tune optimization with Optuna TPE search...")
    print(f"Resources: {n_gpus} GPUs, 8 CPUs per trial")
    print(f"Parallelism: Up to {n_gpus * 2} trials simultaneously")
    print(f"Search strategy: {n_startup} random trials, then intelligent TPE\n")

    analysis = tune.run(
        train_neural_net,
        config=config_space,
        num_samples=n_trials,
        scheduler=scheduler,  # Already has metric="score", mode="min"
        search_alg=search_alg,  # Already has metric="score", mode="min"
        progress_reporter=reporter,
        resources_per_trial={"cpu": 8, "gpu": 1/n_gpus},  # 8 CPUs + shared GPU
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

    # Save results and create visualizations
    _save_raytune_results(analysis, target_type, n_gpus)

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


def _save_raytune_results(analysis, target_type, n_gpus):
    """Save Ray Tune results and create visualizations (similar to Optuna)"""

    # Create separate directory for each target type
    base_dir = os.path.dirname(os.path.dirname(__file__))
    outputs_dir = os.path.join(base_dir, 'outputs', 'raytune_results', target_type)
    os.makedirs(outputs_dir, exist_ok=True)

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

            # Save Optuna study
            study_file = f"optuna_study_{target_type}.pkl"
            study_path = os.path.join(outputs_dir, study_file)
            with open(study_path, 'wb') as f:
                pickle.dump(optuna_study, f)
            print(f"Optuna study saved to: {study_path}")

            # Create Optuna-style plots
            _create_optuna_plots(optuna_study, plots_dir, target_type)
        else:
            print("  Optuna study not accessible (using non-Optuna search)")
    except Exception as e:
        print(f"  Could not save Optuna study or create Optuna plots: {e}")

    # Convert to DataFrame for visualization
    df = analysis.dataframe()

    # Create visualizations
    plots_dir = os.path.join(outputs_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

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
