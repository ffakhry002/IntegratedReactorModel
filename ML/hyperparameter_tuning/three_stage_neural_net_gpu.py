"""
Three-Stage Neural Network Optimization with Aggressive GPU Utilization
========================================================================
Optimized for: 48 CPUs, 4 GPUs
Strategy: 96 parallel processes (2 per CPU) flooding 4 GPUs

Stages:
1. Random Search  - 96 parallel workers flooding GPUs
2. Grid Search    - 96 parallel workers flooding GPUs
3. Bayesian Opt   - Intelligent search with GPU batching
"""

import numpy as np
import torch
import torch.multiprocessing as mp
from multiprocessing import Manager, Queue, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
import os
import pickle
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import itertools

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class GPUOptimizationConfig:
    """Configuration for GPU-optimized three-stage optimization"""
    n_cpus: int = 48
    n_gpus: int = 4
    processes_per_cpu: int = 2
    n_parallel_processes: int = 96  # 48 * 2

    # Stage 1: Random Search
    random_n_iter: int = 2000

    # Stage 2: Grid Search (will be calculated based on param grid)
    grid_n_values_per_param: int = 5

    # Stage 3: Bayesian Optimization
    bayesian_n_iter: int = 100
    bayesian_batch_size: int = 16  # Evaluate 16 configs in parallel on GPUs

    # Cross-validation
    n_cv_folds: int = 5

    # Timeout settings
    single_trial_timeout: int = 600  # 10 minutes per trial
    stage_timeout: int = 7200  # 2 hours per stage

    # Fixed parameters for neural network
    max_epochs: int = 1500
    patience: int = 50
    validation_fraction: float = 0.1
    verbose: bool = False
    random_state: int = 42

config = GPUOptimizationConfig()

# ============================================================================
# PARAMETER SPACE DEFINITIONS
# ============================================================================
class NeuralNetParameterSpace:
    """Define search spaces for neural network hyperparameters"""

    @staticmethod
    def get_random_space():
        """Random search space - broad exploration"""
        return {
            'architecture_type': ['rectangular', 'pyramidal', 'funnel', 'hourglass', 'expanding', 'bottleneck'],
            'depth': list(range(2, 7)),  # 2-6 hidden layers
            'base_width': list(range(50, 451, 10)),  # 50-450 in steps of 10
            'activation_strategy': ['uniform', 'mixed', 'progressive', 'deep_relu'],
            'primary_activation': ['relu', 'elu', 'gelu', 'selu'],
            'learning_rate': (1e-4, 1e-2, 'log-uniform'),
            'weight_decay': (1e-5, 0.1, 'log-uniform'),
            'optimizer': ['adam', 'adamw', 'rmsprop'],
            'batch_size': [64, 128, 256],
            'dropout_rate': (0.0, 0.5, 'uniform'),
            'use_batch_norm': [True, False]
        }

    @staticmethod
    def create_grid_space(best_params):
        """Grid search space - focused around best random search params"""
        grid = {}

        # Architecture parameters - keep best, add nearby variations
        grid['architecture_type'] = [best_params['architecture_type']]

        depth = best_params['depth']
        grid['depth'] = sorted(list(set([max(2, depth-1), depth, min(6, depth+1)])))

        base_width = best_params['base_width']
        width_variations = [
            max(50, base_width - 50),
            max(50, base_width - 25),
            base_width,
            min(450, base_width + 25),
            min(450, base_width + 50)
        ]
        grid['base_width'] = sorted(list(set(width_variations)))

        # Keep best activation strategies
        grid['activation_strategy'] = [best_params['activation_strategy']]
        grid['primary_activation'] = [best_params['primary_activation']]

        # Continuous parameters - ±20% around best
        lr = best_params['learning_rate']
        grid['learning_rate'] = [lr * f for f in [0.5, 0.75, 1.0, 1.5, 2.0]]

        wd = best_params['weight_decay']
        grid['weight_decay'] = [wd * f for f in [0.5, 0.75, 1.0, 1.5, 2.0]]

        # Categorical - keep best
        grid['optimizer'] = [best_params['optimizer']]
        grid['batch_size'] = [best_params['batch_size']]

        # Dropout - fine-tune
        dr = best_params['dropout_rate']
        dropout_variations = [max(0.0, dr - 0.1), dr, min(0.5, dr + 0.1)]
        grid['dropout_rate'] = sorted(list(set(dropout_variations)))

        grid['use_batch_norm'] = [best_params['use_batch_norm']]

        return grid

    @staticmethod
    def create_bayesian_space(best_params):
        """Bayesian search space - tight optimization around best grid params"""
        space = []

        # Architecture - very focused
        depth = best_params['depth']
        space.append(Integer(max(2, depth-1), min(6, depth+1), name='depth'))

        base_width = best_params['base_width']
        space.append(Integer(
            max(50, int(base_width * 0.8)),
            min(450, int(base_width * 1.2)),
            name='base_width'
        ))

        # Continuous parameters - narrow ±20% range
        lr = best_params['learning_rate']
        space.append(Real(lr * 0.8, lr * 1.2, prior='log-uniform', name='learning_rate'))

        wd = best_params['weight_decay']
        space.append(Real(wd * 0.8, wd * 1.2, prior='log-uniform', name='weight_decay'))

        dr = best_params['dropout_rate']
        space.append(Real(max(0.0, dr - 0.1), min(0.5, dr + 0.1), name='dropout_rate'))

        # Categorical - keep best from grid
        space.append(Categorical([best_params['architecture_type']], name='architecture_type'))
        space.append(Categorical([best_params['activation_strategy']], name='activation_strategy'))
        space.append(Categorical([best_params['primary_activation']], name='primary_activation'))
        space.append(Categorical([best_params['optimizer']], name='optimizer'))
        space.append(Categorical([best_params['batch_size']], name='batch_size'))
        space.append(Categorical([best_params['use_batch_norm']], name='use_batch_norm'))

        return space

# ============================================================================
# GPU WORKER FUNCTIONS
# ============================================================================
def worker_evaluate_config(args):
    """
    Worker function to evaluate a single configuration on assigned GPU
    This is the core function that runs in each of the 96 parallel processes
    """
    config_params, X_train, y_train, groups, target_type, use_log_flux, gpu_id, trial_id = args

    try:
        # Set GPU for this worker
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Import here to avoid serialization issues
        from ML_models.neural_net_train import PyTorchFlexibleRegressorWrapper
        from ML_models.neural_architectures import create_heterogeneous_activations

        # Determine activation configuration
        if config_params['activation_strategy'] == 'uniform':
            activations = config_params['primary_activation']
        else:
            activations = create_heterogeneous_activations(
                config_params['depth'],
                pattern=config_params['activation_strategy']
            )
            if config_params['primary_activation'] != 'relu' and config_params['activation_strategy'] != 'progressive':
                activations = [config_params['primary_activation'] if act == 'relu' else act
                             for act in activations]

        # Create model
        model = PyTorchFlexibleRegressorWrapper(
            architecture_type=config_params['architecture_type'],
            base_width=config_params['base_width'],
            depth=config_params['depth'],
            activations=activations,
            optimizer=config_params['optimizer'],
            learning_rate=config_params['learning_rate'],
            weight_decay=config_params['weight_decay'],
            batch_size=config_params['batch_size'],
            dropout_rate=config_params['dropout_rate'],
            use_batch_norm=config_params['use_batch_norm'],
            max_epochs=config.max_epochs,
            patience=config.patience,
            validation_fraction=config.validation_fraction,
            device=device,
            verbose=config.verbose,
            random_state=config.random_state
        )

        # Cross-validation
        cv = GroupKFold(n_splits=config.n_cv_folds)
        cv_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train, groups)):
            X_train_fold = X_train[train_idx]
            X_test_fold = X_train[test_idx]
            y_train_fold = y_train[train_idx]
            y_test_fold = y_train[test_idx]
            groups_train_fold = groups[train_idx]

            # Fit model
            model.fit(X_train_fold, y_train_fold, groups=groups_train_fold)
            predictions = model.predict(X_test_fold)

            # Calculate score
            if target_type == 'flux':
                if use_log_flux:
                    y_true_orig = 10 ** y_test_fold
                    y_pred_orig = 10 ** predictions
                    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-10))) * 100
                else:
                    mape = np.mean(np.abs((y_test_fold - predictions) / (y_test_fold + 1e-10))) * 100
                cv_scores.append(mape)
            else:
                mse = mean_squared_error(y_test_fold, predictions)
                cv_scores.append(mse)

        mean_score = float(np.mean(cv_scores))
        std_score = float(np.std(cv_scores))

        return {
            'trial_id': trial_id,
            'params': config_params,
            'score': mean_score,
            'std': std_score,
            'success': True,
            'gpu_id': gpu_id
        }

    except Exception as e:
        return {
            'trial_id': trial_id,
            'params': config_params,
            'score': float('inf'),
            'std': 0.0,
            'success': False,
            'error': str(e),
            'gpu_id': gpu_id
        }

def sample_random_config(param_space, trial_id):
    """Sample a random configuration from parameter space"""
    config = {'trial_id': trial_id}

    for param_name, param_range in param_space.items():
        if isinstance(param_range, list):
            # Categorical
            config[param_name] = np.random.choice(param_range)
        elif isinstance(param_range, tuple) and len(param_range) == 3:
            # Continuous with distribution
            low, high, dist = param_range
            if dist == 'log-uniform':
                config[param_name] = float(np.exp(np.random.uniform(np.log(low), np.log(high))))
            else:  # uniform
                config[param_name] = float(np.random.uniform(low, high))
        else:
            raise ValueError(f"Unknown parameter range format for {param_name}")

    return config

# ============================================================================
# STAGE 1: RANDOM SEARCH WITH GPU FLOODING
# ============================================================================
def random_search_stage(X_train, y_train, groups, target_type, use_log_flux):
    """
    Stage 1: Random Search
    Uses 96 parallel processes to flood 4 GPUs with evaluations
    """
    print(f"\n{'='*70}")
    print(f"STAGE 1/3: RANDOM SEARCH - GPU FLOODING")
    print(f"{'='*70}")
    print(f"Workers: {config.n_parallel_processes} parallel processes")
    print(f"GPUs: {config.n_gpus}")
    print(f"Trials: {config.random_n_iter}")
    print(f"GPU load: ~{config.n_parallel_processes // config.n_gpus} processes per GPU")
    print(f"{'='*70}\n")

    stage_start = time.time()
    param_space = NeuralNetParameterSpace.get_random_space()

    # Generate all random configurations
    print("Generating random configurations...")
    configs = [sample_random_config(param_space, i) for i in range(config.random_n_iter)]

    # Prepare arguments for parallel evaluation
    # Round-robin GPU assignment
    args_list = []
    for i, cfg in enumerate(configs):
        gpu_id = i % config.n_gpus
        args_list.append((cfg, X_train, y_train, groups, target_type, use_log_flux, gpu_id, i))

    # Parallel evaluation with progress tracking
    print(f"\nEvaluating {len(configs)} configurations across {config.n_gpus} GPUs...")
    print(f"Starting {config.n_parallel_processes} parallel workers...\n")

    results = []
    completed = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=config.n_parallel_processes) as executor:
        futures = {executor.submit(worker_evaluate_config, args): i for i, args in enumerate(args_list)}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (len(configs) - completed) / rate if rate > 0 else 0

            if completed % 10 == 0 or completed == len(configs):
                print(f"  Progress: {completed}/{len(configs)} "
                      f"({100*completed/len(configs):.1f}%) | "
                      f"Rate: {rate:.2f} trials/sec | "
                      f"ETA: {remaining/60:.1f} min | "
                      f"Best: {min(r['score'] for r in results):.4f}")

    stage_time = time.time() - stage_start

    # Find best result
    successful_results = [r for r in results if r['success']]
    if not successful_results:
        raise RuntimeError("All random search trials failed!")

    best_result = min(successful_results, key=lambda x: x['score'])

    print(f"\n{'='*70}")
    print(f"STAGE 1 COMPLETE - Time: {stage_time/60:.2f} minutes")
    print(f"{'='*70}")
    print(f"Successful trials: {len(successful_results)}/{len(results)}")
    print(f"Best score: {best_result['score']:.4f}")
    print(f"Best parameters:")
    for key, value in best_result['params'].items():
        if key != 'trial_id':
            print(f"  {key}: {value}")
    print(f"{'='*70}\n")

    return best_result['params'], best_result['score'], results, stage_time

# ============================================================================
# STAGE 2: GRID SEARCH WITH GPU FLOODING
# ============================================================================
def grid_search_stage(X_train, y_train, groups, target_type, use_log_flux, best_random_params):
    """
    Stage 2: Grid Search
    Focused grid around best random search result, flooding GPUs
    """
    print(f"\n{'='*70}")
    print(f"STAGE 2/3: GRID SEARCH - FOCUSED REFINEMENT")
    print(f"{'='*70}")

    stage_start = time.time()
    param_grid = NeuralNetParameterSpace.create_grid_space(best_random_params)

    # Calculate total combinations
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)

    print(f"Grid size: {total_combinations} combinations")
    print(f"Workers: {config.n_parallel_processes} parallel processes")
    print(f"GPUs: {config.n_gpus}")
    print(f"{'='*70}\n")

    # Generate all grid configurations
    print("Generating grid configurations...")
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    configs = []
    for i, combination in enumerate(itertools.product(*param_values)):
        cfg = dict(zip(param_names, combination))
        cfg['trial_id'] = i
        configs.append(cfg)

    # Prepare arguments for parallel evaluation
    args_list = []
    for i, cfg in enumerate(configs):
        gpu_id = i % config.n_gpus
        args_list.append((cfg, X_train, y_train, groups, target_type, use_log_flux, gpu_id, i))

    # Parallel evaluation
    print(f"\nEvaluating {len(configs)} configurations...\n")

    results = []
    completed = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=config.n_parallel_processes) as executor:
        futures = {executor.submit(worker_evaluate_config, args): i for i, args in enumerate(args_list)}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (len(configs) - completed) / rate if rate > 0 else 0

            if completed % 5 == 0 or completed == len(configs):
                print(f"  Progress: {completed}/{len(configs)} "
                      f"({100*completed/len(configs):.1f}%) | "
                      f"Rate: {rate:.2f} trials/sec | "
                      f"ETA: {remaining/60:.1f} min | "
                      f"Best: {min(r['score'] for r in results):.4f}")

    stage_time = time.time() - stage_start

    # Find best result
    successful_results = [r for r in results if r['success']]
    if not successful_results:
        print("WARNING: All grid search trials failed, using random search result")
        return best_random_params, float('inf'), [], stage_time

    best_result = min(successful_results, key=lambda x: x['score'])

    print(f"\n{'='*70}")
    print(f"STAGE 2 COMPLETE - Time: {stage_time/60:.2f} minutes")
    print(f"{'='*70}")
    print(f"Successful trials: {len(successful_results)}/{len(results)}")
    print(f"Best score: {best_result['score']:.4f}")
    print(f"Best parameters:")
    for key, value in best_result['params'].items():
        if key != 'trial_id':
            print(f"  {key}: {value}")
    print(f"{'='*70}\n")

    return best_result['params'], best_result['score'], results, stage_time

# ============================================================================
# STAGE 3: BAYESIAN OPTIMIZATION WITH GPU BATCHING
# ============================================================================
def bayesian_optimization_stage(X_train, y_train, groups, target_type, use_log_flux, best_grid_params):
    """
    Stage 3: Bayesian Optimization
    Intelligent search with batched GPU evaluation
    """
    print(f"\n{'='*70}")
    print(f"STAGE 3/3: BAYESIAN OPTIMIZATION - INTELLIGENT SEARCH")
    print(f"{'='*70}")
    print(f"Iterations: {config.bayesian_n_iter}")
    print(f"Batch size: {config.bayesian_batch_size} (evaluated in parallel)")
    print(f"GPUs: {config.n_gpus}")
    print(f"{'='*70}\n")

    stage_start = time.time()
    search_space = NeuralNetParameterSpace.create_bayesian_space(best_grid_params)

    # Track all evaluations
    all_params = []
    all_scores = []

    # Objective function for batched evaluation
    @use_named_args(search_space)
    def objective(**params):
        """Evaluate a single configuration"""
        # Convert params to proper types
        config_params = {
            'depth': int(params['depth']),
            'base_width': int(params['base_width']),
            'learning_rate': float(params['learning_rate']),
            'weight_decay': float(params['weight_decay']),
            'dropout_rate': float(params['dropout_rate']),
            'architecture_type': params['architecture_type'],
            'activation_strategy': params['activation_strategy'],
            'primary_activation': params['primary_activation'],
            'optimizer': params['optimizer'],
            'batch_size': int(params['batch_size']),
            'use_batch_norm': bool(params['use_batch_norm'])
        }

        # Evaluate on GPU 0 (Bayesian optimization is sequential)
        args = (config_params, X_train, y_train, groups, target_type, use_log_flux, 0, len(all_params))
        result = worker_evaluate_config(args)

        all_params.append(config_params)
        all_scores.append(result['score'])

        print(f"  Trial {len(all_params)}/{config.bayesian_n_iter}: "
              f"Score = {result['score']:.4f} | "
              f"Best = {min(all_scores):.4f}")

        return result['score']

    # Run Bayesian optimization
    print("Starting Bayesian optimization...\n")

    result = gp_minimize(
        objective,
        search_space,
        n_calls=config.bayesian_n_iter,
        n_random_starts=20,  # Start with 20 random points
        random_state=config.random_state,
        verbose=False,
        n_jobs=1  # Sequential for acquisition function
    )

    stage_time = time.time() - stage_start

    # Extract best parameters
    best_params_list = result.x
    best_score = result.fun

    # Convert to dictionary format
    best_params = {}
    for i, dim in enumerate(search_space):
        best_params[dim.name] = best_params_list[i]

    # Convert to proper types
    best_params['depth'] = int(best_params['depth'])
    best_params['base_width'] = int(best_params['base_width'])
    best_params['batch_size'] = int(best_params['batch_size'])
    best_params['use_batch_norm'] = bool(best_params['use_batch_norm'])

    print(f"\n{'='*70}")
    print(f"STAGE 3 COMPLETE - Time: {stage_time/60:.2f} minutes")
    print(f"{'='*70}")
    print(f"Total evaluations: {len(all_scores)}")
    print(f"Best score: {best_score:.4f}")
    print(f"Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"{'='*70}\n")

    return best_params, best_score, all_params, stage_time

# ============================================================================
# MAIN THREE-STAGE OPTIMIZATION
# ============================================================================
def three_stage_neural_net_optimization(X_train, y_train, groups=None,
                                       target_type='flux', use_log_flux=True,
                                       save_results=True, output_dir=None):
    """
    Three-stage hyperparameter optimization for PyTorch Neural Networks
    Optimized for aggressive GPU utilization: 48 CPUs, 4 GPUs, 96 parallel processes

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    groups : np.ndarray, optional
        Group labels for GroupKFold (prevents augmentation leakage)
    target_type : str
        'flux' or 'keff'
    use_log_flux : bool
        Whether flux data is log-transformed
    save_results : bool
        Whether to save results to disk
    output_dir : str, optional
        Directory to save results

    Returns
    -------
    best_params : dict
        Best hyperparameters found
    optimization_history : dict
        Full optimization history
    """

    print(f"\n{'='*70}")
    print(f"THREE-STAGE NEURAL NETWORK OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Target: {target_type.upper()}")
    print(f"Metric: {'MAPE' if target_type == 'flux' else 'MSE'}")
    print(f"")
    print(f"System Resources:")
    print(f"  CPUs: {config.n_cpus}")
    print(f"  GPUs: {config.n_gpus}")
    print(f"  Parallel Workers: {config.n_parallel_processes}")
    print(f"  Processes per GPU: ~{config.n_parallel_processes // config.n_gpus}")
    print(f"")
    print(f"Data:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    if groups is not None:
        print(f"  Unique configs: {len(np.unique(groups))}")
        print(f"  Augmentation: {X_train.shape[0] / len(np.unique(groups)):.1f}x")
        print(f"  CV: GroupKFold (no leakage)")
    print(f"{'='*70}")

    optimization_start = time.time()

    # Validate groups
    if groups is None:
        raise ValueError("groups parameter is required for GroupKFold cross-validation")

    # Stage 1: Random Search
    best_random_params, best_random_score, random_results, stage1_time = \
        random_search_stage(X_train, y_train, groups, target_type, use_log_flux)

    # Stage 2: Grid Search
    best_grid_params, best_grid_score, grid_results, stage2_time = \
        grid_search_stage(X_train, y_train, groups, target_type, use_log_flux, best_random_params)

    # Use best from grid or random
    if best_grid_score < best_random_score:
        best_params_for_bayes = best_grid_params
        best_score_so_far = best_grid_score
    else:
        best_params_for_bayes = best_random_params
        best_score_so_far = best_random_score

    # Stage 3: Bayesian Optimization
    best_bayes_params, best_bayes_score, bayes_results, stage3_time = \
        bayesian_optimization_stage(X_train, y_train, groups, target_type, use_log_flux, best_params_for_bayes)

    # Select overall best
    if best_bayes_score < best_score_so_far:
        final_best_params = best_bayes_params
        final_best_score = best_bayes_score
        best_stage = "Bayesian"
    elif best_grid_score < best_random_score:
        final_best_params = best_grid_params
        final_best_score = best_grid_score
        best_stage = "Grid"
    else:
        final_best_params = best_random_params
        final_best_score = best_random_score
        best_stage = "Random"

    total_time = time.time() - optimization_start

    # Final summary
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"  Stage 1 (Random):   {stage1_time/60:.2f} min")
    print(f"  Stage 2 (Grid):     {stage2_time/60:.2f} min")
    print(f"  Stage 3 (Bayesian): {stage3_time/60:.2f} min")
    print(f"")
    print(f"Best stage: {best_stage}")
    print(f"Final best score: {final_best_score:.4f}")
    print(f"")
    print(f"Optimal hyperparameters:")

    # Reconstruct activations for final output
    from ML_models.neural_architectures import create_heterogeneous_activations
    if final_best_params['activation_strategy'] == 'uniform':
        final_activations = final_best_params['primary_activation']
    else:
        final_activations = create_heterogeneous_activations(
            final_best_params['depth'],
            pattern=final_best_params['activation_strategy']
        )
        if final_best_params['primary_activation'] != 'relu' and final_best_params['activation_strategy'] != 'progressive':
            final_activations = [final_best_params['primary_activation'] if act == 'relu' else act
                               for act in final_activations]

    return_params = {
        'architecture_type': final_best_params['architecture_type'],
        'base_width': final_best_params['base_width'],
        'depth': final_best_params['depth'],
        'activations': final_activations,
        'learning_rate': final_best_params['learning_rate'],
        'weight_decay': final_best_params['weight_decay'],
        'optimizer': final_best_params['optimizer'],
        'batch_size': final_best_params['batch_size'],
        'dropout_rate': final_best_params['dropout_rate'],
        'use_batch_norm': final_best_params['use_batch_norm'],
        'max_epochs': config.max_epochs,
        'patience': config.patience,
        'device': None,  # Will auto-detect during final training
        # Backward compatibility
        'width': final_best_params['base_width'],
        'activation': final_best_params['primary_activation']
    }

    for key, value in return_params.items():
        if key not in ['max_epochs', 'patience', 'device']:
            print(f"  {key}: {value}")
    print(f"{'='*70}\n")

    # Prepare optimization history
    optimization_history = {
        'target_type': target_type,
        'final_best_params': return_params,
        'final_best_score': final_best_score,
        'best_stage': best_stage,
        'stage1_time': stage1_time,
        'stage2_time': stage2_time,
        'stage3_time': stage3_time,
        'total_time': total_time,
        'random_search_results': random_results,
        'grid_search_results': grid_results,
        'bayesian_results': bayes_results,
        'config': config
    }

    # Save results
    if save_results:
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            output_dir = os.path.join(base_dir, 'outputs', 'three_stage_neural_net', target_type)

        os.makedirs(output_dir, exist_ok=True)

        # Save full history
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_file = os.path.join(output_dir, f'optimization_history_{timestamp}.pkl')
        with open(history_file, 'wb') as f:
            pickle.dump(optimization_history, f)
        print(f"Saved optimization history to: {history_file}")

        # Save best params
        params_file = os.path.join(output_dir, f'best_params_{timestamp}.pkl')
        joblib.dump(return_params, params_file)
        print(f"Saved best parameters to: {params_file}")

    return return_params, optimization_history


if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    print("Three-Stage Neural Network GPU Optimization")
    print("=" * 70)
    print("This module provides GPU-optimized hyperparameter search")
    print("Designed for: 48 CPUs, 4 GPUs, 96 parallel processes")
    print("=" * 70)
