import optuna
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import cross_val_score
import joblib
from concurrent.futures import ProcessPoolExecutor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
import time
from datetime import datetime
import warnings
import gc
import os
import signal
from contextlib import contextmanager

# Global timeout settings
TRIAL_TIMEOUT = 600*3  # 30 minutes per trial
TOTAL_TIMEOUT = 30*60*60  # 30 hours total per model

@contextmanager
def timeout(duration):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {duration} seconds")

    # Set the signal handler and a duration-second alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

def optimize_flux_model(X_train, y_flux_train, model_type='xgboost', n_trials=250, n_jobs=10):
    """Optimize hyperparameters for flux prediction only"""

    print(f"\n{'='*60}")
    print(f"Starting {model_type.upper()} optimization for FLUX")
    print(f"Total trials: {n_trials}, Timeout per trial: {TRIAL_TIMEOUT}s")
    print(f"Total timeout: {TOTAL_TIMEOUT}s")
    print(f"{'='*60}\n")

    start_time = time.time()
    completed_trials = 0

    def objective(trial):
        nonlocal completed_trials
        trial_start = time.time()

        # Check if we've exceeded total timeout
        if time.time() - start_time > TOTAL_TIMEOUT:
            print(f"\n[WARNING] Total timeout reached. Stopping optimization.")
            trial.study.stop()
            return float('inf')

        print(f"\n[Trial {trial.number + 1}/{n_trials}] Starting at {datetime.now().strftime('%H:%M:%S')}")

        try:
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'n_jobs': 1,
                    'verbosity': 2  # High verbosity for XGBoost
                }
                print(f"  XGBoost params: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")
                model = MultiOutputRegressor(xgb.XGBRegressor(**params))

            elif model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 5, 50),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
                    'n_jobs': 1,
                    'verbose': 1  # Verbose output for RF
                }
                print(f"  RF params: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")
                model = RandomForestRegressor(**params)

            elif model_type == 'svm':
                params = {
                    'C': trial.suggest_float('C', 0.001, 1000, log=True),
                    'gamma': trial.suggest_float('gamma', 0.0001, 1, log=True),
                    'epsilon': trial.suggest_float('epsilon', 0.001, 1.0, log=True),
                    'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
                    'verbose': True,  # Verbose output for SVM
                    'max_iter': 10000  # Reasonable max iterations
                }
                if params['kernel'] == 'poly':
                    params['degree'] = trial.suggest_int('degree', 2, 5)
                print(f"  SVM params: C={params['C']:.4f}, gamma={params['gamma']:.6f}, kernel={params['kernel']}")
                model = MultiOutputRegressor(SVR(**params))

            elif model_type == 'neural_net':
                n_layers = trial.suggest_int('n_layers', 1, 5)
                layers = []
                for i in range(n_layers):
                    layers.append(trial.suggest_int(f'layer_{i}_size', 50, 400))

                params = {
                    'hidden_layer_sizes': tuple(layers),
                    'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.01, log=True),
                    'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
                    'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                    'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
                    'max_iter': 500,
                    'verbose': True,  # Verbose output for NN
                    'early_stopping': True,  # Enable early stopping
                    'n_iter_no_change': 10
                }
                print(f"  NN params: layers={params['hidden_layer_sizes']}, solver={params['solver']}")
                model = MLPRegressor(**params)

            # Train and evaluate with CV - keeping all original functionality
            print(f"  Starting cross-validation...")
            cv_scores = []

            # Set environment variable to limit thread usage (helps with deadlocks)
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'

            for fold_idx in range(10):
                fold_start = time.time()

                # Check trial timeout
                if time.time() - trial_start > TRIAL_TIMEOUT:
                    print(f"  [TIMEOUT] Trial exceeded {TRIAL_TIMEOUT}s limit")
                    if cv_scores:  # If we have some scores, use their mean
                        return -np.mean(cv_scores)
                    else:
                        return float('inf')

                print(f"    Fold {fold_idx + 1}/10...", end='', flush=True)

                # Suppress warnings during CV
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    try:
                        score = cross_val_score(model, X_train, y_flux_train, cv=10,
                                              scoring='neg_mean_squared_error', n_jobs=1)
                        cv_scores.extend(score)
                        print(f" done ({time.time() - fold_start:.1f}s)")
                        break  # We got all scores at once
                    except Exception as e:
                        print(f" error: {str(e)[:50]}")
                        continue

            if not cv_scores:
                print(f"  No valid CV scores obtained")
                return float('inf')

            final_score = -np.mean(cv_scores)
            print(f"  Trial {trial.number} score: {final_score:.6f}")
            print(f"  Trial time: {time.time() - trial_start:.1f}s")

            # Increment completed trials
            completed_trials += 1

            # Force garbage collection to free memory
            del model
            gc.collect()

            return final_score

        except TimeoutError:
            print(f"  [TIMEOUT] Trial {trial.number} timed out")
            completed_trials += 1
            return float('inf')
        except Exception as e:
            print(f"  [ERROR] Trial {trial.number} failed: {str(e)[:100]}")
            completed_trials += 1
            return float('inf')

    # Create study with proper exception handling
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(n_startup_trials=30, seed=42),
        pruner=None  # Disable pruning to avoid complexity
    )

    # Add callback to print best value updates
    def callback(study, trial):
        if study.best_trial.number == trial.number:
            print(f"\n[NEW BEST] Trial {trial.number}: {study.best_value:.6f}")
            print(f"Parameters: {trial.params}\n")

    try:
        # Use catch to prevent hanging
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
            callbacks=[callback],
            timeout=TOTAL_TIMEOUT,
            catch=(Exception,),  # Catch all exceptions
            gc_after_trial=True  # Force garbage collection after each trial
        )
        print(f"\nOptimization completed successfully!")
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Optimization interrupted by user")
        print(f"Completed {completed_trials}/{n_trials} trials")
    except Exception as e:
        print(f"\n[ERROR] Optimization error: {str(e)}")
        print(f"Completed {completed_trials}/{n_trials} trials")

    # Always clean up and return results
    print(f"\n{'='*60}")
    print(f"Optimization finished. Completed trials: {len(study.trials)}/{n_trials}")

    if len(study.trials) > 0:
        print(f"Best score: {study.best_value:.6f}")
        print(f"Total time: {time.time() - start_time:.1f}s")
        print(f"{'='*60}\n")
        return study.best_params, study
    else:
        print(f"No trials completed. Returning default parameters.")
        print(f"{'='*60}\n")
        return {}, study

def optimize_keff_model(X_train, y_keff_train, model_type='xgboost', n_trials=250, n_jobs=10):
    """Optimize hyperparameters for k-eff prediction only"""

    print(f"\n{'='*60}")
    print(f"Starting {model_type.upper()} optimization for K-EFF")
    print(f"Total trials: {n_trials}, Timeout per trial: {TRIAL_TIMEOUT}s")
    print(f"Total timeout: {TOTAL_TIMEOUT}s")
    print(f"{'='*60}\n")

    start_time = time.time()
    completed_trials = 0

    def objective(trial):
        nonlocal completed_trials
        trial_start = time.time()

        # Check if we've exceeded total timeout
        if time.time() - start_time > TOTAL_TIMEOUT:
            print(f"\n[WARNING] Total timeout reached. Stopping optimization.")
            trial.study.stop()
            return float('inf')

        print(f"\n[Trial {trial.number + 1}/{n_trials}] Starting at {datetime.now().strftime('%H:%M:%S')}")

        try:
            # Same parameter definitions as above but for single output
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'n_jobs': 1,
                    'verbosity': 2  # High verbosity
                }
                print(f"  XGBoost params: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")
                model = xgb.XGBRegressor(**params)

            elif model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 5, 50),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
                    'n_jobs': 1,
                    'verbose': 1  # Verbose output
                }
                print(f"  RF params: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")
                model = RandomForestRegressor(**params)

            elif model_type == 'svm':
                params = {
                    'C': trial.suggest_float('C', 0.001, 1000, log=True),
                    'gamma': trial.suggest_float('gamma', 0.0001, 1, log=True),
                    'epsilon': trial.suggest_float('epsilon', 0.001, 1.0, log=True),
                    'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
                    'verbose': True,  # Verbose output
                    'max_iter': 10000
                }
                if params['kernel'] == 'poly':
                    params['degree'] = trial.suggest_int('degree', 2, 5)
                print(f"  SVM params: C={params['C']:.4f}, gamma={params['gamma']:.6f}, kernel={params['kernel']}")
                model = SVR(**params)

            elif model_type == 'neural_net':
                n_layers = trial.suggest_int('n_layers', 1, 5)
                layers = []
                for i in range(n_layers):
                    layers.append(trial.suggest_int(f'layer_{i}_size', 50, 400))

                params = {
                    'hidden_layer_sizes': tuple(layers),
                    'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.01, log=True),
                    'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
                    'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                    'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
                    'max_iter': 500,
                    'verbose': True,  # Verbose output
                    'early_stopping': True,
                    'n_iter_no_change': 10
                }
                print(f"  NN params: layers={params['hidden_layer_sizes']}, solver={params['solver']}")
                model = MLPRegressor(**params)

            # Train and evaluate with CV - keeping all original functionality
            print(f"  Starting cross-validation...")
            cv_scores = []

            # Set environment variable to limit thread usage
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'

            for fold_idx in range(10):
                fold_start = time.time()

                # Check trial timeout
                if time.time() - trial_start > TRIAL_TIMEOUT:
                    print(f"  [TIMEOUT] Trial exceeded {TRIAL_TIMEOUT}s limit")
                    if cv_scores:  # If we have some scores, use their mean
                        return -np.mean(cv_scores)
                    else:
                        return float('inf')

                print(f"    Fold {fold_idx + 1}/10...", end='', flush=True)

                # Suppress warnings during CV
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    try:
                        score = cross_val_score(model, X_train, y_keff_train.ravel(), cv=10,
                                              scoring='neg_mean_squared_error', n_jobs=1)
                        cv_scores.extend(score)
                        print(f" done ({time.time() - fold_start:.1f}s)")
                        break  # We got all scores at once
                    except Exception as e:
                        print(f" error: {str(e)[:50]}")
                        continue

            if not cv_scores:
                print(f"  No valid CV scores obtained")
                return float('inf')

            final_score = -np.mean(cv_scores)
            print(f"  Trial {trial.number} score: {final_score:.6f}")
            print(f"  Trial time: {time.time() - trial_start:.1f}s")

            # Increment completed trials
            completed_trials += 1

            # Force garbage collection
            del model
            gc.collect()

            return final_score

        except TimeoutError:
            print(f"  [TIMEOUT] Trial {trial.number} timed out")
            completed_trials += 1
            return float('inf')
        except Exception as e:
            print(f"  [ERROR] Trial {trial.number} failed: {str(e)[:100]}")
            completed_trials += 1
            return float('inf')

    # Create study with proper exception handling
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(n_startup_trials=30, seed=42),
        pruner=None
    )

    # Add callback to print best value updates
    def callback(study, trial):
        if study.best_trial.number == trial.number:
            print(f"\n[NEW BEST] Trial {trial.number}: {study.best_value:.6f}")
            print(f"Parameters: {trial.params}\n")

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
            callbacks=[callback],
            timeout=TOTAL_TIMEOUT,
            catch=(Exception,),
            gc_after_trial=True
        )
        print(f"\nOptimization completed successfully!")
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Optimization interrupted by user")
        print(f"Completed {completed_trials}/{n_trials} trials")
    except Exception as e:
        print(f"\n[ERROR] Optimization error: {str(e)}")
        print(f"Completed {completed_trials}/{n_trials} trials")

    # Always clean up and return results
    print(f"\n{'='*60}")
    print(f"Optimization finished. Completed trials: {len(study.trials)}/{n_trials}")

    if len(study.trials) > 0:
        print(f"Best score: {study.best_value:.6f}")
        print(f"Total time: {time.time() - start_time:.1f}s")
        print(f"{'='*60}\n")
        return study.best_params, study
    else:
        print(f"No trials completed. Returning default parameters.")
        print(f"{'='*60}\n")
        return {}, study
