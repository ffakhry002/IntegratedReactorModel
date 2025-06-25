import optuna
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Global timeout settings
TRIAL_TIMEOUT = 600*10  # 30 minutes per trial
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

# Custom MAPE scorer for flux models
def mape_scorer_flux(y_true, y_pred, use_log_flux=True):
    """Custom MAPE scorer that handles log-transformed flux data"""
    # Convert from log scale to linear scale for MAPE calculation
    if use_log_flux:
        y_true_linear = 10 ** y_true
        y_pred_linear = 10 ** y_pred
    else:
        # If not using log scale, assume data is already scaled
        y_true_linear = y_true * 1e14  # Adjust scale as needed
        y_pred_linear = y_pred * 1e14

    # Calculate MAPE
    n_outputs = y_true_linear.shape[1] if len(y_true_linear.shape) > 1 else 1

    if n_outputs == 1:
        # Single output
        mask = y_true_linear != 0
        if mask.any():
            mape = np.mean(np.abs((y_pred_linear[mask] - y_true_linear[mask]) / y_true_linear[mask])) * 100
        else:
            mape = float('inf')
    else:
        # Multi-output
        mapes = []
        for i in range(len(y_true_linear)):
            sample_errors = []
            for j in range(n_outputs):
                if y_true_linear[i, j] != 0:
                    error = abs((y_pred_linear[i, j] - y_true_linear[i, j]) / y_true_linear[i, j]) * 100
                    sample_errors.append(error)
            if sample_errors:
                mapes.append(np.mean(sample_errors))

        if mapes:
            mape = np.mean(mapes)
        else:
            mape = float('inf')

    return mape

def optimize_flux_model(X_train, y_flux_train, model_type='xgboost', n_trials=250, n_jobs=10, use_log_flux=True, groups=None, flux_mode='total'):
    """Optimize hyperparameters for flux prediction only - NOW USING MAPE or MSE based on mode"""

    print(f"\n{'='*60}")
    print(f"Starting {model_type.upper()} optimization for FLUX")
    print(f"Flux mode: {flux_mode}")
    if flux_mode == 'bin':
        print(f"Optimization metric: MSE (for energy bins)")
    elif flux_mode in ['thermal_only', 'epithermal_only', 'fast_only']:
        energy_group = flux_mode.replace('_only', '')
        print(f"Optimization metric: MAPE (for {energy_group} flux only)")
    else:
        print(f"Optimization metric: MAPE (Mean Absolute Percentage Error)")
    print(f"Total trials: {n_trials}, Timeout per trial: {TRIAL_TIMEOUT}s")
    print(f"Total timeout: {TOTAL_TIMEOUT}s")

    # NEW: Check if groups provided
    if groups is not None:
        print(f"Using GroupKFold to prevent augmentation leakage")
        print(f"Number of unique configurations: {len(np.unique(groups))}")
    else:
        print(f"WARNING: No groups provided - may have augmentation leakage!")

    print(f"{'='*60}\n")

    start_time = time.time()
    completed_trials = 0

    # Create custom MAPE scorer with use_log_flux parameter (matching old implementation)
    def mape_scorer_wrapper(y_true, y_pred):
        """Wrapper to pass use_log_flux parameter to MAPE scorer"""
        return mape_scorer_flux(y_true, y_pred, use_log_flux=use_log_flux)

    custom_mape_scorer = make_scorer(mape_scorer_wrapper, greater_is_better=False)

    def objective(trial):
        nonlocal completed_trials
        trial_start = time.time()

        # Check if we've exceeded total timeout
        if time.time() - start_time > TOTAL_TIMEOUT:
            print(f"\n[WARNING] Total timeout reached. Stopping optimization.")
            trial.study.stop()
            return float('inf')

        print(f"\n[Trial {trial.number + 1}/{n_trials}] Starting at {datetime.now().strftime('%H:%M:%S')}")

        # Set environment variable to limit thread usage
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

        try:
            # [KEEP ALL YOUR EXISTING PARAMETER SELECTION CODE HERE - NO CHANGES]
            if model_type == 'xgboost':
                params = {
                    # ##### For full screening
                    'n_estimators': trial.suggest_int('n_estimators', 50, 5000),
                    'max_depth': trial.suggest_int('max_depth', 2, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
                    'subsample': trial.suggest_float('subsample', 0.3, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                    'gamma': trial.suggest_float('gamma', 0.0001, 0.1, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                    'n_jobs': 1,
                    'verbosity': 1,
                    'tree_method': 'exact'
                }

                print(f"  XGBoost params: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")
                model = MultiOutputRegressor(xgb.XGBRegressor(**params))

            elif model_type == 'random_forest':
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)

                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 1500),
                    'max_depth': trial.suggest_int('max_depth', 10, 40),
                    'min_samples_split': trial.suggest_int('min_samples_split', max(5, 2 * min_samples_leaf), 30),
                    'min_samples_leaf': min_samples_leaf,
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7, 0.9]),
                    'max_samples': trial.suggest_float('max_samples', 0.2, 1.0),
                    'n_jobs': 1,
                    'verbose': 2
                }
                print(f"  RF params: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, min_samples_split={params['min_samples_split']}, min_samples_leaf={params['min_samples_leaf']}")
                model = RandomForestRegressor(**params)

            elif model_type == 'svm':
                kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])

                params = {
                    'C': trial.suggest_float('C', 1.0, 100.0),
                    'epsilon': trial.suggest_float('epsilon', 0.0005, 0.1, log=True),
                    'kernel': kernel,
                    'max_iter': 200000,
                    'tol': trial.suggest_float('tol', 1e-4, 1e-2, log=True),
                    'shrinking': False,
                    'gamma': trial.suggest_float('gamma', 0.0001, 0.1, log=True),
                    'verbose': True,
                }

                # Kernel-specific parameters with unique names
                if kernel == 'poly':
                    params['degree'] = trial.suggest_int('degree', 2, 5)
                    params['coef0'] = trial.suggest_float('coef0', 1, 10)

                print(f"  SVM params: C={params['C']:.4f}, gamma={params['gamma']:.6f}, kernel={params['kernel']}")

                # CRITICAL FIX: Create pipeline with scaling for SVM
                base_svr = SVR(**params)
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('svr', base_svr)
                ])
                model = MultiOutputRegressor(pipeline)

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
                    'verbose': True,
                    'early_stopping': True,
                    'n_iter_no_change': 10
                }
                print(f"  NN params: layers={params['hidden_layer_sizes']}, solver={params['solver']}")
                model = MLPRegressor(**params)

            # Choose scoring based on flux mode
            if flux_mode == 'bin':
                print(f"  Starting MSE-based cross-validation for energy bins...")
                # Use sklearn cross_val_score for bin mode (MSE scoring)
                if groups is not None:
                    from sklearn.model_selection import GroupKFold
                    cv = GroupKFold(n_splits=10)
                    scores = cross_val_score(model, X_train, y_flux_train,
                                           cv=cv, groups=groups,
                                           scoring='neg_mean_squared_error', n_jobs=1)
                else:
                    from sklearn.model_selection import KFold
                    cv = KFold(n_splits=10, shuffle=True, random_state=42)
                    scores = cross_val_score(model, X_train, y_flux_train,
                                           cv=cv, scoring='neg_mean_squared_error', n_jobs=1)

                # Return positive MSE (sklearn returns negative)
                final_score = -np.mean(scores)
                print(f"  Trial {trial.number} MSE: {final_score:.6f}")

            else:
                # MAPE-based scoring using sklearn cross_val_score (fixes model state contamination)
                print(f"  Starting MAPE-based cross-validation...")

                # Set environment variable to limit thread usage
                os.environ['OMP_NUM_THREADS'] = '1'
                os.environ['MKL_NUM_THREADS'] = '1'

                # Use sklearn cross_val_score with custom MAPE scorer
                if groups is not None:
                    from sklearn.model_selection import GroupKFold
                    cv = GroupKFold(n_splits=10)
                    scores = cross_val_score(model, X_train, y_flux_train,
                                           cv=cv, groups=groups,
                                           scoring=custom_mape_scorer, n_jobs=1)
                else:
                    from sklearn.model_selection import KFold
                    cv = KFold(n_splits=10, shuffle=True, random_state=42)
                    scores = cross_val_score(model, X_train, y_flux_train,
                                           cv=cv, scoring=custom_mape_scorer, n_jobs=1)

                # Handle scorer output (custom_mape_scorer returns negative values due to greater_is_better=False)
                final_score = -np.mean(scores)  # Convert back to positive MAPE
                print(f"  Trial {trial.number} MAPE: {final_score:.2f}%")

            print(f"  Trial time: {time.time() - trial_start:.1f}s")

            # Increment completed trials
            completed_trials += 1

            # Force garbage collection
            del model
            gc.collect()

            return final_score  # Return score to minimize

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
        sampler=TPESampler(
            n_startup_trials=150,
            n_ei_candidates=100,
            seed=42
        ),
        pruner=None
    )

    # Add callback to print best value updates
    def callback(study, trial):
        if study.best_trial.number == trial.number:
            if flux_mode == 'bin':
                print(f"\n[NEW BEST] Trial {trial.number}: MSE = {study.best_value:.6f}")
            else:
                print(f"\n[NEW BEST] Trial {trial.number}: MAPE = {study.best_value:.2f}%")
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
        if flux_mode == 'bin':
            print(f"Best MSE: {study.best_value:.6f}")
        else:
            print(f"Best MAPE: {study.best_value:.2f}%")
        print(f"Total time: {time.time() - start_time:.1f}s")

        # Save study for later visualization
        try:
            # Save to outputs folder instead of local folder
            outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'optuna_studies')
            os.makedirs(outputs_dir, exist_ok=True)
            study_filename = f"{model_type}_flux_{flux_mode}_study.pkl"
            study_path = os.path.join(outputs_dir, study_filename)
            joblib.dump(study, study_path)
            print(f"\nStudy saved to: {study_path}")
            print(f"You can load it later for visualization using:")
            print(f"  study = joblib.load('{study_path}')")
        except Exception as e:
            print(f"Could not save study: {str(e)}")

        print(f"{'='*60}\n")
        return study.best_params, study
    else:
        print(f"No trials completed. Returning default parameters.")
        print(f"{'='*60}\n")
        return {}, study

def optimize_keff_model(X_train, y_keff_train, model_type='xgboost', n_trials=250, n_jobs=10, groups=None):
    """Optimize hyperparameters for k-eff prediction only"""

    print(f"\n{'='*60}")
    print(f"Starting {model_type.upper()} optimization for K-EFF")
    print(f"Total trials: {n_trials}, Timeout per trial: {TRIAL_TIMEOUT}s")
    print(f"Total timeout: {TOTAL_TIMEOUT}s")

    # NEW: Check if groups provided
    if groups is not None:
        print(f"Using GroupKFold to prevent augmentation leakage")
        print(f"Number of unique configurations: {len(np.unique(groups))}")
    else:
        print(f"WARNING: No groups provided - may have augmentation leakage!")

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
            # Same parameter definitions as flux optimization - EXACTLY MATCHING
            if model_type == 'xgboost':
                params = {
                    # ##### Matching flux optimization exactly
                    'n_estimators': trial.suggest_int('n_estimators', 50, 5000),
                    'max_depth': trial.suggest_int('max_depth', 2, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
                    'subsample': trial.suggest_float('subsample', 0.3, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                    'gamma': trial.suggest_float('gamma', 0.0001, 0.1, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                    'n_jobs': 1,
                    'verbosity': 1,
                    'tree_method': 'exact'
                }
                print(f"  XGBoost params: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")
                model = xgb.XGBRegressor(**params)

            elif model_type == 'random_forest':
                # Matching flux optimization exactly
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)

                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 1500),
                    'max_depth': trial.suggest_int('max_depth', 10, 40),
                    'min_samples_split': trial.suggest_int('min_samples_split', max(5, 2 * min_samples_leaf), 30),
                    'min_samples_leaf': min_samples_leaf,
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7, 0.9]),
                    'max_samples': trial.suggest_float('max_samples', 0.2, 1.0),
                    'n_jobs': 1,
                    'verbose': 0
                }
                print(f"  RF params: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, min_samples_split={params['min_samples_split']}, min_samples_leaf={params['min_samples_leaf']}")
                model = RandomForestRegressor(**params)

            elif model_type == 'svm':
                # Matching flux optimization exactly
                kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])

                params = {
                    'C': trial.suggest_float('C', 1.0, 100.0),
                    'epsilon': trial.suggest_float('epsilon', 0.0005, 0.1, log=True),
                    'kernel': kernel,
                    'max_iter': 200000,
                    'tol': trial.suggest_float('tol', 1e-4, 1e-2, log=True),
                    'shrinking': False,
                    'verbose': True,
                }

                # Kernel-specific parameters matching flux optimization
                if kernel == 'rbf':
                    params['gamma'] = trial.suggest_float('gamma', 0.0001, 0.1, log=True)
                elif kernel == 'poly':
                    params['gamma'] = trial.suggest_float('gamma', 0.0001, 0.1, log=True)
                    params['degree'] = trial.suggest_int('degree', 2, 5)
                    params['coef0'] = trial.suggest_float('coef0', 1, 10)

                print(f"  SVM params: C={params['C']:.4f}, gamma={params['gamma']:.6f}, kernel={params['kernel']}")

                # Create pipeline with scaling for SVM
                base_svr = SVR(**params)
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('svr', base_svr)
                ])
                model = pipeline

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
                    'verbose': True,
                    'early_stopping': True,
                    'n_iter_no_change': 10
                }
                print(f"  NN params: layers={params['hidden_layer_sizes']}, solver={params['solver']}")
                model = MLPRegressor(**params)

            # Train and evaluate with CV - UPDATED FOR GROUPS
            print(f"  Starting cross-validation...")
            cv_scores = []

            # Set environment variable to limit thread usage
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'

            # NEW: Use GroupKFold if groups provided
            if groups is not None:
                from sklearn.model_selection import GroupKFold, cross_val_score
                cv = GroupKFold(n_splits=10)
            else:
                from sklearn.model_selection import cross_val_score
                cv = 10  # Regular KFold

            # Use cross_val_score with proper CV
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                try:
                    if groups is not None:
                        # Use GroupKFold with groups
                        # CRITICAL FIX: For SVM, we need to scale features
                        if model_type == 'svm':
                            # Create a pipeline that scales then applies SVM
                            model = Pipeline([
                                ('scaler', StandardScaler()),
                                ('svm', model)
                            ])

                        scores = cross_val_score(model, X_train, y_keff_train.ravel(),
                                               cv=cv,
                                               groups=groups,
                                               scoring='neg_mean_squared_error',
                                               n_jobs=1)
                    else:
                        # Regular cross-validation
                        # CRITICAL FIX: For SVM, we need to scale features
                        if model_type == 'svm':
                            # Create a pipeline that scales then applies SVM
                            model = Pipeline([
                                ('scaler', StandardScaler()),
                                ('svm', model)
                            ])

                        scores = cross_val_score(model, X_train, y_keff_train.ravel(),
                                               cv=cv,
                                               scoring='neg_mean_squared_error',
                                               n_jobs=1)
                    cv_scores = scores
                    print(f"  CV completed. Mean score: {np.mean(scores):.6f}")
                except Exception as e:
                    print(f"  CV error: {str(e)[:100]}")
                    return float('inf')

            if len(cv_scores) == 0:
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
        sampler=TPESampler(
            n_startup_trials=150,
            n_ei_candidates=100,
            seed=42
        ),
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

        # Save study for later visualization
        try:
            # Save to outputs folder instead of local folder
            outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'optuna_studies')
            os.makedirs(outputs_dir, exist_ok=True)
            study_filename = f"{model_type}_keff_study.pkl"
            study_path = os.path.join(outputs_dir, study_filename)
            joblib.dump(study, study_path)
            print(f"\nStudy saved to: {study_path}")
            print(f"You can load it later for visualization using:")
            print(f"  study = joblib.load('{study_path}')")
        except Exception as e:
            print(f"Could not save study: {str(e)}")

        print(f"{'='*60}\n")
        return study.best_params, study
    else:
        print(f"No trials completed. Returning default parameters.")
        print(f"{'='*60}\n")
        return {}, study
