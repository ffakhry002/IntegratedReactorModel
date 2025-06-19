import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, GroupKFold, KFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from scipy.stats import uniform, randint
import joblib
import time
from datetime import datetime
import warnings
from functools import wraps
import signal

# Global timeout settings
STAGE_TIMEOUT = 3600*2  # 2 hours per stage
TOTAL_TIMEOUT = 7200*3  # 6 hours total

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")

def with_timeout(timeout_seconds):
    """Decorator to add timeout to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set the signal handler and a timeout alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)
            return result
        return wrapper
    return decorator

def three_stage_optimization(X_train, y_train, model_class, model_type='xgboost',
                           n_jobs=-1, target_type='flux', use_log_flux=True, groups=None):
    """
    Three-stage hyperparameter optimization: Random → Grid → Bayesian
    Complete implementation for all model types with timeouts and verbosity
    NOW WITH PROPER GROUP-AWARE CV

    Args:
        target_type: 'flux' or 'keff' - determines optimization metric
        use_log_flux: whether flux data is log-transformed
        groups: array indicating which samples belong to same original config
    """

    print(f"\n{'='*60}")
    print(f"Three-Stage Optimization for {model_type}")
    print(f"Target: {target_type.upper()}")
    print(f"Optimization metric: {'MAPE' if target_type == 'flux' else 'MSE'}")
    print(f"Stage timeout: {STAGE_TIMEOUT}s, Total timeout: {TOTAL_TIMEOUT}s")
    print(f"{'='*60}")

    # Track overall progress
    optimization_start_time = time.time()
    best_params_so_far = {}
    best_score_so_far = float('-inf')

    # Check if this is multi-output (flux) or single output (keff)
    is_multi_output = len(y_train.shape) > 1 and y_train.shape[1] > 1

    print(f"\n📊 Data Info:")
    print(f"   - Training samples: {X_train.shape[0]}")
    print(f"   - Features: {X_train.shape[1]}")
    print(f"   - Output type: {'Multi-output' if is_multi_output else 'Single output'}")
    if is_multi_output:
        print(f"   - Number of outputs: {y_train.shape[1]}")
    print(f"   - Using {n_jobs} parallel jobs")

    # SET UP CROSS-VALIDATION STRATEGY
    if groups is not None:
        cv = GroupKFold(n_splits=10)
        n_unique_configs = len(np.unique(groups))
        print(f"   - Using GroupKFold with {n_unique_configs} unique configurations")
        print(f"   - Preventing augmentation leakage in CV")
    else:
        cv = 10  # Regular KFold
        print(f"   - WARNING: No groups provided - may have CV leakage!")
        print(f"   - Using regular {cv}-fold cross-validation")

    # Create custom MAPE scorer for flux models
    if target_type == 'flux':
        from sklearn.metrics import make_scorer

        def mape_scorer(y_true, y_pred):
            """Custom MAPE scorer that handles log-transformed data"""
            # Convert from log scale if needed
            if use_log_flux:
                y_true_linear = 10 ** y_true
                y_pred_linear = 10 ** y_pred
            else:
                y_true_linear = y_true
                y_pred_linear = y_pred

            # Calculate MAPE (always positive)
            if len(y_true_linear.shape) > 1:
                mapes = []
                for i in range(len(y_true_linear)):
                    sample_errors = []
                    for j in range(y_true_linear.shape[1]):
                        if y_true_linear[i, j] != 0:
                            error = abs((y_pred_linear[i, j] - y_true_linear[i, j]) / y_true_linear[i, j]) * 100
                            sample_errors.append(error)
                    if sample_errors:
                        mapes.append(np.mean(sample_errors))
                return np.mean(mapes)
            else:
                # Single output
                mask = y_true_linear != 0
                mape = np.mean(np.abs((y_pred_linear[mask] - y_true_linear[mask]) / y_true_linear[mask])) * 100
                return mape

        # Tell sklearn that lower MAPE is better (not greater)
        scoring = make_scorer(mape_scorer, greater_is_better=False)
        print(f"   - Using MAPE scoring (log_flux={use_log_flux}) - lower is better")
    else:
        scoring = 'neg_mean_squared_error'
        print(f"   - Using MSE scoring")

    # Helper function to safely fit with timeout
    def safe_fit_with_progress(search_cv, X, y, stage_name, groups=None):
        """Fit with timeout and progress tracking"""
        try:
            print(f"\n Starting {stage_name} at {datetime.now().strftime('%H:%M:%S')}...")

            # Add custom scoring wrapper for verbose output
            original_fit = search_cv.fit
            fit_count = [0]

            # Calculate total fits
            if hasattr(search_cv, 'cv') and hasattr(search_cv.cv, 'n_splits'):
                cv_folds = search_cv.cv.n_splits
            elif hasattr(search_cv, 'cv') and isinstance(search_cv.cv, int):
                cv_folds = search_cv.cv
            else:
                cv_folds = 10  # Default

            if hasattr(search_cv, 'n_iter'):
                total_fits = search_cv.n_iter * cv_folds
            else:
                # Grid search
                if hasattr(search_cv, 'param_grid'):
                    if isinstance(search_cv.param_grid, dict):
                        total_combinations = 1
                        for param_values in search_cv.param_grid.values():
                            total_combinations *= len(param_values)
                    else:
                        total_combinations = sum(
                            np.prod([len(v) for v in grid.values()])
                            for grid in search_cv.param_grid
                        )
                    total_fits = total_combinations * cv_folds
                else:
                    total_fits = cv_folds

            def verbose_fit(*args, **kwargs):
                start_time = time.time()
                result = original_fit(*args, **kwargs)
                fit_count[0] += 1

                # Print progress every 10%
                if fit_count[0] % max(1, total_fits // 10) == 0:
                    elapsed = time.time() - start_time
                    print(f"   Progress: {fit_count[0]}/{total_fits} fits ({fit_count[0]/total_fits*100:.0f}%) - {elapsed:.1f}s")

                return result

            # Temporarily replace fit method
            search_cv.fit = verbose_fit

            # Set verbose to 2 for underlying estimators if they support it
            if hasattr(search_cv.estimator, 'set_params'):
                try:
                    if model_type == 'xgboost':
                        search_cv.estimator.set_params(verbosity=2)
                    elif model_type == 'random_forest':
                        search_cv.estimator.set_params(verbose=1)
                    elif model_type == 'svm':
                        search_cv.estimator.set_params(verbose=True)
                    elif model_type == 'neural_net':
                        search_cv.estimator.set_params(verbose=True)
                except Exception as e:
                    print(f"   ⚠️  Could not set verbose mode for {model_type}: {str(e)[:50]}")

            # Fit with timeout
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(STAGE_TIMEOUT)
                try:
                    # Pass groups if available
                    if groups is not None:
                        search_cv.fit(X, y, groups=groups)
                    else:
                        search_cv.fit(X, y)
                    print(f"\n {stage_name} completed successfully!")
                    if target_type == 'flux':
                        print(f" Best MAPE: {abs(search_cv.best_score_):.2f}%")
                    else:
                        print(f" Best score: {search_cv.best_score_:.6f}")
                finally:
                    signal.alarm(0)

            return True, search_cv.best_params_, search_cv.best_score_

        except TimeoutException:
            print(f"\n {stage_name} timed out after {STAGE_TIMEOUT}s")
            if hasattr(search_cv, 'cv_results_'):
                scores = search_cv.cv_results_['mean_test_score']
                if len(scores) > 0:
                    if target_type == 'flux':
                        best_idx = np.argmin(scores)
                    else:
                        best_idx = np.argmax(scores)
                    best_params = search_cv.cv_results_['params'][best_idx]
                    best_score = scores[best_idx]
                    print(f"   Using partial results: {len(scores)} combinations tested")
                    if target_type == 'flux':
                        print(f"   Best MAPE so far: {abs(best_score):.2f}%")
                    else:
                        print(f"   Best score so far: {best_score:.6f}")
                    return True, best_params, best_score
            return False, {}, float('-inf')

        except Exception as e:
            print(f"\n {stage_name} failed with error: {str(e)[:100]}")
            return False, {}, float('-inf')

    # Stage 1: Random Search - Cast wide net
    print(f"\n{'='*60}")
    print("STAGE 1/3: RANDOM SEARCH")
    print(f"{'='*60}")
    print("Exploring parameter space broadly...")
    print("This stage tests 250 random parameter combinations")
    stage1_start = time.time()

    if model_type == 'xgboost':
        if is_multi_output:
            param_distributions = {
                'estimator__n_estimators': randint(50, 5000),
                'estimator__max_depth': randint(2, 20),
                'estimator__learning_rate': uniform(0.001, 0.499),
                'estimator__subsample': uniform(0.3, 0.7),
                'estimator__colsample_bytree': uniform(0.3, 0.7),
                'estimator__colsample_bylevel': uniform(0.3, 0.7),
                'estimator__reg_alpha': uniform(0, 10),
                'estimator__reg_lambda': uniform(0, 10),
                'estimator__gamma': uniform(0.0001, 0.0999),
                'estimator__min_child_weight': randint(1, 20)
            }
        else:
            param_distributions = {
                'n_estimators': randint(50, 5000),
                'max_depth': randint(2, 20),
                'learning_rate': uniform(0.001, 0.499),
                'subsample': uniform(0.3, 0.7),
                'colsample_bytree': uniform(0.3, 0.7),
                'colsample_bylevel': uniform(0.3, 0.7),
                'reg_alpha': uniform(0, 10),
                'reg_lambda': uniform(0, 10),
                'gamma': uniform(0.0001, 0.0999),
                'min_child_weight': randint(1, 20)
            }
    elif model_type == 'random_forest':
        param_distributions = {
            'n_estimators': randint(200, 1500),
            'max_depth': randint(3, 40),
            'min_samples_split': randint(2, 30),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, 0.9],
            'max_samples': uniform(0.2, 0.8),
            'bootstrap': [True, False]
        }
    elif model_type == 'svm':
        if is_multi_output:
            param_distributions = {
                'estimator__C': uniform(1.0, 99.0),
                'estimator__gamma': uniform(0.0001, 0.0999),
                'estimator__epsilon': uniform(0.0005, 0.0995),
                'estimator__kernel': ['rbf', 'poly'],
                'estimator__degree': randint(2, 5),
                'estimator__coef0': uniform(1, 9)
            }
        else:
            param_distributions = {
                'C': uniform(1.0, 99.0),
                'gamma': uniform(0.0001, 0.0999),
                'epsilon': uniform(0.0005, 0.0995),
                'kernel': ['rbf', 'poly'],
                'degree': randint(2, 5),
                'coef0': uniform(1, 9)
            }
    elif model_type == 'neural_net':
        if is_multi_output:
            param_distributions = {
                'estimator__hidden_layer_sizes': [
                    (50,), (100,), (200,), (300,), (400,),
                    (100, 50), (200, 100), (300, 150), (400, 200),
                    (200, 100, 50), (300, 200, 100), (400, 300, 200),
                    (300, 200, 100, 50), (400, 300, 200, 100),
                    (400, 300, 200, 100, 50)
                ],
                'estimator__learning_rate_init': uniform(0.0001, 0.0099),
                'estimator__alpha': uniform(0.0001, 0.0999),
                'estimator__activation': ['relu', 'tanh'],
                'estimator__solver': ['adam', 'lbfgs']
            }
        else:
            param_distributions = {
                'hidden_layer_sizes': [
                    (50,), (100,), (200,), (300,), (400,),
                    (100, 50), (200, 100), (300, 150), (400, 200),
                    (200, 100, 50), (300, 200, 100), (400, 300, 200),
                    (300, 200, 100, 50), (400, 300, 200, 100),
                    (400, 300, 200, 100, 50)
                ],
                'learning_rate_init': uniform(0.0001, 0.0099),
                'alpha': uniform(0.0001, 0.0999),
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'lbfgs']
            }

    print(f"\n Random Search Parameters:")
    print(f"   - Candidates to test: 500")
    print(f"   - Cross-validation folds: 10")
    print(f"   - Total fits: 5,000")
    print(f"   - Timeout: {STAGE_TIMEOUT}s")

    random_search = RandomizedSearchCV(
        model_class(),
        param_distributions,
        n_iter=500,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=2,
        random_state=42
    )

    success, best_random_params, best_random_score = safe_fit_with_progress(
        random_search, X_train, y_train, "Random Search", groups
    )

    if success and best_random_params:
        best_params_so_far = best_random_params
        best_score_so_far = best_random_score

    stage1_time = time.time() - stage1_start
    print(f"\nStage 1 took {stage1_time/60:.1f} minutes")
    if target_type == 'flux':
        print(f"Best MAPE: {abs(best_random_score):.2f}%")
    else:
        print(f"Best score: {best_random_score:.6f}")
    print(f"Best params: {best_random_params}")

    # Check if Stage 1 failed
    if not best_random_params:
        print(f"\n Stage 1 failed to find any parameters. Returning default parameters.")
        # Return default parameters...
        return get_default_params(model_type), None

    # Check total timeout
    if time.time() - optimization_start_time > TOTAL_TIMEOUT:
        print(f"\n Total timeout reached. Returning best parameters found so far.")
        return best_params_so_far, None

    # Helper function to get parameter value with or without prefix
    def get_param_value(params, base_key, default=None):
        """Get parameter value checking both with and without estimator__ prefix"""
        prefixed_key = f'estimator__{base_key}'
        if prefixed_key in params:
            return params[prefixed_key]
        elif base_key in params:
            return params[base_key]
        elif default is not None:
            print(f"   Parameter '{base_key}' not found, using default: {default}")
            return default
        else:
            raise KeyError(f"Parameter '{base_key}' not found")

    # Stage 2: Grid Search - Focus on promising region
    print(f"\n{'='*60}")
    print("STAGE 2/3: GRID SEARCH")
    print(f"{'='*60}")
    print("Refining parameters around best region from Random Search...")
    stage2_start = time.time()

    # Create focused grid around best random search params
    if model_type == 'xgboost':
        if is_multi_output:
            n_est = get_param_value(best_random_params, 'n_estimators')
            depth = get_param_value(best_random_params, 'max_depth')
            lr = get_param_value(best_random_params, 'learning_rate')

            param_grid = {
                'estimator__n_estimators': [max(100, n_est-200), n_est-100, n_est, n_est+100, min(1500, n_est+200)],
                'estimator__max_depth': [max(3, depth-2), depth-1, depth, depth+1, min(15, depth+2)],
                'estimator__learning_rate': [max(0.001, lr*0.5), lr*0.75, lr, lr*1.25, min(0.3, lr*1.5)],
                'estimator__subsample': [0.6, 0.7, 0.8, 0.9],
                'estimator__colsample_bytree': [0.6, 0.7, 0.8, 0.9]
            }
        else:
            n_est = get_param_value(best_random_params, 'n_estimators')
            depth = get_param_value(best_random_params, 'max_depth')
            lr = get_param_value(best_random_params, 'learning_rate')

            param_grid = {
                'n_estimators': [max(100, n_est-200), n_est-100, n_est, n_est+100, min(1500, n_est+200)],
                'max_depth': [max(3, depth-2), depth-1, depth, depth+1, min(15, depth+2)],
                'learning_rate': [max(0.001, lr*0.5), lr*0.75, lr, lr*1.25, min(0.3, lr*1.5)],
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
            }

    elif model_type == 'random_forest':
        n_est = get_param_value(best_random_params, 'n_estimators')
        depth = get_param_value(best_random_params, 'max_depth', 20)
        min_split = get_param_value(best_random_params, 'min_samples_split')

        param_grid = {
            'n_estimators': [max(100, n_est-100), n_est-50, n_est, n_est+50, min(1000, n_est+100)],
            'max_depth': [max(5, depth-5), depth-2, depth, depth+2, min(50, depth+5)] if depth else [None, 10, 20, 30, 40],
            'min_samples_split': [max(2, min_split-3), min_split-1, min_split, min_split+1, min(20, min_split+3)],
            'min_samples_leaf': [1, 2, 4, 6, 8]
        }

    elif model_type == 'svm':
        if is_multi_output:
            C = get_param_value(best_random_params, 'C')
            gamma = get_param_value(best_random_params, 'gamma')
            eps = get_param_value(best_random_params, 'epsilon')

            param_grid = {
                'estimator__C': [C/10, C/3, C, C*3, C*10],
                'estimator__gamma': [gamma/10, gamma/3, gamma, gamma*3, gamma*10],
                'estimator__epsilon': [max(0.001, eps*0.5), eps*0.75, eps, eps*1.25, min(1.0, eps*1.5)],
                'estimator__kernel': [get_param_value(best_random_params, 'kernel')]
            }
        else:
            C = get_param_value(best_random_params, 'C')
            gamma = get_param_value(best_random_params, 'gamma')
            eps = get_param_value(best_random_params, 'epsilon')

            param_grid = {
                'C': [C/10, C/3, C, C*3, C*10],
                'gamma': [gamma/10, gamma/3, gamma, gamma*3, gamma*10],
                'epsilon': [max(0.001, eps*0.5), eps*0.75, eps, eps*1.25, min(1.0, eps*1.5)],
                'kernel': [get_param_value(best_random_params, 'kernel')]
            }

    elif model_type == 'neural_net':
        if is_multi_output:
            layers = get_param_value(best_random_params, 'hidden_layer_sizes')
            lr = get_param_value(best_random_params, 'learning_rate_init')
            alpha = get_param_value(best_random_params, 'alpha')

            layer_variations = []
            if isinstance(layers, tuple):
                base_layers = list(layers)
                for delta in [-50, -25, 0, 25, 50]:
                    new_layers = tuple(max(50, min(400, l + delta)) for l in base_layers)
                    if new_layers not in layer_variations:
                        layer_variations.append(new_layers)

            param_grid = {
                'estimator__hidden_layer_sizes': layer_variations[:5],
                'estimator__learning_rate_init': [max(0.0001, lr*0.5), lr*0.75, lr, lr*1.25, min(0.01, lr*1.5)],
                'estimator__alpha': [max(0.0001, alpha*0.5), alpha*0.75, alpha, alpha*1.25, min(0.1, alpha*1.5)],
                'estimator__activation': [get_param_value(best_random_params, 'activation')]
            }
        else:
            layers = get_param_value(best_random_params, 'hidden_layer_sizes')
            lr = get_param_value(best_random_params, 'learning_rate_init')
            alpha = get_param_value(best_random_params, 'alpha')

            layer_variations = []
            if isinstance(layers, tuple):
                base_layers = list(layers)
                for delta in [-50, -25, 0, 25, 50]:
                    new_layers = tuple(max(50, min(400, l + delta)) for l in base_layers)
                    if new_layers not in layer_variations:
                        layer_variations.append(new_layers)

            param_grid = {
                'hidden_layer_sizes': layer_variations[:5],
                'learning_rate_init': [max(0.0001, lr*0.5), lr*0.75, lr, lr*1.25, min(0.01, lr*1.5)],
                'alpha': [max(0.0001, alpha*0.5), alpha*0.75, alpha, alpha*1.25, min(0.1, alpha*1.5)],
                'activation': [get_param_value(best_random_params, 'activation')]
            }

    # Calculate total combinations
    total_combinations = 1
    for param_values in param_grid.values():
        total_combinations *= len(param_values)

    print(f"\n🔍 Grid Search Parameters:")
    print(f"   - Parameter combinations: {total_combinations}")
    print(f"   - Cross-validation folds: 10")
    print(f"   - Total fits: {total_combinations * 10}")
    print(f"   - Timeout: {STAGE_TIMEOUT}s")

    grid_model = model_class()

    grid_search = GridSearchCV(
        grid_model,
        param_grid,
        cv=cv,  # Use the CV strategy defined at the beginning
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=2
    )

    success, best_grid_params, best_grid_score = safe_fit_with_progress(
        grid_search, X_train, y_train, "Grid Search", groups
    )

    if success and best_grid_params:
        if target_type == 'flux':
            improved = best_grid_score > best_score_so_far
        else:
            improved = best_grid_score > best_score_so_far

        if improved:
            best_params_so_far = best_grid_params
            best_score_so_far = best_grid_score
            print(f"\n Grid search improved performance!")
        else:
            print(f"\n Grid search did not improve performance.")

    stage2_time = time.time() - stage2_start
    print(f"\nStage 2 took {stage2_time/60:.1f} minutes")

    # Check total timeout
    if time.time() - optimization_start_time > TOTAL_TIMEOUT:
        print(f"\n Total timeout reached. Returning best parameters found so far.")
        return best_params_so_far, None

    # Stage 3: Bayesian Optimization - Fine tuning
    print(f"\n{'='*60}")
    print("STAGE 3/3: BAYESIAN OPTIMIZATION")
    print(f"{'='*60}")
    print("Fine-tuning with intelligent search...")
    stage3_start = time.time()

    # Update best params with grid results
    best_params = {**best_random_params, **best_grid_params}

    # Define search spaces for Bayesian optimization
    if model_type == 'xgboost':
        if is_multi_output:
            n_est = get_param_value(best_params, 'n_estimators')
            depth = get_param_value(best_params, 'max_depth')
            lr = get_param_value(best_params, 'learning_rate')
            subsample = get_param_value(best_params, 'subsample')
            colsample = get_param_value(best_params, 'colsample_bytree')

            search_spaces = {
                'estimator__n_estimators': Integer(max(100, n_est-100), min(1500, n_est+100)),
                'estimator__max_depth': Integer(max(3, depth-1), min(15, depth+1)),
                'estimator__learning_rate': Real(max(0.001, lr*0.8), min(0.3, lr*1.2), prior='log-uniform'),
                'estimator__subsample': Real(max(0.5, subsample-0.1), min(1.0, subsample+0.1)),
                'estimator__colsample_bytree': Real(max(0.5, colsample-0.1), min(1.0, colsample+0.1))
            }
        else:
            n_est = get_param_value(best_params, 'n_estimators')
            depth = get_param_value(best_params, 'max_depth')
            lr = get_param_value(best_params, 'learning_rate')
            subsample = get_param_value(best_params, 'subsample')
            colsample = get_param_value(best_params, 'colsample_bytree')

            search_spaces = {
                'n_estimators': Integer(max(100, n_est-100), min(1500, n_est+100)),
                'max_depth': Integer(max(3, depth-1), min(15, depth+1)),
                'learning_rate': Real(max(0.001, lr*0.8), min(0.3, lr*1.2), prior='log-uniform'),
                'subsample': Real(max(0.5, subsample-0.1), min(1.0, subsample+0.1)),
                'colsample_bytree': Real(max(0.5, colsample-0.1), min(1.0, colsample+0.1))
            }

    elif model_type == 'random_forest':
        n_est = get_param_value(best_params, 'n_estimators')
        min_split = get_param_value(best_params, 'min_samples_split')
        min_leaf = get_param_value(best_params, 'min_samples_leaf')

        search_spaces = {
            'n_estimators': Integer(max(100, n_est-50), min(1000, n_est+50)),
            'min_samples_split': Integer(max(2, min_split-2), min(20, min_split+2)),
            'min_samples_leaf': Integer(max(1, min_leaf-1), min(10, min_leaf+2))
        }

        depth = get_param_value(best_params, 'max_depth', None)
        if depth is not None:
            search_spaces['max_depth'] = Integer(max(5, depth-3), min(50, depth+3))

    elif model_type == 'svm':
        if is_multi_output:
            C = get_param_value(best_params, 'C')
            gamma = get_param_value(best_params, 'gamma')
            eps = get_param_value(best_params, 'epsilon')
            kernel = get_param_value(best_params, 'kernel')

            search_spaces = {
                'estimator__C': Real(C/5, C*5, prior='log-uniform'),
                'estimator__gamma': Real(gamma/5, gamma*5, prior='log-uniform'),
                'estimator__epsilon': Real(max(0.001, eps*0.5), min(1.0, eps*1.5)),
                'estimator__kernel': Categorical([kernel])
            }
        else:
            C = get_param_value(best_params, 'C')
            gamma = get_param_value(best_params, 'gamma')
            eps = get_param_value(best_params, 'epsilon')
            kernel = get_param_value(best_params, 'kernel')

            search_spaces = {
                'C': Real(C/5, C*5, prior='log-uniform'),
                'gamma': Real(gamma/5, gamma*5, prior='log-uniform'),
                'epsilon': Real(max(0.001, eps*0.5), min(1.0, eps*1.5)),
                'kernel': Categorical([kernel])
            }

    elif model_type == 'neural_net':
        if is_multi_output:
            lr = get_param_value(best_params, 'learning_rate_init')
            alpha = get_param_value(best_params, 'alpha')
            layers = get_param_value(best_params, 'hidden_layer_sizes')
            activation = get_param_value(best_params, 'activation')

            search_spaces = {
                'estimator__learning_rate_init': Real(lr*0.5, lr*2.0, prior='log-uniform'),
                'estimator__alpha': Real(alpha*0.5, alpha*2.0, prior='log-uniform'),
                'estimator__hidden_layer_sizes': Categorical([layers]),
                'estimator__activation': Categorical([activation])
            }
        else:
            lr = get_param_value(best_params, 'learning_rate_init')
            alpha = get_param_value(best_params, 'alpha')
            layers = get_param_value(best_params, 'hidden_layer_sizes')
            activation = get_param_value(best_params, 'activation')

            search_spaces = {
                'learning_rate_init': Real(lr*0.5, lr*2.0, prior='log-uniform'),
                'alpha': Real(alpha*0.5, alpha*2.0, prior='log-uniform'),
                'hidden_layer_sizes': Categorical([layers]),
                'activation': Categorical([activation])
            }

    bayes_search = BayesSearchCV(
        model_class(),
        search_spaces,
        n_iter=100,
        cv=cv,  # Use the CV strategy defined at the beginning
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=2,
        random_state=42
    )

    success, best_bayes_params, best_bayes_score = safe_fit_with_progress(
        bayes_search, X_train, y_train, "Bayesian Optimization", groups
    )

    if success and best_bayes_params:
        if target_type == 'flux':
            improved = best_bayes_score > best_score_so_far
        else:
            improved = best_bayes_score > best_score_so_far

        if improved:
            best_params_so_far = best_bayes_params
            best_score_so_far = best_bayes_score
            print(f"\n Bayesian search improved performance!")
        else:
            print(f"\n Bayesian search did not improve performance.")

    stage3_time = time.time() - stage3_start
    total_time = time.time() - optimization_start_time

    print(f"\nStage 3 took {stage3_time/60:.1f} minutes")
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE!")
    print(f"{'='*60}")

    # Clean up parameters for final model creation
    if is_multi_output and model_type in ['xgboost', 'svm', 'neural_net']:
        clean_params = {}
        for key, value in best_params_so_far.items():
            if key.startswith('estimator__'):
                clean_params[key.replace('estimator__', '')] = value
            else:
                clean_params[key] = value
        print(f"\n Best parameters (cleaned):")
        for param, value in clean_params.items():
            print(f"   - {param}: {value}")
        return clean_params, None
    else:
        print(f"\n Best parameters:")
        for param, value in best_params_so_far.items():
            print(f"   - {param}: {value}")
        return best_params_so_far, None

def get_default_params(model_type):
    """Get default parameters for model type"""
    defaults = {
        'xgboost': {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'random_forest': {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        },
        'svm': {
            'C': 1.0,
            'gamma': 0.1,
            'epsilon': 0.1,
            'kernel': 'rbf'
        },
        'neural_net': {
            'hidden_layer_sizes': (100,),
            'learning_rate_init': 0.001,
            'alpha': 0.001,
            'activation': 'relu'
        }
    }
    return defaults.get(model_type, {})
