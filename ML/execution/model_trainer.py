import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from hyperparameter_tuning.optuna_optimization import optimize_flux_model, optimize_keff_model, get_model_specific_jobs
from hyperparameter_tuning.three_stage_optimization import three_stage_optimization
from hyperparameter_tuning.raytune_neural_net import optimize_neural_net_raytune
from hyperparameter_tuning.three_stage_neural_net_gpu import three_stage_neural_net_optimization
from ML_models.xgboost_train import XGBoostReactorModel
from ML_models.random_forest_train import RandomForestReactorModel
from ML_models.svm_train import SVMReactorModel
from ML_models.neural_net_train import NeuralNetReactorModel
import joblib
import os
import time
from datetime import datetime

class ModelTrainer:
    """Handle model training and evaluation"""

    def __init__(self, data_handler=None):
        """Initialize the model trainer.

        Parameters
        ----------
        data_handler : object, optional
            Data handler object to access flux transform settings

        Returns
        -------
        None
        """
        # Store reference to data_handler to access flux transform settings
        self.data_handler = data_handler

    def train_model(self, model_type, target, data_splits, config, encoding):
        """Train a single model with hyperparameter optimization"""

        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} for {target.upper()}")
        print(f"Optimization method: {config.optimization}")
        if target == 'flux' and hasattr(config, 'flux_mode'):
            print(f"Flux mode: {config.flux_mode}")
        print(f"{'='*60}")

        # Get appropriate data
        X_train = data_splits['X_train']
        X_test = data_splits['X_test']

        # NEW: Get groups if available
        groups_train = data_splits.get('groups_train', None)

        # NEW: Get lattices for lambda optimization (if available)
        lattices_train = data_splits.get('lattices_train', None)

        if target == 'flux':
            y_train = data_splits['y_flux_train']
            y_test = data_splits['y_flux_test']
        else:  # keff
            y_train = data_splits['y_keff_train']
            y_test = data_splits['y_keff_test']

        # Get flux mode
        flux_mode = config.flux_mode if hasattr(config, 'flux_mode') and target == 'flux' else 'total'

        # NEW: Get irradiation_mode and nci_mode for lambda optimization
        # Read from encoding_methods.py module-level settings
        from ML_models.encodings.encoding_methods import IRRADIATION_MODE, NCI_MODE, NCI_DISTANCE_CUTOFF
        irradiation_mode = IRRADIATION_MODE
        nci_mode = NCI_MODE

        # Check if lambda optimization will be enabled
        # Don't optimize lambda if NCI features are disabled (mode 2)
        optimize_lambda = (encoding == 'physics' and lattices_train is not None and NCI_DISTANCE_CUTOFF != 2)
        if optimize_lambda:
            print(f"✅ Lambda optimization ENABLED: irradiation_mode={irradiation_mode}, nci_mode={nci_mode}")
            print(f"   Will optimize lambda parameters in range [0.5, 2.0]")
            print(f"   NCI formula: exp(-(distance - 1) / lambda)")
            if NCI_DISTANCE_CUTOFF == 1:
                print(f"   Distance cutoff: ENABLED (d > sqrt(5) → 0)")
            else:
                print(f"   Distance cutoff: DISABLED (all distances contribute)")
        elif NCI_DISTANCE_CUTOFF == 2:
            print(f"⚠️  NCI features DISABLED (NCI_DISTANCE_CUTOFF=2)")
            print(f"   Using only global + local features (no lambda optimization)")
        else:
            if encoding != 'physics':
                print(f"⚠️  Lambda optimization disabled: encoding='{encoding}' (need 'physics')")
            elif lattices_train is None:
                print(f"⚠️  Lambda optimization disabled: no lattices provided")
            else:
                print(f"⚠️  Lambda optimization disabled: unknown reason")

        # Get best hyperparameters
        optimization_start = time.time()
        best_cv_score = None  # Track CV score from optimization

        if config.optimization == 'optuna':
            print(f"Starting Optuna optimization...")

            # NEW: Get model-specific job allocation
            n_jobs, cores_per_trial = get_model_specific_jobs(model_type, config)

            if target == 'flux':
                best_params, study = optimize_flux_model(
                    X_train, y_train,
                    model_type=model_type,
                    n_trials=config.n_trials,
                    n_jobs=n_jobs,              # Use model-specific n_jobs
                    cores_per_trial=cores_per_trial,  # NEW parameter
                    groups=groups_train,
                    flux_mode=flux_mode,
                    encoding=encoding,
                    lattices_train=lattices_train if optimize_lambda else None,      # Only pass if lambda optimization enabled
                    irradiation_mode=irradiation_mode,  # NEW: From encoding_methods.py
                    nci_mode=nci_mode                   # NEW: From encoding_methods.py
                    # Note: n_gpus not needed - Optuna uses sklearn MLP (CPU-only)
                )
            else:  # keff
                best_params, study = optimize_keff_model(
                    X_train, y_train,
                    model_type=model_type,
                    n_trials=config.n_trials,
                    n_jobs=n_jobs,              # Use model-specific n_jobs
                    cores_per_trial=cores_per_trial,  # NEW parameter
                    groups=groups_train,
                    encoding=encoding,
                    lattices_train=lattices_train if optimize_lambda else None,      # Only pass if lambda optimization enabled
                    irradiation_mode=irradiation_mode,  # NEW: From encoding_methods.py
                    nci_mode=nci_mode                   # NEW: From encoding_methods.py
                    # Note: n_gpus not needed - Optuna uses sklearn MLP (CPU-only)
                )

            # Check if optimization completed or timed out
            if not best_params:
                print(f"  Optimization failed or timed out. Using default parameters.")
                best_params = self._get_default_params(model_type)
            else:
                print(f" Optimization complete!")
                print(f"  Best parameters found: {best_params}")
                # Capture CV score from Optuna
                if study is not None:
                    best_cv_score = study.best_value
                    print(f"  Best CV MAPE: {best_cv_score:.2f}%")

        elif config.optimization == 'three_stage':
            print(f"Starting three-stage optimization...")

            # NEW: Get model-specific job allocation
            n_jobs, cores_per_trial = get_model_specific_jobs(model_type, config)

            # Three-stage optimization
            model_class = self._get_model_class(model_type, target)
            best_params, search = three_stage_optimization(
                X_train, y_train,
                model_class,
                model_type=model_type,
                n_jobs=n_jobs,              # Use model-specific n_jobs
                cores_per_trial=cores_per_trial,  # NEW parameter
                target_type=target,
                use_log_flux=self.data_handler.use_log_flux if target == 'flux' else False,
                groups=groups_train,  # NEW: Pass groups
                n_gpus=config.n_gpus,  # NEW: Pass GPU count
                encoding=encoding,                  # NEW: Enable lambda optimization
                lattices_train=lattices_train if optimize_lambda else None,      # Only pass if lambda optimization enabled
                irradiation_mode=irradiation_mode,  # NEW: From encoding_methods.py
                nci_mode=nci_mode,                  # NEW: From encoding_methods.py
                skip_grid_search=True               # NEW: Skip grid search for lambda optimization
            )

            # Check if optimization completed or timed out
            if not best_params:
                print(f"  Optimization failed or timed out. Using default parameters.")
                best_params = self._get_default_params(model_type)
            else:
                print(f" Optimization complete!")
                # Capture CV score from Three-Stage
                if search is not None and hasattr(search, 'best_score_'):
                    best_cv_score = abs(search.best_score_)  # Negative MSE, so take abs
                    print(f"  Best CV score: {best_cv_score:.6f}")

        elif config.optimization == 'raytune':
            print(f"Starting Ray Tune optimization...")
            # Ray Tune optimization (neural_net only!)
            if model_type == 'neural_net':
                best_params, analysis = optimize_neural_net_raytune(
                    X_train, y_train,
                    groups=groups_train,
                    n_trials=config.n_trials if hasattr(config, 'n_trials') else 100,
                    n_gpus=config.n_gpus,
                    target_type=target,
                    use_log_flux=self.data_handler.use_log_flux if target == 'flux' else False,
                    encoding=encoding
                )
                print(f" Ray Tune complete!")
                # Capture CV score from Ray Tune
                if analysis is not None:
                    best_trial = analysis.best_trial
                    if best_trial is not None:
                        best_cv_score = best_trial.last_result.get('mape', None)
                        if best_cv_score is not None:
                            print(f"  Best CV MAPE: {best_cv_score:.2f}%")
            else:
                print(f"  Ray Tune only supports neural_net. Using default parameters.")
                best_params = self._get_default_params(model_type)

        elif config.optimization == 'three_stage_neural_net':
            print(f"Starting Three-Stage Neural Net GPU optimization...")
            # Three-Stage Neural Net optimization (neural_net only!)
            if model_type == 'neural_net':
                # Get configuration parameters
                random_iter = config.random_iter if hasattr(config, 'random_iter') else 2000
                bayesian_iter = config.bayesian_iter if hasattr(config, 'bayesian_iter') else 100

                # Update the GPU configuration in the three_stage module
                from hyperparameter_tuning.three_stage_neural_net_gpu import config as gpu_config

                # Set system resources
                if hasattr(config, 'n_jobs'):
                    if config.n_jobs == -1:
                        import multiprocessing
                        n_cpus = multiprocessing.cpu_count()
                    else:
                        n_cpus = config.n_jobs
                else:
                    n_cpus = 48  # Default

                n_gpus = config.n_gpus if hasattr(config, 'n_gpus') else 4

                # Update GPU optimization config
                gpu_config.n_cpus = n_cpus
                gpu_config.n_gpus = n_gpus
                gpu_config.n_parallel_processes = n_cpus * 2  # 2 processes per CPU
                gpu_config.random_n_iter = random_iter
                gpu_config.bayesian_n_iter = bayesian_iter

                print(f"\n  System Configuration:")
                print(f"    CPUs: {n_cpus}")
                print(f"    GPUs: {n_gpus}")
                print(f"    Parallel processes: {gpu_config.n_parallel_processes}")
                print(f"    Random search: {random_iter} iterations")
                print(f"    Bayesian: {bayesian_iter} iterations\n")

                best_params, history = three_stage_neural_net_optimization(
                    X_train, y_train,
                    groups=groups_train,
                    target_type=target,
                    use_log_flux=self.data_handler.use_log_flux if target == 'flux' else False,
                    save_results=True
                )
                print(f" Three-Stage Neural Net GPU optimization complete!")

                # Capture CV score from optimization history
                if history is not None and 'final_best_score' in history:
                    best_cv_score = history['final_best_score']
                    print(f"  Best CV score: {best_cv_score:.4f}")
                    print(f"  Best stage: {history.get('best_stage', 'Unknown')}")
            else:
                print(f"  Three-Stage Neural Net only supports neural_net. Using default parameters.")
                best_params = self._get_default_params(model_type)

        else:  # No optimization
            best_params = self._get_default_params(model_type)
            print(f"  Using default parameters: {best_params}")

        optimization_time = time.time() - optimization_start
        print(f"\nOptimization took {optimization_time/60:.1f} minutes")

        # Train final model
        print(f"\n Training final model...")
        training_start = time.time()

        model = self._create_and_train_model(model_type, target, X_train, y_train, best_params)

        training_time = time.time() - training_start
        print(f"  Final model training took {training_time:.1f} seconds")

        # Evaluate
        # Check if we have a test set (test_size > 0) or if using CV only (test_size = 0)
        if X_test is not None and len(X_test) > 0:
            print(f"\n Evaluating on test set...")
            eval_start = time.time()
            metrics = self._evaluate_model(model, X_test, y_test, target)
            eval_time = time.time() - eval_start
            print(f"  Evaluation took {eval_time:.1f} seconds")
        else:
            # No test set - model was validated via CV during hyperparameter optimization
            print(f"\n Model validated via cross-validation during hyperparameter optimization")
            print(f"  Model trained on {len(X_train)} configurations")
            if best_cv_score is not None:
                print(f"  Best CV MAPE: {best_cv_score:.2f}%")
            print(f"  No held-out test set (test_size=0.0)")
            print(f"  For evaluation on external test data, use test.py with external test configs")

            # Create metrics with CV score
            metrics = {
                'mse': None,
                'rmse': None,
                'mae': None,
                'r2': None,
                'mape': best_cv_score,  # Use CV MAPE
                'relative_error': best_cv_score / 100 if best_cv_score else None,
                'cv_score': best_cv_score,
                'note': 'Model validated via CV. Use test.py for external test evaluation.'
            }

        total_time = time.time() - optimization_start
        print(f"\n Total time for {model_type} {target}: {total_time/60:.1f} minutes")

        return model, metrics, best_params

    def _transform_nn_params(self, params):
        """Transform neural network parameters to PyTorch format.

        PyTorch models use rectangular architecture, so no transformation needed
        for depth/width parameters. This is mainly for backward compatibility.
        """
        # PyTorch models already use the correct parameter names
        # (depth, width, etc.) so just return a copy
        return params.copy()

    def _get_model_class(self, model_type, target):
        """Get appropriate model class for three-stage optimization"""
        from sklearn.multioutput import MultiOutputRegressor
        import xgboost as xgb
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        from ML_models.neural_net_train import PyTorchRegressorWrapper

        if target == 'flux':
            # Multi-output for flux - return lambdas that accept **kwargs
            if model_type == 'xgboost':
                return lambda **kwargs: MultiOutputRegressor(xgb.XGBRegressor(**kwargs))
            elif model_type == 'random_forest':
                # Random Forest has native multi-output support
                return lambda **kwargs: RandomForestRegressor(**kwargs)
            elif model_type == 'svm':
                # CRITICAL FIX: Return raw SVR for optimization
                # The optimization stages will handle Pipeline + MultiOutputRegressor wrapping
                return SVR
            else:  # neural_net
                # PyTorch wrapper already handles multi-output natively
                return PyTorchRegressorWrapper
        else:  # keff - single output
            if model_type == 'xgboost':
                return xgb.XGBRegressor
            elif model_type == 'random_forest':
                return RandomForestRegressor
            elif model_type == 'svm':
                return SVR
            else:  # neural_net
                return PyTorchRegressorWrapper

    def _get_default_params(self, model_type):
        """Get default parameters for each model type - updated with better defaults"""
        defaults = {
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.01,
                'reg_lambda': 0.01,
                'min_child_weight': 1,
                'verbosity': 1
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'verbose': 1
            },
            'svm': {
                'kernel': 'rbf',
                'C': 10.0,
                'gamma': 0.005,
                'epsilon': 0.01,   # CRITICAL FIX: Reduced from 0.1 - large epsilon causes trivial solutions
                'cache_size': 50000,  # Large cache for better performance
                'max_iter': 100000,
                'tol': 1e-4,       # Final model tolerance (0.0001, stricter than optimization tol=1e-3)
                'shrinking': False,
                'verbose': True
            },
            'neural_net': {
                'depth': 2,
                'width': 100,
                'activation': 'relu',
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'weight_decay': 0.001,
                'batch_size': 128,
                'max_epochs': 1000,
                'patience': 20,
                'device': None,  # Auto-detect GPU
                'verbose': False,
                'random_state': 42
            }
        }
        return defaults.get(model_type, {})

    def _create_and_train_model(self, model_type, target, X_train, y_train, params):
        """Create and train the appropriate model using new model classes"""
        # Import the new model classes
        from ML_models import (
            RandomForestReactorModel,
            NeuralNetReactorModel,
            SVMReactorModel,
            XGBoostReactorModel
        )

        # Create the appropriate model
        if model_type == 'xgboost':
            model = XGBoostReactorModel(**params)
        elif model_type == 'random_forest':
            model = RandomForestReactorModel(**params)
        elif model_type == 'svm':
            # IMPORTANT: Since optimization used scaled data via Pipeline,
            # the hyperparameters are optimized for scaled features.
            # SVMReactorModel will handle scaling internally, so we keep it enabled.
            model = SVMReactorModel(**params)
        elif model_type == 'neural_net':
            model = NeuralNetReactorModel(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Set flux mode if it's a flux model
        if target == 'flux' and hasattr(self.data_handler, 'flux_mode'):
            if hasattr(model, 'set_flux_mode'):
                model.set_flux_mode(self.data_handler.flux_mode)
            else:
                # Direct setting for backward compatibility
                if self.data_handler.flux_mode in ['total', 'thermal_only', 'epithermal_only', 'fast_only']:
                    model._n_flux_outputs = 4
                else:
                    model._n_flux_outputs = 12

        # Train the appropriate target with progress tracking
        print(f"  Training {model_type} model...")
        if target == 'flux':
            model.fit_flux(X_train, y_train)
        else:  # keff
            model.fit_keff(X_train, y_train)

        print(f"  Training complete!")

        return model

    def _evaluate_model(self, model, X_test, y_test, target):
        """Evaluate model performance"""
        # Use the model's predict methods
        if target == 'flux':
            predictions = model.predict_flux(X_test)
        else:
            predictions = model.predict_keff(X_test)

        # Get flux mode if available
        flux_mode = self.data_handler.flux_mode if hasattr(self.data_handler, 'flux_mode') else 'total'

        # Calculate metrics
        if target == 'flux' and len(y_test.shape) > 1 and y_test.shape[1] > 1:
            # Multi-output metrics
            if flux_mode == 'bin':
                # Use MSE for bins
                mse = mean_squared_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)

                # No MAPE for bins - use relative MSE instead
                mape = np.sqrt(mse) * 100  # Convert RMSE to percentage-like metric

            else:  # total or energy flux
                # Average metrics across outputs
                n_outputs = y_test.shape[1]
                mse = np.mean([mean_squared_error(y_test[:, i], predictions[:, i])
                            for i in range(n_outputs)])
                mae = np.mean([mean_absolute_error(y_test[:, i], predictions[:, i])
                            for i in range(n_outputs)])
                r2 = np.mean([r2_score(y_test[:, i], predictions[:, i])
                            for i in range(n_outputs)])

                # Calculate MAPE for flux
                if self.data_handler and self.data_handler.use_log_flux:
                    # Convert from log scale to original scale for MAPE
                    y_test_original = 10 ** y_test
                    predictions_original = 10 ** predictions
                    mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100
                else:
                    # Direct MAPE calculation
                    mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-10))) * 100

        else:
            # Single output metrics (k-eff)
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            # MAPE for single output
            mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-10))) * 100

        # Store metrics
        metrics = {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'relative_error': float(mape / 100)  # Keep for backward compatibility
        }

        print(f"  Test MSE: {mse:.6f}")
        print(f"  Test RMSE: {np.sqrt(mse):.6f}")
        print(f"  Test MAE: {mae:.6f}")
        print(f"  Test R²: {r2:.4f}")
        if flux_mode == 'bin' and target == 'flux':
            print(f"  Test RMSE%: {mape:.2f}%")
        else:
            print(f"  Test MAPE: {mape:.2f}%")

        return metrics

    def save_model(self, model, filepath, metadata, model_type, target, encoding, optimization):
        """Save model with correct flux transform metadata"""
        # Get flux transform settings from data_handler
        if self.data_handler:
            use_log_flux = self.data_handler.use_log_flux if target == 'flux' else False
            flux_scale = self.data_handler.flux_scale if not use_log_flux else 1.0
            flux_mode = self.data_handler.flux_mode if hasattr(self.data_handler, 'flux_mode') else 'total'
        else:
            # Fallback values
            use_log_flux = True if target == 'flux' else False
            flux_scale = 1e14
            flux_mode = 'total'

        # Capture irradiation and NCI modes from encoding module (physics encoding only)
        irradiation_mode = 'vacuum'  # Fallback default
        nci_mode = 'single'          # Fallback default
        if encoding == 'physics':
            try:
                from ML_models.encodings.encoding_methods import IRRADIATION_MODE, NCI_MODE
                irradiation_mode = IRRADIATION_MODE
                nci_mode = NCI_MODE
            except:
                pass  # Use fallback defaults if import fails

        # Use the model's own save_model method
        saved_path = model.save_model(
            filepath=filepath,
            model_type=target,  # 'flux' or 'keff'
            encoding=encoding,
            optimization_method=optimization,
            flux_scale=flux_scale,
            use_log_flux=use_log_flux,
            flux_mode=flux_mode,
            irradiation_mode=irradiation_mode,  # NEW
            nci_mode=nci_mode,  # NEW
            **metadata  # Pass any additional metadata
        )

        print(f"\n✓ Model saved:")
        print(f"  Path: {saved_path}")
        print(f"  Metadata:")
        print(f"    - use_log_flux: {use_log_flux}")
        print(f"    - flux_scale: {flux_scale}")
        print(f"    - flux_mode: {flux_mode}")
        if encoding == 'physics':
            print(f"    - irradiation_mode: {irradiation_mode}")
            print(f"    - nci_mode: {nci_mode}")

        return saved_path
