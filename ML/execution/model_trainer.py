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

        if (
            model_type == 'neural_net'
            and config.optimization == 'raytune'
            and target == 'flux'
        ):
            nn_cfg = getattr(config, 'nn_config', None)
            if nn_cfg is None:
                nn_cfg = int(os.environ.get('NN_CONFIG', '1'))
            fm = getattr(config, 'flux_mode', 'total')
            if fm == 'energy_sixteen' and 1 <= nn_cfg <= 5:
                return self._train_nn_ray_thesis(
                    data_splits, config, encoding, model_type, target,
                    X_train, y_train, y_test, X_test, groups_train,
                    lattices_train, flux_mode, irradiation_mode, nci_mode,
                    optimize_lambda, optimization_start, nn_cfg)

        if config.optimization == 'optuna':
            print(f"Starting Optuna optimization...")

            # NEW: Get model-specific job allocation
            n_jobs, cores_per_trial = get_model_specific_jobs(model_type, config)

            if target == 'flux':
                # Determine flux loss function from environment variable (default: MAPE)
                flux_loss = os.environ.get('FLUX_LOSS', 'mape').lower()

                # Check if restructured mode applies for this model
                use_restructured = (getattr(config, 'use_restructured', False)
                                    and model_type == 'xgboost')

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
                    nci_mode=nci_mode,                  # NEW: From encoding_methods.py
                    flux_loss=flux_loss,                # NEW: MSE or MAPE
                    use_restructured=use_restructured,  # Position pooling
                    nci_disabled=(NCI_DISTANCE_CUTOFF == 2),
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
                    uses_mse = (target == 'keff'
                                or (target == 'flux' and (use_restructured or flux_loss == 'mse')))
                    if uses_mse:
                        print(f"  Best CV MSE: {best_cv_score:.6f}")
                    else:
                        print(f"  Best CV MAPE: {best_cv_score:.2f}%")

        elif config.optimization == 'three_stage':
            print(f"Starting three-stage optimization...")

            # NEW: Get model-specific job allocation
            n_jobs, cores_per_trial = get_model_specific_jobs(model_type, config)

            # Three-stage optimization
            model_class = self._get_model_class(model_type, target)

            # Determine flux loss function from environment variable (default: MAPE)
            flux_loss = os.environ.get('FLUX_LOSS', 'mape').lower() if target == 'flux' else 'mse'

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
                skip_grid_search=True,              # NEW: Skip grid search for lambda optimization
                flux_loss=flux_loss                 # NEW: MSE or MAPE
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

        # Extract lambda params from best_params and apply to features
        lambda_param_names = ['lambda_P', 'lambda_B', 'lambda_G', 'lambda_decay']
        best_lambdas = {k: best_params[k] for k in lambda_param_names if k in best_params}
        model_params = {k: v for k, v in best_params.items() if k not in lambda_param_names}

        if best_lambdas and optimize_lambda:
            print(f"\n  Applying optimized lambda values to features:")
            for k, v in best_lambdas.items():
                print(f"    {k}: {v:.3f}")

            from utils.lambda_feature_regenerator import LambdaFeatureRegenerator
            feature_regenerator = LambdaFeatureRegenerator(encoding)
            feature_data_train = feature_regenerator.separate_features(
                X_train, lattices_train, irradiation_mode, nci_mode)
            X_train = feature_regenerator.regenerate_features(
                feature_data_train, **best_lambdas)
            print(f"  Regenerated training features with optimized lambdas")

            if X_test is not None and len(X_test) > 0:
                lattices_test = data_splits.get('lattices_test', None)
                if lattices_test is not None:
                    feature_data_test = feature_regenerator.separate_features(
                        X_test, lattices_test, irradiation_mode, nci_mode)
                    X_test = feature_regenerator.regenerate_features(
                        feature_data_test, **best_lambdas)
                    print(f"  Regenerated test features with optimized lambdas")
                else:
                    print(f"  WARNING: No lattices_test available -- test data uses default lambdas")

            # Apply lambdas to Test Set C as well
            lattices_test_c = data_splits.get('lattices_test_c', None)
            X_test_c = data_splits.get('X_test_c', None)
            if X_test_c is not None and len(X_test_c) > 0 and lattices_test_c is not None:
                if len(lattices_test_c) != X_test_c.shape[0]:
                    raise ValueError(
                        f"lattices_test_c length ({len(lattices_test_c)}) != X_test_c rows "
                        f"({X_test_c.shape[0]}). Set C features/targets would be misaligned."
                    )
                feature_data_test_c = feature_regenerator.separate_features(
                    X_test_c, lattices_test_c, irradiation_mode, nci_mode)
                data_splits['X_test_c'] = feature_regenerator.regenerate_features(
                    feature_data_test_c, **best_lambdas)
                print(f"  Regenerated Test Set C features with optimized lambdas")
            elif (
                X_test_c is not None
                and len(X_test_c) > 0
                and lattices_test_c is None
            ):
                print(
                    "\n  ⚠️  CRITICAL: Test Set C has no lattices in data_splits — "
                    "CANNOT apply optimized lambdas to Set C.\n"
                    "      Model was trained on lambda-regenerated NCI features; "
                    "Set C still uses DEFAULT NCI from the initial encoder.\n"
                    "      Set C metrics will be systematically biased (usually worse). "
                    "Fix load_and_prepare_data to return augmented_lattices for test_c."
                )

        # Restructure training data for final model if needed
        use_restructured_final = (getattr(config, 'use_restructured', False)
                                  and target == 'flux' and model_type == 'xgboost')
        nci_disabled = (NCI_DISTANCE_CUTOFF == 2)
        if use_restructured_final:
            from utils.data_restructure import restructure_for_position_pooling
            X_train_final, y_train_final, _ = restructure_for_position_pooling(
                X_train, y_train, groups_train, irradiation_mode, nci_mode,
                nci_disabled=nci_disabled)
            print(f"  Restructured training data: {X_train.shape} -> {X_train_final.shape}")
        else:
            X_train_final, y_train_final = X_train, y_train

        # Train final model
        print(f"\n Training final model...")
        training_start = time.time()

        model = self._create_and_train_model(
            model_type, target, X_train_final, y_train_final, model_params,
            use_restructured=use_restructured_final,
            irradiation_mode=irradiation_mode, nci_mode=nci_mode,
            nci_disabled=nci_disabled)

        training_time = time.time() - training_start
        print(f"  Final model training took {training_time:.1f} seconds")

        # Evaluate
        # Check if we have a test set (test_size > 0) or if using CV only (test_size = 0)
        if X_test is not None and len(X_test) > 0:
            print(f"\n Evaluating on test set...")
            eval_start = time.time()

            fill_categories_test = data_splits.get('fill_categories_test', None)
            test_split_run = data_splits.get('test_split_run', None)

            metrics = self._evaluate_model(
                model, X_test, y_test, target,
                fill_categories=fill_categories_test,
                test_split_run=test_split_run)
            eval_time = time.time() - eval_start
            print(f"  Evaluation took {eval_time:.1f} seconds")

            # Evaluate on Test Set C (random cores) if available
            X_test_c = data_splits.get('X_test_c', None)
            if X_test_c is not None and len(X_test_c) > 0:
                y_test_c = data_splits.get('y_flux_test_c' if target == 'flux' else 'y_keff_test_c')
                fc_test_c = data_splits.get('fill_categories_test_c', None)
                test_c_metrics = self._evaluate_test_set_c(
                    model, X_test_c, y_test_c, target, fill_categories=fc_test_c)
                metrics['test_set_c'] = test_c_metrics
        else:
            # No test set - model was validated via CV during hyperparameter optimization
            print(f"\n Model validated via cross-validation during hyperparameter optimization")
            print(f"  Model trained on {len(X_train_final)} samples")
            if best_cv_score is not None:
                if use_restructured_final:
                    print(f"  Best CV MSE: {best_cv_score:.6f} (log-scale)")
                else:
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

    def _create_and_train_model(self, model_type, target, X_train, y_train, params,
                               use_restructured=False, irradiation_mode='fill',
                               nci_mode='separate', nci_disabled=False):
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
            model = XGBoostReactorModel(
                use_restructured=use_restructured,
                irradiation_mode=irradiation_mode,
                nci_mode=nci_mode,
                nci_disabled=nci_disabled,
                **params)
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
                elif self.data_handler.flux_mode == 'energy_sixteen':
                    model._n_flux_outputs = 16
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

    def _compute_metrics_for_subset(self, y_true, y_pred, target):
        """Compute MSE, MAPE, R^2 for a subset of test data.

        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        target : str
            'flux' or 'keff'

        Returns
        -------
        dict
            Dictionary with mse, rmse, mae, r2, mape keys
        """
        flux_mode = self.data_handler.flux_mode if hasattr(self.data_handler, 'flux_mode') else 'total'

        if len(y_true) == 0:
            return {
                'mse': None, 'rmse': None, 'mae': None, 'r2': None, 'mape': None,
                'mse_log': None, 'n_samples': 0,
            }

        mse_log = None
        if target == 'flux' and len(y_true.shape) > 1 and y_true.shape[1] > 1:
            if flux_mode == 'bin':
                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
                mape = np.sqrt(mse) * 100
            else:
                n_outputs = y_true.shape[1]
                mse = np.mean([mean_squared_error(y_true[:, i], y_pred[:, i])
                            for i in range(n_outputs)])
                mae = np.mean([mean_absolute_error(y_true[:, i], y_pred[:, i])
                            for i in range(n_outputs)])
                r2 = np.mean([r2_score(y_true[:, i], y_pred[:, i])
                            for i in range(n_outputs)]) if len(y_true) > 1 else 0.0

                if self.data_handler and self.data_handler.use_log_flux:
                    y_orig = 10 ** y_true
                    p_orig = 10 ** y_pred
                    mape = np.mean(np.abs((y_orig - p_orig) / y_orig)) * 100
                else:
                    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

            if flux_mode == 'energy_sixteen' and self.data_handler and self.data_handler.use_log_flux:
                mse_log = float(mean_squared_error(y_true, y_pred))
        else:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

        out = {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'n_samples': int(len(y_true)),
        }
        if mse_log is not None:
            out['mse_log'] = mse_log
        return out

    def _evaluate_model(self, model, X_test, y_test, target,
                        fill_categories=None, test_split_run=None):
        """Evaluate model performance with optional test-set splitting and per-fill-type breakdown.

        Parameters
        ----------
        model : object
            Trained model
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets
        target : str
            'flux' or 'keff'
        fill_categories : np.ndarray, optional
            Fill type category for each test sample (e.g., '4G', '3G1P', etc.)
        test_split_run : int, optional
            RUN number at which to split test data into Set A and Set B.
            Set A = RUNs 1..test_split_run, Set B = RUNs test_split_run+1..end.
            Split index = test_split_run * 8 (due to 8-fold augmentation).
        """
        if target == 'flux':
            predictions = model.predict_flux(X_test)
        else:
            predictions = model.predict_keff(X_test)

        flux_mode = self.data_handler.flux_mode if hasattr(self.data_handler, 'flux_mode') else 'total'

        # Overall metrics
        overall = self._compute_metrics_for_subset(y_test, predictions, target)
        overall['relative_error'] = float(overall['mape'] / 100) if overall['mape'] else None

        print(f"  Test MSE: {overall['mse']:.10f}")
        if overall.get('mse_log') is not None:
            print(f"  Test MSE (log space, all outputs): {overall['mse_log']:.10f}")
        print(f"  Test RMSE: {overall['rmse']:.10f}")
        print(f"  Test MAE: {overall['mae']:.10f}")
        print(f"  Test R²: {overall['r2']:.6f}")
        if flux_mode == 'bin' and target == 'flux':
            print(f"  Test RMSE%: {overall['mape']:.8f}%")
        else:
            print(f"  Test MAPE: {overall['mape']:.8f}%")

        metrics = {'overall': overall}
        metrics.update(overall)  # Flat access for backward compatibility with results_manager

        # Split test evaluation if boundary provided
        if test_split_run is not None:
            split_idx = test_split_run * 8
            if split_idx < len(y_test):
                # Test Set A (random lattice configs)
                y_a, p_a = y_test[:split_idx], predictions[:split_idx]
                X_a = X_test[:split_idx]
                fc_a = fill_categories[:split_idx] if fill_categories is not None else None

                # Test Set B (self-adjusted configs)
                y_b, p_b = y_test[split_idx:], predictions[split_idx:]
                X_b = X_test[split_idx:]
                fc_b = fill_categories[split_idx:] if fill_categories is not None else None

                metrics_a = self._compute_metrics_for_subset(y_a, p_a, target)
                metrics_b = self._compute_metrics_for_subset(y_b, p_b, target)

                print(f"\n  --- Test Set A (random lattice, {metrics_a['n_samples']} samples) ---")
                print(f"  MSE: {metrics_a['mse']:.10f}  R²: {metrics_a['r2']:.6f}  MAPE: {metrics_a['mape']:.8f}%")

                print(f"\n  --- Test Set B (self-adjusted, {metrics_b['n_samples']} samples) ---")
                print(f"  MSE: {metrics_b['mse']:.10f}  R²: {metrics_b['r2']:.6f}  MAPE: {metrics_b['mape']:.8f}%")

                # Per-fill-type breakdown for each test set
                fill_types = ['4G', '3G1P', '3G1B', '2G2P', '2G2B', '2G1P1B']

                for set_label, y_set, p_set, fc_set in [
                    ('test_set_a', y_a, p_a, fc_a),
                    ('test_set_b', y_b, p_b, fc_b)
                ]:
                    set_metrics = {'overall': self._compute_metrics_for_subset(y_set, p_set, target)}

                    if fc_set is not None:
                        set_name = "Set A (random)" if set_label == 'test_set_a' else "Set B (self-adj)"
                        print(f"\n  Per-fill-type breakdown for {set_name}:")
                        for ft in fill_types:
                            mask = (fc_set == ft)
                            if np.any(mask):
                                ft_metrics = self._compute_metrics_for_subset(
                                    y_set[mask], p_set[mask], target)
                                set_metrics[ft] = ft_metrics
                                print(f"    {ft:>7s}: MSE={ft_metrics['mse']:.10f}  "
                                      f"R²={ft_metrics['r2']:.6f}  "
                                      f"MAPE={ft_metrics['mape']:.8f}%  "
                                      f"(n={ft_metrics['n_samples']})")
                            else:
                                set_metrics[ft] = {'mse': None, 'r2': None, 'mape': None, 'n_samples': 0}

                    metrics[set_label] = set_metrics

        return metrics

    def _evaluate_test_set_c(self, model, X_test, y_test, target, fill_categories=None):
        """Evaluate model on Test Set C (random cores from test_c.txt).

        Same per-fill-type breakdown as Sets A/B but no internal splitting.
        """
        if target == 'flux':
            predictions = model.predict_flux(X_test)
        else:
            predictions = model.predict_keff(X_test)

        overall = self._compute_metrics_for_subset(y_test, predictions, target)

        print(f"\n  --- Test Set C (random cores, {overall['n_samples']} samples) ---")
        print(f"  MSE: {overall['mse']:.10f}  R²: {overall['r2']:.6f}  MAPE: {overall['mape']:.8f}%")

        set_metrics = {'overall': overall}

        if fill_categories is not None:
            fill_types = ['4G', '3G1P', '3G1B', '2G2P', '2G2B', '2G1P1B']
            print(f"\n  Per-fill-type breakdown for Set C (random cores):")
            for ft in fill_types:
                mask = (fill_categories == ft)
                if np.any(mask):
                    ft_metrics = self._compute_metrics_for_subset(
                        y_test[mask], predictions[mask], target)
                    set_metrics[ft] = ft_metrics
                    print(f"    {ft:>7s}: MSE={ft_metrics['mse']:.10f}  "
                          f"R²={ft_metrics['r2']:.6f}  "
                          f"MAPE={ft_metrics['mape']:.8f}%  "
                          f"(n={ft_metrics['n_samples']})")
                else:
                    set_metrics[ft] = {'mse': None, 'r2': None, 'mape': None, 'n_samples': 0}

        return set_metrics

    def _train_nn_ray_thesis(
        self,
        data_splits,
        config,
        encoding,
        model_type,
        target,
        X_train,
        y_train,
        y_test,
        X_test,
        groups_train,
        lattices_train,
        flux_mode,
        irradiation_mode,
        nci_mode,
        optimize_lambda,
        optimization_start,
        nn_cfg,
    ):
        """Thesis NN configs 1–5 with joint Ray Tune + lambda (``energy_sixteen`` flux only).

        When ``config.flux_group`` is set (0–3) and the config is multi-study
        (2, 3, 5), only that single flux-group HPO is executed.  The resulting
        submodel(s) are persisted to a deterministic path under
        ``config.models_dir`` and the method returns early (no composite
        assembly, no evaluation).
        """
        from ML_models.encodings.encoding_methods import NCI_DISTANCE_CUTOFF
        from neural_net_configs.composite import NeuralNetCompositeThesis
        from neural_net_configs.data_pipeline import (
            prepare_xy_after_lambda,
            slice_y_for_flux_group,
            slice_y_single_target,
        )
        from utils.lambda_feature_regenerator import LambdaFeatureRegenerator

        nci_disabled = (NCI_DISTANCE_CUTOFF == 2)
        n_trials = config.n_trials if hasattr(config, 'n_trials') else 100
        n_gpus = config.n_gpus if hasattr(config, 'n_gpus') else 1
        flux_group = getattr(config, 'flux_group', None)
        models_dir = getattr(config, 'models_dir', 'outputs/models')

        def run_one_ray_tune(flux_group_index=None, y_override=None):
            yt = y_override if y_override is not None else y_train
            return optimize_neural_net_raytune(
                X_train,
                yt,
                groups=groups_train,
                n_trials=n_trials,
                n_gpus=n_gpus,
                target_type='flux',
                use_log_flux=self.data_handler.use_log_flux,
                encoding=encoding,
                lattices_train=lattices_train,
                irradiation_mode=irradiation_mode,
                nci_mode=nci_mode,
                nci_disabled=nci_disabled,
                nn_config=nn_cfg,
                flux_mode=flux_mode,
                optimize_lambda=optimize_lambda,
                flux_group_index=flux_group_index,
                score_metric='mse_log',
            )

        def apply_lambdas_to_matrices(best_lambdas, mutate_splits=True):
            """Regenerate X from λ. If ``mutate_splits`` is True, update test matrices in ``data_splits``."""
            if not best_lambdas or not optimize_lambda:
                return X_train, X_test
            fr = LambdaFeatureRegenerator(encoding)
            fd = fr.separate_features(X_train, lattices_train, irradiation_mode, nci_mode)
            Xt = fr.regenerate_features(fd, **best_lambdas)
            Xte = X_test
            if X_test is not None and len(X_test) > 0:
                lt = data_splits.get('lattices_test', None)
                if lt is not None:
                    fd_te = fr.separate_features(X_test, lt, irradiation_mode, nci_mode)
                    Xte = fr.regenerate_features(fd_te, **best_lambdas)
            if mutate_splits:
                if Xte is not None and len(Xte) > 0:
                    data_splits['X_test'] = Xte
                ltc = data_splits.get('lattices_test_c', None)
                Xtc = data_splits.get('X_test_c', None)
                if Xtc is not None and len(Xtc) > 0 and ltc is not None:
                    fd_c = fr.separate_features(Xtc, ltc, irradiation_mode, nci_mode)
                    data_splits['X_test_c'] = fr.regenerate_features(
                        fd_c, **best_lambdas)
            return Xt, Xte

        def strip_lambda(bp):
            names = ['lambda_P', 'lambda_B', 'lambda_G', 'lambda_decay']
            lam = {k: bp[k] for k in names if k in bp}
            mp = {k: v for k, v in bp.items() if k not in names}
            return lam, mp

        def fit_nn(mp, Xt, yt, gr, use_rest, nn_c, lambdas_for_eval=None):
            m = NeuralNetReactorModel(
                nn_config=nn_c,
                use_restructured=use_rest,
                irradiation_mode=irradiation_mode,
                nci_mode=nci_mode,
                nci_disabled=nci_disabled,
                **mp,
            )
            m.set_flux_mode('energy_sixteen')
            m.fit_flux(Xt, yt, groups=gr)
            m.regen_lambdas_ = lambdas_for_eval
            return m

        def _save_submodel(model, tag):
            """Save a submodel with a deterministic name under *models_dir*."""
            os.makedirs(models_dir, exist_ok=True)
            path = os.path.join(models_dir, f'nn_config{nn_cfg}_{tag}_submodel.pkl')
            model.save_model(
                filepath=path,
                model_type='flux',
                encoding=encoding,
                optimization_method='raytune',
                flux_scale=1.0,
                use_log_flux=self.data_handler.use_log_flux,
                flux_mode='energy_sixteen',
                irradiation_mode=irradiation_mode,
                nci_mode=nci_mode,
            )
            print(f"  Submodel saved → {path}")
            return path

        # ── single-group mode for configs 2, 3, 5 ──────────────────────
        if nn_cfg in (2, 3, 5) and flux_group is not None:
            g = flux_group
            print(f"\n{'='*60}")
            print(f"Single-HPO mode: nn_config={nn_cfg}, flux_group={g}")
            print(f"{'='*60}")

            if nn_cfg == 2:
                best_params, _ = run_one_ray_tune(flux_group_index=g)
                lam, mp = strip_lambda(best_params)
                Xt, _ = apply_lambdas_to_matrices(lam, mutate_splits=False)
                y_sub = slice_y_for_flux_group(y_train, g)
                sub = fit_nn(mp, Xt, y_sub, groups_train, False, 2, lam if lam else None)
                _save_submodel(sub, f'group{g}')

            elif nn_cfg == 5:
                best_params, _ = run_one_ray_tune(flux_group_index=g)
                lam, mp = strip_lambda(best_params)
                Xt, _ = apply_lambdas_to_matrices(lam, mutate_splits=False)
                Xt, y1, gr = prepare_xy_after_lambda(
                    Xt, y_train, groups_train, 5, irradiation_mode, nci_mode,
                    nci_disabled, flux_group_index=g,
                )
                sub = fit_nn(mp, Xt, y1, gr, True, 5, lam if lam else None)
                _save_submodel(sub, f'group{g}')

            else:  # nn_cfg == 3
                best_params, _ = run_one_ray_tune(
                    y_override=slice_y_single_target(y_train, 0, g))
                lam, mp = strip_lambda(best_params)
                Xt, _ = apply_lambdas_to_matrices(lam, mutate_splits=False)
                for pos in range(4):
                    y_col = slice_y_single_target(y_train, pos, g)
                    sub = fit_nn(mp, Xt, y_col, groups_train, False, 3, lam if lam else None)
                    _save_submodel(sub, f'group{g}_pos{pos}')

            total_time = time.time() - optimization_start
            print(f"\n Submodel(s) saved.  Time: {total_time/60:.1f} min")
            print("Run assemble_composite.py after all groups finish to build & evaluate.")
            return None, {}, {'nn_config': nn_cfg, 'flux_group': g}

        # ── full-run mode (all groups in one process) ───────────────────
        best_cv_score = None
        submodels = []
        all_best = {}

        if nn_cfg in (1, 4):
            best_params, analysis = run_one_ray_tune()
            if analysis is not None:
                bt = analysis.get_best_trial(metric='score', mode='min')
                if bt is not None and bt.last_result:
                    best_cv_score = bt.last_result.get('score')
            lam, mp = strip_lambda(best_params)
            Xt, _ = apply_lambdas_to_matrices(lam, mutate_splits=True)
            y_tr = y_train
            gr = groups_train
            use_rest = nn_cfg == 4
            if use_rest:
                Xt, y_tr, gr = prepare_xy_after_lambda(
                    Xt, y_train, groups_train, 4, irradiation_mode, nci_mode, nci_disabled
                )
            model = fit_nn(mp, Xt, y_tr, gr, use_rest, nn_cfg, lam if lam else None)
            all_best = best_params

        elif nn_cfg == 2:
            for g in range(4):
                best_params, analysis = run_one_ray_tune(flux_group_index=g)
                lam, mp = strip_lambda(best_params)
                Xt, _ = apply_lambdas_to_matrices(lam, mutate_splits=False)
                y_sub = slice_y_for_flux_group(y_train, g)
                submodels.append(
                    fit_nn(mp, Xt, y_sub, groups_train, False, 2, lam if lam else None))
            model = NeuralNetCompositeThesis(
                2, submodels, encoding, irradiation_mode, nci_mode, nci_disabled,
                optimize_lambda=optimize_lambda,
            )
            model.set_flux_mode('energy_sixteen')

        elif nn_cfg == 5:
            for g in range(4):
                best_params, _ = run_one_ray_tune(flux_group_index=g)
                lam, mp = strip_lambda(best_params)
                Xt, _ = apply_lambdas_to_matrices(lam, mutate_splits=False)
                Xt, y1, gr = prepare_xy_after_lambda(
                    Xt, y_train, groups_train, 5, irradiation_mode, nci_mode,
                    nci_disabled, flux_group_index=g,
                )
                submodels.append(
                    fit_nn(mp, Xt, y1, gr, True, 5, lam if lam else None))
            model = NeuralNetCompositeThesis(
                5, submodels, encoding, irradiation_mode, nci_mode, nci_disabled,
                optimize_lambda=optimize_lambda,
            )
            model.set_flux_mode('energy_sixteen')

        else:
            # nn_cfg == 3: sixteen independent single-output nets
            # Thesis HPO: 4 runs grouped by flux type — all 4 position
            # networks within the same flux group share hyperparameters.
            for g in range(4):
                best_params, _ = run_one_ray_tune(
                    y_override=slice_y_single_target(y_train, 0, g))
                lam, mp = strip_lambda(best_params)
                Xt, _ = apply_lambdas_to_matrices(lam, mutate_splits=False)
                for pos in range(4):
                    y_col = slice_y_single_target(y_train, pos, g)
                    submodels.append(
                        fit_nn(mp, Xt, y_col, groups_train, False, 3, lam if lam else None))
            model = NeuralNetCompositeThesis(
                3, submodels, encoding, irradiation_mode, nci_mode, nci_disabled,
                optimize_lambda=optimize_lambda,
            )
            model.set_flux_mode('energy_sixteen')

        X_test_eval = data_splits.get('X_test', X_test)
        metrics = {}
        if X_test_eval is not None and len(X_test_eval) > 0:
            fill_categories_test = data_splits.get('fill_categories_test', None)
            test_split_run = data_splits.get('test_split_run', None)
            if isinstance(model, NeuralNetCompositeThesis):
                model._lattices_eval = data_splits.get('lattices_test', None)
            metrics = self._evaluate_model(
                model, X_test_eval, y_test, target,
                fill_categories=fill_categories_test,
                test_split_run=test_split_run)
            X_test_c = data_splits.get('X_test_c', None)
            if X_test_c is not None and len(X_test_c) > 0:
                y_tc = data_splits.get('y_flux_test_c', None)
                fc_tc = data_splits.get('fill_categories_test_c', None)
                if isinstance(model, NeuralNetCompositeThesis):
                    model._lattices_eval = data_splits.get('lattices_test_c', None)
                metrics['test_set_c'] = self._evaluate_test_set_c(
                    model, X_test_c, y_tc, target, fill_categories=fc_tc)

        total_time = time.time() - optimization_start
        print(f"\n Total time for neural_net thesis nn_config={nn_cfg}: {total_time/60:.1f} minutes")

        return model, metrics, all_best if nn_cfg in (1, 4) else {'nn_config': nn_cfg}

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
