import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from hyperparameter_tuning.optuna_optimization import optimize_flux_model, optimize_keff_model
from hyperparameter_tuning.three_stage_optimization import three_stage_optimization
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

        if target == 'flux':
            y_train = data_splits['y_flux_train']
            y_test = data_splits['y_flux_test']
        else:  # keff
            y_train = data_splits['y_keff_train']
            y_test = data_splits['y_keff_test']

        # Get flux mode
        flux_mode = config.flux_mode if hasattr(config, 'flux_mode') and target == 'flux' else 'total'

        # Get best hyperparameters
        optimization_start = time.time()

        if config.optimization == 'optuna':
            print(f"Starting Optuna optimization...")
            if target == 'flux':
                best_params, study = optimize_flux_model(
                    X_train, y_train,
                    model_type=model_type,
                    n_trials=config.n_trials,
                    n_jobs=config.n_jobs,
                    groups=groups_train,  # NEW: Pass groups
                    flux_mode=flux_mode   # NEW: Pass flux mode
                )
            else:  # keff
                best_params, study = optimize_keff_model(
                    X_train, y_train,
                    model_type=model_type,
                    n_trials=config.n_trials,
                    n_jobs=config.n_jobs,
                    groups=groups_train  # NEW: Pass groups
                )

            # Check if optimization completed or timed out
            if not best_params:
                print(f"  Optimization failed or timed out. Using default parameters.")
                best_params = self._get_default_params(model_type)
            else:
                print(f" Optimization complete!")
                # Transform neural network parameters if needed
                if model_type == 'neural_net' and 'n_layers' in best_params:
                    best_params = self._transform_nn_params(best_params)
                print(f"  Best parameters found: {best_params}")

        elif config.optimization == 'three_stage':
            print(f"Starting three-stage optimization...")
            # Three-stage optimization
            model_class = self._get_model_class(model_type, target)
            best_params, search = three_stage_optimization(
                X_train, y_train,
                model_class,
                model_type=model_type,
                n_jobs=config.n_jobs,
                target_type=target,
                use_log_flux=self.data_handler.use_log_flux if target == 'flux' else False,
                groups=groups_train  # NEW: Pass groups
            )

            # Check if optimization completed or timed out
            if not best_params:
                print(f"  Optimization failed or timed out. Using default parameters.")
                best_params = self._get_default_params(model_type)
            else:
                print(f" Optimization complete!")
                # Transform neural network parameters if needed
                if model_type == 'neural_net' and 'n_layers' in best_params:
                    best_params = self._transform_nn_params(best_params)

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
        print(f"\n Evaluating on test set...")
        eval_start = time.time()

        metrics = self._evaluate_model(model, X_test, y_test, target)

        eval_time = time.time() - eval_start
        print(f"  Evaluation took {eval_time:.1f} seconds")

        total_time = time.time() - optimization_start
        print(f"\n Total time for {model_type} {target}: {total_time/60:.1f} minutes")

        return model, metrics, best_params

    def _transform_nn_params(self, params):
        """Transform Optuna neural network parameters to MLPRegressor format"""
        transformed = {}

        # Extract n_layers if present
        if 'n_layers' in params:
            n_layers = params['n_layers']
            # Build the hidden_layer_sizes tuple
            layers = []
            for i in range(n_layers):
                layer_key = f'layer_{i}_size'
                if layer_key in params:
                    layers.append(params[layer_key])

            transformed['hidden_layer_sizes'] = tuple(layers)

            # Copy other parameters (excluding layer-specific ones)
            for key, value in params.items():
                if not key.startswith('layer_') and key != 'n_layers':
                    transformed[key] = value
        else:
            # If no n_layers, params are already in correct format
            transformed = params.copy()

        return transformed

    def _get_model_class(self, model_type, target):
        """Get appropriate model class for three-stage optimization"""
        from sklearn.multioutput import MultiOutputRegressor
        import xgboost as xgb
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        from sklearn.neural_network import MLPRegressor

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
                return lambda **kwargs: MultiOutputRegressor(MLPRegressor(**kwargs))
        else:  # keff - single output
            if model_type == 'xgboost':
                return xgb.XGBRegressor
            elif model_type == 'random_forest':
                return RandomForestRegressor
            elif model_type == 'svm':
                return SVR
            else:  # neural_net
                return MLPRegressor

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
                'gamma': 0.01,
                'epsilon': 0.1,
                'cache_size': 1000,
                'max_iter': 100000,
                'tol': 1e-4,
                'shrinking': False,
                'verbose': True
            },
            'neural_net': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'learning_rate_init': 0.001,
                'max_iter': 1000,
                'early_stopping': True,
                'n_iter_no_change': 20,
                'verbose': True
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

        # Use the model's own save_model method
        saved_path = model.save_model(
            filepath=filepath,
            model_type=target,  # 'flux' or 'keff'
            encoding=encoding,
            optimization_method=optimization,
            flux_scale=flux_scale,
            use_log_flux=use_log_flux,
            flux_mode=flux_mode,  # NEW
            **metadata  # Pass any additional metadata
        )

        print(f"\n Model saved:")
        print(f"  Path: {saved_path}")
        print(f"  Flux metadata:")
        print(f"    - use_log_flux: {use_log_flux}")
        print(f"    - flux_scale: {flux_scale}")
        print(f"    - flux_mode: {flux_mode}")

        return saved_path
