import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from hyperparameter_tuning.optuna_optimization import optimize_flux_model, optimize_keff_model
from hyperparameter_tuning.three_stage_optimization import three_stage_optimization
import joblib
import os

class ModelTrainer:
    """Handle model training and evaluation"""

    def __init__(self, data_handler=None):
        # Store reference to data_handler to access flux transform settings
        self.data_handler = data_handler

    def train_model(self, model_type, target, data_splits, config, encoding):
        """Train a single model with hyperparameter optimization"""

        # Get appropriate data
        X_train = data_splits['X_train']
        X_test = data_splits['X_test']

        if target == 'flux':
            y_train = data_splits['y_flux_train']
            y_test = data_splits['y_flux_test']
        else:  # keff
            y_train = data_splits['y_keff_train']
            y_test = data_splits['y_keff_test']

        # Get best hyperparameters
        if config.optimization == 'optuna':
            if target == 'flux':
                best_params, study = optimize_flux_model(
                    X_train, y_train,
                    model_type=model_type,
                    n_trials=config.n_trials,
                    n_jobs=config.n_jobs
                )
            else:  # keff
                best_params, study = optimize_keff_model(
                    X_train, y_train,
                    model_type=model_type,
                    n_trials=config.n_trials,
                    n_jobs=config.n_jobs
                )
            print(f"  Best parameters found: {best_params}")

        elif config.optimization == 'three_stage':
            # Three-stage optimization
            model_class = self._get_model_class(model_type, target)
            best_params, search = three_stage_optimization(
                X_train, y_train,
                model_class,
                model_type=model_type,
                n_jobs=config.n_jobs
            )

        else:  # No optimization
            best_params = self._get_default_params(model_type)
            print(f"  Using default parameters: {best_params}")

        # Train final model
        print(f"  Training final model...")
        model = self._create_and_train_model(model_type, target, X_train, y_train, best_params)

        # Evaluate
        print(f"  Evaluating on test set...")
        metrics = self._evaluate_model(model, X_test, y_test, target)

        return model, metrics, best_params

    def _get_model_class(self, model_type, target):
        """Get appropriate model class for three-stage optimization"""
        from sklearn.multioutput import MultiOutputRegressor
        import xgboost as xgb
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        from sklearn.neural_network import MLPRegressor

        if target == 'flux':
            # Multi-output for flux
            if model_type == 'xgboost':
                return lambda: MultiOutputRegressor(xgb.XGBRegressor())
            elif model_type == 'random_forest':
                return lambda: RandomForestRegressor()
            elif model_type == 'svm':
                return lambda: MultiOutputRegressor(SVR())
            else:  # neural_net
                return lambda: MultiOutputRegressor(MLPRegressor())
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
        """Get default parameters for each model type"""
        defaults = {
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            },
            'svm': {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'epsilon': 0.1
            },
            'neural_net': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'learning_rate_init': 0.001
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
            model = SVMReactorModel(**params)
        elif model_type == 'neural_net':
            model = NeuralNetReactorModel(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train the appropriate target
        if target == 'flux':
            model.fit_flux(X_train, y_train)
        else:  # keff
            model.fit_keff(X_train, y_train)

        return model

    def _evaluate_model(self, model, X_test, y_test, target):
        """Evaluate model performance"""
        # Use the model's predict methods
        if target == 'flux':
            predictions = model.predict_flux(X_test)
        else:
            predictions = model.predict_keff(X_test)

        # Calculate metrics
        if target == 'flux' and len(y_test.shape) > 1 and y_test.shape[1] > 1:
            # Multi-output metrics - average across positions
            mse = np.mean([mean_squared_error(y_test[:, i], predictions[:, i])
                        for i in range(y_test.shape[1])])
            mae = np.mean([mean_absolute_error(y_test[:, i], predictions[:, i])
                        for i in range(y_test.shape[1])])
            r2 = np.mean([r2_score(y_test[:, i], predictions[:, i])
                        for i in range(y_test.shape[1])])

            # Calculate MAPE for flux
            # Check if log transform was used
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
        print(f"  Test R²: {r2:.4f}")
        print(f"  Test MAPE: {mape:.2f}%")

        return metrics

    def save_model(self, model, filepath, metadata, model_type, target, encoding, optimization):
        """Save model with correct flux transform metadata"""
        # Get flux transform settings from data_handler
        if self.data_handler:
            use_log_flux = self.data_handler.use_log_flux if target == 'flux' else False
            flux_scale = self.data_handler.flux_scale if not use_log_flux else 1.0
        else:
            # Fallback values
            use_log_flux = True if target == 'flux' else False
            flux_scale = 1e14

        # Use the model's own save_model method
        saved_path = model.save_model(
            filepath=filepath,
            model_type=target,  # 'flux' or 'keff'
            encoding=encoding,
            optimization_method=optimization,
            flux_scale=flux_scale,
            use_log_flux=use_log_flux,
            **metadata  # Pass any additional metadata
        )

        print(f"  Model saved with flux metadata:")
        print(f"    - use_log_flux: {use_log_flux}")
        print(f"    - flux_scale: {flux_scale}")

        return saved_path
