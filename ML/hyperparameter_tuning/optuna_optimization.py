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

def optimize_flux_model(X_train, y_flux_train, model_type='xgboost', n_trials=250, n_jobs=10):
    """Optimize hyperparameters for flux prediction only"""

    def objective(trial):
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
                'n_jobs': 1
            }
            model = MultiOutputRegressor(xgb.XGBRegressor(**params))

        elif model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
                'n_jobs': 1
            }
            model = RandomForestRegressor(**params)

        elif model_type == 'svm':
            params = {
                'C': trial.suggest_float('C', 0.001, 1000, log=True),
                'gamma': trial.suggest_float('gamma', 0.0001, 1, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.001, 1.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
            }
            if params['kernel'] == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)
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
                'verbose': 2,
            }
            # model = MultiOutputRegressor(MLPRegressor(**params))
            model = MLPRegressor(**params)

        # Train and evaluate on flux only
        scores = cross_val_score(model, X_train, y_flux_train, cv=10,
                               scoring='neg_mean_squared_error', n_jobs=1)
        return -scores.mean()

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(n_startup_trials=30, seed=42)
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    return study.best_params, study

def optimize_keff_model(X_train, y_keff_train, model_type='xgboost', n_trials=250, n_jobs=10):
    """Optimize hyperparameters for k-eff prediction only"""

    def objective(trial):
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
                'n_jobs': 1
            }
            model = xgb.XGBRegressor(**params)

        elif model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
                'n_jobs': 1
            }
            model = RandomForestRegressor(**params)

        elif model_type == 'svm':
            params = {
                'C': trial.suggest_float('C', 0.001, 1000, log=True),
                'gamma': trial.suggest_float('gamma', 0.0001, 1, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.001, 1.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
            }
            if params['kernel'] == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)
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
                'verbose': 2,
            }
            model = MLPRegressor(**params)

        # Train and evaluate on k-eff only
        scores = cross_val_score(model, X_train, y_keff_train.ravel(), cv=10,
                               scoring='neg_mean_squared_error', n_jobs=1)
        return -scores.mean()

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(n_startup_trials=30, seed=42)
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    return study.best_params, study
