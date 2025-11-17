"""
Lambda-aware estimator wrapper for three-stage optimization.

This wrapper allows sklearn's RandomizedSearchCV, GridSearchCV, and BayesSearchCV
to optimize lambda parameters alongside model hyperparameters by regenerating
features with new lambda values before fitting.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Dict, Any, List
from utils.lambda_feature_regenerator import LambdaFeatureRegenerator


class LambdaAwareEstimator(BaseEstimator, RegressorMixin):
    """Wrapper estimator that regenerates features based on lambda parameters.

    This allows lambda to be treated as a hyperparameter in grid search,
    random search, and Bayesian optimization.

    Parameters
    ----------
    base_estimator : estimator object
        The base estimator to wrap
    feature_data : Dict
        Feature data from LambdaFeatureRegenerator.separate_features()
    irradiation_mode : str
        'vacuum' or 'fill'
    nci_mode : str
        'single' or 'separate'
    lambda_decay : float, optional
        Default lambda for single NCI mode
    lambda_P : float, optional
        Default lambda for P-type vehicles
    lambda_B : float, optional
        Default lambda for B-type vehicles
    lambda_G : float, optional
        Default lambda for G-type vehicles
    """

    def __init__(self, base_estimator=None, feature_data=None,
                 irradiation_mode='vacuum', nci_mode='single',
                 lambda_decay=1.5, lambda_P=1.5, lambda_B=1.5, lambda_G=1.5):
        self.base_estimator = base_estimator
        self.feature_data = feature_data
        self.irradiation_mode = irradiation_mode
        self.nci_mode = nci_mode
        self.lambda_decay = lambda_decay
        self.lambda_P = lambda_P
        self.lambda_B = lambda_B
        self.lambda_G = lambda_G

        # Initialize feature regenerator
        self.regenerator = LambdaFeatureRegenerator('physics')

    def fit(self, X, y):
        """Fit the model with regenerated features.

        Parameters
        ----------
        X : array-like
            Input features (will be regenerated)
        y : array-like
            Target values

        Returns
        -------
        self
        """
        # Regenerate features with current lambda values
        X_regenerated = self._regenerate_features()

        # Fit base estimator
        self.base_estimator.fit(X_regenerated, y)

        return self

    def predict(self, X):
        """Predict using regenerated features.

        Parameters
        ----------
        X : array-like
            Input features (will be regenerated)

        Returns
        -------
        array-like
            Predicted values
        """
        # Regenerate features with current lambda values
        X_regenerated = self._regenerate_features()

        return self.base_estimator.predict(X_regenerated)

    def _regenerate_features(self):
        """Regenerate features based on current lambda parameters."""
        if self.irradiation_mode == 'vacuum' or self.nci_mode == 'single':
            # Use lambda_decay
            X_regenerated = self.regenerator.regenerate_features(
                self.feature_data, lambda_decay=self.lambda_decay
            )
        else:
            # Use lambda_P, lambda_B, lambda_G
            X_regenerated = self.regenerator.regenerate_features(
                self.feature_data,
                lambda_P=self.lambda_P,
                lambda_B=self.lambda_B,
                lambda_G=self.lambda_G
            )

        return X_regenerated

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters for sub-estimators

        Returns
        -------
        dict
            Parameter names mapped to their values
        """
        params = {
            'base_estimator': self.base_estimator,
            'feature_data': self.feature_data,
            'irradiation_mode': self.irradiation_mode,
            'nci_mode': self.nci_mode,
            'lambda_decay': self.lambda_decay,
            'lambda_P': self.lambda_P,
            'lambda_B': self.lambda_B,
            'lambda_G': self.lambda_G
        }

        if deep and self.base_estimator is not None:
            # Get base estimator params with prefix
            base_params = self.base_estimator.get_params(deep=True)
            for key, value in base_params.items():
                params[f'base_estimator__{key}'] = value

        return params

    def set_params(self, **params):
        """Set parameters for this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters

        Returns
        -------
        self
        """
        base_params = {}
        wrapper_params = {}

        for key, value in params.items():
            if key.startswith('base_estimator__'):
                # Remove prefix and pass to base estimator
                base_key = key.replace('base_estimator__', '')
                base_params[base_key] = value
            else:
                wrapper_params[key] = value

        # Set wrapper params
        for key, value in wrapper_params.items():
            setattr(self, key, value)

        # Set base estimator params
        if base_params and self.base_estimator is not None:
            self.base_estimator.set_params(**base_params)

        return self

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like
            Test samples
        y : array-like
            True values for X

        Returns
        -------
        float
            R^2 score
        """
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


def add_lambda_to_param_distributions(param_distributions: Dict,
                                      irradiation_mode: str,
                                      nci_mode: str) -> Dict:
    """Add lambda parameters to parameter distributions for random search.

    Parameters
    ----------
    param_distributions : Dict
        Original parameter distributions
    irradiation_mode : str
        'vacuum' or 'fill'
    nci_mode : str
        'single' or 'separate'

    Returns
    -------
    Dict
        Updated parameter distributions with lambda parameters
    """
    from scipy.stats import uniform

    # Prefix all existing params with base_estimator__
    prefixed_params = {f'base_estimator__{k}': v
                      for k, v in param_distributions.items()}

    # Add lambda parameters (uniform between 0.5 and 2.0)
    if irradiation_mode == 'vacuum' or nci_mode == 'single':
        prefixed_params['lambda_decay'] = uniform(0.5, 1.5)  # 0.5 to 2.0
    else:
        prefixed_params['lambda_P'] = uniform(0.5, 1.5)  # 0.5 to 2.0
        prefixed_params['lambda_B'] = uniform(0.5, 1.5)  # 0.5 to 2.0
        prefixed_params['lambda_G'] = uniform(0.5, 1.5)  # 0.5 to 2.0

    return prefixed_params


def add_lambda_to_grid_params(param_grid: Dict,
                              best_lambda_values: Dict,
                              irradiation_mode: str,
                              nci_mode: str) -> Dict:
    """Add lambda parameters to grid search parameters.

    Parameters
    ----------
    param_grid : Dict
        Original parameter grid
    best_lambda_values : Dict
        Best lambda values from random search
    irradiation_mode : str
        'vacuum' or 'fill'
    nci_mode : str
        'single' or 'separate'

    Returns
    -------
    Dict
        Updated parameter grid with lambda parameters
    """
    # Prefix all existing params with base_estimator__
    prefixed_params = {f'base_estimator__{k}': v
                      for k, v in param_grid.items()}

    # Add lambda grid around best values (±20%)
    if irradiation_mode == 'vacuum' or nci_mode == 'single':
        best_lambda = best_lambda_values.get('lambda_decay', 1.5)
        prefixed_params['lambda_decay'] = [
            max(0.5, best_lambda * 0.8),
            max(0.5, best_lambda * 0.9),
            best_lambda,
            min(2.0, best_lambda * 1.1),
            min(2.0, best_lambda * 1.2)
        ]
    else:
        for vehicle_type in ['P', 'B', 'G']:
            key = f'lambda_{vehicle_type}'
            best_lambda = best_lambda_values.get(key, 1.5)
            prefixed_params[key] = [
                max(0.5, best_lambda * 0.8),
                max(0.5, best_lambda * 0.9),
                best_lambda,
                min(2.0, best_lambda * 1.1),
                min(2.0, best_lambda * 1.2)
            ]

    return prefixed_params


def add_lambda_to_bayesian_spaces(search_spaces: Dict,
                                  irradiation_mode: str,
                                  nci_mode: str) -> Dict:
    """Add lambda parameters to Bayesian search spaces.

    Parameters
    ----------
    search_spaces : Dict
        Original search spaces
    irradiation_mode : str
        'vacuum' or 'fill'
    nci_mode : str
        'single' or 'separate'

    Returns
    -------
    Dict
        Updated search spaces with lambda parameters
    """
    from skopt.space import Real

    # Prefix all existing params with base_estimator__
    prefixed_spaces = {f'base_estimator__{k}': v
                       for k, v in search_spaces.items()}

    # Add lambda search spaces
    if irradiation_mode == 'vacuum' or nci_mode == 'single':
        prefixed_spaces['lambda_decay'] = Real(0.5, 2.0, prior='uniform')
    else:
        prefixed_spaces['lambda_P'] = Real(0.5, 2.0, prior='uniform')
        prefixed_spaces['lambda_B'] = Real(0.5, 2.0, prior='uniform')
        prefixed_spaces['lambda_G'] = Real(0.5, 2.0, prior='uniform')

    return prefixed_spaces
