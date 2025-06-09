from sklearn.ensemble import RandomForestRegressor
import numpy as np
from .base_model import ReactorModelBase

class RandomForestReactorModel(ReactorModelBase):
    def __init__(self, **kwargs):
        super().__init__()  # Initialize base class
        self.model_class_name = 'random_forest'

        # Set n_jobs for parallelization if not specified
        if 'n_jobs' not in kwargs:
            kwargs['n_jobs'] = -1  # Use all cores

        self.params = kwargs

    def fit_flux(self, X_train, y_flux):
        """Train flux model only"""
        # Use base class validation
        y_flux = self.validate_flux_output(y_flux)

        self.flux_model = RandomForestRegressor(**self.params)
        self.flux_model.fit(X_train, y_flux)
        return self

    def fit_keff(self, X_train, y_keff):
        """Train k-eff model only"""
        self.keff_model = RandomForestRegressor(**self.params)
        self.keff_model.fit(X_train, y_keff.ravel())
        return self

    def predict_flux(self, X_test):
        """Predict flux values"""
        if self.flux_model is None:
            raise ValueError("Flux model not trained")

        predictions = self.flux_model.predict(X_test)

        # Use base class validation
        predictions = self.validate_prediction_shape(
            predictions, X_test.shape[0], 'flux'
        )

        return predictions

    def predict_keff(self, X_test):
        """Predict k-effective"""
        if self.keff_model is None:
            raise ValueError("K-eff model not trained")

        predictions = self.keff_model.predict(X_test)

        # Use base class validation
        predictions = self.validate_prediction_shape(
            predictions, X_test.shape[0], 'keff'
        )

        return predictions

    def get_feature_importance(self, model_type='flux'):
        """Get feature importances for interpretation"""
        if model_type == 'flux' and self.flux_model is not None:
            return self.flux_model.feature_importances_
        elif model_type == 'keff' and self.keff_model is not None:
            return self.keff_model.feature_importances_
        else:
            return None

    # No need to override _get_model_specific_metadata or _restore_model_specific_attributes
    # RandomForest has no extra attributes beyond the base class
