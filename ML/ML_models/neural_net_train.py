from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from .base_model import ReactorModelBase

class NeuralNetReactorModel(ReactorModelBase):
    def __init__(self, scale_features=True, **kwargs):
        super().__init__()  # Initialize base class
        self.model_class_name = 'neural_net'

        # Set default parameters if not provided
        if 'max_iter' not in kwargs:
            kwargs['max_iter'] = 1000
        if 'early_stopping' not in kwargs:
            kwargs['early_stopping'] = True
        if 'validation_fraction' not in kwargs:
            kwargs['validation_fraction'] = 0.1

        self.params = kwargs
        self.scale_features = scale_features

        if scale_features:
            self.flux_scaler = StandardScaler()
            self.keff_scaler = StandardScaler()

    def fit_flux(self, X_train, y_flux):
        """Train flux model only"""
        # Use base class validation
        y_flux = self.validate_flux_output(y_flux)

        if self.scale_features:
            X_scaled = self.flux_scaler.fit_transform(X_train)
        else:
            X_scaled = X_train

        self.flux_model = MLPRegressor(**self.params)
        self.flux_model.fit(X_scaled, y_flux)
        return self

    def fit_keff(self, X_train, y_keff):
        """Train k-eff model only"""
        if self.scale_features:
            X_scaled = self.keff_scaler.fit_transform(X_train)
        else:
            X_scaled = X_train

        self.keff_model = MLPRegressor(**self.params)
        self.keff_model.fit(X_scaled, y_keff.ravel())
        return self

    def predict_flux(self, X_test):
        """Predict flux values"""
        if self.flux_model is None:
            raise ValueError("Flux model not trained")

        if self.scale_features:
            X_scaled = self.flux_scaler.transform(X_test)
        else:
            X_scaled = X_test

        predictions = self.flux_model.predict(X_scaled)

        # Use base class validation
        predictions = self.validate_prediction_shape(
            predictions, X_test.shape[0], 'flux'
        )

        return predictions

    def predict_keff(self, X_test):
        """Predict k-effective"""
        if self.keff_model is None:
            raise ValueError("K-eff model not trained")

        if self.scale_features:
            X_scaled = self.keff_scaler.transform(X_test)
        else:
            X_scaled = X_test

        predictions = self.keff_model.predict(X_scaled)

        # Use base class validation
        predictions = self.validate_prediction_shape(
            predictions, X_test.shape[0], 'keff'
        )

        return predictions

    def _get_model_specific_metadata(self):
        """Add Neural Net-specific metadata"""
        return {
            'scale_features': self.scale_features
        }

    def _restore_model_specific_attributes(self, data):
        """Restore Neural Net-specific attributes"""
        self.scale_features = data.get('scale_features', True)
        if 'scaler' in data:
            if data['model_type'] == 'flux':
                self.flux_scaler = data['scaler']
            else:
                self.keff_scaler = data['scaler']
