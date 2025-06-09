from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from .base_model import ReactorModelBase

class SVMReactorModel(ReactorModelBase):
    def __init__(self, scale_features=True, **kwargs):
        super().__init__()  # Initialize base class
        self.model_class_name = 'svm'

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

        # Use MultiOutputRegressor for flux (multiple outputs)
        self.flux_model = MultiOutputRegressor(SVR(**self.params), n_jobs=-1)
        self.flux_model.fit(X_scaled, y_flux)
        return self

    def fit_keff(self, X_train, y_keff):
        """Train k-eff model only"""
        if self.scale_features:
            X_scaled = self.keff_scaler.fit_transform(X_train)
        else:
            X_scaled = X_train

        self.keff_model = SVR(**self.params)
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
        """Add SVM-specific metadata"""
        metadata = {
            'scale_features': self.scale_features
        }
        # SVM needs to manually store n_flux_outputs
        if hasattr(self, '_n_flux_outputs'):
            metadata['n_flux_outputs'] = self._n_flux_outputs
        return metadata

    def _restore_model_specific_attributes(self, data):
        """Restore SVM-specific attributes"""
        self.scale_features = data.get('scale_features', True)
        if 'scaler' in data:
            if data['model_type'] == 'flux':
                self.flux_scaler = data['scaler']
            else:
                self.keff_scaler = data['scaler']
        # SVM stored n_flux_outputs separately
        if 'n_flux_outputs' in data and data['model_type'] == 'flux':
            self._n_flux_outputs = data['n_flux_outputs']
