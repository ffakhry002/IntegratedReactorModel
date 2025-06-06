from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
from datetime import datetime
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

    def save_model(self, filepath, model_type, encoding, optimization_method,
                   flux_scale=1e14, use_log_flux=False, **extra_metadata):
        """Save trained model with comprehensive metadata"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Construct filename with all identifiers
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        dir_name = os.path.dirname(filepath)

        # Format: svm_flux_physics_optuna.pkl
        new_filename = f"{self.model_class_name}_{model_type}_{encoding}_{optimization_method}.pkl"
        full_path = os.path.join(dir_name, new_filename)

        save_dict = {
            'model_class': self.model_class_name,
            'model_type': model_type,  # 'flux' or 'keff'
            'encoding': encoding,
            'optimization_method': optimization_method,
            'params': self.params,
            'scale_features': self.scale_features,
            # CRITICAL: Add flux transformation metadata
            'flux_scale': flux_scale,
            'use_log_flux': use_log_flux,
            # Add timestamp for tracking
            'saved_at': datetime.now().isoformat(),
            # Add any extra metadata
            **extra_metadata
        }

        if model_type == 'flux':
            save_dict['model'] = self.flux_model
            if self.scale_features:
                save_dict['scaler'] = self.flux_scaler
            # Store information about output shape
            save_dict['n_flux_outputs'] = self._n_flux_outputs
        else:  # keff
            save_dict['model'] = self.keff_model
            if self.scale_features:
                save_dict['scaler'] = self.keff_scaler

        joblib.dump(save_dict, full_path)
        print(f"Model saved to: {full_path}")

        return full_path

    @classmethod
    def load_model(cls, filepath):
        """Load model with all metadata"""
        data = joblib.load(filepath)

        # Verify this is the correct model class
        if data.get('model_class') != 'svm':
            raise ValueError(f"Model file is for {data.get('model_class')}, not svm")

        # Create instance with original parameters
        model = cls(scale_features=data.get('scale_features', True), **data.get('params', {}))

        # Restore the trained model and scaler
        if data['model_type'] == 'flux':
            model.flux_model = data['model']
            if model.scale_features and 'scaler' in data:
                model.flux_scaler = data['scaler']
            if 'n_flux_outputs' in data:
                model._n_flux_outputs = data['n_flux_outputs']
        else:
            model.keff_model = data['model']
            if model.scale_features and 'scaler' in data:
                model.keff_scaler = data['scaler']

        # Return model and metadata
        return model, {
            'flux_scale': data.get('flux_scale', 1e14),
            'use_log_flux': data.get('use_log_flux', False),
            'encoding': data.get('encoding'),
            'optimization_method': data.get('optimization_method'),
            'saved_at': data.get('saved_at')
        }
