import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import joblib
import os
from datetime import datetime
from .base_model import ReactorModelBase

class XGBoostReactorModel(ReactorModelBase):
    def __init__(self, use_multioutput=True, **kwargs):
        super().__init__()  # Initialize base class
        self.model_class_name = 'xgboost'

        # Set n_jobs for parallelization
        if 'n_jobs' not in kwargs:
            kwargs['n_jobs'] = -1  # Use all cores

        self.params = kwargs
        self.use_multioutput = use_multioutput

    def fit_flux(self, X_train, y_flux):
        """Train flux model only"""
        # Use base class validation
        y_flux = self.validate_flux_output(y_flux)

        if self.use_multioutput:
            # Use MultiOutputRegressor for multiple flux outputs
            self.flux_model = MultiOutputRegressor(xgb.XGBRegressor(**self.params))
        else:
            # This won't work well for multiple outputs!
            print("WARNING: use_multioutput=False for flux prediction may cause issues")
            self.flux_model = xgb.XGBRegressor(**self.params)

        self.flux_model.fit(X_train, y_flux)
        return self

    def fit_keff(self, X_train, y_keff):
        """Train k-eff model only"""
        self.keff_model = xgb.XGBRegressor(**self.params)
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
            if self.use_multioutput:
                # MultiOutputRegressor - average importances across outputs
                importances = []
                for estimator in self.flux_model.estimators_:
                    importances.append(estimator.feature_importances_)
                return np.mean(importances, axis=0)
            else:
                return self.flux_model.feature_importances_
        elif model_type == 'keff' and self.keff_model is not None:
            return self.keff_model.feature_importances_
        else:
            return None

    def save_model(self, filepath, model_type, encoding, optimization_method,
                   flux_scale=1e14, use_log_flux=False, **extra_metadata):
        """Save trained model with comprehensive metadata"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Construct filename with all identifiers
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        dir_name = os.path.dirname(filepath)

        # Format: xgboost_flux_physics_optuna.pkl
        new_filename = f"{self.model_class_name}_{model_type}_{encoding}_{optimization_method}.pkl"
        full_path = os.path.join(dir_name, new_filename)

        save_dict = {
            'model_class': self.model_class_name,
            'model_type': model_type,  # 'flux' or 'keff'
            'encoding': encoding,
            'optimization_method': optimization_method,
            'params': self.params,
            'use_multioutput': self.use_multioutput,
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
            save_dict['n_flux_outputs'] = self._n_flux_outputs
        else:  # keff
            save_dict['model'] = self.keff_model

        joblib.dump(save_dict, full_path)
        print(f"Model saved to: {full_path}")

        return full_path

    @classmethod
    def load_model(cls, filepath):
        """Load model with all metadata"""
        data = joblib.load(filepath)

        # Verify this is the correct model class
        if data.get('model_class') != 'xgboost':
            raise ValueError(f"Model file is for {data.get('model_class')}, not xgboost")

        # Create instance with original parameters
        model = cls(use_multioutput=data.get('use_multioutput', True), **data.get('params', {}))

        # Restore the trained model
        if data['model_type'] == 'flux':
            model.flux_model = data['model']
            if 'n_flux_outputs' in data:
                model._n_flux_outputs = data['n_flux_outputs']
        else:
            model.keff_model = data['model']

        # Return model and metadata
        return model, {
            'flux_scale': data.get('flux_scale', 1e14),
            'use_log_flux': data.get('use_log_flux', False),
            'encoding': data.get('encoding'),
            'optimization_method': data.get('optimization_method'),
            'saved_at': data.get('saved_at')
        }
