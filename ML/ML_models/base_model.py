# Option 1: Create a new file ML_models/base_model.py
# ML_models/base_model.py
from abc import ABC, abstractmethod
import numpy as np

class ReactorModelBase(ABC):
    """Base class for all reactor ML models to ensure consistency"""

    def __init__(self):
        self.flux_model = None
        self.keff_model = None
        self._n_flux_outputs = 4  # Standard number of flux positions
        self.model_class_name = None  # Must be set by subclasses

    @abstractmethod
    def fit_flux(self, X_train, y_flux):
        """Train flux model - must be implemented by subclasses"""
        pass

    @abstractmethod
    def fit_keff(self, X_train, y_keff):
        """Train k-eff model - must be implemented by subclasses"""
        pass

    @abstractmethod
    def predict_flux(self, X_test):
        """Predict flux - must be implemented by subclasses"""
        pass

    @abstractmethod
    def predict_keff(self, X_test):
        """Predict k-eff - must be implemented by subclasses"""
        pass

    def validate_flux_output(self, y_flux):
        """Validate and reshape flux training data if needed"""
        if len(y_flux.shape) == 1:
            # Attempt to reshape
            if y_flux.shape[0] % self._n_flux_outputs == 0:
                n_samples = y_flux.shape[0] // self._n_flux_outputs
                y_flux = y_flux.reshape(n_samples, self._n_flux_outputs)
                print(f"Reshaped flux data from {y_flux.shape[0]} to {y_flux.shape}")
            else:
                raise ValueError(f"Cannot reshape flux data: {y_flux.shape}")

        if y_flux.shape[1] != self._n_flux_outputs:
            print(f"Warning: Expected {self._n_flux_outputs} flux outputs, got {y_flux.shape[1]}")
            self._n_flux_outputs = y_flux.shape[1]

        return y_flux

    def validate_prediction_shape(self, predictions, n_samples, target_type='flux'):
        """Ensure predictions have correct shape"""
        if target_type == 'flux':
            expected_shape = (n_samples, self._n_flux_outputs)

            if len(predictions.shape) == 1:
                if predictions.shape[0] == n_samples * self._n_flux_outputs:
                    predictions = predictions.reshape(n_samples, self._n_flux_outputs)
                else:
                    raise ValueError(f"Cannot reshape {target_type} predictions: {predictions.shape}")

            if predictions.shape != expected_shape:
                raise ValueError(f"Expected {target_type} shape {expected_shape}, got {predictions.shape}")

        else:  # keff
            if len(predictions.shape) > 1:
                predictions = predictions.ravel()

            if predictions.shape[0] != n_samples:
                raise ValueError(f"Expected {n_samples} k-eff predictions, got {predictions.shape[0]}")

        return predictions
