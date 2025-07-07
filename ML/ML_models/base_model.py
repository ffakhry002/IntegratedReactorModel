from abc import ABC, abstractmethod
import numpy as np
import os
import joblib
from datetime import datetime

class ReactorModelBase(ABC):
    """Base class for all reactor ML models to ensure consistency.

    Provides common interface and functionality for flux and k-effective prediction
    models, including model saving/loading, flux mode handling, and validation.
    """

    def __init__(self):
        """Initialize the base reactor model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.flux_model = None
        self.keff_model = None
        self._n_flux_outputs = 4  # Standard number of flux positions (will be updated based on flux_mode)
        self.model_class_name = None  # Must be set by subclasses
        self.flux_mode = 'total'  # Default
        self.params = {}  # Initialize params dict

    def set_flux_mode(self, flux_mode):
        """Set the flux mode and update expected outputs.

        Parameters
        ----------
        flux_mode : str
            Flux prediction mode ('total', 'energy', 'bin', 'thermal_only',
            'epithermal_only', 'fast_only')
        """
        self.flux_mode = flux_mode
        if flux_mode == 'total':
            self._n_flux_outputs = 4
        elif flux_mode in ['thermal_only', 'epithermal_only', 'fast_only']:
            self._n_flux_outputs = 4  # Single energy group, 4 positions
        else:  # energy or bin
            self._n_flux_outputs = 12

    def get_flux_mode(self):
        """Get the current flux mode.

        Returns
        -------
        str
            Current flux prediction mode
        """
        return self.flux_mode

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
        """Validate and reshape flux training data if needed.

        Parameters
        ----------
        y_flux : numpy.ndarray
            Flux training data

        Returns
        -------
        numpy.ndarray
            Validated and properly shaped flux data
        """
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
        """Ensure predictions have correct shape.

        Parameters
        ----------
        predictions : numpy.ndarray
            Model predictions
        n_samples : int
            Number of samples
        target_type : str, optional
            Type of prediction ('flux' or 'keff')

        Returns
        -------
        numpy.ndarray
            Predictions with validated shape
        """
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

    def save_model(self, filepath, model_type, encoding, optimization_method,
                   flux_scale=1e14, use_log_flux=False, flux_mode='total', **extra_metadata):
        """Save trained model with comprehensive metadata.

        Parameters
        ----------
        filepath : str
            Path where to save the model
        model_type : str
            Type of model ('flux' or 'keff')
        encoding : str
            Encoding method used
        optimization_method : str
            Optimization method used
        flux_scale : float, optional
            Flux scaling factor (default: 1e14)
        use_log_flux : bool, optional
            Whether logarithmic flux scaling was used (default: False)
        flux_mode : str, optional
            Flux prediction mode (default: 'total')
        **extra_metadata
            Additional metadata to save

        Returns
        -------
        str
            Path to saved model file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Construct filename with all identifiers
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        dir_name = os.path.dirname(filepath)

        # Format: random_forest_flux_physics_optuna.pkl or random_forest_flux_energy_physics_optuna.pkl
        if model_type == 'flux' and flux_mode != 'total':
            new_filename = f"{self.model_class_name}_{model_type}_{flux_mode}_{encoding}_{optimization_method}.pkl"
        else:
            new_filename = f"{self.model_class_name}_{model_type}_{encoding}_{optimization_method}.pkl"
        full_path = os.path.join(dir_name, new_filename)

        save_dict = {
            'model_class': self.model_class_name,
            'model_type': model_type,  # 'flux' or 'keff'
            'encoding': encoding,
            'optimization_method': optimization_method,
            'params': self.params,
            # CRITICAL: Add flux transformation metadata
            'flux_scale': flux_scale,
            'use_log_flux': use_log_flux,
            'flux_mode': flux_mode,  # NEW
            # Add timestamp for tracking
            'saved_at': datetime.now().isoformat(),
            # Add any extra metadata
            **extra_metadata
        }

        # Let subclasses add their specific metadata
        specific_metadata = self._get_model_specific_metadata()
        save_dict.update(specific_metadata)

        if model_type == 'flux':
            save_dict['model'] = self.flux_model  # Save the actual trained model
            # Store information about output shape
            if hasattr(self.flux_model, 'n_outputs_'):
                save_dict['n_flux_outputs'] = self.flux_model.n_outputs_
            else:
                # Infer from flux_mode
                if flux_mode == 'total':
                    save_dict['n_flux_outputs'] = 4
                else:  # energy or bin
                    save_dict['n_flux_outputs'] = 12

            # Handle scalers for models that use them
            if hasattr(self, 'scale_features') and self.scale_features and hasattr(self, 'flux_scaler'):
                save_dict['scaler'] = self.flux_scaler

        else:  # keff
            save_dict['model'] = self.keff_model  # Save the actual trained model

            # Handle scalers for models that use them
            if hasattr(self, 'scale_features') and self.scale_features and hasattr(self, 'keff_scaler'):
                save_dict['scaler'] = self.keff_scaler

        joblib.dump(save_dict, full_path)
        print(f"Model saved to: {full_path}")

        return full_path

    def _get_model_specific_metadata(self):
        """Override in subclasses to add model-specific metadata"""
        return {}

    @classmethod
    def load_model(cls, filepath):
        """Load model with all metadata.

        Parameters
        ----------
        filepath : str
            Path to saved model file

        Returns
        -------
        tuple
            (model_instance, metadata_dict) where model_instance is the loaded
            model and metadata_dict contains saved metadata
        """
        data = joblib.load(filepath)

        # Create a temporary instance to get the model_class_name
        temp_instance = cls()
        expected_class = temp_instance.model_class_name

        # Verify this is the correct model class
        if data.get('model_class') != expected_class:
            raise ValueError(f"Model file is for {data.get('model_class')}, not {expected_class}")

        # Create instance with original parameters
        model = cls(**data.get('params', {}))

        # Restore model-specific attributes
        model._restore_model_specific_attributes(data)

        # Restore flux mode if present
        if 'flux_mode' in data:
            model.flux_mode = data['flux_mode']
            if data['flux_mode'] == 'total':
                model._n_flux_outputs = 4
            elif data['flux_mode'] in ['thermal_only', 'epithermal_only', 'fast_only']:
                model._n_flux_outputs = 4  # Single energy group, 4 positions
            else:  # energy or bin
                model._n_flux_outputs = 12
        else:
            # Backward compatibility - check n_flux_outputs
            model._n_flux_outputs = data.get('n_flux_outputs', 4)
            model.flux_mode = 'total' if model._n_flux_outputs == 4 else 'energy'

        # Restore the trained model
        if data['model_type'] == 'flux':
            model.flux_model = data['model']
        else:
            model.keff_model = data['model']

        # Return model and metadata
        return model, {
            'flux_scale': data.get('flux_scale', 1e14),
            'use_log_flux': data.get('use_log_flux', False),
            'flux_mode': data.get('flux_mode', 'total'),
            'encoding': data.get('encoding'),
            'optimization_method': data.get('optimization_method'),
            'saved_at': data.get('saved_at')
        }

    def _restore_model_specific_attributes(self, data):
        """Override in subclasses to restore model-specific attributes"""
        pass
