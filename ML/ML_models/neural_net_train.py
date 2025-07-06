from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from .base_model import ReactorModelBase

class NeuralNetReactorModel(ReactorModelBase):
    """Neural network implementation of the reactor model."""

    def __init__(self, scale_features=True, **kwargs):
        """Initialize the neural network reactor model.

        Parameters
        ----------
        scale_features : bool, optional
            Whether to apply feature scaling
        **kwargs : dict
            Additional parameters for MLPRegressor

        Returns
        -------
        None
        """
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
        """Train flux model only.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training feature data
        y_flux : numpy.ndarray
            Training flux target data

        Returns
        -------
        self
            Fitted model instance
        """
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
        """Train k-eff model only.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training feature data
        y_keff : numpy.ndarray
            Training k-effective target data

        Returns
        -------
        self
            Fitted model instance
        """
        if self.scale_features:
            X_scaled = self.keff_scaler.fit_transform(X_train)
        else:
            X_scaled = X_train

        self.keff_model = MLPRegressor(**self.params)
        self.keff_model.fit(X_scaled, y_keff.ravel())
        return self

    def predict_flux(self, X_test):
        """Predict flux values.

        Parameters
        ----------
        X_test : numpy.ndarray
            Test feature data

        Returns
        -------
        numpy.ndarray
            Predicted flux values
        """
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
        """Predict k-effective.

        Parameters
        ----------
        X_test : numpy.ndarray
            Test feature data

        Returns
        -------
        numpy.ndarray
            Predicted k-effective values
        """
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
        """Add Neural Net-specific metadata.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Neural network specific metadata
        """
        return {
            'scale_features': self.scale_features
        }

    def _restore_model_specific_attributes(self, data):
        """Restore Neural Net-specific attributes.

        Parameters
        ----------
        data : dict
            Loaded model data dictionary

        Returns
        -------
        None
        """
        self.scale_features = data.get('scale_features', True)
        if 'scaler' in data:
            if data['model_type'] == 'flux':
                self.flux_scaler = data['scaler']
            else:
                self.keff_scaler = data['scaler']
