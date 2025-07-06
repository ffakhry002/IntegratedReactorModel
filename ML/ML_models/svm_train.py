# from sklearn.svm import SVR
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# from .base_model import ReactorModelBase

# class SVMReactorModel(ReactorModelBase):
#     def __init__(self, scale_features=True, **kwargs):
#         super().__init__()  # Initialize base class
#         self.model_class_name = 'svm'

#         self.params = kwargs
#         self.scale_features = scale_features

#         if scale_features:
#             self.flux_scaler = StandardScaler()
#             self.keff_scaler = StandardScaler()

#     def fit_flux(self, X_train, y_flux):
#         """Train flux model only"""
#         # Use base class validation
#         y_flux = self.validate_flux_output(y_flux)

#         if self.scale_features:
#             X_scaled = self.flux_scaler.fit_transform(X_train)
#         else:
#             X_scaled = X_train

#         # Use MultiOutputRegressor for flux (multiple outputs)
#         # Use n_jobs=1 to avoid conflicts with Optuna parallelization
#         self.flux_model = MultiOutputRegressor(SVR(**self.params), n_jobs=1)
#         self.flux_model.fit(X_scaled, y_flux)
#         return self

#     def fit_keff(self, X_train, y_keff):
#         """Train k-eff model only"""
#         if self.scale_features:
#             X_scaled = self.keff_scaler.fit_transform(X_train)
#         else:
#             X_scaled = X_train

#         self.keff_model = SVR(**self.params)
#         self.keff_model.fit(X_scaled, y_keff.ravel())
#         return self

#     def predict_flux(self, X_test):
#         """Predict flux values"""
#         if self.flux_model is None:
#             raise ValueError("Flux model not trained")

#         if self.scale_features:
#             X_scaled = self.flux_scaler.transform(X_test)
#         else:
#             X_scaled = X_test

#         predictions = self.flux_model.predict(X_scaled)

#         # Use base class validation
#         predictions = self.validate_prediction_shape(
#             predictions, X_test.shape[0], 'flux'
#         )

#         return predictions

#     def predict_keff(self, X_test):
#         """Predict k-effective"""
#         if self.keff_model is None:
#             raise ValueError("K-eff model not trained")

#         if self.scale_features:
#             X_scaled = self.keff_scaler.transform(X_test)
#         else:
#             X_scaled = X_test

#         predictions = self.keff_model.predict(X_scaled)

#         # Use base class validation
#         predictions = self.validate_prediction_shape(
#             predictions, X_test.shape[0], 'keff'
#         )

#         return predictions

#     def _get_model_specific_metadata(self):
#         """Add SVM-specific metadata"""
#         metadata = {
#             'scale_features': self.scale_features
#         }
#         # SVM needs to manually store n_flux_outputs
#         if hasattr(self, '_n_flux_outputs'):
#             metadata['n_flux_outputs'] = self._n_flux_outputs
#         return metadata

#     def _restore_model_specific_attributes(self, data):
#         """Restore SVM-specific attributes"""
#         self.scale_features = data.get('scale_features', True)
#         if 'scaler' in data:
#             if data['model_type'] == 'flux':
#                 self.flux_scaler = data['scaler']
#             else:
#                 self.keff_scaler = data['scaler']
#         # SVM stored n_flux_outputs separately
#         if 'n_flux_outputs' in data and data['model_type'] == 'flux':
#             self._n_flux_outputs = data['n_flux_outputs']




#####Here is a potentially better way of doing it don't forget to change the n_jobs to 1 but it fixes some of the SVM scaling issues
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from .base_model import ReactorModelBase

class SVMReactorModel(ReactorModelBase):
    def __init__(self, scale_features=True, **kwargs):
        """Initialize SVM reactor model.

        Parameters
        ----------
        scale_features : bool, optional
            Whether to apply feature scaling, by default True
        **kwargs : dict
            Additional parameters for SVR model

        Returns
        -------
        None
        """
        super().__init__()  # Initialize base class
        self.model_class_name = 'svm'

        self.params = kwargs
        # Add better defaults to help with convergence
        if 'shrinking' not in self.params:
            self.params['shrinking'] = False
        if 'max_iter' not in self.params:
            self.params['max_iter'] = 100000  # Increase default max_iter
        if 'tol' not in self.params:
            self.params['tol'] = 1e-4  # Slightly relaxed tolerance

        self.scale_features = scale_features

    def fit_flux(self, X_train, y_flux):
        """Train flux model only.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training input features
        y_flux : numpy.ndarray
            Training flux target values

        Returns
        -------
        SVMReactorModel
            Self for method chaining
        """
        # Use base class validation
        y_flux = self.validate_flux_output(y_flux)

        # Create base SVR
        base_svr = SVR(**self.params)

        if self.scale_features:
            # Use Pipeline to match optimization code
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', base_svr)
            ])
            # Use n_jobs=1 to avoid parallel conflicts and improve convergence
            self.flux_model = MultiOutputRegressor(pipeline, n_jobs=1)
        else:
            self.flux_model = MultiOutputRegressor(base_svr, n_jobs=1)

        self.flux_model.fit(X_train, y_flux)
        return self

    def fit_keff(self, X_train, y_keff):
        """Train k-eff model only.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training input features
        y_keff : numpy.ndarray
            Training k-effective target values

        Returns
        -------
        SVMReactorModel
            Self for method chaining
        """
        if self.scale_features:
            # Use Pipeline for keff too
            self.keff_model = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(**self.params))
            ])
        else:
            self.keff_model = SVR(**self.params)

        self.keff_model.fit(X_train, y_keff.ravel())
        return self

    def predict_flux(self, X_test):
        """Predict flux values.

        Parameters
        ----------
        X_test : numpy.ndarray
            Test input features

        Returns
        -------
        numpy.ndarray
            Predicted flux values
        """
        if self.flux_model is None:
            raise ValueError("Flux model not trained")

        # Pipeline handles scaling automatically
        predictions = self.flux_model.predict(X_test)

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
            Test input features

        Returns
        -------
        numpy.ndarray
            Predicted k-effective values
        """
        if self.keff_model is None:
            raise ValueError("K-eff model not trained")

        # Pipeline handles scaling automatically
        predictions = self.keff_model.predict(X_test)

        # Use base class validation
        predictions = self.validate_prediction_shape(
            predictions, X_test.shape[0], 'keff'
        )

        return predictions

    def _get_model_specific_metadata(self):
        """Add SVM-specific metadata.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary containing SVM-specific metadata
        """
        metadata = {
            'scale_features': self.scale_features,
            'svm_params': self.params
        }
        # SVM needs to manually store n_flux_outputs
        if hasattr(self, '_n_flux_outputs'):
            metadata['n_flux_outputs'] = self._n_flux_outputs
        return metadata

    def _restore_model_specific_attributes(self, data):
        """Restore SVM-specific attributes.

        Parameters
        ----------
        data : dict
            Dictionary containing model data to restore

        Returns
        -------
        None
        """
        self.scale_features = data.get('scale_features', True)
        self.params = data.get('svm_params', {})
        # Pipeline handles scaling internally, no need to restore scalers
        # SVM stored n_flux_outputs separately
        if 'n_flux_outputs' in data and data['model_type'] == 'flux':
            self._n_flux_outputs = data['n_flux_outputs']
