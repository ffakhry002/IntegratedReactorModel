import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from .base_model import ReactorModelBase

class XGBoostReactorModel(ReactorModelBase):
    def __init__(self, use_multioutput=True, use_restructured=False,
                 irradiation_mode='fill', nci_mode='separate',
                 nci_disabled=False, **kwargs):
        """Initialize XGBoost reactor model.

        Parameters
        ----------
        use_multioutput : bool, optional
            Whether to use MultiOutputRegressor for flux prediction, by default True
        use_restructured : bool, optional
            Whether to use position-pooled restructured data (single model), by default False
        irradiation_mode : str, optional
            'vacuum' or 'fill', needed for restructuring feature layout at prediction time
        nci_mode : str, optional
            'single' or 'separate', needed for restructuring feature layout at prediction time
        **kwargs : dict
            Additional parameters for XGBRegressor

        Returns
        -------
        None
        """
        super().__init__()  # Initialize base class
        self.model_class_name = 'xgboost'

        # Set n_jobs for parallelization
        # Use n_jobs=1 to avoid conflicts with Optuna parallelization
        if 'n_jobs' not in kwargs:
            kwargs['n_jobs'] = 1  # Single core per trial

        self.params = kwargs
        self.use_multioutput = use_multioutput
        self.use_restructured = use_restructured
        self.irradiation_mode = irradiation_mode
        self.nci_mode = nci_mode
        self.nci_disabled = nci_disabled

    def fit_flux(self, X_train, y_flux):
        """Train flux model only.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training input features.
            If use_restructured=True: shape (4N, F) with 1D targets (already restructured).
            If use_restructured=False: shape (N, F) with (N, 4) targets.
        y_flux : numpy.ndarray
            Training flux target values.

        Returns
        -------
        XGBoostReactorModel
            Self for method chaining
        """
        if self.use_restructured:
            # Restructured: single-output model, y is 1D
            self.flux_model = xgb.XGBRegressor(**self.params)
            self.flux_model.fit(X_train, y_flux.ravel())
        else:
            # Use base class validation
            y_flux = self.validate_flux_output(y_flux)

            if self.use_multioutput:
                self.flux_model = MultiOutputRegressor(xgb.XGBRegressor(**self.params))
            else:
                print("WARNING: use_multioutput=False for flux prediction may cause issues")
                self.flux_model = xgb.XGBRegressor(**self.params)

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
        XGBoostReactorModel
            Self for method chaining
        """
        self.keff_model = xgb.XGBRegressor(**self.params)
        self.keff_model.fit(X_train, y_keff.ravel())
        return self

    def predict_flux(self, X_test):
        """Predict flux values.

        Parameters
        ----------
        X_test : numpy.ndarray
            Test input features in ORIGINAL layout (n, F).
            If use_restructured=True, internally restructures to (4n, F),
            predicts scalars, and reshapes back to (n, 4).

        Returns
        -------
        numpy.ndarray
            Predicted flux values, shape (n, 4)
        """
        if self.flux_model is None:
            raise ValueError("Flux model not trained")

        if self.use_restructured:
            from utils.data_restructure import restructure_single_for_prediction
            n_samples = X_test.shape[0]
            X_pooled = restructure_single_for_prediction(
                X_test, self.irradiation_mode, self.nci_mode, self.nci_disabled)
            raw_predictions = self.flux_model.predict(X_pooled)
            predictions = raw_predictions.reshape(n_samples, 4)
        else:
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

        predictions = self.keff_model.predict(X_test)

        # Use base class validation
        predictions = self.validate_prediction_shape(
            predictions, X_test.shape[0], 'keff'
        )

        return predictions

    def get_feature_importance(self, model_type='flux'):
        """Get feature importances for interpretation.

        Parameters
        ----------
        model_type : str, optional
            Type of model ('flux' or 'keff'), by default 'flux'

        Returns
        -------
        numpy.ndarray or None
            Feature importances if model exists, None otherwise
        """
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

    def _get_model_specific_metadata(self):
        """Add XGBoost-specific metadata.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary containing XGBoost-specific metadata
        """
        return {
            'use_multioutput': self.use_multioutput,
            'use_restructured': self.use_restructured,
            'restructured_irradiation_mode': self.irradiation_mode,
            'restructured_nci_mode': self.nci_mode,
            'restructured_nci_disabled': self.nci_disabled,
        }

    def _restore_model_specific_attributes(self, data):
        """Restore XGBoost-specific attributes.

        Parameters
        ----------
        data : dict
            Dictionary containing model data to restore

        Returns
        -------
        None
        """
        self.use_multioutput = data.get('use_multioutput', True)
        self.use_restructured = data.get('use_restructured', False)
        self.irradiation_mode = data.get('restructured_irradiation_mode', 'fill')
        self.nci_mode = data.get('restructured_nci_mode', 'separate')
        self.nci_disabled = data.get('restructured_nci_disabled', False)