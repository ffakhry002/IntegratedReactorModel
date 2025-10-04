import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from .base_model import ReactorModelBase
import warnings


class PyTorchRectangularNet(nn.Module):
    """PyTorch neural network with rectangular (uniform width) architecture."""

    def __init__(self, input_dim, output_dim, depth=2, width=100, activation='relu'):
        """Initialize rectangular neural network.

        Parameters
        ----------
        input_dim : int
            Number of input features
        output_dim : int
            Number of output targets
        depth : int
            Number of hidden layers
        width : int
            Number of neurons in each hidden layer (uniform across all layers)
        activation : str
            Activation function ('relu', 'tanh', 'sigmoid', 'elu')
        """
        super(PyTorchRectangularNet, self).__init__()

        # Select activation function
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU()
        }
        self.activation = activation_map.get(activation.lower(), nn.ReLU())

        # Build network layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, width))
        layers.append(self.activation)

        # Hidden layers (depth - 1 additional layers, all with same width)
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(self.activation)

        # Output layer (no activation)
        layers.append(nn.Linear(width, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)


class PyTorchRegressorWrapper(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper for PyTorch neural network.

    This wrapper makes PyTorch models compatible with sklearn's cross_val_score
    and hyperparameter optimization tools while supporting GPU training.
    """

    def __init__(self, depth=2, width=100, activation='relu', learning_rate=0.001,
                 batch_size=128, max_epochs=1000, weight_decay=0.0001,
                 patience=20, optimizer='adam', validation_fraction=0.1,
                 device=None, n_gpus=1, verbose=False, random_state=42):
        """Initialize PyTorch regressor wrapper.

        Parameters
        ----------
        depth : int
            Number of hidden layers
        width : int
            Number of neurons per hidden layer (uniform)
        activation : str
            Activation function ('relu', 'tanh', 'sigmoid', 'elu', 'leaky_relu')
        learning_rate : float
            Learning rate for optimizer
        batch_size : int
            Batch size for training
        max_epochs : int
            Maximum number of training epochs
        weight_decay : float
            L2 regularization strength
        patience : int
            Early stopping patience (epochs without improvement)
        optimizer : str
            Optimizer type ('adam', 'sgd', 'adamw', 'rmsprop')
        validation_fraction : float
            Fraction of training data to use for validation
        device : str
            Device to use ('cuda' or 'cpu')
        verbose : bool
            Whether to print training progress
        random_state : int
            Random seed for reproducibility
        """
        self.depth = depth
        self.width = width
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.optimizer = optimizer
        self.validation_fraction = validation_fraction
        self.n_gpus = n_gpus  # Store for later
        self.device = device  # Store, but assign lazily in fit()
        self.verbose = verbose
        self.random_state = random_state

        self.model = None
        self.input_dim = None
        self.output_dim = None

    def _set_random_seed(self):
        """Set random seeds for reproducibility (with PID variation for workers)."""
        import os
        # Add PID to seed for worker diversity in parallel CV
        pid = os.getpid()
        seed = self.random_state + (pid % 10000)  # Vary by PID but keep deterministic
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _create_optimizer(self, model):
        """Create optimizer based on specified type."""
        optimizer_map = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'adamw': optim.AdamW,
            'rmsprop': optim.RMSprop
        }

        optimizer_class = optimizer_map.get(self.optimizer.lower(), optim.Adam)

        # SGD needs momentum for better convergence
        if self.optimizer.lower() == 'sgd':
            return optimizer_class(model.parameters(), lr=self.learning_rate,
                                 weight_decay=self.weight_decay, momentum=0.9)
        else:
            return optimizer_class(model.parameters(), lr=self.learning_rate,
                                 weight_decay=self.weight_decay)

    def fit(self, X, y):
        """Fit the neural network.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values

        Returns
        -------
        self
            Fitted estimator
        """
        self._set_random_seed()

        # Lazy device assignment (happens in worker process, not parent!)
        if self.device is None:
            if torch.cuda.is_available():
                if self.n_gpus > 1:
                    # Multi-GPU assignment in worker process
                    import os
                    import random
                    pid = os.getpid()
                    random.seed(pid)
                    gpu_id = random.randint(0, self.n_gpus - 1)
                    self.device = f'cuda:{gpu_id}'
                    print(f"  [Worker PID {pid}] Assigned to GPU {gpu_id}")
                else:
                    self.device = 'cuda'
            else:
                self.device = 'cpu'

        # Convert to numpy arrays
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Handle target shape
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]

        # Split into train/validation (skip if validation_fraction=0)
        n_samples = X.shape[0]
        n_val = int(n_samples * self.validation_fraction)

        # Check if using internal validation
        use_validation = n_val > 0

        if use_validation:
            # Shuffle indices for train/val split
            indices = np.arange(n_samples)
            np.random.seed(self.random_state)
            np.random.shuffle(indices)

            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
        else:
            # No validation split - use all data for training
            X_train, y_train = X, y
            train_indices = np.arange(n_samples)

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                 shuffle=True, drop_last=False, num_workers=0)

        # Only create validation loader if using validation
        if use_validation:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                   shuffle=False, num_workers=0)

        # Create model
        self.model = PyTorchRectangularNet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            depth=self.depth,
            width=self.width,
            activation=self.activation
        )
        # Only move to device if not CPU (avoid Mac segfault)
        if self.device != 'cpu':
            self.model = self.model.to(self.device)

        # Create optimizer and loss function
        optimizer = self._create_optimizer(self.model)
        criterion = nn.MSELoss()

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                # Skip .to(device) for CPU to avoid Mac segfault
                if self.device != 'cpu':
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)

            train_loss /= len(train_indices)

            # Validation phase (only if using internal validation)
            if use_validation:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        # Skip .to(device) for CPU to avoid Mac segfault
                        if self.device != 'cpu':
                            batch_X = batch_X.to(self.device)
                            batch_y = batch_y.to(self.device)

                        predictions = self.model(batch_X)
                        loss = criterion(predictions, batch_y)
                        val_loss += loss.item() * batch_X.size(0)

                val_loss /= len(val_indices)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                # Verbose output
                if self.verbose and (epoch % 50 == 0 or epoch == self.max_epochs - 1):
                    print(f"  Epoch {epoch+1}/{self.max_epochs} - "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

                # Early stopping (only if patience is set)
                if self.patience is not None and patience_counter >= self.patience:
                    if self.verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break
            else:
                # No validation - just print training loss if verbose
                if self.verbose and (epoch % 50 == 0 or epoch == self.max_epochs - 1):
                    print(f"  Epoch {epoch+1}/{self.max_epochs} - Train Loss: {train_loss:.6f}")

        # Restore best model
        if hasattr(self, 'best_model_state'):
            if self.device != 'cpu':
                self.model.load_state_dict({k: v.to(self.device)
                                           for k, v in self.best_model_state.items()})
            else:
                # Direct load for CPU (avoid .to() call)
                self.model.load_state_dict(self.best_model_state)

        return self

    def predict(self, X):
        """Predict using the neural network.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns
        -------
        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")

        X = np.asarray(X, dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            # Only move to device if not CPU (avoid Mac segfault)
            if self.device != 'cpu':
                X_tensor = X_tensor.to(self.device)
            predictions = self.model(X_tensor)
            # Only move to CPU if on another device
            if self.device != 'cpu':
                predictions = predictions.cpu()
            predictions = predictions.numpy()

        # Return shape consistent with sklearn
        if self.output_dim == 1:
            return predictions.ravel()
        else:
            return predictions


class NeuralNetReactorModel(ReactorModelBase):
    """PyTorch neural network implementation of the reactor model."""

    def __init__(self, scale_features=True, **kwargs):
        """Initialize the PyTorch neural network reactor model.

        Parameters
        ----------
        scale_features : bool, optional
            Whether to apply feature scaling (recommended for neural networks)
        **kwargs : dict
            Additional parameters for PyTorchRegressorWrapper
            - depth: number of hidden layers
            - width: neurons per layer
            - activation: activation function
            - learning_rate: learning rate
            - batch_size: batch size
            - max_epochs: maximum epochs
            - weight_decay: L2 regularization
            - patience: early stopping patience
            - optimizer: optimizer type ('adam', 'sgd', 'adamw', 'rmsprop')
            - device: 'cuda' or 'cpu'

        Returns
        -------
        None
        """
        super().__init__()  # Initialize base class
        self.model_class_name = 'neural_net'

        # Set default parameters if not provided
        if 'max_epochs' not in kwargs:
            kwargs['max_epochs'] = 1000
        if 'patience' not in kwargs:
            kwargs['patience'] = 20
        if 'validation_fraction' not in kwargs:
            kwargs['validation_fraction'] = 0.1
        if 'device' not in kwargs:
            # Auto-detect: prefer GPU if available
            kwargs['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Auto-detected device: {kwargs['device']}")
        if 'verbose' not in kwargs:
            kwargs['verbose'] = False

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

        self.flux_model = PyTorchRegressorWrapper(**self.params)
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

        self.keff_model = PyTorchRegressorWrapper(**self.params)
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
            'scale_features': self.scale_features,
            'pytorch_version': torch.__version__
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
