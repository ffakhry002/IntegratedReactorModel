class TrainingConfig:
    """Configuration container for training settings"""

    def __init__(self):
        """Initialize TrainingConfig with default values.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.targets = []
        self.models = []
        self.encoding = 'physics'
        self.optimization = 'optuna'
        self.n_trials = 250
        self.n_jobs = -1
        self.n_gpus = 1  # Number of GPUs to use for neural network training
        self.test_size = 0.0  # Use all data for training; validation via CV
        self.random_state = 42
        self.use_restructured = False  # Position-pooling for flux (XGBoost + physics + fill only)