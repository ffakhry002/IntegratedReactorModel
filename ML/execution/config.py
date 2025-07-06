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
        self.test_size = 0.15
        self.random_state = 42
