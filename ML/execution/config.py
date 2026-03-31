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
        self.flux_mode = 'total'
        # Thesis neural-net layout (1–5) for energy_sixteen + Ray Tune; see neural_net_configs.data_pipeline
        self.nn_config = None
        # Flux group index (0–3) for single-HPO-per-job runs (configs 2, 3, 5)
        self.flux_group = None