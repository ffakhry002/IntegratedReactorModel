import os
import sys
from datetime import datetime
from .data_handler import DataHandler
from .model_trainer import ModelTrainer
from .results_manager import ResultsManager
from .config import TrainingConfig

class InteractiveTrainer:
    """Interactive menu system for model training"""

    def __init__(self, outputs_dir=None):
        self.config = TrainingConfig()
        self.data_handler = DataHandler()
        self.model_trainer = ModelTrainer(data_handler=self.data_handler)
        self.results_manager = ResultsManager()

        # Set outputs directory (default to current directory structure)
        if outputs_dir is None:
            self.outputs_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.outputs_dir = outputs_dir

        # Define subdirectories
        self.models_dir = os.path.join(self.outputs_dir, "models")
        self.results_dir = os.path.join(self.outputs_dir, "results")
        self.logs_dir = os.path.join(self.outputs_dir, "logs")

    def get_yes_no(self, prompt, default='n'):
        """Get yes/no input from user"""
        while True:
            response = input(f"{prompt} (y/n, default: {default}): ").strip().lower()
            if response == '':
                response = default
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            print("Please enter 'y' or 'n'")

    def get_choice(self, prompt, options, default=None):
        """Get choice from list of options"""
        print(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")

        while True:
            if default:
                choice = input(f"Enter choice (1-{len(options)}, default: {default}): ").strip()
                if choice == '':
                    return options[default-1]
            else:
                choice = input(f"Enter choice (1-{len(options)}): ").strip()

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            except ValueError:
                pass
            print(f"Please enter a number between 1 and {len(options)}")

    def select_target(self):
        """Select what to predict"""
        print("\n" + "-"*40)
        print("PREDICTION TARGET SELECTION")
        print("-"*40)

        targets = []

        if self.get_yes_no("Train model for flux prediction?", 'y'):
            # NEW: Ask for flux mode
            print("\n" + "-"*30)
            print("FLUX PREDICTION MODE")
            print("-"*30)
            flux_modes = [
                ('total', 'Total Flux - Single value per position (current behavior)'),
                ('energy', 'Energy Flux - Absolute flux for thermal/epithermal/fast'),
                ('bin', 'Energy Bins - Percentage distribution (thermal/epithermal/fast)')
            ]

            print("Select flux prediction mode:")
            for i, (mode, desc) in enumerate(flux_modes, 1):
                print(f"  {i}. {desc}")

            while True:
                choice = input(f"Enter choice (1-3, default: 1): ").strip()
                if choice == '':
                    self.flux_mode = 'total'
                    break
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(flux_modes):
                        self.flux_mode = flux_modes[idx][0]
                        break
                except ValueError:
                    pass
                print("Please enter 1, 2, or 3")

            print(f"\nSelected flux mode: {self.flux_mode}")
            targets.append('flux')

        if self.get_yes_no("Train model for k-effective prediction?", 'y'):
            targets.append('keff')

        if not targets:
            print("\nNo targets selected. Please select at least one target.")
            return self.select_target()

        return targets

    def select_models(self):
        """Select which models to train"""
        print("\n" + "-"*40)
        print("MODEL SELECTION")
        print("-"*40)

        models = []

        if self.get_yes_no("Train XGBoost model?", 'y'):
            models.append('xgboost')

        if self.get_yes_no("Train Random Forest model?", 'y'):
            models.append('random_forest')

        if self.get_yes_no("Train SVM model?", 'n'):
            models.append('svm')

        if self.get_yes_no("Train Neural Network model?", 'n'):
            models.append('neural_net')

        if not models:
            print("\nNo models selected. Please select at least one model.")
            return self.select_models()

        return models

    def select_encodings(self):
        """Select multiple encoding methods"""
        print("\n" + "-"*40)
        print("ENCODING METHOD SELECTION")
        print("-"*40)
        print("Select encoding methods to test:")

        encodings = {
            'one_hot': 'One-Hot Encoding (Baseline)',
            'categorical': 'Categorical Encoding (Compact)',
            'physics': 'Physics-Based Encoding (RECOMMENDED)',
            'spatial': 'Spatial Convolution Encoding (3x3 patterns)',
            'graph': 'Graph-Based Encoding (Network features)'
        }

        selected_encodings = []

        for encoding_key, encoding_desc in encodings.items():
            if self.get_yes_no(f"Use {encoding_desc}?", 'y' if encoding_key == 'physics' else 'n'):
                selected_encodings.append(encoding_key)

        if not selected_encodings:
            print("\nNo encodings selected. Please select at least one encoding.")
            return self.select_encodings()

        print(f"\nSelected {len(selected_encodings)} encoding method(s): {', '.join(selected_encodings)}")
        return selected_encodings

    def select_optimizations(self):
        """Select multiple hyperparameter optimization methods"""
        print("\n" + "-"*40)
        print("HYPERPARAMETER OPTIMIZATION SELECTION")
        print("-"*40)
        print("Select optimization methods to use:")

        optimizations = {
            'optuna': 'Optuna (Bayesian optimization - RECOMMENDED)',
            'three_stage': 'Three-Stage (Random → Grid → Bayesian)',
            'none': 'No optimization (use default parameters)'
        }

        selected_optimizations = []

        for opt_key, opt_desc in optimizations.items():
            default = 'y' if opt_key == 'optuna' else 'n'
            if self.get_yes_no(f"Use {opt_desc}?", default):
                selected_optimizations.append(opt_key)

        if not selected_optimizations:
            print("\nNo optimization methods selected. Please select at least one method.")
            return self.select_optimizations()

        print(f"\nSelected {len(selected_optimizations)} optimization method(s): {', '.join(selected_optimizations)}")
        return selected_optimizations

    def get_parallel_settings(self):
        """Get parallel computing settings"""
        print("\n" + "-"*40)
        print("PARALLEL COMPUTING SETTINGS")
        print("-"*40)

        if self.get_yes_no("Use parallel computing?", 'y'):
            try:
                import multiprocessing
                max_cores = multiprocessing.cpu_count()
                print(f"\nDetected {max_cores} CPU cores")

                while True:
                    cores = input(f"Number of cores to use (1-{max_cores}, default: all): ").strip()
                    if cores == '':
                        return -1  # Use all cores
                    try:
                        cores = int(cores)
                        if 1 <= cores <= max_cores:
                            return cores
                    except ValueError:
                        pass
                    print(f"Please enter a number between 1 and {max_cores}")
            except:
                print("Could not detect CPU cores, using single core")
                return 1
        else:
            return 1

    def run(self):
        """Run the interactive training process"""
        # Create necessary directories (now under outputs)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Initialize flux mode
        self.flux_mode = 'total'  # Default

        # Get user selections
        self.config.targets = self.select_target()
        self.config.models = self.select_models()
        self.config.encodings = self.select_encodings()
        self.config.optimizations = self.select_optimizations()  # Now returns a list!

        # Store flux mode in config
        self.config.flux_mode = self.flux_mode

        # Get additional settings for optuna if selected
        if 'optuna' in self.config.optimizations:
            n_trials = input("\nNumber of Optuna trials (default: 250): ").strip()
            self.config.n_trials = int(n_trials) if n_trials else 250

        self.config.n_jobs = self.get_parallel_settings()

        # Data file selection
        print("\n" + "-"*40)
        print("DATA FILE SELECTION")
        print("-"*40)

        data_file = input("Path to training data (default: data/train.txt): ").strip()
        if not data_file:
            data_file = 'data/train.txt'

        # Calculate total training jobs
        total_jobs = (len(self.config.targets) * len(self.config.models) *
                    len(self.config.encodings) * len(self.config.optimizations))

        # Summary of selections
        print("\n" + "="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)
        print(f"Targets: {', '.join(self.config.targets)}")
        if 'flux' in self.config.targets:
            print(f"Flux mode: {self.flux_mode}")
        print(f"Models: {', '.join(self.config.models)}")
        print(f"Encodings: {', '.join(self.config.encodings)}")
        print(f"Optimizations: {', '.join(self.config.optimizations)}")
        if 'optuna' in self.config.optimizations:
            print(f"Optuna trials: {self.config.n_trials}")
        print(f"Parallel cores: {'All' if self.config.n_jobs == -1 else self.config.n_jobs}")
        print(f"Data file: {data_file}")
        print(f"Total training jobs: {total_jobs}")

        # Time estimate
        time_per_job = {
            'optuna': {'xgboost': 3, 'random_forest': 3, 'svm': 15, 'neural_net': 20},
            'three_stage': {'xgboost': 2, 'random_forest': 2, 'svm': 10, 'neural_net': 15},
            'none': {'xgboost': 0.1, 'random_forest': 0.1, 'svm': 0.2, 'neural_net': 0.3}
        }

        estimated_time = 0
        for opt in self.config.optimizations:
            for model in self.config.models:
                estimated_time += time_per_job.get(opt, {}).get(model, 5) * len(self.config.targets) * len(self.config.encodings)

        print(f"Estimated time: {estimated_time:.0f}-{estimated_time*1.5:.0f} minutes")
        print("="*60)

        if not self.get_yes_no("\nProceed with training?", 'y'):
            print("Training cancelled.")
            return

        # Start training
        print("\n" + "="*60)
        print("STARTING TRAINING PROCESS")
        print("="*60)

        start_time = datetime.now()

        try:
            # Initialize results tracking
            all_results = {
                'training_info': {
                    'date': start_time.isoformat(),
                    'encodings': self.config.encodings,
                    'optimizations': self.config.optimizations,
                    'targets': self.config.targets,
                    'models': self.config.models,
                    'flux_mode': self.flux_mode if 'flux' in self.config.targets else None  # NEW
                }
            }

            # Add result dictionaries for each combination
            for encoding in self.config.encodings:
                for optimization in self.config.optimizations:
                    all_results[f'{encoding}_{optimization}_flux_results'] = {}
                    all_results[f'{encoding}_{optimization}_keff_results'] = {}

            job_counter = 0

            # Loop through all combinations
            for optimization in self.config.optimizations:
                print(f"\n{'='*60}")
                print(f"OPTIMIZATION METHOD: {optimization.upper()}")
                print(f"{'='*60}")

                # Update config for this optimization
                self.config.optimization = optimization

                for encoding in self.config.encodings:
                    print(f"\n{'='*50}")
                    print(f"ENCODING: {encoding.upper()} | OPTIMIZATION: {optimization.upper()}")
                    print(f"{'='*50}")

                    # Train models for each target
                    for target in self.config.targets:
                        print(f"\n{'-'*40}")
                        print(f"{target.upper()} MODELS - {encoding.upper()} - {optimization.upper()}")
                        if target == 'flux':
                            print(f"Flux mode: {self.flux_mode}")
                        print(f"{'-'*40}")

                        # Load and prepare data with current encoding
                        print(f"\nLoading data with {encoding} encoding...")
                        # Pass flux mode for flux targets
                        flux_mode_to_use = self.flux_mode if target == 'flux' else 'total'
                        result = self.data_handler.load_and_prepare_data(
                            data_file,
                            encoding,
                            flux_mode=flux_mode_to_use
                        )

                        # Handle both old and new return formats
                        if len(result) == 4:
                            X, y_flux, y_keff, groups = result
                        else:
                            X, y_flux, y_keff = result
                            groups = None  # Fallback for compatibility
                            print("WARNING: No groups returned - may have data leakage!")

                        # Split data
                        print("Splitting data...")
                        data_splits = self.data_handler.split_data(X, y_flux, y_keff, groups)

                        # Update training info
                        if 'n_samples' not in all_results['training_info']:
                            all_results['training_info'].update({
                                'n_samples': X.shape[0],
                                'train_size': data_splits['X_train'].shape[0],
                                'test_size': data_splits['X_test'].shape[0]
                            })

                        all_results['training_info'][f'{encoding}_features'] = X.shape[1]

                        for model_type in self.config.models:
                            job_counter += 1
                            print(f"\nJob {job_counter}/{total_jobs}: {model_type} for {target}")
                            print(f"Encoding: {encoding} | Optimization: {optimization}")
                            if target == 'flux':
                                print(f"Flux mode: {self.flux_mode}")
                            print("-"*30)

                            # Train model
                            model, metrics, best_params = self.model_trainer.train_model(
                                model_type=model_type,
                                target=target,
                                data_splits=data_splits,
                                config=self.config,
                                encoding=encoding
                            )

                            # Save results with encoding and optimization specific key
                            result_key = f'{encoding}_{optimization}_{target}_results'
                            if result_key not in all_results:
                                all_results[result_key] = {}

                            all_results[result_key][model_type] = {
                                'best_params': best_params,
                                'metrics': metrics,
                                'encoding': encoding,
                                'optimization': optimization,
                                'flux_mode': self.flux_mode if target == 'flux' else None  # NEW
                            }

                            # Save model with all identifiers in filename
                            if target == 'flux' and self.flux_mode != 'total':
                                # Include flux mode in filename for non-total modes
                                model_path = os.path.join(self.models_dir,
                                    f'{model_type}_{target}_{self.flux_mode}_{encoding}_{optimization}.pkl')
                            else:
                                model_path = os.path.join(self.models_dir,
                                    f'{model_type}_{target}_{encoding}_{optimization}.pkl')

                            # Fixed save_model call - removed flux_mode from metadata since it's passed separately
                            metadata = {
                                'params': best_params,
                                'metrics': metrics,
                                'model_class': model_type
                            }
                            self.model_trainer.save_model(
                                model,
                                model_path,
                                metadata,
                                model_type,     # model type (xgboost, random_forest, etc.)
                                target,         # target (flux or keff)
                                encoding,       # encoding method
                                optimization    # optimization method
                            )

            # Save all results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Save JSON results
            results_json = os.path.join(self.results_dir,
                f'training_results_complete_{end_time.strftime("%Y%m%d_%H%M%S")}.json')
            self.results_manager.save_json_results(all_results, results_json)

            # Save text summary
            summary_file = os.path.join(self.results_dir,
                f'training_summary_complete_{end_time.strftime("%Y%m%d_%H%M%S")}.txt')
            self.results_manager.save_text_summary_complete(all_results, summary_file, duration)

            # Print summary
            print("\n" + "="*60)
            print("TRAINING COMPLETE!")
            print("="*60)
            print(f"Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            print(f"Total jobs completed: {job_counter}")
            print(f"Models saved: {job_counter}")
            print(f"\nResults saved to:")
            print(f"  - {results_json}")
            print(f"  - {summary_file}")

            # Display best models
            self.results_manager.print_best_models_complete(all_results)

        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()

            # Save error log
            error_log = os.path.join(self.logs_dir,
                f'error_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            with open(error_log, 'w') as f:
                f.write(f"Error during training: {e}\n\n")
                traceback.print_exc(file=f)
            print(f"\nError details saved to: {error_log}")
