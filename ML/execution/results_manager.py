import json
from datetime import datetime

class ResultsManager:
    """Manage training results and summaries"""

    @staticmethod
    def _get_comparable_metric(result):
        """Get a comparable metric for model ranking.

        When test_size=0.0 (CV-only validation), MSE is None.
        In that case, use MAPE (CV score) instead. Both are "lower is better".

        Parameters
        ----------
        result : dict
            Result dictionary with 'metrics' key

        Returns
        -------
        float
            Comparable metric value (MSE if available, otherwise MAPE)
        """
        metrics = result.get('metrics') or {}
        if not metrics:
            return float('inf')
        mse = metrics.get('mse')
        if mse is not None:
            return mse
        mape = metrics.get('mape', float('inf'))
        return mape if mape is not None else float('inf')

    def initialize_results(self, config, data_splits):
        """Initialize results dictionary with training configuration.

        Parameters
        ----------
        config : TrainingConfig
            Training configuration object
        data_splits : dict
            Dictionary containing train/test data splits

        Returns
        -------
        dict
            Initialized results dictionary with training info and empty result containers
        """
        return {
            'training_info': {
                'date': datetime.now().isoformat(),
                'encoding': config.encoding,
                'optimization': config.optimization,
                'n_samples': data_splits['X_train'].shape[0] + data_splits['X_test'].shape[0],
                'n_features': data_splits['X_train'].shape[1],
                'train_size': data_splits['X_train'].shape[0],
                'test_size': data_splits['X_test'].shape[0],
                'targets': config.targets,
                'models': config.models
            },
            'flux_results': {},
            'keff_results': {}
        }

    def add_result(self, target, model_type, model, metrics, best_params):
        """Add a model's results to the internal results storage.

        Parameters
        ----------
        target : str
            Target type ('flux' or 'keff')
        model_type : str
            Type of model (e.g. 'xgboost', 'random_forest')
        model : object
            Trained model object
        metrics : dict
            Dictionary of performance metrics
        best_params : dict
            Dictionary of best hyperparameters

        Returns
        -------
        None
        """
        if not hasattr(self, 'all_results'):
            self.all_results = {'flux_results': {}, 'keff_results': {}}

        result_key = f'{target}_results'
        self.all_results[result_key][model_type] = {
            'best_params': best_params,
            'metrics': metrics
        }

    def save_json_results(self, results, filepath):
        """Save results to JSON file.

        Parameters
        ----------
        results : dict
            Results dictionary to save
        filepath : str
            Path to save the JSON file

        Returns
        -------
        None
        """
        # Merge with stored results
        if hasattr(self, 'all_results'):
            results.update(self.all_results)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

    def save_text_summary(self, results, filepath, duration):
        """Save human-readable summary to text file.

        Parameters
        ----------
        results : dict
            Results dictionary to summarize
        filepath : str
            Path to save the text summary
        duration : float
            Training duration in seconds

        Returns
        -------
        None
        """
        if hasattr(self, 'all_results'):
            results.update(self.all_results)

        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("NUCLEAR REACTOR ML TRAINING SUMMARY\n")
            f.write("="*80 + "\n\n")

            # Training info
            info = results['training_info']
            f.write("TRAINING CONFIGURATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Date: {info['date']}\n")
            f.write(f"Duration: {duration:.1f} seconds\n")
            f.write(f"Encoding: {info['encoding']}\n")
            f.write(f"Optimization: {info['optimization']}\n")
            f.write(f"Total samples: {info['n_samples']}\n")
            f.write(f"Features: {info['n_features']}\n")
            f.write(f"Train size: {info['train_size']}\n")
            f.write(f"Test size: {info['test_size']}\n")
            f.write(f"Targets: {', '.join(info['targets'])}\n")
            f.write(f"Models: {', '.join(info['models'])}\n")
            f.write("\n")

            # Results for each target
            for target in ['flux', 'keff']:
                target_results = results.get(f'{target}_results', {})
                if target_results:
                    f.write(f"\n{target.upper()} PREDICTION RESULTS\n")
                    f.write("="*60 + "\n\n")

                    # Find best model
                    if target_results:
                        best_model = min(target_results.items(),
                                       key=lambda x: self._get_comparable_metric(x[1]))
                        f.write(f"Best Model: {best_model[0]}\n")
                        bm = best_model[1].get('metrics') or {}
                        if bm.get('mse') is not None:
                            f.write(f"Best MSE: {bm['mse']:.6f}\n")
                            f.write(f"Best R²: {bm.get('r2', 0):.4f}\n\n")
                        elif bm.get('mape') is not None:
                            f.write(f"Best CV MAPE: {bm['mape']:.2f}%\n")
                            f.write(f"(Validated via CV only, no test set)\n\n")
                        else:
                            f.write(f"(Submodel-only run — no test metrics)\n\n")

                    # All models
                    f.write("All Models:\n")
                    f.write("-"*40 + "\n")
                    f.write(f"{'Model':<20} {'MSE':<12} {'RMSE':<12} {'R²':<8} {'Rel Error':<10}\n")
                    f.write("-"*60 + "\n")

                    for model_name, result in sorted(target_results.items()):
                        metrics = result.get('metrics') or {}
                        if not metrics:
                            f.write(f"{model_name:<20} (no test metrics)\n")
                        elif metrics.get('mse') is not None:
                            f.write(f"{model_name:<20} "
                                   f"{metrics['mse']:<12.6f} "
                                   f"{metrics.get('rmse', 0):<12.6f} "
                                   f"{metrics.get('r2', 0):<8.4f} "
                                   f"{metrics.get('relative_error', 0):<10.4%}\n")
                        else:
                            mape = metrics.get('mape', 0.0)
                            f.write(f"{model_name:<20} "
                                   f"{'CV':<12} "
                                   f"{mape:<12.2f}% "
                                   f"{'N/A':<8} "
                                   f"{'N/A':<10}\n")

                    # Best parameters
                    f.write("\nBest Parameters:\n")
                    f.write("-"*40 + "\n")
                    for model_name, result in target_results.items():
                        f.write(f"\n{model_name}:\n")
                        for param, value in result['best_params'].items():
                            f.write(f"  {param}: {value}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("="*80 + "\n")

    def print_best_models(self, results):
        """Print best models to console.

        Parameters
        ----------
        results : dict
            Results dictionary containing model performance data

        Returns
        -------
        None
        """
        if hasattr(self, 'all_results'):
            results.update(self.all_results)

        print("\n" + "-"*60)
        print("BEST MODELS SUMMARY")
        print("-"*60)

        for target in ['flux', 'keff']:
            target_results = results.get(f'{target}_results', {})
            if target_results:
                best_model = min(target_results.items(),
                               key=lambda x: self._get_comparable_metric(x[1]))
                print(f"\n{target.upper()} Prediction:")
                print(f"  Best Model: {best_model[0]}")
                bm = best_model[1].get('metrics') or {}
                if bm.get('mse') is not None:
                    print(f"  MSE: {bm['mse']:.6f}")
                    print(f"  R²: {bm.get('r2', 0):.4f}")
                elif bm.get('mape') is not None:
                    print(f"  CV MAPE: {bm['mape']:.2f}%")
                    print(f"  (Validated via CV only)")
                else:
                    print(f"  (Submodel-only run — no test metrics)")

    def save_text_summary_multi_encoding(self, results, filepath, duration):
        """Save human-readable summary for multi-encoding results.

        Parameters
        ----------
        results : dict
            Results dictionary containing multi-encoding training results
        filepath : str
            Path to save the text summary
        duration : float
            Training duration in seconds

        Returns
        -------
        None
        """
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("NUCLEAR REACTOR ML TRAINING SUMMARY - MULTI-ENCODING\n")
            f.write("="*80 + "\n\n")

            # Training info
            info = results['training_info']
            f.write("TRAINING CONFIGURATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Date: {info['date']}\n")
            f.write(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)\n")
            f.write(f"Encodings: {', '.join(info['encodings'])}\n")
            f.write(f"Optimization: {info['optimization']}\n")
            f.write(f"Total samples: {info['n_samples']}\n")
            f.write(f"Train size: {info['train_size']}\n")
            f.write(f"Test size: {info['test_size']}\n")
            f.write(f"Targets: {', '.join(info['targets'])}\n")
            f.write(f"Models: {', '.join(info['models'])}\n\n")

            # Feature counts per encoding
            f.write("FEATURES PER ENCODING\n")
            f.write("-"*40 + "\n")
            for encoding in info['encodings']:
                if f'{encoding}_features' in info:
                    f.write(f"{encoding}: {info[f'{encoding}_features']} features\n")
            f.write("\n")

            # Results by encoding
            for encoding in info['encodings']:
                f.write(f"\n{'='*60}\n")
                f.write(f"ENCODING: {encoding.upper()}\n")
                f.write(f"{'='*60}\n")

                # Flux results
                flux_key = f'{encoding}_flux_results'
                if flux_key in results and results[flux_key]:
                    f.write(f"\nFLUX PREDICTION RESULTS ({encoding})\n")
                    f.write("-"*40 + "\n")

                    best_model = min(results[flux_key].items(),
                                key=lambda x: self._get_comparable_metric(x[1]))
                    f.write(f"Best Model: {best_model[0]}\n")
                    bm = best_model[1].get('metrics') or {}
                    if bm.get('mse') is not None:
                        f.write(f"Best MSE: {bm['mse']:.6f}\n")
                        f.write(f"Best R²: {bm.get('r2', 0):.4f}\n\n")
                    elif bm.get('mape') is not None:
                        f.write(f"Best CV MAPE: {bm['mape']:.2f}%\n")
                        f.write(f"(Validated via CV only)\n\n")
                    else:
                        f.write(f"(Submodel-only run — no test metrics)\n\n")

                    f.write(f"{'Model':<20} {'MSE':<12} {'RMSE':<12} {'R²':<8} {'Rel Error':<10}\n")
                    f.write("-"*60 + "\n")

                    for model_name, result in sorted(results[flux_key].items()):
                        metrics = result.get('metrics') or {}
                        if not metrics:
                            f.write(f"{model_name:<20} (no test metrics)\n")
                        elif metrics.get('mse') is not None:
                            f.write(f"{model_name:<20} "
                                f"{metrics['mse']:<12.6f} "
                                f"{metrics.get('rmse', 0):<12.6f} "
                                f"{metrics.get('r2', 0):<8.4f} "
                                f"{metrics.get('relative_error', 0):<10.4%}\n")
                        else:
                            mape = metrics.get('mape', 0.0)
                            f.write(f"{model_name:<20} "
                                f"{'CV':<12} "
                                f"{mape:<12.2f}% "
                                f"{'N/A':<8} "
                                f"{'N/A':<10}\n")

                # K-eff results
                keff_key = f'{encoding}_keff_results'
                if keff_key in results and results[keff_key]:
                    f.write(f"\nK-EFF PREDICTION RESULTS ({encoding})\n")
                    f.write("-"*40 + "\n")

                    best_model = min(results[keff_key].items(),
                                key=lambda x: self._get_comparable_metric(x[1]))
                    f.write(f"Best Model: {best_model[0]}\n")
                    bm = best_model[1].get('metrics') or {}
                    if bm.get('mse') is not None:
                        f.write(f"Best MSE: {bm['mse']:.6f}\n")
                        f.write(f"Best R²: {bm.get('r2', 0):.4f}\n\n")
                    elif bm.get('mape') is not None:
                        f.write(f"Best CV MAPE: {bm['mape']:.2f}%\n")
                        f.write(f"(Validated via CV only)\n\n")
                    else:
                        f.write(f"(Submodel-only run — no test metrics)\n\n")

                    f.write(f"{'Model':<20} {'MSE':<12} {'RMSE':<12} {'R²':<8} {'Rel Error':<10}\n")
                    f.write("-"*60 + "\n")

                    for model_name, result in sorted(results[keff_key].items()):
                        metrics = result.get('metrics') or {}
                        if not metrics:
                            f.write(f"{model_name:<20} (no test metrics)\n")
                        elif metrics.get('mse') is not None:
                            f.write(f"{model_name:<20} "
                                f"{metrics['mse']:<12.6f} "
                                f"{metrics.get('rmse', 0):<12.6f} "
                                f"{metrics.get('r2', 0):<8.4f} "
                                f"{metrics.get('relative_error', 0):<10.4%}\n")
                        else:
                            mape = metrics.get('mape', 0.0)
                            f.write(f"{model_name:<20} "
                                f"{'CV':<12} "
                                f"{mape:<12.2f}% "
                                f"{'N/A':<8} "
                                f"{'N/A':<10}\n")

            # Overall best models
            f.write("\n" + "="*80 + "\n")
            f.write("OVERALL BEST MODELS\n")
            f.write("="*80 + "\n")

            # Find best across all encodings
            best_flux = None
            best_flux_mse = float('inf')
            best_keff = None
            best_keff_mse = float('inf')

            for encoding in info['encodings']:
                flux_key = f'{encoding}_flux_results'
                if flux_key in results and results[flux_key]:
                    for model_name, result in results[flux_key].items():
                        metric = self._get_comparable_metric(result)
                        if metric < best_flux_mse:
                            best_flux_mse = metric
                            best_flux = (model_name, encoding, result)

                keff_key = f'{encoding}_keff_results'
                if keff_key in results and results[keff_key]:
                    for model_name, result in results[keff_key].items():
                        metric = self._get_comparable_metric(result)
                        if metric < best_keff_mse:
                            best_keff_mse = metric
                            best_keff = (model_name, encoding, result)

            if best_flux:
                f.write(f"\nBest Flux Model: {best_flux[0]} with {best_flux[1]} encoding\n")
                bm = best_flux[2].get('metrics') or {}
                if bm.get('mse') is not None:
                    f.write(f"  MSE: {bm['mse']:.6f}\n")
                    f.write(f"  R²: {bm.get('r2', 0):.4f}\n")
                elif bm.get('mape') is not None:
                    f.write(f"  CV MAPE: {bm['mape']:.2f}%\n")
                    f.write(f"  (Validated via CV only)\n")
                else:
                    f.write(f"  (Submodel-only run — no test metrics)\n")

            if best_keff:
                f.write(f"\nBest K-eff Model: {best_keff[0]} with {best_keff[1]} encoding\n")
                bm = best_keff[2].get('metrics') or {}
                if bm.get('mse') is not None:
                    f.write(f"  MSE: {bm['mse']:.6f}\n")
                    f.write(f"  R²: {bm.get('r2', 0):.4f}\n")
                elif bm.get('mape') is not None:
                    f.write(f"  CV MAPE: {bm['mape']:.2f}%\n")
                    f.write(f"  (Validated via CV only)\n")
                else:
                    f.write(f"  (Submodel-only run — no test metrics)\n")

            f.write("\n" + "="*80 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("="*80 + "\n")

    def print_best_models_multi_encoding(self, results):
        """Print best models across all encodings to console.

        Parameters
        ----------
        results : dict
            Results dictionary containing multi-encoding training results

        Returns
        -------
        None
        """
        print("\n" + "-"*60)
        print("BEST MODELS SUMMARY")
        print("-"*60)

        # Find best for each target across all encodings
        for target in ['flux', 'keff']:
            best_model = None
            best_mse = float('inf')
            best_encoding = None

            for encoding in results['training_info']['encodings']:
                result_key = f'{encoding}_{target}_results'
                if result_key in results and results[result_key]:
                    for model_name, result in results[result_key].items():
                        metric = self._get_comparable_metric(result)
                        if metric < best_mse:
                            best_mse = metric
                            best_model = model_name
                            best_encoding = encoding
                            m = result.get('metrics') or {}
                            best_r2 = m.get('r2')
                            best_mape = m.get('mape')

            if best_model:
                print(f"\n{target.upper()} Prediction:")
                print(f"  Best Model: {best_model}")
                print(f"  Best Encoding: {best_encoding}")
                if best_r2 is not None:
                    print(f"  MSE: {best_mse:.6f}")
                    print(f"  R²: {best_r2:.4f}")
                else:
                    print(f"  CV MAPE: {best_mape:.2f}%")
                    print(f"  (Validated via CV only)")

    def save_text_summary_complete(self, results, filepath, duration):
        """Save comprehensive summary for all combinations.

        Parameters
        ----------
        results : dict
            Results dictionary containing all encoding/optimization combinations
        filepath : str
            Path to save the comprehensive text summary
        duration : float
            Training duration in seconds

        Returns
        -------
        None
        """
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("NUCLEAR REACTOR ML TRAINING SUMMARY - COMPLETE\n")
            f.write("="*80 + "\n\n")

            # Training info
            info = results['training_info']
            f.write("TRAINING CONFIGURATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Date: {info['date']}\n")
            f.write(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)\n")
            f.write(f"Encodings: {', '.join(info['encodings'])}\n")
            f.write(f"Optimizations: {', '.join(info['optimizations'])}\n")
            f.write(f"Total samples: {info['n_samples']}\n")
            f.write(f"Train size: {info['train_size']}\n")
            f.write(f"Test size: {info['test_size']}\n")
            f.write(f"Targets: {', '.join(info['targets'])}\n")
            f.write(f"Models: {', '.join(info['models'])}\n")
            f.write(f"Total combinations: {len(info['encodings']) * len(info['optimizations']) * len(info['targets']) * len(info['models'])}\n\n")

            # Best overall models
            f.write("="*80 + "\n")
            f.write("BEST OVERALL MODELS\n")
            f.write("="*80 + "\n\n")

            best_flux = None
            best_flux_mse = float('inf')
            best_keff = None
            best_keff_mse = float('inf')

            # Find best across all combinations
            for encoding in info['encodings']:
                for optimization in info['optimizations']:
                    flux_key = f'{encoding}_{optimization}_flux_results'
                    if flux_key in results and results[flux_key]:
                        for model_name, result in results[flux_key].items():
                            metric = self._get_comparable_metric(result)
                            if metric < best_flux_mse:
                                best_flux_mse = metric
                                best_flux = (model_name, encoding, optimization, result)

                    keff_key = f'{encoding}_{optimization}_keff_results'
                    if keff_key in results and results[keff_key]:
                        for model_name, result in results[keff_key].items():
                            metric = self._get_comparable_metric(result)
                            if metric < best_keff_mse:
                                best_keff_mse = metric
                                best_keff = (model_name, encoding, optimization, result)

            if best_flux:
                f.write(f"Best Flux Model: {best_flux[0]}\n")
                f.write(f"  Encoding: {best_flux[1]}\n")
                f.write(f"  Optimization: {best_flux[2]}\n")
                bm = best_flux[3].get('metrics') or {}
                if bm.get('mse') is not None:
                    f.write(f"  MSE: {bm['mse']:.6f}\n")
                    f.write(f"  R²: {bm.get('r2', 0):.4f}\n\n")
                elif bm.get('mape') is not None:
                    f.write(f"  CV MAPE: {bm['mape']:.2f}%\n")
                    f.write(f"  (Validated via CV only)\n\n")
                else:
                    f.write(f"  (Submodel-only run — no test metrics)\n\n")

            if best_keff:
                f.write(f"Best K-eff Model: {best_keff[0]}\n")
                f.write(f"  Encoding: {best_keff[1]}\n")
                f.write(f"  Optimization: {best_keff[2]}\n")
                bm = best_keff[3].get('metrics') or {}
                if bm.get('mse') is not None:
                    f.write(f"  MSE: {bm['mse']:.6f}\n")
                    f.write(f"  R²: {bm.get('r2', 0):.4f}\n\n")
                elif bm.get('mape') is not None:
                    f.write(f"  CV MAPE: {bm['mape']:.2f}%\n")
                    f.write(f"  (Validated via CV only)\n\n")
                else:
                    f.write(f"  (Submodel-only run — no test metrics)\n\n")

            # Results by optimization method
            for optimization in info['optimizations']:
                f.write(f"\n{'='*70}\n")
                f.write(f"OPTIMIZATION METHOD: {optimization.upper()}\n")
                f.write(f"{'='*70}\n")

                for encoding in info['encodings']:
                    f.write(f"\nENCODING: {encoding.upper()}\n")
                    f.write("-"*50 + "\n")

                    # Flux results
                    flux_key = f'{encoding}_{optimization}_flux_results'
                    if flux_key in results and results[flux_key]:
                        f.write(f"\nFlux Models ({encoding} - {optimization}):\n")
                        f.write(f"{'Model':<20} {'MSE':<12} {'RMSE':<12} {'R²':<8}\n")
                        f.write("-"*52 + "\n")

                        for model_name, result in sorted(results[flux_key].items()):
                            metrics = result.get('metrics') or {}
                            if not metrics:
                                f.write(f"{model_name:<20} (submodel-only, no test metrics)\n")
                            elif metrics.get('mse') is not None:
                                f.write(f"{model_name:<20} "
                                    f"{metrics['mse']:<12.6f} "
                                    f"{metrics.get('rmse', 0):<12.6f} "
                                    f"{metrics.get('r2', 0):<8.4f}\n")
                            else:
                                mape = metrics.get('mape', 0.0)
                                f.write(f"{model_name:<20} "
                                    f"{'CV MAPE':<12} "
                                    f"{mape:<12.2f}% "
                                    f"{'N/A':<8}\n")

                    # K-eff results
                    keff_key = f'{encoding}_{optimization}_keff_results'
                    if keff_key in results and results[keff_key]:
                        f.write(f"\nK-eff Models ({encoding} - {optimization}):\n")
                        f.write(f"{'Model':<20} {'MSE':<12} {'RMSE':<12} {'R²':<8}\n")
                        f.write("-"*52 + "\n")

                        for model_name, result in sorted(results[keff_key].items()):
                            metrics = result.get('metrics') or {}
                            if not metrics:
                                f.write(f"{model_name:<20} (submodel-only, no test metrics)\n")
                            elif metrics.get('mse') is not None:
                                f.write(f"{model_name:<20} "
                                    f"{metrics['mse']:<12.6f} "
                                    f"{metrics.get('rmse', 0):<12.6f} "
                                    f"{metrics.get('r2', 0):<8.4f}\n")
                            else:
                                mape = metrics.get('mape', 0.0)
                                f.write(f"{model_name:<20} "
                                    f"{'CV MAPE':<12} "
                                    f"{mape:<12.2f}% "
                                    f"{'N/A':<8}\n")

    def print_best_models_complete(self, results):
        """Print best models across all combinations.

        Parameters
        ----------
        results : dict
            Results dictionary containing all encoding/optimization combinations

        Returns
        -------
        None
        """
        print("\n" + "-"*60)
        print("BEST MODELS SUMMARY - ALL COMBINATIONS")
        print("-"*60)

        # Find best for each target
        for target in ['flux', 'keff']:
            best_model = None
            best_mse = float('inf')
            best_encoding = None
            best_optimization = None
            best_r2 = None
            best_mape = None

            for encoding in results['training_info']['encodings']:
                for optimization in results['training_info']['optimizations']:
                    result_key = f'{encoding}_{optimization}_{target}_results'
                    if result_key in results and results[result_key]:
                        for model_name, result in results[result_key].items():
                            metric = self._get_comparable_metric(result)
                            if metric < best_mse:
                                best_mse = metric
                                best_model = model_name
                                best_encoding = encoding
                                best_optimization = optimization
                                m = result.get('metrics') or {}
                                best_r2 = m.get('r2')
                                best_mape = m.get('mape')

            if best_model:
                print(f"\n{target.upper()} Prediction:")
                print(f"  Best Model: {best_model}")
                print(f"  Encoding: {best_encoding}")
                print(f"  Optimization: {best_optimization}")
                if best_mse < float('inf') and best_r2 is not None:
                    print(f"  MSE: {best_mse:.6f}")
                    print(f"  R²: {best_r2:.4f}")
                elif best_mape is not None:
                    print(f"  CV MAPE: {best_mape:.2f}%")
                    print(f"  (Validated via CV only)")
                else:
                    print(f"  (Submodel-only run — no test metrics)")
