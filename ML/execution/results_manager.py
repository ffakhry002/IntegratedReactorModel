import json
from datetime import datetime

class ResultsManager:
    """Manage training results and summaries"""

    def initialize_results(self, config, data_splits):
        """Initialize results dictionary"""
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
        """Add a model's results"""
        if not hasattr(self, 'all_results'):
            self.all_results = {'flux_results': {}, 'keff_results': {}}

        result_key = f'{target}_results'
        self.all_results[result_key][model_type] = {
            'best_params': best_params,
            'metrics': metrics
        }

    def save_json_results(self, results, filepath):
        """Save results to JSON file"""
        # Merge with stored results
        if hasattr(self, 'all_results'):
            results.update(self.all_results)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

    def save_text_summary(self, results, filepath, duration):
        """Save human-readable summary to text file"""
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
                                       key=lambda x: x[1]['metrics']['mse'])
                        f.write(f"Best Model: {best_model[0]}\n")
                        f.write(f"Best MSE: {best_model[1]['metrics']['mse']:.6f}\n")
                        f.write(f"Best R²: {best_model[1]['metrics']['r2']:.4f}\n\n")

                    # All models
                    f.write("All Models:\n")
                    f.write("-"*40 + "\n")
                    f.write(f"{'Model':<20} {'MSE':<12} {'RMSE':<12} {'R²':<8} {'Rel Error':<10}\n")
                    f.write("-"*60 + "\n")

                    for model_name, result in sorted(target_results.items()):
                        metrics = result['metrics']
                        f.write(f"{model_name:<20} "
                               f"{metrics['mse']:<12.6f} "
                               f"{metrics['rmse']:<12.6f} "
                               f"{metrics['r2']:<8.4f} "
                               f"{metrics['relative_error']:<10.4%}\n")

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
        """Print best models to console"""
        if hasattr(self, 'all_results'):
            results.update(self.all_results)

        print("\n" + "-"*60)
        print("BEST MODELS SUMMARY")
        print("-"*60)

        for target in ['flux', 'keff']:
            target_results = results.get(f'{target}_results', {})
            if target_results:
                best_model = min(target_results.items(),
                               key=lambda x: x[1]['metrics']['mse'])
                print(f"\n{target.upper()} Prediction:")
                print(f"  Best Model: {best_model[0]}")
                print(f"  MSE: {best_model[1]['metrics']['mse']:.6f}")
                print(f"  R²: {best_model[1]['metrics']['r2']:.4f}")

    def save_text_summary_multi_encoding(self, results, filepath, duration):
        """Save human-readable summary for multi-encoding results"""
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

                    # Find best model
                    best_model = min(results[flux_key].items(),
                                key=lambda x: x[1]['metrics']['mse'])
                    f.write(f"Best Model: {best_model[0]}\n")
                    f.write(f"Best MSE: {best_model[1]['metrics']['mse']:.6f}\n")
                    f.write(f"Best R²: {best_model[1]['metrics']['r2']:.4f}\n\n")

                    # All models table
                    f.write(f"{'Model':<20} {'MSE':<12} {'RMSE':<12} {'R²':<8} {'Rel Error':<10}\n")
                    f.write("-"*60 + "\n")

                    for model_name, result in sorted(results[flux_key].items()):
                        metrics = result['metrics']
                        f.write(f"{model_name:<20} "
                            f"{metrics['mse']:<12.6f} "
                            f"{metrics['rmse']:<12.6f} "
                            f"{metrics['r2']:<8.4f} "
                            f"{metrics['relative_error']:<10.4%}\n")

                # K-eff results
                keff_key = f'{encoding}_keff_results'
                if keff_key in results and results[keff_key]:
                    f.write(f"\nK-EFF PREDICTION RESULTS ({encoding})\n")
                    f.write("-"*40 + "\n")

                    # Find best model
                    best_model = min(results[keff_key].items(),
                                key=lambda x: x[1]['metrics']['mse'])
                    f.write(f"Best Model: {best_model[0]}\n")
                    f.write(f"Best MSE: {best_model[1]['metrics']['mse']:.6f}\n")
                    f.write(f"Best R²: {best_model[1]['metrics']['r2']:.4f}\n\n")

                    # All models table
                    f.write(f"{'Model':<20} {'MSE':<12} {'RMSE':<12} {'R²':<8} {'Rel Error':<10}\n")
                    f.write("-"*60 + "\n")

                    for model_name, result in sorted(results[keff_key].items()):
                        metrics = result['metrics']
                        f.write(f"{model_name:<20} "
                            f"{metrics['mse']:<12.6f} "
                            f"{metrics['rmse']:<12.6f} "
                            f"{metrics['r2']:<8.4f} "
                            f"{metrics['relative_error']:<10.4%}\n")

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
                        if result['metrics']['mse'] < best_flux_mse:
                            best_flux_mse = result['metrics']['mse']
                            best_flux = (model_name, encoding, result)

                keff_key = f'{encoding}_keff_results'
                if keff_key in results and results[keff_key]:
                    for model_name, result in results[keff_key].items():
                        if result['metrics']['mse'] < best_keff_mse:
                            best_keff_mse = result['metrics']['mse']
                            best_keff = (model_name, encoding, result)

            if best_flux:
                f.write(f"\nBest Flux Model: {best_flux[0]} with {best_flux[1]} encoding\n")
                f.write(f"  MSE: {best_flux[2]['metrics']['mse']:.6f}\n")
                f.write(f"  R²: {best_flux[2]['metrics']['r2']:.4f}\n")

            if best_keff:
                f.write(f"\nBest K-eff Model: {best_keff[0]} with {best_keff[1]} encoding\n")
                f.write(f"  MSE: {best_keff[2]['metrics']['mse']:.6f}\n")
                f.write(f"  R²: {best_keff[2]['metrics']['r2']:.4f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("="*80 + "\n")

    def print_best_models_multi_encoding(self, results):
        """Print best models across all encodings to console"""
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
                        if result['metrics']['mse'] < best_mse:
                            best_mse = result['metrics']['mse']
                            best_model = model_name
                            best_encoding = encoding
                            best_r2 = result['metrics']['r2']

            if best_model:
                print(f"\n{target.upper()} Prediction:")
                print(f"  Best Model: {best_model}")
                print(f"  Best Encoding: {best_encoding}")
                print(f"  MSE: {best_mse:.6f}")
                print(f"  R²: {best_r2:.4f}")

    def save_text_summary_complete(self, results, filepath, duration):
        """Save comprehensive summary for all combinations"""
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
                            if result['metrics']['mse'] < best_flux_mse:
                                best_flux_mse = result['metrics']['mse']
                                best_flux = (model_name, encoding, optimization, result)

                    keff_key = f'{encoding}_{optimization}_keff_results'
                    if keff_key in results and results[keff_key]:
                        for model_name, result in results[keff_key].items():
                            if result['metrics']['mse'] < best_keff_mse:
                                best_keff_mse = result['metrics']['mse']
                                best_keff = (model_name, encoding, optimization, result)

            if best_flux:
                f.write(f"Best Flux Model: {best_flux[0]}\n")
                f.write(f"  Encoding: {best_flux[1]}\n")
                f.write(f"  Optimization: {best_flux[2]}\n")
                f.write(f"  MSE: {best_flux[3]['metrics']['mse']:.6f}\n")
                f.write(f"  R²: {best_flux[3]['metrics']['r2']:.4f}\n\n")

            if best_keff:
                f.write(f"Best K-eff Model: {best_keff[0]}\n")
                f.write(f"  Encoding: {best_keff[1]}\n")
                f.write(f"  Optimization: {best_keff[2]}\n")
                f.write(f"  MSE: {best_keff[3]['metrics']['mse']:.6f}\n")
                f.write(f"  R²: {best_keff[3]['metrics']['r2']:.4f}\n\n")

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
                            metrics = result['metrics']
                            f.write(f"{model_name:<20} "
                                f"{metrics['mse']:<12.6f} "
                                f"{metrics['rmse']:<12.6f} "
                                f"{metrics['r2']:<8.4f}\n")

                    # K-eff results
                    keff_key = f'{encoding}_{optimization}_keff_results'
                    if keff_key in results and results[keff_key]:
                        f.write(f"\nK-eff Models ({encoding} - {optimization}):\n")
                        f.write(f"{'Model':<20} {'MSE':<12} {'RMSE':<12} {'R²':<8}\n")
                        f.write("-"*52 + "\n")

                        for model_name, result in sorted(results[keff_key].items()):
                            metrics = result['metrics']
                            f.write(f"{model_name:<20} "
                                f"{metrics['mse']:<12.6f} "
                                f"{metrics['rmse']:<12.6f} "
                                f"{metrics['r2']:<8.4f}\n")

    def print_best_models_complete(self, results):
        """Print best models across all combinations"""
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

            for encoding in results['training_info']['encodings']:
                for optimization in results['training_info']['optimizations']:
                    result_key = f'{encoding}_{optimization}_{target}_results'
                    if result_key in results and results[result_key]:
                        for model_name, result in results[result_key].items():
                            if result['metrics']['mse'] < best_mse:
                                best_mse = result['metrics']['mse']
                                best_model = model_name
                                best_encoding = encoding
                                best_optimization = optimization
                                best_r2 = result['metrics']['r2']

            if best_model:
                print(f"\n{target.upper()} Prediction:")
                print(f"  Best Model: {best_model}")
                print(f"  Encoding: {best_encoding}")
                print(f"  Optimization: {best_optimization}")
                print(f"  MSE: {best_mse:.6f}")
                print(f"  R²: {best_r2:.4f}")
