"""
Updated task execution functions that properly handle k-medoids metrics.
"""

import os
import sys
import json
import pickle
import time
from io import StringIO

from .constants import SAMPLER_MAP, get_sampler_map


def save_sampling_results(method_name, results):
    """Save sampling results without needing to load all data."""
    # Create output directories if they don't exist
    os.makedirs('output/samples_picked/pkl', exist_ok=True)
    os.makedirs('output/samples_picked/txt', exist_ok=True)

    # Save pickle file
    pkl_path = f'output/samples_picked/pkl/{method_name}_samples.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)

    # Save JSON file (simplified version without configurations)
    json_data = {
        'method': method_name,
        'n_samples': len(results.get('selected_indices', [])),
        'selected_indices': [int(idx) for idx in results.get('selected_indices', [])],
        'diversity_score': float(results.get('diversity_score', 0)),
        'best_run': results.get('best_run'),
        'total_runs': results.get('total_runs')
    }

    # Add algorithm-specific metrics
    if 'algorithm' in results:
        json_data['algorithm'] = results['algorithm']
    if 'distance' in results:
        json_data['distance_metric'] = results['distance']
    if 'selection_metric' in results:
        json_data['selection_metric'] = results['selection_metric']

    # Add inertia for k-medoids
    if 'inertia' in results and results['inertia'] is not None:
        json_data['inertia'] = float(results['inertia'])

    # Add any distance metrics
    for key in ['min_jaccard_distance', 'min_euclidean_distance', 'min_manhattan_distance']:
        if key in results:
            json_data[key] = float(results[key])

    json_path = f'output/samples_picked/pkl/{method_name}_samples.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    # Save text summary
    txt_path = f'output/samples_picked/txt/{method_name}_summary.txt'
    with open(txt_path, 'w') as f:
        f.write(f"{method_name.upper()} SAMPLING RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Number of samples: {len(results.get('selected_indices', []))}\n")

        # Show appropriate metric based on algorithm
        if results.get('algorithm') == 'kmeans_nearest' and 'inertia' in results and results['inertia'] is not None:
            f.write(f"K-means inertia (lower is better): {results['inertia']:.4f}\n")
            f.write("Note: For lattice methods, this is inertia in MDS embedding space\n")

        if 'diversity_score' in results:
            f.write(f"Diversity score (min pairwise distance): {results['diversity_score']:.4f}\n")

        if 'best_run' in results:
            f.write(f"Best run: #{results['best_run']} of {results['total_runs']}\n")
            if results.get('selection_metric'):
                f.write(f"Selection based on: {results['selection_metric']}\n")

        f.write("\nSelected configuration indices:\n")
        indices = results.get('selected_indices', [])
        f.write(str(indices[:20]))
        if len(indices) > 20:
            f.write(f"\n... and {len(indices) - 20} more\n")


def run_single_method(method_args):
    """Run a single sampling method. Designed to be called by multiprocessing."""
    method_name, n_samples, n_runs, base_seed, use_6x6_restriction, selected_parameters = method_args

    print(f"\n[Process {os.getpid()}] Starting {method_name.upper()}")

    # Get sampler map with 6x6 restriction if needed
    sampler_map = get_sampler_map(use_6x6_restriction)

    # Create sampler instance
    sampler_class = sampler_map[method_name]
    # Check if this is a geometric method that needs selected parameters
    if selected_parameters and method_name in ['lhs', 'sobol', 'halton', 'jaccard_geometric',
                                              'euclidean_geometric', 'manhattan_geometric', 'random_geometric']:
        sampler = sampler_class(selected_parameters=selected_parameters)
    else:
        sampler = sampler_class()

    # Load data for this sampler
    sampler.load_data()

    # Check sample size
    n_configs = len(sampler.configurations)
    n_samples = min(n_samples, n_configs)

    if n_samples < method_args[1]:
        print(f"[{method_name}] Warning: Requested {method_args[1]} samples but only {n_configs} configurations available")

    # Run sampling
    start_time = time.time()
    results = sampler.sample(n_samples, n_runs=n_runs, base_seed=base_seed)
    elapsed = time.time() - start_time

    # Save results
    sampler.save_results(results)

    print(f"[Process {os.getpid()}] Completed {method_name.upper()} in {elapsed:.1f}s")

    return method_name, results, elapsed


def run_single_stochastic_run(args):
    """Run a single run of a stochastic method."""
    sampler, n_samples, run_idx, seed = args

    # Check if this is a k-medoids method
    method_name = getattr(sampler, 'method_name', '')
    is_kmedoids = 'kmedoids' in method_name

    # Get selected indices for this run
    if hasattr(sampler, '_run_single_sample'):
        # Use the method's internal single run method if available
        indices, quality_value = sampler._run_single_sample(n_samples, seed)

        # For k-medoids, quality_value is inertia; for others, it's diversity
        if is_kmedoids:
            inertia = quality_value
            distance = 0
            # Calculate diversity separately
            # Get the distance type from the sampler
            if hasattr(sampler, 'distance_calculator'):
                distance_type = sampler.distance_calculator.name
            else:
                distance_type = 'euclidean'  # default

            if 'lattice' in method_name:
                diversity = sampler.calculate_diversity_score_lattice_generic(indices, distance_type)
            else:
                diversity = sampler.calculate_diversity_score_generic(indices, distance_type)
        else:
            inertia = None
            distance = quality_value
            diversity = quality_value
    else:
        # Fallback: run the full sample method with n_runs=1
        temp_results = sampler.sample(n_samples, n_runs=1, base_seed=seed)
        indices = temp_results['selected_indices']
        distance = temp_results.get('min_jaccard_distance',
                                   temp_results.get('min_euclidean_distance',
                                   temp_results.get('min_manhattan_distance', 0)))
        diversity = temp_results.get('diversity_score')
        inertia = temp_results.get('inertia')

        # Calculate diversity if not provided
        if diversity is None:
            # Get the distance type from the sampler
            if hasattr(sampler, 'distance_calculator'):
                distance_type = sampler.distance_calculator.name
            else:
                distance_type = 'euclidean'  # default

            if 'lattice' in method_name:
                diversity = sampler.calculate_diversity_score_lattice_generic(indices, distance_type)
            else:
                diversity = sampler.calculate_diversity_score_generic(indices, distance_type)

    configurations = [sampler.configurations[i] for i in indices]
    irradiation_sets = [sampler.irradiation_sets[i] for i in indices]

    result = {
        'run_idx': run_idx,
        'indices': indices,
        'distance': distance,
        'diversity': diversity,
        'configurations': configurations,
        'irradiation_sets': irradiation_sets
    }

    # Add k-medoids specific info
    if inertia is not None:
        result['inertia'] = inertia
        result['algorithm'] = 'kmedoids'

    return result

def run_single_task_with_progress(task_args):
    """Run a single task and update progress."""
    # Extract args - now includes use_6x6_restriction and selected_parameters
    method_name, n_samples, run_idx, seed, total_runs, progress_dict, use_6x6_restriction, selected_parameters = task_args

    # Get sampler map with 6x6 restriction if needed
    sampler_map = get_sampler_map(use_6x6_restriction)

    # Create sampler instance
    sampler_class = sampler_map[method_name]
    # Check if this is a geometric method that needs selected parameters
    if selected_parameters and method_name in ['lhs', 'sobol', 'halton', 'jaccard_geometric',
                                              'euclidean_geometric', 'manhattan_geometric', 'random_geometric']:
        sampler = sampler_class(selected_parameters=selected_parameters)
    else:
        sampler = sampler_class()

    # Load data
    sampler.load_data()

    # Update progress to "0%"
    progress_dict[(method_name, run_idx)] = ("", "0%")

    # Check if this is a k-medoids method
    is_kmedoids = 'kmedoids' in method_name

    # Check if this is a sequence-based method
    is_sequence = any(seq in method_name for seq in ['sobol', 'lhs', 'halton'])

    # For sequence methods, show "SEQUENCING" instead of 0%
    if is_sequence:
        progress_dict[(method_name, run_idx)] = ("", "SEQUENCING")

    # Create a custom output capture to intercept progress messages
    import io
    import re
    captured_output = io.StringIO()

    # Run single sample - capture output to extract progress
    old_stdout = sys.stdout

    try:
        # Don't set global random seed - pass seed to sampler
        if hasattr(sampler, '_run_single_sample'):
            # Create a progress callback for k-medoids
            use_progress_callback = False

            def progress_callback(progress_str):
                progress_dict[(method_name, run_idx)] = ("", progress_str)

            # Pass the callback to the sampler if it supports it
            if hasattr(sampler, 'algorithm') and hasattr(sampler.algorithm, 'select_samples'):
                # Store the callback on the sampler so it can pass it to the algorithm
                sampler._progress_callback = progress_callback
                # Check if this is k-medoids which supports progress callback
                if 'kmedoids' in method_name:
                    use_progress_callback = True

            # Only use monitor thread if we don't have a progress callback
            if not use_progress_callback:
                # Start a thread to monitor progress output
                import threading
                stop_monitoring = threading.Event()

                def monitor():
                    last_pos = 0
                    no_progress_count = 0
                    while not stop_monitoring.is_set():
                        captured_output.seek(last_pos)
                        new_output = captured_output.read()
                        if new_output:
                            # Look for progress patterns
                            progress_match = re.search(r'Progress:\s*(\d+(?:\.\d+)?)%', new_output)
                            if progress_match:
                                progress_dict[(method_name, run_idx)] = ("", f"{progress_match.group(1)}%")
                                no_progress_count = 0
                            else:
                                no_progress_count += 1
                            last_pos = captured_output.tell()
                        else:
                            no_progress_count += 1

                        # If no progress for a while, show RUNNING
                        if no_progress_count > 10 and progress_dict.get((method_name, run_idx), ("", ""))[1] == "0%":
                            # Don't override SEQUENCING status
                            current_status = progress_dict.get((method_name, run_idx), ("", ""))[1]
                            if current_status not in ["SEQUENCING", "ERROR"]:
                                progress_dict[(method_name, run_idx)] = ("", "RUNNING")

                        time.sleep(0.1)

                monitor_thread = threading.Thread(target=monitor)
                monitor_thread.daemon = True
                monitor_thread.start()

                # NOW redirect stdout after monitor thread is started
                sys.stdout = captured_output
            else:
                # For methods with progress callback, still redirect stdout to suppress output
                sys.stdout = captured_output

            indices, quality_value = sampler._run_single_sample(n_samples, seed)

            if not use_progress_callback:
                stop_monitoring.set()
                monitor_thread.join(timeout=0.5)

            # For unified samplers with k-medoids, quality_value is inertia
            # For unified samplers with greedy, quality_value is diversity
            if is_kmedoids:
                inertia = quality_value
                algorithm = 'kmedoids'
                # Need to calculate diversity separately for k-medoids
                # Get the distance type from the sampler
                if hasattr(sampler, 'distance_calculator'):
                    distance_type = sampler.distance_calculator.name
                else:
                    distance_type = 'euclidean'  # default

                if 'lattice' in method_name:
                    diversity = sampler.calculate_diversity_score_lattice_generic(indices, distance_type)
                else:
                    diversity = sampler.calculate_diversity_score_generic(indices, distance_type)
            else:
                inertia = None
                algorithm = 'greedy'
                diversity = quality_value

            # Set distance to 0 (not used for unified samplers)
            distance = 0
        else:
            # Pass seed as base_seed to the sampler
            temp_results = sampler.sample(n_samples, n_runs=1, base_seed=seed)
            indices = temp_results['selected_indices']
            distance = temp_results.get('min_jaccard_distance',
                                       temp_results.get('min_euclidean_distance',
                                       temp_results.get('min_manhattan_distance', 0)))
            # Extract k-medoids specific metrics
            inertia = temp_results.get('inertia')
            algorithm = temp_results.get('algorithm')
            diversity = temp_results.get('diversity_score')

            # If diversity not in results, calculate it
            if diversity is None:
                # Get the distance type from the sampler
                if hasattr(sampler, 'distance_calculator'):
                    distance_type = sampler.distance_calculator.name
                else:
                    distance_type = 'euclidean'  # default

                if 'lattice' in method_name:
                    diversity = sampler.calculate_diversity_score_lattice_generic(indices, distance_type)
                else:
                    diversity = sampler.calculate_diversity_score_generic(indices, distance_type)

        configurations = [sampler.configurations[i] for i in indices]
        irradiation_sets = [sampler.irradiation_sets[i] for i in indices]
    except Exception as e:
        # If there's an error, update progress to show error
        progress_dict[(method_name, run_idx)] = ("", "ERROR")
        sys.stdout = old_stdout
        raise
    finally:
        sys.stdout = old_stdout

    # Update progress with appropriate metric
    if is_kmedoids and inertia is not None:
        progress_dict[(method_name, run_idx)] = ("", f"i:{inertia:.1f}")
    else:
        progress_dict[(method_name, run_idx)] = ("", f"{diversity:.4f}")

    result = {
        'method': method_name,
        'run_idx': run_idx,
        'indices': indices,
        'distance': distance,
        'diversity': diversity,
        'configurations': configurations,
        'irradiation_sets': irradiation_sets
    }

    # Add k-medoids specific info
    if inertia is not None:
        result['inertia'] = inertia
    if algorithm is not None:
        result['algorithm'] = algorithm

    return result


def run_single_task(task_args):
    """Fallback version for when Rich is not available."""
    # Extract only the needed args (without progress_dict)
    method_name, n_samples, run_idx, seed, total_runs = task_args[:5]
    use_6x6_restriction = task_args[6] if len(task_args) > 6 else False
    selected_parameters = task_args[7] if len(task_args) > 7 else None

    # Get sampler map with 6x6 restriction if needed
    sampler_map = get_sampler_map(use_6x6_restriction)

    # Create sampler instance
    sampler_class = sampler_map[method_name]
    # Check if this is a geometric method that needs selected parameters
    if selected_parameters and method_name in ['lhs', 'sobol', 'halton', 'jaccard_geometric',
                                              'euclidean_geometric', 'manhattan_geometric', 'random_geometric']:
        sampler = sampler_class(selected_parameters=selected_parameters)
    else:
        sampler = sampler_class()

    # Load data
    sampler.load_data()

    print(f"[PID {os.getpid()}] Running {method_name} run {run_idx+1}/{total_runs}")

    # Check if this is a k-medoids method
    is_kmedoids = 'kmedoids' in method_name

    # Run single sample - suppress the internal print statements
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Don't set global random seed - pass seed to sampler
        if hasattr(sampler, '_run_single_sample'):
            indices, quality_value = sampler._run_single_sample(n_samples, seed)

            # For unified samplers with k-medoids, quality_value is inertia
            # For unified samplers with greedy, quality_value is diversity
            if is_kmedoids:
                inertia = quality_value
                algorithm = 'kmedoids'
                # Need to calculate diversity separately for k-medoids
                # Get the distance type from the sampler
                if hasattr(sampler, 'distance_calculator'):
                    distance_type = sampler.distance_calculator.name
                else:
                    distance_type = 'euclidean'  # default

                if 'lattice' in method_name:
                    diversity = sampler.calculate_diversity_score_lattice_generic(indices, distance_type)
                else:
                    diversity = sampler.calculate_diversity_score_generic(indices, distance_type)
            else:
                inertia = None
                algorithm = 'greedy'
                diversity = quality_value

            # Set distance to 0 (not used for unified samplers)
            distance = 0
        else:
            # Pass seed as base_seed to the sampler
            temp_results = sampler.sample(n_samples, n_runs=1, base_seed=seed)
            indices = temp_results['selected_indices']
            distance = temp_results.get('min_jaccard_distance',
                                       temp_results.get('min_euclidean_distance',
                                       temp_results.get('min_manhattan_distance', 0)))
            # Extract k-medoids specific metrics
            inertia = temp_results.get('inertia')
            algorithm = temp_results.get('algorithm')
            diversity = temp_results.get('diversity_score')

            # If diversity not in results, calculate it
            if diversity is None:
                # Get the distance type from the sampler
                if hasattr(sampler, 'distance_calculator'):
                    distance_type = sampler.distance_calculator.name
                else:
                    distance_type = 'euclidean'  # default

                if 'lattice' in method_name:
                    diversity = sampler.calculate_diversity_score_lattice_generic(indices, distance_type)
                else:
                    diversity = sampler.calculate_diversity_score_generic(indices, distance_type)

        configurations = [sampler.configurations[i] for i in indices]
        irradiation_sets = [sampler.irradiation_sets[i] for i in indices]
    finally:
        sys.stdout = old_stdout

    # Print appropriate metric
    if is_kmedoids and inertia is not None:
        print(f"[PID {os.getpid()}] Completed {method_name} run {run_idx+1}/{total_runs} (inertia: {inertia:.2f}, diversity: {diversity:.4f})")
    else:
        print(f"[PID {os.getpid()}] Completed {method_name} run {run_idx+1}/{total_runs} (diversity: {diversity:.4f})")

    result = {
        'method': method_name,
        'run_idx': run_idx,
        'indices': indices,
        'distance': distance,
        'diversity': diversity,
        'configurations': configurations,
        'irradiation_sets': irradiation_sets
    }

    # Add k-medoids specific info
    if inertia is not None:
        result['inertia'] = inertia
    if algorithm is not None:
        result['algorithm'] = algorithm

    # ADDED: Save cluster assignments if available
    if hasattr(sampler, 'algorithm') and hasattr(sampler.algorithm, 'cluster_assignments'):
        result['cluster_assignments'] = sampler.algorithm.cluster_assignments.tolist()
        result['n_clusters'] = sampler.algorithm.n_clusters

    return result
