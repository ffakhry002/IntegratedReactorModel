"""
Parallel execution strategies for sampling methods.
"""

import time
from multiprocessing import Pool, Manager, Process
from collections import defaultdict
import queue
import threading
import sys

from .constants import SAMPLER_MAP, RICH_AVAILABLE, get_sampler_map
from .task_runner import (
    run_single_method,
    run_single_stochastic_run,
    run_single_task_with_progress,
    run_single_task,
    save_sampling_results
)
from .progress_display import create_progress_table, create_method_progress_table, create_method_parallel_table

if RICH_AVAILABLE:
    from rich.console import Console
    from rich.live import Live


def method_worker_with_progress(args):
    """Worker function that runs a method with progress tracking."""
    method_name, n_samples, n_runs, base_seed, use_6x6_restriction, selected_parameters, method_status = args

    # Update to running
    method_status[method_name] = {
        'state': 'RUNNING',
        'runs': {}
    }

    try:
        # Get sampler map with 6x6 restriction if needed
        sampler_map = get_sampler_map(use_6x6_restriction)

        # Create sampler
        sampler_class = sampler_map[method_name]
        # Check if this is a geometric method that needs selected parameters
        if selected_parameters and method_name in ['lhs', 'sobol', 'halton', 'jaccard_geometric',
                                                  'euclidean_geometric', 'manhattan_geometric', 'random_geometric']:
            sampler = sampler_class(selected_parameters=selected_parameters)
        else:
            sampler = sampler_class()
        sampler.load_data()

        # Run the method with progress capture
        # We'll capture stdout and parse progress
        import io
        import re
        import time as time_module

        start_time = time.time()

        # For methods that run multiple times, we need to track each run
        all_results = []

        for run_idx in range(n_runs):
            # Update current run as running
            runs_info = method_status[method_name]['runs']
            runs_info[run_idx] = {'status': '0%'}
            method_status[method_name] = {
                'state': 'RUNNING',
                'runs': runs_info
            }

            # Capture output
            captured = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured

            try:
                # Run single iteration
                seed = base_seed + run_idx if base_seed else None

                # Monitor output in background
                import threading
                stop_monitor = threading.Event()

                def monitor():
                    last_pos = 0
                    while not stop_monitor.is_set():
                        captured.seek(last_pos)
                        new_output = captured.read()
                        if new_output:
                            # Look for progress
                            match = re.search(r'Progress:\s*(\d+)%', new_output)
                            if match:
                                runs_info[run_idx] = {'status': f"{match.group(1)}%"}
                                method_status[method_name] = {
                                    'state': 'RUNNING',
                                    'runs': runs_info
                                }
                            last_pos = captured.tell()
                        time_module.sleep(0.1)

                monitor_thread = threading.Thread(target=monitor)
                monitor_thread.daemon = True
                monitor_thread.start()

                # Run the sampler
                if hasattr(sampler, '_run_single_sample'):
                    indices, quality = sampler._run_single_sample(n_samples, seed)
                    # Calculate diversity if needed
                    if 'kmedoids' in method_name:
                        diversity = sampler.calculate_diversity_score(indices)
                        runs_info[run_idx] = {'inertia': quality, 'diversity': diversity}
                    else:
                        runs_info[run_idx] = {'diversity': quality}
                else:
                    temp_result = sampler.sample(n_samples, n_runs=1, base_seed=seed)
                    diversity = temp_result.get('diversity_score', 0)
                    if 'inertia' in temp_result:
                        runs_info[run_idx] = {'inertia': temp_result['inertia'], 'diversity': diversity}
                    else:
                        runs_info[run_idx] = {'diversity': diversity}

                stop_monitor.set()
                monitor_thread.join(timeout=0.5)

            except Exception as e:
                runs_info[run_idx] = {'status': 'ERROR'}
                raise
            finally:
                sys.stdout = old_stdout

        # Mark as completed
        method_status[method_name] = {
            'state': 'COMPLETED',
            'runs': runs_info
        }

        # Run the full method to get proper results
        results = sampler.sample(n_samples, n_runs=n_runs, base_seed=base_seed)
        elapsed = time.time() - start_time

        return method_name, results, elapsed

    except Exception as e:
        # Mark as error
        method_status[method_name] = {
            'state': 'ERROR',
            'runs': {}
        }
        print(f"Error in {method_name}: {str(e)}")
        raise


def run_method_parallel_execution(selected_methods, n_samples, n_runs, base_seed, n_workers, use_6x6_restriction=False, selected_parameters=None):
    """Run methods in parallel with Rich table progress display."""
    # Create method args in sorted order for consistency
    method_args = [(method, n_samples, n_runs, base_seed, use_6x6_restriction, selected_parameters)
                   for method in sorted(selected_methods)]

    print(f"Starting method-parallel execution with {n_workers} workers...")
    print("(Each worker handles a different method)")

    if RICH_AVAILABLE:
        print("Visual progress table will appear below:")
    print("="*80)

    if RICH_AVAILABLE:
        manager = Manager()
        method_status = manager.dict()

        # Initialize all methods as waiting
        for method in selected_methods:
            method_status[method] = {
                'state': 'WAITING',
                'runs': {}
            }

        console = Console()

        # Use process pool with status updates
        with Pool(n_workers) as pool:
            # Start async processing
            async_results = []
            for args in method_args:
                # Add method_status to args
                args_with_status = args + (method_status,)
                async_result = pool.apply_async(method_worker_with_progress, (args_with_status,))
                async_results.append(async_result)

            # Show live progress
            with Live(create_method_parallel_table(selected_methods, n_runs, method_status),
                     console=console, refresh_per_second=2) as live:
                # Wait for all results
                while not all(r.ready() for r in async_results):
                    live.update(create_method_parallel_table(selected_methods, n_runs, method_status))
                    time.sleep(0.5)

                # Final update
                live.update(create_method_parallel_table(selected_methods, n_runs, method_status))

            # Collect results
            results_list = []
            for async_result in async_results:
                try:
                    results_list.append(async_result.get())
                except Exception as e:
                    print(f"Error collecting result: {e}")
    else:
        # Fallback without Rich
        with Pool(n_workers) as pool:
            results_list = pool.map(run_single_method, method_args)

    # Sort results by method name for consistent ordering
    results_list.sort(key=lambda x: x[0])

    # Convert results to dictionary
    results_dict = {method: (results, elapsed)
                   for method, results, elapsed in results_list}

    return results_dict


def run_hybrid_parallel_execution(selected_methods, n_samples, n_runs, base_seed, n_workers, use_6x6_restriction=False, selected_parameters=None):
    """Run all method+run combinations in a single work queue (hybrid parallel mode)."""
    print(f"Starting hybrid-parallel execution with {n_workers} workers...")
    print("(All workers grab tasks from shared queue)")

    # Get sampler map with 6x6 restriction if needed
    sampler_map = get_sampler_map(use_6x6_restriction)

    # Create shared progress dictionary if Rich is available
    if RICH_AVAILABLE:
        manager = Manager()
        progress_dict = manager.dict()
    else:
        progress_dict = None

    # All methods run multiple times (including Sobol/Halton with different seeds)
    n_runs_per_method = {method: n_runs for method in selected_methods}

    # Create all tasks in deterministic order
    all_tasks = []
    # Sort methods to ensure consistent ordering
    for method in sorted(selected_methods):
        method_runs = n_runs_per_method[method]
        for run_idx in range(method_runs):
            if progress_dict is not None:
                task = (method, n_samples, run_idx, base_seed + run_idx, method_runs, progress_dict, use_6x6_restriction, selected_parameters)
            else:
                task = (method, n_samples, run_idx, base_seed + run_idx, method_runs, {}, use_6x6_restriction, selected_parameters)
            all_tasks.append(task)

    print(f"Total tasks in queue: {len(all_tasks)}")
    if RICH_AVAILABLE:
        print("Visual progress table will appear below:")
    print("="*80)

    # Process all tasks with visual progress
    if RICH_AVAILABLE and progress_dict is not None:
        console = Console()

        # Use context manager for proper pool handling with Live display
        with Pool(n_workers) as pool:
            # Start async processing
            async_result = pool.map_async(run_single_task_with_progress, all_tasks)

            # Show live progress with sorted methods
            sorted_methods = sorted(selected_methods)
            with Live(create_progress_table(sorted_methods, n_runs_per_method, progress_dict),
                     console=console, refresh_per_second=4) as live:
                while not async_result.ready():
                    live.update(create_progress_table(sorted_methods, n_runs_per_method, progress_dict))
                    time.sleep(0.25)

                # Final update
                live.update(create_progress_table(sorted_methods, n_runs_per_method, progress_dict))

            # Get results
            task_results = async_result.get()
    else:
        # Fallback without Rich
        with Pool(n_workers) as pool:
            task_results = pool.map(run_single_task, all_tasks)

    # Remove None results
    task_results = [r for r in task_results if r is not None]

    # Sort results by method and run_idx for deterministic processing
    task_results.sort(key=lambda x: (x['method'], x['run_idx']))

    # Group results by method
    method_results = defaultdict(list)
    for result in task_results:
        method_results[result['method']].append(result)

    # Process results for each method
    print("\n" + "="*80)
    print("RESULTS SUMMARY:")
    print("="*80)

    results_dict = {}
    # Process methods in sorted order for consistency
    for method in sorted(selected_methods):
        method_start = time.time()
        runs = method_results[method]

        if not runs:
            continue

        # Sort runs by run_idx to ensure consistent ordering
        runs.sort(key=lambda x: x['run_idx'])

        # Check if this is k-medoids
        is_kmedoids = 'kmedoids' in method

        if is_kmedoids and any('inertia' in r for r in runs):
            # For k-medoids: find run with LOWEST inertia
            best_run = min(runs, key=lambda x: (x.get('inertia', float('inf')), x['run_idx']))
            selection_metric = 'inertia'
        else:
            # For other methods: find run with HIGHEST diversity
            best_run = max(runs, key=lambda x: (x['diversity'], -x['run_idx']))
            selection_metric = 'diversity'

        final_results = {
            'method': method,
            'n_samples': n_samples,
            'selected_indices': best_run['indices'],
            'diversity_score': best_run['diversity'],
            'best_run': best_run['run_idx'] + 1,
            'total_runs': len(runs),
            'all_diversity_scores': [r['diversity'] for r in runs],
            'configurations': best_run['configurations'],
            'irradiation_sets': best_run['irradiation_sets'],
            'selection_metric': selection_metric
        }

        # Add algorithm-specific data
        if 'algorithm' in best_run:
            final_results['algorithm'] = best_run['algorithm']
        if 'inertia' in best_run:
            final_results['inertia'] = best_run['inertia']
            final_results['all_inertias'] = [r.get('inertia', float('inf')) for r in runs]

        # Add distance metrics
        if 'jaccard' in method:
            final_results['min_jaccard_distance'] = best_run['distance']
            final_results['all_jaccard_distances'] = [r['distance'] for r in runs]
        elif 'euclidean' in method:
            final_results['min_euclidean_distance'] = best_run['distance']
            final_results['all_euclidean_distances'] = [r['distance'] for r in runs]
        elif 'manhattan' in method:
            final_results['min_manhattan_distance'] = best_run['distance']
            final_results['all_manhattan_distances'] = [r['distance'] for r in runs]

        # Save cluster assignments from best run
        if 'cluster_assignments' in best_run:
            final_results['cluster_assignments'] = best_run['cluster_assignments']
            final_results['n_clusters'] = best_run.get('n_clusters', len(set(best_run['cluster_assignments'])))

        # Save results using standalone function (no data loading needed)
        save_sampling_results(method, final_results)

        elapsed = time.time() - method_start
        results_dict[method] = (final_results, elapsed)

        # Print appropriate summary based on method type
        if is_kmedoids and 'inertia' in best_run:
            print(f"\n{method}: Best inertia = {best_run['inertia']:.2f}, Diversity = {final_results['diversity_score']:.4f} (run {final_results['best_run']}/{final_results['total_runs']})")
        else:
            print(f"\n{method}: Best diversity = {final_results['diversity_score']:.4f} (run {final_results['best_run']}/{final_results['total_runs']})")

    return results_dict


def run_method_task_parallel(method_name, n_samples, n_runs, base_seed, n_workers):
    """Run a single method using all workers for parallel runs (task parallelism)."""
    print(f"\n{'='*60}")
    print(f"Running {method_name.upper()} with {n_workers} workers...")
    print(f"{'='*60}")

    # Create sampler instance
    sampler_class = SAMPLER_MAP[method_name]
    sampler = sampler_class()

    # Load data
    sampler.load_data()

    # Check sample size
    n_configs = len(sampler.configurations)
    n_samples = min(n_samples, n_configs)

    start_time = time.time()

    # Run all methods with multiple runs (including Sobol/Halton with different seeds)
    print(f"Running {n_runs} runs in parallel across {n_workers} workers...")

    # Prepare arguments for parallel runs
    run_args = [(sampler, n_samples, i, base_seed + i) for i in range(n_runs)]

    # Run in parallel
    with Pool(n_workers) as pool:
        run_results = pool.map(run_single_stochastic_run, run_args)

    # Check if k-medoids
    is_kmedoids = 'kmedoids' in method_name

    if is_kmedoids and any('inertia' in r for r in run_results):
        # Find best run by lowest inertia
        best_run = min(run_results, key=lambda x: x.get('inertia', float('inf')))
        selection_metric = 'inertia'
    else:
        # Find best run by highest diversity
        best_run = max(run_results, key=lambda x: x['diversity'])
        selection_metric = 'diversity'

    # Format results like the original sample method
    results = {
        'method': method_name,
        'n_samples': n_samples,
        'selected_indices': best_run['indices'],
        'diversity_score': best_run['diversity'],
        'best_run': best_run['run_idx'] + 1,
        'total_runs': n_runs,
        'all_diversity_scores': [r['diversity'] for r in run_results],
        'configurations': best_run['configurations'],
        'irradiation_sets': best_run['irradiation_sets'],
        'selection_metric': selection_metric
    }

    # Add k-medoids specific data
    if 'algorithm' in best_run:
        results['algorithm'] = best_run['algorithm']
    if 'inertia' in best_run:
        results['inertia'] = best_run['inertia']
        results['all_inertias'] = [r.get('inertia', float('inf')) for r in run_results]

    # Add appropriate distance metric
    if 'jaccard' in method_name:
        results['min_jaccard_distance'] = best_run['distance']
        results['all_jaccard_distances'] = [r['distance'] for r in run_results]
    elif 'euclidean' in method_name:
        results['min_euclidean_distance'] = best_run['distance']
        results['all_euclidean_distances'] = [r['distance'] for r in run_results]
    elif 'manhattan' in method_name:
        results['min_manhattan_distance'] = best_run['distance']
        results['all_manhattan_distances'] = [r['distance'] for r in run_results]

    # Print summary
    print(f"\nRun summary:")
    for i, r in enumerate(run_results):
        if is_kmedoids and 'inertia' in r:
            print(f"  Run {i+1}: inertia = {r['inertia']:.2f}, diversity = {r['diversity']:.4f}")
        else:
            print(f"  Run {i+1}: diversity = {r['diversity']:.4f}")

    if is_kmedoids and 'inertia' in best_run:
        print(f"Best run: #{best_run['run_idx']+1} with inertia = {best_run['inertia']:.2f} (diversity = {best_run['diversity']:.4f})")
    else:
        print(f"Best run: #{best_run['run_idx']+1} with diversity = {best_run['diversity']:.4f}")

    elapsed = time.time() - start_time

    # Save results using standalone function
    save_sampling_results(method_name, results)

    print(f"✓ Completed {method_name} in {elapsed:.1f}s")

    return method_name, results, elapsed
