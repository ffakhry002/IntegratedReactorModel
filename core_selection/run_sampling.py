"""
Main script to run all sampling methods with optional parallel execution.
"""

import argparse
import time
import os
from multiprocessing import cpu_count
import sys
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

from sampler_execution import (
    SAMPLER_MAP,
    run_serial_execution,
    run_method_parallel_execution,
    run_hybrid_parallel_execution,
    create_comparison_summary,
    create_diversity_analysis
)
from interactive_parameter_selection import get_parameter_selection


def validate_arguments(args):
    """Validate command line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments

    Returns
    -------
    tuple
        (is_valid, selected_methods, n_workers, methods_list) where:
        - is_valid: Boolean indicating if arguments are valid
        - selected_methods: List of validated method names
        - n_workers: Number of worker processes
        - methods_list: Copy of selected methods list
    """
    # Validate parallel options
    if args.parallel and args.hybrid_parallel:
        print("ERROR: Cannot use both --parallel and --hybrid-parallel modes")
        print("Choose one:")
        print("  --parallel: Each core runs a different method")
        print("  --hybrid-parallel: All cores grab from a shared queue of all tasks")
        return False, None, None, None

    # Determine number of workers for parallel modes
    if args.parallel or args.hybrid_parallel:
        n_workers = args.workers if args.workers else cpu_count()
    else:
        n_workers = None

    # Parse methods selection
    if args.methods:
        if args.methods.lower() == "all":
            selected_methods = list(SAMPLER_MAP.keys())
        else:
            selected_methods = [method.strip() for method in args.methods.split(',')]
    else:
        # Default to all methods
        selected_methods = list(SAMPLER_MAP.keys())

    # Validate selected methods
    valid_methods = []
    for method in selected_methods:
        if method in SAMPLER_MAP:
            valid_methods.append(method)
        else:
            print(f"Warning: Unknown method '{method}' - skipping")
    selected_methods = valid_methods

    if not selected_methods:
        print("ERROR: No valid methods selected!")
        return False, None, None, None

    return True, selected_methods, n_workers, selected_methods


def print_header(args, selected_methods, n_workers):
    """Print execution header.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
    selected_methods : list
        List of selected sampling method names
    n_workers : int or None
        Number of worker processes for parallel execution
    """
    if args.parallel:
        print("PARALLEL CORE CONFIGURATION SAMPLING (Method Parallel)")
    elif args.hybrid_parallel:
        print("PARALLEL CORE CONFIGURATION SAMPLING (Hybrid/Work Queue)")
    else:
        print("CORE CONFIGURATION SAMPLING")
    print("="*80)
    print(f"Generating {args.n_samples} samples using {len(selected_methods)} methods")
    print(f"Stochastic methods will run {args.runs} times")
    print(f"Base random seed: {args.seed}")
    if args.parallel or args.hybrid_parallel:
        print(f"Number of workers: {n_workers}")
        if args.hybrid_parallel:
            print("Mode: Hybrid parallel (work queue with all tasks)")
    print(f"Selected methods: {', '.join(selected_methods)}\n")


def check_required_data(restrict_6x6=False):
    """Check if required data files exist.

    Parameters
    ----------
    restrict_6x6 : bool, optional
        Whether to check for 6x6 restricted files (default: False)

    Returns
    -------
    bool
        True if all required files exist, exits if files are missing
    """
    suffix = "_6x6" if restrict_6x6 else ""

    config_file = SCRIPT_DIR / f'output/data/core_configurations_optimized{suffix}.pkl'
    physics_file = SCRIPT_DIR / f'output/data/physics_parameters{suffix}.pkl'

    if not config_file.exists():
        print("ERROR: Core configurations not found!")
        print(f"Looking for: {config_file.relative_to(SCRIPT_DIR)}")
        print(f"Please run: python generate_core_configurations.py{' --restrict-6x6' if restrict_6x6 else ''}")
        sys.exit(1)

    if not physics_file.exists():
        print("ERROR: Physics parameters not found!")
        print(f"Looking for: {physics_file.relative_to(SCRIPT_DIR)}")
        print(f"Please run: python calculate_geometric_parameters.py{' --restrict-6x6' if restrict_6x6 else ''}")
        sys.exit(1)

    return True


def print_final_summary(args, selected_methods, results_dict, total_time, n_workers):
    """Print final execution summary.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
    selected_methods : list
        List of selected sampling method names
    results_dict : dict
        Dictionary containing results from all sampling methods
    total_time : float
        Total execution time in seconds
    n_workers : int or None
        Number of worker processes used
    """
    print("\n" + "="*80)
    if args.parallel:
        print("METHOD-PARALLEL SAMPLING COMPLETE!")
    elif args.hybrid_parallel:
        print("HYBRID-PARALLEL SAMPLING COMPLETE!")
    else:
        print("SAMPLING COMPLETE!")
    print("="*80)

    if any([args.parallel, args.hybrid_parallel]):
        print(f"Total execution time: {total_time:.1f}s")
        if args.parallel:
            print(f"Speedup vs serial: ~{len(selected_methods)/max(1, total_time/60):.1f}x")

    print(f"\nRan {len(selected_methods)} methods")
    if args.parallel:
        print(" in method-parallel mode")
    elif args.hybrid_parallel:
        print(" in hybrid-parallel mode")
    else:
        for method in selected_methods:
            print(f"  - {method}")

    print("\nResults saved in:")
    print("  output/samples_picked/pkl/       - Pickle and JSON files")
    print("  output/samples_picked/txt/       - Text summaries")
    print("  output/samples_picked/results/   - Comparison and analysis")

    # Print quick summary
    print("\nQuick Summary:")
    print("-"*40)

    if args.parallel or args.hybrid_parallel:
        # Print timing summary for parallel modes
        print("\nTiming Summary:")
        for method, (results, elapsed) in sorted(results_dict.items(),
                                                key=lambda x: x[1][1],
                                                reverse=True):
            diversity = results.get('diversity_score', 'N/A')
            if isinstance(diversity, float):
                print(f"  {method}: {elapsed:.1f}s, Diversity = {diversity:.4f}")
            else:
                print(f"  {method}: {elapsed:.1f}s, Diversity = {diversity}")
    else:
        # Show methods by category for serial mode
        _print_serial_summary(results_dict)


def _print_serial_summary(results_dict):
    """Print summary for serial execution.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing results from all sampling methods
    """
    # Show lattice methods if any were run
    lattice_methods = ['lhs_lattice', 'sobol_lattice', 'halton_lattice',
                      'jaccard_lattice', 'euclidean_lattice', 'manhattan_lattice', 'random_lattice',
                      'euclidean_lattice_geometric_diversity']
    selected_lattice = [m for m in lattice_methods if m in results_dict]

    if selected_lattice:
        print("\nLattice-based methods (Configuration Space):")
        for method in selected_lattice:
            diversity = results_dict[method].get('diversity_score', 'N/A')
            if isinstance(diversity, float):
                print(f"  {method}: Diversity = {diversity:.4f}")
            else:
                print(f"  {method}: Diversity = {diversity}")

    # Show geometric methods if any were run
    geometric_methods = ['lhs', 'sobol', 'halton',
                        'jaccard_geometric', 'euclidean_geometric', 'manhattan_geometric', 'random_geometric']
    selected_geometric = [m for m in geometric_methods if m in results_dict]

    if selected_geometric:
        print("\nGeometric/Physics-based methods (Parameter Space):")
        for method in selected_geometric:
            diversity = results_dict[method].get('diversity_score', 'N/A')
            if isinstance(diversity, float):
                print(f"  {method}: Diversity = {diversity:.4f}")
            else:
                print(f"  {method}: Diversity = {diversity}")


def main():
    """Main function to run sampling methods for core configuration selection.

    Parses command line arguments, validates inputs, loads required data,
    and executes the selected sampling methods in serial or parallel mode.
    """
    parser = argparse.ArgumentParser(
        description='Run sampling methods for core configuration selection'
    )
    parser.add_argument('n_samples', type=int, help='Number of samples to generate')
    parser.add_argument('--runs', type=int, default=10,
                       help='Number of runs for stochastic methods (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--methods', type=str, default=None,
                       help='Comma-separated list of methods to run (default: all methods)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run methods in parallel using multiprocessing')
    parser.add_argument('--hybrid-parallel', action='store_true',
                       help='Run all method+run combinations in a single work queue')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: number of CPUs)')
    parser.add_argument('--restrict-6x6', action='store_true',
                       help='Use 6x6 central square restriction')
    parser.add_argument('--interactive-params', action='store_true',
                       help='Interactively select geometric parameters to use')
    parser.add_argument('--param-selection', type=str, default=None,
                       help='Comma-separated list of parameters to use (overrides interactive)')

    args = parser.parse_args()

    # Validate arguments
    valid, selected_methods, n_workers, _ = validate_arguments(args)
    if not valid:
        return

    # Print header
    print_header(args, selected_methods, n_workers)

    # Check required data
    if not check_required_data(args.restrict_6x6):
        return

    # Handle parameter selection for geometric methods
    selected_parameters = None
    geometric_methods = ['lhs', 'sobol', 'halton', 'jaccard_geometric',
                        'euclidean_geometric', 'manhattan_geometric', 'random_geometric']

    has_geometric = any(method in selected_methods for method in geometric_methods)

    if has_geometric:
        if args.param_selection:
            # Use command-line specified parameters
            selected_parameters = [p.strip() for p in args.param_selection.split(',')]
            print(f"\nUsing specified parameters: {', '.join(selected_parameters)}")
        elif args.interactive_params:
            # Interactive selection
            selected_parameters, _ = get_parameter_selection(
                interactive=True,
                restrict_6x6=args.restrict_6x6
            )
        else:
            # Use all available parameters (default)
            print("\nUsing all available geometric parameters")

    # Execute sampling
    results_dict = {}
    start_time = time.time()

    if args.parallel:
        # Method parallel execution
        results_dict = run_method_parallel_execution(
            selected_methods, args.n_samples, args.runs, args.seed, n_workers,
            args.restrict_6x6, selected_parameters
        )
    elif args.hybrid_parallel:
        # Hybrid parallel execution
        results_dict = run_hybrid_parallel_execution(
            selected_methods, args.n_samples, args.runs, args.seed, n_workers,
            args.restrict_6x6, selected_parameters
        )
    else:
        # Serial execution
        results_dict = run_serial_execution(
            selected_methods, args.n_samples, args.runs, args.seed,
            args.restrict_6x6, selected_parameters
        )

    total_time = time.time() - start_time

    # Create comparison summary
    print(f"\n{'='*60}")
    print("Creating comparison reports...")
    print(f"{'='*60}")

    create_comparison_summary(results_dict, args.n_samples, total_time, n_workers,
                            args.parallel or args.hybrid_parallel)
    create_diversity_analysis(results_dict, args.parallel or args.hybrid_parallel)

    # Print final summary
    print_final_summary(args, selected_methods, results_dict, total_time, n_workers)


if __name__ == "__main__":
    main()
