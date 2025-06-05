"""
Main script to run complete core configuration sampling workflow.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

def run_command(command, description):
    """Run a command and handle errors with real-time output."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    print()  # Add blank line before output

    try:
        # Run with real-time output instead of capturing
        # Change to script directory before running command
        result = subprocess.run(command, shell=True, check=True, cwd=str(SCRIPT_DIR))
        print("✓ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: Command failed with return code {e.returncode}")
        return False

def check_required_files(suffix):
    """Check if required data files exist."""
    required_files = [
        SCRIPT_DIR / f'output/data/core_configurations_optimized{suffix}.pkl',
        SCRIPT_DIR / f'output/data/physics_parameters{suffix}.pkl'
    ]

    # Check for optional but recommended files
    optional_files = [
        SCRIPT_DIR / f'output/data/all_configurations_before_symmetry{suffix}.pkl',
        SCRIPT_DIR / f'output/data/physics_parameters_full{suffix}.pkl'
    ]

    missing_files = []
    for file in required_files:
        if not file.exists():
            missing_files.append(str(file.relative_to(SCRIPT_DIR)))

    missing_optional = []
    for file in optional_files:
        if not file.exists():
            missing_optional.append(str(file.relative_to(SCRIPT_DIR)))

    return missing_files, missing_optional

def get_method_selection():
    """Get user selection of which methods to run."""
    print("\n" + "="*60)
    print("METHOD SELECTION")
    print("="*60)

    # Define all available methods
    lattice_methods = [
        ("lhs_lattice", "LHS Lattice"),
        ("sobol_lattice", "Sobol Lattice"),
        ("halton_lattice", "Halton Lattice"),
        ("jaccard_lattice", "Jaccard Lattice (Greedy)"),
        ("euclidean_lattice", "Euclidean Lattice (Greedy)"),
        ("manhattan_lattice", "Manhattan Lattice (Greedy)"),
        ("euclidean_lattice_kmedoids", "Euclidean Lattice (K-Means)"),
        ("euclidean_lattice_geometric_diversity", "Euclidean Lattice w/ Geometric Diversity (Greedy)"),
        ("random_lattice", "Random Lattice (Baseline)")
    ]

    geometric_methods = [
        ("lhs", "LHS"),
        ("sobol", "Sobol"),
        ("halton", "Halton"),
        ("jaccard_geometric", "Jaccard Geometric (Greedy)"),
        ("euclidean_geometric", "Euclidean Geometric (Greedy)"),
        ("manhattan_geometric", "Manhattan Geometric (Greedy)"),
        ("euclidean_geometric_kmedoids", "Euclidean Geometric (K-Means)"),
        ("random_geometric", "Random Geometric (Baseline)")
    ]

    print("Choose methods to run:")
    print("1. All methods (17 total)")
    print("2. All lattice methods (9 total)")
    print("3. All geometric methods (8 total)")
    print("4. All greedy methods (7 total)")
    print("5. All k-means methods (2 total)")
    print("6. Custom selection")

    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()

            if choice == "1":
                # All methods
                return [method[0] for method in lattice_methods + geometric_methods]

            elif choice == "2":
                # All lattice methods
                return [method[0] for method in lattice_methods]

            elif choice == "3":
                # All geometric methods
                return [method[0] for method in geometric_methods]

            elif choice == "4":
                # All greedy methods
                greedy_methods = [m[0] for m in lattice_methods + geometric_methods
                                 if 'kmedoids' not in m[0] and 'random' not in m[0]
                                 and any(dist in m[0] for dist in ['euclidean', 'manhattan', 'jaccard'])]
                return greedy_methods

            elif choice == "5":
                # All k-means methods
                kmeans_methods = [m[0] for m in lattice_methods + geometric_methods
                                   if 'kmedoids' in m[0]]
                return kmeans_methods

            elif choice == "6":
                # Custom selection
                selected_methods = []

                print("\nLattice-based methods (Configuration Space):")
                for i, (method_id, method_name) in enumerate(lattice_methods):
                    answer = input(f"  {i+1}. {method_name} (y/n): ").strip().lower()
                    if answer in ['y', 'yes']:
                        selected_methods.append(method_id)

                print("\nGeometric/Physics-based methods (Parameter Space):")
                for i, (method_id, method_name) in enumerate(geometric_methods):
                    answer = input(f"  {i+len(lattice_methods)+1}. {method_name} (y/n): ").strip().lower()
                    if answer in ['y', 'yes']:
                        selected_methods.append(method_id)

                if not selected_methods:
                    print("No methods selected. Please select at least one method.")
                    continue

                return selected_methods

            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")

        except KeyboardInterrupt:
            print("\nAborted by user.")
            sys.exit(0)

def main():
    """Main workflow execution."""

    print("REACTOR CORE CONFIGURATION SAMPLING WORKFLOW")
    print("="*80)
    print("This script will run the complete sampling workflow:")
    print("1. Generate core configurations (if needed)")
    print("2. Calculate geometric parameters (if needed)")
    print("3. Run selected sampling methods")
    print("4. Create comprehensive visualizations")
    print("="*80)

    # Ask about 6x6 restriction
    print("\nConfiguration Space:")
    print("1. Full 8x8 grid (52 valid positions)")
    print("2. Central 6x6 square only")
    restriction_choice = input("Choose configuration space (1-2, default: 1): ").strip()
    use_6x6_restriction = (restriction_choice == "2")

    if use_6x6_restriction:
        print("\n✓ Using 6x6 central square restriction")
        print("  - Sampling will be restricted to central 6x6 square")
        print("  - Physics parameters will consider full 8x8 grid context")
        suffix = "_6x6"
    else:
        print("\n✓ Using full 8x8 grid")
        suffix = ""

    # Check if we need to generate data first
    missing_files, missing_optional = check_required_files(suffix)

    if missing_files:
        print("\nMissing required data files:")
        for file in missing_files:
            print(f"  - {file}")

        print("\nGenerating missing data files...")

        # Generate core configurations if needed
        if f'output/data/core_configurations_optimized{suffix}.pkl' in missing_files:
            cmd = f"{sys.executable} generate_core_configurations.py"
            if use_6x6_restriction:
                cmd += " --restrict-6x6"
            success = run_command(
                cmd,
                f"GENERATING CORE CONFIGURATIONS{' (6x6 Restricted)' if use_6x6_restriction else ''}"
            )
            if not success:
                print("Failed to generate core configurations. Exiting.")
                return

        # Calculate geometric parameters if needed
        if f'output/data/physics_parameters{suffix}.pkl' in missing_files:
            cmd = f"{sys.executable} calculate_geometric_parameters.py"
            if use_6x6_restriction:
                cmd += " --restrict-6x6"
            success = run_command(
                cmd,
                f"CALCULATING GEOMETRIC PARAMETERS{' (6x6 Restricted)' if use_6x6_restriction else ''}"
            )
            if not success:
                print("Failed to calculate geometric parameters. Exiting.")
                return

            # ALWAYS generate full physics parameters too
            print("\nNow generating physics parameters for FULL configuration set...")
            cmd_full = f"{sys.executable} calculate_geometric_parameters.py --full"
            if use_6x6_restriction:
                cmd_full += " --restrict-6x6"
            success = run_command(
                cmd_full,
                f"CALCULATING GEOMETRIC PARAMETERS FOR FULL SET{' (6x6 Restricted)' if use_6x6_restriction else ''}"
            )
            if not success:
                print("Warning: Failed to calculate full physics parameters.")

    # Re-check for optional files after potentially generating them
    _, missing_optional = check_required_files(suffix)

    # Get method selection first
    selected_methods = get_method_selection()

    print(f"\nSelected methods ({len(selected_methods)}):")
    for method in selected_methods:
        print(f"  - {method}")

    # Check if we need full physics parameters for geometric methods
    geometric_methods_selected = any(method in ['lhs', 'sobol', 'halton', 'jaccard_geometric',
                                               'euclidean_geometric', 'manhattan_geometric',
                                               'random_geometric'] for method in selected_methods)

    full_params_path = SCRIPT_DIR / f'output/data/physics_parameters_full{suffix}.pkl'
    all_configs_path = SCRIPT_DIR / f'output/data/all_configurations_before_symmetry{suffix}.pkl'

    if geometric_methods_selected and not full_params_path.exists() and all_configs_path.exists():
        print("\nGenerating physics parameters for full configuration set...")
        print("This enables geometric methods to sample from all 270,725 configurations")
        success = run_command(
            f"{sys.executable} calculate_geometric_parameters.py --full",
            "CALCULATING GEOMETRIC PARAMETERS FOR FULL SET"
        )
        if not success:
            print("Warning: Failed to calculate full physics parameters.")
            print("Geometric methods will use reduced set instead.")

    # Ask about execution mode
    use_parallel = False
    use_hybrid_parallel = False
    n_cores = 1

    if len(selected_methods) > 1:
        print("\nExecution mode:")
        print("1. Sequential (one method at a time, clean output)")
        print("2. Method Parallel (different methods on different cores)")
        print("3. Hybrid Parallel (work queue - all cores always busy, best efficiency)")

        exec_choice = input("Choose execution mode (1-3, default: 1): ").strip()

        if exec_choice == '2':
            use_parallel = True
            try:
                import multiprocessing
                available_cores = multiprocessing.cpu_count()
                print(f"\nYou have {available_cores} CPU cores available")
                print("Each core will handle a different method")

                core_input = input(f"How many cores to use (1-{available_cores}, default: {available_cores}): ").strip()
                if core_input:
                    try:
                        n_cores = int(core_input)
                        n_cores = max(1, min(n_cores, available_cores))
                    except:
                        n_cores = available_cores
                else:
                    n_cores = available_cores
            except:
                print("Could not detect CPU cores, using 1")
                n_cores = 1

        elif exec_choice == '3':
            use_hybrid_parallel = True
            try:
                import multiprocessing
                available_cores = multiprocessing.cpu_count()
                print(f"\nYou have {available_cores} CPU cores available")
                print("All cores will process a shared queue of all method+run combinations")
                print("This ensures maximum CPU utilization with visual progress tracking")

                core_input = input(f"How many cores to use (1-{available_cores}, default: {available_cores}): ").strip()
                if core_input:
                    try:
                        n_cores = int(core_input)
                        n_cores = max(1, min(n_cores, available_cores))
                    except:
                        n_cores = available_cores
                else:
                    n_cores = available_cores
            except:
                print("Could not detect CPU cores, using 1")
                n_cores = 1

    else:
        # Single method selected - offer hybrid parallel for multiple runs
        print("\nExecution mode for single method:")
        print("1. Sequential (all runs on one core)")
        print("2. Hybrid Parallel (parallelize runs across cores with visual progress)")

        exec_choice = input("Choose execution mode (1-2, default: 1): ").strip()

        if exec_choice == '2':
            use_hybrid_parallel = True
            try:
                import multiprocessing
                available_cores = multiprocessing.cpu_count()
                print(f"\nYou have {available_cores} CPU cores available")
                print("Runs will be distributed across cores")

                core_input = input(f"How many cores to use (1-{available_cores}, default: {available_cores}): ").strip()
                if core_input:
                    try:
                        n_cores = int(core_input)
                        n_cores = max(1, min(n_cores, available_cores))
                    except:
                        n_cores = available_cores
                else:
                    n_cores = available_cores
            except:
                print("Could not detect CPU cores, using 1")
                n_cores = 1

    # Get sampling parameters from user
    try:
        n_samples = int(input("\nEnter number of samples to generate (default: 16): ") or "16")
        n_runs = int(input("Enter number of runs for stochastic methods (default: 10): ") or "10")
        seed = int(input("Enter random seed (default: 42): ") or "42")
    except ValueError:
        print("Invalid input. Using default values.")
        n_samples = 20
        n_runs = 10
        seed = 42

    print(f"\nConfiguration:")
    print(f"  Samples: {n_samples}")
    print(f"  Runs: {n_runs}")
    print(f"  Seed: {seed}")
    print(f"  Methods: {len(selected_methods)} selected")
    if use_parallel:
        print(f"  Mode: Method Parallel ({n_cores} cores, each handles different method)")
    elif use_hybrid_parallel:
        total_tasks = sum(1 if m in ['sobol_lattice', 'halton_lattice', 'sobol', 'halton'] else n_runs for m in selected_methods)
        print(f"  Mode: Hybrid Parallel ({n_cores} cores, {total_tasks} total tasks)")
    else:
        print(f"  Mode: Sequential")

    # Run sampling
    print(f"\n{'='*60}")
    print(f"RUNNING {len(selected_methods)} SELECTED SAMPLING METHODS")
    print(f"{'='*60}")

    start_time = time.time()

    # Create methods string for command line
    methods_str = ",".join(selected_methods)

    if use_parallel:
        cmd = f"{sys.executable} run_sampling.py {n_samples} --runs {n_runs} --seed {seed} --methods {methods_str} --parallel --workers {n_cores}"
        if use_6x6_restriction:
            cmd += " --restrict-6x6"
        success = run_command(
            cmd,
            f"METHOD-PARALLEL SAMPLING WITH {n_samples} SAMPLES ON {n_cores} CORES"
        )
    elif use_hybrid_parallel:
        cmd = f"{sys.executable} run_sampling.py {n_samples} --runs {n_runs} --seed {seed} --methods {methods_str} --hybrid-parallel --workers {n_cores}"
        if use_6x6_restriction:
            cmd += " --restrict-6x6"
        success = run_command(
            cmd,
            f"HYBRID-PARALLEL SAMPLING WITH {n_samples} SAMPLES ON {n_cores} CORES"
        )
    else:
        cmd = f"{sys.executable} run_sampling.py {n_samples} --runs {n_runs} --seed {seed} --methods {methods_str}"
        if use_6x6_restriction:
            cmd += " --restrict-6x6"
        success = run_command(
            cmd,
            f"SAMPLING WITH {n_samples} SAMPLES"
        )

    if not success:
        print("Sampling failed. Please check the error messages above.")
        return

    sampling_time = time.time() - start_time
    print(f"\n✓ Sampling completed in {sampling_time:.1f} seconds")

    # Create visualizations
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    print("Generating organized visualizations:")
    print("  - Lattice method plots → visualizations/lattice/")
    print("  - Geometric method plots → visualizations/geometric/")
    print("  - Combined analysis → visualizations/")

    viz_start_time = time.time()

    success = run_command(
        f"{sys.executable} visualize_all_samples.py",
        "CREATING COMPREHENSIVE VISUALIZATIONS"
    )

    if not success:
        print("Visualization failed. Please check the error messages above.")
        return

    viz_time = time.time() - viz_start_time
    total_time = time.time() - start_time

    print(f"\n✓ Visualizations completed in {viz_time:.1f} seconds")

    # Summary
    print(f"\n{'='*80}")
    print("WORKFLOW COMPLETE!")
    print(f"{'='*80}")
    print(f"Total execution time: {total_time:.1f} seconds")
    print(f"Generated {n_samples} samples using {len(selected_methods)} methods")


    # Quick results summary
    print(f"\n{'='*60}")
    print("QUICK RESULTS SUMMARY")
    print(f"{'='*60}")

    try:
        # Try to read and display summary statistics
        summary_file = SCRIPT_DIR / "visualizations/summary_statistics.txt"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                print(f.read())
        else:
            print("Summary statistics not found.")
    except Exception as e:
        print(f"Could not read summary: {e}")

    print("All done! Check the output directories for results.")

    print("\nSample results will be saved in:")
    print(f"{SCRIPT_DIR}/output/samples_picked/")

if __name__ == "__main__":
    main()
