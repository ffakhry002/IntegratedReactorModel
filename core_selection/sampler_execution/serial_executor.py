"""
Serial execution of sampling methods.
"""

from .constants import SAMPLER_MAP, get_sampler_map
from .task_runner import save_sampling_results


def run_serial_execution(selected_methods, n_samples, n_runs, base_seed, use_6x6_restriction=False):
    """Run sampling methods in serial mode."""
    results_dict = {}

    # Get sampler map with 6x6 restriction if needed
    sampler_map = get_sampler_map(use_6x6_restriction)

    # Initialize samplers
    samplers = {method: sampler_map[method]() for method in selected_methods}

    # Run each sampling method
    for name, sampler in samplers.items():
        print(f"\n{'='*60}")
        print(f"Running {name.upper().replace('_', ' ')} sampling...")
        print(f"{'='*60}")

        # Load data for this sampler
        sampler.load_data()

        # Check sample size
        n_configs = len(sampler.configurations)
        actual_n_samples = min(n_samples, n_configs)

        if n_samples > n_configs:
            print(f"Warning: Requested {n_samples} samples but only {n_configs} configurations available")

        # Run sampling
        results = sampler.sample(actual_n_samples, n_runs=n_runs, base_seed=base_seed)

        # Save results using standalone function
        save_sampling_results(name, results)
        results_dict[name] = results

        print(f"✓ {name.upper().replace('_', ' ')} sampling complete")

    return results_dict
