"""
Analysis and reporting functions for sampling results.
"""

import time
import os
from datetime import datetime


def create_comparison_summary(results_dict, n_samples, total_time=None, n_workers=None, parallel=False):
    """Create a comparison summary of all sampling methods."""
    # Create results directory if needed
    os.makedirs('output/samples_picked/results', exist_ok=True)

    summary_path = 'output/samples_picked/results/sampling_comparison.txt'
    if parallel:
        summary_path = 'output/samples_picked/results/sampling_comparison_parallel.txt'

    with open(summary_path, 'w') as f:
        if parallel:
            f.write("PARALLEL SAMPLING METHODS COMPARISON\n")
        else:
            f.write("SAMPLING METHODS COMPARISON\n")
        f.write("="*80 + "\n\n")
        f.write(f"Number of samples: {n_samples}\n")
        if parallel:
            f.write(f"Number of workers: {n_workers}\n")
            f.write(f"Total execution time: {total_time:.1f}s\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("METHOD PERFORMANCE:\n")
        f.write("-"*80 + "\n")
        if parallel:
            f.write(f"{'Method':<25} {'Diversity Score':<20} {'Time (s)':<15} {'Notes':<20}\n")
        else:
            f.write(f"{'Method':<25} {'Diversity Score':<20} {'Min Distance':<20} {'Notes':<20}\n")
        f.write("-"*80 + "\n")

        for method, data in results_dict.items():
            if parallel:
                results, elapsed = data
            else:
                results = data
                elapsed = None

            diversity = results.get('diversity_score', 'N/A')
            if isinstance(diversity, float):
                diversity_str = f"{diversity:.4f}"
            else:
                diversity_str = str(diversity)

            if not parallel:
                # Get the appropriate distance metric
                min_dist = results.get('min_jaccard_distance') or \
                          results.get('min_euclidean_distance') or \
                          results.get('min_manhattan_distance', 'N/A')
                if isinstance(min_dist, float):
                    min_dist_str = f"{min_dist:.4f}"
                else:
                    min_dist_str = str(min_dist)

            notes = ""
            if results.get('deterministic'):
                notes = "Deterministic"
            elif results.get('best_run'):
                notes = f"Best of {results['total_runs']} runs"

            if parallel:
                f.write(f"{method:<25} {diversity_str:<20} {elapsed:<15.1f} {notes:<20}\n")
            else:
                f.write(f"{method:<25} {diversity_str:<20} {min_dist_str:<20} {notes:<20}\n")

        f.write("\n\nDETAILED RESULTS:\n")
        f.write("="*80 + "\n\n")

        for method, data in results_dict.items():
            if parallel:
                results, elapsed = data
            else:
                results = data

            f.write(f"\n{method.upper()}:\n")
            f.write("-"*40 + "\n")

            if 'selected_indices' in results:
                f.write(f"Selected indices: {results['selected_indices'][:10]}")
                if len(results['selected_indices']) > 10:
                    f.write(f"... ({len(results['selected_indices'])-10} more)")
                f.write("\n")

            for key, value in results.items():
                if key not in ['selected_indices', 'configurations', 'irradiation_sets']:
                    f.write(f"{key}: {value}\n")

            if parallel:
                f.write(f"execution_time: {elapsed:.1f}s\n")

    print(f"✓ Saved comparison summary to {summary_path}")


def create_diversity_analysis(results_dict, parallel=False):
    """Create a detailed diversity analysis."""
    # Create results directory if needed
    os.makedirs('output/samples_picked/results', exist_ok=True)

    analysis_path = 'output/samples_picked/results/diversity_analysis.txt'

    with open(analysis_path, 'w') as f:
        f.write("DIVERSITY ANALYSIS\n")
        f.write("="*80 + "\n\n")

        # Extract results from tuples if parallel
        if parallel:
            diversity_scores = [(m, r[0].get('diversity_score', 0)) for m, r in results_dict.items()]
            results_only = {m: r[0] for m, r in results_dict.items()}
        else:
            diversity_scores = [(m, r.get('diversity_score', 0)) for m, r in results_dict.items()]
            results_only = results_dict

        # Rank methods by diversity score
        methods_by_diversity = sorted(diversity_scores, key=lambda x: x[1], reverse=True)

        f.write("METHODS RANKED BY DIVERSITY SCORE:\n")
        f.write("-"*40 + "\n")
        for i, (method, score) in enumerate(methods_by_diversity, 1):
            f.write(f"{i}. {method}: {score:.4f}\n")

        # Separate by distance type
        f.write("\n\nMETHODS BY DISTANCE TYPE:\n")
        f.write("-"*40 + "\n")

        # Jaccard methods
        jaccard_methods = [(m, r) for m, r in results_only.items()
                          if 'min_jaccard_distance' in r]
        if jaccard_methods:
            f.write("\nJaccard Distance Methods:\n")
            for method, results in sorted(jaccard_methods,
                                         key=lambda x: x[1].get('min_jaccard_distance', 0),
                                         reverse=True):
                f.write(f"  {method}: min_dist={results['min_jaccard_distance']:.4f}, "
                       f"diversity={results['diversity_score']:.4f}\n")

        # Euclidean methods
        euclidean_methods = [(m, r) for m, r in results_only.items()
                            if 'min_euclidean_distance' in r]
        if euclidean_methods:
            f.write("\nEuclidean Distance Methods:\n")
            for method, results in sorted(euclidean_methods,
                                         key=lambda x: x[1].get('min_euclidean_distance', 0),
                                         reverse=True):
                f.write(f"  {method}: min_dist={results['min_euclidean_distance']:.4f}, "
                       f"diversity={results['diversity_score']:.4f}\n")

        # Manhattan methods
        manhattan_methods = [(m, r) for m, r in results_only.items()
                            if 'min_manhattan_distance' in r]
        if manhattan_methods:
            f.write("\nManhattan Distance Methods:\n")
            for method, results in sorted(manhattan_methods,
                                         key=lambda x: x[1].get('min_manhattan_distance', 0),
                                         reverse=True):
                f.write(f"  {method}: min_dist={results['min_manhattan_distance']:.4f}, "
                       f"diversity={results['diversity_score']:.4f}\n")

        f.write("\n\nINTERPRETATION:\n")
        f.write("-"*40 + "\n")
        f.write("- Diversity Score: Minimum pairwise Euclidean distance in normalized parameter space\n")
        f.write("- Min Distance: Method-specific minimum distance between selected configs\n")
        f.write("  - Jaccard: Based on position sets (lattice) or discretized parameters (geometric)\n")
        f.write("  - Euclidean: L2 norm in 5D parameter space\n")
        f.write("  - Manhattan: L1 norm in 5D parameter space\n")
        f.write("- LHS and Sobol map continuous samples to discrete configurations\n")
        f.write("- Greedy methods directly optimize for maximum diversity\n")

    print(f"✓ Saved diversity analysis to {analysis_path}")
