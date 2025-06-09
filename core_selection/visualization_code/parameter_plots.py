"""
Parameter plotting functions for sampling method analysis.
Handles physics parameters comparisons and diversity plots.
UPDATED: Bottom-right panel now shows both diversity and inertia with dual y-axes
UPDATED: Support for 6x6 mode detection in titles
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from .data_loader import detect_6x6_mode


def plot_physics_parameters_comparison(methods, physics_params, results_dict, output_dir):
    """Plot comparison of physics parameters across sampling methods."""
    # Detect 6x6 mode
    is_6x6 = detect_6x6_mode()
    mode_str = " (6x6 Central Square)" if is_6x6 else ""

    # Dynamically determine available parameters
    # Handle both list and dict formats for physics_params
    if isinstance(physics_params, dict):
        sample_params = next(iter(physics_params.values()))
    else:
        # If it's a list, get the first element
        sample_params = physics_params[0] if physics_params else {}

    param_names = []
    param_labels = []

    # Define all possible parameters and their labels
    all_params = {
        'avg_distance_from_core_center': 'Avg Distance from Core Center',
        'min_inter_position_distance': 'Min Inter-Position Distance',
        'clustering_coefficient': 'Clustering Coefficient',
        'symmetry_balance': 'Symmetry Balance',
        'local_fuel_density': 'Local Fuel Density',
        'avg_distance_to_edge': 'Avg Distance to Edge'
    }

    # Only include parameters that exist in the data
    for param_key, param_label in all_params.items():
        if param_key in sample_params:
            param_names.append(param_key)
            param_labels.append(param_label)

    # Adjust subplot layout based on number of parameters
    n_params = len(param_names)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    # Colorful scheme
    method_colors = {
        'lhs': '#FD79A8',
        'sobol': '#6C5CE7',
        'halton': '#A29BFE',
        'jaccard_geometric': '#74B9FF',
        'euclidean_geometric': '#00B894',
        'manhattan_geometric': '#FDCB6E',
        'random_geometric': '#FF8C69',
        'lhs_lattice': '#FF6B6B',
        'sobol_lattice': '#4ECDC4',
        'halton_lattice': '#45B7D1',
        'jaccard_lattice': '#96CEB4',
        'euclidean_lattice': '#FFEAA7',
        'manhattan_lattice': '#DDA0DD',
        'random_lattice': '#FFB6C1'
    }

    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[idx]

        for method_idx, method in enumerate(methods):
            if method in results_dict:
                indices = results_dict[method]['selected_indices']
                # Extract parameter values using proper indexing
                values = [physics_params[int(idx)][param] for idx in indices]

                # Plot as scatter WITH jitter for better visibility
                x_pos = method_idx + 1
                # Add small random jitter to x-coordinates
                jittered_x = x_pos + np.random.normal(0, 0.02, len(values))
                ax.scatter(jittered_x, values,
                          alpha=0.7, s=40,
                          color=method_colors.get(method, '#2C3E50'),
                          label=method if idx == 0 else "",
                          edgecolors='white', linewidth=0.5)

        ax.set_title(label, fontweight='bold')
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

    # Add legend to first subplot
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Remove empty subplots
    for idx in range(n_params, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f'Physics Parameters Distribution by Sampling Method{mode_str}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Create appropriate filename based on output directory
    if 'lattice' in output_dir:
        filename = f'{output_dir}/lattice_parameters_comparison.png'
    elif 'geometric' in output_dir:
        filename = f'{output_dir}/geometric_parameters_comparison.png'
    else:
        filename = f'{output_dir}/parameters_comparison.png'

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    return filename


def plot_diversity_comparison(methods, results_dict, output_dir):
    """Plot diversity scores and distance comparisons."""
    # Detect 6x6 mode
    is_6x6 = detect_6x6_mode()
    mode_str = " (6x6 Central Square)" if is_6x6 else ""

    fig = plt.figure(figsize=(16, 8))

    # Create three subplots
    ax1 = plt.subplot(131)  # Diversity scores
    ax2 = plt.subplot(132)  # Method-specific distances
    ax3 = plt.subplot(133)  # Distance type comparison

    # 1. Diversity scores bar plot
    diversity_scores = []
    method_labels = []
    colors = []

    method_colors = {
        'lhs': 'blue',
        'lhs_lattice': 'darkblue',
        'sobol': 'green',
        'sobol_lattice': 'darkgreen',
        'halton': 'cyan',
        'jaccard_geometric': 'red',
        'jaccard_lattice': 'purple',
        'euclidean_geometric': 'orange',
        'manhattan_geometric': 'brown'
    }

    for method in methods:
        if method in results_dict and 'diversity_score' in results_dict[method]:
            diversity_scores.append(results_dict[method]['diversity_score'])
            method_labels.append(method.replace('_', '\n'))
            colors.append(method_colors.get(method, 'gray'))

    bars = ax1.bar(range(len(diversity_scores)), diversity_scores, color=colors)
    ax1.set_xticks(range(len(diversity_scores)))
    ax1.set_xticklabels(method_labels, rotation=45, ha='right')
    ax1.set_ylabel('Diversity Score')
    ax1.set_title('Diversity Scores by Method')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, score in zip(bars, diversity_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom')

    # 2. Method-specific distances
    distance_data = []
    distance_labels = []
    distance_colors = []

    for method in methods:
        if method in results_dict:
            if 'min_jaccard_distance' in results_dict[method]:
                distance_data.append(results_dict[method]['min_jaccard_distance'])
                distance_labels.append(f"{method.replace('_', ' ')}\n(Jaccard)")
                distance_colors.append(method_colors.get(method, 'gray'))
            elif 'min_euclidean_distance' in results_dict[method]:
                distance_data.append(results_dict[method]['min_euclidean_distance'])
                distance_labels.append(f"{method.replace('_', ' ')}\n(Euclidean)")
                distance_colors.append(method_colors.get(method, 'gray'))
            elif 'min_manhattan_distance' in results_dict[method]:
                distance_data.append(results_dict[method]['min_manhattan_distance'])
                distance_labels.append(f"{method.replace('_', ' ')}\n(Manhattan)")
                distance_colors.append(method_colors.get(method, 'gray'))
            elif 'avg_lattice_distance' in results_dict[method]:
                distance_data.append(results_dict[method]['avg_lattice_distance'])
                distance_labels.append(f"{method.replace('_', ' ')}\n(Lattice)")
                distance_colors.append(method_colors.get(method, 'gray'))

    if distance_data:
        bars2 = ax2.bar(range(len(distance_data)), distance_data, color=distance_colors)
        ax2.set_xticks(range(len(distance_data)))
        ax2.set_xticklabels(distance_labels, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Min Distance')
        ax2.set_title('Method-Specific Min Distances')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, dist in zip(bars2, distance_data):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{dist:.3f}', ha='center', va='bottom')

    # 3. Group by approach type
    approach_types = {
        'Physics Space': [],
        'Lattice Space': [],
        'Distance-Based': []
    }

    for method in methods:
        if method in results_dict:
            diversity = results_dict[method]['diversity_score']
            if method in ['lhs', 'sobol', 'halton']:
                approach_types['Physics Space'].append((method, diversity))
            elif method in ['lhs_lattice', 'sobol_lattice']:
                approach_types['Lattice Space'].append((method, diversity))
            elif 'jaccard' in method or 'euclidean' in method or 'manhattan' in method:
                approach_types['Distance-Based'].append((method, diversity))

    # Plot grouped bars
    n_groups = len([k for k, v in approach_types.items() if v])
    if n_groups > 0:
        group_width = 0.8
        bar_width = group_width / max([len(v) for v in approach_types.values() if v], default=1)

        x = np.arange(n_groups)
        offset = 0

        for approach, methods_data in approach_types.items():
            if methods_data:
                for i, (method, diversity) in enumerate(methods_data):
                    ax3.bar(offset + i * bar_width, diversity, bar_width,
                           label=method, color=method_colors.get(method, 'gray'))
                offset += 1

        ax3.set_xlabel('Approach Type')
        ax3.set_ylabel('Diversity Score')
        ax3.set_title('Performance by Approach Type')
        ax3.set_xticks(np.arange(n_groups) + group_width/2)
        ax3.set_xticklabels([k for k, v in approach_types.items() if v])
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Sampling Method Performance Comparison{mode_str}', fontsize=16)
    plt.tight_layout()

    filename = f'{output_dir}/diversity_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    return filename


def create_diversity_comparison_by_type(samples_data: Dict, lattice_methods: List[str],
                                      geometric_methods: List[str], method_colors: Dict):
    """Create diversity comparison plots separated by method type."""
    # Detect 6x6 mode
    is_6x6 = detect_6x6_mode()
    mode_str = " (6x6 Central Square)" if is_6x6 else ""

    # Lattice methods comparison
    lattice_data = {method: samples_data[method] for method in lattice_methods if method in samples_data}
    if lattice_data:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        methods = list(lattice_data.keys())
        diversities = [lattice_data[m]['diversity_score'] for m in methods]
        colors = [method_colors.get(m, '#2C3E50') for m in methods]

        bars = ax.bar(range(len(methods)), diversities, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xlabel('Sampling Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Diversity Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Diversity Comparison - Lattice-based Methods{mode_str}\n(Configuration Space Sampling)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, diversity in zip(bars, diversities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{diversity:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('visualizations/lattice/diversity_comparison_lattice.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Geometric methods comparison
    geometric_data = {method: samples_data[method] for method in geometric_methods if method in samples_data}
    if geometric_data:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        methods = list(geometric_data.keys())
        diversities = [geometric_data[m]['diversity_score'] for m in methods]
        colors = [method_colors.get(m, '#2C3E50') for m in methods]

        bars = ax.bar(range(len(methods)), diversities, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xlabel('Sampling Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Diversity Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Diversity Comparison - Geometric/Physics-based Methods{mode_str}\n(Parameter Space Sampling)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, diversity in zip(bars, diversities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{diversity:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('visualizations/geometric/diversity_comparison_geometric.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_parameter_scatter(physics_params, selected_indices, method_name, output_file, color, sample_data=None):
    """Create parameter space scatter plots."""
    # Detect 6x6 mode
    is_6x6 = detect_6x6_mode()
    mode_str = " (6x6 Central Square)" if is_6x6 else ""

    selected_params = [physics_params[i] for i in selected_indices]

    # Dynamically determine available parameters
    sample_params = selected_params[0] if selected_params else {}

    param_names = []
    param_labels = []

    # Define all possible parameters and their labels
    all_params = {
        'avg_distance_from_core_center': 'Avg Distance from Core Center',
        'min_inter_position_distance': 'Min Inter-Position Distance',
        'clustering_coefficient': 'Clustering Coefficient',
        'symmetry_balance': 'Symmetry Balance',
        'local_fuel_density': 'Local Fuel Density',
        'avg_distance_to_edge': 'Avg Distance to Edge'
    }

    # Only include parameters that exist in the data
    for param_key, param_label in all_params.items():
        if param_key in sample_params:
            param_names.append(param_key)
            param_labels.append(param_label)

    # Check if this is a k-means/k-medoids method that has cluster assignments
    is_kmeans_with_clusters = (sample_data and
                              (sample_data.get('algorithm') in ['kmedoids', 'kmeans_nearest'] or
                               'kmedoids' in method_name) and
                              'cluster_assignments' in sample_data)

    n_params = len(param_names)

    # Create appropriate subplot layout
    if is_kmeans_with_clusters:
        # For k-means: create panels for all parameters + 2 extra panels
        total_panels = n_params + 2  # params + diversity/inertia + cluster distribution
        n_cols = 3
        n_rows = (total_panels + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(20, 6 * n_rows), constrained_layout=True)
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)

        # Create axes
        axes = []
        for row in range(n_rows):
            for col in range(n_cols):
                if row * n_cols + col < total_panels:
                    axes.append(fig.add_subplot(gs[row, col]))
    else:
        # Non-k-means layout: params + 1 extra panel for diversity
        total_panels = n_params + 1
        n_cols = 3
        n_rows = (total_panels + n_cols - 1) // n_cols

        fig, axes_arr = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        axes = axes_arr.flatten() if n_rows > 1 else [axes_arr] if n_cols == 1 else axes_arr

    fig.suptitle(f'{method_name} - Parameter Space Distribution{mode_str}', fontsize=16, fontweight='bold')

    # Plot parameter distributions (first n_params panels)
    for i, (param_name, param_label) in enumerate(zip(param_names, param_labels)):
        ax = axes[i]
        param_values = [params[param_name] for params in selected_params]

        ax.hist(param_values, bins=min(10, len(param_values)),
                color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel(param_label, fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{param_label} Distribution')
        ax.grid(True, alpha=0.3)

    # Panel n_params+1: Diversity/Inertia scores by run plot
    ax = axes[n_params]

    # Check if this is a k-means/k-medoids method
    is_kmeans = sample_data and (sample_data.get('algorithm') in ['kmedoids', 'kmeans_nearest'] or
                                'kmedoids' in method_name)

    # Get diversity scores
    diversity_scores = sample_data.get('all_diversities', sample_data.get('all_diversity_scores', []))

    if diversity_scores:
        run_numbers = list(range(1, len(diversity_scores) + 1))

        # Check for both possible field names for backward compatibility
        if is_kmeans and ('all_inertias' in sample_data or 'all_inertia_scores' in sample_data):
            # K-means/K-medoids: Plot both inertia and diversity with dual y-axes
            inertia_scores = sample_data.get('all_inertias', sample_data.get('all_inertia_scores', []))

            # Plot diversity on primary y-axis (left)
            color_div = 'green'
            ax.plot(run_numbers, diversity_scores, 'o-', color=color_div,
                    markersize=8, linewidth=2, alpha=0.8)  # Removed label
            ax.set_xlabel('Run Number', fontweight='bold')
            ax.set_ylabel('Diversity Score', color=color_div, fontweight='bold')
            ax.tick_params(axis='y', labelcolor=color_div)

            # Create second y-axis for inertia
            ax2 = ax.twinx()
            color_inertia = 'blue'
            ax2.plot(run_numbers, inertia_scores, 's-', color=color_inertia,
                    markersize=8, linewidth=2, alpha=0.8)  # Removed label
            ax2.set_ylabel('Inertia', color=color_inertia, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor=color_inertia)

            # Highlight the best run (lowest inertia)
            if 'best_run' in sample_data:
                best_run_idx = sample_data['best_run'] - 1
                if 0 <= best_run_idx < len(inertia_scores):
                    # Mark best run on both axes
                    ax.plot(sample_data['best_run'], diversity_scores[best_run_idx],
                        'o', color='red', markersize=14,
                        markeredgecolor='black', markeredgewidth=2, zorder=5)
                    ax2.plot(sample_data['best_run'], inertia_scores[best_run_idx],
                            's', color='red', markersize=14,
                            markeredgecolor='black', markeredgewidth=2, zorder=5)

            # Title
            algorithm_name = 'K-Medoids' if sample_data.get('algorithm') == 'kmedoids' else 'K-Means'
            ax.set_title(f'{algorithm_name}: Diversity & Inertia Across Runs')

        else:
            # Non-k-means/k-medoids: Plot only diversity
            ax.plot(run_numbers, diversity_scores, 'o-', color=color,
                    markersize=8, linewidth=2, alpha=0.8)

            # Highlight the best run
            if 'best_run' in sample_data:
                best_run_idx = sample_data['best_run'] - 1
                if 0 <= best_run_idx < len(diversity_scores):
                    ax.plot(sample_data['best_run'], diversity_scores[best_run_idx],
                        'o', color='red', markersize=12, zorder=5)

            ax.set_xlabel('Run Number', fontweight='bold')
            ax.set_ylabel('Diversity Score', fontweight='bold')
            ax.set_title('Diversity Scores Across Runs')

        # Common formatting
        ax.grid(True, alpha=0.3)
        ax.set_xticks(run_numbers)
        ax.set_xlim(0.5, len(run_numbers) + 0.5)

    else:
        # Fallback if no run data available
        ax.text(0.5, 0.5, 'Run scores data\nnot available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Scores by Run')

    # Panel n_params+2: Cluster distribution (only for k-means with cluster data)
    if is_kmeans_with_clusters:
        ax = axes[n_params + 1]

        cluster_assignments = sample_data['cluster_assignments']
        n_clusters = sample_data.get('n_clusters', len(set(cluster_assignments)))

        # Count items in each cluster
        cluster_counts = np.zeros(n_clusters)
        for cluster_id in cluster_assignments:
            if 0 <= cluster_id < n_clusters:
                cluster_counts[cluster_id] += 1

        # Bar plot
        cluster_ids = list(range(n_clusters))
        bars = ax.bar(cluster_ids, cluster_counts, color=color, alpha=0.7, edgecolor='black')

        # Add value labels on bars
        for bar, count in zip(bars, cluster_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{int(count)}', ha='center', va='bottom', fontweight='bold')

        # Labels and title
        ax.set_xlabel('Cluster Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Points per Cluster', fontsize=12, fontweight='bold')
        ax.set_title(f'Cluster Distribution (Best Run #{sample_data.get("best_run", "?")})\n' +
                    f'Total: {len(cluster_assignments)} points in {n_clusters} clusters',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Set x-axis ticks
        ax.set_xticks(cluster_ids)

        # Add mean line
        mean_size = np.mean(cluster_counts)
        ax.axhline(y=mean_size, color='red', linestyle='--', alpha=0.5,
                  label=f'Mean: {mean_size:.1f}')
        ax.legend()

    # Use appropriate layout method
    if not is_kmeans_with_clusters:
        plt.tight_layout()
    # For k-means with clusters, constrained_layout is already set on the figure

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_cluster_distribution_plot(sample_data, method_name, output_file, color='blue'):
    """Create a bar plot showing cluster sizes for k-means/k-medoids methods."""

    # Check if cluster assignments exist
    if 'cluster_assignments' not in sample_data:
        print(f"No cluster assignments found for {method_name}")
        return None

    cluster_assignments = sample_data['cluster_assignments']
    n_clusters = sample_data.get('n_clusters', len(set(cluster_assignments)))

    # Count items in each cluster
    cluster_counts = np.zeros(n_clusters)
    for cluster_id in cluster_assignments:
        if 0 <= cluster_id < n_clusters:
            cluster_counts[cluster_id] += 1

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Bar plot
    cluster_ids = list(range(n_clusters))
    bars = ax.bar(cluster_ids, cluster_counts, color=color, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar, count in zip(bars, cluster_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
               f'{int(count)}', ha='center', va='bottom', fontweight='bold')

    # Labels and title
    ax.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Points', fontsize=12, fontweight='bold')
    ax.set_title(f'{method_name} - Points per Cluster\n(Total: {len(cluster_assignments)} points in {n_clusters} clusters)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Set x-axis ticks
    ax.set_xticks(cluster_ids)
    ax.set_xticklabels([f'C{i}' for i in cluster_ids])

    # Add statistics
    mean_size = np.mean(cluster_counts)
    std_size = np.std(cluster_counts)
    ax.axhline(y=mean_size, color='red', linestyle='--', alpha=0.5,
              label=f'Mean: {mean_size:.1f} ± {std_size:.1f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file
