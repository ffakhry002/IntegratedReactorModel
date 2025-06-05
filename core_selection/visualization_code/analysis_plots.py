"""
Analysis and combined plotting functions.
Handles method visualizations, combined analysis, and summary statistics.
UPDATED: Support for 6x6 mode detection in titles
"""
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path
from .data_loader import detect_6x6_mode
from .config_visualizer import (
    create_core_grid_visualization,
    create_irradiation_analysis,
    create_method_visualization
)
from .parameter_plots import (
    plot_physics_parameters_comparison,
    create_diversity_comparison_by_type,
    create_parameter_scatter
)
from .data_loader import create_summary_statistics_data

# Get the script directory (parent of visualization_code)
SCRIPT_DIR = Path(__file__).parent.parent.absolute()

def create_method_visualizations(method, sample_data, configurations,
                               physics_params, irradiation_sets, color, output_dir):
    """Create comprehensive visualizations for a single method."""
    selected_indices = sample_data['selected_indices']

    print(f"\nCreating visualizations for {method}...")

    # 1. Configuration grid
    config_file = f'{output_dir}/{method}_configurations.png'
    create_core_grid_visualization(configurations, selected_indices,
                                 method.upper(), config_file, color)

    # 2. Irradiation analysis
    irrad_file = f'{output_dir}/{method}_irradiation.png'
    create_irradiation_analysis(irradiation_sets, selected_indices,
                              method.upper(), irrad_file, color)

    # 3. Parameter scatter plots
    scatter_file = f'{output_dir}/{method}_parameters.png'
    create_parameter_scatter(physics_params, selected_indices,
                           method.upper(), scatter_file, color, sample_data)

    # 4. Copy individual core configurations to all_cores folder
    # Create all_cores directory structure
    if 'lattice' in output_dir:
        all_cores_dir = SCRIPT_DIR / 'visualizations/all_cores/lattice'
    elif 'geometric' in output_dir:
        all_cores_dir = SCRIPT_DIR / 'visualizations/all_cores/geometric'
    else:
        all_cores_dir = SCRIPT_DIR / 'visualizations/all_cores'

    all_cores_dir.mkdir(parents=True, exist_ok=True)

    # Copy the configurations file to all_cores
    config_source = Path(output_dir) / f'{method}_configurations.png'
    config_dest = all_cores_dir / f'{method}_configurations.png'

    if config_source.exists():
        shutil.copy2(config_source, config_dest)
        print(f"  ✓ Copied configurations to all_cores folder")

    print(f"  ✓ Created all visualizations for {method}")

def create_combined_analysis(samples_data, configurations, physics_params,
                           lattice_methods, geometric_methods, method_colors):
    """Create combined analysis plots comparing both method types."""
    # Detect 6x6 mode
    is_6x6 = detect_6x6_mode()
    mode_str = " (6x6 Central Square)" if is_6x6 else ""

    # 1. Overall diversity comparison
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    all_methods = lattice_methods + geometric_methods
    available_methods = [m for m in all_methods if m in samples_data]

    if available_methods:
        diversities = [samples_data[m]['diversity_score'] for m in available_methods]
        colors = [method_colors.get(m, '#808080') for m in available_methods]

        bars = ax.bar(range(len(available_methods)), diversities,
                     color=colors, alpha=0.8, edgecolor='black')

        ax.set_xlabel('Sampling Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Diversity Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Overall Diversity Comparison - All Methods{mode_str}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(available_methods)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in available_methods],
                          rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, diversity in zip(bars, diversities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{diversity:.3f}', ha='center', va='bottom', fontweight='bold')

        # Add separating line between lattice and geometric methods
        if len(lattice_methods) > 0 and len(geometric_methods) > 0:
            available_lattice = [m for m in lattice_methods if m in available_methods]
            lattice_count = len(available_lattice)

            if lattice_count > 0 and lattice_count < len(available_methods):
                ax.axvline(lattice_count - 0.5, color='black', linestyle='--', alpha=0.7)

                # Add method type labels
                if diversities:  # Check that we have diversity values
                    ax.text(lattice_count/2 - 0.5, max(diversities)*0.9, 'Lattice-based',
                           ha='center', va='center', fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
                    ax.text(lattice_count + (len(available_methods) - lattice_count)/2 - 0.5,
                           max(diversities)*0.9, 'Geometric/Physics',
                           ha='center', va='center', fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    else:
        ax.text(0.5, 0.5, 'No sampling results available',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'Overall Diversity Comparison - No Results{mode_str}')

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'visualizations/overall_diversity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Method type comparison
    _create_method_type_comparison(samples_data, lattice_methods, geometric_methods, method_colors)

    # 3. Performance metrics summary
    _create_performance_summary(samples_data, available_methods, method_colors)


def _create_method_type_comparison(samples_data, lattice_methods, geometric_methods, method_colors):
    """Create comparison between method types."""
    # Detect 6x6 mode
    is_6x6 = detect_6x6_mode()
    mode_str = " (6x6 Central Square)" if is_6x6 else ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Collect data for each type
    lattice_data = [(m, samples_data[m]['diversity_score'])
                    for m in lattice_methods if m in samples_data]
    geometric_data = [(m, samples_data[m]['diversity_score'])
                      for m in geometric_methods if m in samples_data]

    # Box plot comparison
    if lattice_data or geometric_data:
        data_to_plot = []
        labels = []

        if lattice_data:
            data_to_plot.append([d[1] for d in lattice_data])
            labels.append('Lattice-based')

        if geometric_data:
            data_to_plot.append([d[1] for d in geometric_data])
            labels.append('Geometric/Physics')

        if data_to_plot:
            bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)

            # Color the boxes
            colors_box = ['lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax1.set_ylabel('Diversity Score', fontweight='bold')
            ax1.set_title('Method Type Comparison', fontweight='bold')
            ax1.grid(True, alpha=0.3)

    # Average diversity by type
    if lattice_data or geometric_data:
        avg_data = []
        std_data = []
        type_labels = []

        if lattice_data:
            scores = [d[1] for d in lattice_data]
            avg_data.append(np.mean(scores))
            std_data.append(np.std(scores))
            type_labels.append('Lattice-based')

        if geometric_data:
            scores = [d[1] for d in geometric_data]
            avg_data.append(np.mean(scores))
            std_data.append(np.std(scores))
            type_labels.append('Geometric/Physics')

        if avg_data:
            x = np.arange(len(type_labels))
            bars = ax2.bar(x, avg_data, yerr=std_data, capsize=5,
                          color=['lightblue', 'lightgreen'][:len(avg_data)],
                          alpha=0.7, edgecolor='black')

            ax2.set_xticks(x)
            ax2.set_xticklabels(type_labels)
            ax2.set_ylabel('Average Diversity Score', fontweight='bold')
            ax2.set_title('Average Performance by Method Type', fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # Add value labels
            for bar, avg, std in zip(bars, avg_data, std_data):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                        f'{avg:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle(f'Method Type Analysis{mode_str}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'visualizations/method_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def _create_performance_summary(samples_data, available_methods, method_colors):
    """Create performance summary visualization."""
    # Detect 6x6 mode
    is_6x6 = detect_6x6_mode()
    mode_str = " (6x6 Central Square)" if is_6x6 else ""

    if not available_methods:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Sort methods by diversity score
    method_scores = [(m, samples_data[m]['diversity_score']) for m in available_methods]
    method_scores.sort(key=lambda x: x[1], reverse=True)

    methods = [m[0] for m in method_scores]
    scores = [m[1] for m in method_scores]
    colors = [method_colors.get(m, '#808080') for m in methods]

    # Create horizontal bar chart
    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.8, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace('_', ' ').title() for m in methods])
    ax.set_xlabel('Diversity Score', fontweight='bold')
    ax.set_title(f'Sampling Methods Ranked by Performance{mode_str}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
               f'{score:.3f}', ha='left', va='center', fontweight='bold')

    # Add performance tiers
    if scores:
        max_score = max(scores)
        min_score = min(scores)
        score_range = max_score - min_score

        if score_range > 0:
            # Define tier boundaries
            tier_boundaries = [
                min_score + 0.75 * score_range,  # Top tier
                min_score + 0.5 * score_range,   # Mid tier
                min_score + 0.25 * score_range   # Low tier
            ]

            # Add tier lines
            for boundary in tier_boundaries:
                ax.axvline(boundary, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'visualizations/performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_statistics(samples_data, lattice_methods, geometric_methods):
    """Create and save summary statistics."""
    # Detect 6x6 mode
    is_6x6 = detect_6x6_mode()
    mode_str = " (6x6 Central Square)" if is_6x6 else ""

    summary = create_summary_statistics_data(samples_data, lattice_methods, geometric_methods)

    # Save summary to file
    summary_file = SCRIPT_DIR / 'visualizations/summary_statistics.txt'
    with open(summary_file, 'w') as f:
        f.write(f"SAMPLING METHODS SUMMARY STATISTICS{mode_str}\n")
        f.write("="*50 + "\n\n")

        if is_6x6:
            f.write("Configuration Space: 6x6 Central Square Only\n")
            f.write("Note: Physics parameters calculated considering full 8x8 grid context\n\n")

        for method_type, stats in summary.items():
            f.write(f"{method_type}:\n")
            f.write(f"  Count: {stats['count']}\n")
            f.write(f"  Mean Diversity: {stats['mean_diversity']:.4f}\n")
            f.write(f"  Std Diversity: {stats['std_diversity']:.4f}\n")
            f.write(f"  Min Diversity: {stats['min_diversity']:.4f}\n")
            f.write(f"  Max Diversity: {stats['max_diversity']:.4f}\n\n")

        f.write("Individual Method Results:\n")
        f.write("-" * 30 + "\n")

        # Sort by diversity score
        sorted_methods = sorted(samples_data.items(),
                              key=lambda x: x[1]['diversity_score'],
                              reverse=True)

        for method, data in sorted_methods:
            f.write(f"{method}: {data['diversity_score']:.4f}\n")

    # Create summary visualization
    # _create_summary_visualization(summary, samples_data, lattice_methods, geometric_methods)


def _create_summary_visualization(summary, samples_data, lattice_methods, geometric_methods):
    """Create a visual summary of the statistics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Count comparison
    counts = [summary['Lattice Methods']['count'],
              summary['Geometric Methods']['count']]
    labels = ['Lattice Methods', 'Geometric Methods']

    ax1.bar(labels, counts, color=['lightblue', 'lightgreen'],
            alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Methods')
    ax1.set_title('Method Count by Type')
    ax1.grid(True, alpha=0.3)

    # 2. Mean diversity comparison
    means = [summary['Lattice Methods']['mean_diversity'],
             summary['Geometric Methods']['mean_diversity']]
    stds = [summary['Lattice Methods']['std_diversity'],
            summary['Geometric Methods']['std_diversity']]

    ax2.bar(labels, means, yerr=stds, capsize=5,
            color=['lightblue', 'lightgreen'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Mean Diversity Score')
    ax2.set_title('Average Performance by Type')
    ax2.grid(True, alpha=0.3)

    # 3. Range comparison
    lattice_range = [summary['Lattice Methods']['min_diversity'],
                    summary['Lattice Methods']['max_diversity']]
    geometric_range = [summary['Geometric Methods']['min_diversity'],
                      summary['Geometric Methods']['max_diversity']]

    ax3.plot([0, 1], lattice_range, 'o-', color='blue', linewidth=3,
             markersize=10, label='Lattice')
    ax3.plot([2, 3], geometric_range, 'o-', color='green', linewidth=3,
             markersize=10, label='Geometric')
    ax3.set_xticks([0.5, 2.5])
    ax3.set_xticklabels(labels)
    ax3.set_ylabel('Diversity Score Range')
    ax3.set_title('Score Range by Type')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Distribution histogram
    lattice_scores = [samples_data[m]['diversity_score']
                     for m in lattice_methods if m in samples_data]
    geometric_scores = [samples_data[m]['diversity_score']
                       for m in geometric_methods if m in samples_data]

    if lattice_scores:
        ax4.hist(lattice_scores, bins=5, alpha=0.5, color='blue',
                label='Lattice', edgecolor='black')
    if geometric_scores:
        ax4.hist(geometric_scores, bins=5, alpha=0.5, color='green',
                label='Geometric', edgecolor='black')

    ax4.set_xlabel('Diversity Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Score Distribution by Type')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Summary Statistics Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/summary_statistics_visual.png', dpi=300, bbox_inches='tight')
    plt.close()
