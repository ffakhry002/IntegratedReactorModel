"""
Visualization code package for sampling method analysis.
"""

from .data_loader import (
    load_core_configurations,
    load_physics_parameters,
    load_sample_data,
    load_all_results,
    get_method_colors,
    get_method_lists,
    load_physics_parameters_for_method,
    load_configurations_for_method
)

from .config_visualizer import (
    visualize_configuration,
    create_method_visualization,
    create_core_grid_visualization,
    create_irradiation_analysis
)

from .parameter_plots import (
    plot_physics_parameters_comparison,
    plot_diversity_comparison,
    create_diversity_comparison_by_type,
    create_parameter_scatter
)

from .analysis_plots import (
    create_method_visualizations,
    create_combined_analysis,
    create_summary_statistics
)

__all__ = [
    # Data loading
    'load_core_configurations',
    'load_physics_parameters',
    'load_sample_data',
    'load_all_results',
    'get_method_colors',
    'get_method_lists',

    # Configuration visualization
    'visualize_configuration',
    'create_method_visualization',
    'create_core_grid_visualization',
    'create_irradiation_analysis',

    # Parameter plots
    'plot_physics_parameters_comparison',
    'plot_diversity_comparison',
    'create_diversity_comparison_by_type',
    'create_parameter_scatter',

    # Analysis plots
    'create_method_visualization',
    'create_combined_analysis',
    'create_summary_statistics'
]
