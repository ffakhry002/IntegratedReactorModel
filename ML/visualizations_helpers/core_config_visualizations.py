"""
Core configuration visualization module for Nuclear Reactor ML.
Generates visual representations of reactor core layouts from train/test files.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Optional


def parse_config_file(file_path: str) -> List[Dict]:
    """
    Parse a configuration file (train.txt or test.txt) to extract configurations and descriptions.

    Returns:
        List of dictionaries with 'config' (8x8 array) and 'description' fields
    """
    configurations = []

    with open(file_path, 'r') as f:
        content = f.read()

    # Split by RUN entries
    runs = content.split('RUN ')

    for run in runs[1:]:  # Skip the first split which is before "RUN"
        lines = run.split('\n')

        # Extract run number
        run_num = lines[0].replace(':', '').strip()

        # Find description
        description = None
        core_lattice = None

        for i, line in enumerate(lines):
            if 'Description:' in line:
                description = line.split('Description:')[1].strip()
            elif 'core_lattice:' in line:
                # Extract the core lattice - it might span multiple lines
                lattice_str = line.split('core_lattice:')[1].strip()

                # If it doesn't end with ]], keep reading lines
                j = i + 1
                while j < len(lines) and not lattice_str.rstrip().endswith(']]'):
                    lattice_str += ' ' + lines[j].strip()
                    j += 1

                # Parse the Python list string
                try:
                    # Replace I_1, I_2, etc. with just 'I' for visualization
                    lattice_str = lattice_str.replace("'I_1'", "'I'")
                    lattice_str = lattice_str.replace("'I_2'", "'I'")
                    lattice_str = lattice_str.replace("'I_3'", "'I'")
                    lattice_str = lattice_str.replace("'I_4'", "'I'")

                    # Safely evaluate the string as a Python list
                    import ast
                    core_lattice = ast.literal_eval(lattice_str)

                    # Convert to numpy array
                    if len(core_lattice) == 8 and all(len(row) == 8 for row in core_lattice):
                        config_array = np.array(core_lattice)

                        configurations.append({
                            'number': f'Run {run_num}',
                            'description': description if description else f'Run {run_num}',
                            'config': config_array
                        })
                except Exception as e:
                    print(f"  Warning: Could not parse core lattice for Run {run_num}: {e}")

                break  # Found core_lattice, no need to continue

    # If no RUN format found, try the old format
    if not configurations:
        print("  No RUN format found, trying alternative format...")

        with open(file_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for configuration headers
            if line.startswith('Config') and ':' in line:
                # Extract description
                parts = line.split(':', 1)
                config_num = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ""

                # Read the 8x8 grid
                config_grid = []
                i += 1

                # Read 8 lines for the configuration
                for row in range(8):
                    if i < len(lines):
                        row_data = lines[i].strip().split()
                        if len(row_data) == 8:
                            config_grid.append(row_data)
                        i += 1

                # If we got a complete 8x8 grid, add it
                if len(config_grid) == 8:
                    config_array = np.array(config_grid)
                    configurations.append({
                        'number': config_num,
                        'description': description,
                        'config': config_array
                    })
            else:
                i += 1

    return configurations


def create_core_visualization(config: np.ndarray, title: str, ax: plt.Axes) -> None:
    """
    Create a visualization of a single core configuration.

    Args:
        config: 8x8 numpy array with 'C', 'F', 'I' values
        title: Title for the configuration
        ax: Matplotlib axes to draw on
    """
    # Color mapping
    color_map = {
        'C': [0.7, 0.9, 1.0],  # Light blue for coolant
        'F': [0.7, 1.0, 0.7],  # Light green for fuel
        'I': [1.0, 0.4, 0.4],  # Red for irradiation
    }

    # Create color array
    config_colors = np.zeros((8, 8, 3))
    for row in range(8):
        for col in range(8):
            element = config[row, col]
            if element in color_map:
                config_colors[row, col] = color_map[element]
            else:
                config_colors[row, col] = [0.9, 0.9, 0.9]  # Gray for unknown

    # Display the configuration
    ax.imshow(config_colors, aspect='equal')
    ax.set_title(title, fontsize=8, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add grid lines
    for x in range(9):
        ax.axvline(x-0.5, color='black', linewidth=0.5)
    for y in range(9):
        ax.axhline(y-0.5, color='black', linewidth=0.5)

    # Removed text labels - no more C, F, I text in cells


def create_config_grid_visualization(
    configurations: List[Dict],
    output_file: str,
    title: str,
    max_per_page: int = 20
) -> None:
    """
    Create a grid visualization of multiple core configurations.

    Args:
        configurations: List of configuration dictionaries
        output_file: Path to save the visualization
        title: Main title for the figure
        max_per_page: Maximum configurations per page (ignored - all on one page)
    """
    n_configs = len(configurations)

    if n_configs == 0:
        print("No configurations to visualize")
        return

    # Calculate grid dimensions to fit all on one page
    # Aim for roughly square layout
    cols = int(np.ceil(np.sqrt(n_configs)))
    rows = int(np.ceil(n_configs / cols))

    # Adjust figure size based on number of configurations
    # Make each subplot smaller when there are many
    subplot_size = max(1.5, min(2.5, 20.0 / max(rows, cols)))

    # Create figure with all configurations
    fig, axes = plt.subplots(rows, cols, figsize=(subplot_size*cols, subplot_size*rows))

    # Add main title
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Flatten axes for easier indexing
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Plot each configuration
    for i, config_dict in enumerate(configurations):
        if i < len(axes):
            ax = axes[i]
            # Show both run number and description
            title = f"{config_dict['description']}"
            create_core_visualization(config_dict['config'], title, ax)

    # Hide unused subplots
    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)

    # Add legend at the bottom
    legend_elements = [
        patches.Patch(color=[0.7, 0.9, 1.0], label='C - Coolant'),
        patches.Patch(color=[0.7, 1.0, 0.7], label='F - Fuel'),
        patches.Patch(color=[1.0, 0.4, 0.4], label='I - Irradiation')
    ]

    # Adjust legend position based on figure size
    legend_y = -0.02 if rows > 10 else -0.05
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
              bbox_to_anchor=(0.5, legend_y), frameon=True)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {os.path.basename(output_file)} ({n_configs} configurations)")


def extract_irradiation_positions(configurations: List[Dict]) -> List[List[Tuple[int, int]]]:
    """
    Extract irradiation positions from configurations.

    Args:
        configurations: List of configuration dictionaries

    Returns:
        List of irradiation position lists, each containing (row, col) tuples
    """
    irradiation_sets = []

    for config_dict in configurations:
        config = config_dict['config']
        positions = []

        for row in range(8):
            for col in range(8):
                if config[row, col] == 'I':
                    positions.append((row, col))

        irradiation_sets.append(positions)

    return irradiation_sets


def create_irradiation_heatmap(
    configurations: List[Dict],
    output_file: str,
    title: str
) -> None:
    """
    Create irradiation position frequency heatmap.

    Args:
        configurations: List of configuration dictionaries
        output_file: Path to save the heatmap
        title: Title for the heatmap
    """
    # Extract irradiation positions
    irradiation_sets = extract_irradiation_positions(configurations)

    # Create position frequency matrix
    position_counts = np.zeros((8, 8))

    for irrad_set in irradiation_sets:
        for row, col in irrad_set:
            position_counts[row, col] += 1

    # Create the heatmap
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Create heatmap
    im = ax.imshow(position_counts, cmap='Reds', aspect='equal', origin='upper')
    ax.set_title('Irradiation Position Frequency', fontsize=14)
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Frequency')
    cbar.ax.tick_params(labelsize=10)

    # Add grid
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(np.arange(8))
    ax.set_yticklabels(np.arange(8))

    # Add grid lines
    ax.set_xticks(np.arange(8.5), minor=True)
    ax.set_yticks(np.arange(8.5), minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    # Add text annotations for non-zero values
    for row in range(8):
        for col in range(8):
            if position_counts[row, col] > 0:
                text = ax.text(col, row, f'{int(position_counts[row, col])}',
                             ha='center', va='center', color='white' if position_counts[row, col] > position_counts.max()/2 else 'black',
                             fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {os.path.basename(output_file)}")


def generate_core_config_visualizations(output_dir: str) -> None:
    """
    Main function to generate core configuration visualizations.

    Args:
        output_dir: Directory to save visualizations
    """
    print("\n📊 Generating Core Configuration Visualizations...")

    # Create core_images subdirectory
    core_images_dir = os.path.join(output_dir, 'core_images')
    os.makedirs(core_images_dir, exist_ok=True)

    # Get the absolute path of the script and find the project root
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Try to find the ML directory (could be current dir or parent)
    ml_dir = None
    if os.path.basename(script_dir) == 'ML':
        ml_dir = script_dir
    elif os.path.exists(os.path.join(script_dir, 'ML')):
        ml_dir = os.path.join(script_dir, 'ML')
    elif os.path.exists(os.path.join(os.path.dirname(script_dir), 'ML')):
        ml_dir = os.path.join(os.path.dirname(script_dir), 'ML')

    # Possible locations for config files
    possible_paths = []

    # Add ML/data paths if we found ML directory
    if ml_dir:
        possible_paths.extend([
            os.path.join(ml_dir, 'data', 'train.txt'),
            os.path.join(ml_dir, 'data', 'test.txt')
        ])

    # Add other possible paths
    possible_paths.extend([
        os.path.join(script_dir, 'data', 'train.txt'),
        os.path.join(script_dir, 'data', 'test.txt'),
        os.path.join(script_dir, '..', 'data', 'train.txt'),
        os.path.join(script_dir, '..', 'data', 'test.txt'),
        os.path.join(script_dir, '..', 'ML', 'data', 'train.txt'),
        os.path.join(script_dir, '..', 'ML', 'data', 'test.txt'),
        '/root/IntegratedReactorModel/ML/data/train.txt',
        '/root/IntegratedReactorModel/ML/data/test.txt',
        'ML/data/train.txt',
        'ML/data/test.txt',
        'data/train.txt',
        'data/test.txt'
    ])

    train_file = None
    test_file = None

    # Find train.txt
    for path in possible_paths:
        if 'train.txt' in path and os.path.exists(path):
            train_file = path
            print(f"  Found train.txt at: {train_file}")
            break

    # Find test.txt
    for path in possible_paths:
        if 'test.txt' in path and os.path.exists(path):
            test_file = path
            print(f"  Found test.txt at: {test_file}")
            break

    configs_found = False

    # Process train configurations
    if train_file and os.path.exists(train_file):
        print(f"\n  Processing training configurations from: {train_file}")
        try:
            train_configs = parse_config_file(train_file)
            if train_configs:
                print(f"  Found {len(train_configs)} training configurations")
                create_config_grid_visualization(
                    train_configs,
                    os.path.join(core_images_dir, 'train_cores.png'),
                    'Training Core Configurations'
                )
                create_irradiation_heatmap(
                    train_configs,
                    os.path.join(core_images_dir, 'train_irradiation_heatmap.png'),
                    'Training Irradiation Heatmap'
                )
                configs_found = True
            else:
                print("  No configurations found in train.txt")
        except Exception as e:
            print(f"  ERROR processing train.txt: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  Train.txt file not found")
        print(f"  Searched in: {[p for p in possible_paths if 'train.txt' in p][:5]}")

    # Process test configurations
    if test_file and os.path.exists(test_file):
        print(f"\n  Processing test configurations from: {test_file}")
        try:
            test_configs = parse_config_file(test_file)
            if test_configs:
                print(f"  Found {len(test_configs)} test configurations")
                create_config_grid_visualization(
                    test_configs,
                    os.path.join(core_images_dir, 'test_cores.png'),
                    'Test Core Configurations'
                )
                create_irradiation_heatmap(
                    test_configs,
                    os.path.join(core_images_dir, 'test_irradiation_heatmap.png'),
                    'Test Irradiation Heatmap'
                )
                configs_found = True
            else:
                print("  No configurations found in test.txt")
        except Exception as e:
            print(f"  ERROR processing test.txt: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  Test.txt file not found")
        print(f"  Searched in: {[p for p in possible_paths if 'test.txt' in p][:5]}")

    if configs_found:
        print(f"\n  ✓ Core configuration visualizations saved to: {core_images_dir}")
    else:
        print("\n  ⚠ No configuration files found to visualize")
        print("    Expected files: ML/data/train.txt and ML/data/test.txt")


# For testing the module independently
if __name__ == "__main__":
    # Test with current directory
    generate_core_config_visualizations(".")
