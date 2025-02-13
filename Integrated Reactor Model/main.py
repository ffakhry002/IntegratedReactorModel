"""
Main script to run the integrated reactor simulation.
"""

import os
import sys
import shutil
from execution.run import create_simulation_directories, run_eigenvalue
from Reactor.geometry import plot_geometry
from ThermalHydraulics.TH_refactored import THSystem
from plotting.plotall import plot_all_flux_distributions
from inputs import inputs

def cleanup_all_pycache():
    """Remove all __pycache__ directories in the entire project structure."""
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Walk through all directories in the project
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__pycache__' in dirnames:
            pycache_path = os.path.join(dirpath, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                # print(f"Removed: {pycache_path}")
            except Exception as e:
                print(f"Error removing {pycache_path}: {e}")

def main():
    """Run the complete reactor simulation workflow."""
    # Clean up all __pycache__ directories first
    print("\nCleaning up __pycache__ directories...")
    cleanup_all_pycache()

    # Get root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Create simulation directory structure
    print("\nCreating simulation directory structure...")
    # Create main simulation directory
    sim_dir = os.path.join(root_dir, 'simulation_data')
    if os.path.exists(sim_dir):
        shutil.rmtree(sim_dir)
    os.makedirs(sim_dir)

    # Create subdirectories
    dirs = {
        'geometry_materials': os.path.join(sim_dir, 'Geometry_and_Materials'),
        'thermal_hydraulics': os.path.join(sim_dir, 'ThermalHydraulics'),
        'xml_h5': os.path.join(sim_dir, 'xml_and_h5'),
        'flux_plots': os.path.join(sim_dir, 'flux_plots')
    }

    # Create each subdirectory
    for dir_path in dirs.values():
        os.makedirs(dir_path)

    # Run geometry and materials generation
    print("\nGenerating geometry and materials...")
    plot_geometry(dirs['geometry_materials'])

    # Run thermal hydraulics
    print("\nRunning thermal hydraulics analysis...")
    th_system = THSystem(inputs)
    thermal_state = th_system.calculate_temperature_distribution()
    th_system.write_results(dirs['thermal_hydraulics'])

    # Run OpenMC simulation
    print("\nRunning OpenMC simulation...")
    k_eff, k_std = run_eigenvalue()
    print(f"\nSimulation completed successfully!")
    print(f"k-effective = {k_eff:.6f} ± {k_std:.6f}")

    # Generate flux plots
    print("\nGenerating flux plots...")
    plot_all_flux_distributions()

    # Final cleanup of any new __pycache__ directories created during the run
    print("\nFinal cleanup of __pycache__ directories...")
    cleanup_all_pycache()

if __name__ == "__main__":
    main()
