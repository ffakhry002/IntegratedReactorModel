import openmc
import os
import sys
import numpy as np
import shutil

# Add root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
workspace_root = os.path.dirname(root_dir)

# Add all necessary paths
sys.path.append(root_dir)
sys.path.append(workspace_root)
sys.path.append(os.path.join(root_dir, 'Reactor'))


from inputs import inputs
from Reactor.materials import make_materials
from Reactor.geometry_helpers.core import build_core_uni
from execution.tallies.irradiation_tallies import create_irradiation_tallies, create_irradiation_axial_tallies
from execution.tallies.core_tallies import create_nutotal_tallies, create_coreflux_tallys
from execution.tallies.power_tallies import create_power_tallies
from execution.outputs import process_results


def make_and_run_openmc_model(model, statepoint_name, folder='Output/'):
    """Run OpenMC simulation with given parameters.

    Parameters
    ----------
    model : openmc.Model
        OpenMC model object containing geometry, settings, and tallies
    statepoint_name : str
        Name for the statepoint file
    folder : str, optional
        Output directory, defaults to 'Output/'
    """
    # Create output directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ['OPENMC_CROSS_SECTIONS'] = os.path.join(workspace_root, 'cross_sections', 'cross_sections.xml')
    model.export_to_xml(folder)
    openmc.run(cwd=folder,geometry_debug=False)

    # Rename statepoint file
    old_sp = os.path.join(folder, f'statepoint.{model.settings.batches}.h5')
    new_sp = os.path.join(folder, f'statepoint.{statepoint_name}.h5')
    if os.path.exists(old_sp):
        shutil.copy2(old_sp, new_sp)  # Copy instead of rename to keep original
        os.remove(old_sp)  # Remove original after successful copy

    # Reset auto IDs for next run
    openmc.mixin.reset_auto_ids()

def run_eigenvalue():
    """Run eigenvalue calculation.

    Returns
    -------
    tuple
        (k_effective, standard_deviation)
    """
    # Use values from inputs file
    batches = inputs['batches']
    inactive = inputs['inactive']
    particles = inputs['particles']

    print(f"Starting eigenvalue calculation with {batches} batches, {inactive} inactive, {particles} particles")

    # Create materials and geometry
    mat_dict, materials = make_materials(mat_list=None)
    core_universe, first_irr_universe = build_core_uni(mat_dict)
    geometry = openmc.Geometry(core_universe)

    # Create settings
    settings = openmc.Settings()
    settings.verbosity = 7
    settings.seed = 1
    settings.batches = batches
    settings.inactive = inactive
    settings.particles = particles

    # Calculate source region based on maximum fuel assembly row
    max_assemblies = max([len(row) - list(row).count('C') for row in inputs['core_lattice']])

    # Calculate the side length based on assembly type
    if inputs['assembly_type'] == 'Plate':
        assembly_unit_width = (inputs['fuel_plate_width'] + 2*inputs['clad_structure_width']) * 100  # cm
    else:  # Pin type
        assembly_unit_width = inputs['pin_pitch'] * inputs['n_side_pins'] * 100  # cm

    total_width = max_assemblies * assembly_unit_width
    half_width = total_width / 2
    half_height = inputs['fuel_height'] * 50   # Convert to cm

    # Create initial source distribution - use actual fuel region size
    uniform_dist = openmc.stats.Box(
        [-half_width, -half_width, -half_height],
        [half_width, half_width, half_height],
        only_fissionable=True
    )

    # Create source
    source = openmc.IndependentSource()
    source.space = uniform_dist
    source.strength = 1.0
    settings.source = source

    # Create entropy mesh matching the source region
    entropy_mesh = openmc.RegularMesh()
    entropy_mesh.lower_left = [-half_width, -half_width, -half_height]
    entropy_mesh.upper_right = [half_width, half_width, half_height]
    entropy_mesh.dimension = [20, 20, 20]
    settings.entropy_mesh = entropy_mesh

    settings.temperature = {'method': 'interpolation', 'tolerance': 100}
    settings.run_mode = 'eigenvalue'

    # Add tallies
    tallies = openmc.Tallies()
    tallies.extend(create_irradiation_tallies())
    tallies.extend(create_irradiation_axial_tallies())
    tallies.extend(create_nutotal_tallies())
    tallies.extend(create_coreflux_tallys())
    tallies.extend(create_power_tallies())

    # Create model
    model = openmc.model.Model()
    model.geometry = geometry
    model.settings = settings
    model.tallies = tallies

    # Determine output directory based on how we're running
    running_directly = os.path.basename(sys.argv[0]) == 'run.py'
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not running_directly and os.path.exists(os.path.join(root_dir, 'simulation_data')):
        # Running from main.py
        output_dir = os.path.join(root_dir, 'simulation_data', 'transport_data')
    else:
        # Running directly or simulation_data doesn't exist
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Output')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run the simulation
    statepoint_name = 'eigenvalue'
    make_and_run_openmc_model(model, statepoint_name, output_dir)

    # Extract results
    sp_path = os.path.join(output_dir, f'statepoint.{statepoint_name}.h5')
    with openmc.StatePoint(sp_path) as sp:
        k_effective = sp.keff
        # Process and save results
        process_results(sp, k_effective)

    return k_effective.nominal_value, k_effective.std_dev

if __name__ == '__main__':
    # Run with default parameters
    run_eigenvalue()
