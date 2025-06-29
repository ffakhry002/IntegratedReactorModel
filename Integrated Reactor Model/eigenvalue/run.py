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
from eigenvalue.tallies.irradiation_tallies import create_irradiation_tallies, create_irradiation_axial_tallies
from eigenvalue.tallies.core_tallies import create_nutotal_tallies, create_coreflux_tallys
from eigenvalue.tallies.power_tallies import create_power_tallies
from eigenvalue.outputs import process_results


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

def run_eigenvalue(inputs_dict=None):
    """Run eigenvalue calculation.

    Parameters
    ----------
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    tuple
        (k_effective, standard_deviation)
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    # Use values from inputs file
    batches = inputs_dict['batches']
    inactive = inputs_dict['inactive']
    particles = inputs_dict['particles']

    print(f"Starting eigenvalue calculation with {batches} batches, {inactive} inactive, {particles} particles")

    # Reset OpenMC auto IDs before creating any geometry objects
    openmc.mixin.reset_auto_ids()
    print("OpenMC auto IDs reset before geometry creation")

    # Create materials and geometry
    mat_dict, materials = make_materials(mat_list=None, inputs_dict=inputs_dict)
    core_universe, first_irr_universe = build_core_uni(mat_dict, inputs_dict=inputs_dict)
    geometry = openmc.Geometry(core_universe)

    # Create settings
    settings = openmc.Settings()
    settings.verbosity = 7
    settings.seed = 1
    settings.batches = batches
    settings.inactive = inactive
    settings.particles = particles

    # Calculate source region based on maximum fuel assembly row
    lattice_array = np.array(inputs_dict['core_lattice'])
    n_rows, n_cols = lattice_array.shape

    # Count the effective number of assemblies in each row/column (excluding coolant)
    row_counts = [np.sum([1 for cell in row if cell != 'C']) for row in lattice_array]
    col_counts = [np.sum([1 for row in lattice_array if row[j] != 'C']) for j in range(n_cols)]

    max_row_assemblies = max(row_counts)
    max_col_assemblies = max(col_counts)

    # Calculate the side length based on assembly type
    if inputs_dict['assembly_type'] == 'Plate':
        assembly_unit_width = (inputs_dict['fuel_plate_width'] + 2*inputs_dict['clad_structure_width']) * 100  # cm
    else:  # Pin type
        assembly_unit_width = inputs_dict['pin_pitch'] * inputs_dict['n_side_pins'] * 100  # cm

    # Calculate total width of active core region
    width_x = max_col_assemblies * assembly_unit_width
    width_y = max_row_assemblies * assembly_unit_width

    half_width_x = width_x / 2
    half_width_y = width_y / 2
    half_height = inputs_dict['fuel_height'] * 50   # Convert to cm

    # Create initial source distribution - use actual fuel region size
    uniform_dist = openmc.stats.Box(
        [-half_width_x, -half_width_y, -half_height],
        [half_width_x, half_width_y, half_height],
        only_fissionable=True
    )

    # Create source
    source = openmc.IndependentSource()
    source.space = uniform_dist
    source.strength = 1.0
    settings.source = source

    # Create entropy mesh matching the source region
    entropy_mesh = openmc.RegularMesh()
    entropy_mesh.lower_left = [-half_width_x, -half_width_y, -half_height]
    entropy_mesh.upper_right = [half_width_x, half_width_y, half_height]
    entropy_mesh.dimension = inputs_dict['entropy_mesh_dimension']
    settings.entropy_mesh = entropy_mesh

    settings.temperature = {'method': 'interpolation', 'tolerance': 100}
    settings.run_mode = 'eigenvalue'

    # Add tallies
    tallies = openmc.Tallies()
    tallies.extend(create_irradiation_tallies(inputs_dict=inputs_dict))
    tallies.extend(create_irradiation_axial_tallies(inputs_dict=inputs_dict))
    tallies.extend(create_nutotal_tallies())
    tallies.extend(create_coreflux_tallys(inputs_dict=inputs_dict))

    # Only add power tallies if tally_power is enabled
    if inputs_dict.get('tally_power', True):
        tallies.extend(create_power_tallies(inputs_dict=inputs_dict))

    # Create model
    model = openmc.model.Model()
    model.geometry = geometry
    model.settings = settings
    model.tallies = tallies

    # Determine output directory based on how we're running
    running_directly = os.path.basename(sys.argv[0]) == 'run.py'
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check if we're in a parametric study run directory
    current_dir = os.getcwd()
    if 'parametric_simulation_' in current_dir and 'run_' in current_dir:
        # We're in a parametric study run directory - use transport_data subdirectory
        output_dir = os.path.join(current_dir, 'transport_data')
    elif not running_directly and os.path.exists(os.path.join(root_dir, 'simulation_data')):
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
        process_results(sp, k_effective, inputs_dict)

    return k_effective.nominal_value, k_effective.std_dev

if __name__ == '__main__':
    # Run with default parameters
    run_eigenvalue()
