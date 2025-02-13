"""
Main script for running full core depletion calculations.
"""

import os
import sys
import numpy as np
import openmc
import openmc.deplete
import multiprocessing
import matplotlib.pyplot as plt
import h5py

# Add parent directory to path to find inputs.py and make depletion package importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inputs import inputs
import depletion.depletion_operator as depletion_operator
from Reactor.materials import make_materials
from Reactor.geometry_helpers.core import build_core_uni

def configure_multiprocessing():
    """Configure multiprocessing settings based on environment.

    Returns
    -------
    bool
        Whether multiprocessing should be enabled
    """
    # Check if running with MPI
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        using_mpi = size > 1
    except ImportError:
        using_mpi = False
        rank = 0

    # Get number of CPU cores
    cpu_count = multiprocessing.cpu_count()

    # Determine if we should use multiprocessing
    if using_mpi:
        print(f"\nRank {rank}: Running with MPI ({size} processes)")
        print(f"Rank {rank}: Disabling multiprocessing for depletion to avoid MPI conflicts")
        return False
    else:
        print(f"\nRunning in serial mode with {cpu_count} CPU cores available")
        print("Enabling multiprocessing for depletion to utilize all cores")
        return True

def create_model(depletion_type='core'):
    """Create a new OpenMC model for depletion calculations.

    Parameters
    ----------
    depletion_type : str
        Type of depletion model to create: 'core', 'assembly', 'assembly_enhanced',
        'element', or 'element_enhanced'

    Returns
    -------
    openmc.Model
        The OpenMC model instance
    """
    # Create materials and geometry
    mat_dict, materials = make_materials(None)  # Get both materials dict and collection

    # Debug print for material temperatures
    print("\nMaterial temperatures:")
    for mat in materials:
        if hasattr(mat, 'temperature'):
            print(f"Material {mat.id} ({mat.name}): {mat.temperature} K")
        else:
            print(f"Material {mat.id} ({mat.name}): No temperature set")

    # Set boundary type based on calculation type
    # Core uses vacuum boundaries, everything else uses reflective for k-infinity
    boundary_type = 'vacuum' if depletion_type == 'core' else 'reflective'
    calc_type = 'keff' if depletion_type == 'core' else 'k∞'

    # Create geometry based on depletion type
    if depletion_type == 'core':
        print("\nCreating full core model for depletion...")
        root_universe, _ = build_core_uni(mat_dict)
    elif depletion_type in ['assembly', 'assembly_enhanced']:
        is_enhanced = (depletion_type == 'assembly_enhanced')
        print(f"\nCreating single {'enhanced ' if is_enhanced else ''}assembly model for {calc_type} calculation...")
        if inputs['assembly_type'] == 'Pin':
            from Reactor.geometry_helpers.pin_fuel import build_fuel_assembly_uni
            root_universe = build_fuel_assembly_uni(mat_dict, is_enhanced=is_enhanced)

            # Calculate assembly bounds
            pin_pitch = inputs['pin_pitch'] * 100  # m to cm
            n_pins = inputs['n_side_pins']
            half_width = (pin_pitch * n_pins) / 2

            # Create bounding surfaces with specified BC
            left = openmc.XPlane(x0=-half_width, boundary_type=boundary_type)
            right = openmc.XPlane(x0=half_width, boundary_type=boundary_type)
            bottom = openmc.YPlane(y0=-half_width, boundary_type=boundary_type)
            top = openmc.YPlane(y0=half_width, boundary_type=boundary_type)

            # Create bottom and top axial surfaces
            fuel_height = inputs['fuel_height'] * 100  # m to cm
            bottom_z = openmc.ZPlane(z0=-fuel_height/2, boundary_type=boundary_type)
            top_z = openmc.ZPlane(z0=fuel_height/2, boundary_type=boundary_type)

            # Create cell with bounds
            main_cell = openmc.Cell(fill=root_universe)
            main_cell.region = +left & -right & +bottom & -top & +bottom_z & -top_z
            root_universe = openmc.Universe(cells=[main_cell])

        else:  # Plate assembly
            from Reactor.geometry_helpers.plate_fuel import build_fuel_assembly_uni
            root_universe = build_fuel_assembly_uni(mat_dict, is_enhanced=is_enhanced)

            # Calculate assembly bounds
            plate_pitch = inputs['fuel_plate_pitch'] * 100  # m to cm
            n_plates = inputs['plates_per_assembly']
            clad_width = inputs['clad_structure_width'] * 100  # m to cm
            assembly_width = n_plates * plate_pitch + 2 * clad_width
            half_width = assembly_width / 2

            # Create bounding surfaces with specified BC
            left = openmc.XPlane(x0=-half_width, boundary_type=boundary_type)
            right = openmc.XPlane(x0=half_width, boundary_type=boundary_type)
            bottom = openmc.YPlane(y0=-half_width, boundary_type=boundary_type)
            top = openmc.YPlane(y0=half_width, boundary_type=boundary_type)

            # Create bottom and top axial surfaces
            fuel_height = inputs['fuel_height'] * 100  # m to cm
            bottom_z = openmc.ZPlane(z0=-fuel_height/2, boundary_type=boundary_type)
            top_z = openmc.ZPlane(z0=fuel_height/2, boundary_type=boundary_type)

            # Create cell with bounds
            main_cell = openmc.Cell(fill=root_universe)
            main_cell.region = +left & -right & +bottom & -top & +bottom_z & -top_z
            root_universe = openmc.Universe(cells=[main_cell])
    elif depletion_type in ['element', 'element_enhanced']:
        is_enhanced = (depletion_type == 'element_enhanced')
        element_type = 'pin' if inputs['assembly_type'] == 'Pin' else 'plate'
        print(f"\nCreating single {'enhanced ' if is_enhanced else ''}{element_type} model for {calc_type} calculation...")

        if inputs['assembly_type'] == 'Pin':
            from Reactor.geometry_helpers.pin_fuel import build_pin_cell_fuel_uni
            root_universe = build_pin_cell_fuel_uni(mat_dict, is_enhanced=is_enhanced)

            # Calculate pin bounds
            pin_pitch = inputs['pin_pitch'] * 100  # m to cm
            half_width = pin_pitch / 2

        else:  # Plate
            from Reactor.geometry_helpers.plate_fuel import build_plate_cell_fuel_uni
            root_universe = build_plate_cell_fuel_uni(mat_dict, is_enhanced=is_enhanced)

            # Calculate plate bounds
            plate_pitch = inputs['fuel_plate_pitch'] * 100  # m to cm
            half_width = plate_pitch / 2

        # Create bounding surfaces with specified BC
        left = openmc.XPlane(x0=-half_width, boundary_type=boundary_type)
        right = openmc.XPlane(x0=half_width, boundary_type=boundary_type)
        bottom = openmc.YPlane(y0=-half_width, boundary_type=boundary_type)
        top = openmc.YPlane(y0=half_width, boundary_type=boundary_type)

        # Create bottom and top axial surfaces
        fuel_height = inputs['fuel_height'] * 100  # m to cm
        bottom_z = openmc.ZPlane(z0=-fuel_height/2, boundary_type=boundary_type)
        top_z = openmc.ZPlane(z0=fuel_height/2, boundary_type=boundary_type)

        # Create cell with bounds
        main_cell = openmc.Cell(fill=root_universe)
        main_cell.region = +left & -right & +bottom & -top & +bottom_z & -top_z
        root_universe = openmc.Universe(cells=[main_cell])
    else:
        raise ValueError(f"Invalid depletion type: {depletion_type}. Must be 'core', 'assembly', 'assembly_enhanced', 'element', or 'element_enhanced'")

    geometry = openmc.Geometry(root_universe)

    # Create settings
    settings = openmc.Settings()

    # Set particles and batches from inputs
    settings.particles = inputs.get('depletion_particles', 10000)
    settings.batches = inputs.get('depletion_batches', 100)
    settings.inactive = inputs.get('depletion_inactive', 20)

    # Set source distribution
    bounds = geometry.bounding_box
    lower_left = bounds[0]
    upper_right = bounds[1]

    # Create uniform spatial distribution over core using IndependentSource
    source = openmc.IndependentSource()
    source.space = openmc.stats.Box(
        lower_left=lower_left,
        upper_right=upper_right,
        only_fissionable=True  # Only sample points in fissionable regions
    )
    settings.source = source

    # Set other settings
    settings.output = {'tallies': False}

    # Point to cross sections
    cross_sections = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'cross_sections', 'cross_sections.xml'
    )
    os.environ['OPENMC_CROSS_SECTIONS'] = cross_sections

    # Create model and set components
    model = openmc.Model()
    model.materials = materials  # Set materials first
    model.geometry = geometry   # Then geometry
    settings.temperature = {'method': 'interpolation', 'tolerance': 100}
    settings.output = {'tallies': False}
    model.settings = settings   # Then settings
    model.settings.output = {'tallies': False}

    # Determine paths after all components are set
    model.geometry.determine_paths(instances_only=False)

    return model

def run_all_depletions(output_dir=None):
    """Run all enabled depletion calculations.

    Parameters
    ----------
    output_dir : str, optional
        Base directory to write output files. If None, uses ./depletion/outputs/

    Returns
    -------
    dict
        Dictionary containing results for each enabled depletion type
    """
    results = {}

    # Set up base output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')

    # Clean up old outputs if they exist
    if os.path.exists(output_dir):
        print("\nCleaning up old depletion output files...")
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Run core depletion if enabled
    if inputs['deplete_core']:
        print("\n" + "="*80)
        print("Starting full core depletion calculation")
        print("="*80)
        results['core'] = run_depletion(output_dir=output_dir, depletion_type='core')

    # Run regular assembly depletion if enabled
    if inputs['deplete_assembly']:
        print("\n" + "="*80)
        print("Starting fuel assembly k-infinity depletion calculation")
        print("="*80)
        results['assembly'] = run_depletion(output_dir=output_dir, depletion_type='assembly')

    # Run enhanced assembly depletion if enabled
    if inputs['deplete_assembly_enhanced']:
        print("\n" + "="*80)
        print("Starting enhanced fuel assembly k-infinity depletion calculation")
        print("="*80)
        results['assembly_enhanced'] = run_depletion(output_dir=output_dir, depletion_type='assembly_enhanced')

    # Run single element depletion if enabled
    if inputs['deplete_element']:
        element_type = 'pin' if inputs['assembly_type'] == 'Pin' else 'plate'
        print("\n" + "="*80)
        print(f"Starting single {element_type} k-infinity depletion calculation")
        print("="*80)
        results['element'] = run_depletion(output_dir=output_dir, depletion_type='element')

    # Run enhanced single element depletion if enabled
    if inputs['deplete_element_enhanced']:
        element_type = 'pin' if inputs['assembly_type'] == 'Pin' else 'plate'
        print("\n" + "="*80)
        print(f"Starting enhanced single {element_type} k-infinity depletion calculation")
        print("="*80)
        results['element_enhanced'] = run_depletion(output_dir=output_dir, depletion_type='element_enhanced')

    return results

def run_depletion(model=None, output_dir=None, depletion_type='core'):
    """Run a depletion calculation.

    Parameters
    ----------
    model : openmc.Model, optional
        OpenMC model instance. If not provided, creates a new one
    output_dir : str, optional
        Directory to write output files. If None, uses ./depletion/outputs/
    depletion_type : str
        Type of depletion to run: 'core', 'assembly', 'assembly_enhanced',
        'element', or 'element_enhanced'

    Returns
    -------
    openmc.deplete.Results
        The depletion calculation results
    """
    # Set default output directory
    base_output_dir = output_dir or os.path.join(os.path.dirname(__file__), 'outputs')
    calc_type = 'keff' if depletion_type == 'core' else 'k∞'
    output_dir = os.path.join(base_output_dir, f"{depletion_type}_{calc_type}")

    # Clean up old outputs if they exist
    if os.path.exists(output_dir):
        print(f"\nCleaning up old {depletion_type} {calc_type} output files...")
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create new model if not provided
    if model is None:
        print(f"\nCreating new OpenMC model for {depletion_type} {calc_type} depletion...")
        model = create_model(depletion_type)

    # Update transport settings from inputs
    for setting in ['particles', 'batches', 'inactive']:
        input_key = f'depletion_{setting}'
        if input_key in inputs:
            setattr(model.settings, setting, inputs[input_key])

    # Print transport settings
    print(f"\n{depletion_type.capitalize()} {calc_type} depletion transport settings:")
    print(f"- Particles per batch: {model.settings.particles}")
    print(f"- Active batches: {model.settings.batches}")
    print(f"- Inactive batches: {model.settings.inactive}")

    # Set up depletion calculation
    chain_type = inputs.get('depletion_chain', 'endfb71')  # Default to ENDF/B-VII.1 chain
    chain_file = os.path.join(
        os.path.dirname(__file__),
        "depletion chains",
        f"chain_{chain_type}_pwr.xml"
    )

    # Configure output paths
    model.settings.output = {'path': output_dir}
    model.export_to_xml(directory=output_dir)

    # Create operator and set up integrator
    dep_operator = depletion_operator.create_operator(model=model, chain_file=chain_file, depletion_type=depletion_type)
    integrator, timesteps = depletion_operator.setup_depletion(dep_operator, depletion_type=depletion_type)

    # Configure integrator output paths
    integrator.output_dir = os.path.abspath(output_dir)
    integrator.operator.output_dir = os.path.abspath(output_dir)

    # Write simulation parameters to text file
    params_file = os.path.join(output_dir, "simulation_parameters.txt")
    with open(params_file, 'w') as f:
        f.write(f"Depletion Parameters for {depletion_type} calculation\n")
        f.write("="*50 + "\n\n")

        f.write("Heavy Metal Mass:\n")
        f.write(f"- Total HM mass: {dep_operator.heavy_metal:.2f} g ({dep_operator.heavy_metal/1000:.2f} kg)\n\n")

        f.write("Power Settings:\n")
        f.write(f"- Core power: {inputs['core_power']:.2f} MW ({inputs['core_power']*1e6:.2f} W)\n")
        f.write(f"- Power density: {integrator.power_density:.2f} W/gHM\n")
        total_power = integrator.power_density * dep_operator.heavy_metal
        f.write(f"- Total power being used: {total_power/1e6:.3f} MW\n\n")

        f.write("Time Steps:\n")
        f.write(f"- Number of steps: {len(timesteps)}\n")
        f.write(f"- Units: {inputs['depletion_timestep_units']}\n")
        f.write(f"- Step configurations:\n")
        for i, config in enumerate(inputs['depletion_timesteps'], 1):
            f.write(f"  {i}. {config['steps']} steps of {config['size']} {inputs['depletion_timestep_units']}\n")

    # Print calculation parameters
    print(f"\nRunning {depletion_type} {calc_type} depletion calculation:")
    print(f"- Power density: {integrator.power_density:.3f} W/gHM")
    print(f"- Time steps: {len(timesteps)}")
    print(f"- Total time: {sum(timesteps)/(24*60*60):.1f} days")
    print(f"- Step size: {timesteps[0]/(24*60*60):.1f} days\n")

    # Run the calculation
    integrator.integrate()

    # Store heavy metal mass in the results file (in grams)
    results_file = os.path.join(output_dir, "depletion_results.h5")
    heavy_metal_mass = dep_operator.heavy_metal  # Already in grams
    with h5py.File(results_file, 'a') as f:
        if 'heavy_metal' in f:
            del f['heavy_metal']
        f.create_dataset('heavy_metal', data=heavy_metal_mass)
        print(f"\nStored heavy metal mass for {depletion_type}: {heavy_metal_mass:.2f} g ({heavy_metal_mass/1000:.2f} kg)")

    # Process results
    results = openmc.deplete.Results(results_file)

    # Append k-effective results to simulation parameters file
    with open(params_file, 'a') as f:
        f.write("\nDepletion Results:\n")
        f.write("="*50 + "\n")
        f.write("Time (days)  Burnup (MWd/kgHM)    k-eff ± std dev\n")
        f.write("-"*50 + "\n")

        # Get time, k-eff and burnup data
        time_days = results.get_times() / (24*60*60)  # Convert seconds to days
        k_eff = results.get_eigenvalue()
        burnup = results.get_times() * integrator.power_density / (24*60*60*1000)  # Convert to MWd/kgHM

        # Write results in formatted table
        for t, b, k in zip(time_days, burnup, k_eff):
            f.write(f"{t:10.2f} {b:16.2f}    {k[0]:.5f} ± {k[1]:.5f}\n")

    return results

if __name__ == "__main__":
    run_all_depletions()
