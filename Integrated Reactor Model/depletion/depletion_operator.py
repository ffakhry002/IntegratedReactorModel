"""
Functions for handling depletion calculations with OpenMC transport.
"""

import os
import warnings
import numpy as np
import openmc
import openmc.deplete
import sys
import tempfile
import shutil
import h5py

# Add parent directory to path to find inputs.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inputs import inputs

current_dir = os.path.dirname(os.path.abspath(__file__))  # Current directory (depletion)
parent_dir = os.path.dirname(current_dir)  # Up one level to "Integrated Reactor Model"
root_dir = os.path.dirname(parent_dir)  # Up another level to IntegratedReactorModel

# List of possible cross_sections.xml locations
cross_sections_path = os.path.join(root_dir, "cross_sections", "cross_sections.xml")


def calculate_volumes(geometry, inputs_dict):
    """Calculate volumes for all depletable materials considering full core geometry."""
    debug_file = os.path.join(os.path.dirname(__file__), 'outputs', 'volume_debug.txt')
    os.makedirs(os.path.dirname(debug_file), exist_ok=True)

    with open(debug_file, 'w') as f:
        f.write("VOLUME CALCULATION DEBUG LOG\n")
        f.write("===========================\n\n")

        depletable_cells = []
        used_materials = geometry.get_all_materials()
        material_volumes = {}  # Track volume for each material ID

        def process_universe(universe, level=0):
            indent = "  " * level
            f.write(f"{indent}Processing universe {universe.id}\n")
            for cell in universe.cells.values():
                if isinstance(cell.fill, openmc.RectLattice):
                    f.write(f"{indent}Found lattice in cell {cell.id}\n")
                    f.write(f"{indent}Lattice shape: {cell.fill.universes.shape}\n")
                    for sub_universe in np.ravel(cell.fill.universes):
                        process_universe(sub_universe, level + 1)
                elif isinstance(cell.fill, openmc.Universe):
                    f.write(f"{indent}Found nested universe in cell {cell.id}\n")
                    process_universe(cell.fill, level + 1)
                elif cell.fill is not None and cell.fill in used_materials.values():
                    if hasattr(cell.fill, 'depletable') and cell.fill.depletable:
                        depletable_cells.append(cell)
                        mat_id = cell.fill.id
                        if mat_id not in material_volumes:
                            material_volumes[mat_id] = 0
                        f.write(f"{indent}Found depletable cell {cell.id} with material {cell.fill.name}\n")

        f.write("Starting volume calculation...\n")
        f.write("Processing root universe...\n")
        process_universe(geometry.root_universe)

        # Calculate single cell volume
        fuel_height = inputs_dict['fuel_height'] * 100
        if inputs_dict['assembly_type'] == 'Plate':
            fuel_meat_width = inputs_dict['fuel_meat_width'] * 100
            fuel_meat_thickness = inputs_dict['fuel_meat_thickness'] * 100
            single_volume = fuel_meat_width * fuel_meat_thickness * fuel_height
        else:
            fuel_radius = inputs_dict['r_fuel'] * 100
            single_volume = np.pi * fuel_radius**2 * fuel_height

        # Count how many cells use each material
        material_cell_counts = {}
        for cell in depletable_cells:
            mat_id = cell.fill.id
            material_cell_counts[mat_id] = material_cell_counts.get(mat_id, 0) + 1

        # Calculate and set correct volume for each material
        for material in used_materials.values():
            if hasattr(material, 'depletable') and material.depletable:
                if material.id in material_cell_counts:
                    # Each material gets volume * number of cells using that material
                    material.volume = single_volume * material_cell_counts[material.id]
                    f.write(f"\nMaterial {material.id} ({material.name}):\n")
                    f.write(f"  Cells using this material: {material_cell_counts[material.id]}\n")
                    f.write(f"  Total volume: {material.volume:.4f} cm³\n")

def create_operator(model, chain_file=None, depletion_type='core', inputs_dict=None):
    """Create a depletion operator for full core calculations.

    Parameters
    ----------
    model : openmc.Model
        OpenMC model instance containing geometry, settings, and materials
    chain_file : str, optional
        Path to depletion chain XML file. If None, uses the chain specified in inputs.
    depletion_type : str, optional
        Type of depletion: 'core', 'assembly', 'assembly_enhanced', 'element', or 'element_enhanced'
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    openmc.deplete.CoupledOperator
        The depletion operator
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    # Select chain file based on inputs if not explicitly provided
    if chain_file is None:
        chain_type = inputs_dict.get('depletion_chain', 'endfb71').lower()
        if chain_type == 'endfb71':
            chain_file = os.path.join(
                os.path.dirname(__file__),
                "depletion chains/chain_endfb71_pwr.xml"
            )
        elif chain_type == 'casl':
            chain_file = os.path.join(
                os.path.dirname(__file__),
                "depletion chains/chain_casl_pwr.xml"
            )
        else:
            raise ValueError(f"Unknown chain type '{chain_type}'. Must be 'endfb71' or 'casl'")

    os.environ['OPENMC_CROSS_SECTIONS'] = cross_sections_path

    # Ensure paths are determined
    model.geometry.determine_paths(instances_only=False)

    # Get list of materials actually used in the geometry
    used_materials = model.geometry.get_all_materials()

    # Only mark materials as depletable if they are both marked as depletable AND used in the geometry
    for mat in model.materials:
        if mat.depletable and mat not in used_materials.values():
            print(f"Setting material {mat.id} ({mat.name}) as non-depletable since it is not used in geometry")
            mat.depletable = False

    # Calculate volumes for depletable materials
    calculate_volumes(model.geometry, inputs_dict)

    # Create operator with recommended settings for full core
    operator = openmc.deplete.CoupledOperator(
        model=model,
        chain_file=chain_file,
        diff_burnable_mats=False,
        normalization_mode="fission-q"
    )

    # Verify volumes are set for burnable materials
    for material in model.geometry.get_all_materials().values():
        if material.depletable and not material.volume:
            warnings.warn(f"Material {material.id} is depletable but has no volume specified")

    # Modify the operator's __call__ method to handle the "No fission sites banked" error
    original_call = operator.__call__

    def wrapped_call(*args, **kwargs):
        try:
            return original_call(*args, **kwargs)
        except Exception as e:
            if "No fission sites banked" in str(e):
                print("\nWarning: No fission sites banked in this step - continuing to next step")
                # Return some default values that will allow the calculation to continue
                # These are placeholder values that indicate the step failed
                return {
                    'k-effective': (0.01, 0.0),  # k-eff of 0.01 with 0 uncertainty
                    'tallies': {}  # Empty tallies dictionary
                }
            else:
                # Re-raise any other exceptions
                raise

    # Replace the operator's __call__ method with our wrapped version
    operator.__call__ = wrapped_call

    return operator

def setup_depletion(operator, depletion_type='core', inputs_dict=None):
    """Set up depletion calculation parameters.

    Parameters
    ----------
    operator : openmc.deplete.CoupledOperator
        The depletion operator
    depletion_type : str, optional
        Type of depletion: 'core', 'assembly', 'assembly_enhanced', 'element', or 'element_enhanced'
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    tuple
        (integrator, timesteps in seconds or MWd/kgHM)
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    # Get parameters from inputs - use direct access since these are required
    timestep_units = inputs_dict['depletion_timestep_units']
    timestep_configs = inputs_dict['depletion_timesteps']

    # Generate timesteps based on configurations
    timesteps = []
    for config in timestep_configs:
        steps = config['steps']  # Direct access since these are required
        size = config['size']
        timesteps.extend([size] * steps)

    if not timesteps:
        raise ValueError("No timesteps specified in inputs")

    # Convert units for integrator
    if timestep_units == 'days':
        # Keep timesteps in days - OpenMC expects days directly when using 'd' units
        timesteps = timesteps  # No conversion needed
        timestep_units = 'd'  # OpenMC expects 'd' for days
    elif timestep_units == 'MWd/kgHM':
        # Keep as is - OpenMC expects MWd/kgHM directly
        timestep_units = 'MWd/kg'  # OpenMC expects 'MWd/kg' for burnup
    else:
        raise ValueError(f"Unknown timestep units: {timestep_units}")

    # Print timestep information
    print("\nDepletion timestep information:")
    print(f"- Units: {timestep_units}")
    print(f"- Total number of steps: {len(timesteps)}")
    print("- Step configurations:")
    for i, config in enumerate(timestep_configs, 1):
        print(f"  Config {i}: {config['steps']} steps of {config['size']} {inputs_dict['depletion_timestep_units']}")

    # Select integrator based on input
    integrator_type = inputs_dict['depletion_integrator'].lower()

    # Map of integrator names to their classes
    integrator_map = {
        'predictor': openmc.deplete.PredictorIntegrator,
        'cecm': openmc.deplete.CECMIntegrator,
        'celi': openmc.deplete.CELIIntegrator,
        'cf4': openmc.deplete.CF4Integrator,
        'epcrk4': openmc.deplete.EPCRK4Integrator,
        'leqi': openmc.deplete.LEQIIntegrator,
        'siceli': openmc.deplete.SICELIIntegrator,
        'sileqi': openmc.deplete.SILEQIIntegrator
    }

    # Get the integrator class
    if integrator_type not in integrator_map:
        raise ValueError(f"Unknown integrator type '{integrator_type}'. "
                       f"Must be one of: {', '.join(integrator_map.keys())}")

    integrator_class = integrator_map[integrator_type]

    # Calculate power density in W/gHM based on core values
    core_power = inputs_dict['core_power'] * 1e6  # MW to W

    # For assembly and element cases, we want to use the same power density as the core
    if depletion_type == 'core':
        power_density = core_power / operator.heavy_metal  # W/gHM
    else:
        # Get number of assemblies from inputs
        num_assemblies = inputs_dict['num_assemblies']
        if depletion_type.startswith('element'):
            # For pin fuel, divide by number of pins per assembly
            if inputs_dict['assembly_type'] == 'Pin':
                num_elements = inputs_dict['n_side_pins'] ** 2 - len(inputs_dict['guide_tube_positions'])
            else:  # Plate fuel
                num_elements = inputs_dict['plates_per_assembly']
            # Scale by both number of assemblies and elements per assembly
            total_elements = num_assemblies * num_elements
            power_density = (core_power / total_elements) / operator.heavy_metal
        else:  # assembly case
            # Scale by number of assemblies
            power_density = (core_power / num_assemblies) / operator.heavy_metal

    # Create integrator with power density
    integrator = integrator_class(
        operator=operator,
        timesteps=timesteps,
        power_density=power_density,
        timestep_units=timestep_units
    )

    # Store power density as an attribute for later use
    integrator.power_density = power_density

    return integrator, timesteps
