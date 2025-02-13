"""
Functions for handling depletion calculations with OpenMC transport.
"""

import os
import warnings
import numpy as np
import openmc
import openmc.deplete
import sys

# Add parent directory to path to find inputs.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inputs import inputs

def calculate_volumes(geometry):
    """Calculate volumes for pin cell fuel."""
    # Get all materials used in geometry
    used_materials = geometry.get_all_materials()

    # Calculate fuel pin volume
    fuel_height = inputs['fuel_height'] * 100  # m to cm
    fuel_radius = inputs['r_fuel'] * 100       # m to cm
    fuel_volume = np.pi * fuel_radius**2 * fuel_height

    # Set volume for depletable materials
    for material in used_materials.values():
        if hasattr(material, 'depletable') and material.depletable:
            material.volume = fuel_volume
            print(f"Set volume for material {material.name}: {fuel_volume:.2f} cm³")

def create_operator(model):
    """Create a depletion operator for pin cell calculations."""
    # Copy chain file to output directory
    output_dir = model.settings.output['path']
    chain_file = os.path.join(output_dir, "chain_casl_pwr.xml")

    # Copy chain file if it doesn't exist in output directory
    if not os.path.exists(chain_file):
        import shutil
        src_chain = os.path.join(os.path.dirname(__file__), "chain_casl_pwr.xml")
        shutil.copy2(src_chain, chain_file)

    # Calculate volumes for depletable materials
    calculate_volumes(model.geometry)

    # Create operator
    operator = openmc.deplete.CoupledOperator(
        model=model,
        chain_file=chain_file,
        diff_burnable_mats=False,
        normalization_mode="fission-q"
    )

    return operator

def setup_depletion(operator):
    """Set up depletion calculation parameters."""
    # Get timestep information from inputs
    timestep_units = inputs['depletion_timestep_units']
    timestep_configs = inputs['depletion_timesteps']

    # Generate timesteps
    timesteps = []
    for config in timestep_configs:
        timesteps.extend([config['size']] * config['steps'])

    # Convert units for integrator
    if timestep_units == 'days':
        timestep_units = 'd'
    elif timestep_units == 'MWd/kgHM':
        timestep_units = 'MWd/kg'
    else:
        raise ValueError(f"Unknown timestep units: {timestep_units}")

    # Get power density from inputs (W/gHM)
    power_density = inputs['power_density']

    # Create integrator
    integrator = openmc.deplete.CECMIntegrator(
        operator=operator,
        timesteps=timesteps,
        power_density=power_density,
        timestep_units=timestep_units
    )
    integrator.power_density = power_density
    return integrator, timesteps
