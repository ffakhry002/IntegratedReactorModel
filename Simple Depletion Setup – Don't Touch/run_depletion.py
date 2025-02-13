"""
Main script for running pin cell depletion calculations.
"""

import os
import sys
import openmc
import openmc.deplete
import matplotlib.pyplot as plt

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inputs import inputs
import depletion_operator
from materials import make_materials
from pin_fuel import build_pin_cell_fuel_uni

# Set cross sections path (using absolute path)
cross_sections_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'cross_sections', 'cross_sections.xml')
openmc.config['cross_sections'] = cross_sections_path

def create_model():
    """Create a new OpenMC model for pin cell depletion calculations.

    Returns
    -------
    openmc.Model
        The OpenMC model instance
    """
    # Create materials and geometry
    mat_dict, materials = make_materials()

    # Create pin cell universe with reflective boundaries
    root_universe = build_pin_cell_fuel_uni(mat_dict)

    # Calculate pin bounds
    pin_pitch = inputs['pin_pitch'] * 100  # m to cm
    half_width = pin_pitch / 2
    fuel_height = inputs['fuel_height'] * 100  # m to cm

    # Create bounding surfaces with reflective BC
    left = openmc.XPlane(x0=-half_width, boundary_type='reflective')
    right = openmc.XPlane(x0=half_width, boundary_type='reflective')
    bottom = openmc.YPlane(y0=-half_width, boundary_type='reflective')
    top = openmc.YPlane(y0=half_width, boundary_type='reflective')
    bottom_z = openmc.ZPlane(z0=-fuel_height/2, boundary_type='reflective')
    top_z = openmc.ZPlane(z0=fuel_height/2, boundary_type='reflective')

    # Create cell with bounds
    main_cell = openmc.Cell(fill=root_universe)
    main_cell.region = +left & -right & +bottom & -top & +bottom_z & -top_z
    root_universe = openmc.Universe(cells=[main_cell])

    geometry = openmc.Geometry(root_universe)

    # Create settings
    settings = openmc.Settings()
    settings.particles = inputs.get('depletion_particles', 1000)
    settings.temperature = {'method': 'interpolation', 'tolerance': 100}
    settings.batches = inputs.get('depletion_batches', 100)
    settings.inactive = inputs.get('depletion_inactive', 20)

    # Set source distribution
    bounds = geometry.bounding_box
    source = openmc.IndependentSource()
    source.space = openmc.stats.Box(
        lower_left=bounds[0],
        upper_right=bounds[1],
        only_fissionable=True
    )
    settings.source = source
    settings.output = {'tallies': False}

    # Create model
    model = openmc.Model()
    model.materials = materials
    model.geometry = geometry
    model.settings = settings

    return model

def run_depletion(output_dir=None):
    """Run a pin cell depletion calculation.

    Parameters
    ----------
    output_dir : str, optional
        Directory to write output files. If None, uses ./outputs/

    Returns
    -------
    openmc.deplete.Results
        The depletion calculation results
    """
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')

    # Clean up old outputs if they exist
    if os.path.exists(output_dir):
        print("\nCleaning up old output files...")
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create model
    print("\nCreating OpenMC model for pin cell depletion...")
    model = create_model()

    # Configure output paths for all files
    model.settings.output = {
        'path': output_dir,
        'summary': True,
        'tallies': False
    }

    # Create operator and set up integrator
    operator = depletion_operator.create_operator(model)
    integrator, timesteps = depletion_operator.setup_depletion(operator)

    # Export all XML files to output directory
    model.export_to_xml(directory=output_dir)

    # Set paths for depletion results
    h5_path = os.path.join(output_dir, "depletion_results.h5")
    if os.path.exists(h5_path):
        os.remove(h5_path)

    # Print calculation parameters
    print(f"\nRunning pin cell depletion calculation:")
    print(f"- Power density: {inputs['power_density']:.3f} W/gHM")
    print(f"- Time steps: {len(timesteps)}")
    if inputs['depletion_timestep_units'] == 'days':
        total_time = sum(timesteps)/(24*60*60)  # Convert seconds to days
        step_size = timesteps[0]/(24*60*60)
        print(f"- Total time: {total_time:.1f} days")
        print(f"- Step size: {step_size:.1f} days")
    else:  # MWd/kgHM
        total_time = sum(timesteps)
        step_size = timesteps[0]
        print(f"- Total burnup: {total_time:.1f} MWd/kgHM")
        print(f"- Step size: {step_size:.1f} MWd/kgHM")

    # Change working directory to output directory for all file operations
    original_dir = os.getcwd()
    os.chdir(output_dir)

    try:
        # Run the calculation
        integrator.integrate()

        # Load and return results
        results = openmc.deplete.Results("depletion_results.h5")
        return results
    finally:
        # Change back to original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    run_depletion()
