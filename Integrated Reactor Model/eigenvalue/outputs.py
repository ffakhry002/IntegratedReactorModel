"""
Functions for processing and saving OpenMC calculation results.
"""

import os
import numpy as np
import openmc
from inputs import inputs
from eigenvalue.tallies.energy_groups import get_energy_bins, get_group_indices
from eigenvalue.tallies.normalization import calc_norm_factor
import sys
from plotting.functions.utils import get_tally_volume

# Add root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

def collapse_to_three_groups(mean, std_dev, inputs_dict):
    """Collapse energy group fluxes into three groups (thermal, epithermal, fast).

    Parameters
    ----------
    mean : numpy.ndarray
        Mean values from the energy group tally
    std_dev : numpy.ndarray
        Standard deviations from the energy group tally
    inputs_dict : dict
        Custom inputs dictionary.

    Returns
    -------
    tuple
        (means, std_devs) for the three collapsed groups
    """
    # Get indices for group boundaries from inputs
    thermal_inds = get_group_indices(inputs_dict['thermal_cutoff'], inputs_dict)
    fast_inds = get_group_indices(inputs_dict['fast_cutoff'], inputs_dict)

    # Create masks for each group
    energy_bins = get_energy_bins(inputs_dict)
    n_bins = len(energy_bins) - 1  # Number of energy bins is one less than boundaries

    # Filter out any indices that would be out of bounds
    thermal_inds = thermal_inds[thermal_inds < n_bins]
    fast_inds = fast_inds[fast_inds < n_bins]

    # Create the masks
    thermal_mask = np.zeros(n_bins, dtype=bool)
    epithermal_mask = np.zeros(n_bins, dtype=bool)
    fast_mask = np.zeros(n_bins, dtype=bool)

    # Since get_group_indices returns indices where E > boundary:
    # First mark all indices where E > thermal cutoff
    thermal_mask[thermal_inds] = True
    # Then invert to get E < thermal cutoff
    thermal_mask = ~thermal_mask

    # Fast: E > 100 keV (indices where E > 100 keV ARE fast)
    fast_mask[fast_inds] = True

    # Epithermal: everything else
    epithermal_mask = ~thermal_mask & ~fast_mask

    # Sum fluxes in each group
    means = np.array([
        np.sum(mean[thermal_mask]),
        np.sum(mean[epithermal_mask]),
        np.sum(mean[fast_mask])
    ])

    # Propagate uncertainties (quadrature sum)
    std_devs = np.array([
        np.sqrt(np.sum(std_dev[thermal_mask]**2)),
        np.sqrt(np.sum(std_dev[epithermal_mask]**2)),
        np.sqrt(np.sum(std_dev[fast_mask]**2))
    ])

    return means, std_devs

def process_results(sp, k_effective, inputs_dict=None):
    """Process and save all results from the OpenMC calculation.

    Parameters
    ----------
    sp : openmc.StatePoint
        StatePoint file containing the tally results
    k_effective : openmc.keff
        k-effective result from the simulation
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    None

    Notes
    -----
    Results are written to a results.txt file in the appropriate output directory.
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    # Determine output directory based on how we're running
    running_directly = os.path.basename(sys.argv[0]) == 'run.py'
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check if we're in a parametric study run directory
    current_dir = os.getcwd()
    if 'parametric_simulation_' in current_dir and 'run_' in current_dir:
        # We're in a parametric study run directory - write results.txt directly to run directory
        output_dir = current_dir
    elif not running_directly and os.path.exists(os.path.join(root_dir, 'simulation_data')):
        # Running from main.py - write results.txt to simulation_data root
        output_dir = os.path.join(root_dir, 'simulation_data')
    else:
        # Running directly - write to execution/Output
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Output')

    # Write results to the same directory as the statepoint file
    output_file = os.path.join(output_dir, 'results.txt')

    # Get power from inputs
    power_mw = inputs_dict.get('core_power', 1.0)

    # Calculate normalization factor using tallied values
    norm_factor = float(calc_norm_factor(power_mw, sp))

    # Calculate nu from tallies
    nu_fission_tally = sp.get_tally(name='nu-fission')
    fission_tally = sp.get_tally(name='fission')
    nu = nu_fission_tally.mean.flatten()[0] / fission_tally.mean.flatten()[0]

    with open(output_file, 'w') as f:
        # Write k-effective results
        f.write("CALCULATION RESULTS\n")
        f.write("===================\n\n")
        f.write("K-effective Results:\n")
        f.write("-----------------\n")
        f.write(f"K-effective: {k_effective.nominal_value:.6f} ± {k_effective.std_dev:.6f}\n\n")

        # Write fixed values used
        f.write("Fixed Parameters Used:\n")
        f.write("-------------------\n")
        f.write("Energy per fission: 200.0 MeV\n")
        f.write(f"Neutrons per fission (from tallies): {nu:.3f}\n")
        f.write(f"Normalization factor: {norm_factor:.3e} n/s/cm³\n")

        # Write irradiation position results
        f.write("Irradiation Position Results:\n")
        f.write("--------------------------\n")
        f.write(f"Results normalized to {power_mw:.1f} MW core power\n\n")

        # Process each irradiation position tally
        for tally in sp.tallies.values():
            if tally.name.startswith('I_'):
                # Skip axial tallies - they don't need energy group collapsing
                if tally.name.endswith('_axial'):
                    continue

                # Get volume for normalization
                volume = get_tally_volume(tally, sp, inputs_dict)

                # Get the flux data and normalize
                mean = tally.mean.ravel() * norm_factor / volume
                std_dev = tally.std_dev.ravel() * norm_factor / volume

                # Calculate three-group fluxes
                means, std_devs = collapse_to_three_groups(mean, std_dev, inputs_dict)

                # Calculate total flux
                total_flux = np.sum(means)
                total_std_dev = np.sqrt(np.sum(std_devs**2))

                # Write results
                f.write(f"Position {tally.name}:\n")
                f.write(f"Total flux: {total_flux:.5e} ± {total_std_dev:.5e} n/cm²/s\n")
                group_names = ["Thermal   ", "Epithermal", "Fast      "]
                for name, m, s in zip(group_names, means, std_devs):
                    f.write(f"{name}  {m:10.5e}  {s:10.5e}\n")
                f.write("\n")

    print(f"\nResults have been written to: {output_file}")
