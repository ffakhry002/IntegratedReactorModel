"""
Functions for writing depletion calculation output text files.
"""

def process_depletion_results(time_seconds, k_eff, power_density):
    """Process depletion calculation results to get derived quantities.

    Parameters
    ----------
    time_seconds : numpy.ndarray
        Array of time points in seconds
    k_eff : numpy.ndarray
        Array of k-effective values and uncertainties
    power_density : float
        Power density in W/gHM

    Returns
    -------
    dict
        Dictionary containing processed results including time_days, burnup, k_eff,
        k_means, pcm_per_burnup, pcm_per_EFPD, and k=1 crossing information
    """
    # Convert time to days and calculate burnup
    time_days = time_seconds / (24 * 60 * 60)
    burnup = time_seconds * power_density / (24*60*60*1000)  # Convert to MWd/kgHM
    k_means = [k[0] for k in k_eff]

    results = {
        'time_days': time_days,
        'burnup': burnup,
        'k_eff': k_eff,
        'k_means': k_means
    }

    # Get last two points for all calculations
    k1, k2 = k_means[-2], k_means[-1]  # Last two k values
    b1, b2 = burnup[-2], burnup[-1]    # Last two burnup values
    t1, t2 = time_days[-2], time_days[-1]  # Last two time values

    drho = (k2 - k1)*1e5
    dburnup = b2 - b1

    # Calculate coefficients using last two points
    results['pcm_per_burnup'] = drho / dburnup
    days_per_burnup = (t2 - t1) / (b2-b1)
    results['pcm_per_EFPD'] = results['pcm_per_burnup'] / days_per_burnup

    # Handle k=1 crossing or extrapolation
    k_start = k_means[0]
    k_end = k_means[-1]

    # Case 1: Crosses k=1 during depletion
    if (k_start > 1 and k_end < 1) or (k_start < 1 and k_end > 1):
        for i in range(len(k_means)-1):
            if (k_means[i] > 1.0 and k_means[i+1] < 1.0) or (k_means[i] < 1.0 and k_means[i+1] > 1.0):
                k1, k2 = k_means[i], k_means[i+1]
                b1, b2 = burnup[i], burnup[i+1]
                t1, t2 = time_days[i], time_days[i+1]

                frac = (1.0 - k1) / (k2 - k1)
                results['keff_1_burnup'] = b1 + frac * (b2 - b1)
                results['keff_1_time'] = t1 + frac * (t2 - t1)
                results['keff_1_method'] = 'interpolation'
                break

    # Case 2: All points above k=1, extrapolate down
    elif k_start > 1 and k_end > 1:
        # Use last two points to extrapolate
        slope = (k2 - k1) / (b2 - b1)
        b_intercept = k1 - slope * b1
        burnup_at_k1 = (1 - b_intercept) / slope

        # Also get time at k=1
        time_slope = (t2 - t1) / (b2 - b1)
        time_at_k1 = t1 + (burnup_at_k1 - b1) * time_slope

        results['keff_1_burnup'] = burnup_at_k1
        results['keff_1_time'] = time_at_k1
        results['keff_1_method'] = 'extrapolation_down'

    # Case 3: All points below k=1, don't extrapolate up
    elif k_start < 1 and k_end < 1:
        results['keff_1_burnup'] = None
        results['keff_1_time'] = None
        results['keff_1_method'] = 'never_crosses_k1'

    return results

def write_output(params_file, depletion_type, dep_operator, integrator, timesteps, inputs_dict, results_data):
    """Write all depletion output to a single file.

    Parameters
    ----------
    params_file : str
        Path to the output parameter file
    depletion_type : str
        Type of depletion calculation being performed
    dep_operator : openmc.deplete.Operator
        The depletion operator instance
    integrator : openmc.deplete.Integrator
        The depletion integrator instance
    timesteps : list
        List of timesteps for the calculation
    inputs_dict : dict
        Dictionary of input parameters
    results_data : dict
        Dictionary of processed results data
    """
    with open(params_file, 'w') as f:
        # Write initial parameters
        f.write(f"Depletion Parameters for {depletion_type} calculation\n")
        f.write("="*50 + "\n\n")

        f.write("Heavy Metal Mass:\n")
        f.write(f"- Total HM mass: {dep_operator.heavy_metal:.2f} g ({dep_operator.heavy_metal/1000:.2f} kg)\n\n")

        f.write("Power Settings:\n")
        f.write(f"- Core power: {inputs_dict['core_power']:.2f} MW ({inputs_dict['core_power']*1e6:.2f} W)\n")
        f.write(f"- Power density: {integrator.power_density:.2f} W/gHM\n")
        total_power = integrator.power_density * dep_operator.heavy_metal
        f.write(f"- Total power being used: {total_power/1e6:.3f} MW\n\n")

        f.write("Time Steps:\n")
        f.write(f"- Number of steps: {len(timesteps)}\n")
        f.write(f"- Units: {inputs_dict['depletion_timestep_units']}\n")
        f.write(f"- Step configurations:\n")
        for i, config in enumerate(inputs_dict['depletion_timesteps'], 1):
            f.write(f"  {i}. {config['steps']} steps of {config['size']} {inputs_dict['depletion_timestep_units']}\n")

        # Write depletion results
        f.write("\nDepletion Results:\n")
        f.write("="*50 + "\n")

        # Write keff=1 crossing point if found
        if 'keff_1_burnup' in results_data:
            method = results_data['keff_1_method']
            if method == 'never_crosses_k1':
                f.write("k=1 crossing point: Never crosses k=1\n\n")
            else:
                if method == 'interpolation':
                    f.write(f"k=1 crossing point (interpolated):\n")
                elif method == 'extrapolation_down':
                    f.write(f"k=1 point (extrapolated from above):\n")
                else:
                    f.write(f"k=1 point (never crossed)):\n")

                if results_data['keff_1_burnup'] is not None:
                    f.write(f"- Burnup: {results_data['keff_1_burnup']:.2f} MWd/kgHM\n")
                    f.write(f"- Time: {results_data['keff_1_time']:.2f} days\n\n")

        # Write reactivity coefficients
        if 'pcm_per_burnup' in results_data:
            f.write("Reactivity Coefficients (using last two points):\n")
            f.write(f"- {results_data['pcm_per_burnup']:.2f} pcm/(MWd/kgHM)\n")
            f.write(f"- {results_data['pcm_per_EFPD']:.2f} pcm/EFPD\n\n")

        # Write k-eff table
        f.write("Time (days)  Burnup (MWd/kgHM)    k-eff ± std dev\n")
        f.write("-"*50 + "\n")
        for t, b, k in zip(results_data['time_days'], results_data['burnup'], results_data['k_eff']):
            k_mean, k_std = k[0], k[1]
            f.write(f"{t:10.2f} {b:16.2f}    {k_mean:.5f} ± {k_std:.5f}\n")
