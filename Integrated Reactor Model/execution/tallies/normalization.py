"""
Calculate normalization factors for tallies.
"""

import numpy as np
from inputs import inputs

def calc_norm_factor(power_mw, sp):
    """Calculate the normalization factor based on reactor parameters.

    This converts OpenMC's per-source-particle values to absolute values (n/s)
    based on the actual reactor power and fixed values.

    Parameters
    ----------
    power_mw : float
        Reactor power in MW
    sp : openmc.StatePoint
        StatePoint file containing the tally results

    Returns
    -------
    float
        Normalization factor (neutrons/second)

    Notes
    -----
    Uses the equation:
    C = P * 6.2415e18 / (kappa * keff)
    where:
    - P is reactor power [MW]
    - 6.2415e18 is MeV/MW-s conversion
    - kappa is energy per fission [MeV] (fixed at 200 MeV)
    - keff is from the simulation
    - nu is calculated from tallies as (nu-fission)/(fission)
    """
    # Use fixed value for energy per fission
    kappa_fission = 200.0  # MeV per fission
    keff = sp.keff.nominal_value

    # Calculate nu from tallies
    nu_fission_tally = sp.get_tally(name='nu-fission')
    fission_tally = sp.get_tally(name='fission')

    nu = (nu_fission_tally.mean.flatten()[0] /
          fission_tally.mean.flatten()[0])

    # Calculate normalization factor
    C = power_mw * 6.2415e18 * nu / (kappa_fission * keff)

    return float(C)
