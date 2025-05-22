"""
Energy group structures for tallies.
"""

import numpy as np
from inputs import inputs

# SCALE 238-group structure (in eV), from lowest to highest energy
SCALE_238 = np.flip(np.array([
    2.0000e+07, 1.7333e+07, 1.5683e+07, 1.4550e+07, 1.3840e+07, 1.2840e+07,
    1.1618e+07, 1.0000e+07, 8.1873e+06, 6.4340e+06, 4.8000e+06, 4.3040e+06,
    3.0000e+06, 2.4790e+06, 2.3540e+06, 1.8500e+06, 1.5000e+06, 1.4000e+06,
    1.3560e+06, 1.3170e+06, 1.2500e+06, 1.2000e+06, 1.1000e+06, 1.0100e+06,
    9.2000e+05, 8.7500e+05, 8.6110e+05, 8.2000e+05, 7.5000e+05, 6.7900e+05,
    6.7000e+05, 6.0000e+05, 5.7300e+05, 5.5000e+05, 4.9952e+05, 4.7000e+05,
    4.4000e+05, 4.2000e+05, 4.0000e+05, 3.3000e+05, 2.7000e+05, 2.0000e+05,
    1.5000e+05, 1.2830e+05, 1.0000e+05, 8.5000e+04, 8.2000e+04, 7.5000e+04,
    7.3000e+04, 6.0000e+04, 5.2000e+04, 5.0000e+04, 4.5000e+04, 3.0000e+04,
    2.5000e+04, 1.7000e+04, 1.3000e+04, 9.5000e+03, 8.0300e+03, 6.0000e+03,
    3.9000e+03, 3.7400e+03, 3.0000e+03, 2.5800e+03, 2.2900e+03, 2.2000e+03,
    1.8000e+03, 1.5500e+03, 1.5000e+03, 1.1500e+03, 9.5000e+02, 6.8300e+02,
    6.7000e+02, 5.5000e+02, 3.0500e+02, 2.8500e+02, 2.4000e+02, 2.1000e+02,
    2.0750e+02, 1.9250e+02, 1.8600e+02, 1.2200e+02, 1.1900e+02, 1.1500e+02,
    1.0800e+02, 1.0000e+02, 9.0000e+01, 8.2000e+01, 8.0000e+01, 7.6000e+01,
    7.2000e+01, 6.7500e+01, 6.5000e+01, 6.1000e+01, 5.9000e+01, 5.3400e+01,
    5.2000e+01, 5.0600e+01, 4.8300e+01, 4.7000e+01, 4.5200e+01, 4.4000e+01,
    4.2400e+01, 4.1000e+01, 3.9600e+01, 3.9100e+01, 3.8000e+01, 3.7000e+01,
    3.5500e+01, 3.4600e+01, 3.3750e+01, 3.3250e+01, 3.1750e+01, 3.1250e+01,
    3.0000e+01, 2.7500e+01, 2.5000e+01, 2.2500e+01, 2.1000e+01, 2.0000e+01,
    1.9000e+01, 1.8500e+01, 1.7000e+01, 1.6000e+01, 1.5100e+01, 1.4400e+01,
    1.3750e+01, 1.2900e+01, 1.1900e+01, 1.1500e+01, 1.0000e+01, 9.1000e+00,
    8.1000e+00, 7.1500e+00, 7.0000e+00, 6.7500e+00, 6.5000e+00, 6.2500e+00,
    6.0000e+00, 5.4000e+00, 5.0000e+00, 4.7500e+00, 4.0000e+00, 3.7300e+00,
    3.5000e+00, 3.1500e+00, 3.0500e+00, 3.0000e+00, 2.9700e+00, 2.8700e+00,
    2.7700e+00, 2.6700e+00, 2.5700e+00, 2.4700e+00, 2.3800e+00, 2.3000e+00,
    2.2100e+00, 2.1200e+00, 2.0000e+00, 1.9400e+00, 1.8500e+00, 1.7700e+00,
    1.6800e+00, 1.5900e+00, 1.5000e+00, 1.4500e+00, 1.4000e+00, 1.3500e+00,
    1.3000e+00, 1.2500e+00, 1.2250e+00, 1.2000e+00, 1.1750e+00, 1.1500e+00,
    1.1400e+00, 1.1300e+00, 1.1200e+00, 1.1100e+00, 1.1000e+00, 1.0900e+00,
    1.0800e+00, 1.0700e+00, 1.0600e+00, 1.0500e+00, 1.0400e+00, 1.0300e+00,
    1.0200e+00, 1.0100e+00, 1.0000e+00, 9.7500e-01, 9.5000e-01, 9.2500e-01,
    9.0000e-01, 8.5000e-01, 8.0000e-01, 7.5000e-01, 7.0000e-01, 6.5000e-01,
    6.2500e-01, 6.0000e-01, 5.5000e-01, 5.0000e-01, 4.5000e-01, 4.0000e-01,
    3.7500e-01, 3.5000e-01, 3.2500e-01, 3.0000e-01, 2.7500e-01, 2.5000e-01,
    2.2500e-01, 2.0000e-01, 1.7500e-01, 1.5000e-01, 1.2500e-01, 1.0000e-01,
    9.0000e-02, 8.0000e-02, 7.0000e-02, 6.0000e-02, 5.0000e-02, 4.0000e-02,
    3.0000e-02, 2.5300e-02, 1.0000e-02, 7.5000e-03, 5.0000e-03, 4.0000e-03,
    3.0000e-03, 2.5000e-03, 2.0000e-03, 1.5000e-03, 1.2000e-03, 1.0000e-03,
    7.5000e-04, 5.0000e-04, 1.0000e-04, 1.0000e-05
]))

# Fine logarithmic energy group structure (in eV), from lowest to highest energy
LOG_501 = np.logspace(np.log10(1e-5), np.log10(20.0e6), 501)

LOG_1001 = np.logspace(np.log10(1e-5), np.log10(20.0e6), 1001)


def get_energy_bins(inputs_dict=None):
    """Get the energy group structure based on inputs setting.

    Parameters
    ----------
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    numpy.ndarray
        Energy bin boundaries in eV
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        from inputs import inputs
        inputs_dict = inputs

    energy_structure = inputs_dict.get('energy_structure', 'log501')  # Default to log501 if not specified
    if energy_structure.lower() == 'scale238':
        return SCALE_238
    elif energy_structure.lower() == 'log1001':
        return LOG_1001
    else:  # Default to log501
        return LOG_501

def get_group_indices(energy, inputs_dict=None):
    """Get the group indices for collapsing into few-group structure.

    Parameters
    ----------
    energy : float
        Energy boundary in eV
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    numpy.ndarray
        Indices in energy bins array where energies are greater than the boundary
    """
    energy_bins = get_energy_bins(inputs_dict)
    return np.where(energy_bins > energy)[0]
