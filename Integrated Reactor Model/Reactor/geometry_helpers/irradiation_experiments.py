"""
Centralized configuration for irradiation experiment geometries.

This module contains all the geometric parameters for different irradiation
experiment types (HTWL, SIGMA, etc.) to ensure consistency across geometry
creation, tallying, and plotting functions.
"""

# Irradiation Experiment Configurations
IRRADIATION_EXPERIMENTS = {
    'HTWL': {  # For PWR_loop and BWR_loop
        'description': 'High Temperature Water Loop experiment',
        'reference_outer_radius': 2.585,  # MCNP reference radius for scaling [cm]

        'radii': {  # All radii in cm (unscaled MCNP dimensions)
            'spine': 0.24,
            'sample_ti': 0.1,
            'sample_inner': 0.3175,
            'sample_outer': 0.45,
            'sample_center': 1.03,  # Distance from center to sample centers
            'capsule_inner': 1.698,
            'capsule_outer': 1.778,
            'autoclave_inner': 1.93675,
            'autoclave_outer': 2.15265,
            'thimble_inner': 2.2987,  # Also CO2 outer boundary
            'thimble_outer': 2.54,
            'water_gap_outer': 2.585
        },

        'z_planes': {  # All z-coordinates in cm (fixed experimental positions)
            'autoclave_bottom_bot': -28.305,
            'autoclave_bottom_top': -27.305,
            'capsule_bottom_bot': -24.0,
            'capsule_bottom_top': -23.5,
            'capsule_top_bot': -0.5,
            'capsule_top_top': 0.0
        },

        'sample_positions': [  # 4 sample positions at clock positions
            (0, 1),      # 12:00 (y = +sample_center)
            (1, 0),      # 3:00  (x = +sample_center)
            (0, -1),     # 6:00  (y = -sample_center)
            (-1, 0),     # 9:00  (x = -sample_center)
        ],

        'sample_height': 23.0,  # Height of sample region [cm] (-23.5 to -0.5)
    },

    'SIGMA': {  # For Gas_capsule
        'description': 'SIGMA gas capsule experiment',
        'reference_outer_radius': 2.618,  # MCNP reference radius for scaling [cm]

        'radii': {  # All radii in cm (unscaled MCNP dimensions)
            'spine': 0.25,
            'inner_he': 0.5,
            'sample_inner': 1.3,
            'sample_outer': 1.8,
            'outer_graphite': 2.35,
            'outer_he': 2.4511,
            'thimble_inner': 2.4511,  # Same as outer_he
            'thimble_outer': 2.54,
            'water_gap_outer': 2.618
        }
    }
}


def get_experiment_config(irradiation_type):
    """Get experiment configuration based on irradiation type.

    Parameters
    ----------
    irradiation_type : str
        Type of irradiation ('PWR_loop', 'BWR_loop', 'Gas_capsule')

    Returns
    -------
    dict
        Configuration dictionary for the experiment
    """
    if irradiation_type in ['PWR_loop', 'BWR_loop']:
        return IRRADIATION_EXPERIMENTS['HTWL']
    elif irradiation_type == 'Gas_capsule':
        return IRRADIATION_EXPERIMENTS['SIGMA']
    else:
        raise ValueError(f"Unknown irradiation type: {irradiation_type}")


def get_scaled_radii(irradiation_type, target_diameter):
    """Get scaled radii for an experiment type.

    Parameters
    ----------
    irradiation_type : str
        Type of irradiation ('PWR_loop', 'BWR_loop', 'Gas_capsule')
    target_diameter : float
        Target diameter for the experiment [cm]

    Returns
    -------
    tuple
        (scaled_radii_dict, scale_factor)
    """
    config = get_experiment_config(irradiation_type)
    scale_factor = target_diameter / (2 * config['reference_outer_radius'])

    scaled_radii = {}
    for name, radius in config['radii'].items():
        scaled_radii[name] = radius * scale_factor

    return scaled_radii, scale_factor


def get_sample_positions(irradiation_type, scale_factor):
    """Get scaled sample positions for HTWL experiments.

    Parameters
    ----------
    irradiation_type : str
        Type of irradiation ('PWR_loop', 'BWR_loop')
    scale_factor : float
        Scaling factor for dimensions

    Returns
    -------
    list
        List of (x, y) tuples for sample positions [cm]
    """
    if irradiation_type not in ['PWR_loop', 'BWR_loop']:
        return []

    config = get_experiment_config(irradiation_type)
    sample_center_radius = config['radii']['sample_center'] * scale_factor

    positions = []
    for x_mult, y_mult in config['sample_positions']:
        x = x_mult * sample_center_radius
        y = y_mult * sample_center_radius
        positions.append((x, y))

    return positions


def get_scaled_z_planes(irradiation_type):
    """Get z-plane coordinates for HTWL experiments.

    Parameters
    ----------
    irradiation_type : str
        Type of irradiation ('PWR_loop', 'BWR_loop')

    Returns
    -------
    dict
        Dictionary of z-plane coordinates [cm]
    """
    if irradiation_type not in ['PWR_loop', 'BWR_loop']:
        return {}

    config = get_experiment_config(irradiation_type)
    return config['z_planes'].copy()  # Return a copy to prevent modification


def get_reference_axial_bounds(inputs_dict=None):
    """Get the reference axial bounds from PWR/BWR experiments for height matching.

    This function extracts the axial extent that PWR/BWR experiments use for their
    sample regions, which can be used to match the gas experiment height when
    match_GS_height is enabled.

    Parameters
    ----------
    inputs_dict : dict, optional
        Inputs dictionary (not used currently, but kept for consistency)

    Returns
    -------
    tuple
        (z_bottom, z_top, height) where:
        - z_bottom: Bottom of sample region [cm]
        - z_top: Top of sample region [cm]
        - height: Total height of sample region [cm]
    """
    # Get z-planes from HTWL configuration
    z_planes = get_scaled_z_planes('PWR_loop')  # Same for PWR and BWR

    z_bottom = z_planes['capsule_bottom_top']  # -23.5 cm
    z_top = z_planes['capsule_top_bot']        # -0.5 cm
    height = z_top - z_bottom                  # 23.0 cm

    return z_bottom, z_top, height
