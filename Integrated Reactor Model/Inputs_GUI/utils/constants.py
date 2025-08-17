# Shared constants for the GUI application

"""
Constants and Configuration
Material colors and other constants used throughout the GUI
"""

MATERIAL_COLORS = {
    "Light Water": "#40E0D0",      # Turquoise
    "Heavy Water": "#1E90FF",      # Dodger Blue
    "U3Si2": "#32CD32",           # Lime Green
    "U10Mo": "#FFD700",           # Gold
    "UO2": "#FF6347",             # Tomato
    "Al6061": "#C0C0C0",          # Silver
    "Zirc4": "#BC8F8F",           # Rosy Brown
    "Zirc2": "#CD853F",           # Peru
    "Concrete": "#8B7355",         # Dark Tan
    "Steel": "#708090",            # Slate Gray
    "mgo": "#9370DB",              # Medium Purple - MgO reflector
    "beryllium": "#E6E6FA",        # Lavender - Be reflector
    "Bioshield": "#8B4513",        # Saddle Brown
    "Reflector": "#9370DB",        # Medium Purple
    "Plenum": "#87CEEB",           # Sky Blue
    "Coolant": "#40E0D0",         # Turquoise
    "Feed": "#4682B4",            # Steel Blue
    "Vacuum": "#FFFAFA",          # Snow - very light for vacuum
    "fill": "#87CEFA",            # Light Sky Blue - for Al-water mix
    "Test pos": "#87CEFA",        # Light Sky Blue - for Al-water mix
    "PWR_loop": "#FF69B4",        # Hot Pink - for PWR loop
    "BWR_loop": "#32CD32",        # Lime Green - for BWR loop
    "Gas_capsule": "#FFA500",     # Orange - for Gas capsule
}


def get_irradiation_material_type(lattice_position, inputs_dict):
    """Get irradiation material type from lattice position.

    Parameters
    ----------
    lattice_position : str
        Position string from core lattice (e.g., 'I_1', 'I_2P', 'I_3B', 'I_4G')
    inputs_dict : dict
        Inputs dictionary containing irradiation_fill

    Returns
    -------
    str
        Material name for the irradiation position
    """
    if not lattice_position.startswith('I_'):
        raise ValueError(f"Invalid irradiation position: {lattice_position}")

    # Check for suffix
    if lattice_position.endswith('P'):
        return 'PWR_loop'
    elif lattice_position.endswith('B'):
        return 'BWR_loop'
    elif lattice_position.endswith('G'):
        return 'Gas_capsule'
    else:
        # No suffix, use default irradiation_fill
        return inputs_dict.get('irradiation_fill', 'Vacuum')


def get_material_color(material):
    """Get color for a material.

    Parameters
    ----------
    material : str
        Name of the material

    Returns
    -------
    str
        Hex color code for the material, or gray default if not found
    """
    return MATERIAL_COLORS.get(material, "#808080")  # Gray default
