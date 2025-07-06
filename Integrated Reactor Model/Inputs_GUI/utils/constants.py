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
}


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
