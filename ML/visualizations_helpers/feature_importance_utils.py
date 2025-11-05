"""
Utility functions to detect and handle variable feature counts in physics encoding
Supports vacuum (22), fill+single (29), and fill+separate (37) feature counts
"""

import numpy as np

def detect_physics_mode(n_features):
    """Detect irradiation and NCI mode from feature count.

    Parameters
    ----------
    n_features : int
        Number of features in the physics encoding

    Returns
    -------
    tuple
        (irradiation_mode, nci_mode, is_valid)
    """
    if n_features == 22:
        return 'vacuum', 'single', True
    elif n_features == 29:
        return 'fill', 'single', True
    elif n_features == 37:
        return 'fill', 'separate', True
    else:
        return 'unknown', 'unknown', False


def get_feature_names(irradiation_mode, nci_mode):
    """Get feature names based on mode.

    Parameters
    ----------
    irradiation_mode : str
        'vacuum' or 'fill'
    nci_mode : str
        'single' or 'separate'

    Returns
    -------
    list
        List of feature names in order
    """
    names = []

    # Global features
    names.append('Global: Avg Distance')
    names.append('Global: Symmetry')

    if irradiation_mode == 'fill':
        names.append('Global: Fraction P')
        names.append('Global: Fraction B')
        names.append('Global: Fraction G')

    # Local features per position
    local_feature_types = ['Fuel Density', 'Coolant Contact', 'Edge Distance', 'Center Distance']
    if irradiation_mode == 'fill':
        local_feature_types.append('Vehicle Type')

    for pos in range(1, 5):
        for feat_type in local_feature_types:
            names.append(f'Pos{pos}: {feat_type}')

    # NCI features
    if irradiation_mode == 'fill' and nci_mode == 'separate':
        for pos in range(1, 5):
            names.append(f'Pos{pos}: NCI_P')
            names.append(f'Pos{pos}: NCI_B')
            names.append(f'Pos{pos}: NCI_G')
    else:
        for pos in range(1, 5):
            names.append(f'Pos{pos}: NCI')

    return names


def reorder_importances_for_plot(importances, irradiation_mode, nci_mode):
    """Reorder feature importances for better visualization.

    Groups features by type rather than by position for easier comparison.

    Parameters
    ----------
    importances : array-like
        Feature importance values
    irradiation_mode : str
        'vacuum' or 'fill'
    nci_mode : str
        'single' or 'separate'

    Returns
    -------
    tuple
        (reordered_importances, reordered_names)
    """
    n_global = 2 if irradiation_mode == 'vacuum' else 5
    n_local_per_pos = 4 if irradiation_mode == 'vacuum' else 5
    n_nci_per_pos = 1 if nci_mode == 'single' else 3

    # Start with global features
    reordered = list(importances[:n_global])

    # Extract local features grouped by type
    local_start_idx = n_global
    for feat_idx in range(n_local_per_pos):
        for pos in range(4):
            idx = local_start_idx + pos * n_local_per_pos + feat_idx
            reordered.append(importances[idx])

    # Extract NCI features
    nci_start_idx = n_global + 4 * n_local_per_pos
    for nci_idx in range(n_nci_per_pos):
        for pos in range(4):
            idx = nci_start_idx + pos * n_nci_per_pos + nci_idx
            reordered.append(importances[idx])

    # Generate corresponding names
    names = []

    # Global
    names.append('Global: Avg Distance')
    names.append('Global: Symmetry')
    if irradiation_mode == 'fill':
        names.extend(['Global: Fraction P', 'Global: Fraction B', 'Global: Fraction G'])

    # Local by type
    local_types = ['Fuel Density', 'Coolant Contact', 'Edge Distance', 'Center Distance']
    if irradiation_mode == 'fill':
        local_types.append('Vehicle Type')

    for feat_type in local_types:
        for pos in range(1, 5):
            names.append(f'Pos{pos}: {feat_type}')

    # NCI
    if irradiation_mode == 'fill' and nci_mode == 'separate':
        for nci_type in ['P', 'B', 'G']:
            for pos in range(1, 5):
                names.append(f'Pos{pos}: NCI_{nci_type}')
    else:
        for pos in range(1, 5):
            names.append(f'Pos{pos}: NCI')

    return reordered, names


def get_aggregated_features(importances, irradiation_mode, nci_mode):
    """Aggregate local features by averaging across positions.

    Parameters
    ----------
    importances : array-like
        Feature importance values
    irradiation_mode : str
        'vacuum' or 'fill'
    nci_mode : str
        'single' or 'separate'

    Returns
    -------
    tuple
        (aggregated_importances, feature_names)
    """
    n_global = 2 if irradiation_mode == 'vacuum' else 5
    n_local_per_pos = 4 if irradiation_mode == 'vacuum' else 5
    n_nci_per_pos = 1 if nci_mode == 'single' else 3

    # Global features (unchanged)
    aggregated = list(importances[:n_global])

    # Average local features across positions
    local_start_idx = n_global
    for feat_idx in range(n_local_per_pos):
        feat_values = []
        for pos in range(4):
            idx = local_start_idx + pos * n_local_per_pos + feat_idx
            feat_values.append(importances[idx])
        aggregated.append(np.mean(feat_values))

    # Average NCI features
    nci_start_idx = n_global + 4 * n_local_per_pos
    for nci_idx in range(n_nci_per_pos):
        nci_values = []
        for pos in range(4):
            idx = nci_start_idx + pos * n_nci_per_pos + nci_idx
            nci_values.append(importances[idx])
        aggregated.append(np.mean(nci_values))

    # Generate names
    names = []
    names.append('Global: Avg Distance')
    names.append('Global: Symmetry')
    if irradiation_mode == 'fill':
        names.extend(['Global: Fraction P', 'Global: Fraction B', 'Global: Fraction G'])

    local_types = ['Fuel Density (avg)', 'Coolant Contact (avg)',
                   'Edge Distance (avg)', 'Center Distance (avg)']
    if irradiation_mode == 'fill':
        local_types.append('Vehicle Type (avg)')
    names.extend([f'Local: {t}' for t in local_types])

    if irradiation_mode == 'fill' and nci_mode == 'separate':
        names.extend(['Local: NCI_P (avg)', 'Local: NCI_B (avg)', 'Local: NCI_G (avg)'])
    else:
        names.append('Local: NCI (avg)')

    return aggregated, names


def get_feature_colors(irradiation_mode, nci_mode):
    """Get color scheme for feature importance plots.

    Returns
    -------
    tuple
        (full_plot_colors, aggregated_plot_colors)
    """
    # Define base colors
    blue = '#1f77b4'     # Global
    orange = '#ff7f0e'   # Fuel density
    green = '#2ca02c'    # Coolant contact
    red = '#d62728'      # Edge distance
    purple = '#9467bd'   # Center distance
    yellow = '#e377c2'   # Vehicle type
    brown = '#8c564b'    # NCI
    teal = '#17becf'     # NCI_P
    olive = '#bcbd22'    # NCI_B
    gray = '#7f7f7f'     # NCI_G

    n_global = 2 if irradiation_mode == 'vacuum' else 5

    # Full plot colors
    full_colors = [blue] * n_global
    full_colors += [orange] * 4  # Fuel density
    full_colors += [green] * 4   # Coolant contact
    full_colors += [red] * 4     # Edge distance
    full_colors += [purple] * 4  # Center distance

    if irradiation_mode == 'fill':
        full_colors += [yellow] * 4  # Vehicle type

        if nci_mode == 'separate':
            full_colors += [teal] * 4    # NCI_P
            full_colors += [olive] * 4   # NCI_B
            full_colors += [gray] * 4    # NCI_G
        else:
            full_colors += [brown] * 4   # NCI
    else:
        full_colors += [brown] * 4   # NCI

    # Aggregated plot colors
    agg_colors = [blue] * n_global
    if irradiation_mode == 'fill':
        agg_colors += [orange] * 5  # 5 local features
        if nci_mode == 'separate':
            agg_colors += [brown] * 3  # 3 NCI types
        else:
            agg_colors += [brown] * 1  # 1 NCI
    else:
        agg_colors += [orange] * 4  # 4 local features
        agg_colors += [brown] * 1  # 1 NCI

    return full_colors, agg_colors
