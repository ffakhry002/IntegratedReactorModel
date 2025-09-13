def generate_cell_id(cell_type, position, is_enhanced=False, clad_part=None):
    """Generate a unique ID for a cell based on its type and position.

    Parameters
    ----------
    cell_type : str
        Type of cell ('fuel' or 'irradiation')
    position : tuple
        (i, j) position in the core lattice
    is_enhanced : bool, optional
        Whether this is an enhanced fuel position
    clad_part : str, optional
        For irradiation cells with cladding, specifies which part ('bottom', 'top', 'left', 'right')
        If None, assumes this is the center cell or a non-cladding cell

    Returns
    -------
    int
        Unique ID for the cell
    """
    i, j = position

    # For fuel cells, use base 1000000 for regular fuel, 2000000 for enhanced
    if cell_type == 'fuel':
        base = 2000000 if is_enhanced else 1000000
        return base + i * 10000 + j * 100

    # For irradiation cells, use base 3000000 and standardized position multipliers
    else:  # irradiation
        part_map = {
            None: 0,  # center cell
            'bottom': 1,
            'top': 2,
            'left': 3,
            'right': 4
        }
        part_num = part_map[clad_part]
        # FIXED: Use standardized encoding scheme consistent with fuel cells and filters
        return 3000000 + i * 10000 + j * 100 + part_num


def generate_filter_id(filter_type, position=None, component=None, index=0):
    """Generate a unique ID for a filter based on its type and usage.

    This prevents OpenMC filter ID conflicts by using a systematic numbering scheme
    similar to the cell ID system.

    Parameters
    ----------
    filter_type : str
        Type of filter ('energy', 'mesh', 'cell')
    position : tuple, optional
        (i, j) position in the core lattice for position-specific filters
    component : str, optional
        Component identifier (e.g., 'irradiation', 'core', 'power', 'axial')
    index : int, optional
        Sequential index for multiple filters of the same type

    Returns
    -------
    int
        Unique ID for the filter

                    Notes
    -----
    ID Ranges:
    - Energy filters: 10000000-19999999 (10M range for positions + components)
    - Mesh filters: 20000000-29999999 (10M range for positions + components)
    - Cell filters: 30000000-39999999 (10M range for positions + components)
    """
    base_ids = {
        'energy': 10000000,
        'mesh': 20000000,
        'cell': 30000000
    }

    if filter_type not in base_ids:
        raise ValueError(f"Unknown filter type: {filter_type}")

    base = base_ids[filter_type]

    # Add position-based offset if provided
    if position is not None:
        i, j = position
        position_offset = i * 10000 + j * 100  # Standardized encoding: 4 digits for i, 2 for j
    else:
        position_offset = 0

    # Add component-based offset
    # All filter types have 10M range → use 1M component spacing
    component_offsets = {
        'irradiation': 0,
        'core': 1000000,
        'power': 2000000,
        'axial': 3000000,
        'assembly': 4000000,
        'element': 5000000
    }

    component_offset = component_offsets.get(component, 0)

    return base + component_offset + position_offset + index


def get_irradiation_cell_name(position, core_lattice):
    """Get the irradiation cell name from the core lattice.

    Parameters
    ----------
    position : tuple
        (i, j) position in the core lattice
    core_lattice : list
        2D list representing the core layout

    Returns
    -------
    str
        Name of the irradiation cell (e.g., 'irradiation_cell_1')
    """
    i, j = position
    cell_id = core_lattice[i][j].split('_')[1]  # Extract number from I_X format
    return f"irradiation_cell_{cell_id}"
