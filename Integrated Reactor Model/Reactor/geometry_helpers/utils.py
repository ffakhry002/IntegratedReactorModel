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

    # For irradiation cells, use base 3000000 and larger position multipliers
    else:  # irradiation
        part_map = {
            None: 0,  # center cell
            'bottom': 1,
            'top': 2,
            'left': 3,
            'right': 4
        }
        part_num = part_map[clad_part]
        return 3000000 + i * 100000 + j * 1000 + part_num

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
