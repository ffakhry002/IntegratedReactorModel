"""
Helper functions for plotting.
"""

import numpy as np
import openmc
from inputs import inputs

def get_cell_volume(cell_id, sp, is_irradiation=False, inputs_dict=None):
    """Get the volume of a cell.

    Parameters
    ----------
    cell_id : int
        ID of the cell
    sp : openmc.StatePoint
        StatePoint file containing the geometry information
    is_irradiation : bool, optional
        Whether this is an irradiation position cell (default: False)
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    float
        Volume of the cell in cm³
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    cell = sp.summary.geometry.get_all_cells()[cell_id]
    if hasattr(cell, 'volume') and cell.volume is not None:
        return cell.volume
    else:
        # If volume not available, calculate from dimensions
        if inputs_dict['assembly_type'] == 'Pin':
            width = inputs_dict['pin_pitch'] * inputs_dict['n_side_pins'] * 100  # Convert m to cm
        else:
            width = (inputs_dict['fuel_plate_width'] + 2 * inputs_dict['clad_structure_width']) * 100  # Convert m to cm

        # Only subtract cladding thickness for irradiation positions
        if is_irradiation and inputs_dict['irradiation_clad']:
            clad_thickness = inputs_dict['irradiation_clad_thickness'] * 100  # Convert m to cm
            width = width - (2 * clad_thickness)  # Subtract cladding from both sides

        height = inputs_dict['fuel_height'] * 100  # Convert m to cm
        return width * width * height

def get_mesh_volume(mesh):
    """Calculate the volume of a mesh element.

    Parameters
    ----------
    mesh : openmc.RegularMesh
        The mesh to calculate volume for

    Returns
    -------
    float
        Volume of a single mesh element in cm³
    """
    return np.prod(np.array(mesh.upper_right) - np.array(mesh.lower_left))/np.prod(mesh.dimension)

def get_tally_volume(tally, sp, inputs_dict=None):
    """Get the volume associated with a tally.

    Parameters
    ----------
    tally : openmc.Tally
        The tally to get volume for
    sp : openmc.StatePoint
        StatePoint file containing geometry information
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    float
        Volume in cm³
    """
    for filter in tally.filters:
        if isinstance(filter, openmc.MeshFilter):
            return get_mesh_volume(sp.meshes[filter._mesh.id])
        elif isinstance(filter, openmc.CellFilter):
            # Check if this is an irradiation position tally
            is_irradiation = tally.name.startswith('I_')
            return get_cell_volume(filter.bins[0], sp, is_irradiation, inputs_dict)

    raise ValueError(f"No volume information found for tally {tally.name}")
