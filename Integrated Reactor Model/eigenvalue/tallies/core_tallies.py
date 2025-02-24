"""
Functions for creating core-wide tallies (nu-fission, fission, and core flux mesh).
"""

import openmc
from inputs import inputs

def create_nutotal_tallies():
    """Create tallies for whole-core quantities (nu-fission, fission).

    Returns
    -------
    openmc.Tallies
        Collection of tallies for core-wide reactions
    """
    tallies = openmc.Tallies()

    # Create tally for nu-fission
    nu_fission_tally = openmc.Tally(name='nu-fission')
    nu_fission_tally.scores = ['nu-fission']
    tallies.append(nu_fission_tally)

    # Create tally for fission
    fission_tally = openmc.Tally(name='fission')
    fission_tally.scores = ['fission']
    tallies.append(fission_tally)

    return tallies

def create_coreflux_tallys():
    """Create a mesh tally covering the entire core including shields."""
    tallies = openmc.Tallies()

    # Create mesh covering core + reflector + bioshield
    mesh = openmc.RegularMesh()
    mesh.dimension = [201, 201, 201]  # Higher resolution mesh

    # Calculate mesh boundaries from inputs (in cm)
    core_radius = inputs['tank_radius'] * 100
    reflector_thickness = inputs['reflector_thickness'] * 100
    total_radius = core_radius + reflector_thickness

    half_height = inputs['fuel_height']* 50

    mesh.lower_left = [-total_radius, -total_radius, -half_height]
    mesh.upper_right = [total_radius, total_radius, half_height]

    # Calculate mesh volume for normalization
    dx = 2 * total_radius / mesh.dimension[0]
    dy = 2 * total_radius / mesh.dimension[1]
    dz = 2 * half_height / mesh.dimension[2]
    mesh_volume = dx * dy * dz

    # Create mesh filter and tally
    mesh_filter = openmc.MeshFilter(mesh)
    mesh_tally = openmc.Tally(name='flux_mesh')
    mesh_tally.filters = [mesh_filter]
    mesh_tally.scores = ['flux']

    tallies.append(mesh_tally)
    return tallies
