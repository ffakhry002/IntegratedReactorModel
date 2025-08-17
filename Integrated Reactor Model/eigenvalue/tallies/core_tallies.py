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

def create_coreflux_tallys(inputs_dict=None):
    """Create a mesh tally covering the entire core including shields.

    The tally includes three energy groups:
    - Thermal: E < thermal_cutoff (default: 0.625 eV)
    - Epithermal: thermal_cutoff < E < fast_cutoff (default: 0.625 eV to 100 keV)
    - Fast: E > fast_cutoff (default: 100 keV)

    Parameters
    ----------
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    openmc.Tallies
        Collection of mesh tallies for core flux
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    tallies = openmc.Tallies()

    # Create mesh covering core + reflector + bioshield
    mesh = openmc.RegularMesh()
    mesh.dimension = inputs_dict['core_mesh_dimension']  # Use mesh dimensions from inputs

    # Calculate mesh boundaries from inputs (in cm)
    core_radius = inputs_dict['tank_radius'] * 100
    reflector_thickness = inputs_dict['reflector_thickness'] * 100
    total_radius = core_radius + reflector_thickness

    half_height = inputs_dict['fuel_height']* 50

    mesh.lower_left = [-total_radius, -total_radius, -half_height]
    mesh.upper_right = [total_radius, total_radius, half_height]

    mesh_filter = openmc.MeshFilter(mesh)
    print(f"Created core flux mesh filter - ID: {mesh_filter.id}, Dimensions: {mesh.dimension}")

    # Define three energy groups using cutoffs from inputs
    thermal_cutoff = inputs_dict.get('thermal_cutoff', 0.625)  # Default: 0.625 eV
    fast_cutoff = inputs_dict.get('fast_cutoff', 100000.0)     # Default: 100 keV

    # Create energy bins for three groups: [0, thermal, fast, 20M]
    energy_bins = [0.0, thermal_cutoff, fast_cutoff, 20.0e6]
    energy_filter = openmc.EnergyFilter(energy_bins)
    print(f"Created core flux energy filter - ID: {energy_filter.id}, 3-group structure")

    # Create mesh tally with three energy groups
    mesh_tally = openmc.Tally(name='flux_mesh')
    if(inputs_dict['Core_Three_Group_Energy_Bins']):
        mesh_tally.filters = [mesh_filter, energy_filter]
    else:
        mesh_tally.filters = [mesh_filter]
    mesh_tally.scores = ['flux']
    tallies.append(mesh_tally)
    return tallies
