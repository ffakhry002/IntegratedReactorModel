# Create base inputs dictionary
base_inputs = {
    ###########################################
    # Parametric Study Configuration
    ###########################################
    "parametric_study": False,        # Toggle for parametric study mode

    ###########################################
    # Core Configuration
    ###########################################
    # Core Layout
    "core_lattice": [  # C: coolant, F: fuel assembly, E: enriched fuel assembly, I: irradiation position
        ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'I_1B', 'I_2', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'I_3P', 'I_4G', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
    ],

    # Core Operating Parameters
    "core_power": 10.0,              # Core power [MW]
    "assembly_type": 'Plate',        # Assembly type: 'Pin' or 'Plate'

    ###########################################
    # Geometry Specifications
    ###########################################
    # Radial Core Geometry
    "tank_radius": 0.25,            # Reactor tank radius [m]
    "reflector_thickness": 0.1,    # Radial reflector thickness [m]
    "bioshield_thickness": 0.25,    # Radial bioshield thickness [m]

    # Axial Core Geometry
    "bottom_bioshield_thickness": 0.9,  # Bottom bioshield thickness [m]
    "bottom_reflector_thickness": 0.3,  # Bottom reflector thickness [m]
    "feed_thickness": 0,            # Feed region thickness [m]
    "plenum_height": 1.7,             # Plenum height [m]
    "top_reflector_thickness": 0.0,   # Top reflector thickness [m]
    "top_bioshield_thickness": 0.0,    # Top bioshield thickness [m]

    "fuel_height": 0.6,            # Active fuel height [m]

    # Pin Fuel Assembly Parameters
    "pin_pitch": 0.0126,          # Pin-to-pin pitch [m]
    "r_fuel": 0.0041,            # Fuel pellet radius [m]
    "r_clad_inner": 0.0042,      # Cladding inner radius [m]
    "r_clad_outer": 0.00475,     # Cladding outer radius [m]
    "n_side_pins": 3,            # Number of pins per assembly side
    "guide_tube_positions": [(1, 1)],   # List of (x,y) tuples for guide tube positions

    # Plate Fuel Assembly Parameters
    "fuel_meat_width": 0.0391,      # Fuel meat width [m]
    "fuel_plate_width": 0.0481,       # Fuel plate width [m]
    "fuel_plate_pitch": 0.0037,       # Plate-to-plate pitch [m]
    "fuel_meat_thickness": 0.00147,    # Fuel meat thickness [m]
    "clad_thickness": 0.00025,       # coolant to fuel meat in y direction cladding thickness [m]
    "plates_per_assembly": 13,         # Number of plates per assembly
    "clad_structure_width": 0.0015,  # Support structure width [m]

    ###########################################
    # Materials Configuration
    ###########################################
    # Material Choices
    "coolant_type": 'Light Water',     # Coolant: 'Light Water' or 'Heavy Water'
    "clad_type": 'Al6061',             # Cladding: 'Al6061', 'Zirc2', or 'Zirc4'
    "fuel_type": 'U3Si2',             # Fuel: 'U3Si2', 'UO2', or 'U10Mo'
    "reflector_material": "mgo", # Reflector material
    "bioshield_material": "Concrete",  # Bioshield material

    # Fuel Specifications
    "n%": 19.75,                      # Standard fuel enrichment [%]
    "n%E": 93,                     # Enhanced fuel enrichment [%]

    ###########################################
    # Thermal Hydraulics Parameters
    ###########################################
    "reactor_pressure": 300000.0,          # System pressure [Pa]
    "flow_rate": 3,                   # Coolant flow rate [m/s]
    "T_inlet": 315.15,          # Inlet temperature [K]

    # Direct Thermal hydraulics calculation mode only
    "input_power_density": 100,        # Power density [kW/L]
    "max_linear_power": 70,           # Maximum linear power [kW/m]
    "average_linear_power": 50,       # Average linear power [kW/m]
    "cos_curve_squeeze": 0,           # Axial power shape parameter [0-1]
    "CP_PD_MLP_ALP": "CP",          # CP: core power (MW), PD: power density (kW/L)
                                     # MLP: max linear power (kW/m), ALP: avg linear power (kW/m)

    ###########################################
    # Irradiation Position Parameters
    ###########################################
    "irradiation_clad": False,              # Include irradiation position cladding
    "irradiation_clad_thickness": 0.0015, # Irradiation cladding thickness [m]
    "irradiation_fill": "Vacuum",  # Default fill for irradiation positions without suffix: e.g. "Vacuum" or "BWR_fluid"

    # Complexity
    "irradiation_cell_complexity": "Complex", # Simple: smeared channels, Complex: MCNP provided positions

    # Complex fills
    "PWR_sample_fill": "Vacuum", # PWR loop sample fill
    "BWR_sample_fill": "Vacuum", # BWR loop sample fill
    "Gas_capsule_fill": "Vacuum", # Gas capsule sample fill

    # Loop Diameters
    "PWR_loop_diameter": 0.9, # PWR loop diameter [% cell width]
    "BWR_loop_diameter": 0.9, # BWR loop diameter [% cell width]
    "Gas_capsule_diameter": 0.9, # Gas capsule diameter [% cell width]

    ###########################################
    # OpenMC Transport Parameters
    ###########################################
    # Standard Transport Settings
    "batches": int(150),                   # Number of active batches
    "inactive": int(20),                   # Number of inactive batches
    "particles": int(100000),            # Particles per batch
    "energy_structure": 'log1001',    # Energy group structure

    # Energy Group Boundaries
    "thermal_cutoff": float(0.625),          # Thermal/epithermal boundary [eV]
    "fast_cutoff": float(100000.0),          # Epithermal/fast boundary [eV]

    # Tally Granularity Settings
    "power_tally_axial_segments": 50,     # Number of axial segments for power tallies
    "irradiation_axial_segments": 100,    # Number of axial segments for irradiation tallies
    "core_mesh_dimension": [201, 201, 201], # Mesh resolution for core flux tallies
    "entropy_mesh_dimension": [20, 20, 20],  # Mesh resolution for entropy calculation

    # Additional Tallies
    "Core_Three_Group_Energy_Bins": True, # True: use three energy bins for core tallies, False: don't tally energy groups
    "tally_power": True,                  # True: calculate power tallies and TH, False: skip power calculations
    "element_level_power_tallies": False, # True: tally power for individual fuel elements, False: tally power for assemblies

    ###########################################
    # Depletion Calculation Parameters
    ###########################################
    # Depletion Scenario Selection (only one should be True)
    "deplete_core": False,                    # Full core depletion
    "deplete_assembly": False,               # Single assembly with reflective BC
    "deplete_assembly_enhanced": False,      # Single enhanced assembly with reflective BC
    "deplete_element": False,                # Single fuel element with reflective BC
    "deplete_element_enhanced": False,       # Single enhanced fuel element with reflective BC

    # Time Steps Configuration
    "depletion_timestep_units": "MWd/kgHM",  # Units for timesteps: 'MWd/kgHM' or 'days'
    "depletion_timesteps": [{'steps': 10, 'size': 0.01},{'steps': 10, 'size': 0.1}, {'steps': 10, 'size': 0.5}, {'steps': 5, 'size': 2.5}, {'steps': 5, 'size': 5.0}, {'steps': 5, 'size': 10.0}],

    # Transport Settings for Depletion
    "depletion_particles": int(1000),       # Particles per batch for depletion
    "depletion_batches": int(100),         # Active batches for depletion
    "depletion_inactive": int(20),         # Inactive batches for depletion

    # Depletion Options
    "depletion_integrator": "predictor",  # Integration algorithm
                                         # Options: predictor, cecm, celi, cf4,
                                         #         epcrk4, leqi, siceli, sileqi
    "depletion_chain": "casl",       # Depletion chain type ('casl' or 'endfb71')

    # Nuclides to Extract and Plot
    "depletion_nuclides": ['U235', 'U238', 'Pu239', 'Xe135', 'Sm149', 'Cs137', 'Sr90', 'I131'],

    ###########################################
    # Miscellaneous Settings
    ###########################################
    "outputs_folder": "local_outputs",  # Base output directory
    "pixels": (1600, 1600),            # Plot resolution
}


############################ UPDATED INPUTS #####################################


def calculate_derived_values(core_lattice, guide_tube_positions):
    """Calculate derived values from the core lattice and guide tube positions.

    Parameters
    ----------
    core_lattice : list
        2D list representing the core layout
    guide_tube_positions : list
        List of (x,y) tuples for guide tube positions

    Returns
    -------
    tuple
        (num_assemblies, n_guide_tubes)
    """
    # Flatten the core lattice and count 'F's for number of assemblies
    flattened = [item for row in core_lattice for item in row]
    num_assemblies_F = flattened.count('F')
    num_assemblies_E = flattened.count('E')
    num_assemblies=num_assemblies_E+num_assemblies_F

    # Get number of guide tubes from length of positions list
    n_guide_tubes = len(guide_tube_positions)

    return num_assemblies, n_guide_tubes


# Calculate derived values from the base inputs
num_assemblies, n_guide_tubes = calculate_derived_values(base_inputs["core_lattice"],
                                                       base_inputs["guide_tube_positions"])

# Add derived values to create final inputs dictionary
inputs = {
    **base_inputs,
    "n_guide_tubes": 1,  # number of guide tubes per assembly
    "num_assemblies": 48  # Automatically calculated from core_lattice
}
