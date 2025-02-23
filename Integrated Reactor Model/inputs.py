# Create base inputs dictionary
base_inputs = {
    ###########################################
    # Core Configuration
    ###########################################
    # Core Layout
    "core_lattice": [  # C: coolant, F: fuel assembly
        ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C']
    ],
    # Core Operating Parameters
    "core_power": 10,              # Core power [MW]
    "fuel_height": 0.6,            # Active fuel height [m]
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
    "feed_thickness": 0.0,            # Feed region thickness [m]
    "plenum_height": 1.7,             # Plenum height [m]
    "top_reflector_thickness": 0.00,   # Top reflector thickness [m]
    "top_bioshield_thickness": 0.0,    # Top bioshield thickness [m]

    # Pin Fuel Assembly Parameters
    "pin_pitch": 0.0126,          # Pin-to-pin pitch [m]
    "r_fuel": 0.0041,            # Fuel pellet radius [m]
    "r_clad_inner": 0.0042,      # Cladding inner radius [m]
    "r_clad_outer": 0.00475,     # Cladding outer radius [m]
    "n_side_pins": 3,            # Number of pins per assembly side
    "guide_tube_positions": [],   # List of (x,y) tuples for guide tube positions

    # Plate Fuel Assembly Parameters
    "fuel_meat_width": 3.91/100,      # Fuel meat width [m]
    "fuel_plate_width": 4.81/100,       # Fuel plate width [m]
    "fuel_plate_pitch": 0.37/100,       # Plate-to-plate pitch [m]
    "fuel_meat_thickness": 0.147/100,    # Fuel meat thickness [m]
    "clad_thickness": 0.025/100,       # coolant to fuel meat in y direction cladding thickness [m]
    "plates_per_assembly": 13,         # Number of plates per assembly
    "clad_structure_width": 0.15/100,  # Support structure width [m]

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
    "input_power_density": 100,        # Power density [kW/L]
    "max_linear_power": 70,           # Maximum linear power [kW/m]
    "average_linear_power": 50,       # Average linear power [kW/m]
    "reactor_pressure": 3e5,          # System pressure [Pa]
    "flow_rate": 3,                   # Coolant flow rate [m/s]
    "T_inlet": 273.15 + 42,          # Inlet temperature [K]
    "cos_curve_squeeze": 0,           # Axial power shape parameter [0-1]

    # Power calculation mode for thermal hydraulics
    "CP_PD_MLP_ALP": "CP",           # CP: core power, PD: power density
                                     # MLP: max linear power, ALP: avg linear power

    ###########################################
    # Irradiation Position Parameters
    ###########################################
    "irradiation_clad": False,              # Include irradiation position cladding
    "irradiation_clad_thickness": 0.15/100, # Irradiation cladding thickness [m]
    "irradiation_cell_fill": "Vacuum",      # Fill: "Vacuum" or "fill" (Al-water mix)

    ###########################################
    # OpenMC Transport Parameters
    ###########################################
    # Standard Transport Settings
    "batches": 150,                   # Number of active batches
    "inactive": 20,                   # Number of inactive batches
    "particles": int(1e2),            # Particles per batch
    "energy_structure": 'log1001',    # Energy group structure

    # Energy Group Boundaries
    "thermal_cutoff": 0.625,          # Thermal/epithermal boundary [eV]
    "fast_cutoff": 100.0e3,          # Epithermal/fast boundary [eV]

    ###########################################
    # Depletion Calculation Parameters
    ###########################################
    # Depletion Scenario Selection (only one should be True)
    "deplete_core": True,                    # Full core depletion
    "deplete_assembly": False,               # Single assembly with reflective BC
    "deplete_assembly_enhanced": False,      # Single enhanced assembly with reflective BC
    "deplete_element": False,                # Single fuel element with reflective BC
    "deplete_element_enhanced": False,       # Single enhanced fuel element with reflective BC

    # Time Steps Configuration
    "depletion_timestep_units": "MWd/kgHM",  # Units for timesteps: 'MWd/kgHM' or 'days'
    "depletion_timesteps": [
        {"steps": 1, "size": 0.1},  # 5 steps of 0.2 MWd/kgHM or days
        # {"steps": 5, "size": 0.5},  # 10 steps of 1.0 MWd/kgHM or days
        # {"steps": 5, "size": 2.5},   # 5 steps of 2.0 MWd/kgHM or days
        # {"steps": 5, "size": 5},   # 5 steps of 2.0 MWd/kgHM or days
        # {"steps": 5, "size": 10},   # 5 steps of 2.0 MWd/kgHM or days
        # {"steps": 5, "size": 0}    # 5 steps of 10.0 MWd/kgHM or days
    ],

    # Transport Settings for Depletion
    "depletion_particles": 100,       # Particles per batch for depletion
    "depletion_batches": 100,         # Active batches for depletion
    "depletion_inactive": 20,         # Inactive batches for depletion

    # Depletion Options
    "depletion_integrator": "predictor",  # Integration algorithm
                                         # Options: predictor, cecm, celi, cf4,
                                         #         epcrk4, leqi, siceli, sileqi
    "depletion_chain": "casl",       # Depletion chain type ('casl' or 'endfb71')

    # Nuclides to Extract and Plot
    "depletion_nuclides": [
        'U235',   # Primary fissile fuel
        'U238',   # Fertile fuel
        'Pu239',  # Primary plutonium breeding product
        'Xe135',  # Important neutron poison (short-lived)
        'Sm149',  # Important neutron poison (long-lived)
        'Cs137',  # Important fission product (long-lived)
        'Sr90',   # Important fission product (long-lived)
        'I131'    # Important fission product (short-lived)
    ],

    ###########################################
    # Miscellaneous Settings
    ###########################################
    "outputs_folder": "local_outputs",  # Base output directory
    "pixels": (1000, 1000),            # Plot resolution
}


############################ UPDATED INPUTS #####################################


def calculate_derived_values(core_lattice, guide_tube_positions):
    """Calculate derived values from the core lattice and guide tube positions.

    Args:
        core_lattice (list): 2D list representing the core layout
        guide_tube_positions (list): List of (x,y) tuples for guide tube positions

    Returns:
        tuple: (num_assemblies, n_guide_tubes)
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
    "n_guide_tubes": n_guide_tubes,  # number of guide tubes per assembly
    "num_assemblies": num_assemblies  # Automatically calculated from core_lattice
}
