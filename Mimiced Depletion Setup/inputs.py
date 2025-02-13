# Create base inputs dictionary
inputs = {
    ###########################################
    # Pin Cell Geometry
    ###########################################
    "fuel_height": 0.6,            # Active fuel height [m]
    "pin_pitch": 0.0126,          # Pin-to-pin pitch [m]
    "r_fuel": 0.0041,            # Fuel pellet radius [m]
    "r_clad_inner": 0.0042,      # Cladding inner radius [m]
    "r_clad_outer": 0.00475,     # Cladding outer radius [m]

    ###########################################
    # Materials Configuration
    ###########################################
    # Material Choices
    "fuel_type": 'UO2',             # Fuel: 'UO2', 'U3Si2', or 'U10Mo'
    "clad_type": 'Zircaloy',        # Cladding: 'Zircaloy' or 'Al6061'
    "n%": 19.75,                    # Fuel enrichment [%]

    # Temperatures
    "T_fuel": 682.19,                  # Fuel temperature [°C]
    "T_clad": 429.60,                  # Cladding temperature [°C]
    "T_cool": 324.94,                  # Coolant temperature [°C]

    ###########################################
    # Depletion Parameters
    ###########################################
    # Power and timesteps
    "power_density": 74.28,          # Power density [W/gHM]
    # Time Steps Configuration
    "depletion_timestep_units": "MWd/kgHM",  # Units for timesteps: 'MWd/kgHM' or 'days'
    "depletion_timesteps": [
        {"steps": 5, "size": 0.1},  # 5 steps of 0.2 MWd/kgHM or days
        {"steps": 5, "size": 0.5},  # 10 steps of 1.0 MWd/kgHM or days
        {"steps": 5, "size": 2.5},   # 5 steps of 2.0 MWd/kgHM or days
        {"steps": 5, "size": 5},   # 5 steps of 2.0 MWd/kgHM or days
        {"steps": 5, "size": 10},   # 5 steps of 2.0 MWd/kgHM or days
        # {"steps": 5, "size": 0}    # 5 steps of 10.0 MWd/kgHM or days
    ],

    # Transport Settings for Depletion
    "depletion_particles": 1000,       # Particles per batch for depletion
    "depletion_batches": 100,         # Active batches for depletion
    "depletion_inactive": 20,        }
