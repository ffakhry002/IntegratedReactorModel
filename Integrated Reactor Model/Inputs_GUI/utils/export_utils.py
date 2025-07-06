# Export utility functions

"""
Export Utilities
Functions for exporting reactor configurations
"""
import time
import os


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
    # Flatten the core lattice and count 'F's and 'E's
    flattened = [item for row in core_lattice for item in row]
    num_assemblies_F = flattened.count('F')
    num_assemblies_E = flattened.count('E')
    num_assemblies = num_assemblies_E + num_assemblies_F

    # Get number of guide tubes
    n_guide_tubes = len(guide_tube_positions)

    return num_assemblies, n_guide_tubes


def export_current_values(inputs):
    """Export current parameter values in the exact format of inputs.py.

    Parameters
    ----------
    inputs : dict
        Dictionary of input parameters

    Returns
    -------
    str
        Filepath of the exported file
    """
    # Get the Integrated Reactor Model directory (parent of parent of current file)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))  # Inputs_GUI/utils/
    inputs_gui_dir = os.path.dirname(current_file_dir)  # Inputs_GUI/
    integrated_reactor_dir = os.path.dirname(inputs_gui_dir)  # Integrated Reactor Model/

    filename = f"inputs_{int(time.time())}.py"
    filepath = os.path.join(integrated_reactor_dir, filename)
    export_inputs_to_file(inputs, filepath)
    return filepath


def export_inputs_to_file(inputs, filepath):
    """Export inputs to a file in the exact format of inputs.py.

    Parameters
    ----------
    inputs : dict
        Dictionary of input parameters
    filepath : str
        Full path where to save the file

    Returns
    -------
    None
    """
    # Calculate derived values
    num_assemblies, n_guide_tubes = calculate_derived_values(
        inputs["core_lattice"],
        inputs["guide_tube_positions"]
    )

    # Create output
    output = f'''# Create base inputs dictionary
base_inputs = {{
    ###########################################
    # Parametric Study Configuration
    ###########################################
    "parametric_study": {inputs['parametric_study']},        # Toggle for parametric study mode

    ###########################################
    # Core Configuration
    ###########################################
    # Core Layout
    "core_lattice": [  # C: coolant, F: fuel assembly, E: enriched fuel assembly, I: irradiation position
'''

    # Add core_lattice with proper formatting
    lattice = inputs['core_lattice']
    for row in lattice:
        output += "        [" + ", ".join(f"'{cell}'" for cell in row) + "],\n"

    output += f'''    ],

    # Core Operating Parameters
    "core_power": {inputs['core_power']},              # Core power [MW]
    "assembly_type": '{inputs['assembly_type']}',        # Assembly type: 'Pin' or 'Plate'

    ###########################################
    # Geometry Specifications
    ###########################################
    # Radial Core Geometry
    "tank_radius": {inputs['tank_radius']},            # Reactor tank radius [m]
    "reflector_thickness": {inputs['reflector_thickness']},    # Radial reflector thickness [m]
    "bioshield_thickness": {inputs['bioshield_thickness']},    # Radial bioshield thickness [m]

    # Axial Core Geometry
    "bottom_bioshield_thickness": {inputs['bottom_bioshield_thickness']},  # Bottom bioshield thickness [m]
    "bottom_reflector_thickness": {inputs['bottom_reflector_thickness']},  # Bottom reflector thickness [m]
    "feed_thickness": {inputs['feed_thickness']},            # Feed region thickness [m]
    "plenum_height": {inputs['plenum_height']},             # Plenum height [m]
    "top_reflector_thickness": {inputs['top_reflector_thickness']},   # Top reflector thickness [m]
    "top_bioshield_thickness": {inputs['top_bioshield_thickness']},    # Top bioshield thickness [m]

    "fuel_height": {inputs['fuel_height']},            # Active fuel height [m]

    # Pin Fuel Assembly Parameters
    "pin_pitch": {inputs['pin_pitch']},          # Pin-to-pin pitch [m]
    "r_fuel": {inputs['r_fuel']},            # Fuel pellet radius [m]
    "r_clad_inner": {inputs['r_clad_inner']},      # Cladding inner radius [m]
    "r_clad_outer": {inputs['r_clad_outer']},     # Cladding outer radius [m]
    "n_side_pins": {inputs['n_side_pins']},            # Number of pins per assembly side
    "guide_tube_positions": {inputs['guide_tube_positions']},   # List of (x,y) tuples for guide tube positions

    # Plate Fuel Assembly Parameters
    "fuel_meat_width": {inputs['fuel_meat_width']},      # Fuel meat width [m]
    "fuel_plate_width": {inputs['fuel_plate_width']},       # Fuel plate width [m]
    "fuel_plate_pitch": {inputs['fuel_plate_pitch']},       # Plate-to-plate pitch [m]
    "fuel_meat_thickness": {inputs['fuel_meat_thickness']},    # Fuel meat thickness [m]
    "clad_thickness": {inputs['clad_thickness']},       # coolant to fuel meat in y direction cladding thickness [m]
    "plates_per_assembly": {inputs['plates_per_assembly']},         # Number of plates per assembly
    "clad_structure_width": {inputs['clad_structure_width']},  # Support structure width [m]

    ###########################################
    # Materials Configuration
    ###########################################
    # Material Choices
    "coolant_type": '{inputs['coolant_type']}',     # Coolant: 'Light Water' or 'Heavy Water'
    "clad_type": '{inputs['clad_type']}',             # Cladding: 'Al6061', 'Zirc2', or 'Zirc4'
    "fuel_type": '{inputs['fuel_type']}',             # Fuel: 'U3Si2', 'UO2', or 'U10Mo'
    "reflector_material": "{inputs['reflector_material']}", # Reflector material
    "bioshield_material": "{inputs['bioshield_material']}",  # Bioshield material

    # Fuel Specifications
    "n%": {inputs['n%']},                      # Standard fuel enrichment [%]
    "n%E": {inputs['n%E']},                     # Enhanced fuel enrichment [%]

    ###########################################
    # Thermal Hydraulics Parameters
    ###########################################
    "reactor_pressure": {inputs['reactor_pressure']},          # System pressure [Pa]
    "flow_rate": {inputs['flow_rate']},                   # Coolant flow rate [m/s]
    "T_inlet": {inputs['T_inlet']},          # Inlet temperature [K]

    # Direct Thermal hydraulics calculation mode only
    "input_power_density": {inputs['input_power_density']},        # Power density [kW/L]
    "max_linear_power": {inputs['max_linear_power']},           # Maximum linear power [kW/m]
    "average_linear_power": {inputs['average_linear_power']},       # Average linear power [kW/m]
    "cos_curve_squeeze": {inputs['cos_curve_squeeze']},           # Axial power shape parameter [0-1]
    "CP_PD_MLP_ALP": "{inputs['CP_PD_MLP_ALP']}",          # CP: core power (MW), PD: power density (kW/L)
                                     # MLP: max linear power (kW/m), ALP: avg linear power (kW/m)

    ###########################################
    # Irradiation Position Parameters
    ###########################################
    "irradiation_clad": {inputs['irradiation_clad']},              # Include irradiation position cladding
    "irradiation_clad_thickness": {inputs['irradiation_clad_thickness']}, # Irradiation cladding thickness [m]
    "irradiation_cell_fill": "{inputs['irradiation_cell_fill']}",      # Fill: "Vacuum" or "fill" (Al-water mix)

    ###########################################
    # OpenMC Transport Parameters
    ###########################################
    # Standard Transport Settings
    "batches": int({inputs['batches']}),                   # Number of active batches
    "inactive": int({inputs['inactive']}),                   # Number of inactive batches
    "particles": int({inputs['particles']}),            # Particles per batch
    "energy_structure": '{inputs['energy_structure']}',    # Energy group structure

    # Energy Group Boundaries
    "thermal_cutoff": float({inputs['thermal_cutoff']}),          # Thermal/epithermal boundary [eV]
    "fast_cutoff": float({inputs['fast_cutoff']}),          # Epithermal/fast boundary [eV]

    # Tally Granularity Settings
    "power_tally_axial_segments": {inputs['power_tally_axial_segments']},     # Number of axial segments for power tallies
    "irradiation_axial_segments": {inputs['irradiation_axial_segments']},    # Number of axial segments for irradiation tallies
    "core_mesh_dimension": {inputs['core_mesh_dimension']}, # Mesh resolution for core flux tallies
    "entropy_mesh_dimension": {inputs['entropy_mesh_dimension']},  # Mesh resolution for entropy calculation

    # Additional Tallies
    "Core_Three_Group_Energy_Bins": {inputs['Core_Three_Group_Energy_Bins']}, # True: use three energy bins for core tallies, False: don't tally energy groups
    "tally_power": {inputs['tally_power']},                  # True: calculate power tallies and TH, False: skip power calculations
    "element_level_power_tallies": {inputs['element_level_power_tallies']}, # True: tally power for individual fuel elements, False: tally power for assemblies

    ###########################################
    # Depletion Calculation Parameters
    ###########################################
    # Depletion Scenario Selection (only one should be True)
    "deplete_core": {inputs['deplete_core']},                    # Full core depletion
    "deplete_assembly": {inputs['deplete_assembly']},               # Single assembly with reflective BC
    "deplete_assembly_enhanced": {inputs['deplete_assembly_enhanced']},      # Single enhanced assembly with reflective BC
    "deplete_element": {inputs['deplete_element']},                # Single fuel element with reflective BC
    "deplete_element_enhanced": {inputs['deplete_element_enhanced']},       # Single enhanced fuel element with reflective BC

    # Time Steps Configuration
    "depletion_timestep_units": "{inputs['depletion_timestep_units']}",  # Units for timesteps: 'MWd/kgHM' or 'days'
    "depletion_timesteps": {inputs['depletion_timesteps']},

    # Transport Settings for Depletion
    "depletion_particles": int({inputs['depletion_particles']}),       # Particles per batch for depletion
    "depletion_batches": int({inputs['depletion_batches']}),         # Active batches for depletion
    "depletion_inactive": int({inputs['depletion_inactive']}),         # Inactive batches for depletion

    # Depletion Options
    "depletion_integrator": "{inputs['depletion_integrator']}",  # Integration algorithm
                                         # Options: predictor, cecm, celi, cf4,
                                         #         epcrk4, leqi, siceli, sileqi
    "depletion_chain": "{inputs['depletion_chain']}",       # Depletion chain type ('casl' or 'endfb71')

    # Nuclides to Extract and Plot
    "depletion_nuclides": {inputs['depletion_nuclides']},

    ###########################################
    # Miscellaneous Settings
    ###########################################
    "outputs_folder": "{inputs['outputs_folder']}",  # Base output directory
    "pixels": {inputs['pixels']},            # Plot resolution
}}


############################ UPDATED INPUTS #####################################


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
inputs = {{
    **base_inputs,
    "n_guide_tubes": {n_guide_tubes},  # number of guide tubes per assembly
    "num_assemblies": {num_assemblies}  # Automatically calculated from core_lattice
}}
'''

    # Write to file
    with open(filepath, 'w') as f:
        f.write(output)
