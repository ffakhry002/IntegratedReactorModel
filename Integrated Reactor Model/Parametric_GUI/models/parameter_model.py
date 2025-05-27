"""
Parameter Model
Handles parameter data and metadata
"""


class ParameterModel:
    """Model for managing parameters and their metadata"""

    def __init__(self, current_inputs):
        self.current_inputs = current_inputs
        self.available_params = self._get_available_parameters()

    def _get_available_parameters(self):
        """Get list of available parameters from the current inputs with categorization"""
        # Exclude certain parameters that shouldn't be varied parametrically
        excluded_params = {
            'guide_tube_positions',
            'num_assemblies', 'n_guide_tubes', 'core_mesh_dimension',
            'entropy_mesh_dimension', 'outputs_folder', 'pixels'
        }

        # Parameter categories for better organization
        categories = {
            'Core Configuration': [
                'core_power', 'assembly_type', 'fuel_height', 'core_lattice'
            ],
            'Reactor Geometry': [
                'tank_radius', 'reflector_thickness', 'bioshield_thickness',
                'bottom_bioshield_thickness', 'bottom_reflector_thickness',
                'feed_thickness', 'plenum_height', 'top_reflector_thickness',
                'top_bioshield_thickness'
            ],
            'Pin Assembly Parameters': [
                'pin_pitch', 'r_fuel', 'r_clad_inner', 'r_clad_outer', 'n_side_pins'
            ],
            'Plate Assembly Parameters': [
                'fuel_meat_width', 'fuel_plate_width', 'fuel_plate_pitch',
                'fuel_meat_thickness', 'clad_thickness', 'plates_per_assembly',
                'clad_structure_width'
            ],
            'Materials': [
                'coolant_type', 'clad_type', 'fuel_type', 'reflector_material',
                'bioshield_material', 'n%', 'n%E'
            ],
            'Thermal Hydraulics': [
                'reactor_pressure', 'flow_rate', 'T_inlet', 'input_power_density',
                'max_linear_power', 'average_linear_power', 'cos_curve_squeeze',
                'CP_PD_MLP_ALP'
            ],
            'Irradiation Positions': [
                'irradiation_clad', 'irradiation_clad_thickness', 'irradiation_cell_fill'
            ],
            'Monte Carlo Transport': [
                'batches', 'inactive', 'particles', 'energy_structure',
                'thermal_cutoff', 'fast_cutoff', 'power_tally_axial_segments',
                'irradiation_axial_segments', 'Core_Three_Group_Energy_Bins',
                'tally_power', 'element_level_power_tallies'
            ],
            'Depletion Parameters': [
                'deplete_core', 'deplete_assembly', 'deplete_assembly_enhanced',
                'deplete_element', 'deplete_element_enhanced', 'depletion_timestep_units',
                'depletion_particles', 'depletion_batches', 'depletion_inactive',
                'depletion_integrator', 'depletion_chain', 'depletion_nuclides',
                'depletion_timesteps'
            ],
            'Miscellaneous': [
                'parametric_study'
            ]
        }

        available = {}
        for key, value in self.current_inputs.items():
            if key not in excluded_params:
                param_type = type(value).__name__
                if param_type in ['int', 'float', 'str', 'bool']:
                    # Find category for this parameter
                    category = 'Other'
                    for cat_name, cat_params in categories.items():
                        if key in cat_params:
                            category = cat_name
                            break

                    available[key] = {
                        'current_value': value,
                        'type': param_type,
                        'description': self._get_parameter_description(key),
                        'category': category
                    }
                elif key == 'core_lattice' and param_type == 'list':
                    # Special handling for core_lattice
                    available[key] = {
                        'current_value': value,
                        'type': 'core_lattice',
                        'description': self._get_parameter_description(key),
                        'category': 'Core Configuration'
                    }
                elif key == 'depletion_timesteps' and param_type == 'list':
                    # Special handling for depletion_timesteps
                    available[key] = {
                        'current_value': value,
                        'type': 'depletion_timesteps',
                        'description': self._get_parameter_description(key),
                        'category': 'Depletion Parameters'
                    }

        return available

    def _get_parameter_description(self, param_name):
        """Get description for a parameter"""
        descriptions = {
            'core_power': 'Core power [MW]',
            'assembly_type': 'Assembly type: Pin or Plate',
            'tank_radius': 'Reactor tank radius [m]',
            'reflector_thickness': 'Radial reflector thickness [m]',
            'bioshield_thickness': 'Radial bioshield thickness [m]',
            'fuel_height': 'Active fuel height [m]',
            'pin_pitch': 'Pin-to-pin pitch [m]',
            'r_fuel': 'Fuel pellet radius [m]',
            'r_clad_inner': 'Cladding inner radius [m]',
            'r_clad_outer': 'Cladding outer radius [m]',
            'n_side_pins': 'Number of pins per assembly side',
            'fuel_meat_width': 'Fuel meat width [m]',
            'fuel_plate_width': 'Fuel plate width [m]',
            'fuel_plate_pitch': 'Plate-to-plate pitch [m]',
            'fuel_meat_thickness': 'Fuel meat thickness [m]',
            'clad_thickness': 'Cladding thickness [m]',
            'plates_per_assembly': 'Number of plates per assembly',
            'coolant_type': 'Coolant type',
            'clad_type': 'Cladding material',
            'fuel_type': 'Fuel type',
            'reflector_material': 'Reflector material',
            'bioshield_material': 'Bioshield material',
            'n%': 'Standard fuel enrichment [%]',
            'n%E': 'Enhanced fuel enrichment [%]',
            'reactor_pressure': 'System pressure [Pa]',
            'flow_rate': 'Coolant flow rate [m/s]',
            'T_inlet': 'Inlet temperature [K]',
            'batches': 'Number of active batches',
            'inactive': 'Number of inactive batches',
            'particles': 'Particles per batch',
            'energy_structure': 'Energy group structure',
            'thermal_cutoff': 'Thermal/epithermal boundary [eV]',
            'fast_cutoff': 'Epithermal/fast boundary [eV]',
            'core_lattice': 'Core layout configuration [2D array]',
            'depletion_timesteps': 'Depletion time steps [list]'
        }
        return descriptions.get(param_name, f'{param_name}')

    def get_string_options(self, param_name):
        """Get string options for a parameter"""
        options_map = {
            'coolant_type': ['Light Water', 'Heavy Water'],
            'clad_type': ['Al6061', 'Zirc2', 'Zirc4'],
            'fuel_type': ['U3Si2', 'UO2', 'U10Mo'],
            'assembly_type': ['Pin', 'Plate'],
            'reflector_material': ['mgo', 'Graphite', 'Light Water', 'Heavy Water'],
            'bioshield_material': ['Concrete', 'Steel'],
            'energy_structure': ['log1001', 'log501', 'Scale238'],
            'depletion_integrator': ['predictor', 'cecm', 'celi', 'cf4'],
            'depletion_chain': ['casl', 'endfb71'],
            'CP_PD_MLP_ALP': ['CP', 'PD', 'MLP', 'ALP'],
            'depletion_timestep_units': ['MWd/kgHM', 'days'],
            'irradiation_cell_fill': ['Vacuum', 'fill']
        }
        return options_map.get(param_name, [])
