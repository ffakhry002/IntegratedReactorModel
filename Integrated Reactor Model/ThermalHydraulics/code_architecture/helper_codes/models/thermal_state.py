import numpy as np
from ..material_properties.coolant_properties import get_coolant_properties, calculate_heat_transfer_coeff_coolant, calculate_mass_flow_rate
from ..material_properties.clad_properties import calculate_k_clad
from ..material_properties.fuel_properties import calculate_k_fuel_vector
from ..material_properties.gap_properties import calculate_h_gap_vector
from ..power_calculations import calculate_Q_dot_z

class ThermalState:
    """Class representing the thermal state of the reactor system.

    This class maintains all temperature distributions, thermal conductivities,
    heat transfer coefficients, coolant properties, power distributions, and
    critical heat flux parameters throughout the calculation process.

    Attributes:
        z_points (int): Number of axial mesh points
        T_coolant_z (np.array): Coolant temperatures along z-axis
        T_clad_out_z (np.array): Outer cladding temperatures along z-axis
        T_clad_middle_z (np.array): Middle cladding temperatures along z-axis
        T_clad_in_z (np.array): Inner cladding temperatures along z-axis
        T_fuel_surface_z (np.array): Fuel surface temperatures along z-axis
        T_fuel_centerline_z (np.array): Fuel centerline temperatures along z-axis
        T_fuel_avg_z (np.array): Average fuel temperatures along z-axis
        T_fuel_y (np.array): 2D array of fuel temperatures (z, y)
        y_fuel (np.array): Radial mesh points
        T_gap_z (np.array): Gap temperatures along z-axis
        k_fuel (np.array): Fuel thermal conductivities
        k_fuel_bulk (np.array): Bulk fuel thermal conductivities
        k_fuel_centerline (np.array): Centerline fuel thermal conductivities
        k_clad_out (np.array): Outer cladding thermal conductivities
        k_clad_mid (np.array): Middle cladding thermal conductivities
        k_clad_in (np.array): Inner cladding thermal conductivities
        h_gap (np.array): Gap heat transfer coefficients
        h_coolant (np.array): Coolant heat transfer coefficients
        coolant_density (np.array): Coolant densities
        specific_heat_capacity (np.array): Coolant specific heat capacities
        thermal_conductivity (np.array): Coolant thermal conductivities
        viscosity (np.array): Coolant viscosities
        Q_dot_z (np.array): Heat generation rates along z-axis
        mass_flow_rate (float): Mass flow rate
        q_dnb (np.array): Critical heat fluxes
        heat_flux_z (np.array): Heat fluxes along z-axis
        MDNBR (float): Minimum departure from nucleate boiling ratio
    """
    def __init__(self, z_points=1000):
        """Initialize the thermal state object.

        Parameters
        ----------
        z_points : int, optional
            Number of axial mesh points for discretization (default: 1000)

        Returns
        -------
        None
        """
        self.z_points = z_points
        # Temperature arrays
        self.T_coolant_z = None
        self.T_clad_out_z = None
        self.T_clad_middle_z = None
        self.T_clad_in_z = None
        self.T_fuel_surface_z = None
        self.T_fuel_centerline_z = None
        self.T_fuel_avg_z = None
        self.T_fuel_y = None
        self.y_fuel = None
        self.T_gap_z = None

        # Thermal conductivities
        self.k_fuel = None
        self.k_fuel_bulk = None
        self.k_fuel_centerline = None
        self.k_clad_out = None
        self.k_clad_mid = None
        self.k_clad_in = None

        # Heat transfer coefficients
        self.h_gap = None
        self.h_coolant = None

        # Coolant properties
        self.coolant_density = None
        self.specific_heat_capacity = None
        self.thermal_conductivity = None
        self.viscosity = None

        # Power and flow
        self.Q_dot_z = None
        self.mass_flow_rate = None

        # Critical heat flux
        self.q_dnb = None
        self.heat_flux_z = None
        self.MDNBR = None

    def initialize_state(self, initial_temp, n_axial, n_radial, th_system):
        """Initialize thermal state with starting values.

        This method sets up initial temperature distributions, thermal conductivities,
        coolant properties, and other parameters needed to start the convergence process.

        Args:
            initial_temp (float): Initial temperature to use for all components
            n_axial (int): Number of axial mesh points
            n_radial (int): Number of radial mesh points
            th_system: THSystem object containing geometry and material information
        """
        # Initialize temperatures
        self.T_coolant_z = np.full(n_axial, initial_temp)
        self.k_fuel = calculate_k_fuel_vector(th_system, np.full((n_axial, n_radial), initial_temp))
        k_clad = calculate_k_clad(th_system, initial_temp)
        self.k_clad_out = np.full(n_axial, k_clad)
        self.k_clad_mid = np.full(n_axial, k_clad)
        self.k_clad_in = np.full(n_axial, k_clad)

        # Initialize coolant properties
        self.coolant_density, self.specific_heat_capacity, self.thermal_conductivity, self.viscosity = get_coolant_properties(th_system, self.T_coolant_z)

        # Initialize mass flow rate
        self.mass_flow_rate = calculate_mass_flow_rate(th_system)

        # Initialize Q_dot_z first
        self.Q_dot_z = calculate_Q_dot_z(th_system)

        # Initialize heat transfer coefficients
        self.h_coolant = calculate_heat_transfer_coeff_coolant(th_system)
        if th_system.thermal_hydraulics.assembly_type == 'Pin':
            self.h_gap = calculate_h_gap_vector(th_system)
