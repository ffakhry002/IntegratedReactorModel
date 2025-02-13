import numpy as np
from scipy.integrate import trapezoid

def get_TH_data(th_system):
    """Get all thermal-hydraulic data as a dictionary.

    This function extracts and calculates various thermal-hydraulic parameters
    from the THSystem object and returns them in a structured dictionary.

    Args:
        th_system: THSystem object containing geometry, material, and thermal state information

    Returns:
        dict: Dictionary containing the following categories of data:
            - Temperature distributions (coolant, cladding, fuel)
            - Thermal conductivities
            - Heat transfer coefficients
            - Coolant properties
            - Heat generation and flow rates
            - Geometry parameters
            - Flow parameters
            - Heat transfer parameters
            - Temperature gradients
            - Critical heat flux data
            - Power and energy balance
            - Reactor parameters
            - Assembly-specific geometry parameters
    """
    data = {}

    # Calculate derived quantities
    if th_system.thermal_hydraulics.assembly_type == 'Pin':
        coolant_to_total_area_ratio = th_system.geometry.coolant_area / (th_system.pin_geometry.pin_pitch**2)
    else:
        coolant_to_total_area_ratio = th_system.geometry.coolant_area / (
            th_system.geometry.coolant_area + th_system.plate_geometry.fuel_plate_thickness * th_system.plate_geometry.fuel_plate_width)

    data = {
        # Temperatures along the fuel rod/plate length (z)
        'T_fuel_centerline_z': th_system.thermal_state.T_fuel_centerline_z,
        'T_fuel_surface_z': th_system.thermal_state.T_fuel_surface_z,
        'T_gap_z': th_system.thermal_state.T_gap_z,
        'T_clad_in_z': th_system.thermal_state.T_clad_in_z,
        'T_clad_middle_z': th_system.thermal_state.T_clad_middle_z,
        'T_clad_out_z': th_system.thermal_state.T_clad_out_z,
        'T_coolant_z': th_system.thermal_state.T_coolant_z,
        'T_fuel_avg_z': th_system.thermal_state.T_fuel_avg_z,
        'T_fuel_y': th_system.thermal_state.T_fuel_y,
        'y_fuel': th_system.thermal_state.y_fuel,

        # Thermal conductivities
        'k_fuel_centerline': th_system.thermal_state.k_fuel_centerline,
        'k_fuel_bulk': th_system.thermal_state.k_fuel_bulk,
        'k_clad_out': th_system.thermal_state.k_clad_out,
        'k_clad_mid': th_system.thermal_state.k_clad_mid,
        'k_clad_in': th_system.thermal_state.k_clad_in,

        # Heat transfer coefficients
        'h_gap': th_system.thermal_state.h_gap,
        'heat_transfer_coeff_coolant': th_system.thermal_state.h_coolant,

        # Coolant properties
        'coolant_density': th_system.thermal_state.coolant_density,
        'specific_heat_capacity_coolant': th_system.thermal_state.specific_heat_capacity,
        'thermal_conductivity_coolant': th_system.thermal_state.thermal_conductivity,
        'viscosity_coolant': th_system.thermal_state.viscosity,

        # Heat generation and flow rates
        'Q_dot_z': th_system.thermal_state.Q_dot_z,
        'mass_flow_rate': th_system.thermal_state.mass_flow_rate,

        # Geometry parameters
        'hydraulic_diameter': th_system.geometry.hydraulic_diameter * 1000,  # mm
        'coolant_area': th_system.geometry.coolant_area * 1e6,  # mm^2
        'coolant_to_total_area_ratio': coolant_to_total_area_ratio,

        # Flow parameters
        'coolant_velocity': th_system.thermal_hydraulics.flow_rate,  # m/s
        'volume_flow_rate': th_system.geometry.coolant_area * th_system.thermal_hydraulics.flow_rate * 1e6,  # cm^3/s
        'mean_mass_flux': np.mean(th_system.thermal_state.coolant_density) * th_system.thermal_hydraulics.flow_rate,  # kg/m^2-s

        # Heat transfer parameters
        'average_linear_heat_rate': np.mean(th_system.thermal_state.Q_dot_z),  # W/m
        'average_heat_flux': np.mean(th_system.thermal_state.heat_flux_z),  # W/m^2
        'maximum_heat_flux': np.max(th_system.thermal_state.heat_flux_z),  # W/m^2

        # Temperature gradients at z = 0 (middle of the fuel rod/plate)
        'fuel_delta_T': th_system.thermal_state.T_fuel_centerline_z[len(th_system.z)//2] - th_system.thermal_state.T_fuel_surface_z[len(th_system.z)//2],  # K
        'cladding_delta_T_inner_to_outer': th_system.thermal_state.T_clad_in_z[len(th_system.z)//2] - th_system.thermal_state.T_clad_out_z[len(th_system.z)//2],  # K
        'cladding_delta_T_inner_to_middle': th_system.thermal_state.T_clad_in_z[len(th_system.z)//2] - th_system.thermal_state.T_clad_middle_z[len(th_system.z)//2],  # K
        'cladding_delta_T_middle_to_outer': th_system.thermal_state.T_clad_middle_z[len(th_system.z)//2] - th_system.thermal_state.T_clad_out_z[len(th_system.z)//2],  # K
        'coolant_delta_T_z0_to_inlet': th_system.thermal_state.T_coolant_z[len(th_system.z)//2] - th_system.thermal_state.T_coolant_z[0],  # K

        # Critical Heat Flux and Safety Margins
        'minimum_DNBR': np.min(th_system.thermal_state.MDNBR),
        'location_of_min_DNBR': th_system.z[np.argmin(th_system.thermal_state.MDNBR)],  # m
        'average_CHF': np.mean(th_system.thermal_state.q_dnb),  # W/m^2
        'minimum_CHF': np.min(th_system.thermal_state.q_dnb),  # W/m^2
        'mean_MDNBR': np.mean(th_system.thermal_state.MDNBR),

        # Power and Energy Balance
        'total_power_per_element': trapezoid(th_system.thermal_state.Q_dot_z, th_system.z),  # W
        'total_power_per_assembly': trapezoid(th_system.thermal_state.Q_dot_z * th_system.geometry.n_elements_per_assembly / 1000, th_system.z),  # kW
        'total_core_power': trapezoid(th_system.thermal_state.Q_dot_z, th_system.z) * th_system.geometry.n_elements_per_assembly * th_system.reactor_power.num_assemblies / 1e6,  # MW
        'coolant_energy_gain_per_element': np.mean(th_system.thermal_state.mass_flow_rate) * np.mean(th_system.thermal_state.specific_heat_capacity) * (th_system.thermal_state.T_coolant_z[-1] - th_system.thermal_hydraulics.T_inlet),  # W
        'energy_balance_error': (trapezoid(th_system.thermal_state.Q_dot_z, th_system.z) - th_system.thermal_state.mass_flow_rate * np.mean(th_system.thermal_state.specific_heat_capacity) * (th_system.thermal_state.T_coolant_z[-1] - th_system.thermal_hydraulics.T_inlet)) / trapezoid(th_system.thermal_state.Q_dot_z, th_system.z) * 100,  # %

        # Additional parameters
        'T_inlet': th_system.thermal_hydraulics.T_inlet,  # K
        'T_outlet': th_system.thermal_state.T_coolant_z[-1],  # K
        'fuel_count': th_system.reactor_power.num_assemblies,
        'number_of_elements_per_assembly': th_system.geometry.n_elements_per_assembly,
        'z': th_system.z,  # m

        # Reactor parameters
        'assembly_type': th_system.thermal_hydraulics.assembly_type,
        'coolant_type': th_system.material.coolant_type,
        'reactor_pressure': th_system.thermal_hydraulics.reactor_pressure,  # Pa
    }

    # Add geometry-specific parameters
    if th_system.thermal_hydraulics.assembly_type == 'Pin':
        data.update({
            'fuel_outer_diameter': 2 * th_system.pin_geometry.r_fuel * 1000,  # mm
            'clad_outer_diameter': 2 * th_system.pin_geometry.r_clad_outer * 1000,  # mm
            'pin_pitch': th_system.pin_geometry.pin_pitch * 1000,  # mm
            'pitch_to_fuel_OD_ratio': th_system.pin_geometry.pin_pitch / (2 * th_system.pin_geometry.r_fuel),
            'pitch_to_clad_OD_ratio': th_system.pin_geometry.pin_pitch / (2 * th_system.pin_geometry.r_clad_outer),
            'n_side_pins': th_system.pin_geometry.n_side_pins,
        })
    else:
        data.update({
            'fuel_meat_thickness': th_system.plate_geometry.fuel_meat_thickness * 1000,  # mm
            'fuel_plate_thickness': th_system.plate_geometry.fuel_plate_thickness * 1000,  # mm
            'fuel_plate_width': th_system.plate_geometry.fuel_plate_width * 1000,  # mm
            'fuel_plate_pitch': th_system.plate_geometry.fuel_plate_pitch * 1000,  # mm
        })

    return data
