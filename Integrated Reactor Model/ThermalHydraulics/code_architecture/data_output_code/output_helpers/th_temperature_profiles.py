import os
import numpy as np
import pandas as pd

def extract_temperature_profiles_to_csv(th_system, output_dir=None):
    """Extract temperature profiles and material properties to a CSV file.

    This function saves detailed temperature distributions and material properties
    along the axial length of the fuel element to a CSV file for further analysis.

    Args:
        th_system: THSystem object containing thermal state and material properties
        output_dir (str, optional): Directory to save the CSV file. If None,
            uses the default output directory from th_system. Defaults to None.

    The CSV file includes:
        - Axial position (z)
        - Temperature distributions (coolant, cladding, fuel)
        - Thermal conductivities (fuel, cladding)
        - Heat transfer coefficients
        - Coolant properties
        - Heat generation rates
        - Heat fluxes
        - Critical heat flux data
        - For pin assemblies: gap temperature and conductance
    """
    # Set up output directory
    if output_dir is None:
        # Use the ThermalHydraulics directory as root
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        output_dir = os.path.join(root_dir, 'ThermalHydraulics', th_system.thermal_hydraulics.outputs_folder)

    # Create output directories
    data_dir = os.path.join(output_dir, 'Data')
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, 'temperature_profiles.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Create a dictionary to store all the data
    data = {
        'z (m)': th_system.z,
        'T_coolant (K)': th_system.thermal_state.T_coolant_z,
        'T_clad_out (K)': th_system.thermal_state.T_clad_out_z,
        'T_clad_middle (K)': th_system.thermal_state.T_clad_middle_z,
        'T_clad_in (K)': th_system.thermal_state.T_clad_in_z,
        'T_fuel_surface (K)': th_system.thermal_state.T_fuel_surface_z,
        'T_fuel_centerline (K)': th_system.thermal_state.T_fuel_centerline_z,
        'T_fuel_avg (K)': th_system.thermal_state.T_fuel_avg_z,
        'k_fuel_centerline (W/m-K)': th_system.thermal_state.k_fuel_centerline,
        'k_fuel_bulk (W/m-K)': th_system.thermal_state.k_fuel_bulk,
        'k_clad_out (W/m-K)': th_system.thermal_state.k_clad_out,
        'k_clad_mid (W/m-K)': th_system.thermal_state.k_clad_mid,
        'k_clad_in (W/m-K)': th_system.thermal_state.k_clad_in,
        'h_coolant (W/m^2-K)': th_system.thermal_state.h_coolant,
        'coolant_density (kg/m^3)': th_system.thermal_state.coolant_density,
        'specific_heat_capacity (J/kg-K)': th_system.thermal_state.specific_heat_capacity,
        'thermal_conductivity (W/m-K)': th_system.thermal_state.thermal_conductivity,
        'viscosity (Pa-s)': th_system.thermal_state.viscosity,
        'Q_dot (W/m)': th_system.thermal_state.Q_dot_z,
        'heat_flux (W/m^2)': th_system.thermal_state.heat_flux_z,
        'q_dnb (W/m^2)': th_system.thermal_state.q_dnb,
        'MDNBR': th_system.thermal_state.MDNBR
    }

    # Add gap temperature and conductance for pin assemblies
    if th_system.thermal_hydraulics.assembly_type == 'Pin':
        data.update({
            'T_gap (K)': th_system.thermal_state.T_gap_z,
            'h_gap (W/m^2-K)': th_system.thermal_state.h_gap
        })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Temperature profiles have been written to temperature_profiles.csv")
