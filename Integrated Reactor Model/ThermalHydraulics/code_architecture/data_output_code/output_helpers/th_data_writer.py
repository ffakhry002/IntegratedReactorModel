import numpy as np
from tabulate import tabulate
import os
import sys

# Add root directory to path to access inputs
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)

from inputs import inputs

def write_TH_results(th_system, output_dir=None):
    """Write thermal-hydraulic analysis results to a text file.

    This function writes a comprehensive report of the thermal-hydraulic analysis
    including geometry parameters, flow parameters, temperature profiles, material
    properties, coolant properties, critical heat flux data, and power/energy balance.

    Args:
        th_system: THSystem object containing all thermal-hydraulic data
        output_dir (str, optional): Directory to save the output file. If None,
            uses the default output directory from th_system. Defaults to None.

    The output file includes the following sections:
    1. Geometry and Flow Parameters
    2. Flow and Heat Transfer Parameters
    3. Temperature Profiles
    4. Temperature Gradients
    5. Thermal Properties
    6. Coolant Properties
    7. Critical Heat Flux and Safety Margins
    8. Power and Energy Balance
    9. Axial Power Profile
    10. Reactor Parameters
    """
    TH_data = th_system.get_data()

    if output_dir is None:
        # Use the ThermalHydraulics directory as root
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        output_dir = os.path.join(root_dir, 'ThermalHydraulics', th_system.thermal_hydraulics.outputs_folder)

    data_dir = os.path.join(output_dir, 'Data')
    data_path = os.path.join(data_dir, 'TH_printed_output.txt')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(data_path):
        os.remove(data_path)

    with open(data_path, 'w') as file:
        def write_section(title):
            file.write(f"\n{title}\n{'=' * len(title)}\n")

        def write_param(name, value, unit=""):
            if isinstance(value, (int, float)):
                file.write(f"{name:<40} {value:>10.3f} {unit}\n")
            else:
                file.write(f"{name:<40} {value:>10} {unit}\n")

        def write_table_header(columns):
            file.write(f"\n{'Parameter':<30}" + "".join(f"{col:>12}" for col in columns) + "\n")
            file.write("-" * (30 + 12 * len(columns)) + "\n")

        def write_table_row(name, values):
            file.write(f"{name:<30}" + "".join(f"{val:12.2f}" for val in values) + "\n")

        write_section("Thermal-Hydraulic Analysis Results")

        write_section("1. Geometry and Flow Parameters")
        if TH_data['assembly_type'] == 'Pin':
            write_param("Fuel outer diameter", TH_data['fuel_outer_diameter'], "mm")
            write_param("Clad outer diameter", TH_data['clad_outer_diameter'], "mm")
            write_param("Pin pitch", TH_data['pin_pitch'], "mm")
            write_param("Pitch to fuel OD ratio", TH_data['pitch_to_fuel_OD_ratio'])
            write_param("Pitch to clad OD ratio", TH_data['pitch_to_clad_OD_ratio'])
        else:
            write_param("Fuel meat thickness", TH_data['fuel_meat_thickness'], "mm")
            write_param("Fuel plate thickness", TH_data['fuel_plate_thickness'], "mm")
            write_param("Fuel plate width", TH_data['fuel_plate_width'], "mm")
            write_param("Fuel plate pitch", TH_data['fuel_plate_pitch'], "mm")
        write_param("Hydraulic diameter (D_h)", TH_data['hydraulic_diameter'], "mm")
        write_param("Coolant area per element", TH_data['coolant_area'], "mm^2")
        write_param("Coolant to total area ratio", TH_data['coolant_to_total_area_ratio'])

        write_section("2. Flow and Heat Transfer Parameters")
        write_param("Coolant velocity", TH_data['coolant_velocity'], "m/s")
        write_param("Volume flow rate per element", TH_data['volume_flow_rate'], "cm^3/s")
        write_param("Mass flow rate", TH_data['mass_flow_rate'], "kg/s")
        write_param("Mass flux (G)", TH_data['mean_mass_flux'], "kg/m^2-s")
        write_param("Average linear heat rate", TH_data['average_linear_heat_rate'], "W/m")
        write_param("Average heat flux", TH_data['average_heat_flux'], "W/m^2")
        write_param("Maximum heat flux", TH_data['maximum_heat_flux'], "W/m^2")

        write_section("3. Temperature Profiles")
        write_table_header(["Min", "Max", "Avg"])
        for param in ['T_coolant_z', 'T_clad_out_z', 'T_clad_middle_z', 'T_clad_in_z',
                     'T_fuel_surface_z', 'T_fuel_centerline_z', 'T_fuel_avg_z']:
            name = param[2:-2].replace('_', ' ').title()
            write_table_row(name, [np.min(TH_data[param]), np.max(TH_data[param]), np.mean(TH_data[param])])
        if TH_data['assembly_type'] == 'Pin':
            write_table_row("T Gap", [np.min(TH_data['T_gap_z']), np.max(TH_data['T_gap_z']), np.mean(TH_data['T_gap_z'])])

        file.write(f"\n")
        write_param("Inlet temperature", TH_data['T_inlet'], "K")
        write_param("Outlet temperature", TH_data['T_outlet'], "K")

        write_section("4. Temperature Gradients at z = 0 (middle of the fuel element)")
        write_param("Fuel ΔT", TH_data['fuel_delta_T'], "K")
        write_param("Cladding ΔT (Inner to Outer)", TH_data['cladding_delta_T_inner_to_outer'], "K")
        write_param("Cladding ΔT (Middle to Outer)", TH_data['cladding_delta_T_middle_to_outer'], "K")
        write_param("Coolant ΔT (z=0 to inlet)", TH_data['coolant_delta_T_z0_to_inlet'], "K")

        write_section("5. Thermal Properties")
        write_table_header(["Min", "Max", "Avg"])
        for param in ['k_fuel_centerline', 'k_fuel_bulk', 'k_clad_out', 'k_clad_mid', 'k_clad_in']:
            name = param.replace('_', ' ').title()
            write_table_row(name, [np.min(TH_data[param]), np.max(TH_data[param]), np.mean(TH_data[param])])
        if TH_data['assembly_type'] == 'Pin':
            write_table_row("H Gap", [np.min(TH_data['h_gap']), np.max(TH_data['h_gap']), np.mean(TH_data['h_gap'])])

        write_section("6. Coolant Properties")
        write_table_header(["Min", "Max", "Avg"])
        coolant_params = ['coolant_density', 'specific_heat_capacity_coolant', 'thermal_conductivity_coolant',
                         'viscosity_coolant', 'heat_transfer_coeff_coolant']
        units = ['kg/m^3', 'J/kg-K', 'W/m-K', 'Pa-s', 'W/m^2-K']
        for param, unit in zip(coolant_params, units):
            name = param.replace('_', ' ').title()
            write_table_row(f"{name} ({unit})", [np.min(TH_data[param]), np.max(TH_data[param]), np.mean(TH_data[param])])

        write_section("7. Critical Heat Flux and Safety Margins")
        if TH_data['assembly_type'] == 'Pin':
            write_param("Minimum DNBR", TH_data['minimum_DNBR'])
            write_param("Location of min DNBR", TH_data['location_of_min_DNBR'], "m")
            write_param("Average CHF (q_dnb)", TH_data['average_CHF'], "W/m^2")
            write_param("Minimum CHF", TH_data['minimum_CHF'], "W/m^2")
            write_param("Maximum heat flux", TH_data['maximum_heat_flux'], "W/m^2")
            write_param("Mean MDNBR", TH_data['mean_MDNBR'])
        else:  # Plate assembly
            file.write("CHF and MDNBR not calculated for plate assembly\n")
            write_param("Maximum heat flux", TH_data['maximum_heat_flux'], "W/m^2")

        write_section("8. Power and Energy Balance")
        write_param("Total power per element", TH_data['total_power_per_element'], "W")
        write_param("Total power per assembly", TH_data['total_power_per_assembly'], "kW")
        write_param("Total core power", TH_data['total_core_power'], "MW")

        # Calculate average power density in kW/L
        fuel_height = TH_data['z'][-1] - TH_data['z'][0]  # m
        if TH_data['assembly_type'] == 'Pin':
            # For pin assembly, use the full assembly volume
            assembly_pitch = TH_data['pin_pitch'] * TH_data['n_side_pins']  # mm
            # Convert to L: (mm->m)² * m * 1000 L/m³
            assembly_volume = (assembly_pitch/1000)**2 * fuel_height * 1000  # L
            avg_power_density_kW_L = TH_data['total_power_per_assembly'] / assembly_volume  # kW/L
        else:  # Plate assembly
            # For plate assembly, use the assembly volume (including all plates)
            assembly_width = TH_data['fuel_plate_width'] + 2*(inputs['clad_structure_width']*1000)  # mm
            plate_pitch = TH_data['fuel_plate_pitch']  # mm
            n_plates = inputs['plates_per_assembly']
            # Convert to L: (mm->m) * (mm->m) * m * n_plates * 1000 L/m³
            assembly_volume = (assembly_width/1000) * (plate_pitch/1000) * fuel_height * n_plates * 1000  # L
            avg_power_density_kW_L = TH_data['total_power_per_assembly'] / assembly_volume  # kW/L

        write_param("Average power density", avg_power_density_kW_L, "kW/L")
        write_param("Coolant energy gain per element", TH_data['coolant_energy_gain_per_element'], "W")
        write_param("Energy balance error", TH_data['energy_balance_error'], "%")
        write_param("Number of assemblies", TH_data['fuel_count'])
        write_param("Number of elements per assembly", TH_data['number_of_elements_per_assembly'])

        write_section("9. Axial Power Profile")
        file.write(f"{'Position (m)':<15}{'Linear Power (W/m)':>20}\n")
        file.write("-" * 35 + "\n")
        for i in range(0, len(TH_data['z']), len(TH_data['z'])//10):
            file.write(f"{TH_data['z'][i]:15.3f}{TH_data['Q_dot_z'][i]:20.2f}\n")

        write_section("10. Reactor Parameters")
        write_param("Assembly type", TH_data['assembly_type'])
        write_param("Coolant type", TH_data['coolant_type'])
        write_param("Reactor pressure", TH_data['reactor_pressure']/1e5, "bar")

        file.write("\nNote: All temperatures are in Kelvin (K) unless otherwise stated.\n")

    print(f"TH information has been written to TH_printed_output.txt")
