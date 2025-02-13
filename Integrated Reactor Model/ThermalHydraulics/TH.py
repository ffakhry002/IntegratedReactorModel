import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from tabulate import tabulate
from scipy.integrate import quad
from scipy.integrate import trapezoid
import csv
from inputs import inputs
from code_architecture.data_output_code.plottingcode.plotting_plate import plot_results_plate, calculate_cladding_temperature_profile
from code_architecture.data_output_code.plottingcode.plotting_coeffs import plot_material_properties, plot_conductivity_vs_temperature
from code_architecture.data_output_code.plottingcode.plotting_pin import plot_results_pin
from code_architecture.data_output_code.plottingcode.plotting_geometry import plot_pin, plot_pin_assembly, plot_plate, plot_plate_assembly
from scipy.interpolate import interp2d

#################### GEOMETRY & GLOBALS ####################

def initialize_globals():
    global pin_pitch, r_fuel, r_clad_inner, r_clad_outer, n_side_pins, n_guide_tubes
    global fuel_meat_width, fuel_plate_width, fuel_plate_pitch, fuel_meat_thickness, plates_per_assembly, clad_structure_width, clad_thickness, gap_width
    global coolant_type, clad_type, fuel_type
    global core_power, num_assemblies, reactor_pressure, flow_rate, T_inlet, fuel_height, cos_curve_squeeze, assembly_type
    global z, output_folder, n_radial, n_axial, Q_dot_z_toggle, input_max_lp, input_avg_lp, input_avg_power_density

    # Pin Fuel Geometry
    pin_pitch = inputs["pin_pitch"]
    r_fuel = inputs["r_fuel"]
    r_clad_inner = inputs["r_clad_inner"]
    r_clad_outer = inputs["r_clad_outer"]
    n_side_pins = inputs["n_side_pins"]
    n_guide_tubes = inputs["n_guide_tubes"]

    # Plate Fuel Geometry
    fuel_meat_width = inputs["fuel_meat_width"]
    fuel_plate_width = inputs["fuel_plate_width"]
    fuel_plate_pitch = inputs["fuel_plate_pitch"]
    fuel_meat_thickness = inputs["fuel_meat_thickness"]
    clad_thickness = inputs["clad_thickness"]
    plates_per_assembly = inputs["plates_per_assembly"]
    clad_structure_width = inputs["clad_structure_width"]

    # Material profile
    coolant_type = inputs["coolant_type"]
    clad_type = inputs["clad_type"]
    fuel_type = inputs["fuel_type"]

    # Reactor Parameters
    core_power = inputs["core_power"]
    num_assemblies = inputs["num_assemblies"]
    reactor_pressure = inputs["reactor_pressure"]
    flow_rate = inputs["flow_rate"]
    T_inlet = inputs["T_inlet"]
    fuel_height = inputs["fuel_height"]
    cos_curve_squeeze = inputs["cos_curve_squeeze"]
    assembly_type = inputs["assembly_type"]
    Q_dot_z_toggle = inputs["CP_PD_MLP_ALP"]
    input_avg_power_density = inputs["input_power_density"]
    input_max_lp = inputs["max_linear_power"]
    input_avg_lp = inputs["average_linear_power"]

    z = np.linspace(-fuel_height/2, fuel_height/2, 1000)
    n_axial = len(z)
    n_radial = 500
    output_folder = inputs["outputs_folder"]
    gap_width = (r_clad_inner-r_fuel)

#################### CONDUCTIVITY COEFFICENTS ####################

def calculate_k_fuel(fuel_temp):
    if fuel_type == 'U3Si2':
        # Linear interpolation
        T1, k1 = 300, 10
        T2, k2 = 1773, 32.0
        return k1 + (k2 - k1) * (fuel_temp - T1) / (T2 - T1)
    elif fuel_type == 'U10Mo':
        return 10.2+(3.51*(10**(-2)))*(fuel_temp-273.15)
    elif fuel_type == 'UO2':
        A1, A2, A3, A4 = 0.0375, 2.165e-4, 1.70e-3, 0.058
        Bu = 0  # GWd/Mt_IHM
        f = (1 + math.exp((fuel_temp - 900) / 80)) ** (-1)
        g = 4.715e9 * fuel_temp ** (-2) * math.exp(-16361 / fuel_temp)
        return 1 / (A1 + A2 * fuel_temp + A3 * Bu + A4 * f) + g

def calculate_k_clad(clad_temp):
    if clad_type == 'Al6061':
        # Linear interpolation
        data_points = [ # W/m-K
            (4, 5.347),
            (50, 62.05),
            (100, 97.7),
            (150, 120.4),
            (200, 136),
            (300, 156.3),
            (400, 171.6),
            (500, 176.8),
            (600, 179.3),
            (673, 180)
        ]

        data_points.sort(key=lambda x: x[0])

        if clad_temp <= data_points[0][0]:
            return data_points[0][1]
        elif clad_temp >= data_points[-1][0]:
            return data_points[-1][1]

        for i in range(len(data_points) - 1):
            T1, k1 = data_points[i]
            T2, k2 = data_points[i + 1]

            if T1 <= clad_temp <= T2:
                slope = (k2 - k1) / (T2 - T1)
                k = k1 + slope * (clad_temp - T1)
                return k

    elif clad_type == 'Zirc2':
        k = 0.138 - 3.90e-5 * clad_temp + 1.184e-7 * clad_temp ** 2
        return k * 100  # Convert to W/m-K
    elif clad_type == 'Zirc4':
        k = 0.113 + 2.25e-5 * clad_temp + 0.725e-8 * clad_temp ** 2
        return k * 100  # Convert to W/m-K

def calculate_h_gap_vector(Q_dot_z):
    power_W_cm = Q_dot_z / 100
    data_points = {
        0.00005: [(170, 1.9), (330, 7.0), (380, 11.0), (450, 11.5), (500, 11.5)],
        0.0001:  [(170, 0.48), (330, 0.85), (380, 1.1), (450, 1.7), (500, 1.7)],
        0.0002:  [(170, 0.275), (330, 0.385), (380, 0.415), (450, 0.5), (500, 0.585)],
        0.00025: [(170, 0.2), (330, 0.3), (380, 0.33), (450, 0.385), (500, 0.4)],
        0.0005:  [(170, 0.35), (330, 0.5), (380, 0.75), (450, 0.95), (500, 0.95)]
    }

    def interpolate_single_point(power, gap):
        gap_widths = sorted(data_points.keys())
        if gap <= gap_widths[0]:
            lower_gap, upper_gap = gap_widths[0], gap_widths[0]
        elif gap >= gap_widths[-1]:
            lower_gap, upper_gap = gap_widths[-1], gap_widths[-1]
        else:
            for i in range(len(gap_widths) - 1):
                if gap_widths[i] <= gap < gap_widths[i + 1]:
                    lower_gap, upper_gap = gap_widths[i], gap_widths[i + 1]
                    break

        def interpolate_power(power, gap_data):
            for i in range(len(gap_data) - 1):
                t1, h1 = gap_data[i]
                t2, h2 = gap_data[i + 1]

                if t1 <= power <= t2:
                    # Linear interpolation for power
                    slope = (h2 - h1) / (t2 - t1)
                    return h1 + slope * (power - t1)
            return gap_data[0][1] if power < gap_data[0][0] else gap_data[-1][1]

        h_gap_lower = interpolate_power(power, data_points[lower_gap])
        h_gap_upper = interpolate_power(power, data_points[upper_gap])

        if lower_gap != upper_gap:
            gap_fraction = (gap - lower_gap) / (upper_gap - lower_gap)
            h_gap = h_gap_lower + gap_fraction * (h_gap_upper - h_gap_lower)
        else:
            h_gap = h_gap_lower

        return h_gap * 10000  # Convert to W/m^2-K

    vectorized_interpolate = np.vectorize(interpolate_single_point)
    return vectorized_interpolate(power_W_cm, gap_width)

def heat_transfer_coeff_coolant_calculation_vector(coolant_density, specific_heat_capacity_coolant, thermal_conductivity_coolant, viscosity_coolant, flow_rate):
    global D_h, fuel_plate_thickness

    if assembly_type == 'Plate':
        fuel_plate_thickness = fuel_meat_thickness + 2*clad_thickness
        D_h = 4 * fuel_plate_width*(fuel_plate_pitch-fuel_plate_thickness)/(2*(fuel_plate_width+fuel_plate_pitch-fuel_plate_thickness))
        Pr = specific_heat_capacity_coolant * viscosity_coolant / thermal_conductivity_coolant
        Re = coolant_density * flow_rate * D_h / viscosity_coolant
        heat_transfer_coeff_coolant = 0.036 * Re**0.76 * Pr**(1/3) * thermal_conductivity_coolant / D_h
        return heat_transfer_coeff_coolant

    elif assembly_type == 'Pin':
        coolant_area = pin_pitch**2 - np.pi * r_clad_outer**2
        wetted_perimeter = 2 * np.pi * r_clad_outer
        D_h = 4 * coolant_area / wetted_perimeter
        Pr = specific_heat_capacity_coolant * viscosity_coolant / thermal_conductivity_coolant
        Re = coolant_density * flow_rate * D_h / viscosity_coolant
        heat_transfer_coeff_coolant = 0.023 * Re**0.8 * Pr**0.4 * thermal_conductivity_coolant / D_h
        return heat_transfer_coeff_coolant

def calculate_k_fuel_vector(T_fuel):
    if T_fuel.ndim == 1:
        return np.vectorize(calculate_k_fuel)(T_fuel)
    elif T_fuel.ndim == 2:
        return np.apply_along_axis(lambda x: np.vectorize(calculate_k_fuel)(x), axis=1, arr=T_fuel)
    else:
        raise ValueError("T_fuel must be 1D or 2D array")

def calculate_k_clad_vector(T_clad_average_z):
    return np.vectorize(calculate_k_clad)(T_clad_average_z)

#################### COOLANT PROPERTIES ####################

def get_coolant_properties_vector(coolant_type, temp_vector):
    data = None
    if coolant_type == 'Light Water':
        data = pd.read_csv('code_architecture/helper_codes/material_properties/coolant_data/isobaric_lightwater_1atm.csv')
    elif coolant_type == 'Heavy Water':
        data = pd.read_csv('code_architecture/helper_codes/material_properties/coolant_data/isobaric_heavywater_1atm.csv')

    data['Temperature [K]'] = data['Temperature [K]'].astype(float)

    coolant_density = np.interp(temp_vector, data['Temperature [K]'], data['Density [g/ml]']) * 1000
    specific_heat_capacity_coolant = np.interp(temp_vector, data['Temperature [K]'], data['Specific Heat Capacity [J/kg.K]'])
    thermal_conductivity_coolant = np.interp(temp_vector, data['Temperature [K]'], data['Thermal Conductivity [W/m.K]'])
    viscosity_coolant = np.interp(temp_vector, data['Temperature [K]'], data['Dynamic Viscosity [Pa.s]'])
    return coolant_density, specific_heat_capacity_coolant, thermal_conductivity_coolant, viscosity_coolant

def get_saturated_values():
    def Ts(P):
        Tsat = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                         210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 374.15]) + 273.15
        Psat = np.array([0.006112, 0.012271, 0.023368, 0.042418, 0.073750, 0.12335, 0.19919, 0.31161, 0.47358, 0.70109, 1.01325,
                         1.4327, 1.9854, 2.7011, 3.6136, 4.7597, 6.1804, 7.9202, 10.027, 12.553, 15.550, 19.080, 23.202, 27.979,
                         33.480, 39.776, 46.941, 55.052, 64.191, 74.449, 85.917, 98.694, 112.89, 128.64, 146.08, 165.37, 186.74,
                         210.53, 221.2]) * 1e5  # Convert to Pa
        return np.interp(P, Psat, Tsat)

    def hfg(T):
        Tsat = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                         210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 374.15]) + 273.15
        deltah = np.array([2501, 2477, 2454, 2430, 2406, 2382, 2358, 2333, 2308, 2283, 2257, 2230, 2203, 2174, 2145, 2114, 2083,
                           2050, 2015, 1979, 1941, 1900, 1857, 1811, 1762, 1709, 1652, 1590, 1523, 1450, 1370, 1282, 1184, 1074,
                           948, 801, 623, 374, 0]) * 1000  # Convert to J/kg
        return np.interp(T, Tsat, deltah)

    def muf(T):
        Tsat = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                         210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 374.15]) + 273.15
        muliq = np.array([1786, 1304, 1002, 798.3, 653.9, 547.8, 467.3, 404.8, 355.4, 315.6, 283.1, 254.8, 231.0, 210.9, 194.1, 179.8,
                          167.7, 157.4, 148.5, 140.7, 133.9, 127.9, 122.4, 117.5, 112.9, 108.7, 104.8, 101.1, 97.5, 94.1, 90.7,
                          87.2, 83.5, 79.5, 75.4, 69.4, 62.1, 51.8, 41.4]) * 1e-6  # Already in Pa*s
        return np.interp(T, Tsat, muliq)

    def Cpf(T):
        Tsat = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                         210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 374.15]) + 273.15
        Cpliq = np.array([4.218, 4.194, 4.182, 4.179, 4.179, 4.181, 4.185, 4.191, 4.198, 4.207, 4.218, 4.230, 4.244, 4.262,
                          4.282, 4.306, 4.334, 4.366, 4.403, 4.446, 4.494, 4.550, 4.613, 4.685, 4.769, 4.866, 4.985, 5.134,
                          5.307, 5.520, 5.794, 6.143, 6.604, 7.241, 8.225, 10.07, 15, 55, 1e6]) * 1000  # Convert to J/(kg*K)
        return np.interp(T, Tsat, Cpliq)

    T_sat = Ts(reactor_pressure)
    h_fg_value = hfg(T_sat)
    mu_f_value = muf(T_sat)
    Cp_f_value = Cpf(T_sat)

    return T_sat, h_fg_value, mu_f_value, Cp_f_value

def calculate_mass_flow_rate(coolant_density):
    global coolant_area, mass_flow_rate
    if assembly_type == 'Plate':
        fuel_plate_thickness = fuel_meat_thickness + 2*clad_thickness
        coolant_area = fuel_plate_width * (fuel_plate_pitch-fuel_plate_thickness)
    elif assembly_type == 'Pin':
        coolant_area = pin_pitch**2 - np.pi * r_clad_outer**2
    mass_flow_rate = coolant_area * coolant_density * flow_rate
    return mass_flow_rate

#################### HEAT GENERATION AND FLUXES ####################

def calculate_Q_dot_z(z, fuel_height):
    global TH_toggle, lattice_array, n_side_pins, avg_power_density_kW_L, assembly_pitch, fuel_count, n_elements_per_assembly, Q_dot_z

    if assembly_type == 'Plate':
        n_elements_per_assembly = plates_per_assembly
        single_assembly_volume = (fuel_plate_width + 2 * clad_structure_width) * fuel_plate_pitch * fuel_height * plates_per_assembly
        total_assembly_volume = single_assembly_volume * num_assemblies
    elif assembly_type == 'Pin':
        n_elements_per_assembly = n_side_pins**2 - n_guide_tubes
        single_assembly_volume = (pin_pitch * n_side_pins)**2 * fuel_height
        total_assembly_volume = single_assembly_volume * num_assemblies  # m^3

    if Q_dot_z_toggle == 'PD':
        avg_power_density_kW_L = input_avg_power_density
        avg_power_density_W_m3 = avg_power_density_kW_L * 1e6  # Convert to W/m^3
    else:
        avg_power_density_kW_L = core_power / total_assembly_volume  # MW/m^3 = kW/L
        avg_power_density_W_m3 = avg_power_density_kW_L * 1e6  # W/m^3

    # Calculate total power per assembly and power per element
    total_power_per_assembly = avg_power_density_W_m3 * single_assembly_volume  # W
    power_per_element = total_power_per_assembly / n_elements_per_assembly  # Integrated power of element

    # Toggle-specific calculations for Average and Peak Linear Power
    if Q_dot_z_toggle == 'CP' or Q_dot_z_toggle == 'PD':
        Average_linear_power = power_per_element / fuel_height
        Peak_linear_power = Average_linear_power * np.pi / 2
    elif Q_dot_z_toggle == 'ALP':
        Average_linear_power = input_avg_lp * 1e3
        Peak_linear_power = Average_linear_power * np.pi / 2
    elif Q_dot_z_toggle == 'MLP':
        Peak_linear_power = input_max_lp * 1e3
        Average_linear_power = Peak_linear_power * 2 / np.pi

    # Calculate the original cosine curve
    original_curve = np.cos(np.pi * z / fuel_height)
    original_power = trapezoid(original_curve, z)

    # Apply the curve squeeze
    adjusted_curve = (1 - cos_curve_squeeze) * np.cos(np.pi * z / fuel_height) + cos_curve_squeeze

    new_power = trapezoid(adjusted_curve, z)
    adjusted_curve *= original_power / new_power

    if Q_dot_z_toggle == 'MLP':
        adjusted_curve = adjusted_curve / np.max(adjusted_curve)  # Normalize the curve
        Q_dot_z = Peak_linear_power * adjusted_curve  # Scale the curve to maintain peak power
    else:
        Q_dot_z = Peak_linear_power * adjusted_curve

    return Q_dot_z

def calculate_q_dnb_vector(T_coolant_z):
    global reactor_pressure, mass_flow_rate, coolant_area, D_h
    T_sat, h_fg, mu_f, Cp_f = get_saturated_values()
    x_e = - Cp_f * (T_sat - T_coolant_z) / h_fg  # Dimensionless
    G = mass_flow_rate / coolant_area
    P_MPa = reactor_pressure * 1e-6  # Convert Pa to MPa
    K_tong = (1.76 - 7.433*x_e + 12.222*(x_e**2)) * (1 - (52.3 + 80*x_e - 50*(x_e**2)) / (60.5 + (10*P_MPa)**1.4))
    q_dnb = K_tong * (G**0.4) * (mu_f**0.6) * h_fg / (D_h**0.6)
    return q_dnb

def get_MDNBR(T_coolant_z):
    q_dnb_vector=calculate_q_dnb_vector(T_coolant_z)
    Q_dot_z = calculate_Q_dot_z(z, fuel_height)
    heat_flux_z=Q_dot_z/(2*np.pi*r_clad_outer)
    MDNBR = q_dnb_vector/heat_flux_z
    return q_dnb_vector,heat_flux_z,MDNBR

#################### TEMPERATURE CALCS ####################

def calculate_temperature_points_pins(Q_dot_z, z, mass_flow_rate, specific_heat_capacity_coolant, heat_transfer_coeff_coolant, k_clad_out, k_clad_mid, k_clad_in, k_fuel, h_gap):
    global r_fuel_mesh
    integral_Q_dot_z = integrate.cumulative_trapezoid(Q_dot_z, z, initial=0)
    T_coolant_z = (1 / (mass_flow_rate * specific_heat_capacity_coolant)) * integral_Q_dot_z + T_inlet
    T_clad_out = T_coolant_z + Q_dot_z / (2 * np.pi * r_clad_outer * heat_transfer_coeff_coolant)
    r_clad_middle = (r_clad_inner + r_clad_outer) / 2
    T_clad_middle = T_clad_out + Q_dot_z / (2 * np.pi * k_clad_out) * np.log(r_clad_outer / r_clad_middle)
    T_clad_in = T_clad_middle + Q_dot_z / (2 * np.pi * k_clad_mid) * np.log(r_clad_middle / r_clad_inner)
    T_fuel_surface = T_clad_in + Q_dot_z / (2 * np.pi * r_fuel * h_gap)

    q_v = Q_dot_z / (np.pi * r_fuel**2)
    r_fuel_mesh = np.linspace(0, r_fuel, n_radial)
    T_fuel_y = np.zeros((len(z), len(r_fuel_mesh)))
    for i in range(len(z)):
        T_fuel_y[i] = q_v[i] / (4 * k_fuel[i]) * r_fuel**2 * (1 - (r_fuel_mesh / r_fuel)**2) + T_fuel_surface[i]

    T_fuel_centerline = T_fuel_y[:, 0]
    T_fuel_avg = np.mean(T_fuel_y, axis=1)

    return T_coolant_z, T_clad_out, T_clad_middle, T_clad_in, T_fuel_surface, T_fuel_centerline, T_fuel_avg, T_fuel_y, r_fuel_mesh

def calculate_temperature_points_plates(Q_dot_z, z, mass_flow_rate, specific_heat_capacity_coolant, heat_transfer_coeff_coolant, k_clad, k_fuel):
    integral_Q_dot_z = integrate.cumulative_trapezoid(Q_dot_z, z, initial=0)
    T_coolant_z = (1 / (mass_flow_rate * specific_heat_capacity_coolant)) * integral_Q_dot_z + T_inlet
    T_clad_out = T_coolant_z + Q_dot_z / (2 * fuel_plate_width * heat_transfer_coeff_coolant)
    T_clad_middle = T_clad_out + Q_dot_z / (2*fuel_plate_width) * (clad_thickness/(2*k_clad))
    T_clad_in = T_clad_middle + Q_dot_z / (2*fuel_plate_width) * (clad_thickness/(2*k_clad))

    Q_triple_dot = Q_dot_z / (fuel_meat_thickness * fuel_meat_width)
    y_fuel = np.linspace(0, fuel_meat_thickness / 2, n_radial)
    T_fuel_y = np.zeros((len(z), len(y_fuel)))

    for i in range(len(z)):
        for j in range(len(y_fuel)):
            T_fuel_y[i, j] = Q_triple_dot[i] / k_fuel[i,j] * ((fuel_meat_thickness / 2) ** 2 - y_fuel[j] ** 2) + T_clad_in[i]

    T_fuel_centerline = T_fuel_y[:, 0]
    T_fuel_avg = np.mean(T_fuel_y, axis=1)
    T_fuel_surface = T_fuel_y[:, -1]

    return T_coolant_z, T_clad_out, T_clad_middle, T_clad_in, T_fuel_surface, T_fuel_centerline, T_fuel_avg, T_fuel_y, y_fuel

#################### CONVERGENCE CODE ####################

def single_iteration(T_coolant_z, k_fuel, k_clad_out, k_clad_mid, k_clad_in, h_gap=None):
    coolant_density, specific_heat_capacity_coolant, thermal_conductivity_coolant, viscosity_coolant = get_coolant_properties_vector(coolant_type, T_coolant_z)
    heat_transfer_coeff_coolant = heat_transfer_coeff_coolant_calculation_vector(coolant_density, specific_heat_capacity_coolant, thermal_conductivity_coolant, viscosity_coolant, flow_rate)

    Q_dot_z = calculate_Q_dot_z(z, fuel_height)
    mass_flow_rate = calculate_mass_flow_rate(np.mean(coolant_density))

    if assembly_type == 'Pin':
        T_coolant_z, T_clad_out_z, T_clad_middle_z, T_clad_in_z, T_fuel_surface_z, T_fuel_centerline_z, T_fuel_avg_z, T_fuel_y, y_fuel = calculate_temperature_points_pins(
            Q_dot_z, z, mass_flow_rate, specific_heat_capacity_coolant, heat_transfer_coeff_coolant, k_clad_out, k_clad_mid, k_clad_in, k_fuel, h_gap)
        return T_coolant_z, T_clad_out_z, T_clad_middle_z, T_clad_in_z, T_fuel_surface_z, T_fuel_centerline_z, T_fuel_avg_z, T_fuel_y, y_fuel, Q_dot_z, mass_flow_rate, heat_transfer_coeff_coolant
    elif assembly_type == 'Plate':
        T_coolant_z, T_clad_out_z, T_clad_middle_z, T_clad_in_z, T_fuel_surface_z, T_fuel_centerline_z, T_fuel_avg_z, T_fuel_y, y_fuel = calculate_temperature_points_plates(
            Q_dot_z, z, mass_flow_rate, specific_heat_capacity_coolant, heat_transfer_coeff_coolant, k_clad_out, k_fuel)
        return T_coolant_z, T_clad_out_z, T_clad_middle_z, T_clad_in_z, T_fuel_surface_z, T_fuel_centerline_z, T_fuel_avg_z, T_fuel_y, y_fuel, Q_dot_z, mass_flow_rate, heat_transfer_coeff_coolant

def converge_coolant(initial_T_coolant_z, k_fuel, k_clad_out, k_clad_mid, k_clad_in, h_gap=None, tolerance=0.001, max_iterations=100):
    current_T_coolant_z = initial_T_coolant_z
    iteration = 0
    convergence_history = []
    print("Starting coolant convergence process:")
    headers = ["Iteration", "Min Temp (K)", "Max Temp (K)", "Avg Temp (K)", "Max Diff (K)"]
    rows = [headers]

    while iteration < max_iterations:
        coolant_density, specific_heat_capacity_coolant, thermal_conductivity_coolant, viscosity_coolant = get_coolant_properties_vector(coolant_type, current_T_coolant_z)

        if assembly_type == 'Pin':
            new_T_coolant_z, _, _, _, _, _, _, _, _, _, _, _ = single_iteration(
                current_T_coolant_z, k_fuel, k_clad_out, k_clad_mid, k_clad_in, h_gap
            )
        elif assembly_type == 'Plate':
            new_T_coolant_z, _, _, _, _, _, _, _, _, _, _, _ = single_iteration(
                current_T_coolant_z, k_fuel, k_clad_out, k_clad_mid, k_clad_in
            )

        max_diff = np.max(np.abs(new_T_coolant_z - current_T_coolant_z))

        row = [iteration + 1, f"{np.min(new_T_coolant_z):.2f}", f"{np.max(new_T_coolant_z):.2f}", f"{np.mean(new_T_coolant_z):.2f}", f"{max_diff:.6f}"]
        rows.append(row)
        convergence_history.append(max_diff)

        if max_diff < tolerance:
            print(tabulate(rows, headers="firstrow", tablefmt="grid"))
            print(f"\nCoolant converged after {iteration + 1} iterations")
            return new_T_coolant_z, np.mean(new_T_coolant_z)

        current_T_coolant_z = new_T_coolant_z
        iteration += 1

    print(tabulate(rows, headers="firstrow", tablefmt="grid"))
    print(f"\nWarning: Coolant did not converge after {max_iterations} iterations")
    return current_T_coolant_z, np.mean(current_T_coolant_z)


def converge_cladding(T_coolant_z, k_fuel, initial_k_clad, h_gap=None, tolerance=0.001, max_iterations=100):
    current_k_clad_out = np.full_like(T_coolant_z, initial_k_clad)
    current_k_clad_mid = np.full_like(T_coolant_z, initial_k_clad)
    current_k_clad_in = np.full_like(T_coolant_z, initial_k_clad)
    iteration = 0

    print("\nStarting cladding convergence process:")

    rows_out = [["Iteration", "Min Temp (K)", "Max Temp (K)", "Avg Temp (K)", "Coeff (W/m-K)", "Max Temp Diff (K)"]]
    rows_mid = [["Iteration", "Min Temp (K)", "Max Temp (K)", "Avg Temp (K)", "Coeff (W/m-K)", "Max Temp Diff (K)"]]
    rows_in = [["Iteration", "Min Temp (K)", "Max Temp (K)", "Avg Temp (K)", "Coeff (W/m-K)", "Max Temp Diff (K)"]]

    while iteration < max_iterations:
        if assembly_type == 'Pin':
            _, T_clad_out_z, T_clad_middle_z, T_clad_in_z, _, _, _, _, _, _, _, _ = single_iteration(T_coolant_z, k_fuel, current_k_clad_out, current_k_clad_mid, current_k_clad_in, h_gap)
        elif assembly_type == 'Plate':
            _, T_clad_out_z, T_clad_middle_z, T_clad_in_z, _, _, _, _, _, _, _, _ = single_iteration(T_coolant_z, k_fuel, current_k_clad_out, current_k_clad_mid, current_k_clad_in)

        new_k_clad_out = calculate_k_clad_vector(T_clad_out_z)
        new_k_clad_mid = calculate_k_clad_vector(T_clad_middle_z)
        new_k_clad_in = calculate_k_clad_vector(T_clad_in_z)

        max_temp_diff_out = np.max(np.abs(new_k_clad_out - current_k_clad_out))
        max_temp_diff_mid = np.max(np.abs(new_k_clad_mid - current_k_clad_mid))
        max_temp_diff_in = np.max(np.abs(new_k_clad_in - current_k_clad_in))

        row_out = [iteration + 1, f"{np.min(T_clad_out_z):.2f}", f"{np.max(T_clad_out_z):.2f}",
                   f"{np.mean(T_clad_out_z):.2f}", f"{np.mean(new_k_clad_out):.2f}", f"{max_temp_diff_out:.6f}"]
        row_mid = [iteration + 1, f"{np.min(T_clad_middle_z):.2f}", f"{np.max(T_clad_middle_z):.2f}",
                   f"{np.mean(T_clad_middle_z):.2f}", f"{np.mean(new_k_clad_mid):.2f}", f"{max_temp_diff_mid:.6f}"]
        row_in = [iteration + 1, f"{np.min(T_clad_in_z):.2f}", f"{np.max(T_clad_in_z):.2f}",
                  f"{np.mean(T_clad_in_z):.2f}", f"{np.mean(new_k_clad_in):.2f}", f"{max_temp_diff_in:.6f}"]

        rows_out.append(row_out)
        rows_mid.append(row_mid)
        rows_in.append(row_in)

        if max(max_temp_diff_out, max_temp_diff_mid, max_temp_diff_in) < tolerance:
            print("Outer Cladding Convergence:")
            print(tabulate(rows_out, headers="firstrow", tablefmt="grid"))
            print("\nMiddle Cladding Convergence:")
            print(tabulate(rows_mid, headers="firstrow", tablefmt="grid"))
            print("\nInner Cladding Convergence:")
            print(tabulate(rows_in, headers="firstrow", tablefmt="grid"))
            print(f"\nCladding converged after {iteration + 1} iterations")
            return T_clad_out_z, T_clad_middle_z, T_clad_in_z, new_k_clad_out, new_k_clad_mid, new_k_clad_in

        current_k_clad_out, current_k_clad_mid, current_k_clad_in = new_k_clad_out, new_k_clad_mid, new_k_clad_in
        iteration += 1

    print("Outer Cladding Convergence:")
    print(tabulate(rows_out, headers="firstrow", tablefmt="grid"))
    print("\nMiddle Cladding Convergence:")
    print(tabulate(rows_mid, headers="firstrow", tablefmt="grid"))
    print("\nInner Cladding Convergence:")
    print(tabulate(rows_in, headers="firstrow", tablefmt="grid"))
    print(f"\nWarning: Cladding did not converge after {max_iterations} iterations")
    return T_clad_out_z, T_clad_middle_z, T_clad_in_z, current_k_clad_out, current_k_clad_mid, current_k_clad_in

def converge_fuel(T_coolant_z, T_clad_out_z, T_clad_middle_z, T_clad_in_z, k_clad_out, k_clad_mid, k_clad_in, initial_k_fuel, h_gap=None, tolerance=0.001, max_iterations=100):

    current_k_fuel = np.copy(initial_k_fuel)
    iteration = 0

    print("\nStarting fuel convergence process:")
    rows = [["Iteration", "Min Temp (K)", "Max Temp (K)", "Avg Temp (K)", "Avg Coeff (W/m-K)", "Max Temp Diff (K)"]]

    while iteration < max_iterations:
        if assembly_type == 'Pin':
            _, _, _, _, _, _, T_fuel_avg, T_fuel_y, _, _, _, _ = single_iteration(
                T_coolant_z, current_k_fuel, k_clad_out, k_clad_mid, k_clad_in, h_gap
            )
        elif assembly_type == 'Plate':
            _, _, _, _, _, _, T_fuel_avg, T_fuel_y, _, _, _, _ = single_iteration(
                T_coolant_z, current_k_fuel, k_clad_out, k_clad_mid, k_clad_in
            )

        new_k_fuel = calculate_k_fuel_vector(T_fuel_y)
        max_temp_diff = np.max(np.abs(new_k_fuel - current_k_fuel))

        row = [iteration + 1,
               f"{np.min(T_fuel_y):.2f}",
               f"{np.max(T_fuel_y):.2f}",
               f"{np.mean(T_fuel_avg):.2f}",
               f"{np.mean(new_k_fuel):.2f}",
               f"{max_temp_diff:.6f}"]

        rows.append(row)

        if max_temp_diff < tolerance:
            print(tabulate(rows, headers="firstrow", tablefmt="grid"))
            print(f"\nFuel converged after {iteration + 1} iterations")
            return T_fuel_avg, T_fuel_y, new_k_fuel

        current_k_fuel = new_k_fuel
        iteration += 1

    print(tabulate(rows, headers="firstrow", tablefmt="grid"))
    print(f"\nWarning: Fuel did not converge after {max_iterations} iterations")
    return T_fuel_avg, T_fuel_y, current_k_fuel

def converge_model(initial_temp, initial_k_fuel, initial_k_clad, h_gap=None, tolerance=0.001, max_iterations=100):
    initial_T_coolant_z = np.full(len(z), initial_temp)

    T_coolant_z, avg_temp_coolant = converge_coolant(initial_T_coolant_z, initial_k_fuel, initial_k_clad, initial_k_clad, initial_k_clad, h_gap, tolerance, max_iterations)
    T_clad_out_z, T_clad_middle_z, T_clad_in_z, k_clad_out, k_clad_mid, k_clad_in = converge_cladding(T_coolant_z, initial_k_fuel, initial_k_clad, h_gap, tolerance, max_iterations)
    T_fuel_avg, T_fuel_y, k_fuel = converge_fuel(T_coolant_z, T_clad_out_z, T_clad_middle_z, T_clad_in_z, k_clad_out, k_clad_mid, k_clad_in, initial_k_fuel, h_gap, tolerance, max_iterations)

    if assembly_type == 'Pin':
        T_coolant_z, T_clad_out_z, T_clad_middle_z, T_clad_in_z, _, _, _, T_fuel_y, y_fuel, Q_dot_z, mass_flow_rate, heat_transfer_coeff_coolant = single_iteration(
            T_coolant_z, k_fuel, k_clad_out, k_clad_mid, k_clad_in, h_gap)
    else:
        T_coolant_z, T_clad_out_z, T_clad_middle_z, T_clad_in_z, _, _, _, T_fuel_y, y_fuel, Q_dot_z, mass_flow_rate, heat_transfer_coeff_coolant = single_iteration(
            T_coolant_z, k_fuel, k_clad_out, k_clad_mid, k_clad_in)

    k_fuel_bulk = np.mean(k_fuel, axis=1)
    Q_dot_z = calculate_Q_dot_z(z, fuel_height)

    if assembly_type == 'Pin':
        h_gap = calculate_h_gap_vector(Q_dot_z)
    else:
        h_gap = None

    common_returns = (T_coolant_z, T_clad_out_z, T_clad_middle_z, T_clad_in_z, Q_dot_z,
                      mass_flow_rate, k_fuel_bulk, k_clad_out, k_clad_mid, k_clad_in)

    return (*common_returns, T_fuel_y, y_fuel, k_fuel, h_gap)

#################### DATA PREPARATION AND EXTRACTION ####################

def get_TH_data(plotting=False):
    initialize_globals()
    initial_temp = 310  # K
    initial_k_fuel = np.full((n_axial, n_radial), calculate_k_fuel(initial_temp))
    initial_k_clad = calculate_k_clad(initial_temp)

    Q_dot_z = calculate_Q_dot_z(z, fuel_height)
    h_gap = calculate_h_gap_vector(Q_dot_z) if assembly_type == 'Pin' else None

    T_coolant_z, T_clad_out_z, T_clad_middle_z, T_clad_in_z, Q_dot_z, mass_flow_rate, k_fuel_bulk, k_clad_out, k_clad_mid, k_clad_in, T_fuel_y, y_fuel, k_fuel, h_gap = converge_model(
        initial_temp, initial_k_fuel, initial_k_clad, h_gap=h_gap, tolerance=0.001, max_iterations=100)

    # Extract surface, centerline, and average temperatures
    T_fuel_surface_z = T_fuel_y[:, -1]
    T_fuel_centerline_z = T_fuel_y[:, 0]
    T_fuel_avg_z = np.mean(T_fuel_y, axis=1)
    k_fuel_centerline = k_fuel[:, 0]

    if assembly_type == 'Pin':
        T_gap_z = (T_clad_in_z + T_fuel_surface_z) / 2
    else:
        T_gap_z = None

    coolant_density, specific_heat_capacity_coolant, thermal_conductivity_coolant, viscosity_coolant = get_coolant_properties_vector(coolant_type, T_coolant_z)
    heat_transfer_coeff_coolant = heat_transfer_coeff_coolant_calculation_vector(coolant_density, specific_heat_capacity_coolant, thermal_conductivity_coolant, viscosity_coolant, flow_rate)

    q_dnb_vector, heat_flux_z, MDNBR = get_MDNBR(T_coolant_z)

    TH_data = {
        # Temperatures along the fuel rod/plate length (z)
        'T_fuel_centerline_z': T_fuel_centerline_z,
        'T_fuel_surface_z': T_fuel_surface_z,
        'T_gap_z': T_gap_z,
        'T_clad_in_z': T_clad_in_z,
        'T_clad_middle_z': T_clad_middle_z,
        'T_clad_out_z': T_clad_out_z,
        'T_coolant_z': T_coolant_z,
        'T_fuel_avg_z': T_fuel_avg_z,
        'T_fuel_y': T_fuel_y,
        'y_fuel': y_fuel,

        # Thermal conductivities
        'k_fuel_centerline': k_fuel_centerline,
        'k_fuel_bulk': k_fuel_bulk,
        'k_clad_in': k_clad_in,
        'k_clad_mid': k_clad_mid,
        'k_clad_out': k_clad_out,

        # Heat transfer coefficients
        'h_gap': h_gap,
        'heat_transfer_coeff_coolant': heat_transfer_coeff_coolant,

        # Coolant properties
        'coolant_density': coolant_density,
        'specific_heat_capacity_coolant': specific_heat_capacity_coolant,
        'thermal_conductivity_coolant': thermal_conductivity_coolant,
        'viscosity_coolant': viscosity_coolant,

        # Heat generation and flow rates
        'Q_dot_z': Q_dot_z,
        'mass_flow_rate': mass_flow_rate,

        # Geometry parameters
        'hydraulic_diameter': D_h * 1000,  # mm
        'coolant_area': coolant_area * 1e6,  # mm^2
        'coolant_to_total_area_ratio': coolant_area / (pin_pitch**2 if assembly_type == 'Pin' else (coolant_area+fuel_plate_thickness*fuel_plate_width)),

        # Flow parameters
        'coolant_velocity': flow_rate,  # m/s
        'volume_flow_rate': coolant_area * flow_rate * 1e6,  # cm^3/s
        'mean_mass_flux': np.mean(coolant_density) * flow_rate,  # kg/m^2-s

        # Heat transfer parameters
        'average_linear_heat_rate': np.mean(Q_dot_z),  # W/m
        'average_heat_flux': np.mean(Q_dot_z) / (2 * np.pi * r_clad_outer if assembly_type == 'Pin' else 2 * fuel_plate_width),  # W/m^2
        'maximum_heat_flux': np.max(Q_dot_z) / (2 * np.pi * r_clad_outer if assembly_type == 'Pin' else 2 * fuel_plate_width),  # W/m^2

        # Temperature gradients at z = 0 (middle of the fuel rod/plate)
        'fuel_delta_T': T_fuel_centerline_z[len(z)//2] - T_fuel_surface_z[len(z)//2],  # K
        'cladding_delta_T_inner_to_outer': T_clad_in_z[len(z)//2] - T_clad_out_z[len(z)//2],  # K
        'cladding_delta_T_inner_to_middle': T_clad_in_z[len(z)//2] - T_clad_middle_z[len(z)//2],  # K
        'cladding_delta_T_middle_to_outer': T_clad_middle_z[len(z)//2] - T_clad_out_z[len(z)//2],  # K
        'coolant_delta_T_z0_to_inlet': T_coolant_z[len(z)//2] - T_coolant_z[0],  # K

        # Critical Heat Flux and Safety Margins
        'minimum_DNBR': np.min(MDNBR),
        'location_of_min_DNBR': z[np.argmin(MDNBR)],  # m
        'average_CHF': np.mean(q_dnb_vector),  # W/m^2
        'minimum_CHF': np.min(q_dnb_vector),  # W/m^2
        'mean_MDNBR': np.mean(q_dnb_vector) / (np.mean(Q_dot_z) / (2 * np.pi * r_clad_outer if assembly_type == 'Pin' else 2 * fuel_plate_width)),

        # Power and Energy Balance
        'total_power_per_element': trapezoid(Q_dot_z, z),  # W
        'total_power_per_assembly': trapezoid(Q_dot_z * n_elements_per_assembly / 1000, z),  # kW
        'total_core_power': trapezoid(Q_dot_z, z) * n_elements_per_assembly * num_assemblies / 1e6,  # MW
        'coolant_energy_gain_per_element': np.mean(mass_flow_rate) * np.mean(specific_heat_capacity_coolant) * (T_coolant_z[-1] - T_inlet),  # W
        'energy_balance_error': (trapezoid(Q_dot_z, z) - mass_flow_rate * np.mean(specific_heat_capacity_coolant) * (T_coolant_z[-1] - T_inlet)) / trapezoid(Q_dot_z, z) * 100,  # %

        # Additional parameters
        'T_inlet': T_inlet,  # K
        'T_outlet': T_coolant_z[-1],  # K
        'fuel_count': num_assemblies,
        'number_of_elements_per_assembly': n_elements_per_assembly,
        'power_density': avg_power_density_kW_L,  # kW/L
        'z': z,  # m (axial positions)

        # Reactor parameters
        'assembly_type': assembly_type,
        'coolant_type': coolant_type,
        'reactor_pressure': reactor_pressure,  # Pa
    }

    if assembly_type == 'Pin':
        TH_data.update({
            'fuel_outer_diameter': 2 * r_fuel * 1000,  # mm
            'clad_outer_diameter': 2 * r_clad_outer * 1000,  # mm
            'pin_pitch': pin_pitch * 1000,  # mm
            'pitch_to_fuel_OD_ratio': pin_pitch / (2 * r_fuel),
            'pitch_to_clad_OD_ratio': pin_pitch / (2 * r_clad_outer),
        })
    elif assembly_type == 'Plate':
        TH_data.update({
            'fuel_meat_thickness': fuel_meat_thickness * 1000,  # mm
            'fuel_plate_thickness': fuel_plate_thickness * 1000,  # mm
            'fuel_plate_width': fuel_plate_width * 1000,  # mm
            'fuel_plate_pitch': fuel_plate_pitch * 1000,  # mm
        })

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_folder)
    os.makedirs(output_dir, exist_ok=True)

    extract_temperature_profiles_to_csv(TH_data, output_dir)
    write_TH_results(TH_data, output_dir)

    if plotting:
        if assembly_type == 'Plate':
            T_clad_y = calculate_cladding_temperature_profile(Q_dot_z, T_clad_in_z, T_clad_out_z)
            plot_results_plate(Q_dot_z, T_coolant_z, T_clad_out_z, T_clad_middle_z, T_clad_in_z, T_fuel_y, T_clad_y)
            plot_material_properties(z, k_fuel_bulk, k_clad_out, k_clad_mid, k_clad_in, heat_transfer_coeff_coolant, h_gap=None)
            plot_conductivity_vs_temperature(calculate_k_fuel, calculate_k_clad, calculate_h_gap_vector=None)
            plot_plate()
            plot_plate_assembly()
        elif assembly_type == 'Pin':
            T_clad_y = calculate_cladding_temperature_profile(Q_dot_z, T_clad_in_z, T_clad_out_z)
            plot_results_pin(Q_dot_z, T_coolant_z, T_clad_out_z, T_clad_middle_z, T_clad_in_z, T_fuel_surface_z, T_fuel_centerline_z, T_fuel_y, y_fuel, MDNBR)
            plot_material_properties(z, k_fuel_bulk, k_clad_out, k_clad_mid, k_clad_in, heat_transfer_coeff_coolant, h_gap)
            plot_conductivity_vs_temperature(calculate_k_fuel, calculate_k_clad, calculate_h_gap_vector)
            plot_pin()
            plot_pin_assembly()
    return TH_data

def extract_temperature_profiles_to_csv(TH_data, output_dir,output_file='temperature_profiles.csv'):
    # Extract the required data from TH_data
    z = TH_data['z']
    T_coolant = TH_data['T_coolant_z']
    T_clad_out = TH_data['T_clad_out_z']
    T_clad_middle = TH_data['T_clad_middle_z']
    T_clad_in = TH_data['T_clad_in_z']
    T_fuel_surface = TH_data['T_fuel_surface_z']
    T_fuel_centerline = TH_data['T_fuel_centerline_z']
    T_fuel_avg = TH_data['T_fuel_avg_z']

    h_coolant = TH_data['heat_transfer_coeff_coolant']
    k_clad_out = TH_data['k_clad_out']
    k_clad_mid = TH_data['k_clad_mid']
    k_clad_in = TH_data['k_clad_in']
    k_fuel_centerline = TH_data['k_fuel_centerline']
    k_fuel_bulk = TH_data['k_fuel_bulk']

    # Calculate k_fuel_surface based on T_fuel_surface
    k_fuel_surface = calculate_k_fuel_vector(T_fuel_surface)

    # Check if it's a pin or plate assembly
    if TH_data['assembly_type'] == 'Pin':
        T_gap = TH_data['T_gap_z']
        h_gap = TH_data['h_gap']
    else:
        T_gap = np.full_like(T_coolant, np.nan)
        h_gap = np.full_like(T_coolant, np.nan)

    output_file = 'temperature_profiles.csv'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_folder)
    data_dir = os.path.join(output_dir,'Data')
    data_path = os.path.join(data_dir,output_file)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(data_path):
        os.remove(data_path)

    # Write data to CSV file
    with open(data_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow([
            "Element Number", "Axial Position (m)",
            "Coolant Temp (K)", "Coolant HTC (W/m²-K)",
            "Clad Out Temp (K)", "Clad Out k (W/m-K)",
            "Clad Middle Temp (K)", "Clad Middle k (W/m-K)",
            "Clad In Temp (K)", "Clad In k (W/m-K)",
            "Gap Temp (K)", "Gap HTC (W/m²-K)",
            "Fuel Surface Temp (K)", "Fuel Surface k (W/m-K)",
            "Fuel Centerline Temp (K)", "Fuel Centerline k (W/m-K)",
            "Fuel Bulk Temp (K)", "Fuel Bulk k (W/m-K)"
        ])

        # Write data
        for i in range(len(z)):
            writer.writerow([
                i+1, f"{z[i]:.6f}",
                f"{T_coolant[i]:.2f}", f"{h_coolant[i]:.2f}",
                f"{T_clad_out[i]:.2f}", f"{k_clad_out[i]:.2f}",
                f"{T_clad_middle[i]:.2f}", f"{k_clad_mid[i]:.2f}",
                f"{T_clad_in[i]:.2f}", f"{k_clad_in[i]:.2f}",
                f"{T_gap[i]:.2f}", f"{h_gap[i]:.2f}",
                f"{T_fuel_surface[i]:.2f}", f"{k_fuel_surface[i]:.2f}",
                f"{T_fuel_centerline[i]:.2f}", f"{k_fuel_centerline[i]:.2f}",
                f"{T_fuel_avg[i]:.2f}", f"{k_fuel_bulk[i]:.2f}"
            ])

    print(f"Temperature profiles have been written to {output_file}")

def write_TH_results(TH_data, output_dir, output_file='TH_printed_output.txt'):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_folder)
    data_dir = os.path.join(output_dir,'Data')
    data_path = os.path.join(data_dir,'TH_printed_output.txt')
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
        elif TH_data['assembly_type'] == 'Plate':
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
        coolant_params = ['coolant_density', 'specific_heat_capacity_coolant', 'thermal_conductivity_coolant', 'viscosity_coolant', 'heat_transfer_coeff_coolant']
        units = ['kg/m^3', 'J/kg-K', 'W/m-K', 'Pa-s', 'W/m^2-K']
        for param, unit in zip(coolant_params, units):
            name = param.replace('_', ' ').title()
            write_table_row(f"{name} ({unit})", [np.min(TH_data[param]), np.max(TH_data[param]), np.mean(TH_data[param])])

        write_section("7. Critical Heat Flux and Safety Margins")
        write_param("Minimum DNBR", TH_data['minimum_DNBR'])
        write_param("Location of min DNBR", TH_data['location_of_min_DNBR'], "m")
        write_param("Average CHF (q_dnb)", TH_data['average_CHF'], "W/m^2")
        write_param("Minimum CHF", TH_data['minimum_CHF'], "W/m^2")
        write_param("Maximum heat flux", TH_data['maximum_heat_flux'], "W/m^2")
        write_param("Mean MDNBR", TH_data['mean_MDNBR'])

        write_section("8. Power and Energy Balance")
        write_param("Total power per element", TH_data['total_power_per_element'], "W")
        write_param("Total power per assembly", TH_data['total_power_per_assembly'], "kW")
        write_param("Total core power", TH_data['total_core_power'], "MW")
        write_param("Coolant energy gain per element", TH_data['coolant_energy_gain_per_element'], "W")
        write_param("Energy balance error", TH_data['energy_balance_error'], "%")
        write_param("Power density", TH_data['power_density'], "kW/L")
        write_param("Number of assemblies", TH_data['fuel_count'])
        write_param("Number of elements per assembly", TH_data['number_of_elements_per_assembly'])

        write_section("9. Axial Power Profile")
        file.write(f"{'Position (m)':<15}{'Power Density (W/m)':>20}\n")
        file.write("-" * 35 + "\n")
        for i in range(0, len(TH_data['z']), len(TH_data['z'])//10):
            file.write(f"{TH_data['z'][i]:15.3f}{TH_data['Q_dot_z'][i]:20.2f}\n")

        write_section("10. Reactor Parameters")
        write_param("Assembly type", TH_data['assembly_type'])
        write_param("Coolant type", TH_data['coolant_type'])
        write_param("Reactor pressure", TH_data['reactor_pressure']/1e5, "bar")

        file.write("\nNote: All temperatures are in Kelvin (K) unless otherwise stated.\n")

    print(f"TH information has been written to {output_file}")

def validate_geometry():
    initialize_globals()
    r_fuel = inputs["r_fuel"] * 100
    r_clad_inner = inputs["r_clad_inner"] * 100
    r_clad_outer = inputs["r_clad_outer"] * 100
    pin_pitch = inputs['pin_pitch'] * 100

    # Validate pin geometry: smaller radii must be smaller than larger radii
    if r_fuel > r_clad_inner:
        raise ValueError(f"Fuel radius ({r_fuel} cm) cannot be greater than or equal to inner cladding radius ({r_clad_inner} cm).")

    if r_clad_inner >= r_clad_outer:
        raise ValueError(f"Inner cladding radius ({r_clad_inner} cm) cannot be greater than or equal to outer cladding radius ({r_clad_outer} cm).")

    if r_clad_outer>pin_pitch/2:
            raise ValueError(f"Outer cladding radius ({r_clad_outer} cm) cannot be greater than or equal to the pitch divided by 2 ({pin_pitch/2} cm).")

    # Plate geometry validation
    fuel_meat_width = inputs["fuel_meat_width"] * 100
    fuel_plate_width = inputs["fuel_plate_width"] * 100
    fuel_meat_thickness = inputs["fuel_meat_thickness"] * 100
    clad_thickness = inputs["clad_thickness"] * 100
    fuel_plate_pitch = inputs["fuel_plate_pitch"] * 100

    if fuel_meat_width >= fuel_plate_width:
        raise ValueError(f"Fuel meat width ({fuel_meat_width} cm) cannot be greater than or equal to plate width ({fuel_plate_width} cm).")

    required_pitch = fuel_meat_thickness + 2 * clad_thickness
    if fuel_plate_pitch < required_pitch:
        raise ValueError(f"Fuel plate pitch ({fuel_plate_pitch} cm) must be greater than or equal to meat thickness + 2 * clad thickness ({required_pitch} cm).")

    print("Geometry validation passed")

if __name__ == "__main__":
    validate_geometry()
    TH_data = get_TH_data(plotting=True)
