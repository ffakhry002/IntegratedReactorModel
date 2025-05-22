import numpy as np
import pandas as pd
import os
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

def read_power_from_csv(th_system):
    """Read power distribution from CSV file.

    Args:
        th_system: THSystem object containing power source settings

    Returns:
        np.array: Power distribution array in W/m
    """
    csv_path = th_system.reactor_power.csv_path
    if not csv_path or not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}, falling back to cosine approximation")
        return calculate_cosine_power(th_system)

    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Check for and remove 'Total' row
        if 'z_position_cm' in df.columns:
            # First convert to string to safely check
            df['z_position_cm'] = df['z_position_cm'].astype(str)
            # Filter out rows containing 'Total'
            df = df[~df['z_position_cm'].str.contains('Total', case=False, na=False)]
            # Convert back to numeric
            df['z_position_cm'] = pd.to_numeric(df['z_position_cm'], errors='coerce')

        # Convert all numeric columns to float, replacing non-numeric values with NaN
        for col in df.columns:
            if col != 'z_position_cm':  # Skip z position column which we already handled
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaN values
        df = df.dropna()

        if df.empty:
            print("CSV file contains no valid numeric data, falling back to cosine approximation")
            return calculate_cosine_power(th_system)

        # Get z-axis values from the CSV
        z_csv = df['z_position_cm'].values

        # Get the power source as specified
        power_source = th_system.reactor_power.power_source

        # Determine if we're dealing with element-level or assembly-level data
        is_element_level = any(col.startswith(('Pin_', 'Plate_')) for col in df.columns)

        if power_source == 'HOT_ELEMENT':
            # Find the hot element column (starts with 'Hot_')
            hot_col = next((col for col in df.columns if col.startswith('Hot_')), None)
            if hot_col:
                power_data = df[hot_col].values
            else:
                print("Hot element column not found in CSV, using core average")
                avg_col = next((col for col in df.columns if col.startswith('Average_')), None)
                if avg_col:
                    power_data = df[avg_col].values
                elif 'Core_Total' in df.columns:
                    # Calculate per element power
                    if is_element_level:
                        # For element-level data, divide by total number of elements
                        if th_system.thermal_hydraulics.assembly_type == 'Pin':
                            n_elements = th_system.reactor_power.num_assemblies * (th_system.pin_geometry.n_side_pins**2 - th_system.pin_geometry.n_guide_tubes)
                        else:  # Plate
                            n_elements = th_system.reactor_power.num_assemblies * th_system.plate_geometry.plates_per_assembly
                    else:
                        # For assembly-level data, divide by number of assemblies
                        n_elements = th_system.reactor_power.num_assemblies

                    power_data = df['Core_Total'].values / float(n_elements)
                else:
                    print("Neither hot element nor core total column found in CSV, using cosine approximation")
                    return calculate_cosine_power(th_system)

        elif power_source == 'CORE_AVERAGE':
            # Use the Average_X column where X is Pin, Plate, or Assembly
            avg_col = next((col for col in df.columns if col.startswith('Average_')), None)
            if avg_col:
                power_data = df[avg_col].values
            else:
                # If no Average column, take Core_Total and divide by number of elements
                if 'Core_Total' in df.columns:
                    if is_element_level:
                        # For element-level data, divide by total number of elements
                        if th_system.thermal_hydraulics.assembly_type == 'Pin':
                            n_elements = th_system.reactor_power.num_assemblies * (th_system.pin_geometry.n_side_pins**2 - th_system.pin_geometry.n_guide_tubes)
                        else:  # Plate
                            n_elements = th_system.reactor_power.num_assemblies * th_system.plate_geometry.plates_per_assembly
                    else:
                        # For assembly-level data, divide by number of assemblies
                        n_elements = th_system.reactor_power.num_assemblies

                    power_data = df['Core_Total'].values / float(n_elements)
                else:
                    print("Core average data not found in CSV, using cosine approximation")
                    return calculate_cosine_power(th_system)
        else:
            print(f"Unknown power source {power_source}, using cosine approximation")
            return calculate_cosine_power(th_system)

        # Convert power values from kW/m to W/m
        power_data = power_data * 1000

        # Convert z_csv from cm to m to match th_system.z
        z_csv_m = z_csv / 100

        # Use 1D linear interpolation to create a smooth power distribution
        # Sort the z values and corresponding power values to ensure the interpolator works correctly
        sort_indices = np.argsort(z_csv_m)
        z_csv_m_sorted = z_csv_m[sort_indices]
        power_data_sorted = power_data[sort_indices]

        # Create the interpolation function
        # Use 'linear' for linear interpolation between points
        # Use 'extrapolate' to handle any z values outside the range of the CSV data
        try:
            power_interpolator = interp1d(z_csv_m_sorted, power_data_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')

            # Apply the interpolation to get values at each z position in the thermal system
            Q_dot_z = power_interpolator(th_system.z)
            return Q_dot_z

        except Exception as e:
            print(f"Error creating interpolation: {str(e)}, falling back to closest-point method")

            # Fallback to closest-point method if interpolation fails
            Q_dot_z = np.zeros_like(th_system.z)
            for i, z in enumerate(th_system.z):
                closest_idx = np.abs(z_csv_m - z).argmin()
                Q_dot_z[i] = power_data[closest_idx]
            return Q_dot_z

    except Exception as e:
        print(f"Error reading power from CSV: {str(e)}, falling back to cosine approximation")
        return calculate_cosine_power(th_system)

def calculate_cosine_power(th_system):
    """Calculate axial power distribution using cosine approximation.

    Args:
        th_system: THSystem object containing geometry, power, and material information

    Returns:
        np.array: Array of heat generation rates along z-axis in W/m
    """
    # Calculate assembly volumes
    if th_system.thermal_hydraulics.assembly_type == 'Plate':
        single_assembly_volume = (
            (th_system.plate_geometry.fuel_plate_width + 2 * th_system.plate_geometry.clad_structure_width) *
            th_system.plate_geometry.fuel_plate_pitch *
            th_system.plate_geometry.fuel_height *
            th_system.plate_geometry.plates_per_assembly
        )
    else:
        single_assembly_volume = (
            (th_system.pin_geometry.pin_pitch * th_system.pin_geometry.n_side_pins)**2 *
            th_system.pin_geometry.fuel_height
        )

    total_assembly_volume = single_assembly_volume * th_system.reactor_power.num_assemblies

    # Calculate power density
    if th_system.reactor_power.Q_dot_z_toggle == 'PD':
        avg_power_density_kW_L = th_system.reactor_power.input_avg_power_density
        avg_power_density_W_m3 = avg_power_density_kW_L * 1e6
    else:
        avg_power_density_kW_L = th_system.reactor_power.core_power / total_assembly_volume
        avg_power_density_W_m3 = avg_power_density_kW_L * 1e6

    # Calculate power per element
    total_power_per_assembly = avg_power_density_W_m3 * single_assembly_volume
    power_per_element = total_power_per_assembly / th_system.geometry.n_elements_per_assembly

    # Calculate linear power based on toggle
    if th_system.reactor_power.Q_dot_z_toggle in ['CP', 'PD']:
        Average_linear_power = power_per_element / th_system.geometry.fuel_height
        Peak_linear_power = Average_linear_power * np.pi / 2
    elif th_system.reactor_power.Q_dot_z_toggle == 'ALP':
        Average_linear_power = th_system.reactor_power.input_avg_lp * 1e3
        Peak_linear_power = Average_linear_power * np.pi / 2
    elif th_system.reactor_power.Q_dot_z_toggle == 'MLP':
        Peak_linear_power = th_system.reactor_power.input_max_lp * 1e3
        Average_linear_power = Peak_linear_power * 2 / np.pi

    # Calculate power distribution
    original_curve = np.cos(np.pi * th_system.z / th_system.geometry.fuel_height)
    original_power = trapezoid(original_curve, th_system.z)

    adjusted_curve = (1 - th_system.reactor_power.cos_curve_squeeze) * original_curve + th_system.reactor_power.cos_curve_squeeze
    new_power = trapezoid(adjusted_curve, th_system.z)
    adjusted_curve *= original_power / new_power

    # Calculate final Q_dot_z
    if th_system.reactor_power.Q_dot_z_toggle == 'MLP':
        adjusted_curve = adjusted_curve / np.max(adjusted_curve)
        Q_dot_z = Peak_linear_power * adjusted_curve
    else:
        Q_dot_z = Peak_linear_power * adjusted_curve

    return Q_dot_z

def calculate_Q_dot_z(th_system):
    """Calculate axial power distribution.

    Args:
        th_system: THSystem object containing geometry, power, and material information

    Returns:
        np.array: Array of heat generation rates along z-axis
    """
    # Check if we should use CSV power distribution
    if th_system.reactor_power.power_source != 'COSINE' and th_system.reactor_power.csv_path:
        return read_power_from_csv(th_system)
    else:
        return calculate_cosine_power(th_system)
