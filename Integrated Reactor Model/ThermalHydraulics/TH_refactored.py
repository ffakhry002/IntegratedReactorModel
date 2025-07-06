import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import shutil
from tabulate import tabulate
from scipy.integrate import quad, trapezoid
import csv
import sys

# Add parent directory to Python path to access inputs.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_inputs import inputs

# Use absolute imports for code_architecture
from ThermalHydraulics.code_architecture.data_output_code.output_helpers.th_data_writer import write_TH_results
from ThermalHydraulics.code_architecture.data_output_code.output_helpers.th_data_extractor import get_TH_data
from ThermalHydraulics.code_architecture.data_output_code.output_helpers.th_temperature_profiles import extract_temperature_profiles_to_csv
from ThermalHydraulics.code_architecture.data_output_code.output_helpers.th_plotting import generate_plots
from ThermalHydraulics.code_architecture.helper_codes.convergence import (
    single_iteration,
    converge_temperatures
)
from ThermalHydraulics.code_architecture.helper_codes.temperature_points import (
    calculate_temperature_points_pins,
    calculate_temperature_points_plates
)
from ThermalHydraulics.code_architecture.helper_codes.power_calculations import (
    calculate_Q_dot_z,
    calculate_critical_heat_flux
)
from ThermalHydraulics.code_architecture.helper_codes.models.material import Material
from ThermalHydraulics.code_architecture.helper_codes.models.geometry import PinGeometry, PlateGeometry
from ThermalHydraulics.code_architecture.helper_codes.models.reactor import ReactorPower, ThermalHydraulics
from ThermalHydraulics.code_architecture.helper_codes.models.thermal_state import ThermalState
from ThermalHydraulics.code_architecture.helper_codes.models.geometry_validator import validate_geometry

def cleanup_pycache():
    """Remove all __pycache__ directories in the project.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Walk through all directories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__pycache__' in dirnames:
            pycache_path = os.path.join(dirpath, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                print(f"Removed: {pycache_path}")
            except Exception as e:
                print(f"Error removing {pycache_path}: {e}")

def cleanup_local_outputs():
    """Remove any local output directories in code_architecture folder.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    code_arch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code_architecture')

    # List of output directory names to check for
    output_dirs = ['TH_plots', 'geometry_plots', 'Data']

    # Walk through all directories in code_architecture
    for dirpath, dirnames, filenames in os.walk(code_arch_dir):
        for dirname in dirnames:
            if dirname in output_dirs:
                output_path = os.path.join(dirpath, dirname)
                try:
                    shutil.rmtree(output_path)
                    print(f"Removed output directory: {output_path}")
                except Exception as e:
                    print(f"Error removing {output_path}: {e}")

class THSystem:
    def __init__(self, inputs_dict):
        # Store the inputs_dict for later use
        self.inputs_dict = inputs_dict

        self.material = Material(
            inputs_dict["coolant_type"],
            inputs_dict["clad_type"],
            inputs_dict["fuel_type"]
        )

        self.pin_geometry = PinGeometry(
            inputs_dict["pin_pitch"],
            inputs_dict["r_fuel"],
            inputs_dict["r_clad_inner"],
            inputs_dict["r_clad_outer"],
            inputs_dict["n_side_pins"],
            inputs_dict["n_guide_tubes"],
            inputs_dict["fuel_height"]
        )

        self.plate_geometry = PlateGeometry(
            inputs_dict["fuel_meat_width"],
            inputs_dict["fuel_plate_width"],
            inputs_dict["fuel_plate_pitch"],
            inputs_dict["fuel_meat_thickness"],
            inputs_dict["clad_thickness"],
            inputs_dict["plates_per_assembly"],
            inputs_dict["clad_structure_width"],
            inputs_dict["fuel_height"]
        )

        self.reactor_power = ReactorPower(
            inputs_dict["core_power"],
            inputs_dict["num_assemblies"],
            inputs_dict["CP_PD_MLP_ALP"],
            inputs_dict["input_power_density"],
            inputs_dict["max_linear_power"],
            inputs_dict["average_linear_power"],
            inputs_dict["cos_curve_squeeze"]
        )

        self.thermal_hydraulics = ThermalHydraulics(
            inputs_dict["reactor_pressure"],
            inputs_dict["flow_rate"],
            inputs_dict["T_inlet"],
            inputs_dict["assembly_type"],
            inputs_dict["outputs_folder"]
        )

        # Initialize thermal state
        self.thermal_state = ThermalState()

        # Set up geometry based on assembly type
        self.geometry = self.pin_geometry if self.thermal_hydraulics.assembly_type == 'Pin' else self.plate_geometry

        # Validate geometry parameters
        validate_geometry(self)

        # Set up z array
        self.z = np.linspace(-self.geometry.fuel_height/2, self.geometry.fuel_height/2, self.thermal_state.z_points)

        # Initialize Q_dot_z
        self.thermal_state.Q_dot_z = calculate_Q_dot_z(self)

    def calculate_Q_dot_z(self):
        """Calculate power distribution.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            Power distribution array
        """
        return calculate_Q_dot_z(self)

    def calculate_temperature_points_pins(self):
        """Calculate temperature points for pin geometry.

        Parameters
        ----------
        None

        Returns
        -------
        tuple
            Temperature points for pin geometry calculation
        """
        return calculate_temperature_points_pins(self)

    def calculate_temperature_points_plates(self):
        """Calculate temperature points for plate geometry.

        Parameters
        ----------
        None

        Returns
        -------
        tuple
            Temperature points for plate geometry calculation
        """
        return calculate_temperature_points_plates(self)

    def calculate_power_distribution(self):
        """Calculate power distribution and store in thermal state.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.thermal_state.Q_dot_z = calculate_Q_dot_z(self)

    def calculate_critical_heat_flux(self):
        """Calculate critical heat flux and MDNBR.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Updates thermal_state with q_dnb, heat_flux_z, and MDNBR values
        """
        q_dnb, heat_flux_z, MDNBR = calculate_critical_heat_flux(self)
        self.thermal_state.q_dnb = q_dnb
        self.thermal_state.heat_flux_z = heat_flux_z
        self.thermal_state.MDNBR = MDNBR

    def single_iteration(self):
        """Perform a single iteration of temperature calculations using current thermal state.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if iteration completed successfully
        """
        return single_iteration(self)

    def calculate_temperature_distribution(self):
        """Calculate temperature distribution with convergence.

        Parameters
        ----------
        None

        Returns
        -------
        ThermalState
            Converged thermal state object
        """
        return converge_temperatures(self)

    def write_results(self, output_dir=None, plotting=True):
        """Write all results to files and generate plots.

        Parameters
        ----------
        output_dir : str, optional
            Directory to save outputs. If None, saves in local ThermalHydraulics directory.
        plotting : bool, optional
            Whether to generate plots. Defaults to True.
        """
        if output_dir is None:
            # Use local ThermalHydraulics directory
            root_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(root_dir, self.thermal_hydraulics.outputs_folder)

        write_TH_results(self, output_dir)
        extract_temperature_profiles_to_csv(self, output_dir)
        if plotting:
            generate_plots(self, output_dir, inputs_dict=self.inputs_dict)

    def get_data(self):
        """Get all thermal-hydraulic data as a dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary containing all thermal-hydraulic data
        """
        return get_TH_data(self)

if __name__ == "__main__":
    try:
        th_system = THSystem(inputs)
        thermal_state = th_system.calculate_temperature_distribution()
        th_system.write_results(plotting=True)
    finally:
        # Clean up __pycache__ directories and local outputs even if an error occurs
        cleanup_pycache()
        cleanup_local_outputs()
