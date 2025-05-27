#!/usr/bin/env python3
"""
Standalone Parametric GUI for Reactor Model
Allows creation and configuration of parametric studies independent of the main GUI
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the parametric app
from parametric_app import ParametricApp

# For backward compatibility
ParametricTab = ParametricApp


class MockMainGUI:
    """Mock main GUI class for standalone operation"""

    def __init__(self):
        # Load base inputs from inputs.py
        self.current_inputs = self.load_base_inputs()

    def load_base_inputs(self):
        """Load base inputs from inputs.py in the Integrated Reactor Model folder"""
        try:
            # Import inputs from the parent directory (Integrated Reactor Model)
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from inputs import inputs
            return inputs.copy()
        except ImportError as e:
            print(f"Warning: Could not load inputs.py: {e}")
            # Return default inputs for testing
            return self.get_default_inputs()

    def get_default_inputs(self):
        """Get default inputs for testing"""
        return {
            # Core Configuration
            'core_power': 10.0,
            'assembly_type': 'Plate',
            'fuel_height': 0.6,
            'core_lattice': [
                ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
                ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
                ['F', 'F', 'I_1', 'F', 'F', 'I_4', 'F', 'F'],
                ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                ['F', 'F', 'I_2', 'F', 'F', 'I_3', 'F', 'F'],
                ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
                ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C']
            ],

            # Geometry
            'tank_radius': 0.25,
            'reflector_thickness': 0.1,
            'bioshield_thickness': 0.25,

            # Pin Assembly
            'pin_pitch': 0.0126,
            'r_fuel': 0.0041,
            'r_clad_inner': 0.0042,
            'r_clad_outer': 0.00475,
            'n_side_pins': 3,

            # Plate Assembly
            'fuel_meat_width': 3.91/100,
            'fuel_plate_width': 4.81/100,
            'fuel_plate_pitch': 0.37/100,
            'fuel_meat_thickness': 0.147/100,
            'clad_thickness': 0.025/100,
            'plates_per_assembly': 13,
            'clad_structure_width': 0.15/100,

            # Materials
            'coolant_type': 'Light Water',
            'clad_type': 'Al6061',
            'fuel_type': 'U3Si2',
            'reflector_material': 'mgo',
            'bioshield_material': 'Concrete',
            'n%': 19.75,
            'n%E': 93,

            # Thermal Hydraulics
            'reactor_pressure': 3e5,
            'flow_rate': 3,
            'T_inlet': 273.15 + 42,
            'input_power_density': 100,
            'max_linear_power': 70,
            'average_linear_power': 50,
            'cos_curve_squeeze': 0,
            'CP_PD_MLP_ALP': 'CP',

            # Irradiation
            'irradiation_clad': False,
            'irradiation_clad_thickness': 0.15/100,
            'irradiation_cell_fill': 'Vacuum',

            # Monte Carlo
            'batches': 150,
            'inactive': 20,
            'particles': int(2e4),
            'energy_structure': 'log1001',
            'thermal_cutoff': 0.625,
            'fast_cutoff': 100.0e3,
            'power_tally_axial_segments': 50,
            'irradiation_axial_segments': 100,
            'Core_Three_Group_Energy_Bins': True,
            'tally_power': True,
            'element_level_power_tallies': True,

            # Depletion
            'deplete_core': False,
            'deplete_assembly': False,
            'deplete_assembly_enhanced': False,
            'deplete_element': False,
            'deplete_element_enhanced': False,
            'depletion_timestep_units': 'MWd/kgHM',
            'depletion_particles': 5000,
            'depletion_batches': 120,
            'depletion_inactive': 20,
            'depletion_integrator': 'predictor',
            'depletion_chain': 'casl',
            'depletion_nuclides': ['U235', 'U238', 'Pu239', 'Xe135', 'Sm149', 'Cs137', 'Sr90', 'I131'],

            # Misc
            'parametric_study': True,
            'outputs_folder': 'local_outputs'
        }


class ParametricGUIStandalone:
    """Standalone Parametric GUI Application"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Reactor Model - Parametric Study Generator")
        self.root.geometry("1400x900")

        # Create mock main GUI for parameter access
        self.mock_main_gui = MockMainGUI()

        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI"""
        # Title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=20, pady=10)

        title_label = ttk.Label(title_frame, text="Reactor Model - Parametric Study Generator",
                               font=('TkDefaultFont', 16, 'bold'))
        title_label.pack()

        subtitle_label = ttk.Label(title_frame,
                                  text="Design and configure parametric studies for reactor simulations",
                                  font=('TkDefaultFont', 10))
        subtitle_label.pack(pady=(5, 0))

        # Info frame
        info_frame = ttk.LabelFrame(self.root, text="About", padding=10)
        info_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

        info_text = """This tool allows you to create parametric studies for the Integrated Reactor Model:
• Simple Parameter Studies: Vary individual parameters with different values
• Multi-Parameter Loops: Create complex combinations of multiple parameters
• Core Lattice Designer: Visual interface for designing reactor core layouts
• Export Functionality: Generate run_dictionaries.py files for automated simulations"""

        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack(anchor=tk.W)

        # Main parametric tab area
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Create parametric tab
        self.parametric_tab = ParametricTab(main_frame, self.mock_main_gui)
        self.parametric_tab.setup()

        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = ttk.Label(status_frame, text="Ready - Select parameters to configure parametric studies")
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)

        # Parameter count display
        param_count = len(self.parametric_tab.available_params)
        count_label = ttk.Label(status_frame, text=f"{param_count} parameters available")
        count_label.pack(side=tk.RIGHT, padx=10, pady=5)

    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Application interrupted by user")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


def main():
    """Main function"""
    try:
        app = ParametricGUIStandalone()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
