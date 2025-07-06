# Main reactor GUI class and application logic

"""
Main ReactorGUI class that coordinates all GUI components
"""
import tkinter as tk
from tkinter import ttk, messagebox
import copy
import queue
import sys
import os
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.base_inputs import inputs as base_inputs
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the Integrated Reactor Model directory")
    sys.exit(1)

from Inputs_GUI.components.visualization_tab import VisualizationTab
from Inputs_GUI.components.design_tab import DesignTab
from Inputs_GUI.components.thermal_tab import ThermalTab
from Inputs_GUI.components.advanced_tab import AdvancedTab
from Inputs_GUI.components.geometry_tab import GeometryTab


class ReactorGUI:
    def __init__(self, root):
        """Initialize the reactor GUI application.

        Parameters
        ----------
        root : tkinter.Tk
            Root window for the application

        Returns
        -------
        None
        """
        self.root = root
        self.root.title("Interactive Reactor Design Studio")
        self.root.geometry("1800x1200")

        # Current inputs (copy of base inputs that we'll modify)
        self.current_inputs = copy.deepcopy(base_inputs)

        # View state
        self.current_view = "Core XY"
        self.view_options = [
            "Core XY", "Core YZ", "Core XZ",
            "Assembly XY", "Assembly YZ", "Assembly XZ",
            "Element XY", "Element YZ", "Element XZ"
        ]

        # Update queue for threaded updates
        self.update_queue = queue.Queue()
        self.updating = False

        # Control variables
        self.control_vars = {}

        # Setup GUI
        self.setup_gui()

        # Initialize components
        self.init_components()

        # Initial plot
        self.viz_tab.update_visualization()

        # Start checking for updates
        self.check_update_queue()

    def setup_gui(self):
        """Setup the main GUI layout with tabs and menu.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Create menu bar
        self.setup_menu_bar()

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tab frames
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Reactor Visualization")

        self.design_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.design_frame, text="Layout Designer")

        self.thermal_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.thermal_frame, text="Thermal Hydraulics")

        self.advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.advanced_frame, text="Advanced Inputs")

        self.geometry_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.geometry_frame, text="OpenMC Geometry")

    def setup_menu_bar(self):
        """Setup the menu bar.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        tools_menu.add_command(label="Parametric Study Generator",
                              command=self.launch_parametric_gui)
        tools_menu.add_separator()
        tools_menu.add_command(label="Export Inputs", command=self.export_inputs)
        tools_menu.add_command(label="Import Inputs", command=self.import_inputs)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)

        help_menu.add_command(label="About", command=self.show_about)

    def launch_parametric_gui(self):
        """Launch the separate parametric study GUI.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        try:
            # Get the path to the parametric GUI
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            parametric_gui_path = os.path.join(current_dir, "Parametric_GUI", "main.py")

            if os.path.exists(parametric_gui_path):
                # Launch the parametric GUI as a separate process
                subprocess.Popen([sys.executable, parametric_gui_path])
                messagebox.showinfo("Parametric GUI",
                                   "Parametric Study Generator launched in a separate window.")
            else:
                messagebox.showerror("Error",
                                   f"Parametric GUI not found at: {parametric_gui_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Parametric GUI: {e}")

    def export_inputs(self):
        """Export current inputs to a file.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Placeholder for export functionality
        messagebox.showinfo("Export", "Export functionality coming soon!")

    def import_inputs(self):
        """Import inputs from a file.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Placeholder for import functionality
        messagebox.showinfo("Import", "Import functionality coming soon!")

    def show_about(self):
        """Show about dialog.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        about_text = """Interactive Reactor Design Studio

A comprehensive tool for reactor design and analysis.

Features:
• Visual reactor core layout design
• Real-time geometry visualization
• Thermal hydraulics analysis
• Advanced parameter configuration
• OpenMC geometry generation
• Parametric study generation (separate tool)

Version: 1.0.0
"""
        messagebox.showinfo("About", about_text)

    def init_components(self):
        """Initialize all tab components.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Create tab instances
        self.viz_tab = VisualizationTab(self.viz_frame, self)
        self.design_tab = DesignTab(self.design_frame, self)
        self.thermal_tab = ThermalTab(self.thermal_frame, self)
        self.advanced_tab = AdvancedTab(self.advanced_frame, self)
        self.geometry_tab = GeometryTab(self.geometry_frame, self)

        # Setup tabs
        self.viz_tab.setup()
        self.design_tab.setup()
        self.thermal_tab.setup()
        self.advanced_tab.setup()
        self.geometry_tab.setup()

    def schedule_update(self):
        """Schedule a visualization update.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not self.updating:
            self.update_queue.put("update")

    def check_update_queue(self):
        """Check for queued updates.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        try:
            while True:
                self.update_queue.get_nowait()
                if not self.updating:
                    self.updating = True
                    self.root.after(50, self.update_visualization)
                    break
        except queue.Empty:
            pass

        # Check again in 50ms
        self.root.after(50, self.check_update_queue)

    def update_visualization(self):
        """Update the visualization.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        try:
            self.viz_tab.update_visualization()
        finally:
            self.updating = False
