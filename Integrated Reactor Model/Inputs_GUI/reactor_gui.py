# Main reactor GUI class and application logic

"""
Main ReactorGUI class that coordinates all GUI components
"""
import tkinter as tk
from tkinter import ttk
import copy
import queue
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from inputs import inputs as base_inputs
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
        """Setup the main GUI layout with tabs"""
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

    def init_components(self):
        """Initialize all tab components"""
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
        """Schedule a visualization update"""
        if not self.updating:
            self.update_queue.put("update")

    def check_update_queue(self):
        """Check for queued updates"""
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
        """Update the visualization"""
        try:
            self.viz_tab.update_visualization()
        finally:
            self.updating = False
