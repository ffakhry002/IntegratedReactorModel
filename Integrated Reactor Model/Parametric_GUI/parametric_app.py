"""
Main Parametric Application Controller
Manages the overall parametric study interface
"""
import tkinter as tk
from tkinter import ttk
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Use absolute imports when running as script
try:
    from components.simple_tab import SimpleTab
    from components.complex_tab import ComplexTab
    from components.preview_tab import PreviewTab
    from models.parameter_model import ParameterModel
    from models.run_configuration import RunConfiguration
except ImportError:
    # Fallback for package imports
    from .components.simple_tab import SimpleTab
    from .components.complex_tab import ComplexTab
    from .components.preview_tab import PreviewTab
    from .models.parameter_model import ParameterModel
    from .models.run_configuration import RunConfiguration


class ParametricApp:
    """Main parametric application controller"""

    def __init__(self, parent, main_gui):
        self.parent = parent
        self.main_gui = main_gui

        # Initialize models
        self.parameter_model = ParameterModel(main_gui.current_inputs)
        self.run_config = RunConfiguration()

        # Expose available_params for backward compatibility
        self.available_params = self.parameter_model.available_params

        # Tab references
        self.simple_tab = None
        self.complex_tab = None
        self.preview_tab = None

    def setup(self):
        """Setup the parametric interface"""
        # Main container
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create notebook for different study types
        self.param_notebook = ttk.Notebook(main_frame)
        self.param_notebook.pack(fill=tk.BOTH, expand=True)

        # Simple runs tab
        simple_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(simple_frame, text="Simple Parameter Studies")
        self.simple_tab = SimpleTab(simple_frame, self.parameter_model, self.run_config)
        self.simple_tab.setup()

        # Complex runs tab
        complex_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(complex_frame, text="Multi-Parameter Loops")
        self.complex_tab = ComplexTab(complex_frame, self.parameter_model, self.run_config)
        self.complex_tab.setup()

        # Preview/Export tab
        preview_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(preview_frame, text="Preview & Export")
        self.preview_tab = PreviewTab(preview_frame, self.run_config)
        self.preview_tab.setup()


# For backward compatibility
ParametricTab = ParametricApp
