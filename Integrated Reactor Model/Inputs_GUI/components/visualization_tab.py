# Visualization tab component

"""
Visualization Tab Component
Handles the main reactor visualization display
"""
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from Inputs_GUI.visualization.core_view import CoreView
from Inputs_GUI.visualization.assembly_view import AssemblyView
from Inputs_GUI.visualization.element_view import ElementView
from Inputs_GUI.controls.parameter_controls import ParameterControls
from Inputs_GUI.utils.constants import MATERIAL_COLORS


class VisualizationTab:
    def __init__(self, parent, main_gui):
        self.parent = parent
        self.main_gui = main_gui

        # View handlers
        self.core_view = CoreView(main_gui)
        self.assembly_view = AssemblyView(main_gui)
        self.element_view = ElementView(main_gui)

        # Control variables
        self.auto_update_var = tk.BooleanVar(value=True)
        self.show_legend_var = tk.BooleanVar(value=True)

    def setup(self):
        """Setup the visualization tab"""
        # Main container
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel for controls (wider)
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.configure(width=600)  # Made wider

        # Right panel for visualization
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_controls(control_frame)
        self.setup_visualization(viz_frame)

    def setup_controls(self, parent):
        """Setup control panel"""
        # Create scrollable frame for controls
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # View selection
        view_frame = ttk.LabelFrame(scrollable_frame, text="View Selection", padding=10)
        view_frame.pack(fill=tk.X, pady=(0, 10))

        self.view_var = tk.StringVar(value=self.main_gui.current_view)
        view_combo = ttk.Combobox(view_frame, textvariable=self.view_var,
                                 values=self.main_gui.view_options, state="readonly")
        view_combo.pack(fill=tk.X)
        view_combo.bind('<<ComboboxSelected>>', self.on_view_change)

        # Display options
        display_frame = ttk.LabelFrame(scrollable_frame, text="Display Options", padding=10)
        display_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(display_frame, text="Show Legend",
                       variable=self.show_legend_var,
                       command=self.main_gui.schedule_update).pack(anchor=tk.W)

        ttk.Checkbutton(display_frame, text="Auto-update visualization",
                       variable=self.auto_update_var).pack(anchor=tk.W)

        # Create parameter controls with proper ordering
        self.param_controls = ParameterControls(scrollable_frame, self.main_gui)
        self.param_controls.setup()

        # Manual update button
        ttk.Button(scrollable_frame, text="Update Visualization",
                  command=self.main_gui.schedule_update).pack(fill=tk.X, pady=(10, 5))

        # Export current values button
        ttk.Button(scrollable_frame, text="Export Current Values",
                  command=self.export_values).pack(fill=tk.X, pady=(5, 0))

    def setup_visualization(self, parent):
        """Setup the visualization panel"""
        # Title
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        self.title_label = ttk.Label(title_frame, text=self.main_gui.current_view,
                                    font=('Arial', 16, 'bold'))
        self.title_label.pack()

        # Matplotlib figure
        self.fig = Figure(figsize=(14, 12), dpi=150)
        self.ax = self.fig.add_subplot(111)

        # Canvas
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Navigation toolbar
        try:
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar_frame = ttk.Frame(canvas_frame)
            toolbar_frame.pack(fill=tk.X)
            toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
            toolbar.update()
        except ImportError:
            pass

        # Enable mouse interactions
        self.setup_mouse_interactions()

        # Status bar
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.dimensions_label = ttk.Label(status_frame, text="", relief=tk.SUNKEN)
        self.dimensions_label.pack(side=tk.RIGHT, padx=(5, 0))

    def setup_mouse_interactions(self):
        """Setup mouse wheel zoom and pan"""
        def on_scroll(event):
            if event.inaxes != self.ax:
                return
            scale_factor = 1.1
            if event.step > 0:
                scale_factor = 1 / scale_factor
            elif event.step < 0:
                scale_factor = scale_factor
            else:
                return

            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            xdata = event.xdata
            ydata = event.ydata

            new_width = (xlim[1] - xlim[0]) * scale_factor
            new_height = (ylim[1] - ylim[0]) * scale_factor

            relx = (xlim[1] - xdata)/(xlim[1] - xlim[0])
            rely = (ylim[1] - ydata)/(ylim[1] - ylim[0])

            self.ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * relx])
            self.ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * rely])
            self.canvas.draw()

        self.canvas.mpl_connect('scroll_event', on_scroll)

        # Pan functionality
        pan_start = [None]

        def on_press(event):
            if event.inaxes != self.ax:
                return
            pan_start[0] = (event.xdata, event.ydata)

        def on_release(event):
            pan_start[0] = None

        def on_motion(event):
            if pan_start[0] is None or event.inaxes != self.ax:
                return

            dx = event.xdata - pan_start[0][0]
            dy = event.ydata - pan_start[0][1]

            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            self.ax.set_xlim([xlim[0] - dx, xlim[1] - dx])
            self.ax.set_ylim([ylim[0] - dy, ylim[1] - dy])
            self.canvas.draw()

        self.canvas.mpl_connect('button_press_event', on_press)
        self.canvas.mpl_connect('button_release_event', on_release)
        self.canvas.mpl_connect('motion_notify_event', on_motion)

    def on_view_change(self, event=None):
        """Handle view selection change"""
        self.main_gui.current_view = self.view_var.get()
        self.title_label.config(text=self.main_gui.current_view)
        if self.auto_update_var.get():
            self.main_gui.schedule_update()

    def update_visualization(self):
        """Update the visualization based on current parameters and view"""
        try:
            self.status_label.config(text="Updating visualization...")
            self.main_gui.root.update()

            # Clear the current plot
            self.ax.clear()

            # Generate the appropriate visualization
            if "Core" in self.main_gui.current_view:
                self.core_view.plot(self.ax, self.main_gui.current_view)
            elif "Assembly" in self.main_gui.current_view:
                self.assembly_view.plot(self.ax, self.main_gui.current_view)
            elif "Element" in self.main_gui.current_view:
                self.element_view.plot(self.ax, self.main_gui.current_view)

            # Add legend if enabled
            if self.show_legend_var.get():
                self.add_legend()

            # Update dimensions label
            self.update_dimensions_label()

            # Adjust layout
            self.fig.tight_layout()

            # Update the canvas
            self.canvas.draw()
            self.status_label.config(text="Ready")

        except Exception as e:
            self.ax.text(0.5, 0.5, f"Visualization Error:\n{str(e)}",
                       transform=self.ax.transAxes, ha='center', va='center',
                       fontsize=12, color='red')
            self.canvas.draw()
            self.status_label.config(text=f"Error: {str(e)}")
            print(f"Visualization error: {e}")

    def add_legend(self):
        """Add material legend to the plot"""
        from matplotlib.patches import Patch
        from Inputs_GUI.visualization.plotting_utils import get_material_legend_info

        legend_items = get_material_legend_info(self.main_gui.current_inputs,
                                               self.main_gui.current_view)

        if legend_items:
            patches = []
            labels = []

            for color, label in legend_items:
                patches.append(Patch(facecolor=color, edgecolor='black', linewidth=0.5))
                labels.append(label)

            legend = self.ax.legend(patches, labels, loc='upper left',
                                  bbox_to_anchor=(1.02, 1), frameon=True,
                                  framealpha=0.9, edgecolor='black')
            legend.get_frame().set_facecolor('white')

    def update_dimensions_label(self):
        """Update the dimensions label based on current view"""
        inputs = self.main_gui.current_inputs

        if "Core" in self.main_gui.current_view:
            lattice = inputs['core_lattice']
            n_rows, n_cols = len(lattice), len(lattice[0])

            if inputs['assembly_type'] == 'Pin':
                assembly_pitch = inputs['pin_pitch'] * inputs['n_side_pins']
            else:
                assembly_pitch = (inputs['plates_per_assembly'] *
                                inputs['fuel_plate_pitch'] +
                                2 * inputs['clad_structure_width'])

            tank_radius = inputs['tank_radius']
            self.dimensions_label.config(
                text=f"Core: {n_cols}×{n_rows}, Assembly pitch: {assembly_pitch:.3f}m, "
                     f"Tank: ⌀{tank_radius*2:.3f}m"
            )

        elif "Assembly" in self.main_gui.current_view:
            if inputs['assembly_type'] == 'Pin':
                n_pins = inputs['n_side_pins']
                pitch = inputs['pin_pitch'] * 100  # cm
                size = n_pins * pitch
                self.dimensions_label.config(
                    text=f"Assembly: {size:.1f}×{size:.1f}cm, Pin pitch: {pitch:.2f}cm"
                )
            else:
                n_plates = inputs['plates_per_assembly']
                width = (inputs['fuel_plate_width'] +
                        2 * inputs['clad_structure_width']) * 100
                height = (n_plates * inputs['fuel_plate_pitch'] +
                         2 * inputs['clad_structure_width']) * 100
                self.dimensions_label.config(
                    text=f"Assembly: {width:.1f}×{height:.1f}cm, Plates: {n_plates}"
                )

        elif "Element" in self.main_gui.current_view:
            if inputs['assembly_type'] == 'Pin':
                r_clad = inputs['r_clad_outer'] * 100
                r_fuel = inputs['r_fuel'] * 100
                self.dimensions_label.config(
                    text=f"Pin: ⌀{r_clad*2:.3f}cm, Fuel: ⌀{r_fuel*2:.3f}cm"
                )
            else:
                width = inputs['fuel_plate_width'] * 100
                thickness = (inputs['fuel_meat_thickness'] +
                           2 * inputs['clad_thickness']) * 100
                meat_width = inputs['fuel_meat_width'] * 100
                meat_thickness = inputs['fuel_meat_thickness'] * 100
                self.dimensions_label.config(
                    text=f"Plate: {width:.1f}×{thickness:.2f}cm, "
                         f"Meat: {meat_width:.1f}×{meat_thickness:.2f}cm"
                )

    def export_values(self):
        """Export current parameter values"""
        from Inputs_GUI.utils.export_utils import export_current_values
        from tkinter import messagebox

        filename = export_current_values(self.main_gui.current_inputs)
        messagebox.showinfo("Export Complete", f"Configuration saved to {filename}")
