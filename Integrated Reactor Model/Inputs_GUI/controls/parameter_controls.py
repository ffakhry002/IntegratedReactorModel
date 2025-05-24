# Parameter control widgets and handlers

"""
Parameter Controls
Handles parameter input controls for the GUI
"""
import tkinter as tk
from tkinter import ttk, messagebox

from Inputs_GUI.utils.export_utils import export_current_values


def add_text_control(parent, label, param_key, main_gui, callback=None, scale=1, offset=0):
    """Add a labeled text input control"""
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=2)

    # Label
    ttk.Label(frame, text=label).pack(anchor=tk.W)

    # Current value from inputs
    current_val = (main_gui.current_inputs[param_key] + offset) * scale

    # Text entry
    var = tk.StringVar(value=f"{current_val:.6g}")
    main_gui.control_vars[param_key] = (var, scale, offset)

    entry = ttk.Entry(frame, textvariable=var, width=15)
    entry.pack(fill=tk.X)

    # Update function
    def update_value(*args):
        try:
            val = float(var.get())
            # Update inputs (reverse scaling and offset)
            main_gui.current_inputs[param_key] = (val / scale) - offset
            entry.config(style="TEntry")  # Reset to normal style
            if callback:
                callback()
        except ValueError:
            entry.config(style="Error.TEntry")  # Set error style

    var.trace('w', update_value)

    # Bind Enter key
    entry.bind('<Return>', lambda e: update_value())

    return frame


class ParameterControls:
    """Parameter control panel"""

    def __init__(self, parent, main_gui):
        self.parent = parent
        self.main_gui = main_gui

        # Control variables
        self.assembly_var = tk.StringVar(value=main_gui.current_inputs["assembly_type"])
        self.coolant_var = tk.StringVar(value=main_gui.current_inputs["coolant_type"])
        self.fuel_var = tk.StringVar(value=main_gui.current_inputs["fuel_type"])
        self.clad_var = tk.StringVar(value=main_gui.current_inputs["clad_type"])
        self.reflector_var = tk.StringVar(value=main_gui.current_inputs["reflector_material"])
        self.bioshield_var = tk.StringVar(value=main_gui.current_inputs["bioshield_material"])
        self.irrad_clad_var = tk.BooleanVar(value=main_gui.current_inputs["irradiation_clad"])
        self.irrad_fill_var = tk.StringVar(value=main_gui.current_inputs["irradiation_cell_fill"])

    def setup(self):
        """Setup parameter controls in proper order"""
        # Assembly Type
        assembly_frame = ttk.LabelFrame(self.parent, text="Assembly Type", padding=10)
        assembly_frame.pack(fill=tk.X, pady=(0, 10))

        assembly_type_frame = ttk.Frame(assembly_frame)
        assembly_type_frame.pack(fill=tk.X, pady=5)
        ttk.Label(assembly_type_frame, text="Assembly Type:").pack(anchor=tk.W)
        assembly_combo = ttk.Combobox(assembly_type_frame, textvariable=self.assembly_var,
                                    values=["Pin", "Plate"], state="readonly")
        assembly_combo.pack(fill=tk.X)
        assembly_combo.bind('<<ComboboxSelected>>', self.on_assembly_change)

        add_text_control(assembly_frame, "Core Power (MW)", "core_power",
                        self.main_gui, self.schedule_update)

        # Reactor Geometry
        geom_frame = ttk.LabelFrame(self.parent, text="Reactor Geometry", padding=10)
        geom_frame.pack(fill=tk.X, pady=(0, 10))

        # Radial geometry
        ttk.Label(geom_frame, text="Radial Geometry:",
                 font=('Arial', 9, 'bold')).pack(anchor=tk.W)
        add_text_control(geom_frame, "Tank Radius (m)", "tank_radius",
                        self.main_gui, self.schedule_update)
        add_text_control(geom_frame, "Reflector Thickness (m)", "reflector_thickness",
                        self.main_gui, self.schedule_update)
        add_text_control(geom_frame, "Bioshield Thickness (m)", "bioshield_thickness",
                        self.main_gui, self.schedule_update)

        # Axial geometry
        ttk.Label(geom_frame, text="Axial Geometry:",
                 font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(10, 0))
        add_text_control(geom_frame, "Bottom Bioshield Thickness (m)", "bottom_bioshield_thickness",
                        self.main_gui, self.schedule_update)
        add_text_control(geom_frame, "Bottom Reflector Thickness (m)", "bottom_reflector_thickness",
                        self.main_gui, self.schedule_update)
        add_text_control(geom_frame, "Feed Thickness (m)", "feed_thickness",
                        self.main_gui, self.schedule_update)
        add_text_control(geom_frame, "Fuel Height (m)", "fuel_height",
                        self.main_gui, self.schedule_update)
        add_text_control(geom_frame, "Plenum Height (m)", "plenum_height",
                        self.main_gui, self.schedule_update)
        add_text_control(geom_frame, "Top Reflector Thickness (m)", "top_reflector_thickness",
                        self.main_gui, self.schedule_update)
        add_text_control(geom_frame, "Top Bioshield Thickness (m)", "top_bioshield_thickness",
                        self.main_gui, self.schedule_update)

        # Assembly Parameters (conditionally shown)
        self.setup_assembly_parameters()

        # Material Configuration
        mat_frame = ttk.LabelFrame(self.parent, text="Materials", padding=10)
        mat_frame.pack(fill=tk.X, pady=(0, 10))

        self.setup_material_controls(mat_frame)

        # Irradiation Position Parameters
        self.setup_irradiation_controls()

    def setup_assembly_parameters(self):
        """Setup assembly-specific parameter controls"""
        # Pin Assembly Parameters
        self.pin_frame = ttk.LabelFrame(self.parent, text="Pin Assembly Parameters", padding=10)

        add_text_control(self.pin_frame, "Pin Pitch (mm)", "pin_pitch",
                        self.main_gui, self.schedule_update, scale=1000)
        add_text_control(self.pin_frame, "Fuel Radius (mm)", "r_fuel",
                        self.main_gui, self.schedule_update, scale=1000)
        add_text_control(self.pin_frame, "Clad Inner Radius (mm)", "r_clad_inner",
                        self.main_gui, self.schedule_update, scale=1000)
        add_text_control(self.pin_frame, "Clad Outer Radius (mm)", "r_clad_outer",
                        self.main_gui, self.schedule_update, scale=1000)
        add_text_control(self.pin_frame, "Pins per Side", "n_side_pins",
                        self.main_gui, self.on_pin_count_change)

        # Plate Assembly Parameters
        self.plate_frame = ttk.LabelFrame(self.parent, text="Plate Assembly Parameters", padding=10)

        add_text_control(self.plate_frame, "Fuel Meat Width (cm)", "fuel_meat_width",
                        self.main_gui, self.schedule_update, scale=100)
        add_text_control(self.plate_frame, "Fuel Plate Width (cm)", "fuel_plate_width",
                        self.main_gui, self.schedule_update, scale=100)
        add_text_control(self.plate_frame, "Plate Pitch (mm)", "fuel_plate_pitch",
                        self.main_gui, self.schedule_update, scale=1000)
        add_text_control(self.plate_frame, "Fuel Meat Thickness (mm)", "fuel_meat_thickness",
                        self.main_gui, self.schedule_update, scale=1000)
        add_text_control(self.plate_frame, "Clad Thickness (mm)", "clad_thickness",
                        self.main_gui, self.schedule_update, scale=1000)
        add_text_control(self.plate_frame, "Plates per Assembly", "plates_per_assembly",
                        self.main_gui, self.schedule_update)
        add_text_control(self.plate_frame, "Clad Structure Width (mm)", "clad_structure_width",
                        self.main_gui, self.schedule_update, scale=1000)

        # Show correct frame
        self.update_assembly_controls()

    def setup_material_controls(self, parent):
        """Setup material selection controls"""
        # Coolant Type
        coolant_frame = ttk.Frame(parent)
        coolant_frame.pack(fill=tk.X, pady=2)
        ttk.Label(coolant_frame, text="Coolant:").pack(anchor=tk.W)
        coolant_combo = ttk.Combobox(coolant_frame, textvariable=self.coolant_var,
                                   values=["Light Water", "Heavy Water"], state="readonly")
        coolant_combo.pack(fill=tk.X)
        coolant_combo.bind('<<ComboboxSelected>>', self.on_material_change)

        # Fuel Type
        fuel_frame = ttk.Frame(parent)
        fuel_frame.pack(fill=tk.X, pady=2)
        ttk.Label(fuel_frame, text="Fuel:").pack(anchor=tk.W)
        fuel_combo = ttk.Combobox(fuel_frame, textvariable=self.fuel_var,
                                values=["U3Si2", "UO2", "U10Mo"], state="readonly")
        fuel_combo.pack(fill=tk.X)
        fuel_combo.bind('<<ComboboxSelected>>', self.on_material_change)

        # Cladding Type
        clad_frame = ttk.Frame(parent)
        clad_frame.pack(fill=tk.X, pady=2)
        ttk.Label(clad_frame, text="Cladding:").pack(anchor=tk.W)
        clad_combo = ttk.Combobox(clad_frame, textvariable=self.clad_var,
                                values=["Al6061", "Zirc2", "Zirc4"], state="readonly")
        clad_combo.pack(fill=tk.X)
        clad_combo.bind('<<ComboboxSelected>>', self.on_material_change)

        # Reflector Material
        refl_frame = ttk.Frame(parent)
        refl_frame.pack(fill=tk.X, pady=2)
        ttk.Label(refl_frame, text="Reflector:").pack(anchor=tk.W)
        reflector_combo = ttk.Combobox(refl_frame, textvariable=self.reflector_var,
                                     values=["mgo", "beryllium"], state="readonly")
        reflector_combo.pack(fill=tk.X)
        reflector_combo.bind('<<ComboboxSelected>>', self.on_material_change)

        # Bioshield Material
        bio_frame = ttk.Frame(parent)
        bio_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bio_frame, text="Bioshield:").pack(anchor=tk.W)
        bioshield_combo = ttk.Combobox(bio_frame, textvariable=self.bioshield_var,
                                     values=["Concrete", "Steel"], state="readonly")
        bioshield_combo.pack(fill=tk.X)
        bioshield_combo.bind('<<ComboboxSelected>>', self.on_material_change)

        # Fuel Enrichment
        ttk.Label(parent, text="Fuel Enrichment:",
                 font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(10, 0))
        add_text_control(parent, "Standard Enrichment (%)", "n%",
                        self.main_gui, self.schedule_update)
        add_text_control(parent, "Enhanced Enrichment (%)", "n%E",
                        self.main_gui, self.schedule_update)

    def setup_irradiation_controls(self):
        """Setup irradiation position controls"""
        irrad_frame = ttk.LabelFrame(self.parent, text="Irradiation Positions", padding=10)
        irrad_frame.pack(fill=tk.X, pady=(0, 10))

        # Irradiation cladding checkbox
        ttk.Checkbutton(irrad_frame, text="Include Irradiation Cladding",
                       variable=self.irrad_clad_var,
                       command=self.on_irrad_change).pack(anchor=tk.W)

        add_text_control(irrad_frame, "Irradiation Clad Thickness (mm)", "irradiation_clad_thickness",
                        self.main_gui, self.schedule_update, scale=1000)

        # Irradiation cell fill
        irrad_fill_frame = ttk.Frame(irrad_frame)
        irrad_fill_frame.pack(fill=tk.X, pady=2)
        ttk.Label(irrad_fill_frame, text="Irradiation Cell Fill:").pack(anchor=tk.W)
        irrad_fill_combo = ttk.Combobox(irrad_fill_frame, textvariable=self.irrad_fill_var,
                                      values=["Vacuum", "fill"], state="readonly")
        irrad_fill_combo.pack(fill=tk.X)
        irrad_fill_combo.bind('<<ComboboxSelected>>', self.on_irrad_change)

    def on_irrad_change(self, event=None):
        """Handle irradiation position parameter changes"""
        self.main_gui.current_inputs["irradiation_clad"] = self.irrad_clad_var.get()
        self.main_gui.current_inputs["irradiation_cell_fill"] = self.irrad_fill_var.get()
        self.schedule_update()

    def on_pin_count_change(self):
        """Handle pin count change - update design tab if needed"""
        # Update design tab pin grid if it exists
        if hasattr(self.main_gui, 'design_tab') and hasattr(self.main_gui.design_tab, 'pin_buttons'):
            # Sync guide tube positions first
            self.main_gui.design_tab.design_inputs['guide_tube_positions'] = self.main_gui.current_inputs.get('guide_tube_positions', [])[:]
            # Find the pin grid parent and refresh it
            if self.main_gui.design_tab.pin_buttons:
                parent = self.main_gui.design_tab.pin_buttons[0][0].master.master
                self.main_gui.design_tab.setup_pin_grid(parent)

        self.schedule_update()

    def schedule_update(self):
        """Schedule visualization update if auto-update enabled"""
        if hasattr(self.main_gui, 'viz_tab') and self.main_gui.viz_tab.auto_update_var.get():
            self.main_gui.schedule_update()

    def on_assembly_change(self, event=None):
        """Handle assembly type change"""
        self.main_gui.current_inputs["assembly_type"] = self.assembly_var.get()
        self.update_assembly_controls()

        # Update design tab visibility if it exists
        if hasattr(self.main_gui, 'design_tab'):
            self.main_gui.design_tab.design_inputs["assembly_type"] = self.assembly_var.get()
            self.main_gui.design_tab.update_pin_layout_visibility()

        self.schedule_update()

    def on_material_change(self, event=None):
        """Handle material changes"""
        self.main_gui.current_inputs["coolant_type"] = self.coolant_var.get()
        self.main_gui.current_inputs["fuel_type"] = self.fuel_var.get()
        self.main_gui.current_inputs["clad_type"] = self.clad_var.get()
        self.main_gui.current_inputs["reflector_material"] = self.reflector_var.get()
        self.main_gui.current_inputs["bioshield_material"] = self.bioshield_var.get()
        self.schedule_update()

    def update_assembly_controls(self):
        """Show/hide assembly-specific controls"""
        if self.main_gui.current_inputs["assembly_type"] == "Pin":
            self.pin_frame.pack(fill=tk.X, pady=(0, 10))
            self.plate_frame.pack_forget()
        else:
            self.plate_frame.pack(fill=tk.X, pady=(0, 10))
            self.pin_frame.pack_forget()
