# Advanced settings tab component

"""
Advanced Tab Component
Handles advanced OpenMC and depletion parameters
"""
import tkinter as tk
from tkinter import ttk, messagebox


class AdvancedTab:
    def __init__(self, parent, main_gui):
        self.parent = parent
        self.main_gui = main_gui

        # Advanced parameter variables
        self.timestep_entries = []
        self.advanced_vars = {}

    def setup(self):
        """Setup the advanced inputs tab with two-panel layout"""
        # Main container to use full window
        main_container = ttk.Frame(self.parent)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create left and right panels
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Create scrollable frames for both panels
        # Left panel with scrolling
        left_canvas = tk.Canvas(left_panel)
        left_scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=left_canvas.yview)
        left_scrollable = ttk.Frame(left_canvas)

        left_scrollable.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )

        left_canvas.create_window((0, 0), window=left_scrollable, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")

        # Right panel with scrolling
        right_canvas = tk.Canvas(right_panel)
        right_scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=right_canvas.yview)
        right_scrollable = ttk.Frame(right_canvas)

        right_scrollable.bind(
            "<Configure>",
            lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        )

        right_canvas.create_window((0, 0), window=right_scrollable, anchor="nw")
        right_canvas.configure(yscrollcommand=right_scrollbar.set)

        right_canvas.pack(side="left", fill="both", expand=True)
        right_scrollbar.pack(side="right", fill="y")

        # Add mouse wheel support for both panels
        def on_left_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def on_right_mousewheel(event):
            right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        # Bind mousewheel to each canvas individually
        left_canvas.bind("<MouseWheel>", on_left_mousewheel)
        left_scrollable.bind("<MouseWheel>", on_left_mousewheel)
        right_canvas.bind("<MouseWheel>", on_right_mousewheel)
        right_scrollable.bind("<MouseWheel>", on_right_mousewheel)

        # Create sections in left panel
        self.create_parametric_section(left_scrollable)
        self.create_transport_section(left_scrollable)

        # Export button in left panel
        ttk.Button(left_scrollable, text="Export Current Configuration",
                  command=self.export_configuration).pack(fill='x', padx=10, pady=10)

        # Create sections in right panel
        self.create_depletion_section(right_scrollable)
        self.create_misc_section(right_scrollable)

    def create_parametric_section(self, parent):
        """Create parametric study section"""
        ttk.Label(parent, text="Parametric Study Configuration",
                 font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(10,5))

        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(frame, text="Parametric Study:").grid(row=0, column=0, sticky='w')
        self.parametric_var = tk.StringVar(
            value=str(self.main_gui.current_inputs.get('parametric_study', True))
        )
        combo = ttk.Combobox(frame, textvariable=self.parametric_var,
                           values=["True", "False"], width=15, state="readonly")
        combo.grid(row=0, column=1, sticky='w', padx=5)
        combo.bind('<<ComboboxSelected>>',
                  lambda e: self.update_input('parametric_study', self.parametric_var.get() == "True"))

    def create_transport_section(self, parent):
        """Create OpenMC transport section"""
        ttk.Label(parent, text="OpenMC Transport Parameters",
                 font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(15,5))

        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=10, pady=5)

        row = 0

        # Basic settings
        self.add_number_control(frame, "Active Batches:", 'batches', row)
        row += 1
        self.add_number_control(frame, "Inactive Batches:", 'inactive', row)
        row += 1
        self.add_number_control(frame, "Particles per Batch:", 'particles', row)
        row += 1

        # Energy structure
        ttk.Label(frame, text="Energy Structure:").grid(row=row, column=0, sticky='w')
        self.energy_var = tk.StringVar(
            value=self.main_gui.current_inputs.get('energy_structure', 'log1001')
        )
        combo = ttk.Combobox(frame, textvariable=self.energy_var,
                           values=['log1001', 'log501', 'scale238'],
                           width=15, state="readonly")
        combo.grid(row=row, column=1, sticky='w', padx=5)
        combo.bind('<<ComboboxSelected>>',
                  lambda e: self.update_input('energy_structure', self.energy_var.get()))
        row += 1

        # Energy boundaries
        self.add_number_control(frame, "Thermal Cutoff (eV):", 'thermal_cutoff', row)
        row += 1
        self.add_number_control(frame, "Fast Cutoff (eV):", 'fast_cutoff', row)
        row += 1

        # Tally settings subtitle
        ttk.Label(frame, text="Tally Configuration",
                 font=('TkDefaultFont', 9, 'bold')).grid(
                     row=row, column=0, columnspan=2, sticky='w', pady=(10,5))
        row += 1

        # Tally segments
        self.add_number_control(frame, "Power Tally Axial Segments:",
                               'power_tally_axial_segments', row)
        row += 1
        self.add_number_control(frame, "Irradiation Axial Segments:",
                               'irradiation_axial_segments', row)
        row += 1
        self.add_number_control(frame, "Core Mesh Dimension:",
                               'core_mesh_dimension', row)
        row += 1
        self.add_number_control(frame, "Entropy Mesh Dimension:",
                               'entropy_mesh_dimension', row)
        row += 1

        # Boolean tallies
        self.add_boolean_control(frame, "Core Three Group Energy Bins:",
                                'Core_Three_Group_Energy_Bins', row)
        row += 1
        self.add_boolean_control(frame, "Tally Power:", 'tally_power', row)
        row += 1
        self.add_boolean_control(frame, "Element Level Power Tallies:",
                                'element_level_power_tallies', row)

    def create_depletion_section(self, parent):
        """Create depletion calculation section"""
        ttk.Label(parent, text="Depletion Calculation Parameters",
                 font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(15,5))

        # Depletion scenario selection
        scenario_frame = ttk.LabelFrame(parent, text="Depletion Scenarios (Multiple Can Be Selected)", padding=10)
        scenario_frame.pack(fill='x', padx=10, pady=5)

        # Depletion scenario checkboxes
        self.depletion_vars = {}
        depletion_keys = {
            'deplete_core': 'Full Core Depletion',
            'deplete_assembly': 'Single Assembly',
            'deplete_assembly_enhanced': 'Enhanced Assembly',
            'deplete_element': 'Single Element',
            'deplete_element_enhanced': 'Enhanced Element'
        }

        # Create checkboxes for each scenario
        for key, label in depletion_keys.items():
            var = tk.BooleanVar(value=self.main_gui.current_inputs.get(key, False))
            self.depletion_vars[key] = var
            checkbox = ttk.Checkbutton(scenario_frame, text=label, variable=var,
                                     command=lambda k=key: self.update_depletion_checkbox(k))
            checkbox.pack(anchor='w')

        # Depletion settings
        settings_frame = ttk.Frame(parent)
        settings_frame.pack(fill='x', padx=10, pady=5)

        row = 0

        # Transport settings
        self.add_number_control(settings_frame, "Depletion Particles:",
                               'depletion_particles', row)
        row += 1
        self.add_number_control(settings_frame, "Depletion Batches:",
                               'depletion_batches', row)
        row += 1
        self.add_number_control(settings_frame, "Depletion Inactive:",
                               'depletion_inactive', row)
        row += 1

        # Integrator
        ttk.Label(settings_frame, text="Depletion Integrator:").grid(
            row=row, column=0, sticky='w')
        self.integrator_var = tk.StringVar(
            value=self.main_gui.current_inputs.get('depletion_integrator', 'predictor')
        )
        combo = ttk.Combobox(settings_frame, textvariable=self.integrator_var,
                           values=['predictor', 'cecm', 'celi', 'cf4',
                                  'epcrk4', 'leqi', 'siceli', 'sileqi'],
                           width=15, state="readonly")
        combo.grid(row=row, column=1, sticky='w', padx=5)
        combo.bind('<<ComboboxSelected>>',
                  lambda e: self.update_input('depletion_integrator',
                                            self.integrator_var.get()))
        row += 1

        # Chain
        ttk.Label(settings_frame, text="Depletion Chain:").grid(
            row=row, column=0, sticky='w')
        self.chain_var = tk.StringVar(
            value=self.main_gui.current_inputs.get('depletion_chain', 'casl')
        )
        combo = ttk.Combobox(settings_frame, textvariable=self.chain_var,
                           values=['casl', 'endfb71'], width=15, state="readonly")
        combo.grid(row=row, column=1, sticky='w', padx=5)
        combo.bind('<<ComboboxSelected>>',
                  lambda e: self.update_input('depletion_chain', self.chain_var.get()))
        row += 1

        # Timestep units
        ttk.Label(settings_frame, text="Timestep Units:").grid(
            row=row, column=0, sticky='w')
        self.units_var = tk.StringVar(
            value=self.main_gui.current_inputs.get('depletion_timestep_units', 'MWd/kgHM')
        )
        combo = ttk.Combobox(settings_frame, textvariable=self.units_var,
                           values=['MWd/kgHM', 'days'], width=15, state="readonly")
        combo.grid(row=row, column=1, sticky='w', padx=5)
        combo.bind('<<ComboboxSelected>>',
                  lambda e: self.update_input('depletion_timestep_units',
                                            self.units_var.get()))

        # Timesteps
        self.create_timestep_section(parent)

    def create_timestep_section(self, parent):
        """Create timestep management section"""
        ttk.Label(parent, text="Depletion Timesteps",
                 font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', pady=(10,5))

        self.timestep_frame = ttk.Frame(parent)
        self.timestep_frame.pack(fill='x', padx=10, pady=5)

        # Headers
        ttk.Label(self.timestep_frame, text="Steps").grid(row=0, column=0, padx=5)
        ttk.Label(self.timestep_frame, text="Size").grid(row=0, column=1, padx=5)
        ttk.Label(self.timestep_frame, text="Actions").grid(row=0, column=2, padx=5)

        # Initialize timestep entries
        self.setup_timestep_entries()

    def setup_timestep_entries(self):
        """Setup timestep entry widgets"""
        # Clear existing
        for widget in self.timestep_frame.winfo_children():
            if widget.grid_info()['row'] > 0:
                widget.destroy()
        self.timestep_entries.clear()

        # Get current timesteps
        current_timesteps = self.main_gui.current_inputs.get('depletion_timesteps', [
            {"steps": 5, "size": 1},
            {"steps": 5, "size": 0.5},
            {"steps": 5, "size": 2.5},
            {"steps": 5, "size": 5},
            {"steps": 5, "size": 10},
        ])

        # Create entries
        for i, timestep in enumerate(current_timesteps):
            row = i + 1

            steps_var = tk.StringVar(value=str(timestep['steps']))
            steps_entry = ttk.Entry(self.timestep_frame, textvariable=steps_var, width=10)
            steps_entry.grid(row=row, column=0, padx=5, pady=2)

            size_var = tk.StringVar(value=str(timestep['size']))
            size_entry = ttk.Entry(self.timestep_frame, textvariable=size_var, width=10)
            size_entry.grid(row=row, column=1, padx=5, pady=2)

            # Store index in button for proper deletion
            delete_btn = ttk.Button(self.timestep_frame, text="Delete", width=8)
            delete_btn.timestep_index = i  # Store index as attribute
            delete_btn.configure(command=lambda b=delete_btn: self.delete_timestep_by_button(b))
            delete_btn.grid(row=row, column=2, padx=5, pady=2)

            self.timestep_entries.append((steps_var, size_var, delete_btn))

            # Bind updates
            steps_var.trace('w', self.update_timesteps)
            size_var.trace('w', self.update_timesteps)

        # Add button
        add_btn = ttk.Button(self.timestep_frame, text="Add Timestep",
                           command=self.add_timestep)
        add_btn.grid(row=len(current_timesteps) + 1, column=0, columnspan=3, pady=5)

    def create_misc_section(self, parent):
        """Create miscellaneous settings section"""
        ttk.Label(parent, text="Miscellaneous Settings",
                 font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(15,5))

        # Depletion nuclides section (moved to top)
        nuclides_frame = ttk.LabelFrame(parent, text="Depletion Nuclides to Track", padding=10)
        nuclides_frame.pack(fill='x', padx=10, pady=(5,10))

        # Instructions
        ttk.Label(nuclides_frame, text="Enter nuclides separated by commas (e.g., U235, U238, Pu239):").pack(anchor='w')

        # Text area for nuclides
        self.nuclides_var = tk.StringVar()
        current_nuclides = self.main_gui.current_inputs.get('depletion_nuclides', [])
        self.nuclides_var.set(', '.join(current_nuclides))

        nuclides_entry = ttk.Entry(nuclides_frame, textvariable=self.nuclides_var, width=60)
        nuclides_entry.pack(fill='x', pady=5)

        def update_nuclides(*args):
            nuclides_text = self.nuclides_var.get().strip()
            if nuclides_text:
                nuclides_list = [n.strip().strip("'\"") for n in nuclides_text.split(',') if n.strip()]
            else:
                nuclides_list = []
            self.main_gui.current_inputs['depletion_nuclides'] = nuclides_list

        self.nuclides_var.trace('w', update_nuclides)

        # Plot resolution section (moved to bottom)
        resolution_frame = ttk.Frame(parent)
        resolution_frame.pack(fill='x', padx=10, pady=5)

        # Pixels
        self.add_number_control(resolution_frame, "Plot Resolution (pixels):", 'pixels', 0)

    def add_number_control(self, parent, label, param_key, row):
        """Add a number input control"""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w')

        var = tk.StringVar(value=str(self.main_gui.current_inputs.get(param_key, 0)))
        entry = ttk.Entry(parent, textvariable=var, width=20)
        entry.grid(row=row, column=1, sticky='w', padx=5)

        def update_value(*args):
            try:
                if param_key in ['batches', 'inactive', 'particles',
                               'power_tally_axial_segments', 'irradiation_axial_segments',
                               'core_mesh_dimension', 'entropy_mesh_dimension',
                               'depletion_particles', 'depletion_batches',
                               'depletion_inactive', 'pixels']:
                    value = int(var.get())
                else:
                    value = float(var.get())
                self.main_gui.current_inputs[param_key] = value
            except ValueError:
                pass

        var.trace('w', update_value)
        self.advanced_vars[param_key] = var

    def add_boolean_control(self, parent, label, param_key, row):
        """Add a boolean control"""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w')

        var = tk.StringVar(value=str(self.main_gui.current_inputs.get(param_key, True)))
        combo = ttk.Combobox(parent, textvariable=var, values=["True", "False"],
                           width=15, state="readonly")
        combo.grid(row=row, column=1, sticky='w', padx=5)
        combo.bind('<<ComboboxSelected>>',
                  lambda e: self.update_input(param_key, var.get() == "True"))

        self.advanced_vars[param_key] = var

    def update_input(self, key, value):
        """Update an input parameter"""
        self.main_gui.current_inputs[key] = value

    def update_depletion_checkbox(self, key):
        """Update depletion scenario checkbox"""
        self.main_gui.current_inputs[key] = self.depletion_vars[key].get()

    def update_timesteps(self, *args):
        """Update timesteps from entries"""
        new_timesteps = []
        for entry_tuple in self.timestep_entries:
            steps_var, size_var = entry_tuple[0], entry_tuple[1]
            try:
                steps = int(steps_var.get())
                size = float(size_var.get())
                new_timesteps.append({"steps": steps, "size": size})
            except ValueError:
                continue
        self.main_gui.current_inputs['depletion_timesteps'] = new_timesteps

    def add_timestep(self):
        """Add a new timestep"""
        current = self.main_gui.current_inputs.get('depletion_timesteps', [])
        current.append({"steps": 1, "size": 1.0})
        self.main_gui.current_inputs['depletion_timesteps'] = current
        self.setup_timestep_entries()

    def delete_timestep_by_button(self, button):
        """Delete a timestep by button reference"""
        index = button.timestep_index
        current = self.main_gui.current_inputs.get('depletion_timesteps', [])
        if 0 <= index < len(current):
            current.pop(index)
            self.main_gui.current_inputs['depletion_timesteps'] = current
            self.setup_timestep_entries()

    def delete_timestep(self, index):
        """Delete a timestep"""
        current = self.main_gui.current_inputs.get('depletion_timesteps', [])
        if 0 <= index < len(current):
            current.pop(index)
            self.main_gui.current_inputs['depletion_timesteps'] = current
            self.setup_timestep_entries()

    def export_configuration(self):
        """Export current configuration"""
        from Inputs_GUI.utils.export_utils import export_current_values

        filename = export_current_values(self.main_gui.current_inputs)
        messagebox.showinfo("Export Complete", f"Configuration saved to {filename}")
