"""
Simple Tab Component
Handles simple parameter studies configuration
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import designers and utils
from designers.core_lattice_designer import CoreLatticeDesigner
from designers.depletion_timesteps_designer import DepletionTimestepsDesigner
from utils.parameter_utils import ParameterUtils


class SimpleTab:
    """Simple parameter studies tab"""

    def __init__(self, parent, parameter_model, run_config):
        self.parent = parent
        self.parameter_model = parameter_model
        self.run_config = run_config

        # UI elements
        self.param_tree = None
        self.runs_listbox = None
        self.param_info_label = None
        self.value_input_frame = None

        # Current selection
        self.current_param = None

    def setup(self):
        """Setup the simple parameter studies tab"""
        # Main container
        container = ttk.Frame(self.parent)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - parameter selection
        left_frame = ttk.LabelFrame(container, text="Parameter Selection", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Search/filter
        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        self.search_var.trace('w', self.filter_parameters)

        # Parameter list with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        # Create treeview for parameters with categories
        columns = ('Parameter', 'Current Value', 'Type')
        self.param_tree = ttk.Treeview(list_frame, columns=columns, show='tree headings', height=15)

        # Define headings
        self.param_tree.heading('#0', text='Category')
        self.param_tree.heading('Parameter', text='Parameter')
        self.param_tree.heading('Current Value', text='Current Value')
        self.param_tree.heading('Type', text='Type')

        # Configure column widths
        self.param_tree.column('#0', width=150)
        self.param_tree.column('Parameter', width=200)
        self.param_tree.column('Current Value', width=100)
        self.param_tree.column('Type', width=80)

        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.param_tree.yview)
        self.param_tree.configure(yscrollcommand=tree_scroll.set)

        self.param_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Populate parameter list
        self.populate_parameter_list()

        # Bind selection event
        self.param_tree.bind('<<TreeviewSelect>>', self.on_parameter_select)

        # Right panel - run configuration
        right_frame = ttk.LabelFrame(container, text="Run Configuration", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Create scrollable right panel
        right_canvas = tk.Canvas(right_frame, width=400)
        right_scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=right_canvas.yview)
        self.scrollable_right = ttk.Frame(right_canvas)

        self.scrollable_right.bind(
            "<Configure>",
            lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        )

        right_canvas.create_window((0, 0), window=self.scrollable_right, anchor="nw")
        right_canvas.configure(yscrollcommand=right_scrollbar.set)

        right_canvas.pack(side="left", fill="both", expand=True)
        right_scrollbar.pack(side="right", fill="y")

        # Selected parameter info
        info_frame = ttk.LabelFrame(self.scrollable_right, text="Selected Parameter", padding=5)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.param_info_label = ttk.Label(info_frame, text="Select a parameter from the list")
        self.param_info_label.pack()

        # Value input frame (will be populated dynamically)
        self.value_input_frame = ttk.LabelFrame(self.scrollable_right, text="Values to Test", padding=10)
        self.value_input_frame.pack(fill=tk.X, pady=(0, 10))

        self.no_param_label = ttk.Label(self.value_input_frame, text="No parameter selected")
        self.no_param_label.pack()

        # Run list with improved scrolling
        runs_frame = ttk.LabelFrame(self.scrollable_right, text="Configured Runs", padding=5)
        runs_frame.pack(fill=tk.BOTH, expand=True)

        # Run listbox with scrollbar
        runs_list_frame = ttk.Frame(runs_frame)
        runs_list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.runs_listbox = tk.Listbox(runs_list_frame, height=10)
        runs_scroll = ttk.Scrollbar(runs_list_frame, orient=tk.VERTICAL, command=self.runs_listbox.yview)
        self.runs_listbox.configure(yscrollcommand=runs_scroll.set)

        self.runs_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        runs_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Buttons
        buttons_frame = ttk.Frame(runs_frame)
        buttons_frame.pack(fill=tk.X)

        ttk.Button(buttons_frame, text="Remove Selected",
                  command=self.remove_selected_run).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buttons_frame, text="Clear All",
                  command=self.clear_all_runs).pack(side=tk.LEFT)

        # Load existing runs if any
        self.refresh_runs_display()

    def populate_parameter_list(self):
        """Populate the parameter list treeview with categories"""
        # Clear existing items
        for item in self.param_tree.get_children():
            self.param_tree.delete(item)

        # Group parameters by category
        categories = {}
        for param_name, param_info in self.parameter_model.available_params.items():
            category = param_info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((param_name, param_info))

        # Add category headers and parameters
        for category, params in sorted(categories.items()):
            # Create category header
            category_id = self.param_tree.insert('', 'end', text=category, values=('', '', ''))

            # Add parameters under category
            for param_name, param_info in sorted(params):
                self.param_tree.insert(category_id, 'end', text='', values=(
                    param_name,
                    str(param_info['current_value'])[:20] + ('...' if len(str(param_info['current_value'])) > 20 else ''),
                    param_info['type']
                ))

        # Expand all categories
        for item in self.param_tree.get_children():
            self.param_tree.item(item, open=True)

    def filter_parameters(self, *args):
        """Filter parameters based on search text"""
        search_text = self.search_var.get().lower()

        # Clear existing items
        for item in self.param_tree.get_children():
            self.param_tree.delete(item)

        if not search_text:
            self.populate_parameter_list()
            return

        # Group filtered parameters by category
        categories = {}
        for param_name, param_info in self.parameter_model.available_params.items():
            if (search_text in param_name.lower() or
                search_text in param_info['description'].lower() or
                search_text in param_info['category'].lower()):

                category = param_info['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append((param_name, param_info))

        # Add filtered category headers and parameters
        for category, params in sorted(categories.items()):
            if params:  # Only show categories that have matching parameters
                category_id = self.param_tree.insert('', 'end', text=category, values=('', '', ''))

                for param_name, param_info in sorted(params):
                    self.param_tree.insert(category_id, 'end', text='', values=(
                        param_name,
                        str(param_info['current_value'])[:20] + ('...' if len(str(param_info['current_value'])) > 20 else ''),
                        param_info['type']
                    ))

        # Expand all categories
        for item in self.param_tree.get_children():
            self.param_tree.item(item, open=True)

    def on_parameter_select(self, event):
        """Handle parameter selection"""
        selection = self.param_tree.selection()
        if not selection:
            return

        item = self.param_tree.item(selection[0])
        values = item['values']

        # Check if this is a parameter (has values) or just a category header
        if not values or not values[0]:  # Category header
            return

        param_name = values[0]
        if param_name not in self.parameter_model.available_params:
            return

        self.current_param = param_name
        param_info = self.parameter_model.available_params[param_name]

        # Update info label
        self.param_info_label.config(
            text=f"{param_name}: {param_info['description']}\nCurrent: {param_info['current_value']} ({param_info['type']})"
        )

        # Clear and rebuild value input frame
        for widget in self.value_input_frame.winfo_children():
            widget.destroy()

        self.setup_value_input(param_name, param_info)

    def setup_value_input(self, param_name, param_info):
        """Setup value input widgets based on parameter type"""
        param_type = param_info['type']
        current_value = param_info['current_value']

        if param_type == 'bool':
            self.setup_bool_input(param_name, current_value)
        elif param_type in ['int', 'float']:
            self.setup_numeric_input(param_name, param_type, current_value)
        elif param_type == 'str':
            self.setup_string_input(param_name, current_value)
        elif param_type == 'core_lattice':
            self.setup_core_lattice_input(param_name, current_value)
        elif param_type == 'depletion_timesteps':
            self.setup_depletion_timesteps_input(param_name, current_value)

    def setup_bool_input(self, param_name, current_value):
        """Setup boolean parameter input"""
        ttk.Label(self.value_input_frame, text="Select values to test:").pack(anchor=tk.W)

        self.bool_vars = {}
        for value in [True, False]:
            var = tk.BooleanVar(value=(value == current_value))
            self.bool_vars[value] = var
            cb = ttk.Checkbutton(self.value_input_frame, text=str(value), variable=var)
            cb.pack(anchor=tk.W)

        # Button frame
        button_frame = ttk.Frame(self.value_input_frame)
        button_frame.pack(pady=10, fill=tk.X)

        ttk.Button(button_frame, text="Add as New Run(s)",
                  command=lambda: self.add_bool_runs(param_name)).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(button_frame, text="Stack to Last Run",
                  command=lambda: self.stack_bool_to_last_run(param_name)).pack(side=tk.LEFT)

    def setup_numeric_input(self, param_name, param_type, current_value):
        """Setup numeric parameter input"""
        # Single value input
        single_frame = ttk.LabelFrame(self.value_input_frame, text="Single Value", padding=5)
        single_frame.pack(fill=tk.X, pady=(0, 10))

        entry_frame = ttk.Frame(single_frame)
        entry_frame.pack(fill=tk.X)

        ttk.Label(entry_frame, text="Value:").pack(side=tk.LEFT)
        self.single_value_var = tk.StringVar(value=str(current_value))
        value_entry = ttk.Entry(entry_frame, textvariable=self.single_value_var)
        value_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

        # Buttons
        button_frame = ttk.Frame(single_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(button_frame, text="Add as New Run",
                  command=lambda: self.add_single_numeric_run(param_name, param_type)).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(button_frame, text="Stack to Last Run",
                  command=lambda: self.stack_single_numeric_to_last_run(param_name, param_type)).pack(side=tk.LEFT)

        # Range input
        range_frame = ttk.LabelFrame(self.value_input_frame, text="Value Range", padding=5)
        range_frame.pack(fill=tk.X, pady=(0, 10))

        # Start value
        start_frame = ttk.Frame(range_frame)
        start_frame.pack(fill=tk.X)
        ttk.Label(start_frame, text="Start:").pack(side=tk.LEFT)
        self.range_start_var = tk.StringVar(value=str(current_value * 0.5))
        ttk.Entry(start_frame, textvariable=self.range_start_var, width=15).pack(side=tk.LEFT, padx=(5, 10))

        # End value
        ttk.Label(start_frame, text="End:").pack(side=tk.LEFT)
        self.range_end_var = tk.StringVar(value=str(current_value * 1.5))
        ttk.Entry(start_frame, textvariable=self.range_end_var, width=15).pack(side=tk.LEFT, padx=(5, 10))

        # Number of steps
        ttk.Label(start_frame, text="Steps:").pack(side=tk.LEFT)
        self.range_steps_var = tk.StringVar(value="5")
        ttk.Entry(start_frame, textvariable=self.range_steps_var, width=8).pack(side=tk.LEFT, padx=(5, 5))

        ttk.Button(start_frame, text="Add Range",
                  command=lambda: self.add_numeric_range_runs(param_name, param_type)).pack(side=tk.LEFT, padx=(5, 0))

        # List input
        list_frame = ttk.LabelFrame(self.value_input_frame, text="Value List", padding=5)
        list_frame.pack(fill=tk.X)

        ttk.Label(list_frame, text="Values (comma-separated):").pack(anchor=tk.W)
        self.list_values_var = tk.StringVar(value=f"{current_value}, {current_value*0.8}, {current_value*1.2}")
        list_entry = ttk.Entry(list_frame, textvariable=self.list_values_var)
        list_entry.pack(fill=tk.X, pady=(2, 5))

        ttk.Button(list_frame, text="Add List",
                  command=lambda: self.add_numeric_list_runs(param_name, param_type)).pack()

    def setup_string_input(self, param_name, current_value):
        """Setup string parameter input"""
        string_options = self.parameter_model.get_string_options(param_name)

        if string_options:
            # Dropdown with options
            ttk.Label(self.value_input_frame, text="Options:").pack(anchor=tk.W)

            self.string_vars = {}
            for option in string_options:
                var = tk.BooleanVar(value=(option == current_value))
                self.string_vars[option] = var
                cb = ttk.Checkbutton(self.value_input_frame, text=option, variable=var)
                cb.pack(anchor=tk.W)

            # Add buttons
            button_frame = ttk.Frame(self.value_input_frame)
            button_frame.pack(pady=10, fill=tk.X)

            ttk.Button(button_frame, text="Add as New Run(s)",
                      command=lambda: self.add_string_runs(param_name)).pack(side=tk.LEFT, padx=(0, 5))

            ttk.Button(button_frame, text="Stack to Last Run",
                      command=lambda: self.stack_string_to_last_run(param_name)).pack(side=tk.LEFT)

            # Custom value for parameters that allow it
            if param_name not in ['depletion_timestep_units', 'irradiation_cell_fill', 'energy_structure']:
                custom_frame = ttk.LabelFrame(self.value_input_frame, text="Custom Value", padding=5)
                custom_frame.pack(fill=tk.X, pady=(10, 0))

                entry_frame = ttk.Frame(custom_frame)
                entry_frame.pack(fill=tk.X)

                ttk.Label(entry_frame, text="Value:").pack(side=tk.LEFT)
                self.custom_string_var = tk.StringVar(value=current_value)
                string_entry = ttk.Entry(entry_frame, textvariable=self.custom_string_var)
                string_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

                # Buttons
                custom_button_frame = ttk.Frame(custom_frame)
                custom_button_frame.pack(fill=tk.X, pady=(5, 0))

                ttk.Button(custom_button_frame, text="Add as New Run",
                          command=lambda: self.add_custom_string_run(param_name)).pack(side=tk.LEFT, padx=(0, 5))

                ttk.Button(custom_button_frame, text="Stack to Last Run",
                          command=lambda: self.stack_custom_string_to_last_run(param_name)).pack(side=tk.LEFT)
        else:
            # Only custom value input
            entry_frame = ttk.Frame(self.value_input_frame)
            entry_frame.pack(fill=tk.X)

            ttk.Label(entry_frame, text="Value:").pack(side=tk.LEFT)
            self.custom_string_var = tk.StringVar(value=current_value)
            string_entry = ttk.Entry(entry_frame, textvariable=self.custom_string_var)
            string_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

            # Buttons
            button_frame = ttk.Frame(self.value_input_frame)
            button_frame.pack(fill=tk.X, pady=(5, 0))

            ttk.Button(button_frame, text="Add as New Run",
                      command=lambda: self.add_custom_string_run(param_name)).pack(side=tk.LEFT, padx=(0, 5))

            ttk.Button(button_frame, text="Stack to Last Run",
                      command=lambda: self.stack_custom_string_to_last_run(param_name)).pack(side=tk.LEFT)


    def setup_core_lattice_input(self, param_name, current_value):
        """Setup core_lattice parameter input with popup designer"""
        # Clear any existing widgets first
        for widget in self.value_input_frame.winfo_children():
            widget.destroy()

        # Create a simple vertical layout
        ttk.Label(self.value_input_frame, text="Visual Core Designer",
                font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        ttk.Label(self.value_input_frame,
                text="Design core lattices visually with an interactive grid:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))

        # Button with grid layout
        designer_button = ttk.Button(
            self.value_input_frame,
            text="Open Core Lattice Designer",
            command=lambda: self.open_core_designer(param_name, current_value)
        )
        designer_button.grid(row=2, column=0, pady=10, padx=20, sticky=tk.W)

        # Alternative: Also add a text entry as backup
        backup_frame = ttk.LabelFrame(self.value_input_frame, text="Alternative: Manual Entry", padding=5)
        backup_frame.grid(row=3, column=0, sticky=tk.EW, pady=10)

        ttk.Label(backup_frame, text="Enter lattice (e.g., [['C','F'],['F','C']]):").pack(anchor=tk.W)
        self.manual_lattice_var = tk.StringVar(value=str(current_value) if current_value else "")
        manual_entry = ttk.Entry(backup_frame, textvariable=self.manual_lattice_var, width=40)
        manual_entry.pack(fill=tk.X, pady=(2, 5))

        manual_buttons = ttk.Frame(backup_frame)
        manual_buttons.pack(fill=tk.X)

        ttk.Button(manual_buttons, text="Add as New Run",
                command=lambda: self.add_manual_lattice_run(param_name)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(manual_buttons, text="Stack to Last Run",
                command=lambda: self.stack_manual_lattice_run(param_name)).pack(side=tk.LEFT)

        # Instructions
        ttk.Label(self.value_input_frame,
                text="Click the button above to open the visual core designer").grid(row=4, column=0, sticky=tk.W, pady=(5, 5))

        # Features
        features_text = """• Interactive grid with click-to-edit cells
    - Live preview with color coding
    - Preset configurations (4x4, 6x6, 7x7, 8x8)
    - Automatic irradiation position numbering
    - Apply as new run or stack to existing run"""

        ttk.Label(self.value_input_frame, text=features_text, font=('TkDefaultFont', 8),
                justify=tk.LEFT).grid(row=5, column=0, sticky=tk.W, padx=(10, 0))

        # Configure grid weights
        self.value_input_frame.grid_columnconfigure(0, weight=1)








    def setup_depletion_timesteps_input(self, param_name, current_value):
        """Setup depletion timesteps input"""
        # Visual designer button
        designer_frame = ttk.LabelFrame(self.value_input_frame, text="Depletion Timesteps Designer", padding=10)
        designer_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(designer_frame, text="Configure depletion timesteps with the visual designer:",
                 font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))

        # Current value display
        current_frame = ttk.LabelFrame(designer_frame, text="Current Value", padding=5)
        current_frame.pack(fill=tk.X, pady=(0, 10))

        current_str = str(current_value) if current_value else "[]"
        ttk.Label(current_frame, text=current_str, font=('Courier', 9)).pack(anchor=tk.W)

        # Single button that opens designer
        designer_button = ttk.Button(designer_frame, text="Open Timesteps Designer",
                                   command=lambda: self.open_timesteps_designer(param_name, current_value),
                                   style='Accent.TButton')
        designer_button.pack(pady=(5, 10))

        # Instructions
        features_text = """• Configure target timesteps and number of steps
- Automatic generation of intermediate steps
- Apply as new run or stack to existing run"""

        ttk.Label(designer_frame, text=features_text, font=('TkDefaultFont', 8),
                 justify=tk.LEFT, foreground='gray').pack(anchor=tk.W, padx=(10, 0))

    def open_core_designer(self, param_name, current_value):
        """Open core designer with callback for action"""
        try:
            def handle_designer_result(lattice, action):
                if lattice:
                    description = f"Designed core ({len(lattice)}x{len(lattice[0])})"
                    self.add_single_value_to_run(param_name, lattice, description, stack=(action == 'stack'))

            designer = CoreLatticeDesigner(self.parent, current_value, handle_designer_result)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open core designer: {str(e)}")

    def open_timesteps_designer(self, param_name, current_value):
        """Open timesteps designer with callback for action"""
        def handle_designer_result(timesteps, action):
            description = f"Timesteps: {len(timesteps)} steps"
            self.add_single_value_to_run(param_name, timesteps, description, stack=(action == 'stack'))

        designer = DepletionTimestepsDesigner(self.parent, current_value, handle_designer_result)

    def add_single_value_to_run(self, param_name, value, description=None, stack=False):
        """Add a single parameter value to runs with optional stacking"""
        if stack and self.run_config.simple_runs:
            # Stack onto the last run
            last_run = self.run_config.simple_runs[-1]
            last_run[param_name] = value

            # Update description
            existing_desc = last_run.get('description', '')
            new_param_desc = description or f"{param_name} = {value}"

            # Parse existing description to add new parameter
            if ' + ' in existing_desc:
                last_run['description'] = existing_desc + f" + {new_param_desc}"
            else:
                last_run['description'] = existing_desc + f" + {new_param_desc}"
        else:
            # Create new run
            run_dict = {
                param_name: value,
                "description": description or f"{param_name} = {value}"
            }
            self.run_config.add_simple_run(run_dict)

        self.refresh_runs_display()

    def add_bool_runs(self, param_name):
        """Add boolean parameter runs"""
        selected_values = [value for value, var in self.bool_vars.items() if var.get()]

        for value in selected_values:
            run_dict = {param_name: value, "description": f"{param_name} = {value}"}
            self.run_config.add_simple_run(run_dict)

        self.refresh_runs_display()

    def stack_bool_to_last_run(self, param_name):
        """Stack boolean selection to last run"""
        selected_values = [value for value, var in self.bool_vars.items() if var.get()]

        if not selected_values:
            messagebox.showwarning("No Selection", "Please select at least one value")
            return

        if len(selected_values) > 1:
            messagebox.showwarning("Multiple Values", "Can only stack one value at a time. Adding as new runs.")
            self.add_bool_runs(param_name)
            return

        self.add_single_value_to_run(param_name, selected_values[0], stack=True)

    def add_single_numeric_run(self, param_name, param_type):
        """Add single numeric value run"""
        try:
            if param_type == 'int':
                value = int(self.single_value_var.get())
            else:
                value = float(self.single_value_var.get())

            run_dict = {param_name: value, "description": f"{param_name} = {value}"}
            self.run_config.add_simple_run(run_dict)
            self.refresh_runs_display()

        except ValueError:
            messagebox.showerror("Error", "Invalid numeric value")

    def stack_single_numeric_to_last_run(self, param_name, param_type):
        """Stack single numeric value to last run"""
        try:
            if param_type == 'int':
                value = int(self.single_value_var.get())
            else:
                value = float(self.single_value_var.get())

            self.add_single_value_to_run(param_name, value, stack=True)

        except ValueError:
            messagebox.showerror("Error", "Invalid numeric value")

    def add_numeric_range_runs(self, param_name, param_type):
        """Add numeric range runs"""
        try:
            values = ParameterUtils.parse_numeric_range(
                self.range_start_var.get(),
                self.range_end_var.get(),
                self.range_steps_var.get(),
                param_type
            )

            for value in values:
                run_dict = {param_name: value, "description": f"{param_name} = {value}"}
                self.run_config.add_simple_run(run_dict)

            self.refresh_runs_display()

        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def add_numeric_list_runs(self, param_name, param_type):
        """Add numeric list runs"""
        try:
            values = ParameterUtils.parse_numeric_list(self.list_values_var.get(), param_type)

            for value in values:
                run_dict = {param_name: value, "description": f"{param_name} = {value}"}
                self.run_config.add_simple_run(run_dict)

            self.refresh_runs_display()

        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def add_string_runs(self, param_name):
        """Add string parameter runs"""
        selected_values = [value for value, var in self.string_vars.items() if var.get()]

        for value in selected_values:
            run_dict = {param_name: value, "description": f"{param_name} = {value}"}
            self.run_config.add_simple_run(run_dict)

        self.refresh_runs_display()

    def stack_string_to_last_run(self, param_name):
        """Stack string selection to last run"""
        selected_values = [value for value, var in self.string_vars.items() if var.get()]

        if not selected_values:
            messagebox.showwarning("No Selection", "Please select at least one value")
            return

        if len(selected_values) > 1:
            messagebox.showwarning("Multiple Values", "Can only stack one value at a time. Adding as new runs.")
            self.add_string_runs(param_name)
            return

        self.add_single_value_to_run(param_name, selected_values[0], stack=True)

    def add_custom_string_run(self, param_name):
        """Add custom string value run"""
        value = self.custom_string_var.get().strip()
        if value:
            run_dict = {param_name: value, "description": f"{param_name} = {value}"}
            self.run_config.add_simple_run(run_dict)
            self.refresh_runs_display()

    def stack_custom_string_to_last_run(self, param_name):
        """Stack custom string value to last run"""
        value = self.custom_string_var.get().strip()
        if value:
            self.add_single_value_to_run(param_name, value, stack=True)

    def remove_selected_run(self):
        """Remove selected run from list"""
        selection = self.runs_listbox.curselection()
        if selection:
            index = selection[0]
            self.run_config.remove_simple_run(index)
            self.refresh_runs_display()

    def clear_all_runs(self):
        """Clear all runs"""
        self.run_config.clear_simple_runs()
        self.refresh_runs_display()

    def refresh_runs_display(self):
        """Refresh the runs listbox display"""
        self.runs_listbox.delete(0, tk.END)

        for run in self.run_config.simple_runs:
            self.runs_listbox.insert(tk.END, run.get('description', 'Unnamed run'))

        # Select last item
        if self.run_config.simple_runs:
            self.runs_listbox.select_clear(0, tk.END)
            self.runs_listbox.select_set(tk.END)

    def add_manual_lattice_run(self, param_name):
        """Add manually entered lattice as new run"""
        try:
            lattice_str = self.manual_lattice_var.get().strip()
            if lattice_str:
                lattice = eval(lattice_str)
                if isinstance(lattice, list) and all(isinstance(row, list) for row in lattice):
                    description = f"Manual core ({len(lattice)}x{len(lattice[0])})"
                    self.add_single_value_to_run(param_name, lattice, description, stack=False)
                else:
                    messagebox.showerror("Error", "Invalid lattice format")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse lattice: {str(e)}")

    def stack_manual_lattice_run(self, param_name):
        """Stack manually entered lattice to last run"""
        try:
            lattice_str = self.manual_lattice_var.get().strip()
            if lattice_str:
                lattice = eval(lattice_str)
                if isinstance(lattice, list) and all(isinstance(row, list) for row in lattice):
                    description = f"Manual core ({len(lattice)}x{len(lattice[0])})"
                    self.add_single_value_to_run(param_name, lattice, description, stack=True)
                else:
                    messagebox.showerror("Error", "Invalid lattice format")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse lattice: {str(e)}")
