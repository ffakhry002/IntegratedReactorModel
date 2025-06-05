"""
Complex Tab Component
Handles multi-parameter loops configuration
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


class ComplexTab:
    """Multi-parameter loops tab"""

    def __init__(self, parent, parameter_model, run_config):
        self.parent = parent
        self.parameter_model = parameter_model
        self.run_config = run_config

        # UI elements
        self.loop_sets_notebook = None
        self.current_loop_set_index = 0

    def setup(self):
        """Setup the complex multi-parameter loops tab"""
        # Main container
        container = ttk.Frame(self.parent)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Loop sets management
        mgmt_frame = ttk.Frame(container)
        mgmt_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(mgmt_frame, text="Add New Loop Set",
                  command=self.add_loop_set).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(mgmt_frame, text="Remove Current Set",
                  command=self.remove_current_loop_set).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(mgmt_frame, text="Clear All Sets",
                  command=self.clear_all_loop_sets).pack(side=tk.LEFT)

        # Notebook for loop sets
        self.loop_sets_notebook = ttk.Notebook(container)
        self.loop_sets_notebook.pack(fill=tk.BOTH, expand=True)

        # Add initial loop set
        self.add_loop_set()

    def add_loop_set(self):
        """Add a new loop set tab"""
        set_index = len(self.run_config.loop_sets)
        set_name = f"Loop Set {set_index + 1}"

        # Create frame for this loop set
        set_frame = ttk.Frame(self.loop_sets_notebook)
        self.loop_sets_notebook.add(set_frame, text=set_name)

        # Create loop set configuration
        loop_set = {
            'name': set_name,
            'loops': [],
            'ui_elements': {}
        }

        # Setup the loop set UI
        self.setup_loop_set_ui(set_frame, loop_set)

        # Add to run configuration
        self.run_config.add_loop_set(loop_set)

        # Switch to new tab
        self.loop_sets_notebook.select(set_frame)

    def setup_loop_set_ui(self, parent_frame, loop_set):
        """Setup UI for a single loop set"""
        # Main container
        main_container = ttk.Frame(parent_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Loop configuration area
        config_frame = ttk.LabelFrame(main_container, text="Parameter Loop Configuration", padding=10)
        config_frame.pack(fill=tk.BOTH, expand=True)

        # Create scrollable container
        canvas = tk.Canvas(config_frame, height=400)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Store UI elements
        loop_set['ui_elements']['scrollable_frame'] = scrollable_frame
        loop_set['ui_elements']['preview_label'] = None

        # Add/remove loop buttons
        button_frame = ttk.Frame(config_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(button_frame, text="Add Parameter Loop",
                  command=lambda: self.add_parameter_loop(loop_set)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Clear All Loops",
                  command=lambda: self.clear_loops_in_set(loop_set)).pack(side=tk.LEFT)

        # Preview area
        preview_frame = ttk.LabelFrame(main_container, text="Loop Set Preview", padding=10)
        preview_frame.pack(fill=tk.X, pady=(10, 0))

        preview_label = ttk.Label(preview_frame, text="No loops configured")
        preview_label.pack()
        loop_set['ui_elements']['preview_label'] = preview_label

    def add_parameter_loop(self, loop_set):
        """Add a new parameter loop to the loop set"""
        scrollable_frame = loop_set['ui_elements']['scrollable_frame']
        loop_index = len(loop_set['loops'])

        loop_frame = ttk.LabelFrame(scrollable_frame, text=f"Loop {loop_index + 1}", padding=10)
        loop_frame.pack(fill=tk.X, pady=(0, 10))

        # Parameter selection with categories
        param_frame = ttk.Frame(loop_frame)
        param_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(param_frame, text="Parameter:").pack(side=tk.LEFT)

        # Create a frame for the parameter selection
        param_selection_frame = ttk.Frame(param_frame)
        param_selection_frame.pack(side=tk.LEFT, padx=(5, 10))

        # Button to open parameter selection dialog
        param_var = tk.StringVar()
        param_display_var = tk.StringVar(value="Click to select parameter...")

        def open_parameter_selector():
            self.open_parameter_selection_dialog(param_var, param_display_var, on_param_change)

        param_button = ttk.Button(param_selection_frame, textvariable=param_display_var,
                                 command=open_parameter_selector, width=15)
        param_button.pack()

        # Values input
        ttk.Label(param_frame, text="Values:").pack(side=tk.LEFT)
        values_var = tk.StringVar()
        values_entry = ttk.Entry(param_frame, textvariable=values_var, width=30)
        values_entry.pack(side=tk.LEFT, padx=(5, 10))

        # Container for dynamic buttons
        button_container = ttk.Frame(param_frame)
        button_container.pack(side=tk.LEFT)

        def on_param_change(*args):
            param_name = param_var.get()
            if param_name:
                param_display_var.set(param_name)

            # Clear existing special buttons
            for widget in button_container.winfo_children():
                widget.destroy()

            # Add special buttons for certain parameters
            if param_name == 'core_lattice':
                designer_btn = ttk.Button(button_container, text="Design",
                                        command=lambda: self.open_multi_core_designer(values_var),
                                        style='Accent.TButton')
                designer_btn.pack(side=tk.LEFT, padx=(0, 5))

                preview_btn = ttk.Button(button_container, text="Preview",
                                       command=lambda: self.preview_core_lattices(values_var.get()))
                preview_btn.pack(side=tk.LEFT, padx=(0, 5))

            elif param_name == 'depletion_timesteps':
                designer_btn = ttk.Button(button_container, text="Write",
                                        command=lambda: self.open_multi_timesteps_designer(values_var))
                designer_btn.pack(side=tk.LEFT, padx=(0, 5))

        param_var.trace('w', on_param_change)

        # Remove button
        ttk.Button(param_frame, text="Remove",
                  command=lambda: self.remove_parameter_loop(loop_set, loop_frame, loop_data)).pack(side=tk.LEFT, padx=(10, 0))

        # Store loop configuration
        loop_data = {
            'frame': loop_frame,
            'param_var': param_var,
            'values_var': values_var
        }
        loop_set['loops'].append(loop_data)

        # Update preview when values change
        param_var.trace('w', lambda *args: self.update_loop_preview(loop_set))
        values_var.trace('w', lambda *args: self.update_loop_preview(loop_set))

        # Force canvas to update scroll region
        scrollable_frame.update_idletasks()

    def open_parameter_selection_dialog(self, param_var, param_display_var, on_param_change):
        """Open a dialog for parameter selection with categories"""
        dialog = tk.Toplevel(self.parent)
        dialog.title("Select Parameter")
        dialog.geometry("600x500")
        dialog.transient(self.parent)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (dialog.winfo_screenheight() // 2) - (500 // 2)
        dialog.geometry(f"600x500+{x}+{y}")

        # Main frame
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Select Parameter",
                               font=('TkDefaultFont', 12, 'bold'))
        title_label.pack(pady=(0, 10))

        # Search frame
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # Parameter tree
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        columns = ('Parameter', 'Current Value', 'Type')
        param_tree = ttk.Treeview(tree_frame, columns=columns, show='tree headings', height=15)

        # Define headings
        param_tree.heading('#0', text='Category')
        param_tree.heading('Parameter', text='Parameter')
        param_tree.heading('Current Value', text='Current Value')
        param_tree.heading('Type', text='Type')

        # Configure column widths
        param_tree.column('#0', width=150)
        param_tree.column('Parameter', width=200)
        param_tree.column('Current Value', width=100)
        param_tree.column('Type', width=80)

        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=param_tree.yview)
        param_tree.configure(yscrollcommand=tree_scroll.set)

        param_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        def populate_tree(search_text=""):
            """Populate the parameter tree with categories"""
            # Clear existing items
            for item in param_tree.get_children():
                param_tree.delete(item)

            # Group parameters by category
            categories = {}
            for param_name, param_info in self.parameter_model.available_params.items():
                if not search_text or (search_text.lower() in param_name.lower() or
                                     search_text.lower() in param_info['description'].lower() or
                                     search_text.lower() in param_info['category'].lower()):
                    category = param_info['category']
                    if category not in categories:
                        categories[category] = []
                    categories[category].append((param_name, param_info))

            # Add category headers and parameters
            for category, params in sorted(categories.items()):
                if params:  # Only show categories that have matching parameters
                    # Create category header
                    category_id = param_tree.insert('', 'end', text=category, values=('', '', ''))

                    # Add parameters under category
                    for param_name, param_info in sorted(params):
                        param_tree.insert(category_id, 'end', text='', values=(
                            param_name,
                            str(param_info['current_value'])[:20] + ('...' if len(str(param_info['current_value'])) > 20 else ''),
                            param_info['type']
                        ))

            # Expand all categories
            for item in param_tree.get_children():
                param_tree.item(item, open=True)

        # Initial population
        populate_tree()

        # Search functionality
        def on_search(*args):
            populate_tree(search_var.get())

        search_var.trace('w', on_search)

        # Selection handling
        selected_param = tk.StringVar()

        def on_tree_select(event):
            selection = param_tree.selection()
            if not selection:
                return

            item = param_tree.item(selection[0])
            values = item['values']

            # Check if this is a parameter (has values) or just a category header
            if values and values[0]:  # Parameter
                selected_param.set(values[0])

        param_tree.bind('<<TreeviewSelect>>', on_tree_select)

        # Double-click to select
        def on_double_click(event):
            if selected_param.get():
                param_var.set(selected_param.get())
                param_display_var.set(selected_param.get())
                on_param_change()
                dialog.destroy()

        param_tree.bind('<Double-1>', on_double_click)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        def on_select():
            if selected_param.get():
                param_var.set(selected_param.get())
                param_display_var.set(selected_param.get())
                on_param_change()
                dialog.destroy()
            else:
                messagebox.showwarning("No Selection", "Please select a parameter")

        def on_cancel():
            dialog.destroy()

        ttk.Button(button_frame, text="Select", command=on_select).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT)

        # Focus on search entry
        search_entry.focus_set()

    def open_multi_core_designer(self, values_var):
        """Open core designer for adding multiple lattices"""
        # Enhanced designer that adds to values field
        def handle_designer_result(lattice):
            # Get current values
            current_values = values_var.get().strip()
            lattice_str = str(lattice)

            if current_values:
                # Parse and add to existing
                try:
                    # Check if we already have multiple lattices
                    if current_values.startswith('[[['):
                        # Multiple lattices - parse and add
                        existing = eval(current_values)
                        existing.append(lattice)
                        values_var.set(str(existing))
                    elif current_values.startswith('[['):
                        # Single lattice - convert to multiple
                        existing = eval(current_values)
                        values_var.set(str([existing, lattice]))
                    else:
                        # Try to parse as single lattice
                        values_var.set(current_values + ', ' + lattice_str)
                except:
                    # If parsing fails, just append
                    values_var.set(current_values + ', ' + lattice_str)
            else:
                values_var.set(lattice_str)

        designer = CoreLatticeDesigner(self.parent, None, handle_designer_result)

    def open_multi_timesteps_designer(self, values_var):
        """Open timesteps designer for adding multiple timestep configurations"""
        def handle_designer_result(timesteps, action):
            # Get current values
            current_values = values_var.get().strip()
            timesteps_str = str(timesteps)

            if current_values:
                # Add to existing
                values_var.set(current_values + ', ' + timesteps_str)
            else:
                values_var.set(timesteps_str)

        designer = DepletionTimestepsDesigner(self.parent, None, handle_designer_result)

    def preview_core_lattices(self, values_str):
        """Preview core lattices in a popup"""
        if not values_str.strip():
            messagebox.showinfo("No Lattices", "No core lattices to preview")
            return

        # Create preview window
        preview_window = tk.Toplevel(self.parent)
        preview_window.title("Core Lattice Preview")
        preview_window.geometry("800x600")

        # Parse lattices safely
        lattices = []
        try:
            # Try different parsing approaches
            values_str = values_str.strip()

            # First try direct eval for properly formatted strings
            if values_str.startswith('[[[') or values_str.startswith('[['):
                lattices = eval(values_str)
                if not isinstance(lattices[0][0], list):  # Single lattice
                    lattices = [lattices]
            else:
                # Try to parse as comma-separated lattices
                # Split by '], [' pattern but be careful about nested brackets
                parts = []
                current_part = ""
                bracket_depth = 0

                for char in values_str:
                    if char == '[':
                        bracket_depth += 1
                    elif char == ']':
                        bracket_depth -= 1

                    current_part += char

                    # Check if we've completed a lattice
                    if bracket_depth == 0 and current_part.strip().endswith(']'):
                        if current_part.strip():
                            parts.append(current_part.strip())
                        current_part = ""

                # Parse each part
                for part in parts:
                    part = part.strip()
                    if part.startswith(','):
                        part = part[1:].strip()
                    try:
                        lattice = eval(part)
                        if isinstance(lattice, list) and all(isinstance(row, list) for row in lattice):
                            lattices.append(lattice)
                    except:
                        pass

        except Exception as e:
            messagebox.showerror("Parse Error", f"Error parsing lattices: {e}")
            preview_window.destroy()
            return

        if not lattices:
            messagebox.showinfo("No Valid Lattices", "No valid lattices found to preview")
            preview_window.destroy()
            return

        # Create scrollable frame
        canvas = tk.Canvas(preview_window)
        scrollbar = ttk.Scrollbar(preview_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Title
        title_label = ttk.Label(scrollable_frame, text=f"Core Lattice Preview ({len(lattices)} lattices)",
                               font=('TkDefaultFont', 14, 'bold'))
        title_label.pack(pady=10)

        # Create preview for each lattice
        for i, lattice in enumerate(lattices):
            frame = ttk.LabelFrame(scrollable_frame, text=f"Lattice {i+1}", padding=10)
            frame.pack(fill=tk.X, padx=10, pady=5)

            # Create mini preview
            self.create_lattice_preview(frame, lattice)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_lattice_preview(self, parent, lattice):
        """Create a mini preview of a lattice"""
        rows = len(lattice)
        cols = len(lattice[0]) if rows > 0 else 0

        # Info label
        info_label = ttk.Label(parent, text=f"Size: {rows}x{cols}")
        info_label.pack()

        # Grid frame
        grid_frame = ttk.Frame(parent)
        grid_frame.pack(pady=5)

        # Create mini grid
        for i in range(rows):
            for j in range(cols):
                cell = lattice[i][j]
                # Determine color
                if cell == 'C':
                    bg = '#87CEEB'
                elif cell == 'F':
                    bg = '#90EE90'
                elif cell == 'E':
                    bg = '#FFB6C1'
                elif cell.startswith('I_'):
                    bg = '#F0F0F0'
                else:
                    bg = '#FFFFFF'

                label = tk.Label(grid_frame, text=cell, width=3, height=1,
                               bg=bg, relief=tk.RAISED, font=('TkDefaultFont', 8))
                label.grid(row=i, column=j, padx=1, pady=1)

    def remove_parameter_loop(self, loop_set, loop_frame, loop_data):
        """Remove a parameter loop from the set"""
        # Remove from loops list
        loop_set['loops'].remove(loop_data)

        # Destroy frame
        loop_frame.destroy()

        # Update preview
        self.update_loop_preview(loop_set)

    def clear_loops_in_set(self, loop_set):
        """Clear all loops in a set"""
        # Destroy all loop frames
        for loop in loop_set['loops']:
            loop['frame'].destroy()

        # Clear loops list
        loop_set['loops'].clear()

        # Update preview
        self.update_loop_preview(loop_set)

    def update_loop_preview(self, loop_set):
        """Update the loop preview for a set"""
        preview_label = loop_set['ui_elements']['preview_label']

        if not loop_set['loops']:
            preview_label.config(text="No loops configured")
            return

        # Count valid loops and calculate total runs
        valid_loops = []
        total_runs = 1

        for loop in loop_set['loops']:
            param = loop['param_var'].get()
            values_str = loop['values_var'].get()

            if param and values_str:
                try:
                    # Special handling for different parameter types
                    param_info = self.parameter_model.available_params.get(param, {})
                    param_type = param_info.get('type', 'str')

                    if param_type == 'core_lattice':
                        # Count lattices in the string
                        count = self._count_lattices(values_str)
                    elif param_type == 'depletion_timesteps':
                        # Count timestep lists
                        count = values_str.count('[')
                    else:
                        # Normal comma-separated values
                        values = [v.strip() for v in values_str.split(',') if v.strip()]
                        count = len(values)

                    if count > 0:
                        valid_loops.append((param, count))
                        total_runs *= count
                except:
                    pass

        if valid_loops:
            loop_desc = " × ".join([f"{param} ({count} values)" for param, count in valid_loops])
            preview_text = f"{loop_desc} = {total_runs} total runs"
        else:
            preview_text = "No valid loops configured"

        preview_label.config(text=preview_text)

    def _count_lattices(self, values_str):
        """Count the number of lattices in a string"""
        try:
            values_str = values_str.strip()
            if values_str.startswith('[[['):
                # Multiple lattices
                lattices = eval(values_str)
                return len(lattices)
            elif values_str.startswith('[['):
                # Single lattice
                return 1
            else:
                # Count by balanced brackets
                count = 0
                bracket_depth = 0
                for char in values_str:
                    if char == '[':
                        bracket_depth += 1
                    elif char == ']':
                        bracket_depth -= 1
                        if bracket_depth == 0:
                            count += 1
                return max(count, 0)
        except:
            return 0

    def remove_current_loop_set(self):
        """Remove the currently selected loop set"""
        if len(self.run_config.loop_sets) <= 1:
            messagebox.showwarning("Cannot Remove", "Must have at least one loop set")
            return

        # Get current tab index
        current_index = self.loop_sets_notebook.index("current")

        # Remove from notebook
        self.loop_sets_notebook.forget(current_index)

        # Remove from run config
        self.run_config.remove_loop_set(current_index)

        # Update tab names
        for i in range(len(self.run_config.loop_sets)):
            tab_id = self.loop_sets_notebook.tabs()[i]
            self.loop_sets_notebook.tab(tab_id, text=f"Loop Set {i + 1}")
            self.run_config.loop_sets[i]['name'] = f"Loop Set {i + 1}"

    def clear_all_loop_sets(self):
        """Clear all loop sets"""
        # Remove all tabs except the first
        while len(self.loop_sets_notebook.tabs()) > 1:
            self.loop_sets_notebook.forget(1)

        # Clear run config
        self.run_config.clear_loop_sets()

        # Reset the first tab
        self.loop_sets_notebook.forget(0)
        self.add_loop_set()
