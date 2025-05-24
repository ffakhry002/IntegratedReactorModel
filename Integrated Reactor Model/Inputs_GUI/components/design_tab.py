# Design tab component

"""
Design Tab Component
Handles core layout and pin layout design
"""
import tkinter as tk
from tkinter import ttk
import copy


class DesignTab:
    def __init__(self, parent, main_gui):
        self.parent = parent
        self.main_gui = main_gui

        # Design state
        self.design_inputs = copy.deepcopy(main_gui.current_inputs)
        self.core_buttons = []
        self.pin_buttons = []

    def setup(self):
        """Setup the design tab with combined layout"""
        # Main container
        design_main = ttk.Frame(self.parent)
        design_main.pack(fill=tk.BOTH, expand=True)

        # Combined layout (core + pin in same tab)
        self.setup_combined_layout_tab(design_main)

    def setup_combined_layout_tab(self, parent):
        """Setup combined core and pin layout tab"""
        # Main horizontal split
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side - Core Layout
        core_frame = ttk.LabelFrame(main_frame, text="Core Layout Designer", padding=10)
        core_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.setup_core_layout_content(core_frame)

        # Right side - Pin Layout (conditionally shown)
        self.pin_layout_frame = ttk.LabelFrame(main_frame, text="Pin Layout Designer", padding=10)
        self.pin_layout_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.setup_pin_layout_content(self.pin_layout_frame)

        # Update visibility based on assembly type
        self.update_pin_layout_visibility()

    def setup_core_layout_content(self, parent):
        """Setup core layout content"""
        # Grid area
        grid_frame = ttk.Frame(parent)
        grid_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Controls below grid
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X)

        # Instructions
        instr_frame = ttk.LabelFrame(control_frame, text="Instructions", padding=5)
        instr_frame.pack(fill=tk.X, pady=(0, 5))

        instructions = """Click cells to cycle through:
1 click: Irradiation Position
2 clicks: Fuel Assembly
3 clicks: Enhanced Fuel Assembly
4 clicks: Coolant (empty)"""

        ttk.Label(instr_frame, text=instructions, justify=tk.LEFT, font=('Arial', 8)).pack()

        # Core size controls in a row
        size_frame = ttk.LabelFrame(control_frame, text="Core Size", padding=5)
        size_frame.pack(fill=tk.X, pady=(0, 5))

        size_row = ttk.Frame(size_frame)
        size_row.pack(fill=tk.X)

        ttk.Label(size_row, text="Rows:").pack(side=tk.LEFT)
        self.rows_var = tk.StringVar(value=str(len(self.design_inputs['core_lattice'])))
        rows_entry = ttk.Entry(size_row, textvariable=self.rows_var, width=8)
        rows_entry.pack(side=tk.LEFT, padx=(5, 10))

        ttk.Label(size_row, text="Columns:").pack(side=tk.LEFT)
        self.cols_var = tk.StringVar(value=str(len(self.design_inputs['core_lattice'][0])))
        cols_entry = ttk.Entry(size_row, textvariable=self.cols_var, width=8)
        cols_entry.pack(side=tk.LEFT, padx=(5, 10))

        ttk.Button(size_row, text="Resize", command=self.resize_core).pack(side=tk.LEFT, padx=(10, 0))

        # Assembly type
        assembly_frame = ttk.LabelFrame(control_frame, text="Assembly Type", padding=5)
        assembly_frame.pack(fill=tk.X, pady=(0, 5))

        self.design_assembly_var = tk.StringVar(value=self.design_inputs["assembly_type"])
        assembly_combo = ttk.Combobox(assembly_frame, textvariable=self.design_assembly_var,
                                    values=["Pin", "Plate"], state="readonly")
        assembly_combo.pack(fill=tk.X)
        assembly_combo.bind('<<ComboboxSelected>>', self.on_design_assembly_change)

        # Apply button
        ttk.Button(control_frame, text="Apply Core Design",
                  command=self.apply_design).pack(fill=tk.X, pady=(5, 0))

        # Setup the grid
        self.setup_core_grid(grid_frame)

    def setup_pin_layout_content(self, parent):
        """Setup pin layout content"""
        # Grid area
        grid_frame = ttk.Frame(parent)
        grid_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Controls below grid
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X)

        # Instructions
        instr_frame = ttk.LabelFrame(control_frame, text="Instructions", padding=5)
        instr_frame.pack(fill=tk.X, pady=(0, 5))

        instructions = """Click pin positions to cycle:
1 click: Fuel Pin (default)
2 clicks: Guide Tube"""

        ttk.Label(instr_frame, text=instructions, justify=tk.LEFT, font=('Arial', 8)).pack()

        # Note about pin size
        note_frame = ttk.LabelFrame(control_frame, text="Note", padding=5)
        note_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(note_frame, text="Change 'Pins per Side' in the Reactor Visualization tab",
                 font=('Arial', 8), foreground='gray').pack()

        # Apply button
        ttk.Button(control_frame, text="Apply Pin Layout",
                  command=self.apply_pin_design).pack(fill=tk.X, pady=(5, 0))

        # Setup the pin grid
        self.setup_pin_grid(grid_frame)

    def setup_core_layout_tab(self, parent):
        """Setup the core layout designer"""
        # Main container
        core_main = ttk.Frame(parent)
        core_main.pack(fill=tk.BOTH, expand=True)

        # Left panel for core grid
        grid_frame = ttk.LabelFrame(core_main, text="Core Layout Grid", padding=10)
        grid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Right panel for controls
        control_frame = ttk.Frame(core_main)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Instructions
        instr_frame = ttk.LabelFrame(control_frame, text="Instructions", padding=10)
        instr_frame.pack(fill=tk.X, pady=(0, 10))

        instructions = """Click on grid cells to cycle through:
1 click: Irradiation Position
2 clicks: Fuel Assembly
3 clicks: Enhanced Fuel Assembly
4 clicks: Coolant (empty)

For irradiation positions, use the
panel below to add labels."""

        ttk.Label(instr_frame, text=instructions, justify=tk.LEFT).pack()

        # Core size controls
        size_frame = ttk.LabelFrame(control_frame, text="Core Size", padding=10)
        size_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(size_frame, text="Rows:").pack(anchor=tk.W)
        self.rows_var = tk.StringVar(value=str(len(self.design_inputs['core_lattice'])))
        rows_entry = ttk.Entry(size_frame, textvariable=self.rows_var, width=10)
        rows_entry.pack(fill=tk.X)

        ttk.Label(size_frame, text="Columns:").pack(anchor=tk.W)
        self.cols_var = tk.StringVar(value=str(len(self.design_inputs['core_lattice'][0])))
        cols_entry = ttk.Entry(size_frame, textvariable=self.cols_var, width=10)
        cols_entry.pack(fill=tk.X)

        ttk.Button(size_frame, text="Resize Core", command=self.resize_core).pack(pady=5)

        # Assembly type toggle
        assembly_frame = ttk.LabelFrame(control_frame, text="Assembly Type", padding=10)
        assembly_frame.pack(fill=tk.X, pady=(0, 10))

        self.design_assembly_var = tk.StringVar(value=self.design_inputs["assembly_type"])
        assembly_combo = ttk.Combobox(assembly_frame, textvariable=self.design_assembly_var,
                                    values=["Pin", "Plate"], state="readonly")
        assembly_combo.pack(fill=tk.X)
        assembly_combo.bind('<<ComboboxSelected>>', self.on_design_assembly_change)

        # Apply button
        ttk.Button(control_frame, text="Apply to Visualization",
                  command=self.apply_design).pack(fill=tk.X, pady=5)

        # Setup the grid
        self.setup_core_grid(grid_frame)

    def setup_pin_layout_tab(self, parent):
        """Setup the pin layout designer tab"""
        # Main container
        pin_main = ttk.Frame(parent)
        pin_main.pack(fill=tk.BOTH, expand=True)

        # Left panel for pin grid
        pin_grid_frame = ttk.LabelFrame(pin_main, text="Pin Layout Grid", padding=10)
        pin_grid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Right panel for controls
        pin_control_frame = ttk.Frame(pin_main)
        pin_control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Instructions
        pin_instr_frame = ttk.LabelFrame(pin_control_frame, text="Instructions", padding=10)
        pin_instr_frame.pack(fill=tk.X, pady=(0, 10))

        instructions = """Click on pin positions to cycle:
1 click: Fuel Pin (default)
2 clicks: Guide Tube
3 clicks: Fuel Pin"""

        ttk.Label(pin_instr_frame, text=instructions, justify=tk.LEFT).pack()

        # Pin array size controls
        size_frame = ttk.LabelFrame(pin_control_frame, text="Assembly Size", padding=10)
        size_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(size_frame, text="Pins per Side:").pack(anchor=tk.W)
        self.pin_size_var = tk.StringVar(value=str(int(self.design_inputs['n_side_pins'])))
        pin_size_entry = ttk.Entry(size_frame, textvariable=self.pin_size_var, width=10)
        pin_size_entry.pack(fill=tk.X)

        ttk.Button(size_frame, text="Resize Pin Array", command=self.resize_pin_array).pack(pady=5)

        # Apply button
        ttk.Button(pin_control_frame, text="Apply to Visualization",
                  command=self.apply_pin_design).pack(fill=tk.X, pady=5)

        # Setup the pin grid
        self.setup_pin_grid(pin_grid_frame)

    def setup_core_grid(self, parent):
        """Create the core layout grid"""
        # Clear existing buttons
        for widget in parent.winfo_children():
            widget.destroy()
        self.core_buttons = []

        lattice = self.design_inputs['core_lattice']
        n_rows, n_cols = len(lattice), len(lattice[0])

        # Create grid
        grid_container = ttk.Frame(parent)
        grid_container.pack(expand=True)

        for i in range(n_rows):
            row_buttons = []
            for j in range(n_cols):
                cell_type = lattice[i][j]

                # Create button
                btn = tk.Button(grid_container, text=cell_type, width=5, height=2)
                btn.grid(row=i, column=j, padx=1, pady=1)

                # Set initial color
                self.update_cell_color(btn, cell_type)

                # Bind click event
                btn.bind('<Button-1>', lambda e, r=i, c=j: self.on_core_cell_click(r, c))

                row_buttons.append(btn)
            self.core_buttons.append(row_buttons)

    def setup_pin_grid(self, parent):
        """Create the pin layout grid"""
        # Clear existing buttons
        for widget in parent.winfo_children():
            widget.destroy()
        self.pin_buttons = []

        # Use main GUI inputs to get current pin count
        n_pins = int(self.main_gui.current_inputs['n_side_pins'])
        guide_tube_positions = self.main_gui.current_inputs.get('guide_tube_positions', [])

        # Update design inputs to match
        self.design_inputs['n_side_pins'] = n_pins
        self.design_inputs['guide_tube_positions'] = guide_tube_positions[:]

        # Create grid
        grid_container = ttk.Frame(parent)
        grid_container.pack(expand=True)

        for i in range(n_pins):
            row_buttons = []
            for j in range(n_pins):
                is_guide_tube = (i, j) in guide_tube_positions

                # Create button
                btn = tk.Button(grid_container,
                              text='G' if is_guide_tube else 'F',
                              width=3, height=1)
                btn.grid(row=i, column=j, padx=1, pady=1)

                # Set initial color
                if is_guide_tube:
                    btn.configure(bg='white', fg='black')  # White background for guide tubes
                else:
                    btn.configure(bg='#32CD32', fg='black')

                # Bind click event
                btn.bind('<Button-1>', lambda e, r=i, c=j: self.on_pin_cell_click(r, c))

                row_buttons.append(btn)
            self.pin_buttons.append(row_buttons)

    def on_core_cell_click(self, row, col):
        """Handle core cell click"""
        current = self.design_inputs['core_lattice'][row][col]

        # Cycle through cell types
        if current == 'C':
            new_type = 'I_1'  # Start with I_1
        elif current.startswith('I_'):
            new_type = 'F'
        elif current == 'F':
            new_type = 'E'
        elif current == 'E':
            new_type = 'C'
        else:
            new_type = 'C'

        # Handle irradiation position naming
        if new_type == 'I_1':
            # Find next available irradiation number
            irrad_nums = []
            for row_data in self.design_inputs['core_lattice']:
                for cell in row_data:
                    if cell.startswith('I_'):
                        try:
                            num = int(cell.split('_')[1])
                            irrad_nums.append(num)
                        except:
                            pass

            next_num = 1
            if irrad_nums:
                next_num = max(irrad_nums) + 1
            new_type = f'I_{next_num}'

        # Update the lattice
        self.design_inputs['core_lattice'][row][col] = new_type

        # Update button
        btn = self.core_buttons[row][col]
        btn.config(text=new_type)
        self.update_cell_color(btn, new_type)

    def on_pin_cell_click(self, row, col):
        """Handle pin cell click"""
        guide_tubes = self.design_inputs.get('guide_tube_positions', [])

        if (row, col) in guide_tubes:
            # Remove guide tube
            guide_tubes.remove((row, col))
            self.pin_buttons[row][col].configure(text='F', bg='#32CD32', fg='black')
        else:
            # Add guide tube
            guide_tubes.append((row, col))
            self.pin_buttons[row][col].configure(text='G', bg='white', fg='black')  # White background for guide tubes

        self.design_inputs['guide_tube_positions'] = guide_tubes

        # Update the main GUI inputs to sync
        self.main_gui.current_inputs['guide_tube_positions'] = guide_tubes[:]

    def update_cell_color(self, button, cell_type):
        """Update button color based on cell type"""
        colors = {
            'C': ('#40E0D0', 'black'),     # Coolant - Turquoise
            'F': ('#32CD32', 'black'),     # Fuel - Lime Green
            'E': ('#8B0000', 'black'),     # Enhanced fuel - Dark Red with black text
        }

        if cell_type.startswith('I_'):
            bg, fg = ('#FFD700', 'black')  # Irradiation - Gold
        else:
            bg, fg = colors.get(cell_type, ('#808080', 'white'))

        button.configure(bg=bg, fg=fg)

    def resize_core(self):
        """Resize the core lattice"""
        try:
            new_rows = int(self.rows_var.get())
            new_cols = int(self.cols_var.get())

            if new_rows < 1 or new_cols < 1:
                raise ValueError("Size must be positive")

            # Get current lattice
            current = self.design_inputs['core_lattice']
            old_rows = len(current)
            old_cols = len(current[0]) if old_rows > 0 else 0

            # Create new lattice
            new_lattice = []
            for i in range(new_rows):
                row = []
                for j in range(new_cols):
                    if i < old_rows and j < old_cols:
                        row.append(current[i][j])
                    else:
                        row.append('C')  # Default to coolant
                new_lattice.append(row)

            self.design_inputs['core_lattice'] = new_lattice
            self.setup_core_grid(self.core_buttons[0][0].master.master)

        except ValueError:
            from tkinter import messagebox
            messagebox.showerror("Invalid Size", "Please enter valid positive integers")

    def resize_pin_array(self):
        """Resize the pin array"""
        try:
            new_size = int(self.pin_size_var.get())

            if new_size < 1:
                raise ValueError("Size must be positive")

            self.design_inputs['n_side_pins'] = new_size

            # Clear guide tubes that are out of bounds
            guide_tubes = self.design_inputs.get('guide_tube_positions', [])
            valid_tubes = [(r, c) for r, c in guide_tubes if r < new_size and c < new_size]
            self.design_inputs['guide_tube_positions'] = valid_tubes

            self.setup_pin_grid(self.pin_buttons[0][0].master.master)

        except ValueError:
            from tkinter import messagebox
            messagebox.showerror("Invalid Size", "Please enter a valid positive integer")

    def apply_design(self):
        """Apply design to main visualization"""
        # Update main GUI inputs
        self.main_gui.current_inputs['core_lattice'] = copy.deepcopy(
            self.design_inputs['core_lattice']
        )
        self.main_gui.current_inputs['assembly_type'] = self.design_assembly_var.get()

        # Update and schedule visualization
        self.main_gui.schedule_update()

        from tkinter import messagebox
        messagebox.showinfo("Design Applied", "Core design has been applied to visualization")

    def apply_pin_design(self):
        """Apply pin design to main visualization"""
        # Update main GUI inputs
        self.main_gui.current_inputs['n_side_pins'] = self.design_inputs['n_side_pins']
        self.main_gui.current_inputs['guide_tube_positions'] = copy.deepcopy(
            self.design_inputs.get('guide_tube_positions', [])
        )

        # Update and schedule visualization
        self.main_gui.schedule_update()

        from tkinter import messagebox
        messagebox.showinfo("Design Applied", "Pin layout has been applied to visualization")

    def on_design_assembly_change(self, event=None):
        """Handle assembly type change in designer"""
        self.design_inputs["assembly_type"] = self.design_assembly_var.get()
        self.update_pin_layout_visibility()

    def update_pin_layout_visibility(self):
        """Show/hide pin layout frame based on assembly type"""
        if self.design_inputs["assembly_type"] == "Pin":
            self.pin_layout_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        else:
            self.pin_layout_frame.pack_forget()
