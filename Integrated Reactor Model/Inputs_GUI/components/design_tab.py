"""
Design Tab Component
Handles core layout and pin layout design with unified visual tools
"""
import tkinter as tk
from tkinter import ttk, messagebox
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class DesignTab:
    """Unified design tab for core and pin layouts"""

    def __init__(self, parent, main_gui):
        self.parent = parent
        self.main_gui = main_gui

        # Design state - copy from main GUI
        self.design_inputs = copy.deepcopy(main_gui.current_inputs)

        # Core lattice state
        self.core_lattice = copy.deepcopy(self.design_inputs['core_lattice'])

        # Pin layout state
        self.n_side_pins = int(self.design_inputs.get('n_side_pins', 17))
        self.guide_tube_positions = copy.deepcopy(self.design_inputs.get('guide_tube_positions', []))

    def setup(self):
        """Setup the unified design interface"""
        # Main container
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title and description
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(header_frame, text="Reactor Core Designer",
                 font=('TkDefaultFont', 14, 'bold')).pack(anchor=tk.W)
        ttk.Label(header_frame,
                 text="Design core layouts and pin configurations with real-time preview",
                 font=('TkDefaultFont', 10)).pack(anchor=tk.W)

        # Assembly type selector at top
        assembly_frame = ttk.LabelFrame(main_frame, text="Assembly Type", padding=10)
        assembly_frame.pack(fill=tk.X, pady=(0, 10))

        assembly_row = ttk.Frame(assembly_frame)
        assembly_row.pack()

        ttk.Label(assembly_row, text="Type:").pack(side=tk.LEFT, padx=(0, 10))
        self.assembly_type_var = tk.StringVar(value=self.design_inputs["assembly_type"])
        assembly_combo = ttk.Combobox(assembly_row, textvariable=self.assembly_type_var,
                                    values=["Pin", "Plate"], state="readonly", width=15)
        assembly_combo.pack(side=tk.LEFT)
        assembly_combo.bind('<<ComboboxSelected>>', self.on_assembly_type_change)

        ttk.Label(assembly_row, text="(Pin assemblies allow guide tube configuration)",
                 font=('TkDefaultFont', 9), foreground='gray').pack(side=tk.LEFT, padx=(20, 0))

        # Main content area with two panes
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left pane - Core Layout Designer
        self.setup_core_designer(content_frame)

        # Right pane - Pin Layout Designer (conditional)
        self.pin_frame_container = ttk.Frame(content_frame)
        self.pin_frame_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.setup_pin_designer(self.pin_frame_container)

        # Bottom action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(action_frame, text="Apply All Changes to Reactor",
                  command=self.apply_all_changes,
                  style='Accent.TButton').pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(action_frame, text="Reset to Current",
                  command=self.reset_to_current).pack(side=tk.LEFT)

        self.status_label = ttk.Label(action_frame, text="", foreground='green')
        self.status_label.pack(side=tk.RIGHT)

        # Update visibility based on assembly type
        self.update_pin_designer_visibility()

    def setup_core_designer(self, parent):
        """Setup the core layout designer panel"""
        # Core designer frame
        core_frame = ttk.LabelFrame(parent, text="Core Layout Designer", padding=10)
        core_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Controls
        controls_frame = ttk.Frame(core_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        # Size controls
        size_frame = ttk.LabelFrame(controls_frame, text="Grid Size", padding=5)
        size_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        size_grid = ttk.Frame(size_frame)
        size_grid.pack()

        ttk.Label(size_grid, text="Rows:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.core_rows_var = tk.IntVar(value=len(self.core_lattice))
        rows_spin = ttk.Spinbox(size_grid, from_=2, to=20, textvariable=self.core_rows_var,
                               width=8, command=self.update_core_size)
        rows_spin.grid(row=0, column=1)

        ttk.Label(size_grid, text="Columns:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.core_cols_var = tk.IntVar(value=len(self.core_lattice[0]) if self.core_lattice else 4)
        cols_spin = ttk.Spinbox(size_grid, from_=2, to=20, textvariable=self.core_cols_var,
                               width=8, command=self.update_core_size)
        cols_spin.grid(row=1, column=1)

        # Preset buttons
        preset_frame = ttk.LabelFrame(controls_frame, text="Presets", padding=5)
        preset_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        preset_grid = ttk.Frame(preset_frame)
        preset_grid.pack()

        presets = [
            ("4x4 Simple", self.load_4x4_preset),
            ("6x6 Standard", self.load_6x6_preset),
            ("7x7 Research", self.load_7x7_preset),
            ("8x8 Complex", self.load_8x8_preset)
        ]

        for i, (name, command) in enumerate(presets):
            row = i // 2
            col = i % 2
            ttk.Button(preset_grid, text=name, command=command,
                      width=15).grid(row=row, column=col, padx=2, pady=2)

        # Quick actions
        action_frame = ttk.LabelFrame(controls_frame, text="Quick Fill", padding=5)
        action_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))

        ttk.Button(action_frame, text="All Fuel",
                  command=lambda: self.fill_all_core('F')).pack(fill=tk.X, pady=1)
        ttk.Button(action_frame, text="All Control",
                  command=lambda: self.fill_all_core('C')).pack(fill=tk.X, pady=1)

        # Core grid container with scrolling
        grid_container = ttk.LabelFrame(core_frame, text="Core Grid (Click to Edit)", padding=10)
        grid_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Create scrollable area
        canvas = tk.Canvas(grid_container, bg='white')
        v_scrollbar = ttk.Scrollbar(grid_container, orient="vertical", command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(grid_container, orient="horizontal", command=canvas.xview)

        self.core_grid_frame = ttk.Frame(canvas)

        self.core_grid_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.core_grid_frame, anchor="nw")
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        grid_container.grid_rowconfigure(0, weight=1)
        grid_container.grid_columnconfigure(0, weight=1)

        # Create initial grid
        self.create_core_grid()

    def setup_pin_designer(self, parent):
        """Setup the pin layout designer panel"""
        # Pin designer frame
        self.pin_designer_frame = ttk.LabelFrame(parent, text="Pin Layout Designer", padding=10)
        self.pin_designer_frame.pack(fill=tk.BOTH, expand=True)

        # Info frame
        info_frame = ttk.LabelFrame(self.pin_designer_frame, text="Information", padding=5)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        info_text = f"""Current assembly size: {self.n_side_pins}x{self.n_side_pins} pins
(Change pin count in Reactor Visualization tab)

Click pins to toggle between:
- Fuel Pin (green)
- Guide Tube (white)"""

        self.info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        self.info_label.pack()

        # Pin grid container
        grid_container = ttk.LabelFrame(self.pin_designer_frame, text="Pin Grid (Click to Toggle)", padding=10)
        grid_container.pack(fill=tk.BOTH, expand=True)

        # Create scrollable area
        canvas = tk.Canvas(grid_container, bg='white')
        scrollbar = ttk.Scrollbar(grid_container, orient="vertical", command=canvas.yview)

        self.pin_grid_frame = ttk.Frame(canvas)

        self.pin_grid_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.pin_grid_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Create initial grid
        self.create_pin_grid()

    def create_core_grid(self):
        """Create the interactive core grid"""
        # Clear existing widgets
        for widget in self.core_grid_frame.winfo_children():
            widget.destroy()

        rows = len(self.core_lattice)
        cols = len(self.core_lattice[0]) if rows > 0 else 0

        for i in range(rows):
            for j in range(cols):
                cell_value = self.core_lattice[i][j]

                # Create cell button
                cell = tk.Label(self.core_grid_frame, text=cell_value,
                              width=5, height=2,
                              relief=tk.RAISED, bd=2,
                              bg=self.get_core_cell_color(cell_value),
                              font=('TkDefaultFont', 10, 'bold'),
                              cursor="hand2")
                cell.grid(row=i, column=j, padx=2, pady=2)
                cell.bind('<Button-1>', lambda e, r=i, c=j: self.toggle_core_cell(r, c))

    def create_pin_grid(self):
        """Create the interactive pin grid"""
        # Clear existing widgets
        for widget in self.pin_grid_frame.winfo_children():
            widget.destroy()

        # Update pin count from main GUI
        self.n_side_pins = int(self.main_gui.current_inputs.get('n_side_pins', 17))

        # Update info label
        if hasattr(self, 'info_label'):
            info_text = f"""Current assembly size: {self.n_side_pins}x{self.n_side_pins} pins
(Change pin count in Reactor Visualization tab)

Click pins to toggle between:
- Fuel Pin (green)
- Guide Tube (white)"""
            self.info_label.config(text=info_text)

        for i in range(self.n_side_pins):
            for j in range(self.n_side_pins):
                is_guide = (i, j) in self.guide_tube_positions

                # Create pin button
                pin = tk.Label(self.pin_grid_frame,
                             text='G' if is_guide else 'F',
                             width=3, height=1,
                             relief=tk.RAISED, bd=2,
                             bg='white' if is_guide else '#90EE90',
                             font=('TkDefaultFont', 8),
                             cursor="hand2")
                pin.grid(row=i, column=j, padx=1, pady=1)
                pin.bind('<Button-1>', lambda e, r=i, c=j: self.toggle_pin(r, c))

    def get_core_cell_color(self, value):
        """Get color for core cell based on value"""
        if value == 'C':
            return '#87CEEB'  # Sky blue for control
        elif value == 'F':
            return '#90EE90'  # Light green for fuel
        elif value == 'E':
            return '#FFB6C1'  # Light pink for enhanced
        elif value.startswith('I_'):
            return '#F0F0F0'  # Light gray for irradiation
        else:
            return '#FFFFFF'  # White for unknown

    def toggle_core_cell(self, row, col):
        """Toggle core cell value on click"""
        current = self.core_lattice[row][col]

        # Define toggle sequence
        if current == 'F':
            new_value = 'C'
        elif current == 'C':
            new_value = 'E'
        elif current == 'E':
            new_value = self.get_next_irradiation_position()
        elif current.startswith('I_'):
            new_value = 'F'
        else:
            new_value = 'F'

        # Update lattice
        self.core_lattice[row][col] = new_value

        # If we removed an irradiation position, renumber
        if current.startswith('I_') and not new_value.startswith('I_'):
            self.renumber_irradiation_positions()

        # Refresh display
        self.create_core_grid()

    def toggle_pin(self, row, col):
        """Toggle pin between fuel and guide tube"""
        if (row, col) in self.guide_tube_positions:
            self.guide_tube_positions.remove((row, col))
        else:
            self.guide_tube_positions.append((row, col))

        # Refresh display
        self.create_pin_grid()

    def get_next_irradiation_position(self):
        """Get the next available irradiation position number"""
        irradiation_nums = []
        for row in self.core_lattice:
            for cell in row:
                if cell.startswith('I_'):
                    num = int(cell.split('_')[1])
                    irradiation_nums.append(num)

        if irradiation_nums:
            return f'I_{max(irradiation_nums) + 1}'
        else:
            return 'I_1'

    def renumber_irradiation_positions(self):
        """Renumber all irradiation positions to be sequential"""
        # Find all irradiation positions
        irradiation_positions = []
        for i, row in enumerate(self.core_lattice):
            for j, cell in enumerate(row):
                if cell.startswith('I_'):
                    num = int(cell.split('_')[1])
                    irradiation_positions.append((i, j, num))

        # Sort by current number
        irradiation_positions.sort(key=lambda x: x[2])

        # Renumber sequentially
        for idx, (i, j, old_num) in enumerate(irradiation_positions):
            new_num = idx + 1
            self.core_lattice[i][j] = f'I_{new_num}'

    def update_core_size(self):
        """Update core lattice size based on spinbox values"""
        new_rows = self.core_rows_var.get()
        new_cols = self.core_cols_var.get()

        old_rows = len(self.core_lattice)
        old_cols = len(self.core_lattice[0]) if old_rows > 0 else 0

        # Create new lattice
        new_lattice = []
        for i in range(new_rows):
            row = []
            for j in range(new_cols):
                if i < old_rows and j < old_cols:
                    row.append(self.core_lattice[i][j])
                else:
                    row.append('F')  # Default to fuel
            new_lattice.append(row)

        self.core_lattice = new_lattice
        self.create_core_grid()

    def fill_all_core(self, cell_type):
        """Fill all core cells with specified type"""
        for i in range(len(self.core_lattice)):
            for j in range(len(self.core_lattice[0])):
                self.core_lattice[i][j] = cell_type

        self.create_core_grid()

    def clear_guide_tubes(self):
        """Clear all guide tube positions"""
        self.guide_tube_positions = []
        self.create_pin_grid()

    def load_4x4_preset(self):
        """Load 4x4 preset"""
        self.core_lattice = [
            ['C', 'F', 'F', 'C'],
            ['F', 'I_1', 'I_2', 'F'],
            ['F', 'I_3', 'I_4', 'F'],
            ['C', 'F', 'F', 'C']
        ]
        self.core_rows_var.set(4)
        self.core_cols_var.set(4)
        self.create_core_grid()

    def load_6x6_preset(self):
        """Load 6x6 preset"""
        self.core_lattice = [
            ['C', 'C', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1', 'I_2', 'F', 'C'],
            ['F', 'I_3', 'F', 'F', 'I_4', 'F'],
            ['F', 'I_5', 'F', 'F', 'I_6', 'F'],
            ['C', 'F', 'I_7', 'I_8', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'C', 'C']
        ]
        self.core_rows_var.set(6)
        self.core_cols_var.set(6)
        self.create_core_grid()

    def load_7x7_preset(self):
        """Load 7x7 research preset"""
        self.core_lattice = [
            ['C', 'C', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'I_1', 'F', 'F', 'C'],
            ['F', 'F', 'E', 'I_2', 'E', 'F', 'F'],
            ['F', 'I_3', 'I_4', 'C', 'I_5', 'I_6', 'F'],
            ['F', 'F', 'E', 'I_7', 'E', 'F', 'F'],
            ['C', 'F', 'F', 'I_8', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'C', 'C']
        ]
        self.core_rows_var.set(7)
        self.core_cols_var.set(7)
        self.create_core_grid()

    def load_8x8_preset(self):
        """Load 8x8 complex preset"""
        self.core_lattice = [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'I_1', 'I_2', 'F', 'F', 'C'],
            ['F', 'F', 'I_3', 'F', 'F', 'I_4', 'F', 'F'],
            ['F', 'I_5', 'F', 'C', 'C', 'F', 'I_6', 'F'],
            ['F', 'I_7', 'F', 'C', 'C', 'F', 'I_8', 'F'],
            ['F', 'F', 'I_9', 'F', 'F', 'I_10', 'F', 'F'],
            ['C', 'F', 'F', 'I_11', 'I_12', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C']
        ]
        self.core_rows_var.set(8)
        self.core_cols_var.set(8)
        self.create_core_grid()

    def on_assembly_type_change(self, event=None):
        """Handle assembly type change"""
        self.update_pin_designer_visibility()

    def update_pin_designer_visibility(self):
        """Show/hide pin designer based on assembly type"""
        if self.assembly_type_var.get() == "Pin":
            self.pin_frame_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        else:
            self.pin_frame_container.pack_forget()

    def apply_all_changes(self):
        """Apply all design changes to the main GUI"""
        # Update main GUI inputs
        self.main_gui.current_inputs['core_lattice'] = copy.deepcopy(self.core_lattice)
        self.main_gui.current_inputs['assembly_type'] = self.assembly_type_var.get()

        if self.assembly_type_var.get() == "Pin":
            self.main_gui.current_inputs['guide_tube_positions'] = copy.deepcopy(self.guide_tube_positions)

        # Update visualization
        self.main_gui.schedule_update()

        # Show status
        self.status_label.config(text="✓ Changes applied to reactor", foreground='green')
        self.parent.after(3000, lambda: self.status_label.config(text=""))

    def reset_to_current(self):
        """Reset all designs to current reactor configuration"""
        # Reset from main GUI
        self.design_inputs = copy.deepcopy(self.main_gui.current_inputs)
        self.core_lattice = copy.deepcopy(self.design_inputs['core_lattice'])
        self.n_side_pins = int(self.design_inputs.get('n_side_pins', 17))
        self.guide_tube_positions = copy.deepcopy(self.design_inputs.get('guide_tube_positions', []))

        # Update UI
        self.assembly_type_var.set(self.design_inputs["assembly_type"])
        self.core_rows_var.set(len(self.core_lattice))
        self.core_cols_var.set(len(self.core_lattice[0]) if self.core_lattice else 4)

        # Refresh displays
        self.create_core_grid()
        self.create_pin_grid()
        self.update_pin_designer_visibility()

        # Show status
        self.status_label.config(text="✓ Reset to current configuration", foreground='blue')
        self.parent.after(3000, lambda: self.status_label.config(text=""))
