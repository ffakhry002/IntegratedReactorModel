"""
Core Lattice Designer
Visual designer for reactor core lattice configurations
"""
import tkinter as tk
from tkinter import ttk, messagebox


class CoreLatticeDesigner:
    """Designer window for core lattice configuration"""

    def __init__(self, parent, current_lattice=None, callback=None):
        self.parent = parent
        self.callback = callback

        # Initialize with current lattice or default
        if current_lattice and isinstance(current_lattice, list):
            self.lattice = [row[:] for row in current_lattice]  # Deep copy
        else:
            # Default 4x4 lattice
            self.lattice = [
                ['C', 'F', 'F', 'C'],
                ['F', 'I_1', 'I_2', 'F'],
                ['F', 'I_3', 'I_4', 'F'],
                ['C', 'F', 'F', 'C']
            ]

        # Create window
        self.window = tk.Toplevel(parent)
        self.window.title("Core Lattice Designer")
        self.window.geometry("800x650")  # Made taller to ensure buttons are visible

        # Make modal
        self.window.transient(parent)
        self.window.grab_set()

        # Setup UI
        self.setup_ui()

        # Center window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (self.window.winfo_width() // 2)
        y = (self.window.winfo_screenheight() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")

    def setup_ui(self):
        """Setup the designer UI"""
        # Main container with explicit pack to ensure it doesn't overlap buttons
        main_container = ttk.Frame(self.window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Content frame (everything except buttons)
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - controls
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Instructions
        instr_frame = ttk.LabelFrame(left_panel, text="Instructions", padding=5)
        instr_frame.pack(fill=tk.X, pady=(0, 10))

        instructions = """Click cells to cycle through:
F → C → E → I_N → F

Where:
- F = Fuel
- C = Coolant
- E = Extra Enriched Fuel
- I_N = Irradiation Position N

Irradiation positions are
automatically numbered."""

        ttk.Label(instr_frame, text=instructions, justify=tk.LEFT).pack()

        # Size controls
        size_frame = ttk.LabelFrame(left_panel, text="Grid Size", padding=5)
        size_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(size_frame, text="Rows:").grid(row=0, column=0, sticky=tk.W)
        self.rows_var = tk.IntVar(value=len(self.lattice))
        rows_spin = ttk.Spinbox(size_frame, from_=2, to=12, textvariable=self.rows_var,
                               width=10, command=self.update_size)
        rows_spin.grid(row=0, column=1, padx=(5, 0))

        ttk.Label(size_frame, text="Columns:").grid(row=1, column=0, sticky=tk.W)
        self.cols_var = tk.IntVar(value=len(self.lattice[0]) if self.lattice else 4)
        cols_spin = ttk.Spinbox(size_frame, from_=2, to=12, textvariable=self.cols_var,
                               width=10, command=self.update_size)
        cols_spin.grid(row=1, column=1, padx=(5, 0))

        # Preset configurations
        preset_frame = ttk.LabelFrame(left_panel, text="Presets", padding=5)
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        self.preset_var = tk.StringVar(value="Select preset...")
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                   values=["Select preset...", "4x4 Simple", "6x6 with Center",
                                          "7x7 Symmetric", "8x8 Complex"],
                                   state="readonly", width=15)
        preset_combo.pack(fill=tk.X)
        preset_combo.bind('<<ComboboxSelected>>', lambda e: self.load_preset())

        # Quick fill buttons
        fill_frame = ttk.LabelFrame(left_panel, text="Quick Fill", padding=5)
        fill_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(fill_frame, text="Fill All Fuel",
                  command=lambda: self.fill_all('F')).pack(fill=tk.X, pady=2)
        ttk.Button(fill_frame, text="Fill All Coolant",
                  command=lambda: self.fill_all('C')).pack(fill=tk.X, pady=2)
        ttk.Button(fill_frame, text="Fill All Extra Enriched Fuel",
                  command=lambda: self.fill_all('E')).pack(fill=tk.X, pady=2)

        # Right panel - grid and preview
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Grid frame
        grid_container = ttk.LabelFrame(right_panel, text="Core Layout (Click to Edit)", padding=10)
        grid_container.pack(fill=tk.BOTH, expand=True)

        # Create scrollable frame for grid
        canvas = tk.Canvas(grid_container, bg='white')
        scrollbar_v = ttk.Scrollbar(grid_container, orient="vertical", command=canvas.yview)
        scrollbar_h = ttk.Scrollbar(grid_container, orient="horizontal", command=canvas.xview)
        self.grid_frame = ttk.Frame(canvas)

        self.grid_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar_v.grid(row=0, column=1, sticky="ns")
        scrollbar_h.grid(row=1, column=0, sticky="ew")

        grid_container.grid_rowconfigure(0, weight=1)
        grid_container.grid_columnconfigure(0, weight=1)

        # Preview frame
        preview_frame = ttk.LabelFrame(right_panel, text="Array Preview", padding=5)
        preview_frame.pack(fill=tk.X, pady=(10, 0))

        self.preview_text = tk.Text(preview_frame, height=6, width=50, font=('Courier', 9))
        self.preview_text.pack(fill=tk.X)

        # Action buttons - separate frame at bottom
        button_frame = ttk.Frame(main_container)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        # Show different buttons based on callback type
        if self.callback:
            # Check if we're in multi-loop mode (callback expects only lattice)
            try:
                # Test if callback expects 2 arguments
                import inspect
                sig = inspect.signature(self.callback)
                num_params = len(sig.parameters)

                if num_params == 2:
                    # Single parameter mode - show both buttons
                    ttk.Button(button_frame, text="Apply as New Run",
                              command=lambda: self.apply_lattice('new')).pack(side=tk.LEFT, padx=(0, 5))
                    ttk.Button(button_frame, text="Stack to Last Run",
                              command=lambda: self.apply_lattice('stack')).pack(side=tk.LEFT, padx=(0, 5))
                else:
                    # Multi-loop mode - just add button
                    ttk.Button(button_frame, text="Add Lattice",
                              command=lambda: self.apply_lattice_multi()).pack(side=tk.LEFT, padx=(0, 5))
            except:
                # Default to single parameter mode
                ttk.Button(button_frame, text="Apply as New Run",
                          command=lambda: self.apply_lattice('new')).pack(side=tk.LEFT, padx=(0, 5))
                ttk.Button(button_frame, text="Stack to Last Run",
                          command=lambda: self.apply_lattice('stack')).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(button_frame, text="Cancel",
                  command=self.window.destroy).pack(side=tk.RIGHT)

        # Create initial grid
        self.create_grid()
        self.update_preview()

    def create_grid(self):
        """Create the interactive grid"""
        # Clear existing grid
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        # Create grid cells
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice[0])):
                cell_value = self.lattice[i][j]

                cell = tk.Label(self.grid_frame, text=cell_value,
                              width=4, height=2,
                              relief=tk.RAISED, bd=2,
                              bg=self.get_cell_color(cell_value),
                              font=('TkDefaultFont', 12, 'bold'))
                cell.grid(row=i, column=j, padx=2, pady=2)
                cell.bind('<Button-1>', self.toggle_cell)

    def get_cell_color(self, value):
        """Get color for cell based on value"""
        if value == 'C':
            return '#87CEEB'  # Sky blue for control
        elif value == 'F':
            return '#90EE90'  # Light green for fuel
        elif value == 'E':
            return '#FFB6C1'  # Light pink for extra enriched fuel
        elif value.startswith('I_'):
            return '#F0F0F0'  # Light gray for irradiation
        else:
            return '#FFFFFF'  # White for unknown

    def toggle_cell(self, event):
        """Toggle cell value on click"""
        widget = event.widget
        row = widget.grid_info()['row']
        col = widget.grid_info()['column']

        current = self.lattice[row][col]

        # Define toggle sequence
        if current == 'F':
            new_value = 'C'
        elif current == 'C':
            new_value = 'E'
        elif current == 'E':
            # Get next available irradiation number
            new_value = self.get_next_irradiation_position()
        elif current.startswith('I_'):
            # Cycle back to F
            new_value = 'F'
        else:
            new_value = 'F'

        # Update the lattice
        self.lattice[row][col] = new_value

        # If we removed an irradiation position, renumber all remaining ones
        if current.startswith('I_') and not new_value.startswith('I_'):
            self.renumber_irradiation_positions()

        # Update all cells display (in case renumbering occurred)
        self.refresh_grid_display()

        # Update preview
        self.update_preview()

    def get_next_irradiation_position(self):
        """Get the next available irradiation position number"""
        # Find all current irradiation positions
        irradiation_nums = []
        for row in self.lattice:
            for cell in row:
                if cell.startswith('I_'):
                    num = int(cell.split('_')[1])
                    irradiation_nums.append(num)

        # Return next number
        if irradiation_nums:
            return f'I_{max(irradiation_nums) + 1}'
        else:
            return 'I_1'

    def renumber_irradiation_positions(self):
        """Renumber all irradiation positions to be sequential"""
        # Find all irradiation positions with their coordinates
        irradiation_positions = []
        for i, row in enumerate(self.lattice):
            for j, cell in enumerate(row):
                if cell.startswith('I_'):
                    num = int(cell.split('_')[1])
                    irradiation_positions.append((i, j, num))

        # Sort by current number
        irradiation_positions.sort(key=lambda x: x[2])

        # Renumber sequentially
        for idx, (i, j, old_num) in enumerate(irradiation_positions):
            new_num = idx + 1
            self.lattice[i][j] = f'I_{new_num}'

    def refresh_grid_display(self):
        """Refresh the display of all grid cells"""
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice[0])):
                # Get the cell widget at this position
                for widget in self.grid_frame.grid_slaves(row=i, column=j):
                    value = self.lattice[i][j]
                    widget.config(text=value, bg=self.get_cell_color(value))

    def update_size(self):
        """Update lattice size based on spinbox values"""
        new_rows = self.rows_var.get()
        new_cols = self.cols_var.get()

        old_rows = len(self.lattice)
        old_cols = len(self.lattice[0]) if old_rows > 0 else 0

        # Create new lattice with new size
        new_lattice = []
        for i in range(new_rows):
            row = []
            for j in range(new_cols):
                if i < old_rows and j < old_cols:
                    # Copy existing value
                    row.append(self.lattice[i][j])
                else:
                    # New cell, default to fuel
                    row.append('F')
            new_lattice.append(row)

        self.lattice = new_lattice
        self.create_grid()
        self.update_preview()

    def load_preset(self):
        """Load selected preset configuration"""
        preset_name = self.preset_var.get()
        if preset_name == "Select preset...":
            return

        if preset_name == "4x4 Simple":
            self.lattice = [
                ['C', 'F', 'F', 'C'],
                ['F', 'I_1', 'I_2', 'F'],
                ['F', 'I_3', 'I_4', 'F'],
                ['C', 'F', 'F', 'C']
            ]
        elif preset_name == "6x6 with Center":
            self.lattice = [
                ['C', 'F', 'F', 'F', 'F', 'C'],
                ['F', 'E', 'I_1', 'I_2', 'E', 'F'],
                ['F', 'I_3', 'C', 'C', 'I_4', 'F'],
                ['F', 'I_5', 'C', 'C', 'I_6', 'F'],
                ['F', 'E', 'I_7', 'I_8', 'E', 'F'],
                ['C', 'F', 'F', 'F', 'F', 'C']
            ]
        elif preset_name == "7x7 Symmetric":
            self.lattice = [
                ['C', 'F', 'F', 'F', 'F', 'F', 'C'],
                ['F', 'C', 'I_1', 'I_2', 'I_3', 'C', 'F'],
                ['F', 'I_4', 'E', 'I_5', 'E', 'I_6', 'F'],
                ['F', 'I_7', 'I_8', 'C', 'I_9', 'I_10', 'F'],
                ['F', 'I_11', 'E', 'I_12', 'E', 'I_13', 'F'],
                ['F', 'C', 'I_14', 'I_15', 'I_16', 'C', 'F'],
                ['C', 'F', 'F', 'F', 'F', 'F', 'C']
            ]
        elif preset_name == "8x8 Complex":
            # Create 8x8 with automatic numbering
            self.lattice = []
            irrad_num = 1
            for i in range(8):
                row = []
                for j in range(8):
                    if (i == 0 or i == 7) and (j == 0 or j == 7):
                        row.append('C')
                    elif i in [3, 4] and j in [3, 4]:
                        row.append('C')
                    elif (i in [1, 6] or j in [1, 6]) and not ((i == 0 or i == 7) or (j == 0 or j == 7)):
                        if i % 2 == j % 2:
                            row.append('E')
                        else:
                            row.append(f'I_{irrad_num}')
                            irrad_num += 1
                    else:
                        row.append('F')
                self.lattice.append(row)

        # Update size spinboxes
        self.rows_var.set(len(self.lattice))
        self.cols_var.set(len(self.lattice[0]))

        self.create_grid()
        self.update_preview()

    def fill_all(self, value):
        """Fill all cells with specified value"""
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice[0])):
                self.lattice[i][j] = value

        self.create_grid()
        self.update_preview()

    def clear_irradiation(self):
        """Clear all irradiation positions"""
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice[0])):
                if self.lattice[i][j].startswith('I_'):
                    self.lattice[i][j] = 'F'

        self.create_grid()
        self.update_preview()

    def update_preview(self):
        """Update the array preview"""
        # Generate Python array representation
        preview_lines = ["["]
        for i, row in enumerate(self.lattice):
            row_str = "    " + str(row)
            if i < len(self.lattice) - 1:
                row_str += ","
            preview_lines.append(row_str)
        preview_lines.append("]")

        # Update preview text
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(tk.END, "\n".join(preview_lines))

    def apply_lattice(self, action):
        """Apply the designed lattice for single parameter mode"""
        if self.callback:
            try:
                self.callback(self.lattice, action)
            except TypeError:
                # Callback might only expect one argument (multi-loop mode)
                self.callback(self.lattice)

        self.window.destroy()

    def apply_lattice_multi(self):
        """Apply the designed lattice for multi-loop mode"""
        if self.callback:
            self.callback(self.lattice)

        self.window.destroy()
