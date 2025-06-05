"""
Depletion Timesteps Designer
Visual designer for depletion timesteps configuration
"""
import tkinter as tk
from tkinter import ttk, messagebox


class DepletionTimestepsDesigner:
    """Designer window for depletion timesteps configuration"""

    def __init__(self, parent, current_value=None, callback=None):
        self.parent = parent
        self.current_value = current_value or []
        self.callback = callback

        # Create window
        self.window = tk.Toplevel(parent)
        self.window.title("Depletion Timesteps Designer")
        self.window.geometry("600x500")

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
        # Main container
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Instructions
        instr_frame = ttk.LabelFrame(main_frame, text="Instructions", padding=5)
        instr_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(instr_frame, text="Configure depletion timesteps by specifying steps and size for each segment.\n" +
                                    "Size represents the timestep size in MWd/kgHM or days (depending on units).",
                 justify=tk.LEFT).pack()

        # Timesteps configuration
        config_frame = ttk.LabelFrame(main_frame, text="Timestep Configuration", padding=5)
        config_frame.pack(fill=tk.BOTH, expand=True)

        # Create scrollable frame for timestep entries
        canvas = tk.Canvas(config_frame, height=200)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Header
        header_frame = ttk.Frame(config_frame)
        header_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(header_frame, text="Steps", width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(header_frame, text="Size", width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(header_frame, text="Total", width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(header_frame, text="", width=10).pack(side=tk.LEFT)  # For remove button

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Buttons for adding timesteps
        button_frame = ttk.Frame(config_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(button_frame, text="Add Timestep",
                  command=self.add_timestep_row).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Clear All",
                  command=self.clear_all_timesteps).pack(side=tk.LEFT)

        # Preview
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding=5)
        preview_frame.pack(fill=tk.X, pady=(10, 0))

        self.preview_text = tk.Text(preview_frame, height=5, width=50, wrap=tk.WORD)
        self.preview_text.pack(fill=tk.X)

        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(action_frame, text="Apply as New Run",
                  command=lambda: self.apply_timesteps('new'),
                  style='Accent.TButton').pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(action_frame, text="Stack to Last Run",
                  command=lambda: self.apply_timesteps('stack')).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(action_frame, text="Cancel",
                  command=self.window.destroy).pack(side=tk.RIGHT)

        # Initialize with current value or default
        self.timestep_rows = []
        if self.current_value and isinstance(self.current_value, list):
            # Parse current value
            for item in self.current_value:
                if isinstance(item, dict) and 'steps' in item and 'size' in item:
                    self.add_timestep_row(item['steps'], item['size'])
                elif isinstance(item, (int, float)):
                    # Legacy format - single value
                    self.add_timestep_row(1, item)
        else:
            # Add a few default rows
            self.add_timestep_row(5, 1.0)
            self.add_timestep_row(5, 2.0)
            self.add_timestep_row(5, 5.0)

        self.update_preview()

    def add_timestep_row(self, steps=5, size=1.0):
        """Add a timestep configuration row"""
        row_frame = ttk.Frame(self.scrollable_frame)
        row_frame.pack(fill=tk.X, pady=2)

        # Steps entry
        steps_var = tk.IntVar(value=steps)
        steps_entry = ttk.Entry(row_frame, textvariable=steps_var, width=10)
        steps_entry.pack(side=tk.LEFT, padx=(0, 5))

        # Size entry
        size_var = tk.DoubleVar(value=size)
        size_entry = ttk.Entry(row_frame, textvariable=size_var, width=10)
        size_entry.pack(side=tk.LEFT, padx=(0, 5))

        # Total label
        total_label = ttk.Label(row_frame, text=f"{steps * size:.1f}", width=10)
        total_label.pack(side=tk.LEFT, padx=(0, 5))

        # Update total when values change
        def update_total(*args):
            try:
                total = steps_var.get() * size_var.get()
                total_label.config(text=f"{total:.1f}")
                self.update_preview()
            except:
                pass

        steps_var.trace('w', update_total)
        size_var.trace('w', update_total)

        # Remove button
        ttk.Button(row_frame, text="Remove",
                  command=lambda: self.remove_timestep_row(row_frame)).pack(side=tk.LEFT)

        # Store row data
        self.timestep_rows.append({
            'frame': row_frame,
            'steps_var': steps_var,
            'size_var': size_var,
            'total_label': total_label
        })

    def remove_timestep_row(self, row_frame):
        """Remove a timestep row"""
        # Find and remove from list
        for row in self.timestep_rows:
            if row['frame'] == row_frame:
                self.timestep_rows.remove(row)
                break

        # Destroy frame
        row_frame.destroy()

        # Update preview
        self.update_preview()

    def clear_all_timesteps(self):
        """Clear all timestep rows"""
        for row in self.timestep_rows:
            row['frame'].destroy()

        self.timestep_rows.clear()
        self.update_preview()

    def update_preview(self):
        """Update the preview text"""
        if not self.timestep_rows:
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, "No timesteps configured")
            return

        # Generate timesteps array
        timesteps = []
        for row in self.timestep_rows:
            try:
                steps = row['steps_var'].get()
                size = row['size_var'].get()
                timesteps.append({"steps": steps, "size": size})
            except:
                pass

        # Format for display
        preview_lines = ["["]
        for i, ts in enumerate(timesteps):
            line = f'    {{"steps": {ts["steps"]}, "size": {ts["size"]}}}'
            if i < len(timesteps) - 1:
                line += ","
            line += f'  # {ts["steps"]} steps of {ts["size"]} MWd/kgHM or days'
            preview_lines.append(line)
        preview_lines.append("]")

        # Update preview
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(tk.END, "\n".join(preview_lines))

    def apply_timesteps(self, action):
        """Apply the configured timesteps"""
        # Collect timesteps
        timesteps = []
        for row in self.timestep_rows:
            try:
                steps = row['steps_var'].get()
                size = row['size_var'].get()
                if steps > 0 and size > 0:
                    timesteps.append({"steps": steps, "size": size})
            except:
                pass

        if not timesteps:
            messagebox.showwarning("No Timesteps", "Please configure at least one timestep")
            return

        # Call callback with timesteps and action
        if self.callback:
            self.callback(timesteps, action)

        # Close window
        self.window.destroy()
