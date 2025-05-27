"""
Preview Tab Component
Handles preview and export of run configurations
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from datetime import datetime
import sys

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import utils
from utils.export_utils import ExportUtils


class PreviewTab:
    """Preview and export tab"""

    def __init__(self, parent, run_config):
        self.parent = parent
        self.run_config = run_config

        # UI elements
        self.preview_text = None

    def setup(self):
        """Setup the preview and export tab"""
        # Main container
        container = ttk.Frame(self.parent)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Controls
        control_frame = ttk.Frame(container)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(control_frame, text="Generate Preview",
                  command=self.generate_preview).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Export run_dictionaries.py",
                  command=self.export_run_dictionaries).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Load Existing File",
                  command=self.load_existing_file).pack(side=tk.LEFT)

        # Summary info
        summary_frame = ttk.LabelFrame(container, text="Configuration Summary", padding=10)
        summary_frame.pack(fill=tk.X, pady=(0, 10))

        self.summary_label = ttk.Label(summary_frame, text="Click 'Generate Preview' to see configuration summary")
        self.summary_label.pack()

        # Preview area
        preview_container = ttk.LabelFrame(container, text="Generated run_dictionaries.py Preview", padding=10)
        preview_container.pack(fill=tk.BOTH, expand=True)

        # Text widget with scrollbar
        text_frame = ttk.Frame(preview_container)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_text = tk.Text(text_frame, wrap=tk.NONE, font=('Courier', 10))

        # Scrollbars
        v_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.preview_text.yview)
        h_scroll = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=self.preview_text.xview)

        self.preview_text.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        # Pack components
        self.preview_text.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

    def generate_preview(self):
        """Generate preview of the run_dictionaries.py file"""
        # Update summary
        self.update_summary()

        # Generate content
        content = ExportUtils.generate_run_dictionaries_content(self.run_config)

        # Clear and insert content
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(tk.END, content)

    def update_summary(self):
        """Update the configuration summary"""
        simple_count = len(self.run_config.simple_runs)
        loop_set_count = len(self.run_config.loop_sets)

        # Calculate total runs from loop sets
        loop_runs_total = 0
        for loop_set in self.run_config.loop_sets:
            set_runs = 1
            valid_loops = 0

            for loop in loop_set.get('loops', []):
                param = loop.get('param_var')
                values = loop.get('values_var')

                if param and values:
                    try:
                        param_val = param.get() if hasattr(param, 'get') else loop.get('param', '')
                        values_val = values.get() if hasattr(values, 'get') else loop.get('values', '')

                        if param_val and values_val:
                            # Count values (simplified)
                            count = values_val.count(',') + 1 if ',' in values_val else 1
                            if count > 0:
                                set_runs *= count
                                valid_loops += 1
                    except:
                        pass

            if valid_loops > 0:
                loop_runs_total += set_runs

        total_runs = simple_count + loop_runs_total

        summary_text = f"""Configuration Summary:
- Simple parameter runs: {simple_count}
- Loop sets configured: {loop_set_count}
- Estimated total runs from loops: {loop_runs_total}
- Total runs (including default): {total_runs + 1}"""

        self.summary_label.config(text=summary_text)

    def export_run_dictionaries(self):
        """Export the run_dictionaries.py file"""
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        default_filename = f"run_dictionaries_{timestamp}.py"

        # Get the path to the main folder
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_path = os.path.join(current_dir, default_filename)

        # Get save location
        filename = filedialog.asksaveasfilename(
            title="Save run_dictionaries.py",
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            initialdir=current_dir,
            initialfile=default_filename
        )

        if filename:
            try:
                content = ExportUtils.generate_run_dictionaries_content(self.run_config)
                with open(filename, 'w') as f:
                    f.write(content)

                messagebox.showinfo("Success", f"Exported run_dictionaries.py to:\n{filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to export file:\n{e}")

    def load_existing_file(self):
        """Load an existing run_dictionaries.py file"""
        # Get the path to the main folder
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        filename = filedialog.askopenfilename(
            title="Load run_dictionaries.py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            initialdir=current_dir
        )

        if filename:
            try:
                with open(filename, 'r') as f:
                    content = f.read()

                # Clear preview and load content
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(tk.END, content)

                # Update summary to show loaded file
                self.summary_label.config(text=f"Loaded file: {os.path.basename(filename)}")

                messagebox.showinfo("Success", "File loaded successfully")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{e}")
