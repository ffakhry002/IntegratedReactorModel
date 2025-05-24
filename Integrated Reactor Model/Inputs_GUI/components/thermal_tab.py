"""
Thermal Tab Component
Handles thermal hydraulics parameters and analysis
"""
import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import glob
import tempfile
import threading
from PIL import Image, ImageTk

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Inputs_GUI.controls.parameter_controls import add_text_control
from ThermalHydraulics.TH_refactored import THSystem


class ThermalTab:
    def __init__(self, parent, main_gui):
        self.parent = parent
        self.main_gui = main_gui

        # TH analysis state
        self.run_thermal_button = None
        self.thermal_progress_bar = None
        self.thermal_status = None
        self.thermal_output_dir = None

        # Image viewer state
        self.current_images = []
        self.current_image_index = 0
        self.image_label = None

    def setup(self):
        """Setup the thermal hydraulics tab"""
        # Main container
        thermal_main = ttk.Frame(self.parent)
        thermal_main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for controls
        left_frame = ttk.Frame(thermal_main)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        control_frame = ttk.LabelFrame(left_frame, text="Thermal Hydraulics Controls", padding=10)
        control_frame.pack(fill=tk.BOTH, expand=True)

        # Create scrollable frame
        canvas = tk.Canvas(control_frame, width=350)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Add controls
        self.setup_thermal_controls(scrollable_frame)

        # Right panel for image display
        viewer_frame = ttk.LabelFrame(thermal_main, text="Thermal Analysis Results", padding=10)
        viewer_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Setup image viewer
        self.setup_image_viewer(viewer_frame)

    def setup_thermal_controls(self, parent):
        """Setup thermal hydraulics controls"""
        inputs = self.main_gui.current_inputs

        # Operating Conditions
        ttk.Label(parent, text="Operating Conditions",
                 font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(5, 5))

        add_text_control(parent, "System Pressure (Pa):", 'reactor_pressure',
                        self.main_gui, lambda: self.on_param_change())
        add_text_control(parent, "Coolant Flow Rate (m/s):", 'flow_rate',
                        self.main_gui, lambda: self.on_param_change())
        add_text_control(parent, "Inlet Temperature (K):", 'T_inlet',
                        self.main_gui, lambda: self.on_param_change())

        # Power Distribution
        ttk.Label(parent, text="Power Distribution",
                 font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(15, 5))

        add_text_control(parent, "Input Power Density (kW/L):", 'input_power_density',
                        self.main_gui, lambda: self.on_param_change())
        add_text_control(parent, "Max Linear Power (kW/m):", 'max_linear_power',
                        self.main_gui, lambda: self.on_param_change())
        add_text_control(parent, "Average Linear Power (kW/m):", 'average_linear_power',
                        self.main_gui, lambda: self.on_param_change())
        add_text_control(parent, "Cosine Curve Squeeze (0-1):", 'cos_curve_squeeze',
                        self.main_gui, lambda: self.on_param_change())

        # TH Profile Type
        ttk.Label(parent, text="TH Profile Type:").pack(anchor=tk.W, pady=(5, 2))
        self.th_profile_var = tk.StringVar(value=inputs.get('CP_PD_MLP_ALP', 'CP'))
        th_profile_combo = ttk.Combobox(parent, textvariable=self.th_profile_var,
                                       values=['CP', 'PD', 'MLP', 'ALP'],
                                       state="readonly", width=20)
        th_profile_combo.pack(fill=tk.X, pady=(0, 10))
        th_profile_combo.bind('<<ComboboxSelected>>',
                             lambda e: self.update_thermal_input('CP_PD_MLP_ALP',
                                                               self.th_profile_var.get()))

        # Instructions
        instr_text = """Profile Types:
• CP: Core Power (MW)
• PD: Power Density (kW/L)
• MLP: Max Linear Power (kW/m)
• ALP: Avg Linear Power (kW/m)

"""

        ttk.Label(parent, text=instr_text, justify=tk.LEFT, font=('Arial', 8)).pack(pady=(0, 10))

        # Run button
        self.run_thermal_button = ttk.Button(parent, text="Run Thermal Hydraulics",
                                           command=self.run_thermal_hydraulics,
                                           style='Accent.TButton')
        self.run_thermal_button.pack(fill=tk.X, pady=(0, 10))

        # Progress bar
        self.thermal_progress_bar = ttk.Progressbar(parent, mode='indeterminate')
        self.thermal_progress_bar.pack(fill=tk.X, pady=(0, 10))

        # Status label
        self.thermal_status = ttk.Label(parent, text="Ready to run")
        self.thermal_status.pack(fill=tk.X)

    def setup_image_viewer(self, parent):
        """Setup the image viewer panel with improved configuration"""
        # Navigation controls
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(fill=tk.X, pady=(0, 10))

        # Previous button
        self.prev_button = ttk.Button(
            nav_frame,
            text="◀ Previous",
            command=self.previous_image,
            state=tk.DISABLED
        )
        self.prev_button.pack(side=tk.LEFT, padx=(0, 5))

        # Image name label
        self.image_name_label = ttk.Label(
            nav_frame,
            text="No images loaded",
            font=('TkDefaultFont', 10, 'bold')
        )
        self.image_name_label.pack(side=tk.LEFT, expand=True)

        # Next button
        self.next_button = ttk.Button(
            nav_frame,
            text="Next ▶",
            command=self.next_image,
            state=tk.DISABLED
        )
        self.next_button.pack(side=tk.RIGHT, padx=(5, 0))

        # Image display area with scrollbars
        image_container = ttk.Frame(parent)
        image_container.pack(fill=tk.BOTH, expand=True)

        # Create canvas and scrollbars
        self.canvas = tk.Canvas(image_container, bg='#f0f0f0')  # Light gray background
        v_scrollbar = ttk.Scrollbar(image_container, orient="vertical", command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(image_container, orient="horizontal", command=self.canvas.xview)

        self.canvas.configure(
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set,
            highlightthickness=0  # Remove border
        )

        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind mouse events for pan and zoom
        self.canvas.bind("<ButtonPress-1>", self.on_pan_start)
        self.canvas.bind("<B1-Motion>", self.on_pan_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_pan_end)
        self.canvas.bind("<MouseWheel>", self.on_zoom)  # Windows/Linux
        self.canvas.bind("<Button-4>", self.on_zoom)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_zoom)  # Linux scroll down

        # Configure canvas for better responsiveness
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Pan and zoom state
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.is_panning = False
        self.zoom_level = 1.0
        self.min_zoom = 0.1  # Allow zooming out to 10% for overview
        self.max_zoom = 10.0  # Allow zooming in to 10x for detail

        # Initial message
        self.canvas.create_text(
            400, 300,
            text="Run thermal hydraulics analysis to view results",
            font=('Arial', 14), fill='gray', tags="placeholder"
        )

        # Image info label at bottom
        info_label = ttk.Label(parent, text="Use mouse wheel to zoom, click and drag to pan",
                             font=('TkDefaultFont', 9), foreground='gray')
        info_label.pack(side=tk.BOTTOM, pady=(5, 0))

    def _on_canvas_configure(self, event):
        """Handle canvas resize events"""
        # If we have an image displayed and canvas size changed significantly
        if hasattr(self, 'current_images') and self.current_images:
            # You could add logic here to re-center image on window resize if desired
            pass

    def on_param_change(self):
        """Handle parameter change"""
        if hasattr(self.main_gui, 'viz_tab') and self.main_gui.viz_tab.auto_update_var.get():
            self.main_gui.schedule_update()

    def update_thermal_input(self, key, value):
        """Update thermal input parameter"""
        self.main_gui.current_inputs[key] = value

    def run_thermal_hydraulics(self):
        """Run thermal hydraulics analysis in a separate thread"""
        # Disable button and show progress
        self.run_thermal_button.config(state=tk.DISABLED)
        self.thermal_progress_bar.start()
        self.thermal_status.config(text="Initializing thermal hydraulics...")

        # Run in separate thread
        thread = threading.Thread(target=self._run_thermal_analysis)
        thread.daemon = True
        thread.start()

    def _run_thermal_analysis(self):
        """Actual thermal hydraulics analysis execution"""
        try:
            # Step 1: Initialize thermal hydraulics system
            self.parent.after(0, lambda: self.thermal_status.config(
                text="Setting up thermal hydraulics system..."))

            # Create temporary directory for outputs
            self.thermal_output_dir = tempfile.mkdtemp(prefix="thermal_hydraulics_")

            # Update outputs folder in inputs
            th_inputs = self.main_gui.current_inputs.copy()
            th_inputs['outputs_folder'] = 'local_outputs'

            # Step 2: Run thermal hydraulics
            self.parent.after(0, lambda: self.thermal_status.config(
                text="Running thermal hydraulics simulation..."))

            # Create THSystem with current GUI inputs
            th_system = THSystem(th_inputs)

            # Calculate temperature distribution
            thermal_state = th_system.calculate_temperature_distribution()

            # Step 3: Generate plots
            self.parent.after(0, lambda: self.thermal_status.config(
                text="Generating plots..."))

            # Write results and generate plots
            th_system.write_results(output_dir=self.thermal_output_dir, plotting=True)

            # Step 4: Find generated plots
            self.parent.after(0, lambda: self.thermal_status.config(
                text="Loading results..."))

            # Look for plots in the output directory
            plot_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                plot_files.extend(glob.glob(os.path.join(self.thermal_output_dir, '**', ext), recursive=True))

            if plot_files:
                # Sort by filename for consistent order
                plot_files.sort()

                # Update UI in main thread
                self.parent.after(0, lambda: self.load_thermal_results(plot_files))
            else:
                raise FileNotFoundError("No plot files were generated")

        except Exception as e:
            # Show error in main thread
            error_message = str(e)
            self.parent.after(0, lambda msg=error_message: self.show_thermal_error(msg))

        finally:
            # Cleanup in main thread
            self.parent.after(0, self.cleanup_thermal_run)

    def load_thermal_results(self, plot_files):
        """Load and display thermal results"""
        self.current_images = []
        for img_path in plot_files:
            # Store both the path and the image name
            img_name = os.path.basename(img_path).replace('.png', '').replace('_', ' ').title()
            self.current_images.append({
                'path': img_path,
                'name': img_name,
                'pil_image': None  # Will load on demand
            })

        # Reset to first image
        self.current_image_index = 0
        self.zoom_level = 1.0
        self.display_current_image()

        # Enable navigation
        self.update_navigation_buttons()

        self.thermal_status.config(text=f"Loaded {len(plot_files)} thermal plots")

    def display_current_image(self, center_image=True):
        """Display the current image with improved quality and positioning"""
        if not self.current_images:
            return

        current_img = self.current_images[self.current_image_index]

        # Load image if not already loaded
        if current_img['pil_image'] is None:
            img = Image.open(current_img['path'])
            if img.mode != 'RGB':
                img = img.convert('RGB')
            current_img['pil_image'] = img

        # Remove placeholder text
        self.canvas.delete("placeholder")

        # Get original image dimensions
        img_width, img_height = current_img['pil_image'].size

        # Calculate fit-to-canvas zoom if zoom level is 1.0 (initial load)
        if self.zoom_level == 1.0 and center_image:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                # Always fit the entire image in view with some margin
                scale_x = (canvas_width * 0.9) / img_width  # 90% of canvas width
                scale_y = (canvas_height * 0.9) / img_height  # 90% of canvas height
                # Use the smaller scale to ensure entire image fits
                self.zoom_level = min(scale_x, scale_y)
                # But don't zoom in more than 2x original size for initial view
                self.zoom_level = min(self.zoom_level, 2.0)
                # And don't zoom out more than 20% for readability
                self.zoom_level = max(self.zoom_level, 0.2)

        # Apply zoom
        display_width = int(img_width * self.zoom_level)
        display_height = int(img_height * self.zoom_level)

        # Create display image with high quality
        # For better quality when zoomed out, we'll use a higher quality threshold
        if abs(self.zoom_level - 1.0) > 0.001:  # More sensitive threshold
            # Use LANCZOS for downsampling (zoom < 1) and BICUBIC for upsampling
            resample_filter = Image.Resampling.LANCZOS if self.zoom_level < 1.0 else Image.Resampling.BICUBIC
            display_image = current_img['pil_image'].resize(
                (display_width, display_height),
                resample_filter
            )
        else:
            display_image = current_img['pil_image']

        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(display_image)

        # Clear canvas
        self.canvas.delete("all")

        # Always create image at origin (0,0) for consistent positioning
        self.canvas_image = self.canvas.create_image(
            0, 0,
            image=photo,
            anchor=tk.NW
        )

        # Keep reference to prevent garbage collection
        self.canvas.image = photo

        # Set scrollregion to match image size exactly
        self.canvas.configure(scrollregion=(0, 0, display_width, display_height))

        # Center the view if requested
        if center_image:
            self.canvas.update_idletasks()  # Ensure canvas geometry is updated
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Calculate scroll positions to center the image
            if display_width > canvas_width:
                x_center = (display_width - canvas_width) / 2
                self.canvas.xview_moveto(x_center / display_width)
            else:
                self.canvas.xview_moveto(0)

            if display_height > canvas_height:
                y_center = (display_height - canvas_height) / 2
                self.canvas.yview_moveto(y_center / display_height)
            else:
                self.canvas.yview_moveto(0)

        # Update image name
        self.image_name_label.config(
            text=f"{current_img['name']} ({self.current_image_index + 1}/{len(self.current_images)})"
        )

    def previous_image(self):
        """Show previous image"""
        if self.current_images and self.current_image_index > 0:
            self.current_image_index -= 1
            self.zoom_level = 1.0  # Reset zoom when changing images
            self.display_current_image(center_image=True)
            self.update_navigation_buttons()

    def next_image(self):
        """Show next image"""
        if self.current_images and self.current_image_index < len(self.current_images) - 1:
            self.current_image_index += 1
            self.zoom_level = 1.0  # Reset zoom when changing images
            self.display_current_image(center_image=True)
            self.update_navigation_buttons()

    def update_navigation_buttons(self):
        """Update navigation button states"""
        if not self.current_images:
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
        else:
            self.prev_button.config(
                state=tk.NORMAL if self.current_image_index > 0 else tk.DISABLED
            )
            self.next_button.config(
                state=tk.NORMAL if self.current_image_index < len(self.current_images) - 1 else tk.DISABLED
            )

    def show_thermal_error(self, error_msg):
        """Show thermal analysis error"""
        self.thermal_status.config(text="Error occurred")
        messagebox.showerror("Thermal Analysis Error",
                           f"An error occurred during thermal analysis:\n\n{error_msg}")

    def cleanup_thermal_run(self):
        """Clean up after thermal run"""
        self.thermal_progress_bar.stop()
        self.run_thermal_button.config(state=tk.NORMAL)

    def on_pan_start(self, event):
        """Start panning"""
        self.canvas.scan_mark(event.x, event.y)
        self.is_panning = True
        self.canvas.config(cursor="fleur")  # Change cursor to move icon

    def on_pan_move(self, event):
        """Pan the image"""
        if self.is_panning:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_pan_end(self, event):
        """End panning"""
        self.is_panning = False
        self.canvas.config(cursor="")  # Reset cursor

    def on_zoom(self, event):
        """Handle zoom with mouse wheel - fixed for proper scrolling"""
        if not self.current_images:
            return

        # Determine zoom direction
        if event.delta:
            zoom_in = event.delta > 0
        else:
            zoom_in = event.num == 4

        # Calculate new zoom level
        zoom_factor = 1.1  # 10% per scroll for good responsiveness
        if zoom_in:
            new_zoom = min(self.zoom_level * zoom_factor, self.max_zoom)
        else:
            new_zoom = max(self.zoom_level / zoom_factor, self.min_zoom)

        if abs(new_zoom - self.zoom_level) < 0.001:  # No significant change
            return

        # Get current image
        current_img = self.current_images[self.current_image_index]
        if current_img['pil_image'] is None:
            return

        # Get mouse position in canvas coordinates
        mouse_canvas_x = self.canvas.canvasx(event.x)
        mouse_canvas_y = self.canvas.canvasy(event.y)

        # Calculate mouse position in original image coordinates (0-1 range)
        img_width, img_height = current_img['pil_image'].size
        old_display_width = img_width * self.zoom_level
        old_display_height = img_height * self.zoom_level

        # Relative position of mouse in the image (0-1)
        rel_x = mouse_canvas_x / old_display_width if old_display_width > 0 else 0.5
        rel_y = mouse_canvas_y / old_display_height if old_display_height > 0 else 0.5

        # Clamp to valid range
        rel_x = max(0, min(1, rel_x))
        rel_y = max(0, min(1, rel_y))

        # Update zoom level
        old_zoom = self.zoom_level
        self.zoom_level = new_zoom

        # Calculate new dimensions
        new_display_width = int(img_width * self.zoom_level)
        new_display_height = int(img_height * self.zoom_level)

        # Create resized image
        if abs(self.zoom_level - 1.0) > 0.001:
            # Use LANCZOS for downsampling (zoom < 1) and BICUBIC for upsampling
            resample_filter = Image.Resampling.LANCZOS if self.zoom_level < 1.0 else Image.Resampling.BICUBIC
            display_image = current_img['pil_image'].resize(
                (new_display_width, new_display_height),
                resample_filter
            )
        else:
            display_image = current_img['pil_image']

        # Update canvas image
        photo = ImageTk.PhotoImage(display_image)
        self.canvas.itemconfig(self.canvas_image, image=photo)
        self.canvas.image = photo

        # Update scrollregion
        self.canvas.configure(scrollregion=(0, 0, new_display_width, new_display_height))

        # Calculate new scroll position to keep mouse point in same place
        new_mouse_x = rel_x * new_display_width
        new_mouse_y = rel_y * new_display_height

        # Calculate required scroll to keep mouse position stable
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # New scroll positions
        new_scroll_x = new_mouse_x - event.x
        new_scroll_y = new_mouse_y - event.y

        # Apply scroll positions if image is larger than canvas
        if new_display_width > canvas_width:
            x_fraction = new_scroll_x / new_display_width
            x_fraction = max(0, min(x_fraction, 1 - canvas_width/new_display_width))
            self.canvas.xview_moveto(x_fraction)

        if new_display_height > canvas_height:
            y_fraction = new_scroll_y / new_display_height
            y_fraction = max(0, min(y_fraction, 1 - canvas_height/new_display_height))
            self.canvas.yview_moveto(y_fraction)
