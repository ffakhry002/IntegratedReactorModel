"""
Geometry Tab Component
Handles OpenMC geometry visualization and generation
"""
import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import tempfile
import threading
from PIL import Image, ImageTk
import glob

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Reactor.geometry import plot_geometry


class GeometryTab:
    def __init__(self, parent, main_gui):
        self.parent = parent
        self.main_gui = main_gui

        # State variables
        self.current_images = []
        self.current_image_index = 0
        self.image_label = None
        self.image_name_label = None
        self.geometry_output_dir = None

    def setup(self):
        """Setup the geometry tab"""
        # Main container
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control panel at top
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Generate geometry button
        self.generate_button = ttk.Button(
            control_frame,
            text="Generate OpenMC Geometry",
            command=self.generate_geometry,
            style='Accent.TButton'
        )
        self.generate_button.pack(side=tk.LEFT, padx=(0, 10))

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            control_frame,
            mode='indeterminate',
            length=200
        )
        self.progress_bar.pack(side=tk.LEFT, padx=(0, 10))

        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready to generate geometry")
        self.status_label.pack(side=tk.LEFT)

        # Image viewer frame
        viewer_frame = ttk.LabelFrame(main_frame, text="Geometry Visualization", padding=10)
        viewer_frame.pack(fill=tk.BOTH, expand=True)

        # Setup image viewer
        self.setup_image_viewer(viewer_frame)

        # Image info label at bottom
        info_label = ttk.Label(main_frame, text="Use mouse wheel to zoom, click and drag to pan",
                             font=('TkDefaultFont', 9), foreground='gray')
        info_label.pack(side=tk.BOTTOM, pady=(5, 0))

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

        # Bind canvas resize event
        self.canvas.bind('<Configure>', self._on_canvas_configure)

        # Bind mouse events for pan and zoom
        self.canvas.bind("<ButtonPress-1>", self.on_pan_start)
        self.canvas.bind("<B1-Motion>", self.on_pan_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_pan_end)
        self.canvas.bind("<MouseWheel>", self.on_zoom)  # Windows/Linux
        self.canvas.bind("<Button-4>", self.on_zoom)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_zoom)  # Linux scroll down

        # Pan state
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.is_panning = False

        # Zoom state
        self.zoom_level = 1.0
        self.min_zoom = 0.1  # Allow zooming out to 10% for overview
        self.max_zoom = 10.0  # Allow zooming in to 10x for detail

        # Initial placeholder message
        self.canvas.create_text(
            400, 300,
            text="Click 'Generate OpenMC Geometry' to create geometry plots",
            font=('Arial', 14), fill='gray', tags="placeholder"
        )

    def _on_canvas_configure(self, event):
        """Handle canvas resize events"""
        # If we have an image displayed and canvas size changed significantly
        if hasattr(self, 'current_images') and self.current_images:
            # You could add logic here to re-center image on window resize if desired
            pass

    def generate_geometry(self):
        """Generate OpenMC geometry plots in a separate thread"""
        # Disable generate button during generation
        self.generate_button.config(state=tk.DISABLED)
        self.progress_bar.start()
        self.status_label.config(text="Generating geometry plots...")

        # Run generation in separate thread
        thread = threading.Thread(target=self._generate_geometry_thread)
        thread.daemon = True
        thread.start()

    def _generate_geometry_thread(self):
        """Thread function to generate geometry"""
        try:
            # Set matplotlib to use non-interactive backend for thread safety
            import matplotlib
            matplotlib.use('Agg')  # Use Anti-Grain Geometry backend (non-interactive)

            # Create temporary directory for outputs
            self.geometry_output_dir = tempfile.mkdtemp(prefix="openmc_geometry_")

            # Call plot_geometry with current inputs
            plot_geometry(
                output_dir=self.geometry_output_dir,
                inputs_dict=self.main_gui.current_inputs
            )

            # Schedule UI update in main thread
            self.parent.after(0, self._load_generated_images)

        except Exception as e:
            error_msg = f"Error generating geometry: {str(e)}"
            self.parent.after(0, lambda: self._show_error(error_msg))

    def _load_generated_images(self):
        """Load the generated images"""
        try:
            # Find all PNG files in output directory
            image_files = sorted(glob.glob(os.path.join(self.geometry_output_dir, "*.png")))

            if not image_files:
                self._show_error("No geometry images were generated")
                return

            # Load images
            self.current_images = []
            for img_path in image_files:
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

            # Update status
            self.status_label.config(text=f"Generated {len(self.current_images)} geometry plots")

        except Exception as e:
            self._show_error(f"Error loading images: {str(e)}")

        finally:
            # Re-enable generate button and stop progress
            self.generate_button.config(state=tk.NORMAL)
            self.progress_bar.stop()

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

    def _show_error(self, message):
        """Show error message"""
        self.status_label.config(text="Error occurred")
        self.generate_button.config(state=tk.NORMAL)
        self.progress_bar.stop()
        messagebox.showerror("Geometry Generation Error", message)

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
