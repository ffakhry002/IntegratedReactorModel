# Element view visualization functions

"""
Element View Visualization
Handles individual fuel element visualization
"""
from matplotlib.patches import Circle, Rectangle

from Inputs_GUI.utils.constants import get_material_color


class ElementView:
    def __init__(self, main_gui):
        self.main_gui = main_gui

    def plot(self, ax, view_type):
        """Plot element view"""
        inputs = self.main_gui.current_inputs

        if inputs['assembly_type'] == 'Pin':
            self.plot_single_pin(ax, view_type, inputs)
        else:
            self.plot_single_plate(ax, view_type, inputs)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Single {inputs['assembly_type']} - {view_type}")

    def plot_single_pin(self, ax, view_type, inputs):
        """Plot single pin"""
        r_fuel = inputs['r_fuel'] * 100  # Convert to cm
        r_clad_inner = inputs['r_clad_inner'] * 100
        r_clad_outer = inputs['r_clad_outer'] * 100
        pin_pitch = inputs['pin_pitch'] * 100
        fuel_height = inputs['fuel_height'] * 100

        if "XY" in view_type:
            # Coolant region
            coolant = Rectangle((-pin_pitch/2, -pin_pitch/2), pin_pitch, pin_pitch,
                              facecolor=get_material_color("Coolant"),
                              alpha=0.8, edgecolor='black', linewidth=1)
            ax.add_patch(coolant)

            # Cladding
            clad = Circle((0, 0), r_clad_outer,
                        facecolor=get_material_color(inputs['clad_type']),
                        edgecolor='black', linewidth=1, alpha=0.9)
            ax.add_patch(clad)

            # Gap
            gap = Circle((0, 0), r_clad_inner,
                       facecolor='white', edgecolor='gray', linewidth=0.5, alpha=0.9)
            ax.add_patch(gap)

            # Fuel
            fuel = Circle((0, 0), r_fuel,
                        facecolor=get_material_color(inputs['fuel_type']),
                        edgecolor='darkred', linewidth=0.5, alpha=0.95)
            ax.add_patch(fuel)

            # Set limits (increased for better visibility)
            plot_limit = pin_pitch * 1.2  # Increased from 0.8 to 1.2 for more space
            ax.set_xlim(-plot_limit, plot_limit)
            ax.set_ylim(-plot_limit, plot_limit)
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Y (cm)')

        else:  # Side view
            # Coolant
            coolant_rect = Rectangle((-pin_pitch/2, 0), pin_pitch, fuel_height,
                                   facecolor=get_material_color("Coolant"),
                                   alpha=0.8, edgecolor='black', linewidth=1)
            ax.add_patch(coolant_rect)

            # Cladding
            clad_rect = Rectangle((-r_clad_outer, 0), 2*r_clad_outer, fuel_height,
                                facecolor=get_material_color(inputs['clad_type']),
                                edgecolor='black', linewidth=0.5, alpha=0.9)
            ax.add_patch(clad_rect)

            # Gap
            gap_rect = Rectangle((-r_clad_inner, 0), 2*r_clad_inner, fuel_height,
                               facecolor='white', edgecolor='gray',
                               linewidth=0.3, alpha=0.9)
            ax.add_patch(gap_rect)

            # Fuel
            fuel_rect = Rectangle((-r_fuel, 0), 2*r_fuel, fuel_height,
                                facecolor=get_material_color(inputs['fuel_type']),
                                edgecolor='darkred', linewidth=0.3, alpha=0.95)
            ax.add_patch(fuel_rect)

            # Set limits (increased x_limit for better visibility)
            x_limit = pin_pitch * 1.5  # Increased from 0.8 to 1.5 for more space
            y_margin = fuel_height * 0.15
            ax.set_xlim(-x_limit, x_limit)
            ax.set_ylim(-y_margin, fuel_height + y_margin)
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Z (cm)')

    def plot_single_plate(self, ax, view_type, inputs):
        """Plot single plate"""
        # Get dimensions in cm
        fuel_meat_thickness = inputs['fuel_meat_thickness'] * 100
        fuel_plate_width = inputs['fuel_plate_width'] * 100
        fuel_plate_pitch = inputs['fuel_plate_pitch'] * 100
        clad_thickness = inputs['clad_thickness'] * 100
        fuel_meat_width = inputs['fuel_meat_width'] * 100
        fuel_height = inputs['fuel_height'] * 100

        # Calculate dimensions
        total_thickness = fuel_meat_thickness + 2 * clad_thickness
        coolant_channel = fuel_plate_pitch - total_thickness

        if "XY" in view_type:
            # Define boundaries
            x0 = -fuel_plate_width/2
            x1 = -fuel_meat_width/2
            x2 = fuel_meat_width/2
            x3 = fuel_plate_width/2
            y0 = -total_thickness/2
            y1 = y0 + clad_thickness
            y2 = y1 + fuel_meat_thickness
            y3 = y2 + clad_thickness
            y_top = y3 + coolant_channel/2
            y_bottom = y0 - coolant_channel/2

            # Coolant channels
            top_coolant = Rectangle((x0, y3), fuel_plate_width, coolant_channel/2,
                                  facecolor=get_material_color(inputs['coolant_type']),
                                  alpha=0.8, edgecolor='black', linewidth=1)
            ax.add_patch(top_coolant)

            bottom_coolant = Rectangle((x0, y_bottom), fuel_plate_width, coolant_channel/2,
                                     facecolor=get_material_color(inputs['coolant_type']),
                                     alpha=0.8, edgecolor='black', linewidth=1)
            ax.add_patch(bottom_coolant)

            # Cladding regions
            # Left clad
            left_clad = Rectangle((x0, y0), x1 - x0, y3 - y0,
                                facecolor=get_material_color(inputs['clad_type']),
                                edgecolor='black', linewidth=1, alpha=0.9)
            ax.add_patch(left_clad)

            # Right clad
            right_clad = Rectangle((x2, y0), x3 - x2, y3 - y0,
                                 facecolor=get_material_color(inputs['clad_type']),
                                 edgecolor='black', linewidth=1, alpha=0.9)
            ax.add_patch(right_clad)

            # Bottom clad
            bottom_clad = Rectangle((x1, y0), x2 - x1, y1 - y0,
                                  facecolor=get_material_color(inputs['clad_type']),
                                  edgecolor='black', linewidth=1, alpha=0.9)
            ax.add_patch(bottom_clad)

            # Top clad
            top_clad = Rectangle((x1, y2), x2 - x1, y3 - y2,
                               facecolor=get_material_color(inputs['clad_type']),
                               edgecolor='black', linewidth=1, alpha=0.9)
            ax.add_patch(top_clad)

            # Fuel meat
            fuel_meat = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                facecolor=get_material_color(inputs['fuel_type']),
                                edgecolor='darkred', linewidth=0.5, alpha=0.95)
            ax.add_patch(fuel_meat)

            # Set limits (increased for better visibility)
            plot_limit_x = max(fuel_plate_width, fuel_meat_width) * 1.2  # Increased from 0.8 to 1.2
            plot_limit_y = fuel_plate_pitch * 1.2  # Increased from 0.8 to 1.2 for more space
            ax.set_xlim(-plot_limit_x, plot_limit_x)
            ax.set_ylim(-plot_limit_y, plot_limit_y)
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Y (cm)')

        else:  # Side view
            # Coolant region
            coolant_rect = Rectangle((-fuel_plate_pitch/2, 0),
                                   fuel_plate_pitch, fuel_height,
                                   facecolor=get_material_color(inputs['coolant_type']),
                                   alpha=0.8, edgecolor='black', linewidth=1)
            ax.add_patch(coolant_rect)

            # Plate cladding
            plate_rect = Rectangle((-total_thickness/2, 0),
                                 total_thickness, fuel_height,
                                 facecolor=get_material_color(inputs['clad_type']),
                                 edgecolor='black', linewidth=0.5, alpha=0.9)
            ax.add_patch(plate_rect)

            # Fuel meat
            meat_rect = Rectangle((-fuel_meat_thickness/2, 0),
                                fuel_meat_thickness, fuel_height,
                                facecolor=get_material_color(inputs['fuel_type']),
                                edgecolor='darkred', linewidth=0.3, alpha=0.95)
            ax.add_patch(meat_rect)

            # Set limits (increased x_limit for better visibility)
            x_limit = fuel_plate_pitch * 1.5  # Increased from 0.8 to 1.5 for more space
            y_margin = fuel_height * 0.15
            ax.set_xlim(-x_limit, x_limit)
            ax.set_ylim(-y_margin, fuel_height + y_margin)
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Z (cm)')
