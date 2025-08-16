# Core view visualization functions

"""
Core View Visualization
Handles reactor core level visualization
"""
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection

from Inputs_GUI.utils.constants import get_material_color, get_irradiation_material_type
from Inputs_GUI.visualization.plotting_utils import (
    draw_pin_assembly_detailed, draw_plate_assembly_detailed,
    draw_core_pins_side_view, draw_core_plates_side_view
)


class CoreView:
    def __init__(self, main_gui):
        self.main_gui = main_gui

    def plot(self, ax, view_type):
        """Plot core view"""
        inputs = self.main_gui.current_inputs

        if "XY" in view_type:
            self.plot_core_xy(ax, inputs)
        elif "YZ" in view_type or "XZ" in view_type:
            self.plot_core_side(ax, inputs, view_type)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Detailed Reactor Core - {view_type}")

    def plot_core_xy(self, ax, inputs):
        """Plot core XY view"""
        lattice = inputs['core_lattice']
        n_rows, n_cols = len(lattice), len(lattice[0])

        # Calculate assembly pitch
        if inputs['assembly_type'] == 'Pin':
            assembly_pitch = inputs['pin_pitch'] * inputs['n_side_pins']
        else:
            assembly_pitch = (inputs['plates_per_assembly'] *
                            inputs['fuel_plate_pitch'] +
                            2 * inputs['clad_structure_width'])

        # Get reactor dimensions
        tank_radius = inputs['tank_radius']
        reflector_thickness = inputs['reflector_thickness']
        bioshield_thickness = inputs['bioshield_thickness']

        # Calculate extents
        core_lattice_extent = max(n_cols * assembly_pitch / 2,
                                 n_rows * assembly_pitch / 2)

        # Draw reactor regions from outside to inside
        # 1. Bioshield
        bioshield_radius = tank_radius + reflector_thickness + bioshield_thickness
        bioshield = Circle((0, 0), bioshield_radius,
                          facecolor=get_material_color(inputs['bioshield_material']),
                          alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(bioshield)

        # 2. Reflector
        reflector_radius = tank_radius + reflector_thickness
        reflector = Circle((0, 0), reflector_radius,
                          facecolor=get_material_color(inputs['reflector_material']),
                          alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(reflector)

        # 3. Tank
        tank = Circle((0, 0), tank_radius,
                     facecolor=get_material_color(inputs['coolant_type']),
                     alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(tank)

        # 4. Plot core lattice - properly centered for odd/even layouts
        # For proper centering, the core should be centered at (0,0)
        core_width = n_cols * assembly_pitch
        core_height = n_rows * assembly_pitch

        for i, row in enumerate(lattice):
            for j, cell in enumerate(row):
                if cell == 'C':
                    continue

                # Calculate assembly center - properly centered around (0,0)
                # Flip Y axis to match design grid orientation (i=0 at top)
                assy_x = (j - (n_cols - 1) / 2) * assembly_pitch
                assy_y = ((n_rows - 1 - i) - (n_rows - 1) / 2) * assembly_pitch

                # Handle different cell types
                if cell == 'F':
                    fuel_color = get_material_color(inputs['fuel_type'])
                    self.draw_fuel_assembly(ax, assy_x, assy_y, fuel_color,
                                          assembly_pitch, inputs)
                elif cell == 'E':
                    fuel_color = '#8B0000'  # Enhanced fuel
                    self.draw_fuel_assembly(ax, assy_x, assy_y, fuel_color,
                                          assembly_pitch, inputs)
                elif 'I_' in cell:
                    self.draw_irradiation_position(ax, assy_x, assy_y,
                                                 assembly_pitch, inputs, cell)

        # Set plot extent
        plot_extent = max(bioshield_radius * 1.1, core_lattice_extent * 1.2)
        ax.set_xlim(-plot_extent, plot_extent)
        ax.set_ylim(-plot_extent, plot_extent)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

    def plot_core_side(self, ax, inputs, view_type):
        """Plot core side view (YZ or XZ)"""
        # Get dimensions
        fuel_height = inputs['fuel_height']
        plenum_height = inputs['plenum_height']
        top_refl_thickness = inputs['top_reflector_thickness']
        top_bio_thickness = inputs['top_bioshield_thickness']
        feed_thickness = inputs['feed_thickness']
        bottom_refl_thickness = inputs['bottom_reflector_thickness']
        bottom_bio_thickness = inputs['bottom_bioshield_thickness']

        tank_radius = inputs['tank_radius']
        reflector_thickness = inputs['reflector_thickness']
        bioshield_thickness = inputs['bioshield_thickness']

        # Calculate axial positions
        half_fuel = fuel_height / 2
        z_bottom_bio = -half_fuel - feed_thickness - bottom_refl_thickness - bottom_bio_thickness
        z_bottom_refl = -half_fuel - feed_thickness - bottom_refl_thickness
        z_feed = -half_fuel - feed_thickness
        z_bottom_fuel = -half_fuel
        z_top_fuel = half_fuel
        z_plenum = half_fuel + plenum_height
        z_top_refl = half_fuel + plenum_height + top_refl_thickness
        z_top_bio = half_fuel + plenum_height + top_refl_thickness + top_bio_thickness

        # Radial boundaries
        bioshield_radius = tank_radius + reflector_thickness + bioshield_thickness
        reflector_radius = tank_radius + reflector_thickness

        # Draw regions
        self.draw_axial_regions(ax, z_bottom_bio, z_bottom_refl, z_feed,
                              z_bottom_fuel, z_top_fuel, z_plenum, z_top_refl,
                              z_top_bio, tank_radius, reflector_radius,
                              bioshield_radius, inputs)

        # Draw fuel elements - differentiate XZ vs YZ views
        if inputs['assembly_type'] == 'Pin':
            assembly_pitch = inputs['pin_pitch'] * inputs['n_side_pins']
            draw_core_pins_side_view(ax, tank_radius, z_bottom_fuel,
                                   fuel_height, assembly_pitch, inputs, view_type)
        else:
            assembly_pitch = (inputs['plates_per_assembly'] *
                            inputs['fuel_plate_pitch'] +
                            2 * inputs['clad_structure_width'])
            draw_core_plates_side_view(ax, tank_radius, z_bottom_fuel,
                                     fuel_height, assembly_pitch, inputs, view_type)

        # Set limits
        total_height = z_top_bio - z_bottom_bio
        ax.set_xlim(-bioshield_radius*1.1, bioshield_radius*1.1)
        ax.set_ylim(z_bottom_bio - total_height*0.1, z_top_bio + total_height*0.1)
        ax.set_xlabel('X (m)' if "XZ" in view_type else 'Y (m)')
        ax.set_ylabel('Z (m)')

    def draw_fuel_assembly(self, ax, x, y, fuel_color, assembly_pitch, inputs):
        """Draw a fuel assembly"""
        if inputs['assembly_type'] == 'Pin':
            draw_pin_assembly_detailed(ax, x, y, fuel_color, assembly_pitch, inputs)
        else:
            draw_plate_assembly_detailed(ax, x, y, fuel_color, assembly_pitch, inputs)

    def draw_irradiation_position(self, ax, x, y, assembly_pitch, inputs, label):
        """Draw an irradiation position"""
        if inputs['irradiation_clad']:
            # Determine cladding thickness
            if inputs['assembly_type'] == 'Plate':
                cladding_thickness = inputs['clad_structure_width']
            else:
                cladding_thickness = inputs['irradiation_clad_thickness']

            # Outer cladding
            clad_rect = Rectangle((x-assembly_pitch/2, y-assembly_pitch/2),
                                assembly_pitch, assembly_pitch,
                                facecolor=get_material_color(inputs['clad_type']),
                                alpha=0.9, edgecolor='black', linewidth=1.0)
            ax.add_patch(clad_rect)

            # Inner space
            inner_size = assembly_pitch - 2 * cladding_thickness
            inner_rect = Rectangle((x-inner_size/2, y-inner_size/2),
                                 inner_size, inner_size,
                                 facecolor=get_material_color(inputs['irradiation_cell_fill']),
                                 alpha=0.9, edgecolor='black', linewidth=1.0)
            ax.add_patch(inner_rect)
        else:
            # No cladding
            irrad_rect = Rectangle((x-assembly_pitch/2, y-assembly_pitch/2),
                                 assembly_pitch, assembly_pitch,
                                 facecolor=get_material_color(inputs['irradiation_cell_fill']),
                                 alpha=0.9, edgecolor='black', linewidth=1.0)
            ax.add_patch(irrad_rect)

        # Add label
        ax.text(x, y, label, ha='center', va='center',
               fontsize=6, fontweight='bold', color='black')

    def draw_axial_regions(self, ax, z_bottom_bio, z_bottom_refl, z_feed,
                          z_bottom_fuel, z_top_fuel, z_plenum, z_top_refl,
                          z_top_bio, tank_radius, reflector_radius,
                          bioshield_radius, inputs):
        """Draw axial regions for side view"""
        # Bottom bioshield
        bio_bottom = Rectangle((-bioshield_radius, z_bottom_bio),
                             2*bioshield_radius, z_bottom_refl - z_bottom_bio,
                             facecolor=get_material_color(inputs['bioshield_material']),
                             alpha=0.8, edgecolor='black')
        ax.add_patch(bio_bottom)

        # Top bioshield
        bio_top = Rectangle((-bioshield_radius, z_top_refl),
                          2*bioshield_radius, z_top_bio - z_top_refl,
                          facecolor=get_material_color(inputs['bioshield_material']),
                          alpha=0.8, edgecolor='black')
        ax.add_patch(bio_top)

        # Radial bioshield
        bio_radial = Rectangle((-bioshield_radius, z_bottom_refl),
                             2*bioshield_radius, z_top_refl - z_bottom_refl,
                             facecolor=get_material_color(inputs['bioshield_material']),
                             alpha=0.8, edgecolor='black')
        ax.add_patch(bio_radial)

        # Bottom reflector
        refl_bottom = Rectangle((-tank_radius, z_bottom_refl),
                              2*tank_radius, z_feed - z_bottom_refl,
                              facecolor=get_material_color(inputs['reflector_material']),
                              alpha=0.8, edgecolor='black')
        ax.add_patch(refl_bottom)

        # Top reflector
        refl_top = Rectangle((-tank_radius, z_plenum),
                           2*tank_radius, z_top_refl - z_plenum,
                           facecolor=get_material_color(inputs['reflector_material']),
                           alpha=0.8, edgecolor='black')
        ax.add_patch(refl_top)

        # Radial reflector
        refl_radial = Rectangle((-reflector_radius, z_bottom_refl),
                              2*reflector_radius, z_top_refl - z_bottom_refl,
                              facecolor=get_material_color(inputs['reflector_material']),
                              alpha=0.8, edgecolor='black')
        ax.add_patch(refl_radial)

        # Feed region
        if inputs['feed_thickness'] > 0:
            feed_rect = Rectangle((-tank_radius, z_feed),
                                2*tank_radius, z_bottom_fuel - z_feed,
                                facecolor=get_material_color("Feed"),
                                alpha=0.8, edgecolor='black')
            ax.add_patch(feed_rect)

        # Plenum
        # Plenum
        plenum_rect = Rectangle((-tank_radius, z_top_fuel),
                              2*tank_radius, z_plenum - z_top_fuel,
                              facecolor=get_material_color("Plenum"),
                              alpha=0.8, edgecolor='black')
        ax.add_patch(plenum_rect)

        # Core coolant background
        core_coolant = Rectangle((-tank_radius, z_bottom_fuel),
                               2*tank_radius, z_top_fuel - z_bottom_fuel,
                               facecolor=get_material_color(inputs['coolant_type']),
                               alpha=0.6, edgecolor='black')
        ax.add_patch(core_coolant)
