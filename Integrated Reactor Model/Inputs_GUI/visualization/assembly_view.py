# Assembly view visualization functions

"""
Assembly View Visualization
Handles fuel assembly level visualization
"""
from matplotlib.patches import Circle, Rectangle
import numpy as np

from Inputs_GUI.utils.constants import get_material_color


class AssemblyView:
    def __init__(self, main_gui):
        self.main_gui = main_gui

    def plot(self, ax, view_type):
        """Plot assembly view"""
        inputs = self.main_gui.current_inputs

        if inputs['assembly_type'] == 'Pin':
            self.plot_pin_assembly(ax, view_type, inputs)
        else:
            self.plot_plate_assembly(ax, view_type, inputs)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{inputs['assembly_type']} Assembly - {view_type}")

    def plot_pin_assembly(self, ax, view_type, inputs):
        """Plot pin assembly"""
        n_pins = int(inputs['n_side_pins'])
        pin_pitch = inputs['pin_pitch']
        r_fuel = inputs['r_fuel']
        r_clad_inner = inputs['r_clad_inner']
        r_clad_outer = inputs['r_clad_outer']

        # Convert to cm
        pin_pitch_cm = pin_pitch * 100
        r_fuel_cm = r_fuel * 100
        r_clad_inner_cm = r_clad_inner * 100
        r_clad_outer_cm = r_clad_outer * 100

        assembly_size = n_pins * pin_pitch_cm
        guide_tube_positions = inputs.get('guide_tube_positions', [])

        if "XY" in view_type:
            # Background coolant
            coolant_rect = Rectangle((-assembly_size/2, -assembly_size/2),
                                   assembly_size, assembly_size,
                                   facecolor=get_material_color("Coolant"),
                                   alpha=0.7, edgecolor='black', linewidth=1)
            ax.add_patch(coolant_rect)

            # Plot pins
            for i in range(n_pins):
                for j in range(n_pins):
                    x = (i - (n_pins-1)/2) * pin_pitch_cm
                    y = (j - (n_pins-1)/2) * pin_pitch_cm

                    is_guide_tube = (i, j) in guide_tube_positions

                    if is_guide_tube:
                        # Guide tube
                        guide_tube = Circle((x, y), r_clad_outer_cm,
                                          facecolor='#696969', edgecolor='black',
                                          linewidth=0.5, alpha=0.9)
                        ax.add_patch(guide_tube)
                    else:
                        # Cladding
                        clad = Circle((x, y), r_clad_outer_cm,
                                    facecolor=get_material_color(inputs['clad_type']),
                                    edgecolor='black', linewidth=0.5, alpha=0.9)
                        ax.add_patch(clad)

                        # Gap
                        gap = Circle((x, y), r_clad_inner_cm,
                                   facecolor='white', edgecolor='none', alpha=0.9)
                        ax.add_patch(gap)

                        # Fuel
                        fuel = Circle((x, y), r_fuel_cm,
                                    facecolor=get_material_color(inputs['fuel_type']),
                                    edgecolor='darkred', linewidth=0.5, alpha=0.95)
                        ax.add_patch(fuel)

            # Set limits
            plot_margin = assembly_size * 0.15
            ax.set_xlim(-assembly_size/2 - plot_margin, assembly_size/2 + plot_margin)
            ax.set_ylim(-assembly_size/2 - plot_margin, assembly_size/2 + plot_margin)
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Y (cm)')

        else:  # Side view
            fuel_height_cm = inputs['fuel_height'] * 100

            # Draw pins in side view
            for i in range(n_pins):
                x = (i - (n_pins-1)/2) * pin_pitch_cm

                # Check if guide tube (use center pin for side view)
                is_guide_tube = (i, n_pins//2) in guide_tube_positions

                if is_guide_tube:
                    # Guide tube
                    guide_rect = Rectangle((x-r_clad_outer_cm, 0),
                                         2*r_clad_outer_cm, fuel_height_cm,
                                         facecolor='#696969', edgecolor='black',
                                         linewidth=0.3, alpha=0.9)
                    ax.add_patch(guide_rect)
                else:
                    # Cladding
                    clad_rect = Rectangle((x-r_clad_outer_cm, 0),
                                        2*r_clad_outer_cm, fuel_height_cm,
                                        facecolor=get_material_color(inputs['clad_type']),
                                        edgecolor='black', linewidth=0.3, alpha=0.9)
                    ax.add_patch(clad_rect)

                    # Gap
                    gap_rect = Rectangle((x-r_clad_inner_cm, 0),
                                       2*r_clad_inner_cm, fuel_height_cm,
                                       facecolor='white', edgecolor='gray',
                                       linewidth=0.3, alpha=0.9)
                    ax.add_patch(gap_rect)

                    # Fuel
                    fuel_rect = Rectangle((x-r_fuel_cm, 0),
                                        2*r_fuel_cm, fuel_height_cm,
                                        facecolor=get_material_color(inputs['fuel_type']),
                                        edgecolor='darkred', linewidth=0.3, alpha=0.95)
                    ax.add_patch(fuel_rect)

            # Set limits
            x_margin = assembly_size * 0.15
            y_margin = fuel_height_cm * 0.1
            ax.set_xlim(-assembly_size/2 - x_margin, assembly_size/2 + x_margin)
            ax.set_ylim(-y_margin, fuel_height_cm + y_margin)
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Z (cm)')

    def plot_plate_assembly(self, ax, view_type, inputs):
        """Plot plate assembly"""
        n_plates = int(inputs['plates_per_assembly'])
        plate_pitch = inputs['fuel_plate_pitch']
        plate_width = inputs['fuel_plate_width']
        meat_width = inputs['fuel_meat_width']
        meat_thickness = inputs['fuel_meat_thickness']
        clad_thickness = inputs['clad_thickness']
        clad_structure_width = inputs['clad_structure_width']

        # Convert to cm
        plate_pitch_cm = plate_pitch * 100
        plate_width_cm = plate_width * 100
        meat_width_cm = meat_width * 100
        meat_thickness_cm = meat_thickness * 100
        clad_thickness_cm = clad_thickness * 100
        clad_structure_width_cm = clad_structure_width * 100

        # Calculate dimensions
        assembly_width_cm = plate_width_cm + 2 * clad_structure_width_cm
        assembly_height_cm = n_plates * plate_pitch_cm + 2 * clad_structure_width_cm

        if "XY" in view_type:
            # Assembly structure
            assembly_rect = Rectangle((-assembly_width_cm/2, -assembly_height_cm/2),
                                    assembly_width_cm, assembly_height_cm,
                                    facecolor=get_material_color(inputs['clad_type']),
                                    alpha=0.7, edgecolor='black', linewidth=1)
            ax.add_patch(assembly_rect)

            # Coolant region
            fuel_region_rect = Rectangle((-plate_width_cm/2,
                                        -assembly_height_cm/2 + clad_structure_width_cm),
                                       plate_width_cm, n_plates * plate_pitch_cm,
                                       facecolor=get_material_color(inputs['coolant_type']),
                                       alpha=0.6, edgecolor='black', linewidth=0.5)
            ax.add_patch(fuel_region_rect)

            # Draw plates
            for i in range(n_plates):
                plate_y = (-assembly_height_cm/2 + clad_structure_width_cm +
                          (i + 0.5) * plate_pitch_cm)

                total_thickness = meat_thickness_cm + 2 * clad_thickness_cm

                # Plate cladding structure
                # Left clad
                left_clad = Rectangle((-plate_width_cm/2, plate_y - total_thickness/2),
                                    (plate_width_cm - meat_width_cm)/2, total_thickness,
                                    facecolor=get_material_color(inputs['clad_type']),
                                    edgecolor='black', linewidth=0.2, alpha=0.9)
                ax.add_patch(left_clad)

                # Right clad
                right_clad = Rectangle((meat_width_cm/2, plate_y - total_thickness/2),
                                     (plate_width_cm - meat_width_cm)/2, total_thickness,
                                     facecolor=get_material_color(inputs['clad_type']),
                                     edgecolor='black', linewidth=0.2, alpha=0.9)
                ax.add_patch(right_clad)

                # Bottom clad
                bottom_clad = Rectangle((-meat_width_cm/2, plate_y - total_thickness/2),
                                      meat_width_cm, clad_thickness_cm,
                                      facecolor=get_material_color(inputs['clad_type']),
                                      edgecolor='black', linewidth=0.2, alpha=0.9)
                ax.add_patch(bottom_clad)

                # Top clad
                top_clad = Rectangle((-meat_width_cm/2,
                                    plate_y + total_thickness/2 - clad_thickness_cm),
                                   meat_width_cm, clad_thickness_cm,
                                   facecolor=get_material_color(inputs['clad_type']),
                                   edgecolor='black', linewidth=0.2, alpha=0.9)
                ax.add_patch(top_clad)

                # Fuel meat
                meat_rect = Rectangle((-meat_width_cm/2, plate_y - meat_thickness_cm/2),
                                    meat_width_cm, meat_thickness_cm,
                                    facecolor=get_material_color(inputs['fuel_type']),
                                    edgecolor='none', alpha=0.95)
                ax.add_patch(meat_rect)

            # Set limits
            plot_margin = max(assembly_width_cm, assembly_height_cm) * 0.15
            ax.set_xlim(-assembly_width_cm/2 - plot_margin,
                       assembly_width_cm/2 + plot_margin)
            ax.set_ylim(-assembly_height_cm/2 - plot_margin,
                       assembly_height_cm/2 + plot_margin)
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Y (cm)')

        else:  # Side view
            fuel_height_cm = inputs['fuel_height'] * 100

            # Assembly structure background
            structure_rect = Rectangle((-assembly_width_cm/2, 0),
                                     assembly_width_cm, fuel_height_cm,
                                     facecolor=get_material_color(inputs['clad_type']),
                                     alpha=0.4, edgecolor='black', linewidth=1)
            ax.add_patch(structure_rect)

            # Fuel region
            fuel_region_rect = Rectangle((-plate_width_cm/2, 0),
                                       plate_width_cm, fuel_height_cm,
                                       facecolor=get_material_color(inputs['coolant_type']),
                                       alpha=0.6, edgecolor='black', linewidth=0.5)
            ax.add_patch(fuel_region_rect)

            # Draw plates
            plate_thickness = meat_thickness_cm + 2*clad_thickness_cm
            total_plates_thickness = n_plates * plate_thickness
            available_space = plate_width_cm - total_plates_thickness

            if n_plates > 1:
                gap_between_plates = available_space / (n_plates + 1)
            else:
                gap_between_plates = available_space / 2

            for i in range(n_plates):
                plate_x = (-plate_width_cm/2 + gap_between_plates +
                          i * (plate_thickness + gap_between_plates) + plate_thickness/2)

                # Plate cladding
                plate_rect = Rectangle((plate_x - plate_thickness/2, 0),
                                     plate_thickness, fuel_height_cm,
                                     facecolor=get_material_color(inputs['clad_type']),
                                     edgecolor='black', linewidth=0.3, alpha=0.9)
                ax.add_patch(plate_rect)

                # Fuel meat
                meat_rect = Rectangle((plate_x - meat_thickness_cm/2, 0),
                                    meat_thickness_cm, fuel_height_cm,
                                    facecolor=get_material_color(inputs['fuel_type']),
                                    edgecolor='none', alpha=0.95)
                ax.add_patch(meat_rect)

            # Set limits
            x_margin = assembly_width_cm * 0.2
            y_margin = fuel_height_cm * 0.1
            ax.set_xlim(-assembly_width_cm/2 - x_margin,
                       assembly_width_cm/2 + x_margin)
            ax.set_ylim(-y_margin, fuel_height_cm + y_margin)
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Z (cm)')
