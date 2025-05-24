# Plotting utility functions

"""
Plotting Utilities
Common plotting functions used across different views
"""
from matplotlib.patches import Circle, Rectangle

from Inputs_GUI.utils.constants import get_material_color


def draw_pin_assembly_detailed(ax, center_x, center_y, fuel_color, assembly_pitch, inputs):
    """Draw individual pins within an assembly at the specified center"""
    n_pins = int(inputs['n_side_pins'])
    pin_pitch = inputs['pin_pitch']
    r_fuel = inputs['r_fuel']
    r_clad_inner = inputs['r_clad_inner']
    r_clad_outer = inputs['r_clad_outer']

    assy_size = assembly_pitch

    # Draw assembly background (coolant)
    coolant_rect = Rectangle((center_x-assy_size/2, center_y-assy_size/2),
                           assy_size, assy_size,
                           facecolor=get_material_color("Coolant"),
                           alpha=0.6, edgecolor='black', linewidth=0.5)
    ax.add_patch(coolant_rect)

    # Get guide tube positions
    guide_tube_positions = inputs.get('guide_tube_positions', [])

    # Draw individual pins
    for i in range(n_pins):
        for j in range(n_pins):
            pin_x = center_x + (i - (n_pins-1)/2) * pin_pitch
            pin_y = center_y + (j - (n_pins-1)/2) * pin_pitch

            # Check if this position has a guide tube
            is_guide_tube = (i, j) in guide_tube_positions

            if is_guide_tube:
                # Draw guide tube
                guide_tube = Circle((pin_x, pin_y), r_clad_outer,
                                  facecolor='#696969', edgecolor='black',
                                  linewidth=0.3, alpha=0.9)
                ax.add_patch(guide_tube)
            else:
                # Draw fuel pin with high detail
                # Cladding
                clad = Circle((pin_x, pin_y), r_clad_outer,
                            facecolor=get_material_color(inputs['clad_type']),
                            edgecolor='black', linewidth=0.2, alpha=0.9)
                ax.add_patch(clad)

                # Gap (only if significant)
                if r_clad_inner > r_fuel * 1.05:
                    gap = Circle((pin_x, pin_y), r_clad_inner,
                               facecolor='white', edgecolor='none', alpha=0.9)
                    ax.add_patch(gap)

                # Fuel
                fuel = Circle((pin_x, pin_y), r_fuel,
                            facecolor=fuel_color, edgecolor='none', alpha=0.95)
                ax.add_patch(fuel)


def draw_plate_assembly_detailed(ax, center_x, center_y, fuel_color, assembly_pitch, inputs):
    """Draw individual plates within an assembly at the specified center"""
    n_plates = int(inputs['plates_per_assembly'])
    plate_pitch = inputs['fuel_plate_pitch']
    plate_width = inputs['fuel_plate_width']
    meat_width = inputs['fuel_meat_width']
    meat_thickness = inputs['fuel_meat_thickness']
    clad_thickness = inputs['clad_thickness']
    clad_structure_width = inputs['clad_structure_width']

    assembly_height = assembly_pitch
    assembly_width = plate_width + 2 * clad_structure_width

    # Assembly structure background
    assy_rect = Rectangle((center_x-assembly_width/2, center_y-assembly_height/2),
                        assembly_width, assembly_height,
                        facecolor=get_material_color(inputs['clad_type']),
                        alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.add_patch(assy_rect)

    # Draw individual plates with high detail
    for i in range(n_plates):
        plate_y = center_y + (i - (n_plates-1)/2) * plate_pitch

        # Coolant channel
        coolant_rect = Rectangle((center_x-plate_width/2, plate_y-plate_pitch/2),
                               plate_width, plate_pitch,
                               facecolor=get_material_color("Coolant"),
                               alpha=0.7, edgecolor='none')
        ax.add_patch(coolant_rect)

        # Plate cladding
        plate_thickness = meat_thickness + 2 * clad_thickness
        plate_rect = Rectangle((center_x-plate_width/2, plate_y-plate_thickness/2),
                             plate_width, plate_thickness,
                             facecolor=get_material_color(inputs['clad_type']),
                             edgecolor='black', linewidth=0.1, alpha=0.9)
        ax.add_patch(plate_rect)

        # Fuel meat
        meat_rect = Rectangle((center_x-meat_width/2, plate_y-meat_thickness/2),
                            meat_width, meat_thickness,
                            facecolor=fuel_color, edgecolor='none', alpha=0.95)
        ax.add_patch(meat_rect)


def draw_core_pins_side_view(ax, tank_radius, fuel_start, fuel_height, assembly_pitch, inputs, view_type="YZ"):
    """Draw detailed pin structure in side view"""
    pin_pitch = inputs['pin_pitch']
    r_fuel = inputs['r_fuel']
    r_clad_outer = inputs['r_clad_outer']

    # Get actual core lattice dimensions
    lattice = inputs['core_lattice']
    n_rows, n_cols = len(lattice), len(lattice[0])

    # Choose which slice of the core to show based on view type
    if "XZ" in view_type:
        # XZ view: slice along Y direction, show middle row across X
        slice_idx = len(lattice) // 2  # Middle row
        n_assemblies = len(lattice[0])  # Number of columns (X direction)
        core_lattice_width = n_assemblies * assembly_pitch
        assembly_positions = []
        for j in range(n_assemblies):
            assembly_x = (j - (n_assemblies - 1) / 2) * assembly_pitch
            cell_type = lattice[slice_idx][j]  # Middle row, column j
            assembly_positions.append((assembly_x, cell_type))
    else:
        # YZ view: slice along X direction, show middle column across Y
        slice_idx = len(lattice[0]) // 2  # Middle column
        n_assemblies = len(lattice)  # Number of rows (Y direction)
        core_lattice_width = n_assemblies * assembly_pitch
        assembly_positions = []
        for i in range(n_assemblies):
            assembly_x = (i - (n_assemblies - 1) / 2) * assembly_pitch
            cell_type = lattice[i][slice_idx]  # Row i, middle column
            assembly_positions.append((assembly_x, cell_type))

    # Draw assemblies
    for assembly_x, cell_type in assembly_positions:

        # Check if this is an irradiation position
        if 'I_' in cell_type:
            # Draw irradiation position in side view
            if inputs['irradiation_clad']:
                # Get cladding thickness
                irrad_clad_thickness = inputs['irradiation_clad_thickness']

                # Draw outer cladding structure
                clad_rect = Rectangle((assembly_x-assembly_pitch/2, fuel_start),
                                    assembly_pitch, fuel_height,
                                    facecolor=get_material_color(inputs['clad_type']),
                                    edgecolor='black', linewidth=0.5, alpha=0.9)
                ax.add_patch(clad_rect)

                # Draw inner irradiation space
                inner_width = assembly_pitch - 2 * irrad_clad_thickness
                inner_rect = Rectangle((assembly_x-inner_width/2, fuel_start),
                                     inner_width, fuel_height,
                                     facecolor=get_material_color(inputs['irradiation_cell_fill']),
                                     edgecolor='black', linewidth=0.3, alpha=0.9)
                ax.add_patch(inner_rect)
            else:
                # No cladding
                irrad_rect = Rectangle((assembly_x-assembly_pitch/2, fuel_start),
                                     assembly_pitch, fuel_height,
                                     facecolor=get_material_color(inputs['irradiation_cell_fill']),
                                     edgecolor='black', linewidth=0.5, alpha=0.9)
                ax.add_patch(irrad_rect)
            continue

        # Skip coolant cells
        if cell_type == 'C':
            continue

        # Draw all pins within this fuel assembly cross-section
        n_pins = int(inputs['n_side_pins'])

        for j in range(n_pins):
            pin_x = assembly_x + (j - (n_pins-1)/2) * pin_pitch

            # Cladding
            clad_rect = Rectangle((pin_x-r_clad_outer, fuel_start),
                                2*r_clad_outer, fuel_height,
                                facecolor=get_material_color(inputs['clad_type']),
                                edgecolor='black', linewidth=0.1, alpha=0.9)
            ax.add_patch(clad_rect)

            # Fuel (enhanced fuel for 'E' cells)
            fuel_color = '#8B0000' if cell_type == 'E' else get_material_color(inputs['fuel_type'])
            fuel_rect = Rectangle((pin_x-r_fuel, fuel_start),
                                2*r_fuel, fuel_height,
                                facecolor=fuel_color, edgecolor='none', alpha=0.95)
            ax.add_patch(fuel_rect)


def draw_core_plates_side_view(ax, tank_radius, fuel_start, fuel_height, assembly_pitch, inputs, view_type="YZ"):
    """Draw detailed plate structure in side view"""
    plate_pitch = inputs['fuel_plate_pitch']
    meat_thickness = inputs['fuel_meat_thickness']
    clad_thickness = inputs['clad_thickness']
    clad_structure_width = inputs['clad_structure_width']
    n_plates = int(inputs['plates_per_assembly'])

    # Get actual core lattice dimensions
    lattice = inputs['core_lattice']
    n_rows, n_cols = len(lattice), len(lattice[0])

    # Choose which slice of the core to show based on view type
    if "XZ" in view_type:
        # XZ view: slice along Y direction, show middle row across X
        slice_idx = len(lattice) // 2  # Middle row
        n_assemblies = len(lattice[0])  # Number of columns (X direction)
        core_lattice_width = n_assemblies * assembly_pitch
        assembly_positions = []
        for j in range(n_assemblies):
            assembly_x = (j - (n_assemblies - 1) / 2) * assembly_pitch
            cell_type = lattice[slice_idx][j]  # Middle row, column j
            assembly_positions.append((assembly_x, cell_type))
    else:
        # YZ view: slice along X direction, show middle column across Y
        slice_idx = len(lattice[0]) // 2  # Middle column
        n_assemblies = len(lattice)  # Number of rows (Y direction)
        core_lattice_width = n_assemblies * assembly_pitch
        assembly_positions = []
        for i in range(n_assemblies):
            assembly_x = (i - (n_assemblies - 1) / 2) * assembly_pitch
            cell_type = lattice[i][slice_idx]  # Row i, middle column
            assembly_positions.append((assembly_x, cell_type))

    # Draw assemblies
    for assembly_x, cell_type in assembly_positions:

        # Check if this is an irradiation position
        if 'I_' in cell_type:
            # Draw irradiation position in side view
            if inputs['irradiation_clad']:
                # For plate assemblies, use clad_structure_width
                cladding_thickness = clad_structure_width

                # Draw outer cladding structure
                clad_rect = Rectangle((assembly_x-assembly_pitch/2, fuel_start),
                                    assembly_pitch, fuel_height,
                                    facecolor=get_material_color(inputs['clad_type']),
                                    edgecolor='black', linewidth=0.5, alpha=0.9)
                ax.add_patch(clad_rect)

                # Draw inner irradiation space
                inner_width = assembly_pitch - 2 * cladding_thickness
                inner_rect = Rectangle((assembly_x-inner_width/2, fuel_start),
                                     inner_width, fuel_height,
                                     facecolor=get_material_color(inputs['irradiation_cell_fill']),
                                     edgecolor='black', linewidth=0.3, alpha=0.9)
                ax.add_patch(inner_rect)
            else:
                # No cladding
                irrad_rect = Rectangle((assembly_x-assembly_pitch/2, fuel_start),
                                     assembly_pitch, fuel_height,
                                     facecolor=get_material_color(inputs['irradiation_cell_fill']),
                                     edgecolor='black', linewidth=0.5, alpha=0.9)
                ax.add_patch(irrad_rect)
            continue

        # Skip coolant cells
        if cell_type == 'C':
            continue

        # Calculate assembly structural dimensions
        fuel_region_width = n_plates * plate_pitch
        total_assembly_width = fuel_region_width + 2 * clad_structure_width

        # Draw complete assembly structure background
        assembly_structure = Rectangle((assembly_x - total_assembly_width/2, fuel_start),
                                     total_assembly_width, fuel_height,
                                     facecolor=get_material_color(inputs['clad_type']),
                                     edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.add_patch(assembly_structure)

        # Draw fuel region background (coolant within assembly)
        fuel_region = Rectangle((assembly_x - fuel_region_width/2, fuel_start),
                              fuel_region_width, fuel_height,
                              facecolor=get_material_color(inputs['coolant_type']),
                              edgecolor='none', alpha=0.6)
        ax.add_patch(fuel_region)

        # Draw individual fuel plates
        for j in range(n_plates):
            plate_x = assembly_x + (j - (n_plates-1)/2) * plate_pitch

            # Calculate plate dimensions
            total_plate_thickness = meat_thickness + 2 * clad_thickness

            # Draw complete cladding structure
            clad_rect = Rectangle((plate_x-total_plate_thickness/2, fuel_start),
                                total_plate_thickness, fuel_height,
                                facecolor=get_material_color(inputs['clad_type']),
                                edgecolor='black', linewidth=0.1, alpha=0.9)
            ax.add_patch(clad_rect)

            # Draw fuel meat (enhanced fuel for 'E' cells)
            fuel_color = '#8B0000' if cell_type == 'E' else get_material_color(inputs['fuel_type'])
            fuel_rect = Rectangle((plate_x-meat_thickness/2, fuel_start),
                                meat_thickness, fuel_height,
                                facecolor=fuel_color, edgecolor='none', alpha=0.95)
            ax.add_patch(fuel_rect)


def get_material_legend_info(inputs, view_type):
    """Get legend information for current materials"""
    legend_items = []

    if "Core" in view_type:
        legend_items.extend([
            (get_material_color(inputs['fuel_type']), f"Fuel ({inputs['fuel_type']})"),
            ("#8B0000", "Enhanced Fuel"),
            (get_material_color(inputs['clad_type']), f"Cladding ({inputs['clad_type']})"),
            (get_material_color("Coolant"), f"Coolant ({inputs['coolant_type']})"),
        ])

        # Add irradiation position legend
        if inputs['irradiation_clad']:
            legend_items.extend([
                (get_material_color(inputs['clad_type']),
                 f"Irradiation Cladding ({inputs['clad_type']})"),
                (get_material_color(inputs['irradiation_cell_fill']),
                 f"Irradiation Fill ({inputs['irradiation_cell_fill']})"),
            ])
        else:
            legend_items.append((get_material_color(inputs['irradiation_cell_fill']),
                               f"Irradiation Position ({inputs['irradiation_cell_fill']})"))

        legend_items.extend([
            (get_material_color(inputs['reflector_material']),
             f"Reflector ({inputs['reflector_material']})"),
            (get_material_color(inputs['bioshield_material']),
             f"Bioshield ({inputs['bioshield_material']})"),
            (get_material_color("Plenum"), "Plenum"),
            (get_material_color("Feed"), "Feed Region")
        ])
    elif "Assembly" in view_type:
        if inputs['assembly_type'] == 'Pin':
            legend_items.extend([
                (get_material_color(inputs['fuel_type']), f"Fuel ({inputs['fuel_type']})"),
                (get_material_color(inputs['clad_type']), f"Cladding ({inputs['clad_type']})"),
                (get_material_color("Coolant"), f"Coolant ({inputs['coolant_type']})"),
                ("#696969", "Guide Tube"),
                ("#FFFFFF", "Gap"),
            ])
        else:
            legend_items.extend([
                (get_material_color(inputs['fuel_type']), f"Fuel ({inputs['fuel_type']})"),
                (get_material_color(inputs['clad_type']), f"Cladding ({inputs['clad_type']})"),
                (get_material_color("Coolant"), f"Coolant ({inputs['coolant_type']})"),
            ])
    elif "Element" in view_type:
        if inputs['assembly_type'] == 'Pin':
            legend_items.extend([
                (get_material_color(inputs['fuel_type']), f"Fuel ({inputs['fuel_type']})"),
                (get_material_color(inputs['clad_type']), f"Cladding ({inputs['clad_type']})"),
                (get_material_color("Coolant"), f"Coolant ({inputs['coolant_type']})"),
                ("#FFFFFF", "Gap"),
            ])
        else:
            legend_items.extend([
                (get_material_color(inputs['fuel_type']), f"Fuel ({inputs['fuel_type']})"),
                (get_material_color(inputs['clad_type']), f"Cladding ({inputs['clad_type']})"),
                (get_material_color("Coolant"), f"Coolant ({inputs['coolant_type']})"),
            ])

    return legend_items
