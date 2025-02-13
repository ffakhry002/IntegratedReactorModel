"""Geometry validation module for thermal-hydraulic analysis."""

def validate_pin_geometry(pin_geometry):
    """Validate pin geometry parameters.

    Args:
        pin_geometry: PinGeometry object containing pin dimensions

    Raises:
        ValueError: If any geometry constraints are violated
    """
    # Convert to cm for error messages (same as original code)
    r_fuel = pin_geometry.r_fuel * 100
    r_clad_inner = pin_geometry.r_clad_inner * 100
    r_clad_outer = pin_geometry.r_clad_outer * 100
    pin_pitch = pin_geometry.pin_pitch * 100

    if r_fuel > r_clad_inner:
        raise ValueError(
            f"Fuel radius ({r_fuel:.3f} cm) cannot be greater than or equal to "
            f"inner cladding radius ({r_clad_inner:.3f} cm)."
        )

    if r_clad_inner >= r_clad_outer:
        raise ValueError(
            f"Inner cladding radius ({r_clad_inner:.3f} cm) cannot be greater than or equal to "
            f"outer cladding radius ({r_clad_outer:.3f} cm)."
        )

    if r_clad_outer > pin_pitch/2:
        raise ValueError(
            f"Outer cladding radius ({r_clad_outer:.3f} cm) cannot be greater than or equal to "
            f"the pitch divided by 2 ({pin_pitch/2:.3f} cm)."
        )

def validate_plate_geometry(plate_geometry):
    """Validate plate geometry parameters.

    Args:
        plate_geometry: PlateGeometry object containing plate dimensions

    Raises:
        ValueError: If any geometry constraints are violated
    """
    # Convert to cm for error messages (same as original code)
    fuel_meat_width = plate_geometry.fuel_meat_width * 100
    fuel_plate_width = plate_geometry.fuel_plate_width * 100
    fuel_meat_thickness = plate_geometry.fuel_meat_thickness * 100
    clad_thickness = plate_geometry.clad_thickness * 100
    fuel_plate_pitch = plate_geometry.fuel_plate_pitch * 100

    if fuel_meat_width >= fuel_plate_width:
        raise ValueError(
            f"Fuel meat width ({fuel_meat_width:.3f} cm) cannot be greater than or equal to "
            f"plate width ({fuel_plate_width:.3f} cm)."
        )

    required_pitch = fuel_meat_thickness + 2 * clad_thickness
    if fuel_plate_pitch < required_pitch:
        raise ValueError(
            f"Fuel plate pitch ({fuel_plate_pitch:.3f} cm) must be greater than or equal to "
            f"meat thickness + 2 * clad thickness ({required_pitch:.3f} cm)."
        )

def validate_geometry(th_system):
    """Validate all geometry parameters for a thermal-hydraulic system.

    Args:
        th_system: THSystem object containing geometry information

    Raises:
        ValueError: If any geometry constraints are violated
    """
    if th_system.thermal_hydraulics.assembly_type == 'Pin':
        validate_pin_geometry(th_system.pin_geometry)
    else:
        validate_plate_geometry(th_system.plate_geometry)
    print("Geometry validation passed")
