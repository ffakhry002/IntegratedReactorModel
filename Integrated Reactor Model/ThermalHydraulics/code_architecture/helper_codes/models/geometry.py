import numpy as np

class PinGeometry:
    def __init__(self, pin_pitch, r_fuel, r_clad_inner, r_clad_outer, n_side_pins, n_guide_tubes, fuel_height):
        """Initialize pin geometry parameters.

        Parameters
        ----------
        pin_pitch : float
            Distance between pin centers
        r_fuel : float
            Radius of fuel pellet
        r_clad_inner : float
            Inner radius of cladding
        r_clad_outer : float
            Outer radius of cladding
        n_side_pins : int
            Number of pins on each side of square assembly
        n_guide_tubes : int
            Number of guide tubes in assembly
        fuel_height : float
            Height of active fuel region

        Returns
        -------
        None
        """
        self.pin_pitch = pin_pitch
        self.r_fuel = r_fuel
        self.r_clad_inner = r_clad_inner
        self.r_clad_outer = r_clad_outer
        self.n_side_pins = n_side_pins
        self.n_guide_tubes = n_guide_tubes
        self.gap_width = r_clad_inner - r_fuel
        self.fuel_height = fuel_height
        self.n_elements_per_assembly = n_side_pins**2 - n_guide_tubes

        # Calculate derived quantities
        self.coolant_area = pin_pitch**2 - np.pi * r_clad_outer**2
        self.wetted_perimeter = 2 * np.pi * r_clad_outer
        self.hydraulic_diameter = 4 * self.coolant_area / self.wetted_perimeter

class PlateGeometry:
    """Class representing the geometry of a plate-type fuel assembly.

    Attributes:
        fuel_meat_width: Width of the fuel meat region
        fuel_plate_width: Total width of the fuel plate
        fuel_plate_pitch: Distance between centers of adjacent plates
        fuel_meat_thickness: Thickness of the fuel meat region
        clad_thickness: Thickness of the cladding
        plates_per_assembly: Number of plates in one assembly
        clad_structure_width: Width of the cladding structure
        fuel_height: Height of the fuel region
        n_elements_per_assembly: Number of fuel elements per assembly (same as plates_per_assembly)
        fuel_plate_thickness: Total thickness of fuel plate (meat + cladding)
        coolant_area: Flow area for coolant
        hydraulic_diameter: Hydraulic diameter for flow calculations
    """
    def __init__(self, fuel_meat_width, fuel_plate_width, fuel_plate_pitch, fuel_meat_thickness,
                 clad_thickness, plates_per_assembly, clad_structure_width, fuel_height):
        """Initialize plate geometry parameters.

        Parameters
        ----------
        fuel_meat_width : float
            Width of the fuel meat region
        fuel_plate_width : float
            Total width of the fuel plate
        fuel_plate_pitch : float
            Distance between centers of adjacent plates
        fuel_meat_thickness : float
            Thickness of the fuel meat region
        clad_thickness : float
            Thickness of the cladding
        plates_per_assembly : int
            Number of plates in one assembly
        clad_structure_width : float
            Width of the cladding structure
        fuel_height : float
            Height of the fuel region

        Returns
        -------
        None
        """
        self.fuel_meat_width = fuel_meat_width
        self.fuel_plate_width = fuel_plate_width
        self.fuel_plate_pitch = fuel_plate_pitch
        self.fuel_meat_thickness = fuel_meat_thickness
        self.clad_thickness = clad_thickness
        self.plates_per_assembly = plates_per_assembly
        self.clad_structure_width = clad_structure_width
        self.fuel_height = fuel_height
        self.n_elements_per_assembly = plates_per_assembly

        # Calculate derived quantities
        self.fuel_plate_thickness = fuel_meat_thickness + 2*clad_thickness
        self.coolant_area = fuel_plate_width * (fuel_plate_pitch-self.fuel_plate_thickness)
        self.hydraulic_diameter = 4 * fuel_plate_width*(fuel_plate_pitch-self.fuel_plate_thickness)/(
            2*(fuel_plate_width+fuel_plate_pitch-self.fuel_plate_thickness))
