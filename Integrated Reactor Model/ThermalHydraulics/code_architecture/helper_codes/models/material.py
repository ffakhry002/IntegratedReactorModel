class Material:
    def __init__(self, coolant_type, clad_type, fuel_type):
        """Initialize Material class with material types.

        Parameters
        ----------
        coolant_type : str
            Type of coolant material
        clad_type : str
            Type of cladding material
        fuel_type : str
            Type of fuel material

        Returns
        -------
        None
        """
        self.coolant_type = coolant_type
        self.clad_type = clad_type
        self.fuel_type = fuel_type
