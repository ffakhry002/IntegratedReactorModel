class ReactorPower:
    """Class representing reactor power characteristics.

    Attributes:
        core_power: Total reactor core power
        num_assemblies: Number of fuel assemblies
        Q_dot_z_toggle: Power distribution calculation method ('CP', 'PD', 'ALP', or 'MLP')
        input_avg_power_density: Input average power density
        input_max_lp: Input maximum linear power
        input_avg_lp: Input average linear power
        cos_curve_squeeze: Cosine curve squeeze factor for power distribution
        power_source: Source of power distribution ('COSINE', 'HOT_ELEMENT', or 'CORE_AVERAGE')
        csv_path: Path to the power distribution CSV file (only used if power_source is not 'COSINE')
    """
    def __init__(self, core_power, num_assemblies, Q_dot_z_toggle, input_avg_power_density,
                 input_max_lp, input_avg_lp, cos_curve_squeeze, power_source='COSINE', csv_path=None):
        """Initialize reactor power characteristics.

        Parameters
        ----------
        core_power : float
            Total reactor core power
        num_assemblies : int
            Number of fuel assemblies
        Q_dot_z_toggle : str
            Power distribution calculation method ('CP', 'PD', 'ALP', or 'MLP')
        input_avg_power_density : float
            Input average power density
        input_max_lp : float
            Input maximum linear power
        input_avg_lp : float
            Input average linear power
        cos_curve_squeeze : float
            Cosine curve squeeze factor for power distribution
        power_source : str, optional
            Source of power distribution ('COSINE', 'HOT_ELEMENT', or 'CORE_AVERAGE')
        csv_path : str, optional
            Path to the power distribution CSV file

        Returns
        -------
        None
        """
        self.core_power = core_power
        self.num_assemblies = num_assemblies
        self.Q_dot_z_toggle = Q_dot_z_toggle
        self.input_avg_power_density = input_avg_power_density
        self.input_max_lp = input_max_lp
        self.input_avg_lp = input_avg_lp
        self.cos_curve_squeeze = cos_curve_squeeze
        self.power_source = power_source
        self.csv_path = csv_path

class ThermalHydraulics:
    """Class representing thermal-hydraulic parameters and settings.

    Attributes:
        reactor_pressure: Operating pressure of the reactor
        flow_rate: Coolant flow rate
        T_inlet: Inlet coolant temperature
        assembly_type: Type of fuel assembly ('Pin' or 'Plate')
        outputs_folder: Directory for output files
        n_radial: Number of radial mesh points
        n_axial: Number of axial mesh points
    """
    def __init__(self, reactor_pressure, flow_rate, T_inlet, assembly_type, outputs_folder):
        """Initialize thermal-hydraulic parameters and settings.

        Parameters
        ----------
        reactor_pressure : float
            Operating pressure of the reactor
        flow_rate : float
            Coolant flow rate
        T_inlet : float
            Inlet coolant temperature
        assembly_type : str
            Type of fuel assembly ('Pin' or 'Plate')
        outputs_folder : str
            Directory for output files

        Returns
        -------
        None
        """
        self.reactor_pressure = reactor_pressure
        self.flow_rate = flow_rate
        self.T_inlet = T_inlet
        self.assembly_type = assembly_type
        self.outputs_folder = outputs_folder
        self.n_radial = 500
        self.n_axial = 1000
