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
    """
    def __init__(self, core_power, num_assemblies, Q_dot_z_toggle, input_avg_power_density,
                 input_max_lp, input_avg_lp, cos_curve_squeeze):
        self.core_power = core_power
        self.num_assemblies = num_assemblies
        self.Q_dot_z_toggle = Q_dot_z_toggle
        self.input_avg_power_density = input_avg_power_density
        self.input_max_lp = input_max_lp
        self.input_avg_lp = input_avg_lp
        self.cos_curve_squeeze = cos_curve_squeeze

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
        self.reactor_pressure = reactor_pressure
        self.flow_rate = flow_rate
        self.T_inlet = T_inlet
        self.assembly_type = assembly_type
        self.outputs_folder = outputs_folder
        self.n_radial = 500
        self.n_axial = 1000
