import numpy as np

def calculate_h_gap_vector(th_system):
    """Calculate gap heat transfer coefficient vector based on power and gap width.

    Args:
        th_system: THSystem object containing power and geometry information

    Returns:
        np.array: Array of gap heat transfer coefficients in W/m^2-K
    """
    power_W_cm = th_system.thermal_state.Q_dot_z / 100
    gap_width = th_system.geometry.gap_width

    data_points = {
        0.00005: [(170, 1.9), (330, 7.0), (380, 11.0), (450, 11.5), (500, 11.5)],
        0.0001:  [(170, 0.48), (330, 0.85), (380, 1.1), (450, 1.7), (500, 1.7)],
        0.0002:  [(170, 0.275), (330, 0.385), (380, 0.415), (450, 0.5), (500, 0.585)],
        0.00025: [(170, 0.2), (330, 0.3), (380, 0.33), (450, 0.385), (500, 0.4)],
        0.0005:  [(170, 0.35), (330, 0.5), (380, 0.75), (450, 0.95), (500, 0.95)]
    }

    def interpolate_single_point(power, gap):
        """Interpolate gap heat transfer coefficient for a single power and gap width point.

        Args:
            power (float): Linear power in W/cm
            gap (float): Gap width in microns

        Returns:
            float: Interpolated gap heat transfer coefficient in W/m^2-K
        """
        gap_widths = sorted(data_points.keys())
        if gap <= gap_widths[0]:
            lower_gap, upper_gap = gap_widths[0], gap_widths[0]
        elif gap >= gap_widths[-1]:
            lower_gap, upper_gap = gap_widths[-1], gap_widths[-1]
        else:
            for i in range(len(gap_widths) - 1):
                if gap_widths[i] <= gap < gap_widths[i + 1]:
                    lower_gap, upper_gap = gap_widths[i], gap_widths[i + 1]
                    break

        def interpolate_power(power, gap_data):
            for i in range(len(gap_data) - 1):
                t1, h1 = gap_data[i]
                t2, h2 = gap_data[i + 1]

                if t1 <= power <= t2:
                    slope = (h2 - h1) / (t2 - t1)
                    return h1 + slope * (power - t1)
            return gap_data[0][1] if power < gap_data[0][0] else gap_data[-1][1]

        h_gap_lower = interpolate_power(power, data_points[lower_gap])
        h_gap_upper = interpolate_power(power, data_points[upper_gap])

        if lower_gap != upper_gap:
            gap_fraction = (gap - lower_gap) / (upper_gap - lower_gap)
            h_gap = h_gap_lower + gap_fraction * (h_gap_upper - h_gap_lower)
        else:
            h_gap = h_gap_lower

        return h_gap * 10000  # Convert to W/m^2-K

    vectorized_interpolate = np.vectorize(interpolate_single_point)
    return vectorized_interpolate(power_W_cm, gap_width)
