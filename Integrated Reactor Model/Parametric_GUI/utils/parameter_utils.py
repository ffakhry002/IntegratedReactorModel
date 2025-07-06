"""
Parameter Utilities
Helper functions for parameter handling
"""
import numpy as np


class ParameterUtils:
    """Utility functions for parameter handling"""

    @staticmethod
    def parse_numeric_range(start, end, steps, param_type='float'):
        """Generate numeric range values.

        Parameters
        ----------
        start : str or float or int
            Start value of the range
        end : str or float or int
            End value of the range
        steps : str or int
            Number of steps in the range
        param_type : str, optional
            Type of parameter ('int' or 'float'), by default 'float'

        Returns
        -------
        list
            List of numeric values in the specified range

        Raises
        ------
        ValueError
            If range generation fails
        """
        try:
            if param_type == 'int':
                start = int(start)
                end = int(end)
                steps = int(steps)
                if steps <= 1:
                    values = [start, end]
                else:
                    values = [int(x) for x in np.linspace(start, end, steps)]
            else:
                start = float(start)
                end = float(end)
                steps = int(steps)
                if steps <= 1:
                    values = [start, end]
                else:
                    values = list(np.linspace(start, end, steps))
            return values
        except (ValueError, ImportError) as e:
            raise ValueError(f"Error generating range: {e}")

    @staticmethod
    def parse_numeric_list(values_str, param_type='float'):
        """Parse comma-separated numeric values.

        Parameters
        ----------
        values_str : str
            Comma-separated string of numeric values
        param_type : str, optional
            Type of parameter ('int' or 'float'), by default 'float'

        Returns
        -------
        list
            List of parsed numeric values

        Raises
        ------
        ValueError
            If parsing fails
        """
        try:
            value_strings = [v.strip() for v in values_str.split(',') if v.strip()]
            values = []

            for value_str in value_strings:
                if param_type == 'int':
                    values.append(int(value_str))
                else:
                    values.append(float(value_str))

            return values
        except ValueError:
            raise ValueError("Invalid values in list")

    @staticmethod
    def parse_lattice_string(lattice_str):
        """Parse a lattice string safely.

        Parameters
        ----------
        lattice_str : str
            String representation of a lattice configuration

        Returns
        -------
        list
            Parsed lattice configuration as nested list

        Raises
        ------
        ValueError
            If lattice parsing fails
        """
        try:
            # Clean up the string
            lattice_str = lattice_str.strip()

            # Handle different formats
            if lattice_str.startswith('[[['):
                # Multiple lattices
                return eval(lattice_str)
            elif lattice_str.startswith('[['):
                # Single lattice
                return eval(lattice_str)
            else:
                # Try to reconstruct
                if not lattice_str.startswith('['):
                    lattice_str = '[' + lattice_str
                if not lattice_str.endswith(']'):
                    lattice_str = lattice_str + ']'

                return eval('[' + lattice_str + ']')
        except Exception as e:
            raise ValueError(f"Error parsing lattice: {e}")

    @staticmethod
    def parse_timesteps_string(timesteps_str):
        """Parse timesteps string.

        Parameters
        ----------
        timesteps_str : str
            Comma-separated string of timestep values

        Returns
        -------
        list
            List of parsed timestep values (int or float)

        Raises
        ------
        ValueError
            If timesteps parsing fails
        """
        try:
            timesteps_str = timesteps_str.strip()

            # Handle different formats
            if not timesteps_str:
                return []

            # Parse as comma-separated values
            timesteps = [float(x.strip()) for x in timesteps_str.split(',') if x.strip()]

            # Convert to int if all are integers
            if all(x.is_integer() for x in timesteps):
                timesteps = [int(x) for x in timesteps]

            return timesteps
        except ValueError as e:
            raise ValueError(f"Invalid timesteps format: {e}")
