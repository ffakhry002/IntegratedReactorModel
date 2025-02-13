# Helper Codes

Core functionality modules for thermal-hydraulic calculations and analysis.

## Directory Structure

### Convergence
- Temperature convergence algorithms
- Coolant, fuel, and cladding convergence routines
- Iteration control and management

### Material Properties
- Coolant properties (density, viscosity, thermal conductivity)
- Fuel properties (thermal conductivity, temperature dependence)
- Cladding properties (thermal conductivity)
- Gap properties (heat transfer coefficient)

### Models
- Reactor power characteristics
- Thermal-hydraulic parameters
- Geometry definitions (pin and plate configurations)
- Thermal state management

## Key Features

- Robust convergence algorithms for thermal calculations
- Comprehensive material property calculations
- Modular design for easy maintenance and extension
- Support for both pin and plate fuel assembly types

## Usage Example

```python
from models.thermal_state import ThermalState
from models.reactor import ReactorPower, ThermalHydraulics
from convergence.temperature_convergence import converge_temperatures

# Initialize components
thermal_state = ThermalState()
reactor_power = ReactorPower(core_power=100e6)  # 100 MW
thermal_hydraulics = ThermalHydraulics(reactor_pressure=15.5e6)  # 15.5 MPa

# Perform convergence
converged_state = converge_temperatures(th_system)
```
