# Integrated Reactor Model

A comprehensive nuclear reactor simulation framework integrating OpenMC neutronics with thermal-hydraulics coupling and depletion analysis.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Module Architecture](#module-architecture)
- [Configuration](#configuration)
- [Reactor Geometry](#reactor-geometry)
- [Simulation Modes](#simulation-modes)
- [Thermal-Hydraulics](#thermal-hydraulics)
- [Depletion Analysis](#depletion-analysis)
- [Output and Visualization](#output-and-visualization)
- [GUI Interfaces](#gui-interfaces)
- [API Reference](#api-reference)
- [Examples](#examples)

---

## Overview

The Integrated Reactor Model provides a complete simulation framework for nuclear reactor analysis. It combines:

- **OpenMC Monte Carlo Transport**: High-fidelity neutronics calculations
- **Thermal-Hydraulics**: Temperature distribution and heat transfer
- **Depletion Analysis**: Fuel burnup and isotopic evolution
- **Parametric Studies**: Automated configuration sweeps

### Simulation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Configuration                                │
│  inputs.py → Core lattice, materials, geometry parameters           │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────┐
│                    Geometry & Materials                              │
│  Build OpenMC geometry → Pin/Plate assemblies, core, reflectors     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────┐
│                    Eigenvalue Calculation                            │
│  OpenMC transport → k-effective, flux distributions                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    v                       v
┌───────────────────────────┐ ┌───────────────────────────────────────┐
│    Thermal-Hydraulics     │ │         Depletion Analysis            │
│  Power profiles →         │ │  Burnup calculations →                │
│  Temperature distribution │ │  Isotopic evolution                   │
└───────────────────────────┘ └───────────────────────────────────────┘
                    │                       │
                    └───────────┬───────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────┐
│                    Results & Visualization                           │
│  Flux maps, power distributions, temperature profiles               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

- **Flexible Core Design**: Pin or plate fuel assemblies
- **Multiple Fuel Types**: UO₂, U₃Si₂, U-10Mo
- **Irradiation Experiments**: PWR loop, BWR loop, gas capsule support
- **Energy-Resolved Tallies**: Thermal, epithermal, fast groups
- **Fast Mode**: Quick k-effective calculations for optimization
- **Parametric Studies**: Automated sweeps with parallel execution
- **GUI Support**: Visual configuration tools

---

## Quick Start

### Standard Simulation

```bash
cd "Integrated Reactor Model"
python main.py
```

This runs the complete workflow:
1. Generate geometry and materials
2. Run thermal-hydraulics (cosine approximation)
3. Run OpenMC eigenvalue calculation
4. Generate result plots

### Fast Mode

Edit `inputs.py`:
```python
"fast_mode": True
```

Then run:
```bash
python main.py
```

Fast mode:
- Uses 3-group energy structure
- Minimal tallies (irradiation positions only)
- Skips power tallies and mesh tallies
- ~10× faster execution

---

## Module Architecture

```
Integrated Reactor Model/
├── main.py                    # Main simulation orchestrator
├── inputs.py                  # Configuration parameters
│
├── Reactor/                   # Core geometry and materials
│   ├── geometry.py           # Geometry plotting
│   ├── materials.py          # Material definitions
│   └── geometry_helpers/     # Geometry construction
│       ├── core.py           # Core universe builder
│       ├── pin_fuel.py       # Pin assembly builder
│       ├── plate_fuel.py     # Plate assembly builder
│       ├── irradiation_cell.py
│       └── irradiation_experiments.py
│
├── eigenvalue/               # Neutronics calculations
│   ├── run.py               # Main eigenvalue driver
│   ├── outputs.py           # Result processing
│   ├── parametric_study.py  # Parametric sweeps
│   └── tallies/             # Tally definitions
│       ├── core_tallies.py
│       ├── irradiation_tallies.py
│       ├── power_tallies.py
│       └── energy_groups.py
│
├── depletion/               # Burnup calculations
│   ├── run_depletion.py
│   ├── depletion_operator.py
│   └── depletion_output_text.py
│
├── ThermalHydraulics/       # T/H analysis
│   ├── TH_refactored.py     # Main T/H system
│   └── code_architecture/   # T/H modules
│       ├── helper_codes/
│       │   ├── convergence/
│       │   ├── material_properties/
│       │   ├── models/
│       │   ├── power_calculations/
│       │   └── temperature_points/
│       └── data_output_code/
│
├── plotting/                # Visualization
│   ├── plotall.py          # Plot orchestrator
│   └── functions/
│       ├── flux_maps.py
│       ├── power.py
│       ├── depletion.py
│       └── normalized_flux_profiles.py
│
├── Inputs_GUI/              # Configuration GUI
├── Parametric_GUI/          # Parametric study GUI
│
└── utils/                   # Utilities
    ├── base_inputs.py
    └── parametric_results_analyzer.py
```

---

## Configuration

### inputs.py Structure

The main configuration file `inputs.py` contains all simulation parameters:

#### Core Configuration

```python
{
    # Core layout (8×8 grid)
    "core_lattice": [
        ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'I_1P', 'I_2B', 'F', 'F', 'F'],  # P=PWR, B=BWR, G=Gas
        ['F', 'F', 'F', 'I_3', 'I_4G', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
    ],
    
    "core_power": 10.0,           # MW
    "assembly_type": 'Plate',     # 'Pin' or 'Plate'
}
```

#### Lattice Cell Types

| Symbol | Description |
|--------|-------------|
| `C` | Coolant channel |
| `F` | Standard fuel assembly |
| `E` | Enhanced enrichment fuel |
| `I_1`, `I_2`, etc. | Irradiation positions |
| `I_1P` | PWR loop irradiation |
| `I_1B` | BWR loop irradiation |
| `I_1G` | Gas capsule irradiation |

#### Geometry Parameters

```python
{
    # Radial dimensions (m)
    "tank_radius": 0.25,
    "reflector_thickness": 0.1,
    "bioshield_thickness": 0.25,
    
    # Axial dimensions (m)
    "fuel_height": 0.6,
    "bottom_reflector_thickness": 0.3,
    "plenum_height": 1.7,
    
    # Pin assembly (if assembly_type='Pin')
    "pin_pitch": 0.0126,
    "r_fuel": 0.0041,
    "r_clad_inner": 0.0042,
    "r_clad_outer": 0.00475,
    "n_side_pins": 3,
    
    # Plate assembly (if assembly_type='Plate')
    "fuel_meat_width": 0.0391,
    "fuel_plate_width": 0.0481,
    "fuel_plate_pitch": 0.0037,
    "plates_per_assembly": 13,
}
```

#### Materials

```python
{
    "coolant_type": 'Light Water',    # or 'Heavy Water'
    "clad_type": 'Al6061',            # or 'Zirc2', 'Zirc4'
    "fuel_type": 'U10Mo',             # or 'U3Si2', 'UO2'
    "reflector_material": "mgo",
    "bioshield_material": "Concrete",
    
    "n%": 19.75,                      # Standard enrichment (%)
    "n%E": 93,                        # Enhanced enrichment (%)
}
```

#### Transport Settings

```python
{
    "batches": 250,
    "inactive": 50,
    "particles": 250000,
    "energy_structure": 'log1001',    # or 'three_group', 'scale238'
    
    "thermal_cutoff": 0.625,          # eV
    "fast_cutoff": 100000.0,          # eV
    
    "core_mesh_dimension": [201, 201, 201],
    "entropy_mesh_dimension": [20, 20, 20],
}
```

---

## Reactor Geometry

### Pin Fuel Assembly

```
     ┌─────────────────┐
     │  ●  ●  ●  ●  ● │
     │  ●  ●  ●  ●  ● │
     │  ●  ●  ○  ●  ● │  ● = Fuel pin
     │  ●  ●  ●  ●  ● │  ○ = Guide tube
     │  ●  ●  ●  ●  ● │
     └─────────────────┘
```

Pin structure (radial):
1. Fuel pellet (UO₂, U₃Si₂, or U-10Mo)
2. Gap (helium)
3. Cladding (Al6061, Zircaloy)
4. Coolant

### Plate Fuel Assembly

```
     ┌─────────────────────┐
     │ ═══════════════════ │
     │                     │
     │ ═══════════════════ │  ═ = Fuel plate
     │                     │
     │ ═══════════════════ │
     │                     │
     │ ═══════════════════ │
     └─────────────────────┘
```

Plate structure:
1. Fuel meat
2. Cladding (both sides)
3. Coolant channels

### Core Structure

```
          ┌───────────────────────────┐
          │       Top Reflector       │
          ├───────────────────────────┤
          │         Plenum            │
          ├───────────────────────────┤
          │                           │
      ────┤     Active Core           ├────  Radial
 Reflector│   (Fuel + Irradiation)    │      Reflector
          │                           │
          ├───────────────────────────┤
          │     Bottom Reflector      │
          ├───────────────────────────┤
          │     Bottom Bioshield      │
          └───────────────────────────┘
```

---

## Simulation Modes

### Standard Mode

Full simulation with all tallies and analyses:

```python
"fast_mode": False,
"parametric_study": False,
```

Includes:
- Full energy structure tallies
- Core mesh flux maps
- Power distributions
- Axial flux profiles
- Thermal-hydraulics with multiple profiles

### Fast Mode

Optimized for quick k-effective calculations:

```python
"fast_mode": True,
```

Optimizations:
- Three-group energy structure
- No power tallies
- No mesh tallies
- No entropy mesh
- ~10× speedup

### Parametric Study Mode

Automated sweeps over configuration space:

```python
"parametric_study": True,
```

See [Parametric Studies](#parametric-studies) section.

---

## Thermal-Hydraulics

### THSystem Class

```python
from ThermalHydraulics.TH_refactored import THSystem

# Initialize
th_system = THSystem(inputs)

# Calculate temperatures
thermal_state = th_system.calculate_temperature_distribution()

# Write results
th_system.write_results(output_dir)
```

### Power Sources

| Mode | Description |
|------|-------------|
| `COSINE` | Cosine axial power shape (initial approximation) |
| `HOT_ELEMENT` | Maximum power element from tallies |
| `CORE_AVERAGE` | Average power across all elements |

### Temperature Points Calculated

- Coolant temperature (axial profile)
- Cladding outer surface
- Cladding inner surface
- Fuel surface
- Fuel centerline (maximum)

### Material Properties

Temperature-dependent properties for:
- **Coolant**: Density, viscosity, thermal conductivity, specific heat
- **Cladding**: Thermal conductivity, specific heat
- **Fuel**: Thermal conductivity (with burnup correlation)

---

## Depletion Analysis

### Depletion Modes

```python
{
    "deplete_core": True,              # Full core depletion
    "deplete_assembly": False,         # Single assembly (reflective BC)
    "deplete_element": False,          # Single element (reflective BC)
}
```

### Timestep Configuration

```python
{
    "depletion_timestep_units": "MWd/kgHM",  # or 'days'
    "depletion_timesteps": [
        {'steps': 10, 'size': 0.01},   # Fine initial steps
        {'steps': 10, 'size': 0.1},
        {'steps': 10, 'size': 0.5},
        {'steps': 5, 'size': 2.5},
        {'steps': 5, 'size': 5.0},
        {'steps': 5, 'size': 10.0},    # Larger steps at high burnup
    ],
}
```

### Integration Methods

| Method | Description |
|--------|-------------|
| `predictor` | Simple predictor (fast) |
| `cecm` | CE/CM predictor-corrector |
| `cf4` | 4th-order predictor-corrector |
| `epcrk4` | Stochastic Runge-Kutta |

### Nuclide Tracking

```python
"depletion_nuclides": [
    'U235', 'U238', 'Pu239', 
    'Xe135', 'Sm149', 
    'Cs137', 'Sr90', 'I131'
]
```

---

## Output and Visualization

### Directory Structure

```
simulation_data/
├── Geometry_and_Materials/
│   ├── core_images/
│   │   ├── core_xy.png
│   │   ├── core_yz.png
│   │   └── core_xz.png
│   ├── fuel_images/
│   │   ├── pin_assembly_xy.png  (or plate_assembly_xy.png)
│   │   └── single_pin_xy.png
│   ├── irradiation_P/           # PWR loop plots
│   ├── irradiation_B/           # BWR loop plots
│   ├── irradiation_G/           # Gas capsule plots
│   └── materials.txt
│
├── transport_data/
│   ├── statepoint.eigenvalue.h5
│   └── results.txt
│
├── ThermalHydraulics/
│   ├── cosine_calculation/
│   ├── hot_element/
│   └── core_average/
│
├── flux_plots/
│   ├── flux_map_thermal.png
│   ├── flux_map_epithermal.png
│   ├── flux_map_fast.png
│   └── axial_flux_profiles.png
│
├── power_plots/
│   ├── assembly_power_distribution.png
│   └── detailed_power_distribution.csv
│
├── depletion_data/
│   └── depletion_results.h5
│
└── depletion_plots/
    ├── keff_vs_burnup.png
    └── nuclide_evolution.png
```

### Results File (results.txt)

```text
================================================================================
INTEGRATED REACTOR SIMULATION RESULTS
================================================================================

K-EFFECTIVE
-----------
k-effective = 1.05432 ± 0.00023

IRRADIATION POSITION RESULTS
----------------------------
Position 1 (3, 3):
  Total Flux: 2.34e+14 ± 1.2e+12 n/cm²·s
  Thermal:    1.06e+14 (45.2%)
  Epithermal: 7.51e+13 (32.1%)
  Fast:       5.31e+13 (22.7%)

Position 2 (3, 4):
  Total Flux: 2.56e+14 ± 1.3e+12 n/cm²·s
  ...

POWER SUMMARY
-------------
Total Power: 10.00 MW
Peak-to-Average: 1.23
...
```

---

## GUI Interfaces

### Inputs GUI

Visual interface for configuring reactor parameters:

```bash
cd Inputs_GUI
python main.py
```

Features:
- Core lattice editor (click to place fuel/irradiation)
- Geometry parameter sliders
- Material selection dropdowns
- 2D/3D visualization
- Export to inputs.py

### Parametric GUI

Interface for setting up parametric studies:

```bash
cd Parametric_GUI
python main.py
```

Features:
- Parameter range specification
- Latin Hypercube Sampling
- Progress tracking
- Result analysis

---

## API Reference

### run_eigenvalue

```python
def run_eigenvalue(inputs_dict=None):
    """
    Run eigenvalue calculation with OpenMC.
    
    Parameters
    ----------
    inputs_dict : dict, optional
        Configuration dictionary. Uses global inputs if None.
    
    Returns
    -------
    tuple
        (k_effective, k_std_dev)
    """
```

### THSystem

```python
class THSystem:
    """Thermal-hydraulics analysis system."""
    
    def __init__(self, inputs_dict):
        """Initialize with configuration."""
        
    def calculate_temperature_distribution(self):
        """
        Calculate axial temperature distribution.
        
        Returns
        -------
        ThermalState
            Contains all temperature arrays
        """
        
    def write_results(self, output_dir):
        """Write results to files."""
```

### build_core_uni

```python
def build_core_uni(mat_dict, inputs_dict=None):
    """
    Build complete core universe for OpenMC.
    
    Parameters
    ----------
    mat_dict : dict
        Material dictionary from make_materials()
    inputs_dict : dict, optional
        Configuration dictionary
    
    Returns
    -------
    tuple
        (core_universe, irradiation_universes)
    """
```

### make_materials

```python
def make_materials(mat_list=None, inputs_dict=None):
    """
    Create OpenMC materials.
    
    Parameters
    ----------
    mat_list : list, optional
        Specific materials to create. None for all.
    inputs_dict : dict, optional
        Configuration dictionary
    
    Returns
    -------
    tuple
        (mat_dict, materials_collection)
    """
```

---

## Examples

### Example 1: Basic Eigenvalue Calculation

```python
from eigenvalue.run import run_eigenvalue

# Simple calculation with default inputs
k_eff, k_std = run_eigenvalue()
print(f"k-effective = {k_eff:.5f} ± {k_std:.5f}")
```

### Example 2: Custom Configuration

```python
from inputs import inputs
import copy

# Modify configuration
my_inputs = copy.deepcopy(inputs)
my_inputs['core_power'] = 15.0  # 15 MW
my_inputs['fuel_type'] = 'UO2'
my_inputs['n%'] = 4.95  # 5% enrichment

# Run with custom inputs
from eigenvalue.run import run_eigenvalue
k_eff, _ = run_eigenvalue(inputs_dict=my_inputs)
```

### Example 3: Thermal-Hydraulics Analysis

```python
from ThermalHydraulics.TH_refactored import THSystem
from inputs import inputs

# Initialize T/H system
th_system = THSystem(inputs)

# Calculate temperature distribution
thermal_state = th_system.calculate_temperature_distribution()

# Access results
print(f"Max fuel temp: {thermal_state.T_fuel_max:.1f} K")
print(f"Max clad temp: {thermal_state.T_clad_outer_max:.1f} K")
print(f"Coolant outlet: {thermal_state.T_coolant_outlet:.1f} K")

# Write results
th_system.write_results("output/thermal_hydraulics")
```

### Example 4: Depletion Calculation

```python
from depletion.run_depletion import run_all_depletions
from inputs import inputs
import copy

# Enable core depletion
my_inputs = copy.deepcopy(inputs)
my_inputs['deplete_core'] = True
my_inputs['depletion_timesteps'] = [
    {'steps': 5, 'size': 1.0},
    {'steps': 10, 'size': 5.0},
]

# Run depletion
results = run_all_depletions(
    output_dir="depletion_results",
    inputs_dict=my_inputs
)
```

### Example 5: Parametric Study

```python
from eigenvalue.parametric_study import run_parametric_study
from inputs import inputs
import copy

# Configure parametric study
study_inputs = copy.deepcopy(inputs)
study_inputs['parametric_study'] = True
study_inputs['fast_mode'] = True  # Use fast mode for speed

# Define parameter variations (in parametric_study.py)
# Then run:
run_parametric_study()
```

### Example 6: Generate Geometry Plots

```python
from Reactor.geometry import plot_geometry
from inputs import inputs

# Generate geometry visualizations
plot_geometry(
    output_dir="geometry_plots",
    inputs_dict=inputs
)
```

---

## Performance Tips

### Memory Management

```python
# For large simulations, reduce mesh resolution
"core_mesh_dimension": [101, 101, 101],  # vs [201, 201, 201]
"entropy_mesh_dimension": [10, 10, 10],   # vs [20, 20, 20]
```

### Fast Mode Optimization

```python
{
    "fast_mode": True,
    "particles": 100000,      # Reduced from 250000
    "batches": 150,           # Reduced from 250
}
```

### Parallel Depletion

```python
# OpenMC automatically uses available cores
# Set OMP_NUM_THREADS for thread count
import os
os.environ['OMP_NUM_THREADS'] = '16'
```

---

## Troubleshooting

### Common Issues

**Issue**: `No cross sections found`
**Solution**: Set environment variable:
```bash
export OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml
```

**Issue**: Memory error during mesh tally
**Solution**: Reduce `core_mesh_dimension` in inputs.py

**Issue**: Geometry overlap errors
**Solution**: Check irradiation cell dimensions match assembly pitch

**Issue**: Slow convergence
**Solution**: Increase inactive batches or use better source distribution

---

## References

- OpenMC Documentation: https://docs.openmc.org
- Nuclear Data: ENDF/B-VIII.0
- Thermal-Hydraulics: Todreas & Kazimi, "Nuclear Systems"
- Depletion: ORIGEN methodology
