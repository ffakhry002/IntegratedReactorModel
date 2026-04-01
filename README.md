# Nuclear Reactor Analysis Framework

A comprehensive Python-based framework for nuclear reactor core configuration analysis, simulation, and machine learning-based prediction. This project integrates three major components:

1. **Core Selection** - Intelligent sampling and optimization of reactor core configurations
2. **Machine Learning (ML)** - ML models for predicting reactor physics parameters (flux, k-effective)
3. **Integrated Reactor Model** - Full reactor simulation using OpenMC with thermal-hydraulics coupling

---

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Module Documentation](#module-documentation)
  - [Core Selection](#core-selection)
  - [Machine Learning](#machine-learning)
  - [Integrated Reactor Model](#integrated-reactor-model)
- [Workflow Examples](#workflow-examples)
- [Contributing](#contributing)

---

## Overview

This framework provides an end-to-end solution for nuclear reactor core analysis:

### Workflow Pipeline

```
┌──────────────────┐     ┌──────────────────┐     ┌────────────────────────┐
│  Core Selection  │ --> │ Integrated       │ --> │   Machine Learning     │
│  (Configuration  │     │ Reactor Model    │     │   (Prediction Models)  │
│   Sampling)      │     │ (OpenMC + T/H)   │     │                        │
└──────────────────┘     └──────────────────┘     └────────────────────────┘
        │                         │                          │
        v                         v                          v
 Optimized core           Neutronics &              Fast flux/k-eff
 configurations           thermal results            predictions
```

### Key Features

- **D4 Symmetry-Aware Sampling**: Generate unique core configurations with symmetry reduction
- **Multiple Sampling Methods**: LHS, Sobol, Halton, Greedy MaxMin, K-Means clustering
- **OpenMC Integration**: Full Monte Carlo neutron transport simulations
- **Thermal-Hydraulics Coupling**: Temperature distribution calculations
- **Depletion Analysis**: Fuel burnup calculations
- **Neural Network & Tree-Based Models**: XGBoost, Random Forest, SVM, PyTorch neural networks
- **GPU-Accelerated Training**: Ray Tune and Optuna hyperparameter optimization

---

## System Requirements

### Minimum Requirements
- Python 3.8+
- 16 GB RAM
- Multi-core CPU (8+ cores recommended)

### For Full Functionality
- OpenMC (for reactor simulations)
- CUDA-capable GPU (for neural network training)
- 64+ GB RAM (for large-scale sampling)

### Python Dependencies

```bash
# Core dependencies
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0

# Machine Learning
xgboost>=1.5.0
torch>=1.10.0
optuna>=3.0.0

# Reactor Simulation
openmc>=0.13.0

# Optional: GPU acceleration
ray[tune]>=2.0.0
```

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd workspace

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For OpenMC simulations, ensure cross-sections are available
export OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml
```

---

## Project Structure

```
workspace/
├── core_selection/           # Core configuration sampling module
│   ├── main.py              # Main workflow orchestrator
│   ├── generate_core_configurations.py
│   ├── calculate_geometric_parameters.py
│   ├── run_sampling.py
│   ├── sampling_methods/    # Sampling algorithms
│   │   ├── geometric/       # LHS, Sobol, Halton, Random
│   │   ├── lattice/         # Lattice-based methods
│   │   ├── algorithms/      # Greedy, K-Means
│   │   └── distances/       # Distance metrics
│   └── visualization_code/  # Plotting utilities
│
├── ML/                      # Machine Learning module
│   ├── main.py              # Interactive training interface
│   ├── ML_models/           # Model implementations
│   │   ├── base_model.py    # Abstract base class
│   │   ├── neural_architectures.py
│   │   ├── xgboost_train.py
│   │   ├── random_forest_train.py
│   │   ├── svm_train.py
│   │   └── encodings/       # Feature encoding methods
│   ├── execution/           # Training pipeline
│   ├── hyperparameter_tuning/
│   └── visualizations_helpers/
│
├── Integrated Reactor Model/ # OpenMC reactor simulation
│   ├── main.py              # Simulation orchestrator
│   ├── inputs.py            # Configuration parameters
│   ├── Reactor/             # Geometry and materials
│   │   ├── geometry.py
│   │   ├── materials.py
│   │   └── geometry_helpers/
│   ├── eigenvalue/          # Criticality calculations
│   │   ├── run.py
│   │   └── tallies/
│   ├── depletion/           # Fuel burnup analysis
│   ├── ThermalHydraulics/   # T/H coupling
│   ├── Inputs_GUI/          # GUI for input creation
│   ├── Parametric_GUI/      # Parametric study GUI
│   └── plotting/            # Result visualization
│
└── cross_sections/          # Nuclear data files
```

---

## Quick Start

### 1. Core Configuration Sampling

```bash
cd core_selection
python main.py
```

This launches an interactive workflow to:
- Generate all possible core configurations
- Apply D4 symmetry reduction
- Run sampling methods (LHS, Sobol, Greedy, etc.)
- Create visualizations

### 2. Run Reactor Simulations

```bash
cd "Integrated Reactor Model"
python main.py
```

Runs the complete simulation workflow:
- Geometry and materials setup
- OpenMC eigenvalue calculation
- Thermal-hydraulics analysis
- Result visualization

### 3. Train ML Models

```bash
cd ML
python main.py
```

Interactive menu for:
- Target selection (flux, k-effective)
- Model selection (XGBoost, Random Forest, Neural Net)
- Encoding methods (physics-based, one-hot, spatial)
- Hyperparameter optimization

---

## Module Documentation

### Core Selection

The Core Selection module generates and samples reactor core configurations for training data or optimization studies.

**Key Concepts:**
- **8x8 Lattice**: Reactor core represented as 8x8 grid
- **Position Types**: Fuel (F), Coolant (C), Irradiation (I)
- **D4 Symmetry**: 8-fold symmetry group (rotations + reflections)
- **4 Irradiation Positions**: Each configuration has 4 irradiation positions

**Sampling Methods:**
| Method | Space | Description |
|--------|-------|-------------|
| LHS Lattice | Configuration | Latin Hypercube in lattice space |
| Sobol Lattice | Configuration | Quasi-random Sobol sequence |
| Halton Lattice | Configuration | Quasi-random Halton sequence |
| Euclidean Greedy | Both | Maximum diversity selection |
| Jaccard Greedy | Configuration | Set-based diversity |
| K-Means | Both | Cluster-based selection |

📖 [Full Core Selection Documentation](core_selection/README.md)

---

### Machine Learning

The ML module provides multiple model architectures for predicting reactor physics parameters.

**Prediction Targets:**
- **Flux**: Neutron flux at irradiation positions (total or energy-resolved)
- **k-effective**: Core criticality value

**Models:**
- **XGBoost**: Gradient boosted trees (fast, interpretable)
- **Random Forest**: Ensemble tree method
- **Neural Network**: PyTorch-based with flexible architectures
- **SVM**: Support vector machines

**Encoding Methods:**
- **Physics-Based**: Domain-informed features (NCI, fuel density, symmetry)
- **One-Hot**: Binary encoding of cell types
- **Categorical**: Integer encoding with spatial features
- **Spatial**: 3x3 convolution patterns
- **Graph**: Network-based features

📖 [Full ML Documentation](ML/README.md)

---

### Integrated Reactor Model

The Integrated Reactor Model provides full reactor simulation capabilities using OpenMC.

**Components:**
- **Geometry**: Pin or plate fuel assembly construction
- **Materials**: Full material library with temperature dependence
- **Eigenvalue**: Monte Carlo criticality calculations
- **Thermal-Hydraulics**: Temperature distribution analysis
- **Depletion**: Fuel burnup calculations

**Key Features:**
- Parametric study support
- Fast mode for quick k-effective calculations
- Multiple irradiation experiment types (PWR loop, BWR loop, Gas capsule)
- GUI interfaces for input creation

📖 [Full Integrated Reactor Model Documentation](Integrated%20Reactor%20Model/README.md)

---

## Workflow Examples

### Example 1: Generate Training Data

```bash
# Step 1: Generate core configurations
cd core_selection
python generate_core_configurations.py

# Step 2: Sample 100 configurations
python run_sampling.py 100 --methods lhs_lattice,sobol_lattice --runs 10

# Step 3: Run simulations for each configuration
cd "../Integrated Reactor Model"
python main.py  # Configure parametric study mode

# Step 4: Train ML models
cd ../ML
python main.py  # Select models and training options
```

### Example 2: Quick Prediction

```python
from ML.predict import load_model, predict_flux

# Load trained model
model = load_model("outputs/models/xgboost_flux_physics_optuna.pkl")

# Define core configuration
lattice = [
    ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
    ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
    ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
    ['F', 'F', 'F', 'I_1', 'I_2', 'F', 'F', 'F'],
    ['F', 'F', 'F', 'I_3', 'I_4', 'F', 'F', 'F'],
    ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
    ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
    ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
]

# Predict flux
flux_values = predict_flux(model, lattice)
print(f"Predicted flux at positions: {flux_values}")
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

For questions or support, please open an issue on the repository.
