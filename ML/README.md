# Machine Learning Module

A comprehensive machine learning framework for predicting nuclear reactor physics parameters (neutron flux, k-effective) from core configurations.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Module Architecture](#module-architecture)
- [Models](#models)
- [Encoding Methods](#encoding-methods)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Training Pipeline](#training-pipeline)
- [Prediction](#prediction)
- [API Reference](#api-reference)
- [Examples](#examples)

---

## Overview

The ML module provides a complete pipeline for training machine learning models to predict reactor physics parameters from core configuration lattices. This enables fast evaluation of new configurations without expensive Monte Carlo simulations.

### Prediction Targets

| Target | Description | Output Shape |
|--------|-------------|--------------|
| **Flux** | Neutron flux at irradiation positions | 4 values (total) or 12 values (energy-resolved) |
| **k-effective** | Core criticality (multiplication factor) | 1 value |

### Supported Models

- **XGBoost**: Gradient boosted decision trees
- **Random Forest**: Ensemble of decision trees
- **Neural Network**: PyTorch-based flexible architectures
- **SVM**: Support Vector Machines with RBF kernel

---

## Key Features

- **Multiple Encoding Methods**: Physics-based, one-hot, categorical, spatial, graph
- **Hyperparameter Optimization**: Optuna, Ray Tune, Three-Stage optimization
- **GPU Acceleration**: CUDA support for neural network training
- **Energy-Resolved Predictions**: Thermal, epithermal, and fast flux groups
- **Symmetry-Aware Augmentation**: D4 symmetry for data augmentation
- **Interactive Training**: Menu-driven training interface
- **Comprehensive Logging**: Training logs and result summaries

---

## Quick Start

### Interactive Training

```bash
cd ML
python main.py
```

This launches an interactive menu where you can:
1. Select prediction targets (flux, k-effective)
2. Choose models to train
3. Select encoding methods
4. Configure hyperparameter optimization
5. Set parallel computing options

### Command Line Prediction

```python
from ML.predict import load_and_predict

# Load trained model and predict
flux = load_and_predict(
    model_path="outputs/models/xgboost_flux_physics_optuna.pkl",
    lattice=[
        ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'I_1', 'I_2', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'I_3', 'I_4', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
    ]
)
print(f"Predicted flux: {flux}")
```

---

## Module Architecture

```
ML/
├── main.py                          # Interactive training entry point
├── predict.py                       # Prediction utilities
│
├── execution/                       # Training pipeline
│   ├── config.py                    # Configuration dataclass
│   ├── data_handler.py              # Data loading and preparation
│   ├── interactive_menu.py          # Interactive UI
│   ├── model_trainer.py             # Training orchestration
│   └── results_manager.py           # Results saving and display
│
├── ML_models/                       # Model implementations
│   ├── base_model.py                # Abstract base class
│   ├── xgboost_train.py            # XGBoost implementation
│   ├── random_forest_train.py       # Random Forest implementation
│   ├── svm_train.py                 # SVM implementation
│   ├── neural_net_train.py          # Neural network implementation
│   ├── neural_architectures.py      # PyTorch architectures
│   │
│   └── encodings/                   # Feature encoding
│       ├── encoding_methods.py      # All encoding implementations
│       └── test_edge_distance.py
│
├── hyperparameter_tuning/           # Optimization methods
│   ├── optuna_optimization.py       # Optuna Bayesian optimization
│   ├── raytune_neural_net.py        # Ray Tune for neural nets
│   ├── three_stage_optimization.py  # Three-stage for sklearn models
│   ├── three_stage_neural_net_gpu.py # Three-stage for neural nets
│   └── lambda_aware_estimator.py    # Lambda parameter optimization
│
├── data/                            # Training data
│   ├── train.txt                    # Training configurations
│   ├── test.txt                     # Test configurations
│   └── all_reactor_configurations_D4.txt
│
├── test_execution/                  # Model testing
│   ├── main.py
│   ├── model_tester.py
│   └── excel_reporter.py
│
├── utils/                           # Utilities
│   ├── txt_to_data.py              # Data file parsing
│   ├── log_structure.py            # Logging setup
│   ├── log_viewer.py
│   └── lambda_feature_regenerator.py
│
└── visualizations_helpers/          # Visualization
    ├── config_error_plots.py
    ├── feature_importance.py
    ├── performance_heatmaps.py
    ├── optuna_visualizations.py
    └── spatial_error_heatmaps.py
```

---

## Models

### XGBoost

Gradient boosted decision trees optimized for tabular data.

**Hyperparameters:**
| Parameter | Default | Range |
|-----------|---------|-------|
| `n_estimators` | 100 | 50-500 |
| `max_depth` | 6 | 3-12 |
| `learning_rate` | 0.1 | 0.01-0.3 |
| `subsample` | 0.8 | 0.5-1.0 |
| `colsample_bytree` | 0.8 | 0.5-1.0 |

**Best for:** General prediction, fast training, interpretability

### Random Forest

Ensemble of decision trees with bagging.

**Hyperparameters:**
| Parameter | Default | Range |
|-----------|---------|-------|
| `n_estimators` | 100 | 50-500 |
| `max_depth` | None | 5-50 |
| `min_samples_split` | 2 | 2-20 |
| `min_samples_leaf` | 1 | 1-10 |

**Best for:** Robust predictions, feature importance analysis

### Neural Network

PyTorch-based flexible architecture supporting multiple patterns.

**Architecture Types:**
- `rectangular`: Uniform width across layers
- `pyramidal`: Gradually narrowing layers
- `funnel`: Aggressive narrowing (encoder-style)
- `hourglass`: Narrow in middle, wide at ends
- `bottleneck`: Wide-narrow-wide (autoencoder-style)

**Hyperparameters:**
| Parameter | Options |
|-----------|---------|
| `base_width` | 64-512 |
| `depth` | 2-8 layers |
| `activation` | relu, elu, tanh, gelu, selu |
| `dropout_rate` | 0.0-0.5 |
| `learning_rate` | 1e-5 to 1e-2 |
| `batch_size` | 16-128 |

**Best for:** Complex patterns, large datasets, GPU acceleration

### SVM

Support Vector Machine with RBF kernel.

**Hyperparameters:**
| Parameter | Default | Range |
|-----------|---------|-------|
| `C` | 1.0 | 0.1-1000 |
| `gamma` | 'scale' | 1e-4 to 1 |
| `epsilon` | 0.1 | 0.01-0.5 |

**Best for:** Small datasets, when margin-based learning is preferred

---

## Encoding Methods

Feature engineering is critical for reactor ML. The module provides five encoding methods:

### 1. Physics-Based Encoding (Recommended)

Domain-informed features capturing reactor physics relationships.

**Features:**
- **Global Features** (2):
  - Average distance between irradiation positions
  - Symmetry balance (D4 deviation)
  
- **Local Features per Position** (4 each):
  - Local fuel density (3×3 neighborhood)
  - Coolant contact (adjacent coolant cells)
  - Edge distance (distance to core boundary)
  - Center distance (distance from core center)

- **Neighbor Contribution Index (NCI)** (4):
  - Weighted sum of fuel contributions with exponential decay
  - λ parameter controls decay rate

**Total Features:** 22 (vacuum mode) or 27+ (fill mode)

```python
from ML.ML_models.encodings.encoding_methods import ReactorEncodings

features, irr_positions, position_order = ReactorEncodings.physics_based_encoding(
    lattice,
    lambda_decay=1.5  # NCI decay parameter
)
```

### 2. One-Hot Encoding

Binary encoding of cell types with position features.

**Features:**
- Cell type one-hot (3 bits per cell): Fuel, Coolant, Irradiation
- Normalized position coordinates

**Total Features:** 8×8×3 + 8×8×2 = 320

### 3. Categorical Encoding

Integer encoding with radial distance features.

**Features:**
- Cell type integer (0-2 per cell)
- Radial distance from center

**Total Features:** 8×8×2 = 128

### 4. Spatial Encoding

3×3 convolution patterns capturing local structure.

**Features:**
- 3×3 neighborhood encoding for each cell
- Pattern-based features

**Total Features:** Variable based on patterns

### 5. Graph-Based Encoding

Network features treating core as a graph.

**Features:**
- Node degree
- Clustering coefficient
- Betweenness centrality
- Path lengths

**Total Features:** Variable based on graph metrics

### Encoding Comparison

| Encoding | Features | Training Speed | Accuracy | Interpretability |
|----------|----------|----------------|----------|------------------|
| Physics | ~22 | ★★★★★ | ★★★★★ | ★★★★★ |
| One-Hot | 320 | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ |
| Categorical | 128 | ★★★★☆ | ★★★☆☆ | ★★★☆☆ |
| Spatial | ~150 | ★★★☆☆ | ★★★★☆ | ★★★☆☆ |
| Graph | ~50 | ★★★★☆ | ★★★★☆ | ★★★★☆ |

---

## Hyperparameter Optimization

### Optuna (Recommended)

Bayesian optimization with Tree-structured Parzen Estimator (TPE).

```python
# Configuration
n_trials = 250        # Number of optimization trials
n_jobs = -1           # Use all CPU cores
```

**Features:**
- Automatic hyperparameter importance analysis
- Early stopping of poor trials (pruning)
- Visualization of optimization history

### Ray Tune (Neural Networks + GPU)

Distributed hyperparameter tuning with GPU support.

```python
# Configuration
n_gpus = 2            # Number of GPUs
trials_per_gpu = 2    # Concurrent trials per GPU
```

**Features:**
- Multi-GPU parallelization
- ASHA scheduler for early stopping
- Population-based training

### Three-Stage Optimization

Hybrid approach: Random → Grid → Bayesian

**Stages:**
1. **Random Search** (2000 iterations): Explore parameter space
2. **Grid Search**: Refine around best parameters
3. **Bayesian Optimization** (100 iterations): Fine-tune

**Best for:** When you want to ensure thorough exploration

---

## Training Pipeline

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Loading                                 │
│  train.txt → Parse lattices → Extract flux/k-eff targets            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────┐
│                         Encoding                                     │
│  Lattice → Physics/One-Hot/Graph features → Feature matrix          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Splitting                               │
│  Group-aware split (configurations with same canonical form)        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────┐
│                    Hyperparameter Optimization                       │
│  Optuna/Ray Tune/Three-Stage → Best parameters                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────┐
│                         Model Training                               │
│  Train with best params → Evaluate on test set                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────┐
│                         Model Saving                                 │
│  Save model + metadata + results                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Training Data Format

```text
# train.txt format
Lattice:
C C F F F F C C
C F F F F F F C
F F F F F F F F
F F F I_1 I_2 F F F
F F F I_3 I_4 F F F
F F F F F F F F
C F F F F F F C
C C F F F F C C

Simulation Results:
k-effective: 1.05432 +/- 0.00023
Position 1 Flux: 2.34e14 [thermal: 45.2%, epithermal: 32.1%, fast: 22.7%]
Position 2 Flux: 2.56e14 [thermal: 43.1%, epithermal: 33.5%, fast: 23.4%]
Position 3 Flux: 2.41e14 [thermal: 44.8%, epithermal: 32.8%, fast: 22.4%]
Position 4 Flux: 2.38e14 [thermal: 45.5%, epithermal: 31.9%, fast: 22.6%]
---
```

### Flux Modes

| Mode | Description | Output Dimension |
|------|-------------|------------------|
| `total` | Total integrated flux | 4 (one per position) |
| `energy` | Absolute flux per energy group | 12 (3 groups × 4 positions) |
| `bin` | Percentage distribution | 12 (3 groups × 4 positions) |
| `thermal_only` | Only thermal flux | 4 |
| `epithermal_only` | Only epithermal flux | 4 |
| `fast_only` | Only fast flux | 4 |

---

## Prediction

### Loading a Trained Model

```python
import joblib

# Load model with metadata
data = joblib.load("outputs/models/xgboost_flux_physics_optuna.pkl")

model = data['model']
encoding = data['encoding']
flux_scale = data['flux_scale']
flux_mode = data['flux_mode']
```

### Making Predictions

```python
from ML.ML_models.encodings.encoding_methods import ReactorEncodings
import numpy as np

# Define lattice
lattice = np.array([
    ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
    ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
    ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
    ['F', 'F', 'F', 'I_1', 'I_2', 'F', 'F', 'F'],
    ['F', 'F', 'F', 'I_3', 'I_4', 'F', 'F', 'F'],
    ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
    ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
    ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
])

# Encode features
features, _, _ = ReactorEncodings.physics_based_encoding(lattice)

# Predict
flux_normalized = model.predict(features.reshape(1, -1))
flux_actual = flux_normalized * flux_scale  # Denormalize
```

---

## API Reference

### ReactorModelBase

```python
class ReactorModelBase(ABC):
    """Abstract base class for all reactor ML models."""
    
    @abstractmethod
    def fit_flux(self, X_train, y_flux):
        """Train flux prediction model."""
        pass
    
    @abstractmethod
    def fit_keff(self, X_train, y_keff):
        """Train k-effective prediction model."""
        pass
    
    @abstractmethod
    def predict_flux(self, X_test):
        """Predict flux values."""
        pass
    
    @abstractmethod
    def predict_keff(self, X_test):
        """Predict k-effective."""
        pass
    
    def save_model(self, filepath, model_type, encoding, optimization_method, **kwargs):
        """Save model with metadata."""
        pass
    
    @classmethod
    def load_model(cls, filepath):
        """Load model from file."""
        pass
```

### DataHandler

```python
class DataHandler:
    """Handle data loading and preparation."""
    
    def load_and_prepare_data(self, data_file, encoding, flux_mode='total'):
        """
        Load and encode training data.
        
        Returns
        -------
        X : np.ndarray
            Feature matrix
        y_flux : np.ndarray
            Flux targets
        y_keff : np.ndarray
            k-effective targets
        groups : np.ndarray
            Configuration group IDs for splitting
        lattices : list
            Original lattice arrays
        """
        pass
    
    def split_data(self, X, y_flux, y_keff, groups, test_size=0.2):
        """
        Split data with group-aware stratification.
        
        Returns
        -------
        dict
            Contains X_train, X_test, y_flux_train, y_flux_test, etc.
        """
        pass
```

### FlexibleNeuralNet

```python
class FlexibleNeuralNet(nn.Module):
    """Flexible neural network with multiple architecture patterns."""
    
    def __init__(
        self,
        input_dim,
        output_dim,
        architecture_type='rectangular',  # pyramidal, funnel, hourglass, bottleneck
        base_width=100,
        depth=2,
        activations='relu',  # or list of activations
        dropout_rate=0.0,
        use_batch_norm=False
    ):
        pass
    
    def forward(self, x):
        """Forward pass."""
        pass
    
    def get_architecture_info(self):
        """Return architecture details."""
        pass
```

---

## Examples

### Example 1: Train XGBoost Model

```python
from ML.execution.data_handler import DataHandler
from ML.ML_models.xgboost_train import XGBoostModel
from ML.hyperparameter_tuning.optuna_optimization import OptunaOptimizer

# Load data
handler = DataHandler()
X, y_flux, y_keff, groups, lattices = handler.load_and_prepare_data(
    "data/train.txt",
    encoding="physics",
    flux_mode="total"
)

# Split data
splits = handler.split_data(X, y_flux, y_keff, groups)

# Optimize hyperparameters
optimizer = OptunaOptimizer(n_trials=100, n_jobs=-1)
best_params = optimizer.optimize(
    XGBoostModel,
    splits['X_train'],
    splits['y_flux_train'],
    target='flux'
)

# Train final model
model = XGBoostModel(**best_params)
model.fit_flux(splits['X_train'], splits['y_flux_train'])

# Evaluate
predictions = model.predict_flux(splits['X_test'])
mse = np.mean((predictions - splits['y_flux_test'])**2)
print(f"Test MSE: {mse:.6f}")

# Save model
model.save_model(
    "outputs/models/xgboost_flux_physics_optuna.pkl",
    model_type='flux',
    encoding='physics',
    optimization_method='optuna'
)
```

### Example 2: Train Neural Network with GPU

```python
import torch
from ML.ML_models.neural_net_train import NeuralNetModel
from ML.hyperparameter_tuning.raytune_neural_net import RayTuneOptimizer

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Optimize with Ray Tune
optimizer = RayTuneOptimizer(
    n_trials=100,
    n_gpus=2,
    trials_per_gpu=2
)

best_params = optimizer.optimize(
    NeuralNetModel,
    X_train, y_flux_train,
    target='flux'
)

# Train with best parameters
model = NeuralNetModel(**best_params, device=device)
model.fit_flux(X_train, y_flux_train)
```

### Example 3: Compare Encodings

```python
from ML.ML_models.encodings.encoding_methods import ReactorEncodings
import numpy as np

lattice = np.array([...])  # Your lattice

# Physics-based (recommended)
physics_features, _, _ = ReactorEncodings.physics_based_encoding(lattice)
print(f"Physics features: {len(physics_features)}")

# One-hot
onehot_features, _, _ = ReactorEncodings.one_hot_encoding(lattice)
print(f"One-hot features: {len(onehot_features)}")

# Graph-based
graph_features, _, _ = ReactorEncodings.graph_based_encoding(lattice)
print(f"Graph features: {len(graph_features)}")
```

### Example 4: Lambda Parameter Optimization

```python
from ML.hyperparameter_tuning.lambda_aware_estimator import LambdaAwareEstimator

# Create estimator that optimizes lambda for NCI features
estimator = LambdaAwareEstimator(
    base_model_class=XGBoostModel,
    lambda_range=(0.5, 3.0)
)

# Fit with lambda optimization
estimator.fit(X_train, y_train, lattices=train_lattices)

# Best lambda value
print(f"Optimal lambda: {estimator.best_lambda_}")
```

---

## Output Files

### Model Files

```
outputs/
├── models/
│   ├── xgboost_flux_physics_optuna.pkl
│   ├── random_forest_keff_physics_optuna.pkl
│   └── neural_net_flux_energy_physics_raytune.pkl
│
├── results/
│   ├── training_results_complete_20240101_120000.json
│   └── training_summary_complete_20240101_120000.txt
│
├── logs/
│   └── training_20240101_120000.log
│
└── excel_reports/
    ├── train_data_20240101.xlsx
    └── test_results_20240101.xlsx
```

### Model Metadata

Each saved model includes:
```python
{
    'model': <trained_model>,
    'model_class': 'xgboost',
    'model_type': 'flux',
    'encoding': 'physics',
    'optimization_method': 'optuna',
    'params': {...},
    'flux_scale': 1e14,
    'use_log_flux': False,
    'flux_mode': 'total',
    'saved_at': '2024-01-01T12:00:00'
}
```

---

## Performance Tips

### GPU Training

```bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0,1

# Enable TF32 for faster training (Ampere+ GPUs)
export TORCH_ALLOW_TF32_CUBLAS=1
```

### Memory Optimization

```python
# For large datasets, use data streaming
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
```

### Parallel Training

```python
# XGBoost with optimal core allocation
model = XGBoostModel(
    n_jobs=8,                    # Cores per tree
    tree_method='hist',          # Fast histogram algorithm
    predictor='cpu_predictor'
)
```

---

## Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
**Solution**: Reduce batch size or use gradient accumulation

**Issue**: Poor prediction accuracy
**Solution**: 
1. Check data quality
2. Try physics-based encoding
3. Increase optimization trials
4. Ensure proper train/test split (group-aware)

**Issue**: Training takes too long
**Solution**:
1. Use GPU for neural networks
2. Reduce Optuna trials
3. Use XGBoost (faster than Random Forest)

---

## References

- XGBoost: Chen & Guestrin (2016)
- Optuna: Akiba et al. (2019)
- Ray Tune: Liaw et al. (2018)
- Physics-Informed ML: Karniadakis et al. (2021)
