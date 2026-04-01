# Core Selection Module

A comprehensive framework for generating, sampling, and analyzing nuclear reactor core configurations with D4 symmetry awareness.

---

## Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Quick Start](#quick-start)
- [Module Architecture](#module-architecture)
- [Sampling Methods](#sampling-methods)
- [Distance Metrics](#distance-metrics)
- [Output Files](#output-files)
- [API Reference](#api-reference)
- [Examples](#examples)

---

## Overview

The Core Selection module addresses the challenge of intelligently sampling reactor core configurations for machine learning training data or optimization studies. Given the vast configuration space (270,725 possible configurations for a full 8x8 grid), efficient sampling strategies are crucial.

### Key Features

- **D4 Symmetry Reduction**: Reduces 270,725 configurations to ~34,000 unique equivalence classes
- **Multiple Sampling Methods**: 17 different algorithms across geometric and lattice spaces
- **Parallel Execution**: Multi-core support for high-throughput sampling
- **Physics-Based Parameters**: Geometric features for configuration characterization
- **Comprehensive Visualization**: Automated plot generation for analysis

---

## Key Concepts

### Core Lattice Representation

The reactor core is represented as an 8×8 grid:

```
    0   1   2   3   4   5   6   7
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
0 │ C │ C │ F │ F │ F │ F │ C │ C │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
1 │ C │ F │ F │ F │ F │ F │ F │ C │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
2 │ F │ F │ F │ F │ F │ F │ F │ F │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
3 │ F │ F │ F │ I │ I │ F │ F │ F │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
4 │ F │ F │ F │ I │ I │ F │ F │ F │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
5 │ F │ F │ F │ F │ F │ F │ F │ F │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
6 │ C │ F │ F │ F │ F │ F │ F │ C │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
7 │ C │ C │ F │ F │ F │ F │ C │ C │
  └───┴───┴───┴───┴───┴───┴───┴───┘

Legend:
  C = Coolant (fixed, 12 corner positions)
  F = Fuel assembly
  I = Irradiation position (4 per configuration)
```

### D4 Symmetry Group

The D4 symmetry group includes 8 transformations:
1. **Identity** (no change)
2. **90° rotation**
3. **180° rotation**
4. **270° rotation**
5. **Horizontal reflection**
6. **Vertical reflection**
7. **Main diagonal reflection**
8. **Anti-diagonal reflection**

Configurations related by these transformations are considered equivalent.

### Configuration Space

| Grid Type | Fuel Positions | Total Combinations | After D4 Reduction |
|-----------|----------------|--------------------|--------------------|
| Full 8×8  | 52             | 270,725            | ~34,000            |
| Central 6×6 | 36           | 58,905             | ~7,400             |

---

## Quick Start

### Complete Workflow

```bash
python main.py
```

This interactive script guides you through:
1. Configuration space selection (8×8 or 6×6)
2. Data generation (if needed)
3. Sampling method selection
4. Execution mode (sequential/parallel)
5. Visualization generation

### Individual Scripts

```bash
# Generate all configurations
python generate_core_configurations.py [--restrict-6x6]

# Calculate geometric parameters
python calculate_geometric_parameters.py [--restrict-6x6] [--full]

# Run sampling methods
python run_sampling.py <n_samples> [options]

# Visualize results
python visualize_all_samples.py
```

---

## Module Architecture

```
core_selection/
├── main.py                          # Main workflow orchestrator
├── generate_core_configurations.py  # Configuration generator
├── calculate_geometric_parameters.py # Physics parameter calculator
├── run_sampling.py                  # Sampling execution
├── visualize_all_samples.py         # Visualization generator
├── interactive_parameter_selection.py
│
├── sampling_methods/                # Sampling algorithms
│   ├── base.py                      # BaseSampler class
│   ├── symmetry_utils.py            # D4 symmetry operations
│   │
│   ├── geometric/                   # Parameter-space methods
│   │   ├── lhs.py                   # Latin Hypercube Sampling
│   │   ├── sobol.py                 # Sobol sequence
│   │   ├── halton.py                # Halton sequence
│   │   └── random_geometric.py      # Random sampling
│   │
│   ├── lattice/                     # Configuration-space methods
│   │   ├── lhs_lattice.py
│   │   ├── sobol_lattice.py
│   │   ├── halton_lattice.py
│   │   └── random_lattice.py
│   │
│   ├── algorithms/                  # Selection algorithms
│   │   ├── greedy_maxmin.py         # Greedy maximum-minimum
│   │   └── kmeans_nearest.py        # K-Means clustering
│   │
│   ├── distances/                   # Distance metrics
│   │   ├── base_distance.py
│   │   ├── euclidean_distances.py
│   │   ├── jaccard_distances.py
│   │   └── manhattan_distances.py
│   │
│   └── cache/                       # Distance caching
│       └── distance_cache.py
│
├── sampler_execution/               # Parallel execution
│   ├── parallel_executor.py
│   ├── serial_executor.py
│   └── task_runner.py
│
└── visualization_code/              # Plotting utilities
    ├── analysis_plots.py
    ├── config_visualizer.py
    └── parameter_plots.py
```

---

## Sampling Methods

### Method Categories

#### 1. Geometric/Physics-Based Methods
Sample from continuous physics parameter space, then map to nearest configuration.

| Method | Description | Use Case |
|--------|-------------|----------|
| `lhs` | Latin Hypercube Sampling | Uniform coverage of parameter space |
| `sobol` | Sobol quasi-random sequence | Low-discrepancy coverage |
| `halton` | Halton quasi-random sequence | Alternative quasi-random |
| `euclidean_geometric` | Greedy max-min in parameter space | Maximum diversity |
| `manhattan_geometric` | Manhattan distance greedy | Grid-aligned diversity |
| `jaccard_geometric` | Continuous Jaccard similarity | Feature overlap |
| `random_geometric` | Random selection | Baseline |

#### 2. Lattice-Based Methods
Sample directly from discrete configuration space.

| Method | Description | Use Case |
|--------|-------------|----------|
| `lhs_lattice` | LHS in configuration indices | Uniform index coverage |
| `sobol_lattice` | Sobol on configuration indices | Low-discrepancy indices |
| `halton_lattice` | Halton on configuration indices | Alternative indexing |
| `euclidean_lattice` | Greedy on position coordinates | Spatial diversity |
| `manhattan_lattice` | Manhattan distance on lattice | Grid-based diversity |
| `jaccard_lattice` | Jaccard distance (symmetry-aware) | Set-based diversity |
| `euclidean_lattice_kmedoids` | K-Means clustering | Representative centroids |
| `random_lattice` | Random configuration selection | Baseline |

### Method Selection Guide

```
                    ┌─────────────────────────────────┐
                    │ What's your priority?           │
                    └─────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          v                   v                   v
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │ Maximum  │        │ Uniform  │        │ Fast     │
    │ Diversity│        │ Coverage │        │ Baseline │
    └──────────┘        └──────────┘        └──────────┘
          │                   │                   │
          v                   v                   v
    euclidean_lattice    lhs_lattice         random_lattice
    jaccard_lattice      sobol_lattice       random_geometric
```

---

## Distance Metrics

### Euclidean Distance (Lattice Space)

For two configurations with irradiation positions P₁ and P₂:

```
d_euclidean = mean(min_matching_distances)
```

Uses greedy matching to pair positions between configurations.

### Jaccard Distance (Configuration Space)

For position sets A and B:

```
d_jaccard = 1 - |A ∩ B| / |A ∪ B|
```

Symmetry-aware: compares canonical forms under D4 transformations.

### Manhattan Distance

```
d_manhattan = Σ|x₁ᵢ - x₂ᵢ| + |y₁ᵢ - y₂ᵢ|
```

Grid-aligned distance metric.

---

## Physics Parameters

The following geometric/physics parameters are calculated for each configuration:

| Parameter | Description | Range |
|-----------|-------------|-------|
| `avg_distance_from_core_center` | Mean radial distance of irradiation positions | 0-5 |
| `min_inter_position_distance` | Minimum distance between any two positions | 1-7 |
| `clustering_coefficient` | Measure of position clustering | 0-1 |
| `symmetry_balance` | D4 symmetry deviation | 0-1 |
| `local_fuel_density` | Fuel fraction in neighborhood | 0-1 |
| `avg_distance_to_edge` | Mean distance to core boundary | 0-4 |

---

## Output Files

### Directory Structure

```
output/
├── data/
│   ├── core_configurations_optimized.pkl    # D4-reduced configurations
│   ├── all_configurations_before_symmetry.pkl
│   ├── physics_parameters.pkl               # Geometric features
│   └── physics_parameters_full.pkl
│
├── core_configs/
│   ├── all_configurations_before_symmetry.txt
│   ├── configurations_after_symmetry.txt
│   └── generation_summary.txt
│
└── samples_picked/
    ├── pkl/                                  # Pickle files
    │   ├── lhs_lattice_samples.pkl
    │   └── *.json                           # JSON summaries
    │
    ├── txt/                                  # Text summaries
    │   └── *_summary.txt
    │
    └── results/                              # Comparison results

visualizations/
├── lattice/                                  # Lattice method plots
├── geometric/                                # Geometric method plots
└── summary_statistics.txt
```

### Sample Result Format (JSON)

```json
{
  "method": "lhs_lattice",
  "n_samples": 16,
  "selected_indices": [123, 456, 789, ...],
  "diversity_score": 2.847,
  "best_run": 3,
  "total_runs": 10
}
```

---

## API Reference

### BaseSampler

```python
class BaseSampler:
    """Base class for all sampling methods."""
    
    def __init__(self, use_6x6_restriction=False, selected_parameters=None):
        """
        Parameters
        ----------
        use_6x6_restriction : bool
            Restrict to central 6x6 square
        selected_parameters : list
            Subset of physics parameters to use
        """
    
    def sample(self, n_samples, seed=None, **kwargs) -> Dict:
        """
        Sample n configurations.
        
        Returns
        -------
        dict
            Contains 'selected_indices', 'diversity_score', etc.
        """
    
    def save_results(self, results: Dict):
        """Save sampling results to disk."""
```

### CoreConfigGenerator

```python
class CoreConfigGenerator:
    """Generate all possible core configurations."""
    
    def __init__(self, use_6x6_restriction=False):
        """Initialize generator."""
    
    def generate_configurations(self):
        """Generate all configurations with D4 symmetry reduction."""
    
    def save_configurations(self):
        """Save to pickle and text files."""
    
    def get_canonical_form(self, positions) -> FrozenSet:
        """Get canonical representation under D4 symmetry."""
```

---

## Examples

### Example 1: Basic Sampling

```python
from sampling_methods.lattice.lhs_lattice import LHSLatticeSampler

# Create sampler
sampler = LHSLatticeSampler()

# Sample 20 configurations
results = sampler.sample(n_samples=20, seed=42)

# Access results
print(f"Selected indices: {results['selected_indices']}")
print(f"Diversity score: {results['diversity_score']:.4f}")

# Save results
sampler.save_results(results)
```

### Example 2: Custom Parameter Selection

```python
from sampling_methods.geometric.lhs import LHSSampler

# Select specific physics parameters
params = ['avg_distance_from_core_center', 'min_inter_position_distance']

# Create sampler with parameter selection
sampler = LHSSampler(selected_parameters=params)

# Sample using only these parameters
results = sampler.sample(n_samples=30, seed=123)
```

### Example 3: Greedy Maximum Diversity

```python
from sampling_methods.lattice.lhs_lattice import EuclideanLatticeSampler

# Create greedy sampler
sampler = EuclideanLatticeSampler()

# Sample with maximum spatial diversity
results = sampler.sample(n_samples=16, seed=42)

# Get configuration details
for idx in results['selected_indices']:
    positions = sampler.irradiation_sets[idx]
    print(f"Config {idx}: Positions = {positions}")
```

### Example 4: Parallel Execution

```bash
# Run multiple methods in parallel with 8 cores
python run_sampling.py 20 \
    --methods lhs_lattice,sobol_lattice,euclidean_lattice \
    --runs 10 \
    --hybrid-parallel \
    --workers 8
```

### Example 5: 6x6 Restricted Sampling

```python
from sampling_methods.lattice.lhs_lattice import LHSLatticeSampler

# Restrict to central 6x6 square (positions 1-6 in each dimension)
sampler = LHSLatticeSampler(use_6x6_restriction=True)

# Sample from reduced configuration space
results = sampler.sample(n_samples=16, seed=42)
```

---

## Performance Considerations

### Memory Usage

| Dataset | Configurations | Memory Estimate |
|---------|----------------|-----------------|
| Full 8×8 (D4 reduced) | ~34,000 | ~500 MB |
| Full 8×8 (before symmetry) | 270,725 | ~4 GB |
| Central 6×6 (D4 reduced) | ~7,400 | ~100 MB |

### Execution Time

| Method | 16 samples | 100 samples |
|--------|------------|-------------|
| LHS Lattice | <1 sec | <1 sec |
| Sobol Lattice | <1 sec | <1 sec |
| Euclidean Greedy | ~5 sec | ~30 sec |
| K-Means | ~10 sec | ~60 sec |

### Parallel Speedup

| Cores | Speedup (17 methods) |
|-------|---------------------|
| 1 | 1.0× |
| 4 | 3.5× |
| 8 | 6.5× |
| 16 | 11× |

---

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: core_configurations_optimized.pkl`
**Solution**: Run `python generate_core_configurations.py` first

**Issue**: Out of memory during full configuration generation
**Solution**: Use `--restrict-6x6` flag for smaller dataset

**Issue**: Slow distance calculations
**Solution**: Distance caching is automatic; ensure sufficient RAM

---

## References

- Latin Hypercube Sampling: McKay et al. (1979)
- Sobol Sequences: Sobol (1967)
- Greedy Maximum Diversity: Kennard-Stone algorithm
- D4 Symmetry Group: Dihedral group of order 8
