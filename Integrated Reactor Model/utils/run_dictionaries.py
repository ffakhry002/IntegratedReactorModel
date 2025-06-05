"""
Run dictionaries for parametric studies.
Each dictionary entry represents a set of parameters to modify from the base inputs.
"""

import numpy as np

# Example parametric runs - modify these as needed for your studies
all_runs = [
    {
        "description": "Default inputs"
    },

    {
        "fuel_height": 0.2,
        "description": "Fuel height 0.2 m",
    },

    {
        "fuel_type": "UO2",
        "description": "UO2 fuel",
    },

    {
        "n%": 50.0,
        "description": "50% enriched fuel",
    },

    {
        "fuel_type": "UO2",
        "n%": 50.0,
        "description": "50% enriched UO2 fuel",
    },

    {
        "fuel_height": 1.0,
        "description": "Fuel height 1.0 m",
    },

    {
        "deplete_core": True,
        "fuel_height": 1.0,
    },

    {
        "fuel_height": 0.2,
        "core_lattice": [  # C: coolant, F: fuel assembly, E: enriched fuel assembly, I: irradiation position
        ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C']
        ],
        "description": "Core lattice with 0.2 m fuel height",
    }
]

# ==============================================================================
# COMPLEX PARAMETRIC STUDY EXAMPLES USING LOOPS
# ==============================================================================
# Uncomment and modify any of the sections below to use more comprehensive
# parametric studies generated programmatically

# Example 1: Enrichment sweep
"""
enrichment_values = [5.0, 10.0, 15.0, 19.75, 25.0, 30.0]
enrichment_runs = []
for enrich in enrichment_values:
    enrichment_runs.append({
        "n%": enrich,
        "description": f"Enrichment {enrich}% study"
    })
# all_runs = enrichment_runs
"""

# Example 2: Multi-parameter study (enrichment + power combinations)
"""
enrichments = [10.0, 19.75, 30.0]
powers = [10, 15, 20]
combo_runs = []
for enrich in enrichments:
    for power in powers:
        combo_runs.append({
            "n%": enrich,
            "core_power": power,
            "description": f"Enrichment {enrich}% + Power {power} MW"
        })
# all_runs = combo_runs
"""

# Example 3: Coolant and material combinations
"""
coolants = ["Light Water", "Heavy Water"]
clad_materials = ["Al6061", "Zirc2", "Zirc4"]
fuel_types = ["U3Si2", "UO2", "U10Mo"]

material_runs = []
for coolant in coolants:
    for clad in clad_materials:
        for fuel in fuel_types:
            material_runs.append({
                "coolant_type": coolant,
                "clad_type": clad,
                "fuel_type": fuel,
                "description": f"{coolant} + {clad} clad + {fuel} fuel"
            })
# all_runs = material_runs
"""

# Example 4: Statistical sampling study (Latin Hypercube or random sampling)
"""
import random
np.random.seed(42)  # For reproducible results

# Define parameter ranges for sampling
param_ranges = {
    "n%": (10.0, 30.0),
    "core_power": (8, 25),
    "fuel_height": (0.4, 0.8),
    "flow_rate": (2.0, 4.0),
    "T_inlet": (310, 340)
}

# Generate random samples
n_samples = 20
sampling_runs = []
for i in range(n_samples):
    run_dict = {"description": f"Random sample {i+1}"}
    for param, (min_val, max_val) in param_ranges.items():
        if param in ["core_power"]:
            # Integer parameters
            run_dict[param] = random.randint(int(min_val), int(max_val))
        else:
            # Float parameters
            run_dict[param] = random.uniform(min_val, max_val)
    sampling_runs.append(run_dict)
# all_runs = sampling_runs
"""
