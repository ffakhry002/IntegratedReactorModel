"""
Run dictionaries for parametric studies with suffix permutations.
Each dictionary entry represents a set of parameters to modify from the base inputs.
Generated with at least 2 G suffixes per core configuration.
"""
import numpy as np

# Expanded run dictionary with all permutations
cluster_1 = [
    {
        "description": "Complete cluster --- permutation 1",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1P', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2P', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 2",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1P', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2B', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 3",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1P', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3P', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 4",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1P', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3B', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 5",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1P', 'I_4P', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 6",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1P', 'I_4B', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 7",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1P', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 8",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1B', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2P', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 9",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1B', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2B', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 10",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1B', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3P', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 11",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1B', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3B', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 12",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1B', 'I_4P', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 13",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1B', 'I_4B', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 14",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1B', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 15",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2P', 'I_3P', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 16",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2P', 'I_3B', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 17",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4P', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2P', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 18",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4B', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2P', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 19",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2P', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 20",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2B', 'I_3P', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 21",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2B', 'I_3B', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 22",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4P', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2B', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 23",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4B', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2B', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 24",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2B', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 25",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4P', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3P', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 26",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4B', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3P', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 27",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3P', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 28",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4P', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3B', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 29",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4B', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3B', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 30",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3B', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 31",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4P', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 32",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4B', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Complete cluster --- permutation 33",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_4G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'I_3G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
]

semi_spaced_1 = [
    {
        "description": "Semi Spaced  --- permutation 1",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1P', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2P', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 2",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1P', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2B', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 3",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1P', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3P', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 4",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1P', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3B', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 5",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1P', 'F', 'I_4P', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 6",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1P', 'F', 'I_4B', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 7",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1P', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 8",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1B', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2P', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 9",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1B', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2B', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 10",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1B', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3P', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 11",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1B', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3B', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 12",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1B', 'F', 'I_4P', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 13",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1B', 'F', 'I_4B', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 14",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1B', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 15",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2P', 'F', 'I_3P', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 16",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2P', 'F', 'I_3B', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 17",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4P', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2P', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 18",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4B', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2P', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 19",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2P', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 20",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2B', 'F', 'I_3P', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 21",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2B', 'F', 'I_3B', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 22",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4P', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2B', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 23",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4B', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2B', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 24",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2B', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 25",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4P', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3P', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 26",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4B', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3P', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 27",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3P', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 28",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4P', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3B', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 29",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4B', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3B', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 30",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3B', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 31",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4P', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 32",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4B', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Semi Spaced  --- permutation 33",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'I_1G', 'F', 'I_4G', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_2G', 'F', 'I_3G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
]

full_spaced_1 = [
    {
        "description": "Full Spaced  --- permutation 1",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2P', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1P', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 2",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2B', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1P', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 3",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3P', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1P', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 4",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3B', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1P', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 5",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1P', 'F', 'F', 'I_4P', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 6",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1P', 'F', 'F', 'I_4B', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 7",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1P', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 8",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2P', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1B', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 9",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2B', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1B', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 10",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3P', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1B', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 11",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3B', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1B', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 12",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1B', 'F', 'F', 'I_4P', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 13",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1B', 'F', 'F', 'I_4B', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 14",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1B', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 15",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2P', 'F', 'F', 'I_3P', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 16",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2P', 'F', 'F', 'I_3B', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 17",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2P', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4P', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 18",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2P', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4B', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 19",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2P', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 20",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2B', 'F', 'F', 'I_3P', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 21",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2B', 'F', 'F', 'I_3B', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 22",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2B', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4P', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 23",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2B', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4B', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 24",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2B', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 25",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3P', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4P', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 26",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3P', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4B', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 27",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3P', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 28",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3B', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4P', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 29",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3B', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4B', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 30",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3B', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 31",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4P', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 32",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4B', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
    ,{
        "description": "Full Spaced  --- permutation 33",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_2G', 'F', 'F', 'I_3G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_4G', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
]

p_spot = [
    {
        "description": "P, 1 spot, P",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1P', 'I_2P', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
        {
        "description": "P, 1 spot, B",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1P', 'I_2B', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
            {
        "description": "P, 1 spot, G",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1P', 'I_2G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
                {
        "description": "P, 2 spot, P",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1P', 'F', 'I_2P', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
                    {
        "description": "P, 2 spot, B",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1P', 'F', 'I_2B', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
                        {
        "description": "P, 2 spot, G",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1P', 'F', 'I_2G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
        {
        "description": "P, 3 spot, P",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1P', 'F', 'F', 'I_2P', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },    {
        "description": "P, 3 spot, B",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1P', 'F', 'F', 'I_2B', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
        {
        "description": "P, 3 spot, G",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1P', 'F', 'F', 'I_2G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
]

nci_runs = [
    {
        "description": "B, 1 spot, P",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1B', 'I_2P', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
        {
        "description": "B, 1 spot, B",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1B', 'I_2B', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
            {
        "description": "B, 1 spot, G",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1B', 'I_2G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
                {
        "description": "B, 2 spot, P",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1B', 'F', 'I_2P', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
                    {
        "description": "B, 2 spot, B",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1B', 'F', 'I_2B', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
                        {
        "description": "B, 2 spot, G",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1B', 'F', 'I_2G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
        {
        "description": "B, 3 spot, P",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1B', 'F', 'F', 'I_2P', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },    {
        "description": "B, 3 spot, B",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1B', 'F', 'F', 'I_2B', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
        {
        "description": "B, 3 spot, G",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1B', 'F', 'F', 'I_2G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
]

g_spot = [
    {
        "description": "G, 1 spot, P",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_2P', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
        {
        "description": "G, 1 spot, B",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_2B', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
            {
        "description": "G, 1 spot, G",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'I_2G', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
                {
        "description": "G, 2 spot, P",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'F', 'I_2P', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
                    {
        "description": "G, 2 spot, B",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'F', 'I_2B', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
                        {
        "description": "G, 2 spot, G",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'F', 'I_2G', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
        {
        "description": "G, 3 spot, P",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_2P', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },    {
        "description": "G, 3 spot, B",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_2B', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    },
        {
        "description": "G, 3 spot, G",
        "core_lattice": [
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'I_1G', 'F', 'F', 'I_2G', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ]
    }
]
all_runs = cluster_1
