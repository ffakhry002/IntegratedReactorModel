"""
Generate all permutations of core configurations with suffix constraints.
Reads from run_dictionaries.py and creates expanded run dictionary.
"""
import numpy as np
from itertools import product
import copy

# Import the base cores from run_dictionaries
from run_dictionaries import all_runs

def generate_suffix_permutations():
    """
    Generate all valid suffix permutations with at least 2 'G' suffixes.
    Uses complement method: Total - (0 G's) - (1 G)
    Returns list of valid suffix combinations for [I_1, I_2, I_3, I_4]
    """
    suffixes = ['P', 'B', 'G']
    valid_perms = []

    # Generate ALL possible combinations (3^4 = 81)
    for perm in product(suffixes, repeat=4):
        # Count number of 'G' suffixes
        g_count = perm.count('G')
        # Keep only if at least 2 G's (complement: reject 0 or 1 G's)
        if g_count >= 2:
            valid_perms.append(perm)

    return valid_perms

def apply_suffixes_to_lattice(core_lattice, suffix_combo):
    """
    Apply suffix combination to a core lattice.
    Replaces I_1, I_2, I_3, I_4 with I_1{suffix}, etc.
    """
    new_lattice = copy.deepcopy(core_lattice)
    suffix_map = {
        'I_1': f'I_1{suffix_combo[0]}',
        'I_2': f'I_2{suffix_combo[1]}',
        'I_3': f'I_3{suffix_combo[2]}',
        'I_4': f'I_4{suffix_combo[3]}'
    }

    for i in range(len(new_lattice)):
        for j in range(len(new_lattice[i])):
            cell = new_lattice[i][j]
            if cell in suffix_map:
                new_lattice[i][j] = suffix_map[cell]

    return new_lattice

def generate_all_permutations():
    """
    Generate all permuted runs from base cores.
    """
    suffix_perms = generate_suffix_permutations()
    print(f"Generated {len(suffix_perms)} valid suffix permutations (at least 2 G's)")
    print(f"Verification: 3^4 = 81 total, rejecting {81 - len(suffix_perms)} with 0 or 1 G")

    expanded_runs = []

    # Add default run
    expanded_runs.append({"description": "Default inputs"})

    # Process each base core (skip the default at index 0)
    for base_run in all_runs[1:]:
        if 'core_lattice' in base_run:
            base_desc = base_run['description']
            core_lattice = base_run['core_lattice']

            # Generate all permutations for this core
            for perm_idx, suffix_combo in enumerate(suffix_perms, start=1):
                new_run = {
                    'description': f'{base_desc} --- permutation {perm_idx}',
                    'core_lattice': apply_suffixes_to_lattice(core_lattice, suffix_combo)
                }
                expanded_runs.append(new_run)

    return expanded_runs

def save_to_file(expanded_runs, filename='run_dictionaries_expanded.py'):
    """
    Save expanded runs to a new Python file.
    """
    with open(filename, 'w') as f:
        f.write('"""\n')
        f.write('Run dictionaries for parametric studies with suffix permutations.\n')
        f.write('Each dictionary entry represents a set of parameters to modify from the base inputs.\n')
        f.write('Generated with at least 2 G suffixes per core configuration.\n')
        f.write('"""\n')
        f.write('import numpy as np\n\n')
        f.write('# Expanded run dictionary with all permutations\n')
        f.write('all_runs = [\n')

        for idx, run in enumerate(expanded_runs):
            if idx == 0:
                # Default run
                f.write('    {\n')
                f.write(f'        "description": "{run["description"]}"\n')
                f.write('    }')
            else:
                f.write('\n    ,{\n')
                f.write(f'        "description": "{run["description"]}",\n')
                f.write('        "core_lattice": [\n')

                for row in run['core_lattice']:
                    f.write(f'            {row},\n')

                f.write('        ]\n')
                f.write('    }')

        f.write('\n]\n')

    print(f"\nSaved to {filename}")

# Generate all permutations
all_runs_expanded = generate_all_permutations()
print(f"\nTotal runs: {len(all_runs_expanded)}")
base_core_count = len([r for r in all_runs[1:] if 'core_lattice' in r])
print(f"Base cores: {base_core_count}")
if base_core_count > 0:
    print(f"Permutations per core: {(len(all_runs_expanded) - 1) // base_core_count}")

# Save to file
save_to_file(all_runs_expanded, 'run_dictionaries_expanded.py')
