#!/usr/bin/env python3
"""
Split runs into 10 groups based on their euclidean_geometric index.
"""

# Read the input file
input_file = 'run_dictionaries_expanded.py'
output_file = 'run_dictionaries.py'

with open(input_file, 'r') as f:
    content = f.read()

# Extract individual run dictionaries
runs = []
current_run = []
brace_depth = 0
in_run = False

for line in content.split('\n'):
    # Skip the import line and all_runs line
    if 'import numpy' in line or line.strip().startswith('all_runs'):
        continue
    
    # Track when we enter a dictionary
    if '{' in line and not in_run:
        in_run = True
        brace_depth = 0
    
    if in_run:
        current_run.append(line)
        brace_depth += line.count('{') - line.count('}')
        
        # When we've closed all braces, we have a complete run
        if brace_depth == 0:
            runs.append('\n'.join(current_run))
            current_run = []
            in_run = False

# Filter out the default run and get only euclidean runs
euclidean_runs = []

for run in runs:
    if 'euclidean_geometric' in run:
        euclidean_runs.append(run)

print(f"Found {len(euclidean_runs)} euclidean runs")

# Split into 25 groups
n_groups = 25
runs_per_group = len(euclidean_runs) // n_groups
remainder = len(euclidean_runs) % n_groups

# Generate output file
with open(output_file, 'w') as f:
    f.write('"""\n')
    f.write('Run dictionaries split into 25 groups.\n')
    f.write('"""\n')
    f.write('import numpy as np\n\n')
    
    for i in range(n_groups):
        start_idx = i * runs_per_group + min(i, remainder)
        end_idx = (i + 1) * runs_per_group + min(i + 1, remainder)
        
        f.write(f'runs_{i+1} = [\n')
        
        for j, run in enumerate(euclidean_runs[start_idx:end_idx]):
            # Strip leading comma and whitespace from the run if present
            run = run.strip()
            if run.startswith(','):
                run = run[1:].strip()
            
            # Add comma before each entry except the first
            if j > 0:
                f.write('    ,\n')
            
            # Write the run with proper indentation
            run_lines = run.split('\n')
            for k, line in enumerate(run_lines):
                if k == 0:
                    f.write('    ' + line.lstrip() + '\n')
                else:
                    f.write('    ' + line + '\n')
        
        f.write(']\n\n')
    
    # Add the all_runs assignment at the end
    f.write('# Change this to runs_1, runs_2, etc. to select which group to run\n')
    f.write('all_runs = runs_1\n')

print(f"Split {len(euclidean_runs)} runs into {n_groups} groups")
print(f"Output written to: {output_file}")
print(f"Runs per group: {[runs_per_group + (1 if i < remainder else 0) for i in range(n_groups)]}")