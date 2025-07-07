#!/usr/bin/env python3
"""
Script to process parametric study log files.
Renumbers runs starting from 1 and keeps only description and core_lattice data.
"""

import re
import sys

def process_log_file(input_file, output_file):
    """
    Process the parametric study log file according to specifications.

    Args:
        input_file: Path to input text file
        output_file: Path to output text file
    """
    with open(input_file, 'r') as f:
        content = f.read()

    # Split content into sections by RUN headers
    run_pattern = r'RUN \d+:'
    run_splits = re.split(f'({run_pattern})', content)

    # Find the header (everything before first RUN)
    if len(run_splits) > 0:
        header = run_splits[0]
    else:
        header = ""

    # Process runs
    processed_content = header
    run_number = 1

    # Iterate through run headers and their content
    for i in range(1, len(run_splits), 2):
        if i + 1 < len(run_splits):
            run_header = run_splits[i]
            run_content = run_splits[i + 1]

            # Replace run number
            new_run_header = f'RUN {run_number}:'

            # Extract description and core_lattice
            description_match = re.search(r'Description: (.+?)(?:\n|Timestamp)', run_content, re.DOTALL)
            description = description_match.group(1).strip() if description_match else ""

            # Extract core_lattice - handle both single line and multi-line formats
            core_lattice_match = re.search(r'core_lattice: (.+?)(?=\n(?:Success:|Timestamp:|Modified Parameters:|RUN \d+:|$))', run_content, re.DOTALL)

            if core_lattice_match:
                core_lattice = core_lattice_match.group(1).strip()
                # Clean up any extra whitespace while preserving the list structure
                core_lattice = re.sub(r'\s+', ' ', core_lattice)
                # Fix spacing around brackets and commas for readability
                core_lattice = re.sub(r'\s*,\s*', ', ', core_lattice)
                core_lattice = re.sub(r'\[\s+', '[', core_lattice)
                core_lattice = re.sub(r'\s+\]', ']', core_lattice)
            else:
                core_lattice = ""

            # Construct new run section
            new_run_section = f"\n{new_run_header}\n"
            new_run_section += "----------------------------------------\n"
            new_run_section += f"Description: {description}\n"
            new_run_section += f"  core_lattice: {core_lattice}\n"
            new_run_section += "\n" + "="*80 + "\n"

            processed_content += new_run_section
            run_number += 1

    # Write output
    with open(output_file, 'w') as f:
        f.write(processed_content)

    print(f"Processed {run_number - 1} runs")
    print(f"Output written to: {output_file}")

def main():
    # Default file names
    input_file = "test_backup.txt"
    output_file = "processed_test.txt"

    # Check command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    try:
        process_log_file(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
