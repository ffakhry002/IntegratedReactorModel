#!/usr/bin/env python3
"""
Split a consolidated parametric study log into per-set training files
for learning curve experiments.

Reads descriptions like "Set 1 -- Core 1/13 -- Fill 1/33 -- 4G" and
groups RUNs by set number. Each output file has the same header as the
original plus only that set's RUNs (renumbered sequentially).

Usage:
    python split_lc_training_data.py <consolidated_log> [--output-dir data/]

Example:
    python split_lc_training_data.py "data2/consolidated_parametric_study_log (1).txt"
    # Produces: data/train_lc_set_13.txt, data/train_lc_set_26.txt, ...
"""

import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

GEOMETRY_STEPS = {
    1: 13, 2: 26, 3: 39, 4: 52, 5: 65, 6: 78, 7: 91, 8: 104,
    9: 117, 10: 130, 11: 143, 12: 156, 13: 169, 14: 182, 15: 195,
    16: 208, 17: 221, 18: 234, 19: 247, 20: 256,
}

RUN_SEPARATOR = '=' * 80


def split_log(input_path, output_dir):
    print(f"Reading: {input_path}")
    with open(input_path, 'r') as f:
        content = f.read()

    # Split into header + run blocks
    # The header is everything before "RUN 1:"
    parts = re.split(r'(RUN \d+:)', content)

    # parts[0] = header, then alternating (RUN label, run body)
    header = parts[0]
    run_blocks = []
    for i in range(1, len(parts), 2):
        run_label = parts[i]        # "RUN 123:"
        run_body = parts[i + 1]     # everything until next "RUN N:"
        run_blocks.append((run_label, run_body))

    print(f"Total RUN blocks parsed: {len(run_blocks)}")

    # Group by set number
    set_runs = defaultdict(list)
    skipped = 0

    for run_label, run_body in run_blocks:
        desc_match = re.search(r'Description:\s*Set (\d+)\s*--', run_body)
        if desc_match:
            set_num = int(desc_match.group(1))
            set_runs[set_num].append(run_body)
        else:
            skipped += 1

    if skipped:
        print(f"WARNING: {skipped} runs had no 'Set N' in description (skipped)")

    print(f"Sets found: {sorted(set_runs.keys())}")
    print()

    # Write one file per set
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for set_num in sorted(set_runs.keys()):
        runs = set_runs[set_num]
        n_cores = GEOMETRY_STEPS.get(set_num, '?')

        filename = f"train_lc_set_{n_cores}.txt"
        output_path = output_dir / filename

        with open(output_path, 'w') as f:
            # Write header with updated run count
            updated_header = re.sub(
                r'Total number of runs: \d+',
                f'Total number of runs: {len(runs)}',
                header
            )
            f.write(updated_header)

            # Write each run with sequential numbering
            for i, run_body in enumerate(runs, 1):
                f.write(f"RUN {i}:")
                f.write(run_body)

        print(f"  Set {set_num:2d} (N={n_cores:>3}): {len(runs):5d} runs → {output_path}")

    # Summary
    total_written = sum(len(v) for v in set_runs.values())
    print(f"\nDone: {len(set_runs)} files, {total_written:,} total runs written")

    # Check for expected counts
    print("\nValidation:")
    all_good = True
    for set_num in sorted(set_runs.keys()):
        n_cores = GEOMETRY_STEPS.get(set_num)
        if n_cores is None:
            continue
        expected = n_cores * 33
        actual = len(set_runs[set_num])
        status = "✓" if actual == expected else f"✗ (expected {expected})"
        if actual != expected:
            all_good = False
        print(f"  Set {set_num:2d}: {actual:5d} / {expected:5d} {status}")

    if not all_good:
        print("\nSome sets have missing runs — simulations may still be in progress.")


def main():
    parser = argparse.ArgumentParser(
        description='Split consolidated parametric study log into per-set training files.')
    parser.add_argument('input', type=str,
                        help='Path to consolidated parametric study log')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for split files (default: data/)')
    args = parser.parse_args()

    split_log(args.input, args.output_dir)


if __name__ == '__main__':
    main()
