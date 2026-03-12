"""
Reorder test.txt so that all runs for the same lattice group are contiguous,
with permutations in order (1-33).  Every field inside each run block
(lattice, k-eff, flux, timestamps, etc.) travels with its run untouched.

The file has 25 concatenated study batches with "PARAMETRIC STUDY COMPLETED"
footers between them. RUN numbering is sequential (1-4686) across batches.

Ordering in the output:
  1. random_geometric_1 … random_geometric_64  (each perms 1-33)
  2. random_lattice_1   … random_lattice_64    (each perms 1-33)
  3. Self adjusted 1    … Self-adjusted 14     (each perms 1-33)
"""

import re
import hashlib

INPUT  = "test.txt"
OUTPUT = "test_reordered.txt"

with open(INPUT, "r") as f:
    lines = f.readlines()

print(f"Read {len(lines)} lines from {INPUT}")

# ── 1. Find all RUN line positions ─────────────────────────────────────────
run_positions = []  # (line_index, run_number)
for i, line in enumerate(lines):
    m = re.match(r'^RUN (\d+):', line)
    if m:
        run_positions.append((i, int(m.group(1))))

print(f"Found {len(run_positions)} RUN entries")

# ── 2. Find the header (everything before first run's separator block) ─────
# The original format before a run is:
#   ================\n\n================\n\nRUN N:\n
# For RUN 1, there's only one === line before it (end of the header).
# We want the header to end just before the first run's opening separator.
first_run_line = run_positions[0][0]

# Walk back: blank, then ===, then blank, then === (the double separator)
pos = first_run_line - 1
while pos >= 0 and lines[pos].strip() == '':
    pos -= 1
while pos >= 0 and lines[pos].startswith('==='):
    pos -= 1
while pos >= 0 and lines[pos].strip() == '':
    pos -= 1
while pos >= 0 and lines[pos].startswith('==='):
    pos -= 1
header_end = pos + 1
header_lines = lines[:header_end]

# ── 3. Define run block boundaries ────────────────────────────────────────
# Each run "owns" everything from its leading separator (=== lines + blanks
# before "RUN N:") through and including its trailing separator (=== after
# the flux data, plus trailing blank), but NOT any COMPLETED footer blocks
# that might follow.
#
# Strategy: each run block = from its leading === down to (but not including)
# the next run's leading === or a COMPLETED line, whichever comes first.
# Then we also skip COMPLETED footer blocks entirely.

# Find COMPLETED footer block ranges (to exclude)
completed_ranges = []  # (start, end) line ranges to skip
for i, line in enumerate(lines):
    if 'PARAMETRIC STUDY COMPLETED' in line:
        # Walk back to find start of this footer block (=== or blank before it)
        start = i
        while start > 0 and (lines[start - 1].strip() == '' or lines[start - 1].startswith('===')):
            start -= 1
        # Walk forward to find end
        end = i + 1
        while end < len(lines) and (lines[end].strip() == '' or lines[end].startswith('===') or lines[end].startswith('Study finished')):
            end += 1
        completed_ranges.append((start, end))

# Also find the CONSOLIDATED footer at the very end
consolidated_start = len(lines)
for i in range(len(lines) - 1, -1, -1):
    if 'PARAMETRIC STUDY CONSOLIDATED' in lines[i]:
        start = i
        while start > 0 and (lines[start - 1].strip() == '' or lines[start - 1].startswith('===')):
            start -= 1
        consolidated_start = start
        break

# For each run, determine its full block (leading separator through trailing separator)
run_blocks = []
for idx, (run_line, run_num) in enumerate(run_positions):
    # Find start: walk back from RUN line over blanks and === lines
    start = run_line
    while start > 0 and lines[start - 1].strip() == '':
        start -= 1
    while start > 0 and lines[start - 1].startswith('==='):
        start -= 1
    while start > 0 and lines[start - 1].strip() == '':
        start -= 1
    while start > 0 and lines[start - 1].startswith('==='):
        start -= 1

    # Find end: next run's start OR a COMPLETED block, whichever comes first
    if idx + 1 < len(run_positions):
        next_run_line = run_positions[idx + 1][0]
        # Walk back from next run to find its leading separator start
        end = next_run_line
        while end > 0 and lines[end - 1].strip() == '':
            end -= 1
        while end > 0 and lines[end - 1].startswith('==='):
            end -= 1
        while end > 0 and lines[end - 1].strip() == '':
            end -= 1
        while end > 0 and lines[end - 1].startswith('==='):
            end -= 1

        # Check if there's a COMPLETED block between this run and the next
        for cs, ce in completed_ranges:
            if start < cs < end:
                end = cs
                break
    else:
        # Last run: end at consolidated footer or end of file
        end = consolidated_start

    # Extract description
    desc = ""
    for j in range(run_line, min(run_line + 5, len(lines))):
        if lines[j].startswith('Description:'):
            desc = lines[j].replace('Description:', '').strip()
            break

    # Parse lattice name and permutation
    perm_match = re.match(r'(.+?)\s*---\s*permutation\s*(\d+)', desc)
    if perm_match:
        lattice_name = perm_match.group(1).strip()
        perm_num = int(perm_match.group(2))
    else:
        lattice_name = desc
        perm_num = 0

    block_lines = lines[start:end]

    run_blocks.append({
        'run_num': run_num,
        'lattice_name': lattice_name,
        'perm_num': perm_num,
        'description': desc,
        'lines': block_lines,
        'start': start,
        'end': end,
    })

print(f"Extracted {len(run_blocks)} run blocks")

# Verify each block contains its description
bad = sum(1 for rb in run_blocks if rb['description'] not in ''.join(rb['lines']))
if bad:
    print(f"  WARNING: {bad} blocks don't contain their description!")
else:
    print(f"  All {len(run_blocks)} blocks contain their expected description.")

# Count total lines in blocks vs original
total_block_lines = sum(len(rb['lines']) for rb in run_blocks)
completed_lines = sum(ce - cs for cs, ce in completed_ranges)
consolidated_lines = len(lines) - consolidated_start
header_line_count = len(header_lines)
print(f"  Header: {header_line_count} lines")
print(f"  Run blocks: {total_block_lines} lines")
print(f"  COMPLETED footers (removed): {completed_lines} lines across {len(completed_ranges)} footers")
print(f"  CONSOLIDATED footer: {consolidated_lines} lines")
print(f"  Accounted for: {header_line_count + total_block_lines + completed_lines + consolidated_lines} / {len(lines)}")

# ── 4. Sort ────────────────────────────────────────────────────────────────
def sort_key(run):
    name = run['lattice_name']
    if name.startswith('random_geometric_'):
        category = 0
        num = int(name.split('_')[-1])
    elif name.startswith('random_lattice_'):
        category = 1
        num = int(name.split('_')[-1])
    elif name.lower().startswith('self'):
        category = 2
        num_match = re.search(r'(\d+)', name)
        num = int(num_match.group(1)) if num_match else 0
    else:
        category = 3
        num = 0
    return (category, num, run['perm_num'])

runs_sorted = sorted(run_blocks, key=sort_key)

# ── 5. Rebuild the file ───────────────────────────────────────────────────
output_lines = list(header_lines)

for new_idx, run in enumerate(runs_sorted, start=1):
    block = list(run['lines'])

    # Replace the RUN number
    for i, line in enumerate(block):
        m = re.match(r'^(RUN )\d+(:.*)$', line)
        if m:
            block[i] = f"{m.group(1)}{new_idx}{m.group(2)}\n"
            break

    output_lines.extend(block)

# Add final footer matching original style
output_lines.append('\n')
output_lines.append('PARAMETRIC STUDY COMPLETED\n')
output_lines.append('Study finished: reordered\n')
output_lines.append('=' * 80 + '\n')
output_lines.append('\n')
output_lines.append('=' * 80 + '\n')
output_lines.append('PARAMETRIC STUDY CONSOLIDATED\n')
output_lines.append(f'Total runs: {len(runs_sorted)}\n')
output_lines.append('=' * 80 + '\n')

with open(OUTPUT, "w") as f:
    f.writelines(output_lines)

print(f"\nWrote {len(output_lines)} lines to {OUTPUT}")
print(f"Original had {len(lines)} lines")
print(f"Difference: {len(lines) - len(output_lines)} lines (should ≈ COMPLETED footers removed)")

# ── 6. Verification ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

# Check groups
from collections import OrderedDict
groups = OrderedDict()
for i, run in enumerate(runs_sorted):
    name = run['lattice_name']
    if name not in groups:
        groups[name] = []
    groups[name].append((i + 1, run['perm_num']))

all_good = True
for name, entries in groups.items():
    new_run_nums = [e[0] for e in entries]
    perm_nums = [e[1] for e in entries]
    is_contiguous = all(new_run_nums[i+1] == new_run_nums[i] + 1
                        for i in range(len(new_run_nums) - 1))
    is_complete = set(perm_nums) == set(range(1, 34))
    is_ordered = perm_nums == list(range(1, 34))
    if not (is_contiguous and is_ordered and is_complete):
        all_good = False
        print(f"\n  ISSUE: {name}")
        if not is_contiguous: print(f"    Not contiguous")
        if not is_ordered: print(f"    Perms not in order: {perm_nums}")
        if not is_complete: print(f"    Missing perms: {set(range(1,34)) - set(perm_nums)}")

if all_good:
    print(f"\n  All {len(groups)} groups contiguous, perms 1-33, in order.")

# Data integrity
print(f"\n  Checking data integrity...")

def extract_fingerprint(block_lines):
    text = ''.join(block_lines)
    desc = re.search(r'Description:\s*(.+)', text)
    lattice = re.search(r'core_lattice:\s*(\[.+?\])', text)
    keff = re.search(r'k-effective:\s*([\d.]+\s*±\s*[\d.]+)', text)
    flux_lines = re.findall(r'(I_\d+[PBG]? Flux .+)', text)
    parts = []
    if desc: parts.append(desc.group(1).strip())
    if lattice: parts.append(lattice.group(1).strip())
    if keff: parts.append(keff.group(1).strip())
    parts.extend(flux_lines)
    return '|'.join(parts)

original_fps = sorted(extract_fingerprint(rb['lines']) for rb in run_blocks)
reordered_fps = sorted(extract_fingerprint(rb['lines']) for rb in runs_sorted)

if original_fps == reordered_fps:
    print(f"  Data integrity PASSED: all {len(original_fps)} runs match exactly.")
else:
    print(f"  WARNING: Data mismatch!")

# Line-count comparison
print(f"\n  Line count comparison:")
print(f"    Original:  {len(lines)}")
print(f"    Reordered: {len(output_lines)}")
print(f"    Diff:      {len(lines) - len(output_lines)} (= {len(completed_ranges)} COMPLETED footers removed)")

# Verify key data counts match
with open(OUTPUT, 'r') as f:
    out_text = f.read()
with open(INPUT, 'r') as f:
    orig_text = f.read()

checks = [
    ('RUN entries', r'^RUN \d+:', re.MULTILINE),
    ('k-effective', r'k-effective:', 0),
    ('Flux lines', r'I_\d+[PBG]? Flux', 0),
    ('Description', r'^Description:', re.MULTILINE),
    ('Success', r'^Success:', re.MULTILINE),
    ('core_lattice', r'core_lattice:', 0),
    ('Timestamp', r'^Timestamp:', re.MULTILINE),
]

print(f"\n  Data line counts:")
all_match = True
for label, pattern, flags in checks:
    orig_count = len(re.findall(pattern, orig_text, flags))
    new_count = len(re.findall(pattern, out_text, flags))
    status = "OK" if orig_count == new_count else "MISMATCH"
    if status == "MISMATCH":
        all_match = False
    print(f"    {label:20s}: {orig_count:>5} -> {new_count:>5}  {status}")

if all_match:
    print(f"\n  All data counts match perfectly.")

print(f"\nDone! Original file untouched: {INPUT}")
print(f"Reordered file: {OUTPUT}")
