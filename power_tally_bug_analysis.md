# Power Tally Bug Analysis Report

## Problem Summary

Two fuel assemblies at positions (2,6) and (6,6) in an 8×8 plate-type reactor core are showing 0 power in their power tallies, despite being marked as fuel positions ('F') in the core lattice. All other fuel assemblies show normal power distribution.

## Root Cause Analysis

After investigating the codebase, I identified a **coordinate system mismatch** between the actual fuel assembly geometry and the power tally mesh positioning.

### Key Findings

#### 1. Assembly Dimension Calculations
Both geometry and power tallies correctly calculate the same assembly dimensions:
- **Geometry (`core.py`)**: `assembly_pitch = (plates_per_assembly * fuel_plate_pitch + 2 * clad_structure_width) * 100`
- **Power Tallies (`power_tallies.py`)**: `assembly_width = (fuel_plate_width + 2 * clad_structure_width) * 100`
- **Result**: Both calculate 5.11 cm (since `fuel_plate_width` = `plates_per_assembly * fuel_plate_pitch`)

#### 2. Coordinate System Discrepancy
The issue lies in how positions are calculated:

**Geometry Lattice Positioning (`core.py`)**:
```python
ll_x = -assembly_pitch * n_cols / 2  # = -4 * assembly_pitch
ll_y = -assembly_pitch * n_rows / 2  # = -4 * assembly_pitch
core_lattice.pitch = (assembly_pitch, assembly_pitch)
```
- Uses OpenMC's lattice indexing where cell (i,j) is at lattice coordinates
- Assembly at (i,j) has center at: `(ll_x + j * assembly_pitch, ll_y + i * assembly_pitch)`

**Power Tally Positioning (`power_tallies.py`)**:
```python
x_pos = (j - len(row)/2 + 0.5) * assembly_width
y_pos = (i - len(core_layout)/2 + 0.5) * assembly_width
```
- Uses a different centering formula with +0.5 offset

#### 3. Position Calculation Comparison

For position (2,6) in an 8×8 grid:

**Actual Geometry Position**:
- x = -4 * pitch + 6 * pitch = 2 * pitch
- y = -4 * pitch + 2 * pitch = -2 * pitch

**Power Tally Position**:
- x = (6 - 4 + 0.5) * width = 2.5 * width  
- y = (2 - 4 + 0.5) * width = -1.5 * width

**Offset**: +0.5 pitch in both X and Y directions

### Why This Affects Specific Assemblies

The +0.5 assembly pitch offset causes power tally meshes to be positioned between actual fuel assemblies rather than over them. Assemblies at positions (2,6) and (6,6) are particularly affected because:

1. They're located where this systematic offset causes complete misalignment
2. The offset may interact with the specific geometry layout to cause the tally volumes to fall entirely in coolant regions
3. Other assemblies might have partial overlap that still captures some power

## Affected Code Files

### 1. `eigenvalue/tallies/power_tallies.py`
**Lines 91-92** - Assembly positioning calculation:
```python
x_pos = (j - len(row)/2 + 0.5) * assembly_width
y_pos = (i - len(core_layout)/2 + 0.5) * assembly_width
```

### 2. `Reactor/geometry_helpers/core.py`  
**Lines 86-90** - Correct lattice positioning:
```python
ll_x = -assembly_pitch * n_cols / 2
ll_y = -assembly_pitch * n_rows / 2
core_lattice.lower_left = (ll_x, ll_y)
core_lattice.pitch = (assembly_pitch, assembly_pitch)
```

## Recommended Fix

Modify the power tally positioning formula in `power_tallies.py` to match the geometry coordinate system:

**Current (incorrect)**:
```python
x_pos = (j - len(row)/2 + 0.5) * assembly_width
y_pos = (i - len(core_layout)/2 + 0.5) * assembly_width
```

**Proposed (correct)**:
```python
x_pos = (j - len(row)/2) * assembly_width
y_pos = (i - len(core_layout)/2) * assembly_width
```

This removes the erroneous +0.5 offset that was causing the misalignment.

## Verification Steps

After applying the fix:

1. **Check Assembly (2,6)**: Verify power tally shows non-zero values across all 13 plates
2. **Check Assembly (6,6)**: Verify power tally shows non-zero values across all 13 plates  
3. **Validate Other Assemblies**: Ensure existing assemblies still show correct power values
4. **Geometry Plots**: Generate and verify that power tally mesh positions align with fuel assembly positions
5. **Power Conservation**: Verify that total core power is conserved after the fix

## Additional Recommendations

1. **Add Coordinate System Tests**: Create unit tests that verify power tally mesh positions align with actual fuel geometry
2. **Visualization Tools**: Develop debugging plots that overlay power tally meshes on geometry plots
3. **Code Review**: Review similar positioning calculations in other tally files (e.g., `irradiation_tallies.py`) for consistency

## Files to Modify

### Primary Fixes Required:

1. **`eigenvalue/tallies/power_tallies.py`** (lines 91-92, 135-136, 200-201):
   ```python
   # Remove +0.5 from these lines:
   x_pos = (j - len(row)/2 + 0.5) * assembly_width      # Line 91
   y_pos = (i - len(core_layout)/2 + 0.5) * assembly_width  # Line 92
   assembly_x = (j - len(row)/2 + 0.5) * assembly_width     # Line 135  
   assembly_y = (i - len(core_layout)/2 + 0.5) * assembly_width # Line 136
   assembly_x = (j - len(row)/2 + 0.5) * assembly_width     # Line 200
   assembly_y = (i - len(core_layout)/2 + 0.5) * assembly_width # Line 201
   ```

2. **`eigenvalue/tallies/irradiation_tallies.py`** (lines 103-104):
   ```python
   # Same issue found - remove +0.5 from these lines:
   x_pos = (j - len(row)/2 + 0.5) * width               # Line 103
   y_pos = (i - len(core_layout)/2 + 0.5) * width       # Line 104
   ```

### Additional Review Needed:
- `plotting/functions/power.py` - Check for similar positioning calculations
- Any other files using assembly positioning formulas

**Impact**: This systematic coordinate offset affects both power tallies AND irradiation tallies, potentially causing incorrect positioning of irradiation position tallies as well.

This fix should resolve the 0 power issue for assemblies (2,6) and (6,6) by ensuring their power tally meshes are correctly positioned over the actual fuel regions, and will also correct any similar issues with irradiation tallies.