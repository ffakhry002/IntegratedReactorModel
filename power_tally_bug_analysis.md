# Power Tally Bug Analysis Report

## Problem Summary

Two fuel assemblies at positions (2,6) and (6,6) in an 8×8 plate-type reactor core are showing 0 power in their power tallies, despite being marked as fuel positions ('F') in the core lattice. All other fuel assemblies show normal power distribution.

## Root Cause Analysis

After investigating the codebase, I identified a **plate positioning offset** between where the actual fuel plates are located in the geometry and where the power tally meshes think they are positioned.

### Key Findings

#### 1. Assembly-Level Positioning is Correct
The assembly-level coordinates are correctly calculated and identical between geometry and power tallies:
- Assembly centers match exactly between geometry and power tallies
- No systematic coordinate offset at the assembly level
- Assembly dimensions are consistently calculated as 5.11 cm

#### 2. Plate-Level Positioning Offset
The issue occurs within individual fuel assemblies - the power tally meshes for individual plates are offset from where the actual fuel plates are positioned:

**Actual Plate Positioning (geometry)**:
- Plates positioned at: `lattice_start_y + plate_k * fuel_plate_pitch`
- Example positions: -2.405, -2.035, -1.665... cm

**Power Tally Mesh Positioning**:
- Meshes positioned at: `plate_region_start + (plate_k + 0.5) * fuel_plate_pitch`
- Example positions: -2.220, -1.850, -1.480... cm

**Critical Offset**: **+0.185 cm** (exactly half of fuel_plate_pitch = 0.37/2)

#### 3. The Problematic `+0.5` Factor
The power tally code incorrectly adds 0.5 to the plate index:
```python
plate_y = plate_region_start + (plate_k + 0.5) * fuel_plate_pitch  # WRONG
```
This shifts all power tally meshes by half a plate pitch away from the actual fuel plates.

### Why This Affects Specific Assemblies

The 0.185 cm offset affects all assemblies, but only causes zero power readings in assemblies (2,6) and (6,6) because:

1. **Geometric sensitivity**: These specific positions may have geometry arrangements where the offset causes meshes to fall entirely in coolant regions between plates
2. **Position-dependent effects**: The offset may interact with adjacent irradiation positions or edge effects
3. **Fuel meat positioning**: The offset might cause meshes to miss the narrow fuel meat regions (1.47 mm thick) within the plates

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

Remove the erroneous `+0.5` factor from the plate positioning calculation in the power tally mesh creation:

**Current (incorrect) - Line 215 in `power_tallies.py`**:
```python
plate_y = plate_region_start + (plate_k + 0.5) * fuel_plate_pitch  # WRONG
```

**Proposed (correct)**:
```python
plate_y = plate_region_start + plate_k * fuel_plate_pitch  # CORRECT
```

This aligns the power tally meshes with the actual fuel plate positions in the geometry.

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

### Primary Fix Required:

**`eigenvalue/tallies/power_tallies.py`** (line 215 in `create_plate_element_tallies`):
```python
# BEFORE (incorrect):
plate_y = plate_region_start + (plate_k + 0.5) * fuel_plate_pitch

# AFTER (correct):  
plate_y = plate_region_start + plate_k * fuel_plate_pitch
```

### Files NOT Affected:
- **Assembly-level positioning** (lines 91-92, 135-136, 200-201) is **CORRECT**
- **`irradiation_tallies.py`** positioning is **CORRECT** (as evidenced by working irradiation tallies)
- The issue is specifically with **plate-level** positioning within fuel assemblies

### Root Cause Clarification:
This is NOT a systematic coordinate system issue. The problem is specifically in the individual plate positioning within fuel assemblies, where the power tally meshes are offset by exactly half a plate pitch (0.185 cm) from the actual fuel plate locations.

This fix should resolve the 0 power issue for assemblies (2,6) and (6,6) by aligning the power tally meshes with the actual fuel plate positions. The fix addresses the root cause of the 0.185 cm offset that was causing power tally meshes to miss the fuel regions in these sensitive assembly positions.