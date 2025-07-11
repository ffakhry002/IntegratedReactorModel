# Power Tally Bug Analysis Report

## Problem Summary

Two fuel assemblies at positions (2,6) and (6,6) in an 8×8 plate-type reactor core are showing 0 power in their power tallies, despite being marked as fuel positions ('F') in the core lattice. All other fuel assemblies show normal power distribution.

## Investigation Summary

After extensive investigation, several systematic causes have been ruled out. The issue appears to be highly position-specific rather than a general systematic problem.

### Theories Investigated and Ruled Out

#### 1. ✗ Systematic Coordinate Offset
- **Theory**: General coordinate system mismatch affecting all assemblies
- **Evidence Against**: Symmetric positions work correctly; irradiation tallies function properly
- **Conclusion**: Assembly-level positioning is identical between geometry and tallies

#### 2. ✗ Systematic Plate Positioning Offset  
- **Theory**: All plate-level tallies offset by half a plate pitch
- **Evidence Against**: Would affect all assemblies, not just these two specific positions
- **Conclusion**: If systematic, all 624 plate tallies would show the same issue

#### 3. ✗ Material Assignment Issues
- **Verified**: U3Si2 fuel materials are created correctly for all fuel positions
- **Verified**: Core layout correctly identifies (2,6) and (6,6) as 'F' positions
- **Verified**: No hidden characters or encoding issues in position strings

#### 4. ✗ Tally Creation Problems
- **Verified**: No duplicate tally names (624 unique plate tallies created)
- **Verified**: Power tally mesh calculations produce reasonable boundary values
- **Verified**: Plate-level tallies are being created (element_level_power_tallies = True)

#### 5. ✗ Cell ID Conflicts
- **Verified**: Unique cell IDs generated for all positions using systematic formula
- **Example**: (2,6) → ID 1020600, (6,6) → ID 1060600

### Observations About Affected Positions

Both problematic positions share a specific geometric relationship:
- **(2,6)** is directly **south** of irradiation position **I_4** at (1,6)  
- **(6,6)** is directly **south** of irradiation position **I_3** at (5,6)

However, many other fuel assemblies are adjacent to irradiation positions and work correctly, suggesting this spatial relationship alone is not the complete explanation.

## Remaining Investigation Areas

Since systematic issues have been ruled out, the cause is likely one of these position-specific problems:

### 1. **Neutronics/Physics Issues**
- **Flux shadowing**: Adjacent irradiation positions may create neutron shadows
- **Neutron streaming**: Specific geometry arrangements may prevent neutron flux from reaching these assemblies
- **Material interaction**: Unique neutron cross-section effects at these locations

### 2. **Geometry Interference** 
- **Boundary overlaps**: Subtle geometry conflicts between fuel assemblies and adjacent irradiation cells
- **OpenMC geometry resolution**: Issues with how OpenMC resolves the geometry at these specific coordinates
- **Mesh validity**: Power tally mesh boundaries may be invalid or outside the geometry at these positions

### 3. **Simulation/Post-Processing Issues**
- **OpenMC mesh filtering**: Mesh filters may not function correctly for these specific boundary coordinates  
- **Data extraction**: Power values may be calculated but not properly extracted from OpenMC results
- **Statistical convergence**: These positions may require more particles/batches to show non-zero power

## Recommended Next Steps

Rather than code changes, the following diagnostic steps are recommended:

### 1. **Geometry Validation**
- Generate OpenMC geometry plots showing the exact boundaries of assemblies (2,6) and (6,6)
- Verify that fuel material actually exists at the expected coordinates
- Check for any geometry overlaps or gaps in these regions

### 2. **Neutronics Analysis**  
- Extract neutron flux values at these positions to see if neutrons are reaching these assemblies
- Compare flux spectra between working and non-working assemblies
- Check if these assemblies are being neutronically shadowed

### 3. **Power Tally Debugging**
- Temporarily create assembly-level (not plate-level) power tallies for these positions to see if the issue persists
- Test with different mesh boundaries or tally types
- Increase particle count significantly to rule out statistical effects

### 4. **Geometry Simplification Test**
- Temporarily replace adjacent irradiation positions with fuel assemblies to test for interference
- Run the simulation with a simplified core layout to isolate the issue

## Summary

This investigation has systematically ruled out several plausible systematic causes for the zero power readings at positions (2,6) and (6,6). The issue is confirmed to be highly position-specific rather than a general code problem.

**Key Insight**: The user's challenges to systematic theories were correct - if there were general coordinate offsets, plate positioning errors, or material issues, many more assemblies would be affected.

**Most Likely Cause**: The issue appears to be related to specific neutronics or geometry interactions at these two positions, possibly involving their spatial relationship to adjacent irradiation cells.

**Next Steps**: Rather than code modifications, diagnostic investigation focusing on geometry validation, neutronics analysis, and simulation debugging is recommended to identify the true root cause.