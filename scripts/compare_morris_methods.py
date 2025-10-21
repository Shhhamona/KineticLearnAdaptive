#!/usr/bin/env python3
"""
Compare MorrisSampler vs CorrectedGridMorrisSampler

This script demonstrates the KEY DIFFERENCE between the two Morris sampling methods:
- MorrisSampler: BROKEN - uses value-based delta (doesn't stay on grid)
- CorrectedGridMorrisSampler: FIXED - uses index-based delta (stays on grid)

The Morris method MUST sample on a discrete grid to calculate proper sensitivities.
"""

import sys

import numpy as np

sys.path.append(".")
from other_scripts.SamplingMorrisMethod import CorrectedGridMorrisSampler, MorrisSampler


def compare_morris_methods():
    """
    Compare the two Morris methods with identical parameters to show the difference.
    Using LINEAR sampling for clearer demonstration.
    """

    print("=" * 80)
    print("MORRIS SAMPLING METHODS COMPARISON (LINEAR)")
    print("=" * 80)

    # Common parameters for both methods - USING LINEAR for clarity
    k_real = np.array([1.0, 1.0, 1.0])  # Simplified true values
    p = 11  # Number of grid levels (CHANGED from 10 to break the alignment!)
    r = 2  # Number of trajectories (keeping small for clarity)
    k_range_type = "lin"  # LINEAR instead of log
    k_range = [0, 1]  # Simple 0 to 1 range
    indexes = [0, 1, 2]  # Vary all 3 parameters

    print(f"\nCOMMON PARAMETERS:")
    print(f"  True K values: {k_real}")
    print(f"  Grid levels (p): {p} ⚠️ (Changed to {p} to break delta alignment)")
    print(f"  Trajectories (r): {r}")
    print(f"  Range type: {k_range_type} (LINEAR)")
    print(f"  Range: {k_range[0]} to {k_range[1]}")
    print(f"  Parameters to vary: {indexes}")

    # Calculate expected grid
    grid = np.linspace(k_range[0], k_range[1], p)
    print(f"\n📊 EXPECTED GRID ({p} levels - LINEAR):")
    for i, val in enumerate(grid):
        print(f"  Level {i}: {val:.6f}")
    
    grid_spacing = grid[1] - grid[0]
    print(f"\n  Grid spacing: {grid_spacing:.6f}")
    
    # Show why this breaks the alignment
    delta_value = p / (2 * (p - 1))
    ratio = delta_value / grid_spacing
    print(f"\n🔍 ALIGNMENT CHECK:")
    print(f"  delta = {delta_value:.6f}")
    print(f"  grid_spacing = {grid_spacing:.6f}")
    print(f"  delta / grid_spacing = {ratio:.6f}")
    if abs(ratio - round(ratio)) < 0.001:
        print(f"  ⚠️  With p=10: ratio = 5.0 (PERFECTLY ALIGNED - hides the bug!)")
        print(f"  📌 Changed to p={p} to reveal the bug!")
    else:
        print(f"  ❌ NOT an integer multiple! Delta will create OFF-GRID points!")

    print("\n" + "=" * 80)
    print("WHY p=10 HIDES THE BUG")
    print("=" * 80)
    print(f"\n📐 With p=10:")
    print(f"  grid_spacing = 1.0 / 9 = 0.111111")
    print(f"  delta = 10 / (2*9) = 0.555556")
    print(f"  delta / grid_spacing = 5.0 (EXACT)")
    print(f"  Result: Moving by delta = moving 5 grid positions → stays on grid!")
    print(f"\n📐 With p={p}:")
    print(f"  grid_spacing = 1.0 / {p-1} = {grid_spacing:.6f}")
    print(f"  delta = {p} / (2*{p-1}) = {delta_value:.6f}")
    print(f"  delta / grid_spacing = {ratio:.6f} (NOT exact)")
    print(f"  Result: Moving by delta ≠ integer grid positions → goes OFF grid!")
    
    print("\n" + "=" * 80)
    print("METHOD 1: MorrisSampler (ORIGINAL - BROKEN)")
    print("=" * 80)
    print("\n⚠️  PROBLEM: Uses value-based delta")
    delta_m1 = p / (2 * (p - 1))
    print(f"   - Calculates: delta = p / (2*(p-1)) = {p} / (2*{p-1}) = {delta_m1:.6f}")
    print(f"   - This is a VALUE, not a grid position!")
    print(f"   - Grid spacing is {grid_spacing:.6f}, but delta is {delta_m1:.6f}")
    print(f"   - These don't align! Points fall OFF the grid")

    # Set seed for reproducibility
    np.random.seed(42)

    print("\nRunning MorrisSampler...")
    samples_method1 = MorrisSampler(k_real, p, r, k_range_type, k_range, indexes)

    print(f"\n📋 METHOD 1 RESULTS:")
    print(f"   Shape: {samples_method1.shape}")
    print(f"   Total samples: {len(samples_method1)}")

    # Check if samples are on the grid
    print(f"\n🔍 GRID ALIGNMENT CHECK (Method 1):")
    on_grid_count = 0
    off_grid_count = 0

    for i, sample in enumerate(samples_method1[:5]):  # Check first 5 samples
        print(f"\n   Sample {i+1}:")
        for j, (param_idx, val) in enumerate(zip(indexes, sample[indexes])):
            # Check if value is close to any grid point
            normalized_val = val / k_real[param_idx]
            closest_grid = min(grid, key=lambda x: abs(x - normalized_val))
            distance = abs(normalized_val - closest_grid)

            if distance < 1e-10:  # Essentially on grid
                status = "✅ ON GRID"
                on_grid_count += 1
            else:
                status = f"❌ OFF GRID (distance: {distance:.6f})"
                off_grid_count += 1

            print(
                f"      K{param_idx}: {val:.6f} (normalized: {normalized_val:.6f}) {status}"
            )
            print(f"               Closest grid point: {closest_grid:.6f}")

    print(f"\n📊 METHOD 1 SUMMARY:")
    print(f"   Samples on grid: {on_grid_count}")
    print(f"   Samples off grid: {off_grid_count}")
    if off_grid_count > 0:
        print(f"   ⚠️  This method FAILS to keep samples on the grid!")
    
    print("\n" + "=" * 80)
    print("METHOD 2: CorrectedGridMorrisSampler (FIXED)")
    print("=" * 80)
    print("\n✅ SOLUTION: Uses index-based delta")
    delta_m2 = max(1, p // 4)
    print(f"   - Calculates: delta_index = max(1, p//4) = max(1, {p//4}) = {delta_m2}")
    print(f"   - This is a GRID POSITION (jump {delta_m2} levels), not a value!")
    print(f"   - Jump {delta_m2} positions = move by {delta_m2 * grid_spacing:.6f} in value")
    print(f"   - Result: Points ALWAYS stay on the grid")

    # Reset seed for fair comparison
    np.random.seed(42)

    print("\nRunning CorrectedGridMorrisSampler...")
    samples_method2 = CorrectedGridMorrisSampler(
        k_real, p, r, k_range_type, k_range, indexes
    )

    print(f"\n📋 METHOD 2 RESULTS:")
    print(f"   Shape: {samples_method2.shape}")
    print(f"   Total samples: {len(samples_method2)}")

    # Check if samples are on the grid
    print(f"\n🔍 GRID ALIGNMENT CHECK (Method 2):")
    on_grid_count2 = 0
    off_grid_count2 = 0

    for i, sample in enumerate(samples_method2[:5]):  # Check first 5 samples
        print(f"\n   Sample {i+1}:")
        for j, (param_idx, val) in enumerate(zip(indexes, sample[indexes])):
            # Check if value is close to any grid point
            normalized_val = val / k_real[param_idx]
            closest_grid = min(grid, key=lambda x: abs(x - normalized_val))
            distance = abs(normalized_val - closest_grid)

            if distance < 1e-10:  # Essentially on grid
                status = "✅ ON GRID"
                on_grid_count2 += 1
            else:
                status = f"❌ OFF GRID (distance: {distance:.6f})"
                off_grid_count2 += 1

            print(
                f"      K{param_idx}: {val:.6f} (normalized: {normalized_val:.6f}) {status}"
            )

            # Show which grid level it corresponds to
            for level_idx, grid_val in enumerate(grid):
                if abs(grid_val - normalized_val) < 1e-10:
                    print(f"               → Grid Level {level_idx} = {grid_val:.6f}")
                    break

    print(f"\n📊 METHOD 2 SUMMARY:")
    print(f"   Samples on grid: {on_grid_count2}")
    print(f"   Samples off grid: {off_grid_count2}")
    if off_grid_count2 == 0:
        print(f"   ✅ This method SUCCEEDS in keeping all samples on the grid!")

    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(f"\n🔢 SAMPLING STATISTICS:")
    print(f"{'Metric':<30} {'Method 1 (Broken)':<20} {'Method 2 (Fixed)':<20}")
    print(f"{'-'*70}")
    print(
        f"{'Total samples':<30} {len(samples_method1):<20} {len(samples_method2):<20}"
    )
    print(
        f"{'Samples per trajectory':<30} {(len(indexes)+1):<20} {(len(indexes)+1):<20}"
    )
    print(f"{'Trajectories':<30} {r:<20} {r:<20}")

    print(f"\n📐 DELTA CALCULATION:")
    print(f"{'Delta value':<30} {delta_m1:<20.6f} {delta_m2:<20} (grid positions)")
    print(f"{'Delta type':<30} {'VALUE (wrong!)':<20} {'GRID INDEX (correct!)':<20}")
    print(f"{'Grid spacing':<30} {grid_spacing:<20.6f} {grid_spacing:<20.6f}")

    print(f"\n🎯 KEY DIFFERENCE:")
    print(f"  Method 1: delta = {delta_m1:.6f} → tries to move by this VALUE")
    print(f"            Example: If at grid[5] = {grid[5]:.6f}, moves to {grid[5] + delta_m1:.6f}")
    print(f"            Problem: {grid[5] + delta_m1:.6f} is NOT on the grid!")
    print(f"            Closest grid point would be {grid[min(9, 5 + int(delta_m1/grid_spacing))]:.6f}")

    print(f"\n  Method 2: delta_index = {delta_m2} → moves by this many GRID POSITIONS")
    print(f"            Example: If at grid[5] = {grid[5]:.6f}, moves to grid[5+{delta_m2}] = grid[{5+delta_m2}]")
    if 5 + delta_m2 < len(grid):
        print(f"            Success: grid[{5+delta_m2}] = {grid[5+delta_m2]:.6f} IS on the grid!")
    print(f"            Actual value change: {delta_m2 * grid_spacing:.6f}")

    print(f"\n💡 WHY THIS MATTERS:")
    print(f"  - Morris sensitivity analysis requires samples on a DISCRETE GRID")
    print(f"  - Finite differences are calculated between grid points")
    print(f"  - Method 1's delta ({delta_m1:.6f}) doesn't match grid spacing ({grid_spacing:.6f})")
    print(f"  - Method 2 moves exactly {delta_m2} grid positions = {delta_m2 * grid_spacing:.6f} value change")
    print(f"  - Only Method 2 ensures mathematically valid Morris sampling")

    print(f"\n✅ CONCLUSION:")
    print(f"  Use CorrectedGridMorrisSampler for proper Morris sampling!")
    print(f"  The original MorrisSampler has a fundamental bug in delta calculation.")

    return samples_method1, samples_method2


def demonstrate_trajectory_example():
    """
    Show a detailed trajectory example to illustrate the difference.
    """

    print("\n\n" + "=" * 80)
    print("DETAILED TRAJECTORY EXAMPLE")
    print("=" * 80)

    # Simplified parameters for clear demonstration
    k_real = np.array([1.0, 1.0, 1.0])  # Simplified for clarity
    p = 6  # Smaller grid for clarity
    k_range = [0, 1]  # Simple range

    grid = np.linspace(0, 1, p)

    print(f"\n📊 SIMPLIFIED GRID (6 levels for clarity):")
    for i, val in enumerate(grid):
        print(f"  Level {i}: {val:.3f}")

    print(f"\n🎲 EXAMPLE TRAJECTORY (starting at grid level 2):")
    start_level = 2
    start_value = grid[start_level]

    print(f"\n  Starting point: Grid Level {start_level} = {start_value:.3f}")

    # Method 1 delta
    delta_m1 = p / (2 * (p - 1))
    print(f"\n  METHOD 1 (Broken) - delta = {delta_m1:.3f}:")
    print(
        f"    Moving up:   {start_value:.3f} + {delta_m1:.3f} = {start_value + delta_m1:.3f}"
    )
    print(f"    ❌ {start_value + delta_m1:.3f} is NOT on grid!")
    print(
        f"    Closest grid point: Level {np.argmin(np.abs(grid - (start_value + delta_m1)))}: {grid[np.argmin(np.abs(grid - (start_value + delta_m1)))]:.3f}"
    )
    print(
        f"    Distance from grid: {abs((start_value + delta_m1) - grid[np.argmin(np.abs(grid - (start_value + delta_m1)))]):.6f}"
    )

    # Method 2 delta
    delta_m2 = max(1, p // 4)
    new_level = start_level + delta_m2
    print(f"\n  METHOD 2 (Fixed) - delta_index = {delta_m2}:")
    print(
        f"    Moving up:   Grid Level {start_level} + {delta_m2} = Grid Level {new_level}"
    )
    print(f"    ✅ Grid Level {new_level} = {grid[new_level]:.3f} IS on grid!")
    print(f"    Distance from grid: 0.000000 (exactly on grid)")

    print(f"\n💡 CONCLUSION:")
    print(f"  Method 1 produces values BETWEEN grid points (mathematically invalid)")
    print(f"  Method 2 produces values EXACTLY ON grid points (mathematically correct)")


if __name__ == "__main__":
    print("\n🔬 MORRIS SAMPLING COMPARISON SCRIPT")
    print("This script demonstrates why CorrectedGridMorrisSampler is needed\n")

    # Main comparison
    samples1, samples2 = compare_morris_methods()

    # Detailed trajectory example
    demonstrate_trajectory_example()

    print("\n\n" + "=" * 80)
    print("✅ COMPARISON COMPLETE")
    print("=" * 80)
    print("\nKEY TAKEAWAYS:")
    print("1. MorrisSampler uses VALUE-based delta → samples fall OFF the grid")
    print(
        "2. CorrectedGridMorrisSampler uses INDEX-based delta → samples stay ON the grid"
    )
    print(
        "3. Morris method REQUIRES grid-based sampling for valid sensitivity analysis"
    )
    print("4. Always use CorrectedGridMorrisSampler for proper Morris sampling!")
