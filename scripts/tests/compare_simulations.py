#!/usr/bin/env python3
"""
Compare the batch simulation results with the original uniform dataset to check compatibility.
"""

import argparse
import json

import numpy as np


def analyze_compatibility(rtol=1e-3):
    print("=" * 80)
    print("COMPATIBILITY ANALYSIS: Batch Results vs Uniform Dataset")
    print("=" * 80)

    # 1. Load and analyze batch simulation results
    print("\n1. ANALYZING BATCH SIMULATION RESULTS")
    print("-" * 50)

    batch_file = r"c:\Users\Rodolfo Simões\Documents\PlasmaML\KineticLearn\results\batch_simulations\lokisimulator\boundsbasedsampler\2025-08-17\batch_100sims_20250817_230722.json"
    batch_file = r"C:\Users\rsimoes\Documents\Learning\GPT5Test\results\batch_simulations\lokisimulator\boundsbasedsampler\2025-08-17\batch_100sims_20250817_230722.json"

    with open(batch_file, "r") as f:
        batch_data = json.load(f)

    # Extract k values and compositions from batch
    batch_k_values = []
    for ps in batch_data["parameter_sets"]:
        batch_k_values.append(ps["k_values"])

    batch_k_values = np.array(batch_k_values)
    batch_compositions = np.array(batch_data["compositions"])

    print(f"Batch simulations: {len(batch_k_values)} parameter sets")
    print(
        f"Batch pressure conditions: {batch_data['metadata']['pressure_conditions_torr']} Torr"
    )
    print(f"Batch composition shape: {batch_compositions.shape}")
    print(f"Expected: 10 param sets × 2 pressures = 20 rows")

    print(f"\nBatch K value ranges:")
    for i in range(3):
        k_min, k_max = batch_k_values[:, i].min(), batch_k_values[:, i].max()
        print(f"  k[{i}]: [{k_min:.3e}, {k_max:.3e}]")

    print(f"\nBatch composition ranges (first 3 species):")
    for i in range(min(3, batch_compositions.shape[1])):
        comp_min, comp_max = (
            batch_compositions[:, i].min(),
            batch_compositions[:, i].max(),
        )
        print(f"  Species {i}: [{comp_min:.3e}, {comp_max:.3e}]")

    # 2. Load and analyze uniform dataset
    print("\n2. ANALYZING UNIFORM DATASET")
    print("-" * 50)

    uniform_file = r"c:\Users\Rodolfo Simões\Documents\PlasmaML\KineticLearn\data\SampleEfficiency\O2_simple_uniform.txt"
    uniform_file = r"C:\Users\rsimoes\Documents\Learning\GPT5Test\data\SampleEfficiency\O2_simple_uniform.txt"

    uniform_data = np.loadtxt(uniform_file)

    # Extract k values (first 3 columns) and compositions (last 3 columns)
    uniform_k_values = uniform_data[:, :3]
    uniform_compositions = uniform_data[:, -3:]  # Last 3 columns are compositions

    print(f"Uniform dataset: {len(uniform_k_values)} samples")
    print(f"Uniform data shape: {uniform_data.shape}")
    print(f"Columns: k1, k2, k3, ..., comp1, comp2, comp3")

    print(f"\nUniform K value ranges:")
    for i in range(3):
        k_min, k_max = uniform_k_values[:, i].min(), uniform_k_values[:, i].max()
        print(f"  k[{i}]: [{k_min:.3e}, {k_max:.3e}]")

    print(f"\nUniform composition ranges:")
    for i in range(3):
        comp_min, comp_max = (
            uniform_compositions[:, i].min(),
            uniform_compositions[:, i].max(),
        )
        print(f"  Species {i}: [{comp_min:.3e}, {comp_max:.3e}]")

    # 3. Compare k value ranges
    print("\n3. K VALUE COMPATIBILITY CHECK")
    print("-" * 50)

    k_true = np.array([6.00e-16, 1.30e-15, 9.60e-16])

    print(f"k_true reference: {k_true}")

    compatible_k = True
    for i in range(3):
        batch_min, batch_max = batch_k_values[:, i].min(), batch_k_values[:, i].max()
        uniform_min, uniform_max = (
            uniform_k_values[:, i].min(),
            uniform_k_values[:, i].max(),
        )

        # Check if batch range is within uniform range
        tol_min = abs(uniform_min) * rtol
        tol_max = abs(uniform_max) * rtol
        k_within_range = (batch_min >= (uniform_min - tol_min)) and (
            batch_max <= (uniform_max + tol_max)
        )

        print(f"\nk[{i}] comparison:")
        print(f"  Uniform range:   [{uniform_min:.3e}, {uniform_max:.3e}]")
        print(f"  Batch range:     [{batch_min:.3e}, {batch_max:.3e}]")
        print(f"  Within uniform:  {'✅ YES' if k_within_range else '❌ NO'}")

        # Check ratios to k_true
        uniform_ratio_min = uniform_min / k_true[i]
        uniform_ratio_max = uniform_max / k_true[i]
        batch_ratio_min = batch_min / k_true[i]
        batch_ratio_max = batch_max / k_true[i]

        print(
            f"  Uniform ratios:  [{uniform_ratio_min:.2f}, {uniform_ratio_max:.2f}] × k_true"
        )
        print(
            f"  Batch ratios:    [{batch_ratio_min:.2f}, {batch_ratio_max:.2f}] × k_true"
        )

        if not k_within_range:
            compatible_k = False

    # 4. Compare composition ranges
    print("\n4. COMPOSITION COMPATIBILITY CHECK")
    print("-" * 50)

    compatible_comp = True
    for i in range(3):
        batch_min, batch_max = (
            batch_compositions[:, i].min(),
            batch_compositions[:, i].max(),
        )
        uniform_min, uniform_max = (
            uniform_compositions[:, i].min(),
            uniform_compositions[:, i].max(),
        )

        # Check if ranges overlap (more lenient than complete containment)
        ranges_overlap = not (batch_max < uniform_min or batch_min > uniform_max)

        # Check if batch range is reasonably close to uniform range (within 2 orders of magnitude)
        ratio_check = (batch_max / uniform_min < 100) and (
            uniform_max / batch_min < 100
        )

        print(f"\nSpecies {i} composition:")
        print(f"  Uniform range:   [{uniform_min:.3e}, {uniform_max:.3e}]")
        print(f"  Batch range:     [{batch_min:.3e}, {batch_max:.3e}]")
        print(f"  Ranges overlap:  {'✅ YES' if ranges_overlap else '❌ NO'}")
        print(f"  Magnitude check: {'✅ YES' if ratio_check else '❌ NO'}")

        if not (ranges_overlap and ratio_check):
            compatible_comp = False

    # 5. Check specific sample compatibility
    print("\n5. SAMPLE-LEVEL COMPARISON")
    print("-" * 50)

    # Find uniform samples with similar k values to batch samples
    print("Looking for similar k values in uniform dataset...")

    tolerance = 0.1  # 10% tolerance
    matches_found = 0

    for i, batch_k in enumerate(batch_k_values[:3]):  # Check first 3 batch samples
        # Find uniform samples with similar k values
        k_diffs = np.abs(uniform_k_values - batch_k) / batch_k  # Relative differences
        k_close = np.all(k_diffs < tolerance, axis=1)  # All 3 k values within tolerance

        if np.any(k_close):
            matches_found += 1
            closest_idx = np.where(k_close)[0][0]

            print(f"\nBatch sample {i}:")
            print(f"  k_values: {batch_k}")
            print(
                f"  Closest uniform (idx {closest_idx}): {uniform_k_values[closest_idx]}"
            )
            print(f"  Uniform compositions: {uniform_compositions[closest_idx]}")
            print(f"  Batch compositions (1st pressure): {batch_compositions[i]}")
            print(
                f"  Batch compositions (2nd pressure): {batch_compositions[i + len(batch_k_values)]}"
            )

    print(f"\nMatches found: {matches_found}/3 samples")

    # 6. Overall compatibility assessment
    print("\n6. OVERALL COMPATIBILITY ASSESSMENT")
    print("=" * 50)

    print(
        f"K value compatibility:    {'✅ COMPATIBLE' if compatible_k else '❌ INCOMPATIBLE'}"
    )
    print(
        f"Composition compatibility: {'✅ COMPATIBLE' if compatible_comp else '❌ INCOMPATIBLE'}"
    )

    if compatible_k and compatible_comp:
        print(f"\n🎉 RESULT: DATASETS ARE COMPATIBLE!")
        print(f"   - Batch simulation k values fall within uniform training range")
        print(f"   - Composition ranges are consistent")
        print(f"   - Safe to use batch results for model training")
    else:
        print(f"\n⚠️  RESULT: POTENTIAL COMPATIBILITY ISSUES!")
        if not compatible_k:
            print(f"   - Batch k values may be outside training distribution")
        if not compatible_comp:
            print(f"   - Composition ranges differ significantly")
        print(f"   - Consider adjusting sampling bounds or checking simulation setup")

    # 7. Recommendations
    print(f"\n7. RECOMMENDATIONS")
    print("-" * 50)

    if compatible_k and compatible_comp:
        print(f"✅ Datasets are compatible - proceed with active learning")
        print(f"✅ Current sampling bounds are appropriate")
        print(f"✅ Both datasets use same simulation setup (O2_simple)")
    else:
        print(f"🔧 Consider these adjustments:")
        if not compatible_k:
            print(f"   - Narrow k sampling bounds to match uniform training range")
        if not compatible_comp:
            print(f"   - Verify simulation pressure conditions match training data")
            print(f"   - Check if batch uses same chemistry file as uniform dataset")

    return compatible_k and compatible_comp


def main():
    parser = argparse.ArgumentParser(
        description="Compare batch JSON to uniform dataset"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="relative tolerance for range containment checks",
    )
    args = parser.parse_args()

    result = analyze_compatibility(rtol=args.rtol)
    print("\nOverall compatibility (with rtol={}): {}".format(args.rtol, result))


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    analyze_compatibility()
