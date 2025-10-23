#!/usr/bin/env python3
"""
Morris Delta Comparison

This script explains and compares different delta choices for Morris sampling.
According to Morris (1991), delta should create "significant" perturbations while
staying on the grid. Different choices are valid!
"""

import numpy as np
import random

def demonstrate_delta_choices():
    """Show how different delta_index values affect Morris sampling."""
    
    print("="*80)
    print("MORRIS METHOD: DELTA CHOICE EXPLANATION")
    print("="*80)
    
    # Setup
    p = 10  # Grid levels
    grid = np.linspace(0, 1, p)
    
    print(f"\n📊 GRID SETUP:")
    print(f"  Levels (p): {p}")
    print(f"  Grid: {grid}")
    print(f"  Grid spacing: {grid[1] - grid[0]:.6f}")
    
    print(f"\n" + "="*80)
    print("MORRIS (1991) ORIGINAL RECOMMENDATION")
    print("="*80)
    
    print(f"\n📖 Morris recommends delta should:")
    print(f"  1. Create 'significant' perturbations (large enough to detect effects)")
    print(f"  2. Stay within the grid bounds")
    print(f"  3. Common choice: p/[2(p-1)] in normalized space")
    
    # Original Morris delta IN GRID POSITIONS
    original_delta_value = p / (2 * (p - 1))
    original_delta_positions = original_delta_value / (grid[1] - grid[0])
    
    print(f"\n📐 Original Morris delta:")
    print(f"  Formula: p / [2(p-1)] = {p} / {2*(p-1)} = {original_delta_value:.6f}")
    print(f"  In grid positions: {original_delta_value:.6f} / {grid[1]-grid[0]:.6f} = {original_delta_positions:.1f} positions")
    print(f"  Interpretation: Move approximately {int(round(original_delta_positions))} grid positions")
    
    print(f"\n🔍 KEY INSIGHT:")
    print(f"  Morris's value-based delta ≈ {original_delta_value:.4f}")
    print(f"  This equals approximately {original_delta_positions:.1f} grid positions")
    print(f"  For p={p}: {original_delta_positions:.1f} ≈ p/2 = {p//2}")
    print(f"  ✅ So delta_index = p//2 is the GRID EQUIVALENT of Morris's formula!")
    
    print(f"\n" + "="*80)
    print("WHY p//2? THE MATHEMATICAL CONNECTION")
    print("="*80)
    
    print(f"\n📐 Morris (1991) original formula:")
    print(f"  delta_value = p / [2(p-1)]")
    
    print(f"\n📐 Grid spacing:")
    print(f"  For linear grid from 0 to 1 with p levels:")
    print(f"  grid_spacing = 1 / (p-1)")
    
    print(f"\n📐 Converting value-based delta to grid positions:")
    print(f"  delta_positions = delta_value / grid_spacing")
    print(f"  delta_positions = [p / 2(p-1)] / [1 / (p-1)]")
    print(f"  delta_positions = [p / 2(p-1)] × [(p-1) / 1]")
    print(f"  delta_positions = p / 2")
    
    print(f"\n✅ RESULT: Morris's value-based delta ALWAYS equals p/2 grid positions!")
    print(f"   This is true for ANY value of p!")
    
    print(f"\n🎯 Verification for p={p}:")
    print(f"  delta_value = {p} / [2×{p-1}] = {original_delta_value:.6f}")
    print(f"  grid_spacing = 1 / {p-1} = {grid[1]-grid[0]:.6f}")
    print(f"  delta_positions = {original_delta_value:.6f} / {grid[1]-grid[0]:.6f} = {original_delta_positions:.1f}")
    print(f"  p/2 = {p}/2 = {p/2:.1f}")
    print(f"  ✅ They match! {original_delta_positions:.1f} = {p/2:.1f}")
    
    print(f"\n" + "="*80)
    print("DIFFERENT DELTA_INDEX CHOICES")
    print("="*80)
    
    delta_choices = [
        (1, "Minimum (smallest perturbation)"),
        (p // 4, "Conservative (p/4)"),
        (p // 2, "Morris-like (p/2)"),
        (int(round(original_delta_positions)), "Exact Morris equivalent"),
    ]
    
    print(f"\n{'Delta':<15} {'Positions':<12} {'Value Change':<15} {'Description':<30}")
    print("-"*80)
    
    for delta_idx, description in delta_choices:
        value_change = delta_idx * (grid[1] - grid[0])
        print(f"{delta_idx:<15} {delta_idx:<12} {value_change:<15.6f} {description:<30}")
    
    # Show examples
    print(f"\n" + "="*80)
    print("TRAJECTORY EXAMPLES (Starting at grid[5] = 0.555556)")
    print("="*80)
    
    start_idx = 5
    start_val = grid[start_idx]
    
    print(f"\nStarting point: Grid[{start_idx}] = {start_val:.6f}")
    print(f"\nMoving UP (adding delta):")
    print(f"{'Delta Index':<15} {'New Position':<15} {'New Value':<15} {'Valid?':<10}")
    print("-"*65)
    
    for delta_idx, description in delta_choices:
        new_idx = start_idx + delta_idx
        if new_idx < len(grid):
            new_val = grid[new_idx]
            valid = "✅ Valid"
        else:
            new_val = "N/A"
            valid = "❌ Out of bounds"
        print(f"{delta_idx:<15} Grid[{new_idx}]{'':<7} {new_val if new_val != 'N/A' else new_val:<15} {valid:<10}")
    
    print(f"\n" + "="*80)
    print("WHICH DELTA TO CHOOSE?")
    print("="*80)
    
    print(f"\n🎯 DELTA = 1 (Minimum):")
    print(f"  ✅ Always stays in bounds")
    print(f"  ✅ Fine-grained exploration")
    print(f"  ❌ Very small perturbations - may miss large effects")
    print(f"  📌 Use when: You want detailed local sensitivity")
    
    print(f"\n🎯 DELTA = p//4 (Conservative):")
    print(f"  ✅ Moderate perturbations")
    print(f"  ✅ Balances exploration and staying in bounds")
    print(f"  ✅ Works well for most cases")
    print(f"  📌 Use when: You want robust all-purpose Morris sampling")
    
    print(f"\n🎯 DELTA = p//2 (Morris-like):")
    print(f"  ✅ Large perturbations (closer to original Morris)")
    print(f"  ✅ Good for detecting significant effects")
    print(f"  ⚠️  May go out of bounds more often")
    print(f"  📌 Use when: You want maximum sensitivity detection")
    
    print(f"\n🎯 DELTA = {int(round(original_delta_positions))} (Exact Morris):")
    print(f"  ✅ Matches original Morris (1991) recommendation")
    print(f"  ✅ Theoretically optimal balance")
    print(f"  ✅ Standard in literature")
    print(f"  📌 Use when: You want to follow the original paper exactly")
    
    print(f"\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    print(f"\n💡 For your K-value analysis:")
    print(f"  1. If p is small (< 20): Use delta = max(1, p//2) for significant perturbations")
    print(f"  2. If p is medium (20-100): Use delta = max(2, p//4) for balance")
    print(f"  3. If p is large (> 100): Use delta = max(p//10, p//4) to avoid tiny changes")
    print(f"  4. To match Morris (1991) exactly: Use delta = max(1, round(p / (2*(p-1)) / grid_spacing))")
    
    print(f"\n📌 KEY POINT: The important thing is that delta is in GRID POSITIONS,")
    print(f"   not values! Any of these choices is valid as long as it's index-based!")
    
    print(f"\n✅ The original MorrisSampler is broken because it uses delta as a VALUE,")
    print(f"   not as a grid index. As long as you use grid indices, you're good!")


def show_practical_example():
    """Show a practical example with p=1000."""
    
    print(f"\n\n" + "="*80)
    print("PRACTICAL EXAMPLE: p = 1000")
    print("="*80)
    
    p = 1000
    grid = np.linspace(0, 1, p)
    grid_spacing = grid[1] - grid[0]
    
    print(f"\n📊 Setup:")
    print(f"  Grid levels: {p}")
    print(f"  Grid spacing: {grid_spacing:.6f}")
    print(f"  Range: [{grid[0]:.6f}, {grid[-1]:.6f}]")
    
    print(f"\n🔢 Delta choices for p={p}:")
    
    delta_options = [
        ("delta = 1", 1),
        ("delta = p//10", p // 10),
        ("delta = p//4", p // 4),
        ("delta = p//2", p // 2),
        (f"delta = Morris = {int(round(p / (2*(p-1)) / grid_spacing))}", 
         int(round(p / (2*(p-1)) / grid_spacing))),
    ]
    
    print(f"\n{'Choice':<25} {'Positions':<12} {'Value Change':<20} {'Percentage of Range':<20}")
    print("-"*85)
    
    for name, delta_idx in delta_options:
        value_change = delta_idx * grid_spacing
        percentage = (value_change / (grid[-1] - grid[0])) * 100
        print(f"{name:<25} {delta_idx:<12} {value_change:<20.6f} {percentage:<20.2f}%")
    
    print(f"\n💡 Analysis for p={p}:")
    print(f"  • delta=1: Only {1*grid_spacing:.6f} change ({0.1:.2f}%) - TOO SMALL!")
    print(f"  • delta={p//10}: {(p//10)*grid_spacing:.6f} change ({10:.2f}%) - Good for fine details")
    print(f"  • delta={p//4}: {(p//4)*grid_spacing:.6f} change ({25:.2f}%) - RECOMMENDED")
    print(f"  • delta={p//2}: {(p//2)*grid_spacing:.6f} change ({50:.2f}%) - Maximum sensitivity")
    
    print(f"\n✅ For large p (like 1000), you WANT a larger delta_index!")
    print(f"   Otherwise, perturbations are too small to detect effects.")
    print(f"   This is why p//4 or p//2 makes sense, not delta=1!")


if __name__ == "__main__":
    demonstrate_delta_choices()
    show_practical_example()
    
    print(f"\n\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print(f"\n1. Morris (1991) formula: delta = p / [2(p-1)]")
    print(f"2. This ALWAYS equals p/2 grid positions (regardless of p!)")
    print(f"3. Proof: [p / 2(p-1)] / [1/(p-1)] = p/2")
    print(f"4. Therefore: delta_index = p//2 is the EXACT grid equivalent!")
    print(f"\n💡 ANSWER TO YOUR QUESTION:")
    print(f"   Yes! p//2 = 5 for p=10, which gives delta ≈ 0.55")
    print(f"   This matches Morris's 0.5556 value-based formula!")
    print(f"   p//2 IS the correct translation of Morris's formula to grid indices!")
    print(f"\n5. The KEY is using GRID INDICES, not values - that's what fixes the bug!")
    print(f"6. p//4 was unnecessarily conservative - p//2 is the true Morris method!")
