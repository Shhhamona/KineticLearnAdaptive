#!/usr/bin/env python3
"""
Morris Method: Analysis of p (grid levels) values

This script analyzes whether different values of p work correctly with
the original Morris delta formula, and whether there are constraints on p.
"""

import numpy as np

def analyze_p_value(p):
    """Analyze if a specific p value works correctly with Morris method."""
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS FOR p = {p}")
    print(f"{'='*80}")
    
    # Create linear grid
    grid = np.linspace(0, 1, p)
    grid_spacing = grid[1] - grid[0] if p > 1 else 0
    
    print(f"\n📊 Grid Setup:")
    print(f"  Levels: {p}")
    print(f"  Grid spacing: {grid_spacing:.10f}")
    print(f"  Range: [0.0, 1.0]")
    
    # Morris original delta (value-based)
    if p > 1:
        delta_value = p / (2 * (p - 1))
        print(f"\n📐 Morris Formula:")
        print(f"  delta = p / [2(p-1)] = {p} / [2×{p-1}] = {delta_value:.10f}")
        
        # Convert to grid positions
        delta_positions = delta_value / grid_spacing
        print(f"\n🔍 Grid Position Analysis:")
        print(f"  delta_positions = {delta_value:.10f} / {grid_spacing:.10f}")
        print(f"  delta_positions = {delta_positions:.10f}")
        
        # Check if it's an integer or close to integer
        rounded_positions = round(delta_positions)
        error = abs(delta_positions - rounded_positions)
        
        print(f"\n✅ Rounding:")
        print(f"  Closest integer: {rounded_positions}")
        print(f"  Rounding error: {error:.10e}")
        
        # Mathematical proof
        print(f"\n📖 Mathematical Proof:")
        print(f"  delta_positions = [p / 2(p-1)] / [1 / (p-1)]")
        print(f"  delta_positions = [p / 2(p-1)] × (p-1)")
        print(f"  delta_positions = p / 2")
        print(f"  delta_positions = {p} / 2 = {p/2}")
        
        print(f"\n🎯 Result:")
        print(f"  Theoretical: p/2 = {p/2}")
        print(f"  Calculated: {delta_positions:.10f}")
        print(f"  Match: {abs(delta_positions - p/2) < 1e-10}")
        
        # Check if p/2 is an integer
        if p % 2 == 0:
            print(f"\n✅ p={p} is EVEN:")
            print(f"  → p/2 = {p//2} is an INTEGER")
            print(f"  → delta_index = {p//2} EXACTLY represents Morris's delta")
            print(f"  → Samples will ALWAYS stay on grid")
            status = "✅ PERFECT - No rounding needed"
        else:
            print(f"\n⚠️  p={p} is ODD:")
            print(f"  → p/2 = {p/2} is NOT an integer")
            print(f"  → Must round: {p//2} or {p//2 + 1}")
            print(f"  → Slight deviation from Morris's exact delta")
            status = "⚠️  WORKS - but requires rounding"
        
        return status, delta_positions, p/2
    else:
        print(f"\n❌ p=1 is not valid (need at least 2 levels for a grid)")
        return "❌ INVALID", None, None


def compare_p_values():
    """Compare different p values to understand constraints."""
    
    print("="*80)
    print("MORRIS METHOD: p VALUE CONSTRAINTS ANALYSIS")
    print("="*80)
    
    print(f"\n📖 According to Morris (1991):")
    print(f"  • p = number of levels in the grid")
    print(f"  • Common values in literature: p = 4, 6, 8, 10")
    print(f"  • No strict upper limit mentioned")
    print(f"  • BUT: p should be chosen such that delta creates meaningful perturbations")
    
    print(f"\n" + "="*80)
    print("TESTING VARIOUS p VALUES")
    print("="*80)
    
    test_values = [4, 6, 8, 10, 11, 12, 20, 50, 100, 1000]
    
    results = []
    for p in test_values:
        status, calc_delta, theoretical_delta = analyze_p_value(p)
        results.append((p, status, calc_delta, theoretical_delta))
    
    # Summary table
    print(f"\n\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    print(f"\n{'p':<10} {'Even/Odd':<12} {'p/2':<12} {'delta_index':<15} {'Status':<30}")
    print("-"*80)
    
    for p, status, calc_delta, theoretical_delta in results:
        if theoretical_delta is not None:
            even_odd = "EVEN ✅" if p % 2 == 0 else "ODD ⚠️"
            delta_idx = p // 2
            print(f"{p:<10} {even_odd:<12} {theoretical_delta:<12.1f} {delta_idx:<15} {status:<30}")
    
    print(f"\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    print(f"\n1️⃣  MATHEMATICAL FACT:")
    print(f"   Morris's delta ALWAYS equals p/2 grid positions (for ANY p > 1)")
    print(f"   Proof: delta_positions = [p/(2(p-1))] / [1/(p-1)] = p/2")
    
    print(f"\n2️⃣  EVEN vs ODD p:")
    print(f"   • EVEN p (4, 6, 8, 10, 12, ...): p/2 is integer → PERFECT!")
    print(f"   • ODD p (5, 7, 9, 11, 13, ...): p/2 is half-integer → Must round!")
    
    print(f"\n3️⃣  IS ODD p ALLOWED?")
    print(f"   ✅ YES! Morris (1991) doesn't prohibit odd p")
    print(f"   ⚠️  BUT: You must round p/2 to an integer for grid-based implementation")
    print(f"   📌 Common practice: Use even p to avoid rounding")
    
    print(f"\n4️⃣  WHAT ABOUT p=11?")
    print(f"   • p=11 is ODD → p/2 = 5.5")
    print(f"   • delta_index must be 5 or 6 (round down or up)")
    print(f"   • Using delta_index=5 gives delta ≈ 0.500 (vs Morris's 0.550)")
    print(f"   • Using delta_index=6 gives delta ≈ 0.600 (vs Morris's 0.550)")
    print(f"   • Either way, slight deviation from Morris's exact value")
    print(f"   ✅ Still valid! Just not exact")
    
    print(f"\n5️⃣  ORIGINAL MorrisSampler BUG:")
    print(f"   The bug is NOT about odd/even p!")
    print(f"   The bug is using VALUE (0.550) instead of INDEX (5 or 6)")
    print(f"   • p=10 (even): Value 0.5556 happens to land on grid (by luck!)")
    print(f"   • p=11 (odd): Value 0.5500 does NOT land on grid → reveals the bug!")


def demonstrate_odd_p_solution():
    """Show how to handle odd p values correctly."""
    
    print(f"\n\n" + "="*80)
    print("SOLUTION FOR ODD p VALUES")
    print("="*80)
    
    p = 11
    grid = np.linspace(0, 1, p)
    grid_spacing = grid[1] - grid[0]
    
    print(f"\n📊 Example: p = {p} (ODD)")
    print(f"  Theoretical: p/2 = {p/2}")
    print(f"  Must choose: {p//2} or {p//2 + 1}")
    
    print(f"\n🎯 Option 1: delta_index = {p//2} (round down)")
    delta_idx_down = p // 2
    delta_value_down = delta_idx_down * grid_spacing
    morris_delta = p / (2 * (p - 1))
    error_down = abs(delta_value_down - morris_delta)
    print(f"  Value change: {delta_idx_down} × {grid_spacing:.4f} = {delta_value_down:.4f}")
    print(f"  Morris delta: {morris_delta:.4f}")
    print(f"  Error: {error_down:.4f} ({(error_down/morris_delta)*100:.1f}%)")
    
    print(f"\n🎯 Option 2: delta_index = {p//2 + 1} (round up)")
    delta_idx_up = p // 2 + 1
    delta_value_up = delta_idx_up * grid_spacing
    error_up = abs(delta_value_up - morris_delta)
    print(f"  Value change: {delta_idx_up} × {grid_spacing:.4f} = {delta_value_up:.4f}")
    print(f"  Morris delta: {morris_delta:.4f}")
    print(f"  Error: {error_up:.4f} ({(error_up/morris_delta)*100:.1f}%)")
    
    print(f"\n✅ RECOMMENDATION:")
    if error_down < error_up:
        print(f"  Use delta_index = {delta_idx_down} (smaller error)")
    else:
        print(f"  Use delta_index = {delta_idx_up} (smaller error)")
    
    print(f"\n💡 OR use round() for best approximation:")
    delta_idx_round = round(p / 2)
    delta_value_round = delta_idx_round * grid_spacing
    error_round = abs(delta_value_round - morris_delta)
    print(f"  delta_index = round({p/2}) = {delta_idx_round}")
    print(f"  Value change: {delta_value_round:.4f}")
    print(f"  Error: {error_round:.4f} ({(error_round/morris_delta)*100:.1f}%)")


def final_recommendations():
    """Provide clear recommendations."""
    
    print(f"\n\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    
    print(f"\n1️⃣  CHOOSE EVEN p WHEN POSSIBLE:")
    print(f"   • p = 4, 6, 8, 10, 12, 20, 50, 100, 1000, ...")
    print(f"   • p/2 will be an exact integer")
    print(f"   • delta_index = p//2 = exact Morris delta")
    
    print(f"\n2️⃣  IF YOU MUST USE ODD p:")
    print(f"   • Use delta_index = round(p/2)")
    print(f"   • Small deviation from Morris, but still valid")
    print(f"   • Example: p=11 → delta_index = round(5.5) = 6")
    
    print(f"\n3️⃣  CORRECTED IMPLEMENTATION:")
    print(f"   ```python")
    print(f"   delta_index = round(p / 2)  # Works for both even and odd p")
    print(f"   # OR for exact match when p is even:")
    print(f"   delta_index = p // 2")
    print(f"   ```")
    
    print(f"\n4️⃣  WHY THE ORIGINAL IS BROKEN:")
    print(f"   ```python")
    print(f"   # WRONG (original MorrisSampler):")
    print(f"   delta = p / (2 * (p - 1))  # Value: 0.550")
    print(f"   new_value = current_value + delta  # May go off grid!")
    print(f"   ")
    print(f"   # CORRECT (CorrectedGridMorrisSampler):")
    print(f"   delta_index = p // 2  # Grid positions: 5")
    print(f"   new_value = grid[current_index + delta_index]  # Stays on grid!")
    print(f"   ```")
    
    print(f"\n5️⃣  NO CONSTRAINT ON p VALUE:")
    print(f"   • Any p ≥ 2 is theoretically valid")
    print(f"   • Practical range: 4 ≤ p ≤ 1000")
    print(f"   • Even p is preferred but odd p works too (with rounding)")


if __name__ == "__main__":
    # Main analysis
    compare_p_values()
    
    # Show solution for odd p
    demonstrate_odd_p_solution()
    
    # Final recommendations
    final_recommendations()
    
    print(f"\n\n" + "="*80)
    print("✅ CONCLUSION")
    print("="*80)
    print(f"\nTO ANSWER YOUR QUESTIONS:")
    print(f"\n1. Is p=11 allowed in Morris method?")
    print(f"   ✅ YES! Any p ≥ 2 is allowed")
    print(f"\n2. Why does p=11 show the bug?")
    print(f"   Because p/2 = 5.5 (not integer)")
    print(f"   Morris's value delta = 0.550 doesn't align with grid spacing = 0.100")
    print(f"   Using VALUE 0.550 → goes off grid")
    print(f"   Using INDEX 5 or 6 → stays on grid")
    print(f"\n3. Is there a limitation on p?")
    print(f"   No strict limit! But prefer EVEN p for exact Morris delta match")
    print(f"\n4. Does p=1000 work?")
    print(f"   ✅ YES! p=1000 is even → p/2 = 500 (integer) → PERFECT!")
