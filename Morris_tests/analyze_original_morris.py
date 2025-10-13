#!/usr/bin/env python3
"""
Analyze the original Morris sampler to understand why it works better
"""

import numpy as np

def analyze_good_morris():
    """Analyze the good Morris file to understand its grid structure"""
    
    # Load the good Morris data
    good_file = r"c:\Users\rodol\Documents\PlasmaML\KineticLearnAdaptive\data\SampleEfficiency\O2_simple_morris.txt"
    data = np.loadtxt(good_file)
    
    print("=== Good Morris Grid Analysis ===")
    print(f"Shape: {data.shape}")
    
    # Analyze first 3 parameters
    for param_idx in range(3):
        values = data[:, param_idx]
        unique_values = np.sort(np.unique(values))
        
        print(f"\nParameter {param_idx + 1}:")
        print(f"  Range: [{values.min():.6e}, {values.max():.6e}]")
        print(f"  Unique values: {len(unique_values)}")
        print(f"  First 5 unique: {unique_values[:5]}")
        print(f"  Last 5 unique: {unique_values[-5:]}")
        
        # Check if values are multiples
        if len(unique_values) > 1:
            ratios = unique_values[1:] / unique_values[:-1]
            print(f"  Ratios between consecutive values: {ratios[:5]}")
            
            # Check for linear progression
            diffs = np.diff(unique_values)
            print(f"  Differences between consecutive: {diffs[:5]}")
            print(f"  Min diff: {diffs.min():.6e}")
            print(f"  Max diff: {diffs.max():.6e}")
            
            # Check if it's a regular grid
            base_step = diffs[0]
            is_regular = np.allclose(diffs, base_step, rtol=1e-10)
            print(f"  Regular grid with step {base_step:.6e}: {is_regular}")

        # Visualize the output (y) distributions (last three columns)
        import matplotlib.pyplot as plt
        y_start = data.shape[1] - 3
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Output (y) Distributions', fontsize=14)
        for i in range(3):
            y_values = data[:, y_start + i]
            axes[i].hist(y_values, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Output y{i+1}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        plt.tight_layout()
        plt.show()

def test_original_morris_reconstruction():
    """Try to reconstruct the original Morris method"""
    
    print("\n=== Testing Original Morris Reconstruction ===")
    
    # Parameters from the original file generation
    k_range = [-15.5, -12.5]  # log range
    p = 20  # number of levels
    
    # Create log grid
    w = np.logspace(k_range[0], k_range[1], p, base=10)
    print(f"Grid values: {w}")
    print(f"Grid in scientific notation:")
    for i, val in enumerate(w):
        print(f"  Level {i}: {val:.6e}")
    
    # Calculate delta as in Morris original
    delta_fraction = 1.0 / (p - 1)
    print(f"Delta fraction: {delta_fraction}")
    
    # Check against good Morris values
    good_file = r"c:\Users\rodol\Documents\PlasmaML\KineticLearnAdaptive\data\SampleEfficiency\O2_simple_morris.txt"
    data = np.loadtxt(good_file)
    param1_unique = np.sort(np.unique(data[:, 0]))
    
    print(f"\nComparison for Parameter 1:")
    print(f"Reconstructed grid matches: {np.allclose(w, param1_unique, rtol=1e-6)}")
    
    if len(w) == len(param1_unique):
        for i, (grid_val, data_val) in enumerate(zip(w, param1_unique)):
            print(f"  Level {i}: Grid={grid_val:.6e}, Data={data_val:.6e}, Match={np.isclose(grid_val, data_val, rtol=1e-6)}")

if __name__ == "__main__":
    analyze_good_morris()
    test_original_morris_reconstruction()
