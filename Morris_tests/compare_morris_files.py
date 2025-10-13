#!/usr/bin/env python3
"""
Compare the two Morris files to understand the difference
"""

import numpy as np
import matplotlib.pyplot as plt

def load_and_analyze_morris_file(filepath, name):
    """Load Morris file and analyze parameter distributions"""
    print(f"\n=== Analyzing {name} ===")
    
    # Load data
    data = np.loadtxt(filepath)
    n_samples, n_params = data.shape
    print(f"Shape: {n_samples} samples × {n_params} parameters")
    
    # Analyze first 3 parameters (most variation expected)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{name} - Parameter Distributions', fontsize=14)
    
    for i in range(3):
        param_values = data[:, i]
        unique_values = np.unique(param_values)
        
        print(f"Parameter {i+1}:")
        print(f"  Range: [{param_values.min():.3e}, {param_values.max():.3e}]")
        print(f"  Unique values: {len(unique_values)}")
        print(f"  First 5 values: {param_values[:5]}")
        
        # Plot histogram
        axes[i].hist(param_values, bins=200, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Parameter {i+1}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(f'Morris_tests/{name.replace(" ", "_")}_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return data

def plot_trajectories(data, name, max_trajectories=10):
    """Plot Morris trajectories to see sampling patterns"""
    n_samples, n_params = data.shape
    
    # Assume each trajectory has n_params+1 points
    trajectory_length = n_params + 1
    n_trajectories = n_samples // trajectory_length
    
    print(f"\nTrajectory analysis for {name}:")
    print(f"Assuming {n_trajectories} trajectories of length {trajectory_length}")
    
    # Plot first few trajectories for first 3 parameters
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{name} - Morris Trajectories (first {min(max_trajectories, n_trajectories)})', fontsize=14)
    
    colors = plt.cm.tab10(np.linspace(0, 1, min(max_trajectories, n_trajectories)))
    
    for param_idx in range(3):
        for traj_idx in range(min(max_trajectories, n_trajectories)):
            start_idx = traj_idx * trajectory_length
            end_idx = start_idx + trajectory_length
            
            if end_idx <= n_samples:
                trajectory = data[start_idx:end_idx, param_idx]
                axes[param_idx].plot(range(trajectory_length), trajectory, 
                                   'o-', color=colors[traj_idx], alpha=0.7, 
                                   linewidth=1, markersize=3)
        
        axes[param_idx].set_title(f'Parameter {param_idx+1}')
        axes[param_idx].set_xlabel('Step in Trajectory')
        axes[param_idx].set_ylabel('Parameter Value')
        axes[param_idx].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        axes[param_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Morris_tests/{name.replace(" ", "_")}_trajectories.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    # Load both files
    good_file = r"c:\Users\rodol\Documents\PlasmaML\KineticLearnAdaptive\data\SampleEfficiency\O2_simple_uniform.txt"
    bad_file = r"c:\Users\rodol\Documents\PlasmaML\KineticLearnAdaptive\data\SampleEfficiency\O2_simple_uniform_morris_continous.txt"
    
    print("Comparing Morris sampling files...")
    
    # Analyze both files
    good_data = load_and_analyze_morris_file(good_file, "Good Morris")
    bad_data = load_and_analyze_morris_file(bad_file, "Bad Morris")
    
    # Plot trajectories
    plot_trajectories(good_data, "Good Morris", max_trajectories=5)
    plot_trajectories(bad_data, "Bad Morris", max_trajectories=5)
    
    # Check if values are on a grid
    print("\n=== Grid Analysis ===")
    for name, data in [("Good Morris", good_data), ("Bad Morris", bad_data)]:
        print(f"\n{name}:")
        for i in range(3):
            values = data[:, i]
            unique_values = np.unique(values)
            
            # Check if values look like they're on a regular grid
            if len(unique_values) > 1:
                diffs = np.diff(np.sort(unique_values))
                min_diff = np.min(diffs)
                max_diff = np.max(diffs)
                
                print(f"  Parameter {i+1}:")
                print(f"    Unique values: {len(unique_values)}")
                print(f"    Min step: {min_diff:.3e}")
                print(f"    Max step: {max_diff:.3e}")
                print(f"    Step ratio: {max_diff/min_diff:.2f}")
                
                # Check for grid-like spacing
                if max_diff/min_diff < 2.0:  # Steps are similar
                    print(f"    → Looks like GRID sampling")
                else:
                    print(f"    → Looks like CONTINUOUS sampling")

if __name__ == "__main__":
    main()
