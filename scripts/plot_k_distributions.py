#!/usr/bin/env python3
"""
Compare K value distributions between uniform dataset, initial training, and newly sampled data.
Creates histograms to visualize if the sampled K values are in the same range as training data.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def load_data():
    """Load train_600_Y.txt, split into initial vs sampled, and load test data."""
    
    # Load train_600_Y.txt (scaled k values)
    train_600_file = r"c:\Users\Rodolfo Simões\Documents\PlasmaML\KineticLearn\results\training_snapshots\train_600_Y.txt"
    train_600_y = np.loadtxt(train_600_file)
    
    # Load test_Y.txt (scaled k values)
    test_file = r"c:\Users\Rodolfo Simões\Documents\PlasmaML\KineticLearn\results\training_snapshots\test_Y.txt"
    test_y = np.loadtxt(test_file)
    
    # Split into initial training (first 50) and newly sampled (last 10)
    initial_k_scaled = train_600_y[:100]  # First 50 samples (initial training)
    sampled_k_scaled = train_600_y[-500:]  # Last 10 samples (newly sampled)
    
    print(f"Loaded data shapes:")
    print(f"  Total samples in train_600_Y: {train_600_y.shape}")
    print(f"  Initial training (scaled): {initial_k_scaled.shape}")
    print(f"  Newly sampled (scaled): {sampled_k_scaled.shape}")
    print(f"  Test data (scaled): {test_y.shape}")
    
    return initial_k_scaled, sampled_k_scaled, test_y

def plot_k_distributions():
    """Create histogram comparisons for each K value including test data."""
    
    initial_k_scaled, sampled_k_scaled, test_k_scaled = load_data()
    
    # Create figure with subplots for each K value
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('K Value Distribution Comparison (Scaled Data)', fontsize=16)
    
    k_names = ['k1', 'k2', 'k3']
    
    for i in range(3):
        ax = axes[i]
        
        # Plot histograms of scaled K values
        ax.hist(initial_k_scaled[:, i], bins=15, alpha=0.7, label=f'Initial Training ({initial_k_scaled.shape[0]})', 
               color='green', density=True)
        ax.hist(sampled_k_scaled[:, i], bins=10, alpha=0.7, label=f'Newly Sampled ({sampled_k_scaled.shape[0]})', 
               color='red', density=True)
        ax.hist(test_k_scaled[:, i], bins=20, alpha=0.7, label=f'Test Data ({test_k_scaled.shape[0]})', 
               color='blue', density=True)
        
        ax.set_xlabel(f'{k_names[i]} (scaled)')
        ax.set_ylabel('Density')
        ax.set_title(f'{k_names[i]} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = r"c:\Users\Rodolfo Simões\Documents\PlasmaML\KineticLearn\results"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "k_distributions_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved to: {plot_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (Scaled K Values)")
    print("="*80)
    
    for i in range(3):
        print(f"\n{k_names[i]} Statistics:")
        print(f"  Initial Training ({initial_k_scaled.shape[0]} samples):")
        print(f"    Range: [{initial_k_scaled[:, i].min():.6f}, {initial_k_scaled[:, i].max():.6f}]")
        print(f"    Mean: {initial_k_scaled[:, i].mean():.6f}")
        print(f"    Std:  {initial_k_scaled[:, i].std():.6f}")
        
        print(f"  Newly Sampled ({sampled_k_scaled.shape[0]} samples):")
        print(f"    Range: [{sampled_k_scaled[:, i].min():.6f}, {sampled_k_scaled[:, i].max():.6f}]")
        print(f"    Mean: {sampled_k_scaled[:, i].mean():.6f}")
        print(f"    Std:  {sampled_k_scaled[:, i].std():.6f}")
        
        print(f"  Test Data ({test_k_scaled.shape[0]} samples):")
        print(f"    Range: [{test_k_scaled[:, i].min():.6f}, {test_k_scaled[:, i].max():.6f}]")
        print(f"    Mean: {test_k_scaled[:, i].mean():.6f}")
        print(f"    Std:  {test_k_scaled[:, i].std():.6f}")
        
        # Check if sampled data overlaps with initial training range
        initial_min, initial_max = initial_k_scaled[:, i].min(), initial_k_scaled[:, i].max()
        sampled_min, sampled_max = sampled_k_scaled[:, i].min(), sampled_k_scaled[:, i].max()
        test_min, test_max = test_k_scaled[:, i].min(), test_k_scaled[:, i].max()
        
        # Check for overlap between training and sampled
        train_sample_overlap = not (sampled_max < initial_min or sampled_min > initial_max)
        print(f"  Training-Sampled overlap: {'✅ YES' if train_sample_overlap else '❌ NO'}")
        
        # Check for overlap between training and test
        train_test_overlap = not (test_max < initial_min or test_min > initial_max)
        print(f"  Training-Test overlap: {'✅ YES' if train_test_overlap else '❌ NO'}")
        
        # Check if sampled data is within initial training range
        within_range = (sampled_min >= initial_min) and (sampled_max <= initial_max)
        print(f"  Sampled data within initial range: {'✅ YES' if within_range else '❌ NO'}")
        
        # Check if test data is within initial training range
        test_within_train = (test_min >= initial_min) and (test_max <= initial_max)
        print(f"  Test data within initial range: {'✅ YES' if test_within_train else '❌ NO'}")
        
        if not within_range:
            if sampled_min < initial_min:
                print(f"    ⚠️ Sampled minimum ({sampled_min:.6f}) below initial minimum ({initial_min:.6f})")
            if sampled_max > initial_max:
                print(f"    ⚠️ Sampled maximum ({sampled_max:.6f}) above initial maximum ({initial_max:.6f})")
        
        # Calculate relative difference in means
        mean_diff = abs(sampled_k_scaled[:, i].mean() - initial_k_scaled[:, i].mean())
        rel_mean_diff = mean_diff / abs(initial_k_scaled[:, i].mean()) * 100
        print(f"  Relative difference in means: {rel_mean_diff:.2f}%")

def main():
    print("K Value Distribution Analysis (train_600_Y + test data)")
    print("="*60)
    
    try:
        plot_k_distributions()
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
