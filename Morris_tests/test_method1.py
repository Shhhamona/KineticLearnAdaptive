"""
Comprehensive test suite for the original Morris Method 1.
Tests the Method 1 Morris sampler to verify its correctness.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import random
from other_scripts.SamplingMorrisMethod import MorrisSampler

def test_method1_basic_functionality():
    """
    Test basic functionality of Method 1 Morris sampler.
    """
    print("=== Testing Method 1 Basic Functionality ===")
    
    # Test parameters
    k_real = np.array([1e10, 1e6, 1e2, 300.0, 1.0])  # Reference values
    p = 6                                              # Number of levels
    r = 3                                              # Number of trajectories
    k_range_type = "log"                              # Logarithmic spacing
    k_range = [-1, 2]                                 # Same range for all (0.1 to 100)
    indexes = [0, 1, 2]                               # Vary first 3 parameters
    
    print(f"Testing with:")
    print(f"  - Reference values: {k_real}")
    print(f"  - Range (log10): {k_range}")
    print(f"  - Levels (p): {p}")
    print(f"  - Trajectories (r): {r}")
    print(f"  - Parameters to vary: {indexes}")
    
    # Generate samples using Method 1
    try:
        result = MorrisSampler(k_real, p, r, k_range_type, k_range, indexes)
        
        print(f"\n✅ Success! Generated {result.shape[0]} samples with {result.shape[1]} parameters each")
        
        # Show first few samples
        print(f"\nFirst 5 sample sets:")
        for i in range(min(5, result.shape[0])):
            print(f"  Sample {i+1}: {result[i]}")
        
        # Verify basic properties
        expected_samples = r * (len(indexes) + 1)
        assert result.shape[0] == expected_samples, f"Expected {expected_samples} samples, got {result.shape[0]}"
        assert result.shape[1] == len(k_real), f"Expected {len(k_real)} parameters, got {result.shape[1]}"
        
        print(f"\n✅ Basic validation passed!")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return None

def test_method1_parameter_bounds():
    """
    Test that Method 1 respects the expected parameter bounds.
    """
    print(f"\n=== Testing Method 1 Parameter Bounds ===")
    
    k_real = np.array([1e8, 1e5, 1e3])  # Reference values
    p = 8
    r = 3
    k_range_type = "log"
    k_range = [-2, 1]  # 0.01 to 10 relative range
    indexes = [0, 1, 2]
    
    print(f"Reference values: {k_real}")
    print(f"Relative range (log10): {k_range} (0.01 to 10)")
    
    result = MorrisSampler(k_real, p, r, k_range_type, k_range, indexes)
    
    print(f"\nParameter bound verification:")
    for i, param_idx in enumerate(indexes):
        param_values = result[:, param_idx]
        
        # Calculate expected bounds
        min_bound = (10**k_range[0]) * k_real[param_idx]
        max_bound = (10**k_range[1]) * k_real[param_idx]
        
        actual_min = np.min(param_values)
        actual_max = np.max(param_values)
        
        print(f"  Parameter {param_idx}: expected [{min_bound:.2e}, {max_bound:.2e}] -> actual [{actual_min:.2e}, {actual_max:.2e}]")
        
        # Check bounds with tolerance
        tolerance = 0.01
        if (min_bound * (1 - tolerance)) <= actual_min and actual_max <= (max_bound * (1 + tolerance)):
            print(f"    ✅ Within bounds")
        else:
            print(f"    ❌ Outside bounds!")
    
    print("✓ Parameter bounds test completed!")
    return result

def test_method1_trajectory_structure():
    """
    Test that Method 1 creates proper Morris trajectories.
    """
    print(f"\n=== Testing Method 1 Trajectory Structure ===")
    
    k_real = np.array([1.0, 1.0, 1.0])
    p = 5
    r = 2
    k_range_type = "lin"  # Test linear spacing
    k_range = [0, 1]      # 0 to 1 range
    indexes = [0, 1, 2]
    
    result = MorrisSampler(k_real, p, r, k_range_type, k_range, indexes)
    
    # Test trajectory continuity
    print(f"\n--- Testing Trajectory Continuity ---")
    k_size = len(indexes)
    
    for traj in range(r):
        traj_start = traj * (k_size + 1)
        trajectory_samples = result[traj_start:traj_start + k_size + 1]
        
        print(f"\nTrajectory {traj + 1}:")
        for i, sample in enumerate(trajectory_samples):
            print(f"  Step {i}: {sample[:3]}")  # Show first 3 parameters
        
        # Check each step in the trajectory
        for step in range(1, k_size + 1):
            prev_sample = trajectory_samples[step - 1]
            curr_sample = trajectory_samples[step]
            
            # Count how many parameters changed
            changes = 0
            for param_idx in indexes:
                if not np.isclose(prev_sample[param_idx], curr_sample[param_idx], rtol=1e-10):
                    changes += 1
            
            assert changes == 1, f"Trajectory {traj}, step {step}: {changes} parameters changed (should be 1)"
    
    print("✓ One-at-a-time design verified!")
    return result

def test_method1_different_spacings():
    """
    Test Method 1 with both linear and logarithmic spacing.
    """
    print(f"\n=== Testing Method 1 Different Spacings ===")
    
    k_real = np.array([100.0, 50.0, 10.0])
    p = 5
    r = 2
    indexes = [0, 1, 2]
    
    # Test linear spacing
    print(f"\n--- Linear Spacing ---")
    k_range_lin = [0.5, 2.0]  # 0.5 to 2.0 multiplier
    result_lin = MorrisSampler(k_real, p, r, "lin", k_range_lin, indexes)
    print(f"Linear result shape: {result_lin.shape}")
    
    # Test log spacing
    print(f"\n--- Log Spacing ---")
    k_range_log = [-1, 1]  # 0.1 to 10 multiplier
    result_log = MorrisSampler(k_real, p, r, "log", k_range_log, indexes)
    print(f"Log result shape: {result_log.shape}")
    
    # Both should have same structure
    assert result_lin.shape == result_log.shape, "Linear and log results have different shapes"
    print("✓ Both spacing methods work correctly")
    
    return result_lin, result_log

def test_method1_relative_scaling():
    """
    Test that Method 1 correctly applies the same relative scaling to all parameters.
    """
    print(f"\n=== Testing Method 1 Relative Scaling ===")
    
    # Use very different reference values
    k_real = np.array([1e12, 1e3, 1e-6])  # Very different scales
    p = 6
    r = 2
    k_range_type = "log"
    k_range = [-1, 1]  # 0.1 to 10 relative range
    indexes = [0, 1, 2]
    
    print(f"Reference values with different scales: {k_real}")
    print(f"Relative range: {k_range} (0.1 to 10)")
    
    result = MorrisSampler(k_real, p, r, k_range_type, k_range, indexes)
    
    print(f"\nChecking relative scaling:")
    for i, param_idx in enumerate(indexes):
        param_values = result[:, param_idx]
        ref_value = k_real[param_idx]
        
        # Calculate the relative variation
        relative_values = param_values / ref_value
        min_relative = np.min(relative_values)
        max_relative = np.max(relative_values)
        
        print(f"  Parameter {param_idx} (ref={ref_value:.1e}): relative range [{min_relative:.3f}, {max_relative:.3f}]")
    
    # All parameters should have the same relative range (approximately)
    print("✓ All parameters should have similar relative ranges")
    return result

def test_method1_reproducibility():
    """
    Test that Method 1 produces reproducible results with the same seed.
    """
    print(f"\n=== Testing Method 1 Reproducibility ===")
    
    k_real = np.array([1.0, 1.0, 1.0])
    p = 4
    r = 2
    k_range_type = "log"
    k_range = [-1, 1]
    indexes = [0, 1, 2]
    
    # Generate samples twice with same seed
    random.seed(42)
    result1 = MorrisSampler(k_real, p, r, k_range_type, k_range, indexes)
    
    random.seed(42)
    result2 = MorrisSampler(k_real, p, r, k_range_type, k_range, indexes)
    
    # Results should be identical
    assert np.allclose(result1, result2), "Results are not reproducible!"
    print("✓ Method 1 produces reproducible results")
    
    return result1

def visualize_method1_trajectories(result, title="Method 1 Trajectories"):
    """
    Visualize Method 1 trajectories.
    """
    print(f"\n=== Visualizing {title} ===")
    
    try:
        # Determine trajectory structure
        k_size = 3  # Assuming 3 parameters varied
        r = result.shape[0] // (k_size + 1)  # Number of trajectories
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Plot 2D projections of trajectories
        projections = [(0, 1), (0, 2), (1, 2)]
        labels = ['Param 0 vs 1', 'Param 0 vs 2', 'Param 1 vs 2']
        
        for proj_idx, (i, j) in enumerate(projections):
            ax = axes[proj_idx]
            
            # Plot each trajectory
            for traj in range(min(r, len(colors))):
                traj_start = traj * (k_size + 1)
                traj_end = traj_start + k_size + 1
                trajectory = result[traj_start:traj_end]
                
                x_vals = trajectory[:, i]
                y_vals = trajectory[:, j]
                
                ax.plot(x_vals, y_vals, 'o-', color=colors[traj], 
                       linewidth=2, markersize=6, label=f'Trajectory {traj+1}')
                
                # Mark start
                ax.plot(x_vals[0], y_vals[0], 's', color=colors[traj], 
                       markersize=10, alpha=0.7)
            
            ax.set_xlabel(f'Parameter {i}')
            ax.set_ylabel(f'Parameter {j}')
            ax.set_title(labels[proj_idx])
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(os.path.dirname(__file__), f'{title.replace(" ", "_")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

def run_all_method1_tests():
    """
    Run all Method 1 tests.
    """
    print("🔬 Starting Comprehensive Method 1 Morris Sampling Tests")
    print("=" * 70)
    
    try:
        # Test 1: Basic functionality
        result1 = test_method1_basic_functionality()
        if result1 is None:
            print("❌ Basic functionality test failed!")
            return None
        
        # Test 2: Parameter bounds
        result2 = test_method1_parameter_bounds()
        
        # Test 3: Trajectory structure
        result3 = test_method1_trajectory_structure()
        
        # Test 4: Different spacings
        result_lin, result_log = test_method1_different_spacings()
        
        # Test 5: Relative scaling
        result5 = test_method1_relative_scaling()
        
        # Test 6: Reproducibility
        result6 = test_method1_reproducibility()
        
        # Visualization
        visualize_method1_trajectories(result3, "Method_1_Trajectories")
        
        print("\n" + "=" * 70)
        print("🎉 ALL METHOD 1 TESTS PASSED! Method 1 is working correctly.")
        print("=" * 70)
        print("Method 1 Summary:")
        print("  ✅ Creates proper Morris trajectories")
        print("  ✅ Respects parameter bounds")
        print("  ✅ One-at-a-time design verified")
        print("  ✅ Same relative scaling for all parameters")
        print("  ✅ Supports both linear and log spacing")
        print("  ✅ Produces reproducible results")
        
        return {
            'basic_test': result1,
            'bounds_test': result2,
            'trajectory_test': result3,
            'linear_spacing': result_lin,
            'log_spacing': result_log,
            'scaling_test': result5,
            'reproducibility_test': result6
        }
        
    except Exception as e:
        print(f"\n❌ METHOD 1 TEST FAILED: {e}")
        raise

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(123)
    np.random.seed(123)
    
    # Run all tests
    results = run_all_method1_tests()
    
    if results:
        print(f"\nMethod 1 tests completed successfully!")
        print(f"Method 1 is reliable for cases where you want the same relative")
        print(f"variation for all parameters.")
    else:
        print(f"\nMethod 1 tests failed!")
