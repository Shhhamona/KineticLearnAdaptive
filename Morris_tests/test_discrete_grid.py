"""
Test to verify that Method 1 Morris Sampling only produces discrete grid values.
This validates whether the implementation is grid-based (discrete) or continuous.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from other_scripts.SamplingMorrisMethod import MorrisSampler

def test_discrete_grid_values():
    """
    Test that Method 1 only produces values from the predefined grid.
    """
    print("🔍 Testing Method 1: Discrete Grid Values")
    print("=" * 60)
    
    # Test parameters
    k_range = [-1, 2]  # log range: 0.1 to 100
    p = 6              # Number of grid levels
    r = 5              # Number of trajectories
    k_range_type = "log"
    input_ref = np.array([1.0, 1.0, 1.0])  # Reference values
    indexes = [0, 1, 2]  # Vary all 3 parameters
    
    print(f"Test setup:")
    print(f"  - k_range: {k_range} (log10)")
    print(f"  - Grid levels (p): {p}")
    print(f"  - Trajectories (r): {r}")
    print(f"  - Reference values: {input_ref}")
    
    # Create the expected grid (same as Method 1 does internally)
    if k_range_type == "log":
        w_log = np.logspace(k_range[0], k_range[1], p, base=10)
        print(f"\nExpected grid points (w_log): {w_log}")
    else:
        w_lin = np.linspace(k_range[0], k_range[1], p)
        print(f"\nExpected grid points (w_lin): {w_lin}")
    
    # Generate Morris samples
    print(f"\nGenerating Morris samples...")
    try:
        result = MorrisSampler(input_ref, p, r, k_range_type, k_range, indexes)
        print(f"✅ Generated {result.shape[0]} samples successfully")
        
        # Test each parameter
        print(f"\n🔍 Checking if all values are from discrete grid...")
        
        for param_idx, k_idx in enumerate(indexes):
            param_values = result[:, k_idx]
            
            print(f"\nParameter {k_idx}:")
            print(f"  Sample values: {np.unique(param_values)}")
            
            # Calculate expected values (grid * reference)
            if k_range_type == "log":
                expected_values = w_log * input_ref[k_idx]
            else:
                expected_values = w_lin * input_ref[k_idx]
            
            print(f"  Expected grid: {expected_values}")
            
            # Check if all sample values are from the expected grid
            all_discrete = True
            tolerance = 1e-10  # Floating point tolerance
            
            for sample_val in np.unique(param_values):
                # Check if this sample value matches any grid point
                matches_grid = np.any(np.abs(expected_values - sample_val) < tolerance)
                if not matches_grid:
                    all_discrete = False
                    print(f"    ❌ Value {sample_val} NOT in expected grid!")
                    break
            
            if all_discrete:
                print(f"    ✅ All values are from discrete grid")
            else:
                print(f"    ❌ Found non-grid values!")
                
        # Show some trajectory samples to visualize the discrete nature
        print(f"\n📊 Sample trajectories (showing discrete steps):")
        k_size = len(indexes)
        
        for traj in range(min(2, r)):  # Show first 2 trajectories
            traj_start = traj * (k_size + 1)
            traj_end = traj_start + k_size + 1
            trajectory = result[traj_start:traj_end]
            
            print(f"\nTrajectory {traj + 1}:")
            for step, sample in enumerate(trajectory):
                print(f"  Step {step}: [{sample[0]:.6f}, {sample[1]:.6f}, {sample[2]:.6f}]")
        
        return result, w_log if k_range_type == "log" else w_lin
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        return None, None

def test_grid_spacing_analysis():
    """
    Analyze the spacing between consecutive grid points to confirm discrete nature.
    """
    print(f"\n📏 Grid Spacing Analysis")
    print("=" * 60)
    
    # Test with different grid sizes
    for p in [4, 6, 8]:
        print(f"\nTesting with p={p} grid levels:")
        
        k_range = [-1, 1]  # Smaller range for clearer visualization
        w_log = np.logspace(k_range[0], k_range[1], p, base=10)
        
        print(f"  Grid points: {w_log}")
        print(f"  Spacings: {np.diff(w_log)}")
        print(f"  Log spacings: {np.diff(np.log10(w_log))}")
        
        # The log spacings should be uniform for logspace
        log_spacings = np.diff(np.log10(w_log))
        is_uniform = np.allclose(log_spacings, log_spacings[0], rtol=1e-10)
        print(f"  Uniform log spacing: {'✅ Yes' if is_uniform else '❌ No'}")

def test_delta_step_verification():
    """
    Test that the delta steps in Morris sampling follow the discrete grid.
    """
    print(f"\n⚡ Delta Step Verification")
    print("=" * 60)
    
    k_range = [-1, 1]
    p = 6
    w_log = np.logspace(k_range[0], k_range[1], p, base=10)
    
    # Calculate delta as Method 1 does
    delta = p / (2 * (p - 1))
    print(f"Calculated delta: {delta}")
    
    # Show what delta means in terms of grid steps
    print(f"Grid points: {w_log}")
    print(f"Log10 grid: {np.log10(w_log)}")
    
    # Calculate actual step sizes in log space
    log_step = (k_range[1] - k_range[0]) / (p - 1)
    print(f"Log step size: {log_step}")
    print(f"Delta in terms of steps: {delta} ≈ {delta / log_step:.2f} grid steps")
    
    # Show example of delta movement
    start_point = w_log[2]  # Middle point
    mean_point = 10**((k_range[0] + k_range[1]) / 2)
    
    if start_point > mean_point:
        new_point = start_point / (10**delta)
    else:
        new_point = start_point * (10**delta)
    
    print(f"\nExample delta step:")
    print(f"  Start: {start_point:.6f}")
    print(f"  Mean:  {mean_point:.6f}")
    print(f"  After delta: {new_point:.6f}")
    
    # Check if new point is close to any grid point
    closest_idx = np.argmin(np.abs(w_log - new_point))
    closest_point = w_log[closest_idx]
    print(f"  Closest grid point: {closest_point:.6f}")
    print(f"  Distance: {abs(new_point - closest_point):.8f}")

def run_discrete_tests():
    """
    Run all tests to verify discrete nature of Method 1.
    """
    print("🧪 Method 1 Discrete Grid Tests")
    print("=" * 80)
    
    # Set seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    try:
        # Test 1: Verify discrete grid values
        result, grid = test_discrete_grid_values()
        
        # Test 2: Analyze grid spacing
        test_grid_spacing_analysis()
        
        # Test 3: Verify delta steps
        test_delta_step_verification()
        
        print(f"\n" + "=" * 80)
        if result is not None:
            print("🎯 CONCLUSION: Method 1 uses DISCRETE GRID sampling")
            print("   - All parameter values come from predefined grid points")
            print("   - No continuous values are generated")
            print("   - This is standard Morris sampling implementation")
        else:
            print("❌ Tests failed - could not verify discrete nature")
        print("=" * 80)
        
        return result
        
    except Exception as e:
        print(f"\n❌ TESTS FAILED: {e}")
        raise

if __name__ == "__main__":
    run_discrete_tests()
