#!/usr/bin/env python3
"""
Test that apply_training_scalers is fully defensive and idempotent.
"""

import numpy as np
import sys
sys.path.append('.')
from active_learning_methods import apply_training_scalers, load_datasets

def test_apply_training_scalers_idempotent():
    """Test that calling apply_training_scalers multiple times gives identical results."""
    
    # Load reference dataset for scalers
    print("Loading reference dataset...")
    dataset_train, dataset_test = load_datasets(
        'data/SampleEfficiency/O2_simple_uniform.txt',
        'data/SampleEfficiency/O2_simple_test.txt',
        nspecies=3,
        num_pressure_conditions=2
    )
    
    # Create test data
    n_sims = 5
    nspecies = 3
    num_pressure_conditions = 2
    
    # Create raw test data
    raw_compositions = np.random.rand(n_sims * num_pressure_conditions, nspecies) * 1e-3
    raw_k_values = np.random.rand(n_sims, 3) * 1e-15
    
    print(f"Original raw_compositions hash: {hash(raw_compositions.tobytes())}")
    print(f"Original raw_k_values hash: {hash(raw_k_values.tobytes())}")
    print(f"Original raw_compositions first row: {raw_compositions[0]}")
    print(f"Original raw_k_values first row: {raw_k_values[0]}")
    
    # Call apply_training_scalers first time
    print("\nFirst call to apply_training_scalers...")
    new_x_1, new_y_1 = apply_training_scalers(
        raw_compositions, raw_k_values, dataset_train, 
        nspecies, num_pressure_conditions, debug=False
    )
    
    # Check if original arrays were modified
    print(f"After 1st call - raw_compositions hash: {hash(raw_compositions.tobytes())}")
    print(f"After 1st call - raw_k_values hash: {hash(raw_k_values.tobytes())}")
    print(f"After 1st call - raw_compositions first row: {raw_compositions[0]}")
    print(f"After 1st call - raw_k_values first row: {raw_k_values[0]}")
    
    # Call apply_training_scalers second time
    print("\nSecond call to apply_training_scalers...")
    new_x_2, new_y_2 = apply_training_scalers(
        raw_compositions, raw_k_values, dataset_train, 
        nspecies, num_pressure_conditions, debug=False
    )
    
    # Check if original arrays were modified again
    print(f"After 2nd call - raw_compositions hash: {hash(raw_compositions.tobytes())}")
    print(f"After 2nd call - raw_k_values hash: {hash(raw_k_values.tobytes())}")
    print(f"After 2nd call - raw_compositions first row: {raw_compositions[0]}")
    print(f"After 2nd call - raw_k_values first row: {raw_k_values[0]}")
    
    # Check if outputs are identical
    print("\nComparing outputs...")
    x_identical = np.allclose(new_x_1, new_x_2, rtol=1e-15, atol=1e-15)
    y_identical = np.allclose(new_y_1, new_y_2, rtol=1e-15, atol=1e-15)
    
    print(f"X outputs identical: {x_identical}")
    print(f"Y outputs identical: {y_identical}")
    
    if x_identical and y_identical:
        print("✅ SUCCESS: apply_training_scalers is idempotent!")
        print(f"   Max X difference: {np.max(np.abs(new_x_1 - new_x_2))}")
        print(f"   Max Y difference: {np.max(np.abs(new_y_1 - new_y_2))}")
    else:
        print("❌ FAILURE: apply_training_scalers is NOT idempotent!")
        print(f"   Max X difference: {np.max(np.abs(new_x_1 - new_x_2))}")
        print(f"   Max Y difference: {np.max(np.abs(new_y_1 - new_y_2))}")
        
        # Show first few differences
        if not x_identical:
            print("   X differences (first 3 rows):")
            print(new_x_1[:3] - new_x_2[:3])
        if not y_identical:
            print("   Y differences (first 3 rows):")
            print(new_y_1[:3] - new_y_2[:3])
    
    return x_identical and y_identical

if __name__ == '__main__':
    success = test_apply_training_scalers_idempotent()
    exit(0 if success else 1)
