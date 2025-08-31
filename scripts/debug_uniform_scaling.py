#!/usr/bin/env python3
"""Quick debug script to check uniform dataset scaling"""

import sys
sys.path.append('.')
from active_learning_methods import load_datasets

# Configuration
config = {
    'nspecies': 3,
    'num_pressure_conditions': 2,
}

# File paths
src_file_train = 'data/SampleEfficiency/O2_simple_uniform.txt'
src_file_test = 'data/SampleEfficiency/O2_simple_test.txt'

print("🔍 QUICK UNIFORM DATASET CHECK")

# Load uniform datasets
uniform_dataset, test_dataset = load_datasets(
    src_file_train, src_file_test, config['nspecies'], config['num_pressure_conditions']
)

# Check what's in the uniform dataset
x_uniform, y_uniform = uniform_dataset.get_data()
print(f"Uniform dataset:")
print(f"   X shape: {x_uniform.shape}")
print(f"   Y shape: {y_uniform.shape}")
print(f"   X range: [{x_uniform.min():.6f}, {x_uniform.max():.6f}]")
print(f"   Y range: [{y_uniform.min():.6f}, {y_uniform.max():.6f}]")
print(f"   X first 3 rows:\n{x_uniform[:3]}")
print(f"   Y first 3 rows:\n{y_uniform[:3]}")

print(f"\nExpected:")
print(f"   X should be in [-1, 1] range (MaxAbsScaler)")
print(f"   Y should be in [0, 1] range (MaxAbsScaler)")

if x_uniform.max() > 2 or x_uniform.min() < -2:
    print(f"🚨 X DATA IS NOT SCALED!")
else:
    print(f"✅ X data appears to be scaled correctly")

if y_uniform.max() > 2 or y_uniform.min() < -2:
    print(f"🚨 Y DATA IS NOT SCALED!")
else:
    print(f"✅ Y data appears to be scaled correctly")
