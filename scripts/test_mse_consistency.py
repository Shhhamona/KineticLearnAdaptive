#!/usr/bin/env python3
"""
Quick test to verify MSE calculation consistency between functions.
"""

from active_learning_methods import load_datasets, calculate_mse_for_dataset, retrain_models_with_new_data
from sklearn.utils import shuffle
import numpy as np

# Load datasets
src_file_train = 'data/SampleEfficiency/O2_simple_uniform.txt'
src_file_test = 'data/SampleEfficiency/O2_simple_test.txt'
nspecies = 3
num_pressure_conditions = 2
best_params = [
    {'C': 10, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'},
    {'C': 20, 'epsilon': 0.005, 'gamma': 5, 'kernel': 'rbf'},
    {'C': 5, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'}
]

print("Loading datasets...")
dataset_train, dataset_test = load_datasets(src_file_train, src_file_test, nspecies, num_pressure_conditions)

# Test with 500 samples using seed=42
print("\n🧪 CONSISTENCY TEST: 500 samples, seed=42")

# Method 1: calculate_mse_for_dataset
print("\n1. Using calculate_mse_for_dataset...")
mse_list, total_mse_list = calculate_mse_for_dataset(dataset_train, dataset_test, best_params, [500], seed=42)
mse_method1 = total_mse_list[0]
print(f"   Method 1 MSE: {mse_method1:.6f}")

# Method 2: retrain_models_with_new_data (simulating initial training)
print("\n2. Using retrain_models_with_new_data (empty new data)...")
x_all, y_all = dataset_train.get_data()
x_shuf, y_shuf = shuffle(x_all, y_all, random_state=42)
current_x_train = x_shuf[:500]
current_y_train = y_shuf[:500]

# Retrain with zero new samples to test consistency
empty_x = np.empty((0, current_x_train.shape[1]))
empty_y = np.empty((0, current_y_train.shape[1]))

new_models, new_mse_per_output, new_total_mse, augmented_size, _, _ = retrain_models_with_new_data(
    current_x_train=current_x_train,
    current_y_train=current_y_train,
    dataset_test=dataset_test,
    new_x=empty_x,
    new_y_scaled=empty_y,
    best_params=best_params,
    seed=42,
    debug=True
)

mse_method2 = new_total_mse
print(f"   Method 2 MSE: {mse_method2:.6f}")

# Compare results
print(f"\n🔍 COMPARISON:")
print(f"   Method 1 (calculate_mse): {mse_method1:.6f}")
print(f"   Method 2 (retrain_models): {mse_method2:.6f}")
print(f"   Difference: {abs(mse_method1 - mse_method2):.6f}")
print(f"   Match: {'✅ YES' if abs(mse_method1 - mse_method2) < 1e-9 else '❌ NO'}")

if abs(mse_method1 - mse_method2) > 1e-9:
    print(f"\n⚠️  Functions don't match! This explains the MSE inconsistencies.")
    print(f"   Individual MSEs method 1: {mse_list}")
    print(f"   Individual MSEs method 2: {new_mse_per_output}")
else:
    print(f"\n✅ Functions are consistent! MSE calculation is correct.")
