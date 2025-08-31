#!/usr/bin/env python3
"""
Debug script to identify MSE calculation issues in active learning pipeline.
"""

import numpy as np
import sys
import os

# Add project root to Python path
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from active_learning_methods import load_datasets, train_initial_models, apply_training_scalers, retrain_models_with_new_data
from sklearn.utils import shuffle

def debug_mse_calculation():
    print("=" * 80)
    print("DEBUG: MSE CALCULATION IN ACTIVE LEARNING")
    print("=" * 80)
    
    # Load the same config as active_learning_train.py
    config = {
        'nspecies': 3,
        'num_pressure_conditions': 2,
        'initial_samples_from_uniform': 500,
        'svr_params': [
            {'C': 10, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'},
            {'C': 20, 'epsilon': 0.005, 'gamma': 5, 'kernel': 'rbf'},
            {'C': 5, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'}
        ]
    }
    
    nspecies = config['nspecies']
    num_pressure_conditions = config['num_pressure_conditions']
    best_params = config['svr_params']
    
    # Load datasets
    src_file_train = 'data/SampleEfficiency/O2_simple_uniform.txt'
    src_file_test = 'data/SampleEfficiency/O2_simple_test.txt'
    
    print("1. Loading datasets...")
    dataset_train, dataset_test = load_datasets(src_file_train, src_file_test, nspecies, num_pressure_conditions)
    
    # Train initial models (exactly like in the pipeline)
    print("2. Training initial models on 500 samples...")
    models, mse_per_output, total_mse = train_initial_models(
        dataset_train, dataset_test, best_params,
        n_initial_samples=config['initial_samples_from_uniform'],
        seed=42
    )
    
    print(f"   Initial MSE: {total_mse:.6f}")
    print(f"   MSE per output: {mse_per_output}")
    
    # Get the actual training data used for initial models
    x_all, y_all = dataset_train.get_data()
    x_shuf, y_shuf = shuffle(x_all, y_all, random_state=42)
    current_x_train = x_shuf[:config['initial_samples_from_uniform']]
    current_y_train = y_shuf[:config['initial_samples_from_uniform']]
    
    print(f"   Current training data shapes: X={current_x_train.shape}, Y={current_y_train.shape}")
    
    # Now let's simulate adding some new training samples
    print("\n3. Simulating addition of new training samples...")
    
    # Take the next 10 samples from the uniform dataset as "new" samples
    # (this simulates what happens when we add new adaptive samples)
    new_samples_start = config['initial_samples_from_uniform']
    new_samples_end = new_samples_start + 10
    
    new_x_from_uniform = x_shuf[new_samples_start:new_samples_end]
    new_y_from_uniform = y_shuf[new_samples_start:new_samples_end]
    
    print(f"   Adding {new_x_from_uniform.shape[0]} new samples (from uniform dataset)")
    print(f"   New sample shapes: X={new_x_from_uniform.shape}, Y={new_y_from_uniform.shape}")
    
    # Method 1: Manual concatenation + retraining (like in the pipeline)
    print("\n4. Method 1: Manual concatenation (current pipeline approach)")
    
    # Manually concatenate like in the pipeline
    manual_x_train = np.vstack([current_x_train, new_x_from_uniform])
    manual_y_train = np.vstack([current_y_train, new_y_from_uniform])
    
    print(f"   Manual concatenation result: X={manual_x_train.shape}, Y={manual_y_train.shape}")
    
    # Use retrain_models_with_new_data
    new_models_method1, new_mse_per_output_method1, new_total_mse_method1, augmented_size_method1, x_retrain_method1, y_retrain_method1 = retrain_models_with_new_data(
        current_x_train=current_x_train,
        current_y_train=current_y_train,
        dataset_test=dataset_test,
        new_x=new_x_from_uniform,
        new_y_scaled=new_y_from_uniform,
        best_params=best_params,
        seed=42,
        debug=True
    )
    
    print(f"   Method 1 Result: MSE={new_total_mse_method1:.6f}")
    print(f"   Method 1 Training size: {augmented_size_method1}")
    
    # Method 2: Direct training on larger subset (ground truth)
    print("\n5. Method 2: Direct training on 510 samples (ground truth)")
    
    ground_truth_models, ground_truth_mse_per_output, ground_truth_total_mse = train_initial_models(
        dataset_train, dataset_test, best_params,
        n_initial_samples=510,  # 500 + 10
        seed=42
    )
    
    print(f"   Ground Truth Result: MSE={ground_truth_total_mse:.6f}")
    
    # Compare results
    print("\n6. COMPARISON OF METHODS")
    print("-" * 50)
    
    print(f"Initial MSE (500 samples):     {total_mse:.6f}")
    print(f"Method 1 MSE (500+10 retrain): {new_total_mse_method1:.6f}")
    print(f"Ground Truth MSE (510 direct): {ground_truth_total_mse:.6f}")
    
    method1_vs_initial = new_total_mse_method1 - total_mse
    ground_truth_vs_initial = ground_truth_total_mse - total_mse
    method1_vs_ground_truth = new_total_mse_method1 - ground_truth_total_mse
    
    print(f"\nChanges from initial:")
    print(f"  Method 1:     {method1_vs_initial:+.6f}")
    print(f"  Ground Truth: {ground_truth_vs_initial:+.6f}")
    
    print(f"\nMethod 1 vs Ground Truth: {method1_vs_ground_truth:+.6f}")
    
    # Check if there's a significant discrepancy
    if abs(method1_vs_ground_truth) > 1e-6:
        print(f"\n🚨 POTENTIAL ISSUE DETECTED!")
        print(f"   Method 1 (retrain pipeline) differs from ground truth by {method1_vs_ground_truth:+.6f}")
        print(f"   This suggests an issue in the retraining logic.")
        
        # Additional debugging: check if the training data is the same
        print(f"\n7. DETAILED DEBUGGING")
        print(f"   Manual concatenation shape: {manual_x_train.shape}")
        print(f"   Retrain function internal shape: {x_retrain_method1.shape}")
        print(f"   Are they equal? {np.array_equal(manual_x_train, x_retrain_method1[:manual_x_train.shape[0]])}")
        
        # Check if shuffling is the issue
        x_ground_truth, y_ground_truth = dataset_train.get_data()
        x_ground_truth_shuf, y_ground_truth_shuf = shuffle(x_ground_truth, y_ground_truth, random_state=42)
        x_ground_truth_510 = x_ground_truth_shuf[:510]
        y_ground_truth_510 = y_ground_truth_shuf[:510]
        
        print(f"   Ground truth 510 samples shape: {x_ground_truth_510.shape}")
        print(f"   First few samples match? {np.allclose(x_retrain_method1[:5], x_ground_truth_510[:5])}")
        
    else:
        print(f"\n✅ Methods match! No MSE calculation issue detected.")
        print(f"   The MSE increase in active learning must be due to other factors.")
    
    # Test with different sample counts to see learning curve behavior
    print(f"\n8. TESTING LEARNING CURVE BEHAVIOR")
    print("-" * 50)
    
    sample_sizes = [500, 510, 520, 530, 540, 550]
    learning_curve_mses = []
    
    for size in sample_sizes:
        models_lc, mse_per_output_lc, total_mse_lc = train_initial_models(
            dataset_train, dataset_test, best_params,
            n_initial_samples=size,
            seed=42
        )
        learning_curve_mses.append(total_mse_lc)
        print(f"   Size {size}: MSE = {total_mse_lc:.6f}")
    
    print(f"\nLearning curve trend:")
    for i in range(1, len(learning_curve_mses)):
        change = learning_curve_mses[i] - learning_curve_mses[i-1]
        trend = "↓ DECREASING" if change < 0 else "↑ INCREASING" if change > 0 else "→ FLAT"
        print(f"   {sample_sizes[i-1]} → {sample_sizes[i]}: {change:+.6f} {trend}")
    
    if all(learning_curve_mses[i] <= learning_curve_mses[i-1] for i in range(1, len(learning_curve_mses))):
        print(f"\n✅ Learning curve is monotonically decreasing as expected!")
    else:
        print(f"\n⚠️ Learning curve is not monotonically decreasing - this might indicate noise or overfitting.")

if __name__ == "__main__":
    debug_mse_calculation()
