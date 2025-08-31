"""
Baseline Test - Using baseline_training_methods module
This keeps our methods safe and reusable.
"""

import numpy as np
import matplotlib.pyplot as plt
from baseline_training_methods import (
    run_mse_analysis,
    load_baseline_datasets
)


if __name__ == "__main__":
    print("🚀 Baseline Test - Using our specific config")
    
    # Configuration (from your active learning script)
    config = {
        'nspecies': 3,
        'num_pressure_conditions': 2,
        'pressure_conditions_pa': [133.322, 1333.22],  # 1 and 10 Torr
        'initial_samples_from_uniform': 500,
        'n_iterations': 1,
        'n_samples_per_iteration': 2,
        'svr_params': [
            {'C': 10, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'},
            {'C': 20, 'epsilon': 0.005, 'gamma': 5, 'kernel': 'rbf'},
            {'C': 5, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'}
        ]
    }
    
    print(f"Configuration: {config}")
    
    # Calculate total samples needed for fair comparison
    initial_samples = config['initial_samples_from_uniform']
    additional_samples = config['n_iterations'] * config['n_samples_per_iteration']
    total_samples_needed = initial_samples + additional_samples
    
    print(f"📊 Sample calculation:")
    print(f"   Initial samples: {initial_samples}")
    print(f"   Additional adaptive samples: {additional_samples}")
    print(f"   Total samples needed: {total_samples_needed}")
    
    # Use our config values
    nspecies = config['nspecies']
    num_pressure_conditions = config['num_pressure_conditions']
    best_params = config['svr_params']
    
    # Create subset sizes: 100, 200, 300, 400, 500, and final total_samples_needed
    base_sizes = list(range(100, initial_samples + 1, 100))  # [100, 200, 300, 400, 500]
    if total_samples_needed not in base_sizes:
        subset_sizes = base_sizes + [total_samples_needed]  # [100, 200, 300, 400, 500, 502]
    else:
        subset_sizes = base_sizes


    subset_sizes.append(500)


    num_seeds = 1  # Start with 1 seed for debugging
    
    print(f"📊 Subset sizes: {subset_sizes}")
    print(f"🎲 Number of seeds: {num_seeds}")
    
    # Use only the uniform dataset (like our active learning)
    dataset = 'O2_simple_uniform.txt'
    
    src_file_train = 'data/SampleEfficiency/' + dataset
    src_file_test = 'data/SampleEfficiency/O2_simple_test.txt'
    
    print(f"📂 Loading training data: {src_file_train}")
    print(f"📂 Loading test data: {src_file_test}")
    
    dataset_train, dataset_test = load_baseline_datasets(src_file_train, src_file_test, nspecies, num_pressure_conditions)

    print(f"✅ Data loaded successfully")
    
    # Get data shapes for debugging
    x_train, y_train = dataset_train.get_data()
    x_test, y_test = dataset_test.get_data()
    print(f"📊 Training data: x_train.shape={x_train.shape}, y_train.shape={y_train.shape}")
    print(f"📊 Test data: x_test.shape={x_test.shape}, y_test.shape={y_test.shape}")
    
    # Run MSE analysis
    mean_total_mse, std_total_mse = run_mse_analysis(dataset_train, dataset_test, best_params, subset_sizes, num_seeds)
    
    print(f"\n🏆 FINAL RESULTS:")
    print(f"📊 Mean Total MSE: {mean_total_mse}")
    print(f"📊 Std Error: {std_total_mse}")
    
    # Compare with expected values from sample_efficiency_with_zones.py
    print(f"\n🔍 COMPARISON:")
    print(f"   Our baseline (size 500): {mean_total_mse[-1]:.6f}")
    print(f"   Expected (sample_efficiency_with_zones): ~0.000105")
    print(f"   Ratio: {mean_total_mse[-1] / 0.000105:.1f}x higher")
    
    # Simple plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(subset_sizes, mean_total_mse, yerr=std_total_mse, label='Uniform Dataset', marker='o')
    plt.xlabel('Dataset size')
    plt.ylabel('MSE on test set')
    plt.title('Baseline Test - Sample Efficiency')
    plt.legend()
    plt.grid(True)
    plt.savefig('baseline_test.pdf')
    plt.show()
    
    print(f"\n✅ Baseline test complete!")
    print(f"📊 Plot saved as: baseline_test.pdf")
