#!/usr/bin/env python3
"""
Debug Sample Selection Test

This script isolates and tests the select_samples_in_k_region method to verify
if it's truly doing random sampling when K-bounds are wide (entire [0,1] range).

We'll test multiple runs with the same parameters to see if results are consistent.
"""

import numpy as np
import json
import os
import sys
from sklearn.utils import shuffle

# Add current directory to path for imports
sys.path.append('.')
from active_learning_methods import (
    load_datasets, 
    apply_training_scalers
)
from scripts.k_centered_adaptive_learning import KCenteredAdaptiveLearner

def test_sample_selection_consistency():
    """Test if sample selection gives consistent results for identical parameters."""
    
    print("🧪 TESTING SAMPLE SELECTION CONSISTENCY")
    print("="*60)
    
    # Configuration
    config = {
        'nspecies': 3,
        'num_pressure_conditions': 2,
        'pressure_conditions_pa': [133.322, 1333.22],
        'svr_params': [
            {'C': 10, 'epsilon': 0.005, 'gamma': 0.1, 'kernel': 'rbf'}  # Lower gamma to prevent overfitting
        ],
        'seeds': [42],
        'random_seed': 42
    }
    
    # File paths - use single file to eliminate multi-file bias
    uniform_file = 'data/SampleEfficiency/O2_simple_uniform.txt'
    test_file = 'data/SampleEfficiency/O2_simple_test_real_K.txt'
    batch_json_paths = [
        'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-08-27/batch_4000sims_20250827_010028.json'
    ]
    
    print(f"\n📂 Loading data...")
    print(f"   Using single batch file: {batch_json_paths[0]}")
    
    # Initialize learner
    learner = KCenteredAdaptiveLearner(config, debug=True)
    learner.load_data(uniform_file, test_file, batch_json_paths)
    
    # Print batch file info
    print(f"\n📊 Batch file status:")
    learner.print_batch_file_status()
    
    # Define very wide K-bounds (essentially no constraint - random sampling)
    predicted_k = np.array([[0.5, 0.5, 0.5]])  # Center prediction
    k_bounds = [
        (0.0, 1.0),  # K1: entire range
        (0.0, 1.0),  # K2: entire range  
        (0.0, 1.0)   # K3: entire range
    ]
    
    print(f"\n🎯 Testing with WIDE K-bounds (random sampling):")
    print(f"   K-bounds: {k_bounds}")
    print(f"   This should be equivalent to random sampling!")
    
    # Test multiple sample sizes
    sample_sizes = [50, 100, 200]
    n_trials = 5  # Multiple trials per sample size
    
    results = {}
    
    for n_samples in sample_sizes:
        print(f"\n🔬 Testing sample_size = {n_samples}")
        print(f"{'Trial':>8} {'Selected':>10} {'K1_mean':>10} {'K2_mean':>10} {'K3_mean':>10} {'K1_std':>10} {'K2_std':>10} {'K3_std':>10}")
        print(f"{'-'*80}")
        
        trial_results = []
        
        for trial in range(n_trials):
            # Reset used indices for fresh sampling
            for batch_info in learner.batch_files_data:
                batch_info['used_indices'] = set()
            
            # Select samples
            selected_x, selected_y, selected_indices = learner.select_samples_in_k_region(
                k_bounds, n_samples
            )
            
            if selected_x is not None:
                # Calculate statistics
                k_means = [selected_y[:, i].mean() for i in range(3)]
                k_stds = [selected_y[:, i].std() for i in range(3)]
                
                trial_results.append({
                    'trial': trial + 1,
                    'n_selected': len(selected_y),
                    'k_means': k_means,
                    'k_stds': k_stds,
                    'selected_indices': selected_indices
                })
                
                print(f"{trial+1:>8} {len(selected_y):>10} {k_means[0]:>10.4f} {k_means[1]:>10.4f} {k_means[2]:>10.4f} {k_stds[0]:>10.4f} {k_stds[1]:>10.4f} {k_stds[2]:>10.4f}")
            else:
                print(f"{trial+1:>8} {'FAILED':>10}")
                
        results[n_samples] = trial_results
    
    # Analyze consistency across trials
    print(f"\n📈 CONSISTENCY ANALYSIS")
    print(f"="*60)
    
    for n_samples, trials in results.items():
        if not trials:
            continue
            
        print(f"\n🎯 Sample size: {n_samples}")
        
        # Calculate statistics across trials
        all_k1_means = [t['k_means'][0] for t in trials]
        all_k2_means = [t['k_means'][1] for t in trials]
        all_k3_means = [t['k_means'][2] for t in trials]
        
        k1_mean_std = np.std(all_k1_means)
        k2_mean_std = np.std(all_k2_means)
        k3_mean_std = np.std(all_k3_means)
        
        print(f"   K1 mean across trials: {np.mean(all_k1_means):.4f} ± {k1_mean_std:.4f}")
        print(f"   K2 mean across trials: {np.mean(all_k2_means):.4f} ± {k2_mean_std:.4f}")
        print(f"   K3 mean across trials: {np.mean(all_k3_means):.4f} ± {k3_mean_std:.4f}")
        
        # For true random sampling from [0,1], expected mean = 0.5, std = 0.289
        expected_mean = 0.5
        expected_sample_std = np.sqrt(1/12)  # ≈ 0.289 for uniform(0,1)
        
        print(f"   Expected mean for random: {expected_mean:.4f}")
        print(f"   Expected std for random: {expected_sample_std:.4f}")
        
        # Check if results are consistent with random sampling
        mean_bias_k1 = abs(np.mean(all_k1_means) - expected_mean)
        mean_bias_k2 = abs(np.mean(all_k2_means) - expected_mean)
        mean_bias_k3 = abs(np.mean(all_k3_means) - expected_mean)
        
        print(f"   Mean bias from 0.5: K1={mean_bias_k1:.4f}, K2={mean_bias_k2:.4f}, K3={mean_bias_k3:.4f}")
        
        # Check trial-to-trial consistency (should be low for random sampling)
        print(f"   Trial-to-trial variation: K1={k1_mean_std:.4f}, K2={k2_mean_std:.4f}, K3={k3_mean_std:.4f}")
        
        if k1_mean_std > 0.05 or k2_mean_std > 0.05 or k3_mean_std > 0.05:
            print(f"   ⚠️ HIGH TRIAL-TO-TRIAL VARIATION - sampling may not be truly random!")
        else:
            print(f"   ✅ Low trial-to-trial variation - consistent with random sampling")
    
    return results

def test_multi_file_vs_single_file():
    """Test if multi-file sampling introduces bias compared to single file."""
    
    print(f"\n\n🧪 TESTING MULTI-FILE vs SINGLE-FILE SAMPLING")
    print(f"="*60)
    
    # Configuration
    config = {
        'nspecies': 3,
        'num_pressure_conditions': 2,
        'pressure_conditions_pa': [133.322, 1333.22],
        'svr_params': [
            {'C': 10, 'epsilon': 0.005, 'gamma': 0.1, 'kernel': 'rbf'}
        ],
        'seeds': [42],
        'random_seed': 42
    }
    
    uniform_file = 'data/SampleEfficiency/O2_simple_uniform.txt'
    test_file = 'data/SampleEfficiency/O2_simple_test_real_K.txt'
    
    # Test 1: Single file
    print(f"\n🔬 Test 1: Single file sampling")
    single_file_paths = [
        'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-08-27/batch_4000sims_20250827_010028.json'
    ]
    
    learner_single = KCenteredAdaptiveLearner(config, debug=False)
    learner_single.load_data(uniform_file, test_file, single_file_paths)
    
    # Test 2: Multiple files
    print(f"\n🔬 Test 2: Multiple file sampling")
    multi_file_paths = [
        'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-08-27/batch_4000sims_20250827_010028.json',
        'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2000sims_20250929_125706.json',
        'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-28/batch_1000sims_20250928_191628.json'
    ]
    
    learner_multi = KCenteredAdaptiveLearner(config, debug=False)
    learner_multi.load_data(uniform_file, test_file, multi_file_paths)
    
    # Wide K-bounds for both
    k_bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    n_samples = 100
    
    # Sample from both
    selected_x_single, selected_y_single, _ = learner_single.select_samples_in_k_region(k_bounds, n_samples)
    selected_x_multi, selected_y_multi, _ = learner_multi.select_samples_in_k_region(k_bounds, n_samples)
    
    # Compare statistics
    if selected_y_single is not None and selected_y_multi is not None:
        print(f"\n📊 Comparison Results:")
        print(f"{'Method':>15} {'K1_mean':>10} {'K2_mean':>10} {'K3_mean':>10} {'K1_std':>10} {'K2_std':>10} {'K3_std':>10}")
        print(f"{'-'*75}")
        
        # Single file stats
        k1_mean_s, k2_mean_s, k3_mean_s = [selected_y_single[:, i].mean() for i in range(3)]
        k1_std_s, k2_std_s, k3_std_s = [selected_y_single[:, i].std() for i in range(3)]
        print(f"{'Single file':>15} {k1_mean_s:>10.4f} {k2_mean_s:>10.4f} {k3_mean_s:>10.4f} {k1_std_s:>10.4f} {k2_std_s:>10.4f} {k3_std_s:>10.4f}")
        
        # Multi file stats
        k1_mean_m, k2_mean_m, k3_mean_m = [selected_y_multi[:, i].mean() for i in range(3)]
        k1_std_m, k2_std_m, k3_std_m = [selected_y_multi[:, i].std() for i in range(3)]
        print(f"{'Multi file':>15} {k1_mean_m:>10.4f} {k2_mean_m:>10.4f} {k3_mean_m:>10.4f} {k1_std_m:>10.4f} {k2_std_m:>10.4f} {k3_std_m:>10.4f}")
        
        # Check for significant differences
        mean_diff_k1 = abs(k1_mean_s - k1_mean_m)
        mean_diff_k2 = abs(k2_mean_s - k2_mean_m)
        mean_diff_k3 = abs(k3_mean_s - k3_mean_m)
        
        print(f"\n📈 Differences:")
        print(f"   Mean differences: K1={mean_diff_k1:.4f}, K2={mean_diff_k2:.4f}, K3={mean_diff_k3:.4f}")
        
        if mean_diff_k1 > 0.05 or mean_diff_k2 > 0.05 or mean_diff_k3 > 0.05:
            print(f"   ⚠️ SIGNIFICANT DIFFERENCES - multi-file introduces bias!")
        else:
            print(f"   ✅ Small differences - multi-file sampling appears consistent")
    
    return selected_y_single, selected_y_multi

if __name__ == '__main__':
    print("🔬 SAMPLE SELECTION DEBUG TEST SUITE")
    print("="*80)
    
    # Test 1: Consistency across trials with same parameters
    results = test_sample_selection_consistency()
    
    # Test 2: Single vs multi-file comparison  
    single_results, multi_results = test_multi_file_vs_single_file()
    
    print(f"\n🎯 SUMMARY")
    print(f"="*40)
    print(f"This test isolates the sample selection method to check:")
    print(f"1. Are results consistent across trials with identical parameters?")
    print(f"2. Does multi-file sampling introduce bias vs single-file?")
    print(f"3. Are the wide K-bounds truly equivalent to random sampling?")
    
    print(f"\nIf sample selection is working correctly:")
    print(f"✅ Trial-to-trial variation should be small (<0.05)")
    print(f"✅ K-means should be close to 0.5 (expected for uniform random)")
    print(f"✅ Single-file vs multi-file should give similar results")
    
    print(f"\n✅ SAMPLE SELECTION DEBUG TEST COMPLETE!")