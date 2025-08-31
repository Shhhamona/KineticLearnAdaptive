#!/usr/bin/env python3
"""
Iterative Batch Training Script - Active Learning Style Loop

This script mimics the active learning training loop by:
1. Starting with an initial uniform dataset (100 samples)
2. Iteratively adding small batches from the 500-batch simulation data (10 samples per iteration)
3. Retraining models after each batch addition using retrain_models_with_new_data()
4. Tracking performance evolution to see if MSE increases with more samples

This helps debug why active learning performance degrades with more samples.
"""

import json
import numpy as np
import os
import sys
sys.path.append('.')  # Add current directory to path
from active_learning_methods import (
    load_datasets, 
    train_initial_models, 
    apply_training_scalers,
    retrain_models_with_new_data,
    evaluate_zones_scaled
)
import matplotlib.pyplot as plt


def sample_from_batch_by_zone(batch_new_x_all, batch_new_y_scaled_all, batch_k_values, 
                             zone_center_scaled_y, zone_thresholds, target_zone_idx, 
                             n_samples, used_indices=None):
    """
    Sample n_samples from batch data that fall within a specific zone.
    
    Args:
        batch_new_x_all: Pre-scaled batch X data (N, features)
        batch_new_y_scaled_all: Pre-scaled batch Y data (N, outputs) 
        batch_k_values: Raw batch K values for reference (N, outputs)
        zone_center_scaled_y: Center point in scaled Y space (outputs,)
        zone_thresholds: Distance thresholds for each zone (n_zones+1,)
        target_zone_idx: Which zone to sample from (0-based)
        n_samples: How many samples to extract
        used_indices: Set of already used indices to avoid (optional)
        
    Returns:
        sampled_x: Selected X samples (n_samples, features)
        sampled_y_scaled: Selected Y samples (n_samples, outputs) 
        sampled_k_values: Selected raw K values (n_samples, outputs)
        selected_indices: Indices that were selected
    """
    if used_indices is None:
        used_indices = set()
    
    # Calculate distances from batch Y data to zone center
    distances = np.linalg.norm(batch_new_y_scaled_all - zone_center_scaled_y, axis=1)
    
    # Find which zone each sample belongs to using thresholds
    zone_assignments = np.digitize(distances, zone_thresholds) - 1
    zone_assignments = np.clip(zone_assignments, 0, len(zone_thresholds) - 2)
    
    # Find samples in target zone that haven't been used yet
    available_mask = (zone_assignments == target_zone_idx)
    available_indices = np.where(available_mask)[0]
    unused_indices = [idx for idx in available_indices if idx not in used_indices]
    
    print(f"   📊 Zone {target_zone_idx+1} sampling:")
    print(f"      Total samples in zone: {len(available_indices)}")
    print(f"      Unused samples in zone: {len(unused_indices)}")
    print(f"      Requested samples: {n_samples}")
    
    if len(unused_indices) < n_samples:
        print(f"      ⚠️ Not enough unused samples in zone {target_zone_idx+1}")
        print(f"         Available: {len(unused_indices)}, Requested: {n_samples}")
        n_samples = len(unused_indices)
        if n_samples == 0:
            print(f"         ❌ No samples available in zone {target_zone_idx+1}")
            return None, None, None, []
    
    # Randomly select from unused samples in the target zone
    seed_val = config.get('random_seed', None)
    if seed_val is None:
        seed_val = config.get('seeds', [42])[0] if isinstance(config, dict) else 42
    rng = np.random.default_rng(seed_val)

    selected_indices = rng.choice(unused_indices, size=n_samples, replace=False)

    sampled_x = batch_new_x_all[selected_indices]
    sampled_y_scaled = batch_new_y_scaled_all[selected_indices] 
    sampled_k_values = batch_k_values[selected_indices]
    
    return sampled_x, sampled_y_scaled, sampled_k_values, selected_indices.tolist()


def run_adaptive_zone_experiment(uniform_dataset, test_dataset,
                                batch_new_x_all, batch_new_y_scaled_all, batch_k_values,
                                config, n_iterations=10, samples_per_iteration=10, 
                                initial_uniform_size=100, n_zones=6):
    """
    Run adaptive zone-based sampling experiment.
    
    This approach:
    1. Train initial model on uniform data
    2. Each iteration: evaluate zones, rank by error, sample from high-error zones
    3. Retrain and repeat
    
    Args:
        uniform_dataset: Uniform training dataset
        test_dataset: Test dataset
        batch_new_x_all: Pre-scaled batch X data
        batch_new_y_scaled_all: Pre-scaled batch Y data  
        batch_k_values: Raw batch K values
        config: Configuration dictionary
        n_iterations: Number of iterations to run
        samples_per_iteration: Number of batch samples to add per iteration
        initial_uniform_size: Initial uniform samples to start with
        n_zones: Number of zones for evaluation
    """
    print(f"\n🎯 ADAPTIVE ZONE-BASED EXPERIMENT")
    print(f"   Initial uniform size: {initial_uniform_size}")
    print(f"   Iterations: {n_iterations}")
    print(f"   Samples per iteration: {samples_per_iteration}")
    print(f"   Number of zones: {n_zones}")
    
    # Start with initial uniform training data
    x_uniform, y_uniform = uniform_dataset.get_data()
    current_x_train = x_uniform[:initial_uniform_size].copy()
    current_y_train = y_uniform[:initial_uniform_size].copy()
    
    print(f"   Starting training size: {current_x_train.shape[0]}")
    
    # Track results and used batch indices
    iteration_results = []
    used_batch_indices = set()
    
    # Train initial models
    print(f"\n🎯 ITERATION 0 - Initial Training (Uniform only)")
    print(f"   Training on {current_x_train.shape[0]} uniform samples...")
    
    class TempDataset:
        def __init__(self, x_data, y_data):
            self.x_data = x_data
            self.y_data = y_data
        def get_data(self):
            return self.x_data, self.y_data
    
    temp_dataset = TempDataset(current_x_train, current_y_train)
    initial_models_per_seed, initial_mse_per_output, initial_total_mse = train_initial_models(
        temp_dataset, test_dataset, config['svr_params'], 
        n_initial_samples=current_x_train.shape[0], seeds=config.get('seeds')
    )
   
    iteration_results.append({
        'iteration': 0,
        'total_samples': current_x_train.shape[0],
        'batch_samples_added': 0,
        'mse_per_output': initial_mse_per_output,
        'total_mse': initial_total_mse,
        'mse_type': 'initial',
        'sampling_strategy': 'uniform_only'
    })
    
    # Store current models for zone evaluation (models_per_seed list-of-lists)
    current_models = initial_models_per_seed
    
    # Now run adaptive iterations
    for iteration in range(1, n_iterations + 1):
        print(f"\n🎯 ITERATION {iteration}")
        
        # Step 1: Evaluate zones with current model to find high-error zones
        print(f"   🔍 Evaluating zones to identify high-error regions...")
        
        # For zone evaluation, we need some recent batch data to define zones
        # Use a sample of unused batch data for zone definition
        eval_sample_size = min(500, len(batch_new_x_all) - len(used_batch_indices))
        if eval_sample_size <= 0:
            print(f"   ❌ No more batch samples available!")
            break
            
        # Get unused indices for zone evaluation
        all_indices = set(range(len(batch_new_x_all)))
        unused_indices = list(all_indices - used_batch_indices)
        rng = np.random.default_rng(42)

        #eval_indices = np.random.choice(unused_indices, size=min(eval_sample_size, len(unused_indices)), replace=False)
        eval_indices = rng.choice(unused_indices, size=min(eval_sample_size, len(unused_indices)), replace=False)

        eval_x = batch_new_x_all[eval_indices]
        eval_y_scaled = batch_new_y_scaled_all[eval_indices]
        
        try:
            zone_indices, center_y_scaled, thresholds, per_output_mse_list, overall_mse_list, counts = \
                evaluate_zones_scaled(current_x_train, current_y_train, eval_x, eval_y_scaled, current_models, n_zones=n_zones)
            
            # Rank zones by overall MSE (higher MSE = higher priority)
            zone_errors = [(i, mse) for i, mse in enumerate(overall_mse_list) if mse is not None and not np.isnan(mse)]
            zone_errors.sort(key=lambda x: x[1], reverse=True)  # Sort by MSE descending
            
            print(f"   📊 Zone error ranking:")
            for rank, (zone_idx, mse) in enumerate(zone_errors):
                print(f"      Rank {rank+1}: Zone {zone_idx+1} - MSE = {mse:.6f}")
            
            if not zone_errors:
                print(f"   ⚠️ No valid zone errors, falling back to random sampling")
                target_zones = list(range(n_zones))
            else:
                # Focus on top 3 highest error zones
                target_zones = [zone_idx for zone_idx, _ in zone_errors[:3]]
                
        except Exception as e:
            print(f"   ⚠️ Zone evaluation failed: {e}")
            print(f"      Falling back to random sampling")
            target_zones = list(range(n_zones))
            center_y_scaled = np.mean(batch_new_y_scaled_all, axis=0)
            # Create simple equal-distance thresholds
            max_dist = np.max(np.linalg.norm(batch_new_y_scaled_all - center_y_scaled, axis=1))
            thresholds = np.linspace(0, max_dist, n_zones + 1)
        
        # Step 2: Sample from high-error zones
        print(f"   🎯 Sampling from high-error zones: {[z+1 for z in target_zones]}")
        
        new_x_list = []
        new_y_scaled_list = []
        new_k_values_list = []
        selected_indices_all = []
        
        samples_per_zone = samples_per_iteration // len(target_zones)
        remaining_samples = samples_per_iteration % len(target_zones)
        
        for i, zone_idx in enumerate(target_zones):
            zone_samples = samples_per_zone + (1 if i < remaining_samples else 0)
            if zone_samples == 0:
                continue
                
            sampled_x, sampled_y_scaled, sampled_k_values, selected_indices = sample_from_batch_by_zone(
                batch_new_x_all, batch_new_y_scaled_all, batch_k_values,
                center_y_scaled, thresholds, zone_idx, zone_samples, used_batch_indices
            )
            
            if sampled_x is not None:
                new_x_list.append(sampled_x)
                new_y_scaled_list.append(sampled_y_scaled)
                new_k_values_list.append(sampled_k_values)
                selected_indices_all.extend(selected_indices)
                used_batch_indices.update(selected_indices)
        
        if not new_x_list:
            print(f"   ❌ No samples could be obtained from target zones!")
            break
            
        # Combine all sampled data
        new_x = np.vstack(new_x_list)
        new_y_scaled = np.vstack(new_y_scaled_list)
        new_k_values = np.vstack(new_k_values_list)
        
        print(f"   ✅ Sampled {len(new_x)} samples from {len(target_zones)} zones")
        print(f"   📊 Sample distribution by target zone: {[len(arr) for arr in new_x_list]}")

        # Step 3: Retrain models with new data
        print(f"   🔄 Retraining models with zone-targeted samples...")
        retrain_models_per_seed, retrain_mse_per_output, retrain_total_mse, augmented_size, x_train_shuffled, y_train_shuffled = retrain_models_with_new_data(
            current_x_train=current_x_train,
            current_y_train=current_y_train,
            dataset_test=test_dataset,
            new_x=new_x,
            new_y_scaled=new_y_scaled,
            best_params=config['svr_params'],
            seeds=config.get('seeds'),
            debug=True
        )
        
        # Step 4: Evaluate final zones for this iteration
        try:
            final_zone_indices, final_center_y_scaled, final_thresholds, final_per_output_mse_list, final_overall_mse_list, final_counts = \
                evaluate_zones_scaled(current_x_train, current_y_train, new_x, new_y_scaled, retrain_models_per_seed, n_zones=n_zones)

            # Convert arrays to JSON-serializable lists
            thresholds_list = [float(x) for x in final_thresholds.tolist()]
            center_list = [float(x) for x in final_center_y_scaled.tolist()]
            per_output_mse_serial = [m.tolist() if m is not None else None for m in final_per_output_mse_list]
            overall_mse_serial = [float(x) if x is not None else None for x in final_overall_mse_list]
            counts_serial = [int(x) for x in final_counts]
        except Exception as e:
            print('   ⚠️ Final zone evaluation failed for iteration', iteration, ':', e)
            thresholds_list = None
            center_list = None
            per_output_mse_serial = None
            overall_mse_serial = None
            counts_serial = None

        # Update current training data and models for next iteration
        current_x_train = np.vstack([current_x_train, new_x])
        current_y_train = np.vstack([current_y_train, new_y_scaled])
        current_models = retrain_models_per_seed

        print(f"   ✅ Iteration {iteration} complete:")
        print(f"      Total training samples: {augmented_size}")
        print(f"      Test MSE per output: {retrain_mse_per_output}")
        print(f"      Total Test MSE: {retrain_total_mse:.6f}")
        print(f"      Used batch indices: {len(used_batch_indices)}")
        
        # Store results
        iteration_results.append({
            'iteration': iteration,
            'total_samples': augmented_size,
            'batch_samples_added': len(new_x),
            'mse_per_output': retrain_mse_per_output,
            'total_mse': retrain_total_mse,
            'mse_type': 'adaptive_retrain',
            'sampling_strategy': 'adaptive_zone_based',
            'target_zones': [z+1 for z in target_zones],  # 1-based for readability
            'zone_thresholds': thresholds_list,
            'zone_center_scaled_y': center_list,
            'zone_per_output_mse_scaled': per_output_mse_serial,
            'zone_overall_mse_scaled': overall_mse_serial,
            'zone_counts': counts_serial,
            'used_batch_indices_count': len(used_batch_indices)
        })
        
        # Check for performance changes
        prev_mse = iteration_results[-2]['total_mse']
        change_pct = ((retrain_total_mse / prev_mse) - 1) * 100
        if change_pct > 10:
            print(f"   ⚠️ Performance degradation: {change_pct:+.1f}%")
        elif change_pct < -5:
            print(f"   ✅ Performance improvement: {change_pct:+.1f}%")
    
    return iteration_results


def load_batch_data(json_path):
    """Load batch simulation data from JSON file."""
    print(f"📂 Loading batch data from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract metadata
    n_simulations = data['n_simulations']
    n_successful = data['n_successful']
    pressure_conditions = data['metadata']['pressure_conditions_pa']
    n_pressure_conditions = data['metadata']['n_pressure_conditions']
    
    print(f"   📊 Batch Info:")
    print(f"      Total simulations: {n_simulations}")
    print(f"      Successful: {n_successful}")
    print(f"      Pressure conditions: {pressure_conditions} Pa")
    
    # Extract parameter sets (k values) and compositions
    k_values = np.array([ps['k_values'] for ps in data['parameter_sets']])
    compositions = np.array(data['compositions'])
    
    print(f"   📊 Data Shapes:")
    print(f"      K values: {k_values.shape}")
    print(f"      Compositions: {compositions.shape}")
    
    return k_values, compositions, n_simulations, n_pressure_conditions


def run_iterative_batch_experiment(uniform_dataset, test_dataset,
                                  batch_new_x_all, batch_new_y_scaled_all,
                                  config, n_iterations=10, samples_per_iteration=10, initial_uniform_size=100):
    """
    Run iterative batch training experiment mimicking active learning loop.
    
    Args:
        uniform_dataset: Uniform training dataset
        test_dataset: Test dataset
        batch_k_values: Batch simulation k values (500, 3)
        batch_compositions: Batch simulation compositions (1000, 3)
        config: Configuration dictionary
        n_iterations: Number of iterations to run
        samples_per_iteration: Number of batch samples to add per iteration
        initial_uniform_size: Initial uniform samples to start with
    """
    print(f"\n🔄 ITERATIVE BATCH EXPERIMENT")
    print(f"   Initial uniform size: {initial_uniform_size}")
    print(f"   Iterations: {n_iterations}")
    print(f"   Samples per iteration: {samples_per_iteration}")
    print(f"   Total batch samples to add: {n_iterations * samples_per_iteration}")
    
    # Start with initial uniform training data
    x_uniform, y_uniform = uniform_dataset.get_data()
    current_x_train = x_uniform[:initial_uniform_size].copy()
    current_y_train = y_uniform[:initial_uniform_size].copy()
    
    print(f"   Starting training size: {current_x_train.shape[0]}")
    print(f"   🔍 UNIFORM DATA DEBUG:")
    print(f"      X_uniform range: [{current_x_train.min():.6f}, {current_x_train.max():.6f}]")
    print(f"      Y_uniform range: [{current_y_train.min():.6f}, {current_y_train.max():.6f}]")
    print(f"      X_uniform first 3 rows:\n{current_x_train[:3]}")
    print(f"      Y_uniform first 3 rows:\n{current_y_train[:3]}")
    
    # Track results
    iteration_results = []
    
    # Train initial models
    print(f"\n🎯 ITERATION 0 - Initial Training (Uniform only)")
    print(f"   Training on {current_x_train.shape[0]} uniform samples...")
    
    # Create temporary dataset for initial training
    class TempDataset:
        def __init__(self, x_data, y_data):
            self.x_data = x_data
            self.y_data = y_data
        def get_data(self):
            return self.x_data, self.y_data
    
    temp_dataset = TempDataset(current_x_train, current_y_train)
    initial_models_per_seed, initial_mse_per_output, initial_total_mse = train_initial_models(
        temp_dataset, test_dataset, config['svr_params'], 
        n_initial_samples=current_x_train.shape[0], seeds=config.get('seeds')
    )
   
    iteration_results.append({
        'iteration': 0,
        'total_samples': current_x_train.shape[0],
        'batch_samples_added': 0,
        'mse_per_output': initial_mse_per_output,
        'total_mse': initial_total_mse,
        'mse_type': 'initial'
    })
    # NOTE: batch_new_x_all and batch_new_y_scaled_all are expected to be provided
    # by the caller (pre-scaled once at script startup). This keeps scaling out
    # of the iterative function so behavior doesn't depend on internal scaling.
    
    # Now run iterative batch additions
    for iteration in range(1, n_iterations + 1):
        print(f"\n🎯 ITERATION {iteration}")
        
        # Calculate which batch samples to use for this iteration
        start_idx = (iteration - 1) * samples_per_iteration
        end_idx = start_idx + samples_per_iteration
        
        # Extract batch samples for this iteration (from pre-scaled arrays)
        new_x = batch_new_x_all[start_idx:end_idx]
        new_y_scaled = batch_new_y_scaled_all[start_idx:end_idx]

        print(f"   Adding batch samples {start_idx} to {end_idx-1}")
        print(f"   Scaled batch X shape: {new_x.shape}, Scaled batch Y shape: {new_y_scaled.shape}")
        print(f"   🔍 BATCH DATA DEBUG:")
        print(f"      X_batch range: [{new_x.min():.6f}, {new_x.max():.6f}]")
        print(f"      Y_batch range: [{new_y_scaled.min():.6f}, {new_y_scaled.max():.6f}]")
        print(f"      X_batch first 3 rows:\n{new_x[:3]}")
        print(f"      Y_batch first 3 rows:\n{new_y_scaled[:3]}")

        print(f"   🔍 COMPARISON:")
        print(f"      Current X range: [{current_x_train.min():.6f}, {current_x_train.max():.6f}]")
        print(f"      Current Y range: [{current_y_train.min():.6f}, {current_y_train.max():.6f}]")

        # Retrain models with new data (exactly like active learning)
        print(f"   Retraining models with current + new data...")
        retrain_models_per_seed, retrain_mse_per_output, retrain_total_mse, augmented_size, x_train_shuffled, y_train_shuffled = retrain_models_with_new_data(
            current_x_train=current_x_train,
            current_y_train=current_y_train,
            dataset_test=test_dataset,
            new_x=new_x,
            new_y_scaled=new_y_scaled,
            best_params=config['svr_params'],
            seeds=config.get('seeds'),
            debug=True  # Enable debug for snapshot saving
        )

        # Compute per-zone evaluation BEFORE we update current training data
        try:
            zone_indices, center_y_scaled, thresholds, per_output_mse_list, overall_mse_list, counts = \
                evaluate_zones_scaled(current_x_train, current_y_train, new_x, new_y_scaled, retrain_models_per_seed, n_zones=6)

            # Convert arrays to JSON-serializable lists
            thresholds_list = [float(x) for x in thresholds.tolist()]
            center_list = [float(x) for x in center_y_scaled.tolist()]
            per_output_mse_serial = [m.tolist() if m is not None else None for m in per_output_mse_list]
            overall_mse_serial = [float(x) if x is not None else None for x in overall_mse_list]
            counts_serial = [int(x) for x in counts]
        except Exception as e:
            print('   ⚠️ Zone evaluation failed for iteration', iteration, ':', e)
            thresholds_list = None
            center_list = None
            per_output_mse_serial = None
            overall_mse_serial = None
            counts_serial = None

        # Update current training data for next iteration
        current_x_train = np.vstack([current_x_train, new_x])
        current_y_train = np.vstack([current_y_train, new_y_scaled])

        print(f"   ✅ Iteration {iteration} complete:")
        print(f"      Total training samples: {augmented_size}")
        print(f"      Test MSE per output: {retrain_mse_per_output}")
        print(f"      Total Test MSE: {retrain_total_mse:.6f}")

        # Store results
        iteration_results.append({
            'iteration': iteration,
            'total_samples': augmented_size,
            'batch_samples_added': samples_per_iteration,
            'mse_per_output': retrain_mse_per_output,
            'total_mse': retrain_total_mse,
            'mse_type': 'retrain',
            'zone_thresholds': thresholds_list,
            'zone_center_scaled_y': center_list,
            'zone_per_output_mse_scaled': per_output_mse_serial,
            'zone_overall_mse_scaled': overall_mse_serial,
            'zone_counts': counts_serial
        })

        # Check for performance degradation
        prev_mse = iteration_results[-2]['total_mse']
        if retrain_total_mse > prev_mse * 1.1:  # 10% increase threshold
            print(f"   ⚠️ Performance degradation detected!")
            print(f"      Previous MSE: {prev_mse:.6f}")
            print(f"      Current MSE: {retrain_total_mse:.6f}")
            print(f"      Increase: {((retrain_total_mse / prev_mse) - 1) * 100:.1f}%")
    
    return iteration_results


def print_experiment_summary(iteration_results):
    """Print a summary of the iterative experiment results."""
    print(f"\n{'='*80}")
    print(f"ITERATIVE BATCH EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n📊 Performance Evolution:")
    print(f"{'Iter':>4} {'Samples':>8} {'Added':>6} {'Total MSE':>12} {'Change':>10} {'Status':>8}")
    print(f"{'-'*60}")
    
    for i, result in enumerate(iteration_results):
        iter_num = result['iteration']
        total_samples = result['total_samples']
        added = result['batch_samples_added']
        total_mse = result['total_mse']
        
        if i == 0:
            change_str = "baseline"
            status = "✅"
        else:
            prev_mse = iteration_results[i-1]['total_mse']
            change_pct = ((total_mse / prev_mse) - 1) * 100
            change_str = f"{change_pct:+.1f}%"
            status = "✅" if change_pct <= 10 else "⚠️" if change_pct <= 50 else "❌"
        
        print(f"{iter_num:>4} {total_samples:>8} {added:>6} {total_mse:>12.6f} {change_str:>10} {status:>8}")
    
    # Find best and worst performance
    best_result = min(iteration_results, key=lambda x: x['total_mse'])
    worst_result = max(iteration_results, key=lambda x: x['total_mse'])
    
    print(f"\n📈 Performance Analysis:")
    print(f"   Best performance:  Iteration {best_result['iteration']} - MSE = {best_result['total_mse']:.6f}")
    print(f"   Worst performance: Iteration {worst_result['iteration']} - MSE = {worst_result['total_mse']:.6f}")
    
    final_mse = iteration_results[-1]['total_mse']
    initial_mse = iteration_results[0]['total_mse']
    overall_change = ((final_mse / initial_mse) - 1) * 100
    
    print(f"   Overall change: {overall_change:+.1f}% (Initial: {initial_mse:.6f} → Final: {final_mse:.6f})")
    
    if overall_change > 10:
        print(f"   🚨 SIGNIFICANT DEGRADATION DETECTED!")
        print(f"      Adding batch samples appears to hurt performance")
    elif overall_change > 0:
        print(f"   ⚠️ Slight performance degradation")
    else:
        print(f"   ✅ Performance improved or maintained")


if __name__ == '__main__':
    # Configuration (same as active_learning_train.py)
    config = {
        'nspecies': 3,
        'num_pressure_conditions': 2,
        'pressure_conditions_pa': [133.322, 1333.22],  # 1 and 10 Torr
        'svr_params': [
            {'C': 10, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'},
            {'C': 20, 'epsilon': 0.005, 'gamma': 5, 'kernel': 'rbf'},
            {'C': 5, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'}
        ]
    }
    # Provide seeds list for multi-seed experiments (default single seed 42)
    config['seeds'] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    print(f"🎯 ITERATIVE BATCH TRAINING EXPERIMENT")
    print(f"Configuration: {config}")
    
    # File paths
    batch_json_path = 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-08-20/batch_500sims_20250820_141641.json'
    batch_json_path = 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-08-27/batch_4000sims_20250827_010028.json'
    src_file_train = 'data/SampleEfficiency/O2_simple_uniform.txt'
    #src_file_train = 'data/SampleEfficiency/O2_simple_latin.txt'
    src_file_test = 'data/SampleEfficiency/O2_simple_test.txt'
    
    nspecies = config['nspecies']
    num_pressure_conditions = config['num_pressure_conditions']
    
    print(f"\n📂 PHASE 1: Loading datasets")
    
    # Load uniform datasets
    print(f"Loading uniform training and test datasets...")
    uniform_dataset, test_dataset = load_datasets(
        src_file_train, src_file_test, nspecies, num_pressure_conditions
    )
    print(f"✅ Uniform datasets loaded")
    
    # DEBUG: Check what's in the uniform dataset
    x_uniform_check, y_uniform_check = uniform_dataset.get_data()
    print(f"🔍 UNIFORM DATASET DEBUG:")
    print(f"   X_uniform total range: [{x_uniform_check.min():.6f}, {x_uniform_check.max():.6f}]")
    print(f"   Y_uniform total range: [{y_uniform_check.min():.6f}, {y_uniform_check.max():.6f}]")
    print(f"   X_uniform first 3 rows:\n{x_uniform_check[:3]}")
    print(f"   Y_uniform first 3 rows:\n{y_uniform_check[:3]}")
    print(f"   Expected: X should be in [-1,1] range, Y should be in [0,1] range")
    
    # Load batch simulation data
    print(f"Loading batch simulation data...")
    batch_k_values, batch_compositions, n_simulations, n_pressure_conditions = load_batch_data(batch_json_path)
    print(f"✅ Batch data loaded")

    print(f"🔍!!!!!!!!!!!!! batch DATASET DEBUG:")

    print(f"   batch k values first 3 rows:\n{batch_k_values[:10]}")
    print(f"   batch compositions first 3 rows:\n{batch_compositions[0:600]}")

    
    
    print(f"\n🔄 PHASE 2: Running iterative experiment")
    
    # Pre-scale full batch once (do it here so run_iterative_batch_experiment
    # receives scaled arrays and doesn't perform scaling internally)
    print('\n🔧 Pre-scaling full batch data once (performed in __main__)')
    batch_new_x_all, batch_new_y_scaled_all = apply_training_scalers(
        raw_compositions=batch_compositions,
        raw_k_values=batch_k_values,
        dataset_train=uniform_dataset,
        nspecies=config['nspecies'],
        num_pressure_conditions=config['num_pressure_conditions'],
        debug=False
    )
    print(f'\n   Pre-scaled batch shapes: X={batch_new_x_all.shape}, Y={batch_new_y_scaled_all.shape}')

    # Run the adaptive zone-based experiment (NEW APPROACH)
    print(f"\n🚀 Running ADAPTIVE ZONE-BASED experiment...")
    adaptive_results = run_adaptive_zone_experiment(
        uniform_dataset=uniform_dataset,
        test_dataset=test_dataset,
        batch_new_x_all=batch_new_x_all,
        batch_new_y_scaled_all=batch_new_y_scaled_all,
        batch_k_values=batch_k_values,
        config=config,
        n_iterations=18,
        samples_per_iteration=100,
        initial_uniform_size=100,
        n_zones=6
    )
    
    print(f"\n📊 PHASE 3: Adaptive results analysis")
    print_experiment_summary(adaptive_results)
    
    # Also run the original random experiment for comparison (BASELINE)
    print(f"\n🔄 Running RANDOM BASELINE experiment for comparison...")
    random_results = run_iterative_batch_experiment(
        uniform_dataset=uniform_dataset,
        test_dataset=test_dataset,
        batch_new_x_all=batch_new_x_all,
        batch_new_y_scaled_all=batch_new_y_scaled_all,
        config=config,
        n_iterations=18,
        samples_per_iteration=100,
        initial_uniform_size=100
    )
    
    print(f"\n📊 PHASE 4: Random baseline results analysis")
    print_experiment_summary(random_results)
    
    # Save both results to JSON
    results_file = 'results/adaptive_vs_random_experiment_results.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'experiment_type': 'adaptive_vs_random_comparison',
            'adaptive_results': adaptive_results,
            'random_results': random_results,
            'comparison': {
                'adaptive_final_mse': adaptive_results[-1]['total_mse'],
                'random_final_mse': random_results[-1]['total_mse'],
                'adaptive_improvement_pct': ((adaptive_results[-1]['total_mse'] / adaptive_results[0]['total_mse']) - 1) * 100,
                'random_improvement_pct': ((random_results[-1]['total_mse'] / random_results[0]['total_mse']) - 1) * 100
            }
        }, f, indent=2)
    
    print(f"\n💾 Comparison results saved to: {results_file}")

    # ----------------------------
    # Visualization: Compare adaptive vs random approaches
    # ----------------------------
    try:
        plt.figure(figsize=(12, 8))

        # Prepare x-axes as cumulative total training samples
        x_adaptive = [r['total_samples'] for r in adaptive_results]
        x_random = [r['total_samples'] for r in random_results]

        # Plot 1: Overall MSE comparison vs total samples
        plt.subplot(2, 2, 1)
        adaptive_mse = [r['total_mse'] for r in adaptive_results]
        random_mse = [r['total_mse'] for r in random_results]

        plt.plot(x_adaptive, adaptive_mse, 'b-o', label='Adaptive Zone-Based', linewidth=2)
        plt.plot(x_random, random_mse, 'r-s', label='Random Sampling', linewidth=2)
        plt.xlabel('Total training samples')
        plt.ylabel('Total MSE')
        plt.title('MSE Evolution: Adaptive vs Random (by samples)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Zone MSE evolution for adaptive approach (x = cumulative samples)
        plt.subplot(2, 2, 2)
        n_iters = len(adaptive_results)
        # Determine number of zones dynamically from the results to avoid hardcoded mismatch
        n_zones = 0
        for it in adaptive_results:
            vals = it.get('zone_overall_mse_scaled')
            if vals:
                try:
                    n_zones = max(n_zones, len(vals))
                except Exception:
                    # ignore malformed entries
                    pass
        if n_zones == 0:
            n_zones = 6  # sensible default

        zone_mse_matrix = np.full((n_iters, n_zones), np.nan)

        for i, it in enumerate(adaptive_results):
            vals = it.get('zone_overall_mse_scaled')
            if vals:
                for z in range(min(n_zones, len(vals))):
                    try:
                        zone_mse_matrix[i, z] = vals[z]
                    except Exception:
                        zone_mse_matrix[i, z] = np.nan

        cmap = plt.get_cmap('tab10')
        for z in range(n_zones):
            plt.plot(x_adaptive, zone_mse_matrix[:, z], marker='o', label=f'Zone {z+1}', color=cmap(z % 10))

        plt.xlabel('Total training samples')
        plt.ylabel('Zone MSE (scaled)')
        plt.title('Adaptive: Per-Zone MSE Evolution (by samples)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # Plot 3: Sampling strategy visualization (counts per zone targeted)
        plt.subplot(2, 2, 3)
        target_zones_per_iter = []
        for r in adaptive_results[1:]:  # Skip iteration 0
            target_zones = r.get('target_zones', [])
            target_zones_per_iter.append(target_zones)

        if target_zones_per_iter:
            # Count which zones were targeted most
            zone_counts = {}
            for zones in target_zones_per_iter:
                for zone in zones:
                    zone_counts[zone] = zone_counts.get(zone, 0) + 1

            zones = sorted(zone_counts.keys())
            counts = [zone_counts[z] for z in zones]
            plt.bar(zones, counts, color='lightblue', edgecolor='navy')
            plt.xlabel('Zone Number')
            plt.ylabel('Times Targeted')
            plt.title('Adaptive: Zone Targeting Frequency')
            plt.grid(True, alpha=0.3)

        # Plot 4: Improvement comparison vs total samples
        plt.subplot(2, 2, 4)
        adaptive_improvement = [(adaptive_mse[i] / adaptive_mse[0] - 1) * 100 for i in range(len(adaptive_mse))]
        random_improvement = [(random_mse[i] / random_mse[0] - 1) * 100 for i in range(len(random_mse))]

        plt.plot(x_adaptive, adaptive_improvement, 'b-o', label='Adaptive Zone-Based', linewidth=2)
        plt.plot(x_random, random_improvement, 'r-s', label='Random Sampling', linewidth=2)
        plt.xlabel('Total training samples')
        plt.ylabel('MSE Change (%)')
        plt.title('Relative Improvement from Baseline (by samples)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        plt.tight_layout()
        comparison_plot = os.path.join('results', 'adaptive_vs_random_comparison.png')
        os.makedirs(os.path.dirname(comparison_plot), exist_ok=True)
        plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'📊 Comparison plot saved to: {comparison_plot}')

        # Print summary comparison
        print(f"\n📊 FINAL COMPARISON SUMMARY:")
        print(f"{'Method':<20} {'Final MSE':<15} {'Change %':<10} {'Best Iter':<10}")
        print(f"{'-'*60}")

        adaptive_final = adaptive_results[-1]['total_mse']
        adaptive_change = ((adaptive_final / adaptive_results[0]['total_mse']) - 1) * 100
        adaptive_best = min(enumerate(adaptive_results), key=lambda x: x[1]['total_mse'])

        random_final = random_results[-1]['total_mse']
        random_change = ((random_final / random_results[0]['total_mse']) - 1) * 100
        random_best = min(enumerate(random_results), key=lambda x: x[1]['total_mse'])

        print(f"{'Adaptive Zone':<20} {adaptive_final:<15.6f} {adaptive_change:<10.1f} {adaptive_best[0]:<10}")
        print(f"{'Random Sampling':<20} {random_final:<15.6f} {random_change:<10.1f} {random_best[0]:<10}")

        if adaptive_final < random_final:
            improvement = ((random_final / adaptive_final) - 1) * 100
            print(f"\n🎉 Adaptive approach is {improvement:.1f}% better than random!")
        else:
            degradation = ((adaptive_final / random_final) - 1) * 100
            print(f"\n⚠️ Adaptive approach is {degradation:.1f}% worse than random")

    except Exception as e:
        print('Could not create comparison plots:', e)

    print(f"\n✅ ADAPTIVE VS RANDOM EXPERIMENT COMPLETE!")
