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


def run_iterative_batch_experiment(uniform_dataset, test_dataset, batch_k_values, batch_compositions, 
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
    
    # Now run iterative batch additions
    for iteration in range(1, n_iterations + 1):
        print(f"\n🎯 ITERATION {iteration}")
        
        # Calculate which batch samples to use for this iteration
        start_idx = (iteration - 1) * samples_per_iteration
        end_idx = start_idx + samples_per_iteration
        
        if end_idx > len(batch_k_values):
            print(f"   ⚠️ Not enough batch samples remaining. Using {len(batch_k_values) - start_idx} samples.")
            end_idx = len(batch_k_values)
        
        # Extract batch samples for this iteration
        iter_k_values = batch_k_values[start_idx:end_idx]
        
        # CRITICAL: batch_compositions is in BLOCK format: [all_pressure_0_rows, all_pressure_1_rows]
        # We need to extract the corresponding slices from each pressure block and concatenate
        # to maintain BLOCK format for apply_training_scalers
        
        n_sims_per_iter = end_idx - start_idx
        total_sims = len(batch_k_values)
        
        # Extract compositions for each pressure condition separately
        iter_compositions_blocks = []
        for pressure_idx in range(config['num_pressure_conditions']):
            # Each pressure block starts at (pressure_idx * total_sims)
            pressure_block_start = pressure_idx * total_sims
            # Extract the slice for this iteration within this pressure block
            pressure_start = pressure_block_start + start_idx
            pressure_end = pressure_block_start + end_idx
            pressure_compositions = batch_compositions[pressure_start:pressure_end]
            iter_compositions_blocks.append(pressure_compositions)
        
        # Concatenate blocks to maintain BLOCK format: [pressure_0_slice, pressure_1_slice, ...]
        iter_compositions = np.vstack(iter_compositions_blocks)
        
        print(f"   Adding batch samples {start_idx} to {end_idx-1}")
        print(f"   Batch k_values shape: {iter_k_values.shape}")
        print(f"   Batch compositions shape: {iter_compositions.shape}")
        print(f"   🔍 BLOCK FORMAT DEBUG:")
        print(f"      Total simulations in batch: {total_sims}")
        print(f"      Simulations this iteration: {n_sims_per_iter}")
        for pressure_idx in range(config['num_pressure_conditions']):
            pressure_block_start = pressure_idx * total_sims
            pressure_start = pressure_block_start + start_idx
            pressure_end = pressure_block_start + end_idx
            print(f"      Pressure {pressure_idx}: extracted rows [{pressure_start}:{pressure_end}] from batch_compositions")
            print(f"      Pressure {pressure_idx} block shape: {iter_compositions_blocks[pressure_idx].shape}")
            print(f"      Pressure {pressure_idx} first 3 rows:\n{iter_compositions_blocks[pressure_idx][:3]}")
        print(f"   Final BLOCK iter_compositions shape: {iter_compositions.shape}")
        print(f"   First 6 rows (should be all pressure 0):\n{iter_compositions[:6]}")
        if iter_compositions.shape[0] > n_sims_per_iter:
            print(f"   Last 6 rows (should be all pressure 1):\n{iter_compositions[-6:]}")
    
        # Scale the new batch samples using apply_training_scalers
        print(f"   Scaling new batch samples...")
        new_x, new_y_scaled = apply_training_scalers(
            raw_compositions=iter_compositions,
            raw_k_values=iter_k_values,
            dataset_train=uniform_dataset,
            nspecies=config['nspecies'],
            num_pressure_conditions=config['num_pressure_conditions'],
            debug=True  # Reduce verbosity
        )
        
        print(f"   Scaled batch data: X={new_x.shape}, Y={new_y_scaled.shape}")
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

    # Use a list for seeds so the training helpers can run multi-seed experiments
    config['seeds'] = [42, 43, 44, 45, 46] 
    
    print(f"🎯 ITERATIVE BATCH TRAINING EXPERIMENT")
    print(f"Configuration: {config}")
    
    # File paths
    #batch_json_path = 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-08-20/batch_500sims_20250820_141641.json'
    batch_json_path = 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-08-27/batch_4000sims_20250827_010028.json'
    src_file_train = 'data/SampleEfficiency/O2_simple_uniform.txt'
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
    
    # Run the iterative experiment
    # Start with 100 uniform, add 10 batch samples per iteration for 20 iterations
    iteration_results = run_iterative_batch_experiment(
        uniform_dataset=uniform_dataset,
        test_dataset=test_dataset,
        batch_k_values=batch_k_values,
        batch_compositions=batch_compositions,
        config=config,
        n_iterations=10,  # 20 iterations
        samples_per_iteration=50,  # 10 samples each
        initial_uniform_size=50
    )
    
    print(f"\n📊 PHASE 3: Results analysis")
    print_experiment_summary(iteration_results)
    
    # Save results to JSON
    results_file = 'results/iterative_batch_experiment_results.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'experiment_type': 'iterative_batch',
            'iteration_results': iteration_results,
            'summary': {
                'total_iterations': len(iteration_results) - 1,
                'initial_mse': iteration_results[0]['total_mse'],
                'final_mse': iteration_results[-1]['total_mse'],
                'overall_change_pct': ((iteration_results[-1]['total_mse'] / iteration_results[0]['total_mse']) - 1) * 100
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")

    # ----------------------------
    # Visualization: per-zone MSE evolution (linear Y scale) + average across zones
    # ----------------------------
    try:
        # Determine number of zones (find first iteration that has zone_overall_mse_scaled)
        n_zones = None
        for it in iteration_results:
            if it.get('zone_overall_mse_scaled'):
                n_zones = len(it['zone_overall_mse_scaled'])
                break
        if n_zones is None:
            print('No zone data found to plot.')
        else:
            # Build matrix iterations x zones
            n_iters = len(iteration_results)
            zone_mse_matrix = np.full((n_iters, n_zones), np.nan)
            for i, it in enumerate(iteration_results):
                vals = it.get('zone_overall_mse_scaled')
                if vals:
                    for z in range(min(n_zones, len(vals))):
                        zone_mse_matrix[i, z] = vals[z]

            # Compute per-iteration average across zones (ignore NaNs)
            avg_per_iter = np.nanmean(zone_mse_matrix, axis=1)

            # Print a compact table including average
            print('\n📈 Zone overall MSE evolution (scaled):')
            header_cols = [f'Z{z+1}' for z in range(n_zones)] + ['Avg']
            header = 'Iter | ' + ' | '.join(header_cols)
            print(header)
            print('-' * len(header))
            for i in range(n_iters):
                row_vals = []
                for x in zone_mse_matrix[i]:
                    if np.isnan(x):
                        row_vals.append('nan')
                    else:
                        row_vals.append(f'{x:.3e}')
                # average
                avg_val = avg_per_iter[i]
                avg_str = 'nan' if np.isnan(avg_val) else f'{avg_val:.3e}'
                row_vals.append(avg_str)
                print(f'{i:4d} | ' + ' | '.join(row_vals))

            # Plot linear-scale per-zone MSE and average line
            plt.figure(figsize=(10, 6))
            x = np.arange(n_iters)
            cmap = plt.get_cmap('tab10')
            for z in range(n_zones):
                plt.plot(x, zone_mse_matrix[:, z], marker='o', label=f'Zone {z+1}', color=cmap(z % 10))
            # Plot average across zones as a bold black line
            plt.plot(x, avg_per_iter, marker='s', color='k', linewidth=2.0, label='Average')
            plt.xlabel('Iteration')
            plt.ylabel('Zone overall MSE (scaled)')
            plt.title('Per-zone overall MSE evolution (scaled)')
            plt.legend(loc='upper right')
            plt.grid(True, which='both', ls='--', lw=0.5)
            out_plot = os.path.join('results', 'zone_mse_evolution.png')
            os.makedirs(os.path.dirname(out_plot), exist_ok=True)
            plt.tight_layout()
            plt.savefig(out_plot)
            plt.close()
            print(f'📉 Zone MSE evolution plot saved to: {out_plot}')
    except Exception as e:
        print('Could not create zone evolution plot:', e)
    print(f"\n✅ ITERATIVE EXPERIMENT COMPLETE!")
