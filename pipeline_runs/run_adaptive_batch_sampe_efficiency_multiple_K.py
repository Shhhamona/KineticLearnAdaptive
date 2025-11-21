"""
Adaptive Batch Sampling Sample Efficiency Analysis Script

This script compares different shrink rates for adaptive batch sampling while
keeping the total number of samples and samples_per_iteration fixed.

The experiments:
- Fixed total sample budget (e.g., 1000 samples)
- Fixed samples_per_iteration values (e.g., 50, 100, 200, 500)
- For each samples_per_iteration:
  * Test multiple shrink rates: 0.0 (baseline/no shrinking), 0.05, 0.10, 0.15, 0.20, 0.30
  * shrink_rate = 0.0 means the window stays constant (baseline)
  * Higher shrink rates mean more aggressive window narrowing

Key comparisons:
- Does adaptive sampling (window shrinking) improve performance vs fixed windows?
- What is the optimal shrink rate for different iteration granularities?
- How does iteration frequency affect the benefit of adaptive sampling?

Usage:
    python pipeline_runs/run_adaptive_batch_sampe_efficiency.py
"""

from pathlib import Path
import sys
import numpy as np
import json
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetic_modelling import (
    MultiPressureDataset,
    NeuralNetModel,
    AdaptiveBatchSamplingPipeline
)


def parse_k_range_factor(k_range_str: str) -> float:
    """
    Parse K range string to extract the range factor.
    
    Examples:
        'K ∈ [K_true/2, K_true×2]' -> 2.0
        'K ∈ [K_true/1.15, K_true×1.15]' -> 1.15
        'K ∈ [K_true/1.005, K_true×1.005]' -> 1.005
    
    Args:
        k_range_str: String describing the K range
        
    Returns:
        The multiplicative factor (e.g., 2.0 for K/2 to K*2)
    """
    import re
    
    # Try to find pattern like "K_true×N" or "K_true*N"
    match = re.search(r'K_true[×*](\d+\.?\d*)', k_range_str)
    if match:
        return float(match.group(1))
    
    # Try to find pattern like "K_true/N"
    match = re.search(r'K_true/(\d+\.?\d*)', k_range_str)
    if match:
        return float(match.group(1))
    
    # Default: assume wide range
    print(f"    ⚠️  Could not parse K range from: {k_range_str}")
    return 1000.0  # Very large factor = no restriction


def main():
    # Configuration
    nspecies = 3
    num_pressure_conditions = 2
    react_idx = [0,1,2]
    
    # Experimental parameters
    # Keys = samples_per_iteration, Values = list of shrink_rates to test
    # Note: shrink_rate = 0.0 means NO shrinking (baseline with multiple iterations)
    iteration_configs = {
        #50: [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],    # 50 samples per iteration, test various shrink rates
        #100: [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],   # 100 samples per iteration
        #200: [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],   # 200 samples per iteration
        800: [0.1],   # 300 samples per iteration
    }
    
    # Total sample budget (will determine n_iterations from samples_per_iteration)
    max_total_samples = 3000  # Fixed total budget
    
    # Training hyperparameters
    n_epochs = 50  # Train for 50 epochs at each iteration
    batch_size = 16  # Batch size for NN training
    initial_window_size = 1.0  # ±100% around center
    num_seeds = 10
    
    # Neural Network hyperparameters
    nn_params = {
        'input_size': None,  # Will be set automatically from data
        'output_size': None,  # Will be set automatically from data
        'hidden_sizes': (64, 32),  # Two hidden layers
        'activation': 'tanh',
        'learning_rate': 0.001,
        'model_name': 'adaptive_batch_sampling_nn'
    }
    
    print("="*70)
    print("Sample Efficiency Analysis: Adaptive vs Baseline")
    print("="*70)
    print(f"Total sample budget: {max_total_samples}")
    print(f"Samples per iteration configurations: {list(iteration_configs.keys())}")
    print(f"Shrink rates to test: {iteration_configs[list(iteration_configs.keys())[0]]}")
    print(f"Epochs per iteration: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Initial window size: {initial_window_size}")
    print(f"Number of seeds: {num_seeds}")
    print(f"NN architecture: {nn_params['hidden_sizes']}")
    print(f"Learning rate: {nn_params['learning_rate']}")
    print("="*70)
    
    # Window sampling batch files with K boundaries
    BATCH_FILES = [
        # K-factor 2.0 - Widest bounds
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-08-27/batch_4000sims_20250827_010028.json',
            'label': 'Window Batch 1 (4000 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/2, K_true×2]'
        },
        
        # K-factor 1.5
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-08/batch_3000sims_20251108_061037.json',
            'label': 'Window Batch 2 (3000 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.5, K_true×1.5]'
        },
        
        # K-factor 1.15
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-28/batch_1000sims_20250928_191628.json',
            'label': 'Window Batch 3 (1000 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.15, K_true×1.15]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2500sims_20250929_031845.json',
            'label': 'Window Batch 4 (2500 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.15, K_true×1.15]'
        },
        
        # K-factor 1.05
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-08/batch_2500sims_20251108_184222.json',
            'label': 'Window Batch 5 (2500 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.05, K_true×1.05]'
        },
        
        # K-factor 1.025
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2500sims_20251109_072738.json',
            'label': 'Window Batch 6 (2500 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.025, K_true×1.025]'
        },
        
        # K-factor 1.01
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-10-27/batch_500sims_20251027_154921.json',
            'label': 'Window Batch 7 (500 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.01, K_true×1.01]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-07/batch_2000sims_20251107_204934.json',
            'label': 'Window Batch 8 (2000 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.01, K_true×1.01]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2500sims_20251109_195233.json',
            'label': 'Window Batch 9 (2500 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.01, K_true×1.01]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2525sims_20251109_195849.json',
            'label': 'Window Batch 10 (2525 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.01, K_true×1.01]'
        },
        
        # K-factor 1.0075
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2550sims_20251109_200200.json',
            'label': 'Window Batch 11 (2550 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.0075, K_true×1.0075]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2575sims_20251109_200551.json',
            'label': 'Window Batch 12 (2575 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.0075, K_true×1.0075]'
        },
        
        # K-factor 1.005
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2000sims_20250929_205429.json',
            'label': 'Window Batch 13 (2000 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.005, K_true×1.005]'
        },
        
        # K-factor 1.0025
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2600sims_20251109_074357.json',
            'label': 'Window Batch 14 (2600 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.0025, K_true×1.0025]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-10/batch_2550sims_20251110_033347.json',
            'label': 'Window Batch 15 (2550 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.0025, K_true×1.0025]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-10/batch_2575sims_20251110_033748.json',
            'label': 'Window Batch 16 (2575 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.0025, K_true×1.0025]'
        },
        
        # K-factor 1.001
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-08/batch_2600sims_20251108_185805.json',
            'label': 'Window Batch 17 (2600 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.001, K_true×1.001]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-10/batch_2500sims_20251110_032616.json',
            'label': 'Window Batch 18 (2500 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.001, K_true×1.001]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-10/batch_2525sims_20251110_033128.json',
            'label': 'Window Batch 19 (2525 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.001, K_true×1.001]'
        },
        
        # K-factor 1.0005
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-28/batch_1500sims_20250928_224858.json',
            'label': 'Window Batch 20 (1500 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.0005, K_true×1.0005]'
        },
        
        # K-factor 1.00025
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2700sims_20251109_075908.json',
            'label': 'Window Batch 21 (2700 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.00025, K_true×1.00025]'
        },
        
        # K-factor 1.0001
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-08/batch_2700sims_20251108_191338.json',
            'label': 'Window Batch 22 (2700 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.0001, K_true×1.0001]'
        },
        
        # K-factor 1.00005
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2000sims_20250929_125706.json',
            'label': 'Window Batch 23 (2000 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.00005, K_true×1.00005]'
        },
        
        # K-factor 1.000025
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2750sims_20251109_080611.json',
            'label': 'Window Batch 24 (2750 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.000025, K_true×1.000025]'
        },
        
        # K-factor 1.00001 - Narrowest bounds
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-08/batch_2750sims_20251108_191922.json',
            'label': 'Window Batch 25 (2750 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.00001, K_true×1.00001]'
        }
    ]

    
    # Test file
    test_file = Path("data/SampleEfficiency/O2_simple_test_real_K.txt")
    
    # Load datasets
    print("\n📂 Loading datasets...")
    
    # Initial dataset file (used only for scaler initialization)
    init_file = Path("data/SampleEfficiency/O2_simple_uniform.txt")
    
    # Load reference dataset to initialize scalers
    if not init_file.exists():
        raise FileNotFoundError(f"Initial file not found: {init_file}")
    
    reference_dataset = MultiPressureDataset(
        nspecies=nspecies,
        num_pressure_conditions=num_pressure_conditions,
        src_file=str(init_file),
        react_idx=react_idx
    )
    
    print(f"✓ Reference dataset: {len(reference_dataset)} samples (for scaler initialization)")
    
    # Load test dataset with same scalers
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    test_dataset = MultiPressureDataset(
        nspecies=nspecies,
        num_pressure_conditions=num_pressure_conditions,
        src_file=str(test_file),
        react_idx=react_idx,
        scaler_input=reference_dataset.scaler_input,
        scaler_output=reference_dataset.scaler_output
    )
    
    print(f"✓ Test dataset: {len(test_dataset)} samples")
    
    # Setup model parameters with correct dimensions
    x_sample, y_sample = test_dataset.get_data()
    nn_params['input_size'] = x_sample.shape[1]
    nn_params['output_size'] = y_sample.shape[1]
    
    print(f"\n🧠 Neural Network Configuration:")
    print(f"   Input size: {nn_params['input_size']}")
    print(f"   Hidden layers: {nn_params['hidden_sizes']}")
    print(f"   Output size: {nn_params['output_size']}")
    print(f"   Learning rate: {nn_params['learning_rate']}")
    
    # Extract K_true values from test dataset (these are the target reaction rates)
    # The output data contains the true K values that we're trying to predict
    _, y_test = test_dataset.get_data()
    # Take the first sample's K values as the true K values (they should be consistent)
    # Shape: (1, n_reactions) -> (n_reactions,)
    k_true_values = y_test[0, :]
    print(f"\n✓ Extracted K_true values from test dataset:")
    print(f"   Shape: {k_true_values.shape}")
    print(f"   Values: {k_true_values}")
    
    # Load ALL pool datasets once (they're reused for all experiments)
    print("\n📂 Loading all pool datasets...")
    pool_datasets = []
    for i, batch_info in enumerate(BATCH_FILES):
        batch_path = Path(batch_info['path'])
        if not batch_path.exists():
            print(f"⚠️  Pool file not found: {batch_path}")
            print(f"   Skipping: {batch_info['label']}")
            continue
        
        pool_ds = MultiPressureDataset(
            nspecies=nspecies,
            num_pressure_conditions=num_pressure_conditions,
            src_file=str(batch_path),
            react_idx=react_idx,
            scaler_input=reference_dataset.scaler_input,
            scaler_output=reference_dataset.scaler_output
        )
        
        # Attach K range metadata for validation
        pool_ds.k_range_factor = parse_k_range_factor(batch_info['k_range'])
        pool_ds.label = batch_info['label']
        pool_ds.k_range_str = batch_info['k_range']
        pool_ds.k_true_values = k_true_values  # Add the true K values
        
        pool_datasets.append(pool_ds)
        print(f"✓ Pool {i+1}: {len(pool_ds)} samples - {batch_info['label']}")
        print(f"          {batch_info['k_range']} (factor: {pool_ds.k_range_factor})")
    
    if len(pool_datasets) == 0:
        raise ValueError("No pool datasets could be loaded!")
    
    print(f"\n✓ Total pool samples available across all files: {sum(len(ds) for ds in pool_datasets)}")
    
    # Store all results for comparison
    # Structure: all_results[samples_per_iteration][shrink_rate] = {...}
    all_results = {}
    
    # Run experiments for each samples_per_iteration and shrink_rate combination
    for samples_per_iteration in iteration_configs.keys():
        all_results[samples_per_iteration] = {}
        
        # Calculate number of iterations needed to reach total sample budget
        n_iterations = int(max_total_samples / samples_per_iteration)
        
        for shrink_rate in iteration_configs[samples_per_iteration]:
            print("\n" + "="*70)
            print(f"EXPERIMENT: {samples_per_iteration} samples/iter, shrink_rate={shrink_rate}")
            print("="*70)
            print(f"Number of iterations: {n_iterations}")
            print(f"Total samples: {samples_per_iteration * n_iterations}")
            if shrink_rate == 0.0:
                print("⚠️  BASELINE: No window shrinking (fixed window)")
            else:
                print(f"✓ ADAPTIVE: Window shrinks by {shrink_rate*100:.1f}% each iteration")
            print("="*70)
            
            # Create and run pipeline
            print(f"\n🚀 Creating pipeline...")
            pipeline = AdaptiveBatchSamplingPipeline(
                pool_datasets=pool_datasets,
                test_dataset=test_dataset,
                model_class=NeuralNetModel,
                model_params=nn_params,
                n_iterations=n_iterations,
                samples_per_iteration=samples_per_iteration,
                n_epochs=n_epochs,
                batch_size=batch_size,
                initial_window_size=initial_window_size,
                shrink_rate=shrink_rate,
                num_seeds=num_seeds,
                use_model_prediction=True,
                window_type='output',
                pipeline_name=f"sample_efficiency_{samples_per_iteration}per_iter_shrink{shrink_rate}",
                results_dir="pipeline_results"
            )
            
            print(f"\n🔄 Training model...")
            results = pipeline.save_and_return(save_results=True)
            
            # Extract final MSE
            agg = results['aggregated_results']
            if len(agg) > 0:
                final_mse = agg[-1]['mean_total_mse']
                final_mse_std = agg[-1]['std_total_mse']
                total_samples_seen = agg[-1]['total_samples_seen']
                
                all_results[samples_per_iteration][shrink_rate] = {
                    'final_mse': final_mse,
                    'final_mse_std': final_mse_std,
                    'total_samples': total_samples_seen,
                    'n_iterations': n_iterations
                }
                
                print(f"\n✓ Final MSE: {final_mse:.6e} ± {final_mse_std:.6e}")
                print(f"✓ Total samples used: {total_samples_seen:.0f}")
    
    # Display comparison summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    for samples_per_iter in sorted(all_results.keys()):
        if len(all_results[samples_per_iter]) == 0:
            continue
            
        print(f"\n📊 Samples per Iteration: {samples_per_iter}")
        print("-" * 70)
        
        baseline_mse = None
        if 0.0 in all_results[samples_per_iter]:
            baseline_mse = all_results[samples_per_iter][0.0]['final_mse']
            n_iters = all_results[samples_per_iter][0.0]['n_iterations']
            print(f"  Baseline (shrink=0.0, {n_iters} iters):  MSE = {baseline_mse:.6e}")
        
        for shrink in sorted(all_results[samples_per_iter].keys()):
            if shrink == 0.0:
                continue
            
            result = all_results[samples_per_iter][shrink]
            mse = result['final_mse']
            n_iters = result['n_iterations']
            improvement = ""
            if baseline_mse is not None and baseline_mse > 0:
                pct = (baseline_mse - mse) / baseline_mse * 100
                improvement = f" ({pct:+.1f}% vs baseline)"
            
            print(f"  Shrink rate {shrink:.2f} ({n_iters} iters):   MSE = {mse:.6e}{improvement}")
    
    # Save summary to file
    summary_file = Path("pipeline_results/sample_efficiency_summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Summary saved to: {summary_file}")
    print("="*70)


if __name__ == "__main__":
    main()
