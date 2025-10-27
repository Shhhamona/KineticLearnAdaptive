"""
Adaptive Sampling Analysis Script

This script demonstrates the AdaptiveSamplingPipeline which implements
iterative window-based sampling inspired by K-centered adaptive learning.

The pipeline:
1. Starts with an initial training set (can be 0 samples!)
2. At each iteration, samples from a window around the average point
3. Window shrinks each iteration to focus sampling
4. Incrementally adds samples and retrains the model
5. Uses multiple seeds for robustness

Key Feature:
- Set initial_training_size = 0 to start completely from scratch
- The pipeline will use the pool dataset's center as the initial sampling point
- Scalers are properly initialized from the reference dataset

Usage:
    python examples/run_adaptive_sampling.py
"""

from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetic_modelling import (
    MultiPressureDataset,
    SVRModel,
    AdaptiveSamplingPipeline
)


def main():
    # Configuration
    nspecies = 3
    num_pressure_conditions = 2
    react_idx = [0, 1, 2]
    
    # Adaptive sampling settings
    n_iterations = 11
    samples_per_iteration = 100
    initial_training_size = 0  # Start with 0 samples (can be changed to any value)
    initial_window_size = 1  # ±100% around center
    shrink_rate = 0.50  # 50% reduction each iteration
    num_seeds = 5
    replace_mode = True
    
    # SVR hyperparameters
    svr_params = [
        {'C': 10, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'},
        {'C': 20, 'epsilon': 0.005, 'gamma': 5, 'kernel': 'rbf'},
        {'C': 5, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'}
    ]
    
    print("="*70)
    print("Adaptive Sampling Analysis - Multiple Pool Files")
    print("="*70)
    print(f"Initial training size: {initial_training_size}")
    print(f"Iterations: {n_iterations}")
    print(f"Samples per iteration: {samples_per_iteration}")
    print(f"Initial window size: {initial_window_size}")
    print(f"Shrink rate: {shrink_rate}")
    print(f"Number of seeds: {num_seeds}")
    print("="*70)
    
    # Initial dataset file
    init_file = Path("data/SampleEfficiency/O2_simple_uniform.txt")
    
    # Window sampling batch files with K boundaries
    BATCH_FILES = [
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-08-27/batch_4000sims_20250827_010028.json',
            'label': 'Window Batch 1 (4000 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/2, K_true×2]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-28/batch_1000sims_20250928_191628.json',
            'label': 'Window Batch 2 (1000 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.15, K_true×1.15]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2500sims_20250929_031845.json',
            'label': 'Window Batch 3 (2500 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.15, K_true×1.15]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2000sims_20250929_205429.json',
            'label': 'Window Batch 4 (2000 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.005, K_true×1.005]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-28/batch_1500sims_20250928_224858.json',
            'label': 'Window Batch 5 (1500 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.0005, K_true×1.0005] '
        },
    ]
    
    # Test file
    test_file = Path("data/SampleEfficiency/O2_simple_test_real_K.txt")
    
    # Load datasets
    print("\n📂 Loading datasets...")
    
    # Load reference dataset to initialize scalers (needed even if initial_training_size=0)
    if not init_file.exists():
        raise FileNotFoundError(f"Initial file not found: {init_file}")
    
    reference_dataset = MultiPressureDataset(
        nspecies=nspecies,
        num_pressure_conditions=num_pressure_conditions,
        src_file=str(init_file),
        react_idx=react_idx
    )
    
    # Create initial training dataset (can be empty if initial_training_size=0)
    if initial_training_size > 0:
        # Use subset of reference dataset for initial training
        ref_x, ref_y = reference_dataset.get_data()
        initial_x = ref_x[:initial_training_size]
        initial_y = ref_y[:initial_training_size]
    else:
        # Create empty arrays with correct shape for 0 samples
        # This ensures the dataset is properly initialized for sampling
        ref_x, ref_y = reference_dataset.get_data()
        initial_x = np.empty((0, ref_x.shape[1]))
        initial_y = np.empty((0, ref_y.shape[1]))
    
    # Create initial dataset with same scalers as reference
    # Even with 0 samples, this preserves the scaling information
    initial_dataset = MultiPressureDataset(
        nspecies=nspecies,
        num_pressure_conditions=num_pressure_conditions,
        processed_x=initial_x,
        processed_y=initial_y,
        scaler_input=reference_dataset.scaler_input,
        scaler_output=reference_dataset.scaler_output
    )
    
    print(f"✓ Initial training: {len(initial_dataset)} samples" + 
          (f" (from {init_file.name})" if initial_training_size > 0 else " (starting from scratch)"))
    
    # Load multiple pool datasets
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
            scaler_input=initial_dataset.scaler_input,  # Use same scalers as initial
            scaler_output=initial_dataset.scaler_output
        )
        pool_datasets.append(pool_ds)
        print(f"✓ Pool {i+1}: {len(pool_ds)} samples - {batch_info['label']}")
        print(f"          {batch_info['k_range']}")
    
    if len(pool_datasets) == 0:
        raise ValueError("No pool datasets could be loaded!")
    
    # Load test dataset with same scalers
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    test_dataset = MultiPressureDataset(
        nspecies=nspecies,
        num_pressure_conditions=num_pressure_conditions,
        src_file=str(test_file),
        react_idx=react_idx,
        scaler_input=initial_dataset.scaler_input,
        scaler_output=initial_dataset.scaler_output
    )
    
    print(f"✓ Test dataset: {len(test_dataset)} samples")
    print(f"\nTotal pool samples across all files: {sum(len(ds) for ds in pool_datasets)}")
    
    # Setup model parameters
    model_params = {
        'params': svr_params,
        'model_name': 'adaptive_sampling_svr'
    }
    
    # Create and run pipeline
    print("\n🚀 Creating adaptive sampling pipeline...")
    pipeline = AdaptiveSamplingPipeline(
        initial_dataset=initial_dataset,
        pool_datasets=pool_datasets,  # Now passing list of datasets
        test_dataset=test_dataset,
        model_class=SVRModel,
        model_params=model_params,
        n_iterations=n_iterations,
        samples_per_iteration=samples_per_iteration,
        initial_window_size=initial_window_size,
        shrink_rate=shrink_rate,
        num_seeds=num_seeds,
        window_type='input',  # Sample based on input (K values)
        pipeline_name=f"adaptive_sampling_multipools_w{initial_window_size}_s{shrink_rate}",
        replace_mode=replace_mode
    )
    
    print("\n🔄 Running adaptive sampling...")
    results = pipeline.save_and_return(save_results=True)
    
    # Display key results
    print("\n" + "="*70)
    print("Key Results")
    print("="*70)
    agg = results['aggregated_results']
    
    if len(agg) > 0:
        initial_mse = agg[0]['mean_total_mse']
        final_mse = agg[-1]['mean_total_mse']
        improvement = (initial_mse - final_mse) / initial_mse * 100
        
        print(f"Initial MSE: {initial_mse:.6e} ± {agg[0]['std_total_mse']:.6e}")
        print(f"Final MSE:   {final_mse:.6e} ± {agg[-1]['std_total_mse']:.6e}")
        print(f"Improvement: {improvement:.1f}%")
        print(f"Total samples used: {agg[-1]['total_samples']:.0f}")
        print(f"Final window size: {agg[-1]['window_size']:.4f}")
    
    print("="*70)


if __name__ == "__main__":
    main()
