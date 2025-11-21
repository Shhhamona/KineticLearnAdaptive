"""
Adaptive Batch Sampling Analysis Script

This script demonstrates the AdaptiveBatchSamplingPipeline which implements
iterative window-based sampling with continuous Neural Network training.

The pipeline:
1. Creates a Neural Network model
2. At each iteration:
   - Samples from a window around the test dataset center
   - Window shrinks each iteration to focus sampling
   - Trains the NN for multiple epochs with new samples (continuous training)
   - Evaluates performance
3. Uses multiple seeds for robustness

Key Features:
- Uses Neural Network instead of SVM (continuous training across iterations)
- Samples from pool datasets at each iteration
- Center point calculated from test dataset (stays constant)
- Model continuously improves with mini-batch gradient descent

Usage:
    python pipeline_runs/run_adaptive_batch_sampling.py
"""

from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetic_modelling import (
    MultiPressureDataset,
    NeuralNetModel,
    AdaptiveBatchSamplingPipeline
)


def main():
    # Configuration
    nspecies = 3
    num_pressure_conditions = 2
    react_idx = [0,1,2]
    
    # Adaptive batch sampling settings
    n_iterations = 10  # Number of iterations (one per pool file)
    samples_per_iteration = int(2000/n_iterations)  # Samples to grab from each pool
    n_epochs = 50  # Train for 50 epochs at each iteration
    batch_size = 16  # Batch size for NN training
    initial_window_size = 1.0  # ±100% around center
    shrink_rate = 1  # 70% reduction each iteration
    num_seeds = 10
    
    # Neural Network hyperparameters
    nn_params = {
        'input_size': None,  # Will be set automatically from data
        'output_size': None,  # Will be set automatically from data
        'hidden_sizes': (64, 32), #(110, 60),#(64, 32),  # Two hidden layers
        'activation': 'tanh',
        'learning_rate': 0.001,
        'model_name': 'adaptive_batch_sampling_nn'
    }
    
    print("="*70)
    print("Adaptive Batch Sampling Analysis - Neural Network")
    print("="*70)
    print(f"Iterations: {n_iterations}")
    print(f"Samples per iteration: {samples_per_iteration}")
    print(f"Epochs per iteration: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Initial window size: {initial_window_size}")
    print(f"Shrink rate: {shrink_rate}")
    print(f"Number of seeds: {num_seeds}")
    print(f"NN architecture: {nn_params['hidden_sizes']}")
    print(f"Learning rate: {nn_params['learning_rate']}")
    print("="*70)
    
    # Initial dataset file (used only for scaler initialization)
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
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-10-27/batch_500sims_20251027_154921.json',
            'label': 'Window Batch 4 (500 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.01, K_true×1.01]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2000sims_20250929_205429.json',
            'label': 'Window Batch 5 (2000 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.005, K_true×1.005]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-28/batch_1500sims_20250928_224858.json',
            'label': 'Window Batch 6 (1500 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.0005, K_true×1.0005]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2000sims_20250929_125706.json',
            'label': 'Window Batch 7 (2000 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.00005, K_true×1.00005] '
        }
    ]
    
    # Test file
    test_file = Path("data/SampleEfficiency/O2_simple_test_real_K.txt")
    
    # Load datasets
    print("\n📂 Loading datasets...")
    
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
    
    # Load multiple pool datasets
    pool_datasets = []
    for i, batch_info in enumerate(BATCH_FILES[:n_iterations]):  # Only load as many as needed
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
            scaler_input=reference_dataset.scaler_input,  # Use same scalers
            scaler_output=reference_dataset.scaler_output
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
        scaler_input=reference_dataset.scaler_input,
        scaler_output=reference_dataset.scaler_output
    )
    
    print(f"✓ Test dataset: {len(test_dataset)} samples")
    print(f"\nTotal pool samples across all files: {sum(len(ds) for ds in pool_datasets)}")
    
    # Setup model parameters with correct dimensions
    x_sample, y_sample = test_dataset.get_data()
    nn_params['input_size'] = x_sample.shape[1]
    nn_params['output_size'] = y_sample.shape[1]
    
    print(f"\n🧠 Neural Network Configuration:")
    print(f"   Input size: {nn_params['input_size']}")
    print(f"   Hidden layers: {nn_params['hidden_sizes']}")
    print(f"   Output size: {nn_params['output_size']}")
    print(f"   Learning rate: {nn_params['learning_rate']}")
    
    # Create and run pipeline
    print("\n🚀 Creating adaptive batch sampling pipeline...")
    pipeline = AdaptiveBatchSamplingPipeline(
        pool_datasets=pool_datasets,  # List of pool datasets
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
        window_type='output',  # Sample based on input (K values)
        pipeline_name=f"adaptive_batch_sampling_w{initial_window_size}_s{shrink_rate}_e{n_epochs}",
        results_dir="pipeline_results"
    )
    
    print("\n🔄 Running adaptive batch sampling with continuous NN training...")
    results = pipeline.save_and_return(save_results=True)
    
    # Display key results
    print("\n" + "="*70)
    print("Key Results")
    print("="*70)
    agg = results['aggregated_results']
    
    if len(agg) > 0:
        # Skip iteration 0 (untrained model) for improvement calculation
        initial_idx = 1 if len(agg) > 1 else 0
        initial_mse = agg[initial_idx]['mean_total_mse']
        final_mse = agg[-1]['mean_total_mse']
        improvement = (initial_mse - final_mse) / initial_mse * 100 if initial_mse > 0 else 0
        
        print(f"Untrained MSE: {agg[0]['mean_total_mse']:.6e} ± {agg[0]['std_total_mse']:.6e}")
        if len(agg) > 1:
            print(f"After 1st iteration: {initial_mse:.6e} ± {agg[initial_idx]['std_total_mse']:.6e}")
        print(f"Final MSE:   {final_mse:.6e} ± {agg[-1]['std_total_mse']:.6e}")
        print(f"Improvement: {improvement:.1f}%")
        print(f"Total samples seen: {agg[-1]['total_samples_seen']:.0f}")
        print(f"Final window size: {agg[-1]['window_size']:.4f}")
        
        # Show training progress
        print(f"\nTraining Loss Evolution:")
        for i, result in enumerate(agg):
            if result['mean_train_loss'] is not None:
                print(f"  Iteration {i}: {result['mean_train_loss']:.6e}")
    
    print("="*70)


if __name__ == "__main__":
    main()
