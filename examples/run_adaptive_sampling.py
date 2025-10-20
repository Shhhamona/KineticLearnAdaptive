"""
Adaptive Sampling Analysis Script

This script demonstrates the AdaptiveSamplingPipeline which implements
iterative window-based sampling inspired by K-centered adaptive learning.

The pipeline:
1. Starts with a small initial training set
2. At each iteration, samples from a window around the average point
3. Window shrinks each iteration to focus sampling
4. Incrementally adds samples and retrains the model
5. Uses multiple seeds for robustness

Usage:
    python examples/run_adaptive_sampling.py
"""

from pathlib import Path
import sys
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
    initial_training_size = 0
    initial_window_size = 0.5  # ±10% around center
    shrink_rate = 1  # 20% reduction each iteration
    num_seeds = 5
    
    # SVR hyperparameters
    svr_params = [
        {'C': 10, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'},
        {'C': 20, 'epsilon': 0.005, 'gamma': 5, 'kernel': 'rbf'},
        {'C': 5, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'}
    ]
    
    print("="*70)
    print("Adaptive Sampling Analysis")
    print("="*70)
    print(f"Initial training size: {initial_training_size}")
    print(f"Iterations: {n_iterations}")
    print(f"Samples per iteration: {samples_per_iteration}")
    print(f"Initial window size: {initial_window_size}")
    print(f"Shrink rate: {shrink_rate}")
    print(f"Number of seeds: {num_seeds}")
    print("="*70)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    pool_file = Path("data/SampleEfficiency/O2_simple_uniform.txt")
    #pool_file = Path("results/batch_simulations/lokisimulator/boundsbasedsampler/2025-08-27/batch_4000sims_20250827_010028.json")
    test_file = Path("data/SampleEfficiency/O2_simple_test.txt")
    test_file = Path("data/SampleEfficiency/O2_simple_test_real_K.txt")


    if not pool_file.exists():
        raise FileNotFoundError(f"Pool file not found: {pool_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    # Load full pool dataset
    pool_dataset = MultiPressureDataset(
        nspecies=nspecies,
        num_pressure_conditions=num_pressure_conditions,
        src_file=str(pool_file),
        react_idx=react_idx
    )
    
    # Create initial training dataset (subset of pool)
    pool_x, pool_y = pool_dataset.get_data()
    initial_x = pool_x[:initial_training_size]
    initial_y = pool_y[:initial_training_size]
    
    initial_dataset = MultiPressureDataset(
        nspecies=nspecies,
        num_pressure_conditions=num_pressure_conditions,
        processed_x=initial_x,
        processed_y=initial_y,
        scaler_input=pool_dataset.scaler_input,
        scaler_output=pool_dataset.scaler_output
    )
    
    # Load test dataset with same scalers
    test_dataset = MultiPressureDataset(
        nspecies=nspecies,
        num_pressure_conditions=num_pressure_conditions,
        src_file=str(test_file),
        react_idx=react_idx,
        scaler_input=pool_dataset.scaler_input,
        scaler_output=pool_dataset.scaler_output
    )
    
    print(f"✓ Initial training: {len(initial_dataset)} samples")
    print(f"✓ Pool dataset: {len(pool_dataset)} samples")
    print(f"✓ Test dataset: {len(test_dataset)} samples")
    
    # Setup model parameters
    model_params = {
        'params': svr_params,
        'model_name': 'adaptive_sampling_svr'
    }
    
    # Create and run pipeline
    print("\n🚀 Creating adaptive sampling pipeline...")
    pipeline = AdaptiveSamplingPipeline(
        initial_dataset=initial_dataset,
        pool_dataset=pool_dataset,
        test_dataset=test_dataset,
        model_class=SVRModel,
        model_params=model_params,
        n_iterations=n_iterations,
        samples_per_iteration=samples_per_iteration,
        initial_window_size=initial_window_size,
        shrink_rate=shrink_rate,
        num_seeds=num_seeds,
        window_type='input',  # Sample based on output (K values)
        pipeline_name=f"adaptive_sampling_w{initial_window_size}_s{shrink_rate}"
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
