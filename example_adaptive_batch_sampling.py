"""
Example script demonstrating the use of AdaptiveBatchSamplingPipeline.

This script shows how to:
1. Load multiple pool datasets
2. Configure a Neural Network model
3. Run adaptive batch sampling with continuous NN training
4. Save and visualize results
"""

import numpy as np
from pathlib import Path

# Import pipeline and model
from kinetic_modelling.pipeline import AdaptiveBatchSamplingPipeline
from kinetic_modelling.model import NeuralNetModel
from kinetic_modelling.data import MultiPressureDataset


def example_adaptive_batch_sampling():
    """
    Example of running adaptive batch sampling with Neural Network.
    """
    
    # Example configuration - ADJUST THESE FOR YOUR ACTUAL DATA
    
    # 1. Load your pool datasets (multiple files)
    # These would be your actual data files
    pool_files = [
        "data/pool_dataset_1.txt",
        "data/pool_dataset_2.txt", 
        "data/pool_dataset_3.txt"
    ]
    
    # You would load these using your data loading code
    # For example:
    # pool_datasets = []
    # for file in pool_files:
    #     pool_ds = MultiPressureDataset.from_file(file, ...)
    #     pool_datasets.append(pool_ds)
    
    # 2. Load test dataset
    # test_dataset = MultiPressureDataset.from_file("data/test_data.txt", ...)
    
    # 3. Configure Neural Network parameters
    model_params = {
        'input_size': 10,  # Number of input features
        'output_size': 5,  # Number of output features
        'hidden_sizes': (64, 32),  # Hidden layer sizes
        'activation': 'tanh',
        'learning_rate': 0.001
    }
    
    # 4. Create and run pipeline
    pipeline = AdaptiveBatchSamplingPipeline(
        pool_datasets=pool_datasets,  # List of pool datasets
        test_dataset=test_dataset,
        model_class=NeuralNetModel,
        model_params=model_params,
        n_iterations=10,  # Number of iterations (one per pool file)
        samples_per_iteration=200,  # Samples to grab from each pool
        n_epochs=10,  # Train for 10 epochs at each iteration
        batch_size=64,  # Batch size for training
        initial_window_size=0.3,  # Start with ±30% window
        shrink_rate=0.8,  # Reduce window by 20% each iteration
        num_seeds=5,  # Run with 5 different seeds
        window_type='output',  # Sample based on output space
        pipeline_name="example_adaptive_batch_sampling",
        results_dir="pipeline_results"
    )
    
    # 5. Run pipeline and save results
    results = pipeline.save_and_return(save_results=True)
    
    # 6. Access results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    aggregated = results['aggregated_results']
    
    print(f"\nNumber of iterations completed: {len(aggregated)}")
    print(f"\nInitial MSE: {aggregated[0]['mean_total_mse']:.6e}")
    print(f"Final MSE: {aggregated[-1]['mean_total_mse']:.6e}")
    
    improvement = aggregated[0]['mean_total_mse'] / aggregated[-1]['mean_total_mse']
    print(f"Improvement factor: {improvement:.2f}x")
    
    print(f"\nTotal samples seen: {aggregated[-1]['total_samples_seen']:.0f}")
    
    return results


def minimal_working_example():
    """
    Minimal working example with dummy data for testing.
    """
    print("Creating dummy data for demonstration...")
    
    # Create dummy datasets
    n_samples_pool = 500
    n_samples_test = 100
    n_features_in = 10
    n_features_out = 5
    
    # Create 3 pool datasets
    pool_datasets = []
    for i in range(3):
        x_pool = np.random.randn(n_samples_pool, n_features_in)
        y_pool = np.random.randn(n_samples_pool, n_features_out)
        
        pool_ds = MultiPressureDataset(
            nspecies=2,
            num_pressure_conditions=1,
            processed_x=x_pool,
            processed_y=y_pool,
            scaler_input=None,
            scaler_output=None
        )
        pool_datasets.append(pool_ds)
    
    # Create test dataset
    x_test = np.random.randn(n_samples_test, n_features_in)
    y_test = np.random.randn(n_samples_test, n_features_out)
    
    test_dataset = MultiPressureDataset(
        nspecies=2,
        num_pressure_conditions=1,
        processed_x=x_test,
        processed_y=y_test,
        scaler_input=None,
        scaler_output=None
    )
    
    print("✓ Dummy data created")
    print(f"  Pool datasets: {len(pool_datasets)}")
    print(f"  Samples per pool: {n_samples_pool}")
    print(f"  Test samples: {n_samples_test}")
    
    # Configure model
    model_params = {
        'input_size': n_features_in,
        'output_size': n_features_out,
        'hidden_sizes': (32, 16),
        'activation': 'tanh',
        'learning_rate': 0.001
    }
    
    # Create pipeline
    pipeline = AdaptiveBatchSamplingPipeline(
        pool_datasets=pool_datasets,
        test_dataset=test_dataset,
        model_class=NeuralNetModel,
        model_params=model_params,
        n_iterations=3,  # Use all 3 pools
        samples_per_iteration=100,
        n_epochs=5,
        batch_size=32,
        initial_window_size=0.5,
        shrink_rate=0.8,
        num_seeds=2,  # Just 2 seeds for quick demo
        window_type='output',
        pipeline_name="demo_adaptive_batch_sampling",
        results_dir="pipeline_results"
    )
    
    # Run pipeline
    print("\nRunning pipeline...")
    results = pipeline.save_and_return(save_results=True)
    
    print("\n✓ Pipeline completed successfully!")
    
    return results


if __name__ == "__main__":
    print("="*70)
    print("ADAPTIVE BATCH SAMPLING PIPELINE - DEMO")
    print("="*70)
    print("\nThis is a minimal working example with dummy data.")
    print("Replace with your actual data for real experiments.\n")
    
    # Run minimal example
    results = minimal_working_example()
    
    print("\n" + "="*70)
    print("To use with your real data, modify the example_adaptive_batch_sampling()")
    print("function and uncomment the data loading code.")
    print("="*70)
