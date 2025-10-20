"""
Batch Training Analysis Script - Optimized Version

This demonstrates efficient batch training with evaluation frequency control.

Usage:
    python examples/run_batch_training_optimized.py
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetic_modelling import (
    MultiPressureDataset,
    NeuralNetModel,
    BatchTrainingPipeline
)


def main():
    # Configuration
    nspecies = 3
    num_pressure_conditions = 2
    react_idx = [0, 1, 2]
    
    # Training settings
    batch_size = 64
    num_epochs = 500      # Train for 100 epochs
    num_seeds = 5         # 5 seeds for robustness
    eval_frequency = 400   # Evaluate every 10 batches (not every batch!)
    
    # Neural network hyperparameters
    hidden_layers = [64, 32]
    learning_rate = 0.001
    
    print("="*70)
    print("Optimized Batch Training Analysis with Neural Networks")
    print("="*70)
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Evaluation frequency: every {eval_frequency} batches")
    print(f"Number of seeds: {num_seeds}")
    print(f"Hidden layers: {hidden_layers}")
    print(f"Learning rate: {learning_rate}")
    print("="*70)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_file = Path("data/SampleEfficiency/O2_simple_uniform.txt")
    test_file = Path("data/SampleEfficiency/O2_simple_test.txt")
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    train_dataset = MultiPressureDataset(
        nspecies=nspecies,
        num_pressure_conditions=num_pressure_conditions,
        src_file=str(train_file),
        react_idx=react_idx
    )
    
    test_dataset = MultiPressureDataset(
        nspecies=nspecies,
        num_pressure_conditions=num_pressure_conditions,
        src_file=str(test_file),
        react_idx=react_idx,
        scaler_input=train_dataset.scaler_input,
        scaler_output=train_dataset.scaler_output
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Get data shapes
    x_train, y_train = train_dataset.get_data()
    print(f"✓ Input features: {x_train.shape[1]}")
    print(f"✓ Output features: {y_train.shape[1]}")
    
    # Calculate batches and evaluations
    num_batches_per_epoch = (len(train_dataset) + batch_size - 1) // batch_size
    total_batches = num_batches_per_epoch * num_epochs
    num_evaluations = total_batches // eval_frequency
    
    print(f"✓ Batches per epoch: {num_batches_per_epoch}")
    print(f"✓ Total batches: {total_batches}")
    print(f"✓ Number of evaluations: {num_evaluations}")
    print(f"   (vs {total_batches} if evaluating every batch)")
    
    # Setup model parameters
    model_params = {
        'input_size': x_train.shape[1],
        'output_size': y_train.shape[1],
        'hidden_sizes': tuple(hidden_layers),
        'learning_rate': learning_rate,
        'model_name': 'batch_training_nn'
    }
    
    # Create and run pipeline
    print("\n🚀 Creating pipeline...")
    pipeline = BatchTrainingPipeline(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_class=NeuralNetModel,
        model_params=model_params,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_seeds=num_seeds,
        eval_frequency=eval_frequency,  # KEY: Evaluate less frequently!
        pipeline_name=f"batch_training_bs{batch_size}_ef{eval_frequency}"
    )
    
    print("\n🔄 Running batch training pipeline...")
    results = pipeline.save_and_return(save_results=True)
    
    # Display key results
    print("\n" + "="*70)
    print("Key Results")
    print("="*70)
    agg = results['aggregated_results']
    print(f"Initial MSE: {agg['initial_mean_mse']:.6e} ± {agg['initial_std_mse']:.6e}")
    print(f"Final MSE:   {agg['final_mean_mse']:.6e} ± {agg['final_std_mse']:.6e}")
    
    improvement = agg['initial_mean_mse'] / agg['final_mean_mse']
    print(f"Improvement: {improvement:.2f}x better")
    print("="*70)


if __name__ == "__main__":
    main()
