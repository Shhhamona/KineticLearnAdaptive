"""
Sample Efficiency Analysis Script

This script demonstrates the StandardSubsetPipeline for analyzing how model
performance scales with dataset size, following the methodology from sample_effiency.py.

Usage:
    python examples/run_subset_pipeline.py
"""

from pathlib import Path
import sys
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetic_modelling import (
    MultiPressureDataset,
    SVRModel,
    StandardSubsetPipeline
)


def main():
    # Configuration matching sample_effiency.py
    nspecies = 3
    num_pressure_conditions = 2
    react_idx = [0, 1, 2]
    
    # Subset sizes to evaluate
    subset_sizes = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    
    # Number of seeds for statistical robustness
    num_seeds = 10
    
    # Best hyperparameters from sample_effiency.py grid search
    best_params = [
        {'C': 10, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'},
        {'C': 20, 'epsilon': 0.005, 'gamma': 5, 'kernel': 'rbf'},
        {'C': 5, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'}
    ]
    
    print("="*70)
    print("Sample Efficiency Analysis with StandardSubsetPipeline")
    print("="*70)
    print(f"Subset sizes: {subset_sizes}")
    print(f"Number of seeds: {num_seeds}")
    print(f"Outputs per model: {len(best_params)}")
    print("="*70)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    #train_file = Path("data/SampleEfficiency/O2_simple_uniform.txt")
    train_file = Path("data/SampleEfficiency/O2_simple_latin.txt")
    #test_file = Path("data/SampleEfficiency/O2_simple_test.txt")
    test_file = Path("data/SampleEfficiency/O2_simple_test_real_K.txt")
    
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
    
    print(f"✓ Training dataset: {len(train_dataset)} samples")
    print(f"✓ Test dataset: {len(test_dataset)} samples")
    
    # Create pipeline
    print("\n🔧 Creating StandardSubsetPipeline...")
    pipeline = StandardSubsetPipeline(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_class=SVRModel,
        model_params={'params': best_params},
        subset_sizes=subset_sizes,
        num_seeds=num_seeds,
        pipeline_name="sample_efficiency_uniform",
        results_dir="pipeline_results"
    )
    
    # Run pipeline
    print("\n🚀 Running pipeline...")
    results = pipeline.save_and_return(save_results=True)
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    agg = results['aggregated_results']
    print(f"Final subset size: {agg['subset_sizes'][-1]}")
    print(f"Final Mean Total MSE: {agg['final_mean_mse']:.6e}")
    print(f"Final Mean Total RMSE: {agg['final_mean_rmse']:.6e}")
    print(f"Final Std Error (RMSE): {agg['final_std_rmse']:.6e}")
    
    print("\nPer-output MSE at final size:")
    for i, (mean, std) in enumerate(zip(
        agg['mean_mse_per_output'][-1],
        agg['std_mse_per_output'][-1]
    )):
        print(f"  K{i+1}: {mean:.6e} ± {std:.6e}")
    
    print("\nPer-output RMSE at final size:")
    for i, (mean, std) in enumerate(zip(
        agg['mean_rmse_per_output'][-1],
        agg['std_rmse_per_output'][-1]
    )):
        print(f"  K{i+1}: {mean:.6e} ± {std:.6e}")
    
    print("\nPer-output Relative Error (%) at final size:")
    for i, (mean, std) in enumerate(zip(
        agg['mean_rel_error_per_output'][-1],
        agg['std_rel_error_per_output'][-1]
    )):
        print(f"  K{i+1}: {mean:.2f}% ± {std:.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()
