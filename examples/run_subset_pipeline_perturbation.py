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

import numpy as np
import matplotlib.pyplot as plt

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
    subset_sizes = [50, 75, 100, 150,  200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    
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
    train_file = Path("data/SampleEfficiency/O2_simple_latin.txt")
    #train_file= Path("results/batch_simulations/lokisimulator/boundsbasedsampler/2025-08-27/batch_4000sims_20250827_010028.json")

    #train_file = Path("data/SampleEfficiency/O2_simple_latin.txt")
    #test_file = Path("data/SampleEfficiency/O2_simple_test.txt")
    test_file = Path("data/SampleEfficiency/O2_simple_test_real_K.txt")

    files = [
        'O2_simple_uniform.txt',
        'O2_simple_latin_log_uniform.txt',
        'O2_simple_latin.txt',
        'O2_simple_morris.txt',
        'O2_simple__morris_continous_discret_corrected.txt',
        'O2_simple__morris_continous_final.txt',
    ]
    
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
    
    # Get the saved JSON path from the pipeline
    saved_json_path = pipeline.json_path if hasattr(pipeline, 'json_path') else None
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    agg = results['aggregated_results']
    print(f"Final subset size: {agg['subset_sizes'][-1]}")
    print(f"Final Mean Total MSE: {agg['final_mean_mse']:.6e}")
    print(f"Final Std Error: {agg['final_std_mse']:.6e}")
    print("\nPer-output MSE at final size:")
    for i, (mean, std) in enumerate(zip(
        agg['mean_mse_per_output'][-1],
        agg['std_mse_per_output'][-1]
    )):
        print(f"  Output {i}: {mean:.6e} ± {std:.6e}")
    print("="*70)
    
    # ========== PERTURBATION ANALYSIS ==========
    print("\n" + "="*70)
    print("PERTURBATION ANALYSIS")
    print("="*70)
    
    RELATIVE_ERROR = 1e-3
    
    # Get test data (already scaled)
    x_test, y_test = test_dataset.get_data()
    n_test = len(x_test)
    print(f"Test samples: {n_test}")
    print(f"Relative error: ±{RELATIVE_ERROR*100:.3f}%")
    
    # For each subset size, predict with perturbed test inputs
    all_models = pipeline.all_trained_models
    
    perturbation_results = []
    
    for idx, n_samples in enumerate(subset_sizes):
        print(f"\n[{idx+1}/{len(subset_sizes)}] n={n_samples}")
        models_for_size = all_models[idx]
        
        # For each test sample
        for s in range(n_test):
            x_sample = x_test[s:s+1]
            y_true = y_test[s]  # Already scaled outputs from test dataset
            
            # Get mean prediction at nominal input (no perturbation)
            mean_preds = []
            for model_wrapper in models_for_size:
                svr_models = model_wrapper.models
                pred_nominal = np.array([svr.predict(x_sample)[0] for svr in svr_models])
                mean_preds.append(pred_nominal)
            y_pred_mean = np.mean(mean_preds, axis=0)
            
            # Create lower/upper perturbed versions (±1e-3)
            x_lower = x_sample * (1 - RELATIVE_ERROR)
            x_upper = x_sample * (1 + RELATIVE_ERROR)
            
            # Collect predictions from all seeds
            all_preds = []
            for model_wrapper in models_for_size:
                svr_models = model_wrapper.models
                
                # Predict for lower/upper (outputs are in scaled space)
                pred_lower = np.array([svr.predict(x_lower)[0] for svr in svr_models])
                pred_upper = np.array([svr.predict(x_upper)[0] for svr in svr_models])
                
                all_preds.append(pred_lower)
                all_preds.append(pred_upper)
            
            # Min/max across all seeds and both perturbations (still in scaled space)
            all_preds = np.array(all_preds)  # (2*n_seeds, n_outputs)
            y_min = np.min(all_preds, axis=0)
            y_max = np.max(all_preds, axis=0)
            
            # Check containment (all in same scaled space)
            containment = (y_true >= y_min) & (y_true <= y_max)
            
            print(f"  Sample {s}:")
            for i in range(len(y_true)):
                status = '✓' if containment[i] else '✗'
                print(f"    Output{i+1}: pred={y_pred_mean[i]:.4e} [{y_min[i]:.4e}, {y_max[i]:.4e}] true={y_true[i]:.4e} {status}")
            
            perturbation_results.append({
                'n_samples': int(n_samples),
                'test_idx': int(s),
                'y_pred_mean': y_pred_mean.tolist() if isinstance(y_pred_mean, np.ndarray) else y_pred_mean,
                'y_min': y_min.tolist() if isinstance(y_min, np.ndarray) else y_min,
                'y_max': y_max.tolist() if isinstance(y_max, np.ndarray) else y_max,
                'y_true': y_true.tolist() if isinstance(y_true, np.ndarray) else y_true,
                'containment': containment.tolist() if isinstance(containment, np.ndarray) else containment
            })
    
    print("\n" + "="*70)
    print("CONTAINMENT SUMMARY")
    print("="*70)
    for n_samples in subset_sizes:
        results_for_n = [r for r in perturbation_results if r['n_samples'] == n_samples]
        frac = np.mean([np.mean(r['containment']) for r in results_for_n]) * 100
        print(f"  n={n_samples:>4}: {frac:>5.1f}% contained")
    print("="*70)
    
    # Save perturbation results to a separate JSON file
    import json
    from datetime import datetime
    
    results_dir = Path("pipeline_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    perturbation_json_path = results_dir / f"perturbation_results_{timestamp}.json"
    
    # Create perturbation results dictionary
    perturbation_output = {
        'pipeline_name': 'perturbation_analysis',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'relative_error': RELATIVE_ERROR,
            'n_test_samples': n_test,
            'subset_sizes': subset_sizes,
            'num_seeds': num_seeds,
            'n_outputs': len(perturbation_results[0]['y_true'])
        },
        'perturbation_results': perturbation_results,
        'summary': {
            n_samples: {
                'containment_percentage': np.mean([np.mean(r['containment']) for r in perturbation_results if r['n_samples'] == n_samples]) * 100
            }
            for n_samples in subset_sizes
        }
    }
    
    # Save perturbation results
    with open(perturbation_json_path, 'w') as f:
        json.dump(perturbation_output, f, indent=2)
    
    print(f"\n💾 Saved perturbation results to: {perturbation_json_path}")


if __name__ == "__main__":
    main()
