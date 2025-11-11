"""
Plot Adaptive Batch Sampling Results

This script loads saved results from AdaptiveBatchSamplingPipeline runs and creates
plots showing MSE evolution across iterations.

Usage:
    python pipeline_plots/plot_adaptive_batch_sampling_results.py
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def load_pipeline_results(json_path):
    """Load results from a saved pipeline JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_adaptive_batch_sampling_results(result_file, output_dir='pipeline_results/plots'):
    """
    Create plots for adaptive batch sampling results:
    1. Total MSE vs Training Samples
    2. MSE per output (K values) vs Training Samples
    
    Args:
        result_file: Path to JSON result file
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = Path(result_file)
    if not result_path.exists():
        print(f"⚠️  File not found: {result_file}")
        return
    
    print("="*70)
    print("Loading and Plotting Adaptive Batch Sampling Results")
    print("="*70)
    print(f"\n📊 Loading: {result_path.name}")
    
    results = load_pipeline_results(result_path)
    
    # Extract configuration
    config = results['config']
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    window_type = config['window_type']
    initial_window = config['initial_window_size']
    shrink_rate = config['shrink_rate']
    num_pool_datasets = config['num_pool_datasets']
    samples_per_iteration = config['samples_per_iteration']
    
    print(f"\nConfiguration:")
    print(f"  Epochs per iteration: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Window type: {window_type}")
    print(f"  Initial window: {initial_window}")
    print(f"  Shrink rate: {shrink_rate}")
    print(f"  Number of pool files: {num_pool_datasets}")
    print(f"  Samples per iteration: {samples_per_iteration}")
    
    # Extract aggregated results
    agg_results = results['aggregated_results']
    
    iterations = []
    training_samples = []
    total_samples_seen = []
    mean_total_mse = []
    std_total_mse = []
    mean_mse_per_output = []
    std_mse_per_output = []
    window_sizes = []
    
    cumulative_training_samples = 0
    
    for result in agg_results:
        iterations.append(result['iteration'])
        
        # Calculate training_samples for this iteration
        if 'training_samples' in result:
            iter_train_samples = result['training_samples']
        else:
            # For old format: calculate training_samples = samples_added * n_epochs
            # Iteration 0 has no training
            iter_train_samples = 0 if result['iteration'] == 0 else result['samples_added'] * n_epochs
        
        # CUMULATIVE training samples (model keeps learning)
        cumulative_training_samples += iter_train_samples
        training_samples.append(cumulative_training_samples)
        
        total_samples_seen.append(result['total_samples_seen'])
        mean_total_mse.append(result['mean_total_mse'])
        std_total_mse.append(result['std_total_mse'])
        mean_mse_per_output.append(result['mean_mse_per_output'])
        std_mse_per_output.append(result['std_mse_per_output'])
        window_sizes.append(result['window_size'])
    
    # Convert to numpy arrays
    iterations = np.array(iterations)
    training_samples = np.array(training_samples)
    total_samples_seen = np.array(total_samples_seen)
    mean_total_mse = np.array(mean_total_mse)
    std_total_mse = np.array(std_total_mse)
    mean_mse_per_output = np.array(mean_mse_per_output)  # Shape: (n_iterations, n_outputs)
    std_mse_per_output = np.array(std_mse_per_output)
    window_sizes = np.array(window_sizes)
    
    n_outputs = mean_mse_per_output.shape[1]
    
    # Extract prediction variance if available
    has_variance = 'mean_prediction_variance_per_output' in agg_results[0]
    if has_variance:
        mean_prediction_variance_per_output = []
        std_prediction_variance_per_output = []
        for result in agg_results:
            mean_prediction_variance_per_output.append(result['mean_prediction_variance_per_output'])
            std_prediction_variance_per_output.append(result['std_prediction_variance_per_output'])
        
        mean_prediction_variance_per_output = np.array(mean_prediction_variance_per_output)
        std_prediction_variance_per_output = np.array(std_prediction_variance_per_output)
    
    print(f"\nResults Summary:")
    print(f"  Number of iterations: {len(iterations)}")
    print(f"  Number of outputs: {n_outputs}")
    print(f"  Initial MSE: {mean_total_mse[0]:.6e} ± {std_total_mse[0]:.6e}")
    print(f"  Final MSE: {mean_total_mse[-1]:.6e} ± {std_total_mse[-1]:.6e}")
    print(f"  Improvement: {mean_total_mse[0] / mean_total_mse[-1]:.2f}x")
    print(f"  Total training samples: {training_samples[-1]:.0f}")
    if has_variance:
        print(f"  Prediction variance tracking: Yes (window size = 5 iterations)")
    
    # Create figure with 2 or 3 subplots depending on variance availability
    if has_variance:
        fig = plt.figure(figsize=(20, 6))
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create main title for the entire figure with dataset info
    dataset_info = f"{num_pool_datasets} pool files × {samples_per_iteration} samples/file"
    fig.suptitle(f'Neural Network Training - Adaptive Batch Sampling\nUniform Sampling with Varying K Range (Training Data: {dataset_info})', 
                 fontsize=15, fontweight='bold', y=1.00)
    
    # ============================================================
    # Plot 1: Total MSE vs Training Samples
    # ============================================================
    ax1.errorbar(training_samples, mean_total_mse, yerr=std_total_mse,
                 marker='o', linewidth=2, markersize=6, capsize=5, 
                 label='Adaptive Batch Sampling', alpha=0.8, color='#2E86AB')
    
    ax1.set_xlabel(f'Training Samples (Total Epochs = {n_epochs}, Batch Size = {batch_size})', fontsize=13)
    ax1.set_ylabel('Total MSE (Sum across outputs)', fontsize=13)
    ax1.set_title('Learning Curve', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')
    
    # ============================================================
    # Plot 2: MSE per Output (K values) vs Training Samples
    # ============================================================
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Matplotlib default colors
    output_labels = [f'K{i+1}' for i in range(n_outputs)]
    
    for i in range(n_outputs):
        ax2.errorbar(training_samples, mean_mse_per_output[:, i], 
                     yerr=std_mse_per_output[:, i],
                     marker='o', linewidth=2, markersize=6, capsize=5,
                     label=output_labels[i], alpha=0.8, color=colors[i % len(colors)])
    
    ax2.set_xlabel(f'Training Samples (Total Epochs = {n_epochs}, Batch Size = {batch_size})', fontsize=13)
    ax2.set_ylabel('MSE per Output', fontsize=13)
    ax2.set_title('MSE per K Value', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')
    
    # ============================================================
    # Plot 3: Prediction Variance per Output (if available)
    # ============================================================
    if has_variance:
        for i in range(n_outputs):
            # Filter out NaN values
            valid_mask = ~np.isnan(mean_prediction_variance_per_output[:, i])
            valid_iterations = training_samples[valid_mask]
            valid_variance = mean_prediction_variance_per_output[valid_mask, i]
            valid_std = std_prediction_variance_per_output[valid_mask, i]
            
            if len(valid_iterations) > 0:
                ax3.errorbar(valid_iterations, valid_variance, 
                            yerr=valid_std,
                            marker='o', linewidth=2, markersize=6, capsize=5,
                            label=output_labels[i], alpha=0.8, color=colors[i % len(colors)])
        
        ax3.set_xlabel(f'Training Samples (Total Epochs = {n_epochs}, Batch Size = {batch_size})', fontsize=13)
        ax3.set_ylabel('Variance of Mean Predictions (Last 5 Iterations)', fontsize=13)
        ax3.set_title('Prediction Variance (Convergence Indicator)', fontsize=14, fontweight='bold')
        ax3.set_yscale('log')
        ax3.legend(loc='best', fontsize=11)
        ax3.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

    # Save plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save to pipeline_results/plots/ directory
    plot_pdf_path = output_dir / f'adaptive_batch_sampling_{timestamp}.pdf'
    plt.savefig(plot_pdf_path, bbox_inches='tight')
    print(f"\n📊 Saved plots to: {plot_pdf_path}")
    
    plot_png_path = output_dir / f'adaptive_batch_sampling_{timestamp}.png'
    plt.savefig(plot_png_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved PNG to: {plot_png_path}")
    
    plt.show()
    
    # ============================================================
    # Print detailed results table
    # ============================================================
    print("\n" + "="*100)
    print("Detailed Results by Iteration")
    print("="*100)
    print(f"{'Iter':<6} {'Seen':<10} {'Train#':<12} {'Total MSE':<15} {'Std Err':<12} {'Window':<10}", end='')
    for i in range(n_outputs):
        print(f"  {output_labels[i]:<12}", end='')
    print()
    print("-"*100)
    
    for idx in range(len(iterations)):
        print(f"{iterations[idx]:<6} "
              f"{total_samples_seen[idx]:<10.0f} "
              f"{training_samples[idx]:<12.0f} "
              f"{mean_total_mse[idx]:<15.6e} "
              f"{std_total_mse[idx]:<12.6e} "
              f"{window_sizes[idx]:<10.4f}", end='')
        for i in range(n_outputs):
            print(f"  {mean_mse_per_output[idx, i]:<12.6e}", end='')
        print()
    
    print("="*100)


def main():
    """Main function to run the plotting script."""
    
    # Path to the adaptive batch sampling results file
    # Update this path to your actual results file
    result_file = 'pipeline_results/adaptive_batch_sampling_w1.0_s0.1_e25_20251028_185800.json'
    result_file = 'pipeline_results/adaptive_batch_sampling_w1.0_s0.5_e100_20251106_172628.json'
    
    if not Path(result_file).exists():
        print(f"⚠️  Result file not found: {result_file}")
        print("\nLooking for available result files in pipeline_results/...")
        
        results_dir = Path('pipeline_results')
        if results_dir.exists():
            json_files = list(results_dir.glob('adaptive_batch_sampling*.json'))
            if json_files:
                print(f"\nFound {len(json_files)} adaptive batch sampling result file(s):")
                for idx, file in enumerate(sorted(json_files, reverse=True)):
                    print(f"  {idx+1}. {file.name}")
                
                # Use the most recent file
                result_file = json_files[0]
                print(f"\n✓ Using most recent file: {result_file.name}")
            else:
                print("No adaptive_batch_sampling*.json files found.")
                return
        else:
            print("pipeline_results/ directory not found.")
            return
    
    # Create plots
    plot_adaptive_batch_sampling_results(result_file)


if __name__ == "__main__":
    main()
