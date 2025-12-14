"""
Plot Batch Training Results

This script loads saved results from BatchTrainingPipeline runs and creates
plots comparing different sampling methods' performance.

Usage:
    python pipeline_plots/plot_batch_training_results.py
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


def extract_sample_counts(results):
    """
    Extract effective sample counts from batch numbers and batch size.
    
    Args:
        results: Loaded JSON results dictionary
        
    Returns:
        Array of sample counts corresponding to evaluation points
    """
    batch_size = results['config']['batch_size']
    batch_numbers = results['aggregated_results']['batch_numbers']
    
    # Approximate samples seen = batch_number * batch_size
    # (batch_number 0 is before training, so samples = 0)
    sample_counts = [batch_num * batch_size for batch_num in batch_numbers]
    
    return np.array(sample_counts)


def plot_mse_vs_samples(results_files, labels=None, output_dir='pipeline_results/plots'):
    """
    Create comparison plot of MSE vs number of training samples across different sampling methods.
    
    Args:
        results_files: List of paths to JSON result files
        labels: List of labels for each result file (optional)
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if labels is None:
        labels = [f"Method {i+1}" for i in range(len(results_files))]
    
    # Create single figure with MSE in log scale
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    all_results = []
    
    print("="*70)
    print("Loading and Plotting Batch Training Results")
    print("="*70)
    
    for idx, (result_file, label) in enumerate(zip(results_files, labels)):
        result_path = Path(result_file)
        if not result_path.exists():
            print(f"⚠️  File not found: {result_file}")
            continue
            
        print(f"\n📊 Loading: {result_path.name}")
        results = load_pipeline_results(result_path)
        
        # Extract data
        agg = results['aggregated_results']
        sample_counts = extract_sample_counts(results)
        total_samples = results['config']['total_train_samples']
        batch_size = results['config']['batch_size']  # Get batch size from config
        
        mean_total_mse = np.array(agg['mean_total_mse'])
        std_total_mse = np.array(agg['std_total_mse'])
        
        all_results.append({
            'label': label,
            'sample_counts': sample_counts,
            'mean_total_mse': mean_total_mse,
            'std_total_mse': std_total_mse,
            'total_samples': total_samples,
            'batch_size': batch_size
        })
        
        # Plot MSE with error bars (log scale)
        ax.errorbar(sample_counts, mean_total_mse, yerr=std_total_mse, 
                    label=label, marker='o', linewidth=2, markersize=6, capsize=5, alpha=0.8)
        
        # Print summary
        initial_mse = agg['initial_mean_mse']
        initial_std = agg['initial_std_mse']
        final_mse = agg['final_mean_mse']
        final_std = agg['final_std_mse']
        improvement = initial_mse / final_mse
        
        print(f"   Dataset: {total_samples} samples")
        print(f"   Batch size: {batch_size}")
        print(f"   Initial MSE: {initial_mse:.6e} ± {initial_std:.6e}")
        print(f"   Final MSE:   {final_mse:.6e} ± {final_std:.6e}")
        print(f"   Improvement: {improvement:.2f}x")
    
    # Get batch size for x-axis label (assuming all use same batch size)
    batch_size_label = all_results[0]['batch_size'] if all_results else 64
    num_epochs = results['config']['num_epochs'] if 'num_epochs' in results['config'] else 1
    train_dataset_size = all_results[0]['total_samples'] if all_results else 2000
    
    # Configure plot
    ax.set_xlabel(f'Training Samples (Total Epochs = {num_epochs}, Batch Size = {batch_size_label})', fontsize=13)
    ax.set_ylabel('Total MSE (Sum across outputs)', fontsize=13)
    ax.set_title(f'Neural Network Training - Uniform Sampling with Varying K Range\nLearning Curves (Training Dataset Size = {train_dataset_size})', 
                 fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = output_dir / f'batch_training_comparison_{timestamp}.pdf'
    plt.savefig(plot_path)
    print(f"\n📊 Saved comparison plot to: {plot_path}")
    
    plot_png_path = output_dir / f'batch_training_comparison_{timestamp}.png'
    plt.savefig(plot_png_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved PNG to: {plot_png_path}")
    
    plt.show()
    
    return all_results


def plot_final_mse_comparison(results_files, labels=None, output_dir='pipeline_results/plots'):
    """
    Create bar plot comparing final MSE across different sampling methods.
    
    Args:
        results_files: List of paths to JSON result files
        labels: List of labels for each result file (optional)
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if labels is None:
        labels = [f"Method {i+1}" for i in range(len(results_files))]
    
    final_mse_values = []
    final_std_values = []
    valid_labels = []
    
    for result_file, label in zip(results_files, labels):
        result_path = Path(result_file)
        if not result_path.exists():
            continue
            
        results = load_pipeline_results(result_path)
        agg = results['aggregated_results']
        
        final_mse_values.append(agg['final_mean_mse'])
        final_std_values.append(agg['final_std_mse'])
        valid_labels.append(label)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x_pos = np.arange(len(valid_labels))
    bars = ax.bar(x_pos, final_mse_values, yerr=final_std_values, 
                   capsize=8, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Color bars
    colors = plt.cm.Set3(np.linspace(0, 1, len(valid_labels)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Sampling Method', fontsize=13, fontweight='bold')
    ax.set_ylabel('Final MSE (after training)', fontsize=13, fontweight='bold')
    ax.set_title('Final Model Performance: Comparison of Sampling Methods', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_labels, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mse, std) in enumerate(zip(final_mse_values, final_std_values)):
        ax.text(i, mse + std, f'{mse:.2e}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = output_dir / f'batch_training_final_mse_{timestamp}.pdf'
    plt.savefig(plot_path)
    print(f"📊 Saved final MSE comparison to: {plot_path}")
    
    plot_png_path = output_dir / f'batch_training_final_mse_{timestamp}.png'
    plt.savefig(plot_png_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved PNG to: {plot_png_path}")
    
    plt.show()


def plot_per_output_mse(results_files, labels=None, output_dir='pipeline_results/plots'):
    """
    Create plots showing MSE for each output (K1, K2, K3) separately across sampling methods.
    
    Args:
        results_files: List of paths to JSON result files
        labels: List of labels for each result file (optional)
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if labels is None:
        labels = [f"Method {i+1}" for i in range(len(results_files))]
    
    # Load first file to get number of outputs
    first_results = load_pipeline_results(results_files[0])
    n_outputs = len(first_results['aggregated_results']['mean_mse_per_output'][0])
    batch_size = first_results['config']['batch_size']
    num_epochs = first_results['config']['num_epochs'] if 'num_epochs' in first_results['config'] else 1
    
    # Create subplots
    fig, axes = plt.subplots(1, n_outputs, figsize=(18, 5))
    
    for idx, (result_file, label) in enumerate(zip(results_files, labels)):
        result_path = Path(result_file)
        if not result_path.exists():
            continue
            
        results = load_pipeline_results(result_path)
        
        # Extract data
        sample_counts = extract_sample_counts(results)
        mean_mse_per_output = np.array(results['aggregated_results']['mean_mse_per_output'])
        std_mse_per_output = np.array(results['aggregated_results']['std_mse_per_output'])
        
        # Plot each output
        for output_idx in range(n_outputs):
            ax = axes[output_idx] if n_outputs > 1 else axes
            
            mean_mse = mean_mse_per_output[:, output_idx]
            std_mse = std_mse_per_output[:, output_idx]
            
            ax.errorbar(sample_counts, mean_mse, yerr=std_mse,
                       label=label, marker='o', linewidth=2, markersize=5, 
                       capsize=4, alpha=0.8)
    
    # Configure subplots
    for output_idx in range(n_outputs):
        ax = axes[output_idx] if n_outputs > 1 else axes
        
        ax.set_xlabel(f'Training Samples (Total Epochs = {num_epochs}, Batch Size = {batch_size})', fontsize=11)
        ax.set_ylabel('MSE', fontsize=11)
        ax.set_title(f'K{output_idx+1} Prediction Error', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
    
    plt.suptitle('Per-Output Learning Curves', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = output_dir / f'batch_training_per_output_{timestamp}.pdf'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"📊 Saved per-output plot to: {plot_path}")
    
    plot_png_path = output_dir / f'batch_training_per_output_{timestamp}.png'
    plt.savefig(plot_png_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved PNG to: {plot_png_path}")
    
    plt.show()


def plot_prediction_variance(results_files, labels=None, output_dir='pipeline_results/plots'):
    """
    Create plots showing prediction variance over time (convergence indicator).
    
    Args:
        results_files: List of paths to JSON result files
        labels: List of labels for each result file (optional)
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if labels is None:
        labels = [f"Method {i+1}" for i in range(len(results_files))]
    
    # Load first file to get number of outputs
    first_results = load_pipeline_results(results_files[0])
    
    # Check if variance data exists
    if 'mean_prediction_variance_per_output' not in first_results['aggregated_results']:
        print("⚠️  Prediction variance data not found in results. Skipping variance plot.")
        return
    
    n_outputs = len(first_results['aggregated_results']['mean_prediction_variance_per_output'][0])
    batch_size = first_results['config']['batch_size']
    num_epochs = first_results['config']['num_epochs'] if 'num_epochs' in first_results['config'] else 1
    
    # Create subplots for each output
    fig, axes = plt.subplots(1, n_outputs, figsize=(18, 5))
    if n_outputs == 1:
        axes = [axes]
    
    for idx, (result_file, label) in enumerate(zip(results_files, labels)):
        result_path = Path(result_file)
        if not result_path.exists():
            continue
            
        results = load_pipeline_results(result_path)
        
        # Check if variance data exists for this file
        if 'mean_prediction_variance_per_output' not in results['aggregated_results']:
            continue
        
        # Extract data
        sample_counts = extract_sample_counts(results)
        mean_variance = np.array(results['aggregated_results']['mean_prediction_variance_per_output'])
        std_variance = np.array(results['aggregated_results']['std_prediction_variance_per_output'])
        
        # Plot each output
        for output_idx in range(n_outputs):
            ax = axes[output_idx]
            
            variance = mean_variance[:, output_idx]
            variance_std = std_variance[:, output_idx]
            
            # Filter out NaN values (from early batches)
            valid_mask = ~np.isnan(variance)
            valid_samples = sample_counts[valid_mask]
            valid_variance = variance[valid_mask]
            valid_std = variance_std[valid_mask]
            
            ax.errorbar(valid_samples, valid_variance, yerr=valid_std,
                       label=label, marker='o', linewidth=2, markersize=5, 
                       capsize=4, alpha=0.8)
    
    # Configure subplots
    for output_idx in range(n_outputs):
        ax = axes[output_idx]
        
        ax.set_xlabel(f'Training Samples (Total Epochs = {num_epochs}, Batch Size = {batch_size})', fontsize=11)
        ax.set_ylabel('Prediction Variance (moving window)', fontsize=11)
        ax.set_title(f'K{output_idx+1} Convergence', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
    
    plt.suptitle('Model Convergence - Prediction Variance Over Last 5 Batches', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = output_dir / f'batch_training_variance_{timestamp}.pdf'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"📊 Saved variance plot to: {plot_path}")
    
    plot_png_path = output_dir / f'batch_training_variance_{timestamp}.png'
    plt.savefig(plot_png_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved PNG to: {plot_png_path}")
    
    plt.show()


def main():
    """Main function to plot batch training results."""
    
    # Define the result files from your training runs
    results_dir = Path("pipeline_results")
    
    # Define label prefixes to search for
    label_prefixes = [
        'NN_batch_Uniform_',
        'NN_batch_Log-Uniform Latin Hypercube_',
        'NN_batch_Uniform Latin Hypercube_',
        'NN_batch_Morris Discret_',
        'NN_batch_Morris Continuous_',
    ]
    
    labels = [
        'Uniform Sampling',
        'Log-Uniform Latin Hypercube',
        'Uniform Latin Hypercube',
        'Morris Discrete',
        'Morris Continuous',
    ]

    """
    # Define label prefixes to search for
    label_prefixes = [
        'Uniform Sampling-K_factor_2',
        'Uniform Sampling-K_factor_1.15',
        'Uniform Sampling-K_factor_1.005',
        'Uniform Sampling-K_factor_1.0005',
        'Uniform Sampling-K_factor_1.00005',
    ]
    
    labels = [
        'K ∈ [K_true/2, K_true×2]',
        'K ∈ [K_true/1.15, K_true×1.15]',
        'K ∈ [K_true/1.005, K_true×1.005]',
        'K ∈ [K_true/1.0005, K_true×1.0005]',
        'K ∈ [K_true/1.00005, K_true×1.00005]',
    ]

        # Define label prefixes to search for
    label_prefixes = [
        'Uniform Sampling-K_factor_2',
        'Uniform Sampling-K_factor_1.15',
        'Uniform Sampling-K_factor_1.005',
        'Uniform Sampling-K_factor_1.00005',
    ]
    
    labels = [
        'K ∈ [K_true/2, K_true×2]',
        'K ∈ [K_true/1.15, K_true×1.15]',
        'K ∈ [K_true/1.005, K_true×1.005]',
        'K ∈ [K_true/1.00005, K_true×1.00005]',
    ]
    """
    # Find the latest file for each label prefix
    results_files = []
    for prefix in label_prefixes:
        matching_files = sorted(results_dir.glob(f'{prefix}*.json'))
        if matching_files:
            latest_file = matching_files[-1]  # Get the most recent one
            results_files.append(latest_file)
            print(f"✓ Found: {latest_file.name}")
        else:
            print(f"⚠️  No files found matching: {prefix}*.json")
    
    if not results_files:
        print("\n❌ No result files found!")
        print("   Please run run_batch_training.py first!")
        return
    
    # Filter labels to match found files
    existing_labels = [labels[i] for i in range(len(results_files))]
    
    print(f"\n✅ Found {len(results_files)} result files\n")
    
    # Create plots
    print("\n" + "="*70)
    print("PLOT 1: Learning Curves - MSE vs Training Samples")
    print("="*70)
    plot_mse_vs_samples(results_files, existing_labels)
    
    print("\n" + "="*70)
    print("PLOT 2: Final MSE Comparison")
    print("="*70)
    plot_final_mse_comparison(results_files, existing_labels)
    
    print("\n" + "="*70)
    print("PLOT 3: Per-Output Learning Curves")
    print("="*70)
    plot_per_output_mse(results_files, existing_labels)
    
    print("\n" + "="*70)
    print("PLOT 4: Model Convergence - Prediction Variance")
    print("="*70)
    plot_prediction_variance(results_files, existing_labels)
    
    print("\n" + "="*70)
    print("✅ All plots generated successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
