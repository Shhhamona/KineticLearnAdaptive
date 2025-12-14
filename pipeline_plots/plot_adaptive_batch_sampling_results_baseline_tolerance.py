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


def plot_adaptive_batch_sampling_results(result_files, output_dir='pipeline_results/plots', labels=None):
    """
    Create comparison plots for batch sampling results:
    1. Total MSE vs Samples Seen
    2. MSE per output (K values) vs Samples Seen
    
    Args:
        result_files: List of paths to JSON result files (or single path for backward compatibility)
        output_dir: Directory to save plots
        labels: List of labels for each result file (optional)
    """
    # Handle single file for backward compatibility
    if isinstance(result_files, (str, Path)):
        result_files = [result_files]
    
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(result_files))]
    elif len(labels) != len(result_files):
        raise ValueError(f"Number of labels ({len(labels)}) must match number of files ({len(result_files)})")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Loading and Plotting Batch Sampling Results - Comparison")
    print("="*70)
    
    # Store data for all files
    all_data = []
    n_outputs = None
    
    for idx, result_file in enumerate(result_files):
        result_path = Path(result_file)
        if not result_path.exists():
            print(f"⚠️  File not found: {result_file}")
            continue
        
        print(f"\n📊 Loading {labels[idx]}: {result_path.name}")
        
        results = load_pipeline_results(result_path)
        
        # Extract configuration
        config = results['config']
        n_iterations = config.get('n_iterations', None)
        samples_per_iteration = config['samples_per_iteration']
        
        print(f"  Samples per iteration: {samples_per_iteration}")
        if n_iterations:
            print(f"  Number of iterations: {n_iterations}")
        
        # Extract aggregated results
        agg_results = results['aggregated_results']
        
        iterations = []
        total_samples_seen = []
        mean_total_mse = []
        std_total_mse = []
        mean_mse_per_output = []
        std_mse_per_output = []
        
        for result in agg_results:
            iterations.append(result['iteration'])
            total_samples_seen.append(result['total_samples_seen'])
            mean_total_mse.append(result['mean_total_mse'])
            std_total_mse.append(result['std_total_mse'])
            mean_mse_per_output.append(result['mean_mse_per_output'])
            std_mse_per_output.append(result['std_mse_per_output'])
        
        # Convert to numpy arrays
        data = {
            'label': labels[idx],
            'iterations': np.array(iterations),
            'total_samples_seen': np.array(total_samples_seen),
            'mean_total_mse': np.array(mean_total_mse),
            'std_total_mse': np.array(std_total_mse),
            'mean_mse_per_output': np.array(mean_mse_per_output),
            'std_mse_per_output': np.array(std_mse_per_output),
            'config': config
        }
        
        all_data.append(data)
        
        # Get n_outputs from first file
        if n_outputs is None:
            n_outputs = data['mean_mse_per_output'].shape[1]
        
        print(f"  Initial MSE: {data['mean_total_mse'][0]:.6e} ± {data['std_total_mse'][0]:.6e}")
        print(f"  Final MSE: {data['mean_total_mse'][-1]:.6e} ± {data['std_total_mse'][-1]:.6e}")
        print(f"  Improvement: {data['mean_total_mse'][0] / data['mean_total_mse'][-1]:.2f}x")
    
    if len(all_data) == 0:
        print("⚠️  No valid data files found!")
        return
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Build title with training data info from the adaptive sampling dataset (last one, or first with shrink_rate > 0)
    # Use the last dataset as it's typically the adaptive one
    adaptive_config = all_data[-1]['config']
    n_iterations = adaptive_config.get('n_iterations')
    samples_per_iteration = adaptive_config['samples_per_iteration']
    shrink_rate = adaptive_config['shrink_rate']
    
    # Calculate shrink percentage (remaining pool size after shrink)
    shrink_percentage = int((1 - shrink_rate) * 100)
    
    if n_iterations:
        dataset_info = f"{n_iterations} iterations, {samples_per_iteration} samples/iteration, {shrink_percentage}% shrink/iter"
    else:
        dataset_info = f"{samples_per_iteration} samples/iteration, {shrink_percentage}% shrink/iter"
    
    # Create main title
    fig.suptitle(f'Neural Network Training - Varying ODE Relative Error Tolerance. \n'
                 f'Relative Error Tolerance Levels: 1e-4 vs 1e-12\n', 
                 fontsize=15, fontweight='bold', y=1.00)
    
    # Define colors and markers for different runs
    plot_colors = ['#2E86AB', '#E63946', '#F77F00', '#06AED5', '#2A9D8F']
    plot_markers = ['o', 's', '^', 'D', 'v']
    
    # ============================================================
    # Plot 1: Total MSE vs Samples Seen
    # ============================================================
    for idx, data in enumerate(all_data):
        color = plot_colors[idx % len(plot_colors)]
        marker = plot_markers[idx % len(plot_markers)]
        
        ax1.errorbar(data['total_samples_seen'], data['mean_total_mse'], 
                     yerr=data['std_total_mse'],
                     marker=marker, linewidth=2, markersize=6, capsize=5, 
                     label=data['label'], alpha=0.8, color=color)
    
    ax1.set_xlabel('Samples Seen', fontsize=13)
    ax1.set_ylabel('Total MSE (Sum across outputs)', fontsize=13)
    ax1.set_title('Learning Curve', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')
    
    # ============================================================
    # Plot 2: MSE per Output (K values) vs Samples Seen
    # ============================================================
    # Plot 2: MSE per Output (K values) vs Samples Seen
    # ============================================================
    k_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Colors for different K values
    output_labels = [f'K{i+1}' for i in range(n_outputs)]
    
    # Plot each K value, showing all runs together
    for i in range(n_outputs):
        for idx, data in enumerate(all_data):
            marker = plot_markers[idx % len(plot_markers)]
            linestyle = '-' if idx == 0 else '--'
            
            ax2.errorbar(data['total_samples_seen'], data['mean_mse_per_output'][:, i], 
                         yerr=data['std_mse_per_output'][:, i],
                         marker=marker, linewidth=2, markersize=6, capsize=5,
                         label=f'{output_labels[i]} ({data["label"]})', 
                         alpha=0.8, color=k_colors[i % len(k_colors)],
                         linestyle=linestyle)
    
    ax2.set_xlabel('Samples Seen', fontsize=13)
    ax2.set_ylabel('MSE per Output', fontsize=13)
    ax2.set_title('MSE per K Value', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
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
    
    plt.show()


def main():
    """Main function to run the plotting script."""
    
    # Example: Compare uniform batching vs adaptive sampling
    result_files = [
        'pipeline_results/sample_efficiency_200per_iter_shrink1_20251202_192414.json',
        'pipeline_results/sample_efficiency_200per_iter_shrink1_20251202_192953.json',
    ]
    
    labels = [
        'Uniform Batching - Relative Error Tolerance Levels: 1e-12',
        'Uniform Batching - Relative Error Tolerance Levels: 1e-4',

    ]
    
    # Check if files exist
    valid_files = []
    valid_labels = []
    for file, label in zip(result_files, labels):
        if Path(file).exists():
            valid_files.append(file)
            valid_labels.append(label)
        else:
            print(f"⚠️  File not found: {file}")
    
    if len(valid_files) == 0:
        print("\n⚠️  No valid result files found!")
        return
    
    # Create comparison plots
    plot_adaptive_batch_sampling_results(valid_files, labels=valid_labels)


if __name__ == "__main__":
    main()
