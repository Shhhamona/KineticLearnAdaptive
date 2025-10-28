"""
Plot Batch Training vs Adaptive Batch Sampling Comparison

This script creates a comparison plot showing batch training with different K ranges
against adaptive batch sampling on the same axes.

Usage:
    python pipeline_plots/plot_batch_vs_adaptive.py
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


def extract_batch_training_data(results):
    """Extract data from batch training results."""
    batch_size = results['config']['batch_size']
    batch_numbers = results['aggregated_results']['batch_numbers']
    
    # Calculate training samples
    sample_counts = np.array([batch_num * batch_size for batch_num in batch_numbers])
    
    mean_total_mse = np.array(results['aggregated_results']['mean_total_mse'])
    std_total_mse = np.array(results['aggregated_results']['std_total_mse'])
    
    return sample_counts, mean_total_mse, std_total_mse


def extract_adaptive_sampling_data(results):
    """Extract data from adaptive batch sampling results."""
    n_epochs = results['config']['n_epochs']
    agg_results = results['aggregated_results']
    
    training_samples = []
    mean_total_mse = []
    std_total_mse = []
    
    cumulative_training_samples = 0
    
    for result in agg_results:
        # Calculate training_samples for this iteration
        if 'training_samples' in result:
            iter_train_samples = result['training_samples']
        else:
            iter_train_samples = 0 if result['iteration'] == 0 else result['samples_added'] * n_epochs
        
        cumulative_training_samples += iter_train_samples
        training_samples.append(cumulative_training_samples)
        mean_total_mse.append(result['mean_total_mse'])
        std_total_mse.append(result['std_total_mse'])
    
    return np.array(training_samples), np.array(mean_total_mse), np.array(std_total_mse)


def plot_comparison(batch_files, batch_labels, adaptive_file, output_dir='pipeline_results/plots'):
    """
    Create comparison plot of batch training vs adaptive batch sampling.
    
    Args:
        batch_files: List of paths to batch training JSON files
        batch_labels: List of labels for batch training methods
        adaptive_file: Path to adaptive batch sampling JSON file
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    print("="*70)
    print("Loading Batch Training Results")
    print("="*70)
    
    # Plot batch training results
    for batch_file, label in zip(batch_files, batch_labels):
        batch_path = Path(batch_file)
        if not batch_path.exists():
            print(f"⚠️  File not found: {batch_file}")
            continue
        
        print(f"📊 Loading: {batch_path.name}")
        results = load_pipeline_results(batch_path)
        
        sample_counts, mean_mse, std_mse = extract_batch_training_data(results)
        
        ax.errorbar(sample_counts, mean_mse, yerr=std_mse,
                   label=f'Batch Training - {label}', 
                   marker='o', linewidth=2, markersize=5, capsize=4, alpha=0.7,
                   linestyle='--')
        
        print(f"   Initial MSE: {mean_mse[0]:.6e}")
        print(f"   Final MSE: {mean_mse[-1]:.6e}")
    
    print("\n" + "="*70)
    print("Loading Adaptive Batch Sampling Results")
    print("="*70)
    
    # Plot adaptive batch sampling results
    adaptive_path = Path(adaptive_file)
    if not adaptive_path.exists():
        print(f"⚠️  File not found: {adaptive_file}")
    else:
        print(f"📊 Loading: {adaptive_path.name}")
        results = load_pipeline_results(adaptive_path)
        
        sample_counts, mean_mse, std_mse = extract_adaptive_sampling_data(results)
        
        ax.errorbar(sample_counts, mean_mse, yerr=std_mse,
                   label='Adaptive Batch Sampling', 
                   marker='s', linewidth=3, markersize=7, capsize=5, alpha=0.9,
                   color='#2E86AB', linestyle='-')
        
        print(f"   Initial MSE: {mean_mse[0]:.6e}")
        print(f"   Final MSE: {mean_mse[-1]:.6e}")
        print(f"   Total training samples: {sample_counts[-1]:.0f}")
    
    # Configure plot
    ax.set_xlabel('Training Samples', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total MSE (Sum across outputs)', fontsize=14, fontweight='bold')
    ax.set_title('Neural Network Training Comparison\nBatch Training vs Adaptive Batch Sampling', 
                 fontsize=15, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    
    plt.tight_layout()
    
    # Save plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = output_dir / f'batch_vs_adaptive_comparison_{timestamp}.pdf'
    plt.savefig(plot_path)
    print(f"\n📊 Saved comparison plot to: {plot_path}")
    
    plot_png_path = output_dir / f'batch_vs_adaptive_comparison_{timestamp}.png'
    plt.savefig(plot_png_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved PNG to: {plot_png_path}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("✅ Comparison plot generated successfully!")
    print("="*70)


def main():
    """Main function to create comparison plot."""
    
    results_dir = Path("pipeline_results")
    
    # Define batch training files (K range variations)
    batch_prefixes = [
        'Uniform Sampling-K_factor_2',
        'Uniform Sampling-K_factor_1.15',
        'Uniform Sampling-K_factor_1.005',
        'Uniform Sampling-K_factor_1.00005',
    ]
    
    batch_labels = [
        'K ∈ [K/2, K×2]',
        'K ∈ [K/1.15, K×1.15]',
        'K ∈ [K/1.005, K×1.005]',
        'K ∈ [K/1.00005, K×1.00005]',
    ]
    
    # Find latest batch training files
    batch_files = []
    found_labels = []
    
    print("Searching for batch training result files...")
    for prefix, label in zip(batch_prefixes, batch_labels):
        matching_files = sorted(results_dir.glob(f'{prefix}*.json'))
        if matching_files:
            latest_file = matching_files[-1]
            batch_files.append(latest_file)
            found_labels.append(label)
            print(f"✓ Found: {latest_file.name}")
        else:
            print(f"⚠️  No files found matching: {prefix}*.json")
    
    # Use specific adaptive batch sampling file
    print("\nUsing adaptive batch sampling result file...")
    adaptive_file = results_dir / 'adaptive_batch_sampling_w1.0_s0.1_e25_20251028_185800.json'
    
    if not adaptive_file.exists():
        print(f"⚠️  Adaptive file not found: {adaptive_file}")
        print("Searching for alternative files...")
        adaptive_files = sorted(results_dir.glob('adaptive_batch_sampling*.json'))
        if adaptive_files:
            adaptive_file = adaptive_files[-1]
            print(f"✓ Using alternative: {adaptive_file.name}")
        else:
            print("⚠️  No adaptive batch sampling files found!")
            return
    else:
        print(f"✓ Found: {adaptive_file.name}")
    
    if not batch_files:
        print("\n❌ No batch training files found!")
        return
    
    # Create comparison plot
    print(f"\n✅ Found {len(batch_files)} batch training files and 1 adaptive sampling file\n")
    plot_comparison(batch_files, found_labels, adaptive_file)


if __name__ == "__main__":
    main()
