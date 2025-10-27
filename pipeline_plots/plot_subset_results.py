"""
Plot Subset Pipeline Results

This script loads saved results from StandardSubsetPipeline runs and creates
comprehensive plots showing sample efficiency using relative error metrics.

Usage:
    python examples/plot_subset_results.py
"""

from pathlib import Path
import sys
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def load_pipeline_results(json_path):
    """Load results from a saved pipeline JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_relative_error_comparison(results_files, labels=None, output_dir='pipeline_results/plots'):
    """
    Create comparison plot of relative errors across multiple pipeline runs.
    
    Args:
        results_files: List of paths to JSON result files
        labels: List of labels for each result file (optional)
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(results_files))]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    all_results = []
    
    for idx, (result_file, label) in enumerate(zip(results_files, labels)):
        print(f"📊 Loading: {result_file}")
        results = load_pipeline_results(result_file)
        
        agg = results['aggregated_results']
        subset_sizes = agg['subset_sizes']
        
        # Calculate mean relative error across all outputs
        mean_rel_error_per_output = np.array(agg['mean_rel_error_per_output'])  # shape: (n_subsets, n_outputs)
        std_rel_error_per_output = np.array(agg['std_rel_error_per_output'])
        
        # Average across outputs to get overall relative error per subset size
        mean_rel_error = np.mean(mean_rel_error_per_output, axis=1)
        # Propagate uncertainty
        std_rel_error = np.sqrt(np.sum(std_rel_error_per_output**2, axis=1)) / mean_rel_error_per_output.shape[1]
        
        all_results.append({
            'label': label,
            'subset_sizes': subset_sizes,
            'mean_rel_error': mean_rel_error,
            'std_rel_error': std_rel_error,
            'mean_rel_error_per_output': mean_rel_error_per_output,
            'std_rel_error_per_output': std_rel_error_per_output
        })
        
        # Plot with error bars
        plt.errorbar(subset_sizes, mean_rel_error, yerr=std_rel_error, 
                    label=label, marker='o', linewidth=2, markersize=6)
        
        print(f"   ✅ {label}: Final Rel Error = {mean_rel_error[-1]:.2f}% ± {std_rel_error[-1]:.2f}%")
    
    plt.xlabel('Training Dataset Size', fontsize=14)
    plt.ylabel('Mean Relative Error (%)', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = output_dir / f'sample_efficiency_comparison_{timestamp}.pdf'
    plt.savefig(plot_path)
    print(f"📊 Saved comparison plot to: {plot_path}")
    
    plot_png_path = output_dir / f'sample_efficiency_comparison_{timestamp}.png'
    plt.savefig(plot_png_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved PNG to: {plot_png_path}")
    
    plt.show()
    
    return all_results


def plot_total_mse_comparison(results_files, labels=None, output_dir='pipeline_results/plots'):
    """
    Create comparison plot of total MSE (sum across outputs) across multiple pipeline runs.
    This matches the original sample_effiency_real_k.py visualization.
    
    Args:
        results_files: List of paths to JSON result files
        labels: List of labels for each result file (optional)
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(results_files))]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    all_results = []
    
    for idx, (result_file, label) in enumerate(zip(results_files, labels)):
        print(f"📊 Loading: {result_file}")
        results = load_pipeline_results(result_file)
        
        agg = results['aggregated_results']
        subset_sizes = agg['subset_sizes']
        # Convert RMSE to MSE by squaring
        mean_total_rmse = np.array(agg['mean_total_rmse'])
        mean_total_mse = mean_total_rmse ** 2
        std_total_rmse = np.array(agg['std_total_rmse'])
        # Proper error propagation for squaring: std(x^2) = 2*mean(x)*std(x)
        std_total_mse = 2 * mean_total_rmse * std_total_rmse
        
        all_results.append({
            'label': label,
            'subset_sizes': subset_sizes,
            'mean_total_mse': mean_total_mse,
            'std_total_mse': std_total_mse
        })
        
        # Plot with error bars
        plt.errorbar(subset_sizes, mean_total_mse, yerr=std_total_mse, 
                    label=label, marker='o', linewidth=2, markersize=6, capsize=5)
        
        print(f"   ✅ {label}: Final Total MSE = {mean_total_mse[-1]:.6e} ± {std_total_mse[-1]:.6e}")
    
    plt.xlabel('Training Dataset Size', fontsize=14)
    plt.ylabel('Total MSE (Sum across outputs)', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = output_dir / f'sample_efficiency_mse_comparison_{timestamp}.pdf'
    plt.savefig(plot_path)
    print(f"📊 Saved MSE comparison plot to: {plot_path}")
    
    plot_png_path = output_dir / f'sample_efficiency_mse_comparison_{timestamp}.png'
    plt.savefig(plot_png_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved PNG to: {plot_png_path}")
    
    plt.show()
    
    return all_results


def plot_per_output_relative_error(results_file, output_dir='pipeline_results/plots'):
    """
    Create plots showing relative error for each output (K1, K2, K3) separately.
    
    Args:
        results_file: Path to JSON result file
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Loading: {results_file}")
    results = load_pipeline_results(results_file)
    
    agg = results['aggregated_results']
    subset_sizes = agg['subset_sizes']
    mean_rel_error_per_output = np.array(agg['mean_rel_error_per_output'])  # shape: (n_subsets, n_outputs)
    std_rel_error_per_output = np.array(agg['std_rel_error_per_output'])
    
    n_outputs = mean_rel_error_per_output.shape[1]
    
    # Create figure with subplots for each output
    fig, axes = plt.subplots(1, n_outputs, figsize=(16, 5))
    
    for i in range(n_outputs):
        ax = axes[i] if n_outputs > 1 else axes
        
        mean_rel_error = mean_rel_error_per_output[:, i]
        std_rel_error = std_rel_error_per_output[:, i]
        
        ax.errorbar(subset_sizes, mean_rel_error, yerr=std_rel_error,
                   marker='o', linewidth=2, markersize=6, capsize=5)
        
        ax.set_xlabel('Training Dataset Size', fontsize=12)
        ax.set_ylabel('Relative Error (%)', fontsize=12)
        ax.set_title(f'K{i+1} Prediction Error', fontsize=13)
        ax.grid(True, alpha=0.3)
        
        # Add final error annotation
        final_error = mean_rel_error[-1]
        final_std = std_rel_error[-1]
        ax.text(0.98, 0.98, f'Final: {final_error:.2f}% ± {final_std:.2f}%',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10)
        
        print(f"   K{i+1}: Final Rel Error = {final_error:.2f}% ± {final_std:.2f}%")
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pipeline_name = results.get('pipeline_name', 'pipeline')
    plot_path = output_dir / f'{pipeline_name}_per_output_{timestamp}.pdf'
    plt.savefig(plot_path)
    print(f"📊 Saved per-output plot to: {plot_path}")
    
    plot_png_path = output_dir / f'{pipeline_name}_per_output_{timestamp}.png'
    plt.savefig(plot_png_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved PNG to: {plot_png_path}")
    
    plt.show()


def main():
    """Main function to demonstrate plotting capabilities."""
    
    # Find the most recent result file
    results_dir = Path("pipeline_results")
    json_files = sorted(results_dir.glob("sample_efficiency_*.json"))
    
    if not json_files:
        print("❌ No result files found in pipeline_results/")
        print("   Please run run_subset_pipeline.py first!")
        return
    
    latest_file = json_files[-1]
    
    # Example 1: Plot Total MSE (matching original visualization)
    print("="*70)
    print("EXAMPLE 1: Total MSE Analysis (Sum across outputs)")
    print("="*70)
    print(f"Using latest result file: {latest_file}\n")
    
    # Single file - just plot that one
    plot_total_mse_comparison([latest_file], labels=["Latin Hypercube Uniform Sampling"])
    
    # Example 2: Plot relative error comparison
    print("\n" + "="*70)
    print("EXAMPLE 2: Relative Error Comparison")
    print("="*70)
    
    # Single file - just plot that one
    plot_relative_error_comparison([latest_file], labels=["Latin Hypercube Uniform Sampling"])
    
    # Example 3: Plot per-output breakdown
    print("\n" + "="*70)
    print("EXAMPLE 3: Per-Output Relative Error Analysis")
    print("="*70)
    
    plot_per_output_relative_error(latest_file)
    
    print("\n✅ Plotting complete!")


if __name__ == "__main__":
    main()
