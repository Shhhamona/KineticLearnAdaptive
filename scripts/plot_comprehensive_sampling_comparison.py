#!/usr/bin/env python3
"""
Comprehensive Sampling Method Comparison Plot

This script combines results from:
1. Sample efficiency experiments (uniform, Latin hypercube, etc.)
2. Adaptive vs random experiments

Creates a unified comparison showing all sampling methods on the same plot.

Usage:
    python scripts/plot_comprehensive_sampling_comparison.py [--sample-efficiency PATH] [--adaptive-random PATH] [--out PATH] [--show]
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_sample_efficiency_results(path: str) -> Dict[str, Any]:
    """Load sample efficiency experiment results."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample efficiency results not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)


def load_adaptive_random_results(path: str) -> Dict[str, Any]:
    """Load adaptive vs random experiment results."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Adaptive vs random results not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)


def extract_sample_efficiency_data(data: Dict[str, Any]) -> Dict[str, Dict[str, List]]:
    """Extract plotting data from sample efficiency results."""
    results = {}
    subset_sizes = data['config']['subset_sizes']
    
    print(f"📊 Sample efficiency subset sizes: {subset_sizes}")
    
    for method_name, method_data in data['results'].items():
        results[method_name] = {
            'sample_sizes': subset_sizes,
            'mean_mse': method_data['mean_total_mse'],
            'std_mse': method_data['std_total_mse']
        }
        print(f"   {method_name}: {len(subset_sizes)} points, range {min(subset_sizes)}-{max(subset_sizes)}")
    
    return results


def extract_adaptive_random_data(data: Dict[str, Any]) -> Dict[str, Dict[str, List]]:
    """Extract plotting data from adaptive vs random results."""
    results = {}
    
    # Extract adaptive data (averaged across seeds) - only adaptive, not random
    adaptive_avg = data.get('adaptive_results_avg', [])
    if adaptive_avg and len(adaptive_avg) > 1:  # Skip first element (initial uniform)
        adaptive_data = adaptive_avg[1:]
        sample_sizes = [r['total_samples'] for r in adaptive_data]
        results['Adaptive Zone-Based (from uniform init)'] = {
            'sample_sizes': sample_sizes,
            'mean_mse': [r['total_mse'] for r in adaptive_data],
            'std_mse': [0] * len(adaptive_data)  # No std available for averaged data
        }
        print(f"📊 Adaptive data: {len(sample_sizes)} points, range {min(sample_sizes)}-{max(sample_sizes)}")
    
    # Note: Removed random sampling as it's redundant with uniform sampling
    
    return results


def create_comprehensive_comparison_plot(sample_eff_data: Dict[str, Dict[str, List]], 
                                       adaptive_data: Dict[str, Dict[str, List]], 
                                       out_path: str, show: bool = False) -> None:
    """Create comprehensive comparison plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define colors and markers for consistency
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H']
    
    # Create both plots with the same data
    all_methods = {}
    
    for ax_idx, (ax, scale_type, title_suffix) in enumerate([(ax1, 'linear', '(Linear Scale)'), (ax2, 'log', '(Log Scale)')]):
        method_idx = 0
        
        # Add sample efficiency methods
        for method_name, method_data in sample_eff_data.items():
            color = colors[method_idx % len(colors)]
            marker = markers[method_idx % len(markers)]
            
            if any(std > 0 for std in method_data['std_mse']):
                ax.errorbar(method_data['sample_sizes'], method_data['mean_mse'], 
                           yerr=method_data['std_mse'], label=method_name, 
                           color=color, marker=marker, linewidth=2, markersize=6)
            else:
                ax.plot(method_data['sample_sizes'], method_data['mean_mse'], 
                       label=method_name, color=color, marker=marker, linewidth=2, markersize=6)
            
            if ax_idx == 0:  # Only store once
                all_methods[method_name] = method_data
            method_idx += 1
        
        # Add adaptive methods
        for method_name, method_data in adaptive_data.items():
            color = colors[method_idx % len(colors)]
            marker = markers[method_idx % len(markers)]
            
            # Make adaptive method very prominent
            if 'Adaptive' in method_name:
                linewidth = 4
                markersize = 10
                alpha = 1.0
                # Use a distinctive color for adaptive
                color = 'red'
            else:
                linewidth = 2.5
                markersize = 6
                alpha = 0.9
            
            ax.plot(method_data['sample_sizes'], method_data['mean_mse'], 
                   label=method_name, color=color, marker=marker, 
                   linewidth=linewidth, markersize=markersize, alpha=alpha)
            
            if ax_idx == 0:  # Only store once
                all_methods[method_name] = method_data
            method_idx += 1
        
        # Configure each subplot
        ax.set_xlabel('Training Dataset Size')
        ax.set_ylabel('MSE on Test Set')
        ax.set_title(f'Comprehensive Sampling Method Comparison {title_suffix}')
        
        # Only show legend on the right plot
        if ax_idx == 1:  # Right plot (log scale)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.grid(True, alpha=0.3)
        
        # Set scale
        if scale_type == 'log':
            ax.set_yscale('log')
            ax.set_xscale('log')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')  # Also save as PDF
    print(f'📊 Comprehensive comparison plot saved to: {out_path}')
    print(f'📊 PDF version saved to: {out_path.replace(".png", ".pdf")}')
    
    # Create summary table
    create_summary_table(all_methods, out_path)
    
    if show:
        plt.show()
    else:
        plt.close()


def create_summary_table(all_methods: Dict[str, Dict[str, List]], out_path: str) -> None:
    """Create a summary table comparing all methods."""
    summary_data = []
    
    for method_name, method_data in all_methods.items():
        sizes = method_data['sample_sizes']
        mses = method_data['mean_mse']
        
        if sizes and mses:
            summary_data.append({
                'Method': method_name,
                'Final_Sample_Size': sizes[-1],
                'Final_MSE': mses[-1],
                'Best_MSE': min(mses),
                'Best_Size': sizes[np.argmin(mses)],
                'Improvement_Factor': mses[0] / min(mses) if len(mses) > 1 else 1.0
            })
    
    # Sort by final MSE
    summary_data.sort(key=lambda x: x['Final_MSE'])
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = out_path.replace('.png', '_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f'📋 Summary table saved to: {csv_path}')
    
    # Print to console
    print(f"\n📋 COMPREHENSIVE SAMPLING METHOD COMPARISON:")
    print("=" * 80)
    print(f"{'Method':<25} {'Final MSE':<12} {'Best MSE':<12} {'Best Size':<10} {'Improvement':<12}")
    print("-" * 80)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Method']:<25} {row['Final_MSE']:<12.2e} {row['Best_MSE']:<12.2e} "
              f"{row['Best_Size']:<10.0f} {row['Improvement_Factor']:<12.1f}x")
    
    # Highlight best performers
    best_final = summary_df.iloc[0]
    best_overall = summary_df.loc[summary_df['Best_MSE'].idxmin()]
    
    print(f"\n🏆 BEST PERFORMERS:")
    print(f"   Best Final MSE: {best_final['Method']} ({best_final['Final_MSE']:.2e})")
    print(f"   Best Overall MSE: {best_overall['Method']} ({best_overall['Best_MSE']:.2e} at size {best_overall['Best_Size']:.0f})")


def find_latest_file(directory: str, pattern: str) -> str:
    """Find the most recent file matching the pattern."""
    import glob
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found in {directory}")
    return max(files, key=os.path.getctime)


def main():
    parser = argparse.ArgumentParser(description='Create comprehensive sampling method comparison plot')
    parser.add_argument('--sample-efficiency', '-s', 
                       help='Path to sample efficiency results JSON (if not provided, will look for latest)')
    parser.add_argument('--adaptive-random', '-a', 
                       default='results/adaptive_vs_random_experiment_results.json',
                       help='Path to adaptive vs random results JSON')
    parser.add_argument('--out', '-o', 
                       default='results/comprehensive_sampling_comparison.png',
                       help='Output plot path')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    
    args = parser.parse_args()
    
    try:
        # Load sample efficiency results
        if args.sample_efficiency:
            sample_eff_path = args.sample_efficiency
        else:
            # Find latest sample efficiency results
            sample_eff_path = find_latest_file('results/sample_efficiency', 'sample_efficiency_results_*.json')
            print(f"📂 Using latest sample efficiency results: {sample_eff_path}")
        
        sample_eff_data = load_sample_efficiency_results(sample_eff_path)
        sample_eff_plot_data = extract_sample_efficiency_data(sample_eff_data)
        
        # Load adaptive vs random results
        adaptive_random_data = load_adaptive_random_results(args.adaptive_random)
        adaptive_plot_data = extract_adaptive_random_data(adaptive_random_data)
        
        # Create comprehensive comparison
        create_comprehensive_comparison_plot(sample_eff_plot_data, adaptive_plot_data, args.out, args.show)
        
        print(f"\n✅ Comprehensive comparison completed successfully!")
        
    except Exception as e:
        print(f'❌ Error creating comparison: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
