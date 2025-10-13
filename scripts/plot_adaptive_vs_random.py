#!/usr/bin/env python3
"""Plot adaptive vs random experiment results with log-scale error.

Loads the JSON produced by `train_batch_iterative_zones_adaptive.py` at
`results/adaptive_vs_random_experiment_results.json` (default) and creates
a multi-panel figure with:
 - overall MSE vs total training samples (log y-scale)
 - per-zone MSE evolution for adaptive run (if available)

Usage:
    python scripts/plot_adaptive_vs_random.py [--results PATH] [--out PATH] [--show]

The script will save a PNG to `results/plot_adaptive_vs_random_log.png` by default.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def load_results(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results JSON not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)


def extract_series(results: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    return results.get(key, [])


def safe_get_total_mse_list(series: List[Dict[str, Any]]) -> List[float]:
    out = []
    for it in series:
        val = it.get('total_mse')
        if val is None:
            out.append(np.nan)
        else:
            out.append(float(val))
    return out


def safe_get_total_samples(series: List[Dict[str, Any]]) -> List[int]:
    out = []
    for it in series:
        out.append(int(it.get('total_samples', 0)))
    return out


def plot_results(results_json: str, out_path: str, show: bool = False) -> None:
    data = load_results(results_json)

    # Try both old and new key formats
    adaptive = extract_series(data, 'adaptive_results_avg') or extract_series(data, 'adaptive_results')
    random = extract_series(data, 'random_results_avg') or extract_series(data, 'random_results')

    if not adaptive and not random:
        raise ValueError('No adaptive_results_avg/adaptive_results or random_results_avg/random_results found in JSON')

    # Prepare series (skip first element - initial uniform training)
    x_adaptive = safe_get_total_samples(adaptive[1:]) if adaptive and len(adaptive) > 1 else []
    y_adaptive = safe_get_total_mse_list(adaptive[1:]) if adaptive and len(adaptive) > 1 else []

    x_random = safe_get_total_samples(random[1:]) if random and len(random) > 1 else []
    y_random = safe_get_total_mse_list(random[1:]) if random and len(random) > 1 else []

    # Begin plotting
    fig = plt.figure(figsize=(12, 8))

    # Plot 1: Overall MSE comparison (normal scale)
    ax1 = fig.add_subplot(2, 2, 1)
    if x_adaptive and y_adaptive:
        ax1.plot(x_adaptive, y_adaptive, 'b-o', label='Adaptive Zone-Based', linewidth=2)
    if x_random and y_random:
        ax1.plot(x_random, y_random, 'r-s', label='Random Sampling', linewidth=2)
    ax1.set_xlabel('Total training samples')
    ax1.set_ylabel('Total MSE')
    ax1.set_title('MSE Evolution: Adaptive vs Random')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Per-zone MSE evolution for adaptive run (if available)
    ax2 = fig.add_subplot(2, 2, 2)
    if adaptive and len(adaptive) > 1:
        n_zones = None
        zone_mse_matrix = None
        adaptive_data = adaptive[1:]  # Skip first element (initial uniform)
        
        for i, it in enumerate(adaptive_data):
            vals = it.get('zone_overall_mse_scaled')
            if vals:
                if zone_mse_matrix is None:
                    n_zones = len(vals)
                    zone_mse_matrix = np.full((len(adaptive_data), n_zones), np.nan)
                for z in range(min(n_zones, len(vals))):
                    try:
                        zone_mse_matrix[i, z] = float(vals[z]) if vals[z] is not None else np.nan
                    except Exception:
                        zone_mse_matrix[i, z] = np.nan

        if zone_mse_matrix is not None:
            cmap = plt.get_cmap('tab10')
            for z in range(zone_mse_matrix.shape[1]):
                ax2.plot(x_adaptive, zone_mse_matrix[:, z], marker='o', label=f'Zone {z+1}', color=cmap(z % 10))
            ax2.set_xlabel('Total training samples')
            ax2.set_ylabel('Zone MSE')
            ax2.set_title('Adaptive: Per-Zone MSE Evolution')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No per-zone data available', ha='center')
    else:
        ax2.text(0.5, 0.5, 'No adaptive results in JSON', ha='center')

    # Plot 3: Sampling frequency per zone (bar chart)
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Try to get targeting data from per-seed results first, then from averaged results
    adaptive_per_seed = data.get('adaptive_per_seed', [])
    targeting_data_found = False
    
    if adaptive_per_seed:
        # Count targeting across all seeds and iterations
        zone_counts = {}
        total_iterations = 0
        
        for seed_results in adaptive_per_seed:
            for r in seed_results[1:]:  # Skip iteration 0 (initial)
                target_zones = r.get('target_zones', [])
                if target_zones:
                    total_iterations += 1
                    for zone in target_zones:
                        zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        if zone_counts and total_iterations > 0:
            targeting_data_found = True
            # Create bar chart
            zones = sorted(zone_counts.keys())
            counts = [zone_counts[z] for z in zones]
            percentages = [(count / total_iterations) * 100 for count in counts]
            
            bars = ax3.bar(zones, percentages, color='lightblue', edgecolor='navy', alpha=0.7)
            ax3.set_xlabel('Zone Number')
            ax3.set_ylabel('Targeting Frequency (%)')
            ax3.set_title('Adaptive: Zone Targeting Frequency\n(3 zones targeted per iteration)')
            ax3.grid(True, axis='y', alpha=0.3)
            ax3.set_ylim(0, max(percentages) * 1.1 if percentages else 100)
            
            # Add percentage labels on bars
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(percentages) * 0.02,
                        f'{pct:.0f}%', ha='center', va='bottom', fontsize=9)
            
            # Add text summary
            most_targeted = max(zone_counts.items(), key=lambda x: x[1])
            num_seeds = len(adaptive_per_seed)
            total_zone_selections = sum(zone_counts.values())
            ax3.text(0.02, 0.98, f'Most targeted: Zone {most_targeted[0]} ({most_targeted[1]}/{total_iterations} times)\nTotal zone selections: {total_zone_selections}\nAcross {num_seeds} seeds', 
                    transform=ax3.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Fallback to averaged results if per-seed data doesn't have targeting info
    if not targeting_data_found and adaptive and len(adaptive) > 1:
        zone_counts = {}
        total_iterations = 0
        
        for r in adaptive[1:]:  # Skip iteration 0 (initial)
            target_zones = r.get('target_zones', [])
            if target_zones:
                total_iterations += 1
                for zone in target_zones:
                    zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        if zone_counts and total_iterations > 0:
            targeting_data_found = True
            # Create bar chart (same code as above)
            zones = sorted(zone_counts.keys())
            counts = [zone_counts[z] for z in zones]
            percentages = [(count / total_iterations) * 100 for count in counts]
            
            bars = ax3.bar(zones, percentages, color='lightblue', edgecolor='navy', alpha=0.7)
            ax3.set_xlabel('Zone Number')
            ax3.set_ylabel('Targeting Frequency (%)')
            ax3.set_title('Adaptive: Zone Targeting Frequency\n(3 zones targeted per iteration)')
            ax3.grid(True, axis='y', alpha=0.3)
            ax3.set_ylim(0, max(percentages) * 1.1 if percentages else 100)
            
            # Add percentage labels on bars
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(percentages) * 0.02,
                        f'{pct:.0f}%', ha='center', va='bottom', fontsize=9)
            
            # Add text summary
            most_targeted = max(zone_counts.items(), key=lambda x: x[1])
            total_zone_selections = sum(zone_counts.values())
            ax3.text(0.02, 0.98, f'Most targeted: Zone {most_targeted[0]} ({most_targeted[1]}/{total_iterations} times)\nTotal zone selections: {total_zone_selections}', 
                    transform=ax3.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if not targeting_data_found:
        ax3.text(0.5, 0.5, 'No targeting data available\nin JSON results', ha='center', va='center', transform=ax3.transAxes)

    # Plot 4: Direct comparison - Adaptive vs Random performance
    ax4 = fig.add_subplot(2, 2, 4)
    if adaptive and random and len(adaptive) > 0 and len(random) > 0:
        # Calculate how much better/worse adaptive is compared to random at each sample size
        comparison_ratios = []
        comparison_samples = []
        comparison_percentages = []
        
        # Match sample sizes between adaptive and random
        for i, adapt_result in enumerate(adaptive):
            adapt_samples = adapt_result.get('total_samples')
            adapt_mse = adapt_result.get('total_mse')
            
            if adapt_samples and adapt_mse:
                # Find corresponding random result with same or similar sample size
                for random_result in random:
                    random_samples = random_result.get('total_samples')
                    random_mse = random_result.get('total_mse')
                    
                    if random_samples == adapt_samples and random_mse and random_mse > 0:
                        # Calculate percentage improvement: (random - adaptive) / random * 100
                        improvement_pct = ((random_mse - adapt_mse) / random_mse) * 100
                        comparison_ratios.append(adapt_mse / random_mse)
                        comparison_samples.append(adapt_samples)
                        comparison_percentages.append(improvement_pct)
                        break
        
        if comparison_percentages:
            # Plot improvement percentages
            colors = ['green' if pct > 0 else 'red' for pct in comparison_percentages]
            bars = ax4.bar(range(len(comparison_percentages)), comparison_percentages, 
                          color=colors, alpha=0.7, edgecolor='black')
            
            # Customize the plot
            ax4.set_xlabel('Sample Size')
            ax4.set_ylabel('Improvement over Random (%)')
            ax4.set_title('Adaptive vs Random: Performance Comparison\n(Positive = Adaptive Better)')
            ax4.grid(True, axis='y', alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
            
            # Set x-axis labels to show actual sample sizes
            tick_labels = [str(s) for s in comparison_samples]
            ax4.set_xticks(range(len(comparison_samples)))
            ax4.set_xticklabels(tick_labels, rotation=45)
            
            # Add percentage labels on bars
            for i, (bar, pct) in enumerate(zip(bars, comparison_percentages)):
                height = bar.get_height()
                label_y = height + (max(comparison_percentages) - min(comparison_percentages)) * 0.01
                if height < 0:
                    label_y = height - (max(comparison_percentages) - min(comparison_percentages)) * 0.01
                ax4.text(bar.get_x() + bar.get_width()/2., label_y,
                        f'{pct:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                        fontsize=8)
            
            # Add summary statistics
            avg_improvement = np.mean(comparison_percentages)
            best_improvement = max(comparison_percentages)
            ax4.text(0.02, 0.02, f'Avg improvement: {avg_improvement:.1f}%\nBest improvement: {best_improvement:.1f}%', 
                    transform=ax4.transAxes, fontsize=8, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightgreen' if avg_improvement > 0 else 'lightcoral', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No matching sample sizes\nfor comparison', ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor comparison', ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'📊 Plot saved to: {out_path}')

    # Print average MSE per iteration (averaged across outputs when available)
    def compute_avg_mse_across_outputs(series):
        rows = []
        for it in series:
            iter_idx = it.get('iteration', None)
            total_samples = it.get('total_samples', None)
            batch_added = it.get('batch_samples_added', None)
            mse_per_output = it.get('mse_per_output', None) or it.get('zone_per_output_mse_scaled', None)
            if mse_per_output:
                try:
                    # convert nested lists to numpy and take mean across outputs
                    arr = np.array(mse_per_output)
                    # if arr is 2D (outputs x subsets) take mean over outputs (axis=0) then mean over subsets
                    if arr.ndim == 2:
                        avg_over_outputs = np.nanmean(arr, axis=0).mean()
                    else:
                        avg_over_outputs = np.nanmean(arr)
                except Exception:
                    avg_over_outputs = float(it.get('total_mse', np.nan))
            else:
                avg_over_outputs = float(it.get('total_mse', np.nan)) if it.get('total_mse') is not None else np.nan

            rows.append((iter_idx, total_samples, batch_added, avg_over_outputs))
        return rows

    print('\n📋 Average MSE per iteration (averaged across outputs when available):')
    if adaptive:
        print('\nAdaptive:')
        rows = compute_avg_mse_across_outputs(adaptive)
        print(f"{'Iter':>4} {'Samples':>8} {'Added':>6} {'AvgMSE':>12}")
        for r in rows:
            print(f"{str(r[0]):>4} {str(r[1]):>8} {str(r[2]):>6} {r[3]:12.6e}")
        # overall mean
        vals = [r[3] for r in rows if not np.isnan(r[3])]
        if vals:
            print(f"Adaptive mean AvgMSE across iterations: {np.mean(vals):.6e}")

    if random:
        print('\nRandom:')
        rows_r = compute_avg_mse_across_outputs(random)
        print(f"{'Iter':>4} {'Samples':>8} {'Added':>6} {'AvgMSE':>12}")
        for r in rows_r:
            print(f"{str(r[0]):>4} {str(r[1]):>8} {str(r[2]):>6} {r[3]:12.6e}")
        vals_r = [r[3] for r in rows_r if not np.isnan(r[3])]
        if vals_r:
            print(f"Random mean AvgMSE across iterations: {np.mean(vals_r):.6e}")

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot adaptive vs random experiment results (log-scale MSE)')
    parser.add_argument('--results', '-r', default='results/adaptive_vs_random_experiment_results.json', help='Path to results JSON')
    parser.add_argument('--out', '-o', default='results/plot_adaptive_vs_random_log.png', help='Output PNG path')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    args = parser.parse_args()

    try:
        plot_results(args.results, args.out, show=args.show)
    except Exception as e:
        print('Error while plotting results:', e)
        sys.exit(2)


if __name__ == '__main__':
    main()
