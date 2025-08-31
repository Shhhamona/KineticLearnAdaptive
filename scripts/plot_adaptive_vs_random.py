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

    adaptive = extract_series(data, 'adaptive_results')
    random = extract_series(data, 'random_results')

    if not adaptive and not random:
        raise ValueError('No adaptive_results or random_results found in JSON')

    # Prepare series
    x_adaptive = safe_get_total_samples(adaptive) if adaptive else []
    y_adaptive = safe_get_total_mse_list(adaptive) if adaptive else []

    x_random = safe_get_total_samples(random) if random else []
    y_random = safe_get_total_mse_list(random) if random else []

    # Begin plotting
    fig = plt.figure(figsize=(12, 8))

    # Plot 1: Overall MSE comparison (log-scale on Y)
    ax1 = fig.add_subplot(2, 2, 1)
    if x_adaptive and y_adaptive:
        ax1.plot(x_adaptive, y_adaptive, 'b-o', label='Adaptive Zone-Based', linewidth=2)
    if x_random and y_random:
        ax1.plot(x_random, y_random, 'r-s', label='Random Sampling', linewidth=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Total training samples')
    ax1.set_ylabel('Total MSE (log scale)')
    ax1.set_title('MSE Evolution: Adaptive vs Random (log-scale)')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()

    # Plot 2: Per-zone MSE evolution for adaptive run (if available)
    ax2 = fig.add_subplot(2, 2, 2)
    if adaptive:
        n_zones = None
        zone_mse_matrix = None
        for i, it in enumerate(adaptive):
            vals = it.get('zone_overall_mse_scaled')
            if vals:
                if zone_mse_matrix is None:
                    n_zones = len(vals)
                    zone_mse_matrix = np.full((len(adaptive), n_zones), np.nan)
                for z in range(min(n_zones, len(vals))):
                    try:
                        zone_mse_matrix[i, z] = float(vals[z]) if vals[z] is not None else np.nan
                    except Exception:
                        zone_mse_matrix[i, z] = np.nan

        if zone_mse_matrix is not None:
            cmap = plt.get_cmap('tab10')
            for z in range(zone_mse_matrix.shape[1]):
                ax2.plot(x_adaptive, zone_mse_matrix[:, z], marker='o', label=f'Zone {z+1}', color=cmap(z % 10))
            ax2.set_yscale('log')
            ax2.set_xlabel('Total training samples')
            ax2.set_ylabel('Zone MSE (log scale)')
            ax2.set_title('Adaptive: Per-Zone MSE Evolution (log-scale)')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, which='both', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No per-zone data available', ha='center')
    else:
        ax2.text(0.5, 0.5, 'No adaptive results in JSON', ha='center')

    # Plot 3: Sampling frequency per zone (bar chart)
    ax3 = fig.add_subplot(2, 2, 3)
    if adaptive and len(adaptive) > 1:
        zone_counts = {}
        for r in adaptive[1:]:
            for zone in r.get('target_zones', []):
                zone_counts[zone] = zone_counts.get(zone, 0) + 1
        if zone_counts:
            zones = sorted(zone_counts.keys())
            counts = [zone_counts[z] for z in zones]
            ax3.bar(zones, counts, color='lightblue', edgecolor='navy')
            ax3.set_xlabel('Zone Number')
            ax3.set_ylabel('Times Targeted')
            ax3.set_title('Adaptive: Zone Targeting Frequency')
        else:
            ax3.text(0.5, 0.5, 'No targeting info', ha='center')
    else:
        ax3.text(0.5, 0.5, 'No adaptive targeting info', ha='center')

    # Plot 4: Relative improvement vs initial
    ax4 = fig.add_subplot(2, 2, 4)
    if adaptive and len(adaptive) > 0 and adaptive[0].get('total_mse'):
        base = float(adaptive[0]['total_mse'])
        adaptive_mse = [float(r['total_mse']) for r in adaptive]
        adaptive_improvement = [(adaptive_mse[i] / base - 1) * 100 for i in range(len(adaptive_mse))]
        ax4.plot(x_adaptive, adaptive_improvement, 'b-o', label='Adaptive', linewidth=2)
    if random and len(random) > 0 and random[0].get('total_mse'):
        base_r = float(random[0]['total_mse'])
        random_mse = [float(r['total_mse']) for r in random]
        random_improvement = [(random_mse[i] / base_r - 1) * 100 for i in range(len(random_mse))]
        ax4.plot(x_random, random_improvement, 'r-s', label='Random', linewidth=2)
    ax4.set_xlabel('Total training samples')
    ax4.set_ylabel('MSE Change (%)')
    ax4.set_title('Relative Improvement from Baseline')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

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
