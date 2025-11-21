"""
Plot Adaptive Batch Sampling Results - Single K Value Analysis

This script loads saved results from AdaptiveBatchSamplingPipeline runs for single K value
experiments and creates plots showing MSE and variance evolution per samples seen.

Simplified version:
- No batch size/epoch multiplication, just samples seen directly
- Focuses on single K value (output)
- Plots both MSE and prediction variance

Usage:
    python pipeline_plots/plot_adaptive_batch_sampling_single_k.py
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


def plot_single_k_results(result_file, output_dir='pipeline_results/plots', reaction_idx=None, 
                         comparison_files=None):
    """
    Create separate plots for single K value adaptive batch sampling results:
    1. MSE vs Samples Seen (separate image)
    2. Prediction Variance vs Samples Seen (separate image, if available)
    
    Args:
        result_file: Path to JSON result file (main/baseline results)
        output_dir: Directory to save plots
        reaction_idx: Reaction index (for filename labeling)
        comparison_files: Optional list of paths to comparison results files (adaptive runs)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = Path(result_file)
    if not result_path.exists():
        print(f"⚠️  File not found: {result_file}")
        return
    
    print("="*70)
    print("Single K Value - Adaptive Batch Sampling Results")
    print("="*70)
    print(f"\n📊 Loading: {result_path.name}")
    
    results = load_pipeline_results(result_path)
    
    # Extract configuration
    config = results['config']
    window_type = config['window_type']
    initial_window = config['initial_window_size']
    shrink_rate = config['shrink_rate']
    num_pool_datasets = config['num_pool_datasets']
    samples_per_iteration = config['samples_per_iteration']
    n_iterations = config['n_iterations']
    num_seeds = config.get('num_seeds', 1)
    
    print(f"\nConfiguration:")
    print(f"  Window type: {window_type}")
    print(f"  Initial window: {initial_window}")
    print(f"  Shrink rate: {shrink_rate}")
    print(f"  Number of pool files: {num_pool_datasets}")
    print(f"  Samples per iteration: {samples_per_iteration}")
    print(f"  Number of iterations: {n_iterations}")
    print(f"  Number of seeds: {num_seeds}")
    
    # Extract aggregated results
    agg_results = results['aggregated_results']
    
    # First pass: collect all data to calculate sample increments
    iterations = []
    total_samples_seen = []
    mean_total_mse = []
    std_total_mse = []
    mean_mse_per_output = []
    std_mse_per_output = []
    window_sizes = []
    
    for result in agg_results:
        iterations.append(result['iteration'])
        total_samples_seen.append(result['total_samples_seen'])
        mean_total_mse.append(result['mean_total_mse'])
        std_total_mse.append(result['std_total_mse'])
        mean_mse_per_output.append(result['mean_mse_per_output'])
        std_mse_per_output.append(result['std_mse_per_output'])
        window_sizes.append(result['window_size'])
    
    # Convert to numpy arrays
    iterations = np.array(iterations)
    total_samples_seen = np.array(total_samples_seen)
    mean_total_mse = np.array(mean_total_mse)
    std_total_mse = np.array(std_total_mse)
    mean_mse_per_output = np.array(mean_mse_per_output)  # Shape: (n_iterations, n_outputs)
    std_mse_per_output = np.array(std_mse_per_output)
    window_sizes = np.array(window_sizes)
    
    # Filter 1: Remove first point if it has 0 samples
    valid_mask = total_samples_seen > 0
    
    # Filter 2: Keep only iterations where samples increment is at least 1/3 of samples_per_iteration
    min_samples_threshold = samples_per_iteration / 3.0
    
    for i in range(1, len(total_samples_seen)):
        if valid_mask[i]:  # Only check if not already filtered
            samples_increment = total_samples_seen[i] - total_samples_seen[i-1]
            if samples_increment < min_samples_threshold:
                valid_mask[i] = False
                print(f"  ⚠️  Filtering out iteration {iterations[i]}: only {samples_increment:.0f} samples (< {min_samples_threshold:.0f})")
    
    # Apply filtering
    iterations = iterations[valid_mask]
    total_samples_seen = total_samples_seen[valid_mask]
    mean_total_mse = mean_total_mse[valid_mask]
    std_total_mse = std_total_mse[valid_mask]
    mean_mse_per_output = mean_mse_per_output[valid_mask]
    std_mse_per_output = std_mse_per_output[valid_mask]
    window_sizes = window_sizes[valid_mask]
    
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
        
        # Apply the same filtering mask
        mean_prediction_variance_per_output = mean_prediction_variance_per_output[valid_mask]
        std_prediction_variance_per_output = std_prediction_variance_per_output[valid_mask]
    
    print(f"\nResults Summary:")
    print(f"  Number of iterations (after filtering): {len(iterations)}")
    print(f"  Number of outputs (K values): {n_outputs}")
    print(f"  Total samples seen: {total_samples_seen[-1]:.0f}")
    print(f"  Initial MSE: {mean_total_mse[0]:.6e} ± {std_total_mse[0]:.6e}")
    print(f"  Final MSE: {mean_total_mse[-1]:.6e} ± {std_total_mse[-1]:.6e}")
    print(f"  Improvement: {mean_total_mse[0] / mean_total_mse[-1]:.2f}x")
    if has_variance:
        print(f"  Prediction variance tracking: Yes")
    
    # Load comparison data if provided
    comparison_datasets = []
    if comparison_files is not None and len(comparison_files) > 0:
        for comp_file in comparison_files:
            comparison_path = Path(comp_file)
            if comparison_path.exists():
                print(f"\n📊 Loading comparison file: {comparison_path.name}")
                comp_results = load_pipeline_results(comparison_path)
                comp_config = comp_results['config']
                comp_agg = comp_results['aggregated_results']
                
                comp_samples_per_iter = comp_config['samples_per_iteration']
                comp_shrink_rate = comp_config['shrink_rate']
                comp_num_seeds = comp_config.get('num_seeds', 1)
                
                # Filter: only include runs with at least 4 seeds
                if comp_num_seeds < 4:
                    print(f"  ⚠️  Skipping: Only {comp_num_seeds} seed(s), need at least 4")
                    continue
                
                comp_samples_seen = []
                comp_mse_mean = []
                comp_mse_std = []
                comp_variance_mean = []
                comp_variance_std = []
                
                for result in comp_agg:
                    comp_samples_seen.append(result['total_samples_seen'])
                    comp_mse_mean.append(result['mean_mse_per_output'][0] if n_outputs == 1 
                                        else result['mean_total_mse'])
                    comp_mse_std.append(result['std_mse_per_output'][0] if n_outputs == 1 
                                       else result['std_total_mse'])
                    
                    if has_variance and 'mean_prediction_variance_per_output' in result:
                        comp_variance_mean.append(result['mean_prediction_variance_per_output'][0] if n_outputs == 1 
                                                 else np.mean(result['mean_prediction_variance_per_output']))
                        comp_variance_std.append(result['std_prediction_variance_per_output'][0] if n_outputs == 1 
                                                else np.mean(result['std_prediction_variance_per_output']))
                
                # Convert to arrays for filtering
                comp_samples_seen = np.array(comp_samples_seen)
                comp_mse_mean = np.array(comp_mse_mean)
                comp_mse_std = np.array(comp_mse_std)
                comp_variance_mean = np.array(comp_variance_mean) if comp_variance_mean else None
                comp_variance_std = np.array(comp_variance_std) if comp_variance_std else None
                
                # Filter 1: Remove first point if it has 0 samples
                comp_valid_mask = comp_samples_seen > 0
                
                # Filter 2: Keep only iterations where samples increment is at least 1/3 of samples_per_iteration
                min_comp_samples_threshold = comp_samples_per_iter / 3.0
                
                for i in range(1, len(comp_samples_seen)):
                    if comp_valid_mask[i]:  # Only check if not already filtered
                        comp_samples_increment = comp_samples_seen[i] - comp_samples_seen[i-1]
                    if comp_samples_increment < min_comp_samples_threshold:
                        comp_valid_mask[i] = False
                
                # Apply filtering
                comp_samples_seen = comp_samples_seen[comp_valid_mask]
                comp_mse_mean = comp_mse_mean[comp_valid_mask]
                comp_mse_std = comp_mse_std[comp_valid_mask]
                if comp_variance_mean is not None:
                    comp_variance_mean = comp_variance_mean[comp_valid_mask]
                    comp_variance_std = comp_variance_std[comp_valid_mask]
                
                comparison_data = {
                    'samples_seen': comp_samples_seen,
                    'mse_mean': comp_mse_mean,
                    'mse_std': comp_mse_std,
                    'variance_mean': comp_variance_mean,
                    'variance_std': comp_variance_std,
                    'samples_per_iter': comp_samples_per_iter,
                    'shrink_rate': comp_shrink_rate,
                    'num_seeds': comp_num_seeds
                }
                
                comparison_datasets.append(comparison_data)
                
                print(f"  Comparison - Samples/iter: {comp_samples_per_iter}, Shrink rate: {comp_shrink_rate}, Seeds: {comp_num_seeds}")
                print(f"  Comparison - Points after filtering: {len(comp_samples_seen)}")
                print(f"  Comparison - Final MSE: {comp_mse_mean[-1]:.6e} ± {comp_mse_std[-1]:.6e}")
            else:
                print(f"⚠️  Comparison file not found: {comp_file}")

    
    # Determine reaction label for titles
    reaction_label = f"Reaction {reaction_idx}" if reaction_idx is not None else "Single K"
    
    # ============================================================
    # SEPARATE FIGURE 1: MSE vs Samples Seen
    # ============================================================
    fig_mse = plt.figure(figsize=(12, 8))
    ax_mse = plt.subplot(1, 1, 1)
    
    if n_outputs == 1:
        # Single K value - plot with filled variance
        mse_values = mean_mse_per_output[:, 0]
        mse_std = std_mse_per_output[:, 0]
        
        # Baseline shrink percentage 
        # shrink_rate means the window size multiplies by (1 - shrink_rate) each iteration
        # So shrink_rate=0.2 means window becomes 80% of previous size each iteration
        baseline_remaining_pct = int((1 - shrink_rate) * 100)
        
        ax_mse.plot(total_samples_seen, mse_values, 
                marker='o', linewidth=2.5, markersize=8,
                label=f'Baseline ({baseline_remaining_pct}% shrink/iter)', alpha=0.9, color='#2E86AB')
        ax_mse.fill_between(total_samples_seen, 
                        mse_values - mse_std,
                        mse_values + mse_std,
                        alpha=0.3, color='#2E86AB')
        
        # Define colors for comparison datasets
        comp_colors = ['#E63946', '#F77F00', '#06AED5', '#2A9D8F', '#E76F51', '#264653']
        comp_markers = ['s', '^', 'D', 'v', 'P', '*']
        
        # Add comparison datasets if available
        if len(comparison_datasets) > 0:
            for idx, comp_data in enumerate(comparison_datasets):
                color = comp_colors[idx % len(comp_colors)]
                marker = comp_markers[idx % len(comp_markers)]
                # Window becomes (1 - shrink_rate) * 100% of previous size each iteration
                remaining_pct = int((1 - comp_data['shrink_rate']) * 100)
                
                ax_mse.plot(comp_data['samples_seen'], comp_data['mse_mean'],
                        marker=marker, linewidth=2.5, markersize=8,
                        label=f'{comp_data["samples_per_iter"]} samp/iter, {remaining_pct}% shrink/iter', 
                        alpha=0.9, color=color)
                ax_mse.fill_between(comp_data['samples_seen'],
                                comp_data['mse_mean'] - comp_data['mse_std'],
                                comp_data['mse_mean'] + comp_data['mse_std'],
                                alpha=0.3, color=color)
        
        ax_mse.set_xlabel('Samples Seen', fontsize=14, fontweight='bold')
        ax_mse.set_ylabel('MSE', fontsize=14, fontweight='bold')
        title_text = f'{reaction_label} - Mean Squared Error'
        if len(comparison_datasets) > 0:
            title_text += f'\nBaseline vs Adaptive Sampling'
        ax_mse.set_title(title_text, fontsize=15, fontweight='bold')
    else:
        # Multiple K values - plot total MSE
        ax_mse.plot(total_samples_seen, mean_total_mse,
                marker='o', linewidth=2.5, markersize=8,
                label='Total MSE (Mean)', alpha=0.9, color='#2E86AB')
        ax_mse.fill_between(total_samples_seen,
                        mean_total_mse - std_total_mse,
                        mean_total_mse + std_total_mse,
                        alpha=0.3, color='#2E86AB', label='±1 Std Dev')
        
        ax_mse.set_xlabel('Samples Seen', fontsize=14, fontweight='bold')
        ax_mse.set_ylabel('Total MSE (Sum across K values)', fontsize=14, fontweight='bold')
        ax_mse.set_title(f'{reaction_label} - Mean Squared Error\n'
                        f'Window: {initial_window}, Shrink Rate: {shrink_rate}', 
                        fontsize=15, fontweight='bold')
    
    ax_mse.set_yscale('log')
    ax_mse.legend(loc='best', fontsize=11, framealpha=0.95)
    ax_mse.grid(True, alpha=0.3, which='both', linestyle='--')
    ax_mse.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    # Save MSE plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    react_suffix = f"_react{reaction_idx}" if reaction_idx is not None else ""
    
    mse_pdf_path = output_dir / f'mse{react_suffix}_{timestamp}.pdf'
    fig_mse.savefig(mse_pdf_path, bbox_inches='tight', dpi=300)
    print(f"\n📊 Saved MSE plot to: {mse_pdf_path}")
    
    mse_png_path = output_dir / f'mse{react_suffix}_{timestamp}.png'
    fig_mse.savefig(mse_png_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved MSE PNG to: {mse_png_path}")
    
    plt.show()
    plt.close(fig_mse)
    
    # ============================================================
    # SEPARATE FIGURE 2: Prediction Variance vs Samples Seen
    # ============================================================
    if has_variance:
        fig_var = plt.figure(figsize=(12, 8))
        ax_var = plt.subplot(1, 1, 1)
        
        if n_outputs == 1:
            # Single K value
            variance_values = mean_prediction_variance_per_output[:, 0]
            variance_std = std_prediction_variance_per_output[:, 0]
            
            # Filter out NaN values
            valid_mask = ~np.isnan(variance_values)
            valid_samples = total_samples_seen[valid_mask]
            valid_variance = variance_values[valid_mask]
            valid_std = variance_std[valid_mask]
            
            # Define colors for comparison datasets (different from MSE plot)
            var_colors = ['#F77F00', '#06AED5', '#2A9D8F', '#E76F51', '#264653', '#E63946']
            var_markers = ['s', '^', 'D', 'v', 'P', '*']
            
            if len(valid_samples) > 0:
                baseline_remaining_pct = int((1 - shrink_rate) * 100)
                
                ax_var.plot(valid_samples, valid_variance,
                        marker='o', linewidth=2.5, markersize=8,
                        label=f'Baseline ({baseline_remaining_pct}% shrink/iter)', alpha=0.9, color='#E63946')
                ax_var.fill_between(valid_samples,
                                valid_variance - valid_std,
                                valid_variance + valid_std,
                                alpha=0.3, color='#E63946')
                
                # Add comparison datasets if available
                if len(comparison_datasets) > 0:
                    for idx, comp_data in enumerate(comparison_datasets):
                        if comp_data['variance_mean'] is not None:
                            color = var_colors[idx % len(var_colors)]
                            marker = var_markers[idx % len(var_markers)]
                            remaining_pct = int((1 - comp_data['shrink_rate']) * 100)
                            
                            comp_var_mask = ~np.isnan(comp_data['variance_mean'])
                            comp_valid_samples = comp_data['samples_seen'][comp_var_mask]
                            comp_valid_var = comp_data['variance_mean'][comp_var_mask]
                            comp_valid_std = comp_data['variance_std'][comp_var_mask]
                            
                            if len(comp_valid_samples) > 0:
                                ax_var.plot(comp_valid_samples, comp_valid_var,
                                        marker=marker, linewidth=2.5, markersize=8,
                                        label=f'{comp_data["samples_per_iter"]} samp/iter, {remaining_pct}% shrink/iter',
                                        alpha=0.9, color=color)
                                ax_var.fill_between(comp_valid_samples,
                                                comp_valid_var - comp_valid_std,
                                                comp_valid_var + comp_valid_std,
                                                alpha=0.3, color=color)
                
                ax_var.set_xlabel('Samples Seen', fontsize=14, fontweight='bold')
                ax_var.set_ylabel('Prediction Variance', fontsize=14, fontweight='bold')
                title_text = f'{reaction_label} - Model Prediction Variance'
                if len(comparison_datasets) > 0:
                    title_text += f'\nBaseline vs Adaptive Sampling'
                ax_var.set_title(title_text, fontsize=15, fontweight='bold')
                ax_var.set_yscale('log')
                ax_var.legend(loc='best', fontsize=11, framealpha=0.95)
                ax_var.grid(True, alpha=0.3, which='both', linestyle='--')
                ax_var.tick_params(axis='both', which='major', labelsize=12)
        else:
            # Multiple K values - plot per output
            colors = ['#E63946', '#F77F00', '#06AED5']
            output_labels = [f'K{i+1}' for i in range(n_outputs)]
            
            for i in range(n_outputs):
                variance_values = mean_prediction_variance_per_output[:, i]
                variance_std = std_prediction_variance_per_output[:, i]
                
                # Filter out NaN values
                valid_mask = ~np.isnan(variance_values)
                valid_samples = total_samples_seen[valid_mask]
                valid_variance = variance_values[valid_mask]
                valid_std = variance_std[valid_mask]
                
                if len(valid_samples) > 0:
                    ax_var.plot(valid_samples, valid_variance,
                            marker='o', linewidth=2.5, markersize=8,
                            label=output_labels[i], alpha=0.9, color=colors[i % len(colors)])
                    ax_var.fill_between(valid_samples,
                                    valid_variance - valid_std,
                                    valid_variance + valid_std,
                                    alpha=0.2, color=colors[i % len(colors)])
            
            ax_var.set_xlabel('Samples Seen', fontsize=14, fontweight='bold')
            ax_var.set_ylabel('Prediction Variance', fontsize=14, fontweight='bold')
            ax_var.set_title(f'{reaction_label} - Model Prediction Variance per K\n'
                            f'Window: {initial_window}, Shrink Rate: {shrink_rate}', 
                            fontsize=15, fontweight='bold')
            ax_var.set_yscale('log')
            ax_var.legend(loc='best', fontsize=11, framealpha=0.95)
            ax_var.grid(True, alpha=0.3, which='both', linestyle='--')
            ax_var.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        
        # Save Variance plot
        var_pdf_path = output_dir / f'variance{react_suffix}_{timestamp}.pdf'
        fig_var.savefig(var_pdf_path, bbox_inches='tight', dpi=300)
        print(f"📊 Saved Variance plot to: {var_pdf_path}")
        
        var_png_path = output_dir / f'variance{react_suffix}_{timestamp}.png'
        fig_var.savefig(var_png_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved Variance PNG to: {var_png_path}")
        
        plt.show()
        plt.close(fig_var)
    
    # ============================================================
    # Print detailed results table
    # ============================================================
    print("\n" + "="*90)
    print("Detailed Results by Samples Seen")
    print("="*90)
    
    output_labels = [f'K{i+1}' for i in range(n_outputs)]
    
    print(f"{'Iter':<6} {'Samples':<10} {'Window':<10} {'Total MSE':<15} {'Std Err':<12}", end='')
    for i in range(n_outputs):
        print(f"  {output_labels[i]:<12}", end='')
    if has_variance:
        print(f"  {'Variance':<12}", end='')
    print()
    print("-"*90)
    
    for idx in range(len(iterations)):
        print(f"{iterations[idx]:<6} "
              f"{total_samples_seen[idx]:<10.0f} "
              f"{window_sizes[idx]:<10.4f} "
              f"{mean_total_mse[idx]:<15.6e} "
              f"{std_total_mse[idx]:<12.6e}", end='')
        for i in range(n_outputs):
            print(f"  {mean_mse_per_output[idx, i]:<12.6e}", end='')
        if has_variance and n_outputs == 1:
            variance_val = mean_prediction_variance_per_output[idx, 0]
            if not np.isnan(variance_val):
                print(f"  {variance_val:<12.6e}", end='')
            else:
                print(f"  {'N/A':<12}", end='')
        print()
    
    print("="*90)


def main():
    """Main function to run the plotting script."""
    
    # List of result files for each reaction
    result_files = [
        {
            'path': 'pipeline_results/Kprediction_runs/react_0_adaptive_batch_sampling_w1.0_s1_e50_20251110_201058.json',
            'reaction_idx': 0,
            'comparisons': [
                'pipeline_results/sample_efficiency_700per_iter_shrink0.15_20251112_111259.json',
                'pipeline_results/sample_efficiency_300per_iter_shrink0.2_20251112_113225.json',
                'pipeline_results/sample_efficiency_200per_iter_shrink0.3_20251112_115915.json'
            ]
        },
        {
            'path': 'pipeline_results/Kprediction_runs/react_1_adaptive_batch_sampling_w1.0_s1_e50_20251110_214803.json',
            'reaction_idx': 1,
            'comparisons': ['pipeline_results/sample_efficiency_400per_iter_shrink0.4_20251112_133210.json',
                            'pipeline_results/sample_efficiency_400per_iter_shrink0.3_20251112_134233.json',
                            'pipeline_results/sample_efficiency_200per_iter_shrink0.3_20251112_135155.json']
        },
        {
            'path': 'pipeline_results/Kprediction_runs/react_2_adaptive_batch_sampling_w1.0_s1_e50_20251110_235744.json',
            'reaction_idx': 2,
            'comparisons': ['pipeline_results/sample_efficiency_500per_iter_shrink0.2_20251112_141543.json',
                            'pipeline_results/sample_efficiency_800per_iter_shrink0.5_20251112_150159.json',
                            'pipeline_results/sample_efficiency_800per_iter_shrink0.4_20251112_151343.json']
        }
    ]
    
    print("="*70)
    print("Processing Multiple Reaction Results")
    print("="*70)
    
    # Process each result file
    for file_info in result_files:
        result_file = file_info['path']
        reaction_idx = file_info['reaction_idx']
        comparison_files = file_info.get('comparisons', [])
        
        print(f"\n{'='*70}")
        print(f"Processing Reaction {reaction_idx}")
        print(f"{'='*70}")
        
        if not Path(result_file).exists():
            print(f"⚠️  Result file not found: {result_file}")
            continue
        
        # Create plots for this reaction
        plot_single_k_results(result_file, reaction_idx=reaction_idx, 
                            comparison_files=comparison_files)
    
    print("\n" + "="*70)
    print("✓ All plots completed!")
    print("="*70)


if __name__ == "__main__":
    main()

