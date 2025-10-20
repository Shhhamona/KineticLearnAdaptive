"""
Plot Perturbation Analysis Results

This script loads perturbation analysis results and creates visualizations
showing prediction intervals vs true values across different dataset sizes.

Usage:
    python examples/plot_perturbation_results.py
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd


def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_prediction_intervals(json_path, output_dir='pipeline_results/plots'):
    """
    Create visualization showing prediction intervals and true values.
    
    Args:
        json_path: Path to the perturbation results JSON file
        output_dir: Directory to save plots
    """
    results = load_results(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract perturbation results
    if 'perturbation_results' not in results:
        print("❌ No perturbation results found in JSON file")
        return
    
    perturbation_data = results['perturbation_results']
    
    # Group by n_samples
    unique_n = sorted(list(set(r['n_samples'] for r in perturbation_data)))
    n_outputs = len(perturbation_data[0]['y_true'])
    
    # Create figure with subplots for each output
    fig, axes = plt.subplots(1, n_outputs, figsize=(18, 6))
    if n_outputs == 1:
        axes = [axes]
    
    output_names = ['K₁', 'K₂', 'K₃']
    colors = ['#2E86AB', "#9C3A6F", '#F18F01']
    
    for output_idx in range(n_outputs):
        ax = axes[output_idx]
        
        y_preds = []
        y_mins = []
        y_maxs = []
        y_trues = []
        
        for n in unique_n:
            results_for_n = [r for r in perturbation_data if r['n_samples'] == n]
            
            # Average across all test samples for this n
            y_pred_avg = np.mean([r['y_pred_mean'][output_idx] for r in results_for_n]) if 'y_pred_mean' in results_for_n[0] else None
            y_min_avg = np.mean([r['y_min'][output_idx] for r in results_for_n])
            y_max_avg = np.mean([r['y_max'][output_idx] for r in results_for_n])
            y_true_avg = np.mean([r['y_true'][output_idx] for r in results_for_n])
            
            y_preds.append(y_pred_avg)
            y_mins.append(y_min_avg)
            y_maxs.append(y_max_avg)
            y_trues.append(y_true_avg)
        
        y_preds = np.array(y_preds)
        y_mins = np.array(y_mins)
        y_maxs = np.array(y_maxs)
        y_trues = np.array(y_trues)
        
        # Calculate interval width
        interval_width = y_maxs - y_mins
        
        # Plot prediction interval as shaded region
        ax.fill_between(unique_n, y_mins, y_maxs, alpha=0.3, color=colors[output_idx],
                        label='Prediction Interval (±0.1% input)')
        
        # Plot bounds
        ax.plot(unique_n, y_mins, '--', color=colors[output_idx], linewidth=1.5, alpha=0.7)
        ax.plot(unique_n, y_maxs, '--', color=colors[output_idx], linewidth=1.5, alpha=0.7)
        
        # Plot predicted value (mean prediction at nominal input)
        if y_preds[0] is not None:
            ax.plot(unique_n, y_preds, '-', color=colors[output_idx], linewidth=2.5, marker='s', 
                   markersize=5, label=f'Predicted {output_names[output_idx]}', zorder=9)
        
        # Plot true value
        ax.plot(unique_n, y_trues, 'k-', linewidth=2.5, marker='o', markersize=5,
               label=f'True {output_names[output_idx]}', zorder=10)
        
        # Adjust y-axis limits to give space above the true K line
        y_min_plot = min(y_mins.min(), y_trues.min(), y_preds.min() if y_preds[0] is not None else y_mins.min())
        y_max_plot = max(y_maxs.max(), y_trues.max(), y_preds.max() if y_preds[0] is not None else y_maxs.max())
        y_range = y_max_plot - y_min_plot
        ax.set_ylim([y_min_plot - 0.05*y_range, y_max_plot + 0.15*y_range])
        
        # Styling
        ax.set_xlabel('Training Dataset Size (N)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{output_names[output_idx]} Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{output_names[output_idx]} Prediction Interval\n(Uniform Latin Hypercube Sampling)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)
        
        # Add text showing final interval width and bias
        final_width = interval_width[-1]
        final_rel_width = (final_width / y_trues[-1]) * 100 if y_trues[-1] != 0 else 0
        if y_preds[-1] is not None:
            final_bias = y_preds[-1] - y_trues[-1]
            final_rel_bias = (final_bias / y_trues[-1]) * 100 if y_trues[-1] != 0 else 0
            ax.text(0.02, 0.98, f'Final width: {final_width:.4e} ({final_rel_width:.2f}%)\nFinal bias: {final_bias:.4e} ({final_rel_bias:.2f}%)',
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                   fontsize=9)
        else:
            ax.text(0.02, 0.98, f'Final width: {final_width:.4e}\n({final_rel_width:.2f}%)',
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                   fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = output_dir / f'perturbation_intervals_{timestamp}.pdf'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"📊 Saved interval plot to: {plot_path}")
    
    plot_png_path = output_dir / f'perturbation_intervals_{timestamp}.png'
    plt.savefig(plot_png_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved PNG to: {plot_png_path}")
    
    plt.show()


def create_prediction_table(json_path, output_dir='pipeline_results/plots'):
    """
    Create a detailed table showing prediction intervals for each N and output.
    
    Args:
        json_path: Path to the perturbation results JSON file
        output_dir: Directory to save table
    """
    results = load_results(json_path)
    
    if 'perturbation_results' not in results:
        print("❌ No perturbation results found in JSON file")
        return
    
    perturbation_data = results['perturbation_results']
    
    # Group by n_samples
    unique_n = sorted(list(set(r['n_samples'] for r in perturbation_data)))
    n_outputs = len(perturbation_data[0]['y_true'])
    
    # Create data for table
    table_data = []
    
    for n in unique_n:
        results_for_n = [r for r in perturbation_data if r['n_samples'] == n]
        
        row = {'N': n}
        
        for output_idx in range(n_outputs):
            # Average across all test samples
            y_pred_avg = np.mean([r['y_pred_mean'][output_idx] for r in results_for_n]) if 'y_pred_mean' in results_for_n[0] else None
            y_min_avg = np.mean([r['y_min'][output_idx] for r in results_for_n])
            y_max_avg = np.mean([r['y_max'][output_idx] for r in results_for_n])
            y_true_avg = np.mean([r['y_true'][output_idx] for r in results_for_n])
            
            # Check if true value is contained
            contained = np.mean([r['containment'][output_idx] for r in results_for_n]) * 100
            
            interval_width = y_max_avg - y_min_avg
            
            row[f'K{output_idx+1}_pred'] = y_pred_avg
            row[f'K{output_idx+1}_min'] = y_min_avg
            row[f'K{output_idx+1}_max'] = y_max_avg
            row[f'K{output_idx+1}_true'] = y_true_avg
            row[f'K{output_idx+1}_width'] = interval_width
            row[f'K{output_idx+1}_contained'] = contained
        
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Print table
    print("\n" + "="*140)
    print("PREDICTION INTERVAL TABLE")
    print("="*140)
    print(f"{'N':<6}", end='')
    for i in range(n_outputs):
        print(f"| K{i+1} Pred {'':>8} | K{i+1} Interval {'':<25} | True K{i+1} {'':>8} | Contained? ", end='')
    print()
    print("-"*140)
    
    for _, row in df.iterrows():
        print(f"{int(row['N']):<6}", end='')
        for i in range(n_outputs):
            k_pred = row[f'K{i+1}_pred']
            k_min = row[f'K{i+1}_min']
            k_max = row[f'K{i+1}_max']
            k_true = row[f'K{i+1}_true']
            k_contained = row[f'K{i+1}_contained']
            
            # Check if contained
            contained_symbol = '✓' if k_contained == 100.0 else '✗'
            
            if k_pred is not None:
                print(f"| {k_pred:.5f} | [{k_min:.5f}, {k_max:.5f}] | {k_true:.5f} | {contained_symbol:^10} ", end='')
            else:
                print(f"| {'N/A':>8} | [{k_min:.5f}, {k_max:.5f}] | {k_true:.5f} | {contained_symbol:^10} ", end='')
        print()
    
    print("="*140)
    
    # Save to CSV
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f'perturbation_table_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved table to: {csv_path}")
    
    return df


def plot_containment_summary(json_path, output_dir='pipeline_results/plots'):
    """
    Create bar plot showing containment percentage for each output and N.
    
    Args:
        json_path: Path to the perturbation results JSON file
        output_dir: Directory to save plots
    """
    results = load_results(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'perturbation_results' not in results:
        print("❌ No perturbation results found in JSON file")
        return
    
    perturbation_data = results['perturbation_results']
    
    # Group by n_samples
    unique_n = sorted(list(set(r['n_samples'] for r in perturbation_data)))
    n_outputs = len(perturbation_data[0]['y_true'])
    
    # Calculate containment percentages
    containment_data = {f'K{i+1}': [] for i in range(n_outputs)}
    
    for n in unique_n:
        results_for_n = [r for r in perturbation_data if r['n_samples'] == n]
        
        for output_idx in range(n_outputs):
            contained_pct = np.mean([r['containment'][output_idx] for r in results_for_n]) * 100
            containment_data[f'K{output_idx+1}'].append(contained_pct)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(unique_n))
    width = 0.25
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, (k_name, values) in enumerate(containment_data.items()):
        offset = (i - n_outputs/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=k_name, color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height < 100:
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{height:.0f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Training Dataset Size (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Containment %', fontsize=12, fontweight='bold')
    ax.set_title('True Value Containment in Prediction Intervals', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(unique_n, rotation=45)
    ax.set_ylim([0, 110])
    ax.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='100% Target')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = output_dir / f'containment_summary_{timestamp}.pdf'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"📊 Saved containment plot to: {plot_path}")
    
    plot_png_path = output_dir / f'containment_summary_{timestamp}.png'
    plt.savefig(plot_png_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved PNG to: {plot_png_path}")
    
    plt.show()


def main():
    """Main function."""
    # Find the most recent perturbation result file
    results_dir = Path("pipeline_results")
    json_files = sorted(results_dir.glob("perturbation_results_*.json"))
    
    if not json_files:
        print("❌ No perturbation result files found in pipeline_results/")
        print("   Please run run_subset_pipeline_perturbation.py first!")
        return
    
    latest_file = json_files[-1]
    
    print("="*70)
    print("PERTURBATION ANALYSIS VISUALIZATION")
    print("="*70)
    print(f"Loading: {latest_file}\n")
    
    # Create all visualizations
    print("\n[1/3] Creating prediction interval plot...")
    plot_prediction_intervals(latest_file)
    
    print("\n[2/3] Creating prediction table...")
    create_prediction_table(latest_file)
    
    print("\n[3/3] Creating containment summary...")
    plot_containment_summary(latest_file)
    
    print("\n✅ All visualizations complete!")


if __name__ == "__main__":
    main()
