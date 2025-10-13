#!/usr/bin/env python3
"""
K-Centered Adaptive Learning Results Plotter

This script loads multiple K-centered adaptive learning result files and creates
comprehensive plots showing performance across different box sizes and shrink rates.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict
import seaborn as sns

def extract_params_from_filename(filename):
    """
    Extract box size and shrink rate from filename.
    Expected format: k_centered_adaptive_multi_seed_results_box{X}_shrink{Y}.json
    """
    match = re.search(r'box(\d+)_shrink(\d+)', filename)
    if match:
        box_size = int(match.group(1))
        shrink_rate = int(match.group(2))
        return box_size, shrink_rate
    return None, None

def load_k_centered_results(result_files):
    """
    Load and organize K-centered adaptive learning results by box size and shrink rate.
    """
    results_by_config = {}
    
    for file_path in result_files:
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue
            
        filename = os.path.basename(file_path)
        box_size, shrink_rate = extract_params_from_filename(filename)
        
        if box_size is None or shrink_rate is None:
            print(f"⚠️  Could not extract parameters from: {filename}")
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            config_key = f"box{box_size}_shrink{shrink_rate}"
            results_by_config[config_key] = {
                'box_size': box_size,
                'shrink_rate': shrink_rate,
                'averaged_results': data['averaged_results'],
                'summary': data['summary'],
                'file_path': file_path
            }
            
            print(f"✅ Loaded {config_key}: {len(data['averaged_results'])} iterations")
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            
    return results_by_config

def create_k_centered_plots(results_by_config):
    """
    Create two separate Test MSE plots for K-centered adaptive learning results.
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('K-Centered Adaptive Learning: Test MSE Performance', fontsize=16, fontweight='bold')
    
    # Separate configs by fixed parameter
    shrink100_configs = []
    box100_configs = []
    
    for config, data in results_by_config.items():
        if data['shrink_rate'] == 100:
            shrink100_configs.append((config, data))
        if data['box_size'] == 100:
            box100_configs.append((config, data))
    
    # Plot 1: Different Box Sizes (Shrink Rate = 100)
    colors1 = plt.cm.tab10(np.linspace(0, 1, len(shrink100_configs)))
    
    for i, (config, data) in enumerate(sorted(shrink100_configs)):
        # Skip first point (iteration 0) and final point with 1 sample added
        filtered_results = [r for r in data['averaged_results'] if r['iteration'] > 0 and r['batch_samples_added'] != 1]
        samples = [r['total_samples'] for r in filtered_results]
        test_mse = [r['test_total_mse'] for r in filtered_results]
        
        # Convert box size to K_true range description with box size
        if data['box_size'] == 50:
            box_label = "0.75×K_true to 1.33×K_true (Box Size 50%)"
        elif data['box_size'] == 75:
            box_label = "0.625×K_true to 1.6×K_true (Box Size 75%)"
        elif data['box_size'] == 90:
            box_label = "0.55×K_true to 1.8×K_true (Box Size 90%)"
        elif data['box_size'] == 100:
            box_label = "0.5×K_true to 2×K_true (Box Size 100%)"
        else:
            box_label = f"Box {data['box_size']}%"
        
        ax1.plot(samples, test_mse, 'o-', color=colors1[i], 
                label=box_label, 
                linewidth=2, markersize=6)
    
    ax1.set_xlabel('Total Training Samples')
    ax1.set_ylabel('Test MSE')
    ax1.set_title('Different Box Sizes (No Shrinking)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Different Shrink Rates (Box Size = 100)
    colors2 = plt.cm.tab10(np.linspace(0, 1, len(box100_configs)))
    
    for i, (config, data) in enumerate(sorted(box100_configs)):
        # Skip first point (iteration 0) and final point with 1 sample added
        filtered_results = [r for r in data['averaged_results'] if r['iteration'] > 0 and r['batch_samples_added'] != 1]
        samples = [r['total_samples'] for r in filtered_results]
        test_mse = [r['test_total_mse'] for r in filtered_results]
        
        # Convert shrink rate to actual shrinking percentage (inverted)
        actual_shrink = 100 - data['shrink_rate']
        shrink_label = f"{actual_shrink}% Shrinking"
        
        ax2.plot(samples, test_mse, 'o-', color=colors2[i], 
                label=shrink_label, 
                linewidth=2, markersize=6)
    
    ax2.set_xlabel('Total Training Samples')
    ax2.set_ylabel('Test MSE')
    ax2.set_title('Different Shrink Rates (0.5×K_true to 2×K_true)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Save the plot
    output_file = 'results/k_centered_results/k_centered_performance_comparison.png'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Performance comparison plot saved: {output_file}")
    
    plt.show()

def print_results_table(results_by_config):
    """
    Print formatted results table similar to the desired output format.
    """
    print("\n" + "="*100)
    print("📊 K-CENTERED ADAPTIVE LEARNING RESULTS COMPARISON")
    print("="*100)
    
    for config in sorted(results_by_config.keys()):
        data = results_by_config[config]
        print(f"\n🎯 Configuration: Box Size {data['box_size']}%, Shrink Rate {data['shrink_rate']}%")
        print(f"{'Iter':>4} {'Samples':>8} {'Added':>6} {'Test MSE':>10} {'K Error':>10} {'K Improve':>10} {'Box Factor':>11}")
        print("-" * 72)
        
        for result in data['averaged_results']:
            iter_num = result['iteration']
            total_samples = result['total_samples']
            added = result['batch_samples_added']
            test_mse = result['test_total_mse']
            k_error = result['k_prediction_error']
            k_improve = result.get('k_improvement_pct', 0.0)
            box_factor = result.get('box_size_factor', 'N/A')
            
            if isinstance(box_factor, (int, float)):
                box_str = f"{box_factor:.3f}"
            else:
                box_str = str(box_factor)
                
            print(f"{iter_num:>4} {total_samples:>8} {added:>6} {test_mse:>10.6f} {k_error:>10.6f} {k_improve:>+9.1f}% {box_str:>11}")
        
        # Print summary
        summary = data['summary']
        print(f"\n📈 Summary:")
        print(f"   Initial K Error: {summary['initial_k_error']:.6f}")
        print(f"   Final K Error:   {summary['final_k_error']:.6f}")
        print(f"   Total K Improvement: {summary['k_improvement_total']:.1f}%")
        print(f"   Final Test MSE: {summary['final_test_mse']:.2e}")
        print(f"   Total Samples: {summary['total_training_samples']}")

def main():
    """
    Main function to run the K-centered results analysis.
    """
    print("🚀 K-Centered Adaptive Learning Results Analysis")
    
    # Define result files
    result_files = [
        "results/k_centered_results/k_centered_adaptive_multi_seed_results_box50_shrink100.json",
        "results/k_centered_results/k_centered_adaptive_multi_seed_results_box75_shrink100.json", 
        "results/k_centered_results/k_centered_adaptive_multi_seed_results_box90_shrink100.json",
        "results/k_centered_results/k_centered_adaptive_multi_seed_results_box100_shrink50.json",
        "results/k_centered_results/k_centered_adaptive_multi_seed_results_box100_shrink60.json",
        "results/k_centered_results/k_centered_adaptive_multi_seed_results_box100_shrink70.json",
        "results/k_centered_results/k_centered_adaptive_multi_seed_results_box100_shrink85.json",
        "results/k_centered_results/k_centered_adaptive_multi_seed_results_box100_shrink100.json",
        "results/k_centered_results/sample_efficiency_results_20250928_155718.json"
    ]
    
    # Load results
    print("\n📂 Loading K-centered adaptive learning results...")
    results_by_config = load_k_centered_results(result_files)
    
    if not results_by_config:
        print("❌ No valid results found!")
        return
    
    print(f"\n✅ Loaded {len(results_by_config)} configurations")
    
    # Print detailed results table
    print_results_table(results_by_config)
    
    # Create plots
    print("\n📊 Creating performance comparison plots...")
    create_k_centered_plots(results_by_config)
    
    print("\n✅ K-Centered Results Analysis Complete!")

if __name__ == "__main__":
    main()