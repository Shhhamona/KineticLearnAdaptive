"""
Calculate window sizes and batch file sampling for iterative adaptive sampling.

This script helps visualize which batch files will be sampled at each iteration
as the window shrinks around a center K value.
"""

import numpy as np
import matplotlib.pyplot as plt

# Batch file information
batch_files = [
    {
        'name': 'Batch 1',
        'file': 'batch_4000sims_20250827_010028.json',
        'n_samples': 4000,
        'k_multiplier': 1.0,
        'k_multiplier_str': 'K × 1'
    },
    {
        'name': 'Batch 2+4',
        'files': ['batch_1000sims_20250928_191628.json', 'batch_2500sims_20250929_031845.json'],
        'n_samples': 1000 + 2500,
        'k_multiplier': 1.15,
        'k_multiplier_str': 'K × 1.15'
    },
    {
        'name': 'Batch 6',
        'file': 'batch_2000sims_20250929_205429.json',
        'n_samples': 2000,
        'k_multiplier': 1.005,
        'k_multiplier_str': 'K × 1.005'
    },
    {
        'name': 'Batch 3',
        'file': 'batch_1500sims_20250928_224858.json',
        'n_samples': 1500,
        'k_multiplier': 1.0005,
        'k_multiplier_str': 'K × 1.0005'
    },
    {
        'name': 'Batch 5',
        'file': 'batch_2000sims_20250929_125706.json',
        'n_samples': 2000,
        'k_multiplier': 1.00005,
        'k_multiplier_str': 'K × 1.00005'
    }
]

def calculate_window_bounds(k_center, window_size):
    """
    Calculate window bounds using multiplicative window.
    
    window_size = 1.0 means: [k_center/2, k_center*2]
    window_size = 0.5 means: [k_center/1.5, k_center*1.5]
    
    Args:
        k_center: Center K value
        window_size: Window size factor
        
    Returns:
        (k_min, k_max): Window bounds
    """
    factor = 1.0 + window_size
    k_min = k_center / factor
    k_max = k_center * factor
    return k_min, k_max

def check_batch_in_window(batch_k, k_min, k_max):
    """
    Check if batch K value falls within window.
    
    Args:
        batch_k: Batch K value (multiplier of true K)
        k_min: Window minimum
        k_max: Window maximum
        
    Returns:
        True if batch is within window
    """
    return k_min <= batch_k <= k_max

def calculate_iterations(initial_window_size=1.0, shrink_rate=0.5, n_iterations=10, 
                        k_center=1.0, samples_per_iteration=100):
    """
    Calculate window sizes and batch sampling for each iteration with sample consumption.
    
    Args:
        initial_window_size: Initial window size factor (1.0 = ±100%)
        shrink_rate: Factor to shrink window each iteration (0.5 = 50% reduction)
        n_iterations: Number of iterations
        k_center: Center K value (normalized to 1.0)
        samples_per_iteration: Number of samples to retrieve per iteration
        
    Returns:
        List of iteration results
    """
    results = []
    
    # Track remaining samples in each batch (deep copy to avoid modifying original)
    batch_remaining = {batch['name']: batch['n_samples'] for batch in batch_files}
    
    print("="*80)
    print("ADAPTIVE SAMPLING WINDOW ANALYSIS WITH SAMPLE CONSUMPTION")
    print("="*80)
    print(f"Initial window size: {initial_window_size}")
    print(f"Shrink rate: {shrink_rate} ({(1-shrink_rate)*100:.0f}% reduction per iteration)")
    print(f"Number of iterations: {n_iterations}")
    print(f"Samples per iteration: {samples_per_iteration}")
    print(f"K center: {k_center}")
    print("="*80)
    print()
    
    for iteration in range(n_iterations):
        # Calculate adaptive window size
        window_size = initial_window_size * (shrink_rate ** iteration)
        
        # Calculate window bounds
        k_min, k_max = calculate_window_bounds(k_center, window_size)
        
        # Window as percentage
        window_percent = window_size * 100
        
        # Check which batches fall within window AND have samples remaining
        batches_in_window = []
        batch_details = []
        total_samples_available = 0
        
        for batch in batch_files:
            batch_k = batch['k_multiplier'] * k_center
            remaining = batch_remaining[batch['name']]
            
            if check_batch_in_window(batch_k, k_min, k_max) and remaining > 0:
                batches_in_window.append(batch['name'])
                batch_details.append({
                    'name': batch['name'],
                    'k_multiplier_str': batch['k_multiplier_str'],
                    'k_value': batch_k,
                    'remaining': remaining
                })
                total_samples_available += remaining
        
        # Calculate how many samples to take from each batch (uniform distribution)
        samples_taken = {}
        if total_samples_available > 0 and len(batches_in_window) > 0:
            samples_to_take = min(samples_per_iteration, total_samples_available)
            
            # Sequential/hierarchical sampling: prioritize batches in order
            # Only move to next batch when current one is exhausted or outside window
            samples_needed = samples_to_take
            
            for detail in batch_details:
                batch_name = detail['name']
                batch_available = detail['remaining']
                
                if samples_needed <= 0:
                    samples_taken[batch_name] = 0
                    continue
                
                # Take as many as possible from this batch (up to what we need)
                samples_from_batch = min(samples_needed, batch_available)
                
                samples_taken[batch_name] = samples_from_batch
                batch_remaining[batch_name] -= samples_from_batch
                samples_needed -= samples_from_batch
                
                # If this batch satisfied all our needs, we're done
                if samples_needed == 0:
                    break
            
            # Fill remaining batches not sampled with 0
            for detail in batch_details:
                if detail['name'] not in samples_taken:
                    samples_taken[detail['name']] = 0
        
        result = {
            'iteration': iteration,
            'window_size': window_size,
            'window_percent': window_percent,
            'k_min': k_min,
            'k_max': k_max,
            'batches_in_window': batches_in_window,
            'n_batches_available': len(batches_in_window),
            'total_samples_available': total_samples_available,
            'samples_taken': samples_taken,
            'total_samples_taken': sum(samples_taken.values()),
            'batch_remaining': dict(batch_remaining)  # Snapshot of remaining samples
        }
        
        results.append(result)
        
        # Print iteration summary
        print(f"Iteration {iteration}")
        print(f"  Window size factor: {window_size:.6f} ({window_percent:.2f}%)")
        print(f"  K bounds: [{k_min:.6f}, {k_max:.6f}]")
        print(f"  Batch files in window: {len(batches_in_window)}")
        
        if batches_in_window:
            for detail in batch_details:
                batch_name = detail['name']
                taken = samples_taken.get(batch_name, 0)
                remaining_after = batch_remaining[batch_name]
                print(f"    ✓ {batch_name}: {detail['k_multiplier_str']} = {detail['k_value']:.6f}")
                print(f"      Samples taken: {taken}, Remaining: {remaining_after}")
        else:
            print(f"    ✗ No batches in window or all samples exhausted!")
        
        print(f"  Total samples in window: {total_samples_available}")
        print(f"  Total samples taken this iteration: {sum(samples_taken.values())}")
        print()
    
    return results, batch_remaining

def plot_window_evolution(results, batch_files):
    """
    Plot the evolution of window size and batch availability.
    """
    iterations = [r['iteration'] for r in results]
    window_percents = [r['window_percent'] for r in results]
    n_batches = [r['n_batches_available'] for r in results]
    total_samples = [r['total_samples_available'] for r in results]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Plot 1: Window size evolution
    axes[0].plot(iterations, window_percents, 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Window Size (%)', fontsize=12)
    axes[0].set_title('Window Size Evolution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot 2: Number of batches available
    axes[1].plot(iterations, n_batches, 'g-s', linewidth=2, markersize=8)
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Number of Batches', fontsize=12)
    axes[1].set_title('Batch Files Available per Iteration', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.5, len(batch_files) + 0.5)
    
    # Plot 3: Total samples available
    axes[2].plot(iterations, total_samples, 'r-^', linewidth=2, markersize=8)
    axes[2].set_xlabel('Iteration', fontsize=12)
    axes[2].set_ylabel('Total Samples', fontsize=12)
    axes[2].set_title('Total Samples Available per Iteration', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('batch_sampling_evolution.png', dpi=300, bbox_inches='tight')
    print("📊 Plot saved to: batch_sampling_evolution.png")
    plt.show()

def plot_batch_ranges(batch_files, results, k_center=1.0):
    """
    Plot batch K ranges and window evolution.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot batch K values as horizontal lines
    for i, batch in enumerate(batch_files):
        batch_k = batch['k_multiplier'] * k_center
        ax.plot([batch_k], [i], 'o', markersize=12, label=f"{batch['name']}: {batch['k_multiplier_str']}")
    
    # Plot window evolution
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for idx, result in enumerate(results):
        k_min = result['k_min']
        k_max = result['k_max']
        y_pos = -1 - idx * 0.3
        
        # Plot window range
        ax.plot([k_min, k_max], [y_pos, y_pos], 'o-', 
                color=colors[idx], linewidth=2, markersize=6,
                label=f"Iter {result['iteration']}: ±{result['window_percent']:.2f}%")
        
        # Fill between
        ax.axvspan(k_min, k_max, ymin=0, ymax=0.1, alpha=0.1, color=colors[idx])
    
    ax.axvline(k_center, color='black', linestyle='--', linewidth=2, label='K center', alpha=0.5)
    
    ax.set_xlabel('K Value (normalized)', fontsize=12)
    ax.set_ylabel('Batch Files / Iterations', fontsize=12)
    ax.set_title('Batch K Ranges and Window Evolution', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('batch_k_ranges.png', dpi=300, bbox_inches='tight')
    print("📊 Plot saved to: batch_k_ranges.png")
    plt.show()

def print_summary_table(results, final_batch_remaining):
    """Print a summary table of all iterations with sample consumption."""
    print("\n" + "="*100)
    print("SUMMARY TABLE WITH SAMPLE CONSUMPTION")
    print("="*100)
    print(f"{'Iter':<6} {'Window %':<12} {'K Min':<12} {'K Max':<12} {'Batches':<10} "
          f"{'Avail':<10} {'Taken':<10} {'Cumulative':<12}")
    print("-"*100)
    
    cumulative_taken = 0
    for r in results:
        cumulative_taken += r['total_samples_taken']
        print(f"{r['iteration']:<6} "
              f"{r['window_percent']:<12.4f} "
              f"{r['k_min']:<12.6f} "
              f"{r['k_max']:<12.6f} "
              f"{r['n_batches_available']:<10} "
              f"{r['total_samples_available']:<10} "
              f"{r['total_samples_taken']:<10} "
              f"{cumulative_taken:<12}")
    
    print("="*100)
    
    # Print final batch status
    print("\n" + "="*80)
    print("FINAL BATCH STATUS")
    print("="*80)
    print(f"{'Batch':<20} {'Initial':<12} {'Remaining':<12} {'Used':<12} {'Usage %':<12}")
    print("-"*80)
    
    total_initial = 0
    total_used = 0
    
    for batch in batch_files:
        batch_name = batch['name']
        initial = batch['n_samples']
        remaining = final_batch_remaining[batch_name]
        used = initial - remaining
        usage_pct = (used / initial * 100) if initial > 0 else 0
        
        total_initial += initial
        total_used += used
        
        print(f"{batch_name:<20} {initial:<12} {remaining:<12} {used:<12} {usage_pct:<12.1f}")
    
    overall_usage = (total_used / total_initial * 100) if total_initial > 0 else 0
    print("-"*80)
    print(f"{'TOTAL':<20} {total_initial:<12} {total_initial - total_used:<12} "
          f"{total_used:<12} {overall_usage:<12.1f}")
    print("="*80)

if __name__ == '__main__':
    # Configuration
    INITIAL_WINDOW_SIZE = 1.0  # 100% window (±100%)
    SHRINK_RATE = 0.5          # 50% reduction per iteration
    N_ITERATIONS = 10
    K_CENTER = 1.0             # Normalized center value
    
    print("\n🎯 Batch File Sampling Analysis\n")
    
    # Calculate iterations
    results, final_batch_remaining = calculate_iterations(
        initial_window_size=INITIAL_WINDOW_SIZE,
        shrink_rate=SHRINK_RATE,
        n_iterations=N_ITERATIONS,
        k_center=K_CENTER
    )
    
    # Print summary table
    print_summary_table(results, final_batch_remaining)
    
    # Generate plots
    print("\n📊 Generating plots...")
    plot_window_evolution(results, batch_files)
    plot_batch_ranges(batch_files, results, K_CENTER)
    
    print("\n✅ Analysis complete!")
    
    # Print key insights
    print("\n🔍 KEY INSIGHTS:")
    print(f"  - Total batch files: {len(batch_files)}")
    print(f"  - Total samples across all batches: {sum(b['n_samples'] for b in batch_files)}")
    print(f"  - K range across batches: [{min(b['k_multiplier'] for b in batch_files):.6f}, {max(b['k_multiplier'] for b in batch_files):.6f}]")
    
    # Find when we run out of batches
    for i, r in enumerate(results):
        if r['n_batches_available'] == 0:
            print(f"  - ⚠️  No batches available starting at iteration {i}")
            print(f"    Window size at failure: {r['window_percent']:.4f}%")
            print(f"    Window bounds: [{r['k_min']:.6f}, {r['k_max']:.6f}]")
            break
    else:
        print(f"  - ✓ All iterations have available batches")
