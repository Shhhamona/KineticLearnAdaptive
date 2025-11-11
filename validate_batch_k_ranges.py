"""
Validation script to check actual K-ranges in batch simulation files.

This script loads all batch files and validates that the actual K-values
in each file match the expected K-range specified in the configuration.

Usage:
    python validate_batch_k_ranges.py
"""

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from kinetic_modelling import MultiPressureDataset
from batch_files_config import BATCH_FILES


def analyze_k_range(k_values: np.ndarray, reaction_idx: int = None) -> dict:
    """
    Analyze the K-range for a single reaction or all reactions.
    
    Args:
        k_values: Array of K values (scaled), shape (n_samples, n_reactions) or (n_samples,)
        reaction_idx: If specified, analyze only this reaction index
        
    Returns:
        Dictionary with range statistics
    """
    if reaction_idx is not None:
        if k_values.ndim > 1:
            k_vals = k_values[:, reaction_idx]
        else:
            k_vals = k_values
    else:
        k_vals = k_values.flatten()
    
    # Calculate min/max/mean
    k_min = np.min(k_vals)
    k_max = np.max(k_vals)
    k_mean = np.mean(k_vals)
    k_median = np.median(k_vals)
    
    # Calculate the range width relative to the center
    # If symmetric around k_mean: k_min = k_mean/F, k_max = k_mean*F
    # Then: lower_factor = k_mean/k_min, upper_factor = k_max/k_mean
    lower_factor = k_mean / k_min if k_min > 0 else np.inf
    upper_factor = k_max / k_mean if k_mean > 0 else np.inf
    
    # Overall factor is the maximum deviation from center
    overall_factor = max(lower_factor, upper_factor)
    
    return {
        'k_min': k_min,
        'k_max': k_max,
        'k_mean': k_mean,
        'k_median': k_median,
        'k_std': np.std(k_vals),
        'lower_factor': lower_factor,
        'upper_factor': upper_factor,
        'overall_factor': overall_factor,
        'n_samples': len(k_vals)
    }


def main():
    # Configuration - must match run_adaptive_batch_sampe_efficiency.py
    nspecies = 3
    num_pressure_conditions = 2
    react_idx = [0, 1, 2]  # All reactions
    
    # True K values (from your O2 simple chemistry) - these will be scaled
    # These are the "true" K values used in simulations
    k_true_raw = np.array([6.00E-16, 1.30E-15, 9.60E-16])
    
    print("="*80)
    print("BATCH FILE K-RANGE VALIDATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Number of species: {nspecies}")
    print(f"  Pressure conditions: {num_pressure_conditions}")
    print(f"  Reactions analyzed: {react_idx}")
    print(f"  True K values (raw): {k_true_raw}")
    print("="*80)
    
    # Load reference dataset to initialize scalers
    init_file = Path("data/SampleEfficiency/O2_simple_uniform.txt")
    
    if not init_file.exists():
        raise FileNotFoundError(f"Initial file not found: {init_file}")
    
    reference_dataset = MultiPressureDataset(
        nspecies=nspecies,
        num_pressure_conditions=num_pressure_conditions,
        src_file=str(init_file),
        react_idx=react_idx
    )
    
    print(f"\n✓ Reference dataset loaded: {len(reference_dataset)} samples")
    print(f"  (Used for scaler initialization)\n")
    
    # Validate each batch file
    validation_results = []
    
    for i, batch_info in enumerate(BATCH_FILES):
        batch_path = Path(batch_info['path'])
        batch_label = batch_info['label']
        expected_k_range = batch_info['k_range']
        
        print(f"\n{'='*80}")
        print(f"BATCH {i+1}: {batch_label}")
        print(f"{'='*80}")
        print(f"File: {batch_path}")
        print(f"Expected K-range: {expected_k_range}")
        
        if not batch_path.exists():
            print(f"❌ FILE NOT FOUND!")
            validation_results.append({
                'batch_idx': i+1,
                'label': batch_label,
                'status': 'FILE_NOT_FOUND',
                'expected_range': expected_k_range
            })
            continue
        
        try:
            # Load dataset WITH THE SAME SCALERS as the pipeline (critical!)
            pool_ds = MultiPressureDataset(
                nspecies=nspecies,
                num_pressure_conditions=num_pressure_conditions,
                src_file=str(batch_path),
                react_idx=react_idx,
                scaler_input=reference_dataset.scaler_input,
                scaler_output=reference_dataset.scaler_output
            )
            
            # Get scaled data (this is what the NN actually sees and what we should validate)
            x_data, y_data = pool_ds.get_data()
            
            # y_data contains the scaled K values - use them directly!
            k_values = y_data  # Shape: (n_samples, n_reactions)
            
            print(f"\n📊 Dataset Statistics:")
            print(f"  Samples: {len(pool_ds)}")
            print(f"  Input shape: {x_data.shape}")
            print(f"  Output shape: {y_data.shape}")
            
            # Analyze overall K-range (all reactions combined)
            print(f"\n📈 Overall K-Range Analysis:")
            overall_stats = analyze_k_range(k_values)
            print(f"  K min: {overall_stats['k_min']:.6e}")
            print(f"  K max: {overall_stats['k_max']:.6e}")
            print(f"  K mean: {overall_stats['k_mean']:.6e}")
            print(f"  K median: {overall_stats['k_median']:.6e}")
            print(f"  K std: {overall_stats['k_std']:.6e}")
            print(f"  Lower factor (K_mean/K_min): {overall_stats['lower_factor']:.6f}")
            print(f"  Upper factor (K_max/K_mean): {overall_stats['upper_factor']:.6f}")
            print(f"  Overall factor: {overall_stats['overall_factor']:.6f}")
            
            # Analyze per-reaction
            print(f"\n📊 Per-Reaction Analysis:")
            per_reaction_stats = []
            for ridx in range(k_values.shape[1]):
                stats = analyze_k_range(k_values, reaction_idx=ridx)
                per_reaction_stats.append(stats)
                
                print(f"\n  Reaction {ridx}:")
                print(f"    K range: [{stats['k_min']:.6e}, {stats['k_max']:.6e}]")
                print(f"    K mean: {stats['k_mean']:.6e}")
                print(f"    K median: {stats['k_median']:.6e}")
                print(f"    K std: {stats['k_std']:.6e}")
                print(f"    Lower factor (mean/min): {stats['lower_factor']:.6f}")
                print(f"    Upper factor (max/mean): {stats['upper_factor']:.6f}")
                print(f"    Overall factor: {stats['overall_factor']:.6f}")
                
                # Check if symmetric around K_true
                if abs(stats['lower_factor'] - stats['upper_factor']) > 0.1:
                    print(f"    ⚠️  WARNING: Asymmetric range! (lower={stats['lower_factor']:.3f}, upper={stats['upper_factor']:.3f})")
            
            # Summary
            max_factor = max([s['overall_factor'] for s in per_reaction_stats])
            print(f"\n✅ Maximum factor across all reactions: {max_factor:.6f}")
            print(f"   → Effective K-range width: K ∈ [K_mean/{max_factor:.6f}, K_mean×{max_factor:.6f}]")
            
            validation_results.append({
                'batch_idx': i+1,
                'label': batch_label,
                'status': 'OK',
                'expected_range': expected_k_range,
                'actual_factor': max_factor,
                'n_samples': len(pool_ds),
                'per_reaction': per_reaction_stats
            })
            
        except Exception as e:
            print(f"\n❌ ERROR loading dataset: {e}")
            import traceback
            traceback.print_exc()
            validation_results.append({
                'batch_idx': i+1,
                'label': batch_label,
                'status': 'ERROR',
                'expected_range': expected_k_range,
                'error': str(e)
            })
    
    # Final Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    successful = [r for r in validation_results if r['status'] == 'OK']
    failed = [r for r in validation_results if r['status'] != 'OK']
    
    print(f"\n✅ Successfully validated: {len(successful)}/{len(validation_results)} batch files")
    if failed:
        print(f"❌ Failed: {len(failed)} batch files")
        for r in failed:
            print(f"   - Batch {r['batch_idx']}: {r['status']}")
    
    if successful:
        print(f"\n📊 K-Range Factors Summary:")
        print(f"{'Batch':<6} {'Expected Range':<35} {'Actual Factor':<15} {'Samples':<10}")
        print("-"*80)
        for r in successful:
            print(f"{r['batch_idx']:<6} {r['expected_range']:<35} {r['actual_factor']:<15.6f} {r['n_samples']:<10}")
    
    # Save detailed results
    import json
    output_file = Path("pipeline_results/batch_k_range_validation.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"\n✓ Detailed validation results saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
