"""
Example script to plot window sampling distributions from JSON batch files.

This script:
1. Loads JSON batch simulation files
2. Extracts species densities (chemistry) from compositions
3. Plots histograms for each batch file showing distribution
4. Overlays true chemical composition values
5. Saves plots to pipeline_results/chemical_plots/window_sampling/
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import json
from kinetic_modelling.data import MultiPressureDataset


def load_batch_json(json_path, nspecies, num_pressures):
    """
    Load batch JSON file and extract compositions and K values.
    
    Args:
        json_path: Path to JSON batch file
        nspecies: Number of species
        num_pressures: Number of pressure conditions
        
    Returns:
        compositions_reshaped: Array of shape (num_pressures, n_samples, nspecies)
        k_values: Array of shape (n_samples, n_k_reactions) - K values (pressure-independent)
    """
    with open(json_path, 'r') as f:
        batch_data = json.load(f)
    
    compositions = np.array(batch_data['compositions'])
    
    # Extract K values from parameter_sets
    parameter_sets = batch_data.get('parameter_sets', [])
    if parameter_sets:
        k_values = np.array([ps['k_values'] for ps in parameter_sets])
    else:
        k_values = None
    
    # Compositions are flat: reshape to (num_pressures, n_samples, nspecies)
    # Data structure: samples are interleaved by pressure condition
    n_samples_total = len(compositions)
    n_samples_per_pressure = n_samples_total // num_pressures
    
    # Extract only the last nspecies columns (species densities)
    # Assuming compositions have format: [..., density1, density2, density3]
    compositions_densities = compositions[:, -nspecies:]
    
    # Reshape to separate pressure conditions
    compositions_reshaped = compositions_densities.reshape(num_pressures, n_samples_per_pressure, nspecies)
    
    return compositions_reshaped, k_values


def plot_window_distribution(compositions_reshaped, dataset_name, file_name, output_dir,
                             nspecies, num_pressures, pressure_values, true_chem, k_range=None,
                             tick_label_size=8, x_sigdigs=None):
    """
    Plot chemistry distribution histograms for a window sampling batch.
    
    Args:
        compositions_reshaped: Array of shape (num_pressures, n_samples, nspecies)
        dataset_name: Name for the plot title (e.g., "Window 1.0")
        file_name: Original file name to display in title
        output_dir: Directory to save plot
        nspecies: Number of species
        num_pressures: Number of pressure conditions
        pressure_values: List of pressure values in Torr
        true_chem: Array of true chemical composition values [pressure, species]
        k_range: String describing the K value range (e.g., "K ∈ [K_true/1.15, K_true×1.15]")
    """
    try:
        # Prepare plot: rows = pressure conditions, cols = species
        fig, axes = plt.subplots(num_pressures, nspecies, 
                                figsize=(5 * nspecies, 4 * num_pressures))
        
        # Add super title with dataset name and K range (no filename)
        title = f'{dataset_name} - Absolute Density Histogram'
        if k_range:
            title += f'\n{k_range}'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        
        # Ensure axes is always 2D
        if num_pressures == 1 and nspecies == 1:
            axes = np.array([[axes]])
        elif num_pressures == 1:
            axes = axes.reshape(1, -1)
        elif nspecies == 1:
            axes = axes.reshape(-1, 1)

        for pressure_idx in range(num_pressures):
            for species_idx in range(nspecies):
                ax = axes[pressure_idx, species_idx]
                vals = compositions_reshaped[pressure_idx, :, species_idx]
                
                ax.hist(vals, bins=50, density=False, alpha=0.75, 
                       color=f'C{pressure_idx}', edgecolor='black')
                ax.set_title(f'{pressure_values[pressure_idx]:.1f} Torr, Species {species_idx+1}')
                ax.set_xlabel('Density')
                ax.set_ylabel('Count')
                ax.grid(alpha=0.3)
                
                # Format x-axis to show full scientific notation without offset
                ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
                ax.xaxis.get_major_formatter().set_powerlimits((0, 0))
                # Optionally control significant digits for x tick labels
                if x_sigdigs is not None:
                    def fmt(x, pos=None):
                        return f"{x:.{x_sigdigs}e}"
                    ax.xaxis.set_major_formatter(FuncFormatter(fmt))
                # Tick label fontsize
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontsize(tick_label_size)
                
                # Add true chemical composition line
                true_val = true_chem[pressure_idx, species_idx]
                ax.axvline(true_val, color='red', linestyle='--', linewidth=2, 
                          label=f'True value: {true_val:.4e}')
                ax.legend(fontsize=8)

        plt.tight_layout()
        
        # Save plot
        output_dir.mkdir(parents=True, exist_ok=True)
        sanitized_name = dataset_name.lower().replace(' ', '_').replace('/', '_')
        out_file = output_dir / f"{sanitized_name}_chemistry_distribution.png"
        plt.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ {dataset_name} plot saved to: {out_file}")
        
    except Exception as e:
        print(f"✗ Could not plot {dataset_name} distribution: {e}")
        import traceback
        traceback.print_exc()


def plot_k_distribution(k_values, dataset_name, file_name, output_dir, true_k=None, k_range=None,
                        tick_label_size=8, x_sigdigs=None):
    """
    Plot K (reaction rate) distribution histograms for a window sampling batch.
    
    Args:
        k_values: Array of shape (n_samples, n_k_reactions)
        dataset_name: Name for the plot title (e.g., "Window 1.0 Sampling")
        file_name: Original file name to display in title
        output_dir: Directory to save plot
        true_k: True K values to overlay (1D array or None)
        k_range: String describing the K value range (e.g., "K ∈ [K_true/1.15, K_true×1.15]")
    """
    try:
        if k_values is None or len(k_values) == 0:
            print(f"  ⚠️ No K values available for {dataset_name}")
            return
        
        n_outputs = k_values.shape[1]
        
        # Prepare plot: single row with K1, K2, K3 columns (K is pressure-independent)
        fig, axes = plt.subplots(1, n_outputs, figsize=(5 * n_outputs, 4))
        
        # Add super title with dataset name and K range (no filename)
        title = f'{dataset_name} - K Value Histogram'
        if k_range:
            title += f'\n{k_range}'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        
        # Normalize axes shape
        if n_outputs == 1:
            axes = np.array([axes])
        
        for out_idx in range(n_outputs):
            ax = axes[out_idx]
            vals = k_values[:, out_idx]
            ax.hist(vals, bins=50, density=False, alpha=0.8, color=f'C{out_idx}', edgecolor='black')
            ax.set_title(f'K{out_idx+1}')
            ax.set_xlabel('K value (raw)')
            ax.set_ylabel('Count')
            ax.grid(alpha=0.3)
            
            # Format x-axis to show full scientific notation without offset
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            ax.xaxis.get_major_formatter().set_powerlimits((0, 0))
            if x_sigdigs is not None:
                def fmt(x, pos=None):
                    return f"{x:.{x_sigdigs}e}"
                ax.xaxis.set_major_formatter(FuncFormatter(fmt))
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(tick_label_size)
            
            # Add true K value line if provided
            if true_k is not None and out_idx < len(true_k):
                true_k_val = true_k[out_idx]
                ax.axvline(true_k_val, color='red', linestyle='--', linewidth=2,
                          label=f'True K: {true_k_val:.4e}')
                ax.legend(fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        output_dir.mkdir(parents=True, exist_ok=True)
        sanitized_name = dataset_name.lower().replace(' ', '_').replace('/', '_')
        out_file = output_dir / f"{sanitized_name}_k_distribution.png"
        plt.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ {dataset_name} K distribution plot saved to: {out_file}")
        
    except Exception as e:
        print(f"✗ Could not plot K distribution for {dataset_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Plot chemistry distributions for window sampling batch files."""
    
    # Configuration
    OUTPUT_DIR = Path(__file__).parent.parent / "pipeline_results" / "chemical_plots" / "window_sampling"
    
    # Window sampling batch files with K boundaries
    BATCH_FILES = [
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-08-27/batch_4000sims_20250827_010028.json',
            'label': 'Window Batch 1 (4000 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/2, K_true×2]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-28/batch_1000sims_20250928_191628.json',
            'label': 'Window Batch 2 (1000 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.15, K_true×1.15]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2500sims_20250929_031845.json',
            'label': 'Window Batch 3 (2500 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.15, K_true×1.15]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2000sims_20250929_205429.json',
            'label': 'Window Batch 4 (2000 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.005, K_true×1.005]'
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-28/batch_1500sims_20250928_224858.json',
            'label': 'Window Batch 5 (1500 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.0005, K_true×1.0005] '
        },
        {
            'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2000sims_20250929_125706.json',
            'label': 'Window Batch 6 (2000 samples) - Uniform Sampling',
            'k_range': 'K ∈ [K_true/1.00005, K_true×1.00005] '
        }
    ]
    
    # O2 simple mechanism parameters
    NSPECIES = 3
    NUM_PRESSURES = 2
    
    # True K values
    TRUE_K = np.array([6.00e-16, 1.30e-15, 9.60e-16])
    
    # True chemical composition from O2_simple_test_real_K.txt
    # Pressure 1: 133.3 Torr - 1.95692653377725e+22, 2.04184831645824e+21, 4.00294188532869e+21
    # Pressure 2: 1333.3 Torr - 2.29232791411886e+23, 3.14188435566196e+21, 2.37654460510635e+22
    PRESSURE_VALUES = [133.3, 1333.3]  # Torr
    TRUE_CHEM = np.array([
        [1.95692653377725e+22, 2.04184831645824e+21, 4.00294188532869e+21],  # Pressure 1
        [2.29232791411886e+23, 3.14188435566196e+21, 2.37654460510635e+22]   # Pressure 2
    ])
    
    print("="*70)
    print("Window Sampling Distribution Plotting")
    print("="*70)
    
    # Load and plot each batch file
    for idx, batch_info in enumerate(BATCH_FILES):
        print(f"\n[{idx+1}/{len(BATCH_FILES)}] Processing {batch_info['label']}...")
        
        batch_path = Path(__file__).parent.parent / batch_info['path']
        
        if not batch_path.exists():
            print(f"  ⚠️ File not found: {batch_path}")
            continue
        
        # Load batch file
        print(f"  Loading batch file: {batch_path.name}")
        try:
            compositions_reshaped, k_values = load_batch_json(
                json_path=batch_path,
                nspecies=NSPECIES,
                num_pressures=NUM_PRESSURES
            )
            
            n_samples_per_pressure = compositions_reshaped.shape[1]
            total_samples = n_samples_per_pressure * NUM_PRESSURES
            print(f"  ✓ Loaded {total_samples} samples ({n_samples_per_pressure} per pressure)")
            
            # Plot chemistry distribution
            print(f"  Plotting chemistry distribution...")
            plot_window_distribution(
                compositions_reshaped=compositions_reshaped,
                dataset_name=batch_info['label'],
                file_name=batch_path.name,
                output_dir=OUTPUT_DIR,
                nspecies=NSPECIES,
                num_pressures=NUM_PRESSURES,
                pressure_values=PRESSURE_VALUES,
                true_chem=TRUE_CHEM,
                k_range=batch_info.get('k_range')
            )
            
            # Plot K distribution
            if k_values is not None:
                print(f"  Plotting K distribution...")
                plot_k_distribution(
                    k_values=k_values,
                    dataset_name=batch_info['label'],
                    file_name=batch_path.name,
                    output_dir=OUTPUT_DIR,
                    true_k=TRUE_K,
                    k_range=batch_info.get('k_range')
                )
            else:
                print(f"  ⚠️ No K values found in batch file")
            
        except Exception as e:
            print(f"  ✗ Error processing file: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("Plotting Complete!")
    print(f"Plots saved to: {OUTPUT_DIR}")
    print(f"Total chemistry plots generated: {len(BATCH_FILES)}")
    print(f"Total K distribution plots generated: {len(BATCH_FILES)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
