"""
Example script to plot dataset distributions for chemistry (species densities).

This script:
1. Loads train and test datasets
2. Plots histograms of species densities for each pressure condition
3. Overlays true chemical composition values
4. Saves plots to pipeline_results/chemical_plots/
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from kinetic_modelling.data import MultiPressureDataset


def plot_chemistry_distribution(dataset, dataset_name, file_name, output_dir, 
                                nspecies, num_pressures, pressure_values, true_chem):
    """
    Plot chemistry distribution histograms for a dataset.
    
    Args:
        dataset: MultiPressureDataset instance
        dataset_name: Name for the plot title (e.g., "Training Dataset")
        file_name: Original file name to display in title
        output_dir: Directory to save plot
        nspecies: Number of species
        num_pressures: Number of pressure conditions
        pressure_values: List of pressure values in Torr
        true_chem: Array of true chemical composition values [pressure, species]
    """
    try:
        raw = dataset.raw_data
        ncols = raw.shape[1]
        # input columns are the last `nspecies` columns
        x_cols = np.arange(ncols - nspecies, ncols)
        x_vals = raw[:, x_cols]
        
        # Reshape to separate pressure conditions
        # Data structure: samples are interleaved by pressure condition
        n_samples_total = len(x_vals)
        n_samples_per_pressure = n_samples_total // num_pressures
        
        x_vals_reshaped = x_vals.reshape(num_pressures, n_samples_per_pressure, nspecies)

        # Prepare plot: rows = pressure conditions, cols = species
        fig, axes = plt.subplots(num_pressures, nspecies, 
                                figsize=(5 * nspecies, 4 * num_pressures))
        
        # Add super title with dataset name and file name
        fig.suptitle(f'{dataset_name} - Absolute Density Histogram\n{file_name}', 
                    fontsize=14, fontweight='bold', y=0.995)
        
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
                vals = x_vals_reshaped[pressure_idx, :, species_idx]
                
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
                
                # Add true chemical composition line
                true_val = true_chem[pressure_idx, species_idx]
                ax.axvline(true_val, color='red', linestyle='--', linewidth=2, 
                          label=f'True value: {true_val:.2e}')
                ax.legend(fontsize=8)

        plt.tight_layout()
        
        # Save plot
        output_dir.mkdir(parents=True, exist_ok=True)
        sanitized_name = dataset_name.lower().replace(' ', '_')
        out_file = output_dir / f"{sanitized_name}_chemistry_distribution.png"
        plt.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ {dataset_name} plot saved to: {out_file}")
        
    except Exception as e:
        print(f"✗ Could not plot {dataset_name} distribution: {e}")


def plot_k_distribution(dataset, dataset_name, file_name, output_dir,
                        nspecies, num_pressures, react_idx, true_k=None):
    """
    Plot K (reaction rate) distribution histograms for a dataset.

    Args:
        dataset: MultiPressureDataset instance
        dataset_name: Name for the plot title
        file_name: Original file name to display in title
        output_dir: Directory to save plot
        nspecies: Number of species (used to find columns)
        num_pressures: Number of pressure conditions
        react_idx: Indices of reaction outputs (list or array)
        true_k: True K values to overlay (1D array or None)
    """
    try:
        raw = dataset.raw_data
        ncols = raw.shape[1]

        # y/ K columns are the first (ncols - nspecies) columns; use react_idx to pick which
        y_cols = np.array(react_idx)
        y_vals = raw[:, y_cols]

        # K values are pressure-independent (same for all pressure conditions)
        # Take only first pressure condition's samples
        n_samples_total = len(y_vals)
        n_samples_per_pressure = n_samples_total // num_pressures
        y_vals_single = y_vals[:n_samples_per_pressure]

        n_outputs = y_vals_single.shape[1]

        # Prepare plot: single row with K1, K2, K3 columns
        fig, axes = plt.subplots(1, n_outputs, figsize=(5 * n_outputs, 4))
        fig.suptitle(f'{dataset_name} - K Value Histogram\n{file_name}', fontsize=14, fontweight='bold', y=0.995)

        # Normalize axes shape
        if n_outputs == 1:
            axes = np.array([axes])

        for out_idx in range(n_outputs):
            ax = axes[out_idx]
            vals = y_vals_single[:, out_idx]
            ax.hist(vals, bins=50, density=False, alpha=0.8, color=f'C{out_idx}', edgecolor='black')
            ax.set_title(f'K{out_idx+1}')
            ax.set_xlabel('K value (raw)')
            ax.set_ylabel('Count')
            ax.grid(alpha=0.3)
            
            # Format x-axis to show full scientific notation without offset
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            ax.xaxis.get_major_formatter().set_powerlimits((0, 0))
            
            # Add true K value line if provided
            if true_k is not None and out_idx < len(true_k):
                true_k_val = true_k[out_idx]
                ax.axvline(true_k_val, color='red', linestyle='--', linewidth=2,
                          label=f'True K: {true_k_val:.2e}')
                ax.legend(fontsize=8)
                true_k_val = true_k[out_idx]
                ax.axvline(true_k_val, color='red', linestyle='--', linewidth=2,
                          label=f'True K: {true_k_val:.2e}')
                ax.legend(fontsize=8)

        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        sanitized_name = dataset_name.lower().replace(' ', '_')
        out_file = output_dir / f"{sanitized_name}_k_distribution.png"
        plt.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ {dataset_name} K distribution saved to: {out_file}")

    except Exception as e:
        print(f"✗ Could not plot K distribution for {dataset_name}: {e}")


def main():
    """Plot chemistry distributions for train and test datasets."""
    
    # Configuration
    DATA_DIR = Path(__file__).parent.parent / "data" / "SampleEfficiency"
    TEST_FILE = DATA_DIR / "O2_simple_test.txt"
    OUTPUT_DIR = Path(__file__).parent.parent / "pipeline_results" / "chemical_plots"
    
    # Dataset files and labels
    TRAIN_FILES = [
        'O2_simple_log.txt',
        'O2_simple__morris_continous_final.txt',
        'O2_simple_uniform.txt',
        'O2_simple_latin_log_uniform.txt',
        'O2_simple_latin.txt',
    ]
    LABELS = [
        'Log-Uniform Sampling',
        'Morris Method Sampling (continuous)',
        'Uniform Sampling',
        'Log-Uniform Latin Hypercube Sampling',
        'Uniform Latin Hypercube Sampling',
    ]
    
    # O2 simple mechanism parameters
    NSPECIES = 3
    NUM_PRESSURES = 2
    REACT_IDX = np.array([0, 1, 2])  # Select first 3 reactions
    
    # True chemical composition from O2_simple_test_real_K.txt
    # Pressure 1: 133.3 Torr - 1.95692653377725e+22, 2.04184831645824e+21, 4.00294188532869e+21
    # Pressure 2: 1333.3 Torr - 2.29232791411886e+23, 3.14188435566196e+21, 2.37654460510635e+22
    PRESSURE_VALUES = [133.3, 1333.3]  # Torr
    TRUE_CHEM = np.array([
        [1.95692653377725e+22, 2.04184831645824e+21, 4.00294188532869e+21],  # Pressure 1
        [2.29232791411886e+23, 3.14188435566196e+21, 2.37654460510635e+22]   # Pressure 2
    ])
    
    # True K values from O2_simple_test_real_K.txt (first 3 columns)
    TRUE_K = np.array([6.00e-16, 1.30e-15, 9.60e-16])  # K1, K2, K3
    
    print("="*70)
    print("Dataset Distribution Plotting")
    print("="*70)
    
    # Load and plot each training dataset
    for idx, (train_file, label) in enumerate(zip(TRAIN_FILES, LABELS)):
        print(f"\n[{idx+1}/{len(TRAIN_FILES)}] Processing {label}...")
        
        train_file_path = DATA_DIR / train_file
        
        # Load training dataset
        print(f"  Loading training dataset: {train_file}")
        train_dataset = MultiPressureDataset(
            nspecies=NSPECIES,
            num_pressure_conditions=NUM_PRESSURES,
            src_file=str(train_file_path),
            react_idx=REACT_IDX
        )
        print(f"  ✓ Loaded {len(train_dataset)} training samples")
        
        # Plot training dataset
        print(f"  Plotting distribution...")
        plot_chemistry_distribution(
            dataset=train_dataset,
            dataset_name=label,
            file_name=train_file,
            output_dir=OUTPUT_DIR,
            nspecies=NSPECIES,
            num_pressures=NUM_PRESSURES,
            pressure_values=PRESSURE_VALUES,
            true_chem=TRUE_CHEM
        )

        # Also plot K distributions for this dataset
        plot_k_distribution(
            dataset=train_dataset,
            dataset_name=label,
            file_name=train_file,
            output_dir=OUTPUT_DIR,
            nspecies=NSPECIES,
            num_pressures=NUM_PRESSURES,
            react_idx=REACT_IDX,
            true_k=TRUE_K
        )
    
    # Load and plot test dataset
    print(f"\n[{len(TRAIN_FILES)+1}/{len(TRAIN_FILES)+1}] Processing Test Dataset...")
    print(f"  Loading test dataset: {TEST_FILE.name}")
    
    # Load with scalers from first training dataset for consistency
    reference_train = MultiPressureDataset(
        nspecies=NSPECIES,
        num_pressure_conditions=NUM_PRESSURES,
        src_file=str(DATA_DIR / TRAIN_FILES[0]),
        react_idx=REACT_IDX
    )
    
    test_dataset = MultiPressureDataset(
        nspecies=NSPECIES,
        num_pressure_conditions=NUM_PRESSURES,
        src_file=str(TEST_FILE),
        react_idx=REACT_IDX,
        scaler_input=reference_train.scaler_input,
        scaler_output=reference_train.scaler_output
    )
    print(f"  ✓ Loaded {len(test_dataset)} test samples")
    
    # Plot test dataset
    print(f"  Plotting distribution...")
    plot_chemistry_distribution(
        dataset=test_dataset,
        dataset_name="Test Dataset",
        file_name=TEST_FILE.name,
        output_dir=OUTPUT_DIR,
        nspecies=NSPECIES,
        num_pressures=NUM_PRESSURES,
        pressure_values=PRESSURE_VALUES,
        true_chem=TRUE_CHEM
    )

    # Plot K distributions for the test dataset
    plot_k_distribution(
        dataset=test_dataset,
        dataset_name="Test Dataset",
        file_name=TEST_FILE.name,
        output_dir=OUTPUT_DIR,
        nspecies=NSPECIES,
        num_pressures=NUM_PRESSURES,
        react_idx=REACT_IDX,
        true_k=TRUE_K
    )
    
    print("\n" + "="*70)
    print("Plotting Complete!")
    print(f"Plots saved to: {OUTPUT_DIR}")
    print(f"Total plots generated: {len(TRAIN_FILES) + 1}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
