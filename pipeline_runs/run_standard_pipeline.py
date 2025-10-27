"""
Example script to run a standard pipeline with O2 simple mechanism data.

This script demonstrates:
1. Loading train/test datasets
2. Creating an SVR model
3. Running pipeline with full training data
4. Evaluating and saving results
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from kinetic_modelling.data import MultiPressureDataset
from kinetic_modelling.model import SVRModel
from kinetic_modelling.pipeline import StandardPipeline


def main():
    """Run the standard pipeline example."""
    
    # Configuration
    DATA_DIR = Path(__file__).parent.parent / "data" / "SampleEfficiency"
    TRAIN_FILE = DATA_DIR / "O2_simple_uniform.txt"
    TEST_FILE = DATA_DIR / "O2_simple_test.txt"
    
    # O2 simple mechanism parameters
    NSPECIES = 3
    NUM_PRESSURES = 2
    REACT_IDX = np.array([0, 1, 2])  # Select first 3 reactions
    # Toggle: plot input chemistry distributions after loading
    PLOT_CHEM_DISTRIBUTION = True
    
    print("="*70)
    print("Standard Pipeline Example: O2 Simple Mechanism")
    print("="*70)
    
    # Step 1: Load datasets
    print("\nLoading datasets...")
    train_dataset = MultiPressureDataset(
        nspecies=NSPECIES,
        num_pressure_conditions=NUM_PRESSURES,
        src_file=str(TRAIN_FILE),
        react_idx=REACT_IDX
    )
    
    test_dataset = MultiPressureDataset(
        nspecies=NSPECIES,
        num_pressure_conditions=NUM_PRESSURES,
        src_file=str(TEST_FILE),
        react_idx=REACT_IDX,
        scaler_input=train_dataset.scaler_input,
        scaler_output=train_dataset.scaler_output
    )
    
    print(f"✓ Loaded {len(train_dataset)} training samples")
    print(f"✓ Loaded {len(test_dataset)} test samples")

    # Optional: plot chemistry (input) distributions to check uniformity
    if PLOT_CHEM_DISTRIBUTION:
        try:
            # True chemical composition from O2_simple_test_real_K.txt
            # Format: K1, K2, K3, ..., density1_p1, density2_p1, density3_p1, density1_p2, density2_p2, density3_p2
            # Pressure 1: 133.3 Torr - 1.95692653377725e+22, 2.04184831645824e+21, 4.00294188532869e+21
            # Pressure 2: 1333.3 Torr - 2.29232791411886e+23, 3.14188435566196e+21, 2.37654460510635e+22
            pressure_values = [133.3, 1333.3]  # Torr
            true_chem = np.array([
                [1.95692653377725e+22, 2.04184831645824e+21, 4.00294188532869e+21],  # Pressure 1
                [2.29232791411886e+23, 3.14188435566196e+21, 2.37654460510635e+22]   # Pressure 2
            ])
            
            raw = train_dataset.raw_data
            ncols = raw.shape[1]
            # input columns are the last `nspecies` columns
            x_cols = np.arange(ncols - train_dataset.nspecies, ncols)
            x_vals = raw[:, x_cols]
            
            # Reshape to separate pressure conditions
            # Data structure: samples are interleaved by pressure condition
            n_samples_total = len(x_vals)
            n_samples_per_pressure = n_samples_total // NUM_PRESSURES
            
            x_vals_reshaped = x_vals.reshape(NUM_PRESSURES, n_samples_per_pressure, NSPECIES)

            # Prepare plot: rows = pressure conditions, cols = species
            fig, axes = plt.subplots(NUM_PRESSURES, NSPECIES, 
                                    figsize=(5 * NSPECIES, 4 * NUM_PRESSURES))
            
            # Ensure axes is always 2D
            if NUM_PRESSURES == 1 and NSPECIES == 1:
                axes = np.array([[axes]])
            elif NUM_PRESSURES == 1:
                axes = axes.reshape(1, -1)
            elif NSPECIES == 1:
                axes = axes.reshape(-1, 1)

            for pressure_idx in range(NUM_PRESSURES):
                for species_idx in range(NSPECIES):
                    ax = axes[pressure_idx, species_idx]
                    vals = x_vals_reshaped[pressure_idx, :, species_idx]
                    
                    ax.hist(vals, bins=50, density=False, alpha=0.75, 
                           color=f'C{pressure_idx}', edgecolor='black')
                    ax.set_title(f'{pressure_values[pressure_idx]:.1f} Torr, Species {species_idx+1}')
                    ax.set_xlabel('Density')
                    ax.set_ylabel('Count')
                    ax.grid(alpha=0.3)
                    
                    # Add true chemical composition line
                    true_val = true_chem[pressure_idx, species_idx]
                    ax.axvline(true_val, color='red', linestyle='--', linewidth=2, 
                              label=f'True value: {true_val:.2e}')
                    ax.legend(fontsize=8)

            plt.tight_layout()
            out_dir = Path(__file__).parent.parent / 'pipeline_results'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "chemistry_distribution_by_pressure.png"
            plt.savefig(out_file, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"\n📊 Chemistry distribution plot saved to: {out_file}")
        except Exception as e:
            print(f"Could not plot chemistry distribution: {e}")
    
    # Print MaxAbsScaler information
    print("\n" + "="*70)
    print("DATA SCALING INFORMATION")
    print("="*70)
    input_scalers, output_scalers = train_dataset.get_scalers()
    
    # Input scalers - K values (reaction rate coefficients)
    print(f"\nInput Scaler (K values - Pressure condition 0):")
    print(f"  Number of K values (inputs): {len(input_scalers[0].scale_)}")
    for i, scale in enumerate(input_scalers[0].scale_):
        print(f"  K_{i+1} MaxAbs scale factor: {scale:.6e}")
        print(f"    → Scaled range: [-1, 1] corresponds to original range: [{-scale:.6e}, {scale:.6e}]")
    
    # Output scalers - species concentrations/pressures
    print(f"\nOutput Scaler (Species concentrations - Pressure condition 0):")
    print(f"  Number of outputs: {len(output_scalers[0].scale_)}")
    for i, scale in enumerate(output_scalers[0].scale_):
        print(f"  Output_{i+1} MaxAbs scale factor: {scale:.6e}")
    
    print("\nNOTE: MSE values reported during training are on SCALED OUTPUT data ([-1, 1] range)")
    print("To convert scaled MSE to original output units:")
    print("  Original_MSE = Scaled_MSE × (output_scale_factor)²")
    print("  Original_RMSE = sqrt(Scaled_MSE) × output_scale_factor")
    print("="*70)
    
    # Step 2: Create SVR model
    print("\nCreating SVR model...")
    # Use parameters from your sample_effiency.py
    svr_params = [
        {'C': 10, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'},
        {'C': 20, 'epsilon': 0.005, 'gamma': 5, 'kernel': 'rbf'},
        {'C': 5, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'}
    ]
    
    model = SVRModel(
        params=svr_params,
        model_name="o2_simple_svr"
    )
    print("✓ SVR model created")
    
    # Step 3: Create and run pipeline
    print("\nCreating pipeline...")
    pipeline = StandardPipeline(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        sampler=None,  # Use full dataset
        n_samples=None,
        pipeline_name="o2_simple_full_data",
        results_dir="pipeline_results"
    )
    
    # Run pipeline and save results
    results = pipeline.save_and_return(save_results=True)
    
    # Display final summary
    print("\n" + "="*70)
    print("Pipeline Execution Complete!")
    print("="*70)
    train_metrics = results['evaluation']['train_metrics']
    test_metrics = results['evaluation']['test_metrics']
    
    if 'r2_score' in train_metrics:
        print(f"Train R²: {train_metrics['r2_score']:.4f}")
        print(f"Test R²:  {test_metrics['r2_score']:.4f}")
    if 'total_mse' in train_metrics:
        print(f"Train Total MSE: {train_metrics['total_mse']:.6e}")
        print(f"Test Total MSE:  {test_metrics['total_mse']:.6e}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
