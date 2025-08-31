#!/usr/bin/env python3
"""
Direct batch training script.

Load the 500-simulation batch JSON data directly and train SVR models using the same
methods as active_learning_train.py. Compare performance with uniform dataset baseline.
"""

import json
import numpy as np
import os
import sys
sys.path.append('.')  # Add current directory to path
from active_learning_methods import (
    load_datasets, 
    train_initial_models, 
    run_mse_analysis,
    apply_training_scalers,
    retrain_models_with_new_data,
    LoadMultiPressureDatasetNumpy
)
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

def load_batch_data(json_path):
    """
    Load batch simulation data from JSON file.
    
    Returns:
        k_values: Array of shape (n_sims, 3) - raw k values
        compositions: Array of shape (n_sims * n_pressures, 3) - compositions
        n_simulations: Number of simulations
        n_pressure_conditions: Number of pressure conditions
    """
    print(f"📂 Loading batch data from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract metadata
    n_simulations = data['n_simulations']
    n_successful = data['n_successful']
    success_rate = data['success_rate']
    pressure_conditions = data['metadata']['pressure_conditions_pa']
    n_pressure_conditions = data['metadata']['n_pressure_conditions']
    
    print(f"   📊 Batch Info:")
    print(f"      Total simulations: {n_simulations}")
    print(f"      Successful: {n_successful}")
    print(f"      Success rate: {success_rate:.1%}")
    print(f"      Pressure conditions: {pressure_conditions} Pa")
    print(f"      Number of pressures: {n_pressure_conditions}")
    
    # Extract parameter sets (k values)
    k_values = []
    for ps in data['parameter_sets']:
        k_values.append(ps['k_values'])
    k_values = np.array(k_values)
    
    # Extract compositions
    compositions = np.array(data['compositions'])
    
    print(f"   📊 Data Shapes:")
    print(f"      K values: {k_values.shape}")
    print(f"      Compositions: {compositions.shape}")
    print(f"      Expected compositions shape: ({n_simulations * n_pressure_conditions}, 3)")
    
    # Verify shapes
    expected_comp_rows = n_simulations * n_pressure_conditions
    if compositions.shape[0] != expected_comp_rows:
        print(f"   ⚠️ Warning: Composition shape mismatch!")
        print(f"      Expected: ({expected_comp_rows}, 3)")
        print(f"      Actual: {compositions.shape}")
    
    # Show data ranges
    print(f"   📊 Data Ranges:")
    print(f"      K values: [{k_values.min():.2e}, {k_values.max():.2e}]")
    print(f"      Compositions: [{compositions.min():.2e}, {compositions.max():.2e}]")
    
    return k_values, compositions, n_simulations, n_pressure_conditions


def create_batch_dataset_using_active_methods(k_values, compositions, nspecies, num_pressure_conditions, 
                                            reference_dataset=None):
    """
    Create a batch dataset using the EXACT same method as active_learning_train.py.
    
    This uses apply_training_scalers() from active_learning_methods to ensure 100% consistency.
    
    Args:
        k_values: Raw k values (n_sims, 3)
        compositions: Raw compositions (n_sims * n_pressures, 3)  
        nspecies: Number of species (3)
        num_pressure_conditions: Number of pressure conditions (2)
        reference_dataset: Reference dataset to use scalers from
        
    Returns:
        Batch dataset object with same interface as LoadMultiPressureDatasetNumpy
    """
    print(f"🔧 Creating batch dataset using ACTIVE LEARNING METHODS")
    print(f"   Input shapes: k_values={k_values.shape}, compositions={compositions.shape}")
    
    if reference_dataset is None:
        raise ValueError("Reference dataset is required for consistent scaling")
    
    # Use the EXACT same method as active_learning_train.py
    print(f"   Using apply_training_scalers() from active_learning_methods...")
    
    # Raw data (exactly like active_learning_train.py processes batch results)
    raw_k_values = k_values  # Shape: (n_sims, 3)
    raw_compositions = compositions  # Shape: (n_sims * n_pressures, 3)
    
    # DEBUG: Print raw data before scaling (same as active learning)
    print("\n==== DEBUG: Batch Data BEFORE Scaling (First 5 Rows) ====")
    print("Raw compositions (first 5):\n", raw_compositions[:5])
    print("Raw k values (first 5):\n", raw_k_values[:5])
    
    # Apply the EXACT same scaling as active_learning_train.py
    new_x, new_y_scaled = apply_training_scalers(
        raw_compositions=raw_compositions,
        raw_k_values=raw_k_values,
        dataset_train=reference_dataset,
        nspecies=nspecies,
        num_pressure_conditions=num_pressure_conditions,
        debug=True
    )
    
    
    # DEBUG: Print scaled data after scaling (same as active learning)
    print("\n==== DEBUG: Batch Data AFTER Scaling (First 5 Rows) ====")
    print("Scaled X (first 5):\n", new_x[:5])
    print("Scaled Y (first 5):\n", new_y_scaled[:5])
    
    print(f"   Final shapes: x_data={new_x.shape}, y_data={new_y_scaled.shape}")
    print(f"   Final ranges: x=[{new_x.min():.3e}, {new_x.max():.3e}]")
    print(f"                 y=[{new_y_scaled.min():.3e}, {new_y_scaled.max():.3e}]")
    
    # Create dataset object with same interface as LoadMultiPressureDatasetNumpy
    class BatchDataset:
        def __init__(self, x_data, y_data, scaler_input, scaler_output):
            self.x_data = x_data
            self.y_data = y_data
            self.scaler_input = scaler_input
            self.scaler_output = scaler_output
            
        def get_data(self):
            return self.x_data, self.y_data
    
    batch_dataset = BatchDataset(
        new_x, new_y_scaled, 
        reference_dataset.scaler_input, 
        reference_dataset.scaler_output
    )
    
    print(f"   ✅ Batch dataset created using ACTIVE LEARNING METHODS")
    return batch_dataset


def train_and_evaluate_batch_models_using_active_methods(batch_dataset, test_dataset, best_params, seeds=None):
    """
    Train SVR models on batch data using the EXACT same method as active_learning_methods.
    
    This uses train_initial_models() but on the batch dataset instead of uniform dataset.
    """
    print(f"🔬 Training models on batch data using ACTIVE LEARNING METHODS")
    
    x_train, y_train = batch_dataset.get_data()
    print(f"   Batch training data: {x_train.shape}, {y_train.shape}")
    
    # Create a temporary dataset object that mimics LoadMultiPressureDatasetNumpy interface
    # so we can use train_initial_models() directly
    class TempDataset:
        def __init__(self, x_data, y_data):
            self.x_data = x_data
            self.y_data = y_data
            
        def get_data(self):
            return self.x_data, self.y_data
    
    temp_batch_dataset = TempDataset(x_train, y_train)
    
    # Use the EXACT same training method as active learning
    # train_initial_models() will handle shuffling, training, and evaluation
    # It now returns models_per_seed (list of model-lists), plus averaged MSEs.
    models_per_seed, mse_per_output, total_mse = train_initial_models(
        temp_batch_dataset, test_dataset, best_params,
        n_initial_samples=x_train.shape[0], seeds=seeds
    )
    
    print(f"   ✅ Training complete using ACTIVE LEARNING METHODS")
    
    # Note: train_initial_models already prints detailed MSE info
    return models_per_seed, mse_per_output, total_mse


def compare_with_uniform_baseline(uniform_dataset, test_dataset, best_params,
                                 batch_size=500, seeds=None):
    """
    Train models on uniform dataset with same sample size as batch for comparison.
    """
    print(f"🔬 Training baseline models on uniform data (size={batch_size})")
    
    # Train on same number of samples as batch
    models_per_seed, mse_per_output, total_mse = train_initial_models(
        uniform_dataset, test_dataset, best_params,
        n_initial_samples=batch_size, seeds=seeds
    )
    
    return models_per_seed, mse_per_output, total_mse


if __name__ == '__main__':
    # Configuration (same as active_learning_train.py)
    config = {
        'nspecies': 3,
        'num_pressure_conditions': 2,
        'pressure_conditions_pa': [133.322, 1333.22],  # 1 and 10 Torr
        'svr_params': [
            {'C': 10, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'},
            {'C': 20, 'epsilon': 0.005, 'gamma': 5, 'kernel': 'rbf'},
            {'C': 5, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'}
        ],
        'seeds': [i for i in range(42, 52)]
    }
    
    print(f"🎯 BATCH DATA TRAINING EXPERIMENT")
    print(f"Configuration: {config}")
    
    # File paths
    batch_json_path = 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-08-20/batch_500sims_20250820_141641.json'
    batch_json_path = 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-08-27/batch_4000sims_20250827_010028.json'
    src_file_train = 'data/SampleEfficiency/O2_simple_uniform.txt'
    src_file_test = 'data/SampleEfficiency/O2_simple_test.txt'
    
    nspecies = config['nspecies']
    num_pressure_conditions = config['num_pressure_conditions']
    best_params = config['svr_params']
    # Multi-seed options (optional)
    num_seeds = config.get('num_seeds', 1)
    seeds = config.get('seeds', None)
    
    print(f"\n📂 PHASE 1: Loading datasets")
    
    # Load uniform datasets (for reference scalers and baseline comparison)
    print(f"Loading uniform training and test datasets...")
    uniform_dataset, test_dataset = load_datasets(
        src_file_train, src_file_test, nspecies, num_pressure_conditions
    )
    print(f"✅ Uniform datasets loaded")
    
    # Load batch simulation data
    print(f"Loading batch simulation data...")
    k_values, compositions, n_simulations, n_pressure_conditions = load_batch_data(batch_json_path)
    print(f"✅ Batch data loaded")
    
    print(f"\n🔧 PHASE 2: Creating batch dataset")
    
    # Create batch dataset using same scalers as uniform dataset
    batch_dataset = create_batch_dataset_using_active_methods(
        k_values, compositions, nspecies, num_pressure_conditions,
        reference_dataset=uniform_dataset
    )

    print("Phase 3 Seed defnition")
    seed_list = [40]

    for seed_shuffle in seed_list:
        uniform_dataset.x_data, uniform_dataset.y_data = shuffle(uniform_dataset.x_data, uniform_dataset.y_data, random_state= seed_shuffle)
        batch_dataset.x_data, batch_dataset.y_data = shuffle(batch_dataset.x_data, batch_dataset.y_data, random_state= seed_shuffle)


    batch_dataset.x_data = batch_dataset.x_data[:1600]
    batch_dataset.y_data = batch_dataset.y_data[:1600]

    
    
    print(f"\n🔬 PHASE 3: Training models")
    
    # Train models on batch data
    print(f"Training on batch data ({n_simulations} samples)...")
    batch_models_per_seed, batch_mse_per_output, batch_total_test_mse = train_and_evaluate_batch_models_using_active_methods(
        batch_dataset, test_dataset, best_params, seeds=seeds
    )
    
    # Train baseline models on uniform data (same sample size)
    print(f"\nTraining baseline on uniform data ({n_simulations} samples)...")
    uniform_models_per_seed, uniform_mse_per_output, uniform_total_test_mse = compare_with_uniform_baseline(
        uniform_dataset, test_dataset, best_params,
        batch_size=1600, seeds=seeds
    )
    
    # HYBRID EXPERIMENT: Train on 100 uniform + 500 batch samples
    print(f"\n🔬 HYBRID EXPERIMENT: Training on 100 uniform + 500 batch samples...")
    
    # Get 100 samples from uniform dataset
    x_uniform, y_uniform = uniform_dataset.get_data()
    x_uniform_100 = x_uniform[:100]
    y_uniform_100 = y_uniform[:100]
    
    # Get batch data (from the dataset we already created)
    x_batch, y_batch = batch_dataset.get_data()

    print("???????????????????????????????????")
    print(f"   new_x first 3 rows:\n{x_batch[:3]}")
    print(f"   new_y_scaled first 3 rows:\n{y_batch[:3]}")
    
    print(f"   Uniform subset: {x_uniform_100.shape}, {y_uniform_100.shape}")
    print(f"   Batch data: {x_batch.shape}, {y_batch.shape}")
    
    # Combine uniform + batch data
    x_hybrid = np.vstack([x_uniform_100, x_batch])
    y_hybrid = np.vstack([y_uniform_100, y_batch])
    
    print(f"   Combined hybrid: {x_hybrid.shape}, {y_hybrid.shape}")
    
    # Create hybrid dataset object
    class HybridDataset:
        def __init__(self, x_data, y_data):
            self.x_data = x_data
            self.y_data = y_data
            
        def get_data(self):
            return self.x_data, self.y_data
    
    hybrid_dataset = HybridDataset(x_hybrid, y_hybrid)
    
    # Train on hybrid dataset
    print(f"   Training on hybrid dataset (600 total samples)...")
    hybrid_models_per_seed, hybrid_mse_per_output, hybrid_total_test_mse = train_initial_models(
        hybrid_dataset, test_dataset, best_params,
        n_initial_samples=500, seeds=seeds  # Use all 600 samples
    )
    
    # RETRAIN EXPERIMENT: Use retrain_models_with_new_data() like active learning
    print(f"\n🔬 RETRAIN EXPERIMENT: Using retrain_models_with_new_data() method...")
    print(f"   This mimics EXACTLY what active_learning_train.py does on first iteration")
    
    # Setup exactly like active learning: start with 100 uniform, add 500 batch
    x_uniform, y_uniform = uniform_dataset.get_data()
    current_x_train = x_uniform[:100]  # Initial 100 uniform samples
    current_y_train = y_uniform[:100]
    
    print("##########################################")
    print(f"   current_x_train first 3 rows:\n{current_x_train[:3]}")
    print(f"   current_y_train first 3 rows:\n{current_y_train[:3]}")
    
    print(f"   Initial training data: {current_x_train.shape}, {current_y_train.shape}")
    
    # Get the NEW data (500 batch samples) - use raw data and apply_training_scalers
    print(f"   Processing new batch data with apply_training_scalers...")
    new_x, new_y_scaled = apply_training_scalers(
        raw_compositions=compositions,
        raw_k_values=k_values,
        dataset_train=uniform_dataset,
        nspecies=nspecies,
        num_pressure_conditions=num_pressure_conditions,
        debug=True  # Less verbose
    )


    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"   new_x first 3 rows:\n{new_x[:3]}")
    print(f"   new_y_scaled first 3 rows:\n{new_y_scaled[:3]}")
    
    print(f"   New batch data (scaled): {new_x.shape}, {new_y_scaled.shape}")
    
    # Call retrain_models_with_new_data EXACTLY like active learning
    print(f"   Calling retrain_models_with_new_data() - EXACT active learning method...")
    retrain_models_per_seed, retrain_mse_per_output, retrain_total_test_mse, augmented_size, _x_shuf, _y_shuf = retrain_models_with_new_data(
        current_x_train=current_x_train,
        current_y_train=current_y_train,
        dataset_test=test_dataset,
        new_x=new_x,
        new_y_scaled=new_y_scaled,
        best_params=best_params,
        seeds=seeds,
        debug=True
    )
    
    print(f"   Retrain method completed. Augmented size: {augmented_size}")
    
    print(f"\n🏆 PHASE 4: Results comparison")
    
    print(f"\n{'='*60}")
    print(f"BATCH vs UNIFORM vs HYBRID vs RETRAIN COMPARISON")
    print(f"{'='*60}")
    
    print(f"\n📊 Test MSE per output:")
    for i in range(len(batch_mse_per_output)):
        batch_mse = batch_mse_per_output[i]
        uniform_mse = uniform_mse_per_output[i]
        hybrid_mse = hybrid_mse_per_output[i]
        retrain_mse = retrain_mse_per_output[i]
        
        batch_ratio = batch_mse / uniform_mse if uniform_mse > 0 else float('inf')
        hybrid_ratio = hybrid_mse / uniform_mse if uniform_mse > 0 else float('inf')
        retrain_ratio = retrain_mse / uniform_mse if uniform_mse > 0 else float('inf')
        
        batch_status = "✅" if batch_ratio <= 1.1 else "⚠️" if batch_ratio <= 2.0 else "❌"
        hybrid_status = "✅" if hybrid_ratio <= 1.1 else "⚠️" if hybrid_ratio <= 2.0 else "❌"
        retrain_status = "✅" if retrain_ratio <= 1.1 else "⚠️" if retrain_ratio <= 2.0 else "❌"
        
        print(f"   Output {i}:")
        print(f"     Batch (500):     {batch_mse:.6f} (ratio: {batch_ratio:.2f}) {batch_status}")
        print(f"     Uniform (500):   {uniform_mse:.6f} (baseline)")
        print(f"     Hybrid (600):    {hybrid_mse:.6f} (ratio: {hybrid_ratio:.2f}) {hybrid_status}")
        print(f"     Retrain (600):   {retrain_mse:.6f} (ratio: {retrain_ratio:.2f}) {retrain_status}")
    
    print(f"\n📊 Total Test MSE:")
    batch_ratio = batch_total_test_mse / uniform_total_test_mse if uniform_total_test_mse > 0 else float('inf')
    hybrid_ratio = hybrid_total_test_mse / uniform_total_test_mse if uniform_total_test_mse > 0 else float('inf')
    retrain_ratio = retrain_total_test_mse / uniform_total_test_mse if uniform_total_test_mse > 0 else float('inf')
    
    batch_status = "✅" if batch_ratio <= 1.1 else "⚠️" if batch_ratio <= 2.0 else "❌"
    hybrid_status = "✅" if hybrid_ratio <= 1.1 else "⚠️" if hybrid_ratio <= 2.0 else "❌"
    retrain_status = "✅" if retrain_ratio <= 1.1 else "⚠️" if retrain_ratio <= 2.0 else "❌"
    
    print(f"   Batch (500):       {batch_total_test_mse:.6f} (ratio: {batch_ratio:.2f}) {batch_status}")
    print(f"   Uniform (500):     {uniform_total_test_mse:.6f} (baseline)")
    print(f"   Hybrid (600):      {hybrid_total_test_mse:.6f} (ratio: {hybrid_ratio:.2f}) {hybrid_status}")
    print(f"   Retrain (600):     {retrain_total_test_mse:.6f} (ratio: {retrain_ratio:.2f}) {retrain_status}")
    
    batch_improvement = (uniform_total_test_mse - batch_total_test_mse) / uniform_total_test_mse * 100
    hybrid_improvement = (uniform_total_test_mse - hybrid_total_test_mse) / uniform_total_test_mse * 100
    retrain_improvement = (uniform_total_test_mse - retrain_total_test_mse) / uniform_total_test_mse * 100
    
    print(f"\n📊 Performance vs Uniform Baseline:")
    print(f"   Batch improvement:   {batch_improvement:+.1f}% {'(batch better)' if batch_improvement > 0 else '(uniform better)'}")
    print(f"   Hybrid improvement:  {hybrid_improvement:+.1f}% {'(hybrid better)' if hybrid_improvement > 0 else '(uniform better)'}")
    print(f"   Retrain improvement: {retrain_improvement:+.1f}% {'(retrain better)' if retrain_improvement > 0 else '(uniform better)'}")
    
    print(f"\n📊 Training vs Test MSE (Overfitting Check):")
    print(f"   Note: Both models trained using same active_learning_methods")
    print(f"   (Detailed train/test MSE already printed above)")
    
    print(f"\n� Key Findings:")
    
    # Compare batch vs uniform
    if batch_ratio <= 1.1:
        print(f"   🎉 Batch data performs as well as uniform sampling!")
    elif batch_ratio <= 1.5:
        print(f"   ✅ Batch data performs reasonably well")
    elif batch_ratio <= 2.0:
        print(f"   ⚠️ Batch data underperforms uniform sampling")
    else:
        print(f"   ❌ Batch data significantly underperforms uniform sampling")
    
    # Compare hybrid vs others
    if hybrid_ratio < batch_ratio and hybrid_ratio < 1.0:
        print(f"   🚀 HYBRID WINS: Combining uniform + batch gives the best performance!")
    elif hybrid_ratio < batch_ratio:
        print(f"   ✅ Hybrid improves over batch alone")
    elif hybrid_ratio < 1.1:
        print(f"   ✅ Hybrid performs as well as uniform baseline")
    else:
        print(f"   ⚠️ Hybrid doesn't improve over uniform baseline")
    
    print(f"\n📊 Sample Efficiency:")
    print(f"   Uniform (500 samples):     MSE = {uniform_total_test_mse:.6f}")
    print(f"   Batch (500 samples):       MSE = {batch_total_test_mse:.6f}")
    print(f"   Hybrid (100+500=600):      MSE = {hybrid_total_test_mse:.6f}")
    
    print(f"\n🔍 Possible Issues to Investigate:")
    print(f"   - Data distribution mismatch between batch and uniform")
    print(f"   - Scaling inconsistencies")
    print(f"   - Sampling bias in batch generation")
    print(f"   - Simulation accuracy issues")
    
    print(f"\n🔧 RETRAIN METHOD DIAGNOSIS:")
    if 'retrain_ratio' in locals():
        if abs(retrain_ratio - hybrid_ratio) < 0.1:
            print(f"   ✅ retrain_models_with_new_data() works correctly (similar to hybrid)")
        elif retrain_ratio > hybrid_ratio * 2:
            print(f"   🚨 BUG FOUND: retrain_models_with_new_data() performs much worse!")
            print(f"      Retrain ratio: {retrain_ratio:.2f}, Hybrid ratio: {hybrid_ratio:.2f}")
            print(f"      This explains the poor active learning performance!")
        elif retrain_ratio > hybrid_ratio * 1.5:
            print(f"   ⚠️ retrain_models_with_new_data() has issues")
            print(f"      Retrain ratio: {retrain_ratio:.2f}, Hybrid ratio: {hybrid_ratio:.2f}")
        else:
            print(f"   ✅ retrain_models_with_new_data() performs reasonably")
    else:
        print(f"   ❓ Retrain experiment not completed")
    
    print(f"\n✅ EXPERIMENT COMPLETE!")
