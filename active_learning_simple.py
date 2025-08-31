"""
Simple Active Learning with Zone-based Sampling for Rate Coefficient Determination

Clean, simple implementation without unnecessary complexity.
Uses real BatchSimulator directly.

Author: AI Assistant  
Date: August 16, 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Add adaptive sampling modules to path
current_dir = os.path.dirname(os.path.abspath(__file__))
adaptive_dir = os.path.join(current_dir, 'adaptive_sampling', 'src')
sys.path.append(adaptive_dir)

from adaptive_sampling.src.batch_simulator import BatchSimulator
from adaptive_sampling.src.base_simulator import LoKISimulator
from adaptive_sampling.src.sampling_strategies import BoundsBasedSampler


class LoadMultiPressureDatasetNumpy:

    def __init__(self, src_file, nspecies, num_pressure_conditions, react_idx=None, m_rows=None, columns=None,
                 scaler_input=None, scaler_output=None):
        self.num_pressure_conditions = num_pressure_conditions

        all_data = np.loadtxt(src_file, max_rows=m_rows,
                              usecols=columns, delimiter="  ",
                              comments="#", skiprows=0, dtype=np.float64)

        ncolumns = len(all_data[0])
        x_columns = np.arange(ncolumns - nspecies, ncolumns, 1)
        y_columns = react_idx
        if react_idx is None:
            y_columns = np.arange(0, ncolumns - nspecies, 1)

        x_data = all_data[:, x_columns]  # densities
        y_data = all_data[:, y_columns] * 1e30  # k's  # *10 to avoid being at float32 precision limit 1e-17

        # Reshape data for multiple pressure conditions
        x_data = x_data.reshape(num_pressure_conditions, -1, x_data.shape[1])
        y_data = y_data.reshape(num_pressure_conditions, -1, y_data.shape[1])

        # Create scalers
        self.scaler_input = scaler_input or [preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)]
        self.scaler_output = scaler_output or [preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)]
        
        for i in range(num_pressure_conditions):
            if scaler_input is None:
                self.scaler_input[i].fit(x_data[i])
            if scaler_output is None:
                self.scaler_output[i].fit(y_data[i])
            x_data[i] = self.scaler_input[i].transform(x_data[i])
            y_data[i] = self.scaler_output[i].transform(y_data[i])

        # Transpose x_data to move the pressure condition axis to the end, then flatten
        x_data = np.transpose(x_data, (1, 0, 2)).reshape(-1, self.num_pressure_conditions * x_data.shape[-1])
        
        # Flatten the output data to be of shape (2000,3)
        y_data = y_data[0]


        # Assign the preprocessed data
        self.x_data = x_data
        self.y_data = y_data
        self.all_data = all_data

    def get_data(self):
        """
        Return the preprocessed input and output data.
        """
        return self.x_data, self.y_data

def create_simulator(config: Dict) -> BatchSimulator:
    """
    Create BatchSimulator - simple and clean
    """
    print(f"🔬 Creating LoKI BatchSimulator...")
    
    # Use pressure conditions from config
    pressure_conditions = config['pressure_conditions_pa']
    
    # Create LoKI simulator
    base_simulator = LoKISimulator(
        setup_file="setup_O2_simple.in",
        chem_file="O2_simple_1.chem", 
        loki_path="C:\\MyPrograms\\LoKI_v3.1.0-v2",
        k_columns=[0, 1, 2],
        simulation_type="simple",
        pressure_conditions=pressure_conditions
    )
    
    # Create sampler for K value bounds
    k_bounds = np.array([
        [1e-19, 1e-15],  # K1 bounds
        [1e-20, 1e-16],  # K2 bounds  
        [1e-18, 1e-14]   # K3 bounds
    ])
    sampler = BoundsBasedSampler(k_bounds, sampling_method='latin_hypercube')
    
    # Create batch simulator
    batch_simulator = BatchSimulator(base_simulator=base_simulator, sampler=sampler)
    
    print(f"✅ BatchSimulator ready with {len(pressure_conditions)} pressure conditions")
    return batch_simulator


class SimpleActiveLearning:
    """
    Simple Active Learning - no unnecessary complexity
    """
    
    def __init__(self, batch_simulator: BatchSimulator, svr_params: List[Dict], config: Dict, 
                 scaler_input=None, scaler_output=None):
        self.batch_simulator = batch_simulator
        self.svr_params = svr_params
        self.config = config
        self.training_x = np.empty((0, 6))  # Will be filled as we collect samples
        self.training_y = np.empty((0, 3))  # 3 outputs for O2_simple
        self.models = []
        
        print(f"🎯 Simple Active Learning initialized")
    
    def add_training_data(self, x_data: np.ndarray, y_data: np.ndarray):
        """Add data to training set"""
        if len(self.training_x) == 0:
            self.training_x = x_data.copy()
            self.training_y = y_data.copy()
        else:
            self.training_x = np.vstack([self.training_x, x_data])
            self.training_y = np.vstack([self.training_y, y_data])
            
        print(f"   Training set now has {len(self.training_x)} samples")
    
    def train_models(self):
        """Train SVR models"""
        print(f"🧠 Training models on {len(self.training_x)} samples...")
        
        self.models = []
        for i, params in enumerate(self.svr_params):
            model = SVR(**params)
            model.fit(self.training_x, self.training_y[:, i])
            self.models.append(model)
        
        print(f"✅ Trained {len(self.models)} models")
    
    def evaluate_performance(self, test_dataset):
        """Evaluate current model performance - matching sample_effiency.py calculation"""
        x_test, y_test = test_dataset.get_data()
        
        predictions = []
        for model in self.models:
            pred = model.predict(x_test)
            predictions.append(pred)
        predictions = np.array(predictions).T
        
        # Calculate MSE for each output
        mse_per_output = []
        for i in range(len(self.svr_params)):
            mse = mean_squared_error(y_test[:, i], predictions[:, i])
            mse_per_output.append(mse)
        
        # Sum MSE across outputs (to match sample_effiency.py exactly)
        total_mse = np.sum(mse_per_output)
        
        return total_mse, mse_per_output
    
    def run_active_learning(self, test_dataset, uniform_data: Tuple[np.ndarray, np.ndarray], 
                           n_iterations: int = 5, n_samples_per_iteration: int = 24) -> Dict:
        """
        Run simple active learning process
        """
        print(f"🔄 Starting Simple Active Learning")
        print(f"   Iterations: {n_iterations}")
        print(f"   Samples per iteration: {n_samples_per_iteration}")
        
        # Track learning curve
        learning_curve = {
            'iteration': [],
            'n_samples': [],
            'mse_per_output': [],
            'total_mse': []
        }
        
        # Start with uniform data
        x_uniform, y_uniform = uniform_data
        self.add_training_data(x_uniform, y_uniform)
        
        # DEBUGGING: Test initial performance with just uniform data
        print(f"🔍 DEBUGGING: Testing initial performance with {len(x_uniform)} uniform samples...")
        self.train_models()
        initial_mse, initial_mse_per_output = self.evaluate_performance(test_dataset)
        print(f"   Initial MSE with uniform data: {initial_mse:.6f}")
        print(f"   MSE per output: {initial_mse_per_output}")
        
        # Active learning iterations
        for iteration in range(n_iterations):
            print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
            
            # Train models
            self.train_models()
            
            # Evaluate performance
            total_mse, mse_per_output = self.evaluate_performance(test_dataset)
            
            # Record learning curve
            learning_curve['iteration'].append(iteration)
            learning_curve['n_samples'].append(len(self.training_x))
            learning_curve['mse_per_output'].append(mse_per_output.copy())
            learning_curve['total_mse'].append(total_mse)
            
            print(f"📊 Current Performance:")
            print(f"   Total samples: {len(self.training_x)}")
            print(f"   Total MSE: {total_mse:.6f}")
            
            # Generate new samples (after evaluation, even on last iteration)
            if iteration < n_iterations:  # Generate samples on ALL iterations
                print(f"🎯 Generating {n_samples_per_iteration} new samples...")
                
                # Use BatchSimulator to generate new samples
                batch_results = self.batch_simulator.run_with_sampling(
                    n_samples=n_samples_per_iteration,
                    parallel_workers=1
                )
                
                # FIXED: Properly convert simulation results to training format
                print(f"   📊 Batch results: {batch_results.compositions.shape} compositions from {len(batch_results.parameter_sets)} parameter sets")
                
                # The simulation outputs densities for each pressure condition
                # We need to flatten these to match the training data format
                densities = batch_results.compositions  # Shape: (n_samples * n_pressures, n_species)
                
                # Reshape to match expected input format: (n_samples, n_pressures * n_species)
                n_pressures = len(self.config['pressure_conditions_pa'])
                n_species = densities.shape[1] 
                n_samples = len(densities) // n_pressures
                
                # Reshape densities to (n_samples, n_pressures * n_species) - same format as original training data
                new_x = densities.reshape(n_samples, n_pressures * n_species)
                
                # Extract K values from ParameterSet objects  
                new_y = np.array([ps.k_values for ps in batch_results.parameter_sets])
                
                print(f"   📊 Converted data: new_x.shape={new_x.shape}, new_y.shape={new_y.shape}")
                print(f"   📊 Training data expects: x={self.training_x.shape}, y={self.training_y.shape}")
                
                self.add_training_data(new_x, new_y)
                
                # Re-evaluate with new samples for learning curve
                if iteration == n_iterations - 1:  # Final evaluation with new samples
                    self.train_models()
                    total_mse, mse_per_output = self.evaluate_performance(test_dataset)
                    learning_curve['iteration'].append(iteration + 0.5)  # Mark as final
                    learning_curve['n_samples'].append(len(self.training_x))
                    learning_curve['mse_per_output'].append(mse_per_output.copy())
                    learning_curve['total_mse'].append(total_mse)
                    print(f"📊 Final Performance with new samples:")
                    print(f"   Total samples: {len(self.training_x)}")
                    print(f"   Total MSE: {total_mse:.6f}")
        
        print(f"\n✅ Simple Active Learning Complete!")
        print(f"   Final samples: {len(self.training_x)}")
        print(f"   Final MSE: {learning_curve['total_mse'][-1]:.6f}")
        
        return learning_curve


def load_uniform_training_data(nspecies: int = 3, num_pressure_conditions: int = 2, 
                               n_initial_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """Load uniform sampling data"""
    src_file_uniform = 'data/SampleEfficiency/O2_simple_uniform.txt'
    
    try:
        print(f"📊 Loading uniform training data from SampleEfficiency/O2_simple_uniform.txt...")
        
        # Load uniform dataset - EXACTLY like sample_effiency.py
        uniform_dataset = LoadMultiPressureDatasetNumpy(
            src_file_uniform, nspecies, num_pressure_conditions, 
            react_idx=[0, 1, 2], m_rows=n_initial_samples
        )
        
        x_uniform, y_uniform = uniform_dataset.get_data()
        x_uniform, y_uniform = shuffle(x_uniform, y_uniform, random_state=42)
        
        print(f"✅ Loaded uniform training data: {x_uniform.shape}, {y_uniform.shape}")
        
        return x_uniform, y_uniform, uniform_dataset.scaler_input, uniform_dataset.scaler_output
        
    except FileNotFoundError:
        print(f"⚠️  Uniform training file not found: {src_file_uniform}")
        return None, None, None, None


def load_test_dataset(nspecies: int = 3, num_pressure_conditions: int = 2,
                      scaler_input=None, scaler_output=None):
    """Load test dataset"""
    src_file_test = 'data/SampleEfficiency/O2_simple_test.txt'
    
    try:
        test_dataset = LoadMultiPressureDatasetNumpy(
            src_file_test, nspecies, num_pressure_conditions, react_idx=[0, 1, 2],
            scaler_input=scaler_input, scaler_output=scaler_output
        )
        print(f"📊 Loaded test dataset: {len(test_dataset.get_data()[0])} samples")
        return test_dataset
        
    except FileNotFoundError:
        print(f"⚠️  Test file not found: {src_file_test}")
        return None


def run_sample_efficiency_baseline(test_dataset, uniform_data: Tuple[np.ndarray, np.ndarray], 
                                  svr_params: List[Dict], config: Dict):
    """Run baseline exactly like sample_effiency.py for debugging"""
    print("📊 Running sample efficiency baseline...")
    
    x_uniform, y_uniform = uniform_data
    x_test, y_test = test_dataset.get_data()
    
    # TEST: Use the same subset sizes as sample_effiency.py for comparison
    test_sizes = [100, 200, 300, 400, 500]  # Smaller range for debugging
    
    print(f"   Baseline will test with: {test_sizes}")
    print(f"   DEBUG: x_uniform.shape={x_uniform.shape}, y_uniform.shape={y_uniform.shape}")
    print(f"   DEBUG: x_test.shape={x_test.shape}, y_test.shape={y_test.shape}")
    
    baseline_curve = {
        'n_samples': [],
        'total_mse': [],
        'mse_per_output': []
    }
    
    for n_samples in test_sizes:
        # Shuffle data exactly like sample_effiency.py (for each subset size)
        x_shuffled, y_shuffled = shuffle(x_uniform, y_uniform, random_state=42)
        
        # Use first n_samples from shuffled data
        x_train = x_shuffled[:n_samples]
        y_train = y_shuffled[:n_samples]
            
        print(f"   Testing with {len(x_train)} uniform samples...")
        print(f"     DEBUG: x_train.shape={x_train.shape}, y_train.shape={y_train.shape}")
        
        # Train SVR models exactly like sample_effiency.py
        mse_per_output = []
        
        for i, params in enumerate(svr_params):
            svr = SVR(**params)
            svr.fit(x_train, y_train[:, i])
            
            # Evaluate on test set
            y_pred = svr.predict(x_test)
            mse = mean_squared_error(y_test[:, i], y_pred)
            mse_per_output.append(mse)
            
            print(f"     DEBUG: Output {i}: MSE={mse:.6f}")
        
        # Sum MSE across outputs (to match sample_effiency.py exactly)
        total_mse = np.sum(mse_per_output)
        
        baseline_curve['n_samples'].append(n_samples)
        baseline_curve['total_mse'].append(total_mse)
        baseline_curve['mse_per_output'].append(mse_per_output.copy())
        
        print(f"      MSE per output: {mse_per_output}")
        print(f"      Total MSE (sum): {total_mse:.6f}")
    
    return baseline_curve


def main():
    """Main function - simple and clean"""
    print("🚀 Simple Active Learning for Rate Coefficient Determination")
    
    # Configuration
    config = {
        'nspecies': 3,
        'num_pressure_conditions': 2,
        'pressure_conditions_pa': [133.322, 1333.22],  # 1 and 10 Torr
        'initial_samples_from_uniform': 500,
        'n_iterations': 1,
        'n_samples_per_iteration': 2,
        'svr_params': [
            {'C': 10, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'},
            {'C': 20, 'epsilon': 0.005, 'gamma': 5, 'kernel': 'rbf'},
            {'C': 5, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'}
        ]
    }
    
    print(f"Configuration: {config}")
    
    # Load data first to see how much we actually have
    x_all_uniform, y_all_uniform, scaler_input, scaler_output = load_uniform_training_data(
        config['nspecies'], 
        config['num_pressure_conditions'],
        1000  # Load more data than config asks for
    )
    
    if x_all_uniform is None:
        print("❌ No training data available")
        return
    
    # Calculate total samples needed for fair comparison
    initial_samples = config['initial_samples_from_uniform']
    additional_samples = config['n_iterations'] * config['n_samples_per_iteration']
    total_samples_needed = initial_samples + additional_samples
    
    print(f"📊 Data loaded: {len(x_all_uniform)} samples available")
    print(f"📊 Active Learning: {initial_samples} initial + {additional_samples} adaptive = {total_samples_needed} total")
    
    # Separate data for active learning vs baseline
    # Active learning gets only initial samples from uniform data
    x_uniform_active = x_all_uniform[:initial_samples]
    y_uniform_active = y_all_uniform[:initial_samples]
    
    # Baseline gets all samples needed for fair comparison
    x_uniform_baseline = x_all_uniform[:total_samples_needed]
    y_uniform_baseline = y_all_uniform[:total_samples_needed]
    
    print(f"📊 Active Learning uniform data: {len(x_uniform_active)} samples")
    print(f"📊 Baseline uniform data: {len(x_uniform_baseline)} samples")
    
    test_dataset = load_test_dataset(
        config['nspecies'], 
        config['num_pressure_conditions'],
        scaler_input=scaler_input, 
        scaler_output=scaler_output
    )
    
    if test_dataset is None:
        print("❌ No test data available")
        return
    
    # COMMENTED OUT ACTIVE LEARNING - FOCUS ON BASELINE ONLY
    # # Create simulator
    # simulator = create_simulator(config)
    # 
    # # Create active learning system
    # active_learner = SimpleActiveLearning(simulator, config['svr_params'], config)
    # 
    # # Run active learning
    # uniform_data_active = (x_uniform_active, y_uniform_active)
    # active_curve = active_learner.run_active_learning(
    #     test_dataset=test_dataset,
    #     uniform_data=uniform_data_active,
    #     n_iterations=config['n_iterations'],
    #     n_samples_per_iteration=config['n_samples_per_iteration']
    # )
    
    print("🚀 BASELINE ONLY - Skipping active learning")
    
    # Create dummy active_curve for compatibility
    active_curve = {'total_mse': [0.1]}  # Dummy data
    
    # Run baseline comparison with separate baseline data
    uniform_data_baseline = (x_uniform_baseline, y_uniform_baseline)
    baseline_curve = run_sample_efficiency_baseline(test_dataset, uniform_data_baseline, config['svr_params'], config)
    
    # Check if we have valid results
    if not baseline_curve['total_mse'] or not active_curve['total_mse']:
        print("❌ No valid results to compare")
        return
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(baseline_curve['n_samples'], baseline_curve['total_mse'], 'r-o', 
             label='Sample Efficiency Baseline', linewidth=2, markersize=8)
    plt.plot(active_curve['n_samples'], active_curve['total_mse'], 'b-o', 
             label='Simple Active Learning', linewidth=2, markersize=8)
    plt.xlabel('Number of Samples')
    plt.ylabel('Total MSE')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final performance comparison
    plt.subplot(1, 2, 2)
    baseline_final = baseline_curve['total_mse'][-1]
    active_final = active_curve['total_mse'][-1]
    improvement = (baseline_final - active_final) / baseline_final * 100
    
    methods = ['Baseline', 'Active Learning']
    mse_values = [baseline_final, active_final]
    colors = ['red', 'blue']
    
    bars = plt.bar(methods, mse_values, color=colors, alpha=0.7)
    plt.ylabel('Final MSE')
    plt.title(f'Final Performance\n(Improvement: {improvement:.1f}%)')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.5f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('simple_active_learning_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Final Results:")
    print(f"   Baseline Final MSE: {baseline_final:.6f}")
    print(f"   Active Learning Final MSE: {active_final:.6f}")
    print(f"   Improvement: {improvement:.1f}%")
    
    if improvement > 0:
        print("✅ Active learning improved performance!")
    else:
        print("⚠️  Active learning needs more work")


if __name__ == "__main__":
    main()
