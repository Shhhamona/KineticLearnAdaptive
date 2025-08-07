"""
Traditional approach: Generate training data once and train a classifier.

This implements the baseline approach where we generate a fixed dataset
and train a model, then evaluate its performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import Dict, Tuple, Any

from base_simulator import BaseSimulator
from ml_models import RateCoefficientPredictor, MultiOutputSVRPredictor
from sampling_strategies import BoundsBasedSampler, LatinHypercubeSampler


class TraditionalApproach:
    """
    Traditional machine learning approach for rate coefficient determination.
    
    Steps:
    1. Define parameter space bounds
    2. Generate training data by sampling K values and running simulations
    3. Train ML model on (C -> K) mapping
    4. Evaluate on test set
    """
    
    def __init__(self, 
                 simulator: BaseSimulator,
                 k_columns: list,
                 model_type: str = 'random_forest',
                 sampling_method: str = 'latin_hypercube',
                 random_state: int = 42,
                 **model_kwargs):
        """
        Initialize traditional approach.
        
        Args:
            simulator: Simulation backend
            k_columns: Which K coefficients to vary
            model_type: Type of ML model to use
            sampling_method: How to sample K values
            random_state: Random seed
            **model_kwargs: Additional arguments for ML model
        """
        self.simulator = simulator
        self.k_columns = k_columns
        self.model_type = model_type
        self.sampling_method = sampling_method
        self.random_state = random_state
        
        # Initialize ML model
        if model_type == 'svm_multi':
            self.model = MultiOutputSVRPredictor(**model_kwargs)
        else:
            self.model = RateCoefficientPredictor(model_type, **model_kwargs)
        
        # Get reference K values and create bounds
        self.ref_k = self.simulator.get_reference_k_values()
        self.k_varied = self.ref_k[k_columns]
        
        # Create reasonable bounds around reference values
        self.k_bounds = self._create_k_bounds()
        
        # Initialize sampler
        self.sampler = BoundsBasedSampler(
            self.k_bounds, 
            sampling_method, 
            random_state
        )
        
        # Results storage
        self.training_history = {}
        self.test_results = {}
        
    def _create_k_bounds(self, bound_factor: float = 2.0) -> np.ndarray:
        """
        Create bounds for K values based on reference values.
        
        Args:
            bound_factor: Factor to multiply/divide reference values for bounds
            
        Returns:
            Bounds array, shape (n_k_varied, 2)
        """
        bounds = np.zeros((len(self.k_columns), 2))
        
        for i, k_val in enumerate(self.k_varied):
            bounds[i, 0] = k_val / bound_factor  # Lower bound
            bounds[i, 1] = k_val * bound_factor  # Upper bound
            
        return bounds
    
    def generate_training_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data by sampling K values and running simulations.
        
        Args:
            n_samples: Number of training samples to generate
            
        Returns:
            Tuple of (compositions, k_values)
        """
        print(f"Generating {n_samples} training samples...")
        start_time = time.time()
        
        # Sample K values
        k_samples = self.sampler.sample_full_space(n_samples)
        
        # Run simulations
        compositions = self.simulator.run_simulations(k_samples)
        
        generation_time = time.time() - start_time
        print(f"Training data generated in {generation_time:.2f} seconds")
        
        return compositions, k_samples
    
    def generate_test_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate test data using same approach as training data.
        
        Args:
            n_samples: Number of test samples
            
        Returns:
            Tuple of (compositions, k_values)
        """
        print(f"Generating {n_samples} test samples...")
        
        # Use different random state for test data
        test_sampler = BoundsBasedSampler(
            self.k_bounds, 
            self.sampling_method, 
            self.random_state + 1000
        )
        
        # Sample K values
        k_samples = test_sampler.sample_full_space(n_samples)
        
        # Run simulations
        compositions = self.simulator.run_simulations(k_samples)
        
        return compositions, k_samples
    
    def train_model(self, compositions: np.ndarray, k_values: np.ndarray) -> Dict[str, Any]:
        """
        Train the ML model on the generated data.
        
        Args:
            compositions: Training compositions, shape (n_samples, n_species)
            k_values: Training K values, shape (n_samples, n_k)
            
        Returns:
            Training metrics
        """
        print(f"Training {self.model_type} model...")
        start_time = time.time()
        
        # Train model
        metrics = self.model.fit(compositions, k_values)
        
        training_time = time.time() - start_time
        metrics['training_time'] = training_time
        
        print(f"Model trained in {training_time:.2f} seconds")
        print(f"Training R²: {metrics['r2_score']:.4f}")
        print(f"Training RMSE: {metrics['rmse']:.2e}")
        
        return metrics
    
    def evaluate_model(self, 
                      test_compositions: np.ndarray, 
                      test_k_values: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            test_compositions: Test compositions
            test_k_values: True test K values
            
        Returns:
            Test metrics
        """
        print("Evaluating model on test data...")
        
        metrics = self.model.evaluate(test_compositions, test_k_values)
        
        print(f"Test R²: {metrics['r2_score']:.4f}")
        print(f"Test RMSE: {metrics['rmse']:.2e}")
        print(f"Test relative error: {metrics['relative_error']:.4f}")
        
        return metrics
    
    def run_complete_study(self, 
                          n_train: int, 
                          n_test: int,
                          save_results: bool = True) -> Dict[str, Any]:
        """
        Run the complete traditional approach study.
        
        Args:
            n_train: Number of training samples
            n_test: Number of test samples
            save_results: Whether to save results
            
        Returns:
            Complete results dictionary
        """
        print("="*60)
        print("TRADITIONAL APPROACH")
        print("="*60)
        
        # Generate data
        train_compositions, train_k = self.generate_training_data(n_train)
        test_compositions, test_k = self.generate_test_data(n_test)
        
        # Train model
        train_metrics = self.train_model(train_compositions, train_k)
        
        # Evaluate model
        test_metrics = self.evaluate_model(test_compositions, test_k)
        
        # Compile results
        results = {
            'approach': 'traditional',
            'model_type': self.model_type,
            'sampling_method': self.sampling_method,
            'n_train': n_train,
            'n_test': n_test,
            'k_columns': self.k_columns,
            'k_bounds': self.k_bounds.tolist(),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'total_simulations': n_train + n_test
        }
        
        self.training_history = train_metrics
        self.test_results = test_metrics
        
        if save_results:
            self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        import json
        
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        filename = f"traditional_{self.model_type}_{int(time.time())}.json"
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def plot_results(self, test_compositions: np.ndarray, test_k_values: np.ndarray):
        """
        Plot prediction results.
        
        Args:
            test_compositions: Test compositions
            test_k_values: True test K values
        """
        # Make predictions
        k_pred = self.model.predict(test_compositions)
        
        # Create plots
        n_k = len(self.k_columns)
        fig, axes = plt.subplots(1, n_k, figsize=(5*n_k, 4))
        
        if n_k == 1:
            axes = [axes]
        
        for i in range(n_k):
            ax = axes[i]
            
            # Scatter plot: true vs predicted
            ax.scatter(test_k_values[:, i], k_pred[:, i], alpha=0.6, s=20)
            
            # Perfect prediction line
            k_min = min(test_k_values[:, i].min(), k_pred[:, i].min())
            k_max = max(test_k_values[:, i].max(), k_pred[:, i].max())
            ax.plot([k_min, k_max], [k_min, k_max], 'r--', alpha=0.8)
            
            # Labels and title
            ax.set_xlabel(f'True K{self.k_columns[i]+1}')
            ax.set_ylabel(f'Predicted K{self.k_columns[i]+1}')
            ax.set_title(f'K{self.k_columns[i]+1} Prediction')
            
            # Calculate R² for this coefficient
            r2 = 1 - np.sum((test_k_values[:, i] - k_pred[:, i])**2) / \
                     np.sum((test_k_values[:, i] - test_k_values[:, i].mean())**2)
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.suptitle(f'Traditional Approach - {self.model_type.title()}', y=1.02)
        
        # Save plot
        plots_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f'traditional_{self.model_type}_predictions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Prediction plot saved to: {plot_path}")
