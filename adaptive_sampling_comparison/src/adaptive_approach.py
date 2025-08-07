"""
Adaptive sampling approach: Iteratively refine sampling based on model predictions.

This implements the adaptive sampling strategy described in the methodology document.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import Dict, Tuple, Any, List

from base_simulator import BaseSimulator
from ml_models import RateCoefficientPredictor, MultiOutputSVRPredictor
from sampling_strategies import AdaptiveSampler


class AdaptiveSamplingApproach:
    """
    Adaptive sampling approach for rate coefficient determination.
    
    Steps:
    1. Start with literature K values (K') and generate target composition C'
    2. Generate initial training data by sampling around K'
    3. Train ML model on (C -> K) mapping
    4. Iteratively:
       - Predict K for target composition C'
       - Sample around predicted K with reduced hypercube
       - Add new data and retrain model
       - Check convergence
    """
    
    def __init__(self, 
                 simulator: BaseSimulator,
                 k_columns: list,
                 true_k_values: np.ndarray,
                 model_type: str = 'random_forest',
                 sampling_method: str = 'latin_hypercube',
                 max_iterations: int = 10,
                 convergence_threshold: float = 1e-3,
                 initial_hypercube_size: float = 0.5,
                 hypercube_reduction: float = 0.8,
                 random_state: int = 42,
                 **model_kwargs):
        """
        Initialize adaptive sampling approach.
        
        Args:
            simulator: Simulation backend
            k_columns: Which K coefficients to vary
            true_k_values: Literature/true K values for the varied coefficients
            model_type: Type of ML model to use
            sampling_method: How to sample K values
            max_iterations: Maximum number of adaptive iterations
            convergence_threshold: Threshold for convergence check
            initial_hypercube_size: Initial size of sampling hypercube
            hypercube_reduction: Factor to reduce hypercube each iteration
            random_state: Random seed
            **model_kwargs: Additional arguments for ML model
        """
        self.simulator = simulator
        self.k_columns = k_columns
        self.true_k_values = true_k_values
        self.model_type = model_type
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.random_state = random_state
        
        # Initialize ML model
        if model_type == 'svm_multi':
            self.model = MultiOutputSVRPredictor(**model_kwargs)
        else:
            self.model = RateCoefficientPredictor(model_type, **model_kwargs)
        
        # Initialize adaptive sampler
        self.sampler = AdaptiveSampler(
            initial_hypercube_size=initial_hypercube_size,
            hypercube_reduction=hypercube_reduction,
            sampling_method=sampling_method,
            random_state=random_state
        )
        
        # Generate target composition from true K values
        self.target_composition = self._generate_target_composition()
        
        # History tracking
        self.history = {
            'iterations': 0,
            'all_compositions': [],
            'all_k_values': [],
            'predictions': [],
            'model_scores': [],
            'convergence_metrics': [],
            'hypercube_sizes': [],
            'simulation_counts': []
        }
        
    def _generate_target_composition(self) -> np.ndarray:
        """
        Generate target composition C' by running simulation with true K values.
        
        Returns:
            Target composition, shape (n_species,)
        """
        print("Generating target composition from true K values...")
        
        # Create single simulation with true K values
        true_k_full = np.array([self.true_k_values])  # Shape (1, n_k)
        target_compositions = self.simulator.run_simulations(true_k_full)
        
        target_composition = target_compositions[0]  # Extract single composition
        print(f"Target composition generated: shape {target_composition.shape}")
        
        return target_composition
    
    def initial_sampling(self, n_initial: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate initial training data around true K values.
        
        Args:
            n_initial: Number of initial samples
            
        Returns:
            Tuple of (compositions, k_values)
        """
        print(f"Initial sampling: {n_initial} samples around true K values")
        
        # Sample around true K values
        k_samples = self.sampler.initial_sample(self.true_k_values, n_initial)
        
        # Run simulations
        compositions = self.simulator.run_simulations(k_samples)
        
        # Store in history
        self.history['all_compositions'].append(compositions)
        self.history['all_k_values'].append(k_samples)
        self.history['simulation_counts'].append(n_initial)
        
        return compositions, k_samples
    
    def train_model(self, compositions: np.ndarray, k_values: np.ndarray) -> Dict[str, float]:
        """
        Train the ML model on current data.
        
        Args:
            compositions: All compositions so far
            k_values: All K values so far
            
        Returns:
            Training metrics
        """
        print(f"Training model on {compositions.shape[0]} samples...")
        
        metrics = self.model.fit(compositions, k_values)
        
        # Store metrics
        self.history['model_scores'].append(metrics['r2_score'])
        
        return metrics
    
    def predict_for_target(self) -> np.ndarray:
        """
        Predict K values for the target composition.
        
        Returns:
            Predicted K values, shape (n_k,)
        """
        target_reshaped = self.target_composition.reshape(1, -1)
        k_pred = self.model.predict(target_reshaped)
        
        return k_pred[0]  # Extract single prediction
    
    def adaptive_iteration(self, k_pred_current: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one adaptive sampling iteration.
        
        Args:
            k_pred_current: Current predicted K values
            n_samples: Number of samples for this iteration
            
        Returns:
            Tuple of (new_compositions, new_k_values)
        """
        print(f"Adaptive iteration {self.history['iterations'] + 1}: "
              f"{n_samples} samples around predicted K")
        
        # Sample around current prediction
        k_samples = self.sampler.adaptive_sample(k_pred_current, n_samples)
        
        # Run simulations
        compositions = self.simulator.run_simulations(k_samples)
        
        # Store in history
        self.history['all_compositions'].append(compositions)
        self.history['all_k_values'].append(k_samples)
        self.history['simulation_counts'].append(n_samples)
        
        return compositions, k_samples
    
    def check_convergence(self, k_pred_new: np.ndarray, k_pred_old: np.ndarray) -> Tuple[bool, float]:
        """
        Check if the adaptive sampling has converged.
        
        Args:
            k_pred_new: New prediction
            k_pred_old: Previous prediction
            
        Returns:
            Tuple of (converged, convergence_metric)
        """
        # Calculate relative change in predictions
        convergence_metric = np.linalg.norm(k_pred_new - k_pred_old) / np.linalg.norm(k_pred_old)
        
        converged = convergence_metric < self.convergence_threshold
        
        self.history['convergence_metrics'].append(convergence_metric)
        
        return converged, convergence_metric
    
    def run_adaptive_sampling(self, 
                             n_initial: int,
                             n_iteration: int,
                             save_results: bool = True) -> Dict[str, Any]:
        """
        Run the complete adaptive sampling procedure.
        
        Args:
            n_initial: Number of initial samples
            n_iteration: Number of samples per iteration
            save_results: Whether to save results
            
        Returns:
            Complete results dictionary
        """
        print("="*60)
        print("ADAPTIVE SAMPLING APPROACH")
        print("="*60)
        
        start_time = time.time()
        
        # Step 1: Initial sampling and training
        initial_compositions, initial_k = self.initial_sampling(n_initial)
        initial_metrics = self.train_model(initial_compositions, initial_k)
        
        # Initial prediction
        k_pred = self.predict_for_target()
        self.history['predictions'].append(k_pred.copy())
        
        print(f"Initial prediction: {k_pred}")
        print(f"True K values: {self.true_k_values}")
        
        initial_error = np.linalg.norm(k_pred - self.true_k_values) / np.linalg.norm(self.true_k_values)
        print(f"Initial relative error: {initial_error:.4f}")
        
        # Step 2: Adaptive iterations
        for iteration in range(self.max_iterations):
            self.history['iterations'] = iteration + 1
            
            # Store current hypercube size
            sampler_history = self.sampler.get_history()
            self.history['hypercube_sizes'] = sampler_history['hypercube_sizes']
            
            print(f"\n--- Iteration {iteration + 1} ---")
            print(f"Current hypercube size: {self.sampler.current_hypercube_size:.4f}")
            
            # Adaptive sampling
            new_compositions, new_k = self.adaptive_iteration(k_pred, n_iteration)
            
            # Combine all data
            all_compositions = np.vstack(self.history['all_compositions'])
            all_k_values = np.vstack(self.history['all_k_values'])
            
            # Retrain model
            metrics = self.train_model(all_compositions, all_k_values)
            
            # New prediction
            k_pred_new = self.predict_for_target()
            self.history['predictions'].append(k_pred_new.copy())
            
            # Check convergence
            converged, conv_metric = self.check_convergence(k_pred_new, k_pred)
            
            current_error = np.linalg.norm(k_pred_new - self.true_k_values) / np.linalg.norm(self.true_k_values)
            
            print(f"New prediction: {k_pred_new}")
            print(f"Convergence metric: {conv_metric:.6f}")
            print(f"Current relative error: {current_error:.4f}")
            print(f"Model R² score: {metrics['r2_score']:.4f}")
            
            # Update prediction for next iteration
            k_pred = k_pred_new
            
            if converged:
                print(f"\nConverged after {iteration + 1} iterations!")
                break
        
        total_time = time.time() - start_time
        total_simulations = sum(self.history['simulation_counts'])
        
        # Final results
        final_error = np.linalg.norm(k_pred - self.true_k_values) / np.linalg.norm(self.true_k_values)
        
        print("\n" + "="*60)
        print("ADAPTIVE SAMPLING RESULTS")
        print("="*60)
        print(f"Final prediction: {k_pred}")
        print(f"True K values: {self.true_k_values}")
        print(f"Final relative error: {final_error:.4f}")
        print(f"Initial relative error: {initial_error:.4f}")
        print(f"Error improvement: {(initial_error - final_error)/initial_error*100:.2f}%")
        print(f"Total simulations: {total_simulations}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Final model R² score: {self.history['model_scores'][-1]:.4f}")
        
        # Compile results
        results = {
            'approach': 'adaptive_sampling',
            'model_type': self.model_type,
            'k_columns': self.k_columns,
            'true_k_values': self.true_k_values.tolist(),
            'final_prediction': k_pred.tolist(),
            'initial_error': initial_error,
            'final_error': final_error,
            'error_improvement': (initial_error - final_error)/initial_error*100,
            'total_simulations': total_simulations,
            'total_time': total_time,
            'iterations_completed': self.history['iterations'],
            'converged': converged if 'converged' in locals() else False,
            'history': self.history
        }
        
        if save_results:
            self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        import json
        
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'adaptive')
        os.makedirs(results_dir, exist_ok=True)
        
        filename = f"adaptive_{self.model_type}_{int(time.time())}.json"
        filepath = os.path.join(results_dir, filename)
        
        # Helper function to convert numpy types to Python types
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert all numpy types to Python types for JSON serialization
        results_copy = convert_numpy_types(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def plot_convergence(self):
        """Plot convergence analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        iterations = range(len(self.history['predictions']))
        predictions_array = np.array(self.history['predictions'])
        
        # Plot 1: Prediction convergence
        ax = axes[0, 0]
        for i, k_col in enumerate(self.k_columns):
            ax.plot(iterations, predictions_array[:, i], 'o-', label=f'K{k_col+1}')
            ax.axhline(y=self.true_k_values[i], color=f'C{i}', linestyle='--', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('K values')
        ax.set_title('Convergence of K Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Relative error over iterations
        relative_errors = [
            np.linalg.norm(pred - self.true_k_values) / np.linalg.norm(self.true_k_values)
            for pred in self.history['predictions']
        ]
        ax = axes[0, 1]
        ax.semilogy(iterations, relative_errors, 'b-o')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Relative Error')
        ax.set_title('Convergence of Relative Error')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Model scores
        ax = axes[1, 0]
        ax.plot(range(len(self.history['model_scores'])), self.history['model_scores'], 'g-o')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Model R² Score')
        ax.set_title('Model Performance Over Iterations')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Hypercube sizes
        ax = axes[1, 1]
        ax.semilogy(range(len(self.history['hypercube_sizes'])), self.history['hypercube_sizes'], 'm-o')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Hypercube Size')
        ax.set_title('Hypercube Size Reduction')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'Adaptive Sampling Convergence - {self.model_type.title()}', y=1.02)
        
        # Save plot
        plots_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f'adaptive_{self.model_type}_convergence.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Convergence plot saved to: {plot_path}")
    
    def plot_final_prediction(self):
        """Plot final prediction comparison."""
        final_pred = self.history['predictions'][-1]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        k_indices = range(len(self.k_columns))
        
        ax.semilogy(k_indices, self.true_k_values, 'ro-', label='True K', markersize=8, linewidth=2)
        ax.semilogy(k_indices, final_pred, 'bs-', label='Predicted K', markersize=8, linewidth=2)
        
        # Add error bars
        errors = np.abs(final_pred - self.true_k_values)
        ax.errorbar(k_indices, final_pred, yerr=errors, fmt='bs', capsize=5, alpha=0.7)
        
        ax.set_xlabel('Rate Coefficient Index')
        ax.set_ylabel('Rate Coefficient Value')
        ax.set_title('Final Prediction vs True Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels to show actual K indices
        ax.set_xticks(k_indices)
        ax.set_xticklabels([f'K{i+1}' for i in self.k_columns])
        
        plt.tight_layout()
        
        # Save plot
        plots_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f'adaptive_{self.model_type}_final_prediction.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Final prediction plot saved to: {plot_path}")
