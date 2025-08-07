#!/usr/bin/env python3
"""
Adaptive Approach Analysis Script

This script runs the adaptive sampling approach for rate coefficient determination,
providing comprehensive analysis, visualization, and organized result saving.
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append('src')

from base_simulator import LoKISimulator, MockSimulator
from adaptive_approach import AdaptiveSamplingApproach


class AdaptiveAnalysis:
    """Comprehensive analysis runner for adaptive sampling approach."""
    
    def __init__(self, config):
        """
        Initialize analysis with configuration.
        
        Args:
            config: Dictionary with analysis configuration
        """
        self.config = config
        
        # Results storage and timestamp
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup directories
        self.setup_directories()
        
        # Initialize simulator
        if config['simulator_type'].lower() == 'loki':
            self.simulator = LoKISimulator(
                config['setup_file'], 
                config['chem_file'], 
                config['loki_path'],
                config['k_columns']
            )
        else:
            self.simulator = MockSimulator(
                config['setup_file'], 
                config['chem_file'], 
                config['loki_path']
            )
            
    def setup_directories(self):
        """Create organized directory structure for results."""
        # Extract chemistry name from file
        chem_name = Path(self.config['chem_file']).stem
        simulator_name = self.config['simulator_type'].lower()
        
        # Create results directory structure
        self.results_dir = Path('results') / 'adaptive' / simulator_name / chem_name / self.timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / 'plots').mkdir(exist_ok=True)
        (self.results_dir / 'data').mkdir(exist_ok=True)
        (self.results_dir / 'models').mkdir(exist_ok=True)
        
        print(f"📁 Results will be saved to: {self.results_dir}")
    
    def run_analysis(self):
        """Run complete adaptive approach analysis."""
        print("=" * 80)
        print("ADAPTIVE SAMPLING ANALYSIS")
        print("=" * 80)
        print(f"Simulator: {self.config['simulator_type']}")
        print(f"Chemistry: {self.config['chem_file']}")
        print(f"True K values: {self.config['true_k_values']}")
        print(f"Initial samples: {self.config['n_initial']}")
        print(f"Iteration samples: {self.config['n_iteration']}")
        print(f"Max iterations: {self.config['max_iterations']}")
        print(f"Models: {', '.join(self.config['models'])}")
        print("=" * 80)
        
        # Run analysis for each model
        for model_type in self.config['models']:
            print(f"\n🔬 Analyzing adaptive sampling with {model_type}...")
            
            # Initialize adaptive approach
            adaptive = AdaptiveSamplingApproach(
                simulator=self.simulator,
                k_columns=self.config['k_columns'],
                true_k_values=np.array(self.config['true_k_values']),
                model_type=model_type,
                sampling_method=self.config['sampling_method'],
                max_iterations=self.config['max_iterations'],
                convergence_threshold=self.config['convergence_threshold'],
                initial_hypercube_size=self.config['initial_hypercube_size'],
                hypercube_reduction=self.config['hypercube_reduction'],
                random_state=self.config['random_state']
            )
            
            # Run the adaptive sampling
            model_results = adaptive.run_adaptive_sampling(
                n_initial=self.config['n_initial'],
                n_iteration=self.config['n_iteration'],
                save_results=False  # We'll handle saving ourselves
            )
            
            # Calculate standardized metrics for comparison
            standardized_metrics = self.calculate_standardized_metrics(model_type, adaptive)
            model_results['standardized_metrics'] = standardized_metrics
            
            # Store results
            self.results[model_type] = model_results
            
            # Print summary
            self.print_model_summary(model_type, model_results)
            
            # Generate model-specific plots
            self.generate_model_plots(model_type, adaptive)
            
        # Generate comparative analysis
        self.generate_comparison_plots()
        self.save_results()
        self.generate_report()
        
        print(f"\n✅ Analysis complete! Results saved to {self.results_dir}")
    
    def calculate_standardized_metrics(self, model_type, adaptive_instance):
        """Calculate standardized metrics for comparison with traditional approach."""
        from sklearn.metrics import r2_score, mean_squared_error
        
        final_prediction = adaptive_instance.history['predictions'][-1]
        true_values = adaptive_instance.true_k_values
        
        # Calculate RMSE (key comparison metric)
        rmse = np.sqrt(mean_squared_error([true_values], [final_prediction]))
        
        # Calculate R² score (treating as single sample prediction)
        # For single sample, we'll use 1 - (residual sum of squares / total sum of squares)
        ss_res = np.sum((true_values - final_prediction) ** 2)
        ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
        r2_score_value = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Calculate relative error (existing)
        relative_error = np.linalg.norm(final_prediction - true_values) / np.linalg.norm(true_values)
        
        # Calculate absolute error
        absolute_error = np.linalg.norm(final_prediction - true_values)
        
        return {
            'rmse': rmse,
            'r2_score': r2_score_value,
            'relative_error': relative_error,
            'absolute_error': absolute_error,
            'final_prediction': final_prediction,
            'true_values': true_values
        }
    
    def print_model_summary(self, model_type, results):
        """Print summary of model performance."""
        print(f"\n📊 {model_type.upper()} ADAPTIVE RESULTS:")
        print(f"   Initial relative error: {results['initial_error']:.4f}")
        print(f"   Final relative error: {results['final_error']:.4f}")
        print(f"   Final RMSE: {results['standardized_metrics']['rmse']:.2e}")
        print(f"   Final R² score: {results['standardized_metrics']['r2_score']:.4f}")
        print(f"   Error improvement: {results['error_improvement']:.2f}%")
        print(f"   Total simulations: {results['total_simulations']}")
        print(f"   Iterations completed: {results['iterations_completed']}")
        print(f"   Converged: {results['converged']}")
        print(f"   Total time: {results['total_time']:.2f}s")
    
    def generate_model_plots(self, model_type, adaptive_instance):
        """Generate plots for a specific model."""
        print(f"   📈 Generating plots for {model_type}...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Convergence plot
        self.plot_convergence(model_type, adaptive_instance)
        
        # 2. Final prediction plot
        self.plot_final_prediction(model_type, adaptive_instance)
        
        # 3. Error evolution plot
        self.plot_error_evolution(model_type, adaptive_instance)
        
        # 4. Sampling evolution plot
        self.plot_sampling_evolution(model_type, adaptive_instance)
    
    def plot_convergence(self, model_type, adaptive_instance):
        """Plot convergence analysis for a model."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        history = adaptive_instance.history
        iterations = range(len(history['predictions']))
        predictions_array = np.array(history['predictions'])
        
        # Plot 1: Prediction convergence
        ax = axes[0, 0]
        for i, k_col in enumerate(adaptive_instance.k_columns):
            ax.plot(iterations, predictions_array[:, i], 'o-', label=f'K{k_col+1}', linewidth=2, markersize=6)
            ax.axhline(y=adaptive_instance.true_k_values[i], color=f'C{i}', linestyle='--', alpha=0.7, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('K values (log scale)')
        ax.set_yscale('log')
        ax.set_title('Convergence of K Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Relative error over iterations
        relative_errors = [
            np.linalg.norm(pred - adaptive_instance.true_k_values) / np.linalg.norm(adaptive_instance.true_k_values)
            for pred in history['predictions']
        ]
        ax = axes[0, 1]
        ax.semilogy(iterations, relative_errors, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Relative Error (log scale)')
        ax.set_title('Convergence of Relative Error')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Model scores
        ax = axes[1, 0]
        model_iterations = range(len(history['model_scores']))
        ax.plot(model_iterations, history['model_scores'], 'g-o', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Model R² Score')
        ax.set_title('Model Performance Over Iterations')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Plot 4: Hypercube sizes
        ax = axes[1, 1]
        if history['hypercube_sizes']:
            ax.semilogy(range(len(history['hypercube_sizes'])), history['hypercube_sizes'], 'm-o', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Hypercube Size (log scale)')
        ax.set_title('Hypercube Size Reduction')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'Adaptive Sampling Convergence - {model_type.title()}', y=1.02, fontsize=16)
        
        # Save plot
        plot_path = self.results_dir / 'plots' / f'{model_type}_convergence.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_final_prediction(self, model_type, adaptive_instance):
        """Plot final prediction comparison."""
        final_pred = adaptive_instance.history['predictions'][-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        k_indices = range(len(adaptive_instance.k_columns))
        
        ax.semilogy(k_indices, adaptive_instance.true_k_values, 'ro-', label='True K', 
                   markersize=10, linewidth=3, alpha=0.8)
        ax.semilogy(k_indices, final_pred, 'bs-', label='Predicted K', 
                   markersize=10, linewidth=3, alpha=0.8)
        
        # Add error bars
        errors = np.abs(final_pred - adaptive_instance.true_k_values)
        ax.errorbar(k_indices, final_pred, yerr=errors, fmt='bs', capsize=8, alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Rate Coefficient Index', fontsize=12)
        ax.set_ylabel('Rate Coefficient Value (log scale)', fontsize=12)
        ax.set_title(f'Final Prediction vs True Values - {model_type.title()}', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels to show actual K indices
        ax.set_xticks(k_indices)
        ax.set_xticklabels([f'K{i+1}' for i in adaptive_instance.k_columns])
        
        # Add relative error text
        relative_error = np.linalg.norm(final_pred - adaptive_instance.true_k_values) / np.linalg.norm(adaptive_instance.true_k_values)
        ax.text(0.05, 0.95, f'Relative Error: {relative_error:.4f}', 
               transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / 'plots' / f'{model_type}_final_prediction.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_evolution(self, model_type, adaptive_instance):
        """Plot how error evolves with simulation count."""
        history = adaptive_instance.history
        
        # Calculate cumulative simulation counts
        cumulative_sims = np.cumsum([0] + history['simulation_counts'])
        
        # Calculate relative errors
        relative_errors = [
            np.linalg.norm(pred - adaptive_instance.true_k_values) / np.linalg.norm(adaptive_instance.true_k_values)
            for pred in history['predictions']
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.semilogy(cumulative_sims[1:], relative_errors, 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('Total Simulations', fontsize=12)
        ax.set_ylabel('Relative Error (log scale)', fontsize=12)
        ax.set_title(f'Error vs Simulation Count - {model_type.title()}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add annotations for key points
        if len(relative_errors) > 1:
            improvement = (relative_errors[0] - relative_errors[-1]) / relative_errors[0] * 100
            ax.text(0.05, 0.95, f'Error Improvement: {improvement:.1f}%', 
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / 'plots' / f'{model_type}_error_evolution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_sampling_evolution(self, model_type, adaptive_instance):
        """Plot how sampling evolves over iterations."""
        history = adaptive_instance.history
        
        if not history['all_k_values']:
            return
        
        fig, axes = plt.subplots(1, len(adaptive_instance.k_columns), figsize=(5*len(adaptive_instance.k_columns), 6))
        if len(adaptive_instance.k_columns) == 1:
            axes = [axes]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(history['all_k_values'])))
        
        for k_idx, k_col in enumerate(adaptive_instance.k_columns):
            ax = axes[k_idx]
            
            for iter_idx, k_values in enumerate(history['all_k_values']):
                ax.scatter(k_values[:, k_idx], [iter_idx] * len(k_values), 
                          c=[colors[iter_idx]], alpha=0.6, s=30, label=f'Iter {iter_idx}')
            
            # Add true value line
            ax.axvline(x=adaptive_instance.true_k_values[k_idx], color='red', 
                      linestyle='--', linewidth=2, alpha=0.8, label='True K')
            
            # Add predictions
            predictions = [pred[k_idx] for pred in history['predictions']]
            ax.scatter(predictions, range(len(predictions)), 
                      c='red', marker='x', s=100, linewidth=3, label='Predictions')
            
            ax.set_xlabel(f'K{k_col+1} Value (log scale)', fontsize=12)
            ax.set_ylabel('Iteration', fontsize=12)
            ax.set_title(f'Sampling Evolution - K{k_col+1}', fontsize=14)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.suptitle(f'Sampling Evolution - {model_type.title()}', y=1.02, fontsize=16)
        
        # Save plot
        plot_path = self.results_dir / 'plots' / f'{model_type}_sampling_evolution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comparison_plots(self):
        """Generate comparison plots between different models."""
        if len(self.results) < 2:
            return
            
        print("\n📈 Generating comparison plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Model comparison plot
        self.plot_model_comparison()
        
        # Efficiency comparison
        self.plot_efficiency_comparison()
    
    def plot_model_comparison(self):
        """Plot comparison of different models."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(self.results.keys())
        
        # Plot 1: Final relative errors
        final_errors = [self.results[m]['final_error'] for m in models]
        initial_errors = [self.results[m]['initial_error'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, initial_errors, width, label='Initial Error', alpha=0.8)
        ax1.bar(x + width/2, final_errors, width, label='Final Error', alpha=0.8)
        ax1.set_xlabel('Model Type')
        ax1.set_ylabel('Relative Error')
        ax1.set_title('Initial vs Final Relative Errors')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Error improvement
        improvements = [self.results[m]['error_improvement'] for m in models]
        bars = ax2.bar(models, improvements, alpha=0.8, color='green')
        ax2.set_xlabel('Model Type')
        ax2.set_ylabel('Error Improvement (%)')
        ax2.set_title('Error Improvement by Model')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:.1f}%', ha='center', va='bottom')
        
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # Plot 3: Total simulations
        total_sims = [self.results[m]['total_simulations'] for m in models]
        ax3.bar(models, total_sims, alpha=0.8, color='orange')
        ax3.set_xlabel('Model Type')
        ax3.set_ylabel('Total Simulations')
        ax3.set_title('Simulation Efficiency')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # Plot 4: Iterations to convergence
        iterations = [self.results[m]['iterations_completed'] for m in models]
        colors = ['green' if self.results[m]['converged'] else 'red' for m in models]
        bars = ax4.bar(models, iterations, alpha=0.8, color=colors)
        ax4.set_xlabel('Model Type')
        ax4.set_ylabel('Iterations Completed')
        ax4.set_title('Convergence Performance')
        ax4.grid(True, alpha=0.3)
        
        # Add convergence status labels
        for bar, converged in zip(bars, [self.results[m]['converged'] for m in models]):
            height = bar.get_height()
            status = 'Y' if converged else 'N'
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    status, ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        plt.setp(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.suptitle('Adaptive Sampling Model Comparison', y=1.02, fontsize=16)
        
        # Save plot
        plot_path = self.results_dir / 'plots' / 'model_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_efficiency_comparison(self):
        """Plot efficiency comparison (performance vs cost)."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        models = list(self.results.keys())
        
        for model in models:
            result = self.results[model]
            x = result['total_simulations']
            y = 1 - result['final_error']  # Convert error to "accuracy"
            
            # Plot point
            ax.scatter(x, y, s=200, alpha=0.7, label=model)
            
            # Add model name annotation
            ax.annotate(model, (x, y), xytext=(10, 10), textcoords='offset points',
                       fontsize=12, alpha=0.8)
        
        ax.set_xlabel('Total Simulations', fontsize=12)
        ax.set_ylabel('Accuracy (1 - Relative Error)', fontsize=12)
        ax.set_title('Efficiency: Accuracy vs Simulation Cost', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / 'plots' / 'efficiency_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _convert_to_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def save_results(self):
        """Save comprehensive results to files."""
        print("\n💾 Saving results...")
        
        # Save raw results
        results_file = self.results_dir / 'data' / 'adaptive_results.json'
        
        # Prepare results for JSON serialization
        json_results = {}
        for model_type, results in self.results.items():
            json_results[model_type] = self._convert_to_serializable(results)
        
        # Add metadata
        json_results['metadata'] = self._convert_to_serializable({
            'timestamp': self.timestamp,
            'simulator_type': self.config['simulator_type'],
            'chem_file': self.config['chem_file'],
            'setup_file': self.config['setup_file'],
            'analysis_config': self.config
        })
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"   Results saved: {results_file}")
        
        # Save summary CSV
        self.save_summary_csv()
    
    def save_summary_csv(self):
        """Save summary results as CSV for easy analysis."""
        import pandas as pd
        
        summary_data = []
        for model_type, results in self.results.items():
            std_metrics = results['standardized_metrics']
            summary_data.append({
                'approach': 'adaptive',
                'model_type': model_type,
                'initial_error': results['initial_error'],
                'final_error': results['final_error'],
                'final_rmse': std_metrics['rmse'],
                'final_r2_score': std_metrics['r2_score'],
                'error_improvement': results['error_improvement'],
                'total_simulations': results['total_simulations'],
                'iterations_completed': results['iterations_completed'],
                'converged': results['converged'],
                'total_time': results['total_time'],
                'efficiency_rmse': 1 / (std_metrics['rmse'] * results['total_simulations']),  # Lower RMSE per simulation is better
                'efficiency_r2': std_metrics['r2_score'] / results['total_simulations']  # Higher R² per simulation is better
            })
        
        df = pd.DataFrame(summary_data)
        summary_file = self.results_dir / 'data' / 'summary.csv'
        df.to_csv(summary_file, index=False)
        
        print(f"   Summary saved: {summary_file}")
    
    def generate_report(self):
        """Generate a comprehensive analysis report."""
        print("\n📝 Generating analysis report...")
        
        report_file = self.results_dir / 'analysis_report.md'
        
        with open(report_file, 'w') as f:
            f.write(f"# Adaptive Sampling Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Configuration\n\n")
            f.write(f"- **Simulator:** {self.config['simulator_type']}\n")
            f.write(f"- **Chemistry File:** {self.config['chem_file']}\n")
            f.write(f"- **Setup File:** {self.config['setup_file']}\n")
            f.write(f"- **True K Values:** {self.config['true_k_values']}\n")
            f.write(f"- **Initial Samples:** {self.config['n_initial']}\n")
            f.write(f"- **Iteration Samples:** {self.config['n_iteration']}\n")
            f.write(f"- **Max Iterations:** {self.config['max_iterations']}\n")
            f.write(f"- **Convergence Threshold:** {self.config['convergence_threshold']}\n")
            f.write(f"- **Sampling Method:** {self.config['sampling_method']}\n\n")
            
            f.write(f"## Results Summary\n\n")
            f.write(f"| Model | Final RMSE | Final R² | Initial Error | Final Error | Improvement (%) | Total Sims | Iterations | Converged | Time (s) | Efficiency (R²/Sim) |\n")
            f.write(f"|-------|------------|----------|---------------|-------------|-----------------|------------|------------|-----------|----------|--------------------|\n")
            
            for model_type, results in self.results.items():
                std_metrics = results['standardized_metrics']
                f.write(f"| {model_type} | {std_metrics['rmse']:.2e} | {std_metrics['r2_score']:.4f} | "
                       f"{results['initial_error']:.4f} | {results['final_error']:.4f} | "
                       f"{results['error_improvement']:.2f} | {results['total_simulations']} | "
                       f"{results['iterations_completed']} | {'Yes' if results['converged'] else 'No'} | "
                       f"{results['total_time']:.2f} | {std_metrics['r2_score']/results['total_simulations']:.2e} |\n")
            
            f.write(f"\n## Best Performing Model\n\n")
            best_model = min(self.results.keys(), key=lambda k: self.results[k]['standardized_metrics']['rmse'])
            best_results = self.results[best_model]
            best_std_metrics = best_results['standardized_metrics']
            f.write(f"**{best_model}** achieved the lowest RMSE of {best_std_metrics['rmse']:.2e}\n")
            f.write(f"- Final R² score: {best_std_metrics['r2_score']:.4f}\n")
            f.write(f"- Final relative error: {best_results['final_error']:.4f}\n")
            f.write(f"- Error improvement: {best_results['error_improvement']:.2f}%\n")
            f.write(f"- Total simulations: {best_results['total_simulations']}\n")
            f.write(f"- Converged: {'Yes' if best_results['converged'] else 'No'}\n\n")
            
            f.write(f"## Files Generated\n\n")
            f.write(f"- `data/adaptive_results.json` - Complete results data\n")
            f.write(f"- `data/summary.csv` - Summary metrics\n")
            f.write(f"- `plots/model_comparison.png` - Model performance comparison\n")
            f.write(f"- `plots/efficiency_comparison.png` - Efficiency analysis\n")
            f.write(f"- `plots/*_convergence.png` - Individual model convergence plots\n")
            f.write(f"- `plots/*_final_prediction.png` - Final prediction comparisons\n")
            f.write(f"- `plots/*_error_evolution.png` - Error evolution plots\n")
            f.write(f"- `plots/*_sampling_evolution.png` - Sampling evolution plots\n")
        
        print(f"   Report saved: {report_file}")


def main():
    """Main analysis runner."""
    
    # Configuration for analysis
    config = {
        # Simulator configuration
        'simulator_type': 'mock',  # 'mock' or 'loki'
        'setup_file': 'setup.in',
        'chem_file': 'chem.chem',
        'loki_path': 'C:\\MyPrograms\\LoKI_v3.1.0-v2',
        
        # Analysis parameters
        'k_columns': [0, 1, 2],  # Which K columns to vary
        'true_k_values': [1e-15, 2e-15, 8e-16],  # True/literature K values for these columns
        
        # Adaptive sampling configuration
        'n_initial': 20,  # Initial samples around true K
        'n_iteration': 10,  # Samples per adaptive iteration
        'max_iterations': 8,
        'convergence_threshold': 1e-3,
        'initial_hypercube_size': 0.5,
        'hypercube_reduction': 0.8,
        'sampling_method': 'latin_hypercube',
        'random_state': 42,
        
        # Models to test
        'models': ['random_forest', 'neural_network']
    }
    
    # For LoKI analysis, update configuration
    loki_config = {
        'simulator_type': 'loki',
        'setup_file': 'setup_O2_simple.in',
        'chem_file': 'O2_simple_1.chem',
        'loki_path': 'C:\\MyPrograms\\LoKI_v3.1.0-v2',
        'k_columns': [0, 1, 2],
        'true_k_values': [6e-16, 1.3e-15, 9.6e-16],  # Reference values for O2 simple
        'n_initial': 10,  # Fewer for LoKI due to computational cost
        'n_iteration': 5,
        'max_iterations': 5,
        'convergence_threshold': 1e-3,
        'initial_hypercube_size': 0.3,
        'hypercube_reduction': 0.8,
        'sampling_method': 'latin_hypercube',
        'random_state': 42,
        'models': ['random_forest', 'neural_network']
    }
    
    print("Adaptive Sampling Analysis")
    print("Choose simulator:")
    print("1. MockSimulator (fast, for testing)")
    print("2. LoKISimulator (slow, real physics)")
    
    # Default to MockSimulator for testing, but allow override
    choice = input("Enter choice (1 or 2) [default=1]: ").strip() or "1"
    
    if choice == '2':
        print("Using LoKI configuration...")
        analysis_config = loki_config
    else:
        print("Using Mock configuration...")
        analysis_config = config
    
    # Run analysis
    analyzer = AdaptiveAnalysis(analysis_config)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
