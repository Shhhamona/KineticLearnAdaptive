"""
Main comparison script: Traditional vs Adaptive Sampling approaches.

This script runs both approaches and compares their performance for 
rate coefficient determination from chemical compositions.
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from base_simulator import LoKISimulator, MockSimulator
from traditional_approach import TraditionalApproach
from adaptive_approach import AdaptiveSamplingApproach


class ComparisonStudy:
    """
    Main class for running comparison between traditional and adaptive approaches.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize comparison study.
        
        Args:
            config_path: Path to configuration JSON file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize simulator
        self.simulator = self._create_simulator()
        
        # Results storage
        self.results = {
            'traditional': {},
            'adaptive': {},
            'comparison': {}
        }
    
    def _create_simulator(self):
        """Create simulator based on configuration."""
        sim_config = self.config['simulation']
        
        use_mock = sim_config.get('use_mock_simulator', True)
        print(f"Configuration use_mock_simulator: {use_mock} (type: {type(use_mock)})")
        
        if use_mock:
            print("Using MockSimulator for testing")
            return MockSimulator(
                sim_config['setup_file'],
                sim_config['chem_file'], 
                sim_config['loki_path'],
                np.array(sim_config['true_k_values'])
            )
        else:
            print("Using LoKISimulator")
            return LoKISimulator(
                sim_config['setup_file'],
                sim_config['chem_file'],
                sim_config['loki_path'],
                sim_config['k_columns']
            )
    
    def run_traditional_approaches(self) -> Dict[str, Any]:
        """
        Run traditional approach with different models.
        
        Returns:
            Results for all traditional models
        """
        print("\n" + "="*80)
        print("RUNNING TRADITIONAL APPROACHES")
        print("="*80)
        
        trad_config = self.config['traditional_approach']
        sim_config = self.config['simulation']
        
        traditional_results = {}
        
        for model_config in trad_config['models']:
            model_type = model_config['type']
            model_params = model_config['params']
            
            print(f"\n--- Traditional Approach: {model_type.upper()} ---")
            
            # Create traditional approach
            traditional = TraditionalApproach(
                simulator=self.simulator,
                k_columns=sim_config['k_columns'],
                model_type=model_type,
                sampling_method=trad_config['sampling_method'],
                **model_params
            )
            
            # Run study
            results = traditional.run_complete_study(
                n_train=trad_config['n_train'],
                n_test=trad_config['n_test'],
                save_results=self.config['comparison']['save_results']
            )
            
            traditional_results[model_type] = results
            
            # Create plots if requested
            if self.config['comparison']['create_plots']:
                # Generate test data for plotting
                test_compositions, test_k = traditional.generate_test_data(trad_config['n_test'])
                traditional.plot_results(test_compositions, test_k)
        
        self.results['traditional'] = traditional_results
        return traditional_results
    
    def run_adaptive_approaches(self) -> Dict[str, Any]:
        """
        Run adaptive sampling approach with different models.
        
        Returns:
            Results for all adaptive models
        """
        print("\n" + "="*80)
        print("RUNNING ADAPTIVE APPROACHES")
        print("="*80)
        
        adapt_config = self.config['adaptive_approach']
        sim_config = self.config['simulation']
        
        adaptive_results = {}
        
        for model_config in adapt_config['models']:
            model_type = model_config['type']
            model_params = model_config['params']
            
            print(f"\n--- Adaptive Approach: {model_type.upper()} ---")
            
            # Create adaptive approach
            adaptive = AdaptiveSamplingApproach(
                simulator=self.simulator,
                k_columns=sim_config['k_columns'],
                true_k_values=np.array(sim_config['true_k_values']),
                model_type=model_type,
                sampling_method=adapt_config['sampling_method'],
                max_iterations=adapt_config['max_iterations'],
                convergence_threshold=adapt_config['convergence_threshold'],
                initial_hypercube_size=adapt_config['initial_hypercube_size'],
                hypercube_reduction=adapt_config['hypercube_reduction'],
                **model_params
            )
            
            # Run study
            results = adaptive.run_adaptive_sampling(
                n_initial=adapt_config['n_initial'],
                n_iteration=adapt_config['n_iteration'],
                save_results=self.config['comparison']['save_results']
            )
            
            adaptive_results[model_type] = results
            
            # Create plots if requested
            if self.config['comparison']['create_plots']:
                adaptive.plot_convergence()
                adaptive.plot_final_prediction()
        
        self.results['adaptive'] = adaptive_results
        return adaptive_results
    
    def compare_results(self) -> Dict[str, Any]:
        """
        Compare traditional and adaptive results.
        
        Returns:
            Comparison results
        """
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        
        comparison = {}
        
        # Get model types that were tested in both approaches
        trad_models = set(self.results['traditional'].keys())
        adapt_models = set(self.results['adaptive'].keys())
        common_models = trad_models.intersection(adapt_models)
        
        for model_type in common_models:
            print(f"\n--- {model_type.upper()} MODEL COMPARISON ---")
            
            trad_result = self.results['traditional'][model_type]
            adapt_result = self.results['adaptive'][model_type]
            
            # Extract key metrics
            trad_metrics = trad_result['test_metrics']
            adapt_metrics = {
                'r2_score': adapt_result['history']['model_scores'][-1],  # Final model score
                'final_error': adapt_result['final_error'],
                'total_simulations': adapt_result['total_simulations'],
                'training_time': adapt_result['total_time']
            }
            
            comparison[model_type] = {
                'traditional': {
                    'test_r2': trad_metrics['r2_score'],
                    'test_rmse': trad_metrics['rmse'],
                    'test_rel_error': trad_metrics['relative_error'],
                    'total_simulations': trad_result['total_simulations'],
                    'training_time': trad_result['train_metrics']['training_time']
                },
                'adaptive': {
                    'final_r2': adapt_metrics['r2_score'],
                    'final_rel_error': adapt_metrics['final_error'],
                    'total_simulations': adapt_metrics['total_simulations'],
                    'training_time': adapt_metrics['training_time'],
                    'error_improvement': adapt_result['error_improvement'],
                    'converged': adapt_result['converged']
                }
            }
            
            # Print comparison
            print(f"Traditional approach:")
            print(f"  Test R²: {trad_metrics['r2_score']:.4f}")
            print(f"  Test RMSE: {trad_metrics['rmse']:.2e}")
            print(f"  Test Rel Error: {trad_metrics['relative_error']:.4f}")
            print(f"  Total simulations: {trad_result['total_simulations']}")
            print(f"  Training time: {trad_result['train_metrics']['training_time']:.2f}s")
            
            print(f"\nAdaptive approach:")
            print(f"  Final Model R²: {adapt_metrics['r2_score']:.4f}")
            print(f"  Final Rel Error: {adapt_metrics['final_error']:.4f}")
            print(f"  Total simulations: {adapt_metrics['total_simulations']}")
            print(f"  Training time: {adapt_metrics['training_time']:.2f}s")
            print(f"  Error improvement: {adapt_result['error_improvement']:.2f}%")
            print(f"  Converged: {adapt_result['converged']}")
            
            # Calculate efficiency metrics
            sim_efficiency = adapt_metrics['total_simulations'] / trad_result['total_simulations']
            print(f"\nEfficiency:")
            print(f"  Simulation ratio (adaptive/traditional): {sim_efficiency:.2f}")
            
            if adapt_metrics['final_error'] < trad_metrics['relative_error']:
                print(f"  ✅ Adaptive approach achieved lower error with {sim_efficiency:.2f}x simulations")
            else:
                print(f"  ❌ Traditional approach achieved lower error")
        
        self.results['comparison'] = comparison
        return comparison
    
    def create_summary_plot(self):
        """Create summary comparison plot."""
        if not self.results['comparison']:
            print("No comparison results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models = list(self.results['comparison'].keys())
        
        # Prepare data
        trad_errors = [self.results['comparison'][m]['traditional']['test_rel_error'] for m in models]
        adapt_errors = [self.results['comparison'][m]['adaptive']['final_rel_error'] for m in models]
        trad_sims = [self.results['comparison'][m]['traditional']['total_simulations'] for m in models]
        adapt_sims = [self.results['comparison'][m]['adaptive']['total_simulations'] for m in models]
        trad_r2 = [self.results['comparison'][m]['traditional']['test_r2'] for m in models]
        adapt_r2 = [self.results['comparison'][m]['adaptive']['final_r2'] for m in models]
        
        # Plot 1: Relative error comparison
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, trad_errors, width, label='Traditional', alpha=0.8)
        axes[0, 0].bar(x + width/2, adapt_errors, width, label='Adaptive', alpha=0.8)
        axes[0, 0].set_xlabel('Model Type')
        axes[0, 0].set_ylabel('Relative Error')
        axes[0, 0].set_title('Relative Error Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([m.replace('_', ' ').title() for m in models])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Number of simulations
        axes[0, 1].bar(x - width/2, trad_sims, width, label='Traditional', alpha=0.8)
        axes[0, 1].bar(x + width/2, adapt_sims, width, label='Adaptive', alpha=0.8)
        axes[0, 1].set_xlabel('Model Type')
        axes[0, 1].set_ylabel('Total Simulations')
        axes[0, 1].set_title('Simulation Count Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([m.replace('_', ' ').title() for m in models])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: R² comparison
        axes[1, 0].bar(x - width/2, trad_r2, width, label='Traditional', alpha=0.8)
        axes[1, 0].bar(x + width/2, adapt_r2, width, label='Adaptive', alpha=0.8)
        axes[1, 0].set_xlabel('Model Type')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('Model Performance (R²) Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([m.replace('_', ' ').title() for m in models])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Efficiency (error vs simulations)
        for i, model in enumerate(models):
            axes[1, 1].scatter(trad_sims[i], trad_errors[i], s=100, 
                             label=f'Traditional {model.replace("_", " ").title()}', 
                             marker='o', alpha=0.8)
            axes[1, 1].scatter(adapt_sims[i], adapt_errors[i], s=100,
                             label=f'Adaptive {model.replace("_", " ").title()}', 
                             marker='s', alpha=0.8)
        
        axes[1, 1].set_xlabel('Total Simulations')
        axes[1, 1].set_ylabel('Relative Error')
        axes[1, 1].set_title('Efficiency: Error vs Simulations')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Traditional vs Adaptive Sampling Comparison', y=1.02)
        
        # Save plot
        plots_dir = os.path.join(current_dir, 'results', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, 'comparison_summary.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Summary plot saved to: {plot_path}")
    
    def save_summary_results(self):
        """Save complete results summary."""
        results_dir = os.path.join(current_dir, 'results', 'comparisons')
        os.makedirs(results_dir, exist_ok=True)
        
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
        
        summary_path = os.path.join(results_dir, f'comparison_summary_{int(time.time())}.json')
        
        # Convert numpy types for JSON serialization
        results_copy = convert_numpy_types(self.results)
        
        with open(summary_path, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"Complete results saved to: {summary_path}")
    
    def run_complete_study(self):
        """Run the complete comparison study."""
        print("Starting Adaptive vs Traditional Sampling Comparison Study")
        print("="*80)
        
        # Run both approaches
        self.run_traditional_approaches()
        self.run_adaptive_approaches()
        
        # Compare results
        self.compare_results()
        
        # Create summary visualizations
        if self.config['comparison']['create_plots']:
            self.create_summary_plot()
        
        # Save results
        if self.config['comparison']['save_results']:
            self.save_summary_results()
        
        print("\n" + "="*80)
        print("STUDY COMPLETED SUCCESSFULLY")
        print("="*80)


def main():
    """Main execution function."""
    import sys
    import time
    
    # Check for command line argument
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Default configuration file path
        config_path = os.path.join(current_dir, 'configs', 'experiment_config.json')
    
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return
    
    print(f"Using configuration: {config_path}")
    
    # Create and run study
    study = ComparisonStudy(config_path)
    study.run_complete_study()


if __name__ == '__main__':
    main()
