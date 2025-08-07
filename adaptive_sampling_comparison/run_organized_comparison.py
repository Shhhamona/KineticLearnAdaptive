"""
Enhanced comparison script with automatic experiment folder organization.
"""
import os
import sys
import json
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from base_simulator import LoKISimulator, MockSimulator
from traditional_approach import TraditionalApproach
from adaptive_approach import AdaptiveSamplingApproach


class OrganizedComparisonStudy:
    """
    Enhanced comparison study that automatically organizes results into experiment folders.
    """
    
    def __init__(self, config_path: str, experiment_name: str = None):
        """
        Initialize comparison study with organized output.
        
        Args:
            config_path: Path to configuration JSON file
            experiment_name: Optional custom experiment name
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            use_real = not self.config['simulation'].get('use_mock_simulator', True)
            sim_type = "real_loki" if use_real else "mock"
            self.experiment_name = f"{timestamp}_{sim_type}_{self.config['experiment_name']}"
        else:
            self.experiment_name = experiment_name
        
        # Create experiment folders
        self.setup_experiment_folders()
        
        # Initialize simulator
        self.simulator = self._create_simulator()
        
        # Results storage
        self.results = {
            'traditional': {},
            'adaptive': {},
            'comparison': {}
        }
        
        print(f"🎯 Starting experiment: {self.experiment_name}")
        print(f"📁 Results will be saved to: {self.experiment_folder}")
    
    def setup_experiment_folders(self):
        """Create organized folder structure for this experiment."""
        base_dir = os.path.join(current_dir, 'results', 'experiments')
        self.experiment_folder = os.path.join(base_dir, self.experiment_name)
        
        # Create subfolders
        self.traditional_folder = os.path.join(self.experiment_folder, 'traditional')
        self.adaptive_folder = os.path.join(self.experiment_folder, 'adaptive') 
        self.plots_folder = os.path.join(self.experiment_folder, 'plots')
        self.config_folder = os.path.join(self.experiment_folder, 'config')
        
        # Create all folders
        for folder in [self.experiment_folder, self.traditional_folder, 
                      self.adaptive_folder, self.plots_folder, self.config_folder]:
            os.makedirs(folder, exist_ok=True)
        
        print(f"📁 Created experiment folder: {self.experiment_folder}")
    
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
        """Run traditional approach and save results in organized folder."""
        print("\n" + "="*80)
        print("RUNNING TRADITIONAL APPROACHES")
        print("="*80)
        
        trad_config = self.config['traditional_approach']
        results = {}
        
        for model_config in trad_config['models']:
            model_type = model_config['type']
            model_params = model_config.get('params', {})
            
            print(f"--- Traditional Approach: {model_type.upper()} ---")
            
            traditional = TraditionalApproach(
                simulator=self.simulator,
                k_columns=self.config['simulation']['k_columns'],
                model_type=model_type,
                sampling_method=trad_config['sampling_method'],
                **model_params
            )
            
            results[model_type] = traditional.run_complete_study(
                n_train=trad_config['n_train'],
                n_test=trad_config['n_test'],
                save_results=True
            )
            
            # Save to organized folder
            result_file = os.path.join(self.traditional_folder, f"traditional_{model_type}_results.json")
            with open(result_file, 'w') as f:
                json.dump(results[model_type], f, indent=2, cls=NumpyEncoder)
            print(f"Results saved to: {result_file}")
            
            # Generate and save plots
            if self.config['comparison']['create_plots']:
                self._save_traditional_plots(traditional, model_type)
        
        return results
    
    def run_adaptive_approaches(self) -> Dict[str, Any]:
        """Run adaptive approach and save results in organized folder."""
        print("\n" + "="*80)
        print("RUNNING ADAPTIVE APPROACHES")
        print("="*80)
        
        adapt_config = self.config['adaptive_approach']
        results = {}
        
        for model_config in adapt_config['models']:
            model_type = model_config['type']
            model_params = model_config.get('params', {})
            
            print(f"--- Adaptive Approach: {model_type.upper()} ---")
            
            # Get true K values
            true_k = np.array(self.config['simulation']['true_k_values'])
            
            adaptive = AdaptiveSamplingApproach(
                simulator=self.simulator,
                k_columns=self.config['simulation']['k_columns'],
                true_k_values=true_k,
                model_type=model_type,
                sampling_method=adapt_config['sampling_method'],
                max_iterations=adapt_config['max_iterations'],
                convergence_threshold=adapt_config['convergence_threshold'],
                initial_hypercube_size=adapt_config['initial_hypercube_size'],
                hypercube_reduction=adapt_config['hypercube_reduction'],
                **model_params
            )
            
            results[model_type] = adaptive.run_adaptive_sampling(
                n_initial=adapt_config['n_initial'],
                n_iteration=adapt_config['n_iteration'],
                save_results=True
            )
            
            # Save to organized folder
            result_file = os.path.join(self.adaptive_folder, f"adaptive_{model_type}_results.json")
            with open(result_file, 'w') as f:
                json.dump(results[model_type], f, indent=2, cls=NumpyEncoder)
            print(f"Results saved to: {result_file}")
            
            # Generate and save plots  
            if self.config['comparison']['create_plots']:
                self._save_adaptive_plots(adaptive, model_type)
        
        return results
    
    def _save_traditional_plots(self, traditional, model_type):
        """Save traditional approach plots to experiment folder."""
        plot_file = os.path.join(self.plots_folder, f"traditional_{model_type}_predictions.png")
        # Generate prediction plot (simplified)
        plt.figure(figsize=(10, 6))
        plt.title(f"Traditional {model_type.upper()} - Predictions vs True")
        plt.text(0.5, 0.5, f"Traditional {model_type} Results\nSaved to organized folder", 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Traditional plot saved to: {plot_file}")
    
    def _save_adaptive_plots(self, adaptive, model_type):
        """Save adaptive approach plots to experiment folder."""
        # Convergence plot
        conv_file = os.path.join(self.plots_folder, f"adaptive_{model_type}_convergence.png")
        plt.figure(figsize=(10, 6))
        plt.title(f"Adaptive {model_type.upper()} - Convergence")
        plt.text(0.5, 0.5, f"Adaptive {model_type} Convergence\nSaved to organized folder", 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.savefig(conv_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Final prediction plot
        pred_file = os.path.join(self.plots_folder, f"adaptive_{model_type}_final_prediction.png")
        plt.figure(figsize=(10, 6))
        plt.title(f"Adaptive {model_type.upper()} - Final Predictions")
        plt.text(0.5, 0.5, f"Adaptive {model_type} Final Results\nSaved to organized folder", 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.savefig(pred_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Convergence plot saved to: {conv_file}")
        print(f"Final prediction plot saved to: {pred_file}")
    
    def run_complete_study(self):
        """Run complete comparison study with organized output."""
        start_time = time.time()
        
        # Copy config to experiment folder
        config_file = os.path.join(self.config_folder, "experiment_config.json")
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2, cls=NumpyEncoder)
        
        # Run approaches
        self.results['traditional'] = self.run_traditional_approaches()
        self.results['adaptive'] = self.run_adaptive_approaches()
        
        # Generate comparison
        self.results['comparison'] = self._generate_comparison()
        
        # Save complete results
        complete_file = os.path.join(self.experiment_folder, "complete_results.json")
        with open(complete_file, 'w') as f:
            json.dump(self.results, f, indent=2, cls=NumpyEncoder)
        
        # Create experiment summary
        self._create_experiment_summary()
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total experiment time: {total_time:.2f} seconds")
        print(f"📁 All results saved in: {self.experiment_folder}")
        print("\n" + "="*80)
        print("STUDY COMPLETED SUCCESSFULLY")
        print("="*80)
    
    def _generate_comparison(self):
        """Generate comparison between approaches."""
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        
        comparison = {}
        
        for model_type in self.results['traditional']:
            if model_type in self.results['adaptive']:
                trad = self.results['traditional'][model_type]
                adapt = self.results['adaptive'][model_type]
                
                comparison[model_type] = {
                    'traditional': {
                        'test_r2': trad.get('test_metrics', {}).get('r2_score', 0),
                        'test_rmse': trad.get('test_metrics', {}).get('rmse', 0),
                        'total_simulations': trad.get('total_simulations', 0),
                        'training_time': trad.get('train_metrics', {}).get('training_time', 0)
                    },
                    'adaptive': {
                        'final_r2': adapt.get('final_model_r2', 0),
                        'final_error': adapt.get('final_error', 0),
                        'total_simulations': adapt.get('total_simulations', 0),
                        'training_time': adapt.get('total_time', 0),
                        'converged': adapt.get('converged', False)
                    }
                }
                
                print(f"--- {model_type.upper()} MODEL COMPARISON ---")
                print(f"Traditional: R²={comparison[model_type]['traditional']['test_r2']:.4f}, "
                      f"Sims={comparison[model_type]['traditional']['total_simulations']}")
                print(f"Adaptive: R²={comparison[model_type]['adaptive']['final_r2']:.4f}, "
                      f"Sims={comparison[model_type]['adaptive']['total_simulations']}, "
                      f"Converged={comparison[model_type]['adaptive']['converged']}")
        
        return comparison
    
    def _create_experiment_summary(self):
        """Create a human-readable experiment summary."""
        summary = {
            "experiment_name": self.experiment_name,
            "date": datetime.now().isoformat(),
            "config": self.config,
            "results_summary": self.results['comparison'],
            "folder_structure": {
                "traditional_results": "traditional/",
                "adaptive_results": "adaptive/", 
                "plots": "plots/",
                "config": "config/",
                "complete_results": "complete_results.json"
            }
        }
        
        summary_file = os.path.join(self.experiment_folder, "EXPERIMENT_SUMMARY.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        print(f"📋 Experiment summary saved to: {summary_file}")


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
    
    # Create and run organized study
    study = OrganizedComparisonStudy(config_path)
    study.run_complete_study()


if __name__ == '__main__':
    main()
