#!/usr/bimport os
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
from traditional_approach import TraditionalApproach
from sampling_strategies import BoundsBasedSampler
from ml_models import RateCoefficientPredictor
"""
Traditional Approach Analysis Script

This script runs the traditional ML approach for rate coefficient determination,
providing comprehensive analysis, visualization, and organized result saving.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append('src')

from base_simulator import LoKISimulator, MockSimulator
from traditional_approach import TraditionalApproach
from sampling_strategies import BoundsBasedSampler


class TraditionalAnalysis:
    """Comprehensive analysis runner for traditional ML approach."""
    
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
        self.results_dir = Path('results') / 'traditional' / simulator_name / chem_name / self.timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / 'plots').mkdir(exist_ok=True)
        (self.results_dir / 'data').mkdir(exist_ok=True)
        (self.results_dir / 'models').mkdir(exist_ok=True)
        
        print(f"📁 Results will be saved to: {self.results_dir}")
    
    def run_analysis(self):
        """Run complete traditional approach analysis."""
        print("=" * 80)
        print("TRADITIONAL APPROACH ANALYSIS")
        print("=" * 80)
        print(f"Simulator: {self.config['simulator_type']}")
        print(f"Chemistry: {self.config['chem_file']}")
        print(f"Training samples: {self.config['n_train']}")
        print(f"Test samples: {self.config['n_test']}")
        print(f"Models: {', '.join(self.config['models'])}")
        print("=" * 80)
        
        # Step 1: Generate simulation data ONCE for all models
        print(f"\n🧪 Generating simulation data...")
        train_data, test_data = self._generate_simulation_data()
        
        # Step 2: Train each model on the same data
        for model_type in self.config['models']:
            print(f"\n🔬 Training {model_type} model...")
            
            # Train model on shared simulation data
            model_results = self._train_model_on_data(model_type, train_data, test_data)
            
            # Store results
            self.results[model_type] = model_results
            
            # Print summary
            self.print_model_summary(model_type, model_results)
        
        # Generate comprehensive analysis
        self.generate_plots()
        self.save_results()
        self.generate_report()
        
        print(f"\n✅ Analysis complete! Results saved to {self.results_dir}")
    
    def _generate_simulation_data(self):
        """Generate training and test simulation data once for all models."""
        
        # Initialize sampler
        if 'k_bounds' in self.config:
            sampler = BoundsBasedSampler(
                self.config['k_bounds'], 
                self.config['sampling_method']
            )
            k_bounds = self.config['k_bounds']
        else:
            # Use default bounds from TraditionalApproach
            traditional_temp = TraditionalApproach(
                simulator=self.simulator,
                k_columns=self.config['k_columns'],
                model_type='random_forest',  # Temporary, just to get bounds
                sampling_method=self.config['sampling_method']
            )
            sampler = traditional_temp.sampler
            k_bounds = traditional_temp.k_bounds
        
        print(f"   📊 K bounds: {k_bounds}")
        
        # Generate training data
        print(f"   🎯 Generating {self.config['n_train']} training samples...")
        train_k_samples = sampler.sample(
            center=np.mean(k_bounds, axis=1),
            bounds=k_bounds,
            n_samples=self.config['n_train']
        )
        
        start_time = time.time()
        train_compositions = self.simulator.run_simulations(train_k_samples)
        train_time = time.time() - start_time
        print(f"   ⏱️  Training simulations completed in {train_time:.2f}s")
        
        # Generate test data
        print(f"   🎯 Generating {self.config['n_test']} test samples...")
        test_k_samples = sampler.sample(
            center=np.mean(k_bounds, axis=1),
            bounds=k_bounds,
            n_samples=self.config['n_test']
        )
        
        start_time = time.time()
        test_compositions = self.simulator.run_simulations(test_k_samples)
        test_time = time.time() - start_time
        print(f"   ⏱️  Test simulations completed in {test_time:.2f}s")
        
        total_sims = self.config['n_train'] + self.config['n_test']
        total_time = train_time + test_time
        print(f"   ✅ Total: {total_sims} simulations in {total_time:.2f}s")
        
        return {
            'train_k': train_k_samples,
            'train_compositions': train_compositions,
            'train_time': train_time,
            'k_bounds': k_bounds
        }, {
            'test_k': test_k_samples,
            'test_compositions': test_compositions,
            'test_time': test_time,
            'k_bounds': k_bounds
        }
    
    def _train_model_on_data(self, model_type, train_data, test_data):
        """Train a specific model on the shared simulation data."""
        from sklearn.metrics import r2_score, mean_squared_error
        
        # Initialize model
        model = RateCoefficientPredictor(model_type)
        
        # Train model
        start_time = time.time()
        train_metrics = model.fit(train_data['train_compositions'], train_data['train_k'])
        train_time = time.time() - start_time
        
        # Evaluate on test data
        test_predictions = model.predict(test_data['test_compositions'])
        
        # Calculate test metrics
        test_r2 = r2_score(test_data['test_k'], test_predictions, multioutput='uniform_average')
        test_rmse = np.sqrt(mean_squared_error(test_data['test_k'], test_predictions, multioutput='uniform_average'))
        
        # Calculate relative error
        relative_errors = np.abs((test_predictions - test_data['test_k']) / test_data['test_k'])
        test_relative_error = np.mean(relative_errors)
        
        test_metrics = {
            'r2_score': test_r2,
            'rmse': test_rmse,
            'relative_error': test_relative_error
        }
        
        # Save trained model
        self.save_model(model_type, model)
        
        # Return results in expected format
        return {
            'approach': 'traditional',
            'model_type': model_type,
            'sampling_method': self.config['sampling_method'],
            'n_train': self.config['n_train'],
            'n_test': self.config['n_test'],
            'k_columns': self.config['k_columns'],
            'k_bounds': train_data['k_bounds'],
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'total_simulations': self.config['n_train'] + self.config['n_test'],
            'simulation_time': train_data['train_time'] + test_data['test_time'],
            'training_time': train_time
        }
    
    def print_model_summary(self, model_type, results):
        """Print summary of model performance."""
        train_metrics = results['train_metrics']
        test_metrics = results['test_metrics']
        
        print(f"\n📊 {model_type.upper()} RESULTS:")
        print(f"   Training R²: {train_metrics['r2_score']:.4f}")
        print(f"   Training RMSE: {train_metrics['rmse']:.2e}")
        print(f"   Test R²: {test_metrics['r2_score']:.4f}")
        print(f"   Test RMSE: {test_metrics['rmse']:.2e}")
        print(f"   Total simulations: {results['total_simulations']}")
    
    def save_model(self, model_type, model):
        """Save trained model."""
        import joblib
        model_path = self.results_dir / 'models' / f"{model_type}_model.joblib"
        joblib.dump(model, model_path)
        print(f"💾 Model saved: {model_path}")
    
    def generate_plots(self):
        """Generate comprehensive plots for analysis."""
        print("\n📈 Generating plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Model comparison plot
        self.plot_model_comparison()
        
        # 2. Performance metrics plot
        self.plot_performance_metrics()
        
        # 3. Training vs test comparison
        self.plot_training_test_comparison()
        
        # 4. Simulation efficiency plot
        self.plot_simulation_efficiency()
        
        print(f"   Plots saved to {self.results_dir / 'plots'}")
    
    def plot_model_comparison(self):
        """Plot comparison of different models."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        models = list(self.results.keys())
        train_r2 = [self.results[m]['train_metrics']['r2_score'] for m in models]
        test_r2 = [self.results[m]['test_metrics']['r2_score'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        # R² comparison
        ax1.bar(x - width/2, train_r2, width, label='Training R²', alpha=0.8)
        ax1.bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8)
        ax1.set_xlabel('Model Type')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Performance Comparison (R²)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RMSE comparison
        train_rmse = [self.results[m]['train_metrics']['rmse'] for m in models]
        test_rmse = [self.results[m]['test_metrics']['rmse'] for m in models]
        
        ax2.bar(x - width/2, train_rmse, width, label='Training RMSE', alpha=0.8)
        ax2.bar(x + width/2, test_rmse, width, label='Test RMSE', alpha=0.8)
        ax2.set_xlabel('Model Type')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Model Performance Comparison (RMSE)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'plots' / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_metrics(self):
        """Plot detailed performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(self.results.keys())
        
        # Training R²
        train_r2 = [self.results[m]['train_metrics']['r2_score'] for m in models]
        axes[0,0].barh(models, train_r2, alpha=0.7)
        axes[0,0].set_title('Training R² Scores')
        axes[0,0].set_xlabel('R² Score')
        axes[0,0].grid(True, alpha=0.3)
        
        # Test R²
        test_r2 = [self.results[m]['test_metrics']['r2_score'] for m in models]
        axes[0,1].barh(models, test_r2, alpha=0.7, color='orange')
        axes[0,1].set_title('Test R² Scores')
        axes[0,1].set_xlabel('R² Score')
        axes[0,1].grid(True, alpha=0.3)
        
        # Training RMSE
        train_rmse = [self.results[m]['train_metrics']['rmse'] for m in models]
        axes[1,0].barh(models, train_rmse, alpha=0.7, color='green')
        axes[1,0].set_title('Training RMSE')
        axes[1,0].set_xlabel('RMSE')
        axes[1,0].set_xscale('log')
        axes[1,0].grid(True, alpha=0.3)
        
        # Test RMSE
        test_rmse = [self.results[m]['test_metrics']['rmse'] for m in models]
        axes[1,1].barh(models, test_rmse, alpha=0.7, color='red')
        axes[1,1].set_title('Test RMSE')
        axes[1,1].set_xlabel('RMSE')
        axes[1,1].set_xscale('log')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'plots' / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_test_comparison(self):
        """Plot training vs test performance scatter."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        models = list(self.results.keys())
        train_r2 = [self.results[m]['train_metrics']['r2_score'] for m in models]
        test_r2 = [self.results[m]['test_metrics']['r2_score'] for m in models]
        train_rmse = [self.results[m]['train_metrics']['rmse'] for m in models]
        test_rmse = [self.results[m]['test_metrics']['rmse'] for m in models]
        
        # R² scatter
        ax1.scatter(train_r2, test_r2, s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax1.annotate(model, (train_r2[i], test_r2[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        # Add diagonal line
        min_r2 = min(min(train_r2), min(test_r2))
        max_r2 = max(max(train_r2), max(test_r2))
        ax1.plot([min_r2, max_r2], [min_r2, max_r2], 'k--', alpha=0.5, label='Perfect correlation')
        
        ax1.set_xlabel('Training R²')
        ax1.set_ylabel('Test R²')
        ax1.set_title('Training vs Test R² Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RMSE scatter
        ax2.scatter(train_rmse, test_rmse, s=100, alpha=0.7, color='orange')
        for i, model in enumerate(models):
            ax2.annotate(model, (train_rmse[i], test_rmse[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        # Add diagonal line
        min_rmse = min(min(train_rmse), min(test_rmse))
        max_rmse = max(max(train_rmse), max(test_rmse))
        ax2.plot([min_rmse, max_rmse], [min_rmse, max_rmse], 'k--', alpha=0.5, label='Perfect correlation')
        
        ax2.set_xlabel('Training RMSE')
        ax2.set_ylabel('Test RMSE')
        ax2.set_title('Training vs Test RMSE Performance')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'plots' / 'training_test_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_simulation_efficiency(self):
        """Plot simulation efficiency metrics."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        models = list(self.results.keys())
        total_sims = [self.results[m]['total_simulations'] for m in models]
        test_r2 = [self.results[m]['test_metrics']['r2_score'] for m in models]
        
        # Efficiency = performance / cost
        efficiency = [r2 / sims for r2, sims in zip(test_r2, total_sims)]
        
        bars = ax.bar(models, efficiency, alpha=0.7)
        ax.set_xlabel('Model Type')
        ax.set_ylabel('Efficiency (R² / Simulations)')
        ax.set_title('Model Efficiency: Performance per Simulation')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, eff in zip(bars, efficiency):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{eff:.2e}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'plots' / 'simulation_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _convert_to_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
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
        results_file = self.results_dir / 'data' / 'traditional_results.json'
        
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
            summary_data.append({
                'approach': 'traditional',
                'model_type': model_type,
                'train_r2': results['train_metrics']['r2_score'],
                'train_rmse': results['train_metrics']['rmse'],
                'test_r2': results['test_metrics']['r2_score'],
                'test_rmse': results['test_metrics']['rmse'],
                'total_simulations': results['total_simulations'],
                'efficiency_r2': results['test_metrics']['r2_score'] / results['total_simulations'],
                'efficiency_rmse': 1 / (results['test_metrics']['rmse'] * results['total_simulations'])
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
            f.write(f"# Traditional Approach Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Configuration\n\n")
            f.write(f"- **Simulator:** {self.config['simulator_type']}\n")
            f.write(f"- **Chemistry File:** {self.config['chem_file']}\n")
            f.write(f"- **Setup File:** {self.config['setup_file']}\n")
            f.write(f"- **Training Samples:** {self.config['n_train']}\n")
            f.write(f"- **Test Samples:** {self.config['n_test']}\n")
            f.write(f"- **K Columns:** {self.config['k_columns']}\n")
            f.write(f"- **Sampling Method:** {self.config['sampling_method']}\n\n")
            
            f.write(f"## Results Summary\n\n")
            f.write(f"| Model | Train RMSE | Test RMSE | Train R² | Test R² | Total Sims | Efficiency (R²/Sim) |\n")
            f.write(f"|-------|------------|-----------|----------|---------|------------|--------------------|\n")
            
            for model_type, results in self.results.items():
                train_r2 = results['train_metrics']['r2_score']
                test_r2 = results['test_metrics']['r2_score']
                train_rmse = results['train_metrics']['rmse']
                test_rmse = results['test_metrics']['rmse']
                total_sims = results['total_simulations']
                efficiency = test_r2 / total_sims
                
                f.write(f"| {model_type} | {train_rmse:.2e} | {test_rmse:.2e} | {train_r2:.4f} | {test_r2:.4f} | {total_sims} | {efficiency:.2e} |\n")
            
            f.write(f"\n## Best Performing Model\n\n")
            best_model = min(self.results.keys(), key=lambda k: self.results[k]['test_metrics']['rmse'])
            best_results = self.results[best_model]
            f.write(f"**{best_model}** achieved the lowest test RMSE of {best_results['test_metrics']['rmse']:.2e}\n")
            f.write(f"- Test R² score: {best_results['test_metrics']['r2_score']:.4f}\n")
            f.write(f"- Train RMSE: {best_results['train_metrics']['rmse']:.2e}\n")
            f.write(f"- Train R² score: {best_results['train_metrics']['r2_score']:.4f}\n")
            f.write(f"- Total simulations: {best_results['total_simulations']}\n\n")
            
            f.write(f"## Files Generated\n\n")
            f.write(f"- `data/traditional_results.json` - Complete results data\n")
            f.write(f"- `data/summary.csv` - Summary metrics\n")
            f.write(f"- `plots/model_comparison.png` - Model performance comparison\n")
            f.write(f"- `plots/performance_metrics.png` - Detailed performance metrics\n")
            f.write(f"- `plots/training_test_comparison.png` - Training vs test comparison\n")
            f.write(f"- `plots/simulation_efficiency.png` - Simulation efficiency analysis\n")
            f.write(f"- `models/*.joblib` - Trained model files\n")
        
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
        'k_bounds': np.array([  # Bounds for K values
            [1e-16, 1e-14],  # K1 bounds
            [5e-16, 5e-15],  # K2 bounds  
            [1e-16, 2e-15]   # K3 bounds
        ]),
        
        # Sampling configuration
        'n_train': 28,  # 80% of 35 simulations to match adaptive total
        'n_test': 7,    # 20% for testing - Total: 35 simulations like adaptive
        'sampling_method': 'latin_hypercube',
        
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
        'k_bounds': np.array([
            [1e-16, 1e-15],   # K1 bounds
            [5e-16, 5e-15],   # K2 bounds
            [1e-16, 2e-15]    # K3 bounds  
        ]),
        'n_train': 28,  # 80% of 35 simulations to match adaptive total
        'n_test': 7,    # 20% for testing - Total: 35 simulations like adaptive
        'sampling_method': 'latin_hypercube',
        'sampling_method': 'latin_hypercube',
        'models': ['random_forest', 'neural_network']  # Fewer models for LoKI
    }
    
    print("Traditional Approach Analysis")
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
    analyzer = TraditionalAnalysis(analysis_config)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
