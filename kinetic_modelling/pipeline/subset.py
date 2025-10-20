"""
Standard Subset Pipeline for sample efficiency analysis.

This pipeline:
1. Shuffles training data with different seeds
2. Trains models on various subset sizes
3. Evaluates on test set
4. Aggregates results across seeds (mean ± std)
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np

from .base import BasePipeline
from kinetic_modelling.data import MultiPressureDataset
from kinetic_modelling.model.base import BaseModel
from kinetic_modelling.sampling import SequentialSampler


class StandardSubsetPipeline(BasePipeline):
    """
    Pipeline for sample efficiency experiments.
    
    Trains models on multiple subset sizes with different random seeds,
    then aggregates results to analyze sample efficiency.
    
    Example:
        ```python
        pipeline = StandardSubsetPipeline(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model_class=SVRModel,
            model_params={'params': [...]},
            subset_sizes=[100, 200, 500, 1000],
            num_seeds=10,
            pipeline_name="sample_efficiency_experiment"
        )
        results = pipeline.run()
        ```
    """
    
    def __init__(
        self,
        train_dataset: MultiPressureDataset,
        test_dataset: MultiPressureDataset,
        model_class: type,
        model_params: Dict[str, Any],
        subset_sizes: List[int],
        num_seeds: int = 10,
        pipeline_name: str = "subset_pipeline",
        results_dir: str = "pipeline_results"
    ):
        """
        Initialize subset pipeline.
        
        Args:
            train_dataset: Full training dataset
            test_dataset: Test dataset
            model_class: Model class (e.g., SVRModel, NeuralNetModel)
            model_params: Parameters to pass to model constructor (e.g., {'params': [...]})
            subset_sizes: List of subset sizes to evaluate
            num_seeds: Number of random seeds for robustness
            pipeline_name: Name for this pipeline
            results_dir: Directory to save results
        """
        super().__init__(pipeline_name, results_dir)
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model_class = model_class
        self.model_params = model_params
        self.subset_sizes = subset_sizes
        self.num_seeds = num_seeds
        
        # Store configuration
        self.results['config'] = {
            'model_type': model_class.__name__,
            'model_params': str(model_params),
            'subset_sizes': subset_sizes,
            'num_seeds': num_seeds,
            'total_train_samples': len(train_dataset),
            'test_samples': len(test_dataset)
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the subset pipeline.
        
        For each seed:
          1. Shuffle training data
          2. For each subset size:
             - Sample subset
             - Train model
             - Evaluate on test set
        
        Then aggregate results across seeds (mean ± std).
        
        Returns:
            Dictionary with aggregated results
        """
        print(f"\n{'='*70}")
        print(f"Running Subset Pipeline: {self.pipeline_name}")
        print(f"{'='*70}")
        print(f"Subset sizes: {self.subset_sizes}")
        print(f"Number of seeds: {self.num_seeds}")
        print(f"Total training samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        
        # Record start time
        start_time = datetime.now()
        self.results['timestamp'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Get test data once
        x_test, y_test = self.test_dataset.get_data()
        n_outputs = y_test.shape[1]
        
        # Store results for each seed
        mse_per_output_all_seeds = []       # Shape: (num_seeds, n_outputs, num_subset_sizes)
        rmse_per_output_all_seeds = []      # Shape: (num_seeds, n_outputs, num_subset_sizes)
        rel_error_per_output_all_seeds = [] # Shape: (num_seeds, n_outputs, num_subset_sizes)
        total_mse_all_seeds = []            # Shape: (num_seeds, num_subset_sizes)
        total_rmse_all_seeds = []           # Shape: (num_seeds, num_subset_sizes)
        models_all_seeds = []               # Shape: (num_seeds, num_subset_sizes) -> list of trained models
        
        # Loop over seeds
        for seed_idx in range(self.num_seeds):
            print(f"\n{'='*70}")
            print(f"Seed {seed_idx + 1}/{self.num_seeds} (seed={seed_idx})")
            print(f"{'='*70}")
            
            # Shuffle training data with this seed using SequentialSampler with shuffle=True
            print(f"  Shuffling training data with seed={seed_idx}...")
            shuffle_sampler = SequentialSampler(sampler_name=f"shuffle_seed_{seed_idx}")
            
            # Sample all data with shuffle=True (this effectively shuffles the entire dataset)
            shuffled_dataset = shuffle_sampler.sample(
                self.train_dataset,
                n_samples=len(self.train_dataset),
                shuffle=True,
                seed=seed_idx
            )
            
            # Storage for this seed
            mse_per_output_this_seed = []       # Shape: (n_outputs, num_subset_sizes)
            rmse_per_output_this_seed = []      # Shape: (n_outputs, num_subset_sizes)
            rel_error_per_output_this_seed = [] # Shape: (n_outputs, num_subset_sizes)
            total_mse_this_seed = []            # Shape: (num_subset_sizes,)
            total_rmse_this_seed = []           # Shape: (num_subset_sizes,)
            models_this_seed = []               # List of trained models for this seed
            
            # Loop over subset sizes
            for subset_size in self.subset_sizes:
                print(f"\n  Training on {subset_size} samples...")
                
                # Sample subset (sequential from shuffled data)
                sampler = SequentialSampler(sampler_name=f"subset_{subset_size}")
                subset_dataset = sampler.sample(shuffled_dataset, n_samples=subset_size)
                x_subset, y_subset = subset_dataset.get_data()
                
                # Create and train model
                model = self.model_class(**self.model_params)
                model.fit(x_subset, y_subset)
                
                # Store the trained model
                models_this_seed.append(model)
                
                # Evaluate on test set - compute MSE, RMSE and relative error per output
                y_pred = model.predict(x_test)
                
                mse_per_output = []
                rmse_per_output = []
                rel_error_per_output = []
                for i in range(n_outputs):
                    # MSE and RMSE for this output
                    mse = np.mean((y_test[:, i] - y_pred[:, i]) ** 2)
                    rmse = np.sqrt(mse)
                    mse_per_output.append(mse)
                    rmse_per_output.append(rmse)
                    
                    # Relative error: mean(|pred - true| / |true|) * 100
                    abs_errors = np.abs(y_test[:, i] - y_pred[:, i])
                    abs_true = np.abs(y_test[:, i])
                    # Avoid division by zero
                    relative_errors = np.where(abs_true > 1e-15, abs_errors / abs_true, 0.0)
                    mean_rel_error = np.mean(relative_errors) * 100  # as percentage
                    rel_error_per_output.append(mean_rel_error)
                
                # Total MSE and RMSE (sum of individual values)
                total_mse = np.sum(mse_per_output)
                total_rmse = np.sum(rmse_per_output)
                
                mse_per_output_this_seed.append(mse_per_output)
                rmse_per_output_this_seed.append(rmse_per_output)
                rel_error_per_output_this_seed.append(rel_error_per_output)
                total_mse_this_seed.append(total_mse)
                total_rmse_this_seed.append(total_rmse)
                
                print(f"    Total MSE: {total_mse:.6e} | Total RMSE: {total_rmse:.6e}")
                print(f"    Per-output RMSE: {[f'{x:.6e}' for x in rmse_per_output]}")
                print(f"    Per-output Rel Error (%): {[f'{x:.2f}' for x in rel_error_per_output]}")
            
            # Store results for this seed
            mse_per_output_all_seeds.append(mse_per_output_this_seed)
            rmse_per_output_all_seeds.append(rmse_per_output_this_seed)
            rel_error_per_output_all_seeds.append(rel_error_per_output_this_seed)
            total_mse_all_seeds.append(total_mse_this_seed)
            total_rmse_all_seeds.append(total_rmse_this_seed)
            models_all_seeds.append(models_this_seed)
        
        # Reorganize models: from (num_seeds, num_subset_sizes) to (num_subset_sizes, num_seeds)
        all_trained_models = []
        for subset_idx in range(len(self.subset_sizes)):
            models_for_subset = []
            for seed_idx in range(self.num_seeds):
                # Get model at subset_idx trained with seed_idx
                models_for_subset.append(models_all_seeds[seed_idx][subset_idx])
            all_trained_models.append(models_for_subset)
        
        # Aggregate results across seeds
        print(f"\n{'='*70}")
        print(f"Aggregating results across {self.num_seeds} seeds...")
        print(f"{'='*70}")
        
        # Convert to numpy arrays for easier manipulation
        # Shape: (num_seeds, num_subset_sizes)
        total_mse_array = np.array(total_mse_all_seeds)
        total_rmse_array = np.array(total_rmse_all_seeds)
        
        # Shape: (num_seeds, num_subset_sizes, n_outputs)
        mse_per_output_array = np.array(mse_per_output_all_seeds)
        rmse_per_output_array = np.array(rmse_per_output_all_seeds)
        rel_error_per_output_array = np.array(rel_error_per_output_all_seeds)
        
        # Compute mean and std across seeds for MSE
        mean_total_mse = np.mean(total_mse_array, axis=0)
        std_total_mse = np.std(total_mse_array, axis=0) / np.sqrt(self.num_seeds)  # Standard error
        
        mean_mse_per_output = np.mean(mse_per_output_array, axis=0)
        std_mse_per_output = np.std(mse_per_output_array, axis=0) / np.sqrt(self.num_seeds)
        
        # Compute mean and std across seeds for RMSE
        mean_total_rmse = np.mean(total_rmse_array, axis=0)
        std_total_rmse = np.std(total_rmse_array, axis=0) / np.sqrt(self.num_seeds)  # Standard error
        
        mean_rmse_per_output = np.mean(rmse_per_output_array, axis=0)
        std_rmse_per_output = np.std(rmse_per_output_array, axis=0) / np.sqrt(self.num_seeds)
        
        # Compute mean and std across seeds for relative error
        mean_rel_error_per_output = np.mean(rel_error_per_output_array, axis=0)
        std_rel_error_per_output = np.std(rel_error_per_output_array, axis=0) / np.sqrt(self.num_seeds)
        
        # Store results
        self.results['raw_results'] = {
            'total_mse_per_seed': total_mse_array.tolist(),
            'total_rmse_per_seed': total_rmse_array.tolist(),
            'mse_per_output_per_seed': mse_per_output_array.tolist(),
            'rmse_per_output_per_seed': rmse_per_output_array.tolist(),
            'rel_error_per_output_per_seed': rel_error_per_output_array.tolist()
        }
        
        self.results['aggregated_results'] = {
            'subset_sizes': self.subset_sizes,
            # MSE metrics (for backward compatibility)
            'mean_total_mse': mean_total_mse.tolist(),
            'std_total_mse': std_total_mse.tolist(),
            'mean_mse_per_output': mean_mse_per_output.tolist(),
            'std_mse_per_output': std_mse_per_output.tolist(),
            'final_mean_mse': float(mean_total_mse[-1]),
            'final_std_mse': float(std_total_mse[-1]),
            # RMSE metrics
            'mean_total_rmse': mean_total_rmse.tolist(),
            'std_total_rmse': std_total_rmse.tolist(),
            'mean_rmse_per_output': mean_rmse_per_output.tolist(),
            'std_rmse_per_output': std_rmse_per_output.tolist(),
            'final_mean_rmse': float(mean_total_rmse[-1]),
            'final_std_rmse': float(std_total_rmse[-1]),
            # Relative error metrics
            'mean_rel_error_per_output': mean_rel_error_per_output.tolist(),
            'std_rel_error_per_output': std_rel_error_per_output.tolist()
        }
        
        # Store trained models as instance variable (not in results dict - models can't be serialized to JSON)
        self.all_trained_models = all_trained_models
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Aggregated Results Summary")
        print(f"{'='*70}")
        print(f"{'Subset Size':<15} {'Mean Total MSE':<20} {'Mean Total RMSE':<20}")
        print(f"{'-'*70}")
        for i, size in enumerate(self.subset_sizes):
            print(f"{size:<15} {mean_total_mse[i]:<20.6e} {mean_total_rmse[i]:<20.6e}")
        
        # Print per-output metrics for final size
        print(f"\n{'='*70}")
        print(f"Final Size ({self.subset_sizes[-1]} samples) - Per-Output Metrics")
        print(f"{'='*70}")
        print(f"{'Output':<10} {'MSE':<20} {'RMSE':<20} {'Rel Error (%)':<20}")
        print(f"{'-'*70}")
        for i in range(n_outputs):
            mse_val = mean_mse_per_output[-1][i]
            rmse_val = mean_rmse_per_output[-1][i]
            rel_err_val = mean_rel_error_per_output[-1][i]
            print(f"K{i+1:<9} {mse_val:<20.6e} {rmse_val:<20.6e} {rel_err_val:<20.2f}")
        
        # Calculate time elapsed
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        self.results['execution_time_seconds'] = elapsed
        print(f"\nExecution time: {elapsed:.2f} seconds")
        print(f"{'='*70}\n")
        
        return self.results
    
    def save_and_return(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Run pipeline and optionally save results.
        
        Args:
            save_results: Whether to save results to JSON
            
        Returns:
            Pipeline results dictionary
        """
        results = self.run()
        
        if save_results:
            filepath = self.save_results()
            print(f"✓ Results saved to: {filepath}\n")
        
        return results
