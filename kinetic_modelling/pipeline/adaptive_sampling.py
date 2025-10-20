"""
Adaptive Sampling Pipeline for kinetic modeling.

This pipeline implements iterative adaptive sampling where:
1. Start with initial training data
2. At each iteration:
   - Define a window around the average point of current training data
   - Sample new points from pool dataset within this window
   - Add new samples to training set
   - Retrain model
   - Evaluate performance
3. Window shrinks each iteration to focus sampling
4. Multiple seeds for robustness (different shuffles of pool data)
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np

from .base import BasePipeline
from kinetic_modelling.data import MultiPressureDataset
from kinetic_modelling.model.base import BaseModel
from kinetic_modelling.sampling import SequentialSampler, WindowSampler


class AdaptiveSamplingPipeline(BasePipeline):
    """
    Pipeline for adaptive sampling experiments.
    
    Inspired by K-centered adaptive learning, this pipeline:
    - Iteratively samples from a pool dataset
    - Uses a shrinking window around the average training point
    - Incrementally adds samples and retrains the model
    - Tracks model evolution across iterations
    
    Example:
        ```python
        pipeline = AdaptiveSamplingPipeline(
            initial_dataset=initial_train,
            pool_dataset=large_pool,
            test_dataset=test_data,
            model_class=SVRModel,
            model_params={'params': [...]},
            n_iterations=10,
            samples_per_iteration=50,
            initial_window_size=0.3,
            shrink_rate=0.8,
            num_seeds=5
        )
        results = pipeline.run()
        ```
    """
    
    def __init__(
        self,
        initial_dataset: MultiPressureDataset,
        pool_dataset: MultiPressureDataset,
        test_dataset: MultiPressureDataset,
        model_class: type,
        model_params: Dict[str, Any],
        n_iterations: int = 10,
        samples_per_iteration: int = 50,
        initial_window_size: float = 0.3,
        shrink_rate: float = 0.8,
        num_seeds: int = 5,
        window_type: str = 'output',  # 'output' or 'input'
        pipeline_name: str = "adaptive_sampling",
        results_dir: str = "pipeline_results"
    ):
        """
        Initialize adaptive sampling pipeline.
        
        Args:
            initial_dataset: Initial training dataset (small)
            pool_dataset: Large pool dataset to sample from
            test_dataset: Test dataset for evaluation
            model_class: Model class (e.g., SVRModel, NeuralNetModel)
            model_params: Parameters to pass to model constructor
            n_iterations: Number of adaptive iterations
            samples_per_iteration: Samples to add per iteration
            initial_window_size: Initial window size as fraction (e.g., 0.3 = ±30%)
            shrink_rate: Factor to shrink window each iteration (0.8 = 20% reduction)
            num_seeds: Number of seeds for robustness
            window_type: 'output' (sample based on y) or 'input' (sample based on x)
            pipeline_name: Name for this pipeline
            results_dir: Directory to save results
        """
        super().__init__(pipeline_name, results_dir)
        
        self.initial_dataset = initial_dataset
        self.pool_dataset = pool_dataset
        self.test_dataset = test_dataset
        self.model_class = model_class
        self.model_params = model_params
        self.n_iterations = n_iterations
        self.samples_per_iteration = samples_per_iteration
        self.initial_window_size = initial_window_size
        self.shrink_rate = shrink_rate
        self.num_seeds = num_seeds
        self.window_type = window_type
        
        # Store configuration
        self.results['config'] = {
            'model_type': model_class.__name__,
            'model_params': str(model_params),
            'n_iterations': n_iterations,
            'samples_per_iteration': samples_per_iteration,
            'initial_window_size': initial_window_size,
            'shrink_rate': shrink_rate,
            'num_seeds': num_seeds,
            'window_type': window_type,
            'initial_train_samples': len(initial_dataset),
            'pool_samples': len(pool_dataset),
            'test_samples': len(test_dataset)
        }
    
    def _calculate_center_point(self, dataset: MultiPressureDataset) -> np.ndarray:
        """
        Calculate the center (average) point of the dataset.
        
        Args:
            dataset: Dataset to calculate center from
            
        Returns:
            center: Average point (1, n_features)
        """
        x_data, y_data = dataset.get_data()
        
        if self.window_type == 'output':
            # Use average of output (y) values
            center = np.mean(y_data, axis=0, keepdims=True)
        else:  # 'input'
            # Use average of input (x) values
            center = np.mean(x_data, axis=0, keepdims=True)
        
        return center
    
    def _run_single_seed(
        self,
        seed: int,
        x_test: np.ndarray,
        y_test: np.ndarray,
        n_outputs: int
    ) -> List[Dict]:
        """
        Run adaptive sampling for a single seed.
        
        Args:
            seed: Random seed for this run
            x_test: Test input features
            y_test: Test output features
            n_outputs: Number of output features
            
        Returns:
            List of results for each iteration
        """
        print(f"\n{'='*70}")
        print(f"Seed {seed}")
        print(f"{'='*70}")
        
        # Shuffle pool dataset with this seed using SequentialSampler with shuffle=True
        shuffle_sampler = SequentialSampler(sampler_name=f"shuffle_seed_{seed}")
        shuffled_pool = shuffle_sampler.sample(
            self.pool_dataset,
            n_samples=len(self.pool_dataset),
            shuffle=True,
            seed=seed
        )
        
        pool_x, pool_y = shuffled_pool.get_data()
        
        # Start with initial training data from the SHUFFLED pool
        # This ensures each seed gets a different initial training set
        initial_size = len(self.initial_dataset)
        
        seed_results = []
        used_indices = set()
        
        # Check if we have initial training data
        if initial_size > 0:
            current_x_train = pool_x[:initial_size].copy()
            current_y_train = pool_y[:initial_size].copy()
            
            # Mark these initial samples as used
            used_indices = set(range(initial_size))
            
            # Create initial training dataset
            current_train_dataset = MultiPressureDataset(
                nspecies=self.initial_dataset.nspecies,
                num_pressure_conditions=self.initial_dataset.num_pressure_conditions,
                processed_x=current_x_train,
                processed_y=current_y_train,
                scaler_input=self.initial_dataset.scaler_input,
                scaler_output=self.initial_dataset.scaler_output
            )
            
            # Iteration 0: Initial model
            print(f"\n  Iteration 0 - Initial Training")
            print(f"    Training samples: {len(current_x_train)}")
            
            model = self.model_class(**self.model_params)
            model.fit(current_x_train, current_y_train)
            
            # Evaluate
            y_pred = model.predict(x_test)
            mse_per_output = []
            for i in range(n_outputs):
                mse = np.mean((y_test[:, i] - y_pred[:, i]) ** 2)
                mse_per_output.append(mse)
            total_mse = np.sum(mse_per_output)
            
            # Calculate center point
            center = self._calculate_center_point(current_train_dataset)
            
            seed_results.append({
                'iteration': 0,
                'seed': seed,
                'total_samples': len(current_x_train),
                'samples_added': 0,
                'mse_per_output': [float(m) for m in mse_per_output],
                'total_mse': float(total_mse),
                'window_size': float(self.initial_window_size),
                'center_point': [float(c) for c in center.flatten()],
                'available_pool_samples': len(pool_x) - len(used_indices)
            })
            
            print(f"    Test MSE: {total_mse:.6e}")
        else:
            # No initial training data - start with empty training set
            print(f"\n  No initial training data - starting adaptive sampling from scratch")
            current_x_train = np.empty((0, pool_x.shape[1]))
            current_y_train = np.empty((0, pool_y.shape[1]))
            # Use pool center as initial center point
            center = np.mean(pool_y, axis=0, keepdims=True) if self.window_type == 'output' else np.mean(pool_x, axis=0, keepdims=True)
        
        # Adaptive iterations
        for iteration in range(1, self.n_iterations + 1):
            print(f"\n  Iteration {iteration}")
            
            # Calculate adaptive window size
            window_size = self.initial_window_size * (self.shrink_rate ** (iteration - 1))
            
            print(f"    Window size: {window_size:.4f}")
            print(f"    Sampling from pool...")
            
            # Create window sampler with current center and window size
            window_sampler = WindowSampler(
                center_point=center,
                window_size=window_size,
                window_type=self.window_type,
                sampler_name=f"window_iter_{iteration}_seed_{seed}"
            )
            
            # Sample from shuffled pool within window, excluding already used indices
            try:
                sampled_subset = window_sampler.sample(
                    shuffled_pool,
                    n_samples=self.samples_per_iteration,
                    shuffle=False,  # Already shuffled
                    seed=None,
                    exclude_indices=used_indices
                )
                
                new_x, new_y = sampled_subset.get_data()
                
                # Update used indices - find which indices were selected
                # This is a bit hacky but necessary to track usage
                for idx in range(len(pool_x)):
                    if idx in used_indices:
                        continue
                    # Check if this sample matches any of the new samples
                    for new_sample_idx in range(len(new_x)):
                        if np.allclose(pool_x[idx], new_x[new_sample_idx]):
                            used_indices.add(idx)
                            break
                
                samples_added = len(new_x)
                print(f"    Samples added: {samples_added}")
                
            except ValueError as e:
                print(f"    ⚠️  {str(e)}")
                break
            
            # Add new samples to training set
            current_x_train = np.vstack([current_x_train, new_x])
            current_y_train = np.vstack([current_y_train, new_y])
            
            # Update training dataset
            current_train_dataset = MultiPressureDataset(
                nspecies=self.initial_dataset.nspecies,
                num_pressure_conditions=self.initial_dataset.num_pressure_conditions,
                processed_x=current_x_train,
                processed_y=current_y_train,
                scaler_input=self.initial_dataset.scaler_input,
                scaler_output=self.initial_dataset.scaler_output
            )
            
            # Recalculate center
            center = self._calculate_center_point(current_train_dataset)
            
            # Retrain model
            model = self.model_class(**self.model_params)
            model.fit(current_x_train, current_y_train)
            
            # Evaluate
            y_pred = model.predict(x_test)
            mse_per_output = []
            for i in range(n_outputs):
                mse = np.mean((y_test[:, i] - y_pred[:, i]) ** 2)
                mse_per_output.append(mse)
            total_mse = np.sum(mse_per_output)
            
            print(f"    Total samples: {len(current_x_train)}")
            print(f"    Test MSE: {total_mse:.6e}")
            
            seed_results.append({
                'iteration': iteration,
                'seed': seed,
                'total_samples': len(current_x_train),
                'samples_added': samples_added,
                'mse_per_output': [float(m) for m in mse_per_output],
                'total_mse': float(total_mse),
                'window_size': float(window_size),
                'center_point': [float(c) for c in center.flatten()],
                'available_pool_samples': len(pool_x) - len(used_indices)
            })
        
        return seed_results
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the adaptive sampling pipeline.
        
        For each seed:
          1. Shuffle pool dataset
          2. Start with initial training data
          3. For each iteration:
             - Calculate center of current training data
             - Define shrinking window around center
             - Sample new points from pool within window
             - Add to training set and retrain
             - Evaluate performance
        
        Then aggregate results across seeds.
        
        Returns:
            Dictionary with aggregated results
        """
        print(f"\n{'='*70}")
        print(f"Running Adaptive Sampling Pipeline: {self.pipeline_name}")
        print(f"{'='*70}")
        print(f"Iterations: {self.n_iterations}")
        print(f"Samples per iteration: {self.samples_per_iteration}")
        print(f"Initial window size: {self.initial_window_size}")
        print(f"Shrink rate: {self.shrink_rate}")
        print(f"Window type: {self.window_type}")
        print(f"Number of seeds: {self.num_seeds}")
        print(f"Initial training: {len(self.initial_dataset)} samples")
        print(f"Pool dataset: {len(self.pool_dataset)} samples")
        
        # Record start time
        start_time = datetime.now()
        self.results['timestamp'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Get test data
        x_test, y_test = self.test_dataset.get_data()
        n_outputs = y_test.shape[1]
        
        # Run for each seed
        all_seed_results = []
        
        for seed_idx in range(self.num_seeds):
            seed = seed_idx + 42  # Start from seed 42
            seed_results = self._run_single_seed(seed, x_test, y_test, n_outputs)
            all_seed_results.append(seed_results)
        
        # Aggregate results across seeds
        print(f"\n{'='*70}")
        print(f"Aggregating results across {self.num_seeds} seeds...")
        print(f"{'='*70}")
        
        max_iterations = max(len(sr) for sr in all_seed_results)
        aggregated_results = []
        
        for iter_idx in range(max_iterations):
            # Collect results for this iteration across seeds
            iter_results = []
            for seed_results in all_seed_results:
                if iter_idx < len(seed_results):
                    iter_results.append(seed_results[iter_idx])
            
            if not iter_results:
                continue
            
            # Average metrics
            avg_total_samples = np.mean([r['total_samples'] for r in iter_results])
            avg_samples_added = np.mean([r['samples_added'] for r in iter_results])
            avg_total_mse = np.mean([r['total_mse'] for r in iter_results])
            std_total_mse = np.std([r['total_mse'] for r in iter_results]) / np.sqrt(len(iter_results))
            
            # Average MSE per output
            mse_per_output_array = np.array([r['mse_per_output'] for r in iter_results])
            avg_mse_per_output = np.mean(mse_per_output_array, axis=0)
            std_mse_per_output = np.std(mse_per_output_array, axis=0) / np.sqrt(len(iter_results))
            
            aggregated_results.append({
                'iteration': iter_idx,
                'total_samples': float(avg_total_samples),
                'samples_added': float(avg_samples_added),
                'mean_total_mse': float(avg_total_mse),
                'std_total_mse': float(std_total_mse),
                'mean_mse_per_output': avg_mse_per_output.tolist(),
                'std_mse_per_output': std_mse_per_output.tolist(),
                'window_size': float(iter_results[0]['window_size']),
                'num_seeds': len(iter_results)
            })
        
        # Store results
        self.results['raw_results'] = {
            'all_seed_results': all_seed_results
        }
        
        self.results['aggregated_results'] = aggregated_results
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Aggregated Results Summary")
        print(f"{'='*70}")
        print(f"{'Iter':<6} {'Samples':<10} {'Added':<8} {'MSE':<15} {'Std Err':<15} {'Window':<10}")
        print(f"{'-'*70}")
        
        for result in aggregated_results:
            print(f"{result['iteration']:<6} "
                  f"{result['total_samples']:<10.0f} "
                  f"{result['samples_added']:<8.0f} "
                  f"{result['mean_total_mse']:<15.6e} "
                  f"{result['std_total_mse']:<15.6e} "
                  f"{result['window_size']:<10.4f}")
        
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
