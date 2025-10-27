"""
Adaptive Batch Sampling Pipeline for kinetic modeling with Neural Networks.

This pipeline implements iterative adaptive sampling with continuous NN training where:
1. Start with a Neural Network model
2. At each iteration:
   - Define a window around the test dataset center
   - Sample new points from pool dataset within this window
   - Train the NN for multiple epochs with the new samples (continuing from previous weights)
   - Evaluate performance
3. Window shrinks each iteration to focus sampling on relevant regions
4. Multiple seeds for robustness

Key differences from adaptive_sampling.py:
- Uses Neural Network instead of SVM
- Continuously trains the same NN model (doesn't reset between iterations)
- Trains with batches and epochs instead of simple fit()
- Samples from pool datasets at each iteration (ignores initial_dataset)
- Center point calculated from test dataset (not training data)
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from .base import BasePipeline
from kinetic_modelling.data import MultiPressureDataset
from kinetic_modelling.model.base import BaseModel
from kinetic_modelling.sampling import WindowSampler


class AdaptiveBatchSamplingPipeline(BasePipeline):
    """
    Pipeline for adaptive batch sampling with continuous Neural Network training.
    
    This pipeline:
    - Iteratively samples from pool datasets using shrinking windows
    - Continuously trains a single NN model across iterations
    - Uses mini-batch gradient descent with configurable epochs
    - Window center is always based on test dataset
    - Tracks model evolution across iterations
    
    Example:
        ```python
        pipeline = AdaptiveBatchSamplingPipeline(
            pool_datasets=[pool1, pool2, pool3],
            test_dataset=test_data,
            model_class=NeuralNetModel,
            model_params={'input_size': 10, 'output_size': 5, 'hidden_sizes': (64, 32)},
            n_iterations=10,
            samples_per_iteration=200,
            n_epochs=10,
            batch_size=64,
            initial_window_size=0.3,
            shrink_rate=0.8,
            num_seeds=5,
            window_type='output'
        )
        results = pipeline.run()
        ```
    """
    
    def __init__(
        self,
        pool_datasets: List[MultiPressureDataset],
        test_dataset: MultiPressureDataset,
        model_class: type,
        model_params: Dict[str, Any],
        n_iterations: int = 10,
        samples_per_iteration: int = 200,
        n_epochs: int = 10,
        batch_size: int = 64,
        initial_window_size: float = 0.3,
        shrink_rate: float = 0.8,
        num_seeds: int = 5,
        window_type: str = 'output',
        pipeline_name: str = "adaptive_batch_sampling",
        results_dir: str = "pipeline_results"
    ):
        """
        Initialize adaptive batch sampling pipeline.
        
        Args:
            pool_datasets: List of pool datasets to sample from sequentially
            test_dataset: Test dataset for evaluation
            model_class: Neural Network model class (e.g., NeuralNetModel)
            model_params: Parameters to pass to model constructor
            n_iterations: Number of adaptive iterations (one per pool file)
            samples_per_iteration: Samples to grab from each pool file
            n_epochs: Number of epochs to train at each iteration
            batch_size: Batch size for NN training
            initial_window_size: Initial window size as fraction (e.g., 0.3 = ±30%)
            shrink_rate: Factor to shrink window each iteration (0.8 = 20% reduction)
            num_seeds: Number of seeds for robustness
            window_type: 'output' (sample based on y) or 'input' (sample based on x)
            pipeline_name: Name for this pipeline
            results_dir: Directory to save results
        """
        super().__init__(pipeline_name, results_dir)
        
        self.pool_datasets = pool_datasets
        self.test_dataset = test_dataset
        self.model_class = model_class
        self.model_params = model_params
        self.n_iterations = n_iterations
        self.samples_per_iteration = samples_per_iteration
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.initial_window_size = initial_window_size
        self.shrink_rate = shrink_rate
        self.num_seeds = num_seeds
        self.window_type = window_type
        
        # Store configuration
        total_pool_samples = sum(len(pool_ds) for pool_ds in pool_datasets)
        self.results['config'] = {
            'model_type': model_class.__name__,
            'model_params': str(model_params),
            'n_iterations': n_iterations,
            'samples_per_iteration': samples_per_iteration,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'initial_window_size': initial_window_size,
            'shrink_rate': shrink_rate,
            'num_seeds': num_seeds,
            'window_type': window_type,
            'num_pool_datasets': len(pool_datasets),
            'pool_samples_per_dataset': [len(pool_ds) for pool_ds in pool_datasets],
            'total_pool_samples': total_pool_samples,
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
    
    def _create_dataloader(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader from numpy arrays.
        
        Args:
            x_data: Input features
            y_data: Output targets
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            PyTorch DataLoader
        """
        # Convert to PyTorch tensors
        x_tensor = torch.FloatTensor(x_data)
        y_tensor = torch.FloatTensor(y_data)
        
        # Create TensorDataset and DataLoader
        tensor_dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False
        )
        
        return dataloader
    
    def _train_for_epochs(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        n_epochs: int
    ) -> List[float]:
        """
        Train model for specified number of epochs.
        
        Args:
            model: Model to train (must have train_single_batch method)
            train_loader: DataLoader for training
            n_epochs: Number of epochs to train
            
        Returns:
            List of average training losses per epoch
        """
        epoch_losses = []
        
        for epoch in range(n_epochs):
            batch_losses = []
            
            for batch_x, batch_y in train_loader:
                # Convert to numpy for model compatibility
                batch_x_np = batch_x.numpy()
                batch_y_np = batch_y.numpy()
                
                # Perform single batch gradient update
                train_loss = model.train_single_batch(batch_x_np, batch_y_np)
                batch_losses.append(train_loss)
            
            # Average loss for this epoch
            avg_epoch_loss = np.mean(batch_losses)
            epoch_losses.append(avg_epoch_loss)
        
        return epoch_losses
    
    def _run_single_seed(
        self,
        seed: int,
        x_test: np.ndarray,
        y_test: np.ndarray,
        n_outputs: int
    ) -> List[Dict]:
        """
        Run adaptive batch sampling for a single seed.
        
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
        
        # Calculate center point from TEST dataset (stays constant)
        center = self._calculate_center_point(self.test_dataset)
        print(f"  Center point calculated from test dataset: {center.flatten()[:3]}... (showing first 3 values)")
        
        # Create a new model for this seed
        print(f"  Initializing Neural Network model...")
        model = self.model_class(**self.model_params)
        
        seed_results = []
        
        # Iteration 0: Evaluate untrained model
        print(f"\n  Iteration 0 - Untrained Model")
        y_pred = model.predict(x_test)
        mse_per_output = []
        for i in range(n_outputs):
            mse = np.mean((y_test[:, i] - y_pred[:, i]) ** 2)
            mse_per_output.append(mse)
        total_mse = np.sum(mse_per_output)
        
        seed_results.append({
            'iteration': 0,
            'seed': seed,
            'total_samples_seen': 0,
            'samples_added': 0,
            'mse_per_output': [float(m) for m in mse_per_output],
            'total_mse': float(total_mse),
            'window_size': float(self.initial_window_size),
            'center_point': [float(c) for c in center.flatten()],
            'pool_idx': None,
            'avg_train_loss': None
        })
        
        print(f"    Test MSE (untrained): {total_mse:.6e}")
        
        # Adaptive iterations - try to sample from pools with fallback
        total_samples_seen = 0
        current_pool_idx = 0  # Start with first pool
        used_indices = set()  # Track which indices we've used from current pool
        
        for iteration in range(1, self.n_iterations + 1):
            print(f"\n  Iteration {iteration}")
            
            # Calculate adaptive window size
            window_size = self.initial_window_size * (self.shrink_rate ** (iteration - 1))
            
            print(f"    Window size: {window_size:.4f}")
            print(f"    Currently using Pool {current_pool_idx + 1}/{len(self.pool_datasets)}")
            
            # Collect samples - may need to use multiple pools to reach samples_per_iteration
            all_sampled_x = []
            all_sampled_y = []
            samples_needed = self.samples_per_iteration
            
            # Keep trying pools until we get enough samples or run out of pools
            while samples_needed > 0 and current_pool_idx < len(self.pool_datasets):
                # Create window sampler with current window size
                window_sampler = WindowSampler(
                    center_point=center,
                    window_size=window_size,
                    window_type=self.window_type,
                    sampler_name=f"window_iter_{iteration}_seed_{seed}_pool_{current_pool_idx}"
                )
                
                try:
                    # Get current pool dataset
                    current_pool = self.pool_datasets[current_pool_idx]
                    
                    print(f"    Attempting to sample {samples_needed} from Pool {current_pool_idx + 1}...")
                    
                    sampled_subset = window_sampler.sample(
                        current_pool,
                        n_samples=samples_needed,
                        shuffle=True,
                        seed=seed + current_pool_idx,  # Different seed for each pool
                        exclude_indices=used_indices if len(used_indices) > 0 else None
                    )
                    
                    sampled_x, sampled_y = sampled_subset.get_data()
                    samples_obtained = len(sampled_x)
                    
                    # Track which indices were used (for next iteration on same pool)
                    if samples_obtained > 0:
                        # Mark these samples as "used" to avoid resampling them
                        pool_x, pool_y = current_pool.get_data()
                        for new_sample_idx in range(len(sampled_x)):
                            for pool_idx_check in range(len(pool_x)):
                                if pool_idx_check not in used_indices:
                                    if np.allclose(pool_x[pool_idx_check], sampled_x[new_sample_idx]):
                                        used_indices.add(pool_idx_check)
                                        break
                        
                        # Add to collection
                        all_sampled_x.append(sampled_x)
                        all_sampled_y.append(sampled_y)
                        samples_needed -= samples_obtained
                        
                        print(f"    ✓ Got {samples_obtained} samples from Pool {current_pool_idx + 1}")
                        print(f"    Total used from this pool: {len(used_indices)}/{len(current_pool)}")
                        
                        # If we still need more samples, try next pool
                        if samples_needed > 0:
                            print(f"    Still need {samples_needed} more samples...")
                            if current_pool_idx + 1 < len(self.pool_datasets):
                                current_pool_idx += 1
                                used_indices = set()  # Reset for new pool
                                print(f"    🔄 Moving to Pool {current_pool_idx + 1}")
                            else:
                                print(f"    ⚠️  No more pools available")
                                break
                    
                except ValueError as e:
                    # Current pool exhausted or no samples in window
                    print(f"    ⚠️  Pool {current_pool_idx + 1}: {str(e)}")
                    
                    # Move to next pool if available
                    if current_pool_idx + 1 < len(self.pool_datasets):
                        current_pool_idx += 1
                        used_indices = set()  # Reset used indices for new pool
                        print(f"    🔄 Switching to Pool {current_pool_idx + 1}")
                    else:
                        # No more pools available
                        print(f"    ❌ All pools exhausted.")
                        break
            
            # Combine all sampled data
            if len(all_sampled_x) == 0:
                print(f"    ❌ No samples could be added. Stopping iterations.")
                break
            
            sampled_x = np.vstack(all_sampled_x)
            sampled_y = np.vstack(all_sampled_y)
            samples_added = len(sampled_x)
            
            total_samples_seen += samples_added
            print(f"    Total samples collected this iteration: {samples_added}")
            print(f"    Total samples seen so far: {total_samples_seen}")
            
            # Create dataloader for the new samples
            train_loader = self._create_dataloader(
                sampled_x,
                sampled_y,
                batch_size=self.batch_size,
                shuffle=True
            )
            
            # Train the model for n_epochs (continuing from previous weights)
            print(f"    Training for {self.n_epochs} epochs with batch size {self.batch_size}...")
            epoch_losses = self._train_for_epochs(model, train_loader, self.n_epochs)
            avg_train_loss = np.mean(epoch_losses)
            
            print(f"    Average training loss: {avg_train_loss:.6e}")
            
            # Evaluate on test set
            y_pred = model.predict(x_test)
            mse_per_output = []
            for i in range(n_outputs):
                mse = np.mean((y_test[:, i] - y_pred[:, i]) ** 2)
                mse_per_output.append(mse)
            total_mse = np.sum(mse_per_output)
            
            print(f"    Test MSE: {total_mse:.6e}")
            
            seed_results.append({
                'iteration': iteration,
                'seed': seed,
                'total_samples_seen': total_samples_seen,
                'samples_added': samples_added,
                'mse_per_output': [float(m) for m in mse_per_output],
                'total_mse': float(total_mse),
                'window_size': float(window_size),
                'center_point': [float(c) for c in center.flatten()],
                'pool_idx': current_pool_idx,
                'avg_train_loss': float(avg_train_loss),
                'epoch_losses': [float(loss) for loss in epoch_losses]
            })
        
        return seed_results
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the adaptive batch sampling pipeline.
        
        For each seed:
          1. Create a new NN model
          2. Calculate center point from test dataset (stays constant)
          3. For each iteration:
             - Define shrinking window around center
             - Sample new points from next pool dataset within window
             - Train NN for n_epochs with the new samples (continuing training)
             - Evaluate performance on test set
        
        Then aggregate results across seeds.
        
        Returns:
            Dictionary with aggregated results
        """
        print(f"\n{'='*70}")
        print(f"Running Adaptive Batch Sampling Pipeline: {self.pipeline_name}")
        print(f"{'='*70}")
        print(f"Model: {self.model_class.__name__}")
        print(f"Iterations: {self.n_iterations}")
        print(f"Samples per iteration: {self.samples_per_iteration}")
        print(f"Epochs per iteration: {self.n_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Initial window size: {self.initial_window_size}")
        print(f"Shrink rate: {self.shrink_rate}")
        print(f"Window type: {self.window_type}")
        print(f"Number of seeds: {self.num_seeds}")
        print(f"Number of pool datasets: {len(self.pool_datasets)}")
        for idx, pool_ds in enumerate(self.pool_datasets):
            print(f"  Pool {idx + 1}: {len(pool_ds)} samples")
        print(f"Total pool samples: {sum(len(pool_ds) for pool_ds in self.pool_datasets)}")
        print(f"Test samples: {len(self.test_dataset)}")
        
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
            avg_total_samples_seen = np.mean([r['total_samples_seen'] for r in iter_results])
            avg_samples_added = np.mean([r['samples_added'] for r in iter_results])
            avg_total_mse = np.mean([r['total_mse'] for r in iter_results])
            std_total_mse = np.std([r['total_mse'] for r in iter_results]) / np.sqrt(len(iter_results))
            
            # Average MSE per output
            mse_per_output_array = np.array([r['mse_per_output'] for r in iter_results])
            avg_mse_per_output = np.mean(mse_per_output_array, axis=0)
            std_mse_per_output = np.std(mse_per_output_array, axis=0) / np.sqrt(len(iter_results))
            
            # Average training loss (if available)
            avg_train_loss = None
            if iter_results[0]['avg_train_loss'] is not None:
                avg_train_loss = float(np.mean([r['avg_train_loss'] for r in iter_results]))
            
            # Calculate total training samples (samples * epochs)
            # For iteration 0 (untrained), training samples = 0
            # For other iterations, training samples = samples_added * n_epochs
            training_samples = 0 if iter_idx == 0 else float(avg_samples_added * self.n_epochs)
            
            aggregated_results.append({
                'iteration': iter_idx,
                'total_samples_seen': float(avg_total_samples_seen),
                'samples_added': float(avg_samples_added),
                'training_samples': training_samples,  # NEW: Total training samples
                'mean_total_mse': float(avg_total_mse),
                'std_total_mse': float(std_total_mse),
                'mean_mse_per_output': avg_mse_per_output.tolist(),
                'std_mse_per_output': std_mse_per_output.tolist(),
                'mean_train_loss': avg_train_loss,
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
        print(f"{'Iter':<6} {'Samples':<12} {'Added':<8} {'Train#':<12} {'MSE':<15} {'Std Err':<15} {'Train Loss':<15} {'Window':<10}")
        print(f"{'-'*100}")
        
        for result in aggregated_results:
            train_loss_str = f"{result['mean_train_loss']:.6e}" if result['mean_train_loss'] is not None else "N/A"
            print(f"{result['iteration']:<6} "
                  f"{result['total_samples_seen']:<12.0f} "
                  f"{result['samples_added']:<8.0f} "
                  f"{result['training_samples']:<12.0f} "
                  f"{result['mean_total_mse']:<15.6e} "
                  f"{result['std_total_mse']:<15.6e} "
                  f"{train_loss_str:<15} "
                  f"{result['window_size']:<10.4f}")
        
        # Calculate time elapsed
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        self.results['execution_time_seconds'] = elapsed
        print(f"\nExecution time: {elapsed:.2f} seconds")
        print(f"{'='*70}\n")
        
        return self.results
    
    def save_and_return(self, save_results: bool = True, filename: str = None) -> Dict[str, Any]:
        """
        Run pipeline and optionally save results.
        
        Args:
            save_results: Whether to save results to JSON
            filename: Optional custom filename for results
            
        Returns:
            Pipeline results dictionary
        """
        results = self.run()
        
        if save_results:
            # Save full results to JSON (includes aggregated_results with training_samples)
            json_filepath = self.save_results(filename=filename)
            print(f"✓ Results saved to: {json_filepath}\n")
        
        return results
