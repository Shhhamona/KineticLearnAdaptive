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
        use_model_prediction: bool = True,
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
            use_model_prediction: If True, use model's prediction as center; if False, use test dataset average
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
        self.use_model_prediction = use_model_prediction
        
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
            'use_model_prediction': use_model_prediction,
            'num_pool_datasets': len(pool_datasets),
            'pool_samples_per_dataset': [len(pool_ds) for pool_ds in pool_datasets],
            'total_pool_samples': total_pool_samples,
            'test_samples': len(test_dataset)
        }
    
    def _calculate_center_point(
        self, 
        dataset: MultiPressureDataset,
        model: Optional[BaseModel] = None
    ) -> np.ndarray:
        """
        Calculate the center point for window sampling.
        
        If use_model_prediction is True:
            - With model: center = mean(model.predict(X_test)) - use predicted K values
            - Without model: center = mean(Y_test) - use true K values from test dataset
            
        If use_model_prediction is False:
            - For window_type='output': center = mean(Y_test)
            - For window_type='input': center = mean(X_test)
        
        Args:
            dataset: Dataset to calculate center from
            model: Optional model for predictions (used if use_model_prediction=True)
            
        Returns:
            center: Center point (1, n_features)
        """
        x_data, y_data = dataset.get_data()
        
        if self.use_model_prediction:
            if model is not None:
                # Use model's prediction as center (predicted K values)
                y_pred = model.predict(x_data)
                center = np.mean(y_pred, axis=0, keepdims=True)
            else:
                # No model yet, use actual K values from test dataset
                center = np.mean(y_data, axis=0, keepdims=True)
        else:
            # Original behavior based on window_type
            if self.window_type == 'output':
                center = np.mean(y_data, axis=0, keepdims=True)
            else:  # 'input'
                center = np.mean(x_data, axis=0, keepdims=True)
        
        return center
    
    def _validate_pool_k_range(
        self,
        pool_dataset: MultiPressureDataset,
        center: np.ndarray,
        window_size: float,
        verbose: bool = False
    ) -> bool:
        """
        Validate that the sampling window is fully contained within the pool's K range.
        
        This prevents "cheating" where we would be sampling from a pool that already
        has K values closer to K_true than what the window size suggests.
        
        The validation checks TWO things:
        1. The pool's range must be at least as wide as the sampling window
        2. The actual sampling interval [center/factor, center*factor] must be 
           contained within the pool's bounds
        
        Args:
            pool_dataset: Pool dataset to validate
            center: Center point for sampling (K_predicted or K_true values)
            window_size: Relative window size (e.g., 0.3 means ±30%)
            verbose: If True, print detailed validation information
            
        Returns:
            True if pool is valid (sampling range ⊆ pool range), False otherwise
        """
        # Check if pool has K range metadata
        if not hasattr(pool_dataset, 'k_range_factor'):
            # No metadata - allow sampling (backward compatibility)
            return True
        
        pool_k_factor = pool_dataset.k_range_factor
        pool_label = getattr(pool_dataset, 'label', 'Unknown Pool')
        
        # Calculate the sampling range factor
        # window_size=0.3 means we sample from [center/1.3, center*1.3]
        # window_size=1.0 means we sample from [center/2.0, center*2.0]
        sampling_factor = 1.0 + window_size
        
        if verbose:
            print(f"\n      DEBUG VALIDATION for {pool_label}:")
            print(f"        Pool K factor: {pool_k_factor}")
            print(f"        Sampling factor: {sampling_factor}")
        
        # CHECK 1: Pool range must be at least as wide as sampling range
        # Example: pool_factor=2.0 (K/2 to K*2) can accommodate sampling_factor=1.3
        # Example: pool_factor=1.005 CANNOT accommodate sampling_factor=2.0
        if pool_k_factor < sampling_factor:
            if verbose:
                print(f"        ❌ CHECK 1 FAILED: {pool_k_factor} < {sampling_factor}")
            return False
        
        if verbose:
            print(f"        ✓ CHECK 1 PASSED: {pool_k_factor} >= {sampling_factor}")
        
        # CHECK 2: The actual sampling interval must be contained in pool bounds
        # We need to check if [center/sampling_factor, center*sampling_factor]
        # is contained in [K_true/pool_k_factor, K_true*pool_k_factor]
        #
        # Get K_true from pool metadata (if available)
        if hasattr(pool_dataset, 'k_true_values'):
            k_true = pool_dataset.k_true_values  # Shape: (n_reactions,)
            
            if verbose:
                print(f"        CHECK 2: Verifying sampling interval within pool bounds")
            
            # For each reaction, check if sampling bounds are within pool bounds
            # Sampling bounds: [center_k / sampling_factor, center_k * sampling_factor]
            # Pool bounds: [k_true_k / pool_k_factor, k_true_k * pool_k_factor]
            
            for k_idx in range(len(k_true)):
                center_k = center[0, k_idx]  # center is shape (1, n_reactions)
                k_true_k = k_true[k_idx]
                
                # Sampling interval
                sample_lower = center_k / sampling_factor
                sample_upper = center_k * sampling_factor
                
                # Pool interval
                pool_lower = k_true_k / pool_k_factor
                pool_upper = k_true_k * pool_k_factor
                
                # Check if sampling interval is contained in pool interval
                # Allow small numerical tolerance (1%)
                tolerance = 0.01
                
                if verbose:
                    print(f"          Reaction {k_idx}:")
                    print(f"            Center K: {center_k:.6e}, True K: {k_true_k:.6e}")
                    print(f"            Sampling interval: [{sample_lower:.6e}, {sample_upper:.6e}]")
                    print(f"            Pool interval:     [{pool_lower:.6e}, {pool_upper:.6e}]")
                
                if sample_lower < pool_lower * (1 - tolerance) or sample_upper > pool_upper * (1 + tolerance):
                    # Sampling interval extends beyond pool bounds
                    if verbose:
                        print(f"            ❌ FAILS: Sampling extends beyond pool!")
                        if sample_lower < pool_lower * (1 - tolerance):
                            print(f"               Lower bound: {sample_lower:.6e} < {pool_lower * (1 - tolerance):.6e}")
                        if sample_upper > pool_upper * (1 + tolerance):
                            print(f"               Upper bound: {sample_upper:.6e} > {pool_upper * (1 + tolerance):.6e}")
                    return False
                elif verbose:
                    print(f"            ✓ OK: Sampling within pool bounds")
            
            if verbose:
                print(f"        ✓ CHECK 2 PASSED: All reactions within bounds")
        
        # If we don't have k_true_values, fall back to just checking the factor
        # (This is the original check - less strict but still catches major issues)
        return True
    
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
        n_epochs: int,
        x_test: np.ndarray = None,
        eval_frequency: int = 10
    ) -> tuple:
        """
        Train model for specified number of epochs with optional variance tracking.
        
        Args:
            model: Model to train (must have train_single_batch method)
            train_loader: DataLoader for training
            n_epochs: Number of epochs to train
            x_test: Optional test input for prediction variance tracking
            eval_frequency: How often to evaluate predictions (every N epochs)
            
        Returns:
            Tuple of (epoch_losses, prediction_variance) where variance is per output or None
        """
        epoch_losses = []
        
        # Variance tracking setup (same as batch_training.py)
        prediction_window_size = 5  # Track variance over last 5 evaluations
        recent_predictions = []  # Stores mean predictions from recent evaluations
        prediction_variance = None
        n_outputs = None
        
        if x_test is not None:
            # Get number of outputs from a test prediction
            y_pred_test = model.predict(x_test)
            n_outputs = y_pred_test.shape[1]
        
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
            
            # Evaluate predictions for variance tracking
            if x_test is not None and (epoch + 1) % eval_frequency == 0:
                y_pred = model.predict(x_test)
                mean_prediction = np.mean(y_pred, axis=0)  # Mean per output
                recent_predictions.append(mean_prediction)
                
                # Keep only the last prediction_window_size predictions
                if len(recent_predictions) > prediction_window_size:
                    recent_predictions = recent_predictions[-prediction_window_size:]
                
                # Calculate variance if we have enough predictions
                if len(recent_predictions) >= prediction_window_size:
                    recent_preds_array = np.array(recent_predictions[-prediction_window_size:])
                    prediction_variance = np.var(recent_preds_array, axis=0)
        
        return epoch_losses, prediction_variance
    
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
        
        # Create a new model for this seed
        print(f"  Initializing Neural Network model...")
        model = self.model_class(**self.model_params)
        
        seed_results = []
        
        # Iteration 0: Evaluate untrained model
        print(f"\n  Iteration 0 - Untrained Model")
        
        # Calculate initial center point (always use test dataset for iteration 0)
        # This is because untrained model predictions are random
        center = self._calculate_center_point(self.test_dataset, model=None)
        print(f"  Initial center from test dataset: {center.flatten()}")
        
        y_pred = model.predict(x_test)
        mse_per_output = []
        for i in range(n_outputs):
            mse = np.mean((y_test[:, i] - y_pred[:, i]) ** 2)
            mse_per_output.append(mse)
        total_mse = np.sum(mse_per_output)
        
        # No training yet, so no variance to report
        prediction_variance = np.full(n_outputs, np.nan)
        
        seed_results.append({
            'iteration': 0,
            'seed': seed,
            'total_samples_seen': 0,
            'samples_added': 0,
            'mse_per_output': [float(m) for m in mse_per_output],
            'total_mse': float(total_mse),
            'window_size': float(self.initial_window_size),
            'center_point': [float(c) for c in center.flatten()],
            'pools_used': [],  # No pools used for untrained model
            'avg_train_loss': None,
            'prediction_variance': [float(v) for v in prediction_variance]
        })
        
        print(f"    Test MSE (untrained): {total_mse:.6e}")
        
        # Adaptive iterations - sample from all valid pools
        total_samples_seen = 0
        pool_used_indices = [set() for _ in range(len(self.pool_datasets))]  # Track used indices per pool
        
        for iteration in range(1, self.n_iterations + 1):
            print(f"\n  Iteration {iteration}")
            
            # Recalculate center point
            # For iteration 1, use test dataset center (using first pool with broad range)
            # For iteration 2+, use model predictions if enabled
            if iteration == 1:
                center = self._calculate_center_point(self.test_dataset, model=None)
                print(f"    Center from test dataset: {center.flatten()}")
            else:
                center = self._calculate_center_point(self.test_dataset, model)
                if self.use_model_prediction:
                    print(f"    Updated center from model prediction: {center.flatten()}")
                else:
                    print(f"    Center from test dataset: {center.flatten()}")
            
            # Calculate adaptive window size
            window_size = self.initial_window_size * (self.shrink_rate ** (iteration - 1))
            
            print(f"    Window size: {window_size:.4f}")
            
            # Validate all pools and identify which ones can be used for this window
            valid_pools = []
            for pool_idx, pool_ds in enumerate(self.pool_datasets):
                if self._validate_pool_k_range(pool_ds, center, window_size, verbose=True):
                    # Check if pool has unused samples
                    unused_count = len(pool_ds) - len(pool_used_indices[pool_idx])
                    if unused_count > 0:
                        valid_pools.append((pool_idx, pool_ds, unused_count))
            
            if len(valid_pools) == 0:
                print(f"    ❌ No valid pools available for window size {window_size:.4f}")
                print(f"    All pools either violate K range constraints or are exhausted.")
                break
            
            print(f"    ✓ Found {len(valid_pools)} valid pool(s) for this window:")
            for pool_idx, pool_ds, unused_count in valid_pools:
                pool_label = getattr(pool_ds, 'label', f'Pool {pool_idx + 1}')
                pool_k_factor = getattr(pool_ds, 'k_range_factor', 'unknown')
                print(f"      • Pool {pool_idx + 1}: {unused_count} unused samples, K factor={pool_k_factor}")
            
            # Collect samples - try all valid pools until we get enough samples
            all_sampled_x = []
            all_sampled_y = []
            samples_needed = self.samples_per_iteration
            pools_used_this_iteration = []
            
            for pool_idx, current_pool, unused_count in valid_pools:
                if samples_needed <= 0:
                    break
                
                pool_label = getattr(current_pool, 'label', f'Pool {pool_idx + 1}')
                
                # Create window sampler with current window size
                window_sampler = WindowSampler(
                    center_point=center,
                    window_size=window_size,
                    window_type=self.window_type,
                    sampler_name=f"window_iter_{iteration}_seed_{seed}_pool_{pool_idx}"
                )
                
                try:
                    print(f"    Attempting to sample {samples_needed} from Pool {pool_idx + 1} ({pool_label})...")
                    
                    # Use pool-specific used indices
                    used_indices = pool_used_indices[pool_idx]
                    
                    sampled_subset = window_sampler.sample(
                        current_pool,
                        n_samples=samples_needed,
                        shuffle=True,
                        seed=seed + pool_idx,  # Different seed for each pool
                        exclude_indices=used_indices if len(used_indices) > 0 else None
                    )
                    
                    sampled_x, sampled_y = sampled_subset.get_data()
                    samples_obtained = len(sampled_x)
                    
                    # Track which indices were used
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
                        pools_used_this_iteration.append(pool_idx)
                        
                        print(f"    ✓ Got {samples_obtained} samples from Pool {pool_idx + 1}")
                        print(f"       Total used from this pool: {len(used_indices)}/{len(current_pool)}")
                        
                        if samples_needed > 0:
                            print(f"       Still need {samples_needed} more samples...")
                
                except ValueError as e:
                    # Current pool exhausted or no samples in window
                    print(f"    ⚠️  Pool {pool_idx + 1}: {str(e)}")
            
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
            epoch_losses, prediction_variance = self._train_for_epochs(
                model, train_loader, self.n_epochs, x_test=x_test, eval_frequency=10
            )
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
            
            # Variance is calculated during training (or None if not enough evaluations)
            if prediction_variance is None:
                prediction_variance = np.full(n_outputs, np.nan)
            else:
                print(f"    Prediction variance (last 5 evals): {prediction_variance}")
            
            seed_results.append({
                'iteration': iteration,
                'seed': seed,
                'total_samples_seen': total_samples_seen,
                'samples_added': samples_added,
                'mse_per_output': [float(m) for m in mse_per_output],
                'total_mse': float(total_mse),
                'window_size': float(window_size),
                'center_point': [float(c) for c in center.flatten()],
                'pools_used': pools_used_this_iteration,  # List of pool indices used
                'avg_train_loss': float(avg_train_loss),
                'epoch_losses': [float(loss) for loss in epoch_losses],
                'prediction_variance': [float(v) for v in prediction_variance]
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
            
            # Average prediction variance per output (handle NaN values)
            prediction_variance_array = np.array([r['prediction_variance'] for r in iter_results])
            avg_prediction_variance = np.nanmean(prediction_variance_array, axis=0)
            std_prediction_variance = np.nanstd(prediction_variance_array, axis=0)
            
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
                'num_seeds': len(iter_results),
                'mean_prediction_variance_per_output': avg_prediction_variance.tolist(),
                'std_prediction_variance_per_output': std_prediction_variance.tolist()
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
