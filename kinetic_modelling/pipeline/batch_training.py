"""
Batch Training Pipeline for neural networks.

This pipeline:
1. Trains neural networks with mini-batch gradient descent
2. Tracks metrics after every batch update
3. Runs with multiple seeds for robustness
4. Aggregates results across seeds (mean ± std)
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from .base import BasePipeline
from kinetic_modelling.data import MultiPressureDataset
from kinetic_modelling.model.base import BaseModel
from kinetic_modelling.sampling import SequentialSampler


class BatchTrainingPipeline(BasePipeline):
    """
    Pipeline for batch training experiments with neural networks.
    
    Trains models using mini-batch gradient descent and tracks metrics
    after every batch to understand model evolution during training.
    
    Example:
        ```python
        pipeline = BatchTrainingPipeline(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model_class=NeuralNetModel,
            model_params={'hidden_layers': [64, 32], 'learning_rate': 0.001},
            batch_size=64,
            num_epochs=1,
            num_seeds=10,
            pipeline_name="batch_training_experiment"
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
        batch_size: int = 64,
        num_epochs: int = 1,
        num_seeds: int = 10,
        eval_frequency: int = 1,
        pipeline_name: str = "batch_training_pipeline",
        results_dir: str = "pipeline_results"
    ):
        """
        Initialize batch training pipeline.
        
        Args:
            train_dataset: Full training dataset
            test_dataset: Test dataset
            model_class: Model class (e.g., NeuralNetModel)
            model_params: Parameters to pass to model constructor
            batch_size: Batch size for training
            num_epochs: Number of epochs to train
            num_seeds: Number of random seeds for robustness
            eval_frequency: Evaluate every N batches (1 = every batch, 10 = every 10 batches)
            pipeline_name: Name for this pipeline
            results_dir: Directory to save results
        """
        super().__init__(pipeline_name, results_dir)
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model_class = model_class
        self.model_params = model_params
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_seeds = num_seeds
        self.eval_frequency = eval_frequency
        
        # Store configuration
        self.results['config'] = {
            'model_type': model_class.__name__,
            'model_params': str(model_params),
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'num_seeds': num_seeds,
            'eval_frequency': eval_frequency,
            'total_train_samples': len(train_dataset),
            'test_samples': len(test_dataset)
        }
    
    def _create_dataloader(
        self,
        dataset: MultiPressureDataset,
        batch_size: int,
        shuffle: bool = False
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader from a MultiPressureDataset.
        
        Args:
            dataset: The dataset to create loader from
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            PyTorch DataLoader
        """
        x_data, y_data = dataset.get_data()
        
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
    
    def _train_and_track(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        x_test: np.ndarray,
        y_test: np.ndarray,
        n_outputs: int
    ) -> Dict[str, List]:
        """
        Train model and track metrics after each batch.
        
        Uses the model's train_single_batch() method for batch updates.
        
        Args:
            model: Model to train (must have train_single_batch method)
            train_loader: DataLoader for training
            x_test: Test input features
            y_test: Test output targets
            n_outputs: Number of output features
            
        Returns:
            Dictionary with batch-wise metrics
        """
        batch_metrics = {
            'batch_numbers': [],
            'train_loss': [],
            'total_mse': [],
            'mse_per_output': []
        }
        
        batch_number = 0
        
        for epoch in range(self.num_epochs):
            for batch_x, batch_y in train_loader:
                # Convert to numpy for model compatibility
                batch_x_np = batch_x.numpy()
                batch_y_np = batch_y.numpy()
                
                # Perform single batch gradient update using model's method
                train_loss = model.train_single_batch(batch_x_np, batch_y_np)
                
                # Evaluate on test set only at specified frequency
                if batch_number % self.eval_frequency == 0:
                    y_pred = model.predict(x_test)
                    
                    # Compute MSE per output
                    mse_per_output = []
                    for i in range(n_outputs):
                        mse = np.mean((y_test[:, i] - y_pred[:, i]) ** 2)
                        mse_per_output.append(mse)
                    
                    total_mse = np.sum(mse_per_output)
                    
                    # Store metrics
                    batch_metrics['batch_numbers'].append(batch_number)
                    batch_metrics['train_loss'].append(train_loss)
                    batch_metrics['total_mse'].append(total_mse)
                    batch_metrics['mse_per_output'].append(mse_per_output)
                
                batch_number += 1
        
        return batch_metrics
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the batch training pipeline.
        
        For each seed:
          1. Shuffle training data
          2. Create data loader with specified batch size
          3. Train model batch by batch
          4. Evaluate on test set after each batch
        
        Then aggregate results across seeds (mean ± std).
        
        Returns:
            Dictionary with aggregated results
        """
        print(f"\n{'='*70}")
        print(f"Running Batch Training Pipeline: {self.pipeline_name}")
        print(f"{'='*70}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Evaluation frequency: every {self.eval_frequency} batch(es)")
        print(f"Number of seeds: {self.num_seeds}")
        print(f"Total training samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        
        # Record start time
        start_time = datetime.now()
        self.results['timestamp'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Get test data once
        x_test, y_test = self.test_dataset.get_data()
        n_outputs = y_test.shape[1]
        
        # Calculate expected number of batches per epoch
        num_batches = int(np.ceil(len(self.train_dataset) / self.batch_size))
        total_batches = num_batches * self.num_epochs
        print(f"Batches per epoch: {num_batches}")
        print(f"Total batches: {total_batches}")
        
        # Store results for each seed
        all_seeds_results = []
        
        # Loop over seeds
        for seed_idx in range(self.num_seeds):
            print(f"\n{'='*70}")
            print(f"Seed {seed_idx + 1}/{self.num_seeds} (seed={seed_idx})")
            print(f"{'='*70}")
            
            # Shuffle training data with this seed
            print(f"  Shuffling training data with seed={seed_idx}...")
            shuffle_sampler = SequentialSampler(sampler_name=f"shuffle_seed_{seed_idx}")
            
            # Sample all data with shuffle=True
            shuffled_dataset = shuffle_sampler.sample(
                self.train_dataset,
                n_samples=len(self.train_dataset),
                shuffle=True,
                seed=seed_idx
            )
            
            # Create data loader (no additional shuffle since data is already shuffled)
            train_loader = self._create_dataloader(
                shuffled_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
            
            # Create and train model
            print(f"  Training model with batch size {self.batch_size}...")
            model = self.model_class(**self.model_params)
            
            # Train and track metrics
            batch_metrics = self._train_and_track(
                model,
                train_loader,
                x_test,
                y_test,
                n_outputs
            )
            
            all_seeds_results.append(batch_metrics)
            
            print(f"  Final MSE: {batch_metrics['total_mse'][-1]:.6e}")
        
        # Aggregate results across seeds
        print(f"\n{'='*70}")
        print(f"Aggregating results across {self.num_seeds} seeds...")
        print(f"{'='*70}")
        
        # All seeds should have the same number of batches
        batch_numbers = all_seeds_results[0]['batch_numbers']
        num_tracked_batches = len(batch_numbers)
        
        # Convert to numpy arrays for easier manipulation
        # Shape: (num_seeds, num_batches)
        train_loss_array = np.array([seed_results['train_loss'] for seed_results in all_seeds_results])
        total_mse_array = np.array([seed_results['total_mse'] for seed_results in all_seeds_results])
        
        # Shape: (num_seeds, num_batches, n_outputs)
        mse_per_output_array = np.array([seed_results['mse_per_output'] for seed_results in all_seeds_results])
        
        # Compute mean and std across seeds
        mean_train_loss = np.mean(train_loss_array, axis=0)
        std_train_loss = np.std(train_loss_array, axis=0) / np.sqrt(self.num_seeds)
        
        mean_total_mse = np.mean(total_mse_array, axis=0)
        std_total_mse = np.std(total_mse_array, axis=0) / np.sqrt(self.num_seeds)  # Standard error
        
        mean_mse_per_output = np.mean(mse_per_output_array, axis=0)
        std_mse_per_output = np.std(mse_per_output_array, axis=0) / np.sqrt(self.num_seeds)
        
        # Store results
        self.results['raw_results'] = {
            'train_loss_per_seed': train_loss_array.tolist(),
            'total_mse_per_seed': total_mse_array.tolist(),
            'mse_per_output_per_seed': mse_per_output_array.tolist()
        }
        
        self.results['aggregated_results'] = {
            'batch_numbers': batch_numbers,
            'mean_train_loss': mean_train_loss.tolist(),
            'std_train_loss': std_train_loss.tolist(),
            'mean_total_mse': mean_total_mse.tolist(),
            'std_total_mse': std_total_mse.tolist(),
            'mean_mse_per_output': mean_mse_per_output.tolist(),
            'std_mse_per_output': std_mse_per_output.tolist(),
            'initial_mean_mse': float(mean_total_mse[0]),
            'initial_std_mse': float(std_total_mse[0]),
            'final_mean_mse': float(mean_total_mse[-1]),
            'final_std_mse': float(std_total_mse[-1])
        }
        
        # Print summary (show first, middle, and last batches)
        print(f"\n{'='*70}")
        print(f"Aggregated Results Summary (Batch-wise)")
        print(f"{'='*70}")
        print(f"{'Batch':<10} {'Mean Total MSE':<20} {'Std Error':<20}")
        print(f"{'-'*70}")
        
        # Show first 3, middle 3, and last 3 batches
        indices_to_show = []
        if num_tracked_batches <= 10:
            indices_to_show = list(range(num_tracked_batches))
        else:
            indices_to_show = (
                [0, 1, 2] +
                [num_tracked_batches // 2 - 1, num_tracked_batches // 2, num_tracked_batches // 2 + 1] +
                [num_tracked_batches - 3, num_tracked_batches - 2, num_tracked_batches - 1]
            )
        
        prev_idx = -2
        for idx in indices_to_show:
            if idx - prev_idx > 1:
                print(f"{'...':<10}")
            print(f"{batch_numbers[idx]:<10} {mean_total_mse[idx]:<20.6e} {std_total_mse[idx]:<20.6e}")
            prev_idx = idx
        
        print(f"\nImprovement: {mean_total_mse[0]:.6e} → {mean_total_mse[-1]:.6e}")
        print(f"Reduction factor: {mean_total_mse[0] / mean_total_mse[-1]:.2f}x")
        
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
