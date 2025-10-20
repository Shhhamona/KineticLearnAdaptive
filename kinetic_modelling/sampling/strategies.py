"""
Sampling strategies for kinetic modeling.

This module provides concrete implementations of various sampling strategies.
"""

import numpy as np
from typing import Optional
from .base import BaseSampler
from kinetic_modelling.data import MultiPressureDataset


class SequentialSampler(BaseSampler):
    """
    Sequential sampling strategy.
    
    Selects the first N samples from the dataset (optionally after shuffling).
    """
    
    def __init__(self, sampler_name: str = "sequential"):
        """Initialize sequential sampler."""
        super().__init__(sampler_name)
    
    def sample(
        self,
        dataset: MultiPressureDataset,
        n_samples: int,
        shuffle: bool = False,
        seed: Optional[int] = None
    ) -> MultiPressureDataset:
        """
        Sample sequentially from the dataset.
        
        Args:
            dataset: The dataset to sample from
            n_samples: Number of samples to select
            shuffle: Whether to shuffle before taking first N samples
            seed: Random seed for reproducibility (used if shuffle=True)
            
        Returns:
            A new MultiPressureDataset with the first n_samples
        """
        # Get data from dataset
        x_data, y_data = dataset.get_data()
        
        # Check n_samples is valid
        if n_samples > len(dataset):
            raise ValueError(f"Cannot sample {n_samples} from dataset of size {len(dataset)}")
        
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        
        # Shuffle if requested
        if shuffle:
            x_data, y_data = self._shuffle_dataset(x_data, y_data, seed)
        
        # Take first n_samples
        sampled_x = x_data[:n_samples]
        sampled_y = y_data[:n_samples]
        
        # Create and return new dataset
        return self._create_sampled_dataset(dataset, sampled_x, sampled_y)


class RandomSampler(BaseSampler):
    """
    Random sampling strategy.
    
    Randomly selects N samples from the dataset without replacement.
    """
    
    def __init__(self, sampler_name: str = "random"):
        """Initialize random sampler."""
        super().__init__(sampler_name)
    
    def sample(
        self,
        dataset: MultiPressureDataset,
        n_samples: int,
        shuffle: bool = False,
        seed: Optional[int] = None
    ) -> MultiPressureDataset:
        """
        Sample randomly from the dataset.
        
        Args:
            dataset: The dataset to sample from
            n_samples: Number of samples to select
            shuffle: Whether to shuffle before random sampling (applied first if True)
            seed: Random seed for reproducibility
            
        Returns:
            A new MultiPressureDataset with randomly selected samples
        """
        # Get data from dataset
        x_data, y_data = dataset.get_data()
        
        # Check n_samples is valid
        if n_samples > len(dataset):
            raise ValueError(f"Cannot sample {n_samples} from dataset of size {len(dataset)}")
        
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        
        # Shuffle first if requested
        if shuffle:
            x_data, y_data = self._shuffle_dataset(x_data, y_data, seed)
            # Use a different seed for random sampling if shuffle was done
            sample_seed = None if seed is None else seed + 1
        else:
            sample_seed = seed
        
        # Set random seed for sampling
        if sample_seed is not None:
            np.random.seed(sample_seed)
        
        # Random sampling without replacement
        indices = np.random.choice(len(x_data), size=n_samples, replace=False)
        indices = np.sort(indices)  # Sort to maintain some order
        
        sampled_x = x_data[indices]
        sampled_y = y_data[indices]
        
        # Create and return new dataset
        return self._create_sampled_dataset(dataset, sampled_x, sampled_y)


class SubsetSampler(BaseSampler):
    """
    Subset sampling strategy using explicit indices.
    
    Allows selection of specific samples by their indices.
    """
    
    def __init__(self, sampler_name: str = "subset"):
        """Initialize subset sampler."""
        super().__init__(sampler_name)
    
    def sample(
        self,
        dataset: MultiPressureDataset,
        n_samples: int = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        indices: Optional[np.ndarray] = None
    ) -> MultiPressureDataset:
        """
        Sample using specific indices from the dataset.
        
        Args:
            dataset: The dataset to sample from
            n_samples: Not used for this sampler (required by base class signature)
            shuffle: Whether to shuffle before selecting indices
            seed: Random seed for reproducibility
            indices: Array of indices to select (required for this sampler)
            
        Returns:
            A new MultiPressureDataset with samples at specified indices
        """
        if indices is None:
            raise ValueError("SubsetSampler requires 'indices' parameter")
        
        # Get data from dataset
        x_data, y_data = dataset.get_data()
        
        # Validate indices
        if np.any(indices >= len(dataset)) or np.any(indices < 0):
            raise ValueError(f"Invalid indices: must be in range [0, {len(dataset)})")
        
        # Shuffle first if requested
        if shuffle:
            x_data, y_data = self._shuffle_dataset(x_data, y_data, seed)
        
        # Select samples at specified indices
        sampled_x = x_data[indices]
        sampled_y = y_data[indices]
        
        # Create and return new dataset
        return self._create_sampled_dataset(dataset, sampled_x, sampled_y)
