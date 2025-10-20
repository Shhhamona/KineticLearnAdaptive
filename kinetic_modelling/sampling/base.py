"""
Base sampler class for kinetic modeling.

This module provides the abstract base class that all sampling strategies must inherit from.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
import sys
from pathlib import Path

# Import the data module
sys.path.insert(0, str(Path(__file__).parent.parent))
from kinetic_modelling.data import MultiPressureDataset


class BaseSampler(ABC):
    """
    Abstract base class for all sampling strategies.
    
    All samplers take a dataset and return a sampled version of it.
    The sampled dataset maintains the same structure but with fewer samples.
    """
    
    def __init__(self, sampler_name: str = "base_sampler"):
        """
        Initialize the base sampler.
        
        Args:
            sampler_name: Name for the sampler (for logging/debugging)
        """
        self.sampler_name = sampler_name
    
    @abstractmethod
    def sample(
        self,
        dataset: MultiPressureDataset,
        n_samples: int,
        shuffle: bool = False,
        seed: Optional[int] = None
    ) -> MultiPressureDataset:
        """
        Sample from the dataset.
        
        Args:
            dataset: The dataset to sample from
            n_samples: Number of samples to select
            shuffle: Whether to shuffle the dataset before sampling
            seed: Random seed for reproducibility (used for shuffling and random sampling)
            
        Returns:
            A new MultiPressureDataset with the sampled data
        """
        pass
    
    def _shuffle_dataset(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        seed: Optional[int] = None
    ) -> tuple:
        """
        Shuffle the dataset arrays.
        
        Args:
            x_data: Input features
            y_data: Output targets
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (shuffled_x, shuffled_y)
        """
        if seed is not None:
            np.random.seed(seed)
        
        indices = np.arange(len(x_data))
        np.random.shuffle(indices)
        
        return x_data[indices].copy(), y_data[indices].copy()
    
    def _create_sampled_dataset(
        self,
        original_dataset: MultiPressureDataset,
        sampled_x: np.ndarray,
        sampled_y: np.ndarray
    ) -> MultiPressureDataset:
        """
        Create a new dataset from sampled data.
        
        Simply uses the processed array initialization - much simpler and faster!
        
        Args:
            original_dataset: The original dataset to copy metadata from
            sampled_x: Sampled input features (already scaled)
            sampled_y: Sampled output targets (already scaled)
            
        Returns:
            A new MultiPressureDataset with the sampled data
        """
        # Get scalers from original dataset
        input_scalers, output_scalers = original_dataset.get_scalers()
        
        # Create new dataset directly from processed data - no unscaling needed!
        new_dataset = MultiPressureDataset(
            nspecies=original_dataset.nspecies,
            num_pressure_conditions=original_dataset.num_pressure_conditions,
            processed_x=sampled_x,
            processed_y=sampled_y,
            scaler_input=input_scalers,
            scaler_output=output_scalers
        )
        
        return new_dataset
    
    def __repr__(self):
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.sampler_name}')"
