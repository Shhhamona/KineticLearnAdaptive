"""
Window-based sampling strategy for adaptive learning.

This sampler selects points that fall within a bounding window around
a center point, useful for focused adaptive sampling.
"""

import numpy as np
from typing import Optional, List, Tuple
from .base import BaseSampler
from kinetic_modelling.data import MultiPressureDataset


class WindowSampler(BaseSampler):
    """
    Window-based sampling strategy.
    
    Samples points from a dataset that fall within a bounding box
    around a specified center point. Useful for adaptive learning
    where you want to focus sampling in a specific region.
    
    Example:
        ```python
        # Create sampler
        sampler = WindowSampler(
            center_point=np.array([[0.5, 0.5, 0.5]]),
            window_size=0.3,
            window_type='output'
        )
        
        # Sample points within window
        subset = sampler.sample(dataset, n_samples=50)
        ```
    """
    
    def __init__(
        self,
        center_point: np.ndarray,
        window_size: float,
        window_type: str = 'output',
        sampler_name: str = "window_sampler"
    ):
        """
        Initialize window sampler.
        
        Args:
            center_point: Center of the window (1, n_features)
            window_size: Window size as fraction of center value (e.g., 0.3 = ±30%)
            window_type: 'output' (apply to y) or 'input' (apply to x)
            sampler_name: Name for the sampler
        """
        super().__init__(sampler_name)
        self.center_point = center_point
        self.window_size = window_size
        self.window_type = window_type
        
        # Calculate bounds once during initialization
        self.bounds = self._calculate_bounds()
    
    def _calculate_bounds(self) -> List[Tuple[float, float]]:
        """
        Calculate bounding box around center point.
        
        Window size interpretation:
        - window_size = 0: bounds = [val, val] (exact value)
        - window_size = 0.3: bounds = [val/1.3, val*1.3] (±30% multiplicative)
        - window_size = 1: bounds = [val/2, val*2] (half to double)
        
        Returns:
            List of (min, max) tuples for each dimension
        """
        bounds = []
        
        for i in range(self.center_point.shape[1]):
            val = self.center_point[0, i]
            
            # Define bounds as multiplicative factor
            # val_min = val / (1 + window_size)
            # val_max = val * (1 + window_size)
            factor = 1.0 + self.window_size
            val_min = val / factor
            val_max = val * factor
            
            # Ensure bounds are within [0, 1] since data is scaled
            val_min = max(0.0, val_min)
            val_max = min(1.0, val_max)
            
            bounds.append((val_min, val_max))
        
        return bounds
    
    def sample(
        self,
        dataset: MultiPressureDataset,
        n_samples: int,
        shuffle: bool = False,
        seed: Optional[int] = None,
        exclude_indices: Optional[set] = None
    ) -> MultiPressureDataset:
        """
        Sample points from dataset that fall within the window.
        
        Args:
            dataset: Dataset to sample from
            n_samples: Maximum number of samples to select
            shuffle: Whether to shuffle before sampling (applied first)
            seed: Random seed for reproducibility
            exclude_indices: Set of indices to exclude from sampling
            
        Returns:
            New dataset with sampled points
        """
        # Get data from dataset
        x_data, y_data = dataset.get_data()
        
        # Shuffle first if requested
        if shuffle:
            x_data, y_data = self._shuffle_dataset(x_data, y_data, seed)

        print("self.window_type", self.window_type)

        print("x_data", x_data)
        print("y_data", y_data)
        
        # Determine which data to check against bounds
        if self.window_type == 'output':
            check_data = y_data
        else:  # 'input'
            check_data = x_data
        
        # Find samples within bounds
        if exclude_indices is None:
            exclude_indices = set()
        
        available_indices = []
        
        for idx in range(len(check_data)):
            if idx in exclude_indices:
                continue
            
            # Check if sample is within all bounds
            within_bounds = True
            for dim_idx, (dim_min, dim_max) in enumerate(self.bounds):
                if not (dim_min <= check_data[idx, dim_idx] <= dim_max):
                    within_bounds = False
                    break
            
            if within_bounds:
                available_indices.append(idx)
        
        # Check if we found any samples
        if len(available_indices) == 0:
            raise ValueError(
                f"No samples found within window bounds. "
                f"Window size: {self.window_size}, "
                f"Center: {self.center_point.flatten()}"
            )
        
        # Select up to n_samples from available
        n_to_select = min(n_samples, len(available_indices))
        selected_indices = available_indices[:n_to_select]
        
        sampled_x = x_data[selected_indices]
        sampled_y = y_data[selected_indices]
        
        # Create and return new dataset
        return self._create_sampled_dataset(dataset, sampled_x, sampled_y)
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        Get the current window bounds.
        
        Returns:
            List of (min, max) tuples for each dimension
        """
        return self.bounds
    
    def update_window(self, new_center: np.ndarray, new_window_size: float):
        """
        Update the window with a new center and size.
        
        Args:
            new_center: New center point (1, n_features)
            new_window_size: New window size
        """
        self.center_point = new_center
        self.window_size = new_window_size
        self.bounds = self._calculate_bounds()
    
    def __repr__(self):
        """String representation."""
        return (f"WindowSampler(center={self.center_point.flatten()}, "
                f"window_size={self.window_size}, type='{self.window_type}')")
