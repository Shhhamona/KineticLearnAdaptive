"""
Sampling strategies for generating K values.

This module implements various sampling strategies for generating rate coefficient
values, including Latin Hypercube, random sampling, and adaptive sampling.
"""

import numpy as np
from scipy.stats import qmc
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod


class BaseSampler(ABC):
    """Abstract base class for sampling strategies."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize sampler.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    @abstractmethod
    def sample(self, center: np.ndarray, bounds: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Generate samples around a center point.
        
        Args:
            center: Center point for sampling, shape (n_k,)
            bounds: Bounds for sampling, shape (n_k, 2) with [min, max] for each K
            n_samples: Number of samples to generate
            
        Returns:
            Sampled K values, shape (n_samples, n_k)
        """
        pass


class RandomSampler(BaseSampler):
    """Random uniform sampling within bounds."""
    
    def sample(self, center: np.ndarray, bounds: np.ndarray, n_samples: int) -> np.ndarray:
        """Generate random uniform samples within bounds."""
        n_k = len(center)
        samples = np.zeros((n_samples, n_k))

        np.random.seed(self.random_state)
        
        for i in range(n_k):
            samples[:, i] = np.random.uniform(bounds[i, 0], bounds[i, 1], n_samples)
        
        return samples


class LatinHypercubeSampler(BaseSampler):
    """Latin Hypercube sampling for better space coverage."""
    
    def sample(self, center: np.ndarray, bounds: np.ndarray, n_samples: int) -> np.ndarray:
        """Generate Latin Hypercube samples within bounds."""
        n_k = len(center)
        
        # Create Latin Hypercube sampler
        sampler = qmc.LatinHypercube(d=n_k, seed=self.random_state)
        unit_samples = sampler.random(n=n_samples)
        
        # Transform to actual bounds
        samples = np.zeros((n_samples, n_k))
        for i in range(n_k):
            samples[:, i] = bounds[i, 0] + unit_samples[:, i] * (bounds[i, 1] - bounds[i, 0])
        
        return samples


class HypercubeSampler(BaseSampler):
    """Hypercube sampling around a center point."""
    
    def __init__(self, sampling_method: str = 'latin_hypercube', random_state: int = 42):
        """
        Initialize hypercube sampler.
        
        Args:
            sampling_method: 'random' or 'latin_hypercube'
            random_state: Random seed
        """
        super().__init__(random_state)
        self.sampling_method = sampling_method
        
        if sampling_method == 'latin_hypercube':
            self.base_sampler = LatinHypercubeSampler(random_state)
        elif sampling_method == 'random':
            self.base_sampler = RandomSampler(random_state)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    def sample(self, center: np.ndarray, hypercube_size: float, n_samples: int) -> np.ndarray:
        """
        Sample within a hypercube around center point.
        
        Args:
            center: Center point, shape (n_k,)
            hypercube_size: Size of hypercube as fraction of center values
            n_samples: Number of samples
            
        Returns:
            Sampled K values, shape (n_samples, n_k)
        """
        # Create bounds based on hypercube size
        k_min = center * (1 - hypercube_size)
        k_max = center * (1 + hypercube_size)
        
        # Ensure positive values (K coefficients should be positive)
        k_min = np.maximum(k_min, 1e-50)
        
        bounds = np.column_stack([k_min, k_max])
        
        return self.base_sampler.sample(center, bounds, n_samples)


class AdaptiveSampler:
    """
    Adaptive sampling strategy that iteratively refines the sampling region.
    """
    
    def __init__(self, 
                 initial_hypercube_size: float = 0.5,
                 hypercube_reduction: float = 0.8,
                 sampling_method: str = 'latin_hypercube',
                 random_state: int = 42):
        """
        Initialize adaptive sampler.
        
        Args:
            initial_hypercube_size: Initial size of hypercube as fraction
            hypercube_reduction: Factor to reduce hypercube size each iteration
            sampling_method: Base sampling method to use
            random_state: Random seed
        """
        self.initial_hypercube_size = initial_hypercube_size
        self.hypercube_reduction = hypercube_reduction
        self.current_hypercube_size = initial_hypercube_size
        
        self.sampler = HypercubeSampler(sampling_method, random_state)
        
        # History tracking
        self.iteration = 0
        self.hypercube_sizes = [initial_hypercube_size]
        self.center_history = []
    
    def initial_sample(self, center: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Generate initial samples around starting point.
        
        Args:
            center: Initial center point (e.g., literature values)
            n_samples: Number of initial samples
            
        Returns:
            Initial K samples, shape (n_samples, n_k)
        """
        self.current_hypercube_size = self.initial_hypercube_size
        self.center_history.append(center.copy())
        
        return self.sampler.sample(center, self.current_hypercube_size, n_samples)
    
    def adaptive_sample(self, new_center: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Generate samples for next iteration around new center.
        
        Args:
            new_center: New center point (predicted K values)
            n_samples: Number of samples for this iteration
            
        Returns:
            New K samples, shape (n_samples, n_k)
        """
        # Reduce hypercube size
        self.current_hypercube_size *= self.hypercube_reduction
        self.iteration += 1
        
        # Track history
        self.hypercube_sizes.append(self.current_hypercube_size)
        self.center_history.append(new_center.copy())
        
        return self.sampler.sample(new_center, self.current_hypercube_size, n_samples)
    
    def get_history(self) -> dict:
        """Get sampling history for analysis."""
        return {
            'iteration': self.iteration,
            'hypercube_sizes': self.hypercube_sizes.copy(),
            'center_history': [c.copy() for c in self.center_history]
        }


class BoundsBasedSampler(BaseSampler):
    """
    Sampler that works with explicit bounds for each parameter.
    Useful when you want to constrain K values to physically meaningful ranges.
    """
    
    def __init__(self, k_bounds: np.ndarray, sampling_method: str = 'latin_hypercube', 
                 random_state: int = 42):
        """
        Initialize bounds-based sampler.
        
        Args:
            k_bounds: Bounds for each K, shape (n_k, 2) with [min, max]
            sampling_method: 'random' or 'latin_hypercube'
            random_state: Random seed
        """
        super().__init__(random_state)
        self.k_bounds = k_bounds
        
        if sampling_method == 'latin_hypercube':
            self.base_sampler = LatinHypercubeSampler(random_state)
        elif sampling_method == 'random':
            self.base_sampler = RandomSampler(random_state)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    def sample(self, center: np.ndarray, bounds: np.ndarray, n_samples: int) -> np.ndarray:
        """Sample within the predefined bounds (center parameter ignored)."""
        return self.base_sampler.sample(center, self.k_bounds, n_samples)
    
    def sample_full_space(self, n_samples: int) -> np.ndarray:
        """Sample across the full parameter space."""
        dummy_center = np.mean(self.k_bounds, axis=1)  # Not used but needed for interface
        return self.sample(dummy_center, self.k_bounds, n_samples)
