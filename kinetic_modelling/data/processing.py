"""
Data preprocessing utilities for kinetic modeling.

This module provides functions for applying scalers and loading datasets
with proper scaler management.
"""

import numpy as np
from typing import Tuple, Optional
from .data import MultiPressureDataset


def apply_training_scalers(
    raw_compositions: np.ndarray,
    raw_k_values: np.ndarray,
    dataset_train: MultiPressureDataset,
    nspecies: int,
    num_pressure_conditions: int,
    debug: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply training dataset scalers to new simulation data.
    
    This is a convenience function that wraps MultiPressureDataset initialization
    from raw arrays. For more flexibility, consider using MultiPressureDataset directly.
    
    Args:
        raw_compositions: Raw density values, shape (n_sims * num_pressure_conditions, nspecies)
        raw_k_values: Raw reaction rates, shape (n_sims, n_k) - NOT yet scaled by 1e30
        dataset_train: Training dataset with fitted scalers
        nspecies: Number of chemical species
        num_pressure_conditions: Number of pressure conditions
        debug: Whether to print debug information (currently not used)
        
    Returns:
        new_x: Scaled and flattened input features, shape (n_sims, num_pressure_conditions * nspecies)
        new_y_scaled: Scaled output targets, shape (n_sims, n_k)
    """
    # Get scalers from training dataset
    input_scalers, output_scalers = dataset_train.get_scalers()
    
    # Create new dataset using array initialization
    new_dataset = MultiPressureDataset(
        nspecies=nspecies,
        num_pressure_conditions=num_pressure_conditions,
        raw_compositions=raw_compositions,
        raw_k_values=raw_k_values,
        scaler_input=input_scalers,
        scaler_output=output_scalers
    )
    
    return new_dataset.get_data()


def load_datasets(
    train_file: str,
    test_file: str,
    nspecies: int,
    num_pressure_conditions: int,
    react_idx: Optional[np.ndarray] = None
) -> Tuple[MultiPressureDataset, MultiPressureDataset]:
    """
    Load training and test datasets with proper scaler reuse.
    
    Args:
        train_file: Path to training data file
        test_file: Path to test data file
        nspecies: Number of chemical species
        num_pressure_conditions: Number of pressure conditions
        react_idx: Indices of reaction rate columns (None = all)
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Load training data and fit scalers
    train_dataset = MultiPressureDataset(
        src_file=train_file,
        nspecies=nspecies,
        num_pressure_conditions=num_pressure_conditions,
        react_idx=react_idx
    )
    
    # Load test data with training scalers
    input_scalers, output_scalers = train_dataset.get_scalers()
    test_dataset = MultiPressureDataset(
        src_file=test_file,
        nspecies=nspecies,
        num_pressure_conditions=num_pressure_conditions,
        react_idx=react_idx,
        scaler_input=input_scalers,
        scaler_output=output_scalers
    )
    
    return train_dataset, test_dataset

