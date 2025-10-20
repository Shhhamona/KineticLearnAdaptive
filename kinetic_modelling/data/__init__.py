"""
Data module for kinetic modeling.

This module provides clean abstractions for loading multi-pressure condition datasets
and handling data scaling in a safe, reusable manner.
"""

from .data import MultiPressureDataset
from .processing import (
    apply_training_scalers,
    load_datasets
)

__all__ = [
    'MultiPressureDataset',
    'apply_training_scalers',
    'load_datasets'
]
