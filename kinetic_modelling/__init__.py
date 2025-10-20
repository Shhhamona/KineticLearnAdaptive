"""
Kinetic Modelling Package

A clean, modular framework for kinetic modeling with adaptive learning.

Modules:
    data: Data loading and preprocessing
    model: Model training and prediction
    sampling: Sampling strategies for data subsets
    pipeline: End-to-end workflow integration
    evaluation: Model evaluation and metrics (coming soon)
"""

from .data import (
    MultiPressureDataset,
    apply_training_scalers,
    load_datasets
)

from .model import (
    BaseModel,
    SVRModel,
    NeuralNetModel
)

from .sampling import (
    BaseSampler,
    SequentialSampler,
    RandomSampler,
    WindowSampler
)

from .pipeline import (
    BasePipeline,
    StandardPipeline,
    StandardSubsetPipeline,
    BatchTrainingPipeline,
    AdaptiveSamplingPipeline
)

__version__ = "0.1.0"

__all__ = [
    "MultiPressureDataset",
    "apply_training_scalers",
    "load_datasets",
    "BaseModel",
    "SVRModel",
    "NeuralNetModel",
    "BaseSampler",
    "SequentialSampler",
    "RandomSampler",
    "WindowSampler",
    "BasePipeline",
    "StandardPipeline",
    "StandardSubsetPipeline",
    "BatchTrainingPipeline",
    "AdaptiveSamplingPipeline"
]
