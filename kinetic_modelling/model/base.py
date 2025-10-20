"""
Base model class for kinetic modeling.

This module provides the abstract base class that all models must inherit from.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error


class BaseModel(ABC):
    """
    Abstract base class for all kinetic modeling ML models.
    
    All models must implement fit(), predict(), save(), and load() methods.
    """
    
    def __init__(self, model_name: str = "model", checkpoint_dir: str = "model_checkpoints"):
        """
        Initialize the base model.
        
        Args:
            model_name: Name for the model (used in checkpoints)
            checkpoint_dir: Directory to save/load checkpoints
        """
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {
            'model_type': self.__class__.__name__,
            'created_at': datetime.now().isoformat(),
            'trained': False
        }
    
    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            x_train: Training input features, shape (n_samples, n_features)
            y_train: Training targets, shape (n_samples, n_outputs)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            x: Input features, shape (n_samples, n_features)
            
        Returns:
            Predictions, shape (n_samples, n_outputs)
        """
        pass
    
    @abstractmethod
    def save(self, checkpoint_name: Optional[str] = None) -> str:
        """
        Save the model to a checkpoint.
        
        Args:
            checkpoint_name: Optional name for the checkpoint (default: timestamped)
            
        Returns:
            Path to the saved checkpoint
        """
        pass
    
    @abstractmethod
    def load(self, checkpoint_path: str):
        """
        Load the model from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        pass
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            x_test: Test input features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics (MSE per output and total)
        """
        y_pred = self.predict(x_test)
        
        mse_per_output = []
        for i in range(y_test.shape[1]):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            mse_per_output.append(mse)
        
        total_mse = np.sum(mse_per_output)
        
        return {
            'mse_per_output': mse_per_output,
            'total_mse': total_mse
        }
