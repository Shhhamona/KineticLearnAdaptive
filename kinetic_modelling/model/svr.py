"""
Support Vector Regression model for kinetic modeling.

This module provides SVR model implementation with multi-output support
and checkpoint management.
"""

import numpy as np
import pickle
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from .base import BaseModel


class SVRModel(BaseModel):
    """
    Support Vector Regression model for multi-output prediction.
    
    Trains separate SVR models for each output dimension.
    """
    
    def __init__(
        self,
        params: List[Dict[str, Any]],
        model_name: str = "svr_model",
        checkpoint_dir: str = "model_checkpoints"
    ):
        """
        Initialize SVR model.
        
        Args:
            params: List of parameter dictionaries, one per output dimension
                    Each dict should contain: 'C', 'epsilon', 'gamma', 'kernel'
            model_name: Name for the model
            checkpoint_dir: Directory to save/load checkpoints
        """
        super().__init__(model_name, checkpoint_dir)
        self.params = params
        self.models = None
        self.n_outputs = len(params)
        self.metadata['params'] = params
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train SVR models on the provided data.
        
        Args:
            x_train: Training input features, shape (n_samples, n_features)
            y_train: Training targets, shape (n_samples, n_outputs)
            **kwargs: Additional parameters (currently unused)
            
        Returns:
            Dictionary with training metrics
        """
        if y_train.shape[1] != self.n_outputs:
            raise ValueError(f"Expected {self.n_outputs} outputs, got {y_train.shape[1]}")
        
        self.models = []
        train_mse_per_output = []
        
        print(f"Training {self.n_outputs} SVR models...")
        for i in range(self.n_outputs):
            print(f"  Training model {i+1}/{self.n_outputs}...")
            model = SVR(
                C=self.params[i]['C'],
                epsilon=self.params[i]['epsilon'],
                gamma=self.params[i]['gamma'],
                kernel=self.params[i]['kernel']
            )
            
            model.fit(x_train, y_train[:, i])
            self.models.append(model)
            
            # Calculate training MSE
            y_pred_train = model.predict(x_train)
            train_mse = mean_squared_error(y_train[:, i], y_pred_train)
            train_mse_per_output.append(train_mse)
            print(f"    Training MSE: {train_mse:.6e}")
        
        self.metadata['trained'] = True
        self.metadata['trained_at'] = datetime.now().isoformat()
        self.metadata['n_samples'] = x_train.shape[0]
        self.metadata['n_features'] = x_train.shape[1]
        
        return {
            'train_mse_per_output': train_mse_per_output,
            'total_train_mse': np.sum(train_mse_per_output)
        }
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            x: Input features, shape (n_samples, n_features)
            
        Returns:
            Predictions, shape (n_samples, n_outputs)
        """
        if self.models is None:
            raise ValueError("Model not trained. Call fit() first or load() a checkpoint.")
        
        predictions = np.zeros((x.shape[0], self.n_outputs))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(x)
        
        return predictions
    
    def save(self, checkpoint_name: Optional[str] = None) -> str:
        """
        Save the SVR models to a checkpoint.
        
        Args:
            checkpoint_name: Optional name for the checkpoint
            
        Returns:
            Path to the saved checkpoint directory
        """
        if self.models is None:
            raise ValueError("No trained models to save.")
        
        # Create checkpoint name
        if checkpoint_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_name = f"{self.model_name}_{timestamp}"
        
        # Create checkpoint directory
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for i, model in enumerate(self.models):
            model_file = checkpoint_path / f"svr_model_{i}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        
        # Save metadata
        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✅ Saved SVR model to: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load(self, checkpoint_path: str):
        """
        Load SVR models from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint directory
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load metadata
        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Load models
        self.models = []
        for i in range(self.n_outputs):
            model_file = checkpoint_path / f"svr_model_{i}.pkl"
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            self.models.append(model)
        
        print(f"✅ Loaded SVR model from: {checkpoint_path}")
