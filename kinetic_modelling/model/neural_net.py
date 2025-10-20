"""
Neural Network model for kinetic modeling.

This module provides PyTorch-based neural network implementation with
checkpoint management.
"""

import numpy as np
import json
import copy
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset, random_split

from .base import BaseModel


class SimpleNeuralNet(nn.Module):
    """Simple feedforward neural network for regression."""
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: Tuple[int, ...], activation: str = 'tanh'):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features
            output_size: Number of output dimensions
            hidden_sizes: Tuple of hidden layer sizes
            activation: Activation function ('tanh' or 'relu')
        """
        super().__init__()
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)


class NeuralNetModel(BaseModel):
    """
    Neural Network model for multi-output regression using PyTorch.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Tuple[int, ...] = (30, 30),
        activation: str = 'tanh',
        learning_rate: float = 0.0001,
        model_name: str = "nn_model",
        checkpoint_dir: str = "model_checkpoints"
    ):
        """
        Initialize Neural Network model.
        
        Args:
            input_size: Number of input features
            output_size: Number of output dimensions
            hidden_sizes: Tuple of hidden layer sizes
            activation: Activation function ('tanh' or 'relu')
            learning_rate: Learning rate for optimizer
            model_name: Name for the model
            checkpoint_dir: Directory to save/load checkpoints
        """
        super().__init__(model_name, checkpoint_dir)
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        
        # Set device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Create model and move to device
        self.model = SimpleNeuralNet(input_size, output_size, hidden_sizes, activation)
        self.model = self.model.to(self.device)
        
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = MSELoss()
        
        # Store metadata
        self.metadata.update({
            'input_size': input_size,
            'output_size': output_size,
            'hidden_sizes': hidden_sizes,
            'activation': activation,
            'learning_rate': learning_rate
        })
    
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int = 32,
        num_epochs: int = 1000,
        patience: int = 15,
        val_split: float = 0.1,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the neural network.
        
        Args:
            x_train: Training input features
            y_train: Training targets
            batch_size: Batch size for training
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            val_split: Validation split fraction
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        # Convert to tensors
        x_tensor = torch.from_numpy(x_train).float()
        y_tensor = torch.from_numpy(y_train).float()
        
        # Create dataset
        dataset = TensorDataset(x_tensor, y_tensor)
        
        # Split into train and validation
        train_len = int((1.0 - val_split) * len(dataset))
        val_len = len(dataset) - train_len
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training variables
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Move data to device
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss /= len(val_loader.dataset)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6e}, Val Loss: {val_loss:.6e}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        
        self.metadata['trained'] = True
        self.metadata['trained_at'] = datetime.now().isoformat()
        self.metadata['n_samples'] = x_train.shape[0]
        self.metadata['n_features'] = x_train.shape[1]
        self.metadata['final_epoch'] = epoch + 1
        self.metadata['best_val_loss'] = best_val_loss
        
        return history
    
    def train_single_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """
        Perform a single batch gradient update.
        
        This method is useful for batch-by-batch training where you want
        fine-grained control over the training process.
        
        Args:
            x_batch: Batch of input features
            y_batch: Batch of target values
            
        Returns:
            Loss value for this batch
        """
        # Set model to training mode
        self.model.train()
        
        # Convert to tensors and move to device
        x_tensor = torch.from_numpy(x_batch).float().to(self.device)
        y_tensor = torch.from_numpy(y_batch).float().to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(x_tensor)
        loss = self.criterion(outputs, y_tensor)
        
        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            x: Input features
            
        Returns:
            Predictions
        """
        self.model.eval()
        
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().to(self.device)
            outputs = self.model(x_tensor)
            predictions = outputs.cpu().numpy()  # Move back to CPU for numpy conversion
        
        return predictions
    
    def save(self, checkpoint_name: Optional[str] = None) -> str:
        """
        Save the neural network to a checkpoint.
        
        Args:
            checkpoint_name: Optional name for the checkpoint
            
        Returns:
            Path to the saved checkpoint directory
        """
        # Create checkpoint name
        if checkpoint_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_name = f"{self.model_name}_{timestamp}"
        
        # Create checkpoint directory
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_file = checkpoint_path / "model.pth"
        torch.save(self.model.state_dict(), model_file)
        
        # Save optimizer state
        optimizer_file = checkpoint_path / "optimizer.pth"
        torch.save(self.optimizer.state_dict(), optimizer_file)
        
        # Save metadata
        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✅ Saved Neural Network model to: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load(self, checkpoint_path: str):
        """
        Load neural network from a checkpoint.
        
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
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Recreate model with saved parameters
        self.model = SimpleNeuralNet(
            self.metadata['input_size'],
            self.metadata['output_size'],
            tuple(self.metadata['hidden_sizes']),
            self.metadata['activation']
        )
        
        # Load model state and move to device
        model_file = checkpoint_path / "model.pth"
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model = self.model.to(self.device)
        
        # Load optimizer state
        self.optimizer = Adam(self.model.parameters(), lr=self.metadata['learning_rate'])
        optimizer_file = checkpoint_path / "optimizer.pth"
        if optimizer_file.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_file, map_location=self.device))
        
        print(f"✅ Loaded Neural Network model from: {checkpoint_path}")
