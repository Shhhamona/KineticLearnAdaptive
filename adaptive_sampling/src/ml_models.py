"""
Machine learning models for rate coefficient prediction.

This module provides a unified interface for different ML models used in 
the inverse problem: predicting K from chemical compositions C.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple, Dict, Any
import joblib
import os


class RateCoefficientPredictor:
    """
    Unified interface for ML models predicting rate coefficients from compositions.
    """
    
    def __init__(self, model_type: str = 'random_forest', **model_kwargs):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model ('random_forest', 'svm', 'neural_network')
            **model_kwargs: Additional arguments for the specific model
        """
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        
        # Initialize scalers
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Initialize model
        self.model = self._create_model()
        self.is_fitted = False
        
    def _create_model(self):
        """Create the appropriate ML model."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.model_kwargs.get('n_estimators', 100),
                max_depth=self.model_kwargs.get('max_depth', 10),
                random_state=self.model_kwargs.get('random_state', 42),
                n_jobs=self.model_kwargs.get('n_jobs', -1)
            )
        elif self.model_type == 'svm':
            return SVR(
                C=self.model_kwargs.get('C', 1.0),
                epsilon=self.model_kwargs.get('epsilon', 0.1),
                kernel=self.model_kwargs.get('kernel', 'rbf'),
                gamma=self.model_kwargs.get('gamma', 'scale')
            )
        elif self.model_type == 'neural_network':
            return MLPRegressor(
                hidden_layer_sizes=self.model_kwargs.get('hidden_layer_sizes', (100, 50)),
                max_iter=self.model_kwargs.get('max_iter', 1000),
                random_state=self.model_kwargs.get('random_state', 42),
                early_stopping=self.model_kwargs.get('early_stopping', True),
                validation_fraction=self.model_kwargs.get('validation_fraction', 0.2)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train the model on composition -> K data.
        
        Args:
            X: Chemical compositions, shape (n_samples, n_species)
            y: Rate coefficients, shape (n_samples, n_k)
            
        Returns:
            Dictionary with training metrics
        """
        # Scale the data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Train the model
        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True
        
        # Calculate training metrics
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        metrics = {
            'r2_score': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'relative_error': np.mean(np.abs((y - y_pred) / (y + 1e-10)))
        }
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict rate coefficients from compositions.
        
        Args:
            X: Chemical compositions, shape (n_samples, n_species)
            
        Returns:
            Predicted rate coefficients, shape (n_samples, n_k)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Test compositions
            y: True rate coefficients
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'r2_score': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'relative_error': np.mean(np.abs((y - y_pred) / (y + 1e-10)))
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'model_type': self.model_type,
            'model_kwargs': self.model_kwargs,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        self.model_type = model_data['model_type']
        self.model_kwargs = model_data['model_kwargs']
        self.is_fitted = model_data['is_fitted']


class MultiOutputSVRPredictor(RateCoefficientPredictor):
    """
    SVR predictor that handles multiple outputs by training separate models.
    """
    
    def __init__(self, **model_kwargs):
        """Initialize multi-output SVR."""
        super().__init__(model_type='svm', **model_kwargs)
        self.models = []  # Will store one model per output
    
    def _create_model(self):
        """Create a single SVR model (template)."""
        return SVR(
            C=self.model_kwargs.get('C', 1.0),
            epsilon=self.model_kwargs.get('epsilon', 0.1),
            kernel=self.model_kwargs.get('kernel', 'rbf'),
            gamma=self.model_kwargs.get('gamma', 'scale')
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train separate SVR models for each output.
        
        Args:
            X: Chemical compositions, shape (n_samples, n_species)
            y: Rate coefficients, shape (n_samples, n_k)
            
        Returns:
            Dictionary with training metrics
        """
        # Scale the data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Train one model per output
        self.models = []
        for i in range(y_scaled.shape[1]):
            model = self._create_model()
            model.fit(X_scaled, y_scaled[:, i])
            self.models.append(model)
        
        self.is_fitted = True
        
        # Calculate training metrics
        y_pred = self.predict(X)
        
        metrics = {
            'r2_score': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'relative_error': np.mean(np.abs((y - y_pred) / (y + 1e-10)))
        }
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using all trained models.
        
        Args:
            X: Chemical compositions, shape (n_samples, n_species)
            
        Returns:
            Predicted rate coefficients, shape (n_samples, n_k)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler_X.transform(X)
        
        # Predict with each model
        predictions = []
        for model in self.models:
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        # Combine predictions and inverse transform
        y_pred_scaled = np.column_stack(predictions)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred
