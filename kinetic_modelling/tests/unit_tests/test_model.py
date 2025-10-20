"""
Tests for the model module.
"""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetic_modelling.model import BaseModel, SVRModel, NeuralNetModel


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    n_outputs = 3
    
    x_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randn(n_samples, n_outputs)
    
    x_test = np.random.randn(20, n_features)
    y_test = np.random.randn(20, n_outputs)
    
    return x_train, y_train, x_test, y_test


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def svr_params():
    """Create sample SVR parameters."""
    return [
        {'C': 10, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'},
        {'C': 20, 'epsilon': 0.005, 'gamma': 5, 'kernel': 'rbf'},
        {'C': 5, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'}
    ]


class TestSVRModel:
    """Tests for SVRModel class."""
    
    def test_initialization(self, svr_params, temp_checkpoint_dir):
        """Test SVR model initialization."""
        model = SVRModel(
            params=svr_params,
            model_name="test_svr",
            checkpoint_dir=temp_checkpoint_dir
        )
        
        assert model is not None
        assert model.n_outputs == 3
        assert model.models is None  # Not trained yet
        assert model.metadata['model_type'] == 'SVRModel'
    
    def test_fit(self, svr_params, sample_training_data, temp_checkpoint_dir):
        """Test SVR model training."""
        x_train, y_train, _, _ = sample_training_data
        
        model = SVRModel(
            params=svr_params,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        history = model.fit(x_train, y_train)
        
        assert model.models is not None
        assert len(model.models) == 3
        assert 'train_mse_per_output' in history
        assert 'total_train_mse' in history
        assert model.metadata['trained'] is True
    
    def test_predict(self, svr_params, sample_training_data, temp_checkpoint_dir):
        """Test SVR model prediction."""
        x_train, y_train, x_test, y_test = sample_training_data
        
        model = SVRModel(
            params=svr_params,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        
        assert predictions.shape == (20, 3)
        assert not np.isnan(predictions).any()
    
    def test_predict_without_training(self, svr_params, sample_training_data, temp_checkpoint_dir):
        """Test that predict raises error when model is not trained."""
        _, _, x_test, _ = sample_training_data
        
        model = SVRModel(
            params=svr_params,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        with pytest.raises(ValueError, match="Model not trained"):
            model.predict(x_test)
    
    def test_evaluate(self, svr_params, sample_training_data, temp_checkpoint_dir):
        """Test SVR model evaluation."""
        x_train, y_train, x_test, y_test = sample_training_data
        
        model = SVRModel(
            params=svr_params,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        model.fit(x_train, y_train)
        metrics = model.evaluate(x_test, y_test)
        
        assert 'mse_per_output' in metrics
        assert 'total_mse' in metrics
        assert len(metrics['mse_per_output']) == 3
        assert metrics['total_mse'] > 0
    
    def test_save_and_load(self, svr_params, sample_training_data, temp_checkpoint_dir):
        """Test saving and loading SVR model."""
        x_train, y_train, x_test, _ = sample_training_data
        
        # Train and save model
        model1 = SVRModel(
            params=svr_params,
            checkpoint_dir=temp_checkpoint_dir
        )
        model1.fit(x_train, y_train)
        pred1 = model1.predict(x_test)
        
        checkpoint_path = model1.save("test_checkpoint")
        
        # Load model and compare predictions
        model2 = SVRModel(
            params=svr_params,
            checkpoint_dir=temp_checkpoint_dir
        )
        model2.load(checkpoint_path)
        pred2 = model2.predict(x_test)
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(pred1, pred2)
        
        # Metadata should be loaded
        assert model2.metadata['trained'] is True
        assert model2.metadata['n_samples'] == 100


class TestNeuralNetModel:
    """Tests for NeuralNetModel class."""
    
    def test_initialization(self, temp_checkpoint_dir):
        """Test Neural Network model initialization."""
        model = NeuralNetModel(
            input_size=10,
            output_size=3,
            hidden_sizes=(30, 30),
            model_name="test_nn",
            checkpoint_dir=temp_checkpoint_dir
        )
        
        assert model is not None
        assert model.input_size == 10
        assert model.output_size == 3
        assert model.hidden_sizes == (30, 30)
        assert model.metadata['model_type'] == 'NeuralNetModel'
    
    def test_fit(self, sample_training_data, temp_checkpoint_dir):
        """Test Neural Network model training."""
        x_train, y_train, _, _ = sample_training_data
        
        model = NeuralNetModel(
            input_size=10,
            output_size=3,
            hidden_sizes=(20, 20),
            checkpoint_dir=temp_checkpoint_dir
        )
        
        history = model.fit(
            x_train, y_train,
            batch_size=16,
            num_epochs=50,
            patience=10,
            verbose=False
        )
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) > 0
        assert model.metadata['trained'] is True
    
    def test_predict(self, sample_training_data, temp_checkpoint_dir):
        """Test Neural Network model prediction."""
        x_train, y_train, x_test, _ = sample_training_data
        
        model = NeuralNetModel(
            input_size=10,
            output_size=3,
            hidden_sizes=(20, 20),
            checkpoint_dir=temp_checkpoint_dir
        )
        
        model.fit(x_train, y_train, num_epochs=50, verbose=False)
        predictions = model.predict(x_test)
        
        assert predictions.shape == (20, 3)
        assert not np.isnan(predictions).any()
    
    def test_evaluate(self, sample_training_data, temp_checkpoint_dir):
        """Test Neural Network model evaluation."""
        x_train, y_train, x_test, y_test = sample_training_data
        
        model = NeuralNetModel(
            input_size=10,
            output_size=3,
            hidden_sizes=(20, 20),
            checkpoint_dir=temp_checkpoint_dir
        )
        
        model.fit(x_train, y_train, num_epochs=50, verbose=False)
        metrics = model.evaluate(x_test, y_test)
        
        assert 'mse_per_output' in metrics
        assert 'total_mse' in metrics
        assert len(metrics['mse_per_output']) == 3
        assert metrics['total_mse'] > 0
    
    def test_save_and_load(self, sample_training_data, temp_checkpoint_dir):
        """Test saving and loading Neural Network model."""
        x_train, y_train, x_test, _ = sample_training_data
        
        # Train and save model
        model1 = NeuralNetModel(
            input_size=10,
            output_size=3,
            hidden_sizes=(20, 20),
            checkpoint_dir=temp_checkpoint_dir
        )
        model1.fit(x_train, y_train, num_epochs=50, verbose=False)
        pred1 = model1.predict(x_test)
        
        checkpoint_path = model1.save("test_nn_checkpoint")
        
        # Load model and compare predictions
        model2 = NeuralNetModel(
            input_size=10,
            output_size=3,
            hidden_sizes=(20, 20),
            checkpoint_dir=temp_checkpoint_dir
        )
        model2.load(checkpoint_path)
        pred2 = model2.predict(x_test)
        
        # Predictions should be very close (allowing for small floating point differences)
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)
        
        # Metadata should be loaded
        assert model2.metadata['trained'] is True
        assert model2.metadata['n_samples'] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
