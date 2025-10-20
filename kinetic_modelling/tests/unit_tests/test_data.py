"""
Tests for the data loading module.
"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

# Add parent directory to path to import kinetic_modelling
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append('.')
from kinetic_modelling.data import (
    MultiPressureDataset,
    apply_training_scalers,
    load_datasets
)


@pytest.fixture
def sample_data_file():
    """Create a temporary data file for testing."""
    # Create sample data: 100 rows, 3 reactions + 5 species * 2 pressure conditions
    # Format: [k1, k2, k3, dens1_p1, dens2_p1, ..., dens5_p1, dens1_p2, ..., dens5_p2]
    n_samples = 50  # 50 simulations
    n_reactions = 3
    n_species = 5
    n_pressures = 2
    
    np.random.seed(42)
    
    # Generate data: reactions (small values) + densities (larger values)
    reactions = np.random.uniform(1e-18, 1e-16, (n_samples * n_pressures, n_reactions))
    densities = np.random.uniform(1e14, 1e18, (n_samples * n_pressures, n_species))
    
    # Combine: k values followed by densities for all pressure conditions
    data = np.hstack([reactions, densities])
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        # Write header
        f.write("# Sample kinetic data\n")
        f.write("# k1  k2  k3  dens1_p1  dens2_p1  dens3_p1  dens4_p1  dens5_p1  dens1_p2  dens2_p2  dens3_p2  dens4_p2  dens5_p2\n")
        
        # Write data
        for row in data:
            f.write("  ".join([f"{val:.6e}" for val in row]) + "\n")
        
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


class TestMultiPressureDataset:
    """Tests for MultiPressureDataset class."""
    
    def test_initialization(self, sample_data_file):
        """Test basic dataset initialization."""
        dataset = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            src_file=sample_data_file
        )
        
        assert dataset is not None
        assert dataset.num_pressure_conditions == 2
        assert dataset.nspecies == 5
        assert len(dataset) == 50  # 100 rows / 2 pressures
    
    def test_data_shapes(self, sample_data_file):
        """Test that data has correct shapes."""
        dataset = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            src_file=sample_data_file
        )
        
        x, y = dataset.get_data()
        
        # x should be (n_samples, num_pressure_conditions * nspecies)
        assert x.shape == (50, 2 * 5)
        
        # y should be (n_samples, n_reactions)
        assert y.shape == (50, 3)
    
    def test_scaling(self, sample_data_file):
        """Test that data is properly scaled."""
        dataset = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            src_file=sample_data_file
        )
        
        x, y = dataset.get_data()
        
        # MaxAbsScaler scales to [-1, 1]
        assert np.all(np.abs(x) <= 1.0)
        assert np.all(np.abs(y) <= 1.0)
    
    def test_scaler_reuse(self, sample_data_file):
        """Test that scalers can be reused for new data."""
        # Load training data
        train_dataset = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            src_file=sample_data_file
        )
        
        # Get scalers
        input_scalers, output_scalers = train_dataset.get_scalers()
        
        # Load test data with same scalers
        test_dataset = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            src_file=sample_data_file,
            scaler_input=input_scalers,
            scaler_output=output_scalers
        )
        
        # Verify scalers are the same objects
        assert test_dataset.scaler_input is input_scalers
        assert test_dataset.scaler_output is output_scalers
    
    def test_react_idx_selection(self, sample_data_file):
        """Test selecting specific reaction indices."""
        # Select only first 2 reactions
        dataset = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            src_file=sample_data_file,
            react_idx=np.array([0, 1])
        )
        
        _, y = dataset.get_data()
        
        # Should only have 2 reactions
        assert y.shape[1] == 2
    
    def test_repr(self, sample_data_file):
        """Test string representation."""
        dataset = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            src_file=sample_data_file
        )
        
        repr_str = repr(dataset)
        assert "samples=50" in repr_str
        assert "features=10" in repr_str
        assert "outputs=3" in repr_str
        assert "pressures=2" in repr_str
    
    def test_init_from_arrays(self, sample_data_file):
        """Test initializing dataset from raw arrays."""
        # First create a training dataset to get scalers
        train_dataset = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            src_file=sample_data_file
        )
        
        # Get scalers
        input_scalers, output_scalers = train_dataset.get_scalers()
        
        # Create new simulation data
        np.random.seed(42)
        n_new_sims = 10
        raw_compositions = np.random.uniform(1e14, 1e18, (n_new_sims * 2, 5))
        raw_k_values = np.random.uniform(1e-18, 1e-16, (n_new_sims, 3))
        
        # Initialize from arrays
        array_dataset = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            raw_compositions=raw_compositions,
            raw_k_values=raw_k_values,
            scaler_input=input_scalers,
            scaler_output=output_scalers
        )
        
        # Check properties
        assert array_dataset is not None
        assert len(array_dataset) == 10
        assert array_dataset.raw_data is None  # No file data in this mode
        
        # Check data shapes
        x, y = array_dataset.get_data()
        assert x.shape == (10, 2 * 5)
        assert y.shape == (10, 3)
        
        # Check that data is scaled
        assert np.all(np.abs(x) <= 2.0)
        assert np.all(np.abs(y) <= 2.0)
    
    def test_init_from_arrays_requires_scalers(self, sample_data_file):
        """Test that array initialization requires scalers."""
        np.random.seed(42)
        n_new_sims = 10
        raw_compositions = np.random.uniform(1e14, 1e18, (n_new_sims * 2, 5))
        raw_k_values = np.random.uniform(1e-18, 1e-16, (n_new_sims, 3))
        
        # Should raise error without scalers
        with pytest.raises(ValueError, match="both scaler_input and scaler_output must be provided"):
            MultiPressureDataset(
                nspecies=5,
                num_pressure_conditions=2,
                raw_compositions=raw_compositions,
                raw_k_values=raw_k_values
            )
    
    def test_init_requires_either_file_or_arrays(self):
        """Test that initialization requires either file or arrays."""
        with pytest.raises(ValueError, match="Must provide either src_file OR"):
            MultiPressureDataset(
                nspecies=5,
                num_pressure_conditions=2
            )


class TestApplyTrainingScalers:
    """Tests for apply_training_scalers function."""
    
    def test_basic_scaling(self, sample_data_file):
        """Test basic scaler application."""
        # Load training dataset
        train_dataset = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            src_file=sample_data_file
        )
        
        # Create new simulation data
        np.random.seed(123)
        n_new_sims = 10
        raw_compositions = np.random.uniform(1e14, 1e18, (n_new_sims * 2, 5))
        raw_k_values = np.random.uniform(1e-18, 1e-16, (n_new_sims, 3))
        
        # Apply scalers
        new_x, new_y = apply_training_scalers(
            raw_compositions=raw_compositions,
            raw_k_values=raw_k_values,
            dataset_train=train_dataset,
            nspecies=5,
            num_pressure_conditions=2,
            debug=False
        )
        
        # Check shapes
        assert new_x.shape == (10, 2 * 5)
        assert new_y.shape == (10, 3)
        
        # Check scaling (should be in [-1, 1] range approximately)
        assert np.all(np.abs(new_x) <= 2.0)  # Allow some margin
        assert np.all(np.abs(new_y) <= 2.0)
    
    def test_idempotency(self, sample_data_file):
        """Test that repeated calls with same data give same results."""
        train_dataset = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            src_file=sample_data_file
        )
        
        # Create new simulation data
        np.random.seed(123)
        n_new_sims = 10
        raw_compositions = np.random.uniform(1e14, 1e18, (n_new_sims * 2, 5))
        raw_k_values = np.random.uniform(1e-18, 1e-16, (n_new_sims, 3))
        
        # Apply scalers twice
        new_x1, new_y1 = apply_training_scalers(
            raw_compositions=raw_compositions.copy(),
            raw_k_values=raw_k_values.copy(),
            dataset_train=train_dataset,
            nspecies=5,
            num_pressure_conditions=2
        )
        
        new_x2, new_y2 = apply_training_scalers(
            raw_compositions=raw_compositions.copy(),
            raw_k_values=raw_k_values.copy(),
            dataset_train=train_dataset,
            nspecies=5,
            num_pressure_conditions=2
        )
        
        # Results should be identical
        np.testing.assert_array_almost_equal(new_x1, new_x2)
        np.testing.assert_array_almost_equal(new_y1, new_y2)
    
    def test_no_mutation(self, sample_data_file):
        """Test that input arrays are not mutated."""
        train_dataset = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            src_file=sample_data_file
        )
        
        # Create new simulation data
        np.random.seed(123)
        n_new_sims = 10
        raw_compositions = np.random.uniform(1e14, 1e18, (n_new_sims * 2, 5))
        raw_k_values = np.random.uniform(1e-18, 1e-16, (n_new_sims, 3))
        
        # Store copies
        comp_copy = raw_compositions.copy()
        k_copy = raw_k_values.copy()
        
        # Apply scalers
        apply_training_scalers(
            raw_compositions=raw_compositions,
            raw_k_values=raw_k_values,
            dataset_train=train_dataset,
            nspecies=5,
            num_pressure_conditions=2
        )
        
        # Original arrays should be unchanged
        np.testing.assert_array_almost_equal(raw_compositions, comp_copy)
        np.testing.assert_array_almost_equal(raw_k_values, k_copy)


class TestProcessedArrayInit:
    """Tests for processed array initialization mode."""
    
    def test_init_from_processed_arrays(self, sample_data_file):
        """Test initializing from already-processed (scaled) arrays."""
        # First create a dataset from file to get processed data
        original = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            src_file=sample_data_file
        )
        
        # Get the processed data and scalers
        x_processed, y_processed = original.get_data()
        input_scalers, output_scalers = original.get_scalers()
        
        # Create a new dataset from the processed arrays
        new_dataset = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            processed_x=x_processed,
            processed_y=y_processed,
            scaler_input=input_scalers,
            scaler_output=output_scalers
        )
        
        # Data should be identical
        x_new, y_new = new_dataset.get_data()
        np.testing.assert_array_almost_equal(x_new, x_processed)
        np.testing.assert_array_almost_equal(y_new, y_processed)
    
    def test_processed_arrays_require_scalers(self):
        """Test that processed arrays require scalers."""
        x = np.random.randn(10, 10)
        y = np.random.randn(10, 3)
        
        with pytest.raises(ValueError, match="both scaler_input and scaler_output must be provided"):
            MultiPressureDataset(
                nspecies=5,
                num_pressure_conditions=2,
                processed_x=x,
                processed_y=y
            )
    
    def test_processed_arrays_defensive_copy(self, sample_data_file):
        """Test that processed array init makes defensive copies."""
        # Create original dataset
        original = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            src_file=sample_data_file
        )
        
        x_orig, y_orig = original.get_data()
        input_scalers, output_scalers = original.get_scalers()
        
        # Keep references to original arrays
        x_ref = x_orig.copy()
        y_ref = y_orig.copy()
        
        # Create new dataset
        new_dataset = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            processed_x=x_orig,
            processed_y=y_orig,
            scaler_input=input_scalers,
            scaler_output=output_scalers
        )
        
        # Modify original arrays
        x_orig[:] = 999
        y_orig[:] = 999
        
        # New dataset should be unaffected
        x_new, y_new = new_dataset.get_data()
        np.testing.assert_array_almost_equal(x_new, x_ref)
        np.testing.assert_array_almost_equal(y_new, y_ref)
    
    def test_processed_arrays_subset(self, sample_data_file):
        """Test creating a subset using processed arrays."""
        # Create original dataset
        original = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            src_file=sample_data_file
        )
        
        x_orig, y_orig = original.get_data()
        input_scalers, output_scalers = original.get_scalers()
        
        # Create subset with first 10 samples
        subset_indices = np.arange(10)
        x_subset = x_orig[subset_indices]
        y_subset = y_orig[subset_indices]
        
        subset_dataset = MultiPressureDataset(
            nspecies=5,
            num_pressure_conditions=2,
            processed_x=x_subset,
            processed_y=y_subset,
            scaler_input=input_scalers,
            scaler_output=output_scalers
        )
        
        # Check size and data
        assert len(subset_dataset) == 10
        x_sub, y_sub = subset_dataset.get_data()
        np.testing.assert_array_almost_equal(x_sub, x_subset)
        np.testing.assert_array_almost_equal(y_sub, y_subset)


class TestLoadDatasets:
    """Tests for load_datasets function."""
    
    def test_load_train_test(self, sample_data_file):
        """Test loading training and test datasets."""
        train_dataset, test_dataset = load_datasets(
            train_file=sample_data_file,
            test_file=sample_data_file,
            nspecies=5,
            num_pressure_conditions=2
        )
        
        assert train_dataset is not None
        assert test_dataset is not None
        assert len(train_dataset) == len(test_dataset)
    
    def test_shared_scalers(self, sample_data_file):
        """Test that train and test datasets share scalers."""
        train_dataset, test_dataset = load_datasets(
            train_file=sample_data_file,
            test_file=sample_data_file,
            nspecies=5,
            num_pressure_conditions=2
        )
        
        # Scalers should be the same objects
        assert train_dataset.scaler_input is test_dataset.scaler_input
        assert train_dataset.scaler_output is test_dataset.scaler_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
