"""
Unit tests for JSON batch simulation file loading in MultiPressureDataset.
"""

import json
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from kinetic_modelling.data import MultiPressureDataset


class TestJSONLoading:
    """Test JSON batch simulation file loading."""
    
    @pytest.fixture
    def sample_json_data(self):
        """Create sample JSON batch simulation data matching real batch file format."""
        np.random.seed(42)
        n_sims = 100
        num_pressure_conditions = 2
        nspecies = 3
        nreactions = 5
        
        # Create sample data - compositions in flat format (n_sims * num_pressure, nspecies)
        compositions_3d = np.random.rand(num_pressure_conditions, n_sims, nspecies)
        compositions = compositions_3d.reshape(num_pressure_conditions * n_sims, nspecies).tolist()
        
        # Create parameter_sets as list of dictionaries (real format)
        k_values_array = np.random.rand(n_sims, nreactions)
        parameter_sets = [{'k_values': k_values_array[i].tolist()} for i in range(n_sims)]
        
        return {
            'compositions': compositions,
            'parameter_sets': parameter_sets
        }
    
    @pytest.fixture
    def json_file(self, sample_json_data):
        """Create a temporary JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_json_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    def test_json_file_detection(self, json_file):
        """Test that JSON files are correctly detected and loaded."""
        dataset = MultiPressureDataset(
            src_file=json_file,
            num_pressure_conditions=2,
            nspecies=3
        )
        
        # Should successfully load
        x_data, y_data = dataset.get_data()
        assert x_data is not None
        assert y_data is not None
        
    def test_json_data_shape(self, json_file):
        """Test that JSON data is loaded with correct shapes."""
        dataset = MultiPressureDataset(
            src_file=json_file,
            num_pressure_conditions=2,
            nspecies=3
        )
        
        x_data, y_data = dataset.get_data()
        
        # x_data should be (n_sims, num_pressure_conditions * nspecies)
        assert x_data.shape == (100, 2 * 3)
        
        # y_data should be (n_sims, nreactions)
        assert y_data.shape == (100, 5)
    
    def test_json_data_scaling(self, json_file):
        """Test that data from JSON is properly scaled."""
        dataset = MultiPressureDataset(
            src_file=json_file,
            num_pressure_conditions=2,
            nspecies=3
        )
        
        x_data, y_data = dataset.get_data()
        
        # Check that data is scaled (should be in range roughly [-1, 1])
        assert np.all(np.abs(x_data) <= 1.0)
        assert np.all(np.abs(y_data) <= 1.0)
    
    def test_json_scalers_created(self, json_file):
        """Test that scalers are created when loading JSON."""
        dataset = MultiPressureDataset(
            src_file=json_file,
            num_pressure_conditions=2,
            nspecies=3
        )
        
        scaler_input, scaler_output = dataset.get_scalers()
        
        # Should have one scaler per pressure condition
        assert len(scaler_input) == 2
        assert len(scaler_output) == 2
    
    def test_json_with_max_rows(self, json_file):
        """Test loading JSON with max_rows parameter."""
        max_rows = 50
        dataset = MultiPressureDataset(
            src_file=json_file,
            num_pressure_conditions=2,
            nspecies=3,
            max_rows=max_rows
        )
        
        x_data, y_data = dataset.get_data()
        
        # Should only load max_rows simulations
        assert x_data.shape[0] == max_rows
        assert y_data.shape[0] == max_rows
    
    def test_json_with_react_idx(self, json_file):
        """Test loading JSON with reaction index selection."""
        react_idx = np.array([0, 2, 4])  # Select reactions 0, 2, 4
        dataset = MultiPressureDataset(
            src_file=json_file,
            num_pressure_conditions=2,
            nspecies=3,
            react_idx=react_idx
        )
        
        x_data, y_data = dataset.get_data()
        
        # Should only have 3 reactions
        assert y_data.shape[1] == 3
    
    def test_json_with_provided_scalers(self, json_file):
        """Test loading JSON with pre-existing scalers."""
        from sklearn.preprocessing import MaxAbsScaler
        
        # Create first dataset to get scalers
        dataset1 = MultiPressureDataset(
            src_file=json_file,
            num_pressure_conditions=2,
            nspecies=3
        )
        scaler_input, scaler_output = dataset1.get_scalers()
        
        # Create second dataset with same scalers
        dataset2 = MultiPressureDataset(
            src_file=json_file,
            num_pressure_conditions=2,
            nspecies=3,
            scaler_input=scaler_input,
            scaler_output=scaler_output
        )
        
        # Data should be scaled identically
        x1, y1 = dataset1.get_data()
        x2, y2 = dataset2.get_data()
        
        np.testing.assert_array_almost_equal(x1, x2)
        np.testing.assert_array_almost_equal(y1, y2)
    
    def test_json_missing_keys_error(self):
        """Test that missing required keys raises an error."""
        # Create JSON with missing required keys (no compositions or k_values)
        invalid_data = {
            'some_other_key': [1, 2, 3]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="must contain 'compositions' key"):
                dataset = MultiPressureDataset(
                    src_file=temp_path,
                    num_pressure_conditions=2,
                    nspecies=3
                )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_json_shape_mismatch_error(self):
        """Test that shape mismatch raises an error."""
        # Create JSON with mismatched shapes: 2 sims but compositions for 3*1 = 3 entries
        invalid_data = {
            'compositions': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # 3 entries (3 * 1 pressure)
            'parameter_sets': [
                {'k_values': [1, 2, 3]},
                {'k_values': [4, 5, 6]}  # Only 2 simulations - MISMATCH
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                dataset = MultiPressureDataset(
                    src_file=temp_path,
                    num_pressure_conditions=1,
                    nspecies=3
                )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_json_vs_txt_consistency(self, json_file):
        """Test that JSON loading produces consistent results with array initialization."""
        # Load from JSON
        dataset_json = MultiPressureDataset(
            src_file=json_file,
            num_pressure_conditions=2,
            nspecies=3
        )
        
        # Get scalers from JSON dataset
        scaler_input, scaler_output = dataset_json.get_scalers()
        
        # Load JSON data manually and use array initialization
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract k_values from parameter_sets (real format)
        compositions = np.array(data['compositions'])
        k_values = np.array([ps['k_values'] for ps in data['parameter_sets']])
        
        dataset_array = MultiPressureDataset(
            num_pressure_conditions=2,
            nspecies=3,
            raw_compositions=compositions,
            raw_k_values=k_values,
            scaler_input=scaler_input,
            scaler_output=scaler_output
        )
        
        # Should produce identical results
        x_json, y_json = dataset_json.get_data()
        x_array, y_array = dataset_array.get_data()
        
        np.testing.assert_array_almost_equal(x_json, x_array)
        np.testing.assert_array_almost_equal(y_json, y_array)
    
    def test_json_raw_data_storage(self, json_file):
        """Test that raw JSON data is stored for reference."""
        dataset = MultiPressureDataset(
            src_file=json_file,
            num_pressure_conditions=2,
            nspecies=3
        )
        
        # raw_data should contain the original JSON dict
        assert dataset.raw_data is not None
        assert isinstance(dataset.raw_data, dict)
        assert 'compositions' in dataset.raw_data
        assert 'parameter_sets' in dataset.raw_data
        # Verify parameter_sets has the expected structure
        assert len(dataset.raw_data['parameter_sets']) > 0
        assert 'k_values' in dataset.raw_data['parameter_sets'][0]


class TestJSONIntegration:
    """Integration tests for JSON loading with other components."""
    
    @pytest.fixture
    def multi_pressure_json_data(self):
        """Create JSON data with multiple pressure conditions."""
        np.random.seed(123)
        n_sims = 200
        num_pressure_conditions = 3
        nspecies = 5
        nreactions = 7
        
        # Create realistic-looking compositions in flat format (n_sims * num_pressure, nspecies)
        compositions = []
        for _ in range(num_pressure_conditions):
            for _ in range(n_sims):
                # Each simulation has composition data
                composition = np.random.uniform(0.1, 1.0, nspecies).tolist()
                compositions.append(composition)
        
        # Create parameter_sets as list of dictionaries (real format)
        k_values_array = np.random.uniform(1e-15, 1e-10, (n_sims, nreactions))
        parameter_sets = [{'k_values': k_values_array[i].tolist()} for i in range(n_sims)]
        
        return {
            'compositions': compositions,
            'parameter_sets': parameter_sets
        }
    
    @pytest.fixture
    def multi_pressure_json_file(self, multi_pressure_json_data):
        """Create temporary multi-pressure JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(multi_pressure_json_data, f)
            temp_path = f.name
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    def test_multi_pressure_json_loading(self, multi_pressure_json_file):
        """Test loading JSON with multiple pressure conditions."""
        dataset = MultiPressureDataset(
            src_file=multi_pressure_json_file,
            num_pressure_conditions=3,
            nspecies=5
        )
        
        x_data, y_data = dataset.get_data()
        
        # x_data should concatenate all pressure conditions
        assert x_data.shape == (200, 3 * 5)
        assert y_data.shape == (200, 7)
    
    def test_json_train_test_split(self, multi_pressure_json_file):
        """Test that JSON data can be split for train/test."""
        dataset = MultiPressureDataset(
            src_file=multi_pressure_json_file,
            num_pressure_conditions=3,
            nspecies=5
        )
        
        x_data, y_data = dataset.get_data()
        
        # Split manually
        n_train = 150
        x_train, x_test = x_data[:n_train], x_data[n_train:]
        y_train, y_test = y_data[:n_train], y_data[n_train:]
        
        assert x_train.shape[0] == 150
        assert x_test.shape[0] == 50
    
    def test_json_scaler_reuse_workflow(self, multi_pressure_json_file):
        """Test typical workflow: train scalers on JSON, apply to new data."""
        # Load training data
        train_dataset = MultiPressureDataset(
            src_file=multi_pressure_json_file,
            num_pressure_conditions=3,
            nspecies=5,
            max_rows=150
        )
        
        scaler_input, scaler_output = train_dataset.get_scalers()
        
        # Load test data with same scalers
        test_dataset = MultiPressureDataset(
            src_file=multi_pressure_json_file,
            num_pressure_conditions=3,
            nspecies=5,
            scaler_input=scaler_input,
            scaler_output=scaler_output
        )
        
        # Test data should be scaled using training scalers
        x_test, y_test = test_dataset.get_data()
        
        # Both datasets should use the same scaler instances
        assert test_dataset.scaler_input is scaler_input
        assert test_dataset.scaler_output is scaler_output


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
