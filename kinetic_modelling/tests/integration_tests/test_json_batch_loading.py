"""
Integration test for loading real batch simulation JSON files.
Tests compatibility with K-centered adaptive learning JSON format.
"""

import pytest
import numpy as np
import os
from pathlib import Path
from kinetic_modelling.data import MultiPressureDataset


class TestRealBatchJSONLoading:
    """Integration tests with real batch simulation JSON files."""
    
    @pytest.fixture
    def batch_json_file(self):
        """Path to real batch simulation JSON file."""
        # Path from k_centered_adaptive_learning.py
        json_path = 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2000sims_20250929_125706.json'
        
        # Convert to absolute path from workspace root
        workspace_root = Path(__file__).parent.parent.parent.parent
        full_path = workspace_root / json_path
        
        if not full_path.exists():
            pytest.skip(f"Batch JSON file not found: {full_path}")
        
        return str(full_path)
    
    def test_load_real_batch_json(self, batch_json_file):
        """Test loading a real batch simulation JSON file."""
        print(f"\n📂 Loading real batch JSON file:")
        print(f"   {batch_json_file}")
        
        # Load with MultiPressureDataset
        dataset = MultiPressureDataset(
            src_file=batch_json_file,
            num_pressure_conditions=2,
            nspecies=3
        )
        
        x_data, y_data = dataset.get_data()
        
        print(f"\n✅ Successfully loaded batch JSON file:")
        print(f"   Input shape (x_data): {x_data.shape}")
        print(f"   Output shape (y_data): {y_data.shape}")
        print(f"   Expected format: (n_sims, num_pressure_conditions * nspecies) for x")
        print(f"   Expected format: (n_sims, nreactions) for y")
        
        # Verify data is loaded
        assert x_data is not None
        assert y_data is not None
        
        # Verify shapes
        assert x_data.ndim == 2
        assert y_data.ndim == 2
        
        # Verify x_data has correct feature dimension
        assert x_data.shape[1] == 2 * 3  # 2 pressures * 3 species
        
        # Verify same number of samples
        assert x_data.shape[0] == y_data.shape[0]
        
        print(f"\n✅ Shape validation passed!")
    
    def test_json_data_scaling(self, batch_json_file):
        """Test that data from real JSON is properly scaled."""
        dataset = MultiPressureDataset(
            src_file=batch_json_file,
            num_pressure_conditions=2,
            nspecies=3
        )
        
        x_data, y_data = dataset.get_data()
        
        print(f"\n📊 Data scaling verification:")
        print(f"   x_data range: [{x_data.min():.6f}, {x_data.max():.6f}]")
        print(f"   y_data range: [{y_data.min():.6f}, {y_data.max():.6f}]")
        
        # Scaled data should be in range roughly [-1, 1] (MaxAbsScaler)
        assert np.all(np.abs(x_data) <= 1.0 + 1e-6), "x_data should be scaled to [-1, 1]"
        assert np.all(np.abs(y_data) <= 1.0 + 1e-6), "y_data should be scaled to [-1, 1]"
        
        print(f"   ✅ Data is properly scaled!")
    
    def test_json_scalers_created(self, batch_json_file):
        """Test that scalers are properly created from real JSON."""
        dataset = MultiPressureDataset(
            src_file=batch_json_file,
            num_pressure_conditions=2,
            nspecies=3
        )
        
        scaler_input, scaler_output = dataset.get_scalers()
        
        print(f"\n⚙️ Scaler verification:")
        print(f"   Number of input scalers: {len(scaler_input)}")
        print(f"   Number of output scalers: {len(scaler_output)}")
        
        # Should have one scaler per pressure condition for input
        assert len(scaler_input) == 2
        # Output has 2 scalers but only first is fitted (convention)
        assert len(scaler_output) == 2
        
        # Verify input scalers are fitted
        for i, scaler in enumerate(scaler_input):
            assert hasattr(scaler, 'max_abs_'), f"Input scaler {i} not fitted"
            print(f"   Input scaler {i} max_abs_: {scaler.max_abs_}")
        
        # Only first output scaler is fitted (K-centered convention)
        assert hasattr(scaler_output[0], 'max_abs_'), "Output scaler 0 not fitted"
        print(f"   Output scaler 0 max_abs_: {scaler_output[0].max_abs_}")
        
        print(f"   ✅ All scalers properly fitted!")
    
    def test_json_format_compatibility(self, batch_json_file):
        """Test that our JSON loading matches k_centered_adaptive_learning.py format."""
        import json
        
        # Load JSON directly to check structure
        with open(batch_json_file, 'r') as f:
            data = json.load(f)
        
        print(f"\n🔍 JSON structure verification:")
        print(f"   Keys in JSON: {list(data.keys())}")
        
        # Should have required keys
        assert 'compositions' in data, "JSON must have 'compositions' key"
        
        # Should have either 'k_values' or 'parameter_sets'
        has_k_values = 'k_values' in data
        has_parameter_sets = 'parameter_sets' in data
        assert has_k_values or has_parameter_sets, "JSON must have 'k_values' or 'parameter_sets'"
        
        if has_parameter_sets:
            print(f"   ✅ Found 'parameter_sets' format (k_centered style)")
            # Extract k_values from parameter_sets
            k_values_from_ps = np.array([ps['k_values'] for ps in data['parameter_sets']])
            print(f"   k_values shape from parameter_sets: {k_values_from_ps.shape}")
        
        if has_k_values:
            print(f"   ✅ Found direct 'k_values' format")
            k_values_direct = np.array(data['k_values'])
            print(f"   k_values shape (direct): {k_values_direct.shape}")
        
        compositions = np.array(data['compositions'])
        print(f"   compositions shape: {compositions.shape}")
        
        # Now load with our dataset class
        dataset = MultiPressureDataset(
            src_file=batch_json_file,
            num_pressure_conditions=2,
            nspecies=3
        )
        
        x_data, y_data = dataset.get_data()
        
        print(f"\n✅ Format compatibility verified!")
        print(f"   Loaded {x_data.shape[0]} samples successfully")
    
    def test_json_with_max_rows(self, batch_json_file):
        """Test loading JSON with max_rows parameter."""
        max_rows = 100
        
        dataset = MultiPressureDataset(
            src_file=batch_json_file,
            num_pressure_conditions=2,
            nspecies=3,
            max_rows=max_rows
        )
        
        x_data, y_data = dataset.get_data()
        
        print(f"\n📏 max_rows test:")
        print(f"   Requested max_rows: {max_rows}")
        print(f"   Actual samples loaded: {x_data.shape[0]}")
        
        # Should have at most max_rows samples
        assert x_data.shape[0] <= max_rows
        assert y_data.shape[0] <= max_rows
        
        print(f"   ✅ max_rows parameter works correctly!")
    
    def test_json_scaler_reuse(self, batch_json_file):
        """Test that scalers can be reused across datasets (k_centered workflow)."""
        # Load first dataset to train scalers
        dataset1 = MultiPressureDataset(
            src_file=batch_json_file,
            num_pressure_conditions=2,
            nspecies=3,
            max_rows=500
        )
        
        scaler_input, scaler_output = dataset1.get_scalers()
        
        # Load SAME SUBSET with pre-trained scalers to verify they produce identical results
        dataset2 = MultiPressureDataset(
            src_file=batch_json_file,
            num_pressure_conditions=2,
            nspecies=3,
            max_rows=500,  # Same max_rows to compare same data
            scaler_input=scaler_input,
            scaler_output=scaler_output
        )
        
        x1, y1 = dataset1.get_data()
        x2, y2 = dataset2.get_data()
        
        print(f"\n🔄 Scaler reuse test:")
        print(f"   Dataset 1 samples: {x1.shape[0]}")
        print(f"   Dataset 2 samples: {x2.shape[0]}")
        
        # Same data with same scalers should produce identical results
        np.testing.assert_array_almost_equal(
            x1,
            x2,
            err_msg="Same samples with same scalers should have identical scaled values"
        )
        
        np.testing.assert_array_almost_equal(
            y1,
            y2,
            err_msg="Same samples with same scalers should have identical scaled values"
        )
        
        # Verify scalers are the same objects
        assert dataset2.scaler_input is scaler_input
        assert dataset2.scaler_output is scaler_output
        
        print(f"   ✅ Scaler reuse works correctly!")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
