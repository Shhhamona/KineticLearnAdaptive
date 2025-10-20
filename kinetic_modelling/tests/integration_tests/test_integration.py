"""
Integration tests using real data files.

These tests verify that the data module works correctly with actual
kinetic modeling data from the O2 simple mechanism.
"""

import numpy as np
import pytest
from pathlib import Path

# Add parent directory to path to import kinetic_modelling
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append('.')
from kinetic_modelling.data import MultiPressureDataset


# Real data file paths
# Path from: kinetic_modelling/tests/integration_tests/test_integration.py
# To:        data/SampleEfficiency/
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "SampleEfficiency"
TRAIN_FILE = DATA_DIR / "O2_simple_uniform.txt"
TEST_FILE = DATA_DIR / "O2_simple_test.txt"

# O2 simple mechanism parameters (from sample_effiency.py)
NSPECIES = 3  # O2, O, O3
NUM_PRESSURES = 2  # Two pressure conditions
N_REACTIONS = 3  # Three reactions (selected from 10 total in file)
REACT_IDX = np.array([0, 1, 2])  # Select first 3 reaction columns


@pytest.fixture
def check_data_files():
    """Check that required data files exist."""
    if not TRAIN_FILE.exists():
        pytest.skip(f"Training file not found: {TRAIN_FILE}")
    if not TEST_FILE.exists():
        pytest.skip(f"Test file not found: {TEST_FILE}")
    return str(TRAIN_FILE), str(TEST_FILE)


class TestRealDataLoading:
    """Test loading real O2 simple mechanism data."""
    
    def test_load_real_training_dataset(self, check_data_files):
        """Test loading real training data."""
        train_file, _ = check_data_files
        
        dataset = MultiPressureDataset(
            nspecies=NSPECIES,
            num_pressure_conditions=NUM_PRESSURES,
            src_file=train_file,
            react_idx=REACT_IDX
        )
        
        # Check basic properties
        assert len(dataset) > 0
        print(f"\nLoaded {len(dataset)} samples from training file")
        
        # Check data shapes
        x, y = dataset.get_data()
        assert x.shape[1] == NSPECIES * NUM_PRESSURES
        assert y.shape[1] == N_REACTIONS
        print(f"Input shape: {x.shape}, Output shape: {y.shape}")
        
        # Check data is properly scaled (should be in [-1, 1] range roughly)
        assert np.abs(x).max() <= 1.0
        assert np.abs(y).max() <= 1.0
        print(f"Data scaling OK: x range [{x.min():.3f}, {x.max():.3f}], y range [{y.min():.3f}, {y.max():.3f}]")
    
    def test_load_train_test_with_shared_scalers(self, check_data_files):
        """Test loading both training and test datasets with shared scalers."""
        train_file, test_file = check_data_files
        
        # Create train dataset first
        train_dataset = MultiPressureDataset(
            nspecies=NSPECIES,
            num_pressure_conditions=NUM_PRESSURES,
            src_file=train_file,
            react_idx=REACT_IDX
        )
        
        # Create test dataset with train scalers
        test_dataset = MultiPressureDataset(
            nspecies=NSPECIES,
            num_pressure_conditions=NUM_PRESSURES,
            src_file=test_file,
            react_idx=REACT_IDX,
            scaler_input=train_dataset.scaler_input,
            scaler_output=train_dataset.scaler_output
        )
        
        # Verify both datasets loaded
        assert len(train_dataset) > 0
        assert len(test_dataset) > 0
        print(f"\nTrain samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        
        # Verify same dimensions
        x_train, y_train = train_dataset.get_data()
        x_test, y_test = test_dataset.get_data()
        assert x_train.shape[1] == x_test.shape[1]
        assert y_train.shape[1] == y_test.shape[1]
        
        # Verify scalers are shared
        assert train_dataset.scaler_input is test_dataset.scaler_input
        assert train_dataset.scaler_output is test_dataset.scaler_output
        print("✓ Scalers properly shared between train and test datasets")
    
    def test_dataset_statistics(self, check_data_files):
        """Test and display statistics about the loaded dataset."""
        train_file, _ = check_data_files
        
        dataset = MultiPressureDataset(
            nspecies=NSPECIES,
            num_pressure_conditions=NUM_PRESSURES,
            src_file=train_file,
            react_idx=REACT_IDX
        )
        
        x, y = dataset.get_data()
        
        print(f"\n{'='*60}")
        print(f"Dataset Statistics for O2 Simple Mechanism")
        print(f"{'='*60}")
        print(f"Number of samples: {len(dataset)}")
        print(f"Number of species: {NSPECIES}")
        print(f"Number of pressure conditions: {NUM_PRESSURES}")
        print(f"Number of reactions: {N_REACTIONS}")
        print(f"\nInput features (scaled densities):")
        print(f"  Shape: {x.shape}")
        print(f"  Mean: {x.mean():.4f}, Std: {x.std():.4f}")
        print(f"  Min: {x.min():.4f}, Max: {x.max():.4f}")
        print(f"\nOutput targets (scaled reaction rates):")
        print(f"  Shape: {y.shape}")
        print(f"  Mean: {y.mean():.4f}, Std: {y.std():.4f}")
        print(f"  Min: {y.min():.4f}, Max: {y.max():.4f}")
        print(f"{'='*60}\n")
        
        # Verify reasonable statistics
        assert x.std() > 0.1  # Data should have some variance
        assert y.std() > 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
