"""
Tests for the sampling module.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetic_modelling.data import MultiPressureDataset
from kinetic_modelling.sampling import BaseSampler, SequentialSampler, RandomSampler


@pytest.fixture
def sample_dataset_file():
    """Create a temporary dataset file for testing."""
    # Create sample data: 100 rows
    n_samples = 50  # 50 simulations
    n_reactions = 3
    n_species = 5
    n_pressures = 2
    
    np.random.seed(42)
    
    # Generate data
    reactions = np.random.uniform(1e-18, 1e-16, (n_samples * n_pressures, n_reactions))
    densities = np.random.uniform(1e14, 1e18, (n_samples * n_pressures, n_species))
    
    # Combine
    data = np.hstack([reactions, densities])
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("# Sample kinetic data\n")
        f.write("# k1  k2  k3  dens1  dens2  dens3  dens4  dens5\n")
        
        for row in data:
            f.write("  ".join([f"{val:.6e}" for val in row]) + "\n")
        
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    import os
    os.unlink(temp_path)


@pytest.fixture
def sample_dataset(sample_dataset_file):
    """Create a MultiPressureDataset for testing."""
    dataset = MultiPressureDataset(
        nspecies=5,
        num_pressure_conditions=2,
        src_file=sample_dataset_file
    )
    return dataset


class TestSequentialSampler:
    """Tests for SequentialSampler class."""
    
    def test_initialization(self):
        """Test sampler initialization."""
        sampler = SequentialSampler()
        assert sampler is not None
        assert sampler.sampler_name == "sequential"
    
    def test_basic_sampling(self, sample_dataset):
        """Test basic sequential sampling."""
        sampler = SequentialSampler()
        
        # Sample 20 from 50
        sampled_dataset = sampler.sample(sample_dataset, n_samples=20)
        
        assert len(sampled_dataset) == 20
        assert sampled_dataset.num_pressure_conditions == 2
        assert sampled_dataset.nspecies == 5
        
        # Check data shapes
        x, y = sampled_dataset.get_data()
        assert x.shape == (20, 10)  # 20 samples, 2 pressures * 5 species
        assert y.shape == (20, 3)   # 20 samples, 3 reactions
    
    def test_sampling_preserves_order(self, sample_dataset):
        """Test that sequential sampling without shuffle preserves order."""
        sampler = SequentialSampler()
        
        # Get original first 10 samples
        x_orig, y_orig = sample_dataset.get_data()
        
        # Sample first 10
        sampled_dataset = sampler.sample(sample_dataset, n_samples=10, shuffle=False)
        x_sampled, y_sampled = sampled_dataset.get_data()
        
        # Should be identical
        np.testing.assert_array_almost_equal(x_sampled, x_orig[:10])
        np.testing.assert_array_almost_equal(y_sampled, y_orig[:10])
    
    def test_sampling_with_shuffle(self, sample_dataset):
        """Test sequential sampling with shuffling."""
        sampler = SequentialSampler()
        
        # Sample with shuffle and seed
        sampled_dataset1 = sampler.sample(sample_dataset, n_samples=20, shuffle=True, seed=42)
        sampled_dataset2 = sampler.sample(sample_dataset, n_samples=20, shuffle=True, seed=42)
        
        # Should be reproducible
        x1, y1 = sampled_dataset1.get_data()
        x2, y2 = sampled_dataset2.get_data()
        
        np.testing.assert_array_almost_equal(x1, x2)
        np.testing.assert_array_almost_equal(y1, y2)
        
        # Should be different from non-shuffled
        sampled_dataset3 = sampler.sample(sample_dataset, n_samples=20, shuffle=False)
        x3, y3 = sampled_dataset3.get_data()
        
        # At least some difference expected (not all identical)
        assert not np.allclose(x1, x3) or not np.allclose(y1, y3)
    
    def test_invalid_n_samples(self, sample_dataset):
        """Test error handling for invalid n_samples."""
        sampler = SequentialSampler()
        
        # Too many samples
        with pytest.raises(ValueError, match="Cannot sample"):
            sampler.sample(sample_dataset, n_samples=100)
        
        # Zero samples
        with pytest.raises(ValueError, match="must be positive"):
            sampler.sample(sample_dataset, n_samples=0)
        
        # Negative samples
        with pytest.raises(ValueError, match="must be positive"):
            sampler.sample(sample_dataset, n_samples=-10)
    
    def test_full_dataset_sampling(self, sample_dataset):
        """Test sampling the entire dataset."""
        sampler = SequentialSampler()
        
        # Sample all
        sampled_dataset = sampler.sample(sample_dataset, n_samples=50)
        
        assert len(sampled_dataset) == 50
        
        # Should be identical to original
        x_orig, y_orig = sample_dataset.get_data()
        x_sampled, y_sampled = sampled_dataset.get_data()
        
        np.testing.assert_array_almost_equal(x_sampled, x_orig)
        np.testing.assert_array_almost_equal(y_sampled, y_orig)


class TestRandomSampler:
    """Tests for RandomSampler class."""
    
    def test_initialization(self):
        """Test sampler initialization."""
        sampler = RandomSampler()
        assert sampler is not None
        assert sampler.sampler_name == "random"
    
    def test_basic_sampling(self, sample_dataset):
        """Test basic random sampling."""
        sampler = RandomSampler()
        
        # Sample 20 from 50
        sampled_dataset = sampler.sample(sample_dataset, n_samples=20, seed=42)
        
        assert len(sampled_dataset) == 20
        assert sampled_dataset.num_pressure_conditions == 2
        assert sampled_dataset.nspecies == 5
        
        # Check data shapes
        x, y = sampled_dataset.get_data()
        assert x.shape == (20, 10)
        assert y.shape == (20, 3)
    
    def test_sampling_reproducibility(self, sample_dataset):
        """Test that random sampling with seed is reproducible."""
        sampler = RandomSampler()
        
        # Sample twice with same seed
        sampled_dataset1 = sampler.sample(sample_dataset, n_samples=20, seed=42)
        sampled_dataset2 = sampler.sample(sample_dataset, n_samples=20, seed=42)
        
        x1, y1 = sampled_dataset1.get_data()
        x2, y2 = sampled_dataset2.get_data()
        
        np.testing.assert_array_almost_equal(x1, x2)
        np.testing.assert_array_almost_equal(y1, y2)
    
    def test_sampling_different_seeds(self, sample_dataset):
        """Test that different seeds produce different samples."""
        sampler = RandomSampler()
        
        # Sample with different seeds
        sampled_dataset1 = sampler.sample(sample_dataset, n_samples=20, seed=42)
        sampled_dataset2 = sampler.sample(sample_dataset, n_samples=20, seed=123)
        
        x1, y1 = sampled_dataset1.get_data()
        x2, y2 = sampled_dataset2.get_data()
        
        # Should be different
        assert not np.allclose(x1, x2) or not np.allclose(y1, y2)
    
    def test_sampling_with_shuffle(self, sample_dataset):
        """Test random sampling with initial shuffling."""
        sampler = RandomSampler()
        
        # Sample with and without shuffle
        sampled_dataset1 = sampler.sample(sample_dataset, n_samples=20, shuffle=True, seed=42)
        sampled_dataset2 = sampler.sample(sample_dataset, n_samples=20, shuffle=False, seed=42)
        
        x1, y1 = sampled_dataset1.get_data()
        x2, y2 = sampled_dataset2.get_data()
        
        # Should be different (shuffle changes the pool)
        assert not np.allclose(x1, x2) or not np.allclose(y1, y2)
    
    def test_sampling_without_replacement(self, sample_dataset):
        """Test that random sampling is without replacement."""
        sampler = RandomSampler()
        
        # Sample half the dataset
        sampled_dataset = sampler.sample(sample_dataset, n_samples=25, seed=42)
        x, y = sampled_dataset.get_data()
        
        # Check for unique rows (no duplicates)
        unique_x = np.unique(x, axis=0)
        assert len(unique_x) == 25
    
    def test_invalid_n_samples(self, sample_dataset):
        """Test error handling for invalid n_samples."""
        sampler = RandomSampler()
        
        # Too many samples
        with pytest.raises(ValueError, match="Cannot sample"):
            sampler.sample(sample_dataset, n_samples=100)
        
        # Zero samples
        with pytest.raises(ValueError, match="must be positive"):
            sampler.sample(sample_dataset, n_samples=0)


class TestSamplerComparison:
    """Tests comparing different samplers."""
    
    def test_samplers_produce_valid_datasets(self, sample_dataset):
        """Test that all samplers produce valid datasets."""
        samplers = [
            SequentialSampler(),
            RandomSampler()
        ]
        
        for sampler in samplers:
            sampled = sampler.sample(sample_dataset, n_samples=15, seed=42)
            
            # Check basic properties
            assert len(sampled) == 15
            assert sampled.num_pressure_conditions == 2
            assert sampled.nspecies == 5
            
            # Check scalers are preserved
            input_scalers_orig, output_scalers_orig = sample_dataset.get_scalers()
            input_scalers_sampled, output_scalers_sampled = sampled.get_scalers()
            
            assert input_scalers_orig is input_scalers_sampled
            assert output_scalers_orig is output_scalers_sampled
    
    def test_sequential_vs_random(self, sample_dataset):
        """Test that sequential and random samplers produce different results."""
        seq_sampler = SequentialSampler()
        rand_sampler = RandomSampler()
        
        # Sample with both
        seq_dataset = seq_sampler.sample(sample_dataset, n_samples=20, shuffle=False)
        rand_dataset = rand_sampler.sample(sample_dataset, n_samples=20, seed=42)
        
        x_seq, y_seq = seq_dataset.get_data()
        x_rand, y_rand = rand_dataset.get_data()
        
        # Should be different (with high probability)
        assert not np.allclose(x_seq, x_rand) or not np.allclose(y_seq, y_rand)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
