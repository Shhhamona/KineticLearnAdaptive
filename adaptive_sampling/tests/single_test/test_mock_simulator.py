"""
Test script for the MockSimulator and basic functionality.
"""
#import sys
#import os
# Add the adaptive_sampling/src directory to path
#sys.path.append(os.path.join('..', 'src'))

from adaptive_sampling.src.base_simulator import MockSimulator
import numpy as np

def test_mock_simulator():
    """Test the MockSimulator basic functionality."""
    print('Testing MockSimulator...')
    
    # Test single simulation
    mock_sim = MockSimulator('setup.in', 'chem.chem', 'path/to/loki')
    k_test = np.array([[1e-15, 2e-15, 1e-16]])
    result = mock_sim.run_simulations(k_test)
    print(f'Single simulation result shape: {result.shape}')
    print(f'Sample result: {result[0, :3]}')
    
    # Test batch simulations
    k_batch = np.random.uniform(1e-16, 1e-14, (5, 3))
    results = mock_sim.run_simulations(k_batch)
    print(f'Batch simulation result shape: {results.shape}')
    
    # Test reference K values
    ref_k = mock_sim.get_reference_k_values()
    print(f'Reference K values shape: {ref_k.shape}')
    
    print('MockSimulator test passed!')
    return True

if __name__ == "__main__":
    test_mock_simulator()
