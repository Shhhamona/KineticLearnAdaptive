#!/usr/bin/env python3
"""
Test script to verify LoKI is available and can run a basic simulation.
"""
import sys
sys.path.append('src')
from base_simulator import LoKISimulator
import numpy as np

print("Testing LoKI availability...")

# Test configuration
setup_file = "setup_O2_simple.in"
chem_file = "O2_simple_1.chem"
loki_path = "C:\\MyPrograms\\LoKI_v3.1.0-v2"
k_columns = [0, 1, 2]

# Create simulator
simulator = LoKISimulator(setup_file, chem_file, loki_path, k_columns)

# Test with a single simulation
print("\\nTesting single LoKI simulation...")
k_samples = np.array([[6e-16, 1.3e-15, 9.6e-16]])  # Reference values

try:
    results = simulator.run_simulations(k_samples)
    print(f"✓ LoKI simulation successful!")
    print(f"Results shape: {results.shape}")
    print(f"Sample results: {results[0][:5]}...")  # First 5 values
    
except Exception as e:
    print(f"✗ LoKI simulation failed: {e}")
    print("This might be expected if LoKI is not installed or configured properly.")
