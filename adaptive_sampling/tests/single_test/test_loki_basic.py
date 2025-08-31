#!/usr/bin/env python3
"""
Test script to verify LoKI is available and can run a basic simulation.
"""

from adaptive_sampling.src.base_simulator import LoKISimulator
import numpy as np

print("Testing LoKI availability...")


# Test configuration - change simulation_type here to test different types
SIMULATION_TYPE = "simple"  # "simple" for O2_simple or "complex" for O2_novib

# Set files based on simulation type
if SIMULATION_TYPE == "simple":
    setup_file = "setup_O2_simple.in"
    chem_file = "O2_simple_1.chem"
elif SIMULATION_TYPE == "complex":
    setup_file = "oxygen_chem_setup_novib.in"  # Uncomment and use for complex
    chem_file = "oxygen_novib.chem"             # Uncomment and use for complex
else:
    raise ValueError(f"Unknown SIMULATION_TYPE: {SIMULATION_TYPE}")

loki_path = "C:\\MyPrograms\\LoKI_v3.1.0-v2"
k_columns = [0, 1, 2]

# Create simulator with specified type and pressure conditions
simulator = LoKISimulator(setup_file, chem_file, loki_path, k_columns, 
                         simulation_type=SIMULATION_TYPE,
                         pressure_conditions=[133.322, 666.66])  # Test with 2 pressure conditions

# Test with a single simulation
print(f"\nTesting single LoKI simulation ({SIMULATION_TYPE})...")
k_samples = np.array([[9.941885789401E-16, 1.800066252209E-15, 1.380839580124E-15]])  # Reference values

try:
    results = simulator.run_simulations(k_samples)
    print(f"✅ LoKI simulation successful!")
    print(f"   📊 Results shape: {results.shape}")
    print(f"   🧬 Chemistry: {simulator.chemistry_name}")
    print(f"   📁 Results dir: {simulator.results_dir}")
    print(f"   🔢 Sample results: {results[0][:5]}...")  # First 5 values
    
except Exception as e:
    print(f"❌ LoKI simulation failed: {e}")
    print("This might be expected if LoKI is not installed or configured properly.")
