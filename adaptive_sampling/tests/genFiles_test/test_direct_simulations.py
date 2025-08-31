"""
Direct test for genFiles parallel functionality - both simple and complex.
Tests if we can run simulations with parallel workers directly for both simulation types.
"""
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add genFiles path
genfiles_path = os.path.join(project_root, 'other_scripts', 'genFiles')
sys.path.append(genfiles_path)

import numpy as np

SIMULATION_TYPE = "complex"

def test_parallel_genfiles_directly(simulation_type="complex"):
    """Direct test: simulations with parallel workers using genFiles directly."""
    
    # Import appropriate module based on simulation type
    if simulation_type == "simple":
        from other_scripts.genFiles.genFiles_O2_simple import Simulations
        print("🧪 Testing genFiles_O2_simple Parallel Directly")
        setup_file = "setup_O2_simple.in"
        chem_file = "O2_simple_1.chem"
        # Simple doesn't use K columns variation
        k_columns = None
        k_true_values = None
    else:  # complex
        from other_scripts.genFiles.genFiles_O2_novib import Simulations
        print("🧪 Testing genFiles_O2_novib Parallel Directly")
        setup_file = "oxygen_chem_setup_novib.in"
        chem_file = "oxygen_novib.chem"
        # Complex uses K columns variation
        k_columns = [0, 1, 2, 3, 4]
        k_true_values = [7.6e-22, 3E-44, 4e-20, 4e-20, 1e-16]
    
    print("=" * 50)
    print(f"📊 Test: {simulation_type} simulation type with 3 parallel workers")
    print()

    # Setup paths and parameters
    loki_path = "C:\\MyPrograms\\LoKI_v3.1.0-v2"
    n_simulations = 5
    
    # Create simulation object directly
    simul = Simulations(setup_file, chem_file, loki_path, n_simulations)
    
    print(f"📁 Setup file: {setup_file}")
    print(f"🧪 Chemistry file: {chem_file}")
    print(f"📊 Number of simulations: {n_simulations}")
    
    if simulation_type == "complex":
        print(f"📍 Varying K coefficients: {k_columns}")
        
        # Generate K set for complex simulations
        simul.set_ChemFile_ON()
        simul.random_kset(k_columns, k_true_values, krange=[0.5, 2], pdf_function='uniform')
        
        print(f"📊 K values generated: {simul.parameters.k_set.shape}")
    else:
        print("📍 Simple simulation - no K coefficient variation")
    
    print()
    
    # Test with 3 parallel workers
    print(f"🚀 Running {simulation_type} simulations with 3 parallel workers...")
    
    try:
        # This should use our new parallel implementation
        simul.runSimulations(parallel_workers=3)
        
        print(f"\n✅ PARALLEL TEST COMPLETED!")
        print(f"=" * 40)
        print(f"🎉 SUCCESS: genFiles_{simulation_type} parallel execution works!")
        print(f"   📈 Ready for production use with 3 workers")
        
    except Exception as e:
        print(f"\n❌ PARALLEL TEST FAILED:")
        print(f"   Error: {e}")
        print(f"   🔧 Check the parallel implementation in genFiles module")
        raise

if __name__ == "__main__":
    # Test simple simulation type specifically
    print("🎯 Testing Simple Simulation Type with Parallel Processing")
    test_parallel_genfiles_directly(SIMULATION_TYPE)
