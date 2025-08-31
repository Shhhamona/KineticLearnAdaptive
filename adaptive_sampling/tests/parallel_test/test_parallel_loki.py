"""
Test for parallel LoKI batch simulation - both simple and complex types.
Tests if we can run simulations with parallel workers for both simulation types.
"""
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from adaptive_sampling.src.base_simulator import LoKISimulator
from adaptive_sampling.src.batch_simulator import BatchSimulator
from adaptive_sampling.src.sampling_strategies import BoundsBasedSampler
import numpy as np
import numpy as np

def test_parallel_loki_batch(simulation_type="simple"):
    """Test parallel LoKI batch simulation for specified simulation type."""
    
    # Configure based on simulation type
    if simulation_type == "simple":
        print("🧪 Testing Simple (O2_simple) Parallel LoKI Batch")
        setup_file = "setup_O2_simple.in"
        chem_file = "O2_simple_1.chem"
        k_columns = [0, 1, 2]  # Simple has fewer K coefficients
    else:  # complex
        print("🧪 Testing Complex (O2_novib) Parallel LoKI Batch")
        setup_file = "oxygen_chem_setup_novib.in"
        chem_file = "oxygen_novib.chem"
        k_columns = [0, 1, 2, 3, 4]  # Complex has more K coefficients
    
    print("=" * 50)
    print(f"📊 Test: {simulation_type} simulation type with 3 parallel workers")
    print()

    # 1. Setup LoKI simulator with multiple pressure conditions
    print(f"1. Setting up {simulation_type} LoKI simulator...")
    
    # Define pressure conditions: 1 Torr and 5 Torr
    pressure_conditions = [133.322, 666.66]  # Pa (1 Torr, 5 Torr)
    
    simulator = LoKISimulator(
        setup_file=setup_file,
        chem_file=chem_file,
        loki_path='C:\\MyPrograms\\LoKI_v3.1.0-v2',
        k_columns=k_columns,
        simulation_type=simulation_type,
        pressure_conditions=pressure_conditions
    )
    
    print(f"🔬 Using {simulator.chemistry_name} chemistry")
    print(f"📁 Setup: {setup_file}")
    print(f"🧪 Chemistry: {chem_file}")
    print(f"🌡️ Pressure conditions: {pressure_conditions} Pa ({[p/133.322 for p in pressure_conditions]} Torr)")
    
    # 2. Setup sampling strategy
    ref_k = simulator.get_reference_k_values()
    k_to_vary = ref_k[k_columns]
    
    # Simple bounds: ±50% around reference values
    k_bounds = np.array([[k*0.5, k*1.5] for k in k_to_vary])
    
    print(f"📍 Varying K coefficients: {k_columns}")
    print(f"📊 K bounds:")
    for i, bounds in enumerate(k_bounds):
        print(f"   K{k_columns[i]}: [{bounds[0]:.2e}, {bounds[1]:.2e}]")
    
    # 3. Create batch simulator
    sampler = BoundsBasedSampler(k_bounds)
    batch_sim = BatchSimulator(base_simulator=simulator, sampler=sampler)
    
    # 4. Run the test!
    print(f"\n🚀 Running 5 simulations × 2 pressures = 10 total outputs with 3 parallel workers...")
    
    try:
        results = batch_sim.run_with_sampling(
            n_samples=5,
            k_bounds=k_bounds,
            parallel_workers=3,  # Key test: 3 parallel workers!
            pressure_conditions=pressure_conditions  # Multi-pressure test!
        )
        
        # 5. Report results
        print(f"\n✅ TEST COMPLETED!")
        print(f"=" * 30)
        print(f"Total simulations: {results.n_simulations}")
        print(f"Successful: {results.n_successful}")
        print(f"Success rate: {results.success_rate:.1%}")
        print(f"Total time: {results.total_time:.2f}s")
        print(f"Avg time per sim: {results.total_time/results.n_simulations:.2f}s")
        print(f"Output shape: {results.compositions.shape}")
        print(f"Expected shape: (10, n_species) = 5 simulations × 2 pressures")
        print(f"Parallel workers used: {results.metadata.get('parallel_workers', 'unknown')}")
        print(f"Pressure conditions: {results.metadata.get('pressure_conditions_pa', 'unknown')} Pa")
        print(f"Pressure conditions: {results.metadata.get('pressure_conditions_torr', 'unknown')} Torr")
        
        # 6. Save simulation data
        print("\n6. Saving simulation results...")
        batch_sim.save_batch_results(results)
        print(f"✅ Results saved to organized directory structure")
        
        # 7. Show some sample data analysis (account for multi-pressure results)
        if results.success_rate > 0:
            print("\n7. Sample data analysis...")
            successful_idx = results.success_mask
            if np.any(successful_idx):
                successful_k = np.array([ps.k_values for ps in results.parameter_sets])[successful_idx]
                
                # Handle multi-pressure results: compositions shape is (n_simulations * n_pressures, n_species)
                n_pressures = len(pressure_conditions)
                n_simulations = len(results.parameter_sets)
                expected_total_outputs = n_simulations * n_pressures
                
                print(f"Successful simulations: {np.sum(successful_idx)}")
                print(f"Expected total outputs: {expected_total_outputs} (from {n_simulations} simulations × {n_pressures} pressures)")
                print(f"Actual composition outputs: {results.compositions.shape[0]}")
                
                # Only analyze if we have multi-pressure detection logic
                if results.compositions.shape[0] == expected_total_outputs:
                    print("✅ Multi-pressure outputs detected correctly!")
                    
                    # Analyze compositions by pressure condition
                    for p_idx, pressure in enumerate(pressure_conditions):
                        pressure_start = p_idx * n_simulations
                        pressure_end = pressure_start + n_simulations
                        pressure_compositions = results.compositions[pressure_start:pressure_end]
                        
                        print(f"\n📊 Pressure {pressure:.1f} Pa ({pressure/133.322:.1f} Torr):")
                        comp_min, comp_max = pressure_compositions.min(axis=0), pressure_compositions.max(axis=0)
                        for i in range(min(3, len(comp_min))):
                            print(f"   Species {i+1}: [{comp_min[i]:.2e}, {comp_max[i]:.2e}]")
                else:
                    print("⚠️  Unexpected composition shape - using simple analysis")
                    successful_compositions = results.compositions[successful_idx]
                
                print(f"K value ranges in successful runs:")
                for i in range(len(successful_k[0])):
                    k_min, k_max = successful_k[:, i].min(), successful_k[:, i].max()
                    print(f"  K{i+1}: [{k_min:.2e}, {k_max:.2e}]")
        
        # 8. Overall assessment
        if results.success_rate > 0.8:  # 80% success
            print(f"\n🎉 SUCCESS: Parallel LoKI batch works great!")
            print(f"   👍 Ready for production use with 3 workers")
        elif results.success_rate > 0.5:  # 50% success  
            print(f"\n⚠️  PARTIAL SUCCESS: Some issues but mostly working")
            print(f"   💡 May need debugging or reduced workers")
        else:
            print(f"\n❌ FAILED: Low success rate")
            print(f"   🔧 Need to debug the parallel implementation")
            
    except Exception as e:
        print(f"\n❌ TEST FAILED with exception:")
        print(f"   Error: {e}")
        print(f"   🔧 Check the parallel implementation")

def test_both_simulation_types():
    """Test both simple and complex simulation types."""
    print("🚀 Testing Both Simulation Types with Parallel Processing")
    print("=" * 60)
    
    # Test simple first (as requested)
    print("\n1️⃣ Testing Simple (O2_simple) simulations:")
    try:
        test_parallel_loki_batch("simple")
        print("\n✅ Simple simulations: PASSED")
    except Exception as e:
        print(f"\n❌ Simple simulations: FAILED - {e}")
    
    print("\n" + "=" * 60)
    
    # Test complex
    print("\n2️⃣ Testing Complex (O2_novib) simulations:")
    try:
        test_parallel_loki_batch("complex")
        print("\n✅ Complex simulations: PASSED")
    except Exception as e:
        print(f"\n❌ Complex simulations: FAILED - {e}")

if __name__ == "__main__":
    # Test simple simulation type first (as requested)
    print("🎯 Testing Simple Simulation Type First")
    test_parallel_loki_batch("simple")
