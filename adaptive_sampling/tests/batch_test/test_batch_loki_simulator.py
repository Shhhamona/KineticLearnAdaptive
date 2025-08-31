"""
Test the batch simulation framework with real LoKI simulator.
"""
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from adaptive_sampling.src.base_simulator import LoKISimulator
from adaptive_sampling.src.batch_simulator import BatchSimulator, ParameterSet
from adaptive_sampling.src.sampling_strategies import BoundsBasedSampler, LatinHypercubeSampler
import numpy as np

def test_batch_loki_simulator():
    """Test the batch simulation framework with real LoKI simulator."""
    print("=" * 60)
    print("TESTING BATCH LOKI SIMULATOR FRAMEWORK")
    print("=" * 60)
    
    # 1. Create a LoKI simulator
    print("\n1. Setting up LoKISimulator...")
    k_columns = [0, 1, 2]  # Which K coefficients to vary (first 3)
    
    # Test configuration - change simulation_type here to test different types
    SIMULATION_TYPE = "simple"  # "simple" for O2_simple or "complex" for O2_novib

    # Set files based on simulation type
    if SIMULATION_TYPE == "simple":
        setup_file = "setup_O2_simple.in"
        chem_file = "O2_simple_1.chem"
    elif SIMULATION_TYPE == "complex":
        setup_file = "oxygen_chem_setup_novib.in"
        chem_file = "oxygen_novib.chem"
    else:
        raise ValueError(f"Unknown SIMULATION_TYPE: {SIMULATION_TYPE}")

    simulator = LoKISimulator(
        setup_file=setup_file,
        chem_file=chem_file,
        loki_path='C:\\MyPrograms\\LoKI_v3.1.0-v2',
        k_columns=k_columns,
        simulation_type=SIMULATION_TYPE,
        pressure_conditions=[133.322, 666.66]  # Test with 2 pressure conditions (1 Torr, 5 Torr)
    )
    
    print(f"🔬 Using {simulator.chemistry_name} simulation type")
    print(f"🔧 Pressure conditions: {simulator.pressure_conditions} Pa ({[p/133.322 for p in simulator.pressure_conditions]} Torr)")
    
    # Get reference K values from simulator
    print("Getting reference K values...")
    ref_k = simulator.get_reference_k_values()
    print(f"Reference K values: {ref_k}")
    print(f"Will vary K coefficients at indices: {k_columns}")
    
    # 2. Create sampling strategy for the K coefficients we want to vary
    print("\n2. Setting up sampling strategy...")
    k_to_vary = ref_k[k_columns]  # Extract the K values we want to vary
    
    # Create bounds around reference values (vary by factor of 1.5)
    bound_factor = 1.5
    k_bounds = np.array([[k/bound_factor, k*bound_factor] for k in k_to_vary])
    
    print(f"K bounds for variation:")
    for i, (k_idx, bounds) in enumerate(zip(k_columns, k_bounds)):
        print(f"  K{k_idx+1}: [{bounds[0]:.2e}, {bounds[1]:.2e}] (ref: {k_to_vary[i]:.2e})")
    
    # Use Latin Hypercube sampling for better space coverage
    sampler = BoundsBasedSampler(k_bounds, sampling_method='latin_hypercube')
    
    # 3. Create batch simulator
    print("\n3. Setting up BatchSimulator...")
    batch_sim = BatchSimulator(base_simulator=simulator, sampler=sampler)
    
    # 4. Test parameter set generation with pressure conditions
    print("\n4. Generating parameter sets...")
    parameter_sets = batch_sim.generate_parameter_sets(
        n_samples=6,  # Small number for testing
        k_bounds=k_bounds,
        pressure_conditions=[133.322, 666.66]  # Test with custom pressure conditions
    )
    
    print(f"Generated {len(parameter_sets)} parameter sets")
    print(f"Sample K values shape: {parameter_sets[0].k_values.shape}")
    print(f"Sample K values: {parameter_sets[0].k_values}")
    print(f"Pressure conditions: {parameter_sets[0].pressure_conditions} Pa")
    
    # 5. Run small batch first with pressure conditions (to test)
    print("\n5. Running small test batch with pressure conditions...")
    batch_results = batch_sim.run_with_sampling(
        n_samples=5, 
        k_bounds=k_bounds, 
        pressure_conditions=[133.322, 666.66],  # Test with 2 pressure conditions
        parallel_workers=1
    )  # Start with 1 parallel worker for testing!
    
    # 6. Print results
    print("\n" + "=" * 60)
    print("BATCH RESULTS")
    print("=" * 60)
    print(f"Total simulations: {batch_results.n_simulations}")
    print(f"Successful simulations: {batch_results.n_successful}")
    print(f"Success rate: {batch_results.success_rate:.1%}")
    print(f"Total execution time: {batch_results.total_time:.2f}s")
    print(f"Average time per simulation: {np.mean(batch_results.execution_times):.4f}s")
    print(f"Compositions shape: {batch_results.compositions.shape}")
    print(f"⚠️  Note: All simulations show same time - LoKI processes batches together")
    print(f"📊 Expected shape with 2 pressure conditions: (5 simulations × 2 pressures, n_species) = ({5*2}, n_species)")
    
    # Show some sample results
    print(f"\nSample composition values (first simulation):")
    print(f"  {batch_results.compositions[0, :5]}...")  # First 5 species
    
    # Analyze pressure effects if we have multiple pressure results
    if batch_results.compositions.shape[0] > len(batch_results.parameter_sets):
        n_pressures = len(batch_results.parameter_sets[0].pressure_conditions)
        n_simulations = len(batch_results.parameter_sets)
        print(f"\n📈 Pressure Effects Analysis:")
        print(f"   Simulations per pressure: {n_simulations}")
        print(f"   Total results: {batch_results.compositions.shape[0]}")
        
        # Compare first simulation at different pressures
        if batch_results.compositions.shape[0] >= 2:
            comp_p1 = batch_results.compositions[0, :3]  # First 3 species at pressure 1
            comp_p2 = batch_results.compositions[n_simulations, :3]  # Same simulation at pressure 2
            print(f"   First simulation at P1 ({batch_results.parameter_sets[0].pressure_conditions[0]:.1f} Pa): {comp_p1}")
            print(f"   First simulation at P2 ({batch_results.parameter_sets[0].pressure_conditions[1]:.1f} Pa): {comp_p2}")
            print(f"   Pressure ratio effect: {comp_p2/comp_p1}")
    
    # 7. Get overall summary
    print("\n7. Overall batch summary...")
    summary = batch_sim.get_batch_summary()
    print(f"Total batches run: {summary['total_batches']}")
    print(f"Total simulations: {summary['total_simulations']}")
    print(f"Overall success rate: {summary['overall_success_rate']:.1%}")
    print(f"Average time per simulation: {summary['average_time_per_simulation']:.4f}s")
    
    # 8. Save results with automatic organization
    print("\n8. Saving results...")
    
    # Save the batch results (automatic path generation)
    batch_sim.save_batch_results(batch_results)  # No filepath needed - auto-organized!
    
    # 9. Analysis of K-Composition relationship
    if batch_results.success_rate > 0:
        print("\n9. K-Composition relationship analysis...")
        successful_idx = batch_results.success_mask
        if np.any(successful_idx):
            successful_k = np.array([ps.k_values for ps in batch_results.parameter_sets])[successful_idx]
            
            # Handle multiple pressure conditions
            n_pressures = len(batch_results.parameter_sets[0].pressure_conditions) if batch_results.parameter_sets else 1
            n_simulations = len(batch_results.parameter_sets)
            
            print(f"Successful simulations: {np.sum(successful_idx)}")
            print(f"Pressure conditions per simulation: {n_pressures}")
            print(f"Total composition results: {batch_results.compositions.shape[0]}")
            
            # For multi-pressure analysis, we need to handle the expanded composition array
            if batch_results.compositions.shape[0] == n_simulations * n_pressures:
                print(f"✅ Detected multi-pressure results: {n_simulations} simulations × {n_pressures} pressures")
                
                # Analyze only the first pressure condition for K-value relationship
                first_pressure_compositions = batch_results.compositions[:n_simulations]
                successful_compositions = first_pressure_compositions[successful_idx]
                
                print(f"K value ranges in successful runs (at first pressure):")
            else:
                # Single pressure case
                successful_compositions = batch_results.compositions[successful_idx]
                print(f"K value ranges in successful runs:")
            for i, k_idx in enumerate(k_columns):
                k_min, k_max = successful_k[:, i].min(), successful_k[:, i].max()
                print(f"  K{k_idx+1}: [{k_min:.2e}, {k_max:.2e}]")
            
            print(f"Composition ranges:")
            # Ensure compositions are numeric
            if successful_compositions.size > 0 and np.issubdtype(successful_compositions.dtype, np.number):
                comp_min, comp_max = successful_compositions.min(axis=0), successful_compositions.max(axis=0)
                for i in range(min(5, len(comp_min))):  # Show first 5 species
                    print(f"  Species {i+1}: [{comp_min[i]:.2e}, {comp_max[i]:.2e}]")
            else:
                print("  Composition data not available or not numeric")
    
    print("\n✅ Batch LoKI simulator test completed!")
    return batch_results

if __name__ == "__main__":
    # First test basic functionality
    test_batch_loki_simulator()
