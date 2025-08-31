"""
Test the pure batch simulation framework.
"""
import sys
import os


from adaptive_sampling.src.base_simulator import MockSimulator
from adaptive_sampling.src.batch_simulator import BatchSimulator, ParameterSet
from adaptive_sampling.src.sampling_strategies import BoundsBasedSampler, LatinHypercubeSampler
import numpy as np

def test_batch_simulator():
    """Test the batch simulation framework."""
    print("=" * 60)
    print("TESTING BATCH SIMULATOR FRAMEWORK")
    print("=" * 60)
    
    # 1. Create a simulator
    print("\n1. Setting up MockSimulator...")
    true_k = np.array([6E-16, 1.3E-15, 9.6E-16, 2.2E-15, 7E-22])  # 5 K coefficients
    simulator = MockSimulator('setup.in', 'chem.chem', 'path/to/loki', true_k=true_k)
    
    # 2. Create a sampling strategy
    print("\n2. Setting up sampling strategy...")
    # Define bounds for first 3 K coefficients (vary by factor of 2)
    k_to_vary = true_k[:3]  # Vary first 3 coefficients
    k_bounds = np.array([[k/2, k*2] for k in k_to_vary])
    
    sampler = BoundsBasedSampler(k_bounds, sampling_method='latin_hypercube')
    
    # 3. Create batch simulator
    print("\n3. Setting up BatchSimulator...")
    batch_sim = BatchSimulator(base_simulator=simulator, sampler=sampler)
    
    # 4. Test parameter set generation
    print("\n4. Generating parameter sets...")
    parameter_sets = batch_sim.generate_parameter_sets(
        n_samples=10,
        k_bounds=k_bounds
    )
    
    print(f"Generated {len(parameter_sets)} parameter sets")
    print(f"Sample K values shape: {parameter_sets[0].k_values.shape}")
    print(f"Sample K values: {parameter_sets[0].k_values}")
    
    # 5. Run simple batch
    print("\n5. Running simple batch...")
    batch_results = batch_sim.run_with_sampling(n_samples=15, k_bounds=k_bounds)
    
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
    
    # 7. Test with multiple files
    print("\n6. Testing with multiple file combinations...")
    setup_files = ['setup1.in', 'setup2.in']
    chem_files = ['chem1.chem', 'chem2.chem'] 
    
    multi_file_results = batch_sim.run_with_sampling(
        n_samples=8,  # Will cycle through file combinations
        k_bounds=k_bounds,
        setup_files=setup_files,
        chem_files=chem_files
    )
    
    print(f"Multi-file batch: {multi_file_results.n_simulations} simulations")
    print(f"File combinations used: {multi_file_results.metadata['unique_file_combinations']}")
    
    # Show file distribution
    file_combos = {}
    for ps in multi_file_results.parameter_sets:
        key = f"{ps.setup_file}/{ps.chem_file}"
        file_combos[key] = file_combos.get(key, 0) + 1
    
    print("File combination distribution:")
    for combo, count in file_combos.items():
        print(f"  {combo}: {count} simulations")
    
    # 8. Get overall summary
    print("\n7. Overall batch summary...")
    summary = batch_sim.get_batch_summary()
    print(f"Total batches run: {summary['total_batches']}")
    print(f"Total simulations: {summary['total_simulations']}")
    print(f"Overall success rate: {summary['overall_success_rate']:.1%}")
    print(f"Average time per simulation: {summary['average_time_per_simulation']:.4f}s")
    
    # 9. Save results with automatic organization
    print("\n8. Saving results...")
    
    # Save the best batch result with automatic path generation
    best_batch = max(batch_sim.batch_history, key=lambda b: b.success_rate)
    batch_sim.save_batch_results(best_batch)  # No filepath needed - auto-organized!
    
    print("\n✅ Batch simulator test completed!")
    return batch_results

if __name__ == "__main__":
    test_batch_simulator()
