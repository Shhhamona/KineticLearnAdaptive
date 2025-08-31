"""
Batch simulation framework for running multiple simulations with parameter variations.

This module provides a clean abstraction for running batches of simulations
with different parameter sets, using various sampling strategies.
Separate from ML models and adaptive approaches.
"""

import numpy as np
import os
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
# Removed: ProcessPoolExecutor, multiprocessing - parallelism handled by base_simulator

from adaptive_sampling.src.base_simulator import BaseSimulator
from adaptive_sampling.src.sampling_strategies import BaseSampler, BoundsBasedSampler


@dataclass
class ParameterSet:
    """Represents a set of parameters for a single simulation."""
    k_values: np.ndarray              # K coefficients to use
    setup_file: str                   # Setup file for this simulation
    chem_file: str                   # Chemistry file for this simulation
    pressure_conditions: Optional[List[float]] = None  # Pressure conditions in Pa
    metadata: Dict[str, Any] = None   # Additional metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.pressure_conditions is None:
            self.pressure_conditions = [133.322]  # Default 1 Torr in Pa


@dataclass
class BatchResults:
    """Results from a batch of simulations."""
    parameter_sets: List[ParameterSet]    # Input parameters used
    compositions: np.ndarray              # Output compositions, shape (n_sims, n_species)
    execution_times: List[float]          # Time for each simulation
    total_time: float                     # Total batch time
    success_mask: np.ndarray              # Boolean mask for successful simulations
    metadata: Dict[str, Any]              # Batch-level metadata
    
    @property
    def n_simulations(self) -> int:
        return len(self.parameter_sets)
    
    @property
    def n_successful(self) -> int:
        return np.sum(self.success_mask)
    
    @property
    def success_rate(self) -> float:
        return self.n_successful / self.n_simulations if self.n_simulations > 0 else 0.0


class BatchSimulator:
    """
    Manages batch simulations with parameter sampling strategies.
    
    This class provides a clean interface for:
    1. Generating parameter sets using sampling strategies
    2. Running batches of simulations with different parameters
    3. Managing file variations (setup/chemistry files)
    4. Collecting and organizing results
    """
    
    def __init__(self, base_simulator: BaseSimulator, sampler: BaseSampler):
        """
        Initialize batch simulator.
        
        Args:
            base_simulator: The simulator to use for running simulations
            sampler: Sampling strategy for generating parameter sets
        """
        self.base_simulator = base_simulator
        self.sampler = sampler
        
        # Track all batch results
        self.batch_history: List[BatchResults] = []
        
    def generate_parameter_sets(self, 
                               n_samples: int,
                               k_bounds: Optional[np.ndarray] = None,
                               setup_files: Optional[List[str]] = None,
                               chem_files: Optional[List[str]] = None,
                               pressure_conditions: Optional[List[float]] = None) -> List[ParameterSet]:
        """
        Generate parameter sets using the sampling strategy.
        
        Args:
            n_samples: Number of parameter sets to generate
            k_bounds: Bounds for K values, shape (n_k, 2). If None, uses sampler defaults
            setup_files: List of setup files to use. If None, uses base_simulator default
            chem_files: List of chemistry files to use. If None, uses base_simulator default
            pressure_conditions: List of pressure values in Pa. If None, uses base_simulator default
            
        Returns:
            List of ParameterSet objects
        """
        # Generate K values using sampling strategy
        if k_bounds is not None:
            # Use provided bounds
            center = np.mean(k_bounds, axis=1)
            k_samples = self.sampler.sample(center, k_bounds, n_samples)
        else:
            # Use sampler's default method (e.g., BoundsBasedSampler.sample_full_space)
            if hasattr(self.sampler, 'sample_full_space'):
                k_samples = self.sampler.sample_full_space(n_samples)
            else:
                raise ValueError("Sampler must implement sample_full_space or k_bounds must be provided")
        
        # Set up file combinations
        if setup_files is None:
            setup_files = [self.base_simulator.setup_file]
        if chem_files is None:
            chem_files = [self.base_simulator.chem_file]
        
        # Set up pressure conditions
        if pressure_conditions is None:
            # Use base_simulator's default pressure conditions
            pressure_conditions = getattr(self.base_simulator, 'pressure_conditions', [133.322])
            
        # Create parameter sets
        parameter_sets = []
        for i in range(n_samples):
            # Cycle through files if we have fewer files than samples
            setup_file = setup_files[i % len(setup_files)]
            chem_file = chem_files[i % len(chem_files)]
            
            param_set = ParameterSet(
                k_values=k_samples[i],
                setup_file=setup_file,
                chem_file=chem_file,
                pressure_conditions=pressure_conditions.copy(),  # Each parameter set gets the same pressures
                metadata={
                    'sample_index': i,
                    'sampling_method': self.sampler.__class__.__name__,
                    'pressure_conditions_pa': pressure_conditions.copy(),
                    'pressure_conditions_torr': [p/133.322 for p in pressure_conditions],
                    'n_pressure_conditions': len(pressure_conditions)
                }
            )
            parameter_sets.append(param_set)
            
        return parameter_sets
    
    def run_batch(self, parameter_sets: List[ParameterSet], parallel_workers: int = 1) -> BatchResults:
        """
        Run a batch of simulations with the given parameter sets.
        
        Args:
            parameter_sets: List of parameter sets to simulate
            parallel_workers: Number of parallel workers for simulation execution
            
        Returns:
            BatchResults object with all results and metadata
        """
        print(f"Running batch of {len(parameter_sets)} simulations...")
        print(f"   � Simulator: {self.base_simulator.__class__.__name__}")
        start_time = time.time()
        
        # Group parameter sets by (setup_file, chem_file) combination
        file_groups = self._group_by_files(parameter_sets)
        
        # Run simulations for each file combination
        all_compositions = []
        all_execution_times = []
        all_success_mask = []
        
        for (setup_file, chem_file), group_param_sets in file_groups.items():
            print(f"  Running {len(group_param_sets)} simulations with {setup_file}/{chem_file}")
            
            # Extract K values and pressure conditions for this group
            k_values_batch = np.array([ps.k_values for ps in group_param_sets])
            
            # Get pressure conditions from first parameter set (they should be the same for all in group)
            pressure_conditions = group_param_sets[0].pressure_conditions
            print(f"    🔧 Pressure conditions: {pressure_conditions} Pa ({[p/133.322 for p in pressure_conditions]} Torr)")
            
            # Update simulator files if needed
            original_setup = self.base_simulator.setup_file
            original_chem = self.base_simulator.chem_file
            
            try:
                self.base_simulator.setup_file = setup_file
                self.base_simulator.chem_file = chem_file
                
                # Run simulations for this group with pressure conditions
                group_start = time.time()
                compositions = self.base_simulator.run_simulations(
                    k_values_batch, 
                    parallel_workers=parallel_workers,
                    pressure_conditions=pressure_conditions
                )
                group_time = time.time() - group_start
                
                # Store results
                all_compositions.append(compositions)
                
                # Distribute group time across individual simulations
                individual_times = [group_time / len(group_param_sets)] * len(group_param_sets)
                all_execution_times.extend(individual_times)
                
                # Mark all as successful
                all_success_mask.extend([True] * len(group_param_sets))
                
            except Exception as e:
                print(f"    Error in group {setup_file}/{chem_file}: {e}")
                
                # Create dummy results for failed simulations
                # Try to infer n_species from successful results, or use a default
                if all_compositions:
                    # Use the same number of species as previous successful groups
                    n_species = all_compositions[0].shape[1]
                else:
                    # Default based on simulator type
                    n_species = getattr(self.base_simulator, '_get_expected_species_count', lambda: 10)()
                
                dummy_compositions = np.zeros((len(group_param_sets), n_species))
                all_compositions.append(dummy_compositions)
                
                # Mark execution times and failures
                all_execution_times.extend([0.0] * len(group_param_sets))
                all_success_mask.extend([False] * len(group_param_sets))
                
            finally:
                # Restore original files
                self.base_simulator.setup_file = original_setup
                self.base_simulator.chem_file = original_chem
        
        # Combine all results
        total_time = time.time() - start_time
        
        if all_compositions:
            combined_compositions = np.vstack(all_compositions)
        else:
            combined_compositions = np.array([])
            
        # Create results object
        batch_results = BatchResults(
            parameter_sets=parameter_sets,
            compositions=combined_compositions,
            execution_times=all_execution_times,
            total_time=total_time,
            success_mask=np.array(all_success_mask),
            metadata={
                'batch_timestamp': datetime.now().isoformat(),
                'sampler_type': self.sampler.__class__.__name__,
                'simulator_type': self.base_simulator.__class__.__name__,
                'simulation_type': getattr(self.base_simulator, 'simulation_type', 'unknown'),
                'chemistry_name': getattr(self.base_simulator, 'chemistry_name', 'unknown'),
                'pressure_conditions_pa': getattr(self.base_simulator, 'pressure_conditions', [133.322]),
                'pressure_conditions_torr': [p/133.322 for p in getattr(self.base_simulator, 'pressure_conditions', [133.322])],
                'n_pressure_conditions': len(getattr(self.base_simulator, 'pressure_conditions', [133.322])),
                'unique_file_combinations': len(file_groups),
                'parallel_workers': parallel_workers
            }
        )
        
        # Store in history
        self.batch_history.append(batch_results)
        
        print(f"Batch completed in {total_time:.2f}s")
        print(f"  Success rate: {batch_results.success_rate:.1%} ({batch_results.n_successful}/{batch_results.n_simulations})")
        
        return batch_results
    
    def _group_by_files(self, parameter_sets: List[ParameterSet]) -> Dict[Tuple[str, str], List[ParameterSet]]:
        """Group parameter sets by (setup_file, chem_file) combination."""
        groups = {}
        
        for param_set in parameter_sets:
            key = (param_set.setup_file, param_set.chem_file)
            if key not in groups:
                groups[key] = []
            groups[key].append(param_set)
            
        return groups
    
    def run_with_sampling(self, 
                         n_samples: int,
                         k_bounds: Optional[np.ndarray] = None,
                         setup_files: Optional[List[str]] = None,
                         chem_files: Optional[List[str]] = None,
                         pressure_conditions: Optional[List[float]] = None,
                         parallel_workers: int = 1) -> BatchResults:
        """
        Generate parameter sets using sampling and run batch simulations in one call.
        
        Args:
            n_samples: Number of simulations to run
            k_bounds: Bounds for K value sampling
            setup_files: List of setup files to cycle through
            chem_files: List of chemistry files to cycle through
            pressure_conditions: List of pressure values in Pa
            parallel_workers: Number of parallel workers (default: 1)
            
        Returns:
            BatchResults object
        """
        # Generate parameter sets
        parameter_sets = self.generate_parameter_sets(
            n_samples=n_samples,
            k_bounds=k_bounds,
            setup_files=setup_files,
            chem_files=chem_files,
            pressure_conditions=pressure_conditions
        )
        
        # Run batch
        return self.run_batch(parameter_sets, parallel_workers=parallel_workers)
    
    def save_batch_results(self, batch_results: BatchResults, filepath: str = None):
        """
        Save batch results to file with better organization.
        
        Args:
            batch_results: Results to save
            filepath: Optional custom path. If None, creates organized path automatically
        """
        import json
        from datetime import datetime
        
        if filepath is None:
            # Create organized path automatically
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sampler_name = batch_results.metadata['sampler_type'].lower()
            simulator_name = batch_results.metadata['simulator_type'].lower()
            
            # Organized structure: results/batch_simulations/simulator/sampler/date/
            results_dir = os.path.join(
                os.getcwd(), 
                'results', 
                'batch_simulations',
                simulator_name,
                sampler_name,
                datetime.now().strftime("%Y-%m-%d")
            )
            os.makedirs(results_dir, exist_ok=True)
            
            filename = f"batch_{batch_results.n_simulations}sims_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
        
        # Prepare data for JSON serialization
        save_data = {
            'metadata': batch_results.metadata,
            'n_simulations': int(batch_results.n_simulations),
            'n_successful': int(batch_results.n_successful),
            'success_rate': float(batch_results.success_rate),
            'total_time': float(batch_results.total_time),
            'parameter_sets': [
                {
                    'k_values': ps.k_values.tolist(),
                    'setup_file': ps.setup_file,
                    'chem_file': ps.chem_file,
                    'metadata': ps.metadata
                }
                for ps in batch_results.parameter_sets
            ],
            'compositions': batch_results.compositions.tolist(),
            'execution_times': [float(t) for t in batch_results.execution_times],
            'success_mask': [bool(s) for s in batch_results.success_mask]
        }
        
        # Save to file
        if filepath and os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        print(f"📁 Batch results saved to: {filepath}")
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """Get summary of all batches run so far."""
        if not self.batch_history:
            return {'total_batches': 0, 'total_simulations': 0}
            
        total_simulations = sum(br.n_simulations for br in self.batch_history)
        total_successful = sum(br.n_successful for br in self.batch_history)
        total_time = sum(br.total_time for br in self.batch_history)
        
        return {
            'total_batches': len(self.batch_history),
            'total_simulations': total_simulations,
            'total_successful': total_successful,
            'overall_success_rate': total_successful / total_simulations if total_simulations > 0 else 0.0,
            'total_time': total_time,
            'average_time_per_simulation': total_time / total_simulations if total_simulations > 0 else 0.0
        }
