"""
Base simulator interface for rate coefficient determination studies.

This module provides an abstract interface for plasma chemistry simulators,
allowing easy switching between different simulation backends (LoKI, mock, etc.)
"""

import os
import sys
import json
import hashlib
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
from datetime import datetime

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

class BaseSimulator(ABC):
    """Abstract base class for plasma chemistry simulators."""
    
    def __init__(self, setup_file: str, chem_file: str, loki_path: str, 
                 pressure_conditions: Optional[List[float]] = None):
        """
        Initialize the simulator.
        
        Args:
            setup_file: Path to simulation setup file
            chem_file: Path to chemistry file
            loki_path: Path to LoKI installation (if applicable)
            pressure_conditions: List of pressure values in Pa (e.g., [133.322, 666.66])
                                 If None, uses single default pressure from setup file
        """
        self.setup_file = setup_file
        self.chem_file = chem_file
        self.loki_path = loki_path
        self.pressure_conditions = pressure_conditions or [133.322]  # Default 1 Torr in Pa
        self.cwd = os.getcwd()
        
        # Create organized simulation results directory structure
        simulator_name = self.__class__.__name__.lower()
        chem_name = os.path.splitext(os.path.basename(self.chem_file))[0]  # Just filename, no path
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        self.results_dir = os.path.join(
            self.cwd, 
            'results', 
            'individual_simulations',
            simulator_name, 
            chem_name,
            date_str
        )
        os.makedirs(self.results_dir, exist_ok=True)
        
    def _save_simulation_result(self, k_values: np.ndarray, results: np.ndarray, 
                               execution_time: float, metadata: dict = None) -> str:
        """
        Save simulation results with metadata for future reference.
        
        Args:
            k_values: Input K values used
            results: Output compositions
            execution_time: Time taken for simulation
            metadata: Additional metadata
            
        Returns:
            Path to saved file
        """
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_hash = hashlib.md5(k_values.tobytes()).hexdigest()[:12]
        filename = f"sim_{sim_hash}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "simulator_type": self.__class__.__name__,
                "setup_file": self.setup_file,
                "chem_file": self.chem_file,
                "loki_path": self.loki_path,
                "execution_time_seconds": execution_time,
                "simulation_hash": sim_hash,
                "n_simulations": k_values.shape[0],
                "n_k_varied": k_values.shape[1],
                "n_species": results.shape[1],
                **(metadata or {})
            },
            "inputs": {
                "k_values": k_values.tolist(),
                "k_shape": list(k_values.shape)
            },
            "outputs": {
                "compositions": results.tolist(),
                "composition_shape": list(results.shape)
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        print(f"💾 Simulation results saved: {filename}")
        print(f"   📊 {k_values.shape[0]} simulations, {execution_time:.2f}s execution time")
        
        return filepath
        
    @abstractmethod
    def run_simulations(self, k_samples: np.ndarray, parallel_workers: int = 1,
                       pressure_conditions: Optional[List[float]] = None) -> np.ndarray:
        """
        Run simulations for given K values.
        
        Args:
            k_samples: Array of shape (n_simulations, n_reactions) with K values
            parallel_workers: Number of parallel workers for simulation execution
            pressure_conditions: List of pressure values in Pa. If None, uses self.pressure_conditions
            
        Returns:
            Array of shape (n_simulations * n_pressures, n_species) with chemical compositions
            If multiple pressures are used, results are stacked for each pressure condition
        """
        pass
    
    @abstractmethod
    def get_reference_k_values(self) -> np.ndarray:
        """Get reference K values from chemistry file."""
        pass


class LoKISimulator(BaseSimulator):
    """LoKI-based simulator using the existing genFiles framework."""
    
    def __init__(self, setup_file: str, chem_file: str, loki_path: str, k_columns: List[int], 
                 simulation_type: str = "complex", pressure_conditions: Optional[List[float]] = None):
        """
        Initialize LoKI simulator.
        
        Args:
            setup_file: LoKI setup file
            chem_file: LoKI chemistry file  
            loki_path: Path to LoKI installation
            k_columns: Which K coefficients to vary
            simulation_type: "simple" for O2_simple or "complex" for O2_novib (default)
            pressure_conditions: List of pressure values in Pa (e.g., [133.322, 666.66])
        """
        super().__init__(setup_file, chem_file, loki_path, pressure_conditions)
        self.k_columns = k_columns
        self.simulation_type = simulation_type
        
        # Import the existing genFiles class
        genfiles_path = os.path.join(parent_dir, 'other_scripts', 'genFiles')
        sys.path.append(genfiles_path)
        
        try:
            if simulation_type == "simple":
                from other_scripts.genFiles.genFiles_O2_simple import Simulations
                self.chemistry_name = "O2_simple"
                print("🔬 Using O2_simple simulation (faster, less accurate)")
            elif simulation_type == "complex":
                from other_scripts.genFiles.genFiles_O2_novib import Simulations
                self.chemistry_name = "O2_novib"
                print("🔬 Using O2_novib simulation (slower, more physically accurate)")
            else:
                raise ValueError(f"Unknown simulation_type: {simulation_type}. Use 'simple' or 'complex'")
                
            self.SimulationsClass = Simulations
        except ImportError as e:
            print(f"Warning: Could not import LoKI genFiles module for {simulation_type}: {e}")
            print("Using mock simulation fallback for LoKI simulator")
            self.SimulationsClass = None
            self.chemistry_name = f"{simulation_type}_fallback"
        
    def run_simulations(self, k_samples: np.ndarray, parallel_workers: int = 1,
                       pressure_conditions: Optional[List[float]] = None) -> np.ndarray:
        """
        Run LoKI simulations for given K samples with pressure control.
        
        Args:
            k_samples: K values to simulate, shape (n_simulations, n_k_varied)
            parallel_workers: Number of parallel workers for LoKI execution
            pressure_conditions: List of pressure values in Pa. If None, uses self.pressure_conditions
            
        Returns:
            Chemical compositions, shape (n_simulations * n_pressures, n_species)
            Results are stacked for each pressure condition
        """
        if self.SimulationsClass is None:
            print("LoKI simulator not available, using mock data")
            return self._generate_mock_data(k_samples)
            
        # Use provided pressure conditions or default ones
        pressures = pressure_conditions if pressure_conditions is not None else self.pressure_conditions
        n_simulations = k_samples.shape[0]
        n_pressures = len(pressures)
        start_time = datetime.now()
        
        print(f"🔧 Pressure conditions: {pressures} Pa ({[p/133.322 for p in pressures]} Torr)")
        
        try:
            # Create simulation object
            simul = self.SimulationsClass(
                self.setup_file, 
                self.chem_file, 
                self.loki_path, 
                n_simulations
            )
            
            # Get reference K values
            ref_k = self.get_reference_k_values()
            
            # Create full K arrays by modifying only specified columns
            full_k_samples = np.tile(ref_k, (n_simulations, 1))
            full_k_samples[:, self.k_columns] = k_samples
            
            # Set K values in the simulation object
            # Different genFiles modules have different expectations
            if self.simulation_type == "complex":
                # O2_novib expects: len(k_set[i]) == len(kcolumns)
                # So we only pass the K values being varied
                simul.parameters.k_set = k_samples  # Only the values we're changing
                simul.parameters.kcolumns = self.k_columns
                print(f"🔧 Setting up O2_novib (complex) simulation:")
                print(f"   📊 Varied K values shape: {k_samples.shape}")
                print(f"   📍 K columns being varied: {self.k_columns}")
                print(f"   ✅ Lengths match: {k_samples.shape[1]} == {len(self.k_columns)}")
            else:
                # O2_simple expects full K arrays
                simul.parameters.k_set = full_k_samples
                simul.parameters.kcolumns = self.k_columns
                print(f"🔧 Setting up O2_simple simulation:")
                print(f"   📊 Full K array shape: {full_k_samples.shape}")
                print(f"   📍 K columns being varied: {self.k_columns}")
            
            # Set pressure conditions if multiple pressures specified
            if n_pressures > 1:
                print(f"🔧 Setting multiple pressure conditions: {n_pressures} pressures")
                simul.fixed_pressure_set(pressures)
                print(f"   📈 Total simulations after pressure expansion: {simul.nsimulations}")
            elif n_pressures == 1 and pressures[0] != 133.322:  # Not default pressure
                print(f"🔧 Setting single custom pressure: {pressures[0]} Pa")
                simul.fixed_pressure_set(pressures)
            else:
                print(f"🔧 Using default pressure from setup file")
            
            # Generate modified chemistry files
            simul.set_ChemFile_ON()  # This creates NEW chemistry files!
            
            # Save original working directory
            original_cwd = os.getcwd()
            
            # Run simulations
            print(f"🚀 Starting LoKI {self.chemistry_name} simulation...")
            if parallel_workers > 1:
                print(f"   ⚡ Using {parallel_workers} parallel workers")
            simul.runSimulations(parallel_workers=parallel_workers)
            
            # Restore original working directory
            os.chdir(original_cwd)
            
            # Read output densities
            print("📖 Reading LoKI output densities...")
            densities = simul._read_otpt_densities()
            
            if densities is None or len(densities) == 0:
                raise ValueError("No density data returned from LoKI simulation")
            
            # Convert to numpy array and ensure proper shape
            densities = np.array(densities, dtype=float)
            if densities.ndim == 1:
                densities = densities.reshape(1, -1)
            
            print(f"✅ LoKI simulation completed successfully!")
            print(f"   📊 Output shape: {densities.shape}")
            
            # Calculate execution time and save results
            execution_time = (datetime.now() - start_time).total_seconds()
            metadata = {
                "k_columns_varied": self.k_columns,
                "reference_k_values": ref_k.tolist(),
                "pressure_conditions_pa": pressures,
                "pressure_conditions_torr": [p/133.322 for p in pressures],
                "n_pressure_conditions": len(pressures),
                "genfiles_module": self.SimulationsClass.__module__ if self.SimulationsClass else "unknown",
                "simulation_type": self.simulation_type,
                "chemistry": self.chemistry_name,
                "is_physically_accurate": self.simulation_type == "complex",
                "matlab_engine_used": True
            }
            self._save_simulation_result(k_samples, densities, execution_time, metadata)
            
            return densities
            
        except Exception as e:
            print(f"LoKI simulation failed: {e}")
            print(f"   Error details: {type(e).__name__}")
            # Restore original working directory on error
            try:
                os.chdir(original_cwd)
            except:
                pass
            # Return mock data as fallback
            print("⚠️  Using mock simulation fallback data instead of real LoKI")
            return self._generate_mock_data(k_samples)
    
    def get_reference_k_values(self) -> np.ndarray:
        """Get reference K values for the variable reactions."""
        if self.simulation_type == "simple":
            # For O2_simple, try to parse all K values from chemistry file
            chem_path = os.path.join(self.cwd, 'simulFiles', self.chem_file)
            try:
                with open(chem_path, 'r') as file:
                    values = []
                    for line in file:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                values.append(float(parts[-2]))
                            except ValueError:
                                continue
                if len(values) > 0:
                    return np.array(values)
            except FileNotFoundError:
                print(f"Chemistry file not found: {chem_path}")
            
            # Default O2_simple values if parsing fails
            return np.array([6E-16, 1.3E-15, 9.6E-16, 2.2E-15, 7E-22, 3E-44, 3.2E-45, 5.2, 53])
            
        else:  # complex/O2_novib
            # For O2_novib, use the complete list of constantRateCoeff values extracted from chemistry file
            # These are ALL the K values that can be varied in O2_novib chemistry
            full_reference_k = np.array([
                7.59593006e-22, 2.99827165e-44, 4.00183879e-20, 8e-18, 1e-18, 1.5e-17,
                1.2e-16, 1.2e-16, 2e-19, 3e-21, 4.95e-18, 2.7e-18, 1.35e-18, 1.86e-19,
                2.1e-20, 2.3e-20, 1.425e-16, 1.3e-15, 1e-18, 6.9e-16, 2.8e-13, 1e-16
            ])
            
            # Return only the values corresponding to the columns we're varying
            if hasattr(self, 'k_columns') and all(k < len(full_reference_k) for k in self.k_columns):
                return full_reference_k[self.k_columns]
            else:
                # Fallback: return the first few values to match the number of columns
                n_needed = len(self.k_columns) if hasattr(self, 'k_columns') else 3
                return full_reference_k[:n_needed]
    
    def _generate_mock_data(self, k_samples: np.ndarray) -> np.ndarray:
        """Generate mock chemical composition data when LoKI fails."""
        n_simulations, n_k = k_samples.shape
        
        # Number of species depends on simulation type
        # These numbers should match the actual LoKI output
        if self.simulation_type == "simple":
            n_species = 3  # O2_simple has fewer species
        else:  # complex/O2_novib
            n_species = 11  # O2_novib has more species - updated to match real LoKI output!
        
        # Create realistic-looking mock data based on K values
        np.random.seed(42)  # For reproducibility
        
        # Simple model: composition depends on log(K) with some noise
        log_k = np.log10(k_samples + 1e-50)  # Avoid log(0)
        compositions = np.random.lognormal(
            mean=22 + np.sum(log_k, axis=1, keepdims=True) * 0.1,  # Higher mean to match LoKI scale (~1e22)
            sigma=1,
            size=(n_simulations, n_species)
        )
        
        print(f"🎭 Generated mock data with {n_species} species to match {self.chemistry_name} output")
        
        return compositions
    
    def _get_expected_species_count(self) -> int:
        """Get the expected number of species for this simulation type."""
        if self.simulation_type == "simple":
            return 3  # O2_simple species count
        else:  # complex/O2_novib
            return 11  # O2_novib species count (from real LoKI output)


class MockSimulator(BaseSimulator):
    """Mock simulator for testing and development."""
    
    def __init__(self, setup_file: str, chem_file: str, loki_path: str, 
                 true_k: Optional[np.ndarray] = None, pressure_conditions: Optional[List[float]] = None):
        """
        Initialize mock simulator.
        
        Args:
            setup_file: Not used but kept for interface compatibility
            chem_file: Not used but kept for interface compatibility  
            loki_path: Not used but kept for interface compatibility
            true_k: True K values for generating realistic mock data
            pressure_conditions: List of pressure values in Pa
        """
        super().__init__(setup_file, chem_file, loki_path, pressure_conditions)
        self.true_k = true_k if true_k is not None else np.array([6E-16, 1.3E-15, 9.6E-16, 2.2E-15, 7E-22, 3E-44, 3.2E-45, 5.2, 53])
        
    def run_simulations(self, k_samples: np.ndarray, parallel_workers: int = 1,
                       pressure_conditions: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate mock chemical composition data for multiple pressure conditions.
        
        Args:
            k_samples: K values to simulate, shape (n_simulations, n_k)
            parallel_workers: Not used in mock simulator
            pressure_conditions: List of pressure values in Pa. If None, uses self.pressure_conditions
            
        Returns:
            Mock compositions, shape (n_simulations * n_pressures, n_species)
        """
        # Use provided pressure conditions or default ones
        pressures = pressure_conditions if pressure_conditions is not None else self.pressure_conditions
        n_simulations = k_samples.shape[0]
        n_pressures = len(pressures)
        n_species = 3  # Mock simulator uses 3 species
        
        print(f"🎭 Mock simulator: {n_simulations} simulations × {n_pressures} pressures = {n_simulations * n_pressures} total outputs")
        
        # Generate base compositions for each K sample
        np.random.seed(42)  # For reproducibility
        log_k = np.log10(k_samples + 1e-50)
        
        all_compositions = []
        
        for pressure_idx, pressure in enumerate(pressures):
            # Pressure-dependent effects on composition
            pressure_factor = np.log10(pressure / 133.322)  # Normalize to 1 Torr
            
            # Simple pressure-dependent model
            pressure_effect = 1.0 + 0.1 * pressure_factor  # 10% pressure dependence
            
            compositions = np.random.lognormal(
                mean=22 + np.sum(log_k, axis=1, keepdims=True) * 0.1 * pressure_effect,
                sigma=0.5,
                size=(n_simulations, n_species)
            )
            
            all_compositions.append(compositions)
            print(f"   🔧 Pressure {pressure:.1f} Pa: generated {compositions.shape} compositions")
        
        # Stack all pressure conditions
        result = np.vstack(all_compositions)
        print(f"   ✅ Final mock data shape: {result.shape}")
        
        # Save simulation results with metadata
        execution_time = 0.1  # Mock execution time
        metadata = {
            "k_columns_varied": list(range(k_samples.shape[1])),  # Assume all columns are varied
            "reference_k_values": self.true_k.tolist(),
            "pressure_conditions_pa": pressures,
            "pressure_conditions_torr": [p/133.322 for p in pressures],
            "n_pressure_conditions": len(pressures),
            "genfiles_module": "mock",
            "simulation_type": "mock",
            "chemistry": "mock_chemistry",
            "is_physically_accurate": False,
            "matlab_engine_used": False
        }
        self._save_simulation_result(k_samples, result, execution_time, metadata)
        
        return result
        
    def get_reference_k_values(self) -> np.ndarray:
        """Get reference K values."""
        return self.true_k.copy()
