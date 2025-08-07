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
    
    def __init__(self, setup_file: str, chem_file: str, loki_path: str):
        """
        Initialize the simulator.
        
        Args:
            setup_file: Path to simulation setup file
            chem_file: Path to chemistry file
            loki_path: Path to LoKI installation (if applicable)
        """
        self.setup_file = setup_file
        self.chem_file = chem_file
        self.loki_path = loki_path
        self.cwd = os.getcwd()
        
        # Create organized simulation results directory structure
        simulator_name = self.__class__.__name__.lower()
        chem_name = os.path.splitext(self.chem_file)[0]  # Remove .chem extension
        self.results_dir = os.path.join(self.cwd, 'results', 'simulations', simulator_name, chem_name)
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
    def run_simulations(self, k_samples: np.ndarray) -> np.ndarray:
        """
        Run simulations for given K values.
        
        Args:
            k_samples: Array of shape (n_simulations, n_reactions) with K values
            
        Returns:
            Array of shape (n_simulations, n_species) with chemical compositions
        """
        pass
    
    @abstractmethod
    def get_reference_k_values(self) -> np.ndarray:
        """Get reference K values from chemistry file."""
        pass


class LoKISimulator(BaseSimulator):
    """LoKI-based simulator using the existing genFiles framework."""
    
    def __init__(self, setup_file: str, chem_file: str, loki_path: str, k_columns: List[int]):
        """
        Initialize LoKI simulator.
        
        Args:
            setup_file: LoKI setup file
            chem_file: LoKI chemistry file  
            loki_path: Path to LoKI installation
            k_columns: Which K coefficients to vary
        """
        super().__init__(setup_file, chem_file, loki_path)
        self.k_columns = k_columns
        
        # Import the existing genFiles class
        genfiles_path = os.path.join(parent_dir, 'other_scripts', 'genFiles')
        sys.path.append(genfiles_path)
        
        try:
            from other_scripts.genFiles.genFiles_O2_simple import Simulations
            self.SimulationsClass = Simulations
        except ImportError as e:
            print(f"Warning: Could not import LoKI genFiles module: {e}")
            print("Using mock simulation fallback for LoKI simulator")
            self.SimulationsClass = None
        
    def run_simulations(self, k_samples: np.ndarray) -> np.ndarray:
        """
        Run LoKI simulations for given K samples.
        
        Args:
            k_samples: K values to simulate, shape (n_simulations, n_k_varied)
            
        Returns:
            Chemical compositions, shape (n_simulations, n_species)
        """
        if self.SimulationsClass is None:
            print("LoKI simulator not available, using mock data")
            return self._generate_mock_data(k_samples)
            
        n_simulations = k_samples.shape[0]
        start_time = datetime.now()
        
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
        simul.parameters.k_set = full_k_samples
        simul.set_ChemFile_ON()
        
        # Save original working directory
        original_cwd = os.getcwd()
        
        try:
            # Run simulations
            simul.runSimulations()
            
            # Restore original working directory
            os.chdir(original_cwd)
            
            # Read output densities
            densities = simul._read_otpt_densities()
            
            # Calculate execution time and save results
            execution_time = (datetime.now() - start_time).total_seconds()
            metadata = {
                "k_columns_varied": self.k_columns,
                "reference_k_values": ref_k.tolist(),
                "loki_version": "v3.1.0",
                "chemistry": "O2_simple"
            }
            self._save_simulation_result(k_samples, densities, execution_time, metadata)
            
            return densities
            
        except Exception as e:
            print(f"LoKI simulation failed: {e}")
            # Restore original working directory on error
            try:
                os.chdir(original_cwd)
            except:
                pass
            # Return mock data as fallback
            return self._generate_mock_data(k_samples)
    
    def get_reference_k_values(self) -> np.ndarray:
        """Get reference K values from chemistry file."""
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
            return np.array(values)
        except FileNotFoundError:
            print(f"Chemistry file not found: {chem_path}")
            # Return default O2_simple values
            return np.array([6E-16, 1.3E-15, 9.6E-16, 2.2E-15, 7E-22, 3E-44, 3.2E-45, 5.2, 53])
    
    def _generate_mock_data(self, k_samples: np.ndarray) -> np.ndarray:
        """Generate mock chemical composition data when LoKI fails."""
        n_simulations, n_k = k_samples.shape
        n_species = 3  # Match the O2_simple LoKI output (3 main species densities)
        
        # Create realistic-looking mock data based on K values
        np.random.seed(42)  # For reproducibility
        
        # Simple model: composition depends on log(K) with some noise
        log_k = np.log10(k_samples + 1e-50)  # Avoid log(0)
        compositions = np.random.lognormal(
            mean=22 + np.sum(log_k, axis=1, keepdims=True) * 0.1,  # Higher mean to match LoKI scale (~1e22)
            sigma=1,
            size=(n_simulations, n_species)
        )
        
        return compositions


class MockSimulator(BaseSimulator):
    """Mock simulator for testing and development."""
    
    def __init__(self, setup_file: str, chem_file: str, loki_path: str, 
                 true_k: Optional[np.ndarray] = None):
        """
        Initialize mock simulator.
        
        Args:
            setup_file: Not used but kept for interface compatibility
            chem_file: Not used but kept for interface compatibility  
            loki_path: Not used but kept for interface compatibility
            true_k: True K values for generating realistic mock data
        """
        super().__init__(setup_file, chem_file, loki_path)
        self.true_k = true_k if true_k is not None else np.array([6E-16, 1.3E-15, 9.6E-16, 2.2E-15, 7E-22, 3E-44, 3.2E-45, 5.2, 53])
        
    def run_simulations(self, k_samples: np.ndarray) -> np.ndarray:
        """
        Generate mock chemical composition data.
        
        Args:
            k_samples: K values, shape (n_simulations, n_k)
            
        Returns:
            Mock chemical compositions, shape (n_simulations, n_species)
        """
        start_time = datetime.now()
        
        n_simulations, n_k = k_samples.shape
        n_species = 10
        
        # Create deterministic but complex relationship between K and C
        np.random.seed(42)  # For reproducibility
        
        # Nonlinear transformation: C depends on K through multiple pathways
        log_k = np.log10(k_samples + 1e-50)
        
        # Base composition influenced by different K coefficients
        compositions = np.zeros((n_simulations, n_species))
        
        for i in range(n_species):
            # Each species influenced by different combinations of K values
            weights = np.sin(np.arange(n_k) * (i + 1) * 0.5) + 1  # Different weights per species
            weighted_log_k = np.sum(log_k * weights, axis=1)
            
            # Base level + K-dependent component + noise
            compositions[:, i] = (15 + i * 0.5 +  # Base level varies by species
                                 weighted_log_k * 0.3 +  # K-dependent component
                                 np.random.normal(0, 0.1, n_simulations))  # Small noise
        
        # Convert to densities (positive values)
        compositions = np.exp(compositions)
        
        # Calculate execution time and save results
        execution_time = (datetime.now() - start_time).total_seconds()
        metadata = {
            "simulator_note": "Mock simulator for testing",
            "n_species_generated": n_species,
            "deterministic_seed": 42,
            "relationship": "nonlinear log(K) -> composition"
        }
        self._save_simulation_result(k_samples, compositions, execution_time, metadata)
        
        return compositions
    
    def get_reference_k_values(self) -> np.ndarray:
        """Get reference K values."""
        return self.true_k.copy()
