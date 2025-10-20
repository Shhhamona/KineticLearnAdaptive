"""
Core data loading class for kinetic modeling.

This module provides the MultiPressureDataset class for loading and preprocessing
multi-pressure kinetic modeling data with proper scaler management.
"""

import json
import numpy as np
from sklearn import preprocessing
from typing import Optional, List, Tuple, Union


class MultiPressureDataset:
    """
    Load and preprocess multi-pressure kinetic modeling data.
    
    This class handles:
    - Loading data from text files OR raw arrays
    - Multi-pressure condition support
    - Safe scaler fitting/reuse
    - Proper data normalization
    
    Attributes:
        num_pressure_conditions (int): Number of pressure conditions in the data
        nspecies (int): Number of species in the composition
        x_data (np.ndarray): Scaled input features, shape (n_samples, num_pressure_conditions * nspecies)
        y_data (np.ndarray): Scaled output targets, shape (n_samples, n_reactions)
        raw_data (np.ndarray): Unprocessed data as loaded from file (None if initialized from arrays)
        scaler_input (List[MaxAbsScaler]): Input scalers (one per pressure condition)
        scaler_output (List[MaxAbsScaler]): Output scalers (one per pressure condition)
    """
    
    def __init__(
        self,
        nspecies: int,
        num_pressure_conditions: int,
        src_file: Optional[str] = None,
        raw_compositions: Optional[np.ndarray] = None,
        raw_k_values: Optional[np.ndarray] = None,
        processed_x: Optional[np.ndarray] = None,
        processed_y: Optional[np.ndarray] = None,
        react_idx: Optional[np.ndarray] = None,
        max_rows: Optional[int] = None,
        columns: Optional[np.ndarray] = None,
        scaler_input: Optional[List[preprocessing.MaxAbsScaler]] = None,
        scaler_output: Optional[List[preprocessing.MaxAbsScaler]] = None,
    ):
        """
        Initialize the dataset from either a file, raw arrays, or processed arrays.
        
        Three initialization modes:
        1. From file: Provide src_file (and optionally react_idx, max_rows, columns)
        2. From raw arrays: Provide raw_compositions and raw_k_values (must also provide scalers)
        3. From processed arrays: Provide processed_x and processed_y (must also provide scalers)
        
        Args:
            nspecies: Number of chemical species
            num_pressure_conditions: Number of pressure conditions
            src_file: Path to the data file (Mode 1)
            raw_compositions: Raw density values, shape (n_sims * num_pressure_conditions, nspecies) (Mode 2)
            raw_k_values: Raw reaction rates, shape (n_sims, n_k) - NOT yet scaled by 1e30 (Mode 2)
            processed_x: Already-scaled input data, shape (n_samples, num_pressure_conditions * nspecies) (Mode 3)
            processed_y: Already-scaled output data, shape (n_samples, n_reactions) (Mode 3)
            react_idx: Indices of reaction rate columns (None = all non-species columns) (Mode 1)
            max_rows: Maximum number of rows to load (None = all) (Mode 1)
            columns: Specific columns to load (None = all) (Mode 1)
            scaler_input: Pre-fitted input scalers (required for Modes 2&3, optional for Mode 1)
            scaler_output: Pre-fitted output scalers (required for Modes 2&3, optional for Mode 1)
        """
        self.num_pressure_conditions = num_pressure_conditions
        self.nspecies = nspecies
        
        # Determine initialization mode
        if src_file is not None:
            # Mode 1: Initialize from file
            self._init_from_file(
                src_file=src_file,
                react_idx=react_idx,
                max_rows=max_rows,
                columns=columns,
                scaler_input=scaler_input,
                scaler_output=scaler_output
            )
        elif raw_compositions is not None and raw_k_values is not None:
            # Mode 2: Initialize from raw arrays
            if scaler_input is None or scaler_output is None:
                raise ValueError("When initializing from raw arrays, both scaler_input and scaler_output must be provided")
            self._init_from_arrays(
                raw_compositions=raw_compositions,
                raw_k_values=raw_k_values,
                scaler_input=scaler_input,
                scaler_output=scaler_output
            )
        elif processed_x is not None and processed_y is not None:
            # Mode 3: Initialize from processed arrays
            if scaler_input is None or scaler_output is None:
                raise ValueError("When initializing from processed arrays, both scaler_input and scaler_output must be provided")
            self._init_from_processed_arrays(
                processed_x=processed_x,
                processed_y=processed_y,
                scaler_input=scaler_input,
                scaler_output=scaler_output
            )
        else:
            raise ValueError("Must provide either src_file OR (raw_compositions AND raw_k_values) OR (processed_x AND processed_y)")
    
    def _init_from_file(
        self,
        src_file: str,
        react_idx: Optional[np.ndarray],
        max_rows: Optional[int],
        columns: Optional[np.ndarray],
        scaler_input: Optional[List[preprocessing.MaxAbsScaler]],
        scaler_output: Optional[List[preprocessing.MaxAbsScaler]]
    ):
        """Initialize from a data file (TXT or JSON)."""
        # Detect file type
        if src_file.lower().endswith('.json'):
            self._init_from_json_file(
                src_file=src_file,
                react_idx=react_idx,
                max_rows=max_rows,
                scaler_input=scaler_input,
                scaler_output=scaler_output
            )
            return
        
        # Load raw data from TXT file
        self.raw_data = np.loadtxt(
            src_file,
            max_rows=max_rows,
            usecols=columns,
            delimiter="  ",
            comments="#",
            skiprows=0,
            dtype=np.float64
        )
        
        # Determine column indices
        ncolumns = len(self.raw_data[0])
        x_columns = np.arange(ncolumns - self.nspecies, ncolumns, 1)
        y_columns = react_idx if react_idx is not None else np.arange(0, ncolumns - self.nspecies, 1)
        
        # Extract and preprocess
        x_data = self.raw_data[:, x_columns]  # Densities
        y_data = self.raw_data[:, y_columns] * 1e30  # Reaction rates (scale to avoid precision limit)
        
        print(f"\n[DEBUG] Data loading from TXT file:")
        print(f"  Raw x_data (densities) sample [0]: {self.raw_data[0, x_columns]}")
        print(f"  Raw y_data (K values) sample [0]: {self.raw_data[0, y_columns]}")
        print(f"  After 1e30 scaling, y_data [0]: {y_data[0]}")
        
        # Reshape for multi-pressure conditions
        x_data = x_data.reshape(self.num_pressure_conditions, -1, x_data.shape[1])
        y_data = y_data.reshape(self.num_pressure_conditions, -1, y_data.shape[1])
        
        # Initialize or reuse scalers
        # CONSISTENT CONVENTION: num_pressure_conditions scalers for both input and output
        if scaler_input is None:
            self.scaler_input = [preprocessing.MaxAbsScaler() for _ in range(self.num_pressure_conditions)]
            for i in range(self.num_pressure_conditions):
                self.scaler_input[i].fit(x_data[i])
                print(f"  Input scaler [pressure {i}] scale factors: {self.scaler_input[i].scale_}")
        else:
            self.scaler_input = scaler_input
            
        if scaler_output is None:
            self.scaler_output = [preprocessing.MaxAbsScaler() for _ in range(self.num_pressure_conditions)]
            for i in range(self.num_pressure_conditions):
                self.scaler_output[i].fit(y_data[i])
                print(f"  Output scaler [pressure {i}] scale factors: {self.scaler_output[i].scale_}")
        else:
            self.scaler_output = scaler_output
        
        # Apply scaling
        for i in range(self.num_pressure_conditions):
            x_data[i] = self.scaler_input[i].transform(x_data[i])
            y_data[i] = self.scaler_output[i].transform(y_data[i])
        
        # Flatten x_data: (pressure, samples, features) -> (samples, pressure * features)
        x_data = np.transpose(x_data, (1, 0, 2)).reshape(-1, self.num_pressure_conditions * x_data.shape[-1])
        
        # Use first pressure condition for output (standard convention)
        y_data = y_data[0]
        
        # Store processed data
        self.x_data = x_data
        self.y_data = y_data
    
    def _init_from_arrays(
        self,
        raw_compositions: np.ndarray,
        raw_k_values: np.ndarray,
        scaler_input: List[preprocessing.MaxAbsScaler],
        scaler_output: List[preprocessing.MaxAbsScaler]
    ):
        """Initialize from raw arrays with defensive copying."""
        self.raw_data = None  # No raw file data in this mode
        
        # Store scalers
        self.scaler_input = scaler_input
        self.scaler_output = scaler_output
        
        # Defensive copy and apply 1e30 multiplier to k values
        k_data = (raw_k_values * 1e30).copy()
        
        # Defensive copy and reshape data
        n_sims = raw_k_values.shape[0]
        x_data = raw_compositions.reshape(self.num_pressure_conditions, n_sims, self.nspecies).copy()
        
        # For y_data: k_values are per-simulation, replicate for each pressure condition
        # Shape: (n_sims, n_reactions) -> (num_pressure_conditions, n_sims, n_reactions)
        y_data = np.tile(k_data, (self.num_pressure_conditions, 1, 1))
        
        # Apply per-pressure scalers (SAME as TXT convention)
        for i in range(self.num_pressure_conditions):
            x_data[i] = self.scaler_input[i].transform(x_data[i])
            y_data[i] = self.scaler_output[i].transform(y_data[i])
        
        # Flatten x_data: (pressure, samples, features) -> (samples, pressure * features)
        x_data_transposed = np.transpose(x_data, (1, 0, 2))
        x_data_flat = x_data_transposed.reshape(n_sims, self.num_pressure_conditions * self.nspecies).copy()
        
        # Use first pressure condition for output (SAME as TXT convention)
        y_data_flat = y_data[0]
        
        # Store processed data
        self.x_data = x_data_flat
        self.y_data = y_data_flat
    
    def _init_from_json_file(
        self,
        src_file: str,
        react_idx: Optional[np.ndarray],
        max_rows: Optional[int],
        scaler_input: Optional[List[preprocessing.MaxAbsScaler]],
        scaler_output: Optional[List[preprocessing.MaxAbsScaler]]
    ):
        """
        Initialize from a JSON batch simulation file.
        
        Follows the same approach as k_centered_adaptive_learning.py:
        1. Load JSON and extract compositions and k_values
        2. Apply the same reshaping and scaling logic as apply_training_scalers
        """
        # Load JSON file
        with open(src_file, 'r') as f:
            data = json.load(f)
        
        # Extract data - same as K-centered approach
        if 'compositions' not in data:
            raise ValueError(f"JSON file {src_file} must contain 'compositions' key")
        
        # Extract k_values from parameter_sets (K-centered format)
        if 'parameter_sets' in data:
            batch_k_values = np.array([ps['k_values'] for ps in data['parameter_sets']])
        elif 'k_values' in data:
            batch_k_values = np.array(data['k_values'])
        else:
            raise ValueError(f"JSON file {src_file} must contain 'parameter_sets' or 'k_values' key")
        
        # Get compositions array - format: (n_sims * num_pressure_conditions, nspecies)
        batch_compositions = np.array(data['compositions'])
        
        # Store raw JSON data
        self.raw_data = data
        
        # Apply max_rows if specified
        if max_rows is not None:
            n_sims = batch_k_values.shape[0]
            expected_rows = n_sims * self.num_pressure_conditions
            batch_compositions = batch_compositions[:max_rows * self.num_pressure_conditions, :]
            batch_k_values = batch_k_values[:max_rows, :]
        
        # Apply react_idx if specified
        if react_idx is not None:
            batch_k_values = batch_k_values[:, react_idx]
        
        # Now apply the same scaling logic as apply_training_scalers
        # Step 1: Apply 1e30 multiplier to k_values
        k_data = batch_k_values * 1e30
        
        # Step 2: Reshape data
        n_sims = batch_k_values.shape[0]
        x_data = batch_compositions.reshape(self.num_pressure_conditions, n_sims, self.nspecies).copy()
        
        # For y_data: k_values are per-simulation, replicate for each pressure condition
        # Shape: (n_sims, n_reactions) -> (num_pressure_conditions, n_sims, n_reactions)
        y_data = np.tile(k_data, (self.num_pressure_conditions, 1, 1))
        
        # Step 3: Create or use provided scalers (SAME as TXT convention)
        if scaler_input is None:
            scaler_input = [preprocessing.MaxAbsScaler() for _ in range(self.num_pressure_conditions)]
            for i in range(self.num_pressure_conditions):
                scaler_input[i].fit(x_data[i])
        
        if scaler_output is None:
            scaler_output = [preprocessing.MaxAbsScaler() for _ in range(self.num_pressure_conditions)]
            for i in range(self.num_pressure_conditions):
                scaler_output[i].fit(y_data[i])
        
        # Store scalers
        self.scaler_input = scaler_input
        self.scaler_output = scaler_output
        
        # Step 4: Apply scalers (SAME as TXT convention)
        for i in range(self.num_pressure_conditions):
            x_data[i] = self.scaler_input[i].transform(x_data[i])
            y_data[i] = self.scaler_output[i].transform(y_data[i])
        
        # Step 5: Flatten x_data (SAME as TXT convention)
        x_data_transposed = np.transpose(x_data, (1, 0, 2))
        new_x = x_data_transposed.reshape(n_sims, self.num_pressure_conditions * self.nspecies).copy()
        
        # Use first pressure condition for output (SAME as TXT convention)
        new_y = y_data[0]
        
        # Store processed data
        self.x_data = new_x
        self.y_data = new_y
    
    def _init_from_processed_arrays(
        self,
        processed_x: np.ndarray,
        processed_y: np.ndarray,
        scaler_input: List[preprocessing.MaxAbsScaler],
        scaler_output: List[preprocessing.MaxAbsScaler]
    ):
        """Initialize from already-processed (scaled) arrays - simple and fast!"""
        self.raw_data = None  # No raw file data in this mode
        
        # Store scalers
        self.scaler_input = scaler_input
        self.scaler_output = scaler_output
        
        # Directly store the processed data (defensive copy)
        self.x_data = processed_x.copy()
        self.y_data = processed_y.copy()
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the preprocessed input and output data.
        
        Returns:
            Tuple of (x_data, y_data)
        """
        return self.x_data, self.y_data
    
    def get_scalers(self) -> Tuple[List[preprocessing.MaxAbsScaler], List[preprocessing.MaxAbsScaler]]:
        """
        Get the fitted scalers.
        
        Returns:
            Tuple of (input_scalers, output_scalers)
        """
        return self.scaler_input, self.scaler_output
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.x_data)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"MultiPressureDataset(samples={len(self)}, "
                f"features={self.x_data.shape[1]}, outputs={self.y_data.shape[1]}, "
                f"pressures={self.num_pressure_conditions})")
