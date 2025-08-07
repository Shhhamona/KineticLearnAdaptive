#!/usr/bin/env python3
"""
Simulation Cache Manager

Utility script to inspect, manage, and analyze cached simulation results.
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

class SimulationCacheManager:
    """Manager for simulation cache files."""
    
    def __init__(self, cache_dir: str = "results/simulations"):
        """Initialize cache manager."""
        self.cache_dir = cache_dir
        
    def list_simulations(self) -> pd.DataFrame:
        """List all cached simulations with metadata."""
        if not os.path.exists(self.cache_dir):
            print(f"Cache directory not found: {self.cache_dir}")
            return pd.DataFrame()
        
        simulations = []
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                try:
                    filepath = os.path.join(self.cache_dir, filename)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    metadata = data['metadata']
                    sim_info = {
                        'filename': filename,
                        'timestamp': metadata['timestamp'],
                        'simulator_type': metadata['simulator_type'],
                        'n_simulations': metadata['n_simulations'],
                        'n_k_varied': metadata['n_k_varied'],
                        'n_species': metadata['n_species'],
                        'execution_time': metadata['execution_time_seconds'],
                        'setup_file': metadata['setup_file'],
                        'chem_file': metadata['chem_file'],
                        'simulation_hash': metadata['simulation_hash']
                    }
                    simulations.append(sim_info)
                    
                except Exception as e:
                    print(f"Warning: Could not read {filename}: {e}")
        
        return pd.DataFrame(simulations)
    
    def get_simulation_details(self, filename: str) -> Dict[str, Any]:
        """Get detailed information about a specific simulation."""
        filepath = os.path.join(self.cache_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Simulation file not found: {filename}")
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def clean_old_simulations(self, days_old: int = 7) -> None:
        """Remove simulation files older than specified days."""
        cutoff_date = datetime.now() - pd.Timedelta(days=days_old)
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.cache_dir, filename)
                file_date = datetime.fromtimestamp(os.path.getctime(filepath))
                
                if file_date < cutoff_date:
                    os.remove(filepath)
                    print(f"Removed old simulation: {filename}")
    
    def print_summary(self) -> None:
        """Print a summary of cached simulations."""
        df = self.list_simulations()
        
        if df.empty:
            print("No cached simulations found.")
            return
        
        print("="*60)
        print("SIMULATION CACHE SUMMARY")
        print("="*60)
        print(f"Total simulations cached: {len(df)}")
        print(f"Total individual runs: {df['n_simulations'].sum()}")
        print(f"Total execution time: {df['execution_time'].sum():.2f} seconds")
        print()
        
        print("By simulator type:")
        print(df.groupby('simulator_type').agg({
            'n_simulations': 'sum',
            'execution_time': 'sum'
        }))
        print()
        
        print("Recent simulations:")
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        recent = df.nlargest(5, 'timestamp_dt')[['filename', 'timestamp', 'simulator_type', 'n_simulations']]
        print(recent.to_string(index=False))

def main():
    """Main function for command line usage."""
    cache_manager = SimulationCacheManager()
    
    print("Simulation Cache Manager")
    print("========================")
    
    # Print summary
    cache_manager.print_summary()
    
    # List all simulations
    df = cache_manager.list_simulations()
    if not df.empty:
        print("\\n" + "="*60)
        print("ALL CACHED SIMULATIONS")
        print("="*60)
        print(df[['filename', 'timestamp', 'simulator_type', 'n_simulations', 'execution_time']].to_string(index=False))

if __name__ == "__main__":
    main()
