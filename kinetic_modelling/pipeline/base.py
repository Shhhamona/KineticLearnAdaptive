"""
Base pipeline class for ML workflows.

This module defines the abstract base class for all pipeline implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime


class BasePipeline(ABC):
    """
    Abstract base class for ML pipelines.
    
    A pipeline orchestrates the complete workflow:
    1. Data loading/preparation
    2. Sampling (if needed)
    3. Model training
    4. Evaluation
    5. Results storage
    
    Attributes:
        pipeline_name (str): Name of the pipeline
        results_dir (str): Directory to save results
        results (dict): Dictionary storing pipeline results
    """
    
    def __init__(
        self,
        pipeline_name: str = "pipeline",
        results_dir: str = "pipeline_results"
    ):
        """
        Initialize base pipeline.
        
        Args:
            pipeline_name: Name for this pipeline
            results_dir: Directory to save results
        """
        self.pipeline_name = pipeline_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results dictionary
        self.results = {
            'pipeline_name': pipeline_name,
            'timestamp': None,
            'config': {},
            'data_info': {},
            'training_info': {},
            'evaluation': {}
        }
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        pass
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save pipeline results to JSON file.
        
        Args:
            filename: Optional filename (default: pipeline_name_timestamp.json)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.pipeline_name}_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return str(filepath)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.pipeline_name}')"
