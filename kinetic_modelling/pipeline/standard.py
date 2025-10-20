"""
Standard pipeline for supervised learning workflows.

This module provides a straightforward pipeline that:
1. Loads train/test datasets
2. Applies sampling strategy
3. Trains a model
4. Evaluates on test set
"""

from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np

from .base import BasePipeline
from kinetic_modelling.data import MultiPressureDataset
from kinetic_modelling.model.base import BaseModel
from kinetic_modelling.sampling.base import BaseSampler


class StandardPipeline(BasePipeline):
    """
    Standard supervised learning pipeline.
    
    Coordinates: Data → Sampling → Training → Evaluation
    
    Example:
        ```python
        # Setup components
        train_dataset = MultiPressureDataset(...)
        test_dataset = MultiPressureDataset(...)
        model = SVRModel(...)
        sampler = SequentialSampler(...)
        
        # Create and run pipeline
        pipeline = StandardPipeline(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            sampler=sampler,
            pipeline_name="my_experiment"
        )
        results = pipeline.run()
        ```
    """
    
    def __init__(
        self,
        train_dataset: MultiPressureDataset,
        test_dataset: MultiPressureDataset,
        model: BaseModel,
        sampler: Optional[BaseSampler] = None,
        n_samples: Optional[int] = None,
        pipeline_name: str = "standard_pipeline",
        results_dir: str = "pipeline_results"
    ):
        """
        Initialize standard pipeline.
        
        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            model: ML model to train
            sampler: Sampling strategy (None = use all data)
            n_samples: Number of samples to use (None = use all)
            pipeline_name: Name for this pipeline
            results_dir: Directory to save results
        """
        super().__init__(pipeline_name, results_dir)
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.sampler = sampler
        self.n_samples = n_samples
        
        # Store configuration
        self.results['config'] = {
            'model_type': model.__class__.__name__,
            'model_name': model.model_name,
            'sampler': sampler.sampler_name if sampler else 'None (full dataset)',
            'n_samples': n_samples if n_samples else len(train_dataset),
            'total_train_samples': len(train_dataset),
            'test_samples': len(test_dataset)
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete pipeline.
        
        Steps:
        1. Apply sampling (if specified)
        2. Train model
        3. Evaluate on test set
        4. Store results
        
        Returns:
            Dictionary with complete pipeline results
        """
        print(f"\n{'='*70}")
        print(f"Running Pipeline: {self.pipeline_name}")
        print(f"{'='*70}")
        
        # Record start time
        start_time = datetime.now()
        self.results['timestamp'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Step 1: Apply sampling
        print("\n[1/3] Applying sampling strategy...")
        if self.sampler and self.n_samples:
            print(f"  Using sampler: {self.sampler.sampler_name}")
            print(f"  Sampling {self.n_samples} from {len(self.train_dataset)} samples")
            training_dataset = self.sampler.sample(self.train_dataset, self.n_samples)
        else:
            print(f"  Using full training dataset: {len(self.train_dataset)} samples")
            training_dataset = self.train_dataset
        
        x_train, y_train = training_dataset.get_data()
        x_test, y_test = self.test_dataset.get_data()
        
        self.results['data_info'] = {
            'train_samples': len(x_train),
            'test_samples': len(x_test),
            'n_features': x_train.shape[1],
            'n_outputs': y_train.shape[1]
        }
        print(f"  ✓ Train: {x_train.shape}, Test: {x_test.shape}")
        
        # Step 2: Train model
        print(f"\n[2/3] Training {self.model.__class__.__name__}...")
        train_history = self.model.fit(x_train, y_train)
        
        self.results['training_info'] = {
            'history': train_history,
            'model_metadata': self.model.metadata
        }
        print(f"  ✓ Training completed")
        
        # Step 3: Evaluate
        print(f"\n[3/3] Evaluating on test set...")
        
        # Train set evaluation
        y_pred_train = self.model.predict(x_train)
        train_metrics = self.model.evaluate(x_train, y_train)
        
        # Test set evaluation
        y_pred_test = self.model.predict(x_test)
        test_metrics = self.model.evaluate(x_test, y_test)
        
        self.results['evaluation'] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Pipeline Results Summary")
        print(f"{'='*70}")
        print(f"Training samples: {len(x_train)}")
        print(f"Test samples: {len(x_test)}")
        print(f"\nTrain Performance:")
        if 'r2_score' in train_metrics:
            print(f"  R² Score: {train_metrics['r2_score']:.4f}")
        if 'mse' in train_metrics:
            print(f"  MSE:      {train_metrics['mse']:.6e}")
        elif 'total_mse' in train_metrics:
            print(f"  Total MSE: {train_metrics['total_mse']:.6e}")
        print(f"\nTest Performance:")
        if 'r2_score' in test_metrics:
            print(f"  R² Score: {test_metrics['r2_score']:.4f}")
        if 'mse' in test_metrics:
            print(f"  MSE:      {test_metrics['mse']:.6e}")
        elif 'total_mse' in test_metrics:
            print(f"  Total MSE: {test_metrics['total_mse']:.6e}")
        
        # Calculate time elapsed
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        self.results['execution_time_seconds'] = elapsed
        print(f"\nExecution time: {elapsed:.2f} seconds")
        print(f"{'='*70}\n")
        
        return self.results
    
    def save_and_return(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Run pipeline and optionally save results.
        
        Args:
            save_results: Whether to save results to JSON
            
        Returns:
            Pipeline results dictionary
        """
        results = self.run()
        
        if save_results:
            filepath = self.save_results()
            print(f"✓ Results saved to: {filepath}\n")
        
        return results
