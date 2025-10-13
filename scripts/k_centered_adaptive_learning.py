#!/usr/bin/env python3
"""
K-Centered Adaptive Learning Script

This script implements a novel approach to active learning that focuses on improving
predictions around the true K value rather than minimizing overall test error.

Key differences from standard adaptive learning:
1. Uses true composition to get predicted K from current model
2. Creates a bounding box around the predicted K 
3. Selects training samples that fall within this K-centered region
4. Iteratively refines the model in this focused area

This approach is designed for scenarios where we know the true composition
but want to accurately determine the corresponding K values.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from sklearn.utils import shuffle

# Add current directory to path for imports
sys.path.append('.')
from active_learning_methods import (
    load_datasets, 
    train_initial_models, 
    apply_training_scalers,
    retrain_models_with_new_data,
    evaluate_zones_scaled
)

class KCenteredAdaptiveLearner:
    """
    K-Centered Adaptive Learning implementation.
    
    This class handles the core logic of:
    1. Predicting K from true composition
    2. Defining K-centered sampling regions
    3. Selecting training samples within those regions
    4. Iterative model improvement
    """
    
    def __init__(self, config, debug=True):
        """Initialize the K-centered adaptive learner."""
        self.config = config
        self.debug = debug
        self.nspecies = config['nspecies']
        self.n_pressure_conditions = config['num_pressure_conditions']
        
        # Store datasets and models
        self.uniform_dataset = None
        self.test_dataset = None
        self.current_models = None
        
        # Store multiple batch files data
        self.batch_files_data = []  # List of batch file info dicts
        
        # Track learning progress
        self.iteration_results = []
        
    def load_data(self, uniform_file, test_file, batch_json_paths):
        """Load all required datasets with multiple batch files."""
        print(f"📂 Loading datasets...")
        
        # Load uniform and test datasets
        self.uniform_dataset, self.test_dataset = load_datasets(
            uniform_file, test_file, self.nspecies, self.n_pressure_conditions
        )
        
        # Handle multiple batch files
        if isinstance(batch_json_paths, str):
            batch_json_paths = [batch_json_paths]  # Convert single file to list
        
        # Store batch file paths for sequential processing
        self.batch_file_paths = batch_json_paths
        self.current_batch_index = 0
        
        # Load all batch files and store them separately
        self.batch_files_data = []
        total_samples = 0
        
        for i, batch_path in enumerate(batch_json_paths):
            print(f"   Loading batch file {i+1}/{len(batch_json_paths)}: {batch_path}")
            
            with open(batch_path, 'r') as f:
                batch_data = json.load(f)
            
            batch_k_values = np.array([ps['k_values'] for ps in batch_data['parameter_sets']])
            batch_compositions = np.array(batch_data['compositions'])
            
            # Pre-scale batch data
            batch_x_scaled, batch_y_scaled = apply_training_scalers(
                raw_compositions=batch_compositions,
                raw_k_values=batch_k_values,
                dataset_train=self.uniform_dataset,
                nspecies=self.nspecies,
                num_pressure_conditions=self.n_pressure_conditions,
                debug=False
            )
            
            # Store processed batch data
            batch_info = {
                'file_path': batch_path,
                'batch_x_scaled': batch_x_scaled,
                'batch_y_scaled': batch_y_scaled,
                'batch_k_values': batch_k_values,
                'used_indices': set(),
                'total_samples': len(batch_x_scaled)
            }
            
            self.batch_files_data.append(batch_info)
            total_samples += len(batch_x_scaled)
            
            print(f"      Samples in this batch: {len(batch_x_scaled)}")
        
        print(f"✅ Data loaded:")
        print(f"   Uniform dataset: {self.uniform_dataset.get_data()[0].shape}")
        print(f"   Test dataset: {self.test_dataset.get_data()[0].shape}")
        print(f"   Total batch files: {len(self.batch_files_data)}")
        print(f"   Total batch samples: {total_samples}")
        
    def get_batch_file_status(self):
        """Get current status of all batch files."""
        status = []
        for i, batch_info in enumerate(self.batch_files_data):
            used_count = len(batch_info['used_indices'])
            total_count = batch_info['total_samples']
            available_count = total_count - used_count
            usage_pct = (used_count / total_count) * 100 if total_count > 0 else 0
            
            status.append({
                'file_index': i,
                'file_path': batch_info['file_path'],
                'total_samples': total_count,
                'used_samples': used_count,
                'available_samples': available_count,
                'usage_percentage': usage_pct
            })
        return status
    
    def print_batch_file_status(self):
        """Print current status of all batch files."""
        status = self.get_batch_file_status()
        print(f"\n📊 BATCH FILE STATUS ({len(status)} files)")
        print(f"{'File':>6} {'Total':>8} {'Used':>8} {'Available':>10} {'Usage %':>8}")
        print(f"{'-'*50}")
        
        for s in status:
            file_num = s['file_index'] + 1
            total = s['total_samples']
            used = s['used_samples']
            available = s['available_samples']
            usage = s['usage_percentage']
            print(f"{file_num:>6} {total:>8} {used:>8} {available:>10} {usage:>7.1f}%")
        
    def get_true_composition(self):
        """
        Get the true composition (first point from test dataset).
        This represents the known/measured composition.
        """
        test_x, test_y = self.test_dataset.get_data()
        true_composition = test_x[0:1]  # Keep as 2D array for prediction
        true_k_scaled = test_y[0:1]     # Corresponding true K (scaled)
        
        if self.debug:
            print(f"🎯 True composition (scaled): {true_composition.flatten()}")
            print(f"🎯 True K (scaled): {true_k_scaled.flatten()}")
            
        return true_composition, true_k_scaled
        
    def predict_k_from_composition(self, composition, models_per_seed):
        """
        Predict K values from given composition using current models.
        
        Args:
            composition: Input composition (1, n_features)
            models_per_seed: List of model lists (one per seed)
            
        Returns:
            predicted_k: Averaged prediction across seeds and outputs (1, n_outputs)
        """
        if not models_per_seed or not models_per_seed[0]:
            raise ValueError("No models available for prediction")
            
        # Collect predictions from all seeds
        seed_predictions = []
        
        for seed_models in models_per_seed:
            # Predict with each output model for this seed
            output_predictions = []
            for output_idx, model in enumerate(seed_models):
                pred = model.predict(composition)
                output_predictions.append(pred[0])  # Single prediction value
            
            seed_predictions.append(output_predictions)
        
        # Average across seeds
        predicted_k = np.mean(seed_predictions, axis=0, keepdims=True)
        
        if self.debug:
            print(f"🔮 Predicted K (scaled): {predicted_k.flatten()}")
            
        return predicted_k
        
    def define_k_bounding_box(self, predicted_k, box_size_factor=0.3, iteration=0, shrink_rate=0.8):
        """
        Define a bounding box around the predicted K values with adaptive shrinking.
        
        Args:
            predicted_k: Predicted K values (1, n_outputs)
            box_size_factor: Initial size of bounding box as fraction of prediction value
            iteration: Current iteration number (0-based)
            shrink_rate: Factor to shrink box each iteration (0.8 = 20% smaller each time)
            
        Returns:
            k_bounds: List of (min, max) tuples for each K dimension
        """
        # Calculate adaptive box size - shrinks each iteration
        adaptive_box_factor = box_size_factor * (shrink_rate ** iteration)
        
        k_bounds = []
        
        for k_idx in range(predicted_k.shape[1]):
            k_val = predicted_k[0, k_idx]
            
            # Define bounds as ± adaptive_box_factor of the predicted value
            delta = abs(k_val) * adaptive_box_factor
            k_min = k_val - delta
            k_max = k_val + delta
            
            # Ensure bounds are within [0, 1] since data is scaled to [0, 1]
            k_min = max(0.0, k_min)
            k_max = min(1.0, k_max)
            
            k_bounds.append((k_min, k_max))
            
            if self.debug:
                print(f"   K{k_idx+1} bounds: [{k_min:.4f}, {k_max:.4f}] (center: {k_val:.4f}, factor: {adaptive_box_factor:.3f})")
                
        if self.debug:
            print(f"   📏 Box size factor: {box_size_factor:.3f} → {adaptive_box_factor:.3f} (iteration {iteration})")
                
        return k_bounds
        
    def select_samples_in_k_region(self, k_bounds, n_samples):
        """
        Select batch samples that fall within the K bounding box from multiple files sequentially.
        
        Args:
            k_bounds: List of (min, max) tuples for each K dimension
            n_samples: Number of samples to select
            
        Returns:
            selected_x: Selected input samples (n_samples, n_features)
            selected_y: Selected output samples (n_samples, n_outputs)  
            selected_indices: List of tuples (file_index, sample_index)
        """
        selected_x_list = []
        selected_y_list = []
        selected_indices = []
        remaining_samples = n_samples
        
        if self.debug:
            print(f"   🔍 Searching for samples across {len(self.batch_files_data)} batch files...")
        
        # Try each batch file sequentially until we have enough samples
        for file_idx, batch_info in enumerate(self.batch_files_data):
            if remaining_samples <= 0:
                break
                
            batch_x = batch_info['batch_x_scaled']
            batch_y = batch_info['batch_y_scaled']
            used_indices = batch_info['used_indices']
            
            # Find samples within K bounds that haven't been used yet
            available_indices = []
            
            for idx in range(len(batch_y)):
                if idx in used_indices:
                    continue  # Skip already used samples
                    
                # Check if sample's K values fall within all bounds
                sample_k = batch_y[idx]
                within_bounds = True
                
                for k_idx, (min_k, max_k) in enumerate(k_bounds):
                    if not (min_k <= sample_k[k_idx] <= max_k):
                        within_bounds = False
                        break
                        
                if within_bounds:
                    available_indices.append(idx)
            
            if self.debug:
                print(f"      File {file_idx+1}: {len(available_indices)} available samples (of {len(batch_y)} total, {len(used_indices)} used)")
                
            # Select samples from this file
            if len(available_indices) > 0:
                n_select_from_file = min(remaining_samples, len(available_indices))
                rng = np.random.default_rng(self.config.get('random_seed', 42))
                file_selected_indices = rng.choice(available_indices, size=n_select_from_file, replace=False)
                
                # Add selected samples
                selected_x_list.append(batch_x[file_selected_indices])
                selected_y_list.append(batch_y[file_selected_indices])
                
                # Track which samples were used
                for idx in file_selected_indices:
                    batch_info['used_indices'].add(idx)
                    selected_indices.append((file_idx, idx))
                
                remaining_samples -= n_select_from_file
                
                if self.debug:
                    print(f"         ✅ Selected {n_select_from_file} samples from file {file_idx+1}")
        
        # Combine results from all files
        if len(selected_x_list) == 0:
            if self.debug:
                print(f"   ❌ No samples found in K-region across all {len(self.batch_files_data)} files")
            return None, None, []
            
        selected_x = np.vstack(selected_x_list)
        selected_y = np.vstack(selected_y_list)
        
        if self.debug:
            print(f"   📊 Selection summary:")
            print(f"      Total selected: {len(selected_x)} samples")
            print(f"      From {len(selected_x_list)} different files")
            if remaining_samples > 0:
                print(f"      ⚠️ Still need {remaining_samples} more samples")
            
        return selected_x, selected_y, selected_indices
    
    def reset_batch_files_usage(self):
        """Reset all batch files to unused state."""
        for batch_info in self.batch_files_data:
            batch_info['used_indices'] = set()
    
    def shuffle_data_with_seed(self, seed):
        """Shuffle both uniform and batch data using the provided seed."""
        print(f"   🔀 Shuffling data with seed {seed}")
        
        # Shuffle uniform dataset
        x_uniform, y_uniform = self.uniform_dataset.get_data()
        x_shuffled, y_shuffled = shuffle(x_uniform, y_uniform, random_state=seed)
        self.uniform_dataset.x_data = x_shuffled
        self.uniform_dataset.y_data = y_shuffled
        
        # Shuffle each batch file's data
        for i, batch_info in enumerate(self.batch_files_data):
            batch_x_shuffled, batch_y_shuffled = shuffle(
                batch_info['batch_x_scaled'], 
                batch_info['batch_y_scaled'], 
                random_state=seed + i  # Different seed per file to avoid identical shuffling
            )
            batch_info['batch_x_scaled'] = batch_x_shuffled
            batch_info['batch_y_scaled'] = batch_y_shuffled

    def run_single_k_centered_experiment(self, n_iterations=10, samples_per_iteration=20, 
                                       initial_uniform_size=100, box_size_factor=0.3, 
                                       shrink_rate=0.8, seed=42):
        """
        Run a single K-centered learning experiment with a specific seed.
        
        Args:
            n_iterations: Number of adaptive iterations
            samples_per_iteration: Samples to add per iteration
            initial_uniform_size: Initial uniform training samples
            box_size_factor: Initial K bounding box size factor
            shrink_rate: Factor to shrink box each iteration
            seed: Random seed for this experiment
            
        Returns:
            iteration_results: List of results for each iteration
        """
        print(f"\n🌱 SINGLE K-CENTERED EXPERIMENT (Seed: {seed})")
        print(f"   Initial uniform size: {initial_uniform_size}")
        print(f"   Iterations: {n_iterations}")
        print(f"   Samples per iteration: {samples_per_iteration}")
        print(f"   Initial K box size factor: {box_size_factor}")
        print(f"   Box shrink rate: {shrink_rate}")
        
        # Reset batch file usage for this seed
        self.reset_batch_files_usage()
        
        # Shuffle data with this seed
        self.shuffle_data_with_seed(seed)
        
        # Set random seed for sample selection
        original_random_seed = self.config.get('random_seed')
        self.config['random_seed'] = seed
        
        # Get true composition (the target we want to predict K for)
        true_composition, true_k_scaled = self.get_true_composition()
        
        # Start with initial uniform training data
        x_uniform, y_uniform = self.uniform_dataset.get_data()
        current_x_train = x_uniform[:initial_uniform_size].copy()
        current_y_train = y_uniform[:initial_uniform_size].copy()
        
        print(f"\n🎯 ITERATION 0 - Initial Training (Seed {seed})")
        
        # Train initial models
        class TempDataset:
            def __init__(self, x_data, y_data):
                self.x_data = x_data
                self.y_data = y_data
            def get_data(self):
                return self.x_data, self.y_data
        
        temp_dataset = TempDataset(current_x_train, current_y_train)
        initial_models_per_seed, initial_mse_per_output, initial_total_mse = train_initial_models(
            temp_dataset, self.test_dataset, self.config['svr_params'], 
            n_initial_samples=current_x_train.shape[0], seeds=self.config.get('seeds')
        )
        
        # Get initial prediction for true composition
        initial_predicted_k = self.predict_k_from_composition(true_composition, initial_models_per_seed)
        
        # Calculate initial prediction error
        initial_k_error = np.linalg.norm(initial_predicted_k - true_k_scaled)
        
        print(f"   Initial prediction error for true composition: {initial_k_error:.6f}")
        
        # Store initial results
        iteration_results = [{
            'iteration': 0,
            'seed': seed,
            'total_samples': int(current_x_train.shape[0]),
            'batch_samples_added': 0,
            'test_mse_per_output': [float(mse) for mse in initial_mse_per_output],
            'test_total_mse': float(initial_total_mse),
            'predicted_k': [float(k) for k in initial_predicted_k.flatten()],
            'true_k': [float(k) for k in true_k_scaled.flatten()],
            'k_prediction_error': float(initial_k_error),
            'box_size_factor': float(box_size_factor),
            'mse_type': 'initial'
        }]
        
        current_models = initial_models_per_seed
        
        # Run adaptive iterations
        for iteration in range(1, n_iterations + 1):
            print(f"\n🎯 ITERATION {iteration} (Seed {seed})")
            
            # Step 1: Get current prediction for true composition
            predicted_k = self.predict_k_from_composition(true_composition, current_models)
            
            # Step 2: Define K-centered bounding box (adaptive shrinking)
            k_bounds = self.define_k_bounding_box(predicted_k, box_size_factor, iteration, shrink_rate)
            
            # Step 3: Select samples within K region (uses self.config['random_seed'] = seed)
            new_x, new_y_scaled, selected_indices = self.select_samples_in_k_region(
                k_bounds, samples_per_iteration
            )
            
            if new_x is None:
                print(f"   ❌ No samples found in K-region for iteration {iteration} (Seed {seed})")
                print(f"      Box may be too small. Current factor: {box_size_factor * (shrink_rate ** iteration):.4f}")
                break
                
            # Step 4: Retrain models with K-centered samples
            print(f"   🔄 Retraining with K-centered samples (Seed {seed})...")
            retrain_models_per_seed, retrain_mse_per_output, retrain_total_mse, augmented_size, _, _ = retrain_models_with_new_data(
                current_x_train=current_x_train,
                current_y_train=current_y_train,
                dataset_test=self.test_dataset,
                new_x=new_x,
                new_y_scaled=new_y_scaled,
                best_params=self.config['svr_params'],
                seeds=self.config.get('seeds'),
                debug=False
            )
            
            # Step 5: Evaluate new prediction for true composition
            new_predicted_k = self.predict_k_from_composition(true_composition, retrain_models_per_seed)
            k_error = np.linalg.norm(new_predicted_k - true_k_scaled)
            
            # Update training data and models
            current_x_train = np.vstack([current_x_train, new_x])
            current_y_train = np.vstack([current_y_train, new_y_scaled])
            current_models = retrain_models_per_seed
            
            # Calculate current adaptive box size
            current_box_factor = box_size_factor * (shrink_rate ** iteration)
            
            print(f"   ✅ Iteration {iteration} complete (Seed {seed}):")
            print(f"      Total training samples: {augmented_size}")
            print(f"      Test MSE: {retrain_total_mse:.6f}")
            print(f"      K prediction error: {k_error:.6f}")
            print(f"      Current box factor: {current_box_factor:.4f}")
            
            # Calculate improvement in K prediction
            prev_k_error = iteration_results[-1]['k_prediction_error']
            k_improvement = ((prev_k_error - k_error) / prev_k_error) * 100
            print(f"      K prediction improvement: {k_improvement:+.1f}%")
            
            # Store results
            iteration_results.append({
                'iteration': iteration,
                'seed': seed,
                'total_samples': int(augmented_size),
                'batch_samples_added': int(len(new_x)),
                'test_mse_per_output': [float(mse) for mse in retrain_mse_per_output],
                'test_total_mse': float(retrain_total_mse),
                'predicted_k': [float(k) for k in new_predicted_k.flatten()],
                'true_k': [float(k) for k in true_k_scaled.flatten()],
                'k_prediction_error': float(k_error),
                'k_improvement_pct': float(k_improvement),
                'k_bounds': [(float(b[0]), float(b[1])) for b in k_bounds],
                'box_size_factor': float(current_box_factor),
                'selected_indices': [(int(file_idx), int(sample_idx)) for file_idx, sample_idx in selected_indices],
                'mse_type': 'k_centered_retrain'
            })
        
        # Restore original random seed
        if original_random_seed is not None:
            self.config['random_seed'] = original_random_seed
        
        return iteration_results

    def average_results_across_seeds(self, all_seed_results):
        """Average results across multiple seed experiments."""
        if not all_seed_results:
            return []
        
        # Determine maximum iterations across all seeds
        max_iterations = max(len(seed_results) for seed_results in all_seed_results)
        averaged_results = []
        
        for iter_idx in range(max_iterations):
            # Collect results from all seeds for this iteration
            iter_data = []
            for seed_results in all_seed_results:
                if iter_idx < len(seed_results):
                    iter_data.append(seed_results[iter_idx])
            
            if not iter_data:
                continue
            
            # Average numeric fields
            avg_result = {
                'iteration': iter_idx,
                'total_samples': int(np.mean([r['total_samples'] for r in iter_data])),
                'batch_samples_added': int(np.mean([r['batch_samples_added'] for r in iter_data])),
                'test_mse_per_output': np.mean([r['test_mse_per_output'] for r in iter_data], axis=0).tolist(),
                'test_total_mse': float(np.mean([r['test_total_mse'] for r in iter_data])),
                'predicted_k': np.mean([r['predicted_k'] for r in iter_data], axis=0).tolist(),
                'true_k': iter_data[0]['true_k'],  # Same for all seeds
                'k_prediction_error': float(np.mean([r['k_prediction_error'] for r in iter_data])),
                'box_size_factor': float(np.mean([r['box_size_factor'] for r in iter_data])),
                'mse_type': iter_data[0]['mse_type'],
                'seeds_used': [r.get('seed', 'unknown') for r in iter_data],
                'n_seeds': len(iter_data)
            }
            
            # Calculate improvement if not first iteration
            if iter_idx > 0:
                prev_k_error = averaged_results[-1]['k_prediction_error']
                k_improvement = ((prev_k_error - avg_result['k_prediction_error']) / prev_k_error) * 100
                avg_result['k_improvement_pct'] = float(k_improvement)
            
            averaged_results.append(avg_result)
        
        return averaged_results

    def run_k_centered_learning_proper_seeds(self, n_iterations=10, samples_per_iteration=20, 
                                            initial_uniform_size=100, box_size_factor=0.3, 
                                            shrink_rate=0.8, seeds=None):
        """
        Run K-centered learning with proper multi-seed sampling and averaging.
        
        Args:
            n_iterations: Number of adaptive iterations
            samples_per_iteration: Samples to add per iteration  
            initial_uniform_size: Initial uniform training samples
            box_size_factor: Initial K bounding box size factor
            shrink_rate: Factor to shrink box each iteration
            seeds: List of seeds for multiple experiments
            
        Returns:
            averaged_results: Results averaged across all seeds
            all_seed_results: Individual results for each seed
        """
        if seeds is None:
            seeds = [42, 43, 44, 45, 46]  # Default 5 seeds
        
        print(f"\n🎯 MULTI-SEED K-CENTERED ADAPTIVE LEARNING")
        print(f"   Number of seeds: {len(seeds)}")
        print(f"   Seeds: {seeds}")
        print(f"   Each seed will get different data shuffling AND sample selection")
        
        all_seed_results = []
        
        for i, seed in enumerate(seeds):
            print(f"\n{'='*60}")
            print(f"🌱 EXPERIMENT {i+1}/{len(seeds)} - SEED {seed}")
            print(f"{'='*60}")
            
            # Run single experiment with this seed
            seed_results = self.run_single_k_centered_experiment(
                n_iterations=n_iterations,
                samples_per_iteration=samples_per_iteration,
                initial_uniform_size=initial_uniform_size,
                box_size_factor=box_size_factor,
                shrink_rate=shrink_rate,
                seed=seed
            )
            
            all_seed_results.append(seed_results)
            
            # Print summary for this seed
            if seed_results:
                final_result = seed_results[-1]
                print(f"\n📊 Seed {seed} Summary:")
                print(f"   Final MSE: {final_result['test_total_mse']:.6f}")
                print(f"   Final K error: {final_result['k_prediction_error']:.6f}")
                print(f"   Total samples: {final_result['total_samples']}")
                print(f"   Iterations completed: {len(seed_results)-1}")
        
        # Average results across all seeds
        print(f"\n{'='*60}")
        print(f"🧮 AVERAGING RESULTS ACROSS {len(seeds)} SEEDS")
        print(f"{'='*60}")
        
        averaged_results = self.average_results_across_seeds(all_seed_results)
        
        return averaged_results, all_seed_results
        
    def run_k_centered_learning(self, n_iterations=10, samples_per_iteration=20, 
                               initial_uniform_size=100, box_size_factor=0.3, shrink_rate=0.8):
        """
        Run the main K-centered adaptive learning loop.
        
        Args:
            n_iterations: Number of adaptive iterations
            samples_per_iteration: Samples to add per iteration
            initial_uniform_size: Initial uniform training samples
            box_size_factor: Initial K bounding box size factor
            shrink_rate: Factor to shrink box each iteration (0.8 = 20% reduction)
        """
        print(f"\n🎯 K-CENTERED ADAPTIVE LEARNING")
        print(f"   Initial uniform size: {initial_uniform_size}")
        print(f"   Iterations: {n_iterations}")
        print(f"   Samples per iteration: {samples_per_iteration}")
        print(f"   Initial K box size factor: {box_size_factor}")
        print(f"   Box shrink rate: {shrink_rate} (adaptive zoom-in)")
        
        # Get true composition (the target we want to predict K for)
        true_composition, true_k_scaled = self.get_true_composition()
        
        # Start with initial uniform training data
        x_uniform, y_uniform = self.uniform_dataset.get_data()
        current_x_train = x_uniform[:initial_uniform_size].copy()
        current_y_train = y_uniform[:initial_uniform_size].copy()
        
        print(f"\n🎯 ITERATION 0 - Initial Training")
        
        # Train initial models
        class TempDataset:
            def __init__(self, x_data, y_data):
                self.x_data = x_data
                self.y_data = y_data
            def get_data(self):
                return self.x_data, self.y_data
        
        temp_dataset = TempDataset(current_x_train, current_y_train)
        initial_models_per_seed, initial_mse_per_output, initial_total_mse = train_initial_models(
            temp_dataset, self.test_dataset, self.config['svr_params'], 
            n_initial_samples=current_x_train.shape[0], seeds=self.config.get('seeds')
        )
        
        # Get initial prediction for true composition
        initial_predicted_k = self.predict_k_from_composition(true_composition, initial_models_per_seed)
        
        # Calculate initial prediction error
        initial_k_error = np.linalg.norm(initial_predicted_k - true_k_scaled)
        
        print(f"   Initial prediction error for true composition: {initial_k_error:.6f}")
        
        # Store initial results
        self.iteration_results.append({
            'iteration': 0,
            'total_samples': int(current_x_train.shape[0]),
            'batch_samples_added': 0,
            'test_mse_per_output': [float(mse) for mse in initial_mse_per_output],
            'test_total_mse': float(initial_total_mse),
            'predicted_k': [float(k) for k in initial_predicted_k.flatten()],
            'true_k': [float(k) for k in true_k_scaled.flatten()],
            'k_prediction_error': float(initial_k_error),
            'box_size_factor': float(box_size_factor),  # Track adaptive box size
            'mse_type': 'initial'
        })
        
        self.current_models = initial_models_per_seed
        
        # Run adaptive iterations
        for iteration in range(1, n_iterations + 1):
            print(f"\n🎯 ITERATION {iteration}")
            
            # Debug: Show current batch file usage before this iteration
            total_used_so_far = sum(len(batch_info['used_indices']) for batch_info in self.batch_files_data)
            total_available = sum(batch_info['total_samples'] for batch_info in self.batch_files_data)
            print(f"   📊 Before iteration {iteration}: {total_used_so_far}/{total_available} batch samples used")
            
            # Step 1: Get current prediction for true composition
            predicted_k = self.predict_k_from_composition(true_composition, self.current_models)
            
            # Step 2: Define K-centered bounding box (adaptive shrinking)
            k_bounds = self.define_k_bounding_box(predicted_k, box_size_factor, iteration, shrink_rate)
            
            # Step 3: Select samples within K region
            new_x, new_y_scaled, selected_indices = self.select_samples_in_k_region(
                k_bounds, samples_per_iteration
            )
            
            if new_x is None:
                print(f"   ❌ No samples found in K-region for iteration {iteration}")
                print(f"      Box may be too small. Current factor: {box_size_factor * (shrink_rate ** iteration):.4f}")
                print(f"      Consider decreasing shrink_rate or increasing initial box_size_factor")
                
                # Debug: Show final batch file status
                total_used_final = sum(len(batch_info['used_indices']) for batch_info in self.batch_files_data)
                print(f"      🔍 Total batch samples used at termination: {total_used_final}")
                self.print_batch_file_status()
                break
                
            # Step 4: Retrain models with K-centered samples
            print(f"   🔄 Retraining with K-centered samples...")
            retrain_models_per_seed, retrain_mse_per_output, retrain_total_mse, augmented_size, _, _ = retrain_models_with_new_data(
                current_x_train=current_x_train,
                current_y_train=current_y_train,
                dataset_test=self.test_dataset,
                new_x=new_x,
                new_y_scaled=new_y_scaled,
                best_params=self.config['svr_params'],
                seeds=self.config.get('seeds'),
                debug=False
            )
            
            # Step 5: Evaluate new prediction for true composition
            new_predicted_k = self.predict_k_from_composition(true_composition, retrain_models_per_seed)
            k_error = np.linalg.norm(new_predicted_k - true_k_scaled)
            
            # Update training data and models
            current_x_train = np.vstack([current_x_train, new_x])
            current_y_train = np.vstack([current_y_train, new_y_scaled])
            self.current_models = retrain_models_per_seed
            
            # Calculate current adaptive box size
            current_box_factor = box_size_factor * (shrink_rate ** iteration)
            
            print(f"   ✅ Iteration {iteration} complete:")
            print(f"      Total training samples: {augmented_size}")
            print(f"      Test MSE: {retrain_total_mse:.6f}")
            print(f"      K prediction error: {k_error:.6f}")
            print(f"      Current box factor: {current_box_factor:.4f}")
            
            # Calculate improvement in K prediction
            prev_k_error = self.iteration_results[-1]['k_prediction_error']
            k_improvement = ((prev_k_error - k_error) / prev_k_error) * 100
            print(f"      K prediction improvement: {k_improvement:+.1f}%")
            
            # Store results
            self.iteration_results.append({
                'iteration': iteration,
                'total_samples': int(augmented_size),
                'batch_samples_added': int(len(new_x)),
                'test_mse_per_output': [float(mse) for mse in retrain_mse_per_output],
                'test_total_mse': float(retrain_total_mse),
                'predicted_k': [float(k) for k in new_predicted_k.flatten()],
                'true_k': [float(k) for k in true_k_scaled.flatten()],
                'k_prediction_error': float(k_error),
                'k_improvement_pct': float(k_improvement),
                'k_bounds': [(float(b[0]), float(b[1])) for b in k_bounds],
                'box_size_factor': float(current_box_factor),  # Track adaptive box size
                'selected_indices': [(int(file_idx), int(sample_idx)) for file_idx, sample_idx in selected_indices],
                'mse_type': 'k_centered_retrain'
            })
            
        return self.iteration_results

def test_k_centered_components():
    """Test individual components of the K-centered approach."""
    print("🧪 TESTING K-CENTERED COMPONENTS")
    
    # Test configuration
    config = {
        'nspecies': 3,
        'num_pressure_conditions': 2,
        'pressure_conditions_pa': [133.322, 1333.22],
        'svr_params': [
            {'C': 10, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'},
            {'C': 20, 'epsilon': 0.005, 'gamma': 5, 'kernel': 'rbf'},
            {'C': 5, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'}
        ],
        'seeds': list(range(42, 46)),  # Use 3 different random seeds for testing
        'random_seed': 42
    }
    
    # Test file paths - multiple batch files from different uniform iterations
    uniform_file = 'data/SampleEfficiency/O2_simple_uniform.txt'
    test_file = 'data/SampleEfficiency/O2_simple_test_real_K.txt'
    batch_json_paths = [
        'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-08-27/batch_4000sims_20250827_010028.json',
        'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-28/batch_1000sims_20250928_191628.json',
        'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2500sims_20250929_031845.json',
        'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2000sims_20250929_205429.json',
        'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-28/batch_1500sims_20250928_224858.json',
        'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2000sims_20250929_125706.json'
        # Each file represents a different uniform distribution iteration
        # File 1: 4000 sims (largest uniform distribution)
        # File 2: 2000 sims (medium uniform distribution) 
        # File 3: 1000 sims (smallest uniform distribution)
    ]
    
    # Initialize learner
    learner = KCenteredAdaptiveLearner(config, debug=True)
    
    print("\n1️⃣ Testing data loading...")
    learner.load_data(uniform_file, test_file, batch_json_paths)

    return learner, None
    
    print("\n2️⃣ Testing true composition extraction...")
    true_comp, true_k = learner.get_true_composition()
    
    print("\n3️⃣ Testing initial model training...")
    x_uniform, y_uniform = learner.uniform_dataset.get_data()
    
    class TempDataset:
        def __init__(self, x_data, y_data):
            self.x_data = x_data
            self.y_data = y_data
        def get_data(self):
            return self.x_data, self.y_data
    
    temp_dataset = TempDataset(x_uniform[:100], y_uniform[:100])
    models_per_seed, _, _ = train_initial_models(
        temp_dataset, learner.test_dataset, config['svr_params'], 
        n_initial_samples=100, seeds=config.get('seeds')
    )
    
    print("\n4️⃣ Testing K prediction...")
    predicted_k = learner.predict_k_from_composition(true_comp, models_per_seed)
    
    print("\n5️⃣ Testing K bounding box...")
    k_bounds = learner.define_k_bounding_box(predicted_k, box_size_factor=0.3)
    
    print("\n6️⃣ Testing sample selection across multiple files...")
    selected_x, selected_y, selected_indices = learner.select_samples_in_k_region(k_bounds, 10)
    
    if selected_x is not None:
        print(f"   ✅ Successfully selected {len(selected_x)} samples")
        print(f"   📊 Sample sources: {len(set(idx[0] for idx in selected_indices))} different files")
        file_counts = {}
        for file_idx, sample_idx in selected_indices:
            file_counts[file_idx] = file_counts.get(file_idx, 0) + 1
        print(f"   📋 File distribution: {[(f'File {i+1}', count) for i, count in file_counts.items()]}")
        print(f"   Selected K values range:")
        for k_idx in range(selected_y.shape[1]):
            k_vals = selected_y[:, k_idx]
            print(f"      K{k_idx+1}: [{k_vals.min():.4f}, {k_vals.max():.4f}]")
    else:
        print("   ❌ No samples selected")
        
    print("\n🎉 Component testing complete!")
    return learner, models_per_seed

if __name__ == '__main__':
    print("🚀 K-CENTERED ADAPTIVE LEARNING SCRIPT")
    
    # First, test components
    print("\n" + "="*80)
    learner, initial_models = test_k_centered_components()
  
    print("\n" + "="*80)
    print("🎯 RUNNING MULTI-SEED K-CENTERED EXPERIMENT")
    
    # Run the experiment with proper multi-seed sampling
    averaged_results, all_seed_results = learner.run_k_centered_learning_proper_seeds(
        n_iterations=10,
        samples_per_iteration=100,
        initial_uniform_size=100,
        box_size_factor=1,  # Proper K-centered box
        shrink_rate=0.50,      # 20% reduction each iteration
        seeds=list(range(42, 50))   # 3 seeds for testing
    )
    
    # Print summary
    print(f"\n📊 MULTI-SEED K-CENTERED LEARNING RESULTS (Averaged)")
    print(f"{'Iter':>4} {'Samples':>8} {'Added':>6} {'Test MSE':>10} {'K Error':>10} {'K Improve':>10} {'Box Factor':>11}")
    print(f"{'-'*72}")
    
    for result in averaged_results:
        iter_num = result['iteration']
        total_samples = result['total_samples']
        added = result['batch_samples_added']
        test_mse = result['test_total_mse']
        k_error = result['k_prediction_error']
        k_improve = result.get('k_improvement_pct', 0.0)
        box_factor = result.get('box_size_factor', 'N/A')
        
        if isinstance(box_factor, (int, float)):
            box_str = f"{box_factor:.3f}"
        else:
            box_str = str(box_factor)
            
        print(f"{iter_num:>4} {total_samples:>8} {added:>6} {test_mse:>10.6f} {k_error:>10.6f} {k_improve:>+9.1f}% {box_str:>11}")
    
    # Print adaptive shrinking summary
    if len(averaged_results) > 1:
        initial_box = averaged_results[1].get('box_size_factor', 'N/A')  # Skip iteration 0
        final_box = averaged_results[-1].get('box_size_factor', 'N/A')
        if isinstance(initial_box, (int, float)) and isinstance(final_box, (int, float)):
            shrinking_ratio = final_box / initial_box
            print(f"\n📦 Box Size Progression:")
            print(f"   Initial box factor: {initial_box:.3f}")
            print(f"   Final box factor:   {final_box:.3f}")
            print(f"   Shrinking ratio:    {shrinking_ratio:.3f} ({(1-shrinking_ratio)*100:.1f}% total reduction)")
        
        # Print detailed box progression
        print(f"\n📏 Box Factor History:")
        for i, result in enumerate(averaged_results[1:], 1):
            box_factor = result.get('box_size_factor', 'N/A')
            if isinstance(box_factor, (int, float)):
                print(f"   Iteration {i}: {box_factor:.4f}")
    
    print(f"   (Shrink rate: 0.75 = 25% reduction per iteration)")
    
    # Show final batch file usage
    learner.print_batch_file_status()
    
    # Print multi-file summary
    print(f"\n📊 MULTI-FILE USAGE SUMMARY:")
    total_used_across_files = sum(len(batch_info['used_indices']) for batch_info in learner.batch_files_data)
    total_available_across_files = sum(batch_info['total_samples'] for batch_info in learner.batch_files_data)
    
    print(f"   Total samples used across all files: {total_used_across_files}")
    print(f"   Total samples available across all files: {total_available_across_files}")
    print(f"   Overall usage percentage: {(total_used_across_files/total_available_across_files)*100:.1f}%")
    
    # Show which files were actually used
    files_used = [i+1 for i, batch_info in enumerate(learner.batch_files_data) if len(batch_info['used_indices']) > 0]
    print(f"   Files utilized: {files_used} of {len(learner.batch_files_data)} total files")
    
    # Save results with proper JSON serialization
    results_file = 'results/k_centered_adaptive_multi_seed_results.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'nspecies': learner.config['nspecies'],
                'num_pressure_conditions': learner.config['num_pressure_conditions'],
                'pressure_conditions_pa': learner.config['pressure_conditions_pa'],
                'svr_params': learner.config['svr_params'],
                'seeds': learner.config['seeds'],
                'random_seed': learner.config['random_seed']
            },
            'experiment_type': 'k_centered_adaptive_learning_multi_seed',
            'batch_files': [batch_info['file_path'] for batch_info in learner.batch_files_data],
            'batch_file_usage': [
                {
                    'file_index': i,
                    'file_path': batch_info['file_path'],
                    'total_samples': int(batch_info['total_samples']),
                    'used_samples': len(batch_info['used_indices']),
                    'usage_percentage': float(len(batch_info['used_indices']) / batch_info['total_samples'] * 100)
                }
                for i, batch_info in enumerate(learner.batch_files_data)
            ],
            'averaged_results': averaged_results,  # Multi-seed averaged results
            'individual_seed_results': all_seed_results,  # Individual seed results for analysis
            'summary': {
                'initial_k_error': float(averaged_results[0]['k_prediction_error']),
                'final_k_error': float(averaged_results[-1]['k_prediction_error']),
                'k_improvement_total': float(((averaged_results[0]['k_prediction_error'] - averaged_results[-1]['k_prediction_error']) / averaged_results[0]['k_prediction_error']) * 100),
                'final_test_mse': float(averaged_results[-1]['test_total_mse']),
                'total_iterations': len(averaged_results) - 1,
                'total_training_samples': int(averaged_results[-1]['total_samples']),
                'adaptive_box_shrinking': {
                    'initial_box_factor': float(averaged_results[1]['box_size_factor']) if len(averaged_results) > 1 else 0.0,
                    'final_box_factor': float(averaged_results[-1]['box_size_factor']) if len(averaged_results) > 1 else 0.0,
                    'shrink_rate': 0.8,  # Updated to match our settings
                    'total_reduction_pct': float((1 - averaged_results[-1]['box_size_factor'] / averaged_results[1]['box_size_factor']) * 100) if len(averaged_results) > 1 else 0.0
                }
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"\n✅ MULTI-SEED K-CENTERED ADAPTIVE LEARNING COMPLETE!")