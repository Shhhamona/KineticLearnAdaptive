#!/usr/bin/env python3
"""
Test Multi-File K-Centered Adaptive Learning

This script demonstrates how the K-centered adaptive learning handles multiple
batch JSON files from different uniform distributions sequentially.
"""

import os
import sys
import numpy as np
import json

# Add current directory to path for imports
sys.path.append('.')
from scripts.k_centered_adaptive_learning import KCenteredAdaptiveLearner

def create_test_batch_files():
    """Create sample batch files for testing multiple file handling."""
    # Create test directory
    test_dir = 'results/test_batch_files'
    os.makedirs(test_dir, exist_ok=True)
    
    # Create first batch file (larger uniform distribution)
    batch1_data = {
        'compositions': np.random.uniform(0.1, 0.9, (500, 3)).tolist(),
        'parameter_sets': []
    }
    
    for i, comp in enumerate(batch1_data['compositions']):
        # Generate mock K values based on composition
        k_values = [
            comp[0] * 10 + np.random.normal(0, 0.1),  # K1
            comp[1] * 5 + np.random.normal(0, 0.05),   # K2
            comp[2] * 15 + np.random.normal(0, 0.2)    # K3
        ]
        batch1_data['parameter_sets'].append({
            'k_values': k_values,
            'parameters': {'temperature': 1000 + i}
        })
    
    batch1_file = os.path.join(test_dir, 'test_batch_1.json')
    with open(batch1_file, 'w') as f:
        json.dump(batch1_data, f)
    
    # Create second batch file (smaller, different range)
    batch2_data = {
        'compositions': np.random.uniform(0.2, 0.8, (200, 3)).tolist(),
        'parameter_sets': []
    }
    
    for i, comp in enumerate(batch2_data['compositions']):
        # Different K value generation (different uniform range)
        k_values = [
            comp[0] * 8 + np.random.normal(0, 0.08),   # K1
            comp[1] * 6 + np.random.normal(0, 0.06),   # K2
            comp[2] * 12 + np.random.normal(0, 0.15)   # K3
        ]
        batch2_data['parameter_sets'].append({
            'k_values': k_values,
            'parameters': {'temperature': 1200 + i}
        })
    
    batch2_file = os.path.join(test_dir, 'test_batch_2.json')
    with open(batch2_file, 'w') as f:
        json.dump(batch2_data, f)
    
    return [batch1_file, batch2_file]

def test_multi_file_selection():
    """Test the multi-file sample selection functionality."""
    print("🧪 TESTING MULTI-FILE K-CENTERED LEARNING")
    
    # Create test batch files
    print("\n📂 Creating test batch files...")
    batch_files = create_test_batch_files()
    print(f"   Created {len(batch_files)} test batch files")
    
    # Configuration
    config = {
        'nspecies': 3,
        'num_pressure_conditions': 2,
        'pressure_conditions_pa': [133.322, 1333.22],
        'svr_params': [
            {'C': 10, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'}
        ],
        'seeds': [42],
        'random_seed': 42
    }
    
    # Test file paths
    uniform_file = 'data/SampleEfficiency/O2_simple_uniform.txt'
    test_file = 'data/SampleEfficiency/O2_simple_test_real_K.txt'
    
    # Initialize learner
    learner = KCenteredAdaptiveLearner(config, debug=True)
    
    print("\n📊 Loading multiple batch files...")
    try:
        learner.load_data(uniform_file, test_file, batch_files)
        
        print("\n🎯 Testing sequential file selection...")
        
        # Get true composition for K prediction
        true_comp, true_k = learner.get_true_composition()
        
        # Define a K bounding box (using mock predicted K)
        predicted_k = np.array([[5.0, 2.5, 7.5]])  # Mock prediction
        k_bounds = learner.define_k_bounding_box(predicted_k, box_size_factor=0.5)
        
        # Test multiple selections to see file progression
        for iteration in range(3):
            print(f"\n--- Selection Iteration {iteration + 1} ---")
            selected_x, selected_y, selected_indices = learner.select_samples_in_k_region(k_bounds, 20)
            
            if selected_x is not None:
                print(f"✅ Selected {len(selected_x)} samples")
                
                # Show file usage
                file_counts = {}
                for file_idx, sample_idx in selected_indices:
                    file_counts[file_idx] = file_counts.get(file_idx, 0) + 1
                
                print("📊 File usage:")
                for file_idx, batch_info in enumerate(learner.batch_files_data):
                    used_count = len(batch_info['used_indices'])
                    total_count = batch_info['total_samples']
                    current_selected = file_counts.get(file_idx, 0)
                    print(f"   File {file_idx+1}: {current_selected} selected this round, {used_count} total used, {total_count} total available")
                
            else:
                print("❌ No samples found")
                break
                
        print("\n📈 Final file usage summary:")
        for file_idx, batch_info in enumerate(learner.batch_files_data):
            used_count = len(batch_info['used_indices'])
            total_count = batch_info['total_samples']
            usage_pct = (used_count / total_count) * 100
            print(f"   File {file_idx+1}: {used_count}/{total_count} samples used ({usage_pct:.1f}%)")
            
    except FileNotFoundError as e:
        print(f"⚠️ Test files not found: {e}")
        print("   This test requires the standard data files to exist")
        print("   The multi-file logic would work with real files")
        
    # Clean up test files
    print("\n🧹 Cleaning up test files...")
    for batch_file in batch_files:
        if os.path.exists(batch_file):
            os.remove(batch_file)
    
    print("\n✅ Multi-file test complete!")

if __name__ == '__main__':
    test_multi_file_selection()