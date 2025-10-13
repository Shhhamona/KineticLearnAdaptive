#!/usr/bin/env python3
"""
Analysis of Morris Sampling: Finding close points in output space     # Calculate input parameter differences
    input_diffs = []
    output_diffs = []

    for pair in close_pairs[:30]:  # Analyze top 30 pairsomparing input parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import pdist, squareform

def load_morris_data():
    """Load both Morris sampling data files."""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'data')

    continuous_file = os.path.join(base_path, 'O2_simple_uniform_morris_continous.txt')
    discrete_file = os.path.join(base_path, 'O2_simple_morris.txt')

    try:
        continuous_data = np.loadtxt(continuous_file)
        discrete_data = np.loadtxt(discrete_file)

        print(f"📁 Data Loading Results:")
        print(f"  Continuous Morris: {continuous_data.shape} samples")
        print(f"  Discrete Morris: {discrete_data.shape} samples")

        return continuous_data, discrete_data

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def find_close_output_points(data, method_name, n_closest=20, distance_threshold=None):
    """
    Find points that are close in output space and analyze their input parameters.
    """
    print(f"\n🔍 Analyzing Close Points in Output Space: {method_name}")
    print("=" * 70)

    # Split into inputs (first 3 columns) and outputs (last 3 columns)
    n_params = 3
    inputs = data[:, :n_params]  # k1, k2, k3
    outputs = data[:, -n_params:]  # y1, y2, y3

    print(f"Input parameters shape: {inputs.shape}")
    print(f"Output values shape: {outputs.shape}")

    # Normalize outputs for distance calculation
    outputs_normalized = (outputs - np.mean(outputs, axis=0)) / np.std(outputs, axis=0)

    # Calculate pairwise distances in output space
    output_distances = squareform(pdist(outputs_normalized, metric='euclidean'))

    # Find closest pairs
    n_samples = len(data)
    close_pairs = []

    for i in range(n_samples):
        # Get distances to all other points
        distances = output_distances[i, :]
        # Sort by distance (excluding self)
        sorted_indices = np.argsort(distances)[1:n_closest+1]  # Skip self (distance 0)
        sorted_distances = distances[sorted_indices]

        for j, dist in zip(sorted_indices, sorted_distances):
            if distance_threshold is None or dist < distance_threshold:
                close_pairs.append({
                    'point1_idx': i,
                    'point2_idx': j,
                    'output_distance': dist,
                    'inputs1': inputs[i],
                    'inputs2': inputs[j],
                    'outputs1': outputs[i],
                    'outputs2': outputs[j]
                })

    # Sort by output distance
    close_pairs.sort(key=lambda x: x['output_distance'])

    print(f"Found {len(close_pairs)} close pairs (top {n_closest} per point)")

    # Analyze the closest pairs
    analyze_close_pairs(close_pairs[:50], method_name)  # Top 50 closest pairs

    return close_pairs

def analyze_close_pairs(close_pairs, method_name):
    """Analyze the input parameter differences for close output pairs."""

    if not close_pairs:
        print("No close pairs found!")
        return

    print(f"\n📊 Analysis of Closest Output Pairs ({method_name})")
    print("-" * 70)

    # Calculate input parameter differences
    input_diffs = []
    output_diffs = []

    for pair in close_pairs[:10]:  # Analyze top 10
        input_diff = np.abs(pair['inputs1'] - pair['inputs2'])
        output_diff = np.abs(pair['outputs1'] - pair['outputs2'])

        input_diffs.append(input_diff)
        output_diffs.append(output_diff)

        print(f"\nPair {close_pairs.index(pair)+1}: Output distance = {pair['output_distance']:.4f}")
        print(f"  Input 1: {pair['inputs1']}")
        print(f"  Input 2: {pair['inputs2']}")
        print(f"  Input diff: {input_diff}")
        print(f"  Max input diff: {np.max(input_diff):.2e}")
        print(f"  Output 1: {pair['outputs1']}")
        print(f"  Output 2: {pair['outputs2']}")
        print(f"  Output diff: {output_diff}")

    # Statistical analysis
    input_diffs = np.array(input_diffs)
    output_diffs = np.array(output_diffs)

    print(f"\n📈 Statistical Summary (Top 30 pairs):")
    print(f"Input parameter differences:")
    for i in range(3):
        mean_diff = np.mean(input_diffs[:, i])
        max_diff = np.max(input_diffs[:, i])
        min_diff = np.min(input_diffs[:, i])
        print(f"  k{i+1}: mean={mean_diff:.2e}, max={max_diff:.2e}, min={min_diff:.2e}")

    print(f"Output value differences:")
    for i in range(3):
        mean_diff = np.mean(output_diffs[:, i])
        max_diff = np.max(output_diffs[:, i])
        min_diff = np.min(output_diffs[:, i])
        print(f"  y{i+1}: mean={mean_diff:.2e}, max={max_diff:.2e}, min={min_diff:.2e}")

    # Check for clustering patterns
    check_clustering_patterns(close_pairs, method_name)

def check_clustering_patterns(close_pairs, method_name):
    """Check if close output points form clusters in input space."""

    print(f"\n🔍 Clustering Analysis ({method_name})")
    print("-" * 50)

    # Extract input coordinates of close pairs
    input_coords = []
    for pair in close_pairs[:50]:  # Top 50 pairs
        input_coords.extend([pair['inputs1'], pair['inputs2']])

    input_coords = np.array(input_coords)

    # Calculate spread in input space for close output points
    for i in range(3):
        param_values = input_coords[:, i]
        spread = np.max(param_values) - np.min(param_values)
        std = np.std(param_values)
        print(f"  k{i+1} spread in close pairs: {spread:.2e} (std: {std:.2e})")

    # Check if close outputs come from similar input regions
    input_ranges = []
    for i in range(3):
        param_values = input_coords[:, i]
        input_ranges.append(np.max(param_values) - np.min(param_values))

    avg_input_range = np.mean(input_ranges)
    print(f"  Average input parameter range in close pairs: {avg_input_range:.2e}")

    # Compare to overall data range
    all_inputs = []
    for pair in close_pairs[:20]:
        all_inputs.extend([pair['inputs1'], pair['inputs2']])
    all_inputs = np.array(all_inputs)

    # Calculate coefficient of variation for each parameter
    cv_params = []
    for i in range(3):
        param_values = all_inputs[:, i]
        cv = np.std(param_values) / np.mean(param_values) if np.mean(param_values) != 0 else 0
        cv_params.append(cv)
        print(f"  k{i+1} coefficient of variation: {cv:.3f}")

    avg_cv = np.mean(cv_params)
    print(f"  Average CV across parameters: {avg_cv:.3f}")

    if avg_cv < 0.1:
        print("  ✅ Close outputs come from very similar input regions (tight clusters)")
    elif avg_cv < 0.5:
        print("  ⚠️ Close outputs come from moderately similar input regions")
    else:
        print("  ❌ Close outputs come from diverse input regions (no clear clustering)")

def compare_methods_close_points():
    """Compare close point analysis between continuous and discrete Morris."""

    print("🔬 COMPARATIVE ANALYSIS: Close Points in Output Space")
    print("=" * 80)

    continuous_data, discrete_data = load_morris_data()

    if continuous_data is None or discrete_data is None:
        return

    # Analyze both methods
    cont_pairs = find_close_output_points(continuous_data, "Continuous Morris")
    disc_pairs = find_close_output_points(discrete_data, "Discrete Morris")

    # Compare the closest pairs
    if cont_pairs and disc_pairs:
        print(f"\n📊 Method Comparison")
        print("-" * 50)

        cont_closest = cont_pairs[0]['output_distance']
        disc_closest = disc_pairs[0]['output_distance']

        print(f"Closest output distance - Continuous: {cont_closest:.4f}")
        print(f"Closest output distance - Discrete: {disc_closest:.4f}")

        # Compare input parameter spreads for closest pairs
        cont_input_spread = np.max(cont_pairs[0]['inputs1']) - np.min(cont_pairs[0]['inputs1'])
        disc_input_spread = np.max(disc_pairs[0]['inputs1']) - np.min(disc_pairs[0]['inputs1'])

        print(f"Input spread in closest pair - Continuous: {cont_input_spread:.2e}")
        print(f"Input spread in closest pair - Discrete: {disc_input_spread:.2e}")

        if cont_closest < disc_closest:
            print("✅ Continuous Morris has closer output points")
        else:
            print("✅ Discrete Morris has closer output points")

def visualize_close_points():
    """Create visualizations of close points in output space."""

    continuous_data, discrete_data = load_morris_data()

    if continuous_data is None or discrete_data is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Close Points Analysis: Output Space Clustering', fontsize=14)

    methods = [("Continuous Morris", continuous_data), ("Discrete Morris", discrete_data)]
    colors = ['blue', 'red']

    for method_idx, (method_name, data) in enumerate(methods):
        inputs = data[:, :3]
        outputs = data[:, -3:]

        # Normalize outputs
        outputs_norm = (outputs - np.mean(outputs, axis=0)) / np.std(outputs, axis=0)

        # Plot output space (first 2 output dimensions)
        ax = axes[method_idx, 0]
        scatter = ax.scatter(outputs_norm[:, 0], outputs_norm[:, 1],
                           c=outputs_norm[:, 2], cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel('Normalized y1')
        ax.set_ylabel('Normalized y2')
        ax.set_title(f'{method_name} - Output Space')
        plt.colorbar(scatter, ax=ax, label='Normalized y3')

        # Plot input space for close output points
        ax = axes[method_idx, 1]

        # Find close points (top 50 closest pairs)
        output_distances = squareform(pdist(outputs_norm, metric='euclidean'))
        n_samples = len(data)
        close_indices = set()

        for i in range(min(50, n_samples)):  # Check first 50 points
            distances = output_distances[i, :]
            closest_idx = np.argsort(distances)[1]  # Skip self
            close_indices.add(i)
            close_indices.add(closest_idx)

        close_indices = list(close_indices)
        close_inputs = inputs[close_indices]
        close_outputs = outputs_norm[close_indices]

        # Color by output similarity
        distances = pdist(close_outputs, metric='euclidean')
        max_dist = np.max(distances)
        colors_close = plt.cm.coolwarm(distances / max_dist) if len(distances) > 0 else 'blue'

        scatter2 = ax.scatter(close_inputs[:, 0], close_inputs[:, 1],
                            c=close_inputs[:, 2], cmap='plasma', alpha=0.8, s=30)
        ax.set_xlabel('k1')
        ax.set_ylabel('k2')
        ax.set_title(f'{method_name} - Close Points in Input Space')
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
        plt.colorbar(scatter2, ax=ax, label='k3')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_methods_close_points()
    visualize_close_points()
