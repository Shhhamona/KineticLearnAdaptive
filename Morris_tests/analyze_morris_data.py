"""
Analysis of Morris Sampling Data Files
Compares the continuous and discrete Morris sampling results with true center values.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def load_morris_data():
    """
    Load both Morris sampling data files.
    """
    base_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Load continuous Morris data
    continuous_file = os.path.join(base_path, 'O2_simple_uniform.txt')
    discrete_file = os.path.join(base_path, 'O2_simple_uniform_morris_continous.txt')
    
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

def analyze_morris_sampling(continuous_data, discrete_data, k_true):
    """
    Analyze the Morris sampling data quality.
    """
    print(f"\\n🔬 Morris Sampling Data Analysis")
    print(f"=" * 60)
    
    # True center values (first 3 parameters)
    print(f"True center values (k_true):")
    for i, val in enumerate(k_true):
        print(f"  k{i+1}: {val:.2e}")
    
    methods = [
        ("Continuous Morris", continuous_data),
        ("Discrete Morris", discrete_data)
    ]
    
    analysis_results = {}
    
    for method_name, data in methods:
        if data is None:
            continue
            
        print(f"\\n--- {method_name} Analysis ---")
        
        # Focus on first 3 parameters (reaction rate coefficients)
        k_params = data[:, :3]  # First 3 columns
        
        print(f"Shape: {data.shape}")
        print(f"Parameter ranges:")
        
        param_analysis = {}
        
        for i in range(3):
            param_values = k_params[:, i]
            
            # Statistical analysis
            mean_val = np.mean(param_values)
            median_val = np.median(param_values)
            std_val = np.std(param_values)
            min_val = np.min(param_values)
            max_val = np.max(param_values)
            
            # Compare with true value
            true_val = k_true[i]
            mean_error = abs(mean_val - true_val) / true_val * 100
            
            # Coverage analysis
            within_range = np.sum((param_values >= true_val * 0.1) & (param_values <= true_val * 10))
            coverage_percent = within_range / len(param_values) * 100
            
            param_analysis[f'k{i+1}'] = {
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'true': true_val,
                'mean_error': mean_error,
                'coverage': coverage_percent,
                'range_factor': max_val / min_val
            }
            
            print(f"  k{i+1}:")
            print(f"    Range: [{min_val:.2e}, {max_val:.2e}]")
            print(f"    Mean: {mean_val:.2e} (error: {mean_error:.1f}% from true)")
            print(f"    True: {true_val:.2e}")
            print(f"    Coverage (0.1×-10× true): {coverage_percent:.1f}%")
            print(f"    Range factor: {max_val/min_val:.1f}×")
        
        analysis_results[method_name] = param_analysis
    
    return analysis_results

def compare_sampling_methods(analysis_results):
    """
    Compare the two Morris sampling methods.
    """
    print(f"\\n📊 Method Comparison")
    print(f"=" * 60)
    
    if len(analysis_results) < 2:
        print(f"❌ Need both methods for comparison")
        return
    
    continuous_analysis = analysis_results.get("Continuous Morris")
    discrete_analysis = analysis_results.get("Discrete Morris")
    
    if not continuous_analysis or not discrete_analysis:
        print(f"❌ Missing analysis data")
        return
    
    print(f"{'Parameter':<12} {'Method':<18} {'Mean Error':<12} {'Coverage':<12} {'Range Factor':<15}")
    print(f"-" * 75)
    
    for param in ['k1', 'k2', 'k3']:
        cont_data = continuous_analysis[param]
        disc_data = discrete_analysis[param]
        
        print(f"{param:<12} {'Continuous':<18} {cont_data['mean_error']:>7.1f}%     {cont_data['coverage']:>7.1f}%     {cont_data['range_factor']:>10.1f}×")
        print(f"{'':12} {'Discrete':<18} {disc_data['mean_error']:>7.1f}%     {disc_data['coverage']:>7.1f}%     {disc_data['range_factor']:>10.1f}×")
        print(f"-" * 75)
    
    # Overall assessment
    cont_avg_error = np.mean([continuous_analysis[p]['mean_error'] for p in ['k1', 'k2', 'k3']])
    disc_avg_error = np.mean([discrete_analysis[p]['mean_error'] for p in ['k1', 'k2', 'k3']])
    
    cont_avg_coverage = np.mean([continuous_analysis[p]['coverage'] for p in ['k1', 'k2', 'k3']])
    disc_avg_coverage = np.mean([discrete_analysis[p]['coverage'] for p in ['k1', 'k2', 'k3']])
    
    print(f"\\n🎯 Overall Assessment:")
    print(f"  Continuous Morris: {cont_avg_error:.1f}% avg error, {cont_avg_coverage:.1f}% avg coverage")
    print(f"  Discrete Morris:   {disc_avg_error:.1f}% avg error, {disc_avg_coverage:.1f}% avg coverage")
    
    # Recommendation
    if cont_avg_error < disc_avg_error and cont_avg_coverage > disc_avg_coverage:
        print(f"  ✅ Continuous Morris performs better overall")
    elif disc_avg_error < cont_avg_error and disc_avg_coverage > cont_avg_coverage:
        print(f"  ✅ Discrete Morris performs better overall")
    else:
        print(f"  ⚖️ Both methods have trade-offs")

def visualize_parameter_distributions(continuous_data, discrete_data, k_true):
    """
    Create visualizations of parameter distributions.
    """
    print(f"\\n📈 Creating Parameter Distribution Plots")
    print(f"=" * 60)
    
    try:
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        methods = [
            ("Continuous Morris", continuous_data),
            ("Discrete Morris", discrete_data)
        ]
        
        colors = ['blue', 'red']
        
        for param_idx in range(3):
            for method_idx, (method_name, data) in enumerate(methods):
                if data is None:
                    continue
                    
                ax = axes[param_idx, method_idx]
                
                param_values = data[:, param_idx]
                true_val = k_true[param_idx]
                
                # Histogram
                ax.hist(param_values, bins=100, alpha=0.7, color=colors[method_idx], 
                       edgecolor='black', linewidth=0.5)
                
                # Mark true value
                ax.axvline(true_val, color='green', linewidth=3, 
                          label=f'True value: {true_val:.2e}')
                
                # Mark mean
                mean_val = np.mean(param_values)
                ax.axvline(mean_val, color='orange', linewidth=2, linestyle='--',
                          label=f'Mean: {mean_val:.2e}')
                
                ax.set_xlabel(f'k{param_idx+1} values')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{method_name} - Parameter k{param_idx+1}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Set log scale if values span many orders of magnitude
                if np.max(param_values) / np.min(param_values) > 100:
                    ax.set_xscale('log')
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(os.path.dirname(__file__), 'morris_data_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

def check_morris_trajectory_structure(data, method_name):
    """
    Check if the data follows Morris sampling trajectory structure.
    """
    print(f"\\n🔍 Trajectory Structure Analysis: {method_name}")
    print(f"-" * 50)
    
    n_samples = data.shape[0]
    n_params = data.shape[1]
    
    # Try to infer trajectory structure
    # Morris sampling should have r*(k+1) samples for r trajectories and k parameters
    # Most likely candidates for k (number of varied parameters)
    possible_k_values = []
    
    for k in range(2, min(10, n_params)):  # Try k from 2 to 9
        for r in range(2, 20):  # Try r from 2 to 19
            if r * (k + 1) == n_samples:
                possible_k_values.append((k, r))
    
    if possible_k_values:
        print(f"  Possible trajectory structures:")
        for k, r in possible_k_values:
            print(f"    {r} trajectories × {k+1} steps = {n_samples} samples (k={k} parameters)")
        
        # Use the most likely structure (smallest reasonable k)
        k, r = possible_k_values[0]
        print(f"  Most likely: {r} trajectories, {k} parameters varied")
        
        # Check one-at-a-time property for first trajectory
        first_traj = data[:k+1, :k]  # First trajectory, first k parameters
        
        violations = 0
        for step in range(1, k + 1):
            prev_step = first_traj[step - 1]
            curr_step = first_traj[step]
            
            # Count parameter changes
            changes = np.sum(~np.isclose(prev_step, curr_step, rtol=1e-10))
            if changes != 1:
                violations += 1
        
        ota_compliance = ((k - violations) / k * 100) if k > 0 else 100
        print(f"  One-at-a-time compliance: {ota_compliance:.1f}%")
        
        return k, r, ota_compliance
    else:
        print(f"  ❌ No valid Morris trajectory structure detected")
        print(f"  Total samples: {n_samples}")
        print(f"  This may not be proper Morris sampling data")
        return None, None, 0

def morris_data_quality_assessment():
    """
    Overall data quality assessment.
    """
    print(f"\\n🎯 Morris Data Quality Assessment")
    print(f"=" * 60)
    
    # True center values
    k_true = np.array([6.00e-16, 1.30e-15, 9.60e-16])
    
    # Load data
    continuous_data, discrete_data = load_morris_data()
    
    if continuous_data is None or discrete_data is None:
        print(f"❌ Could not load data files")
        return
    
    # Check trajectory structure
    print(f"\\n🔍 TRAJECTORY STRUCTURE ANALYSIS")
    cont_k, cont_r, cont_ota = check_morris_trajectory_structure(continuous_data, "Continuous Morris")
    disc_k, disc_r, disc_ota = check_morris_trajectory_structure(discrete_data, "Discrete Morris")
    
    # Analyze parameter distributions
    analysis_results = analyze_morris_sampling(continuous_data, discrete_data, k_true)
    
    # Compare methods
    compare_sampling_methods(analysis_results)
    
    # Create visualizations
    visualize_parameter_distributions(continuous_data, discrete_data, k_true)

    # Visualize output (y) distributions for both methods
    print(f"\n📈 Creating Output (y) Distribution Plots")
    y_start = continuous_data.shape[1] - 3
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Output (y) Distributions: Continuous (top) vs Discrete (bottom)', fontsize=14)
    for i in range(3):
        # Continuous Morris
        y_cont = continuous_data[:, y_start + i]
        axes[0, i].hist(y_cont, bins=150, alpha=0.7, edgecolor='black', color='blue')
        axes[0, i].set_title(f'Continuous y{i+1}')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        # Discrete Morris
        y_disc = discrete_data[:, y_start + i]
        axes[1, i].hist(y_disc, bins=150, alpha=0.7, edgecolor='black', color='red')
        axes[1, i].set_title(f'Discrete y{i+1}')
        axes[1, i].set_xlabel('Value')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    plt.tight_layout()
    plt.show()
    
    # Final assessment
    print(f"\\n🏆 FINAL ASSESSMENT")
    print(f"=" * 60)
    
    assessments = []
    
    # Structure assessment
    if cont_k and disc_k:
        print(f"✅ Both datasets have valid Morris trajectory structure")
        assessments.append("Structure: ✅")
    else:
        print(f"❌ Invalid Morris trajectory structure detected")
        assessments.append("Structure: ❌")
    
    # Coverage assessment
    if analysis_results:
        cont_avg_coverage = np.mean([analysis_results["Continuous Morris"][p]['coverage'] for p in ['k1', 'k2', 'k3']])
        disc_avg_coverage = np.mean([analysis_results["Discrete Morris"][p]['coverage'] for p in ['k1', 'k2', 'k3']])
        
        if cont_avg_coverage > 80 and disc_avg_coverage > 80:
            print(f"✅ Good parameter space coverage (>80%)")
            assessments.append("Coverage: ✅")
        elif cont_avg_coverage > 60 and disc_avg_coverage > 60:
            print(f"⚠️ Moderate parameter space coverage (>60%)")
            assessments.append("Coverage: ⚠️")
        else:
            print(f"❌ Poor parameter space coverage (<60%)")
            assessments.append("Coverage: ❌")
    
    # One-at-a-time assessment
    if cont_ota > 90 and disc_ota > 90:
        print(f"✅ Excellent one-at-a-time compliance (>90%)")
        assessments.append("OAT: ✅")
    elif cont_ota > 70 and disc_ota > 70:
        print(f"⚠️ Good one-at-a-time compliance (>70%)")
        assessments.append("OAT: ⚠️")
    else:
        print(f"❌ Poor one-at-a-time compliance (<70%)")
        assessments.append("OAT: ❌")
    
    # Overall verdict
    good_count = sum(1 for a in assessments if "✅" in a)
    moderate_count = sum(1 for a in assessments if "⚠️" in a)
    
    print(f"\\n📋 Summary: {', '.join(assessments)}")
    
    if good_count >= 2:
        print(f"🎉 Overall: GOOD Morris sampling data quality")
    elif good_count + moderate_count >= 2:
        print(f"👍 Overall: ACCEPTABLE Morris sampling data quality")
    else:
        print(f"👎 Overall: POOR Morris sampling data quality - consider regenerating")
    
    return analysis_results

if __name__ == "__main__":
    results = morris_data_quality_assessment()
