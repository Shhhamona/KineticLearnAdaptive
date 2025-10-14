import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

folder_path = "D:\\Marcelo\\github\\Dissertation\\Images\\"

class LoadMultiPressureDatasetNumpy:

    def __init__(self, src_file, nspecies, num_pressure_conditions, react_idx=None, m_rows=None, columns=None,
                 scaler_input=None, scaler_output=None):
        self.num_pressure_conditions = num_pressure_conditions

        all_data = np.loadtxt(src_file, max_rows=m_rows,
                              usecols=columns, delimiter="  ",
                              comments="#", skiprows=0, dtype=np.float64)

        ncolumns = len(all_data[0])
        x_columns = np.arange(ncolumns - nspecies, ncolumns, 1)
        y_columns = react_idx
        if react_idx is None:
            y_columns = np.arange(0, ncolumns - nspecies, 1)

        x_data = all_data[:, x_columns]  # densities
        y_data = all_data[:, y_columns] * 1e30  # k's  # *10 to avoid being at float32 precision limit 1e-17

        # Reshape data for multiple pressure conditions
        x_data = x_data.reshape(num_pressure_conditions, -1, x_data.shape[1])
        y_data = y_data.reshape(num_pressure_conditions, -1, y_data.shape[1])

        # Create scalers
        self.scaler_input = scaler_input or [preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)]
        self.scaler_output = scaler_output or [preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)]
        
        for i in range(num_pressure_conditions):
            if scaler_input is None:
                self.scaler_input[i].fit(x_data[i])
            if scaler_output is None:
                self.scaler_output[i].fit(y_data[i])
            x_data[i] = self.scaler_input[i].transform(x_data[i])
            y_data[i] = self.scaler_output[i].transform(y_data[i])

        # Transpose x_data to move the pressure condition axis to the end, then flatten
        x_data = np.transpose(x_data, (1, 0, 2)).reshape(-1, self.num_pressure_conditions * x_data.shape[-1])
        
        # Flatten the output data to be of shape (2000,3)
        y_data = y_data[0]

        # Assign the preprocessed data
        self.x_data = x_data
        self.y_data = y_data
        self.all_data = all_data
        
        # Store original K values for zone analysis
        self.y_data_original = all_data[:, y_columns] * 1e30
        self.y_data_original = self.y_data_original.reshape(num_pressure_conditions, -1, self.y_data_original.shape[1])[0]

    def get_data(self):
        """
        Return the preprocessed input and output data.
        """
        return self.x_data, self.y_data
    
    def get_original_k_data(self):
        """
        Return the original K values for zone analysis.
        """
        return self.y_data_original


def calculate_distances_from_true(k_data, k_true):
    """Calculate Euclidean distances from true K values in log space"""
    epsilon = 1e-50
    log_k_data = np.log10(np.abs(k_data) + epsilon)
    log_k_true = np.log10(np.abs(k_true) + epsilon)
    
    distances = np.linalg.norm(log_k_data - log_k_true, axis=1)
    return distances


def create_k_zones_adaptive(k_data, k_true, num_zones=5):
    """Create K value zones based on actual data distribution"""
    # Calculate distances for all data points
    distances = calculate_distances_from_true(k_data, k_true)
    
    # Create zones based on percentiles of actual distances
    percentiles = np.linspace(0, 100, num_zones + 1)
    zone_boundaries = np.percentile(distances, percentiles)
    
    zones = []
    zone_labels = []
    
    for i in range(num_zones):
        zone_info = {
            'inner_radius': zone_boundaries[i],
            'outer_radius': zone_boundaries[i + 1],
            'label': f'Zone {i+1} (d: {zone_boundaries[i]:.1f}-{zone_boundaries[i+1]:.1f})'
        }
        zones.append(zone_info)
        zone_labels.append(zone_info['label'])
    
    return zones, zone_labels


def classify_points_into_zones(k_data, k_true, zones):
    """Classify data points into K value zones"""
    distances = calculate_distances_from_true(k_data, k_true)
    
    # Classify into zones
    zone_assignments = np.full(len(k_data), -1, dtype=int)  # -1 for unassigned
    
    for i, zone in enumerate(zones):
        mask = (distances >= zone['inner_radius']) & (distances < zone['outer_radius'])
        zone_assignments[mask] = i
    
    return zone_assignments, distances


def generate_subsets(x_data, y_data, subset_sizes):
    subsets = []
    for size in subset_sizes:
        x_subset = x_data[:size]
        y_subset = y_data[:size]
        subsets.append((x_subset, y_subset))
    return subsets


def calculate_mse_for_dataset_with_zones(dataset_train, dataset_test, best_params, subset_sizes, zones, k_true, seed=40):
    """Calculate MSE for dataset including zone-based analysis"""
    x_train, y_train = dataset_train.get_data()
    x_test, y_test = dataset_test.get_data()
    test_k_original = dataset_test.get_original_k_data()

    x_train, y_train = shuffle(x_train, y_train, random_state=seed)
    data_subsets = generate_subsets(x_train, y_train, subset_sizes)

    # Classify test points into zones
    zone_assignments, test_distances = classify_points_into_zones(test_k_original, k_true, zones)
    
    # Count points in each zone for contribution calculations
    zone_point_counts = []
    total_test_points = len(test_k_original)
    for zone_idx in range(len(zones)):
        zone_mask = zone_assignments == zone_idx
        zone_count = np.sum(zone_mask)
        zone_point_counts.append(zone_count)

    mse_list = []
    zone_mse_list = []  # MSE for each zone
    zone_contributions_list = []  # Contribution of each zone to total MSE

    for i in range(y_train.shape[1]):
        mse_output = []
        zone_mse_output = []
        zone_contrib_output = []
        
        for (x_subset, y_subset) in data_subsets:
            model = SVR(C=best_params[i]['C'], epsilon=best_params[i]['epsilon'], 
                        gamma=best_params[i]['gamma'], kernel=best_params[i]['kernel'])

            model.fit(x_subset, y_subset[:,i])
            y_pred = model.predict(x_test)

            # Global MSE
            mse = mean_squared_error(y_test[:, i], y_pred)
            mse_output.append(mse)
            
            # Zone-based MSE and contributions
            zone_mse_subset = []
            zone_contrib_subset = []
            
            for zone_idx in range(len(zones)):
                zone_mask = zone_assignments == zone_idx
                if np.sum(zone_mask) > 0:
                    zone_true = y_test[zone_mask, i]
                    zone_pred = y_pred[zone_mask]
                    zone_mse = mean_squared_error(zone_true, zone_pred)
                    
                    # Calculate zone contribution to total MSE
                    # Contribution = zone_MSE * (zone_points / total_points)
                    zone_contribution = zone_mse * (zone_point_counts[zone_idx] / total_test_points)
                    
                    zone_mse_subset.append(zone_mse)
                    zone_contrib_subset.append(zone_contribution)
                else:
                    zone_mse_subset.append(0.0)  # No points in this zone
                    zone_contrib_subset.append(0.0)
            
            zone_mse_output.append(zone_mse_subset)
            zone_contrib_output.append(zone_contrib_subset)
        
        mse_list.append(mse_output)
        zone_mse_output = np.array(zone_mse_output)  # Shape: (subset_sizes, num_zones)
        zone_contrib_output = np.array(zone_contrib_output)  # Shape: (subset_sizes, num_zones)
        zone_mse_list.append(zone_mse_output)
        zone_contributions_list.append(zone_contrib_output)

    # Calculate total MSE (sum across K coefficients)
    total_mse_list = np.sum(np.array(mse_list), axis=0)
    
    # Calculate total zone MSE (sum across K coefficients)
    zone_mse_array = np.array(zone_mse_list)  # Shape: (3, subset_sizes, num_zones)
    total_zone_mse_list = np.sum(zone_mse_array, axis=0)  # Shape: (subset_sizes, num_zones)
    
    # Calculate total zone contributions (sum across K coefficients)
    zone_contrib_array = np.array(zone_contributions_list)  # Shape: (3, subset_sizes, num_zones)
    total_zone_contributions_list = np.sum(zone_contrib_array, axis=0)  # Shape: (subset_sizes, num_zones)

    return mse_list, total_mse_list, zone_mse_list, total_zone_mse_list, total_zone_contributions_list, zone_point_counts


def save_results_to_csv(results_dict, zones, filename):
    """Save results to CSV for further analysis"""
    # Flatten the results dictionary for CSV export
    rows = []
    
    for strategy, data in results_dict.items():
        for seed_idx in range(len(data['global_mse_seeds'])):
            for size_idx, subset_size in enumerate(data['subset_sizes']):
                row = {
                    'strategy': strategy,
                    'seed': seed_idx,
                    'subset_size': subset_size,
                    'global_mse': data['global_mse_seeds'][seed_idx][size_idx]
                }
                
                # Add zone MSEs
                for zone_idx in range(len(zones)):
                    row[f'zone_{zone_idx+1}_mse'] = data['zone_mse_seeds'][seed_idx][size_idx][zone_idx]
                
                # Add zone contributions
                for zone_idx in range(len(zones)):
                    row[f'zone_{zone_idx+1}_contribution'] = data['zone_contributions_seeds'][seed_idx][size_idx][zone_idx]
                
                # Add zone contribution percentages
                total_contrib = sum(data['zone_contributions_seeds'][seed_idx][size_idx])
                if total_contrib > 0:
                    for zone_idx in range(len(zones)):
                        contrib_pct = (data['zone_contributions_seeds'][seed_idx][size_idx][zone_idx] / total_contrib) * 100
                        row[f'zone_{zone_idx+1}_contribution_pct'] = contrib_pct
                else:
                    for zone_idx in range(len(zones)):
                        row[f'zone_{zone_idx+1}_contribution_pct'] = 0.0
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"📊 Results saved to {filename}")


if __name__ == "__main__":
    nspecies = 3
    num_pressure_conditions = 2
    subset_sizes = [i for i in range(200, 2100, 200)]
    num_seeds = 1  # Number of seeds to use
    
    # Create output directory
    output_dir = 'results/sample_efficiency_zones'
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting Sample Efficiency with Zone Analysis")
    print("=" * 60)
    print(f"📂 Output directory: {output_dir}")
    print(f"📊 Subset sizes: {subset_sizes}")
    print(f"🎲 Number of seeds: {num_seeds}")
    print(f"⏱️  Expected runtime: ~{num_seeds * len(subset_sizes) * 5 * 3 / 60:.1f} minutes")
    
    # True K values from O2_simple_1.chem file
    k_true_original = np.array([6.00E-16, 1.30E-15, 9.60E-16])
    
    best_params = [
        {'C': 10, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'},
        {'C': 20, 'epsilon': 0.005, 'gamma': 5, 'kernel': 'rbf'},
        {'C': 5, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'}
    ]

    datasets = [
        'O2_simple_log.txt',
        'O2_simple_morris.txt',
        'O2_simple_uniform.txt',
        'O2_simple_latin_log_uniform.txt',
        'O2_simple_latin.txt',
    ]
    labels = [
        'Log-Uniform',
        'Morris Method',
        'Uniform',
        'Log-Uniform Latin Hypercube',
        'Uniform Latin Hypercube',
    ]

    # Load test dataset to create zones
    print("\n📂 Loading test dataset to create zones...")
    src_file_test = 'data/SampleEfficiency/O2_simple_test.txt'
    dataset_test_temp = LoadMultiPressureDatasetNumpy(src_file_test, nspecies, num_pressure_conditions, react_idx=[0, 1, 2])
    test_k_data = dataset_test_temp.get_original_k_data()
    
    # Create adaptive zones based on test data distribution
    zones, zone_labels = create_k_zones_adaptive(test_k_data, k_true_original, num_zones=5)
    
    print("🎯 Created adaptive K value zones:")
    for i, zone in enumerate(zones):
        print(f"   {zone['label']}")

    # Dictionary to store all results
    all_results = {}

    # Set up plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Sample Efficiency Analysis: Global MSE and Zone-Based MSE', fontsize=16, fontweight='bold')
    
    # Global MSE plot
    ax_global = axes[0, 0]
    ax_global.set_title('Global MSE vs Dataset Size', fontweight='bold')
    ax_global.set_xlabel('Dataset Size')
    ax_global.set_ylabel('Total MSE')
    ax_global.grid(True, alpha=0.3)
    
    # Zone MSE plots
    zone_axes = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
    for i, ax in enumerate(zone_axes):
        if i < len(zones):
            ax.set_title(f'Zone {i+1} MSE vs Dataset Size', fontweight='bold')
            ax.set_xlabel('Dataset Size')
            ax.set_ylabel(f'Zone {i+1} MSE')
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')  # Hide unused subplots

    colors = plt.cm.Set1(np.linspace(0, 1, len(datasets)))
    
    print(f"\n🔬 Processing {len(datasets)} sampling strategies...")
    
    # Process each dataset
    for idx, dataset in enumerate(datasets):
        strategy_name = labels[idx]
        print(f"\n📊 Processing {strategy_name} ({idx+1}/{len(datasets)})...")
        
        src_file_train = 'data/SampleEfficiency/' + dataset
        src_file_test = 'data/SampleEfficiency/O2_simple_test.txt'

        # Load datasets
        dataset_train = LoadMultiPressureDatasetNumpy(src_file_train, nspecies, num_pressure_conditions, react_idx=[0, 1, 2])
        dataset_test = LoadMultiPressureDatasetNumpy(src_file_test, nspecies, num_pressure_conditions, react_idx=[0, 1, 2], 
                                                    scaler_input=dataset_train.scaler_input, scaler_output=dataset_train.scaler_output)

        # Store results for this strategy
        global_mse_seeds = []
        zone_mse_seeds = []
        zone_contributions_seeds = []
        
        start_time = time.time()
        
        for seed in range(num_seeds):
            print(f"   🎲 Seed {seed+1}/{num_seeds}... ", end="", flush=True)
            
            # Calculate MSE with zones (updated function call)
            _, total_mse_list, _, total_zone_mse_list, total_zone_contributions_list, zone_point_counts = calculate_mse_for_dataset_with_zones(
                dataset_train, dataset_test, best_params, subset_sizes, zones, k_true_original, seed=seed)
            
            global_mse_seeds.append(total_mse_list)
            zone_mse_seeds.append(total_zone_mse_list)
            zone_contributions_seeds.append(total_zone_contributions_list)
            
            elapsed = time.time() - start_time
            remaining = (elapsed / (seed + 1)) * (num_seeds - seed - 1)
            print(f"✅ (ETA: {remaining/60:.1f}m)")
        
        # Calculate statistics
        mean_global_mse = np.mean(global_mse_seeds, axis=0)
        std_global_mse = np.std(global_mse_seeds, axis=0) / np.sqrt(num_seeds)
        
        mean_zone_mse = np.mean(zone_mse_seeds, axis=0)  # Shape: (subset_sizes, num_zones)
        std_zone_mse = np.std(zone_mse_seeds, axis=0) / np.sqrt(num_seeds)
        
        mean_zone_contributions = np.mean(zone_contributions_seeds, axis=0)  # Shape: (subset_sizes, num_zones)
        std_zone_contributions = np.std(zone_contributions_seeds, axis=0) / np.sqrt(num_seeds)
        
        # Store results
        all_results[strategy_name] = {
            'subset_sizes': subset_sizes,
            'global_mse_seeds': global_mse_seeds,
            'zone_mse_seeds': zone_mse_seeds,
            'zone_contributions_seeds': zone_contributions_seeds,
            'mean_global_mse': mean_global_mse,
            'std_global_mse': std_global_mse,
            'mean_zone_mse': mean_zone_mse,
            'std_zone_mse': std_zone_mse,
            'mean_zone_contributions': mean_zone_contributions,
            'std_zone_contributions': std_zone_contributions,
            'zone_point_counts': zone_point_counts
        }
        
        # Plot global MSE
        ax_global.errorbar(subset_sizes, mean_global_mse, yerr=std_global_mse, 
                          label=strategy_name, marker='o', color=colors[idx], linewidth=2)
        
        # Plot zone MSEs
        for zone_idx in range(len(zones)):
            if zone_idx < len(zone_axes):
                zone_mean = mean_zone_mse[:, zone_idx]
                zone_std = std_zone_mse[:, zone_idx]
                zone_axes[zone_idx].errorbar(subset_sizes, zone_mean, yerr=zone_std,
                                           label=strategy_name, marker='o', color=colors[idx], linewidth=2)
        
        elapsed_total = time.time() - start_time
        print(f"   ✅ {strategy_name} completed in {elapsed_total/60:.1f} minutes")
        
        # Print summary statistics
        print(f"      📈 Global MSE range: [{min(mean_global_mse):.2e}, {max(mean_global_mse):.2e}]")
        for zone_idx in range(len(zones)):
            zone_mean = mean_zone_mse[:, zone_idx]
            print(f"      🎯 Zone {zone_idx+1} MSE range: [{min(zone_mean):.2e}, {max(zone_mean):.2e}]")

    # Finalize plots
    print(f"\n📈 Creating final plots...")
    
    ax_global.legend(fontsize=10)
    ax_global.set_yscale('log')
    
    # Calculate common Y-axis limits for zone plots
    all_zone_mins = []
    all_zone_maxs = []
    
    for strategy_name, data in all_results.items():
        zone_data = data['mean_zone_mse']
        all_zone_mins.append(np.min(zone_data[zone_data > 0]))  # Exclude zeros
        all_zone_maxs.append(np.max(zone_data))
    
    common_ymin = min(all_zone_mins) * 0.5  # Add some margin
    common_ymax = max(all_zone_maxs) * 2.0  # Add some margin
    
    print(f"   🎯 Setting common Y-axis limits: [{common_ymin:.2e}, {common_ymax:.2e}]")
    
    for zone_idx in range(len(zones)):
        if zone_idx < len(zone_axes):
            zone_axes[zone_idx].legend(fontsize=8)
            zone_axes[zone_idx].set_yscale('log')
            zone_axes[zone_idx].set_ylim(common_ymin, common_ymax)

    plt.tight_layout()
    
    # Save plots
    output_file = os.path.join(output_dir, 'sample_efficiency_with_zones.pdf')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Plots saved as: {output_file}")
    
    # Save results to CSV
    csv_file = os.path.join(output_dir, 'sample_efficiency_with_zones_results.csv')
    save_results_to_csv(all_results, zones, csv_file)
    
    # Print final summary
    print(f"\n🏆 FINAL RESULTS SUMMARY:")
    print("=" * 60)
    
    # Zone point distribution (only need to show once)
    if zone_point_counts:
        total_points = sum(zone_point_counts)
        print(f"\n📊 TEST DATA DISTRIBUTION ACROSS ZONES:")
        for zone_idx in range(len(zones)):
            count = zone_point_counts[zone_idx]
            percentage = (count / total_points) * 100
            print(f"   Zone {zone_idx+1}: {count:4d} points ({percentage:4.1f}%)")
    
    for strategy_name, data in all_results.items():
        mean_global = np.mean(data['mean_global_mse'])
        print(f"\n📊 {strategy_name}:")
        print(f"   🎯 Average Global MSE: {mean_global:.2e}")
        
        # Calculate average zone contributions
        mean_contributions = np.mean(data['mean_zone_contributions'], axis=0)
        total_contribution = np.sum(mean_contributions)
        
        print(f"   📈 Zone MSE and Contributions:")
        for zone_idx in range(len(zones)):
            mean_zone_mse = np.mean(data['mean_zone_mse'][:, zone_idx])
            zone_contribution = mean_contributions[zone_idx]
            contribution_pct = (zone_contribution / total_contribution * 100) if total_contribution > 0 else 0
            
            print(f"      Zone {zone_idx+1}: MSE={mean_zone_mse:.2e}, Contrib={zone_contribution:.2e} ({contribution_pct:.1f}%)")
    
    # Zone contribution analysis across all strategies
    print(f"\n🎯 ZONE CONTRIBUTION ANALYSIS (Across All Strategies):")
    print("=" * 60)
    
    all_contributions = np.zeros(len(zones))
    strategy_count = 0
    
    for strategy_name, data in all_results.items():
        mean_contributions = np.mean(data['mean_zone_contributions'], axis=0)
        all_contributions += mean_contributions
        strategy_count += 1
    
    avg_contributions = all_contributions / strategy_count
    total_avg_contribution = np.sum(avg_contributions)
    
    print(f"📊 Average Zone Contributions to Total MSE:")
    zone_contrib_ranking = []
    for zone_idx in range(len(zones)):
        contribution_pct = (avg_contributions[zone_idx] / total_avg_contribution * 100) if total_avg_contribution > 0 else 0
        zone_contrib_ranking.append((zone_idx + 1, contribution_pct, avg_contributions[zone_idx]))
        print(f"   Zone {zone_idx+1}: {avg_contributions[zone_idx]:.2e} ({contribution_pct:.1f}%)")
    
    # Sort by contribution percentage
    zone_contrib_ranking.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 ZONE CONTRIBUTION RANKING (Highest to Lowest):")
    for rank, (zone_id, contrib_pct, contrib_value) in enumerate(zone_contrib_ranking):
        print(f"   {rank+1}. Zone {zone_id}: {contrib_pct:.1f}% of total MSE")
    
    print(f"\n💡 ZONE INSIGHTS:")
    top_zone = zone_contrib_ranking[0]
    second_zone = zone_contrib_ranking[1]
    print(f"   🚨 Zone {top_zone[0]} contributes most ({top_zone[1]:.1f}%) to total error")
    print(f"   ⚠️  Zone {second_zone[0]} is second highest ({second_zone[1]:.1f}%) contributor")
    
    top_two_contribution = top_zone[1] + second_zone[1]
    print(f"   📊 Top 2 zones account for {top_two_contribution:.1f}% of total error")
    
    if top_two_contribution > 60:
        print(f"   💡 Focus adaptive sampling on Zones {top_zone[0]} and {second_zone[0]} for maximum impact!")
    
    # Find best strategies
    best_global_strategy = min(all_results.items(), key=lambda x: np.mean(x[1]['mean_global_mse']))
    print(f"\n🏆 Best Global Strategy: {best_global_strategy[0]} (MSE: {np.mean(best_global_strategy[1]['mean_global_mse']):.2e})")
    
    for zone_idx in range(len(zones)):
        best_zone_strategy = min(all_results.items(), key=lambda x: np.mean(x[1]['mean_zone_mse'][:, zone_idx]))
        print(f"🏆 Best Zone {zone_idx+1} Strategy: {best_zone_strategy[0]} (MSE: {np.mean(best_zone_strategy[1]['mean_zone_mse'][:, zone_idx]):.2e})")
    
    print(f"\n✅ Sample efficiency with zone analysis completed!")
    print(f"📊 Results saved to: {csv_file}")
    print(f"📈 Plots saved to: {output_file}")
    print(f"🎯 This analysis shows how different sampling strategies perform in different K-value regions!")
