import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
# import time
import os

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

    def get_data(self):
        """
        Return the preprocessed input and output data.
        """
        return self.x_data, self.y_data


def generate_subsets(x_data, y_data, subset_sizes):
    subsets = []
    for size in subset_sizes:
        x_subset = x_data[:size]
        y_subset = y_data[:size]
        subsets.append((x_subset, y_subset))
    return subsets


def calculate_gaussian_weights(k_test, k_true, sigma=1.0):
    """
    Calculate Gaussian weights based on distance from true K values.
    Uses log-space to handle the wide range of K values (1E-44 to 1E+01).
    
    Args:
        k_test: Test K values (N x num_reactions) 
        k_true: True K values (reference point) (num_reactions,)
        sigma: Standard deviation parameter controlling the width of the Gaussian
        
    Returns:
        weights: Gaussian weights for each test point (N,)
    """
    # Convert to log space to handle wide range of values
    # Add small epsilon to avoid log(0)
    epsilon = 1e-50
    log_k_test = np.log10(np.abs(k_test) + epsilon)
    log_k_true = np.log10(np.abs(k_true) + epsilon)
    
    # Calculate Euclidean distance in log space
    distances = np.linalg.norm(log_k_test - log_k_true, axis=1)
    
    # Calculate Gaussian weights
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    
    # Ensure we don't have zero weights (add small minimum)
    weights = np.maximum(weights, 1e-10)
    
    return weights


def gaussian_weighted_mse(y_true, y_pred, weights):
    """
    Calculate Gaussian-weighted Mean Squared Error.
    
    Args:
        y_true: True values (N,)
        y_pred: Predicted values (N,)
        weights: Gaussian weights for each point (N,)
        
    Returns:
        weighted_mse: Gaussian-weighted MSE
    """
    squared_errors = (y_true - y_pred) ** 2
    weighted_squared_errors = weights * squared_errors
    
    # Check for numerical issues
    sum_weights = np.sum(weights)
    if sum_weights == 0 or np.isnan(sum_weights) or np.isinf(sum_weights):
        print(f"Warning: Invalid weights detected. sum_weights={sum_weights}")
        print(f"weights min/max: {np.min(weights):.2e}/{np.max(weights):.2e}")
        # Fallback to regular MSE
        return np.mean(squared_errors)
    
    # Normalize by sum of weights to get proper average
    weighted_mse = np.sum(weighted_squared_errors) / sum_weights
    
    # Check result
    if np.isnan(weighted_mse) or np.isinf(weighted_mse):
        print(f"Warning: Invalid weighted_mse={weighted_mse}. Falling back to regular MSE.")
        return np.mean(squared_errors)
    
    return weighted_mse


def calculate_gaussian_weighted_mse_for_dataset(dataset_train, dataset_test, best_params, subset_sizes, k_true, sigma=0.5, seed=40):
    """
    Calculate Gaussian-weighted MSE for different dataset sizes.
    
    Args:
        dataset_train: Training dataset
        dataset_test: Test dataset  
        best_params: Best hyperparameters for each output
        subset_sizes: List of training set sizes to evaluate
        k_true: True K values (reference point for weighting)
        sigma: Gaussian width parameter
        seed: Random seed for reproducibility
        
    Returns:
        mse_list: List of weighted MSE for each output
        total_mse_list: Total weighted MSE across all outputs
    """
    x_train, y_train = dataset_train.get_data()
    x_test, y_test = dataset_test.get_data()

    x_train, y_train = shuffle(x_train, y_train, random_state=seed)
    data_subsets = generate_subsets(x_train, y_train, subset_sizes)

    print(f"      🔧 Processing {y_train.shape[1]} K coefficients...")
    mse_list = []

    for i in range(y_train.shape[1]):
        print(f"        📊 K coefficient {i+1}/{y_train.shape[1]}...")
        mse_output = []
        
        for j, (x_subset, y_subset) in enumerate(data_subsets):
            print(f"          📈 Subset size {subset_sizes[j]} ({j+1}/{len(data_subsets)})...", end=" ")
            
            # Train model
            model = SVR(C=best_params[i]['C'], epsilon=best_params[i]['epsilon'], 
                        gamma=best_params[i]['gamma'], kernel=best_params[i]['kernel'])
            model.fit(x_subset, y_subset[:,i])
            y_pred = model.predict(x_test)

            # Calculate Gaussian weights
            k_test_original = dataset_test.scaler_output[0].inverse_transform(y_test)
            k_true_single = k_true[i:i+1]
            k_test_single = k_test_original[:, i:i+1]
            weights = calculate_gaussian_weights(k_test_single, k_true_single, sigma)

            # Calculate Gaussian-weighted MSE
            weighted_mse = gaussian_weighted_mse(y_test[:, i], y_pred, weights)
            mse_output.append(weighted_mse)
            
            print(f"GWMSE: {weighted_mse:.2e}")
        
        mse_list.append(mse_output)

    total_mse_list = np.sum(np.array(mse_list), axis=0)
    print(f"      ✅ Completed all K coefficients. Total GWMSE range: [{min(total_mse_list):.2e}, {max(total_mse_list):.2e}]")

    return mse_list, total_mse_list


if __name__ == "__main__":
    nspecies = 3
    num_pressure_conditions = 2
    subset_sizes = [i for i in range(200, 2100, 200)]
    num_seeds = 3
    
    # Create output directory
    output_dir = 'results/sample_efficiency_gaussian'
    os.makedirs(output_dir, exist_ok=True)
    
    # True K values from O2_simple_1.chem file (first 3 reactions we're analyzing)
    # These are the reference values we want to achieve high accuracy around
    k_true = np.array([6.00E-16, 1.30E-15, 9.60E-16]) * 1e30  # Apply same scaling as dataset
    
    # Gaussian width parameter - controls how focused the weighting is
    sigma = 0.3  # Decreased to make weighting more focused around true K values
    
    print("🔬 Gaussian-Weighted Sample Efficiency Analysis")
    print("=" * 60)
    print(f"📂 Output directory: {output_dir}")
    print(f"📊 True K values (original): {[6.00E-16, 1.30E-15, 9.60E-16]}")
    print(f"📊 True K values (scaled): {k_true}")
    print(f"🎯 Gaussian sigma: {sigma}")
    print(f"📈 Subset sizes: {subset_sizes}")
    print(f"🔄 Number of seeds: {num_seeds}")
    print()
    
    best_params = [
        {'C': 10, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'},
        {'C': 20, 'epsilon': 0.005, 'gamma': 5, 'kernel': 'rbf'},
        {'C': 5, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'}
    ]

    # All sampling datasets from the original analysis
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

    print(f"🧪 Testing {len(datasets)} sampling strategies...")
    print(f"   📋 Datasets: {datasets}")
    print()

    # Create figure for plotting all results (like the original sample_effiency.py)
    plt.figure(figsize=(12, 8))
    
    # Process each dataset
    for idx, dataset_name in enumerate(datasets):
        print(f"📊 Processing dataset {idx+1}/{len(datasets)}: {labels[idx]}")
        print(f"   📁 File: {dataset_name}")
        
        src_file_train = 'data/SampleEfficiency/' + dataset_name
        src_file_test = 'data/SampleEfficiency/O2_simple_test.txt'

        # Load data
        print(f"   📥 Loading data...")
        dataset_train = LoadMultiPressureDatasetNumpy(src_file_train, nspecies, num_pressure_conditions, react_idx=[0, 1, 2])
        dataset_test = LoadMultiPressureDatasetNumpy(src_file_test, nspecies, num_pressure_conditions, react_idx=[0, 1, 2], 
                                                    scaler_input=dataset_train.scaler_input, scaler_output=dataset_train.scaler_output)
        print(f"   ✅ Data loaded. Train: {dataset_train.x_data.shape[0]}, Test: {dataset_test.x_data.shape[0]}")

        # Calculate Gaussian-weighted MSE for multiple seeds
        print(f"   🚀 Calculating GWMSE for {num_seeds} seeds...")
        total_gwmse_for_seeds = []
        
        for seed in range(num_seeds):
            print(f"     🔄 Seed {seed+1}/{num_seeds}...", end=" ")
            
            gwmse_list, total_gwmse_list = calculate_gaussian_weighted_mse_for_dataset(
                dataset_train, dataset_test, best_params, subset_sizes, k_true, sigma, seed=seed)
            
            total_gwmse_for_seeds.append(total_gwmse_list)
            print(f"GWMSE: [{min(total_gwmse_list):.2e}, {max(total_gwmse_list):.2e}]")
        
        # Calculate statistics for this dataset
        mean_total_gwmse = np.mean(total_gwmse_for_seeds, axis=0)
        std_total_gwmse = np.std(total_gwmse_for_seeds, axis=0) / np.sqrt(num_seeds)
        
        print(f"   ✅ {labels[idx]} completed! Final GWMSE: [{min(mean_total_gwmse):.2e}, {max(mean_total_gwmse):.2e}]")
        
        # Add to plot (like the original sample_effiency.py)
        plt.errorbar(subset_sizes, mean_total_gwmse, yerr=std_total_gwmse, 
                    label=labels[idx], marker='o', linewidth=2, markersize=6)
        print()

    # Finalize plot styling (like the original sample_effiency.py)
    plt.rcParams.update({'font.size': 14})
    plt.xlabel('Dataset size', fontsize=14)
    plt.ylabel('Gaussian-weighted MSE', fontsize=14)
    plt.title(f'Sample Efficiency Comparison: Gaussian-weighted MSE (σ={sigma})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, 'sample_efficiency_gaussian.pdf')
    plt.savefig(output_file)
    print(f"✅ Comparison plot saved as: {output_file}")
    
    print(f"\n✅ Gaussian-weighted sample efficiency analysis completed!")
    print(f"📊 Analyzed {len(datasets)} sampling strategies:")
    for i, label in enumerate(labels):
        print(f"   {i+1}. {label}")
    print(f"🎯 This metric measures accuracy near true K values, showing adaptive sampling effectiveness")
