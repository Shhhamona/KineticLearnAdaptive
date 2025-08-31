"""
Baseline Training Methods - Safe methods for baseline MSE calculations

This module contains all the baseline training methods extracted from baseline_test.py
to ensure they don't get modified accidentally and can be reused in other scripts.
"""

import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle


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
    """
    Generate data subsets for different training sizes.
    
    Args:
        x_data: Input training data
        y_data: Output training data  
        subset_sizes: List of subset sizes to create
        
    Returns:
        List of (x_subset, y_subset) tuples
    """
    subsets = []
    for size in subset_sizes:
        x_subset = x_data[:size]
        y_subset = y_data[:size]
        subsets.append((x_subset, y_subset))
    return subsets


def calculate_mse_for_dataset(dataset_train, dataset_test, best_params, subset_sizes, seed=40):
    """
    Calculate MSE for different subset sizes using SVR models.
    
    Args:
        dataset_train: Training dataset object with get_data() method
        dataset_test: Test dataset object with get_data() method
        best_params: List of SVR parameters for each output
        subset_sizes: List of training subset sizes to test
        seed: Random seed for reproducibility
        
    Returns:
        mse_list: MSE values per output and subset size
        total_mse_list: Sum of MSE across outputs for each subset size
    """
    x_train, y_train = dataset_train.get_data()
    x_test, y_test = dataset_test.get_data()

    x_train, y_train = shuffle(x_train, y_train, random_state=seed)  # set a random_state for reproducibility

    data_subsets = generate_subsets(x_train, y_train, subset_sizes)

    mse_list = []

    for i in range(y_train.shape[1]):
        mse_output = []
        for (x_subset, y_subset) in data_subsets:
            model = SVR(C=best_params[i]['C'], epsilon=best_params[i]['epsilon'], 
                        gamma=best_params[i]['gamma'], kernel=best_params[i]['kernel'])

            model.fit(x_subset, y_subset[:,i])

            y_pred = model.predict(x_test)

            mse = mean_squared_error(y_test[:, i], y_pred)

            mse_output.append(mse)
        
        mse_list.append(mse_output)

    total_mse_list = np.sum(np.array(mse_list), axis=0)

    return mse_list, total_mse_list


def run_mse_analysis(dataset_train, dataset_test, best_params, subset_sizes, num_seeds):
    """
    Run MSE analysis for multiple seeds and return mean and std error.
    
    Args:
        dataset_train: Training dataset object
        dataset_test: Test dataset object
        best_params: List of SVR parameters for each output
        subset_sizes: List of training subset sizes to test
        num_seeds: Number of random seeds to run
        
    Returns:
        mean_total_mse: Mean total MSE across seeds
        std_total_mse: Standard error of total MSE across seeds
    """
    total_mse_for_seeds = []
    
    for seed in range(num_seeds):
        print(f"\n🎲 Running seed {seed+1}/{num_seeds}...")
        mse_list, total_mse_list = calculate_mse_for_dataset(dataset_train, dataset_test, best_params, subset_sizes, seed=seed)
        total_mse_for_seeds.append(total_mse_list)
        
        # Print detailed results for debugging
        print(f"   📈 Total MSE for each subset size: {total_mse_list}")
        for i, size in enumerate(subset_sizes):
            individual_mses = [mse_list[j][i] for j in range(len(best_params))]
            print(f"   📊 Size {size}: Individual MSEs = {individual_mses}, Sum = {total_mse_list[i]:.6f}")
    
    mean_total_mse = np.mean(total_mse_for_seeds, axis=0)
    std_total_mse = np.std(total_mse_for_seeds, axis=0) / np.sqrt(num_seeds)
    
    return mean_total_mse, std_total_mse


def load_baseline_datasets(src_file_train, src_file_test, nspecies, num_pressure_conditions, react_idx=[0, 1, 2]):
    """
    Load training and test datasets with proper scaling.
    
    Args:
        src_file_train: Path to training data file
        src_file_test: Path to test data file
        nspecies: Number of species
        num_pressure_conditions: Number of pressure conditions
        react_idx: Reaction indices to use
        
    Returns:
        dataset_train: Training dataset object
        dataset_test: Test dataset object
    """
    dataset_train = LoadMultiPressureDatasetNumpy(src_file_train, nspecies, num_pressure_conditions, react_idx=react_idx)
    dataset_test = LoadMultiPressureDatasetNumpy(src_file_test, nspecies, num_pressure_conditions, react_idx=react_idx, 
                                                scaler_input=dataset_train.scaler_input, scaler_output=dataset_train.scaler_output)
    return dataset_train, dataset_test
