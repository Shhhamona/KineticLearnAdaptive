"""
Active learning helper methods - STANDALONE VERSION

These methods are intentionally very similar to the baseline methods but completely independent.
Only the training data will change once adaptive sampling is added.

Functions:
- LoadMultiPressureDatasetNumpy: Standalone data loading class
- generate_subsets: Create data subsets for different training sizes
- calculate_mse_for_dataset: Calculate MSE for different subset sizes using SVR models
- run_mse_analysis: Run MSE analysis for multiple seeds
- load_datasets: Load training and test datasets with proper scaling
- train_initial_models: Train SVR models on initial samples
- evaluate_initial_learning_curve: Build subset sizes and run MSE analysis
"""

import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle


def apply_training_scalers(raw_compositions, raw_k_values, dataset_train, nspecies, num_pressure_conditions, debug=True):
    """
    Apply the exact same scaling logic as LoadMultiPressureDatasetNumpy.__init__ to new simulation data.
    
    Args:
        raw_compositions: Array of shape (n_sims * num_pressure_conditions, nspecies) - raw densities
        raw_k_values: Array of shape (n_sims, n_k) - raw k values (NOT multiplied by 1e30 yet)
        dataset_train: Training dataset object with scaler_input and scaler_output
        nspecies: Number of species
        num_pressure_conditions: Number of pressure conditions
        debug: Whether to print debug information
        
    Returns:
        new_x: Scaled and flattened input features (n_sims, num_pressure_conditions * nspecies)
        new_y_scaled: Scaled output targets (n_sims, n_k)
    """
    if debug:
        print(f"\n🔧 APPLYING TRAINING SCALERS")
        print(f"   raw_compositions shape: {raw_compositions.shape}")
        print(f"   raw_k_values shape: {raw_k_values.shape}")
        print(f"   nspecies: {nspecies}, num_pressure_conditions: {num_pressure_conditions}")
    
    # Step 1: Apply the 1e30 multiplier to k values (exactly like LoadMultiPressureDatasetNumpy)
    # Create a copy to ensure we don't modify the caller's array
    k_data = raw_k_values * 1e30  # This already creates a new array
    if debug:
        print(f"   k_data after *1e30: min={np.min(k_data):.3e}, max={np.max(k_data):.3e}")
    
    # Step 2: Reshape compositions and k values - DEFENSIVE COPIES
    # Make explicit copies to avoid mutating the caller's arrays in-place. 
    # reshape() can produce views into the original arrays, and subsequent
    # assignments would overwrite the original arrays. Copying prevents that and
    # makes repeated calls to this function idempotent.
    n_sims = raw_k_values.shape[0]
    x_data = raw_compositions.reshape(num_pressure_conditions, n_sims, nspecies).copy()
    y_data = k_data.reshape(1, n_sims, k_data.shape[1]).copy()  # Only one "pressure" for k values
    
    if debug:
        print(f"   x_data reshaped: {x_data.shape}")
        print(f"   y_data reshaped: {y_data.shape}")
        for i in range(num_pressure_conditions):
            print(f"   x_data[{i}] min={np.min(x_data[i]):.3e}, max={np.max(x_data[i]):.3e}")
    
    # Step 3: Apply per-pressure input scalers (exactly like LoadMultiPressureDatasetNumpy)
    for i in range(num_pressure_conditions):
        if debug:
            print(f"   Applying scaler_input[{i}] to x_data[{i}]")
            print(f"     Before: min={np.min(x_data[i]):.3e}, max={np.max(x_data[i]):.3e}")
        
        x_data[i] = dataset_train.scaler_input[i].transform(x_data[i])
        
        if debug:
            print(f"     After: min={np.min(x_data[i]):.3e}, max={np.max(x_data[i]):.3e}")
    
    # Step 4: Apply output scaler (use first pressure scaler like LoadMultiPressureDatasetNumpy)
    if debug:
        print(f"   Applying scaler_output[0] to y_data")
        print(f"     Before: min={np.min(y_data):.3e}, max={np.max(y_data):.3e}")
    
    y_data_scaled = dataset_train.scaler_output[0].transform(y_data[0])
    
    if debug:
        print(f"     After: min={np.min(y_data_scaled):.3e}, max={np.max(y_data_scaled):.3e}")
    
    # Step 5: Flatten x_data exactly like LoadMultiPressureDatasetNumpy
    # Transpose to move pressure axis to end, then flatten
    # Use .copy() to ensure we return independent arrays
    x_data_transposed = np.transpose(x_data, (1, 0, 2))  # (n_sims, num_pressure_conditions, nspecies)
    new_x = x_data_transposed.reshape(n_sims, num_pressure_conditions * nspecies).copy()
    
    if debug:
        print(f"   Final new_x shape: {new_x.shape}")
        print(f"   Final new_y_scaled shape: {y_data_scaled.shape}")
        print(f"   new_x range: min={np.min(new_x):.3e}, max={np.max(new_x):.3e}")
        print(f"   new_y_scaled range: min={np.min(y_data_scaled):.3e}, max={np.max(y_data_scaled):.3e}")
    
    return new_x, y_data_scaled


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


def calculate_mse_for_dataset(dataset_train, dataset_test, best_params, subset_sizes, seed=42):
    """
    Calculate MSE for different subset sizes using SVR models.
    
    Args:
        dataset_train: Training dataset object with get_data() method
        dataset_test: Test dataset object with get_data() method
        best_params: List of SVR parameters for each output
        subset_sizes: List of training subset sizes to test
        seed: Random seed for reproducibility (now defaults to 42 to match retrain_models_with_new_data)
        
    Returns:
        mse_list: MSE values per output and subset size
        total_mse_list: Sum of MSE across outputs for each subset size
    """
    x_train, y_train = dataset_train.get_data()
    x_test, y_test = dataset_test.get_data()

    # DEBUG: Print test data consistency check
    print(f"   calculate_mse_for_dataset: Test data checksum y_test.mean()={y_test.mean():.6f}, x_test.shape={x_test.shape}")
    print(f"   calculate_mse_for_dataset: Using seed={seed} for shuffling")

    # CRITICAL FIX: Uncomment shuffling to match other functions
    x_train, y_train = shuffle(x_train, y_train, random_state=seed)  # set a random_state for reproducibility

    data_subsets = generate_subsets(x_train, y_train, subset_sizes)

    # DEBUG: Print first few samples to verify shuffling and subset selection
    print(f"   DEBUG: First 3 x_train samples after shuffle:\n{x_train[:3]}")
    print(f"   DEBUG: First 3 y_train samples after shuffle:\n{y_train[:3]}")

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

    # DEBUG: Print dimensions and structure
    mse_array = np.array(mse_list)
    print(f"   DEBUG: mse_list shape = {mse_array.shape}")  # Should be (n_outputs, n_subset_sizes)
    print(f"   DEBUG: subset_sizes = {subset_sizes}")
    print(f"   DEBUG: mse_array = \n{mse_array}")

    total_mse_list = np.sum(mse_array, axis=0)
    print(f"   DEBUG: total_mse_list = {total_mse_list}")

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
        # Use base seed 42 + seed offset to maintain consistency with other functions
        actual_seed = 42 + seed
        mse_list, total_mse_list = calculate_mse_for_dataset(dataset_train, dataset_test, best_params, subset_sizes, seed=actual_seed)
        total_mse_for_seeds.append(total_mse_list)
        
        # Print detailed results for debugging
        print(f"   📈 Total MSE for each subset size: {total_mse_list}")
        for i, size in enumerate(subset_sizes):
            individual_mses = [mse_list[j][i] for j in range(len(best_params))]
            print(f"   📊 Size {size}: Individual MSEs = {individual_mses}, Sum = {total_mse_list[i]:.6f}")
    
    mean_total_mse = np.mean(total_mse_for_seeds, axis=0)
    std_total_mse = np.std(total_mse_for_seeds, axis=0) / np.sqrt(num_seeds)
    
    return mean_total_mse, std_total_mse


def retrain_models_with_new_data(current_x_train, current_y_train, dataset_test, new_x, new_y_scaled, best_params, seed=42, debug=True):
    """
    Retrain SVR models by appending new samples to the current training data.
    
    Args:
        current_x_train: Current training input data (n_current, n_features)
        current_y_train: Current training output data (n_current, n_outputs)
        dataset_test: Test dataset object  
        new_x: New input samples (n_new, n_features)
        new_y_scaled: New output samples (n_new, n_outputs)
        best_params: List of SVR parameters for each output
        seed: Random seed for shuffling
        debug: Whether to print debug information
        
    Returns:
        new_models: List of retrained SVR models
        new_mse_per_output: MSE per output on test set
        new_total_mse: Total MSE on test set
        augmented_dataset_size: Size of training set after adding new samples
        x_train_augmented: Augmented training input data
        y_train_augmented: Augmented training output data
    """
    if debug:
        print(f"\n🔄 RETRAINING MODELS WITH NEW DATA")
        print(f"   Using seed={seed} for shuffling (same as other functions)")
    
    # Get test data
    x_test, y_test = dataset_test.get_data()
    
    if debug:
        print(f"   Test data shape: x_test={x_test.shape}, y_test={y_test.shape}")
        print(f"   Test data checksum: y_test.mean()={y_test.mean():.6f} (for consistency check)")
        print(f"   Test data hash: {hash(x_test.tobytes())}, {hash(y_test.tobytes())}")  # Exact data verification
    
    # Append new samples to current training data
    x_train_new = np.vstack([current_x_train, new_x])
    y_train_new = np.vstack([current_y_train, new_y_scaled])

    #x_train_new = new_x
    #y_train_new = new_y_scaled

    if debug:
        print(f"   Current training size: {current_x_train.shape[0]}")
        print(f"   New samples: {new_x.shape[0]}")
        print(f"   Augmented training size: {x_train_new.shape[0]}")
        print(f"   Test size: {x_test.shape[0]}")
    
    # Shuffle the augmented training set
    x_train_shuffled, y_train_shuffled = shuffle(x_train_new, y_train_new, random_state=seed)
    x_train_shuffled, y_train_shuffled = x_train_new, y_train_new

    
    if debug:
        print(f"   Training data shuffled with seed={seed}")
        print(f"   Final training shapes: X={x_train_shuffled.shape}, Y={y_train_shuffled.shape}")
    
    # DEBUG: Print actual training and test data details before training
    print(f"DEBUG (retrain_models_with_new_data): TRAINING DATA")
    print(f"   X_train_shuffled first 3 rows:\n{x_train_shuffled[:3]}")
    print(f"   Y_train_shuffled first 3 rows:\n{y_train_shuffled[:3]}")
    print(f"   X_train_shuffled range: [{x_train_shuffled.min():.6f}, {x_train_shuffled.max():.6f}]")
    print(f"   Y_train_shuffled range: [{y_train_shuffled.min():.6f}, {y_train_shuffled.max():.6f}]")
    print(f"   X_train_shuffled.mean()={x_train_shuffled.mean():.8f}, Y_train_shuffled.mean()={y_train_shuffled.mean():.8f}")
    
    print(f"DEBUG (retrain_models_with_new_data): TEST DATA")
    print(f"   X_test first 3 rows:\n{x_test[:3]}")
    print(f"   Y_test first 3 rows:\n{y_test[:3]}")
    print(f"   X_test range: [{x_test.min():.6f}, {x_test.max():.6f}]")
    print(f"   Y_test range: [{y_test.min():.6f}, {y_test.max():.6f}]")
    print(f"   X_test.mean()={x_test.mean():.8f}, Y_test.mean()={y_test.mean():.8f}")
    
    # Compare the data composition (uniform vs batch)
    print(f"DEBUG (retrain_models_with_new_data): DATA COMPOSITION")
    print(f"   Original current_x_train.mean()={current_x_train.mean():.8f}, current_y_train.mean()={current_y_train.mean():.8f}")
    print(f"   Original new_x.mean()={new_x.mean():.8f}, new_y_scaled.mean()={new_y_scaled.mean():.8f}")
    
    # Save training snapshots if debug is enabled
    if debug:
        import os
        import hashlib
        
        # Create hash from training data for unique filename
        combined_data = np.concatenate([x_train_shuffled.flatten(), y_train_shuffled.flatten()])
        data_hash = hashlib.md5(combined_data.tobytes()).hexdigest()[:8]
        
        # Create snapshots directory if it doesn't exist
        snapshot_dir = 'results/training_snapshots'
        os.makedirs(snapshot_dir, exist_ok=True)
        
        # Save the exact training arrays that will enter model.fit()
        x_train_file = os.path.join(snapshot_dir, f'retrain_x_train_shuffled_{data_hash}.txt')
        y_train_file = os.path.join(snapshot_dir, f'retrain_y_train_shuffled_{data_hash}.txt')
        
        np.savetxt(x_train_file, x_train_shuffled, delimiter='  ', fmt='%.8e')
        np.savetxt(y_train_file, y_train_shuffled, delimiter='  ', fmt='%.8e')
        
        print(f"   💾 Training snapshots saved:")
        print(f"      X_train: {x_train_file}")
        print(f"      Y_train: {y_train_file}")
        print(f"      Data hash: {data_hash}")
        print(f"      Size: {x_train_shuffled.shape[0]} samples")
    
    # Retrain each output model
    new_models = []
    new_mse_per_output = []
    train_mse_per_output = []
    
    for i in range(y_train_shuffled.shape[1]):
        if debug:
            print(f"   Training model for output {i}...")
        
        params = best_params[i]
        model = SVR(C=params['C'], epsilon=params['epsilon'], gamma=params['gamma'], kernel=params['kernel'])
        model.fit(x_train_shuffled, y_train_shuffled[:, i])
        
        # Evaluate on test set
        y_pred_test = model.predict(x_test)
        test_mse = mean_squared_error(y_test[:, i], y_pred_test)
        
        # Evaluate on training set
        y_pred_train = model.predict(x_train_shuffled)
        train_mse = mean_squared_error(y_train_shuffled[:, i], y_pred_train)
        
        new_models.append(model)
        new_mse_per_output.append(test_mse)
        train_mse_per_output.append(train_mse)
        
        if debug:
            print(f"     Output {i}: Test MSE={test_mse:.6f}, Train MSE={train_mse:.6f}")
    
    new_total_mse = float(np.sum(new_mse_per_output))
    train_total_mse = float(np.sum(train_mse_per_output))
    
    if debug:
        print(f"   ✅ Retraining complete")
        print(f"   Test MSE per output: {new_mse_per_output}")
        print(f"   Train MSE per output: {train_mse_per_output}")
        print(f"   Total Test MSE: {new_total_mse:.6f}")
        print(f"   Total Train MSE: {train_total_mse:.6f}")
        print(f"   Overfitting ratio (Test/Train): {new_total_mse/train_total_mse:.3f}")
    
    return new_models, new_mse_per_output, new_total_mse, x_train_new.shape[0], x_train_shuffled, y_train_shuffled


def load_datasets(src_file_train, src_file_test, nspecies, num_pressure_conditions, react_idx=[0, 1, 2]):
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


def train_initial_models(dataset_train, dataset_test, best_params, n_initial_samples=500, seed=42):
    """
    Train SVR models on the first n_initial_samples taken from dataset_train (shuffled with seed).
    Returns trained models, mse_per_output list and total_mse (sum).

    This mirrors the baseline training step exactly.
    """
    x_all, y_all = dataset_train.get_data()
    x_test, y_test = dataset_test.get_data()

    # shuffle and take the first n_initial_samples (same behavior as baseline)
    x_shuf, y_shuf = shuffle(x_all, y_all, random_state=seed)
    x_train = x_shuf[:n_initial_samples]
    y_train = y_shuf[:n_initial_samples]

    print(f"DEBUG (active): x_train.shape={x_train.shape}, y_train.shape={y_train.shape}")
    print(f"DEBUG (active): x_test.shape={x_test.shape}, y_test.shape={y_test.shape}")
    print(f"DEBUG (active): Test data hash: {hash(x_test.tobytes())}, {hash(y_test.tobytes())}")  # Exact data verification
    
    # DEBUG: Print actual training and test data details
    print(f"DEBUG (train_initial_models): TRAINING DATA")
    print(f"   X_train first 3 rows:\n{x_train[:3]}")
    print(f"   Y_train first 3 rows:\n{y_train[:3]}")
    print(f"   X_train range: [{x_train.min():.6f}, {x_train.max():.6f}]")
    print(f"   Y_train range: [{y_train.min():.6f}, {y_train.max():.6f}]")
    print(f"   X_train.mean()={x_train.mean():.8f}, Y_train.mean()={y_train.mean():.8f}")
    
    print(f"DEBUG (train_initial_models): TEST DATA")
    print(f"   X_test first 3 rows:\n{x_test[:3]}")
    print(f"   Y_test first 3 rows:\n{y_test[:3]}")
    print(f"   X_test range: [{x_test.min():.6f}, {x_test.max():.6f}]")
    print(f"   Y_test range: [{y_test.min():.6f}, {y_test.max():.6f}]")
    print(f"   X_test.mean()={x_test.mean():.8f}, Y_test.mean()={y_test.mean():.8f}")

    models = []
    mse_per_output = []
    train_mse_per_output = []

    for i in range(y_train.shape[1]):
        params = best_params[i]
        model = SVR(C=params['C'], epsilon=params['epsilon'], gamma=params['gamma'], kernel=params['kernel'])
        model.fit(x_train, y_train[:, i])
        
        # Test MSE
        y_pred_test = model.predict(x_test)
        test_mse = mean_squared_error(y_test[:, i], y_pred_test)
        
        # Training MSE
        y_pred_train = model.predict(x_train)
        train_mse = mean_squared_error(y_train[:, i], y_pred_train)
        
        models.append(model)
        mse_per_output.append(test_mse)
        train_mse_per_output.append(train_mse)
        print(f"DEBUG (active): Output {i}: Test MSE={test_mse:.6f}, Train MSE={train_mse:.6f}")

    total_mse = float(np.sum(mse_per_output))
    total_train_mse = float(np.sum(train_mse_per_output))
    print(f"Test MSE per output: {mse_per_output}")
    print(f"Train MSE per output: {train_mse_per_output}")
    print(f"Total Test MSE (sum): {total_mse:.6f}")
    print(f"Total Train MSE (sum): {total_train_mse:.6f}")
    print(f"Overfitting ratio (Test/Train): {total_mse/total_train_mse:.3f}")

    return models, mse_per_output, total_mse


def _build_subset_sizes(initial_samples, total_samples_needed, step=100):
    """
    Build subset sizes list from step up to initial_samples (inclusive) by step,
    and append total_samples_needed if not present.
    """
    base_sizes = list(range(step, initial_samples + 1, step))
    subset_sizes = base_sizes.copy()
    if total_samples_needed not in subset_sizes:
        subset_sizes.append(total_samples_needed)
    return subset_sizes


def evaluate_initial_learning_curve(dataset_train, dataset_test, best_params, initial_samples, total_samples_needed, num_seeds=1, seed=42):
    """
    Build subset sizes (100,200,...,initial_samples and final total_samples_needed) and
    run the same MSE analysis used by the baseline.

    Returns mean_total_mse, std_total_mse, and subset_sizes used.
    """
    subset_sizes = _build_subset_sizes(initial_samples, total_samples_needed, step=100)
    print(f"DEBUG (active): Subset sizes for evaluation: {subset_sizes}")

    # Use the same run_mse_analysis helper to keep behavior identical
    mean_total_mse, std_total_mse = run_mse_analysis(dataset_train, dataset_test, best_params, subset_sizes, num_seeds)

    return mean_total_mse, std_total_mse, subset_sizes


def evaluate_zones_scaled(current_x_train, current_y_train, new_x, new_y_scaled, models, n_zones=6):
    """
    Split augmented training data into zones in scaled-Y space and evaluate MSE per zone.

    Args:
        current_x_train: array (n_current, n_features)
        current_y_train: array (n_current, n_outputs) - scaled Y
        new_x: array (n_new, n_features)
        new_y_scaled: array (n_new, n_outputs) - scaled Y
        models: list of trained models (one per output) that accept X and predict scaled outputs
        n_zones: int number of zones to create

    Returns:
        zone_indices: array of shape (n_augmented,) with zone id 0..n_zones-1
        center_y_scaled: array (n_outputs,) center used for distance calculations
        thresholds: array (n_zones+1,) percentile thresholds (distance bins)
        per_output_mse_list: list length n_zones of arrays (n_outputs,) per-output MSEs
        overall_mse_list: list length n_zones of overall MSE (mean across outputs)
        counts: list length n_zones with sample counts
    """
    # Build augmented arrays
    X_aug = np.vstack([current_x_train, new_x])
    y_scaled_aug = np.vstack([current_y_train, new_y_scaled])

    # Center in scaled-Y space
    center_y_scaled = np.mean(y_scaled_aug, axis=0)

    # Distances in scaled space
    distances = np.linalg.norm(y_scaled_aug - center_y_scaled.reshape(1, -1), axis=1)

    # Percentile-based thresholds for equal-count zones
    percentiles = np.linspace(0, 100, n_zones + 1)
    thresholds = np.percentile(distances, percentiles)
    bins = thresholds[1:-1]
    zone_indices = np.digitize(distances, bins)

    per_output_mse_list = []
    overall_mse_list = []
    counts = []

    # Evaluate each zone
    for zi in range(n_zones):
        mask = (zone_indices == zi)
        count = int(np.sum(mask))
        counts.append(count)
        if count == 0:
            per_output_mse_list.append(None)
            overall_mse_list.append(None)
            continue

        X_zone = X_aug[mask]
        y_true_scaled_zone = y_scaled_aug[mask]

        # Predict using provided models
        preds_scaled_cols = []
        for m in models:
            preds_scaled_cols.append(m.predict(X_zone))
        preds_scaled = np.vstack(preds_scaled_cols).T

        per_output_mse = mean_squared_error(y_true_scaled_zone, preds_scaled, multioutput='raw_values')
        overall_mse = float(np.mean(per_output_mse))

        per_output_mse_list.append(per_output_mse)
        overall_mse_list.append(overall_mse)

    return zone_indices, center_y_scaled, thresholds, per_output_mse_list, overall_mse_list, counts

