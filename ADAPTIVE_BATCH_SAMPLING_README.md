# Adaptive Batch Sampling Pipeline - Summary

## Files Created

### 1. `kinetic_modelling/pipeline/adaptive_batch_sampling.py`
The main pipeline class for adaptive batch sampling with Neural Networks.

**Key Features:**
- **Continuous NN Training**: Trains a single Neural Network across all iterations (doesn't reset)
- **Mini-batch Gradient Descent**: Uses `train_single_batch()` method for iterative training
- **Window-based Sampling**: Samples from pool datasets using shrinking windows
- **Test-based Center**: Center point calculated from test dataset (stays constant)
- **Multiple Seeds**: Robust evaluation with different initializations

**Key Differences from `adaptive_sampling.py`:**
- Uses `NeuralNetModel` instead of `SVRModel`
- Continuous training (same model instance across iterations)
- Trains with epochs and batches instead of simple `fit()`
- Always samples from pool datasets (ignores `initial_dataset`)
- Center point from test dataset, not training data

### 2. `pipeline_runs/run_adaptive_batch_sampling.py`
Example script to run the adaptive batch sampling pipeline.

**Configuration:**
- `n_iterations`: Number of iterations (one per pool file)
- `samples_per_iteration`: Samples to grab from each pool (e.g., 200)
- `n_epochs`: Epochs to train at each iteration (e.g., 10)
- `batch_size`: Batch size for NN training (e.g., 64)
- `initial_window_size`: Starting window size (e.g., 1.0 = ±100%)
- `shrink_rate`: Window reduction factor (e.g., 0.5 = 50% reduction)
- `num_seeds`: Number of random seeds for robustness

**Neural Network Parameters:**
```python
nn_params = {
    'input_size': auto-detected,
    'output_size': auto-detected,
    'hidden_sizes': (64, 32),  # Configurable
    'activation': 'tanh',
    'learning_rate': 0.001
}
```

### 3. `example_adaptive_batch_sampling.py`
Minimal working example with dummy data for testing.

**Two Functions:**
1. `minimal_working_example()`: Quick test with random data
2. `example_adaptive_batch_sampling()`: Template for real data

## How It Works

### Workflow (Per Seed)

1. **Initialize**: Create a new Neural Network model
2. **Calculate Center**: Get center point from test dataset (constant)
3. **Iteration Loop**:
   - Define shrinking window around center
   - Sample from next pool dataset within window
   - Train NN for `n_epochs` with new samples (continuous training)
   - Evaluate on test set
   - Track metrics

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `pool_datasets` | List of pool datasets to sample from | `[pool1, pool2, pool3]` |
| `n_iterations` | Number of iterations | `5` |
| `samples_per_iteration` | Samples per pool | `200` |
| `n_epochs` | Training epochs per iteration | `10` |
| `batch_size` | Mini-batch size | `64` |
| `initial_window_size` | Starting window | `1.0` (±100%) |
| `shrink_rate` | Window shrinkage | `0.5` (50% reduction) |

### Output Results

**Aggregated Results Include:**
- `iteration`: Iteration number
- `total_samples_seen`: Cumulative samples trained on
- `samples_added`: New samples this iteration
- `mean_total_mse`: Average test MSE
- `std_total_mse`: Standard error of MSE
- `mean_train_loss`: Average training loss
- `window_size`: Current window size

## Usage

### Basic Usage
```bash
python pipeline_runs/run_adaptive_batch_sampling.py
```

### Minimal Test
```bash
python example_adaptive_batch_sampling.py
```

### Customize Parameters
Edit `pipeline_runs/run_adaptive_batch_sampling.py`:
- Change `n_epochs` for more/less training per iteration
- Adjust `samples_per_iteration` for different sample sizes
- Modify `nn_params` for different architectures
- Update `BATCH_FILES` for your data files

## Comparison: SVM vs NN Approach

| Aspect | Adaptive Sampling (SVM) | Adaptive Batch Sampling (NN) |
|--------|------------------------|------------------------------|
| **Model** | SVRModel | NeuralNetModel |
| **Training** | `model.fit()` (one-shot) | `model.train_single_batch()` (iterative) |
| **Model Persistence** | New model each iteration | Same model across iterations |
| **Initial Data** | Uses `initial_dataset` | Ignores `initial_dataset` |
| **Center Point** | From current training data | From test dataset (constant) |
| **Training Method** | Direct fit to all data | Mini-batch gradient descent |
| **Epochs** | N/A | Configurable per iteration |

## Results Files

Results saved to: `pipeline_results/adaptive_batch_sampling_*.json`

**Contains:**
- Configuration parameters
- Raw results per seed
- Aggregated results (mean ± std)
- Execution time
- Timestamp

## Next Steps

1. **Test with Dummy Data**: Run `example_adaptive_batch_sampling.py`
2. **Configure for Real Data**: Edit `run_adaptive_batch_sampling.py`
3. **Adjust Hyperparameters**: Tune NN architecture, learning rate, etc.
4. **Visualize Results**: Create plotting scripts similar to batch_training plots
