# Adaptive Learning Experiment Setups

This directory contains JSON configuration files for running adaptive learning experiments with different parameters.

## Directory Structure

```
adaptive_learning_setups/
└── 800_iteration_040_shrink/
    ├── seed_42.json
    ├── seed_43.json
    ├── seed_44.json
    ├── seed_45.json
    └── seed_46.json
```

## Configuration File Format

Each JSON file contains:

```json
{
  "experiment_name": "unique_experiment_identifier",
  "description": "Human-readable description",
  "parameters": {
    "n_samples_per_iteration": 800,
    "k_mult_factor": 1.4,
    "k_center": [6.27e-16, 1.40e-15, 9.79e-16],
    "loki_version": "v2"
  },
  "predicted_k_info": {
    "seed": 42,
    "seed_index": 0,
    "true_k_values": [...],
    "predicted_k_values": [...],
    "absolute_error": [...],
    "relative_error_percent": [...],
    "mse_original": 1.098800e+28
  },
  "notes": "Additional context about this experiment"
}
```

## Usage

### Run all experiments in a directory

```powershell
# Run all experiments in the 800_iteration_040_shrink directory
.\run_parallel_active_learning_from_json.ps1 -ConfigDir "adaptive_learning_setups\800_iteration_040_shrink"
```

### Dry run (preview commands without executing)

```powershell
# See what commands would be run without actually executing them
.\run_parallel_active_learning_from_json.ps1 -ConfigDir "adaptive_learning_setups\800_iteration_040_shrink" -DryRun
```

### Use default configuration directory

```powershell
# If no ConfigDir specified, defaults to "adaptive_learning_setups\800_iteration_040_shrink"
.\run_parallel_active_learning_from_json.ps1
```

## Creating New Experiment Setups

1. Create a new subdirectory under `adaptive_learning_setups/`
2. Add JSON configuration files for each experiment
3. Run the script with the new directory:

```powershell
.\run_parallel_active_learning_from_json.ps1 -ConfigDir "adaptive_learning_setups\your_new_setup"
```

## Current Setup: 800_iteration_040_shrink

This setup contains 5 experiments using predicted K values from neural network training:

- **seed_42.json**: K center from seed 42, lowest MSE (1.098800e+28)
- **seed_43.json**: K center from seed 43, MSE: 1.367789e+28
- **seed_44.json**: K center from seed 44, MSE: 1.634545e+28
- **seed_45.json**: K center from seed 45, MSE: 1.842078e+28
- **seed_46.json**: K center from seed 46, highest MSE (2.147605e+28)

All experiments use:
- 800 samples per iteration
- K multiplicative factor: 1.4
- LoKI versions: v2-v6

The K center values are the predicted K values from the neural network model trained with different seeds on 800 samples with 40% shrink rate.

## Logs

Experiment logs are saved to `results\logs\` with filenames matching the experiment names.

## Parameters Explained

- **n_samples_per_iteration**: Number of samples to generate in each adaptive learning iteration
- **k_mult_factor**: Multiplicative factor for K bounds (e.g., 1.4 means bounds are [K/1.4, K×1.4])
- **k_center**: The center K values around which to generate samples [K1, K2, K3]
- **loki_version**: Which LoKI installation to use (v2, v3, v4, v5, v6)
