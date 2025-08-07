# Adaptive Sampling Analysis Report

**Generated:** 2025-07-27 17:27:55

## Configuration

- **Simulator:** mock
- **Chemistry File:** chem.chem
- **Setup File:** setup.in
- **True K Values:** [1e-15, 2e-15, 8e-16]
- **Initial Samples:** 20
- **Iteration Samples:** 10
- **Max Iterations:** 8
- **Convergence Threshold:** 0.001
- **Sampling Method:** latin_hypercube

## Results Summary

| Model | Initial Error | Final Error | Improvement (%) | Total Sims | Iterations | Converged | Time (s) | Efficiency |
|-------|---------------|-------------|-----------------|------------|------------|-----------|----------|------------|
| random_forest | 0.0871 | 0.1254 | -44.00 | 100 | 8 | No | 3.48 | 8.75e-03 |
| neural_network | 0.1305 | 0.1324 | -1.48 | 100 | 8 | No | 1.26 | 8.68e-03 |

## Best Performing Model

**random_forest** achieved the lowest final error of 0.1254
- Error improvement: -44.00%
- Total simulations: 100
- Converged: No

## Files Generated

- `data/adaptive_results.json` - Complete results data
- `data/summary.csv` - Summary metrics
- `plots/model_comparison.png` - Model performance comparison
- `plots/efficiency_comparison.png` - Efficiency analysis
- `plots/*_convergence.png` - Individual model convergence plots
- `plots/*_final_prediction.png` - Final prediction comparisons
- `plots/*_error_evolution.png` - Error evolution plots
- `plots/*_sampling_evolution.png` - Sampling evolution plots
