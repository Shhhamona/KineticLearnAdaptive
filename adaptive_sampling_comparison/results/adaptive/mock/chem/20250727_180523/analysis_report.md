# Adaptive Sampling Analysis Report

**Generated:** 2025-07-27 18:05:42

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

| Model | Final RMSE | Final R² | Initial Error | Final Error | Improvement (%) | Total Sims | Iterations | Converged | Time (s) | Efficiency (R²/Sim) |
|-------|------------|----------|---------------|-------------|-----------------|------------|------------|-----------|----------|--------------------|
| random_forest | 1.82e-16 | 0.8800 | 0.0871 | 0.1326 | -52.29 | 100 | 8 | No | 2.49 | 8.80e-03 |
| neural_network | 1.82e-16 | 0.8803 | 0.1305 | 0.1324 | -1.48 | 100 | 8 | No | 1.05 | 8.80e-03 |

## Best Performing Model

**neural_network** achieved the lowest RMSE of 1.82e-16
- Final R² score: 0.8803
- Final relative error: 0.1324
- Error improvement: -1.48%
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
