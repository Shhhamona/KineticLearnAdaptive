# Adaptive Sampling Analysis Report

**Generated:** 2025-07-27 17:44:22

## Configuration

- **Simulator:** loki
- **Chemistry File:** O2_simple_1.chem
- **Setup File:** setup_O2_simple.in
- **True K Values:** [6e-16, 1.3e-15, 9.6e-16]
- **Initial Samples:** 10
- **Iteration Samples:** 5
- **Max Iterations:** 5
- **Convergence Threshold:** 0.001
- **Sampling Method:** latin_hypercube

## Results Summary

| Model | Initial Error | Final Error | Improvement (%) | Total Sims | Iterations | Converged | Time (s) | Efficiency |
|-------|---------------|-------------|-----------------|------------|------------|-----------|----------|------------|
| random_forest | 0.0764 | 0.0518 | 32.23 | 35 | 5 | No | 398.55 | 2.71e-02 |
| neural_network | 0.0160 | 0.0393 | -145.03 | 35 | 5 | No | 313.34 | 2.74e-02 |

## Best Performing Model

**neural_network** achieved the lowest final error of 0.0393
- Error improvement: -145.03%
- Total simulations: 35
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
