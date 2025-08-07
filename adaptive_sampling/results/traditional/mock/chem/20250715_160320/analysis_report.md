# Traditional Approach Analysis Report

**Generated:** 2025-07-15 16:03:25

## Configuration

- **Simulator:** mock
- **Chemistry File:** chem.chem
- **Setup File:** setup.in
- **Training Samples:** 100
- **Test Samples:** 30
- **K Columns:** [0, 1, 2]
- **Sampling Method:** latin_hypercube

## Results Summary

| Model | Train R² | Test R² | Train RMSE | Test RMSE | Total Sims | Efficiency |
|-------|----------|---------|------------|-----------|------------|------------|
| random_forest | 0.9363 | 0.5200 | 4.74e-16 | 1.45e-15 | 130 | 4.00e-03 |
| neural_network | 0.8420 | 0.6014 | 8.33e-16 | 1.26e-15 | 130 | 4.63e-03 |

## Best Performing Model

**neural_network** achieved the highest test R² of 0.6014

## Files Generated

- `data/traditional_results.json` - Complete results data
- `data/summary.csv` - Summary metrics
- `plots/model_comparison.png` - Model performance comparison
- `plots/performance_metrics.png` - Detailed performance metrics
- `plots/training_test_comparison.png` - Training vs test comparison
- `plots/simulation_efficiency.png` - Simulation efficiency analysis
- `models/*.joblib` - Trained model files
