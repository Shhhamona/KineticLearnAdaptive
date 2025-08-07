# Traditional Approach Analysis Report

**Generated:** 2025-07-15 16:06:33

## Configuration

- **Simulator:** loki
- **Chemistry File:** O2_simple_1.chem
- **Setup File:** setup_O2_simple.in
- **Training Samples:** 20
- **Test Samples:** 10
- **K Columns:** [0, 1, 2]
- **Sampling Method:** latin_hypercube

## Results Summary

| Model | Train R² | Test R² | Train RMSE | Test RMSE | Total Sims | Efficiency |
|-------|----------|---------|------------|-----------|------------|------------|
| random_forest | 0.9238 | 0.4290 | 3.17e-16 | 9.03e-16 | 30 | 1.43e-02 |
| neural_network | 0.8703 | 0.4996 | 4.63e-16 | 8.94e-16 | 30 | 1.67e-02 |

## Best Performing Model

**neural_network** achieved the highest test R² of 0.4996

## Files Generated

- `data/traditional_results.json` - Complete results data
- `data/summary.csv` - Summary metrics
- `plots/model_comparison.png` - Model performance comparison
- `plots/performance_metrics.png` - Detailed performance metrics
- `plots/training_test_comparison.png` - Training vs test comparison
- `plots/simulation_efficiency.png` - Simulation efficiency analysis
- `models/*.joblib` - Trained model files
