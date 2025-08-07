# Traditional Approach Analysis Report

**Generated:** 2025-07-15 15:56:02

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
| random_forest | 0.9238 | 0.4666 | 3.17e-16 | 8.30e-16 | 30 | 1.56e-02 |
| neural_network | 0.8703 | 0.3529 | 4.63e-16 | 9.90e-16 | 30 | 1.18e-02 |

## Best Performing Model

**random_forest** achieved the highest test R² of 0.4666

## Files Generated

- `data/traditional_results.json` - Complete results data
- `data/summary.csv` - Summary metrics
- `plots/model_comparison.png` - Model performance comparison
- `plots/performance_metrics.png` - Detailed performance metrics
- `plots/training_test_comparison.png` - Training vs test comparison
- `plots/simulation_efficiency.png` - Simulation efficiency analysis
- `models/*.joblib` - Trained model files
