# Traditional Approach Analysis Report

**Generated:** 2025-07-27 18:32:39

## Configuration

- **Simulator:** loki
- **Chemistry File:** O2_simple_1.chem
- **Setup File:** setup_O2_simple.in
- **Training Samples:** 100
- **Test Samples:** 20
- **K Columns:** [0, 1, 2]
- **Sampling Method:** latin_hypercube

## Results Summary

| Model | Train RMSE | Test RMSE | Train R² | Test R² | Total Sims | Efficiency (R²/Sim) |
|-------|------------|-----------|----------|---------|------------|--------------------|
| random_forest | 3.30e-16 | 7.46e-16 | 0.9320 | 0.6567 | 120 | 5.47e-03 |
| neural_network | 7.35e-16 | 7.49e-16 | 0.6694 | 0.6313 | 120 | 5.26e-03 |

## Best Performing Model

**random_forest** achieved the lowest test RMSE of 7.46e-16
- Test R² score: 0.6567
- Train RMSE: 3.30e-16
- Train R² score: 0.9320
- Total simulations: 120

## Files Generated

- `data/traditional_results.json` - Complete results data
- `data/summary.csv` - Summary metrics
- `plots/model_comparison.png` - Model performance comparison
- `plots/performance_metrics.png` - Detailed performance metrics
- `plots/training_test_comparison.png` - Training vs test comparison
- `plots/simulation_efficiency.png` - Simulation efficiency analysis
- `models/*.joblib` - Trained model files
