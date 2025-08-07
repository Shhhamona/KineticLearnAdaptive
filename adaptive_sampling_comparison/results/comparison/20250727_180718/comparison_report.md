# Approach Comparison Report

**Generated:** 2025-07-27 18:07:25

## Overall Statistics

### Traditional Approach
- Number of runs: 12
- Mean RMSE: 1.08e-15
- Std RMSE: 2.56e-16
- Mean R²: 0.5445
- Std R²: 0.0956
- Mean simulations: 95.0

### Adaptive Approach
- Number of runs: 6
- Mean RMSE: 1.82e-16
- Std RMSE: 2.00e-19
- Mean R²: 0.8801
- Std R²: 0.0003
- Mean simulations: 78.3

## Detailed Results

| Approach | Model | RMSE | R² | Total Sims | Simulator | Chemistry |
|----------|-------|------|----|-----------|-----------|-----------|
| traditional | random_forest | 8.30e-16 | 0.4666 | 30 | O2_simple_1 | 20250715_155004 |
| traditional | neural_network | 9.90e-16 | 0.3529 | 30 | O2_simple_1 | 20250715_155004 |
| traditional | random_forest | 9.03e-16 | 0.4290 | 30 | O2_simple_1 | 20250715_160346 |
| traditional | neural_network | 8.94e-16 | 0.4996 | 30 | O2_simple_1 | 20250715_160346 |
| traditional | random_forest | 7.46e-16 | 0.6567 | 120 | O2_simple_1 | 20250715_161201 |
| traditional | neural_network | 7.49e-16 | 0.6313 | 120 | O2_simple_1 | 20250715_161201 |
| traditional | random_forest | 1.22e-15 | 0.6089 | 130 | chem | 20250715_154835 |
| traditional | neural_network | 1.19e-15 | 0.6469 | 130 | chem | 20250715_154835 |
| traditional | random_forest | 1.45e-15 | 0.5200 | 130 | chem | 20250715_160254 |
| traditional | neural_network | 1.26e-15 | 0.6014 | 130 | chem | 20250715_160254 |
| traditional | random_forest | 1.45e-15 | 0.5200 | 130 | chem | 20250715_160320 |
| traditional | neural_network | 1.26e-15 | 0.6014 | 130 | chem | 20250715_160320 |
| adaptive | random_forest | nan | nan | 35 | O2_simple_1 | 20250727_172905 |
| adaptive | neural_network | nan | nan | 35 | O2_simple_1 | 20250727_172905 |
| adaptive | random_forest | nan | nan | 100 | chem | 20250727_172727 |
| adaptive | neural_network | nan | nan | 100 | chem | 20250727_172727 |
| adaptive | random_forest | 1.82e-16 | 0.8800 | 100 | chem | 20250727_180523 |
| adaptive | neural_network | 1.82e-16 | 0.8803 | 100 | chem | 20250727_180523 |

## Best Performers

**Lowest RMSE:** adaptive approach with neural_network (RMSE: 1.82e-16, R²: 0.8803)

**Highest R²:** adaptive approach with neural_network (R²: 0.8803, RMSE: 1.82e-16)

## Files Generated

- `comparison_summary.csv` - Combined results data
- `rmse_comparison.png` - RMSE comparison plots
- `r2_comparison.png` - R² score comparison plots
- `efficiency_comparison.png` - Efficiency analysis
- `model_specific_comparison.png` - Model-specific comparisons
