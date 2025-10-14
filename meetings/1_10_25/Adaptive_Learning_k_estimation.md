# Adaptive Learning Strategy - True K

01/10/2025



**Performance measured by K prediction accuracy only** - not overall MSE across all outputs.Learning - K Estimation Performance

## K-Centered Adaptive Learning Results

This analysis focuses on **K estimation performance** - measuring how  different sampling strategies predict the true rate coefficient values.

## Sampling Strategy Comparison

<img src="results/sample_efficiency_real_k/sample_efficiency_20250928_155718.png" alt="Sample Efficiency - K Estimation Performance" width="600">

**Performance measured by K prediction accuracy only** - not overall MSE across all outputs.

### Key Observations

The results show performance ranking across different sampling methods when evaluated specifically on rate coefficient estimation:

1. **Uniform Latin Hypercube** - Best traditional sampling method
2. **Morris Continous** - Close second among traditional methods  
3. **Log Uniform Latin Hypercube** 
4. **Uniform** -
5. **Morris Discrete** 


## K-Centered Adaptive Learning Analysis

<img src="results/k_centered_results/k_centered_performance_comparison.png" alt="K-Centered Performance Comparison" width="800">

### Key Insights

Focusing on areas near K_true has no positive impact on model performance (Why??)

#### Box Size Analysis (Left Plot)
- **Larger boxes perform better**: 0.5×K_true to 2×K_true (Box Size 100%) shows best performance
- **Smaller boxes degrade performance**: Reducing the bounding box consistently worsens results
- **Counter-intuitive finding**: Focusing sampling closer to true K values does not improve K prediction accuracy

#### Shrink Rate Analysis (Right Plot)  
- **No shrinking is optimal**: 0% shrinking (maintaining full box size) performs best
- **Adaptive shrinking degrades performance**: All shrinking strategies (15%, 30%, 40%, 50%) show worse results
- **Diminishing returns**: Higher shrinking rates provide progressively worse performance


Next Steps:
- TRY Neural Network instead of SVM


