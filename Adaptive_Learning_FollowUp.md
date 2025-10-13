# Adaptive Learning Strategy - Morris Sampling

01/10/2025

## 1. Morris Sampling Corrections

### Issue Identified
The original Morris sampling implementation violated Morris Sampling by mixing discrete grid creation with continuous delta steps, causing samples to land between grid points rather than staying on the predefined discrete grid.

### Problems Fixed
- **Delta Calculation**: Changed from value-based `delta = p/(2*(p-1))` to index-based `delta_index = max(1, p//4)` 
- **Grid Adherence**: Now all trajectory moves operate on grid indices first, then convert to values
- **Trajectory Generation**: Ensures samples stay exactly on discrete grid points throughout the sampling process

### New Methods Implemented
1. **CorrectedGridMorrisSampler()**: Fixed discrete grid implementation that maintains proper Morris structure
2. **ContinuousMorrisSampler()**: True continuous Morris sampling allowing parameters to take any value within bounds, with scale-appropriate delta (multiplicative for log space, additive for linear space)

Both methods now provide  Morris elementary effects for  sensitivity analysis and  parameter space exploration.


## 2. Results

### Morris Sampling Performance Analysis

<img src="results/toReport2/sample_efficiency_20250928_141619_all_models.png" alt="Sample Efficiency - All Models" width="600">

<img src="results/toReport2/comprehensive_sampling_comparison_all_models.png" alt="Comprehensive Sampling Comparison - All Models" width="1000">


The corrected Morris sampling methods showed mixed results:

**Corrected Grid Morris (Purple)**: Performed **worse** than the original flawed version. The stricter discrete grid adherence created overly constrained sampling that damaged dataset quality, leading to higher MSE values across all sample sizes.

**Continuous Morris (Blue)**: Performed **much better**, achieving similiar performance to the other methods. The continuous approach provides proper space-filling properties while maintaining Morris theoretical validity.



<img src="results/toReport2/comprehensive_sampling_comparison_best_models.png" alt="Comprehensive Sampling Comparison - Best Models" width="1000">


### Parameter Distribution Analysis

![Morris Data Analysis](results/toReport2/morris_data_analysis.png)

The histograms reveal clear differences between continuous and discrete Morris sampling:
- **Continuous Morris**: Shows smooth, well-distributed parameter coverage across the entire range
- **Discrete Morris**: Shows constrained, sparse sampling with poor space-filling properties


Next Steps:
- Double check Latin Hypercube Sampling - Investigate if the current approach of taking a subset of the full set is valid


## 3. Center Point Investigation

Following professor feedback about using K_true as center, I nede to investigate different center point choices:

### Center Point Testing
- **Original Concern**: Using K_true as zone center could provide unfair advantage
- **Mean Center**: Changed to use mean of augmented data `np.mean(y_scaled_aug, axis=0)`
- **Other**: Try other points and see the outcome.

