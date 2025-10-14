# Morris Sampling Corrections and K-Centered Adaptive Learning: Technical Summary

## 1. Introduction

Following professor feedback on the previous work (Summary_VG_2.md), three critical issues were identified that required fundamental corrections to our approach:

1. **Morris Sampling Implementation Issues**: The original Morris sampling was not correctly implemented
2. **K-Centered Strategy Refinement**: The center point should not be K_true but another reference point
3. **Target Focus**: Should target only the true K values, not all parameters

This document addresses Issue #1: the Morris sampling corrections and explains the technical differences between the original flawed implementation and the corrected versions.

## 2. Morris Sampling: The Original Problem

### 2.1 What Was Wrong with the Original Morris Sampler?

The original `MorrisSampler()` function had several critical flaws that violated Morris (1991) methodology:

#### **Problem 1: Incorrect Delta Calculation**
```python
# WRONG: Original implementation
delta = p/(2*(p-1))  # This gives values >> 1.0 for typical p values
```

**Issue**: For typical Morris parameters (p=4, levels), this gives:
- `delta = 4/(2*(4-1)) = 4/6 = 0.667`
- But this delta was applied directly to grid values, not grid positions
- Result: Samples would "jump off" the discrete grid entirely

#### **Problem 2: Mixing Grid Values with Continuous Delta**
```python
# WRONG: Direct arithmetic on grid values
if(new_node[i]/mean_point>1):
    new_node[i] = new_node[i]-delta  # Delta applied to actual values
else:
    new_node[i] = new_node[i]+delta
```

**Issue**: 
- Morris sampling requires staying on discrete grid points
- Original code mixed grid-based sampling with continuous delta steps
- Result: Samples landed between grid points, violating Morris theory

#### **Problem 3: Inconsistent Grid Logic**
- Grid was created correctly using `np.logspace()` or `np.linspace()`
- But trajectory generation ignored the grid structure
- Starting points were selected from grid, but subsequent moves were continuous

### 2.2 Consequences of the Flawed Implementation

The original flawed Morris sampler would:
1. **Generate invalid samples** that don't follow Morris one-at-a-time (OAT) principle
2. **Break grid structure** leading to uneven parameter space coverage  
3. **Produce inconsistent sensitivity analysis** since delta steps varied in size
4. **Fail to provide proper Morris elementary effects** for sensitivity screening

## 3. Corrected Morris Sampling Methods

### 3.1 Method 3: Corrected Grid-Based Morris Sampling

The `CorrectedGridMorrisSampler()` fixes the discrete grid issues:

#### **Fix 1: Grid Index-Based Delta**
```python
# CORRECT: Work with grid indices, not values
delta_index = max(1, p // 4)  # Move by grid positions
print(f"Corrected delta (grid positions): {delta_index}")
```

**Improvement**:
- Delta now represents **grid position movements**, not value changes
- Ensures all samples stay exactly on predefined grid points
- Maintains Morris's discrete grid structure

#### **Fix 2: Index-Based Trajectory Generation**
```python
# CORRECT: Track and modify grid indices
current_indices = start_indices.copy()
for param_order in order:
    new_indices = current_indices.copy()
    
    # Move by grid index, stay on grid
    current_idx = current_indices[param_order]
    
    if current_idx >= p // 2:  # Upper half of grid
        new_idx = max(0, current_idx - delta_index)
    else:  # Lower half of grid
        new_idx = min(p - 1, current_idx + delta_index)
    
    new_indices[param_order] = new_idx
    
    # Convert indices to values
    new_node = [w[idx] for idx in new_indices]
```

**Improvement**:
- All operations performed on **grid indices** first
- Values computed by converting indices to grid points
- **Guarantees discrete grid adherence**

#### **Fix 3: Proper Bounds Checking**
```python
# CORRECT: Index bounds ensure grid adherence
new_idx = max(0, current_idx - delta_index)  # Stay within [0, p-1]
new_idx = min(p - 1, current_idx + delta_index)
```

### 3.2 Method 4: Continuous Morris Sampling

The `ContinuousMorrisSampler()` implements true Morris (1991) continuous theory:

#### **Innovation 1: True Continuous Parameter Space**
```python
# CONTINUOUS: No grid constraints
for j in range(k_size):
    if k_range_type == "log":
        # Log-uniform sampling anywhere in bounds
        log_val = np.random.uniform(k_range[0], k_range[1])
        start_node.append(10**log_val)
    else:
        # Uniform sampling anywhere in bounds
        start_node.append(np.random.uniform(min_bound, max_bound))
```

**Advantage**:
- Parameters can take **any value** within bounds
- No artificial grid limitations
- Better space-filling properties

#### **Innovation 2: Scale-Appropriate Delta**
```python
# CONTINUOUS LOG: Multiplicative delta in log space
if k_range_type == "log":
    log_range = k_range[1] - k_range[0]
    delta_fraction = 1.0 / (2 * (p - 1))  # Morris's original delta
    log_delta = delta_fraction * log_range
    
# CONTINUOUS LINEAR: Additive delta
else:
    param_range = max_bound - min_bound
    delta = param_range / (2 * (p - 1))  # Morris's original delta
```

**Advantage**:
- **Log space**: Delta is multiplicative (proportional changes)
- **Linear space**: Delta is additive (absolute changes)  
- **Scale-appropriate** for both parameter types

#### **Innovation 3: Continuous Trajectory Generation**
```python
# CONTINUOUS: Apply delta in appropriate space
if k_range_type == "log":
    # Log space: multiplicative delta
    current_log = np.log10(current_node[param_idx])
    mean_log = (k_range[0] + k_range[1]) / 2
    
    if current_log > mean_log:
        new_log = current_log - log_delta
    else:
        new_log = current_log + log_delta
    
    new_node[param_idx] = 10**new_log
    new_node[param_idx] = np.clip(new_node[param_idx], min_bound, max_bound)
```

**Advantage**:
- **Mathematically correct** delta application in each space
- **Respects parameter bounds** through clipping
- **Maintains Morris OAT principle** while allowing continuous values

## 4. Technical Comparison Summary

| Aspect | Original (Flawed) | Corrected Grid | Continuous |
|--------|-------------------|----------------|------------|
| **Grid Adherence** | ❌ Broken | ✅ Perfect | N/A |
| **Delta Calculation** | ❌ Value-based | ✅ Index-based | ✅ Scale-based |
| **Parameter Space** | ❌ Mixed | ✅ Discrete | ✅ Continuous |
| **Morris Theory** | ❌ Violated | ✅ Discrete Morris | ✅ Continuous Morris |
| **Sensitivity Analysis** | ❌ Inconsistent | ✅ Valid | ✅ Valid |
| **Space Coverage** | ❌ Uneven | ✅ Grid-limited | ✅ Full coverage |

## 5. Expected Improvements

The corrected Morris sampling methods should provide:

1. **Better Parameter Space Exploration**
   - Corrected grid method: Proper discrete coverage
   - Continuous method: Dense space-filling sampling

2. **Valid Sensitivity Analysis**  
   - Consistent delta steps enable proper Morris elementary effects
   - Reliable parameter ranking and screening

3. **Improved Model Training**
   - Better distributed training samples
   - More representative parameter combinations
   - Enhanced generalization capabilities

4. **Methodological Rigor**
   - Adherence to Morris (1991) theoretical framework
   - Reproducible and defensible sampling strategy
   - Proper comparison with other sampling methods

## 6. Next Steps

With the Morris sampling now corrected, the next phases will address:

1. **K-Centered Strategy Refinement** (Issue #2)
   - Redefine center point away from K_true
   - Explore alternative reference points

2. **True K Targeting** (Issue #3) 
   - Focus adaptive sampling on K prediction only
   - Optimize for rate coefficient accuracy specifically

3. **Performance Validation**
   - Compare corrected Morris vs. original implementation
   - Quantify improvements in sampling efficiency
   - Validate enhanced model performance

The corrected Morris sampling foundation now enables proper investigation of adaptive learning strategies for chemical kinetics problems.