# Morris Sampling Methods Comparison

## Demonstration Setup

### Parameters

```
True K values: [1.0, 1.0, 1.0]
Grid levels (p): 11
Trajectories (r): 2
Range type: linear
Range: [0, 1]
Parameters to vary: [0, 1, 2]
```

### Expected Grid (11 levels)

```
Level 0:  0.000000
Level 1:  0.100000
Level 2:  0.200000
Level 3:  0.300000
Level 4:  0.400000
Level 5:  0.500000
Level 6:  0.600000
Level 7:  0.700000
Level 8:  0.800000
Level 9:  0.900000
Level 10: 1.000000

Grid spacing: 0.100000
```

### With p=11

```
grid_spacing = 1.0 / (p-1) =  1.0 / 10 = 0.100000
delta = p / [2(p-1)] = 11 / (2*10) = 0.550000

- Grid spacing is 0.100000, but delta is 0.550000

Result: Moving by delta does not equal integer grid positions, goes OFF grid
```

## Method 1: MorrisSampler (Original )

### Sample Output

```
Generated samples:
[[0.6  0.2  1.  ]
 [0.05 0.2  1.  ]
 [0.05 0.2  0.45]
 [0.05 0.75 0.45]
 [1.   0.5  0.7 ]
 [1.   1.05 0.7 ]
 [0.45 1.05 0.7 ]
 [0.45 1.05 0.15]]

Shape: (8, 3)
Total samples: 8
```

### Grid Alignment Check

**Sample 1:**

```
K0: 0.600000 - ON GRID (Level 6)
K1: 0.200000 - ON GRID (Level 2)
K2: 1.000000 - ON GRID (Level 10)
```

**Sample 2:**

```
K0: 0.050000 - OFF GRID (distance: 0.050000, closest: Level 1)
K1: 0.200000 - ON GRID (Level 2)
K2: 1.000000 - ON GRID (Level 10)
```

**Sample 3:**

```
K0: 0.050000 - OFF GRID (distance: 0.050000, closest: Level 1)
K1: 0.200000 - ON GRID (Level 2)
K2: 0.450000 - OFF GRID (distance: 0.050000, closest: Level 4)
```

**Sample 4:**

```
K0: 0.050000 - OFF GRID (distance: 0.050000, closest: Level 1)
K1: 0.750000 - OFF GRID (distance: 0.050000, closest: Level 7)
K2: 0.450000 - OFF GRID (distance: 0.050000, closest: Level 4)
```

**Sample 5:**

```
K0: 1.000000 - ON GRID (Level 10)
K1: 0.500000 - ON GRID (Level 5)
K2: 0.700000 - ON GRID (Level 7)
```

### Summary

```
Samples on grid: 9
Samples off grid: 6

Result: This method FAILS to keep samples on the grid.
```

## Method 2: CorrectedGridMorrisSampler (Fixed)

### Solution: Index-Based Delta

- Calculates: `delta_index = max(1, p//4) = max(1, 2) = 2`
- This is a GRID POSITION (jump 2 levels), not a value
- Jump 2 positions = move by 0.200000 in value
- Result: Points ALWAYS stay on the grid

### Sample Output

```
Shape: (8, 3)
Total samples: 8
```

### Grid Alignment Check

**Sample 1:**

```
K0: 0.700000 - ON GRID (Level 7)
K1: 0.100000 - ON GRID (Level 1)
K2: 0.400000 - ON GRID (Level 4)
```

**Sample 2:**

```
K0: 0.700000 - ON GRID (Level 7)
K1: 0.300000 - ON GRID (Level 3)
K2: 0.400000 - ON GRID (Level 4)
```

**Sample 3:**

```
K0: 0.700000 - ON GRID (Level 7)
K1: 0.300000 - ON GRID (Level 3)
K2: 0.600000 - ON GRID (Level 6)
```

**Sample 4:**

```
K0: 0.500000 - ON GRID (Level 5)
K1: 0.300000 - ON GRID (Level 3)
K2: 0.600000 - ON GRID (Level 6)
```

**Sample 5:**

```
K0: 0.300000 - ON GRID (Level 3)
K1: 0.200000 - ON GRID (Level 2)
K2: 0.100000 - ON GRID (Level 1)
```

### Summary

```
Samples on grid: 15
Samples off grid: 0

Result: This method SUCCEEDS in keeping all samples on the grid.
```

## Final Comparison

### Sampling Statistics

| Metric                     | Method 1 (Broken) | Method 2 (Fixed) |
|----------------------------|-------------------|------------------|
| Total samples              | 8                 | 8                |
| Samples per trajectory     | 4                 | 4                |
| Trajectories               | 2                 | 2                |

### Delta Calculation

| Property      | Method 1 (Broken)      | Method 2 (Fixed)                |
|---------------|------------------------|---------------------------------|
| Delta value   | 0.550000               | 2 (grid positions)              |
| Delta type    | VALUE (wrong)          | GRID INDEX (correct)            |
| Grid spacing  | 0.100000               | 0.100000                        |

### Key Difference

**Method 1: Value-Based Delta**

```
delta = 0.550000 (tries to move by this VALUE)

Example: If at grid[5] = 0.500000
         moves to 0.500000 + 0.550000 = 1.050000
         
Problem: 1.050000 is NOT on the grid
Closest grid point: 0.900000 (grid[9]) or 1.000000 (grid[10])
```

**Method 2: Index-Based Delta**

```
delta_index = 2 (moves by this many GRID POSITIONS)

Example: If at grid[5] = 0.500000
         moves to grid[5+2] = grid[7] = 0.700000
         
Success: 0.700000 IS on the grid
Actual value change: 0.200000
```

## Why This Matters

1. **Morris sensitivity analysis requires samples on a discrete grid**
   - The method calculates finite differences between grid points
   - Off-grid samples invalidate the theoretical foundation

2. **Method 1's delta (0.550000) doesn't match grid spacing (0.100000)**
   - The ratio 0.550000/0.100000 = 5.5 is not an integer
   - This causes samples to fall between grid points

3. **Method 2 moves exactly 2 grid positions = 0.200000 value change**
   - The movement is always an integer multiple of grid spacing
   - All samples remain on valid grid points

4. **Only Method 2 ensures mathematically valid Morris sampling**
   - Preserves the discrete grid structure required by Morris (1991)
   - Finite difference calculations are performed between actual grid points

## Conclusion

Use `CorrectedGridMorrisSampler` for proper Morris sampling. The original `MorrisSampler` has a fundamental bug in delta calculation that causes samples to fall off the discrete grid, invalidating the Morris sensitivity analysis method.

## Mathematical Background

### Morris (1991) Delta Formula

The original Morris method specifies:

```
delta = p / [2(p-1)]
```

For a linear grid from 0 to 1 with p levels:

```
grid_spacing = 1 / (p-1)
```

Converting the value-based delta to grid positions:

```
delta_positions = delta / grid_spacing
                = [p / 2(p-1)] / [1 / (p-1)]
                = [p / 2(p-1)] × (p-1)
                = p / 2
```

Therefore, Morris's value-based delta ALWAYS equals p/2 grid positions.

### Why p=10 Works but p=11 Fails

**For p=10 (even):**

```
delta_positions = 10/2 = 5.0 (exact integer)
Grid-based implementation coincidentally works
```

**For p=11 (odd):**

```
delta_positions = 11/2 = 5.5 (not an integer)
Grid-based implementation fails, revealing the bug
```

### Recommendation for p Values

- **Prefer even p values:** p = 4, 6, 8, 10, 12, 20, 100, 1000
  - Ensures p/2 is an exact integer
  - Provides exact match to Morris's formula

- **Odd p values are valid but require rounding:**
  - Use `delta_index = round(p/2)` for best approximation
  - Example: p=11 → delta_index = round(5.5) = 6

### Correct Implementation

```python
# WRONG (original MorrisSampler):
delta = p / (2 * (p - 1))  # Value: 0.550
new_value = current_value + delta  # May go off grid

# CORRECT (CorrectedGridMorrisSampler):
delta_index = p // 2  # Grid positions: 5 (for p=10)
new_value = grid[current_index + delta_index]  # Stays on grid
```

The key difference is using grid indices instead of values for the delta parameter.
