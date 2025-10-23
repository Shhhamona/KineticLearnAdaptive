# Morris Sampling Methods Comparison

### Different Implementation

```python
# (original MorrisSampler):
delta = p / (2 * (p - 1))  # Value: 0.550
new_value = current_value + delta  # May go off grid

#  (CorrectedGridMorrisSampler):
delta_index = p // 2  # Grid positions: 5 (for p=10)
new_value = grid[current_index + delta_index]  # Stays on grid
```

The key difference is using grid indices instead of values for the delta parameter.

## Example

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

Result: Moving by delta does not equal integer grid positions, goes off grid
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

## Method 2: CorrectedGridMorrisSampler

### Solution: Index-Based Delta

- Calculates: `delta_index = max(1, p//4) = max(1, 2) = 2`
- This is a GRID POSITION (jump 2 levels), not a value
- Jump 2 positions = move by 0.200000 in value
- Result: Points ALWAYS stay on the grid

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

Result: This method succeeds in keeping all samples on the grid.
```

### Morris Delta Formula

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

(By mistake I calculated this wrong and reacehd delta_positions = p/4  instead of p/2)

### Why p=10 Works but p=11 Fails

**For p=10 (even):**
Method 1

```
delta = p / [2(p-1)] = 10/(2(10-1)) = 0.5555556
grid_spacing = 1 / (p-1) = 0.111111

delta_positions = delta/grid_spacing = 0.555555/0.111111 = 5
```

Method 2:

```
delta_positions = p//2 = 10/2 = 5
```

Both work

**For p=11 (odd):**

Method 1

```
delta = p / [2(p-1)] = 11/(2(11-1)) = 0.55
grid_spacing = 1 / (p-1) = 1 / 10= 0.10

delta_positions = delta/grid_spacing = 0.555555/0.10= 5.5
```

NOT Equal to Method 2:

```
delta_positions = p//2 = 11//2 = 5
```
