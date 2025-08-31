from pathlib import Path
import numpy as np

file_path = Path(__file__).parents[1] / 'data' / 'SampleEfficiency' / 'O2_simple_uniform.txt'
if not file_path.exists():
    print('File not found:', file_path)
    raise SystemExit(1)

k_vals = []
with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        # take first three columns as k values
        try:
            k1, k2, k3 = float(parts[0]), float(parts[1]), float(parts[2])
            k_vals.append((k1, k2, k3))
        except Exception:
            continue

arr = np.array(k_vals)
mins = arr.min(axis=0)
maxs = arr.max(axis=0)
means = arr.mean(axis=0)
medians = np.median(arr, axis=0)

k_true = np.array([6.00E-16, 1.30E-15, 9.60E-16])

print('Samples parsed:', arr.shape[0])
for i in range(3):
    print(f'k[{i}]: min={mins[i]:.6e}, max={maxs[i]:.6e}, mean={means[i]:.6e}, median={medians[i]:.6e}')

# compute inferred symmetric relative bounds around k_true
# infer rel_width as max(abs(k_true - min), abs(max - k_true)) / k_true
rel_widths = []
for i in range(3):
    lo, hi = mins[i], maxs[i]
    k0 = k_true[i]
    rel_lo = (k0 - lo) / k0
    rel_hi = (hi - k0) / k0
    rel_widths.append((rel_lo, rel_hi))
    print(f'k[{i}] relative distances from center: down={rel_lo:.3f}, up={rel_hi:.3f}')

# check if k_true lies near midpoint
for i in range(3):
    midpoint = 0.5 * (mins[i] + maxs[i])
    dev = (k_true[i] - midpoint) / midpoint
    print(f'k[{i}] center dev from midpoint: {dev:.6f} (fraction of midpoint)')

# Suggest inferred bounds as [min,max] and symmetric approx
print('\nInferred bounds per k (min, max):')
for i in range(3):
    print(f'  k[{i}]: [{mins[i]:.6e}, {maxs[i]:.6e}]')

# Suggest an approximate symmetric rel_width (max of lo/up)
approx_rel = [max(abs(a), abs(b)) for a,b in rel_widths]
print('\nApprox symmetric relative widths around k_true (fraction of k_true):')
for i, w in enumerate(approx_rel):
    print(f'  k[{i}]: {w:.3f} (~{w*100:.1f}% )')
