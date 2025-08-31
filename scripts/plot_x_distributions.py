#!/usr/bin/env python3
"""
Compare composition (X) distributions between initial training and newly sampled data
using `results/training_snapshots/train_600_X.txt`.
Saves a multi-panel PNG and prints summary statistics.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

TRAIN_X_FILE = r"c:\Users\Rodolfo Simões\Documents\PlasmaML\KineticLearn\results\training_snapshots\train_600_X.txt"
TEST_X_FILE = r"c:\Users\Rodolfo Simões\Documents\PlasmaML\KineticLearn\results\training_snapshots\test_X.txt"
OUT_DIR = r"c:\Users\Rodolfo Simões\Documents\PlasmaML\KineticLearn\results"


def load_train_x(path=TRAIN_X_FILE, test_path=TEST_X_FILE):
    arr = np.loadtxt(path)
    # first 100 initial training, last 50 sampled
    initial = arr[:100]
    sampled = arr[-50:]
    print(f"Loaded train X shape: {arr.shape}")
    print(f"  initial (first 100): {initial.shape}")
    print(f"  sampled (last 50): {sampled.shape}")

    test_arr = None
    if os.path.exists(test_path):
        try:
            test_arr = np.loadtxt(test_path)
            print(f"Loaded test X shape: {test_arr.shape}")
        except Exception as e:
            print(f"Warning: couldn't load test X from {test_path}: {e}")
    else:
        print(f"No test X file at {test_path}, continuing without test data")

    return initial, sampled, test_arr


def plot_x_distributions():
    initial, sampled, test_arr = load_train_x()
    n_features = initial.shape[1]
    feature_names = [f"feat{i+1}" for i in range(n_features)]

    cols = 3
    rows = int(np.ceil(n_features / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    axes = axes.flatten()
    fig.suptitle('Composition (X) Distribution: initial (100) vs sampled (50) vs test', fontsize=14)

    for i in range(n_features):
        ax = axes[i]
        ax.hist(initial[:, i], bins=25, alpha=0.6, label='initial (100)', color='green', density=True)
        ax.hist(sampled[:, i], bins=25, alpha=0.5, label='sampled (50)', color='red', density=True)
        if test_arr is not None:
            # plot a subsample of test for visibility if test is large
            test_plot = test_arr
            if test_arr.shape[0] > 2000:
                idx = np.linspace(0, test_arr.shape[0] - 1, 2000).astype(int)
                test_plot = test_arr[idx]
            ax.hist(test_plot[:, i], bins=25, alpha=0.4, label='test (all)', color='blue', density=True)
        ax.set_title(feature_names[i])
        ax.legend()
        ax.grid(alpha=0.3)

    # hide unused axes
    for j in range(n_features, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'x_distributions_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {out_path}")

    # print stats
    print('\n' + '='*60)
    print('SUMMARY STATISTICS (X features)')
    print('='*60)
    for i in range(n_features):
        init = initial[:, i]
        samp = sampled[:, i]
        print(f"\n{feature_names[i]}:")
        print(f"  initial: mean={init.mean():.6e}, std={init.std():.6e}, range=[{init.min():.6e},{init.max():.6e}]")
        print(f"  sampled: mean={samp.mean():.6e}, std={samp.std():.6e}, range=[{samp.min():.6e},{samp.max():.6e}]")
        overlap = not (samp.max() < init.min() or samp.min() > init.max())
        print(f"  overlap: {'YES' if overlap else 'NO'}")


def main():
    try:
        plot_x_distributions()
        print('\n✅ X-distribution analysis complete')
    except Exception as e:
        print('❌ Error:', e)

if __name__ == '__main__':
    main()
