import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
adaptive_dir = os.path.join(current_dir, '..')
sys.path.append(adaptive_dir)

from traditional_approach import TraditionalApproach
from base_simulator import MockSimulator
import numpy as np

print('Testing Traditional Approach...')

# Initialize components
simulator = MockSimulator('setup.in', 'chem.chem', 'path/to/loki')
true_k = np.array([6e-16, 1.3e-15, 9.6e-16])

# Create traditional approach
traditional = TraditionalApproach(
    simulator=simulator,
    true_k_values=true_k,
    k_columns=[0, 1, 2],
    n_train=50,
    n_test=20,
    sampling_method='latin_hypercube'
)

# Run quick test
print('Running traditional approach...')
results = traditional.run_complete_study(n_train=50, n_test=20, save_results=False)
print(f'Results keys: {list(results.keys())}')
print(f'Total simulations: {results["total_simulations"]}')
print(f'Test R²: {results["test_metrics"]["r2_score"]:.4f}')
print('Traditional approach test passed!')
