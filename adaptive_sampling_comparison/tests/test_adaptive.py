import sys
sys.path.append('src')
from adaptive_approach import AdaptiveSamplingApproach
from base_simulator import MockSimulator
import numpy as np

print('Testing Adaptive Sampling Approach...')

# Initialize components
simulator = MockSimulator('setup.in', 'chem.chem', 'path/to/loki')
true_k = np.array([6e-16, 1.3e-15, 9.6e-16])

# Create adaptive approach
adaptive = AdaptiveSamplingApproach(
    simulator=simulator,
    true_k_values=true_k,
    k_columns=[0, 1, 2]
)

# Run quick test
print('Running adaptive sampling approach...')
results = adaptive.run_adaptive_sampling(
    n_initial=20, 
    n_iteration=10,
    save_results=True
)
print(f'Results keys: {list(results.keys())}')
print(f'Total simulations: {results["total_simulations"]}')
print(f'Final error: {results["final_error"]:.6f}')
print(f'Error improvement: {results["error_improvement"]:.2f}%')
print('Adaptive approach test passed!')
