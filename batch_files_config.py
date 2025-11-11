"""
Configuration file for batch simulation files used in adaptive sampling experiments.

Each batch file represents a pool of simulations with specific K-value ranges.
The files are organized by K-range factor (how wide the sampling bounds are around K_true).
"""

BATCH_FILES = [
    # K-factor 2.0 - Widest bounds
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-08-27/batch_4000sims_20250827_010028.json',
        'label': 'Window Batch 1 (4000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/2, K_true×2]'
    },
    
    # K-factor 1.5
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-08/batch_3000sims_20251108_061037.json',
        'label': 'Window Batch 2 (3000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.5, K_true×1.5]'
    },
    
    # K-factor 1.15
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-28/batch_1000sims_20250928_191628.json',
        'label': 'Window Batch 3 (1000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.15, K_true×1.15]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2500sims_20250929_031845.json',
        'label': 'Window Batch 4 (2500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.15, K_true×1.15]'
    },
    
    # K-factor 1.05
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-08/batch_2500sims_20251108_184222.json',
        'label': 'Window Batch 5 (2500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.05, K_true×1.05]'
    },
    
    # K-factor 1.025
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2500sims_20251109_072738.json',
        'label': 'Window Batch 6 (2500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.025, K_true×1.025]'
    },
    
    # K-factor 1.01
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-10-27/batch_500sims_20251027_154921.json',
        'label': 'Window Batch 7 (500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.01, K_true×1.01]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-07/batch_2000sims_20251107_204934.json',
        'label': 'Window Batch 8 (2000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.01, K_true×1.01]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2500sims_20251109_195233.json',
        'label': 'Window Batch 9 (2500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.01, K_true×1.01]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2525sims_20251109_195849.json',
        'label': 'Window Batch 10 (2525 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.01, K_true×1.01]'
    },
    
    # K-factor 1.0075
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2550sims_20251109_200200.json',
        'label': 'Window Batch 11 (2550 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.0075, K_true×1.0075]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2575sims_20251109_200551.json',
        'label': 'Window Batch 12 (2575 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.0075, K_true×1.0075]'
    },
    
    # K-factor 1.005
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2000sims_20250929_205429.json',
        'label': 'Window Batch 13 (2000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.005, K_true×1.005]'
    },
    
    # K-factor 1.0025
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2600sims_20251109_074357.json',
        'label': 'Window Batch 14 (2600 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.0025, K_true×1.0025]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-10/batch_2550sims_20251110_033347.json',
        'label': 'Window Batch 15 (2550 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.0025, K_true×1.0025]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-10/batch_2575sims_20251110_033748.json',
        'label': 'Window Batch 16 (2575 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.0025, K_true×1.0025]'
    },
    
    # K-factor 1.001
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-08/batch_2600sims_20251108_185805.json',
        'label': 'Window Batch 17 (2600 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.001, K_true×1.001]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-10/batch_2500sims_20251110_032616.json',
        'label': 'Window Batch 18 (2500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.001, K_true×1.001]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-10/batch_2525sims_20251110_033128.json',
        'label': 'Window Batch 19 (2525 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.001, K_true×1.001]'
    },
    
    # K-factor 1.0005
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-28/batch_1500sims_20250928_224858.json',
        'label': 'Window Batch 20 (1500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.0005, K_true×1.0005]'
    },
    
    # K-factor 1.00025
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2700sims_20251109_075908.json',
        'label': 'Window Batch 21 (2700 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.00025, K_true×1.00025]'
    },
    
    # K-factor 1.0001
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-08/batch_2700sims_20251108_191338.json',
        'label': 'Window Batch 22 (2700 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.0001, K_true×1.0001]'
    },
    
    # K-factor 1.00005
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2000sims_20250929_125706.json',
        'label': 'Window Batch 23 (2000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.00005, K_true×1.00005]'
    },
    
    # K-factor 1.000025
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2750sims_20251109_080611.json',
        'label': 'Window Batch 24 (2750 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.000025, K_true×1.000025]'
    },
    
    # K-factor 1.00001 - Narrowest bounds
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-08/batch_2750sims_20251108_191922.json',
        'label': 'Window Batch 25 (2750 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.00001, K_true×1.00001]'
    }
]


# Summary of K-range factors in decreasing order:
# 2.0      - Very wide bounds (original uniform sampling)
# 1.5      - Wide bounds
# 1.15     - Medium-wide bounds
# 1.05     - Medium bounds
# 1.025    - Medium-narrow bounds
# 1.01     - Narrow bounds
# 1.005    - Very narrow bounds
# 1.0025   - Ultra-narrow bounds
# 1.001    - Extremely narrow bounds
# 1.0005   - Super-narrow bounds
# 1.00025  - Hyper-narrow bounds
# 1.0001   - Ultra-tight bounds
# 1.00005  - Super-tight bounds
# 1.000025 - Hyper-tight bounds
# 1.00001  - Extremely tight bounds
