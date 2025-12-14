
# Window sampling batch files with K boundaries
BATCH_FILES = [
    # K-factor 2.0 - Widest bounds

    # K-factor 1.5. 2 files. 
    {
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-07/batch_800sims_20251207_212007.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-07/batch_800sims_20251207_212008.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-07/batch_800sims_20251207_212018.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-07/batch_800sims_20251207_212041.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-07/batch_800sims_20251207_212051.json'],
        'label': 'Window Batch 1 (4000 samples) - Uniform Sampling Shifted',
        'k_range': 'K ∈ [K_pred/3, K_pred×3]'
    },
]