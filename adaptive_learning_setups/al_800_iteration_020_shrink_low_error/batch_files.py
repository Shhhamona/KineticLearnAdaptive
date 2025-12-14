# Window sampling batch files with K boundaries
BATCH_FILES = [
    # K-factor 2.0 - Widest bounds
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-08-27/batch_4000sims_20250827_010028.json',
        'label': 'Window Batch 1 (4000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/2, K_true×2]'
    },

    # K-factor 1.5. 2 files. 
    {
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-05/batch_800sims_20251205_184136.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-05/batch_800sims_20251205_210106.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-05/batch_800sims_20251205_184140.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-05/batch_800sims_20251205_184144.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-05/batch_800sims_20251205_184213.json'],
        'label': 'Window Batch 2 (4000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_pred/1.2, K_pred×1.2]'
    },

    # K-factor 1.16. 2 files. 
    {
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-07/batch_800sims_20251207_163935.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-07/batch_800sims_20251207_163959.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-07/batch_800sims_20251207_164100.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-07/batch_800sims_20251207_164125.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-07/batch_800sims_20251207_164140.json'],
        'label': 'Window Batch 3 (4000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_pred/1.04001, K_pred×1.04001]'
    },

    # K-factor 1.16. 2 files. 
    {
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-07/batch_800sims_20251207_185211.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-07/batch_800sims_20251207_185212.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-07/batch_800sims_20251207_185213.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-07/batch_800sims_20251207_185214.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-07/batch_800sims_20251207_185216.json'],
        'label': 'Window Batch 4 (4000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_pred/1.008001, K_pred×1.008001]'
    }
]
