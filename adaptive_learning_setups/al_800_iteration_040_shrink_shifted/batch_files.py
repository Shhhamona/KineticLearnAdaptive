
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
    {
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-08/batch_800sims_20251208_182425.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-08/batch_800sims_20251208_182451.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-08/batch_800sims_20251208_182459.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-08/batch_800sims_20251208_182514.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-08/batch_800sims_20251208_182529.json'],
        'label': 'Window Batch 2 (3200 samples) - Uniform Sampling Shifted',
        'k_range': 'K ∈ [K_pred/1.8, K_pred×1.8]'
    },
        {
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-08/batch_800sims_20251208_231628.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-08/batch_800sims_20251208_231700.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-08/batch_800sims_20251208_231701.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-08/batch_800sims_20251208_231710.json'],
        'label': 'Window Batch 3 (3200 samples) - Uniform Sampling Shifted',
        'k_range': 'K ∈ [K_pred/1.32, K_pred×1.32]'
    },

            {
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-09/batch_800sims_20251209_220922.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-09/batch_800sims_20251209_220928.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-09/batch_800sims_20251209_221007.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-09/batch_800sims_20251209_221014.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-09/batch_800sims_20251209_221044.json'],
        'label': 'Window Batch 4 (4000 samples) - Uniform Sampling Shifted',
        'k_range': 'K ∈ [K_pred/1.12800001, K_pred×1.12800001]'
    },
                {
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-10/batch_800sims_20251210_011432.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-10/batch_800sims_20251210_011433.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-10/batch_800sims_20251210_011444.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-10/batch_800sims_20251210_011514.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-10/batch_800sims_20251210_011535.json'],
        'label': 'Window Batch 5 (4000 samples) - Uniform Sampling Shifted',
        'k_range': 'K ∈ [K_pred/1.051200001, K_pred×1.051200001]'
    },
]

