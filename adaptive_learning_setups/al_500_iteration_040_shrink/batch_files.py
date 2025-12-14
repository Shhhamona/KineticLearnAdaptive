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
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-24/batch_500sims_20251124_190004.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-24/batch_502sims_20251124_190023.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-24/batch_504sims_20251124_190049.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-24/batch_506sims_20251124_190107.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-24/batch_508sims_20251124_190126.json'],
        'label': 'Window Batch 2 (2500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_pred/1.4, K_pred×1.4]'
    },

    # K-factor 1.16. 2 files. 
    {
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-24/batch_500sims_20251124_211946.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-24/batch_502sims_20251124_212010.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-24/batch_504sims_20251124_212050.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-24/batch_506sims_20251124_212046.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-24/batch_508sims_20251124_212106.json'],
        'label': 'Window Batch 3 (2500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_pred/1.16001, K_pred×1.16001]'
    },

    # K-factor 1.16. 2 files. 
    {
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-24/batch_500sims_20251124_235253.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-24/batch_502sims_20251124_235323.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-24/batch_504sims_20251124_235338.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-24/batch_506sims_20251123_235411.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-24/batch_508sims_20251123_235353.json'],
        'label': 'Window Batch 4 (2500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_pred/1.064001, K_pred×1.064001]'
    },

            # K-factor 1.16. 2 files. 
    {
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-25/batch_500sims_20251125_210249.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-25/batch_502sims_20251125_210337.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-25/batch_504sims_20251125_210342.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-25/batch_506sims_20251125_210433.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-25/batch_508sims_20251125_210433.json'],
        'label': 'Window Batch 5 (2500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_pred/1.0256001., K_pred×1.0256001]'
    },


    {
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-26/batch_502sims_20251126_030235.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-26/batch_502sims_20251126_030235.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-26/batch_504sims_20251126_030254.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-26/batch_506sims_20251126_030341.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-26/batch_508sims_20251126_030348.json'],
        'label': 'Window Batch 6 (2500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_pred/1.01024001., K_pred×1.01024001.]'
    },

    {
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-05/batch_500sims_20251205_011047.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-05/batch_502sims_20251205_011014.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-05/batch_504sims_20251205_011051.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-05/batch_506sims_20251205_011231.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-05/batch_508sims_20251205_011159.json'],
        'label': 'Window Batch 7 (2500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_pred/1.004096., K_pred×1.004096.]'
    },

    {
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-05/batch_500sims_20251205_123413.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-05/batch_502sims_20251205_123434.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-05/batch_504sims_20251205_123459.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-05/batch_506sims_20251205_123521.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-12-05/batch_508sims_20251205_123538.json'],
        'label': 'Window Batch 8 (2500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_pred/1.00163840001., K_pred×1.00163840001.]'
    },
    
]
