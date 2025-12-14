
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
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-21/batch_800sims_20251121_153851.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-21/batch_800sims_20251121_153853.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-21/batch_800sims_20251121_153859.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-21/batch_800sims_20251121_153925.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-21/batch_800sims_20251121_154002.json'],
        'label': 'Window Batch 2 (4000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_pred/1.4, K_pred×1.4]'
    },

    # K-factor 1.16. 2 files. 
    {
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-21/batch_800sims_20251121_190228.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-21/batch_800sims_20251121_190243.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-21/batch_800sims_20251121_190258.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-21/batch_800sims_20251121_190315.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-21/batch_800sims_20251121_190352.json'],
        'label': 'Window Batch 3 (4000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_pred/1.16001, K_pred×1.16001]'
    },

    # K-factor 1.16. 2 files. 
    {
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-21/batch_800sims_20251121_214759.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-21/batch_800sims_20251121_214800.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-21/batch_800sims_20251121_214821.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-21/batch_800sims_20251121_214828.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-21/batch_800sims_20251121_214844.json'],
        'label': 'Window Batch 4 (4000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_pred/1.064001, K_pred×1.064001]'
    },

            # K-factor 1.16. 2 files. 
    {
        'path': ['results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-22/batch_800sims_20251122_122200.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-22/batch_800sims_20251122_122218.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-22/batch_800sims_20251122_122221.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-22/batch_800sims_20251122_122241.json',
                    'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-22/batch_800sims_20251122_122241.json'],
        'label': 'Window Batch 5 (4000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_pred/1.0256001., K_pred×1.0256001]'
    },
    
    
    
    # K-factor 1.5
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-08/batch_3000sims_20251108_061037.json',
        'label': 'Window Batch 3 (3000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.5, K_true×1.5]'
    },
    
    # K-factor 1.15
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-28/batch_1000sims_20250928_191628.json',
        'label': 'Window Batch 4 (1000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.15, K_true×1.15]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2500sims_20250929_031845.json',
        'label': 'Window Batch 5 (2500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.15, K_true×1.15]'
    },
    
    # K-factor 1.05
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-08/batch_2500sims_20251108_184222.json',
        'label': 'Window Batch 6 (2500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.05, K_true×1.05]'
    },
    
    # K-factor 1.025
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2500sims_20251109_072738.json',
        'label': 'Window Batch 7 (2500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.025, K_true×1.025]'
    },
    
    # K-factor 1.01
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-10-27/batch_500sims_20251027_154921.json',
        'label': 'Window Batch 8 (500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.01, K_true×1.01]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-07/batch_2000sims_20251107_204934.json',
        'label': 'Window Batch 9 (2000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.01, K_true×1.01]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2500sims_20251109_195233.json',
        'label': 'Window Batch 10 (2500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.01, K_true×1.01]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2525sims_20251109_195849.json',
        'label': 'Window Batch 11 (2525 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.01, K_true×1.01]'
    },
    
    # K-factor 1.0075
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2550sims_20251109_200200.json',
        'label': 'Window Batch 12 (2550 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.0075, K_true×1.0075]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2575sims_20251109_200551.json',
        'label': 'Window Batch 13 (2575 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.0075, K_true×1.0075]'
    },
    
    # K-factor 1.005
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2000sims_20250929_205429.json',
        'label': 'Window Batch 14 (2000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.005, K_true×1.005]'
    },
    
    # K-factor 1.0025
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2600sims_20251109_074357.json',
        'label': 'Window Batch 15 (2600 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.0025, K_true×1.0025]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-10/batch_2550sims_20251110_033347.json',
        'label': 'Window Batch 16 (2550 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.0025, K_true×1.0025]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-10/batch_2575sims_20251110_033748.json',
        'label': 'Window Batch 17 (2575 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.0025, K_true×1.0025]'
    },
    
    # K-factor 1.001
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-08/batch_2600sims_20251108_185805.json',
        'label': 'Window Batch 18 (2600 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.001, K_true×1.001]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-10/batch_2500sims_20251110_032616.json',
        'label': 'Window Batch 19 (2500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.001, K_true×1.001]'
    },
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-10/batch_2525sims_20251110_033128.json',
        'label': 'Window Batch 20 (2525 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.001, K_true×1.001]'
    },
    
    # K-factor 1.0005
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-28/batch_1500sims_20250928_224858.json',
        'label': 'Window Batch 21 (1500 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.0005, K_true×1.0005]'
    },
    
    # K-factor 1.00025
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2700sims_20251109_075908.json',
        'label': 'Window Batch 22 (2700 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.00025, K_true×1.00025]'
    },
    
    # K-factor 1.0001
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-08/batch_2700sims_20251108_191338.json',
        'label': 'Window Batch 23 (2700 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.0001, K_true×1.0001]'
    },
    
    # K-factor 1.00005
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-09-29/batch_2000sims_20250929_125706.json',
        'label': 'Window Batch 24 (2000 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.00005, K_true×1.00005]'
    },
    
    # K-factor 1.000025
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-09/batch_2750sims_20251109_080611.json',
        'label': 'Window Batch 25 (2750 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.000025, K_true×1.000025]'
    },
    
    # K-factor 1.00001 - Narrowest bounds
    {
        'path': 'results/batch_simulations/lokisimulator/boundsbasedsampler/2025-11-08/batch_2750sims_20251108_191922.json',
        'label': 'Window Batch 26 (2750 samples) - Uniform Sampling',
        'k_range': 'K ∈ [K_true/1.00001, K_true×1.00001]'
    }
]