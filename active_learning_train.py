"""
Active learning starter script.

This script performs the initial training step of the active learning workflow using
exactly the same preprocessing and SVR training used in the baseline code (first 500 samples).
It intentionally keeps the adaptive sampling loop out for now.
"""

from active_learning_methods import load_datasets, train_initial_models, run_mse_analysis, apply_training_scalers, retrain_models_with_new_data
import numpy as np
from sklearn.utils import shuffle
import argparse

from adaptive_sampling.src.batch_simulator import BatchSimulator
from adaptive_sampling.src.sampling_strategies import BoundsBasedSampler
from adaptive_sampling.src.base_simulator import MockSimulator, LoKISimulator


def get_scaled_k_center(k_true: np.ndarray, scaler_output) -> np.ndarray:
    """Return the k center scaled using the dataset output scaler.

    scaler_output is expected to be either None or a list of per-pressure scalers
    (as returned by LoadMultiPressureDatasetNumpy). We use the first pressure
    scaler to transform the k vector into the model's scaled space.
    """
    # NOTE: the training pipeline multiplies K by 1e30 before fitting the
    # output scaler (see LoadMultiPressureDatasetNumpy). To get the correct
    # scaled representation we must apply the same multiplier first.
    k_arr = np.array(k_true, dtype=np.float64).reshape(1, -1) * 1e30
    if scaler_output is None:
        return k_arr.ravel()

    scaler = scaler_output[0] if isinstance(scaler_output, (list, tuple)) else scaler_output
    try:
        scaled = scaler.transform(k_arr)
    except Exception:
        # If scaling fails for any reason, return the raw k_true as a fallback
        return k_arr.ravel()
    return scaled.ravel()


def make_k_bounds_around(k_true: np.ndarray, rel_width: float = 0.2, multiplicative_factor: float = None, min_frac: float = 1e-12) -> np.ndarray:
    """Create raw K-space bounds around k_true.

    Two modes supported:
      - Additive relative width (default): rel_width is a fraction, e.g. 0.2 -> [0.8*k, 1.2*k]
      - Multiplicative factor: if ``multiplicative_factor`` is provided, bounds are
        [k / multiplicative_factor, k * multiplicative_factor], e.g. factor=2.0 -> [0.5*k, 2.0*k].

    Returns an array shape (n_k, 2) where each row is [lower, upper] for that k.
    """
    k = np.array(k_true, dtype=np.float64).ravel()

    if multiplicative_factor is not None:
        lower = k / multiplicative_factor
        upper = k * multiplicative_factor
    else:
        lower = k * (1.0 - rel_width)
        upper = k * (1.0 + rel_width)

    # enforce positive lower bounds (can't sample negative rate constants)
    lower = np.maximum(lower, k * min_frac)

    bounds = np.vstack([lower, upper]).T
    return bounds


if __name__ == '__main__':
    # Parse CLI overrides for quick experiments
    parser = argparse.ArgumentParser(description='Active learning training runner')
    parser.add_argument('--n-samples-per-iteration', type=int, default=3000,
                        help='Number of samples to generate per adaptive iteration')
    parser.add_argument('--k-mult-factor', type=float, default=1.5,
                        help='Multiplicative factor for k-bounds (e.g. 2.0 => [k/2, k*2])')
    parser.add_argument('--loki-version', type=str, default='v2',
                        help='LoKI version to use: v2, v3, or custom path. Maps to predefined installation directories.')
    args = parser.parse_args()

    # Map LoKI version to installation path
    LOKI_PATHS = {
        'v2': 'C:\\MyPrograms\\LoKI_v3.1.0-v2',
        'v3': 'C:\\MyPrograms\\LoKI_v3.1.0-v3',
        'v4': 'C:\\MyPrograms\\LoKI_v3.1.0-v4',
        'v5': 'C:\\MyPrograms\\LoKI_v3.1.0-v5',
    }
    
    # If version is in the dictionary, use the mapped path; otherwise treat it as a custom path
    if args.loki_version in LOKI_PATHS:
        loki_path = LOKI_PATHS[args.loki_version]
    else:
        # Assume it's a custom path provided directly
        loki_path = args.loki_version

    # Configuration (same as baseline)
    config = {
        'nspecies': 3,
        'num_pressure_conditions': 2,
        'pressure_conditions_pa': [133.322, 1333.22],  # 1 and 10 Torr
        'initial_samples_from_uniform': 50,
        'n_iterations': 1,  # More iterations with smaller batches
        'n_samples_per_iteration': args.n_samples_per_iteration,  # Smaller, more targeted batches
        'k_multiplicative_factor': args.k_mult_factor,
        'loki_path': loki_path,
        'svr_params': [
            {'C': 10, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'},
            {'C': 20, 'epsilon': 0.005, 'gamma': 5, 'kernel': 'rbf'},
            {'C': 5, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'}
        ]
    }

    print(f"Configuration: {config}")

    nspecies = config['nspecies']
    num_pressure_conditions = config['num_pressure_conditions']
    best_params = config['svr_params']

    total_samples_needed = config['initial_samples_from_uniform'] + config['n_iterations'] * config['n_samples_per_iteration']

    # Data files (same paths used in baseline)
    src_file_train = 'data/SampleEfficiency/O2_simple_uniform.txt'
    src_file_test = 'data/SampleEfficiency/O2_simple_test.txt'

    print(f"📂 Loading training and test data")
    dataset_train, dataset_test = load_datasets(src_file_train, src_file_test, nspecies, num_pressure_conditions)

    print("✅ Data loaded")

    # PHASE 1: Initial MSE Analysis (exactly like baseline_test.py)
    print("\n🔬 PHASE 1: Initial MSE Analysis (100, 200, 300, 400, 500)")
    
    # Create subset sizes: 100, 200, 300, 400, 500, and final total_samples_needed
    base_sizes = list(range(100, config['initial_samples_from_uniform'] + 1, 100))  # [100, 200, 300, 400, 500]
    if total_samples_needed not in base_sizes:
        subset_sizes = base_sizes + [total_samples_needed]  # [100, 200, 300, 400, 500, 502]
    else:
        subset_sizes = base_sizes
    
    num_seeds = 1  # Start with 1 seed for debugging
    
    print(f"📊 Subset sizes: {subset_sizes}")
    print(f"🎲 Number of seeds: {num_seeds}")
    
    # Run MSE analysis (identical to baseline_test.py)
    mean_total_mse, std_total_mse = run_mse_analysis(dataset_train, dataset_test, best_params, subset_sizes, num_seeds)
    
    print(f"\n🏆 PHASE 1 RESULTS:")
    print(f"📊 Mean Total MSE: {mean_total_mse}")
    print(f"📊 Std Error: {std_total_mse}")
    
    # PHASE 2: Train final models on initial samples (ready for adaptive phase)
    print(f"\n🔬 PHASE 2: Training final models on {config['initial_samples_from_uniform']} samples")

    # Train on initial samples exactly like baseline
    models, mse_per_output, total_mse = train_initial_models(dataset_train, dataset_test, best_params,
                                                             n_initial_samples=config['initial_samples_from_uniform'],
                                                             seeds= [42])

    # Get the actual training data used for the initial models (500 samples)
    x_all, y_all = dataset_train.get_data()
    #x_shuf, y_shuf = shuffle(x_all, y_all, random_state=42)  # Same seed as train_initial_models
    x_shuf, y_shuf = x_all, y_all
    x_current_train = x_shuf[:config['initial_samples_from_uniform']]
    y_current_train = y_shuf[:config['initial_samples_from_uniform']]

    # DEBUG: Print raw and scaled data for initial training set
    print("\n==== DEBUG: Initial Training Data (First 5 Rows) ====")
    print("Raw compositions (X, first 5):\n", x_current_train[:100])
    print("Raw k values (Y, first 5):\n", y_current_train[:100])

    print(f"\n✅ Initial training complete")
    print(f"Total samples (initial): {config['initial_samples_from_uniform']}")
    print(f"Total MSE (initial): {total_mse:.6f}")
    print(f"Current training data shapes: X={x_current_train.shape}, Y={y_current_train.shape}")
    # Save snapshot of the training data used for the initial 500 samples
    try:
        import os
        out_dir = os.path.join('results', 'training_snapshots')
        os.makedirs(out_dir, exist_ok=True)
        x_path = os.path.join(out_dir, 'train_500_X.txt')
        y_path = os.path.join(out_dir, 'train_500_Y.txt')
        # Save arrays in a human-readable text format
        np.savetxt(x_path, x_current_train, fmt='%.6e')
        np.savetxt(y_path, y_current_train, fmt='%.6e')
        print(f"   💾 Saved initial training snapshot: {x_path}, {y_path}")
        
        # Also save test data for comparison
        x_test_all, y_test_all = dataset_test.get_data()
        x_test_path = os.path.join(out_dir, 'test_X.txt')
        y_test_path = os.path.join(out_dir, 'test_Y.txt')
        np.savetxt(x_test_path, x_test_all, fmt='%.6e')
        np.savetxt(y_test_path, y_test_all, fmt='%.6e')
        print(f"   💾 Saved test dataset snapshot: {x_test_path}, {y_test_path}")
        
    except Exception as e:
        print(f"   ⚠️ Could not save initial training snapshot: {e}")
    
    # Compare with baseline results
    baseline_500_mse = mean_total_mse[-2] if len(mean_total_mse) > 5 else mean_total_mse[-1]  # Size 500 result
    print(f"\n🔍 COMPARISON:")
    print(f"   Phase 1 MSE (size 500): {baseline_500_mse:.6f}")
    print(f"   Phase 2 MSE (size 500): {total_mse:.6f}")
    print(f"   Match: {'✅' if abs(baseline_500_mse - total_mse) < 0.001 else '❌'}")

    # Placeholder for adaptive sampling loop (to be implemented)
    print(f"\n-- PHASE 3: Adaptive sampling (not implemented)")
    print(f"   We now have trained models ready for adaptive sampling loop")
    print(f"   Next: Add {config['n_samples_per_iteration']} adaptive samples and retrain")
    

    # --- Quick scaling check using the provided k_true ---
    k_true = np.array([9.941885789401E-16, 1.800066252209E-15, 1.380839580124E-15])
    print('\n🔬 SCALING CHECK: Using k_true =', k_true)
    scaler_output = getattr(dataset_train, 'scaler_output', None)
    scaled_center = get_scaled_k_center(k_true, scaler_output)
    bounds = make_k_bounds_around(k_true, multiplicative_factor=1.0005)

    print('   Raw bounds (multiplicative factor 2.0):')
    for i, (lo, hi) in enumerate(bounds):
        print(f'     k[{i}]: {lo:.3e} -> {hi:.3e}')

    print('   Scaled center (first-pressure scaler):', scaled_center)
    # Also show manual scaling using scaler.scale_ when available (MaxAbsScaler)
    if scaler_output is not None:
        scaler0 = scaler_output[0] if isinstance(scaler_output, (list, tuple)) else scaler_output
        scale_attr = getattr(scaler0, 'scale_', None)
        if scale_attr is not None:
            # manual scaling must mirror the transform: first apply *1e30
            manual_scaled = (k_true * 1e30) / scale_attr
            print('   Scaled center (manual using scale_):', manual_scaled)
        else:
            print('   scaler.scale_ not available; transform already used')
    # Print dataset contents for verification
    print('\n🔎 DATASET CHECK:')
    try:
        x_all, y_all = dataset_train.get_data()
        print('   x shape:', x_all.shape)
        print('   y shape:', y_all.shape)
        print('   x (first 5 rows):')
        print(x_all[:5])
        print('   y (first 5 rows):')
        print(y_all[:5])
    except Exception as e:
        print('   Could not read dataset samples:', e)

    # Show scalers (first-pressure) for input/output
    scaler_in = getattr(dataset_train, 'scaler_input', None)
    scaler_out = getattr(dataset_train, 'scaler_output', None)
    print('   scaler_input (first pressure):', scaler_in[0] if scaler_in else None)
    print('   scaler_output (first pressure):', scaler_out[0] if scaler_out else None)

    print('\n✅ All steps completed successfully!')

    # --- PHASE 3: ADAPTIVE SAMPLING ITERATIONS ---
    print(f"\n🔬 PHASE 3: Adaptive Sampling ({config['n_iterations']} iterations)")
    
    # Initialize tracking variables
    current_x_train, current_y_train = dataset_train.get_data()
    # Use initial 500 samples for training
    #current_x_train, current_y_train = shuffle(current_x_train, current_y_train, random_state=42)
    current_x_train = current_x_train[:config['initial_samples_from_uniform']]
    current_y_train = current_y_train[:config['initial_samples_from_uniform']]
    
    # Track performance over iterations
    mse_history = [total_mse]  # Start with initial MSE
    sample_count_history = [config['initial_samples_from_uniform']]
    
    # Initialize BatchSimulator + sampler
    print('\n🔧 INITIALIZING BATCH SIMULATOR (mock)')
    try:
        # Use raw k_true (physical units) for sampler bounds
        k_true = np.array([6.00E-16, 1.30E-15, 9.60E-16])
        #k_true = np.array([9.941885789401E-16, 1.800066252209E-15, 1.380839580124E-15])
        # Use multiplicative factor from config/CLI to build k bounds
        k_mult = config.get('k_multiplicative_factor', 1.5)
        k_bounds = make_k_bounds_around(k_true, multiplicative_factor=k_mult)

        print('   K bounds for sampler:')
        for i, (lo, hi) in enumerate(k_bounds):
            print(f'     k[{i}]: {lo:.3e} -> {hi:.3e}')

        # Create a bounds-based sampler with these bounds using random sampling
        sampler = BoundsBasedSampler(k_bounds=k_bounds, sampling_method='random')

        # Try to create a real LoKI simulator; fall back to MockSimulator if unavailable
        try:
            loki_path = config.get('loki_path', 'C:\\MyPrograms\\LoKI_v3.1.0-v2')  # adapt if your LoKI is elsewhere
            k_columns = [0, 1, 2]
            loki_sim = LoKISimulator(setup_file='setup_O2_simple.in', chem_file='O2_simple_1.chem',
                                    loki_path=loki_path, k_columns=k_columns,
                                    simulation_type='simple', pressure_conditions=config['pressure_conditions_pa'])
            batch_sim = BatchSimulator(base_simulator=loki_sim, sampler=sampler)
            print('   Initialized real LoKISimulator (will use genFiles).')
        except Exception as e:
            print('   Could not initialize LoKISimulator, falling back to MockSimulator:', e)
            mock_sim = MockSimulator(setup_file='setup_O2_simple.in', chem_file='O2_simple_1.chem',
                                     loki_path='', true_k=k_true, pressure_conditions=config['pressure_conditions_pa'])
            batch_sim = BatchSimulator(base_simulator=mock_sim, sampler=sampler)

        # --- ITERATION LOOP ---
        for iteration in range(config['n_iterations']):
            print(f'\n🔄 ITERATION {iteration + 1}/{config["n_iterations"]}')
            print(f'   Current training size: {current_x_train.shape[0]}')
            
            # Generate a small set of parameter sets (do not run actual LoKI here)
            n_samples = config['n_samples_per_iteration']
            param_sets = batch_sim.generate_parameter_sets(n_samples=n_samples, k_bounds=k_bounds,
                                                           pressure_conditions=config['pressure_conditions_pa'])

            print(f'   Generated {len(param_sets)} parameter sets:')

            
            for i, ps in enumerate(param_sets):
                print(f'     PS {i}: k_values={ps.k_values}')
            # Run the mock simulations for these parameter sets
            print(f'   🚀 Running batch simulations...')
            batch_results = batch_sim.run_with_sampling(n_samples=n_samples, k_bounds=k_bounds,
                                                        pressure_conditions=config['pressure_conditions_pa'],
                                                        parallel_workers=1)

            print(f'   Batch Results: {batch_results.n_successful}/{batch_results.n_simulations} successful')
            
            # Save the batch results (automatic path generation like in tests)
            print(f'   💾 Saving batch results...')
            try:
                saved_path = batch_sim.save_batch_results(batch_results)
                print(f'   ✅ Batch results saved to: {saved_path}')
            except Exception as e:
                print(f'   ⚠️ Warning: Could not save batch results: {e}')
            
            if batch_results.compositions.size > 0 and batch_results.n_successful > 0:
                # Convert results to training format
                raw_k_values = np.array([ps.k_values for ps in batch_results.parameter_sets])
                #k_values = np.array([ps['k_values'] for ps in data['parameter_sets']])
                #raw_k_values = np.array(batch_results.k_values)  # Use the k_values from the batch results
                raw_compositions = batch_results.compositions

                # DEBUG: Print raw new batch data before scaling
                print("\n==== DEBUG!!!!!!!!!!!: New Batch Data BEFORE Scaling (First 5 Rows) ====")
                print("Raw compositions (first 5):\n", raw_compositions)
                print("Raw k values (first 5):\n", raw_k_values)
                
                new_x, new_y_scaled = apply_training_scalers(
                    raw_compositions=raw_compositions,
                    raw_k_values=raw_k_values,
                    dataset_train=dataset_train,
                    nspecies=nspecies,
                    num_pressure_conditions=num_pressure_conditions,
                    debug=True  # Reduce verbosity in loop
                )

                # DEBUG: Print scaled new batch data after scaling
                print("\n==== DEBUG !!!!!!!!!!!!!!!!: New Batch Data AFTER Scaling (First 5 Rows) ====")
                print("Scaled X (first 5):\n", new_x)
                print("Scaled Y (first 5):\n", new_y_scaled)
                
                # Retrain models with augmented dataset
                # retrain_models_with_new_data returns 6 values; unpack all and ignore the shuffled copies we don't need here
                new_models, new_mse_per_output, new_total_mse, augmented_size, _x_shuf, _y_shuf = retrain_models_with_new_data(
                    current_x_train=current_x_train,
                    current_y_train=current_y_train,
                    dataset_test=dataset_test,
                    new_x=new_x,
                    new_y_scaled=new_y_scaled,
                    best_params=best_params,
                    seeds= [42],
                    debug=True  # Reduce verbosity in loop
                )
                
                # Update current training data for next iteration
                current_x_train = np.vstack([current_x_train, new_x])
                current_y_train = np.vstack([current_y_train, new_y_scaled])
                
                # Track progress
                mse_history.append(new_total_mse)
                sample_count_history.append(augmented_size)
                
                print(f'   ✅ Iteration {iteration + 1} complete:')
                print(f'     Training size: {current_x_train.shape[0]}')
                print(f'     Total MSE: {new_total_mse:.6f}')
                print(f'     MSE change: {new_total_mse - mse_history[-2]:+.6f}')
                # If we've reached 600 samples (500 initial + 100 new), save a snapshot
                try:
                    if current_x_train.shape[0] >= 50:
                        import os
                        out_dir = os.path.join('results', 'training_snapshots')
                        os.makedirs(out_dir, exist_ok=True)
                        x_path = os.path.join(out_dir, 'train_600_X.txt')
                        y_path = os.path.join(out_dir, 'train_600_Y.txt')
                        np.savetxt(x_path, current_x_train[:600], fmt='%.6e')
                        np.savetxt(y_path, current_y_train[:600], fmt='%.6e')
                        print(f"   💾 Saved augmented training snapshot (600): {x_path}, {y_path}")
                except Exception as e:
                    print(f"   ⚠️ Could not save augmented training snapshot: {e}")
                
            else:
                print(f'   ❌ No successful simulations in iteration {iteration + 1}')

        # --- FINAL RESULTS SUMMARY ---
        print(f'\n🏆 ACTIVE LEARNING COMPLETE!')
        print(f'   📈 Training Evolution:')
        for i, (samples, mse) in enumerate(zip(sample_count_history, mse_history)):
            stage = "Initial" if i == 0 else f"Iter {i}"
            print(f'     {stage:>7}: {samples:>3} samples → MSE = {mse:.6f}')

        total_improvement = mse_history[-1] - mse_history[0]
        pct_improvement = (total_improvement / mse_history[0] * 100) if mse_history[0] > 0 else 0

        # Correctly compute added samples as final - initial
        initial_samples = sample_count_history[0]
        final_samples = sample_count_history[-1]
        added_samples = final_samples - initial_samples

        print(f'   📊 Overall Performance:')
        print(f'     Total samples added: {added_samples}')
        print(f'     Final training size: {final_samples}')
        print(f'     MSE improvement: {total_improvement:+.6f} ({pct_improvement:+.2f}%)')

        status = "✅ IMPROVED" if total_improvement < 0 else "⚠️ DEGRADED" if total_improvement > 0 else "➡️ NO CHANGE"
        print(f'     🎯 Result: {status}')

    except Exception as e:
        print('   Failed to initialize batch simulator or run iterations:', e)

    print('\n🎉 ACTIVE LEARNING PIPELINE COMPLETE!')
    
