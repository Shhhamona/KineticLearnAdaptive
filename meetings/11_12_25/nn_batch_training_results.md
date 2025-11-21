# Neural Network Adaptive Sampling Results

12_11_2025

## Setup

Setup file:

- setup_O2_simple.in

Species:

- Species 1: O2(X)
- Species 2: O2(a)
- Species 3: O(3P)

Single K prediction - Use NN to predict K values.

- K1_true, K2_true, K3_true

Error:

- Mean Squared Error - MSE. Summed accross the 3 ouptuts. Averaged accross 5 seeds

### Neural Network and Training Configuration

- hidden_layers: [64, 32]
- Relu Activation
- Adam Optimizer
- learning_rate: 0.001
- n_epochs: 50
- batch_size: 16

---

## 1. Adaptive Sampling Training 

Training dataset: 2000 samples

### Pool Files - Shrinking K Boundaries

- Pool 1: K ∈ [K_true/2, K_true×2] - 4000 samples
- Pool 2: K ∈ [K_true/1.15, K_true×1.15] - 1000 samples
- Pool 3: K ∈ [K_true/1.15, K_true×1.15] - 2500 samples
- Pool 4: K ∈ [K_true/1.005, K_true×1.005] - 2000 samples
- Pool 5: K ∈ [K_true/1.0005, K_true×1.0005] - 1500 samples
- Pool 6: K ∈ [K_true/1.00005, K_true×1.00005] - 2000 samples
- etc, etc, 


#### Batch 1 - K ∈ [K_true/2, K_true×2]
<img src="meeting_results/window_sampling/window_batch_1_(4000_samples)_-_uniform_sampling_k_distribution.png" width="1000">


#### Batch 2 - K ∈ [K_true/1.15, K_true×1.15]
<img src="meeting_results/window_sampling/window_batch_2_(1000_samples)_-_uniform_sampling_k_distribution.png" width="1000">


#### Batch 4 - K ∈ [K_true/1.005, K_true×1.005]
<img src="meeting_results/window_sampling/window_batch_4_(2000_samples)_-_uniform_sampling_k_distribution.png" width="1000">


#### Batch 5 - K ∈ [K_true/1.0005, K_true×1.0005]
<img src="meeting_results/window_sampling/window_batch_5_(1500_samples)_-_uniform_sampling_k_distribution.png" width="1000">
<img src="meeting_results/window_sampling/window_batch_5_(1500_samples)_-_uniform_sampling_chemistry_distribution.png" width="1000">


## Training One Single Model to learn all K values:

##Example:
  Iteration 1

  Uniform Sampling from first file.

  Iteration 2

    Updated center from model prediction: [0.50979334 0.5074194  0.51228845]
    Shink rate per iteration: 60 %
    Window size: 0.4000
    Sampling factor: 1.4
    Sampling Interval: [Center K / 1.4, Center K * 1. 4 ]

    Reaction 0:
            Center K: 5.097933e-01, True K: 5.000038e-01
            Sampling interval: [3.641381e-01, 7.137107e-01]

    Reaction 1:
            Center K: 5.074194e-01, True K: 5.003280e-01
            Sampling interval: [3.624424e-01, 7.103872e-01]

    Reaction 2:
            Center K: 5.122885e-01, True K: 5.000426e-01
            Sampling interval: [3.659203e-01, 7.172038e-01]

      DEBUG VALIDATION for Window Batch 1 (4000 samples) - Uniform Sampling:
        Pool K factor: 2.0
        Sampling factor: 1.4
        ✓ CHECK 1 PASSED: 2.0 >= 1.4
        CHECK 2: Verifying sampling interval within pool bounds
          Reaction 0:
            Center K: 5.097933e-01, True K: 5.000038e-01
            Sampling interval: [3.641381e-01, 7.137107e-01]
            Pool interval:     [2.500019e-01, 1.000008e+00]
            ✓ OK: Sampling within pool bounds
          Reaction 1:
            Center K: 5.074194e-01, True K: 5.003280e-01
            Sampling interval: [3.624424e-01, 7.103872e-01]
            Pool interval:     [2.501640e-01, 1.000656e+00]
            ✓ OK: Sampling within pool bounds
          Reaction 2:
            Center K: 5.122885e-01, True K: 5.000426e-01
            Sampling interval: [3.659203e-01, 7.172038e-01]
            Pool interval:     [2.500213e-01, 1.000085e+00]
            ✓ OK: Sampling within pool bounds
        ✓ CHECK 2 PASSED: All reactions within bounds

      DEBUG VALIDATION for Window Batch 2 (3000 samples) - Uniform Sampling:
        Pool K factor: 1.5
        Sampling factor: 1.4
        ✓ CHECK 1 PASSED: 1.5 >= 1.4
        CHECK 2: Verifying sampling interval within pool bounds
          Reaction 0:
            Center K: 5.097933e-01, True K: 5.000038e-01
            Sampling interval: [3.641381e-01, 7.137107e-01]
            Pool interval:     [3.333359e-01, 7.500057e-01]
            ✓ OK: Sampling within pool bounds
          Reaction 1:
            Center K: 5.074194e-01, True K: 5.003280e-01
            Sampling interval: [3.624424e-01, 7.103872e-01]
            Pool interval:     [3.335520e-01, 7.504920e-01]
            ✓ OK: Sampling within pool bounds
          Reaction 2:
            Center K: 5.122885e-01, True K: 5.000426e-01
            Sampling interval: [3.659203e-01, 7.172038e-01]
            Pool interval:     [3.333617e-01, 7.500639e-01]
            ✓ OK: Sampling within pool bounds
        ✓ CHECK 2 PASSED: All reactions within bounds


### All three Reactions
<img src="plots/adaptive_batch_sampling_20251112_163504.png" width="1000">

Issues :

- Very sensitive to hyperparameter tuning - Sampler per iteration and the Shrink Per Iteration rate
- Hard to get results for some of the parameters. Since I am runing the simulations before, some K values intervals are not matching.

Example:

      DEBUG VALIDATION for Window Batch 7 (500 samples) - Uniform Sampling:
        Pool K factor: 1.01
        Sampling factor: 1.01
        ✓ CHECK 1 PASSED: 1.01 >= 1.01
        CHECK 2: Verifying sampling interval within pool bounds
          Reaction 0:
            Center K: 4.975668e-01, True K: 5.000038e-01
            Sampling interval: [4.926404e-01, 5.025425e-01]
            Pool interval:     [4.950533e-01, 5.050039e-01]
            ✓ OK: Sampling within pool bounds
          Reaction 1:
            Center K: 5.081260e-01, True K: 5.003280e-01
            Sampling interval: [5.030950e-01, 5.132072e-01]
            Pool interval:     [4.953743e-01, 5.053313e-01]
            ❌ FAILS: Sampling extends beyond pool!
               Upper bound: 5.132072e-01 > 5.103846e-01


### Reaction 1
<img src="plots/mse_react0_20251112_151905.png" width="1000">

### Reaction 2
<img src="plots/mse_react1_20251112_151914.png" width="1000">

### Reaction 3
<img src="plots/mse_react2_20251112_151919.png" width="1000">

---

## Training One single Model to learn K valuess

## Previous Training - Why it was not Correct


Previously, the adaptive sampling was **not validating** whether the sampling window  fit within the available pool file boundaries. 

### Example of the Issue

- True K value: `K_true = 1.0 × 10⁻¹⁰`
- Current prediction: `K_pred = 0.5 × 10⁻¹⁰` (50% off)
- Sampling window size: `±100%` around prediction (factor of 2.0)
- Available pool file: K ∈ [K_true/1.01, K_true×1.01] 

**What Happened:**

The algorithm would try to sample from:
```
Window: [K_pred/2.0, K_pred×2.0] 
      = [0.25 × 10⁻¹⁰, 1.0 × 10⁻¹⁰]
```

But the pool file only contained:
```
Pool: [K_true/1.01, K_true×1.01]
    = [0.99 × 10⁻¹⁰, 1.01 × 10⁻¹⁰]
```

**Result:** The algorithm would try to sample from `[0.99 × 10⁻¹⁰, 1.0 × 10⁻¹⁰]`, which is much closer to the K_true than the original sample interval.



Problemas

Variar com o sample inicial
Verificar outra vez o error rate to loki
verificar o erro inicial e trainar com muito mais hyperparameters
Trainar com o  modelo mais complexo
