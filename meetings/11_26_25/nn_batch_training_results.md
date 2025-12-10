# Neural Network Adaptive Sampling Results

26_11_2025

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

Training dataset: 300 to 3500 samples

## Training One Single Model to learn all K values

## Example

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

<img src="plots/adaptive_batch_sampling_mse_20251126_171712.png" width="1000">
<img src="plots/adaptive_batch_sampling_relative_20251126_171712.png" width="1000">

Next Steps :

- Training with more hyperparameters
- Train with bigger precision. Chemical Composition error rate is 0.1%. Should we study what is the equivilant error rate in the K space?
- Use more complex simulations - For this I might need the servers
-
