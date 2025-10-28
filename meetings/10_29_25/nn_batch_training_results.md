# Neural Network Batch Training Results

29_10_25

## Setup

Setup file:

- setup_O2_simple.in

Species:

- Species 1: O2(X)
- Species 2: O2(a)
- Species 3: O(3P)

Single K prediction:

- K1_true, K2_true, K3_true

Error:

- Mean Squared Error - MSE. Summed across 3 outputs and averaged accross 5 seeds

### Neural Network and Training Configuration

- hidden_layers: [64, 32]
- learning_rate: 0.001
- n_epochs: 25
- batch_size: 64

---

## 1. Uniform Sampling - Batch Training

Training dataset: 2000 samples

### Total MSE vs Training Samples
<img src="plots/batch_training_comparison_20251028_190612.png" width="1000">

### MSE per Output (K values)
<img src="plots/batch_training_per_output_20251028_190623.png" width="1000">

---

## 2. Uniform Sampling - Adaptive Batch Sampling

Training data: 6 pool files × 192 samples/file

### Configuration

- n_iterations: 6
- samples_per_iteration: 192
- n_epochs: 25
- batch_size: 64
- initial_window_size: 1.0
- shrink_rate: 0.10
- num_seeds: 5



### Pool Files - Shrinking K Boundaries

- Pool 1: K ∈ [K_true/2, K_true×2] - 4000 samples
- Pool 2: K ∈ [K_true/1.15, K_true×1.15] - 1000 samples
- Pool 3: K ∈ [K_true/1.15, K_true×1.15] - 2500 samples
- Pool 4: K ∈ [K_true/1.005, K_true×1.005] - 2000 samples
- Pool 5: K ∈ [K_true/1.0005, K_true×1.0005] - 1500 samples
- Pool 6: K ∈ [K_true/1.00005, K_true×1.00005] - 2000 samples

### Results
<img src="plots/adaptive_batch_sampling_20251028_190728.png" width="1000">


### Example Run - Seed 46

```
Center point: [0.89268728 0.48715398 0.62001899]

Iteration 0 - Untrained Model
  Test MSE: 1.253418e+00

Iteration 1 - Window size: 1.0000
  Pool 1: 192 samples (192/4000 used)
  Total samples: 192
  Training loss: 5.652868e-02
  Test MSE: 1.967060e-02

Iteration 2 - Window size: 0.1000
  Pool 1: 59 samples (247/4000 used)
  Pool 2: 133 samples (133/1000 used)
  Total samples: 384
  Training loss: 4.305872e-03
  Test MSE: 4.176914e-04

Iteration 3 - Window size: 0.0100
  Pool 2: 3 samples (134/1000 used)
  Pool 3: 6 samples (6/2500 used)
  Pool 4: 183 samples (183/2000 used)
  Total samples: 576
  Training loss: 1.549628e-05
  Test MSE: 6.078561e-08

Iteration 4 - Window size: 0.0010
  Pool 4: 86 samples (258/2000 used)
  Pool 5: 106 samples (106/1500 used)
  Total samples: 768
  Training loss: 2.860514e-07
  Test MSE: 1.877822e-10

Iteration 5 - Window size: 0.0001
  Pool 5: 66 samples (168/1500 used)
  Pool 6: 126 samples (126/2000 used)
  Total samples: 960
  Training loss: 1.976724e-09
  Test MSE: 2.528646e-12

Iteration 6 - Window size: 0.0000
  Pool 6: 75 samples (201/2000 used)
  Total samples: 1035
  Training loss: 7.138887e-11
  Test MSE: 4.213179e-13
```

---

## 3. Batch Training vs Adaptive Batch Sampling Comparison

<img src="plots/batch_vs_adaptive_comparison_20251028_193559.png" width="1000">

---

## 4. Sampling Strategies Comparison

Batch training comparison across different sampling strategies with K ∈ [K_true/2, K_true×2].

### Sampling Strategies - With K between [0.5 * K_true, 2 * K_true]:

- Uniform Sampling - Training Dataset
- Log-Uniform Sampling
- Uniform Latin Hypercube Sampling
- Morris Method (Continuous) Sampling

---

<img src="plots/sampling_strategies_batch_training_comparison_20251028_191107.png" width="1000">

