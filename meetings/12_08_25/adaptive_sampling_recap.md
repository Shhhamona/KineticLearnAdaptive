# Rate Coefficient Calculation - Sampling Method Comparasion and Adaptive Learning 

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
- Relu Activation
- Adam Optimizer
- learning_rate: 0.001
- n_epochs: 25
- batch_size: 64

Values obtained from hyperparameter exploration done by Marcelo's work.


---


### 1 - Neural Network Training - Different Sampling Methods


- Uniform Sampling -
- Log-Uniform Sampling
- Uniform Latin Hypercube Sampling
- Log-Uniform Latin Hypercube Sampling
- Morris Method (Continuous) Sampling
- Morris Method (Discreet) Sampling

Morris Method - From Discreet approach to Sampling Between Intervals


<img src="plots/adaptive_batch_sampling_20251209_194312.png" width="1000">


### 2 - Neural Network Training - Adaptive Sampling Method Review

Basic Idea: For a budget of N_samples instead of training always wihitin the same  K boundries - Iteratevily narrow boundries. Focus on central point being predicted

Parameters to control

- Number samples per iteration :
- Shrink Rate per Iteration : 



Example


## Training One Single Model to learn all K values:

##Example:

  Iteration 1

    1 - Determine the new sampling interval?

    Sampling interval: [k_toPredict/2, k_toPredict*2]

    2 - Sample 500 points from sampling interval

    The first iteration is the same as previous approaches. 

  Iteration 2

    1 - Determine the new sampling interval?

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

    2 - Sample 500 points from new sampling interval

 Iteraion 3, Iteraion 4, etc, etc
### Results

### All three Reactions

<img src="plots/adaptive_batch_sampling_mse_20251210_083051.png" width="1000">

### Shifted Center

K_center (original) = [6.0e-16, 1.3e-15, 9.6e-16]

K_shifted = [4.0e-16, 8.666e-16,1.44e-15]

<img src="plots/adaptive_batch_sampling_mse_20251210_083639.png" width="1000">

### Other Topics?

- Explore more hyperparameters - Maybe with grid
- 
- Train Neural Network with Uniform sampling with decreasing range



- Training NN Classifier for solving inverse problem for multiple K values 
    - Zone Based Adaptive Sampling


