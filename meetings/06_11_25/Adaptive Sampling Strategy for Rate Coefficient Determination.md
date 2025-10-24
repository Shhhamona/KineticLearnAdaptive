**Adaptive Sampling Strategy for Rate Coefficient Determination**

**1\. Introduction**

This document outlines a simplified strategy for efficiently determining chemical reaction rate coefficients (K) using a regression model and simulations (e.g., LoKI). We aim to improve upon existing methods that predict K from chemical composition data (C) by minimizing the number of computationally expensive simulations required. This is achieved by iteratively refining estimates of K by sampling in the neighborhood of previous estimates.

**2\. Problem Definition**

* **Inverse Problem:** We want to use a regression model to learn the inverse mapping: C \-\> K  
* **LoKI Simulations:** LoKI simulations provide the forward mapping: K \-\> C  
* Simulations are computationally expensive.  
* There is a single, true value for each K, and we take advantage of this when choosing our sampling strategy.

**3\. Methodology**

The core idea is to iteratively:

* Start with the "true" K values (K') obtained from literature. Generate the corresponding chemical composition C'.  
* Generate training data by sampling K values in a hypercube around K' and running LoKI simulations.  
* Train a regression model on this data to predict K from C.  
* Use the model to predict K for the "true" chemical composition (C').  
* Refine the sampling hypercube around the newly predicted K.  
* Repeat steps 2-5 until convergence.

**4\. Detailed Algorithm**

**4.1. Initialization**

* Obtain K\_true (K'): Start with the "true" K values (K') obtained from literature. (podemos comecar com um valor aleatorio, em vez do true K)  
* Generate C': Run a LoKI simulation using K' to obtain the corresponding chemical composition C'.  
* Define Initial Hypercube: Define a hypercube in K-space around K' for initial sampling.  
* Initial Sampling: Generate a set of K values (K\_initial) by sampling within the defined hypercube using Latin Hypercube Sampling, Random Sampling, and/or the Morris method.  
* Run LoKI Simulations: Run LoKI simulations for each K in K\_initial to obtain corresponding chemical compositions (C\_initial).  
* Initial Dataset: Create the initial training dataset: D\_initial \= {(K1, C1), (K2, C2), ..., (KN\_initial, CN\_initial)}  
* Train Regression Model: Train a regression model on D\_initial to predict K from C.

**4.2. Iterative Refinement Loop**

* Repeat until convergence:  
  * Prediction: Use the regression model to predict K\_pred for the target composition C'.  
  * Define New Hypercube: Define a new, smaller hypercube around K\_pred.  The size of the hypercube should decrease with each iteration.  
  * New Sampling: Generate a set of K values (K\_new) within the new hypercube using Latin Hypercube Sampling, Random Sampling, and/or the Morris method.  
  * Run LoKI Simulations: Run LoKI simulations for each K in K\_new to obtain corresponding chemical compositions (C\_new).  
  * Data Augmentation: Add the new simulation results to the training dataset.  
  * Model Retraining: Retrain the regression model.

**5\. Expected Benefits**

* Fewer simulations required compared to random sampling.  
* Improved accuracy by iteratively refining estimates of K.  
* Demonstrably lower RMSE compared to models created without the adaptive sampling strategy.