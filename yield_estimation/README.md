SRAM Yield Estimation Algorithm
=====
This directory implements various rare-event estimation algorithms using importance sampling techniques to evaluate SRAM failure probabilities under process variations, enabling accurate and efficient yield analysis.

Preparation File: spiced.py 
-------
The spice.py file defines threshold settings for yield estimation and establishes sampling boundary constraints across different circuit dimensions to guide the importance sampling process.and includes functions for defining yield criteria to guide the importance sampling process.

Algorithm
--------
### 1.Monte Carlo(MC)
File: MC.py 

Standard Monte Carlo draws samples directly from the original distribution, serving as an unbiased baseline for yield estimation.
- Direct SPICE-based pass/fail simulation
- No distribution modification or learning

Dependencies: Standard libraries (numpy, torch, gpytorch)
### 2. Mean-shifted IS(MNIS)
File: MNIS.py 

Shifts the sampling distribution toward the most probable failure boundary point to improve rare-event sampling focus.
- Computes minimal-norm failure-inducing point
- Focuses on single-mode failure boundaries
  
Dependencies: Standard libraries (numpy, torch, gpytorch)
### 3. Adaptive Compressed Sampling(ACS)
File: ACS.py 

Applies compressed sensing to construct sparse representations of failure regions, reducing reliance on full-distribution sampling.
- Uses L1-regularized recovery methods
- Exploits sparsity in failure patterns
- Best suited for smooth failure boundaries
  
Dependencies: Standard libraries (numpy, torch, gpytorch)
### 4. Adaptive IS(AIS)
File: AIS.py 

Refines the proposal distribution iteratively using cross-entropy minimization to adapt to unknown or complex failure structures.
- Learns sampling distribution from feedback
- Capable of capturing multiple failure modes
- Requires sampling + optimization in loop
  
Dependencies: Standard libraries (numpy, torch, gpytorch)
### 5. High-dimensional Sparse Compressed(HSCS)
File: HSCS.py 

Combines sparsity and compression strategies to model and sample failure modes in high-dimensional parameter spaces.
- Designed for full-array SRAM or large circuits
- Scales well with hundreds of variation parameters
- Incorporates hierarchical or block sparsity
  
Dependencies: Standard libraries (numpy, torch, gpytorch, sklearn.cluster)

Usage
---
### 1. Run All Algorithms
<pre> python demo_run_a_testbench.py </pre>
Run main_estimation.py to select and execute different algorithms within the file, with parameter settings provided for circuits of different dimensionalities.

Output
-----
Each algorithm's results will be saved as a CSV file; use these CSV outputs to generate visualization plots as needed.

Future Algorithm Extensions
-----
We will continue to add more state-of-the-art algorithms for yield estimation in the future, providing additional methods for testing and comparison.
