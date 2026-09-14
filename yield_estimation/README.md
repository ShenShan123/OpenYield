SRAM Yield Estimation Algorithm
=====
This directory implements various rare-event estimation algorithms using importance sampling techniques to evaluate SRAM failure probabilities under process variations, enabling accurate and efficient yield analysis.

V2.1.0 fixes all 15 compiler calls to consume `(delay, pavg, pstc, pdyn)` and treats NaN/infinite delays as failed samples. The supplied testbench keeps its lookup clock fixed across sampling. These adapter checks do not validate the legacy importance-sampling algorithms end to end; their machine-local paths and external ML dependencies remain separate work.

V2.1.1 uses distributed wiring throughout and rejects star configurations.
The fixed timing table and transistor classes remain unchanged. New physical
routes require fresh functional evidence before PVT/mismatch expansion; see
the [evaluation schedule](../docs/plans/V2_1_1_TIMING_FOLLOWUP.md). Historical
measurements and qualification records retain their original version and
physical context.

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
### 1. Run an Algorithm
<pre> python demo_run_a_testbench.py </pre>
Set `RUN_MODEL` at the top of `demo_run_a_testbench.py` (repository root) to select the algorithm; parameter settings are provided for circuits of different dimensionalities. The script uses the explicit custom process-parameter table mode of `Sram6TCoreMcTestbench` (`custom_mc=True`), not the compiler's default per-device mismatch. The former `main_estimation.py` targeted a removed package and testbench API and was deleted in V2.0.6.

Output
-----
Each algorithm's results will be saved as a CSV file; use these CSV outputs to generate visualization plots as needed.

Future Algorithm Extensions
-----
We will continue to add more state-of-the-art algorithms for yield estimation in the future, providing additional methods for testing and comparison.
