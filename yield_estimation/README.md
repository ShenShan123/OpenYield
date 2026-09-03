SRAM Yield Estimation Algorithm
=====
This directory implements various rare-event estimation algorithms using importance sampling techniques to evaluate SRAM failure probabilities under process variations, enabling accurate and efficient yield analysis.

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
See [`USAGE.md`](USAGE.md) for the complete unified interface, per-method
commands, budget semantics, and result format.

### 1. Run All Algorithms
<pre> python demo_run_a_testbench.py </pre>
Run main_estimation.py to select and execute different algorithms within the file, with parameter settings provided for circuits of different dimensionalities.

Output
-----
Each algorithm's results will be saved as a CSV file; use these CSV outputs to generate visualization plots as needed.

Future Algorithm Extensions
-----
We will continue to add more state-of-the-art algorithms for yield estimation in the future, providing additional methods for testing and comparison.
## Unified estimator interface

New integrations should keep using the OpenYield simulation call shape:

```python
from yield_estimation import YieldEstimator

result = YieldEstimator(
    model=testbench,
    algorithm_choice="EFIAL",
    basic_params={
        "mean": process_mean,
        "covariance": process_covariance,
        "threshold": 6.8869e-11,
        "seed": 0,
    },
    algo_params={"pilot_fraction": 0.4, "defensive_ratio": 0.1},
    spice_params={
        "operation": "read",
        "target_row": 3,
        "target_col": 1,
        "temperature": 27,
    },
).run(max_num=5000)
```

The stable choices are `MC`, `MNIS`, `AIS`, `ACS`, `HSCS`, and `EFIAL`.
`FUSIS`, `OPT`, and the multi-condition `BIBD` interface are experimental.
The unified implementations retain distinct proposal rules: MNIS uses a
minimum-norm failure center, AIS uses the observed failure set, ACS uses
failure-region clustering, HSCS clusters standardized failure directions, and
EFIAL uses target-density-weighted failure components. All use the same paid
pilot plus defensive-IS accounting so their simulation budgets are comparable.

Algorithms call `SimulationRunner.run_mc_simulation(...)`, which mirrors the
native testbench signature. The runner allocates isolated directories, parses
the native return tuple, separates simulator errors from physical failures,
and enforces the charged simulation budget. The underlying testbench remains
directly callable and keeps its historical tuple return value.

With `custom_mc=True`, `vars` contains absolute PDK parameter values. To let an
algorithm operate in standard-normal coordinates, construct `SimulationRunner`
with `input_space="standard_normal"`, plus matching `nominal` and `sigma`
vectors. With `custom_mc=False`, sampling remains inside Xyce/model cards.

The reference validation CLI defaults to the 4x2 SRAM, target cell `(3, 1)`,
18-D target-cell variation, and the reference 5 ns clock period. A finite
non-positive `TREAD_TOTAL` is a converged functional read failure; a missing
measurement, NaN, or Xyce process error is a simulator failure and is excluded
from the physical failure estimate. Use `--vary all` for the 144-D interface
smoke test.

Every facade run writes `config.json`, per-batch `samples.csv`, `result.json`,
`summary.csv`, `DONE`, and `MANIFEST.sha256` under an isolated run directory.
Each sample row records both the algorithm-space input and the absolute
physical parameters passed to the testbench.

Validation jobs are independent and may run concurrently. For example:

```bash
python -m yield_estimation.validation --backend xyce --algorithm EFIAL \
  --budget 5000 --output /shared/validation_run
python -m yield_estimation.aggregate_validation --root /shared/validation_run
```

Use one output root per campaign and one method subdirectory per job. Never let
concurrent methods share a testbench `sim_path`.
