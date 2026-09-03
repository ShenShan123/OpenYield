# Unified Yield Estimation: Usage and Interfaces

This document describes the common simulation entry point for OpenYield yield-estimation methods, the two supported algorithm invocation styles, charged-budget and simulator-error semantics, and the generated result artifacts.

## 1. Supported methods

The unified interface currently registers the following methods:

| Category | Method | Proposal rule in the unified implementation |
| --- | --- | --- |
| Stable | `MC` | Direct sampling from the target distribution |
| Stable | `MNIS` | Minimum-norm failure-point center |
| Stable | `AIS` | Mixture over failure points observed during the pilot stage |
| Stable | `ACS` | Farthest-point clustered centers from pilot failure points |
| Stable | `HSCS` | Clustering of standardized failure directions |
| Stable | `EFIAL` | Target-density-weighted mixture over failure points |
| Experimental | `FUSIS` | Mixture over high-target-density failure-boundary points |
| Experimental | `OPT` | One minimum-norm failure-point center |
| Experimental | `BIBD` | Budget-partitioned MC for multiple conditions; returns a multi-condition result |

`Stable` means that the public interface, budget ledger, and result format are supported. It does not guarantee that every finite-budget seed will pass the statistical consistency gate.

## 2. Native OpenYield simulation entry point

The underlying testbench retains the original OpenYield call shape:

```python
raw = testbench.run_mc_simulation(
    operation="read",
    target_row=0,
    target_col=0,
    mc_runs=100,
    temperature=27,
    vars=None,
)
```

- With `custom_mc=True`, `vars` must contain absolute physical process-parameter values, not standardized deviations.
- With `custom_mc=False`, normally pass `vars=None`; sampling is performed by `AGAUSS` in Xyce/model cards.
- The native testbench tuple return value is unchanged, so existing callers can continue to call it directly.
- Algorithms must not launch Xyce directly or manage `sim_path`, NaN handling, or retries. `SimulationRunner` owns those responsibilities.

Algorithms normally generate samples `z` in standard-normal coordinates. The runner applies

```text
physical = nominal + sigma * z
```

and passes the resulting absolute physical parameters to a testbench configured with `custom_mc=True`.

## 3. Recommended invocation: `YieldEstimator`

The following example estimates the read-delay failure probability for one 18-dimensional target SRAM cell:

```python
from pathlib import Path

import numpy as np

from yield_estimation import YieldEstimator

dimension = 18
nominal = np.asarray(process_nominal, dtype=float)  # shape: (18,)
sigma = np.abs(nominal) * 0.05

estimator = YieldEstimator(
    model=testbench,
    algorithm_choice="EFIAL",  # MC/MNIS/AIS/ACS/HSCS/EFIAL
    basic_params={
        "mean": np.zeros(dimension),
        "covariance": np.eye(dimension),
        "threshold": 6.8869e-11,
        "failure_direction": "greater",
        "seed": 20260903,
    },
    algo_params={
        "pilot_fraction": 0.4,
        "defensive_ratio": 0.1,
        "proposal_scale": 1.0,
        "max_components": 64,
        "batch_size": 40,
        # Treat a finite non-positive measurement from a converged Xyce run
        # as a functional failure.
        "failure_if_nonpositive": True,
    },
    spice_params={
        "run_root": Path("results/efial_seed20260903"),
        "operation": "read",
        "target_row": 3,
        "target_col": 1,
        "temperature": 27,
        "metric": "TREAD_TOTAL",
        "input_space": "standard_normal",
        "nominal": nominal,
        "sigma": sigma,
        "max_retries": 0,
        "quiet": True,
    },
)

result = estimator.run(max_num=5000)
print(result.failure_probability)
print(result.yield_probability)
print(result.standard_error)
print(result.status)
```

Use a new `run_root` for every run. The constructor refuses to overwrite an existing run directory, preventing concurrent methods from sharing or replacing simulation files.

## 4. Compatible invocation: `algorithm.start_estimate(...)`

To preserve the historical `start_estimate(max_num=...)` calling pattern, construct the runner and algorithm explicitly:

```python
from pathlib import Path

import numpy as np

from yield_estimation.unified import ACS, GaussianDistribution, SimulationRunner

runner = SimulationRunner(
    testbench,
    Path("results/acs_seed20260903/sim"),
    metric="TREAD_TOTAL",
    input_space="standard_normal",
    nominal=nominal,
    sigma=sigma,
    max_retries=0,
    quiet=True,
)

algorithm = ACS(
    runner=runner,
    distribution=GaussianDistribution(np.zeros(18), np.eye(18)),
    threshold=6.8869e-11,
    seed=20260903,
    failure_direction="greater",
    operation="read",
    target_row=3,
    target_col=1,
    temperature=27,
    pilot_fraction=0.4,
    defensive_ratio=0.1,
    proposal_scale=1.0,
    max_components=64,
    batch_size=40,
    failure_if_nonpositive=True,
)

result = algorithm.start_estimate(max_num=5000)
```

Calling an algorithm class directly returns the unified result object, but it does not create the facade-level `config.json`, `summary.csv`, `DONE`, or top-level `MANIFEST.sha256`. Use `YieldEstimator.run(...)` for formal runs.

## 5. Direct `SimulationRunner` invocation

The runner retains OpenYield parameter names but returns a structured `SimulationBatch`:

```python
from yield_estimation import SimulationRunner

runner = SimulationRunner(
    testbench,
    "results/manual/sim",
    metric="TREAD_TOTAL",
    input_space="standard_normal",
    nominal=nominal,
    sigma=sigma,
)
runner.reset_budget(100)

batch = runner.run_mc_simulation(
    operation="read",
    target_row=3,
    target_col=1,
    mc_runs=100,
    temperature=27,
    vars=z_samples,  # shape: (100, 18); standardized deviations at this layer
    run_name="manual_batch",
)

print(batch.values)
print(batch.statuses)
print(batch.errors)
```

The runner can parse:

- The native testbench four-tuple or legacy two-tuple return value.
- A CSV path.
- A legacy model exposing `.sample()`.
- A callable Python model.

When `input_space="physical"`, the runner performs no coordinate transform and `vars` must already contain absolute physical parameters.

## 6. Command-line invocation

Run a quick analytic smoke test:

```bash
python -m yield_estimation.validation \
  --backend analytic \
  --algorithm EFIAL \
  --budget 2000 \
  --seed 0 \
  --output results/analytic_smoke
```

Run a real Xyce test using a 4x2 testbench while varying only target cell `(3, 1)`, giving an 18-dimensional algorithm problem:

```bash
python -m yield_estimation.validation \
  --backend xyce \
  --algorithm ACS \
  --budget 5000 \
  --seed 20260903 \
  --output results/sram18_seed20260903 \
  --rows 4 \
  --cols 2 \
  --target-row 3 \
  --target-col 1 \
  --vary target \
  --threshold 6.8869e-11 \
  --std-ratio 0.05 \
  --clock-period-ns 5 \
  --pdk-path /absolute/path/to/models_TT_npe.spice
```

Run a 144-dimensional interface smoke test by varying the entire 4x2 array:

```bash
python -m yield_estimation.validation \
  --backend xyce \
  --algorithm MC \
  --budget 16 \
  --seed 20260903 \
  --output results/sram144_smoke \
  --rows 4 \
  --cols 2 \
  --vary all \
  --pdk-path /absolute/path/to/models_TT_npe.spice
```

An external job scheduler may launch multiple methods concurrently, but every method must use a distinct `--output` directory or method subdirectory. Parallel scheduling is not a yield-estimation algorithm and is not part of the algorithm source code.

## 7. Budget, errors, and status values

`max_num` is a strict charged budget:

- Every requested simulation point increments `charged_calls`.
- Retries increment both `charged_calls` and `retry_calls`.
- If the remaining budget is insufficient, the runner raises `BudgetExceeded` before launching the next simulation.
- Simulation exceptions, missing measurements, and NaN values are recorded as `simulator_error`; they are not physical failures.
- A converged Xyce run that returns a finite non-positive measurement retains simulation status `ok` and may be classified as a functional failure with `failure_if_nonpositive=True`.

Single-condition result status values are:

| Status | Meaning |
| --- | --- |
| `ok` | Budget completed and at least one failure was observed |
| `ok_zero_failure` | Budget completed, but no failure was observed |
| `simulator_failure` | The budget ledger contains one or more simulator errors |

`BIBD` returns a `MultiConditionEstimationResult`. Each condition retains an independent `EstimationResult`; the interface does not force them into one failure probability.

## 8. Result fields and artifacts

The primary `EstimationResult` fields are:

```text
algorithm
status
failure_probability
yield_probability
standard_error
budget_target
charged_calls
live_calls
retry_calls
simulator_errors
samples_used
elapsed_seconds
seed
metadata
artifacts
```

Each isolated directory created by `YieldEstimator.run(...)` contains:

```text
config.json
sim/**/samples.csv
result.json
summary.csv
DONE
MANIFEST.sha256
```

Each row in `samples.csv` records the algorithm-space `input_*` values, the absolute `physical_*` parameters passed to the testbench, the measured value, status, and error. Verify run-directory integrity with:

```bash
sha256sum -c MANIFEST.sha256
```

After multiple methods finish, aggregate their results with:

```bash
python -m yield_estimation.aggregate_validation --root results/validation_campaign
```

## 9. Notes on historical implementations

The historical MC, MNIS, AIS, ACS, and HSCS files under `yield_estimation/model_lib/` remain available for provenance and legacy-calling reference. Some of them contain hard-coded paths, modify `sim_path` directly, remove simulation directories, or call a testbench directly, so the unified algorithm layer does not invoke those implementations unchanged.

`YieldEstimator` uses the implementations in `yield_estimation/unified/estimators.py`, which follow the shared runner and strict budget ledger. The current unified ACS is a clustered defensive-IS implementation. Reproducing the iterative update and FOM behavior of the historical ACS would require refactoring that algorithmic logic to call only `SimulationRunner`; it must not restore the historical simulation-directory management.
