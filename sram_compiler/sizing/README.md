# V2.0.5 driver sizing

V2.0.5 resolves precharge, split write-driver, wordline and decoder output scales
with their TIME loads. The real and replica wordlines share a driver, and the
replica includes the real bitline's disabled write-stack load. The full design
and current qualification status are in
[`DRIVER_SIZING_PROPOSAL.md`](../../DRIVER_SIZING_PROPOSAL.md).

## Use

`global.yaml` defaults to `sizing.mode: fixed`. This retains the legacy numeric
array rules, including their small-array write limitation. To opt into the
proposal, set `mode: rules_only` in the YAML or configure it in memory:

```python
from per_device_mc.run import load_config
from sram_compiler.sizing import resolve_driver_sizes
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench

config = load_config(8, 4, "SF")
config.global_config.sizing = {"mode": "rules_only"}
sizes = resolve_driver_sizes(config, mux=False)

# Resolve before changing the baseline into a candidate cell.
config.sram_6t_cell.pmos_width.value *= 1.2
testbench = Sram6TCoreMcTestbench(
    config, corner="SF", choose_columnmux=False, real_cell_mode=0,
    driver_sizes=sizes, mc=False, sim_path="outputs/sizing_example",
)
deck = testbench.create_testbench("write", 7, 3)
metadata = sizes.to_dict()  # JSON-serializable baseline scales, loads and hashes.
```

`DriverSizes` and its nested `DriverLoads` are frozen dataclasses. Testbenches
resolve once at construction unless a result is supplied. Reuse allows changed
cell values and run PVT, but rejects changed geometry, cell type, mux, peripheral
parameters, or PDK contents. The fingerprint includes baseline cell values and
bounds, all peripheral parameters, all five PDK model files, architecture,
rule version, and sizing settings. A rule result is an unverified prediction.
`auto` consults `sizing_table.json` for an exact qualified baseline and physical
context (RC values and equivalent-cell mode); a miss remains `source='rule'`.
A valid match returns `source='table'` and the testbench applies its measured
clock. Table-qualified results reject reuse in a different physical context.
Candidate cells still have to pass their own checks with that frozen periphery.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `mode` | `fixed` | Legacy rules; `rules_only` selects new rules; `auto` consults qualified records |
| `parasitic_factor` | `1.0` | Multiplies proposed precharge/write load terms |
| `wd_floor_margin` | `1.5` | Multiplies the measured write-output box-cell floor of 1.0 |
| `k_w` | `0.0625` | Proposed write-output load coefficient per row |
| `pre_min` | `0.5` | Proposed precharge scale floor |
| `fixed_scales` | omitted | In fixed mode, optional overrides for `pre`, `wd_in`, `wd_out`, `wl_inv`, `wl_nand` |
| `replica` | `K: 1, N: 9` | Active cell count and odd delay-stage count; `matched` defaults true outside fixed mode |
| `scale_decoder` | true outside fixed mode | Scale only final decoder output inverters |
| `effort_buffers` | true outside fixed mode | Select even buffer-chain length from load effort |
| `canonical_read` | true outside fixed mode | Include disabled write-stack loading on real and replica bitlines |
| `table` | package `sizing_table.json` | Optional qualification table path |

The four rule coefficients are inactive in fixed mode. Unknown options,
nonpositive/nonfinite numbers, unsupported modes and invalid array geometry
fail explicitly. Mux fan-in is currently two and requires an even column count.
The 6T write floor is also used provisionally for 10T, as proposed.

Widths are still based on the circuit YAMLs in metres. Write M1–M4 use `wd_in`;
M5–M12 use `wd_out`. PRE and write-enable loads count these actual gate widths;
WL loads use the configured NAND widths and the matched replica input.
Fixed mode retains the old replica-load omission for comparisons.
Precharge/write sweeps now retain the resolved scale in SPICE expressions, as
wordline sweeps already did. Their previous unscaled sweep behavior is corrected
even in fixed mode. TIME buffer sizing remains based on the baseline loads;
peripheral sweeps are not qualified across their complete width ranges.

## Qualification

The MC testbench now defaults to `variation_mode='per-device'`: select a fixed
global PDK corner, then independently perturb `vth0`, `u0`, and `voff` on every
instantiated MOS with 5% relative Gaussian sigma. This includes array cells and
all read/write periphery. `NF` fingers share their MOS draw. There is no added
shared random process term, area scaling, or parameter correlation calibration.
`mc=False` / `variation_mode='nominal'` selects the corner alone;
`variation_mode='shared'` explicitly requests the previous shared-card model.
A single local sample is random; use nominal mode for deterministic extraction.
Custom process tables remain an explicit separate mode.

Specialization occurs in `create_testbench()`, so direct circuit exporters and
`run_mc_simulation()` use the same models. Add the analysis with `add_analysis()`
to enable stochastic sampling. Content-addressed model files and a CSV audit
retain each device's hierarchy and model identity across read/write exports.
The per-device CLI also defaults to `real_cell_mode=0`; equivalent modes remain
available explicitly and are labelled as partial device coverage. Geometric
sweep expressions are preserved. Local mismatch with the legacy `.STEP` sweep
flags raises an explicit error: a minimal Xyce 7.4 check executed only the first
geometry when `.STEP` and `.SAMPLING` were combined. Generate a separate local
deck for each geometry; deterministic sweeps remain available with `mc=False`.
The default sizing mode is still the unqualified legacy `fixed` mode.

Run the simulator-free regression suite from the repository root:

```bash
python3 -m unittest discover -s sram_compiler/tests -v
```

The tests cover rule crossovers, configured gate loads, mode overrides, invalid
inputs, immutable reuse and fingerprint changes, actual split MOS widths, sweep
expressions, TIME integration, and full-array 6T/10T read/write generation with
both mux choices. Equivalent-cell generation can invoke Xyce for extraction;
the fast suite deliberately uses full transistor arrays.

The resumable campaign derives a clock from nominal SS read and SS/SF write
phases, freezes it, then runs full-local variation, corner, hazard and RC
checks. A case's completion marker must match its exact deck/model/seed inputs
and simulator binary before simulation output is reused. Cached waveforms are
rescored on each run. Shared and nominal runs cannot qualify the local policy.
If Xyce aborts an individual transient with `Time step too small`, the runner
detects its missing waveform interval even when Xyce returns success. It retries
the same seeded ensemble once with a tighter 5 ps maximum step, preserving the
original attempt and recording the actual timestep. Failed electrical checks or
nonfinite signals do not trigger this retry.
RC calibration uses a longer initial clock because the historical fit excludes
the explicit 100-ohm/1-fF networks; its final clock still comes from measurement.

```bash
python3 -m sram_compiler.sizing.campaign --dry-run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m sram_compiler.sizing.campaign --sizes 8x4 --pilot --xyce /path/to/Xyce --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m sram_compiler.sizing.campaign --screen --xyce /path/to/Xyce --workers 24
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m sram_compiler.sizing.campaign --xyce /path/to/Xyce --workers 24
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m sram_compiler.sizing.offset --xyce /path/to/Xyce
python3 -m sram_compiler.sizing.report outputs/qualification/V2.0.5 --table sram_compiler/sizing/sizing_table.json
```

The independent gate-load and wide-gate reference experiments are reproducible:

```bash
python3 -m sram_compiler.sizing.gate_cap --xyce /path/to/Xyce
python3 -m sram_compiler.sizing.gate_fingers --xyce /path/to/Xyce
```

The fingering check holds total MOS widths fixed, compares transient edges at
NF=1/16/100, and verifies nearly unchanged DC current. Its reference metadata
includes the model and simulator binary hashes.

Use `--sizes 8x4 --pilot` on the campaign for a small check. Full-array Monte
Carlo is expensive; the timeout defaults to six hours per sample (60 hours for
a ten-sample deck), and incomplete runs remain failures. Process counts change
scheduling, not circuit tolerances.
`--screen` includes PVT, sequences and RC, postponing the dedicated 100-sample
ensembles. Both pilot and screening runs are ineligible for table promotion.
The full schedule has 112 configurations, 336 calibration decks, 1,682 local
verification decks and 17,156 waveform samples. Review failures before expanding
the schedule. A root lock prevents two campaigns writing the same checkpoint.
The report refuses incomplete campaigns and requires both local sensing and
SS/SF write-tail evidence before promotion. The table remains empty.

Large arrays (at least 1,024 cells) use four MPI ranks by default. Set
`--mpi-ranks` and `--parallel-min-cells` to adjust this; `workers × mpi-ranks`
must fit the available cores. Use the MPI launcher beside the Xyce binary,
with one BLAS thread per rank. Xyce 7.4's native random-expression sampling
crashed on the large MPI test, so these runs sample local Gaussian LHS values
before execution and save numeric cards for each transient. Each parameter's
stream is stable across candidate widths and unrelated device insertions.
Serial small-array checks use native Xyce sampling. Both backends record their
provenance; saved MPI cards support exact reproduction on one or more cores.
The four-core 64x64 execution check passes. The three-sample rule screen finished
on 2026-09-08; its outcome (36 RC wordline-budget failures, timeouts and three
DCOP aborts on materialized MPI samples) is in `DRIVER_SIZING_PROPOSAL.md`.

For a bounded representative-array diagnostic before the full campaign:

```bash
python3 -m sram_compiler.sizing.local_review --workers 20 --mpi-ranks 4 --samples 3 --timeout 1800 --xyce /path/to/openyield/bin/Xyce
python3 -m sram_compiler.sizing.local_review --summarize
```

Half-select waveform qualification remains a separate open requirement and
blocks table promotion, even if all currently scheduled cases pass.

The default replica remains `(1, 9)`. Original YAML 200/100 ps access limits are
reported separately; passing the phase-based specification does not waive them.
`exp_utils.py` now reports these constraint violations, freezes baseline sizing,
and uses actual resolved YAML widths for area. `per_device_mc/run.py` records the
same baseline across variation samples. The obsolete rare-event algorithm entry
point `main_estimation.py` still requires a separate API/backend migration.
