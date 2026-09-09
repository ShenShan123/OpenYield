# V2.0.6 driver sizing

V2.0.6 retains the V2.0.5 sizing rules, which resolve precharge, split
write-driver, wordline and decoder output scales
with their TIME loads. The real and replica wordlines share a driver, and the
replica includes the real bitline's disabled write-stack load. The full design
and current qualification status are in
[`docs/DRIVER_SIZING_PROPOSAL.md`](../../docs/DRIVER_SIZING_PROPOSAL.md).

The local mismatch package now lives in `sram_compiler/per_device_mc/`.
The active rule identity remains `v2.0.5-local-1`; local qualification tools in
ignored `dev/` retain their V2.0.5 artifact format and default output paths. The V2.0.6 package move
does not establish new electrical qualification.

## Use

`global.yaml` defaults to `sizing.mode: fixed`. This retains the legacy numeric
array rules, including their small-array write limitation. To opt into the
proposal, set `mode: rules_only` in the YAML or configure it in memory:

```python
from sram_compiler.per_device_mc.run import load_config
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
Fixed mode retains the old replica-load omission for comparisons. With `w_rc`,
the replica wordline driver and the replica bitline's sense input carry the same
RC segments as the real wordline drivers and sense-amplifier inputs in every mode.
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
python3 -m unittest discover -s tests -v
```

The tests cover rule crossovers, configured gate loads, mode overrides, invalid
inputs, immutable reuse and fingerprint changes, actual split MOS widths, sweep
expressions, TIME integration, and full-array 6T/10T read/write generation with
both mux choices. Equivalent-cell generation can invoke Xyce for extraction;
the fast suite deliberately uses full transistor arrays.

Runtime qualification lookup uses `scoring_sources.json`, which pins the
reviewed local scoring sources by content hash. It hashes that manifest together
with the runtime acceptance inputs, and rejects stale or incomplete records.
The compiler never opens files from ignored `dev/`; local qualification runners
verify their source hashes before recording or promoting evidence. Changes to
the scoring manifest invalidate earlier scoring identities.

Development experiments, campaign runners, and their tests are local-only under
ignored `dev/`. See the [development guide](../../docs/DEVELOPMENT.md) for their
inventory, commands, and source-fingerprint maintenance.

Half-select waveform qualification remains a separate open requirement and
blocks table promotion, even if all currently scheduled cases pass.

The default replica remains `(1, 9)`. Original YAML 200/100 ps access limits are
reported separately; passing the phase-based specification does not waive them.
`exp_utils.py` now reports these constraint violations, freezes baseline sizing,
and uses actual resolved YAML widths for area. `sram_compiler/per_device_mc/run.py` and `main_sram.py` record the
same baseline across variation samples. The obsolete rare-event entry point
`main_estimation.py` (a removed API and package) was deleted on 2026-09-09;
`demo_run_a_testbench.py` is the yield-estimation entrance and uses explicit
custom tables.
