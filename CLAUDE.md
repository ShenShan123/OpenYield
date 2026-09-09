# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

OpenYield generates 6T/10T SRAM transistor netlists with PySpice, simulates them with Xyce
(FreePDK45 models in `tran_models/`), and evaluates delay, power, SNM, area, driver sizing
and yield. `AGENTS.md` holds the detailed working conventions; `docs/DRIVER_SIZING_PROPOSAL.md`
is the live design/qualification document (update its sections in place, status at the top;
the original is archived under `docs/design/`). `docs/CHANGELOG.md` is per release (V2.0.x).

Current release: **V2.0.6**, integrating the default local mismatch package into
`sram_compiler/` and reorganizing documentation. The inherited sizing rule
identity and qualification artifact format remain V2.0.5; their versioned
paths in the commands below identify that evidence workflow.

## Environment

- Simulation needs the conda env: `source /proj/workarea/user5/miniconda3/etc/profile.d/conda.sh && conda activate openyield`
  (Python 3.9, PySpice 1.5, Xyce 7.4 MPI build at `/proj/workarea/user5/miniconda3/envs/openyield/bin/Xyce`,
  matching `mpiexec` beside it). The workspace `python3` (3.11) has PySpice/numpy/pandas/PyYAML and
  runs the unit tests and netlist generation, but `Xyce` is not on its PATH. Equivalent-cell modes
  (`real_cell_mode != 0`) run Xyce during netlist construction, so they need the env too.
- Code must stay Python 3.9 compatible (postponed annotations are used); compile with both interpreters.
- Pin threads for any Xyce run you launch in parallel: `OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`
  (one Xyce process otherwise spawns ~98 threads). The host is shared; check `uptime` before a campaign,
  and expect 1,800 s per-sample timeouts to bite when the load is high.
- Run everything from the repository root. Generated decks and results go under ignored `outputs/`, `sim/`,
  `sim1/`, or a temporary directory.

## Commands

```bash
# Simulator-free regression suite for the compiler, sizing and per-device paths (~10 s)
python3 -m unittest discover -s tests -v
# One test
python3 -m unittest tests.test_driver_paths.PathTests.test_matched_replica_and_canonical_read
# Offline optimizer package
python3 -m unittest discover -s size_optimization/openyield_v2/tests -v
# Static checks
python3 -m compileall -q sram_compiler utils tests size_optimization/exp_utils.py && git diff --check

# Generate (and optionally run) one deck; defaults: full array, per-device mismatch, 100 runs
python3 -m sram_compiler.per_device_mc.run --rows 8 --cols 4 --operation read --mc-runs 2 --seed 3 --run-xyce --output-dir outputs/per_device_mc

# Local development-tool tests (requires the ignored dev/ workspace)
python3 -m unittest discover -s dev/tests -v
```

`docs/DEVELOPMENT.md` contains the local qualification and reference-experiment
commands. Keep these scripts in ignored `dev/`; a fresh checkout must run the
compiler and tracked tests without them.

`utils/` holds shared measurement, waveform, plotting, area, and SPICE helpers.
Its package exports preserve `from utils import ...`. The hardcoded plotting
experiment is local-only in `dev/plot_data_demo.py`; comparison plots default
to `outputs/plots/`. Reusable utility tests stay tracked in `tests/`.

Never run repository-wide test discovery: several `test*.py` files under `size_optimization/` and
`yield_estimation/` are long experiments. Do not use `main_sram.py` as a smoke test; it rewrites
`sram_compiler/config_yaml/global.yaml` and `sram_6t_cell.yaml` in place. For an in-memory run use
`sram_compiler.per_device_mc.run.load_config(rows, cols, corner)`, set fields on `config.global_config`, build
`Sram6TCoreMcTestbench(...)` and call `run_mc_simulation(...)` with `sim_path` under `outputs/`.

## Architecture

Data flow: YAML in `sram_compiler/config_yaml/` -> `SRAM_CONFIG` (`sram_compiler/config_yaml/config.py`;
root `config.py` re-exports it) -> factories in `sram_compiler/testbenches/parameter_factor.py` ->
subcircuits in `sram_compiler/subcircuits/` (all derive from `BaseSubcircuit`, which owns the RC helper)
-> `Sram6TCoreTestbench` (array, replica column, decoder, wordline drivers, TIME block, column
periphery) -> `Sram6TCoreMcTestbench` (variation, stimuli, `.MEASURE`/`.PRINT`, Xyce execution,
result parsing) -> metrics consumed by `size_optimization/exp_utils.py` (optimizer objective) or
`yield_estimation/`. YAML widths/lengths are metres; PySpice unit objects inside circuits; sweep
mode substitutes SPICE expression strings, so test numeric and sweep paths together.

Timing model (`subcircuits/time_generate.py`, `TIME`): wordline enable and write enable occupy the
clock-low phase, precharge the clock-high phase, and sense enable is replica-timed (replica column
plus an odd-stage `DelayChain`). The clock period is the only free timing knob; the qualification
runner derives it from measured SS read and SS/SF write phases (`sizing/timing.py`).

Sizing layer (`sram_compiler/sizing/`): `resolve_driver_sizes()` returns an immutable `DriverSizes`
(precharge, split write-driver input/output, wordline inverter/NAND and decoder scales, TIME loads,
replica K/N, baseline fingerprints) once per baseline; both testbenches consume it and reject a
changed geometry, periphery or PDK. `sizing.mode: fixed` (default) reproduces the legacy netlists
byte for byte; `rules_only` opts into the V2.0.4/V2.0.5 rules; `auto` needs an exact record in
`sizing_table.json` (currently empty). Runtime `table.py` checks the tracked
`scoring_sources.json` manifest and runtime code hashes without reading ignored
scripts. Local `dev/sizing/` contains the qualification, campaign, report,
diagnostic, reference-experiment, and MPI execution tools; they verify their
source hashes against that manifest before producing qualification evidence.

Variation: `Sram6TCoreMcTestbench` modes are `nominal`, `shared` (one AGAUSS card per base model),
`custom` (parameter table from the cell YAML) and `per-device` (independent `vth0/u0/voff` per MOS,
specialized in `create_testbench()` by `sram_compiler/per_device_mc/netlist.py`). `mc=True` now means per-device,
so `mc_runs=1` without `mc_seed` is one unseeded random sample, not nominal; deterministic callers
must pass `variation_mode='nominal'`. Per-device cannot be combined with the legacy `.STEP` sweeps.

Parasitics: `w_rc` adds per-pin RC stubs inside each subcircuit (100 ohm / 1 fF) plus two segments on
driver outputs and periphery inputs; row and column nets between cells are ideal (a star, kept by
decision for the wordline sizing rule). The replica wordline/bitline must carry exactly the array's
RC configuration in every mode; `tests/test_driver_paths.py` enforces it.
`real_cell_mode` 0-4 replaces unused cells by the extracted 5-capacitor equivalent
(`subcircuits/sram_cell_add_equivalent.py`), which aggregates the same stubs.

## Project-specific rules

- Keep topology separate from sizing policy; resolve loads from actual scaled widths. Preserve
  positional factory/testbench arguments and append optional ones.
- A change to fixed mode must be intended: compare generated decks against the previous commit
  (detached `git worktree`) before claiming the default is unchanged.
- Generated decks or passing measures alone do not prove correctness; check waveforms (the
  qualification scorer or `.prn` crossings) and say exactly which validation ran.
- Preserve supplied evidence (`DRIVER_SIZING_data.csv`, `TIMING_AUTOCONFIG_data.csv`,
  `docs/qualification/*.json`). Never promote partial or failed qualification runs.
- Xyce specifics: `.SAMPLING useExpr=true` plus `.options samples numsamples=N seed=S` enable
  sampling (AGAUSS returns its mean without it); failed measures print `FAILED`
  (`MEASFAIL=1`); "Time step too small" gets one tighter-step retry in the runner; native MPI
  sampling crashes on large decks, hence the materialized cards.
- `pkill -f <pattern>` also matches the shell that runs it; kill by PID or from a script file.
- Commits use Conventional Commits (`fix(sram_compiler): ...`, `feat(sizing): ...`, `docs: ...`).
