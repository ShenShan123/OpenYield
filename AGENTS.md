# OpenYield project instructions

Current release: **V2.0.7** (RC configuration corrections and distributed interconnect).
The sizing rule and qualification artifact identities remain V2.0.5; preserve historical version labels when referring to measurements or archived proposals. `docs/CHANGELOG.md` is per release (V2.0.x). `AGENTS.md` holds the detailed working conventions;

## Purpose and architecture

* Data flow: YAML in `sram_compiler/config_yaml/` -> `SRAM_CONFIG` (`sram_compiler/config_yaml/config.py`; root `config.py` re-exports it) -> factories in `sram_compiler/testbenches/parameter_factor.py` -> subcircuits in `sram_compiler/subcircuits/` (all derive from `BaseSubcircuit`, which owns the RC helper)
  -> `Sram6TCoreTestbench` (array, replica column, decoder, wordline drivers, TIME block, column periphery) -> `Sram6TCoreMcTestbench` (variation, stimuli, `.MEASURE`/`.PRINT`, Xyce execution, result parsing) -> metrics consumed by `size_optimization/exp_utils.py` (optimizer objective) or `yield_estimation/`.
* YAML widths/lengths are metres; PySpice unit objects inside circuits; sweep
  mode substitutes SPICE expression strings, so test numeric and sweep paths together.
* Timing model (`subcircuits/time_generate.py`, `TIME`): wordline enable and write enable occupy the clock-low phase, precharge the clock-high phase, and sense enable is replica-timed (replica column plus an odd-stage `DelayChain`). The clock period is the only free timing knob; the qualification runner derives it from measured SS read and SS/SF write phases (`sizing/timing.py`).
* Sizing layer (`sram_compiler/sizing/`): `resolve_driver_sizes()` returns an immutable `DriverSizes`(precharge, split write-driver input/output, wordline inverter/NAND and decoder scales, TIME loads, replica K/N, baseline fingerprints) once per baseline; both testbenches consume it and reject a changed geometry, periphery or PDK. `sizing.mode: fixed` (default) retains legacy driver scales; V2.0.7 intentionally corrects the inconsistent local RC defaults (see changelog); `rules_only` opts into the V2.0.4/V2.0.5 rules; `auto` needs an exact record in `sizing_table.json` (currently empty). Runtime `table.py` checks the tracked`scoring_sources.json` manifest and runtime code hashes without reading ignored scripts. Local `dev/sizing/` contains the qualification, campaign, report, diagnostic, reference-experiment, and MPI execution tools; they verify their source hashes against that manifest before producing qualification evidence.
* Variation: `Sram6TCoreMcTestbench` modes are `nominal`, `shared` (one AGAUSS card per base model), `custom` (parameter table from the cell YAML) and `per-device` (independent `vth0/u0/voff` per MOS, specialized in `create_testbench()` by `sram_compiler/per_device_mc/netlist.py`). `mc=True` now means per-device,
* so `mc_runs=1` without `mc_seed` is one unseeded random sample, not nominal; deterministic callers must pass `variation_mode='nominal'`. Per-device cannot be combined with the legacy `.STEP` sweeps.
* Parasitics: `w_rc` controls local storage-node and peripheral stubs (100 ohm / 1 fF by default); custom values propagate through all factories and nested cells. `interconnect.mode: star` remains the default shared-net topology. Opt-in `distributed` adds geometry-based pi ladders with centered cell taps on WL/BL/BLB, independent of `w_rc`; its `cell_pin_rc` defaults to false to avoid duplicating generic cell-pin wire loads. Replica wires match array lengths and loads, and TIME observes the far replica wordline before precharge. Equivalent modes 1-4 retain every wire segment and attach omitted-cell loads locally; extraction is numeric-only and uses the effective PVT and model-content cache identity. See `docs/design/DISTRIBUTED_RC_MODEL.md`.

## Working conventions

- Make sure the SRAM functionality totally CORRECT first, including write, read, hold opertations. Then solve the driver sizings, transistor sizing optimizations, and yield estimations.
- Run commands from the repository root. Resolve data paths from source-file
  locations, not the caller's working directory; avoid new machine-local paths.
- Append optional arguments. Do not reformat unrelated legacy code or comments.
- YAML widths/lengths use SI metres; PySpice uses unit objects, and parameter
  sweeps use SPICE expression strings. Test numeric and sweep paths together.
- A baseline sizing result must remain fixed across cell candidates and PVT
  samples. Changed architecture/peripheral inputs must not silently reuse it.
- Use per-device mismatch as default in MC simulations; equivalent cells are
  approximations. Generated decks or passing measures alone do not prove
  waveform correctness, retention, sensing margin, or timing qualification.
- Keep generated decks/results under ignored `outputs/` or a temporary path.
  Keep ad hoc development scripts under ignored `dev/`, outside `sram_compiler/`.
  Preserve supplied CSV evidence.
- Keep topology separate from sizing policy; resolve loads from actual scaled widths. Preserve
  positional factory/testbench arguments and append optional ones.
- A change to fixed mode must be intended: compare generated decks against the previous commit (detached `git worktree`) before claiming the default is unchanged.
- Generated decks or passing measures alone do not prove correctness; check waveforms (the qualification scorer or `.prn` crossings) and say exactly which validation ran.
- Preserve supplied evidence (`DRIVER_SIZING_data.csv`, `TIMING_AUTOCONFIG_data.csv`,
  `docs/qualification/*.json`). Never promote partial or failed qualification runs.
- Xyce specifics: `.SAMPLING useExpr=true` plus `.options samples numsamples=N seed=S` enable sampling (AGAUSS returns its mean without it); failed measures print `FAILED`
  (`MEASFAIL=1`); "Time step too small" gets one tighter-step retry in the runner; native MPI
  sampling crashes on large decks, hence the materialized cards.
- `pkill -f <pattern>` also matches the shell that runs it; kill by PID or from a script file.
- Commits use Conventional Commits (`fix(sram_compiler): ...`, `feat(sizing): ...`, `docs: ...`).

## Environment and verification

- `environment.yml` specifies the Conda environment (Python 3.9, PySpice 1.5,
  Xyce 7.4); newer runner code uses postponed annotations. This workspace has
  working `python3` with PySpice/numpy/pandas/matplotlib/PyYAML/pytest.
- Xyce may need the `openyield` Conda environment activated; check `command -v Xyce` before simulation. Full-array netlist generation (`real_cell_mode=0`)
  needs no simulator; equivalent modes can run Xyce for parameter extraction.
- Local development checks, if `dev/` is available:
  `python3 -m unittest discover -s dev/tests -v`.
- Keep solver/model/seed provenance with results. Match physical RC and equivalent
  modes when looking up qualified records; never promote partial or failed runs.
  `docs/DEVELOPMENT.md` documents local tools and the tracked scoring-source
  manifest; compiler table lookup must work when `dev/` is absent.
- Check `git diff --check` and review the final diff. Recent commits use
  `fix(sram_compiler): ...` and `docs: ...`; no repository-wide CI/linter config
  or package manifest was found in the initial scan.
