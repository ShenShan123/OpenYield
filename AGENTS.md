# OpenYield project instructions

Current release: **V2.1.2** (V2.1.1 circuit, timing and sizing unchanged; bounded-step Xyce retry for explicit `.TRAN` fields, CLI `--vdd`/`--temperature`, recorded Xyce installation and an audited first-500 write-failure inventory. V2.1.1 added distributed-only signal wiring, fixed array-class timing lookup, complete write-register initialization, frozen access/retention checks and preserved numerical retry evidence; driver transistor classes remain V2.0.9). Known limit from the V2.1.2 Phase 4/5 follow-up (`docs/design/TIMING_FOLLOWUP_V2_1_2.md`): the shared 4 ns class fails for 10T cells with a column mux at SS 0.9 V/125 °C (16x16 nominal, 8x4 under mismatch) and passes at 4.5 ns; the table stays frozen until a separate budget is evidenced.
The `rules_only` rule identity and the qualification artifact format remain V2.0.5; preserve historical version labels when referring to measurements or archived proposals. `docs/CHANGELOG.md` is per release (V2.x.x). `AGENTS.md` holds the detailed working conventions;

## Purpose and architecture

* Data flow: YAML in `sram_compiler/config_yaml/` -> `SRAM_CONFIG` (`sram_compiler/config_yaml/config.py`; root `config.py` re-exports it) -> factories in `sram_compiler/testbenches/parameter_factor.py` -> subcircuits in `sram_compiler/subcircuits/` (all derive from `BaseSubcircuit`, which owns the RC helper)-> `Sram6TCoreTestbench` (array, replica column, decoder, wordline drivers, TIME block, column periphery) -> `Sram6TCoreMcTestbench` (variation, stimuli, `.MEASURE`/`.PRINT`, Xyce execution, result parsing) -> metrics consumed by `size_optimization/exp_utils.py` (optimizer objective) or `yield_estimation/`.
* YAML widths/lengths are metres; PySpice unit objects inside circuits; sweep
  mode substitutes SPICE expression strings, so test numeric and sweep paths together.
* Timing is set from `sram_compiler/sizing/timing_lookup.json` for different array configurations. Use interpolation or extrapolation for unseen arry sizes. Make sure it has enough margin for correct access under the worst PVT variations.
* Critial drivers' sizes are configurated from `sram_compiler/sizing/sizing_lookup.json` for different array sizes. Use interpolation or extrapolation for unseen arry sizes. The drivers' ability can be adjusted through simulation results.
* Variation: `Sram6TCoreMcTestbench` modes are `nominal`, `shared` (one AGAUSS card per base model), `custom` (parameter table from the cell YAML) and `per-device` (independent `vth0/u0/voff` per MOS, specialized in `create_testbench()` by `sram_compiler/per_device_mc/netlist.py`). The per-device mode is default. `mc=True` now means per-device, so `mc_runs=1` without `mc_seed` is one unseeded random sample, not nominal; deterministic callers must pass `variation_mode='nominal'`. Per-device cannot be combined with the legacy `.STEP` sweeps.
* Parasitics: `w_rc` controls local storage-node and peripheral stubs; custom values propagate through all factories and nested cells. `interconnect.mode: distributed` is the only supported topology and the default; explicit star settings fail. Default wire geometry is illustrative (1 ohm / 0.1 fF per pitch), not extracted metal.
* Replica wires should match array lengths and loads, and TIME observes the far replica wordline before precharge. Equivalent modes 1-4 retain every wire segment and attach omitted-cell loads locally; extraction is numeric-only and uses the effective PVT and model-content cache identity. See `docs/design/DISTRIBUTED_RC_MODEL.md`. `main_sram.py` selects a wire YAML through `INTERCONNECT_CONFIG` and the CLI through `--interconnect-config` (`interconnect.load_interconnect()`).

## Working conventions

- Make sure the SRAM read/write/hold operations totally CORRECT first, including write, read, hold opertations. Then solve the driver sizings, transistor sizing optimizations, and yield estimations.
- Run commands from the repository root. Resolve data paths from source-file locations, not the caller's working directory; avoid new machine-local paths.
- Append optional arguments. Do not reformat unrelated legacy code or comments.
- YAML widths/lengths use SI metres; PySpice uses unit objects, and parameter sweeps use SPICE expression strings. Test numeric and sweep paths together.
- A baseline sizing result must remain fixed across cell candidates and PVT
  samples. Changed architecture/peripheral inputs must not silently reuse it.
- Use per-device mismatch as default in MC simulations; equivalent cells are approximations. Generated decks or passing measures alone do not prove waveform correctness, retention, sensing margin, or timing qualification.
- Keep generated decks/results under ignored `outputs/` or a temporary path. Keep ad hoc development scripts under ignored `dev/`, outside `sram_compiler/`. Preserve supplied CSV evidence.
- Keep topology separate from sizing policy; resolve loads from actual scaled widths. Preserve positional factory/testbench arguments and append optional ones.
- A change to the lookup table or to the `rules_only` derivation path must be intended: compare generated decks against the previous commit (detached `git worktree`) before claiming decks are unchanged; changed classes need new waveform evidence.
- Generated decks or passing measures alone do not prove correctness; check waveforms (the qualification scorer or `.prn` crossings) and say exactly which validation ran.
- Preserve supplied evidence (`docs/data/DRIVER_SIZING_data.csv`, `docs/data/TIMING_AUTOCONFIG_data.csv`,
  `docs/qualification/*.json`). Never promote partial or failed qualification runs.
- Xyce specifics: `.SAMPLING useExpr=true` plus `.options samples numsamples=N seed=S` enable sampling (AGAUSS returns its mean without it); failed measures print `FAILED`
  (`MEASFAIL=1`); "Time step too small" gets one tighter-step retry in the runner; native MPI sampling crashes on large decks, hence the materialized cards.
- `pkill -f <pattern>` also matches the shell that runs it; kill by PID or from a script file.
- Commits use Conventional Commits (`fix(sram_compiler): ...`, `feat(sizing): ...`, `docs: ...`).

## Environment and verification

- `environment.yml` specifies the Conda environment (Python 3.9, PySpice 1.5, Xyce 7.4); newer runner code uses postponed annotations. This workspace has working `python3` with PySpice/numpy/pandas/matplotlib/PyYAML/pytest.
- Xyce may need the `openyield` Conda environment activated; check `command -v Xyce` before simulation. Full-array netlist generation (`real_cell_mode=0`) needs no simulator; equivalent modes can run Xyce for parameter extraction.
- Local development checks, if `dev/` is available:
  `python3 -m unittest discover -s dev/tests -v`.
- Keep solver/model/seed provenance with results. Match physical RC and equivalent modes when looking up qualified records; never promote partial or failed runs.
- `docs/DEVELOPMENT.md` documents local tools and the tracked scoring-source
  manifest; compiler table lookup must work when `dev/` is absent.
- Check `git diff --check` and review the final diff. Recent commits use
  `fix(sram_compiler): ...` and `docs: ...`; no repository-wide CI/linter config
  or package manifest was found in the initial scan.
