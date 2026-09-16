# OpenYield project instructions

This project is a open-source SRAM compiler for yield estimation and transistor sizing optimizations. The main functions include SRAM netlist generation with distributed RC loads, global and local process variations, and full DC/TRAN analyses. `REDAME.md` in both root and sub-folders are the tutorials for this projects. `docs/CHANGELOG.md`is per release (V2.x.x).`AGENTS.md` holds the detailed working conventions.

Current release: **V2.1.5** (equivalent model as an input option: the old top-level `equivalent_modeling/` is now `sram_compiler/equivalent_modeling/`, with a `global.yaml` `equivalent:` block that `--real-cell-mode` / `real_cell_mode` / `REAL_CELL_MODE` override, and an accuracy entrance `python3 -m sram_compiler.equivalent_modeling.compare`. 10T round: the 10T pull-down is widened to 287 nm because the read-disturb bump is the read current through two stacked pull-downs, which drops the 512-row 10T class from 14 ns to 10 ns and gains read-output margin at every height; `timing_lookup.json` is `v2.1.5-timing-4`; a variant class is now floored at the shared class beyond the table too; `docs/design/TIMING_10T_BUDGET_V2_1_5.md` and `docs/design/EQUIVALENT_MODEL_V2_1_5.md`). 6T decks, driver sizes and the TIME circuit are unchanged. The only remaining item in `docs/plans/V2_1_3_OPEN_ITEMS.md` is the carried Phase 6 scope (extracted metal, half-select write, yield estimator); the 10T classes below 512 rows now hold more margin than the 250 ps rule needs and a later round may spend it with its own evidence.

## Purpose and architecture

* Data flow: YAML in `sram_compiler/config_yaml/` -> `SRAM_CONFIG` (`sram_compiler/config_yaml/config.py`; root `config.py` re-exports it) -> factories in `sram_compiler/testbenches/parameter_factor.py` -> subcircuits in `sram_compiler/subcircuits/` (all derive from `BaseSubcircuit`, which owns the RC helper)-> `Sram6TCoreTestbench` (array, replica column, decoder, wordline drivers, TIME module, column periphery) -> `Sram6TCoreMcTestbench` (variation, stimuli, `.MEASURE`/`.PRINT`, Xyce execution, result parsing) -> metrics consumed by `size_optimization/exp_utils.py` (optimizer objective) or `yield_estimation/`.
* YAML widths/lengths are metres; PySpice unit objects inside circuits; sweep
  mode substitutes SPICE expression strings, so test numeric and sweep paths together.
* Clock timing is set from `sram_compiler/sizing/timing_lookup.json` for different array configurations. Use interpolation or extrapolation for unseen arry sizes. Make sure it has enough margin for correct access under the worst PVT variations.
* Critial drivers' sizes are configurated from `sram_compiler/sizing/sizing_lookup.json` for different array sizes. Use interpolation or extrapolation for unseen arry sizes. The drivers' ability can be adjusted through simulation results.
* Variation: `Sram6TCoreMcTestbench` modes are `nominal`, `shared` (one AGAUSS card per base model), `custom` (parameter table from the cell YAML) and `per-device` (independent `vth0/u0/voff` per MOS, specialized in `create_testbench()` by `sram_compiler/per_device_mc/netlist.py`). The per-device mode is default. `mc=True` now means per-device, so `mc_runs=1` without `mc_seed` is one unseeded random sample, not nominal; deterministic callers must pass `variation_mode='nominal'`. Per-device cannot be combined with the legacy `.STEP` sweeps.
* Parasitics: `w_rc` controls local storage-node and peripheral stubs; custom values propagate through all factories and nested cells. `interconnect.mode: distributed` is the only supported topology and the default; explicit star settings fail. Default wire geometry is illustrative (1 ohm / 0.1 fF per pitch), not extracted metal.
* Replica wires should match array lengths and loads, and TIME module observes the far replica wordline before precharge. Equivalent modes 1-4 retain every wire segment and attach omitted-cell loads locally; extraction is numeric-only and uses the effective PVT and model-content cache identity. See `docs/design/DISTRIBUTED_RC_MODEL.md`. `main_sram.py` selects a wire YAML through `INTERCONNECT_CONFIG` and the CLI through `--interconnect-config` (`interconnect.load_interconnect()`).
* Equivalent model: `sram_compiler/equivalent_modeling/` resolves the mode (`resolve_equivalent`, the `equivalent:` block of `global.yaml`, overridden by an explicit `real_cell_mode`) and holds the accuracy entrance `compare.py`; the injection itself stays in `subcircuits/sram_cell_add_equivalent.py`. Mode 0 is the only mode with full per-device coverage, so evidence and qualification runs use it; modes 1-4 also need Xyce at netlist-generation time for cell parasitic extraction.

## Working conventions

- Make sure the SRAM read/write/hold operations totally CORRECT first. Then solve the driver sizes, transistor sizing optimizations, and yield estimations.
- Run commands from the repository root. Resolve data paths from source-file locations, not the caller's working directory; avoid new machine-local paths.
- Append optional arguments. Do not reformat unrelated legacy code or comments.
- YAML widths/lengths use SI metres; PySpice uses unit objects, and parameter sweeps use SPICE expression strings. Test numeric and sweep paths together.
- A baseline sizing result must remain fixed across cell candidates and PVT
  samples. Changed architecture/peripheral inputs must not silently reuse it.
- Use per-device mismatch as default in MC simulations; equivalent cells are approximations. Generated decks or passing measures alone do not prove waveform correctness, retention, sensing margin, or timing qualification.
- Keep generated decks/results under ignored `outputs/` or a temporary path. Keep ad hoc development scripts under ignored `dev/`, outside `sram_compiler/`. Preserve supplied CSV evidence.
- Keep topology separate from sizing and timing policy; resolve loads from actual scaled widths. Preserve positional factory/testbench arguments and append optional ones.
- In each debug process, always check the critial pulse signals first that input and output from `TIME` module, such as `clk`, `clk_buf`, `cs`, `we`, `wl_en`, `rbl`, `rbl_delay`, `s_en`, `w_en`, `PRE`, `sa_iso`, or other signals in the timing waveforms. These enable pulses control all crictial subcircuits in SRAM macro.
- Generated decks or passing measures alone do not prove correctness; check waveforms (the qualification scorer or `.prn` crossings) and say exactly which validation ran.
- Preserve supplied evidence (e.g., `docs/data/DRIVER_SIZING_data.csv`, `docs/data/TIMING_AUTOCONFIG_data.csv`, `docs/qualification/*.json`). Never promote partial or failed qualification runs.
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
  `fix(sram_compiler): ...` and `docs: ...`; no repository-wide CI/linter config or package manifest was found in the initial scan.
