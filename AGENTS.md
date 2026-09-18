# OpenYield project instructions

This project is a open-source SRAM compiler for yield estimation and transistor sizing optimizations. The main functions include SRAM netlist generation with distributed RC loads, global and local process variations, and full DC/TRAN analyses. `REDAME.md` in both root and sub-folders are the tutorials for this projects. `docs/CHANGELOG.md`is per release (V2.x.x).`AGENTS.md` holds the detailed working conventions.

Current release: **V2.1.9** (the write drivers stay fully on until the wordline is observed off: `w_en = we_hold & (wordline_busy | selected_slot)` with `wordline_busy = !(wl_en_bar & wordline_off)`, the write-request latch opens on `wordline_idle`, and the next write slot waits for the drivers to be observed off (`selected_slot = cs_pre & write_slot & slot_armed`, a NOR latch set by `enables_off` after four unit stages and reset by `wordline_busy`); the testbench rejects a write whose target driver enable is below 0.9 VDD while its wordline is above 0.1 VDD (`VWEN_ACCESS_ERROR_<cycle>`), the local checker checks it per column and cycle; the testbench seeds the latch's t = 0 state with `.IC`; a per-device Monte-Carlo sweep at SS 0.9 V / 125 C, FF 1.1 V / -40 C, SF 0.9 V / 125 C and FS 1.1 V / -40 C re-derived the read clocks: every class bound keeps 0.02 T plus 10 % of the nominal access between the read output and the deadline at SS, so `timing_lookup.json` `v2.1.9-timing-8` raises the 128/256/512-row classes (shared 2200/2700/3700 ps, 6T-mux 2300/2900/4000 ps, 10T 512 rows 4200 ps); open: the write -> write slot exceeds the qualification scorer's 0.8 x budget at <= 64 rows; `docs/design/WRITE_HOLD_V2_1_9.md`). V2.1.8 removed the enable overlaps (a read wordline is released at the sense trigger, the precharge and the write slot wait for the previous enables; `docs/design/ENABLE_OVERLAP_V2_1_8.md`). V2.1.7 added the rising-edge select gate `cs_pre = cs & cs_delayed` and renamed the block `TIME_CONTROL` (instance `XTIME_CONTROL`, probe paths `XTIME_CONTROL:<node>`, `TimeControlFactory`, `create_time_control_circuit`, inner subcircuits named after their classes, map in `docs/design/TIME_CONTROL_PATH.md` section 6; `docs/design/SELECT_GATE_V2_1_7.md`). V2.1.6 put the write drivers in the precharge slot (`docs/design/WRITE_SLOT_V2_1_6.md`). Driver sizes, the 6T/10T cells and the read path are unchanged. The only remaining item in `docs/plans/V2_1_3_OPEN_ITEMS.md` is the carried Phase 6 scope (extracted metal, half-select write, yield estimator).

## Purpose and architecture

* Data flow: YAML in `sram_compiler/config_yaml/` -> `SRAM_CONFIG` (`sram_compiler/config_yaml/config.py`; root `config.py` re-exports it) -> factories in `sram_compiler/testbenches/parameter_factor.py` -> subcircuits in `sram_compiler/subcircuits/` (all derive from `BaseSubcircuit`, which owns the RC helper)-> `Sram6TCoreTestbench` (array, replica column, decoder, wordline drivers, TIME_CONTROL module, column periphery) -> `Sram6TCoreMcTestbench` (variation, stimuli, `.MEASURE`/`.PRINT`, Xyce execution, result parsing) -> metrics consumed by `size_optimization/exp_utils.py` (optimizer objective) or `yield_estimation/`.
* YAML widths/lengths are metres; PySpice unit objects inside circuits; sweep
  mode substitutes SPICE expression strings, so test numeric and sweep paths together.
* Clock timing is set from `sram_compiler/sizing/timing_lookup.json` for different array configurations. Use interpolation or extrapolation for unseen arry sizes. Make sure it has enough margin for correct access under the worst PVT variations.
* Critial drivers' sizes are configurated from `sram_compiler/sizing/sizing_lookup.json` for different array sizes. Use interpolation or extrapolation for unseen arry sizes. The drivers' ability can be adjusted through simulation results.
* Variation: `Sram6TCoreMcTestbench` modes are `nominal`, `shared` (one AGAUSS card per base model), `custom` (parameter table from the cell YAML) and `per-device` (independent `vth0/u0/voff` per MOS, specialized in `create_testbench()` by `sram_compiler/per_device_mc/netlist.py`). The per-device mode is default. `mc=True` now means per-device, so `mc_runs=1` without `mc_seed` is one unseeded random sample, not nominal; deterministic callers must pass `variation_mode='nominal'`. Per-device cannot be combined with the legacy `.STEP` sweeps.
* Parasitics: `w_rc` controls local storage-node and peripheral stubs; custom values propagate through all factories and nested cells. `interconnect.mode: distributed` is the only supported topology and the default; explicit star settings fail. Default wire geometry is illustrative (1 ohm / 0.1 fF per pitch), not extracted metal.
* Replica wires should match array lengths and loads, and the TIME_CONTROL module observes the far replica wordline before precharge. Equivalent modes 1-4 retain every wire segment and attach omitted-cell loads locally; extraction is numeric-only and uses the effective PVT and model-content cache identity. See `docs/design/DISTRIBUTED_RC_MODEL.md`. `main_sram.py` selects a wire YAML through `INTERCONNECT_CONFIG` and the CLI through `--interconnect-config` (`interconnect.load_interconnect()`).
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
- In each debug process, always check the critial pulse signals first that input and output from `TIME_CONTROL` module, such as `clk`, `clk_buf`, `cs`, `we`, `wl_en`, `rbl`, `rbl_delay`, `s_en`, `w_en`, `PRE`, `sa_iso`, or other signals in the timing waveforms. These enable pulses control all crictial subcircuits in SRAM macro.
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
