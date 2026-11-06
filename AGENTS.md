# OpenYield working conventions

OpenYield is an open-source SRAM compiler for transistor-level 6T and 10T arrays, DC/transient analysis, sizing optimization, and yield research. The compiler version is **V2.2.9**. The [root tutorial](README.md) and subdirectory READMEs explain usage, and [docs/CHANGELOG.md](docs/CHANGELOG.md) carries release details.

The working tree is **V2.2.9**. The supported SRAM architecture includes a standard 2:1 column mux. The [FreePDK45 workflow](sram_compiler/layout/README.md)
covers component layouts, TIME_CONTROL and the default compact-cell,
column-split bank (`layout bank`; `--with-controller` adds the complete macro). The older fixed and scaled
macro generators, and the unrelated "compact macro" floorplans of early
V2.2.8, have been removed with their artifacts. OpenRAM-pitch, legacy,
data-path, unsplit, row-split and beside-controller bank implementations are also retired.
Keep sizing frozen; physical receipts do not establish waveform correctness
or release qualification. Simulation retains its scoped reference RC profiles.
Complete macros end with a supply grid (M7–M9 straps, M3 controller-gap straps, an M4 bitcell VSS mesh over each half-array); worst-PVT post-layout access
verification of the V2.2.8 controller (local `dev/postlayout/`) is recorded in
[docs/design/layout/POST_LAYOUT_V2_2_8.md](docs/design/layout/POST_LAYOUT_V2_2_8.md).
The FreePDK45 xRC deck charges 88–102 Ω per minimum via1/via2/contact; recorded pass/fail results use it unchanged, and scaled-cut runs are sensitivity only.
Compact-layout clocks come from `sram_compiler/sizing/layout_timing_lookup.json`; `timing_lookup.json` describes the schematic driver profile. V2.2.9 6T clocks budget capture through complete recovery, with one 10% margin and separate absolute read deadlines. The [V2.2.9 record](docs/design/timing/ACCESS_SPEEDUP_V2_2_9.md) separates the extracted 32×32 anchor, schematic checks and larger-array estimates. 10T retains the legacy controller and clocks. Post-layout runs feed every macro supply pin; the controller keeps each enable driver chain on its own rows (`time_control.ENABLE_CHAINS`), and the grid merges the decoder's band VSS rails and puts M2 buses on its edge rails.
The M4 array mesh (`power_grid.array_mesh`) ties each half-array's bitcell VSS M3 rails together; without it, rails fed from one end only let a V2.2.8 128×128 FF 1.1 V −40 °C read flip far-end cells. The ten-case passes at 32×32, 64×64 and 128×128 in that historical record qualify its controller only.

## Architecture

YAML under `sram_compiler/config_yaml/` loads through `SRAM_CONFIG` in `config.py`. Factories in `sram_compiler/testbenches/parameter_factor.py` build subcircuits, then `Sram6TCoreTestbench` assembles the array, replica column, decoder, TIME_CONTROL, wires, and periphery. `Sram6TCoreMcTestbench` adds variation, stimuli, measurements, Xyce execution, and result parsing. `main_sram.py` is the unified experiment entrance. `config_yaml/loader.py` loads experiment configurations, `variation/` specializes independent local model cards, and `experiments/` dispatches simulation, layout, equivalent comparisons and sizing workflows. Optimization consumes the resulting metrics through `size_optimization/exp_utils.py`.

The supported timing envelope is 512 rows by 256 columns; `timing_lookup.json` and `sizing_lookup.json` set frozen clock and critical driver classes. Physical wiring is distributed only. Its default 1 ohm / 0.1 fF per pitch is illustrative, not extracted metal; `w_rc` controls separate local stubs. Mode 0 retains every transistor; equivalent modes 1–4 approximate omitted cells and need Xyce for extraction. Per-device mismatch is the default random mode; use `variation_mode='nominal'` for a deterministic corner. See the linked tutorials for inputs and limits.

## Editing rules

- Establish read, write, hold, sense, and recovery correctness before changing driver sizes, transistor optimization, or yield logic. A deck or passing measures alone does not prove waveform correctness.
- Run commands from the repository root. Resolve data paths from source locations; avoid machine-local paths. Put generated decks and results under ignored `outputs/` or a temporary directory.
- Preserve positional factory and testbench arguments; append optional ones. Keep topology separate from sizing and timing policy, and resolve loads from actual layout implementation.
- YAML widths and lengths are SI metres; circuits use PySpice units, and sweeps substitute SPICE expression strings. Check numeric and sweep paths when either changes.
- Freeze baseline sizing and timing across cell candidates and PVT samples. Changed architecture, periphery, model, or physical RC must not silently reuse a stale baseline.
- Default Monte Carlo to per-device `vth0/u0/voff` mismatch. `mc=True` means per-device, so `mc_runs=1` without a seed is random.
- Inspect TIME_CONTROL pulse inputs and outputs first when debugging: `clk`, `clk_buf`, `cs`, `we`, `wl_en`, `rbl`, `rbl_delay`, `s_en`, `w_en`, `PRE`, and `sa_iso`. Check physical far-end crossings and data retention, then state exactly which waveform validation ran.
- Preserve supplied evidence, including `docs/data/*.csv` and `docs/qualification/*.json`. Record solver, model, seed, RC, and equivalent-mode provenance with results.
- Keep ad hoc scripts under ignored `dev/`; regression and waveform Python scripts under ignored `tests/` stay local. The `tests/spice/*.json` case manifests remain tracked. A fresh clone does not include the local Python test runner.
- Commit with Conventional Commits, run `git diff --check`, and review the final diff.

## Environment

`environment.yml` defines the Python 3.9, PySpice 1.5, Xyce 7.4 environment. Check `command -v Xyce` before simulation. Full mode 0 netlist generation needs no simulator; equivalent modes may invoke Xyce during extraction. Xyce reports failed measures as `FAILED` (`MEASFAIL=1`); the runner has one tighter-step retry for "Time step too small". The local development tools and validation history are indexed in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).
