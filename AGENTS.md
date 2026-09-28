# OpenYield working conventions

OpenYield is an open-source SRAM compiler for transistor-level 6T and 10T arrays, DC/transient analysis, sizing optimization, and yield research. The current release is **V2.2.5**. The [root tutorial](README.md) and subdirectory READMEs explain usage; [docs/CHANGELOG.md](docs/CHANGELOG.md) and [the V2.2.5 design record](docs/design/COMPILER_ENTRY_V2_2_5.md) carry release details. No V2.2.5 waveform screen has run, and 256x256 has no working operating point. The last assembled screen, `docs/data/PHASED_CONTROL_V2_2_2.json`, certifies only that earlier tree. Never promote a partial or failed screen.

## Architecture

YAML under `sram_compiler/config_yaml/` loads through `SRAM_CONFIG` in `config.py`. Factories in `sram_compiler/testbenches/parameter_factor.py` build subcircuits, then `Sram6TCoreTestbench` assembles the array, replica column, decoder, TIME_CONTROL, wires, and periphery. `Sram6TCoreMcTestbench` adds variation, stimuli, measurements, Xyce execution, and result parsing. `main_sram.py` and `python -m sram_compiler.per_device_mc.run` are the user entrances. Optimization consumes the resulting metrics through `size_optimization/exp_utils.py`.

The supported timing envelope is 512 rows by 256 columns; `timing_lookup.json` and `sizing_lookup.json` set frozen clock and critical driver classes. Physical wiring is distributed only. Its default 1 ohm / 0.1 fF per pitch is illustrative, not extracted metal; `w_rc` controls separate local stubs. Mode 0 retains every transistor; equivalent modes 1–4 approximate omitted cells and need Xyce for extraction. Per-device mismatch is the default random mode; use `variation_mode='nominal'` for a deterministic corner. See the linked tutorials for inputs and limits.

## Editing rules

- Establish read, write, hold, sense, and recovery correctness before changing driver sizes, transistor optimization, or yield logic. A deck or passing measures alone does not prove waveform correctness.
- Run commands from the repository root. Resolve data paths from source locations; avoid machine-local paths. Put generated decks and results under ignored `outputs/` or a temporary directory.
- Preserve positional factory and testbench arguments; append optional ones. Keep topology separate from sizing and timing policy, and resolve loads from actual scaled widths.
- YAML widths and lengths are SI metres; circuits use PySpice units, and sweeps substitute SPICE expression strings. Check numeric and sweep paths when either changes.
- Freeze baseline sizing and timing across cell candidates and PVT samples. Changed architecture, periphery, model, or physical RC must not silently reuse a stale baseline.
- Default Monte Carlo to per-device `vth0/u0/voff` mismatch. `mc=True` means per-device, so `mc_runs=1` without a seed is random. Equivalent arrays do not cover every MOS and are not full-array yield evidence.
- Inspect TIME_CONTROL pulse inputs and outputs first when debugging: `clk`, `clk_buf`, `cs`, `we`, `wl_en`, `rbl`, `rbl_delay`, `s_en`, `w_en`, `PRE`, and `sa_iso`. Check physical far-end crossings and data retention, then state exactly which waveform validation ran.
- Preserve supplied evidence, including `docs/data/*.csv` and `docs/qualification/*.json`. Record solver, model, seed, RC, and equivalent-mode provenance with results.
- Keep ad hoc scripts under ignored `dev/`; regression and waveform Python scripts under ignored `tests/` stay local. The `tests/spice/*.json` case manifests remain tracked. A fresh clone does not include the local Python test runner.
- Commit with Conventional Commits, run `git diff --check`, and review the final diff.

## Environment

`environment.yml` defines the Python 3.9, PySpice 1.5, Xyce 7.4 environment. Check `command -v Xyce` before simulation. Full mode 0 netlist generation needs no simulator; equivalent modes may invoke Xyce during extraction. Xyce reports failed measures as `FAILED` (`MEASFAIL=1`); the runner has one tighter-step retry for "Time step too small". The local development tools and validation history are indexed in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).
