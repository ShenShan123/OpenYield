# OpenYield project instructions

Current release: **V2.0.6** (compiler integration and documentation organization).
The sizing rule and qualification artifact identities remain V2.0.5; preserve
historical version labels when referring to measurements or archived proposals.

## Purpose and architecture

OpenYield generates 6T/10T SRAM transistor netlists with PySpice, runs Xyce
transient/DC/Monte Carlo simulations, and evaluates sizing and yield.

- `sram_compiler/config_yaml/config.py` is the canonical YAML loader;
  root `config.py` is a compatibility re-export. `Parameter.value`, bounds,
  model choices, and sweep values are distinct inputs.
- `sram_compiler/subcircuits/` contains circuit topology. Factories in
  `sram_compiler/testbenches/parameter_factor.py` translate configuration
  and sweep parameters into those circuits.
- `Sram6TCoreTestbench` builds both cell types and their periphery;
  `Sram6TCoreMcTestbench` adds variation, stimuli, measures, and Xyce execution.
- `sram_compiler/sizing/` owns driver sizing rules and resolved loads.
- Reusable compiler regression tests live in top-level `tests/`.
  Local experiments, qualification runners, and their tests live under ignored
  `dev/`; they are optional and must not become runtime dependencies.
- Root `main_sram.py` is the script entrance: it uses `load_config()`, applies its
  settings in memory, and defaults to seeded per-device mismatch over the full array.
- `sram_compiler/per_device_mc/run.py` provides a CLI and in-memory `load_config()` helper.
  `sram_compiler/per_device_mc/netlist.py` specializes retained MOS devices for local MC.
- `size_optimization/exp_utils.py` connects legacy optimizers to simulation.
  `size_optimization/openyield_v2/` is a separate offline surrogate workflow.
- `yield_estimation/` contains MC/importance-sampling algorithms; some legacy
  entry points still use machine-local paths/imports.
- `utils/` separates SPICE models, measurements, waveform parsing, plotting, and
  area estimates. Its package exports preserve `from utils import ...` callers;
  `tran_models/` contains FreePDK45 models for TT/FF/SS/FS/SF.

Data flow: YAML → configuration objects → factories/subcircuits → testbench
netlist → Xyce → measurements/waveforms → metrics/optimizer or yield estimator.

## Working conventions

- Run commands from the repository root. Resolve data paths from source-file
  locations, not the caller's working directory; avoid new machine-local paths.
- Keep Python names and existing SPICE node/subcircuit names consistent with
  nearby code. Preserve compatibility with positional factory/testbench callers;
  append optional arguments. Do not reformat unrelated legacy code or comments.
- YAML widths/lengths use SI metres; PySpice uses unit objects, and parameter
  sweeps use SPICE expression strings. Test numeric and sweep paths together.
- Keep topology separate from sizing policy. Resolve periphery loads from
  actual scaled base widths, including mux count and replica/control loads.
- A baseline sizing result must remain fixed across cell candidates and PVT
  samples. Changed architecture/peripheral inputs must not silently reuse it.
- Distinguish shared-model MC from per-device mismatch; equivalent cells are
  approximations. Generated decks or passing measures alone do not prove
  waveform correctness, retention, sensing margin, or timing qualification.
- `main_sram.py` reads YAML in memory and runs Xyce, so it is not a simulator-free
  smoke test. Prefer `sram_compiler/per_device_mc/run.py` (generation only) or
  in-memory configuration in tests. `sram_compiler/testbenches/yaml_change.py`
  rewrites tracked YAML files and needs `ruamel.yaml`; do not call it from tests.
- Keep generated decks/results under ignored `outputs/` or a temporary path.
  Keep ad hoc development scripts under ignored `dev/`, outside `sram_compiler/`.
  Preserve supplied CSV evidence. Avoid whole-repository test discovery:
  several files named `test` are expensive experiment/demo entry points.
- Keep `docs/DRIVER_SIZING_PROPOSAL.md` as the full working proposal; update specific
  sections as implementation progresses rather than replacing it with a summary.
  The original is backed up in `docs/design/DRIVER_SIZING_PROPOSAL_V2.0.4.md`.
  `docs/TIMING_AUTOCONFIG.md` retains the original proposal; measured-period support
  now lives in `sram_compiler/sizing/timing.py`. Qualification status is recorded
  at the top of `docs/DRIVER_SIZING_PROPOSAL.md`.

## Environment and verification

- `environment.yml` specifies the Conda environment (Python 3.9, PySpice 1.5,
  Xyce 7.4); newer runner code uses postponed annotations. This workspace has
  working `python3` with PySpice/numpy/pandas/matplotlib/PyYAML/pytest.
- Xyce may need the `openyield` Conda environment activated; check `command -v
  Xyce` before simulation. Full-array netlist generation (`real_cell_mode=0`)
  needs no simulator; equivalent modes can run Xyce for parameter extraction.
- Driver checks: `python3 -m unittest discover -s tests -v`.
- Offline optimizer checks:
  `python3 -m unittest discover -s size_optimization/openyield_v2/tests -v`.
- For generation smoke, use the in-memory example in
  `sram_compiler/sizing/README.md` with `real_cell_mode=0`. When constructing a
  PySpice simulator object, explicitly select `simulator='xyce-serial'`; the
  package default can try to load an unavailable `libngspice.so` even for deck
  export. The updated MC testbench and per-device runner select Xyce explicitly.
- Local development checks, if `dev/` is available:
  `python3 -m unittest discover -s dev/tests -v`.
- Full driver qualification campaign (requires ignored local `dev/`; V2.0.5
  evidence format retained in V2.0.6): `python3 -m dev.sizing.campaign --xyce /path/to/Xyce
  --workers 48`. It is expensive; use `--sizes 8x4 --pilot` for a small check.
  Keep solver/model/seed provenance with results. Match physical RC and equivalent
  modes when looking up qualified records; never promote partial or failed runs.
  `docs/DEVELOPMENT.md` documents local tools and the tracked scoring-source
  manifest; compiler table lookup must work when `dev/` is absent.
- For circuit changes inspect emitted MOS widths, connectivity, and sweep
  expressions, then run focused simulations and waveform checks as appropriate.
  Report exactly which validation ran and which qualification remains pending.
- Check `git diff --check` and review the final diff. Recent commits use
  `fix(sram_compiler): ...` and `docs: ...`; no repository-wide CI/linter config
  or package manifest was found in the initial scan.
