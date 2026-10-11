# OpenYield V2.2.8

OpenYield builds transistor-level 6T and 10T SRAM netlists, runs DC and transient analyses with Xyce, and provides circuit-backed and offline sizing optimizers. The compiler models distributed RC wiring and samples independent local `vth0`, `u0`, and `voff` variation for each retained MOS by default.

**V2.2.8:** [Compact FreePDK45 layouts](sram_compiler/layout/README.md#compact-macro-floorplan)
reuse common compiler subcircuits, pack by actual device height, shorten pin
escapes, and bound the bitline escape channel. `main_sram.py layout compact`
preserves every MOS, finger, transistor size and intentional passive. Historical
calibration generators remain available; compact geometry requires its own
DRC/LVS, extraction and waveform evidence. No full V2.2.8 screen is complete.

The fixed compact 8×8 macro passes fresh Calibre DRC/LVS and coupled xRC:
**17.4% less area and 21.1% higher density** with frozen circuit sizing.
See the [layout record](docs/design/LAYOUT_COMPACTION_V2_2_8.md) for matched
source identities, measured footprints and validation scope.

**V2.2.7:** [FreePDK45 layout and Calibre xRC calibration](sram_compiler/layout/README.md)
now includes the complete 8×8 6T macro, TIME_CONTROL, registers, physical RC
filters and output latch. Controller sizing is legalized in
[`time_control.yaml`](sram_compiler/config_yaml/time_control.yaml).
The [complete macro calibration record](docs/design/FULL_RC_CALIBRATION_V2_2_7.md)
records zero Calibre DRC violations, LVS `CORRECT`, and passing complete
nominal access waveforms. Defaults use 709 independent RC networks, including
separate VDD/VSS meshes for IR drop, within the calibrated 8×8 reference.
Mean edge error against coupled xRC falls from 1,334.66 ps to 3.40 ps.
The [additional-size calibration](docs/design/MULTI_SIZE_RC_CALIBRATION_V2_2_7.md)
compares 8×16, 16×8 and 16×16 macros, including SNM curves and four-access
waveforms. Explicit size-specific profiles reduce mean edge error to
4.05–6.00 ps; all 22,881 waveform checks pass at the fixed 30 ns diagnostic clock.
V2.2.7 has no completed full waveform screen.

**Release status:** V2.2.6 fixes custom SNM construction, model-audit false rejections, sample-count validation, and single-cell coverage metadata. See the [configuration review](docs/design/CONFIGURATION_REVIEW_V2_2_6.md) for reproductions and validation. No full V2.2.6 screen has completed. The V2.2.5 campaign stopped incomplete; the [V2.2.5 simulation summary](docs/V2_2_5_SIMULATION_SUMMARY.md) records it with the corrected-model diagnostics, which retain their original scope. The configured limit is 512 rows by 256 columns, and 256x256 still lacks a converged operating point. The completed [V2.2.2 screen](docs/design/PHASED_CONTROL_V2_2_2.md) certifies only that earlier tree.

## Quick start

Run commands from the repository root. The supplied Conda environment includes Python, PySpice, and Xyce:

```bash
conda env create -f environment.yml
conda activate openyield
command -v Xyce
```

Edit the settings at the top of [`main_sram.py`](main_sram.py). Set `ARRAY = [8, 4, False]`, `OPERATION = "read"`, and `RUN_XYCE = False` to generate a full-transistor deck without a simulator:

```bash
python main_sram.py
```

Set `RUN_XYCE = True` to simulate the configured macro. The default seed makes the local process sample repeatable; set `MC_RUNS` to draw more samples. Decks and results go under `outputs/main_sram/`. Xyce is also needed during deck construction if you select equivalent mode 1–4.

`MC_RUNS` must be a positive integer, or `None` to use the YAML count for sampled
modes. Nominal uses one run; custom mode uses its process-table row count.
SNM operations simulate one cell and do not provide full-array mismatch coverage.

The [unified experiment guide](sram_compiler/experiments/README.md) covers command-line overrides and all workflows:

```bash
python main_sram.py --help
python main_sram.py simulate --rows 8 --cols 4 --operation read --mc-runs 10 --no-run-xyce
python main_sram.py simulate --variation-mode nominal --operation read --run-xyce
python main_sram.py layout --help
```

`simulate` is optional when passing simulation flags directly. CLI options override
the script settings. Per-device mismatch is the default; `--seed random` requests
fresh draws. `--version` reports V2.2.8.

## Configure an array

`main_sram.py` is the SRAM macro entrance. Its settings block controls `ARRAY = [rows, cols, column_mux]`, corner, 6T cell widths, operation, seed, target cell, and whether to run Xyce. It loads YAML into memory; it does not rewrite the tracked configuration files. For an unvaried diagnostic corner, set `VARIATION_MODE = "nominal"`.

Circuit defaults live in `sram_compiler/config_yaml/`. `global.yaml` sets supply, temperature, geometry, cell type, timing, wiring, and equivalent mode. Cell and peripheral YAMLs define transistor sizes in **metres**. Clocks come from [timing_lookup.json](sram_compiler/sizing/timing_lookup.json), and critical driver sizes from [sizing_lookup.json](sram_compiler/sizing/sizing_lookup.json). Freeze both across candidate cells and PVT samples. The default interconnect YAML selects independent extracted signal and power networks for the nominal 8×8 reference. Outside its recorded scope, separate route geometries remain illustrative and supplies remain ideal; run metadata reports that the reference was not applied. `INTERCONNECT_CONFIG` selects another YAML. The [compiler tutorial](sram_compiler/README.md) covers configuration and the Python API.

| Variation mode | Meaning |
|---|---|
| `nominal` | One fixed PDK corner, no Monte Carlo draw |
| `per-device` | Independent `vth0`, `u0`, and `voff` draws per retained MOS; automatic default |
| `shared` | One random model card per base model |
| `custom` | Explicit parameter table from the cell YAML |

`REAL_CELL_MODE = 0` keeps every transistor and is the reference for full device coverage; the YAML default is also 0. Modes 1–4 replace selected unused cells with approximate equivalent loads and need Xyce even during netlist generation. Compare them against mode 0 using the [equivalent-model guide](sram_compiler/equivalent_modeling/README.md). A write currently drives every column in the selected row; half-select writes remain open work.

## Sizing and yield workflows

Circuit-backed optimization is under [`size_optimization/`](size_optimization/README.md). For example, after installing its optional dependencies:

```bash
python main_sram.py circuit-optimize --algorithm sa
python main_sram.py circuit-experiment
```

`experiment.py` offers joint architecture and transistor sizing or sizing at fixed configurations. The separate [OpenYield V2 offline package](size_optimization/openyield_v2/README.md) trains on bundled static data and does not call Xyce during optimization:

```bash
python -m pip install -r size_optimization/openyield_v2/requirements.txt
python main_sram.py experiment --dry-run
```

Legacy importance-sampling estimators are retained in [`yield_estimation/`](yield_estimation/README.md), but they are not a validated V2.2.6 yield workflow. Use `MC_RUNS` in `main_sram.py` for current sampling, and read the [open items](docs/README.md#open-items) before reporting a yield estimate.

## Validate changes

```bash
python -m compileall -q sram_compiler utils size_optimization yield_estimation
python main_sram.py simulate --rows 2 --cols 2 --no-run-xyce
```

The regression and waveform Python scripts are retained only in local, ignored `tests/` workspaces. If you have them, run `python -m pytest -q tests size_optimization/openyield_v2/tests`; some equivalent-model checks need Xyce. The tracked [SPICE manifests](tests/spice/README.md) describe the unfinished V2.2.5 screen; no V2.2.5 or V2.2.6 full screen has completed. Passing software tests or `.MEASURE` cards alone does not establish read, write, retention, sense, and recovery correctness.

## Guides and project map

| Path | Start here for |
|---|---|
| [Compiler tutorial](sram_compiler/README.md) | Configuration, testbenches, results, and common failures |
| [Sizing and timing](sram_compiler/sizing/README.md) | Frozen driver classes and clock budgets |
| [Equivalent models](sram_compiler/equivalent_modeling/README.md) | Modes 0–4 and accuracy comparison |
| [Sizing optimization](size_optimization/README.md) | Circuit-backed and offline optimizer entry points |
| [Shared utilities](utils/README.md) | Measurement, waveform, plot, area, and SPICE helpers |
| [Local checks](tests/README.md) | Retained manifests and optional local regression scripts |
| [Documentation index](docs/README.md) | Release records, evidence, and remaining work |
| [Changelog](docs/CHANGELOG.md) | Version history |

The main flow is YAML → `SRAM_CONFIG` → subcircuit factories → `Sram6TCoreMcTestbench` → Xyce measurements and waveforms → optimization or analysis. Generated decks and results belong under ignored `outputs/`.

## Cite us

```bibtex
@INPROCEEDINGS{OpenYield,
  author={Shen, Shan and Li, Xingyang and Liu, Zhuohua and Ma, Junhao and Wang, Yikai and Wu, Yiheng and Sun, Yuquan and Xing, Wei W.},
  booktitle={2025 IEEE 43rd International Conference on Computer Design (ICCD)},
  title={OpenYield: An Open-Source SRAM Yield Analysis and Optimization Benchmark Suite},
  year={2025},
  pages={167-175},
  doi={10.1109/ICCD65941.2025.00030}
}
```
