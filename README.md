# OpenYield V2.2.4

OpenYield builds transistor-level 6T and 10T SRAM netlists, runs DC and transient analyses with Xyce, and provides circuit-backed and offline sizing optimizers. The compiler models distributed RC wiring and can apply independent local variation to each retained MOS device.

**Release status:** V2.2.4 fixes passive replica load cells and several control and runner issues. Its full waveform screen has not run. The configured limit is 512 rows by 256 columns, but 256x256 still lacks a converged operating point. The completed [V2.2.2 screen](docs/design/PHASED_CONTROL_V2_2_2.md) certifies only that earlier tree. See the [V2.2.4 record](docs/design/PHASED_CONTROL_V2_2_4.md) before treating a generated deck or passing measurement as functional evidence.

## Quick start

Run commands from the repository root. The supplied Conda environment includes Python, PySpice, and Xyce:

```bash
conda env create -f environment.yml
conda activate openyield
command -v Xyce
```

Generate a small full-transistor array without invoking Xyce:

```bash
python -m sram_compiler.per_device_mc.run \
  --rows 8 --cols 4 --operation read \
  --variation-mode nominal --mc-runs 1 \
  --real-cell-mode 0 --output-dir outputs/first_run
```

Add `--run-xyce` to execute the deck and save measurements and a waveform. For a reproducible local-mismatch run, use `--variation-mode per-device --mc-runs 2 --seed 3 --run-xyce`. A single per-device run without a seed is a random draw; use `nominal` when you want no variation. Outputs are placed in a configuration-specific directory under the selected output root.

```bash
python -m sram_compiler.per_device_mc.run \
  --rows 8 --cols 4 --operation write \
  --variation-mode per-device --mc-runs 2 --seed 3 \
  --real-cell-mode 0 --output-dir outputs/first_run --run-xyce
```

The CLI also accepts `read&write`, `hold_snm`, `read_snm`, and `write_snm`. Use `python -m sram_compiler.per_device_mc.run --help` for all options. The [runner guide](sram_compiler/per_device_mc/README.md) explains modes, files, and sample provenance.

## Configure an array

`main_sram.py` is the editable one-run entrance. Its settings block controls `ARRAY = [rows, cols, column_mux]`, corner, 6T cell widths, operation, variation, seed, and target cell. Running `python main_sram.py` generates and simulates the configured array. It loads YAML into memory; it does not rewrite the tracked configuration files.

Circuit defaults live in `sram_compiler/config_yaml/`. `global.yaml` sets supply, temperature, geometry, cell type, timing, wiring, and equivalent mode. Cell and peripheral YAMLs define transistor sizes in **metres**. Clocks come from [timing_lookup.json](sram_compiler/sizing/timing_lookup.json), and critical driver sizes from [sizing_lookup.json](sram_compiler/sizing/sizing_lookup.json). Freeze both across candidate cells and PVT samples. The default wire geometry is illustrative rather than extracted metal; pass an interconnect YAML through `--interconnect-config` when you have measured geometry. The [compiler tutorial](sram_compiler/README.md) covers the configuration and Python API.

| Variation mode | Meaning |
|---|---|
| `nominal` | One fixed PDK corner, no Monte Carlo draw |
| `per-device` | Independent `vth0`, `u0`, and `voff` draws per retained MOS; default |
| `shared` | One random model card per base model |
| `custom` | Explicit parameter table from the cell YAML |

`--real-cell-mode 0` keeps every transistor and is the reference for per-device coverage. Modes 1–4 replace selected unused cells with approximate equivalent loads and need Xyce even during netlist generation. Compare them against mode 0 using the [equivalent-model guide](sram_compiler/equivalent_modeling/README.md). A write currently drives every column in the selected row; half-select writes remain open work.

## Sizing and yield workflows

Circuit-backed optimization is under [`size_optimization/`](size_optimization/README.md). For example, after installing its optional dependencies:

```bash
python size_optimization/demo_sa.py
python size_optimization/experiment.py
```

`experiment.py` offers joint architecture and transistor sizing or sizing at fixed configurations. The separate [OpenYield V2 offline package](size_optimization/openyield_v2/README.md) trains on bundled static data and does not call Xyce during optimization:

```bash
python -m pip install -r size_optimization/openyield_v2/requirements.txt
python -m size_optimization.openyield_v2.run_experiment --dry-run
```

Legacy importance-sampling estimators are retained in [`yield_estimation/`](yield_estimation/README.md), but they are not a validated V2.2.4 yield workflow. Use the compiler's per-device runner for current sampling, and read the [open items](docs/README.md#open-items) before reporting a yield estimate.

## Validate changes

```bash
python -m compileall -q sram_compiler utils size_optimization yield_estimation
python -m sram_compiler.per_device_mc.run --rows 8 --cols 4 \
  --operation read --variation-mode nominal --mc-runs 1 \
  --real-cell-mode 0 --output-dir outputs/smoke
```

The regression and waveform Python scripts are retained only in local, ignored `tests/` workspaces. If you have them, run `python -m pytest -q tests size_optimization/openyield_v2/tests`; some equivalent-model checks need Xyce. The tracked [SPICE manifests](tests/spice/README.md) describe the pending V2.2.4 screen. Passing software tests or `.MEASURE` cards alone does not establish read, write, retention, sense, and recovery correctness.

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
