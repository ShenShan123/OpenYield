# OpenYield V2.2.8

OpenYield builds transistor-level 6T and 10T SRAM netlists, runs DC and transient analyses with Xyce, and provides circuit-backed and offline sizing optimizers. The compiler models distributed RC wiring and samples independent local `vth0`, `u0`, and `voff` variation for each retained MOS by default.

The opt-in [small-cell flow](sram_compiler/layout/README.md#small-cells-and-component-placement)
builds internally routed, pitch-constrained cells, then places and routes their
instances. The [integrated bank flow](sram_compiler/layout/README.md#openram-pitch-bank)
includes row logic, column periphery, the output latch, and a replica column
inside the core. Its decoder is on the left, using OpenRAM’s 1.365 µm row
and 0.705 µm column pitch. See the [pitch and area verification record](docs/design/layout/OPENRAM_PITCH_V2_2_8.md).
The [OpenRAM reference run](docs/design/layout/OPENRAM_REFERENCE_V2_2_8.md) provides a
separate generated 32×32 bank layout. The optional
[TIME_CONTROL flow](docs/design/layout/LAYOUT_SCALING_V2_2_8.md#split-arrays-with-a-centre-periphery)
splits the arrays around the periphery, with control logic between the decoder
halves. The opt-in [compact cell](docs/design/layout/COMPACT_CELL_V2_2_8.md)
uses a 1125 × 395 nm point-symmetric 6T cell and pitch-matched periphery with a
2:1 column mux. Its [annotated bitcell PNG](docs/design/layout/COMPACT_CELL_V2_2_8.png)
identifies all six MOS instances. Physical checks do not qualify autonomous bank operation.

**Release status:** V2.2.6 fixes custom SNM construction, model-audit false rejections, sample-count validation, and single-cell coverage metadata. See the [configuration review](docs/design/compiler/CONFIGURATION_REVIEW_V2_2_6.md) for reproductions and validation. No full V2.2.6 screen has completed. The V2.2.5 campaign stopped incomplete; the [V2.2.5 simulation summary](docs/validation/V2_2_5_SIMULATION_SUMMARY.md) records it with the corrected-model diagnostics, which retain their original scope. The configured limit is 512 rows by 256 columns, and 256x256 still lacks a converged operating point. The completed [V2.2.2 screen](docs/design/timing/PHASED_CONTROL_V2_2_2.md) certifies only that earlier tree.

## Environment setup

The supplied [`environment.yml`](environment.yml) targets **Linux x86_64** and
pins Python 3.9.19, PySpice 1.5, and Xyce 7.4. Run the following commands in
Bash, from the repository root. Other platforms need a compatible Linux
environment, such as WSL2 on Windows.

If Miniconda is already installed, skip the installer and use its existing
`etc/profile.d/conda.sh`. Otherwise, download the Linux x86_64 installer from
the [Tsinghua mirror](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/):

```bash
CONDA_ROOT="$HOME/miniconda3"  # Choose a disk with several GB of free space.
NO_PROXY=mirrors.tuna.tsinghua.edu.cn no_proxy=mirrors.tuna.tsinghua.edu.cn \
  curl -fL https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -o /tmp/Miniconda3-Linux-x86_64.sh
sha256sum /tmp/Miniconda3-Linux-x86_64.sh
```

Compare the checksum with the matching installer in the
[official Miniconda index](https://repo.anaconda.com/miniconda/) before running it:

```bash
bash /tmp/Miniconda3-Linux-x86_64.sh -b -p "$CONDA_ROOT"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda init bash
conda config --set auto_activate_base false
```

`source` enables conda in the current shell; `conda init bash` enables it in
future Bash sessions.

For downloads in China, merge the following keys into `~/.condarc`, preserving
any other settings. This follows the
[Tsinghua conda mirror configuration](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/).
The `vlsida-eda` channel supplies Xyce and still uses its upstream URL.

```yaml
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

Create the project environment (skip creation if `openyield` already exists).
The command below uses the local proxy for
upstream downloads, bypasses it for Tsinghua, and directs the YAML's pip
dependencies to Tsinghua as well. The proxy must be running at
`127.0.0.1:2080`; adjust these values for your network.

```bash
HTTP_PROXY=http://127.0.0.1:2080 HTTPS_PROXY=http://127.0.0.1:2080 \
http_proxy=http://127.0.0.1:2080 https_proxy=http://127.0.0.1:2080 \
NO_PROXY=localhost,127.0.0.1,mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn \
no_proxy=localhost,127.0.0.1,mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn \
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  conda env create -f environment.yml
conda activate openyield
python --version
python -m pip check
command -v Xyce
Xyce -v
python main_sram.py --version
```

Activate `openyield` in each new terminal before running the compiler. If your
home partition is small, set `envs_dirs` and `pkgs_dirs` in `~/.condarc` to
directories on a larger disk before creating the environment. Optional layout
and optimization dependencies are described in their linked guides below.

[`environment.yml`](environment.yml) is an export of the configured
environment, including optional layout and document packages. It pins
`_x86_64-microarch-level=4`, so it solves only on x86-64-v4 (AVX-512) CPUs;
delete that line on older CPUs. [`requirements.txt`](requirements.txt) lists
its Python packages with version pins and the Tsinghua pip index. It does not
install Xyce or native libraries; use `environment.yml` for the conda setup.

## Quick start

Single-bank 512×512 layout diagnostics, extracted-load driver candidates and
TIME_CONTROL without intentional RC guards are described in the
[layout scaling record](docs/design/layout/LAYOUT_SCALING_V2_2_8.md). Automatic timing
retains the 512×256 envelope; 512×512 requires an explicit diagnostic period
and has no qualified operating point. The same record covers the split-array
centre floorplan that `layout bank --with-controller` uses by default.

Run commands from the repository root after completing the environment setup:

```bash
conda activate openyield
python main_sram.py simulate --rows 2 --cols 2 --operation read --variation-mode nominal --real-cell-mode 0 --no-run-xyce
```

This smoke check generates a deterministic full-transistor deck under
`outputs/main_sram/`. It checks environment and netlist construction only;
it does not simulate or validate waveforms.

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
NO_PROXY=pypi.tuna.tsinghua.edu.cn no_proxy=pypi.tuna.tsinghua.edu.cn \
  python -m pip install -r size_optimization/openyield_v2/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
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
