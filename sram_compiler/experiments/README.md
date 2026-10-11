# Unified experiments — V2.2.8

Use [`main_sram.py`](../../main_sram.py) for all supported experiments. Simulation uses the script settings as defaults; explicit command-line options override them. From the repository root, generate a full-transistor deck without Xyce:

```bash
python main_sram.py simulate \
  --rows 8 --cols 4 --target-row 7 --target-col 3 \
  --operation read --mc-runs 1 \
  --real-cell-mode 0 --no-run-xyce --output-dir outputs/example
```

Use `--no-run-xyce` for generation only and `--run-xyce` to simulate (the script default). Xyce must be on `PATH`, or pass `--xyce /path/to/Xyce`. The runner writes a configuration-specific directory under `--output-dir`, keeping `summary.json`, the generated deck and model cards, Xyce logs, measures, and waveform data when available. Failed attempts retain provenance. Per-device model audits are always saved (`--audit` remains accepted); `--no-waveform` omits `.PRINT` waveform output. Use `--help` for the full option list.

## Variation choices

| Mode | Result |
|---|---|
| `per-device` (automatic) | Independent Gaussian `vth0`, `u0`, and `voff` values for each retained MOS at the selected corner |
| `nominal` | Fixed corner with no random draw; use this for a deterministic baseline |
| `shared` | One random card shared by every device using a base model |
| `custom` | Process parameter table from the cell YAML |

The runner uses a fixed seed by default. To choose another repeatable sample and count:

```bash
python main_sram.py simulate \
  --rows 8 --cols 4 --operation write \
  --mc-runs 2 --seed 3 \
  --real-cell-mode 0 --output-dir outputs/example --run-xyce
```

On the CLI, `--seed random` (or in the Python API, `mc_seed=None`) makes even a single run a random sample. The selected PDK corner is fixed across its local draws. The default relative sigma is 5% for all three parameters; this is an illustrative mismatch model, without area scaling or calibrated correlation. Modes 1–4 only vary the transistors they retain, so use mode 0 for full-array mismatch coverage.

## Set circuit inputs

`--corner` accepts `TT`, `FF`, `SS`, `FS`, or `SF`. `--vdd` is in volts, `--temperature` in Celsius, and `--period` in seconds. `--period` is a diagnostic clock override; the default is the frozen lookup clock. `--timing-lookup` loads another class table. `--interconnect-config` loads a wire YAML mapping without rewriting `global.yaml`. The only supported topology is distributed. The supported array envelope is 512 rows by 256 columns.

The runner supports `read`, `write`, `read&write`, `hold_snm`, `read_snm`, and `write_snm`. `--target-row` and `--target-col` select the accessed cell. An omitted target defaults to the last row and column. Per-device mode cannot be combined with legacy `.STEP` geometry sweeps.

SNM operations construct a single cell, even when larger array dimensions are
configured. Their metadata reports `full_device_coverage=false`; use transient
mode 0 arrays for full-array mismatch coverage.

For Python callers:

```python
from sram_compiler.config_yaml.loader import load_config
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench

config = load_config(8, 4, "TT")
testbench = Sram6TCoreMcTestbench(
    config, corner="TT", mc_seed=3,
    choose_columnmux=False, real_cell_mode=0,
    sim_path="outputs/python_example",
)
circuit = testbench.create_testbench("read", 7, 3)
```

`experiments/simulation.py` implements the shared deck/result workflow. `config_yaml/loader.py` loads configuration in memory; `variation/netlist.py` specializes model cards and audits connectivity. Local waveform and MPI sampling scripts can live in ignored `tests/` and `dev/` workspaces; they are not needed to generate a deck.

A generated deck, passing measures, or one Monte Carlo draw does not establish read/write correctness or yield. Inspect waveforms and the reported margins. The [V2.2.5 record](../../docs/design/COMPILER_ENTRY_V2_2_5.md) describes the pending screen and the unresolved 256x256 operating point; [the compiler tutorial](../README.md) explains the full flow.

## Experiment commands

| Command | Workflow |
|---|---|
| `simulate` (also direct flags or no command) | 6T/10T read, write, read/write, and SNM; per-device mismatch by default |
| `equivalent` | Modes 0–4 accuracy and runtime comparison; nominal is its diagnostic comparison default |
| `layout ACTION` | FreePDK45 reference/full/scaled/compact macro generation, export, extraction, RC analysis and comparisons |
| `circuit-optimize` | Circuit-backed SA, PSO, CMA-ES, random, SMAC, CBO, RoSEOpt, CPN, TSSBO, MOBO, MOEA/D or NSGA-II |
| `circuit-experiment` | Interactive joint architecture/sizing or fixed-configuration experiment |
| `optimize --family FAMILY` | Offline evolutionary, Bayesian or proposed optimizer |
| `compare -- ARGS` | Offline algorithm comparison |
| `experiment` | Configured offline comparison campaign; supports `--dry-run` and `--resume-dir` |

```bash
python main_sram.py --help
python main_sram.py simulate --help
python main_sram.py simulate --cell-type SRAM_10T_CELL --rows 2 --cols 2 --no-run-xyce
python main_sram.py simulate --variation-mode nominal --operation hold_snm --no-run-xyce
python main_sram.py equivalent --sizes 16x16 --modes 0,1,4 --operations read,write --plot
python main_sram.py layout full --help
python main_sram.py circuit-optimize --algorithm sa --max-iter 10
python main_sram.py circuit-experiment
python main_sram.py optimize --family evolutionary -- --help
python main_sram.py compare -- --help
python main_sram.py experiment --dry-run
```

Offline optimizers need their optional dependencies from
[`size_optimization/openyield_v2/requirements.txt`](../../size_optimization/openyield_v2/requirements.txt).
Layout generation/extraction needs the tools described in the
[layout guide](../layout/README.md). These packages are loaded only for their
selected workflow. Simulation defaults remain 16×16, TT, a fixed seed, and
`MC_RUNS=None` using `global.yaml` for sampled modes; nominal always uses one run,
and custom mode uses its process-table row count.

`--column-mux` / `--no-column-mux` and `--w-rc` / `--no-w-rc` override mux and
local-stub settings. `--cell-6t PD_W PG_W PU_W L PD_MODEL PG_MODEL PU_MODEL`
applies 6T dimensions in metres without changing YAML. Run identity includes
cell geometry, model content, mux, RC, controller, PVT, timing and variation.
Identical reruns get a separate attempt directory. Summaries preserve failed
and interrupted simulation status and record actual full-device coverage.

Transient and SNM simulations save sample and statistics CSVs. SNM summaries report the mean margin only when every requested sample is finite; undefined or missing samples retain their evidence and reject the run.
