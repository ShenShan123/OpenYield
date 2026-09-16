# OpenYield V2.1.4: SRAM yield analysis and optimization

![](img/logo-cut-openyield.jpg)
**OpenYield** generates 6T and 10T SRAM netlists for Xyce and evaluates noise margin, delay, power, area, and yield. The repository includes transistor-level arrays, an equivalent-cell model for unused cells, selectable process-variation flows, and sizing/architecture optimization drivers.

The circuit generator models parasitic capacitance/resistance, leakage coupling, and variation in peripheral circuits such as sense amplifiers and write drivers.

The main simulation backend is Xyce. FreePDK45 model cards are included under `tran_models/`.

V2.1.0 selects the clock from a fixed row/column lookup table, using the same
round-up class strategy as driver sizing. The period stays fixed across cell
candidates and PVT samples. It also fixes narrow-array RC precharge overlap,
write-register initialization, the distributed output-latch enable, and yield
callers' measurement handling. See the [timing guide](sram_compiler/sizing/README.md#clock-classes-v210)
and [release review](docs/design/TIMING_LOOKUP_V2_1_0.md) for settings and validation limits.

V2.1.4 fixes a TIME control race and re-evidences both 6T clock ladders. The
write request registered on the clock edge that ends an access now passes a
hold latch that is transparent only while the wordline enable is low, so the
write enable no longer pulses (0.38 to 0.72 V at FF 1.1 V / −40 °C) while a
read wordline is still on. With the 250 ps output-margin rule of V2.1.3 the
shared 6T row budgets become 1800/1900/2100/2500/3600 ps (4.5 ns up to 32
rows) and 6T arrays with a column mux get their own variant (rows
1900/2000/2200/2700/3600 ps, 4.75 ns up to 32 rows). See the
[6T budget record](docs/design/TIMING_6T_BUDGET_V2_1_4.md). 10T clocks are
unchanged; the 10T evidence predates the TIME latch and is the next round.

V2.1.3 gives 10T cells their own evidenced timing budget. The V2.1.2 follow-up
had found the shared 4 ns class failing 10T with a column mux at SS 0.9 V /
125 °C; the V2.1.3 [evidence run](docs/design/TIMING_10T_BUDGET_V2_1_3.md)
found the 10T read port alone over that class and its penalty growing with
height, so `timing_lookup.json` now carries a `variants` entry for
`SRAM_10T_CELL` (rows 2000/2200/2400/3200/5600 ps, columns 200 ps above the
shared ladder: 5 ns up to 32 rows / 16 columns, 14 ns at 512 rows) with at
least 250 ps of output margin at every class bound. 6T decks, the shared
classes, driver sizes and checks are unchanged. The per-device pilot is
expanded to ten seeds. See the [timing guide](sram_compiler/sizing/README.md#clock-classes-v210).

V2.1.2 keeps the V2.1.1 circuit, timing and driver sizes. The per-device CLI accepts
`--vdd`/`--temperature`, run metadata records the Xyce installation, and the
timestep retry also bounds decks with a start time or coarse step ceiling. The
supplied [first-500 write-failure inventory](docs/issue_reports/write_failure_cases_first_500.md)
is audited: its entries are unverified numerical or solver-environment
non-completions, not observed write failures. The Phase 4/5
[timing follow-up](docs/design/TIMING_FOLLOWUP_V2_1_2.md) passes 30 of 35
screening cases; the shared 4 ns class fails for 10T with a column mux at
SS 0.9 V / 125 °C (it passes at 4.5 ns), and the table stays frozen.

V2.1.1 makes distributed RC the only signal-wire topology and the default. Explicit
star settings are rejected. Array, replica, bitline periphery, decoder fan-out,
write-data clock and mux-select paths use physical wire taps. The default
1-ohm / 0.1-fF pitch is illustrative, not extracted metal. See the
[V2.1.1 change and validation status](docs/design/DISTRIBUTED_ONLY_V2_1_1.md)
and [evaluation schedule](docs/plans/V2_1_1_TIMING_FOLLOWUP.md). Full retention checks
and CLI metric rejection are included; the V2.1.0 timing table and V2.0.9
transistor classes keep their historical identities.

V2.0.11 fixes V2.0.10's failure handling, measurement windows and evidence
retention, extends distributed wiring to the control lines that span an array
dimension (`PRE`, `w_en`, `w_en_bar`, `s_en`, `sa_iso`, `wl_en`), and adds the
first star-topology write and read waveform screen. Star-topology decks and
every driver size class are unchanged. See the
[V2.0.11 audit and write screen](docs/design/WRITE_VALIDATION_V211.md) for the
tested configurations and the remaining limits.

V2.0.7 corrects RC parameter propagation and equivalent-extraction context,
adds opt-in distributed wordline/bitline wiring, and matches the replica paths
and waveform probes to that topology. See the
[distributed RC guide](docs/design/DISTRIBUTED_RC_MODEL.md).
V2.0.8 audits that implementation, fixes its `cell_pin_rc` default and the
main-entrance configuration path, and extends the waveform checks to 8x4, 4x8
and 16x16 arrays; see the
[validation record](docs/design/DISTRIBUTED_RC_VALIDATION.md).
Fixed global corners plus independent per-device mismatch remain the default.
V2.0.9 sizes the critical drivers from a lookup table of fixed integer size
classes (`sram_compiler/sizing/sizing_lookup.json`), selected by row and column
class and interpolated on the class ladder for unseen arrays, so that a layout
library of fixed transistors can implement every array; the legacy `fixed` rule
mode is removed and `rules_only`/`auto` remain the rule basis and qualified-record
lookup. Extracted-array timing and yield qualification remain in progress. See the
[sizing guide](sram_compiler/sizing/README.md) and
[qualification status](docs/DRIVER_SIZING_PROPOSAL.md).

V2.0.10 corrects distributed precharge timing, sense-control load accounting,
read measurements and simulation failure handling. It retains the V2.0.9
lookup classes. See the [waveform review](docs/design/DISTRIBUTED_RC_V210_REVIEW.md)
for tested arrays and the wire configurations that remain unsafe.

Documentation: [compiler guide](sram_compiler/README.md),
[equivalent models](sram_compiler/equivalent_modeling/README.md),
[sizing optimization](size_optimization/README.md),
[yield estimation](yield_estimation/README.md), and
[plans and release history](docs/README.md).
Shared measurement, waveform, plotting, area, and SPICE helpers are documented
in the [utilities guide](utils/README.md).
Reusable compiler tests live in `tests/`; local development and qualification
scripts live under ignored `dev/`. See the [development guide](docs/DEVELOPMENT.md).

## Key Features

* **Xyce Integration:** Utilizes the Xyce parallel circuit simulator for transistor-level simulations.
* **Monte Carlo Simulation Support:**
  * Built-in Monte Carlo simulations within Xyce.
  * Support for user-defined Monte Carlo simulations, allowing for custom process parameter generation.
* **SRAM Cell Types:** Supports 6T and 10T SRAM cells.
* **Equivalent Circuit Modeling:** Fast approximate equivalent circuits for unused SRAM cells (5-capacitor parasitic model: `c_bl`, `c_blb`, `c_wl`, `c_wl_bl`, `c_wl_blb`) to speed up large-array simulation.
* **Performance Metrics Analysis:** Evaluates critical SRAM performance metrics:
  * Hold / Read / Write Static Noise Margin (SNM)
  * Read and Write Delay
  * Static and Dynamic Power
* **SRAM Sizing Optimization:** Integrated two-stage optimization for transistor sizing and architecture configuration.
* **Output Parsing and Waveform Plotting:** Includes parsers to extract simulation results and tools to visualize signal waveforms.
* **OpenYield V2 optimizers:** An isolated offline optimizer package under `size_optimization/openyield_v2/` with evolutionary, Bayesian, and surrogate-based methods.

![](img/openyield_all-overall.drawio.png)

## Dependencies

* **[FreePDK45](https://eda.ncsu.edu/freepdk/freepdk45/):** Required by SRAM circuit generator and Xyce simulator.
* **[PySpice](https://pyspice.fabrice-salvaire.fr/releases/v1.4/overview.html):** Required by SRAM circuit generator:

  ```bash
  pip install PySpice
  ```
* **[Xyce](https://xyce.sandia.gov/about-xyce/):** A SPICE simulator for fast simulation. Install using conda through vlsida channel (built for [OpenRAM](https://github.com/VLSIDA/OpenRAM.git)):

  ```bash
  conda install -q -y -c vlsida-eda trilinos
  conda install -q -y -c vlsida-eda xyce
  ```

  For building your own Xyce please refer to this [guide](https://xyce.sandia.gov/documentation-tutorials/building-guide/)
* **Python packages for the bundled circuit-backed optimizers** (install via pip; tSS-BO still needs its separate repository):

  ```bash
  pip install numpy scipy matplotlib pandas torch botorch gpytorch \
    smac ConfigSpace cma gymnasium scikit-learn tqdm tabpfn PyYAML
  ```
* **OpenYield V2 extras** (only needed for `size_optimization/openyield_v2/`):

  ```bash
  pip install -r size_optimization/openyield_v2/requirements.txt
  ```

## Usage Examples

### 0. Conda Environment Creation

Create the conda environment from the `yml` file:

```bash
conda env create -f environment.yml
conda activate openyield
```

Or update an existing environment:

```bash
conda env update -f environment.yml
```

### 1. SRAM Circuit Generator

The generation modules of each sub-circuit are located in `sram_compiler/subcircuits/`.

The simulation code is in `sram_compiler/testbenches/`.

Circuit and simulation parameters are configured through YAML files in `sram_compiler/config_yaml/`.

`main_sram.py` is the main entrance: edit its settings block and run
`python main_sram.py` to generate one array and simulate it with Xyce. It reads
the YAML files in memory and defaults to seeded per-device local mismatch over
the full transistor array. `python -m sram_compiler.per_device_mc.run` exposes
the same defaults as command-line options for scripted generation; add
`--run-xyce` to simulate. See the [compiler guide](sram_compiler/README.md)
for both workflows.

#### Configuration via YAML

Key parameters in `sram_compiler/config_yaml/global.yaml`:

```yaml
vdd: 1.0            # Supply voltage (V)
temperature: 27     # Temperature (Celsius)
num_rows: 16        # Number of SRAM rows
num_cols: 16        # Number of SRAM columns
monte_carlo_runs: 2 # Monte Carlo simulation runs
corner: TT          # Process corner (TT/FF/SS/FS/SF)
```

Transistor widths and models for each cell type are in:

- `sram_compiler/config_yaml/sram_6t_cell.yaml`
- `sram_compiler/config_yaml/sram_10t_cell.yaml`
- `sram_compiler/config_yaml/precharge.yaml`, `wordline_driver.yaml`, etc.

#### Running a Simulation

```bash
# Main entrance: edit the settings block at the top of the script first
python main_sram.py
# Command-line runner with the same defaults
python -m sram_compiler.per_device_mc.run --rows 8 --cols 4 --mc-runs 2 --run-xyce
```

Or programmatically:

```python
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
from config import SRAM_CONFIG
from PySpice.Unit import u_Ohm, u_pF

sram_config = SRAM_CONFIG()
sram_config.load_all_configs(
    global_file="sram_compiler/config_yaml/global.yaml",
    circuit_configs={
        "SRAM_6T_CELL": "sram_compiler/config_yaml/sram_6t_cell.yaml",
        "SRAM_10T_CELL": "sram_compiler/config_yaml/sram_10t_cell.yaml",
        "WORDLINEDRIVER": "sram_compiler/config_yaml/wordline_driver.yaml",
        "PRECHARGE": "sram_compiler/config_yaml/precharge.yaml",
        "COLUMNMUX": "sram_compiler/config_yaml/mux.yaml",
        "SENSEAMP": "sram_compiler/config_yaml/sa.yaml",
        "WRITEDRIVER": "sram_compiler/config_yaml/write_driver.yaml",
        "DECODER": "sram_compiler/config_yaml/decoder.yaml",
    }
)

mc_testbench = Sram6TCoreMcTestbench(
    sram_config,
    sram_cell_type="SRAM_6T_CELL",  # or "SRAM_10T_CELL"
    w_rc=True,
    pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
    vth_std=0.05,                  # relative sigma of vth0/u0/voff
    variation_mode='per-device',   # the default; 'nominal', 'shared', 'custom' are explicit
    mc_seed=20260711,              # reproducible sampling; None draws a new seed per run
    real_cell_mode=0,              # full array; 1-4 use the equivalent circuit for unused cells
                                   # (None, the default, takes global.yaml's `equivalent` block)
    corner='TT',
    sim_path='sim/',
)

# Transient analysis: 'write', 'read', or 'read&write'
delay, pavg, pstc, pdyn = mc_testbench.run_mc_simulation(
    operation='write',
    target_row=15, target_col=15,
    mc_runs=10,
    temperature=27,
)

# DC analysis: 'write_snm', 'hold_snm', 'read_snm'
snm = mc_testbench.run_mc_simulation(
    operation='read_snm',
    target_row=15, target_col=15,
    mc_runs=10,
    temperature=27,
)
```

Simulation outputs (netlists, waveforms, results) are saved to the `sim_path` directory.

### 2. Equivalent Circuit Modeling

For large arrays, unused SRAM cells can be replaced with a compact 5-capacitor equivalent circuit to reduce simulation time.

V2.1.5 makes this a simulation input. `global.yaml` carries the default:

```yaml
equivalent:
  mode: 0        # 0 = full transistor array (reference); 1-4 = equivalent cells
```

`--real-cell-mode` on the command line, the `real_cell_mode` testbench argument
and `REAL_CELL_MODE` in `main_sram.py` override it; `None` keeps the YAML value.
Mode `0` keeps the complete transistor array and is the only mode whose
per-device mismatch covers every cell, so modes 1–4 never carry qualification
evidence. Modes 1–4 call Xyce during netlist generation to extract the cell
parasitics, so Xyce must be on PATH even to build the deck.

To measure the approximation against the full transistor array:

```bash
python3 -m sram_compiler.equivalent_modeling.compare --sizes 16x16,32x32 --modes 0,1,4 --plot
```

Results land in `outputs/equivalent_modeling/<timestamp>/` (`result.csv`,
`result_diff.csv`, `settings.json`). See
[`sram_compiler/equivalent_modeling/README.md`](sram_compiler/equivalent_modeling/README.md)
for the model description and
[the V2.1.5 record](docs/design/EQUIVALENT_MODEL_V2_1_5.md) for measured errors.

#### Per-device process variation

The compiler's [per-device runner](sram_compiler/per_device_mc/README.md)
defaults to independent local mismatch and keeps circuit topology and process
variation as separate options:

```bash
python -m sram_compiler.per_device_mc.run \
  --rows 16 --cols 16 \
  --real-cell-mode 1 \
  --variation-mode per-device \
  --mc-runs 100 \
  --operation read \
  --output-dir outputs/per_device_mc \
  --run-xyce
```

Variation modes:

| Mode                     | Behavior                                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `nominal`              | No process variation                                                                                                           |
| `shared`               | Existing model-card Monte Carlo; devices sharing a base model share its random parameters                                      |
| `custom`               | Parameter-table flow using`process_parameters.vars` from the cell YAML; a one-dimensional 10T table is treated as one sample |
| `per-device` (default) | Independent`vth0`, `u0`, and `voff` expressions for every MOS retained in the generated netlist                          |

`--vth-std` is the relative standard deviation used for all three varied parameters; the default is `0.05`.

`real-cell-mode` is `0` (full array), `1` (target-row/target-column cross), `2` (target row), `3` (target column), or `4` (target cell); omitted, it takes the `equivalent` block of `global.yaml`, and the resolved mode is recorded in `summary.json`. In modes 1–4, replaced cells are represented by the existing equivalent circuit; retained cells and peripheral MOS devices receive per-device variation. Write simulation is available in all five modes. In modes 3 and 4, the target cell write transition remains transistor-level, while replaced cells contribute the equivalent RC and WL-controlled static-power model. Their internal write state and whole-row dynamic write power should therefore be treated as approximations rather than full-array transistor-level results.

Generation is the default. Add `--run-xyce` to simulate or `--audit` to save model counts and hierarchy details. The runner accepts `read`, `write`, `read&write`, `hold_snm`, `read_snm`, and `write_snm`.

`.PRINT` output is retained by default. After a successful Xyce run, the same directory contains `deck.sp.prn` and `waveform.png`. `--no-waveform` is available for transient operations; SNM calculations require the DC waveform and reject that option.

Each parameter set gets a deterministic subdirectory under `--output-dir`. Different circuit and variation modes remain separate. Repeating the same configuration refreshes only runner-generated files in that directory, so old PRN or measurement files cannot be mistaken for the current run.

### 3. SRAM Sizing Optimization

OpenYield includes a suite of optimization algorithms for SRAM transistor sizing and architecture configuration. All algorithms share a common interface via `size_optimization/exp_utils.py`.

#### Available Algorithms

| Algorithm | Script              | Description                                                    |
| --------- | ------------------- | -------------------------------------------------------------- |
| SA        | `demo_sa.py`      | Simulated Annealing                                            |
| PSO       | `demo_pso.py`     | Particle Swarm Optimization                                    |
| CBO       | `demo_cbo.py`     | Constrained Bayesian Optimization                              |
| RoSE-Opt  | `demo_roseopt.py` | Reinforcement Learning Enhanced BO                             |
| CMA-ES    | `demo_cmaes.py`   | Covariance Matrix Adaptation Evolution Strategy                |
| SMAC      | `demo_smac.py`    | Sequential Model-based Algorithm Configuration                 |
| NSGA-II   | `demo_nsgaii.py`  | Multi-Objective Genetic Algorithm                              |
| MOEAD     | `demo_moead.py`   | Multi-Objective Evolutionary Algorithm based on Decomposition  |
| MOBO      | `demo_mobo.py`    | Multi-Objective Bayesian Optimization                          |
| CPN       | `demo_cpn.py`     | TabPFN-based Bayesian Optimization (requires`tabpfn`)        |
| tSS-BO    | `demo_tssbo.py`   | Truncated Subspace Sampling BO (requires separate tSS-BO repo) |
| Random    | `demo_random.py`  | Random Search (baseline)                                       |

#### Running an Optimization

```bash
cd /path/to/OpenYield
python size_optimization/demo_sa.py        # Simulated Annealing
python size_optimization/demo_pso.py       # PSO
python size_optimization/demo_cbo.py       # Constrained BO
```

#### Two-Stage Optimization (Architecture + Sizing)

For joint architecture and transistor sizing optimization:

```bash
python size_optimization/experiment.py
```

This runs a two-stage flow:

1. Stage 1 (SMAC): Search over architecture configurations (rows, cols, arrays).
2. Stage 2: Optimize transistor sizing for the best architecture candidates.

#### Optimization Parameter Space

The parameter space is defined in `size_optimization/exp_utils.py`:

- **`ModifiedSRAMParameterSpace`**: 7-dimensional bitcell transistor sizing space.
- **`CompositeSRAMParameterSpace`**: 24-dimensional joint space (bitcell + peripheral circuits).

#### OpenYield V2 offline optimizers

`size_optimization/openyield_v2/` adds a separate surrogate-optimization path without changing the circuit generator or the existing optimization scripts. It includes NSGA2, SPEA2, UNSGA3, CTAEA, GPBO, PAREGO, MACE, and the proposed coarse-search/refinement method.

The package reads `datasets/train_6t.csv` and `datasets/train_10t.csv`. These are static TT/25 °C samples generated with the equivalent circuit enabled and per-device variation disabled; they are not current per-device Monte Carlo results.

`train_10t.csv` also predates the V2.1.5 10T pull-down resize: its `pd_width` column spans 164-246 nm, the old YAML bounds, while the tracked 10T cell is now 287 nm with bounds 230-344 nm. The offline 10T surrogate therefore describes a design space that barely overlaps the current default, and a 10T run of this package must either regenerate the dataset or be read as a study of the old cell. The 6T dataset is unaffected.

```bash
python -m size_optimization.openyield_v2.run_experiment --dry-run
python -m size_optimization.openyield_v2.run_experiment
```

See [`size_optimization/openyield_v2/README.md`](size_optimization/openyield_v2/README.md) for algorithm selection, budgets, and output files.

### 4. SRAM Yield Estimation Algorithms

OpenYield includes SRAM yield estimators based on Monte Carlo and importance sampling.

#### Available Algorithms

- **MC**: Monte Carlo
- **MNIS**: Mean-shifted Importance Sampling
- **ACS**: Adaptive Compressed Sampling
- **AIS**: Adaptive Importance Sampling
- **HSCS**: High-dimensional Sparse Compressed Sampling

## Project Structure

```
OpenYield/
├── main_sram.py                  # Main entrance: one array, seeded per-device mismatch, YAML read in memory
├── config.py                     # Compatibility re-export of the YAML loader
├── utils/                        # Shared runtime utilities (legacy imports preserved)
│   ├── measurements.py           # Monte Carlo measurement parsing and statistics
│   ├── waveforms.py              # Xyce PRN loading and sample splitting
│   ├── plotting.py               # Waveforms, SRAM comparisons, and optimizer plots
│   ├── area.py                   # Bitcell, array, and macro area estimates
│   └── spice.py                  # SPICE model parsing and writing
├── environment.yml               # Conda environment specification
├── docs/
│   ├── README.md                 # Plans and release history index
│   ├── DRIVER_SIZING_PROPOSAL.md # Full working design and qualification status
│   ├── TIMING_AUTOCONFIG.md      # Original timing proposal
│   ├── CHANGELOG.md              # Release history
│   ├── DEVELOPMENT.md            # Regression checks and local development tools
│   └── design/                  # Archived original design proposal
├── sram_compiler/
│   ├── README.md                 # Compiler and simulation guide
│   ├── CIRCUIT_REVIEW.md         # Circuit review and verification evidence
│   ├── config_yaml/              # YAML configuration files for all circuits
│   ├── per_device_mc/            # Default local mismatch generation and execution
│   │   ├── run.py                # CLI and in-memory configuration helper
│   │   ├── netlist.py            # Independent model cards for retained MOS devices
│   │   └── sampling.py           # Materialized local draws for MPI runs
│   ├── sizing/                  # Driver rules, timing, and qualified-table lookup
│   ├── subcircuits/              # Circuit generation modules (6T, 10T, peripherals)
│   └── testbenches/              # Simulation testbench classes
├── tests/                       # Reusable compiler regression tests
├── dev/                         # Local experiments and qualification tools (ignored)
├── size_optimization/
│   ├── README.md                 # Optimization entry points
│   ├── 电路算法说明文档.md        # Detailed sizing algorithm guide
│   ├── exp_utils.py              # Shared optimization utilities and parameter spaces
│   ├── experiment.py             # Two-stage optimization driver
│   ├── demo_sa.py                # Simulated Annealing
│   ├── demo_pso.py               # Particle Swarm Optimization
│   ├── demo_cbo.py               # Constrained Bayesian Optimization
│   ├── demo_roseopt.py           # RoSE-Opt
│   ├── demo_cmaes.py             # CMA-ES
│   ├── demo_smac.py              # SMAC
│   ├── demo_nsgaii.py            # NSGA-II
│   ├── demo_moead.py             # MOEAD
│   ├── demo_mobo.py              # Multi-Objective BO
│   ├── demo_cpn.py               # CPN (TabPFN-based BO)
│   ├── demo_tssbo.py             # tSS-BO
│   ├── demo_random.py            # Random search baseline
│   ├── NSGA-II/                  # NSGA-II implementation
│   ├── MOBO/                     # MOBO implementation
│   ├── moead/                    # MOEAD implementation
│   └── openyield_v2/             # Offline evolutionary/Bayesian optimizer package and datasets
├── tran_models/                  # FreePDK45 transistor model files
└── yield_estimation/             # Yield estimation algorithms
```

## Important Notes

* Ensure Xyce is installed and available in your system PATH.
* The circuit generator, per-device runner, equivalent-model scripts, and OpenYield V2 package resolve repository data from the project root. Legacy `yield_estimation/` demos still contain their original machine-local paths and were not changed in this integration.
* FreePDK45 model files are included in `tran_models/`.
* Simulation output directories (`sim/`, `sim1/`, `outputs/`) are created automatically and are excluded from git.

## Contributing

Contributions and reproducible issue reports are welcome.

# Cite Us

```LaTeX
@INPROCEEDINGS{OpenYield,
  author={Shen, Shan and Li, Xingyang and Liu, Zhuohua and Ma, Junhao and Wang, Yikai and Wu, Yiheng and Sun, Yuquan and Xing, Wei W.},
  booktitle={2025 IEEE 43rd International Conference on Computer Design (ICCD)},
  title={OpenYield: An Open-Source SRAM Yield Analysis and Optimization Benchmark Suite},
  year={2025},
  volume={},
  number={},
  pages={167-175},
  doi={10.1109/ICCD65941.2025.00030}
}

```
