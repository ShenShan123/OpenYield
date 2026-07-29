# OpenYield: SRAM yield analysis and optimization
![](img/logo-cut-openyield.jpg)
**OpenYield** generates 6T and 10T SRAM netlists for Xyce and evaluates noise margin, delay, power, area, and yield. The repository includes transistor-level arrays, an equivalent-cell model for unused cells, selectable process-variation flows, and sizing/architecture optimization drivers.

The circuit generator models parasitic capacitance/resistance, leakage coupling, and variation in peripheral circuits such as sense amplifiers and write drivers.

The main simulation backend is Xyce. FreePDK45 model cards are included under `tran_models/`.

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

The main simulation entry point is `main_sram.py`.

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
python main_sram.py
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
    vth_std=0.05,
    mc=True,
    real_cell_mode=1,  # use the equivalent circuit for unused cells
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

Set `real_cell_mode=1` (or modes 2–4) when creating the testbench. Mode `0` keeps the complete transistor array.

To analyze and characterize the equivalent model for different array sizes:

```bash
python equivalent_modeling/main_sram.py
```

This compares simulation results with and without the equivalent model across different array configurations. See [`等效电路说明文档.md`](等效电路说明文档.md) for the model description.

#### Per-device process variation

`per_device_mc/run.py` keeps circuit topology and process variation as separate options:

```bash
python per_device_mc/run.py \
  --rows 16 --cols 16 \
  --real-cell-mode 1 \
  --variation-mode per-device \
  --mc-runs 100 \
  --operation read \
  --output-dir outputs/per_device_mc \
  --run-xyce
```

Variation modes:

| Mode | Behavior |
|------|----------|
| `nominal` | No process variation |
| `shared` | Existing model-card Monte Carlo; devices sharing a base model share its random parameters |
| `custom` | Parameter-table flow using `process_parameters.vars` from the cell YAML; a one-dimensional 10T table is treated as one sample |
| `per-device` | Independent `vth0`, `u0`, and `voff` expressions for every MOS retained in the generated netlist |

`--vth-std` is the relative standard deviation used for all three varied parameters; the default is `0.05`.

`real-cell-mode` remains `0` (full array), `1` (target-row/target-column cross), `2` (target row), `3` (target column), or `4` (target cell). In modes 1–4, replaced cells are represented by the existing equivalent circuit; retained cells and peripheral MOS devices receive per-device variation. Write simulation is available in all five modes. In modes 3 and 4, the target cell write transition remains transistor-level, while replaced cells contribute the equivalent RC and WL-controlled static-power model. Their internal write state and whole-row dynamic write power should therefore be treated as approximations rather than full-array transistor-level results.

Generation is the default. Add `--run-xyce` to simulate or `--audit` to save model counts and hierarchy details. The runner accepts `read`, `write`, `read&write`, `hold_snm`, `read_snm`, and `write_snm`.

`.PRINT` output is retained by default. After a successful Xyce run, the same directory contains `deck.sp.prn` and `waveform.png`. `--no-waveform` is available for transient operations; SNM calculations require the DC waveform and reject that option.

Each parameter set gets a deterministic subdirectory under `--output-dir`. Different circuit and variation modes remain separate. Repeating the same configuration refreshes only runner-generated files in that directory, so old PRN or measurement files cannot be mistaken for the current run.

### 3. SRAM Sizing Optimization

OpenYield includes a suite of optimization algorithms for SRAM transistor sizing and architecture configuration. All algorithms share a common interface via `size_optimization/exp_utils.py`.

#### Available Algorithms

| Algorithm | Script | Description |
|-----------|--------|-------------|
| SA | `demo_sa.py` | Simulated Annealing |
| PSO | `demo_pso.py` | Particle Swarm Optimization |
| CBO | `demo_cbo.py` | Constrained Bayesian Optimization |
| RoSE-Opt | `demo_roseopt.py` | Reinforcement Learning Enhanced BO |
| CMA-ES | `demo_cmaes.py` | Covariance Matrix Adaptation Evolution Strategy |
| SMAC | `demo_smac.py` | Sequential Model-based Algorithm Configuration |
| NSGA-II | `demo_nsgaii.py` | Multi-Objective Genetic Algorithm |
| MOEAD | `demo_moead.py` | Multi-Objective Evolutionary Algorithm based on Decomposition |
| MOBO | `demo_mobo.py` | Multi-Objective Bayesian Optimization |
| CPN | `demo_cpn.py` | TabPFN-based Bayesian Optimization (requires `tabpfn`) |
| tSS-BO | `demo_tssbo.py` | Truncated Subspace Sampling BO (requires separate tSS-BO repo) |
| Random | `demo_random.py` | Random Search (baseline) |

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
├── main_sram.py                  # Main simulation entry point
├── config.py                     # Centralized YAML config loader
├── utils.py                      # Result parsing, waveform plotting, and area utilities
├── environment.yml               # Conda environment specification
├── per_device_mc/
│   ├── run.py                    # Variation-mode runner and Xyce entry point
│   └── netlist.py                # Independent model cards for retained MOS devices
├── sram_compiler/
│   ├── config_yaml/              # YAML configuration files for all circuits
│   ├── subcircuits/              # Circuit generation modules (6T, 10T, peripherals)
│   └── testbenches/              # Simulation testbench classes
├── size_optimization/
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
├── equivalent_modeling/
│   └── main_sram.py              # Equivalent circuit analysis script
├── 等效电路说明文档.md            # Equivalent circuit modes and accuracy boundary
├── tran_models/                  # FreePDK45 transistor model files
└── yield_estimation/             # Yield estimation algorithms
```

## Important Notes

* Ensure Xyce is installed and available in your system PATH.
* The circuit generator, per-device runner, equivalent-model scripts, and OpenYield V2 package resolve repository data from the project root. Legacy `yield_estimation/` demos still contain their original machine-local paths and were not changed in this integration.
* FreePDK45 model files are included in `tran_models/`.
* Simulation output directories (`sim/`) are created automatically and are excluded from git.

## Contributing

Contributions and reproducible issue reports are welcome.
