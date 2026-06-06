# SRAM Compiler and Test Platform User Guide

This document introduces the basic usage of the SRAM compiler, simulation flow, Monte Carlo testing, waveform plotting, and result statistics. It mainly covers the following files and directories:

- `main_sram.py`: Entry script for a single SRAM simulation. This is the recommended starting point for new users.
- `utils.py`: Utilities for result parsing, statistics generation, waveform plotting, area estimation, and SPICE model read/write operations.
- `sram_compiler/`: Core code for SRAM configuration, subcircuit generation, and testbench construction.

## 1. What This Platform Can Do

This platform uses PySpice to generate transistor-level SRAM netlists and calls Xyce to run simulations. It currently supports:

- 6T / 10T SRAM cell, array, and complete peripheral circuit modeling.
- Read, write, and combined read/write transient simulations.
- Hold / read / write SNM DC analysis.
- Xyce Monte Carlo process variation simulation.
- Custom process-parameter table simulation.
- RC parasitic modeling for wordline, bitline, and other key nodes.
- Equivalent-model acceleration for non-target cells.
- Read/write delay, power, SNM, and area statistics.

## 2. Environment Setup

It is recommended to use the conda environment provided by the project:

```bash
conda env create -f environment_openyield.yml
conda activate openyield
```

If the environment already exists, update it with:

```bash
conda env update -f environment_openyield.yml
```

It is recommended to check that the following commands work:

```bash
Xyce -v
python -c "import PySpice, pandas, numpy, matplotlib"
```

## 3. Directory Structure

```text
OpenYield/
├── main_sram.py                         # Entry script for a single SRAM simulation
├── utils.py                             # Utilities for parsing, statistics, plotting, and area estimation
├── tran_models/                         # TT/FF/SS/FS/SF SPICE model files
├── sram_compiler/
│   ├── config_yaml/                     # Global and module-level YAML parameter files
│   │   ├── global.yaml
│   │   ├── config.py                    # YAML loader that converts data into dot-accessible config objects
│   │   ├── sweep_config.py              # Parameter sweep configuration mapping
│   │   ├── sram_6t_cell.yaml
│   │   ├── sram_10t_cell.yaml
│   │   ├── wordline_driver.yaml
│   │   ├── precharge.yaml
│   │   ├── mux.yaml
│   │   ├── sa.yaml
│   │   ├── write_driver.yaml
│   │   └── decoder.yaml
│   ├── subcircuits/                     # SRAM cells, peripheral circuits, decoders, and other subcircuits
│   │   ├── base_subcircuit.py           # Base subcircuit class with shared functions such as RC insertion
│   │   ├── standard_cell.py             # Standard cells such as inverters, NAND, AND, buffers, and D latches
│   │   ├── sram_6t_core.py              # 6T SRAM bitcell and array generation
│   │   ├── sram_10t_core.py             # 10T SRAM bitcell and array generation
│   │   ├── sram_cell_add_equivalent.py  # Equivalent C/R model extraction and replacement for idle cells
│   │   ├── dummy_row_or_column.py       # Dummy row / dummy column generation
│   │   ├── replica_column.py            # Replica column generation for timing control and reference paths
│   │   ├── precharge_and_write_driver.py # Precharge circuit and write driver
│   │   ├── mux_and_sa.py                # Column mux and sense amplifier
│   │   ├── wordline_driver.py           # Wordline driver
│   │   ├── decoder.py                   # Row decoder and cascaded decoder structures
│   │   └── time_generate.py             # Clock, delay chain, flip-flop, and control timing generation
│   └── testbenches/                     # Testbenches, MC simulation, SNM processing, and YAML updates
│       ├── base_testbench.py            # Base testbench defining power, PDK, default timing, and simulation APIs
│       ├── sram_6t_core_testbench.py    # SRAM array functional testbench that builds read/write peripherals and the full test circuit
│       ├── sram_6t_core_MC_testbench.py # Monte Carlo testbench that generates Xyce MC netlists and parses results
│       ├── parameter_factor.py          # Factory methods that create subcircuit instances from YAML configs
│       ├── snm.py                       # SNM curve parsing, crossing detection, and statistics table generation
│       └── yaml_change.py               # In-place YAML updates and delay/power summary from CSV files
└── sim1/                                # Default output directory for main_sram.py
```

## 4. Quick Start: Run One SRAM Simulation

Run:

```bash
python main_sram.py
```

The execution flow of `main_sram.py` is:

1. Call `update_global_yaml_inplace()` to modify the array size and column mux switch in `global.yaml`.
2. Call `update_sram6t_yaml_inplace()` to modify 6T cell parameters in `sram_6t_cell.yaml`.
3. Load all YAML configuration files.
4. Create a timestamped output directory, such as `sim1/20260529_223000_mc_6t/`.
5. Estimate the SRAM bitcell area.
6. Build `Sram6TCoreMcTestbench`.
7. Run read, write, or SNM simulation according to `operation`.
8. Save the netlist, Xyce outputs, CSV statistics, and waveform plots.

Note: `main_sram.py` currently writes configuration values back to the YAML files. Before running it, make sure the parameters at the top of the script are the values you want. You can also disable the in-place update flow: comment out lines 12-26 and then modify all parameters directly in the YAML files. Steps 1 and 2 mainly demonstrate that `ruamel.yaml` can modify YAML files without breaking their formatting; this feature may be useful for updating parameters in optimization scripts.

## 5. Modify the Array Size

Modify the following in `main_sram.py` or `global.yaml`:

```python
global_config_update = [16, 16, False]
```

The meaning is:

```text
[num_rows, num_cols, choose_columnmux]
```

For example:

```python
global_config_update = [64, 128, False]
```

This builds a `64 x 128` SRAM array and does not enable column mux. Column mux is temporarily unsupported because the timing has not been fully configured.

## 6. Modify the 6T SRAM Cell Size

Modify the following in `main_sram.py` or `global.yaml`:

```python
sram6t_config_update = [
    2.05e-7, 1.35e-7, 9.0e-8, 50.0e-9,
    "NMOS_VTG", "NMOS_VTG", "PMOS_VTG"
]
```

The parameter order is:

```text
[pd_width, pg_width, pu_width, length, pd_model, pg_model, pu_model]
```

Where:

- `pd_width`: Pull-down NMOS width.
- `pg_width`: Pass-gate NMOS width.
- `pu_width`: Pull-up PMOS width.
- `length`: Channel length.
- `NMOS_VTL / NMOS_VTG / NMOS_VTH`: Low-Vt, regular-Vt, and high-Vt NMOS models.
- `PMOS_VTL / PMOS_VTG / PMOS_VTH`: Low-Vt, regular-Vt, and high-Vt PMOS models.

## 7. Modify Process Corner, Voltage, Temperature, and MC Runs

These parameters are located in:

```text
sram_compiler/config_yaml/global.yaml
```

Common fields:

```yaml
vdd: 1.0
temperature: 25
corner: "TT"
sram_cell_type: "SRAM_6T_CELL"
num_rows: 16
num_cols: 16
monte_carlo_runs: 1
choose_columnmux: false
pdk_path_TT: "tran_models/models_TT.spice"
pdk_path_FF: "tran_models/models_FF.spice"
pdk_path_SS: "tran_models/models_SS.spice"
pdk_path_FS: "tran_models/models_FS.spice"
pdk_path_SF: "tran_models/models_SF.spice"
```

Available `corner` values:

```text
TT, FF, SS, FS, SF
```

## 8. Configuration File Description

Each YAML file under `sram_compiler/config_yaml/` corresponds to one module:

| File | Purpose |
| --- | --- |
| `global.yaml` | Global simulation parameters, PVT, array size, MC runs, and PDK paths |
| `sram_6t_cell.yaml` | 6T cell sizes, threshold models, sweep vectors, and process-parameter table |
| `sram_10t_cell.yaml` | 10T cell sizes, threshold models, sweep vectors, and process-parameter table |
| `wordline_driver.yaml` | Wordline driver NAND/INV sizes and models |
| `precharge.yaml` | Precharge PMOS size and model |
| `mux.yaml` | Column mux size and model |
| `sa.yaml` | Sense amplifier size and model |
| `write_driver.yaml` | Write driver size and model |
| `decoder.yaml` | Decoder NAND/INV sizes and models |

Each module parameter usually contains:

- `value`: Default value used by the current simulation.
- `value_sweep`: Candidate values used for parameter sweep.
- `upper` / `lower`: Upper and lower bounds for optimization or constraints.
- `choices`: Discrete model options, such as `NMOS_VTL / NMOS_VTG / NMOS_VTH`.

## 9. Select the Simulation Type

Modify the following in `main_sram.py`:

```python
operation = 'write'
```

Supported values:

```text
read        # Read transient simulation
write       # Write transient simulation
read&write  # Combined read/write transient simulation
hold_snm    # Hold SNM
read_snm    # Read SNM
write_snm   # Write SNM
```

Read and write operations generate `.data.csv` and `.stats.csv`. SNM operations extract noise margin values from `.prn` waveform data.

## 10. Key Testbench Switches

The commonly used parameters of `Sram6TCoreMcTestbench` in `main_sram.py` are:

```python
mc_testbench = Sram6TCoreMcTestbench(
    sram_config,
    sram_cell_type=sram_cell_type,
    w_rc=True,
    pi_res=100 @ u_Ohm,
    pi_cap=0.001 @ u_pF,
    vth_std=0.05,
    mc=True,
    custom_mc=False,
    sweep_cell=False,
    sweep_precharge=False,
    sweep_senseamp=False,
    sweep_wordlinedriver=False,
    sweep_columnmux=False,
    sweep_writedriver=False,
    sweep_decoder=False,
    corner=corner,
    choose_columnmux=choose_columnmux,
    use_equivalent=True,
    q_init_val=0,
    sim_path=sim_path,
)
```

Meanings:

- `w_rc=True`: Add RC parasitic networks to key nodes.
- `pi_res` / `pi_cap`: Resistance and capacitance values of each segment in the pi-shaped RC network.
- `vth_std=0.05`: Gaussian perturbation ratio for `vth0/u0/voff` in Xyce built-in MC.
- `mc=True`: Enable Xyce Monte Carlo.
- `custom_mc=True`: Use a custom parameter table instead of Xyce built-in random sampling.
- `sweep_*`: Enable parameter sweep for the corresponding module.
- `use_equivalent=True`: Use equivalent models to accelerate non-target cell simulation.
- `q_init_val`: Initial stored value of the target cell.

## 11. Output Files

After one run, the output directory is similar to:

```text
sim1/20260529_223000_mc_6t/
```

Common files:

```text
tmp_mc.spice                         # Temporary model file with Monte Carlo expressions
mc_write_16x16_rc1_tb.sp             # Generated Xyce netlist
mc_write_16x16_rc1_tb.sp.prn         # Waveform output
mc_write_16x16_rc1_tb.sp.mt0         # Measure results of MC run 0
mc_write_16x16_rc1_tb.data.csv       # Raw measurement data of all MC samples
mc_write_16x16_rc1_tb.stats.csv      # Statistics such as mean, standard deviation, and percentiles
mc_write_waveform.png                # Key-node waveform plot
```

Read delay summary formula:

```text
Delay = TDECODER + TPRCH + TSA + TSWING + TS_EN + TWLDRV
Power = PSTC + PDYN
```

Write delay summary formula:

```text
Delay = TDECODER + TWDRV + TWLDRV + TWRITE_Q
Power = PSTC + PDYN
```

At the end, `main_sram.py` prints:

```text
[OUTPUT] y[0]=Delay, y[1]=Power, y[2]=Area
```

## 12. Common `utils.py` Tools

`utils.py` provides the following utilities:

- `parse_mc_measurements()`: Parse Xyce `.mt0/.mt1/...` measurement files.
- `generate_mc_statistics()`: Generate statistics such as mean, standard deviation, percentiles, skewness, and kurtosis.
- `save_mc_results()`: Save `.data.csv` and `.stats.csv`.
- `read_prn_with_preprocess()`: Read Xyce `.prn` waveform files.
- `process_simulation_data()`: Split MC waveform data and generate PNG plots.
- `estimate_bitcell_area()`: Estimate 6T/10T bitcell area.
- `estimate_array_area()` / `estimate_total_macro_area()`: Estimate array or macro area.
- `parse_spice_models()` / `write_spice_models()`: Read and write SPICE model cards.

## 13. Parameter Sweep

The testbench supports sweep for the following modules:

```python
sweep_cell=True
sweep_precharge=True
sweep_senseamp=True
sweep_wordlinedriver=True
sweep_columnmux=True
sweep_writedriver=True
sweep_decoder=True
```

Sweep vectors usually come from the `value_sweep` fields in the YAML files. The sweep configuration is managed by:

```text
sram_compiler/config_yaml/sweep_config.py
```

Note that the number of sweep points must match the Monte Carlo run count set in `global.yaml`.

### 13.1 Custom Process Variation

The platform supports two process variation modes:

- `custom_mc=False`: Use Xyce built-in random sampling. The program automatically rewrites `vth0`, `u0`, and `voff` in the PDK model into `AGAUSS(...)` expressions according to `vth_std`, and generates `tmp_mc.spice`.
- `custom_mc=True`: Use a user-provided process-parameter table. The program writes `vars` into a `.data table` file and uses `.STEP data=table` to make Xyce simulate one row at a time.

To use custom process variation, set the testbench initialization parameters in `main_sram.py` as follows:

```python
mc_testbench = Sram6TCoreMcTestbench(
   ...
    mc=True,
    custom_mc=True,
    ...
)
```

At the same time, `run_mc_simulation()` must receive `vars`. `vars` comes from `process_parameters.vars` in `sram_compiler/config_yaml/sram_6t_cell.yaml` and is read in `main_sram.py`:

```python
vars = sram_config.sram_6t_cell.process_parameters.vars
```

Then pass it to `run_mc_simulation()`:

```python
data_csv_path = mc_testbench.run_mc_simulation(
    operation=operation,
    target_row=num_rows - 1,
    target_col=num_cols - 1,
    mc_runs=num_mc,
    temperature=temperature,
    vars=vars,
)
```

`vars` must be a two-dimensional array with the shape:

```text
(mc_runs, number_of_parameters)
```

In other words, the number of rows in `vars` must equal `monte_carlo_runs` in `global.yaml`; each row corresponds to one simulation sample.

#### 13.1.1 `vars` Format for 6T Cells

A 6T cell has 6 transistors, and each transistor has 3 process parameters:

```text
vth0, u0, voff
```

Therefore, each row for a single 6T cell needs `6 x 3 = 18` parameters. The order is:

```text
PGL(vth0, u0, voff),
PGR(vth0, u0, voff),
PDL(vth0, u0, voff),
PUL(vth0, u0, voff),
PDR(vth0, u0, voff),
PUR(vth0, u0, voff)
```

Example:

```python
vars = [
    [
        0.4106, 0.045, -0.13,
        0.4106, 0.045, -0.13,
        0.4106, 0.045, -0.13,
        -0.3842, 0.02, -0.126,
        0.4106, 0.045, -0.13,
        -0.3842, 0.02, -0.126,
    ],
    [
        0.4206, 0.054, -0.13,
        0.4206, 0.045, -0.13,
        0.4206, 0.045, -0.13,
        -0.3842, 0.02, -0.126,
        0.4206, 0.045, -0.13,
        -0.3842, 0.02, -0.126,
    ],
]
```

The example above has 2 rows, so set:

```yaml
monte_carlo_runs: 2
```

#### 13.1.2 `vars` Format for 10T Cells

A 10T cell has 10 transistors, and each transistor has 3 process parameters. Therefore, each row for a single 10T cell needs `10 x 3 = 30` parameters. The order is:

```text
PGL, PGR, PDL1, PDL2, PUL, PDR1, PDR2, PUR, FD_L, FD_R
```

Inside each transistor group, the order is still:

```text
vth0, u0, voff
```

When using custom process variation for 10T, `vars` must also be a two-dimensional array. If the YAML file contains only one row with 30 numbers, write it as:

```yaml
process_parameters:
  vars:
    - [0.4106, 0.045, -0.13, ...]
```

Do not write it directly as a one-dimensional list.

#### 13.1.3 Single-Cell Parameters vs. Full-Array Parameters

Each row of `vars` can have either of the following lengths:

- Number of parameters for one cell: 18 for 6T, 30 for 10T.
- Total number of parameters after expanding all active cells: `18 x active_cell_num` for 6T and `30 x active_cell_num` for 10T.

For `read`, `write`, and `read&write`, `active_cell_num = num_rows x num_cols`. If each row only provides parameters for one cell, the program automatically copies them to all active cells.

For `hold_snm`, `read_snm`, and `write_snm`, `active_cell_num = 1`, so only the target cell parameters are needed.

#### 13.1.4 Generated Files

After enabling `custom_mc=True`, the simulation directory contains an additional file similar to:

```text
mc_write_table.data
```

This file is the `.data table` used by Xyce. The generated netlist contains:

```spice
.STEP data=table
```

If the following errors appear:

```text
vars row mismatch
vars col mismatch
```

They usually mean that the number of rows in `vars` is not equal to `mc_runs`, or the number of parameters in each row is not 18 / `18 x active_cell_num` for 6T, or not 30 / `30 x active_cell_num` for 10T.

## 14. FAQ

### 14.1 Xyce Not Found

Check:

```bash
Xyce -v
```

If the command does not exist, install Xyce and make sure it is available in `PATH`.

### 14.2 PDK File Not Found

This error usually comes from:

```text
Transistor model file not found
```

Check `global.yaml`:

```yaml
pdk_path_TT: "tran_models/models_TT.spice"
```

Also make sure the corresponding file actually exists.

### 14.3 Simulation Is Slow

Try the following first:

- Reduce `num_rows` / `num_cols`.
- Set `monte_carlo_runs` to 1.
- Enable `use_equivalent=True`.
- For large arrays, run only one of `read` or `write` first, instead of enabling all sweep switches at the beginning.
- Keep `w_rc=False` for quick functional verification, then enable RC later.

### 14.4 `.data.csv` Contains NaN or Negative Delay

Possible causes:

- The trigger or target voltage was not reached in one MC sample.
- The measurement window is not suitable.
- RC loading is too strong in a large array, so some nodes did not switch.
- Process perturbation is too large.

You can first inspect the measurement columns of each sample in `.data.csv`, and then use `.stats.csv` to decide whether it is a single-sample anomaly or an overall timing failure.

## 14.5 The SRAM Period Can Be Modified in `sram_compiler/testbenches/base_testbench.py`

`self.t_period` in `sram_compiler/testbenches/base_testbench.py` can be used to modify the SRAM cycle period. The timing of other input signals and internal signals will be adjusted automatically.

