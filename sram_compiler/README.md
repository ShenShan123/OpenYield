# SRAM Compiler and Test Platform User Guide — V2.1.10

V2.1.1 retains V2.1.0’s default `timing.mode: lookup`: fixed row/column classes set a frozen clock before candidate/PVT changes. V2.1.3 adds a separate, evidenced budget for 10T cells (with or without a column mux). V2.1.4 raises the shared 6T row classes, adds a 6T column-mux budget and holds the TIME write request while the wordline enable is high. The [timing guide](sizing/README.md#clock-classes-v210) covers settings, explicit overrides and evidence limits.

V2.1.1 signal interconnect is distributed-only, including the bitline periphery,
decoder, write-data clock and mux selects. Omitting `interconnect` uses the
illustrative reference geometry; `mode: star` is rejected. `w_rc` controls local
series stubs independently of the always-present physical wires. See the
[current topology and evaluation](../docs/design/DISTRIBUTED_ONLY_V2_1_1.md).

This document introduces the basic usage of the SRAM compiler, simulation flow, Monte Carlo testing, waveform plotting, and result statistics. It mainly covers the following files and directories:

- `main_sram.py`: Main entrance for one SRAM generation and simulation run. It reads the YAML files in memory and defaults to seeded per-device local mismatch over the full transistor array.
- `sram_compiler/per_device_mc/run.py`: Command-line runner with the same defaults, for scripted or batch generation.
- `utils/`: Shared modules for measurements, waveforms, plotting, area estimation, and SPICE model read/write operations; existing `from utils import ...` imports remain supported.
- `sram_compiler/`: Core code for SRAM configuration, subcircuit generation, and testbench construction.

Run commands from the repository root; code paths below are relative to that
root. See the [per-device runner](per_device_mc/README.md),
[driver sizing guide](sizing/README.md), [circuit review](CIRCUIT_REVIEW.md),
and [plans and release history](../docs/README.md) for detailed references.

V2.0.7 adds [distributed RC wiring](../docs/design/DISTRIBUTED_RC_MODEL.md) and corrects local RC configuration;
V2.0.8 audits it, adds the `INTERCONNECT_CONFIG` setting to `main_sram.py`, and records the
[extended validation](../docs/design/DISTRIBUTED_RC_VALIDATION.md).
V2.0.9 sizes the precharge, write driver, wordline driver and decoder output from
fixed integer size classes in a [lookup table](sizing/README.md) (`sizing.mode: lookup`);
the legacy `fixed` mode is removed.
V2.0.11 models the control lines that span an array dimension (`PRE`, `w_en`,
`w_en_bar` (`din_en` since V2.1.10), `s_en`, `sa_iso` across the columns; `wl_en` down the rows) as
tapped RC wires in distributed mode, so each precharge cell, write driver,
sense amplifier and wordline driver connects at its own column or row. Star
decks and every driver size class are unchanged. It also fixes the retry,
rejection and measurement-window handling of V2.0.10 and screens the star
topology on waveforms; see the
[V2.0.11 audit and write screen](../docs/design/WRITE_VALIDATION_V211.md).

V2.0.10 adds a distributed precharge settling guard, corrects TIME loads and
read measurements, and preserves failed samples and numerical retry evidence.
See the [waveform review](../docs/design/DISTRIBUTED_RC_V210_REVIEW.md).
V2.0.6 incorporated the default local mismatch runner, model specialization,
and sampling into this compiler directory. This guide was previously the root
`readme_compiler.md`.

Reusable regression tests are in top-level `tests/`. Local development and
qualification scripts are kept in ignored `dev/`; see the
[development guide](../docs/DEVELOPMENT.md). The `testbenches/` modules below
are the runtime simulation API used to generate and run SRAM circuits.

## 1. What This Platform Can Do

This platform uses PySpice to generate transistor-level SRAM netlists and calls Xyce to run simulations. It currently supports:

- 6T / 10T SRAM cell, array, and complete peripheral circuit modeling.
- Read, write, and combined read/write transient simulations.
- Hold / read / write SNM DC analysis.
- Xyce Monte Carlo with independent per-device local mismatch by default.
- Custom process-parameter table simulation.
- RC parasitic modeling for wordline, bitline, and other key nodes.
- Equivalent-model acceleration for non-target cells.
- Read/write delay, power, SNM, and area statistics.

## 2. Environment Setup

It is recommended to use the conda environment provided by the project:

```bash
conda env create -f environment.yml
conda activate openyield
```

If the environment already exists, update it with:

```bash
conda env update -f environment.yml
```

It is recommended to check that the following commands work:

```bash
Xyce -v
python -c "import PySpice, pandas, numpy, matplotlib"
```

## 3. Directory Structure

```text
OpenYield/
├── main_sram.py                         # Main entrance for a single SRAM simulation (settings block at the top)
├── utils/                               # Shared parsing, statistics, plotting, model, and area helpers
├── tran_models/                         # TT/FF/SS/FS/SF SPICE model files
├── sram_compiler/
│   ├── README.md                        # This compiler and testbench guide
│   ├── CIRCUIT_REVIEW.md                # Circuit review and verification evidence
│   ├── per_device_mc/                   # Default local mismatch generation and sampling
│   │   ├── run.py                      # CLI and in-memory load_config() helper
│   │   ├── netlist.py                  # Independent model specialization per retained MOS
│   │   └── sampling.py                 # Materialized local draws for MPI execution
│   ├── sizing/                         # Fixed driver size classes, measured timing, and qualified-table lookup
│   ├── equivalent_modeling/            # Equivalent array model as a simulation input
│   │   ├── README.md                   # Modes, accuracy boundary, and the YAML/CLI/API option
│   │   ├── __init__.py                 # EquivalentConfig and resolve_equivalent (mode validation)
│   │   └── compare.py                  # Accuracy and runtime measurement against the full array
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
│   │   ├── dummy_row_or_column.py       # Dummy cell (replica-wordline load)
│   │   ├── replica_column.py            # Replica column generation for timing control and reference paths
│   │   ├── precharge_and_write_driver.py # Precharge circuit and write driver
│   │   ├── mux_and_sa.py                # Column mux and sense amplifier
│   │   ├── wordline_driver.py           # Wordline driver
│   │   ├── decoder.py                   # Row decoder and cascaded decoder structures
│   │   └── time_generate.py             # TIME_CONTROL block: clock, registers, delay chains and every enable pulse
│   └── testbenches/                     # Testbenches, MC simulation, and SNM processing
│       ├── base_testbench.py            # Base testbench defining power, PDK, default timing, and simulation APIs
│       ├── sram_6t_core_testbench.py    # SRAM array functional testbench that builds read/write peripherals and the full test circuit
│       ├── sram_6t_core_MC_testbench.py # Monte Carlo testbench that generates Xyce MC netlists and parses results
│       ├── parameter_factor.py          # Factory methods that create subcircuit instances from YAML configs
│       └── snm.py                       # SNM curve parsing, crossing detection, and statistics table generation
└── sim1/                                # Default output directory for main_sram.py
```

## 4. Quick Start: Run One SRAM Simulation

`main_sram.py` is the main entrance. Edit the settings block at the top of the
script, then run it from the repository root inside the `openyield` environment:

```bash
python main_sram.py
```

Its defaults are a 16x16 6T array without column mux at TT, a `write` access on
the last row and column, per-device local mismatch with the fixed seed
`MC_SEED = 20260711`, and the full transistor array (`REAL_CELL_MODE = 0`).
The execution flow is:

1. Load all YAML files in memory with `sram_compiler.per_device_mc.run.load_config()`
   and apply `ARRAY`, `CORNER` and `CELL_6T` to the loaded configuration. The
   tracked YAML files are not modified.
2. Resolve the sample count: `MC_RUNS`, or `monte_carlo_runs` from `global.yaml`
   when `MC_RUNS` is `None`. Nominal mode needs exactly one run; custom mode
   needs one run per row of `process_parameters.vars`.
3. Create a timestamped output directory, such as `sim1/20260909_053000_mc_6t/`.
4. Estimate the bitcell area.
5. Build `Sram6TCoreMcTestbench` with `variation_mode=VARIATION_MODE`,
   `mc_seed=MC_SEED` and `real_cell_mode=REAL_CELL_MODE`.
6. Run the read, write, read&write or SNM simulation selected by `OPERATION`.
7. Save the netlist, per-device model cards, Xyce outputs, CSV statistics and
   the waveform plot, then print the mean delay, power and area over the samples.

The command-line runner offers the same defaults as options, for scripted or
batch generation:

```bash
python -m sram_compiler.per_device_mc.run \
  --rows 8 --cols 4 --operation read --mc-runs 2 --seed 3 --audit
```

Add `--run-xyce` to simulate. Outputs are written under `outputs/per_device_mc/`.
Use `--variation-mode nominal` for a deterministic corner run (omit `--mc-runs`
or set it to 1). See its [guide](per_device_mc/README.md) for output and
variation options.

## 5. Modify the Array Size

Modify the following in `main_sram.py` (it overrides `num_rows`, `num_cols` and
`choose_columnmux` from `global.yaml` in memory):

```python
ARRAY = [16, 16, False]
```

The meaning is:

```text
[num_rows, num_cols, choose_columnmux]
```

For example:

```python
ARRAY = [64, 128, False]
```

This builds a `64 x 128` SRAM array and does not enable column mux. Column mux is temporarily unsupported because the timing has not been fully configured.

## 6. Modify the 6T SRAM Cell Size

Modify the following in `main_sram.py` (applied to the loaded configuration in
memory) or edit `sram_6t_cell.yaml` directly:

```python
CELL_6T = [
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

`main_sram.py` takes the corner from its `CORNER` setting and the array size and
column mux from `ARRAY`; `global.yaml` still supplies `vdd`, `temperature`,
`monte_carlo_runs` (used when `MC_RUNS` is `None`), `sram_cell_type`, the
`sizing` block and the PDK paths.

## 8. Configuration File Description

Each YAML file under `sram_compiler/config_yaml/` corresponds to one module:

| File | Purpose |
| --- | --- |
| `global.yaml` | Global simulation parameters, PVT, array size, MC runs, PDK paths, and the `sizing` / `timing` / `interconnect` / `equivalent` blocks |
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
OPERATION = 'write'
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

`main_sram.py` builds `Sram6TCoreMcTestbench` in `build_testbench()` with these
parameters:

```python
mc_testbench = Sram6TCoreMcTestbench(
    config,
    sram_cell_type=config.global_config.sram_cell_type,
    w_rc=True,
    pi_res=100 @ u_Ohm,
    pi_cap=0.001 @ u_pF,
    vth_std=0.05,
    mc=variation_mode in ("shared", "per-device"),
    custom_mc=variation_mode == "custom",
    variation_mode=variation_mode,     # VARIATION_MODE, default "per-device"
    mc_seed=mc_seed,                   # MC_SEED, default 20260711
    sweep_cell=False,
    sweep_precharge=False,
    sweep_senseamp=False,
    sweep_wordlinedriver=False,
    sweep_columnmux=False,
    sweep_writedriver=False,
    sweep_decoder=False,
    corner=config.global_config.corner,
    choose_columnmux=bool(config.global_config.choose_columnmux),
    real_cell_mode=real_cell_mode,     # REAL_CELL_MODE, default 0
    q_init_val=0,
    sim_path=sim_path,
)
```

Meanings:

- `w_rc=True`: Add RC parasitic networks to key nodes.
- `pi_res` / `pi_cap`: Resistance and capacitance values of each segment in the pi-shaped RC network.
- `vth_std=0.05`: Relative Gaussian sigma for `vth0/u0/voff` in the shared and per-device modes.
- `variation_mode='per-device'` (the default): Independent local mismatch for every retained MOS at the fixed corner.
- `variation_mode='nominal'`: The fixed PDK corner without random mismatch; exactly one run.
- `variation_mode='shared'`: The legacy shared-model Monte Carlo flow.
- `variation_mode='custom'`: A user-provided parameter table instead of Xyce random sampling (see 13.1).
- `mc` / `custom_mc`: Derived from `variation_mode`; the testbench rejects inconsistent combinations.
- `mc_seed`: Xyce sampling seed; `None` draws a new seed every run.
- `sweep_*`: Enable parameter sweep for the corresponding module (not combinable with per-device mismatch).
- `real_cell_mode`: `0` keeps the full transistor array; `1`-`4` replace unused cells
  with the equivalent circuit. `None` (the default) takes the `equivalent` block of
  `global.yaml`; the resolved selection is on the testbench as `equivalent` and in every
  run record. Modes `1`-`4` are approximations and call Xyce during netlist generation
  (see [equivalent_modeling/README.md](equivalent_modeling/README.md)).
- `q_init_val`: Initial stored value of the target cell.
- `t_max_step`, `xyce_options`: Xyce solver knobs (see 13.1).
- `next_row`: row address captured at the clock edge that ends a `read` / `write`
  access, to exercise the address-change hold path (default: the address stays
  at `target_row`).

## 11. Output Files

After one run, the output directory is similar to:

```text
sim1/20260529_223000_mc_6t/
```

Common files:

```text
models_per_device_<hash>.spice       # Per-device model cards (default mode; shared mode writes tmp_mc.spice)
model_audit_<hash>.csv               # Device-to-model audit of the specialized netlist
mc_write_16x16_rc1_tb.sp             # Generated Xyce netlist
mc_write_16x16_rc1_tb.sp.variation.json  # Variation mode, seed, sample count, frozen driver sizes
mc_write_16x16_rc1_tb.log            # Xyce console output
mc_write_16x16_rc1_tb.sp.prn         # Waveform output
mc_write_16x16_rc1_tb.sp.mt0         # Measure results of MC run 0
mc_write_16x16_rc1_tb.data.csv       # Raw measurement data of all MC samples
mc_write_16x16_rc1_tb.stats.csv      # Statistics such as mean, standard deviation, and percentiles
mc_write_waveform.png                # Key-node waveform plot
```

Read delay summary formula:

```text
Delay = TREAD_TOTAL      # wl_en rise -> output latch OUT valid (VDD/2)
Power = PSTC + PDYN      # PSTC: restored window 1 ns + [1.6, 1.65]*t_period
```

Write delay summary formula:

```text
Delay = TWRITE_TOTAL     # wl_en rise -> target cell Q reaches 90% VDD
Power = PSTC + PDYN
```

`PAVG = EREAD / t_period` (or `EWRITE / t_period`), where the energy is integrated
over exactly one clock period starting at the first access
(`1 ns + 0.7 * t_period`, the falling clock edge): wordline access, sensing or
writing, and the precharge that restores the bitlines. Precharge is requested
during the clock-high phase and waits for the physical replica wordline to
settle low. V2.1.1 also waits for far-end PRE release before asserting access.
Restoration and access/precharge exclusion are checked at the frozen clock;
their margins depend on geometry and PVT. Before V2.0.2, a roughly 300 ps
self-timed precharge pulse let the floating bitlines droop, for example to
0.74 V at FF / 125 C with the then-default 10 ns period. The sense amplifiers are isolated from
the bitlines (`ISO` pin, driven by `s_en | w_en`) while they are fired and
while the write drivers are on. Every transient testbench carries the full column
periphery (precharge on all columns and on the replica column, column mux,
sense amplifiers); the write testbenches add the write drivers, each fed
through a data-hold latch that is transparent while `w_en` is low, so a write
cycle is `write slot -> write -> precharge (in the next read cycle)` with the
real bitline load and the write data cannot change while the drivers are
enabled. Since V2.1.6 the write drivers take the precharge slot: in a write
cycle PRE is inhibited and `w_en` turns the drivers on in the clock-high
phase, once the previous wordline and the physical precharge are observed off,
so BL/BLB sit at their write rails before the wordline rises; `w_en` deasserts
when the wordline request ends. Write decks write 1 then 0 and measure the
next write's slot as `TWSLOT` (its clock-high work) instead of `TRESTORE`;
`select_every=N` selects one cycle in N to probe the idle → write boundary.
Since V2.1.7 the select that gates the precharge and the write enable
(`cs_pre`) is delayed on its rising edge only, so new write data passes the
hold latch before `w_en` closes it at an idle → write edge and an unselected
cycle stops both enables at once; a `select_every` write deck changes its data
at the selected edges. Since V2.1.8 a read wordline is released at the sense
trigger (`wl_en = buffer(access_clk_bar & !s_en)`; the amplifier is isolated
from the bitlines from then on), the precharge waits for the sense and write
enables of the previous access, the write slot for its sense enable, and
`w_en = we_hold & (wl_en | (cs_pre & write_slot))` ends with the wordline
enable rather than with the deselect, so no two enables of different roles
overlap (`docs/design/ENABLE_OVERLAP_V2_1_8.md`). Since V2.1.9 the write
drivers stay on until the wordline is observed off (`w_en = we_hold &
(wordline_busy | selected_slot)`, `wordline_busy = !(wl_en_bar &
wordline_off)`; the held write request opens on `wordline_idle`), and every
write cycle carries `VWEN_ACCESS_ERROR_<cycle>`, the highest wordline level
while the target driver's enable is below 0.9 VDD, which `access_validity`
limits to 0.1 VDD like the other access checks
(`docs/design/WRITE_HOLD_V2_1_9.md`). Since V2.1.10 the column write-data
latches hold on the registered write (`din_en = wordline_idle & ((we & cs) |
!w_en)`, the `din_en` line): between two writes the drivers stay on (`w_en =
we_hold & (held(wordline_busy & we_hold) | selected_slot)`, the write's busy
term held four unit stages so the next slot takes over) and only
their data changes, once the previous wordline is observed off
(`docs/design/WRITE_LATCH_V2_1_10.md`). The control block is
`TIME_CONTROL` (instance `XTIME_CONTROL`; probe paths read
`XTIME_CONTROL:<node>`); see
[`docs/design/TIME_CONTROL_PATH.md`](../docs/design/TIME_CONTROL_PATH.md).

The segment measures (`TDECODER`, `TPRCH`, `TWLDRV`, `TSWING`, `TSA`, `TS_EN`,
`TWDRV`, `TWRITE_Q`, ...) are still written to `.mt0` / `.data.csv` for inspection,
but they overlap in time and are not summed: the delay is the end-to-end measure.
A measure that Xyce could not evaluate is written as `FAILED` and makes
`run_mc_simulation()` raise instead of reporting `0.0`.

At the end, `main_sram.py` prints the mean over the samples:

```text
[OUTPUT] mean of N sample(s): y[0]=Delay(... ns), y[1]=Power(... uW), y[2]=Area(... µm²)
```

## 12. Common `utils/` Tools

The [utilities package](../utils/README.md) provides the following helpers,
also available through the existing `from utils import ...` API:

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

The platform supports four process variation modes:

- `variation_mode='nominal'` (also selected by `mc=False`): Use the fixed corner
  without random mismatch.
- `variation_mode='shared'`: Explicitly select the legacy flow, which writes
  `AGAUSS(...)` expressions into `tmp_mc.spice`; devices sharing a base model
  share its random parameters.
- `variation_mode='per-device'` (the default with `mc=True` and `custom_mc=False`):
  Keep the global PDK corner fixed and generate independent `vth0`, `u0`, and
  `voff` expressions for every retained array and peripheral MOS. Specialization
  in `sram_compiler/per_device_mc/netlist.py` writes model cards and a device audit.
  `.SAMPLING` is emitted even for `mc_runs=1`, which is one random sample.
  Pass `mc_seed=<int>` to `Sram6TCoreMcTestbench(...)` to make sampling
  reproducible; without it Xyce draws a new seed every run
  (printed in `<netlist>.log`, which now keeps the Xyce console output).
  If Xyce stops a deck with `Time step too small` (seen on a few 512-row
  arrays, where the Newton loop oscillates during the access), the flow
  automatically retries once with a `.TRAN` maximum step of 20 ps
  (`t_max_step=2e-11`): this keeps results within 0.5 % of the default settings
  at ~1.8x the time steps. Pass `t_max_step` explicitly to use it from the
  start, or `xyce_options=['.OPTIONS TIMEINT ERROPTION=1']` for a faster but
  less accurate run (delays of converging decks shift by a few per cent).
  `next_row=<row>` (read / write decks) makes the address register capture a
  different row at the clock edge that ends the access; the wordline and the
  cell of that row are added to the `.PRINT` so the address-change hold
  margin can be inspected (the address is held by a latch while the wordline
  is on, see `docs/CHANGELOG.md` V2.0.2).
- `custom_mc=True` (selects `variation_mode='custom'`): Use a user-provided process-parameter table. The program writes `vars` into a `.data table` file and uses `.STEP data=table` to make Xyce simulate one row at a time.

Per-device mismatch requires a separate deck per geometry; legacy `.STEP`
geometry sweeps cannot be combined with local sampling.

To use custom process variation, set the following in `main_sram.py`:

```python
VARIATION_MODE = "custom"
```

The script then builds the testbench with `custom_mc=True`, reads the table with
`sram_compiler.per_device_mc.run.get_custom_vars()` from `process_parameters.vars`
in `sram_compiler/config_yaml/sram_6t_cell.yaml` (or the 10T file), resolves the
sample count with `resolve_mc_runs()`, and passes the table to
`run_mc_simulation()`:

```python
delay, pavg, pstc, pdyn = testbench.run_mc_simulation(
    operation=OPERATION,
    target_row=target_row,
    target_col=target_col,
    mc_runs=mc_runs,
    temperature=temperature,
    vars=custom_vars,
)
```

`vars` must be a two-dimensional array with the shape:

```text
(mc_runs, number_of_parameters)
```

In other words, the number of rows in `vars` must equal the sample count (`MC_RUNS`, or `monte_carlo_runs` in `global.yaml` when `MC_RUNS` is `None`); each row corresponds to one simulation sample. A mismatch fails before generation.

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
- Set `REAL_CELL_MODE = 1`, `--real-cell-mode 1`, or `equivalent: {mode: 1}` in
  `global.yaml` (equivalent cells for the unused array; an approximation, so measure it
  with `python3 -m sram_compiler.equivalent_modeling.compare` before trusting it).
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

`self.t_period` in `sram_compiler/testbenches/base_testbench.py` can be used to modify the SRAM cycle period (or call `set_timing_parameters()` on the testbench before `run_mc_simulation()`). The timing of other input signals and internal signals will be adjusted automatically.

After a `read` or `write` run the flow prints an estimate of the minimum clock
period from the measured phases:

```text
[INFO] clock-low phase  (clk->wl_en TCLK_WLEN + access TREAD_TOTAL | TWRITE_TOTAL)
[INFO] clock-high phase (bitline restore TRESTORE, decode TCLK_DEC)
[INFO] CLK(min) estimate in this size and PVT (50 % duty, +10 %)
```

`T_min = 2 * max(clock-low, clock-high) * 1.1`. It was validated against
period sweeps in V2.0.2 (e.g. 8x4 6T read: estimate 0.90 ns, the deck passes
at 0.9 ns and fails at 0.8 ns; at TT / 125 C the same array needs ~1.7 ns).
Periods from 0.6 ns to 100 ns were simulated; note that `PSTC` / `PDYN` are
only a quiescent / dynamic split for `t_period >= 5 ns` (a warning is
printed otherwise, because the start-up precharge overlaps the PSTC window).
