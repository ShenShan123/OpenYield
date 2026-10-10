# SRAM compiler tutorial — V2.2.7 development

The compiler builds 6T or 10T transistor-level SRAM arrays with distributed wordline and bitline RC, a replica column, address and control logic, and column periphery. Generate SRAM macros through [`main_sram.py`](../main_sram.py). Run commands from the repository root.

TIME_CONTROL device sizes and its other configuration settings are in
[`config_yaml/time_control.yaml`](config_yaml/time_control.yaml). They include
model names, SI widths/lengths, register and clock-buffer sizes, folding and
buffer policy, delay stages and dummy loads, guard RC parameters, and fallback
load/effort rules. `SRAM_CONFIG.load_all_configs` reads this file beside the
global YAML, or an explicit `TIME_CONTROL` entry in its configuration-file map.
The validated settings are available as `config.time_control`; constructors
and `TimeControlFactory` accept an appended `control_config` override.

Explicit arguments take precedence, including the frozen driver-class loads
and guard settings supplied by the compiler. Connection topology and names
remain in `subcircuits/time_generate.py`. Defaults use a 90 nm minimum active
width and a 5 nm width grid; delay drivers and dummy loads have separate sizes.
Settings changes invalidate a
frozen peripheral baseline and are recorded in CLI run identity and metadata.
See the [physical calibration guide](layout/README.md) for V2.2.7's current
layout coverage and the complete macro back-annotation workflow.

## 1. Set up

```bash
conda env create -f environment.yml
conda activate openyield
command -v Xyce
```

Xyce is required for simulation and for extracting equivalent cells in modes 1–4. A full-transistor mode 0 deck can be generated without it.

## 2. Generate and run a small array

At the top of `main_sram.py`, set `ARRAY = [8, 4, False]`, `TARGET = (7, 3)`, `OPERATION = "read"`, and `RUN_XYCE = False`. Then run:

```bash
python main_sram.py
```

The script reads YAML in memory and writes `deck.sp` and `summary.json` under `outputs/main_sram/`. Set `RUN_XYCE = True` to execute with Xyce. The generated deck and every simulation automatically include independent local device variation. `MC_SEED` makes a run repeatable, and `MC_RUNS` sets the number of samples. `read&write`, `hold_snm`, `read_snm`, and `write_snm` are also supported. Check the Xyce log and waveform crossings before treating a run as a correct access; a generated deck or passing `.MEASURE` alone is not enough. The [batch runner guide](per_device_mc/README.md) covers command-line overrides and output files.

## 3. Change the circuit

| File | Change |
|---|---|
| [`config_yaml/global.yaml`](config_yaml/global.yaml) | Default array, supply, temperature, cell type, timing, sizing, interconnect, and equivalent mode |
| `config_yaml/sram_6t_cell.yaml`, `sram_10t_cell.yaml` | Cell dimensions and model names |
| `config_yaml/{precharge,write_driver,wordline_driver,decoder,mux,sa}.yaml` | Peripheral dimensions |
| [`sizing/sizing_lookup.json`](sizing/sizing_lookup.json) | Critical driver size classes |
| [`sizing/timing_lookup.json`](sizing/timing_lookup.json) | Array clock classes and supported envelope |

YAML widths and lengths are in **metres**. The supported array envelope is at most 512 rows and 256 columns. An unseen size inside it rounds up to the next row and column class. A changed cell candidate or PVT point must retain the baseline driver and clock resolution; changed physical RC or periphery requires fresh evidence. See [sizing and timing](sizing/README.md).

The signal wiring is always distributed. `interconnect.mode: per_net` is the
canonical setting; `distributed` is a legacy spelling and `star` is rejected.
`global.yaml` loads [`interconnect_example.yaml`](config_yaml/interconnect_example.yaml).
It binds 707 separate signal networks and two power meshes for the qualified
FreePDK45 8×8 nominal TT, 1 V, 25 °C, mode-0, no-mux, target-7/7 read/write
reference at 9 ns. Each entry has its own network file, hash and R/C scales.
`power_models.VDD` and `power_models.VSS` connect device supply terminals at
physical taps, producing IR drop and ground bounce. Cross-net coupling remains
explicit. Changed scales/files mark `reference_settings_validated` false.

For other configurations, `routes` defines thirteen independent critical
interconnect classes in addition to array `wl` and `bl` geometry. Their current
1 ohm / 0.1 fF per-pitch fallback values remain illustrative; fallback supplies
remain ideal. Metadata reports `calibrated_rc.applied: false` outside the
reference scope. Explicit `calibrated_profile` requires its exact scope and
rejects incompatible circuits; `reference_profile` automatically selects the
scoped default. `INTERCONNECT_CONFIG` in `main_sram.py` selects another YAML.
`W_RC` separately controls local storage and peripheral stubs. Use
[`interconnect_illustrative.yaml`](config_yaml/interconnect_illustrative.yaml)
for baseline comparisons. Sizes, timing, models and RC changes invalidate
frozen sizing contexts; keep them fixed across candidate comparisons.

## 4. Select diagnostic options

| Input | Meaning |
|---|---|
| `VARIATION_MODE = "nominal"` | Fixed PDK corner, no local draw, for a diagnostic comparison |
| `VARIATION_MODE = "shared"` | One random card per base model |
| `VARIATION_MODE = "custom"` | Explicit parameter table from the cell YAML |
| `REAL_CELL_MODE = 0` | Keep every array transistor; reference mode |
| `REAL_CELL_MODE = 1`–`4` | Approximate increasingly many unused cells with extracted loads |

`MC_SEED = None` makes one sample random on each run. Modes 1–4 do not have full-array local variation coverage and need Xyce during netlist construction. See [equivalent models](equivalent_modeling/README.md).

## 5. Read the results

The main entrance records the resolved configuration, timing and sizing identities, model hash, seed, and variation metadata in `summary.json` for deck-only runs; simulation results and logs are saved beside the deck. The batch runner also records solver path and run status and retains failed attempts separately. Inspect the `.prn` data for `clk`, `clk_buf`, `cs`, `we`, `wl_en`, `rbl`, `rbl_delay`, `s_en`, `w_en`, `PRE`, and `sa_iso` first when an access fails. Clock-high is access; clock-low is recovery and precharge, including after writes.

The configured V2.2.5 envelope is not fully qualified: its waveform screen has not completed and 256x256 has no working operating point. The [release record](../docs/design/COMPILER_ENTRY_V2_2_5.md) explains the current limits. The [root guide](../README.md) links the sizing, optimization, and yield workflows.
