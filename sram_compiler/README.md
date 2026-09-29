# SRAM compiler tutorial — V2.2.5

The compiler builds 6T or 10T transistor-level SRAM arrays with distributed wordline and bitline RC, a replica column, address and control logic, and column periphery. Generate SRAM macros through [`main_sram.py`](../main_sram.py). Run commands from the repository root.

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

The signal wiring is always distributed. `interconnect.mode: star` is rejected. The default wire dimensions give illustrative 1 ohm and 0.1 fF per pitch, not extracted metal. Use `sram_compiler/config_yaml/interconnect_example.yaml` as a template, then set `INTERCONNECT_CONFIG` in `main_sram.py` to your geometry. `W_RC` separately controls local storage and peripheral stubs.

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
