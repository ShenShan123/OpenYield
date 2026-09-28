# SRAM compiler tutorial — V2.2.4

The compiler builds 6T or 10T transistor-level SRAM arrays with distributed wordline and bitline RC, a replica column, address and control logic, and column periphery. The command-line runner can generate a deck without Xyce or run DC and transient analyses with it. Run commands from the repository root.

## 1. Set up

```bash
conda env create -f environment.yml
conda activate openyield
command -v Xyce
```

Xyce is required for simulation and for extracting equivalent cells in modes 1–4. A full-transistor mode 0 deck can be generated without it. If you already have the environment, check `python -m sram_compiler.per_device_mc.run --help`.

## 2. Generate and run a small array

```bash
python -m sram_compiler.per_device_mc.run \
  --rows 8 --cols 4 --target-row 7 --target-col 3 \
  --operation read --variation-mode nominal --mc-runs 1 \
  --real-cell-mode 0 --output-dir outputs/first_run
```

This reads the YAML files in memory and writes a netlist and `summary.json` below the selected output root. It does not run Xyce. Add `--run-xyce` to execute the netlist and save measures and a `.prn` waveform. Check the Xyce log and waveform crossings before treating a run as a correct access; a generated deck or passing `.MEASURE` alone is not enough.

For a reproducible local-mismatch run:

```bash
python -m sram_compiler.per_device_mc.run \
  --rows 8 --cols 4 --operation write \
  --variation-mode per-device --mc-runs 2 --seed 3 \
  --real-cell-mode 0 --output-dir outputs/first_run --run-xyce
```

`read&write`, `hold_snm`, `read_snm`, and `write_snm` are also supported. The [runner guide](per_device_mc/README.md) explains variation and output files. `main_sram.py` is an editable one-run entrance with an `ARRAY`, cell-size, corner, operation, and seed settings block; run it with `python main_sram.py` after editing those values.

## 3. Change the circuit

| File | Change |
|---|---|
| [`config_yaml/global.yaml`](config_yaml/global.yaml) | Default array, supply, temperature, cell type, timing, sizing, interconnect, and equivalent mode |
| `config_yaml/sram_6t_cell.yaml`, `sram_10t_cell.yaml` | Cell dimensions and model names |
| `config_yaml/{precharge,write_driver,wordline_driver,decoder,mux,sa}.yaml` | Peripheral dimensions |
| [`sizing/sizing_lookup.json`](sizing/sizing_lookup.json) | Critical driver size classes |
| [`sizing/timing_lookup.json`](sizing/timing_lookup.json) | Array clock classes and supported envelope |

YAML widths and lengths are in **metres**. The supported array envelope is at most 512 rows and 256 columns. An unseen size inside it rounds up to the next row and column class. A changed cell candidate or PVT point must retain the baseline driver and clock resolution; changed physical RC or periphery requires fresh evidence. See [sizing and timing](sizing/README.md).

The signal wiring is always distributed. `interconnect.mode: star` is rejected. The default wire dimensions give illustrative 1 ohm and 0.1 fF per pitch, not extracted metal. Use `--interconnect-config sram_compiler/config_yaml/interconnect_example.yaml` as a template for your own geometry. `w_rc` separately controls local storage and peripheral stubs.

## 4. Choose variation and equivalent mode

| Input | Meaning |
|---|---|
| `--variation-mode nominal` | Fixed PDK corner, no Monte Carlo draw |
| `--variation-mode per-device` | Independent `vth0`, `u0`, and `voff` draws per retained MOS; default |
| `--variation-mode shared` | One random card per base model |
| `--variation-mode custom` | Explicit parameter table from the cell YAML |
| `--real-cell-mode 0` | Keep every array transistor; reference mode |
| `--real-cell-mode 1`–`4` | Approximate increasingly many unused cells with extracted loads |

A single per-device run without `--seed` is random, so use `nominal` for a deterministic corner check. Modes 1–4 do not have full-array per-device coverage and need Xyce during netlist construction. See [equivalent models](equivalent_modeling/README.md).

## 5. Read the results

The command runner records its resolved configuration, timing and sizing identities, model hashes, seed, solver path, and run status in `summary.json`. Failed attempts are retained separately. Inspect the `.prn` data for `clk`, `clk_buf`, `cs`, `we`, `wl_en`, `rbl`, `rbl_delay`, `s_en`, `w_en`, `PRE`, and `sa_iso` first when an access fails. Clock-high is access; clock-low is recovery and precharge, including after writes.

The configured V2.2.4 envelope is not fully qualified: its waveform screen has not run and 256x256 has no working operating point. The [release record](../docs/design/PHASED_CONTROL_V2_2_4.md) explains the current limits. The [root guide](../README.md) links the sizing, optimization, and yield workflows.
