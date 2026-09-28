# Batch simulation runner — V2.2.4

Use [`main_sram.py`](../../main_sram.py) for one SRAM macro. This runner accepts command-line options for batches and automated experiments. From the repository root, generate a full-transistor deck without Xyce:

```bash
python -m sram_compiler.per_device_mc.run \
  --rows 8 --cols 4 --target-row 7 --target-col 3 \
  --operation read --mc-runs 1 \
  --real-cell-mode 0 --output-dir outputs/example
```

Add `--run-xyce` to simulate. Xyce must be on `PATH`, or pass `--xyce /path/to/Xyce`. The runner writes a configuration-specific directory under `--output-dir`, keeping `summary.json`, the generated deck and model cards, Xyce logs, measures, and waveform data when available. Failed attempts retain provenance. `--audit` writes a per-device model audit; `--no-waveform` omits `.PRINT` waveform output. Use `--help` for the full option list.

## Variation choices

| Mode | Result |
|---|---|
| `per-device` (automatic) | Independent Gaussian `vth0`, `u0`, and `voff` values for each retained MOS at the selected corner |
| `nominal` | Fixed corner with no random draw; use this for a deterministic baseline |
| `shared` | One random card shared by every device using a base model |
| `custom` | Process parameter table from the cell YAML |

The runner uses a fixed seed by default. To choose another repeatable sample and count:

```bash
python -m sram_compiler.per_device_mc.run \
  --rows 8 --cols 4 --operation write \
  --mc-runs 2 --seed 3 \
  --real-cell-mode 0 --output-dir outputs/example --run-xyce
```

In the Python API, `mc_seed=None` makes even a single run a random sample. The selected PDK corner is fixed across its local draws. The default relative sigma is 5% for all three parameters; this is an illustrative mismatch model, without area scaling or calibrated correlation. Modes 1–4 only vary the transistors they retain, so use mode 0 for full-array mismatch coverage.

## Set circuit inputs

`--corner` accepts `TT`, `FF`, `SS`, `FS`, or `SF`. `--vdd` is in volts, `--temperature` in Celsius, and `--period` in seconds. `--period` is a diagnostic clock override; the default is the frozen lookup clock. `--timing-lookup` loads another class table. `--interconnect-config` loads a wire YAML mapping without rewriting `global.yaml`. The only supported topology is distributed. The supported array envelope is 512 rows by 256 columns.

The runner supports `read`, `write`, `read&write`, `hold_snm`, `read_snm`, and `write_snm`. `--target-row` and `--target-col` select the accessed cell. An omitted target defaults to the last row and column. Per-device mode cannot be combined with legacy `.STEP` geometry sweeps.

For Python callers:

```python
from sram_compiler.per_device_mc.run import load_config
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench

config = load_config(8, 4, "TT")
testbench = Sram6TCoreMcTestbench(
    config, corner="TT", mc_seed=3,
    choose_columnmux=False, real_cell_mode=0,
    sim_path="outputs/python_example",
)
circuit = testbench.create_testbench("read", 7, 3)
```

`run.py` provides the batch CLI and in-memory YAML loading. `netlist.py` specializes model cards and audits connectivity. Local waveform and MPI sampling scripts can live in ignored `tests/` and `dev/` workspaces; they are not needed to generate a deck.

A generated deck, passing measures, or one Monte Carlo draw does not establish read/write correctness or yield. Inspect waveforms and the reported margins. The [V2.2.4 record](../../docs/design/PHASED_CONTROL_V2_2_4.md) describes the pending screen and the unresolved 256x256 operating point; [the compiler tutorial](../README.md) explains the full flow.
