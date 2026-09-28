# Driver sizing and timing — V2.2.4

The compiler resolves critical driver widths and the clock **before** a cell candidate or PVT sample is evaluated. Keep that baseline fixed across those evaluations. A changed peripheral, model file, physical RC context, or control architecture needs a new baseline and functional evidence; stale qualified records are rejected.

## Driver classes

`global.yaml` defaults to `sizing.mode: lookup`. [`sizing_lookup.json`](sizing_lookup.json) maps row and column counts to fixed integer multipliers on the peripheral YAML transistor widths. The row ladder sets precharge and write-driver scales; the column ladder sets the wordline and decoder output scales. A 48x20 array uses the `rows ≤ 64` and `cols ≤ 32` classes. TIME_CONTROL buffer widths use the actual resolved loads.

| Rows up to | Precharge | Write input | Write output |
|---:|---:|---:|---:|
| 32 | 1 | 1 | 2 |
| 64 | 2 | 1 | 4 |
| 128 | 4 | 2 | 8 |
| 256 | 8 | 4 | 16 |
| 512 | 16 | 8 | 32 |

| Columns up to | WL inverter | WL NAND | Decoder inverter |
|---:|---:|---:|---:|
| 4 | 1 | 1 | 1 |
| 8 | 2 | 1 | 1 |
| 16 | 4 | 2 | 1 |
| 32 | 8 | 3 | 1 |
| 64 | 16 | 5 | 2 |
| 128 | 32 | 9 | 3 |
| 256 | 64 | 18 | 5 |

The sizing lookup has a historical 512-column entry, but the current timing envelope rejects arrays wider than 256 columns. `rules_only` is an unqualified continuous fallback and `auto` requires a context-matched qualified record or falls back to those rules. Neither mode extends the supported envelope.

## Clock classes

[`timing_lookup.json`](timing_lookup.json) is the V2.2.4 table. It sets a 512-row by 256-column envelope and separate budgets for cell type and column mux. The current cycle captures a request on the rising edge, accesses during clock-high, and recovers and precharges during clock-low. Clock-high access and clock-low recovery each get a half-cycle budget.

For an unseen array inside the envelope, the row and column counts round up to the next class. When both dimensions require time, the joint half-cycle budget is `max(row, column) + min(row_excess, column_excess)`, where each excess is above the first class of its ladder. Variants can add time, never reduce the shared-ladder clock. The default timing margin is 25%; `--period` is a diagnostic override, not qualification evidence. The 512-column class is outside the envelope and was removed from the timing table.

## Resolve once in Python

```python
from sram_compiler.per_device_mc.run import load_config
from sram_compiler.sizing import resolve_driver_sizes, resolve_timing

config = load_config(8, 4, "TT")
sizes = resolve_driver_sizes(config, mux=False)
timing = resolve_timing(config, sizes)
print(sizes.size_class, timing.t_period, timing.source)

# Pass these same objects to testbenches for candidate cells and PVT samples.
```

For a simple CLI run, use the [compiler tutorial](../README.md). YAML widths and lengths are SI metres. Class lookups and physical-context hashes travel with a run's summary. The current [V2.2.4 release record](../../docs/design/PHASED_CONTROL_V2_2_4.md) reports the pending waveform screen and the unresolved 256x256 operating point. The completed [V2.2.2 screen](../../docs/design/PHASED_CONTROL_V2_2_2.md) certifies only that earlier tree.
