# Driver sizing and timing — V2.2.9

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

A class table may carry four optional fields (V2.2.8 layout candidate 3,
[`layout_sizing_lookup.json`](layout_sizing_lookup.json)): a row class may set
`setup_stages`, the even length of TIME_CONTROL's wordline setup chain; a column
class may set `guard_stages`, the even length of its precharge-off, isolation,
enables-off and replica-wordline-off settling guards, and `pre_cap` / `wd_cap`,
upper bounds on the precharge and write-driver classes because every column
loads one control bar. Tables without these fields keep the fixed 8-stage
setup chain and 4/4/2/4-stage guards. The resolved counts travel in
`DriverSizes` and are checked by `validate_for`, so a frozen record from a
different table is rejected.

The legacy V2.2.8 candidate 3 changes only the replica delay chain, `replica.N` 9 → 3: post-layout
reads fired the sense amplifier after a complete bitline swing, and the shorter
chain fires it 220 ps earlier at 64×64 SS. The extracted 64-row SS read still
missed the shared 10.5 ns clock class by 153 ps, so the layout timing table
below gives that class 12 ns (see the
[post-layout record](../../docs/design/layout/POST_LAYOUT_V2_2_8.md)). The
legacy path of `sizing_lookup.json` keeps `N = 9`.

## Self-timed references and clocks

The standard 2:1-mux 6T architecture uses the V2.2.9 controller in the
versioned lookup profiles. `sizing.access_speedup: false` selects the legacy
controller for diagnostics; 10T retains it. Direct `TIME_CONTROL` and factory
callers keep their historical default unless they select `access_speedup`.

A row class's `self_timed` block sets `setup_stages`, `replica.K/N` and optional
`read_completion_stages` / `write_completion_stages`. A column class can set
completion stage floors. Each completion chain uses the larger row/column
reference, with at least eight stages. These are timing references; the
peripheral, decoder and buffer W/L classes remain frozen. A one-row schematic
uses one active replica cell; explicit invalid replica counts are rejected.
The shared `setup_stages`, `guard_stages` and global `replica` fields continue
to describe the legacy path.

`self_timed_joint_references` can set independent `precharge_off_stages`,
`isolation_stages`, `wordline_off_stages`, `write_drive_stages` and
`enable_off_stages` for a row/column class. These even stage counts cover
different frozen control-bar loads. A common guard count would needlessly
delay light paths and can miss the heavy write-drive path. All references
are part of frozen-object validation; changing the clock alone cannot repair
an unsafe handoff.

Both clock JSONs use schema 2. The `self_timed` block holds integer-ps budgets
for capture-to-data (`access_ps`) and capture-to-full-recovery (`cycle_ps`).
Recovery includes all bitlines at 98%, PRE on, physical enables off, and every
completion timer and settling guard reset. A single bitline's `TRESTORE` at
90% is insufficient; the CLK(min) diagnostic requires full-waveform `TCYCLE`.

The period and absolute read deadline each receive one 10% margin, then round
up to 50 ps. Shipped C1 profiles use an explicit joint row/column grid, so a
shared controller path is counted once. The full grid must match its axis
anchors and must not decrease with either dimension. Custom tables without a
grid retain the separable excess rule. Dimensions round up to the next class
inside the 512-row by 256-column envelope. `evidence_status` refers
to the anchor; sharing its budget does not establish a smaller macro's
waveform qualification. Larger classes are explicitly marked extrapolated.
The [V2.2.9 record](../../docs/design/timing/ACCESS_SPEEDUP_V2_2_9.md) separates
the measured compact macro, schematic reference and larger-array estimates.

`layout_timing_lookup.json` belongs to the compact physical driver profile;
`timing_lookup.json` belongs to the schematic driver profile. Published budgets
check the frozen object's sizing lookup and policy identities, periphery,
PDK and reference RC identities and the
replica/timer references. Custom sizing or RC cannot silently reuse them;
use an explicit fixed diagnostic clock when characterizing a changed context.
The cycle and deadline remain fixed across cell candidates and PVT samples.

Legacy entries retain the clock-high access / clock-low recovery contract,
`ceil_to_50ps(2 * half * 1.25)`, and their earlier periods. Historical compact
32×32 / 64×64 / 128×128 clocks were 9.5 / 12.0 / 21.25 ns; their original
[post-layout record](../../docs/design/layout/POST_LAYOUT_V2_2_8.md) remains
unchanged. Fixed periods and the 512×512 layout escape are diagnostics, without
automatic timing qualification.

## Resolve once in Python

```python
from sram_compiler.config_yaml.loader import load_config
from sram_compiler.sizing import resolve_driver_sizes, resolve_timing
from sram_compiler.sizing.table import physical_context

config = load_config(8, 4, "TT")
context = physical_context(w_rc=True, interconnect=config.global_config.interconnect)
sizes = resolve_driver_sizes(config, mux=True, physical_context=context)
timing = resolve_timing(config, sizes, context)
print(sizes.size_class, timing.t_period, timing.source)

# Pass these same objects to testbenches for candidate cells and PVT samples.
```

For a compact source in Python, set both matching lookups before resolving:

```python
config.global_config.sizing = {
    "mode": "lookup", "compact_control": True,
    "lookup": "sram_compiler/sizing/layout_sizing_lookup.json",
}
config.global_config.timing = {
    "lookup": "sram_compiler/sizing/layout_timing_lookup.json",
}
```

For macro generation, use `main_sram.py` as shown in the [compiler tutorial](../README.md). YAML widths and lengths are SI metres. Class lookups and physical-context hashes travel with a run's summary. The [V2.2.5 release record](../../docs/design/compiler/COMPILER_ENTRY_V2_2_5.md) reports the pending waveform screen and the unresolved 256x256 operating point. The completed [V2.2.2 screen](../../docs/design/timing/PHASED_CONTROL_V2_2_2.md) certifies only that earlier tree.
