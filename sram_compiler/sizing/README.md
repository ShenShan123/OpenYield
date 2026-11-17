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

The standard 2:1-mux 6T architecture uses the V2.2.9 self-timed controller in
the versioned lookup profiles; compact layouts default to its
overlapped-recovery form (below). `sizing.access_speedup: false` selects the
legacy controller for diagnostics; 10T retains it. Direct `TIME_CONTROL` and factory
callers keep their historical default unless they select `access_speedup`.

A row class's `self_timed` block sets `setup_stages`, `replica.K/N` and optional
`read_completion_stages` / `write_completion_stages`. A column class can set
completion stage floors. Each completion chain uses the larger row/column
reference, with at least eight stages. These are timing references; the
decoder classes and buffer-sizing policy remain fixed; physical buffer
sizes follow the resolved loads. The four
compact square anchor classes use the separately validated delay/peripheral
overrides below. A one-row schematic
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
Joint entries can also raise `read_completion_stages` or
`write_completion_stages` for one class pair. The larger row/column allowance
and eight-stage minimum still apply, so a joint entry cannot shorten them.
Likewise, `setup_stages` can raise one pair's decoder allowance while retaining
the row-class minimum and the references of other shapes.

An optional `self_timed_joint_sizing` list selects sizing overrides for an
explicit pair of rounded `max_rows` / `max_cols` anchors. It applies only to
compact self-timed 6T control; legacy, schematic and unlisted or extrapolated
classes retain their existing sizes. Entries may override `pre`, `wd_in` and
`wd_out`; the existing column caps still apply afterward. The paired fields
`delay_driver_scale` and `delay_loads_per_stage` set the unit-delay inverter
widths relative to the configured unit inverter and its dummy-load count.
They do not change any guard, setup or completion stage count. These choices
enter the frozen sizing fingerprint, and a changed lookup invalidates a
previously resolved record. The shipped 32/64/128/256 square anchor overrides
use two-unit delay drivers and one dummy per stage; 64/128/256 also use PRE2
and write-driver input1/output4. The 64×256 and 128×256 classes use the
256×256 values: with the row-class PRE4 / write-driver 2/8 sizing their
256-column PRE and `W_EN` lines released too slowly at FF −40 °C, and the
overlapped-recovery controller then released ISO 95–170 ps before far `W_EN`
was off. Short wide classes (32×128, 32×256, 64×128) keep their row-class
sizes: with only two decoder setup stages, the faster override delay chain
let unselected 32×256 wordlines glitch.
A sizing entry is not timing or area qualification;
characterize it at a fixed diagnostic clock before publishing new budgets.

### Overlapped recovery (default compact 6T controller)

When the lookup carries `overlapped_recovery_references`, compact self-timed
6T arrays use the overlapped-recovery `TIME_CONTROL`
(`v2.2.9-overlapped-recovery-1`): the next access is captured, decoded and set
up while the previous access's bitlines restore. `sizing.recovery_overlap:
false` selects the V2.2.9 recovery-before-capture controller. Each array takes
the smallest reference class that covers it (`read_completion_stages`,
`write_completion_stages`, `restore_stages` and optional guard overrides).
512-row classes have explicit entries; the 512×256 entry raises
`enable_off_stages` from 22 to 30, because at 22 stages ISO released 25 ps
(FF −40 °C) before far `W_EN` was off. The
[design record](../../docs/design/timing/OVERLAPPED_RECOVERY_V2_2_9.md) gives
the architecture, the waveform contract and the per-class evidence.

The `overlapped_recovery` block of `layout_timing_lookup.json` carries every
joint class measured post-layout (FineSim, extracted macros). The cycle budget
is the larger of the longest SS 0.9 V 125 °C read occupancy (admission to
`recovered`) plus the stalled recovered-to-admission delay, and the
restore-limited spacing (admission to the last far bitline at 0.96 VDD, worst
measured corner). The access budget is the worst CLK→OUT at that spacing. Both
receive one 10% margin and 50 ps rounding, as in V2.2.9:

| Rows \ Cols | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| 32 | 6.60 / 4.15 | 7.40 / 4.80 | 8.70 / 5.75 | 14.75 / 9.95 |
| 64 | 7.15 / 4.75 | 7.80 / 5.00 | 9.70 / 6.65 | 14.75 / 9.95 |
| 128 | 8.90 / 6.20 | 9.20 / 6.45 | 10.05 / 7.15 | 14.75 / 10.30 |
| 256 | 11.15 / 7.85 | 11.80 / 8.20 | 13.00 / 9.20 | 16.90 / 12.15 |
| 512 | 14.70 / 11.15 | 16.05 / 11.70 | 19.50 / 13.45 | 25.20 / 16.20 |

(clock / read deadline, ns). 64×256 and 128×256 measure 13.05 and 14.00 ns but
are raised to 32×256's budget, because the grid must not decrease with either
dimension.

The V2.2.9 `self_timed` block is unchanged except for its 512-row classes:
512×32 / 512×64 / 512×128 are measured post-layout on the macros with the
read-held latch, the array VDD strap and the 512-row band placement, at
16.65 / 17.25 / 24.30 ns (deadlines 11.05 / 11.55 / 15.10 ns; 512×32 is raised to
the 256×32 cycle). 512×256 is 35.95 / 17.75 ns (capture-to-full-recovery
32,669 ps, capture-to-data 16,095 ps); its earlier 30.10 ns extrapolation left
too little time for the 512-row bitline restore.

Both clock JSONs use schema 2. The `self_timed` block holds integer-ps budgets
for capture-to-data (`access_ps`) and capture-to-full-recovery (`cycle_ps`);
the `overlapped_recovery` block holds capture-to-data and the sustained
capture spacing, with its own frozen baseline (the sizing policy key includes
`recovery_overlap`).
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
The [V2.2.9 area/timing record](../../docs/design/timing/ACCESS_SPEEDUP_AREA_V2_2_9.md) separates
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
