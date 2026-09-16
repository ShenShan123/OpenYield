# V2.1.4 driver sizing and timing: fixed classes

V2.1.1 adds a far-PRE access-start guard: WL, write and sense enable wait for
physical precharge release. The immutable baseline records its observer and
common-gate loads, four settling stages and `precharge_off_tau`, derived from
wire R/C and the baseline load estimate. Candidate cells and PVT do not refit
it. Old snapshots missing these fields/loads cannot be injected. The existing
V2.0.9 transistor class table and V2.1.0 clock table remain unchanged; TIME
hardware and complete sizing fingerprints change. See the
[access-start repair](../../docs/design/DISTRIBUTED_ONLY_V2_1_1.md#access-start-repair).

V2.0.9 introduced a lookup table of fixed
integer size classes, `sizing_lookup.json`. Every array configuration maps to
one row class and one column class, so the precharge, write driver, wordline
driver and decoder output inverter are always one of a small set of fixed
device sizes. The motivation is layout generation: a layout library is drawn
from fixed transistors, so the sizes must not vary continuously with the array.
The legacy `fixed` mode (the original `rows/16`, `sqrt` array rules with the
known 8-row write weakness) was removed in V2.0.9.

V2.0.10 retains the table's `v2.0.9-lookup-1` identity and transistor classes.
It counts both sense-amplifier EN/ISO RC sections and freezes a four-stage
precharge settling guard for distributed arrays (`precharge_guard_stages=4`,
zero for star); see the
[review and limits](../../docs/design/DISTRIBUTED_RC_V210_REVIEW.md).

Because `rc_sa_units` depends on `w_rc` and not on the interconnect mode, that
sense/isolation load correction resized the TIME buffers of **every**
`w_rc=True` array, star topology included, while V2.0.10's waveform cases were
all distributed. V2.0.11 changes no size and screens the star topology
directly: see the [write and read screen](../../docs/design/WRITE_VALIDATION_V211.md),
which also records a pre-existing precharge-release margin miss on four-column
RC arrays.

## Size classes

Scales multiply the base widths in the circuit YAMLs (precharge PMOS 0.27 um,
write driver 0.18/0.36 um, wordline NAND2 0.18/0.27 um, wordline inverter and
decoder output inverter 0.09/0.27 um). Row classes size the bitline drivers,
column classes the wordline path; both are independent of cell type, column
mux, RC stubs and interconnect mode.

| rows ≤ | precharge `pre` | write input `wd_in` (M1-M4) | write output `wd_out` (M5-M12) |
|---|---|---|---|
| 32 | 1 | 1 | 2 |
| 64 | 2 | 1 | 4 |
| 128 | 4 | 2 | 8 |
| 256 | 8 | 4 | 16 |
| 512 | 16 | 8 | 32 |

| cols ≤ | wordline inverter `wl_inv` | wordline NAND2 `wl_nand` | decoder output inverter `dec_inv` |
|---|---|---|---|
| 4 | 1 | 1 | 1 |
| 8 | 2 | 1 | 1 |
| 16 | 4 | 2 | 1 |
| 32 | 8 | 3 | 1 |
| 64 | 16 | 5 | 2 |
| 128 | 32 | 9 | 3 |
| 256 | 64 | 18 | 5 |
| 512 | 128 | 35 | 9 |

Each entry is the V2.0.5 rule (`wd_out = max(1.5, rows/16)`,
`wd_in = max(0.5, wd_out/4)`, `pre = max(0.5, rows/32)`, `wl_inv = max(1, cols/4)`,
`wl_nand = max(1, cols/15)`, `dec_inv = max(1, wl_nand/4)`) evaluated at the
class upper bound and rounded up to an integer, so every array in a class has
at least the driver strength of the screened rule. Tall arrays with at most
eight columns (256x8, 512x4) use the same driver scales as the V2.0.5 rules; small arrays get
the extra margin of the rounding (8x4: precharge 1 instead of 0.5, write output 2
instead of 1.5). The replica stays `(K, N) = (1, 9)`, matched to the real wordline
driver, with the canonical read loading, decoder output scaling and effort-based
control buffers of V2.0.5.

## Interpolation for unseen arrays

`interpolate_class()` resolves any array size on the class ladder:

- Inside the table an array takes the next anchor at or above its row or column
  count (round-up interpolation). A 48x20 array therefore uses the `rows ≤ 64` and
  `cols ≤ 32` classes. The sizes stay on the tabulated ladder and never fall below
  the rule.
- Beyond the last anchor the ladder continues geometrically with the ratio of the
  last two anchors (doubling per doubling of rows for every row class; 35/18 for
  the NAND2 and 9/5 for the decoder inverter), rounded up to integers, and the
  result is flagged `extrapolated=True` with the synthetic bound in `size_class`
  (for example `rows<=1024 (extrapolated)/cols<=4`). Extrapolated sizes have
  only the V2.0.9 three-sample screen at 1024x4, 2048x2 and 8x1024 behind them;
  add measured anchors to the table instead of relying on them.

The TIME control buffers (`PRE`, `w_en`, `s_en`, isolation, `wl_en`, address and
clock) are still sized from the actual loads of the resolved classes; they are
not part of the table. Their per-stage widths remain the effort-based tapers of
V2.0.4 and are folded into fingers of at most 2 um.

## Use

`global.yaml` defaults to `sizing.mode: lookup`. Resolve once per baseline and
pass the same immutable result to every candidate cell and PVT sample:

```python
from sram_compiler.per_device_mc.run import load_config
from sram_compiler.sizing import resolve_driver_sizes
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench

config = load_config(8, 4, "SF")
sizes = resolve_driver_sizes(config, mux=False)
print(sizes.size_class, sizes.pre, sizes.wd_in, sizes.wd_out, sizes.wl_inv, sizes.wl_nand, sizes.dec_inv)

# Resolve before changing the baseline into a candidate cell.
config.sram_6t_cell.pmos_width.value *= 1.2
testbench = Sram6TCoreMcTestbench(
    config, corner="SF", choose_columnmux=False, real_cell_mode=0,
    driver_sizes=sizes, mc=False, sim_path="outputs/sizing_example",
)
deck = testbench.create_testbench("write", 7, 3)
metadata = sizes.to_dict()  # JSON-serializable classes, loads, hashes and the extrapolated flag.
```

`DriverSizes` and its nested `DriverLoads` are frozen dataclasses. Reuse allows
changed cell values and run PVT, but rejects changed geometry, cell type, mux,
peripheral parameters, PDK contents or physical RC context. The fingerprint
includes the lookup file hash and the selected classes.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `mode` | `lookup` | `lookup` selects fixed classes; `rules_only` evaluates the V2.0.5 continuous rules the classes derive from; `auto` consults qualified records, otherwise `rules_only` |
| `lookup` | package `sizing_lookup.json` | Alternative class table (project-relative or absolute path), e.g. a layout library with different classes |
| `replica` | table `K: 1, N: 9`, `matched: true` | Active replica cell count, odd delay-stage count, matched replica driver |
| `scale_decoder` | true | Scale the final decoder output inverters by `dec_inv` |
| `effort_buffers` | true | Even buffer-chain length from load effort, fingered gates |
| `canonical_read` | true | Disabled write-stack loading on real and replica bitlines in read decks |
| `parasitic_factor`, `wd_floor_margin`, `k_w`, `pre_min` | rule defaults | `rules_only` coefficients; accepted but inactive in `lookup` mode |
| `table` | package `sizing_table.json` | Qualification record table for `auto` |

Unknown options, the removed `fixed` mode and `fixed_scales`, nonpositive or
nonfinite numbers, invalid lookup tables (non-increasing bounds, missing or
nonpositive scales, wrong schema) and invalid array geometry fail explicitly.
Mux fan-in is two and requires an even column count.

Precharge, write-driver and wordline sweeps keep the resolved class in their
SPICE expressions. TIME buffer sizing uses the baseline loads; peripheral sweeps
are not qualified across their complete width ranges.

## Evidence and qualification

The classes inherit the V2.0.4 campaign and V2.0.5 Stage C screen evidence of
the rules they round up from, and V2.0.9 adds a three-sample per-device screen
of the fixed classes themselves (seed 82026): 8x4, 16x16 and 32x32 for both
cells and mux choices with explicit-RC 16x16 variants; 64x16 (both cells, mux
on and off); 64x64 and 16x256; the interpolated sizes 3x3, 5x3, 6x6, 12x4,
20x10, 48x20, 100x50 and 128x32; the extrapolated classes at 1024x4, 2048x2 and
8x1024; and pilots at 256x8, 512x4, 16x512, 128x128 and 256x64. Read, write and
hold are checked in every deck (retention after release, no read disturb, quiet
unselected wordlines, restored and equalized bitlines, neighbor-row retention
across address-change hazards); the results are in the V2.0.9 changelog and
under ignored `outputs/qualification/V2.0.9/`. A screen is not tail or yield
qualification: `sizing_table.json` remains empty, half-select waveform
qualification remains a separate open requirement, and the extrapolated
classes have screening evidence only.

The MC testbench defaults to `variation_mode='per-device'` (fixed PDK corner,
independent `vth0`, `u0`, `voff` with 5% relative sigma on every instantiated
MOS). A single local sample is random; use `variation_mode='nominal'` for a
deterministic deck. Local qualification tools take `--sizing-mode lookup`; see
the [development guide](../../docs/DEVELOPMENT.md). Run the simulator-free
regression suite from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

The tests cover class boundaries, the rule lower bound of every class,
independence from cell/mux/RC/wires, interpolation and extrapolation, invalid
tables, alternative table paths, generated MOS widths, TIME loads and full-array
6T/10T read/write generation.

Runtime qualification lookup uses `scoring_sources.json`, which pins the
reviewed local scoring sources by content hash together with the runtime code.
The compiler never opens files from ignored `dev/`. The original YAML 200/100 ps
access limits are reported separately by the qualification scorer; the default
replica `(1, 9)` does not claim compliance with the read limit.


## Clock classes (V2.1.0)

`timing_lookup.json` (`v2.1.6-timing-5`: every class re-evidenced on the
V2.1.6 write slot, see `docs/design/WRITE_SLOT_V2_1_6.md`) uses the same row
and column anchors as driver sizing.
Each entry contains an integer `half_period_ps` budget; choose the next anchor
at or above each dimension, then compute:

```text
T = ceil_to_50ps(2 * max(row_budget, column_budget) * (1 + margin))
```

| Row bound | Half-cycle budget (ps) | Period with 25% margin (ns) |
|---|---|---|
| 32 | 1800 | 4.5 |
| 64 | 1900 | 4.75 |
| 128 | 2100 | 5.25 |
| 256 | 2500 | 6.25 |
| 512 | 3600 | 9.0 |

The row budgets are the V2.1.4 values (V2.1.0 to V2.1.3:
1600/1800/2000/2400/3600 ps). Column bounds ≤4/8/16/32/64/128/256/512
contribute budgets 1600/1600/1600/1800/2000/2400/2800/3200 ps. For example
16x32 uses 4.5 ns and 48x20 uses 4.75 ns. Classes are shared across RC and
PVT; the evidenced 10T and 6T column-mux variants below replace them for
those architectures.
Extrapolation uses the final geometric ratio and is flagged; it is unqualified.
The stored phase fields are budgets, not measurements. These settings are not
full PVT or mismatch qualification; see the
[release review](../../docs/design/TIMING_LOOKUP_V2_1_0.md).

```yaml
timing:
  mode: lookup
  margin: 0.25
  # lookup: path/from/project/root/timing_lookup.json
```

### 10T cells (V2.1.3)

`variants` in the table hold a separate ladder for one cell type, optionally
restricted to one `mux` setting. The loader requires the shared anchors and
rejects any budget below the shared one, so a variant can only add margin.
The V2.1.3 entry for `SRAM_10T_CELL` applies with or without a column mux.
At SS 0.9 V / 125 °C the 10T read port discharges the replica bitline about
1 ps per row slower than 6T and its sense path adds a near-constant 200 ps,
and a column mux costs a further pass-gate delay of about 140 ps that the
shared class absorbs for 6T but not for 10T: the V2.1.2
[follow-up](../../docs/design/TIMING_FOLLOWUP_V2_1_2.md) found 16x16 10T with
a mux failing at 4 ns, and the V2.1.3 evidence run found 16x16 10T without a
mux failing the same read-output checks, a flat 200 ps failing the 128x8
mux read at 5.5 ns and a 4.5 ns class leaving 125 ps at the 32x16 bound
(81 ps under one mismatch seed). The row budget therefore grows with height
and keeps at least 250 ps between the local read output and the 1.2 T
deadline at every class bound, nominal SS 0.9 V / 125 °C. The 512-row class
is set by the storage node instead: the 10T read-disturb bump (0.196 V
against 0.131 V for 6T) decays only as the bitline discharges, and a 512-row
bitline left Q at 0.108 V at the deadline at 11 ns against the 0.1 VDD
tolerance, so that class gives the bitline about 5.6 ns:

| Row bound | 10T budget (ps) | Period (ns) | Column bound | 10T budget (ps) | Period (ns) |
|---|---|---|---|---|---|
| 32 | 2000 | 5.0 | 4 / 8 / 16 | 1800 | 4.5 |
| 64 | 2200 | 5.5 | 32 | 2000 | 5.0 |
| 128 | 2400 | 6.0 | 64 | 2200 | 5.5 |
| 256 | 3200 | 8.0 | 128 | 2600 | 6.5 |
| 512 | 5600 | 14.0 | 256 / 512 | 3000 / 3400 | 7.5 / 8.5 |

`ArrayTiming.budget` records `SRAM_10T_CELL` when the variant applies and
`shared` otherwise (a mux-restricted variant reads `<cell>/mux` or
`<cell>/nomux`). The nominal boundary run and the ten-seed mismatch pilot
behind this budget are in the
[10T budget record](../../docs/design/TIMING_10T_BUDGET_V2_1_3.md); it ran on
the V2.1.3 TIME block, before the V2.1.4 write-request hold latch.

### 6T cells (V2.1.4)

V2.1.4 applies the 10T rule to both 6T ladders. At SS 0.9 V / 125 °C the
V2.1.4 probe reads (on the new TIME block) left 109 ps between the local read
output and the 1.2 T deadline at the 32x16 bound at 4 ns and 206, 203 and
175 ps at 64, 128 and 256 rows, so the shared row budgets rise to the table
above. A column mux delays the 6T output by about 70 ps at 32 rows, 95 ps at
64, 150 ps at 128, 200 ps at 256 and 275 ps at 512 rows: at the old shared
classes the 32x16 (38 ps), 128x8 (53 ps) and 256x4 (29 ps late) mux reads
failed their output checks nominally and the 16x16 mux sequence failed two
of three mismatch seeds. The `SRAM_6T_CELL` entry with `"mux": true` gives
6T with a column mux its own rows and keeps the shared columns; sequences
need about 50 ps more than reads, hence 4.75 ns up to 32 rows:

| Row bound | 6T-mux budget (ps) | Period (ns) |
|---|---|---|
| 32 | 1900 | 4.75 |
| 64 | 2000 | 5.0 |
| 128 | 2200 | 5.5 |
| 256 | 2700 | 6.75 |
| 512 | 3600 | 9.0 |

`ArrayTiming.budget` reads `SRAM_6T_CELL/mux` for these arrays. The boundary
run, the mismatch seeds and the write-waveform gate are in the
[6T budget record](../../docs/design/TIMING_6T_BUDGET_V2_1_4.md).

For a diagnostic override use `timing: {mode: fixed, t_period: 1.0e-8}` (seconds).
`timing.fixed` is independent of the removed legacy driver `sizing.fixed` mode.
No automatic simulation or historical fitted model runs during lookup.

```python
from sram_compiler.sizing import resolve_driver_sizes, resolve_timing

sizes = resolve_driver_sizes(config, mux=False)
timing = resolve_timing(config, sizes)
# Freeze both before changing the cell or run PVT.
testbench = Sram6TCoreMcTestbench(
    config, choose_columnmux=False, driver_sizes=sizes, timing_config=timing,
    variation_mode="nominal", sim_path="outputs/timing_example",
)
```

`ArrayTiming` binds to the driver baseline and timing options. Both testbenches
apply it automatically when no timing override is supplied. The optimizer caches
it before applying candidates. Metadata includes source, class, margin, table
hash/version and extrapolation. Explicit `TimingConfig.apply()` or legacy
`set_timing_parameters()` overrides are recorded as such. Exact qualified-driver
records retain measured timing and their original artifact format.

V2.1.1 supports only distributed wiring and applies the four-stage
precharge guard even when local series stubs are disabled. `sizing.precharge_guard_stages` can select a nonnegative even
stage count for an explicitly validated wire configuration. It is part of the
frozen driver baseline. A longer clock alone cannot repair wordline/precharge
overlap; release checks reject unsafe or missing events. Driver transistor
classes and replica sense timing K=1/N=9 are unchanged. The
[V2.1.1 report](../../docs/design/DISTRIBUTED_ONLY_V2_1_1.md) and
[evaluation schedule](../../docs/plans/V2_1_1_TIMING_FOLLOWUP.md) record current
distributed-only coverage; V2.1.0 waveform passes remain historical evidence.

The V2.1.2 [follow-up](../../docs/design/TIMING_FOLLOWUP_V2_1_2.md) found the
shared 4 ns class exhausted for 10T cells with a column mux at SS 0.9 V /
125 °C (16x16 fails its read outputs, 8x4 fails under per-device mismatch,
both pass at 4.5 ns). V2.1.3 adopts the separate 10T budget above with its
own [evidence run](../../docs/design/TIMING_10T_BUDGET_V2_1_3.md); the shared
classes were unchanged until V2.1.4 re-evidenced both 6T ladders.
