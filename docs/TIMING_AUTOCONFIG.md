# Automatic timing configuration — V2.1.4 guide and historical proposal

**V2.2.1 update:** access is now clock-high and recovery is clock-low. The
current `v2.2.0-timing-3` budgets are twice the V2.1.10 values below, with
threefold 128–512-row budgets and the wider 512-column class (24 ns for 6T, 25.5 ns for muxed 6T and 10T). Read
sensing also waits for physical wordline release and isolation. The
[current record](design/PHASED_CONTROL_V2_2_2.md) is authoritative
for present timing, measurements, and validation; the following numeric history
explains the previous architecture.


V2.1.1 retains the timing table introduced in V2.1.0: **a fixed lookup table with
row and column classes, like driver sizing**. `timing.mode: lookup` is now the
default. `sram_compiler/sizing/timing_lookup.json` contains integer half-cycle
budgets in ps. Select the next row and column anchors, take the larger budget,
apply the configured margin (25% by default), double it and round up to 50 ps.
No model fit, automatic calibration or simulation runs inside the resolver.

The row classes are ≤32/64/128/256/512 with budgets 1800/1900/2200/2700/3700 ps
(V2.1.9; 1800/1900/2100/2500/3600 ps from V2.1.4, 1600/1800/2000/2400/3600 ps before).
Column classes are ≤4/8/16/32/64/128/256/512 with budgets
1600/1600/1600/1800/2000/2400/2800/3200 ps. Thus 8x4 and 16x16 use 4.5 ns,
48x20 uses 4.75 ns, and 512x4 uses 9.25 ns and 8x512 uses 8 ns. Beyond the final anchor the
geometric ladder continues, with `extrapolated: true` and no qualification claim.
V2.1.3 adds one `variants` entry: 10T cells, with or without a column mux,
take row budgets 2000/2200/2400/3200/5600 ps and column budgets 200 ps above
the shared ladder (5 ns up to 32 rows / 16 columns, 14 ns at 512 rows),
after the V2.1.2 follow-up and the V2.1.3 evidence run measured their read
path about 200 ps plus 1 ps per row slower than 6T at SS 0.9 V / 125 °C; see
the [10T budget record](design/TIMING_10T_BUDGET_V2_1_3.md).
V2.1.4 raises the shared row budgets, which now apply to 6T cells without a
mux only, and adds a `SRAM_6T_CELL` variant for 6T with a column mux (rows
1900/2000/2200/2700/3600 ps, shared columns: 4.75 ns up to 32 rows) after the
V2.1.4 probe reads left 109 ps at the 32x16 6T bound and failed the 32x16,
128x8 and 256x4 mux reads at the old classes; see the
[6T budget record](design/TIMING_6T_BUDGET_V2_1_4.md).
V2.1.9 re-derives the 128- to 512-row classes of all three ladders under local
mismatch (shared 2200/2700/3700 ps, 6T-mux 2300/2900/4000 ps, 10T 512 rows
4200 ps): every class bound keeps 0.02 T plus 10 % of the access between the
nominal SS read output and the deadline; see the
[write-hold record](design/WRITE_HOLD_V2_1_9.md), section 6.
V2.1.10 moves the 10T 512-row class to 4300 ps (10.75 ns; the 512x4 read
missed the rule by 7 ps at 10.5 ns) and keeps every other class: the column
write-data latches now hold on the registered write, so the drivers stay on
between two writes and the write -> write slot shrinks by about 0.5 ns at
SS 0.9 V / 125 C, where V2.1.9's slot missed the runtime restore check at the
32x16 and 32x32 bounds; see the [write-latch record](design/WRITE_LATCH_V2_1_10.md).

These are design budgets informed by the historical envelope below, with
allowance for the current guard; they are **not measured phases or full PVT /
local-mismatch qualification**. The original waveform screen is
recorded in [the V2.1.0 review](design/TIMING_LOOKUP_V2_1_0.md). The
[V2.1.1 report](design/DISTRIBUTED_ONLY_V2_1_1.md) records validation of the
new distributed-only signal routes. Physical RC and changed peripherals
require fresh waveform validation.

`resolve_timing(config, driver_sizes)` returns an immutable baseline-bound
`ArrayTiming`. Both testbenches apply it; the optimizer caches it before cell
candidate changes, and reused yield testbenches keep it throughout sampling.
Changed driver baselines or timing options are rejected on injection. The
50% duty cycle, 1% edges, 1 ns startup offset and replica K/N remain unchanged.
`timing: {mode: fixed, t_period: 1.0e-8}` selects an explicit diagnostic clock;
the CLI also accepts `--period` or `--timing-lookup`. The existing measured-phase
`TimingConfig` and qualified driver-record path retain their historical format.

Access checks sample data at the clock deadline. Additional data and retention
checks prevent a late crossing or a cell that flips back from passing. V2.1.1
checks the whole post-deadline retention interval, and every supported
distributed array checks local and far precharge release. The static-power window
is now at the end of the first post-access restore phase. See the
[sizing/timing guide](../sram_compiler/sizing/README.md#clock-classes-v221-re-screened-in-v222).

---

# Automatic timing configuration for all array sizes — proposal (V2.0.3)

Status in V2.0.6: original V2.0.3 proposal with measured basis, relocated into
`docs/`. The measured-phase timing and frozen testbench/table application
introduced in V2.0.4 remain in `sram_compiler/sizing/timing.py`;
full qualification is in progress and tracked
in `docs/DRIVER_SIZING_PROPOSAL.md`. The historical fits below choose a calibration
clock only; they are not reused as qualification for the new periphery.
The numbers come from
the V2.0.2 sweeps (494 rows, 490 unique, in `docs/data/TIMING_AUTOCONFIG_data.csv`) and from the worst-case PVT characterisation run for this
proposal (section 3.4). Everything marked *phase 2* is optional follow-up.

## 0. Recommendation in short

1. **The only free timing knob of this architecture is the clock period**
   (plus, optionally, its duty cycle). Wordline enable, write enable and
   precharge are the two clock phases; sense enable is replica-timed and the
   three latch enables are derived from `s_en`, `w_en` and `wl_en`. Today the
   period is 10 ns for every array (`base_testbench.py`), 5-10x what the
   circuits need, so no timing failure can ever show up in the optimiser or in
   the yield estimator.
2. **Set the period per array from a phase model, derated to the worst-case
   PVT, with an explicit margin:**
   `T = 2 * max(low_wc, high_wc) * (1 + margin)` with
   `low = clk->wl_en + access`, `high = max(restore, decode)`, the phases the
   flow already measures. A fitted model (`a + b*rows + c*log2(cols)` for
   reads, `a + b*rows + c*cols + d*log2(rows)` for the write restore) predicts
   the nominal phases within 20-40 ps over 1x1 .. 512x4 / 16x512; one
   multiplicative PVT factor per phase (measured: 2.2-2.3x at
   SS / 125 C / 0.9 V for every control phase and the read access, up to 3.8x
   at SF / 125 C / 0.9 V for the write access of arrays with fewer than 16
   rows) moves it to the worst case; the default margin is 25 %. Period
   sweeps at those corners confirm the formula within one 100 ps step.
3. **Anchor the model with a characterised table and a self-calibration
   fallback:** every (cell, rows, cols, mux) that was simulated is stored with
   its measured worst-case phases; sizes outside the table or a changed
   periphery (hash of the YAMLs) trigger one nominal read + one write deck at
   the worst-case PVT (the four measures exist), whose result is added to the
   table. No simulation in the optimiser loop.
4. **Freeze the period per array size as the spec** for size optimisation and
   yield analysis: it is derived from the baseline (YAML) transistor sizes at
   the worst-case PVT, and candidate designs or Monte Carlo samples that need
   more time fail (delay measure `FAILED`, wrong data). The optimiser must not
   re-derive the period from its own candidate.
5. *Phase 2:* expose the sense-enable margin (number of active replica cells
   `K`, delay-chain stages `N`) and an asymmetric duty cycle; both are
   architecture changes with their own verification.

## 1. What "timing" means in this compiler

| signal | generated by (`time_generate.py` / testbench) | timing | array-size dependence | set by the proposal |
|---|---|---|---|---|
| `clk` | testbench pulse source | period `T`, 50 % duty, capture edge at `1 ns + 0.2 T`, access (falling) edge at `1 ns + 0.7 T` | none today (10 ns) | **`T` per array** |
| address, data, `csb`, `web` | testbench pulse sources | valid from `0.1 T` to `0.3 T` around the capture edge (setup = hold = `0.1 T`) | scales with `T` | follows `T` (setup must stay > DFF setup at the worst case, section 6) |
| `wl_en` / `WL{row}` | `WORDLINE_ENABLE_BUFFER(access_clk_bar, s_en_bar)` (V2.1.8; `access_clk_bar` alone before, `wl_pdrive` before V2.1.7), wordline driver | write: whole clock-low phase, `T/2`; read: from `TCLK_WLEN` = 112-190 ps after the edge (287-290 ps for 512-column write decks: clock-buffer load) to the sense trigger (released 45-192 ps after `s_en` reaches the amplifier, V2.1.8) | edge time grows with rows / cols (buffer taper) | `T/2 >= low_wc * (1 + m)` |
| `w_en` (V2.1.6) | `AND2(we_hold, held(wordline_busy & we_hold) \| (cs_pre & write_slot))` (the write's busy term through four unit stages, V2.1.10; `wordline_busy \| (cs_pre & write_slot)` in V2.1.9) with `wordline_busy = !(wl_en_bar & pre_ready)` and `write_slot = pre_ready & pre_off_ready & !s_en` (V2.1.10; V2.1.9 also required `slot_armed` in the slot; `wl_en` instead of `wordline_busy` in V2.1.8; `AND3(we_hold, cs_pre, wl_en \| (pre_ready & pre_off_ready))` in V2.1.7); `cs_pre = cs & cs_delayed` since V2.1.7 | the write slot of the clock-high phase (from the previous wordline, the physical precharge and the previous sense enable observed off, and after the select delay at an idle -> write edge) through the access, ending when the wordline is observed off (V2.1.9; with the wordline enable before); between two writes it stays on and only the data changes (V2.1.10; V2.1.9 waited for it to be observed off) | bitline drive grows with rows (driver class) | `T/2 >= TWSLOT_wc * (1 + m)` |
| `PRE` | `NAND3(clk_buf, cs_pre, pre_ready & !we_hold & !(s_en \| w_en))` + `PRECHARGE_BUFFER` (V2.1.8; without the enables term before) | whole clock-high phase of a read cycle, `T/2`, after the replica wordline and both enables are observed off; inhibited in a write cycle (V2.1.6) | restore time 190-430 ps (rows, cols) | `T/2 >= high_wc * (1 + m)` |
| `s_en` | `AND3(rbl_delay, gated_clk_bar, we_bar)`, `rbl_delay` = replica bitline fully discharged by one replica cell + 9-stage delay chain | self-timed: wordline + ~250 ps at TT / 25 C for every size, tracks rows and PVT by construction (the replica column carries `rows + 1` cells of load) | tracks automatically | unchanged (phase 2: `K`, `N`) |
| `sa_iso` | `NOR2(s_en, w_en)` + inverter | `s_en \| w_en` | - | derived |
| output latch `D_LATCH` | EN = `s_en` | opens with the amplifier | - | derived |
| write-data hold latch | EN = `din_en = wordline_idle & ((we & cs) \| !w_en)` (V2.1.10; `w_en_bar` before) | holds while a wordline is open, and while the drivers are on unless the new cycle is a selected write: between two writes only the data changes | - | derived |
| address hold latch | EN = `wl_en_bar` | holds while a wordline is on | - | derived |

So "precharge, WL enable, SA enable, latch enable" reduce to: **the clock
period (and duty)** for precharge / wordline / write enable, and **the
replica configuration** for sense enable and the output latch.

## 2. Requirements

- One call returns the timing for any `(cell type, rows, cols, mux)`, for the
  sizes the compiler accepts (1x1 .. 512 rows, up to 512 columns).
- Worst-case PVT: the period is valid at the slowest process corner, lowest
  supply and highest temperature the design is specified for, not at the
  PVT of the current run.
- Margin suitable for size optimisation and yield analysis: the baseline
  design passes at the worst case with local variation; a design that is
  ~25 % slower does not.
- No simulation inside the optimiser loop; the cost of characterising a new
  size is two nominal decks, once.
- Verifiable with the existing measures (`TCLK_WLEN`, `TCLK_DEC`,
  `TRESTORE`, `TREAD_TOTAL`, `TWRITE_TOTAL`) and period sweeps.

## 3. Measured basis

### 3.1 Constraint equations

With a 50 % duty cycle both phases must fit their work
(`_print_min_period()`, V2.0.2):

```
low  = TCLK_WLEN + max(TREAD_TOTAL, TWRITE_TOTAL)   # clock-low: wordline phase
high = max(TRESTORE, TCLK_DEC)                      # clock-high: precharge + decode
T_min = 2 * max(low, high)
```

Validated by the V2.0.2 period sweeps: 8x4 6T read passes at 0.9 ns and
fails at 0.8 ns (`T_min` = 0.82 ns, +10 % = 0.90); 16x16 6T read passes at
0.9, fails at 0.8 (0.88 / 0.97); writes pass down to 0.6 ns (0.49-0.55 ns).
Re-validated at the worst case for this proposal (6T, nominal devices):

| deck | condition | measured `2 * max(low, high)` | passes at | fails at |
|---|---|---|---|---|
| 8x4 read | SS / 125 C / 0.9 V | 1.81 ns | 1.9 ns | 1.8 ns (output latch not settled) |
| 8x4 write | SF / 125 C / 0.9 V | 1.44 ns | 1.5 ns | 1.4 ns (BLB 0.44 V, cell not written) |
| 16x16 read | SS / 125 C / 0.9 V | 1.95 ns | 2.1 ns | 2.0 ns: `OUT` crosses VDD/2 10 ps after the clock edge and settles while the wordline is still on, i.e. functionally at the limit |

The measured phases under-estimate the functional limit by 0-8 % (the
measures use VDD/2 crossings, the pass criterion needs the amplifier and
latch settled before the edge), which the margin of section 4.3 has to
cover.

### 3.2 Nominal phases against array size (6T, mux off, TT, 25 C, 1.0 V)

Values in ps, `T_nom` = the larger of the read and write minimum periods
with the 10 % margin used by `_print_min_period()`:

| rows | cols | clk->wl_en (r) | read access | low (r) | high (r) | clk->wl_en (w) | write access | low (w) | high (w) | T_nom [ns] |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 113 | 283 | 395 | 217 | 114 | 119 | 234 | 188 | 0.87 |
| 2 | 128 | 121 | 322 | 443 | 276 | 153 | 176 | 329 | 276 | 0.98 |
| 4 | 4 | 116 | 287 | 403 | 242 | 119 | 126 | 245 | 214 | 0.89 |
| 8 | 4 | 120 | 291 | 410 | 247 | 122 | 131 | 253 | 219 | 0.90 |
| 8 | 512 | 125 | 350 | 475 | 307 | 289 | 202 | 491 | 429 | 1.08 |
| 16 | 16 | 132 | 308 | 440 | 257 | 140 | 94 | 235 | 247 | 0.97 |
| 16 | 256 | 127 | 354 | 481 | 287 | 189 | 133 | 322 | 337 | 1.06 |
| 16 | 512 | 127 | 360 | 488 | 297 | 290 | 158 | 447 | 432 | 1.07 |
| 32 | 32 | 133 | 332 | 465 | 261 | 147 | 100 | 247 | 259 | 1.02 |
| 64 | 16 | 135 | 358 | 493 | 255 | 143 | 93 | 236 | 259 | 1.08 |
| 64 | 64 | 133 | 382 | 515 | 266 | 153 | 108 | 261 | 284 | 1.13 |
| 128 | 32 | 136 | 424 | 560 | 268 | 150 | 104 | 254 | 278 | 1.23 |
| 256 | 8 | 137 | 524 | 661 | 267 | 140 | 96 | 236 | 268 | 1.45 |
| 512 | 4 | 146 | 739 | 885 | 279 | 141 | 95 | 236 | 275 | 1.95 |

Reads are limited by the clock-low phase everywhere (the access grows
~0.9 ps per row: bitline capacitance against one cell), writes by the
clock-high phase on most sizes (the restore grows with the columns: precharge
of every bitline plus the `PRE` buffer taper; the row-scaled write driver
keeps the write access itself at 70-130 ps). The 10T cell is 0-135 ps slower
on reads at >= 64 rows (1.2 ps per row) and within 15 ps elsewhere; the mux
changes the phases by -29..+24 ps. The full table (27 sizes x 2 cells x mux)
is `docs/data/TIMING_AUTOCONFIG_data.csv` (source `v202_sweep`).

### 3.3 Model fit (nominal)

Least-squares fits per (cell, operation, mux) over the 27 sizes
(`model_coefs_nominal.json`; residuals in ps):

| cell | op | mux | phase | form | coefficients | rms | worst under-prediction |
|---|---|---|---|---|---|---|---|
| 6T | read | off | low | `a + b*rows + c*log2(cols)` | 390, 0.946, 9.05 | 7 | 18 (16x1) |
| 6T | read | off | high | `a + b*rows + c*log2(cols)` | 221, 0.076, 8.19 | 5 | 12 |
| 6T | write | off | low | `a + b*cols + c*cols/wd_scale` | 234, 0.298, 0.107 | 15 | 29 (2x128) |
| 6T | write | off | high | `a + b*rows + c*cols + d*log2(rows)` | 188, -0.020, 0.410, 10.7 | 8 | 25 (2x128) |
| 10T | read | off | low | `a + b*rows + c*log2(cols)` | 392, 1.20, 9.33 | 9 | 23 (100x50) |
| 10T | write | off | high | `a + b*rows + c*cols + d*log2(rows)` | 188, -0.013, 0.409, 10.6 | 8 | 25 |

(`wd_scale = max(8, rows) / 16`, the write-driver width scale; mux-on
coefficients differ by < 5 %.) A pure `rows + cols` linear model is 2-3x
worse (35-77 ps) because the column dependence of the buffered nets is
logarithmic (taper stages), and the per-row term is the physical bitline
term. The residuals are 2-4 % of the resulting period, i.e. well inside the
margin; the table anchors the exact points anyway.

### 3.4 PVT factors

Per-phase factors relative to TT / 25 C / 1.0 V, mean over 8x4 and 16x16,
both cells (final V2.0.2 corner runs `v202_corners_f`, 1.0 V; clock ->
`wl_en`, decode, restore and high from the read decks):

| condition | clk->wl_en | decode | restore | read access | write access | low (r) | low (w) | high |
|---|---|---|---|---|---|---|---|---|
| TT -40 C | 0.72 | 0.71 | 0.70 | 0.70 | 0.65 | 0.70 | 0.69 | 0.70 |
| SS -40 C | 0.78 | 0.76 | 0.75 | 0.75 | 0.69 | 0.76 | 0.75 | 0.75 |
| FF 25 C | 0.90 | 0.91 | 0.92 | 0.91 | 0.93 | 0.91 | 0.92 | 0.92 |
| FS 25 C | 1.02 | 0.99 | 1.01 | 1.00 | 0.91 | 1.00 | 0.98 | 1.01 |
| SF 25 C | 0.98 | 1.02 | 1.00 | 1.01 | 1.17 (1.33 at 8x4 6T) | 1.00 | 1.08 | 1.00 |
| SS 25 C | 1.11 | 1.10 | 1.10 | 1.11 | 1.09 | 1.11 | 1.11 | 1.10 |
| TT 85 C | 1.37 | 1.38 | 1.40 | 1.42 | 1.50 | 1.40 | 1.44 | 1.40 |
| FF 125 C | 1.47 | 1.50 | 1.54 | 1.55 | 1.71 | 1.53 | 1.60 | 1.54 |
| TT 125 C | 1.65 | 1.69 | 1.72 | 1.75 | 1.90 (2.03 at 8x4 6T) | 1.72 | 1.78 | 1.72 |

Temperature dominates in these models (no temperature inversion at 1.0 V:
-40 C is the fastest condition), the slow process corner adds ~10 %, and the
write access is the most corner-sensitive phase (SF: strong PMOS pull-up
against the row-scaled driver). The control phases
(clock -> `wl_en`, decode, restore) agree within 5 %, the read access is
within 4 % of the largest of them and the write access differs by up to 15 %,
so a per-phase factor table is enough; apart from the 8x4 6T write access no
size dependence was visible between 8x4 and 16x16 (checked again up to 512x4
in the worst-case runs below).

**Worst case for this PDK.** The slowest combination has to be measured, not
assumed: SS / 125 C and SF / 125 C at 1.0 V and at 0.9 V (VDD -10 %) were
not in the V2.0.2 set. They were run for this proposal on 8x4, 16x16, 64x16,
256x8 and 16x256 (6T, read and write; 10T at 16x16), 64x64 (SS / 125 C /
1.0 V) and 512x4 (read); every deck at the default 10 ns period:

<!-- WC_TABLE_BEGIN -->
| condition | op | sizes (6T unless noted) | clk->wl_en | decode | restore | access | low | high | checks |
|---|---|---|---|---|---|---|---|---|---|
| SS / 125 C / 0.9 V | read | 10T 16x16, 16x16, 16x256, 256x8, 512x4, 64x16, 8x4 | 2.12-2.26 | 2.20-2.24 | 2.16-2.22 | 2.27-2.31 | 2.24-2.28 | 2.16-2.22 | all pass |
| SS / 125 C / 0.9 V | write | 10T 16x16, 16x16, 16x256, 256x8, 64x16, 8x4 | 1.96-2.24 | 2.07-2.24 | 2.08-2.20 | 2.31-2.45 | 2.11-2.32 | 2.08-2.20 | all pass |
| SS / 125 C / 1.0 V | read | 10T 16x16, 16x16, 16x256, 256x8, 512x4, 64x16, 64x64, 8x4 | 1.85-1.96 | 1.91-1.94 | 1.88-1.94 | 1.93-1.99 | 1.92-1.98 | 1.88-1.94 | all pass |
| SS / 125 C / 1.0 V | write | 10T 16x16, 16x16, 16x256, 256x8, 64x16, 64x64, 8x4 | 1.74-1.95 | 1.81-1.95 | 1.82-1.92 | 2.01-2.25 | 1.85-2.07 | 1.82-1.92 | all pass |
| SF / 125 C / 1.0 V | read | 16x16, 16x256, 256x8, 64x16, 8x4 | 1.63-1.72 | 1.72-1.74 | 1.69-1.72 | 1.78 | 1.74-1.76 | 1.69-1.72 | all pass |
| SF / 125 C / 1.0 V | write | 16x16, 16x256, 256x8, 64x16, 8x4 | 1.59-1.69 | 1.62-1.74 | 1.63-1.71 | 1.90-3.55 | 1.73-2.64 | 1.63-1.71 | all pass |
| FS / 125 C / 1.0 V | read | 16x16, 16x256, 256x8, 64x16, 8x4 | 1.66-1.74 | 1.67-1.73 | 1.68-1.76 | 1.68-1.73 | 1.69-1.73 | 1.68-1.76 | FAIL: 6T_256x8_read_FS_125C_v1.0; 6T_64x16_read_FS_125C_v1.0 |
| FS / 125 C / 1.0 V | write | 16x16, 16x256, 256x8, 64x16, 8x4 | 1.56-1.73 | 1.62-1.74 | 1.63-1.72 | 1.67-1.72 | 1.61-1.72 | 1.63-1.72 | all pass |
| FF / 125 C / 1.0 V | read | 16x256, 256x8, 64x16 | 1.45-1.52 | 1.53-1.54 | 1.50-1.54 | 1.52-1.55 | 1.52-1.54 | 1.50-1.54 | FAIL: 6T_256x8_read_FF_125C_v1.0; 6T_64x16_read_FF_125C_v1.0 |
| FF / 125 C / 1.0 V | write | 16x256, 256x8, 64x16 | 1.43-1.51 | 1.44-1.55 | 1.45-1.51 | 1.59-1.62 | 1.50-1.55 | 1.45-1.51 | all pass |
| TT / 25 C / 0.9 V | read | 16x16, 16x256, 256x8, 512x4, 64x16, 8x4 | 1.10-1.16 | 1.12-1.14 | 1.12-1.14 | 1.13-1.15 | 1.13-1.15 | 1.12-1.14 | all pass |
| TT / 25 C / 0.9 V | write | 16x16, 16x256, 256x8, 64x16, 8x4 | 1.11-1.17 | 1.12-1.13 | 1.11-1.13 | 1.09-1.14 | 1.12-1.16 | 1.11-1.13 | all pass |

The worst case is per phase, not one corner. SS / 125 C / 0.9 V is the slowest condition for every control phase (clock -> `wl_en` 2.12-2.26x, decode 2.20-2.24x, restore 2.16-2.22x) and for the read access (2.27-2.31x), with the same factor at every size run (the bitline term and the buffer terms scale together). The write access is worst at SF / 125 C / 0.9 V on arrays with fewer than 16 rows, where the write driver is at its 0.5x scale (8x4: 3.82x, `low` 2.89x), and at SS from 16 rows up (2.31-2.41x). The supply term (-10 %) is larger than the process term (SS vs TT). Every deck passes all waveform checks at the 10 ns period. Factors for the model, max over the 125 C / 0.9 V runs: k_wlen = 2.26, k_dec = 2.24, k_restore = 2.22, k_access_read = 2.31, k_access_write = 3.82 (rows < 16, SF) / 2.41 (rows >= 16); resulting k_low_read = 2.28, k_high = 2.22.
<!-- WC_TABLE_END -->

Resulting worst-case phases and periods for the characterised sizes (the
seed of the timing table; max over SS and SF at 125 C / 0.9 V, nominal
devices):

| cell | rows x cols | read low / high [ps] | write low / high [ps] | `2 * max` [ns] | period, 25 % margin, 50 ps grid [ns] | limiting phase |
|---|---|---|---|---|---|---|
| 6T | 8x4 | 925 / 547 | 732 / 483 | 1.85 | **2.35** | read |
| 6T | 16x16 | 987 / 563 | 535 / 537 | 1.97 | **2.50** | read |
| 6T | 16x256 | 1093 / 635 | 678 / 702 | 2.19 | **2.75** | read |
| 6T | 64x16 | 1111 / 550 | 533 / 559 | 2.22 | **2.80** | read |
| 6T | 256x8 | 1507 / 586 | 539 / 590 | 3.01 | **3.80** | read |
| 6T | 512x4 | 1986 / 602 | - / - | 3.97 | **5.00** | read |
| 10T | 16x16 | 1010 / 564 | 567 / 537 | 2.02 | **2.55** | read |

The read phase sets the period on every characterised size (at 512 rows the
bitline term alone is 1.6 ns at the worst case). Within the write operation
the restore is the longer phase except at 8x4, where the 0.5x write driver
makes the write access the longer one, and at 10T 16x16 (567 / 537 ps); at
6T 16x16 the two are within 2 ps.


### 3.5 Local variation

Seeded Monte Carlo (`vth_std = 0.05`, 5 samples, TT / 25 C): read delay
standard deviation 3.4-5.0 ps (1.1-1.6 % of the access, ~1 % of `low`),
write 1.6-3.9 ps at >= 16 rows and 5-12 ps at 8x4 (up to 7 % of the access,
4 % of `low`). At the worst case the spread is larger (5 seeded samples,
seed 2026, this proposal):

| deck | condition, period | limiting phase `low`: mean / sd / max [ps] | failing samples |
|---|---|---|---|
| 8x4 read | SS / 125 C / 0.9 V, 2.35 ns | 929 / 25 (2.7 %) / 962 | 0 of 5 (slack 213 ps) |
| 16x16 read | SS / 125 C / 0.9 V, 2.5 ns | 996 / 27 (2.7 %) / 1031 | 0 of 5 (slack 219 ps) |
| 8x4 write | SS / 125 C / 0.9 V, 10 ns | 636 / 63 (10 %) / 746 | 0 of 5 |
| 8x4 write | SF / 125 C / 0.9 V, 10 ns | 1463 / 1132 / 2768 | 2 of 5: BLB stays at 0.26 V, cell keeps its 0 for the whole 5 ns wordline phase |
| 8x4 write | SF / 125 C / 0.9 V, 1.85 ns | - | 3 of 5 (the same two plus the 2.8 ns sample) |
| 16x16 write | SF / 125 C / 0.9 V, 10 ns | 515 / 20 (4 %) / 548 (write access 241 / 14 / 264) | 0 of 5 |

Three sigma is therefore ~8 % of the limiting phase for reads and ~30 % of
the write `low` phase of an 8-row array at SS / 125 C / 0.9 V (the write
access alone has 16 % sigma). The SF failures are
not a timing effect: with the 10 ns clock the driver still cannot pull the
bitline below 0.26 V against the strong PMOS of a slow-NMOS sample, so it is
a DC write-ability limit of the 0.5x write driver that the row-scaling rule
(`max(8, rows) / 16`) gives to arrays with <= 8 rows at SF / 125 C / 0.9 V;
the 16x16 array (1x driver) writes every sample at the same corner.
It is exactly the kind of failure the yield flow should see at that corner;
it is listed as an open item in `docs/CHANGELOG.md` V2.0.3.

## 4. Proposed method

### 4.1 Options considered

| option | pros | cons |
|---|---|---|
| (a) lookup table of simulated sizes | exact where simulated | 512 rows x 512 cols x 2 cells x mux is not enumerable; interpolation on a log grid is itself a model |
| (b) fitted phase model (this proposal) | 4 coefficients per phase, physical terms (rows: bitline; log2(cols): taper stages), residual < 4 % | must be re-fitted when the periphery sizing rules change |
| (c) analytical RC model from the netlist sizes | no fitting | the buffer scales are step functions (`ceil(load / 8)`, 2 vs 4 stages at scale 16), so it is piecewise anyway and needs the same validation |
| (d) calibrate every size in the loop | always exact | 25 s .. 3 h per size, and the optimiser would move the spec with the candidate |

Recommendation: **(b) fitted on (a), derated by the measured PVT factors,
with (d) as the fallback for sizes outside the fitted range and for a
changed periphery.** (a) alone is what the flow already prints per run; (b)
is what turns it into a configuration.

### 4.2 Algorithm

```
timing_for_array(cell, rows, cols, mux, margin=0.25, worst_case=('SS', 125, 0.9)):
    key = (cell, rows, cols, mux, periphery_hash)
    if key in timing_table:                       # measured worst-case phases
        low_wc, high_wc = timing_table[key]
    elif (rows, cols) inside the fitted range and periphery_hash == fitted_hash:
        low_r  = model(cell, 'read',  mux, 'low')(rows, cols) * k_low_read
        high_r = model(cell, 'read',  mux, 'high')(rows, cols) * k_high
        low_w  = model(cell, 'write', mux, 'low')(rows, cols) * k_low_write
        high_w = model(cell, 'write', mux, 'high')(rows, cols) * k_high
        low_wc, high_wc = max(low_r, low_w), max(high_r, high_w)
    else:
        low_wc, high_wc = calibrate(key, worst_case)      # 1 read + 1 write deck
        timing_table[key] = (low_wc, high_wc)             # cached on disk
    T = 2 * max(low_wc, high_wc) * (1 + margin)
    T = ceil(T / 50 ps) * 50 ps                           # quantise so the spec is stable
    return TimingConfig(t_period=T, duty=0.5, t_rise=t_fall=0.01 T,
                        wl_pulse=T/2, precharge=T/2, w_en=T/2,
                        s_en='replica', latch='s_en', source='table'|'model'|'calibrated')
```

`k_*` are the worst-case factors of section 3.4: one number per phase,
constant over size for the control phases and the read access, and two
regimes for the write access (`rows < 16`: SF / 125 C / 0.9 V, driver at its
0.5x scale; `rows >= 16`: SS / 125 C / 0.9 V). Because the worst corner
differs per phase, `calibrate()` runs the read deck at SS / 125 C / 0.9 V
and the write deck at both SS and SF / 125 C / 0.9 V and keeps the larger
phase. `calibrate()` runs the existing `read` and `write` decks
with `mc_runs=1` at the worst-case corner / temperature / VDD and a
provisional period of `4 x` the model estimate, reads `TCLK_WLEN`,
`TREAD_TOTAL`, `TWRITE_TOTAL`, `TRESTORE`, `TCLK_DEC` from the `.stats.csv`
and stores the phases. The periphery hash covers the periphery YAMLs, the
cell YAML defaults and the sizing rules (`nand_scale`, `TaperedBuffer`), so
a re-sized periphery invalidates the table entry, not the flow.

### 4.3 Margin policy

`T = 2 * max(low_wc, high_wc) * (1 + margin)`, default `margin = 0.25`,
one YAML parameter. What it covers at the worst case:

| item | size | note |
|---|---|---|
| settling of the amplifier / output latch after the VDD/2 crossings the measures use | 0-8 % (section 3.1, worst-case sweeps) | inside every margin |
| local variation, 3 sigma | ~8 % of `low` for reads at the worst case (2.7 % sigma); ~30 % of the write `low` phase of 8-row arrays (10 % sigma; 16 % on the write access alone), < 12 % elsewhere | a 6-sigma target (large macros) needs ~2x these |
| model residual | 2-4 % | zero for table entries |
| stimulus setup / hold | `0.1 T` = 150-250 ps at the resulting periods | exceeds the DFF setup at the worst case (the sweeps of section 3.1 pass at `0.1 T` = 150 ps) |
| design margin | remainder | the usual allowance for un-modelled loads (RC, `w_rc`) |

25 % therefore covers reads with 3 sigma to spare on every size, and covers
writes wherever the restore phase is the limit (all arrays with >= 16 rows
in the table, restore sigma < 2 %). Where the write access itself limits
(8-row arrays, 0.5x driver), 25 % is about one sigma at SF / 125 C / 0.9 V,
and no period fixes the samples that cannot write at all (section 3.5).
`calibrate()` should therefore also run a 5-sample seeded Monte Carlo at the
worst case and print the sigma of the limiting phase; when
`3 * sigma_rel + 8 %` exceeds the configured margin the flow warns and
recommends the larger value (`margin = max(0.25, 3 * sigma_rel + 0.08)`).

The margin is applied to the phases at the worst-case PVT. A run at TT / 25 C
with that period therefore has ~2x slack, and Monte Carlo at TT finds
(almost) no timing failures: that is physically right for a chip timed for
SS / 125 C / 0.9 V, and the yield analysis that is supposed to see timing
failures should run at the same worst-case PVT (both flows accept `corner`,
`temperature` and `vdd`). Raising the margin moves the timing yield up; the
value should be chosen once per project (25 % is the proposal's default,
50 % a conservative alternative for 6-sigma studies).

### 4.4 What each signal becomes for a given array

| signal | configured value | example 16x16 6T, worst case SS / 125 C / 0.9 V, margin 25 % |
|---|---|---|
| clock period `T` | `2 * max(low_wc, high_wc) * 1.25`, 50 ps grid | low 987 ps, high 563 ps -> 2.47 ns -> **2.50 ns** |
| wordline / write-enable pulse | `T / 2` (clock-low) | 1.250 ns |
| precharge | `T / 2` (clock-high), plus the `wl_en_bar` overlap | 1.250 ns |
| sense enable | replica-timed, `TCLK_WLEN + TWL + ~250 ps * k` after the falling edge | unchanged |
| output latch | opens with `s_en` | unchanged |
| address / data setup, hold | `0.1 T` | 250 ps |
| `.TRAN` end, energy window, PSTC window | scale with `T` (see section 5) | - |

### 4.5 Sense-enable margin (phase 2)

`s_en` fires when the replica bitline, discharged by one replica cell, has
crossed the delay-chain input threshold plus nine inverter delays: at that
moment the target bitline is at ~0.02 V (full swing). This is the maximum
possible sensing margin, it tracks rows and PVT, and it costs ~150-200 ps of
read access against a differential-sensing design. For yield analysis it
hides the read-access failure mode (amplifier fired before enough
differential), so the proposal exposes two parameters without changing the
default:

- `n_replica_cells` `K` (1..4): `K` active replica cells in parallel on RBL
  make `rbl_delay` fire when the target bitline has swung ~`VDD / (2K)`;
- `delay_stages` `N` (0..9): the chain length after the replica bitline.

Rule for choosing `(K, N)` per array: the target-bitline differential at
`s_en` at the worst-case PVT must stay above `dV_min` = amplifier offset
(3 sigma of the SA pair from Monte Carlo) + 100 mV. The harness already
records `bl_at_sen`, so the characterisation is one sweep over `K` and `N`
per array size at the worst case; the values go into the same table as the
period. Default stays `K = 1, N = 9` until that sweep has been run.

### 4.6 Duty cycle (phase 2)

Reads waste 40-60 % of the clock-high phase (`high` = 220-300 ps against
`low` = 400-900 ps), so an asymmetric clock (`duty = low / (low + high)`,
clamped to 0.35..0.65) would shorten the period by 15-30 % on read-limited
arrays. The testbench, however, assumes 50 % in the capture-edge position
(`1 ns + 0.2 T` / `0.7 T`), the measure windows and the PSTC window, so this
is a separate change with its own period sweep.

## 5. Integration

- New module `sram_compiler/timing/timing_config.py`: `TimingConfig`
  (fields of section 4.4), `timing_for_array()`, the coefficient set
  (`timing_model.json`, generated from `phases.csv` by a fit script that is
  committed with it), the table (`timing_table.json`), and `calibrate()`
  (wraps `Sram6TCoreMcTestbench.run_mc_simulation` for one read and one
  write deck).
- `global.yaml`: a `timing` block
  (`mode: auto | fixed`, `t_period`, `margin: 0.25`,
  `worst_case: {corner: SS, temperature: 125, vdd: 0.9}`,
  `table: sram_compiler/timing/timing_table.json`).
- `Sram6TCoreMcTestbench.__init__(..., timing=None)`: `None` keeps 10 ns
  (backward compatible), `'auto'` calls `timing_for_array()` and applies it
  through `set_timing_parameters()`; the applied period and its source are
  printed and written next to the `.stats.csv`.
- `size_optimization/exp_utils.py` and `main_estimation.py`: create the
  testbench with `timing='auto'` once, before the loop; the period is part of
  the experiment record.
- Testbench changes needed for periods of 1-3 ns (all already flagged in
  V2.0.2): the PSTC window `1 ns + [0.4, 0.65] T` overlaps the start-up
  precharge for `T < 5 ns` (move the quiescent window to a dedicated idle
  cycle, e.g. run the first access one cycle later); `.TRAN` start-up offset
  of 1 ns is fine; the stimulus setup of `0.1 T` must be checked against the
  DFF setup at the worst case (section 6).
- `_print_min_period()` keeps printing the measured `T_min` of every run;
  when `timing='auto'` it also prints the slack against the configured period
  (negative slack = the candidate failed the spec).

## 6. Validation plan

1. Worst-case characterisation (done for this proposal, section 3.4): read
   and write at SS / SF / FS / FF x 125 C x 1.0 V and SS / 125 C / 0.9 V and
   TT / 25 C / 0.9 V on 7 sizes; confirms which corner is the worst per phase
   and that the factors are size-independent.
2. Period sweep at the worst case on 8x4, 16x16, 64x16 and 256x8: the deck
   must pass at `T = 2 * max(low_wc, high_wc) * 1.0` (the estimate) and fail
   one 100 ps step below; this checks the DFF setup at `0.1 T` as well.
3. Seeded Monte Carlo (5-10 samples) at the worst case with the configured
   period on 8x4, 16x16, 64x16: all samples must pass with the 25 % margin.
4. Model against table: for every characterised size the model prediction
   must be within 5 % of the measured worst-case phases.
5. Optimiser smoke test: `demo_*` runs on 32x1 with `timing='auto'`; a
   candidate with the cell pass gate at its lower bound must fail the read
   spec (delay `FAILED` or wrong data) instead of returning a longer delay.

## 7. Effort

| step | size |
|---|---|
| `timing_config.py` with model + table + calibrate | ~250 lines, 1 day |
| YAML / testbench / optimiser / yield hooks | ~80 lines, 0.5 day |
| PSTC idle-cycle window | ~40 lines, 0.5 day incl. re-validation |
| validation runs (section 6, items 2-5) | ~1 day of machine time |
| phase 2 (`K`, `N`, duty) | 2-3 days incl. sweeps |

## 8. Decisions needed

1. Default margin: 25 % (proposed) or a project value.
2. Worst-case PVT definition: the characterisation supports 125 C / 0.9 V with
   SS for the control phases and the read access and SF for the write access
   of small arrays (section 3.4); if the product spec is different (e.g.
   1.0 V only, 85 C), the factors are re-read from the same runs (the 1.0 V
   values are in the table).
3. Whether the optimiser keeps the baseline spec (proposed) or is allowed to
   trade period against size (that turns the period into an objective, and
   the optimiser would need `T` as a variable with its own cost).
