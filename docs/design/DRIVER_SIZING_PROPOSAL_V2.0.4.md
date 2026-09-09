# Automatic driver sizing for all array sizes — proposal (V2.0.4)

Status: proposal with a measured basis; not implemented. Source revision
`d30a23b` plus the uncommitted V2.0.3 documents (`TIMING_AUTOCONFIG.md`).
The numbers in section 3 come from a characterisation campaign run for this
proposal (163 Xyce decks, `DRIVER_SIZING_data.csv`, harness in the session
scratchpad, see section 3.0);
everything else is derived from the code. This revision replaces the earlier
draft of this file; section 10 lists what was kept from it.

## 0. Recommendation in short

1. **Every critical driver already has a load-proportional rule; turn each
   into `scale = max(floor, load term)` and calibrate both at that driver's
   own worst corner.** Today the precharge scales with `rows/16`, the write
   driver with `max(8, rows)/16`, the wordline driver with `cols/4` (NAND2
   with its square root) and every control buffer with its fan-out. The one
   functional failure found so far — an 8-row array that cannot be written
   at SF / 125 C / 0.9 V — is a *floor* problem, not a load problem: the
   strength a write driver needs to flip a cell does not depend on the row
   count, so a rule that only scales with rows has no floor and gives small
   arrays a driver that is weaker than the cell it must overpower.
2. **Requirement per driver, measured at its worst corner, with an explicit
   margin** (section 2): the precharge restores every bitline inside the
   clock-high phase with slack against the read phase; the write driver
   flips the strongest cell of the optimiser's parameter box at
   SF / 125 C / 0.9 V with every global-variation sample and finishes the
   write before the read would; the wordline driver keeps the wordline path
   below a fixed fraction of the read access; the replica path tracks the
   real wordline and fires the amplifier inside a two-sided window; the
   control buffers keep a fan-out of ~8 per stage. Speed terms are sized at
   SS / 125 C / 0.9 V, write-ability at SF / 125 C / 0.9 V, leakage checks
   at FF and FS / 125 C, sensing margin against local mismatch.
3. **One resolver, frozen per array.** `resolve_driver_sizes(cell, rows,
   cols, mux)` computes all scales from the YAML base widths, then the TIME
   buffer loads *from those scales* (today the testbench recomputes the
   loads from the rules by hand), and the factories consume the immutable
   result. Like the clock period of `TIMING_AUTOCONFIG.md`, the sizes are
   derived once from the baseline cell and frozen as the spec for size
   optimisation and yield analysis; a candidate cell that the frozen
   periphery cannot write or read fails, the periphery is not re-sized around it.
4. **Replica-timing drivers become a copy of the real path:** the replica
   wordline is driven by the same NAND2 + inverter wordline driver as every
   row, loaded by a dummy row of `cols` cells, instead of a fixed AND2 that
   drives one cell (today the replica wordline rises 46-101 ps before the real one at
   256 columns, section 3.3). The number of active replica cells
   `K` and delay-chain stages `N` stay `(1, 9)` by default; the table of
   section 4.3 gives the sensing margin for the alternatives.
5. **Verification is three decks per array** (read at SS, write at SS and
   SF, all 125 C / 0.9 V, with the automatic period) whose waveform checks
   and per-driver acceptance metrics (section 2.2) must pass with 10
   global-variation samples; accepted vectors go into a table keyed by a
   hash of the cell and periphery YAMLs, so the optimiser never simulates
   for sizing.

## 1. What is sized today

### 1.1 The drivers and their rules

All widths in um, channel length 0.05 um. "unit" = one 0.09 / 0.27 um
inverter input (0.36 um of gate). R = rows, C = columns of the subarray.

| driver | rule (scale on the YAML base width) | base widths | drives | is limited by / worst corner | code |
|---|---|---|---|---|---|
| precharge PMOS x3 per column (+ replica column) | `max(0.5, R/16)` | 0.27 | one bitline each (R pass-gate drains); the equaliser bridges BL/BLB | restore time in the clock-high phase, SS / 125 C / 0.9 V | `PrechargeFactory.width_scale`, `Precharge` |
| write driver, 12 transistors per column, one width pair | `max(8, R)/16` | Wn 0.18 / Wp 0.36 | two series stacks per bitline (2 PMOS up, 2 NMOS down) against the cell; the DIN / EN input inverters have the same width | DC write-ability against the cell PMOS through the pass gate, SF / 125 C / 0.9 V; speed at SS | `WriteDriverFactory.width_scale`, `WriteDriver` |
| wordline driver (NAND2 + inverter), one per row | inverter `max(C, 4)/4`, NAND2 `sqrt(inverter scale)` | inv 0.09 / 0.27, NAND 0.18 / 0.27 | C cells x 2 pass-gate gates; the NAND2 is driven by the decoder's fixed 0.09 / 0.27 output inverter and by `wl_en` | wordline rise (read / write start) and fall (address hold), SS / 125 C / 0.9 V | `WordlineDriverFactory.inv_scale / nand_scale` |
| replica wordline driver | fixed AND2 (0.18 / 0.27 NAND2 + 0.09 / 0.27 inverter) | - | one replica cell (the other R replica cells are tied to VSS) | must track the real wordline | `create_and2_for_rwl` |
| replica bitline -> `rbl_delay` | 9 inverter stages, 4 unit loads each, fixed size | 0.09 / 0.27 | AND3 of `s_en` | ~250 ps at TT, ~2.3x at the worst case; sets the sensing margin | `DelayChain` |
| `wl_en` buffer | `wl_pdrive`, scale `ceil(R * nand_scale / 32)` | 0.45 / 1.35 output | R NAND2 inputs + replica AND2 | fan-out <= 8 | `TIME` |
| `PRE`, `w_en`, `s_en`, `sa_iso`, address buffers | `TaperedBuffer`, scale `ceil(load / 8)`, 2 stages up to 16, 4 above | 0.09 / 0.27 x scale | 3 precharge gates per column, write-driver EN gates, SA footers, SA pass gates, decoder inputs | fan-out <= 8 | `TIME` |
| clock buffer | `pdrive`, scale = DFF count / reference count | 4 stages to 2.43 / 7.29 | address, data, CS, WE flip-flops | - | `TIME` |

### 1.2 What the rules produce

Computed from the code for representative sizes (mux off). `h` is the
electrical effort of a stage (gate width driven / gate width of the driving
inverter input); `h_dec` is the effort of the decoder's fixed output
inverter into the wordline-driver NAND2.

| R x C | precharge W (x3/col) | PRE load [units] -> buffer scale | write driver Wn/Wp | w_en load -> buffer | WL inverter Wn/Wp | WL NAND2 Wn/Wp | h_inv | h_nand | h_dec | wl_en load -> scale |
|---|---|---|---|---|---|---|---|---|---|---|
| 8x4 | 0.14 | 5.6 -> 1 | 0.09 / 0.18 | 13 -> direct | 0.09 / 0.27 | 0.18 / 0.27 | 3.0 | 0.8 | 1.2 | 8 -> 1 |
| 16x16 | 0.27 | 38 -> 5 | 0.18 / 0.36 | 60 -> 8 | 0.36 / 1.08 | 0.36 / 0.54 | 3.0 | 1.6 | 2.5 | 32 -> 1 |
| 64x16 | 1.08 | 153 -> 20 | 0.72 / 1.44 | 180 -> 23 | 0.36 / 1.08 | 0.36 / 0.54 | 3.0 | 1.6 | 2.5 | 128 -> 4 |
| 64x64 | 1.08 | 585 -> 74 | 0.72 / 1.44 | 720 -> 90 | 1.44 / 4.32 | 0.72 / 1.08 | 3.0 | 3.2 | 5.0 | 256 -> 8 |
| 256x8 | 4.32 | 324 -> 41 | 2.88 / 5.76 | 332 -> 42 | 0.18 / 0.54 | 0.25 / 0.38 | 3.0 | 1.1 | 1.8 | 362 -> 12 |
| 512x4 | 8.64 | 360 -> 46 | 5.76 / 11.52 | 328 -> 41 | 0.09 / 0.27 | 0.18 / 0.27 | 3.0 | 0.8 | 1.2 | 512 -> 16 |
| 16x256 | 0.27 | 578 -> 73 | 0.18 / 0.36 | 960 -> 120 | 5.76 / 17.28 | 1.44 / 2.16 | 3.0 | 6.4 | 10.0 | 128 -> 4 |
| 16x512 | 0.27 | 1154 -> 145 | 0.18 / 0.36 | 1920 -> 240 | 11.52 / 34.56 | 2.04 / 3.05 | 3.0 | 9.1 | 14.1 | 181 -> 6 |
| 128x128 | 2.16 | 2322 -> 291 | 1.44 / 2.88 | 2720 -> 340 | 2.88 / 8.64 | 1.02 / 1.53 | 3.0 | 4.5 | 7.1 | 724 -> 23 |

Three things stand out:

- **The write driver has no floor.** At 8 rows it is 0.09 / 0.18 um, and
  its pull-down is two of those NMOS in series (EN-gated M7 / M11 above the
  data-gated M8 / M12), i.e. ~0.045 um of effective NMOS against the cell's
  0.09 um PMOS pull-up through a 0.135 um pass gate. At SF (weak NMOS,
  strong PMOS), 125 C and 0.9 V that is not enough (section 3.1).
- **The wordline path is unbalanced at wide arrays.** The inverter keeps
  `h = 3` by construction, but the NAND2 and the decoder output stage see
  `h = 0.8 sqrt(C/4)` and `1.25 sqrt(C/4)`: 9 and 14 at 512 columns. The
  decoder's 0.09 / 0.27 output inverter (fixed in `decoder.yaml`) is then
  the slowest stage of the row path, and the wordline inverter itself is
  larger than a balanced taper would make it.
- **The precharge and write-driver widths feed straight into the largest
  buffers of the timing block** (`PRE` 291x, `w_en` 340x at 128x128):
  the precharge PMOS is scaled by rows *and* instantiated per column, so its
  gate load grows with `R * C`. Where the restore has slack the width can
  come down and the buffer with it.

### 1.3 What the netlist actually loads the drivers with

The transistor instances carry no `AS / AD / PS / PD`, so the bitline sees
per row only the gate-edge sidewall junction (`cjswgd` 0.5 fF/um x 0.135 um
= 0.07 fF) and the overlap / fringe capacitance (`cgdo`, `cgdl`: 0.02-0.05
fF) of one pass-gate drain: **0.06-0.12 fF per row**, against 0.2-0.3 fF
per cell (junction area + wire) in a laid-out 45 nm array. The wordline
sees two pass-gate gates per column (~0.45 fF). Wire RC exists only as the
optional `w_rc` lumped branch per cell (100 Ohm / 1 fF), not as a
distributed line, so there is no far-end wordline or bitline delay.

Consequences for this proposal: the *method* (floors, load terms, worst
corners, margins, resolver) does not depend on the absolute loads, but the
*coefficients* of the load terms fitted in section 3 are optimistic by
2-3x for the bitline and unknown for the wordline wire. Section 5 therefore
carries a `parasitic_factor` on the load terms; a physical design has to
set it from extraction (or run with `w_rc` and calibrated `pi_res / pi_cap`)
and re-run the characterisation of section 7.

## 2. Requirements

### 2.1 What "timing requirement" means for the drivers

`TIMING_AUTOCONFIG.md` sets the clock period per array from the two
clock phases measured at the worst case, `T = 2 * max(low, high) * (1 +
margin)`, with `low = TCLK_WLEN + access` (wordline phase) and `high =
max(TRESTORE, TCLK_DEC)` (precharge / decode phase). The read access is
bitline-limited (one cell discharging `R` rows of bitline, 0.95 ps per row
nominal, 2.3x at the worst case) — the physical floor that no periphery
driver changes. The requirement on the drivers is therefore:

1. **No driver term may become the limiting phase.** `TRESTORE` (precharge),
   `TCLK_WLEN + TWLDRV` (control + wordline path), the write access (write
   driver) and `TCLK_DEC` (decoder) each stay below a fixed fraction
   `alpha` of the bitline-limited read phase at the worst case. Then the
   period of `TIMING_AUTOCONFIG.md` is set by the array, not by the
   periphery, and its 25 % margin covers the drivers as well.
2. **Function first.** Independently of the period, every driver has a DC /
   functional requirement that holds for any clock: the precharge restores
   and equalises, the write driver flips the strongest cell, the wordline
   reaches VDD, the amplifier is fired only after enough differential.
3. **The `global.yaml` limits (`delay.upper` = 200 ps read, 100 ps write)**
   are access limits measured from `wl_en`. The baseline read is 290-310 ps
   at TT for every size up to 32 rows because the replica timing waits for a
   full bitline swing plus nine delay stages (~250 ps); no driver width
   changes that. It is a replica-configuration decision (section 4.3): with
   `(K, N) = (2, 1)` the 8x4 read is 113 ps and the 16x16 read 127 ps at
   TT (measured, section 3.4).
   The proposal keeps `(1, 9)` as the default and reports the read spec as
   not met by the baseline; the optimiser's constraint should use the
   period-derived budget (item 1) or a replica configuration chosen for
   the spec, not silently pass.

### 2.2 Requirement per driver

| driver | functional requirement (any clock) | timing requirement (worst case) | sized at | checked at | acceptance metric (existing measure / waveform check) |
|---|---|---|---|---|---|
| precharge | BL, BLB and RBL back above 0.98 VDD and within 5 mV of each other before the next wordline rises; after a full write swing (BLB at 0) as well as after a read | `TRESTORE_wc <= alpha_pre * low_read_wc` (default `alpha_pre` = 0.8) so the restore is never the limiting phase | SS / 125 C / 0.9 V | SS (speed), FF / 125 C (leakage: bitline held at VDD, unselected bitline droop only during the wordline phase) | `TRESTORE`, `TPRCH`, `bl_at_acc0` / `blb_at_acc0`, `bl_recovered` |
| write driver | flips the strongest cell of the parameter box (max PU, min PG, max PD) at SF / 125 C / 0.9 V with every global-variation sample; the driven bitline reaches < 0.1 VDD; the cell keeps the new data after `w_en` / WL release | `TCLK_WLEN + TWRITE_TOTAL_wc <= alpha_w * low_read_wc` (default 0.8): a write never sets the period | SF / 125 C / 0.9 V (floor), SS / 125 C / 0.9 V (speed) | SF and SS, 10 samples | `TWRITE_TOTAL`, `TWDRV`, `blb_driven_low`, `q_written`, `q_retained` |
| wordline driver | WL reaches > 0.9 VDD; WL falls before the precharge turns on (`wl_en_bar` gating) and before the next decoder output (address latch) | `TWLDRV_wc + slew_wc <= alpha_wl * access_read_wc` (default 0.15); the decoder stage effort bounded (`h_dec <= 8`) | SS / 125 C / 0.9 V | SS; `next_row` hazard deck at 256 / 512 rows | `TWLDRV`, `wl_slew`, `nb_wl_no_glitch` |
| replica wordline | rises within +-10 ps of the real wordline at every size and corner (same driver, same load) | - | - | SS, FF -40 C | `rwl_minus_wl` |
| replica bitline + delay chain (`K`, `N`) | amplifier fired only when the target bitline differential exceeds `dV_min` (0.3 V placeholder: SA offset 3 sigma + replica / cell mismatch 3 sigma + 100 mV) for the slowest cell / fastest replica; and early enough that `OUT` settles before the clock edge at the configured period | `sen_minus_wl + TSA + latch <= low_read_wc - TCLK_WLEN` | SS / 125 C / 0.9 V (late side), FF / -40 C + local mismatch (early side) | both, per-device MC for the mismatch | `dv_at_sen`, `sen_minus_wl`, `TSA`, `out_end_acc` |
| control buffers (`wl_en`, `PRE`, `w_en`, `s_en`, `sa_iso`, address) | edge 10-90 % <= 40 ps at TT regardless of size (V2.0.2 rule), no coupling glitch on `s_en` | included in `TCLK_WLEN`, `TRESTORE` | fan-out rule, checked at SS | SS | `TCLK_WLEN`, `TS_EN`, `sen_no_glitch_before` |

`alpha` values are proposal defaults (section 9); they are fractions of the
worst-case phases, so the drivers inherit the same PVT derating as the
period.

## 3. Measured basis

### 3.0 Harness

The V2.0.3 sweep harness (`run_one.py`) was extended with per-run overrides
of the sizing rules and of the replica hookup, applied by monkeypatching the
factories before the deck is built:

| switch | overrides |
|---|---|
| `--wd-scale s` | `WriteDriverFactory.width_scale` -> `s` (all 12 transistors) |
| `--wd-out-scale o` | the two tristate stacks at `o`, the DIN / EN inverters at `--wd-scale` |
| `--pre-scale s` | `PrechargeFactory.width_scale` -> `s` |
| `--wl-inv-scale s`, `--wl-nand-scale n` | `WordlineDriverFactory.inv_scale / nand_scale` |
| `--rwl matched` | replica wordline driven by the row `WordlineDriver` plus a dummy row of `cols` cells (bitlines tied to VDD) |
| `--rwl-k K` | `K` of the `rows + 1` replica cells on RWL |
| `--dc-stages N` | delay chain of `N` (odd) stages |
| `--cell-pu / --cell-pg / --cell-pd` | cell widths (parameter-box corners) |

The waveform scorer records in addition the wordline 10-90 % slew, the
replica-wordline / wordline skew, the bitline differential and replica
level at `s_en`, and for writes the driven-bitline level and the time from
the wordline to the cell flip. The TIME buffer loads follow the overridden
scales automatically (the testbench derives them from the factory rules).
All decks: 6T unless noted, mux off, nominal devices unless `mc10` (10
seeded global-variation samples, `vth_std` 0.05, seed 2026), 10 ns clock
unless noted, target cell = last row / last column.
`DRIVER_SIZING_data.csv` holds one row per deck (group, tag, waveform
result, the measures and the scorer quantities used in the tables below);
the raw decks and waveforms stay in the scratchpad `runs/` tree.

### 3.1 Write driver

Sweep of the write-driver scale at SF / 125 C / 0.9 V (the write-ability
corner). "rule" marks the value `max(8, rows)/16` gives today. Write access
= `TWRITE_TOTAL` (`wl_en` -> Q at 90 %); MC = 10 seeded global-variation
samples, mean / worst; the driven-bitline level is the worst sample.

| array, cell | scale (Wn / Wp um) | write access nominal [ps] | MC mean / worst [ps] | TWDRV [ps] | driven BLB min [V] | result |
|---|---|---|---|---|---|---|
| 8x4 default | 0.5 (0.09 / 0.18) **rule** | 499 | 364 / 420 (7 samples) | 119 | 0.269 | **3 of 10 samples do not write** |
| | 0.625 | 259 | - | 101 | 0.000 | pass |
| | 0.75 | 215 | 204 / 245 | 89 | 0.001 | 10 / 10 |
| | 0.875 | 192 | - | 80 | 0.000 | pass |
| | 1.0 (0.18 / 0.36) | 177 | 168 / 195 | 72 | 0.000 | 10 / 10 |
| | 1.25 | 160 | - | 62 | 0.000 | pass |
| | 1.5 (0.27 / 0.54) | 148 | 141 / 160 | 55 | 0.000 | 10 / 10 |
| | 2.0 (0.36 / 0.72) | 136 | - | 47 | 0.000 | pass |
| 8x4 box cell (PU 0.108, PG 0.108, PD 0.246) | 0.5 **rule** | - | - | - | 0.253 | **fails with nominal devices** |
| | 0.75 | - | - | - | 0.204 | **fails with nominal devices** |
| | 1.0 | 236 | 232 / 343 | 70 | 0.000 | 10 / 10 |
| | 1.5 | 176 | 168 / 201 | 53 | 0.000 | 10 / 10 |
| | 2.0 | 158 | 151 / 176 | 46 | 0.000 | 10 / 10 |
| | 3.0 | 162 | - | 34 | 0.000 | pass (slower than 2.0: self-loading) |
| 8x4 default, split: inputs 0.5, stacks o | o = 1.0 | 173 | - | 70 | 0.000 | pass |
| | o = 1.5 | 139 | 132 / 151 | 51 | 0.000 | 10 / 10 |
| | o = 2.0 | 124 | - | 40 | 0.000 | pass |
| 8x4 default, SS / 125 C / 0.9 V | 0.5 / 0.75 / 1.0 / 1.5 / 2.0 | 321 / 208 / 177 / 151 / 141 | - | 121 / 90 / 72 / 55 / 48 | 0.000 | pass |
| 8x4 10T | 0.5 **rule** | 344 | 327 / 411 | 120 | 0.001 | 10 / 10 |
| | 1.0 | 204 | - | 72 | 0.000 | pass |
| 16x16 default | 0.5 | 609 | - | 147 | 0.001 | pass (at the DC limit) |
| | 0.75 | 271 | 258 / 308 | 106 | 0.001 | 10 / 10 |
| | 1.0 **rule** | 226 | 214 / 247 | 86 | 0.000 | 10 / 10 |
| | 1.5 / 2.0 | 189 / 173 | - | 66 / 55 | 0.000 | pass |
| 16x16 box cell | 1.0 / 1.5 / 2.0 | 286 / 216 / 194 | - | 82 / 63 / 53 | 0.000 | pass |
| 64x16 default | 1.0 / 2.0 / 4.0 **rule** / 8.0 | 366 / 245 / 211 / 189 | - | 165 / 95 / 58 / 40 | 0.002 | pass; write PAVG 145 / 145 / 157 / 181 uW |
| 256x8 default | 4.0 / 8.0 / 16.0 **rule** | 313 / 247 / 210 | - | 138 / 80 / 52 | 0.002 | pass; write PAVG 207 / 219 / 244 uW |

What the sweep shows:

1. **The floor is set by the cell, not by the rows.** Below a certain scale
   the write access diverges and then the cell does not flip at all: the
   default cell fails 3 of 10 samples at 0.5x (bitline stuck at 0.27 V) and
   writes every sample at 0.75x; the parameter-box cell (strongest PMOS,
   weakest pass gate the optimiser may choose) fails at 0.75x with nominal
   devices and writes every sample at 1.0x (worst 343 ps). The 16-row array
   with the same 0.5x driver takes 609 ps, the 8-row array 499 ps: the row
   count changes the time, the cell decides whether it flips. **Floor
   `wd_floor` = 1.0** (box cell, 10 / 10); with the 1.5x strength margin of
   section 5 the resolved floor is **1.5** (worst box-cell sample 201 ps,
   worst default-cell sample 160 ps). The 10T cell writes at 0.5x
   (10 / 10, worst 411 ps); the 6T floor is kept for it as the
   conservative choice until it has its own box-cell sweep.
2. **Above the floor the load term is a speed choice.** Fitted on 64 and
   256 rows, `write access = 170-190 ps + 2.1-3.2 ps * rows / scale` at
   SF / 125 C / 0.9 V; the current `rows/16` (`rows/scale` = 16) gives
   210-226 ps at every size from 16 rows up, i.e. the ~100 ps TT write
   access the `global.yaml` limit asks for. Half of it (`rows/32`) costs
   +35-40 ps (245-247 ps), a quarter +100-150 ps; all of them are far
   inside the write budget of section 2.2 (`0.8 * low_read_wc` = 0.9-1.2 ns
   at these sizes), so the coefficient is a specification decision
   (section 9), not a functional one. Beyond 2x at 8 rows the driver gets
   slower again (3.0x: 162 ps vs 158 ps): the DIN / EN inverters and the
   `w_en` buffer load grow with the same width.
3. **Only the output stacks need the floor.** Keeping the DIN / EN inverters
   at 0.5x and the stacks at 1.5x writes 6 % faster than 1.5x everywhere
   (139 vs 148 ps; 10 / 10 samples, worst 151 ps) with 25 % less `w_en`
   load per column (`2 Wn_out + Wn_in + Wp_in` = 0.81 um instead of
   `3 Wn + Wp` = 1.08 um).
4. **The driver drains load the bitline.** The two stack transistors on each
   bitline add `cjswg * (Wn + Wp)` of gate-edge junction: 0.27 fF per column
   at 1.0x, 8.6 fF at the 32x the rule gives 512 rows — a third of the
   ~26 fF of cells on that bitline (section 1.3). The read deck has no write
   drivers (the canonical-macro point of the previous draft), so it
   under-estimates the read access of a macro with row-scaled write drivers
   at large row counts; the split of item 3 and the smallest acceptable
   coefficient in item 2 limit that load.

### 3.2 Precharge

Sweep of the precharge PMOS scale at SS / 125 C / 0.9 V, read and write
decks (10 ns clock; `T` marks the period `TIMING_AUTOCONFIG.md` would
assign). `TRESTORE` = rising clock edge -> discharged bitline back at
0.9 VDD; `TPRCH` = `PRE` fall -> bitline at 0.9 VDD from 0 V (start-up
precharge, the raw PMOS charging time); the bitline levels are read at the
next wordline rise (VDD = 0.9 V here).

| array | scale (W um) | rows / scale | TRESTORE read / write [ps] | TPRCH [ps] | BL, BLB at next WL [V] | read access [ps] | read PAVG [uW] |
|---|---|---|---|---|---|---|---|
| 8x4 | 0.25 (0.07) | 32 | 652 / 528 | 269 | 0.900 / 0.900 | 661 | 23.3 |
| | 0.5 (0.14) **rule** | 16 | 548 / 483 | 162 | 0.900 / 0.900 | 665 | 23.5 |
| | 1.0 (0.27) | 8 | 492 / 455 | 103 | 0.900 / 0.900 | 671 | 24.0 |
| | 2.0 (0.54) | 4 | 478 / 456 | 77 | 0.900 / 0.900 | 683 | 24.9 |
| 16x16 | 0.25 | 64 | 743 / 637 (623 at T = 2.5 ns) | 309 | 0.899 / 0.900 | 696 | 39.1 |
| | 0.5 | 32 | 615 / 564 (549) | 180 | 0.900 / 0.900 | 699 | 39.9 |
| | 1.0 **rule** | 16 | 563 / 537 (523) | 120 | 0.900 / 0.900 | 705 | 41.5 |
| | 2.0 | 8 | 536 / 526 | 82 | 0.900 / 0.900 | 717 | 44.7 |
| | 4.0 | 4 | 534 / 540 | 53 | 0.900 / 0.900 | 740 | 52.0 |
| 64x16 | 1.0 | 64 | 609 / 592 | 163 | 0.899 / 0.900 | 790 | 67.8 |
| | 2.0 | 32 | 562 / 558 | 106 | 0.900 / 0.900 | 800 | 71.0 |
| | 4.0 **rule** | 16 | 550 / 559 | 68 | 0.900 / 0.900 | 822 | 78.4 |
| | 8.0 | 8 | 542 / 557 | 49 | 0.900 / 0.900 | 863 | 91.1 |
| 256x8 | 4.0 | 64 | 608 / 616 (604 at T = 3.8 ns) | 120 | 0.899 / 0.900 | 1081 | 112.5 |
| | 8.0 | 32 | 593 / 600 (586) | 79 | 0.900 / 0.900 | 1129 | 120.3 |
| | 16.0 **rule** | 16 | 585 / 590 (578) | 58 | 0.900 / 0.900 | 1209 | 133.7 |
| | 32.0 | 8 | 588 / 594 | 45 | 0.900 / 0.900 | 1364 | 160.5 |

What the sweep shows:

1. **The restore is control-path limited.** `TRESTORE` = 440-590 ps of
   fixed delay (clock buffer, `NAND3`, `PRE` buffer; growing with the
   columns through the buffer taper) plus an RC term that is flat for
   `rows / scale <= 16` (< 15 ps) and adds 30-110 ps at `rows / scale` =
   32-64, most at the small arrays where the 0.25x device is 0.07 um wide.
   Every point is inside the budget `0.8 * low_read_wc` (740 / 790 / 889 /
   1206 ps for the four arrays); the tightest is 8x4 at 0.25x (652 ps).
2. **Width costs read access and power.** The precharge drains sit on the
   bitline: the read access grows 3 % (64x16, 1x -> 4x), 12 % (256x8,
   4x -> 16x) and 26 % (256x8, 32x), and the read power 16-19 % over the
   same steps (the `PRE` buffer scales with the load). Today's rule at 256
   rows (16x, 4.32 um x 3 per column) therefore costs 12 % of the read
   access for a restore that is 23 ps faster than 4x.
3. **Equalisation and restore are complete at every width**: BL and BLB
   are at VDD within 1 mV at the next wordline in every deck, also at the
   automatic periods (1.25 ns and 1.9 ns clock-high phases).

Rule: the smallest width whose RC term stays on the flat part,
`pre = max(0.5, p * rows / 32)` — half of today's rule in the netlist
(`p` = 1), today's rule at `p` = 2 — plus the budget check
`TRESTORE_wc <= alpha_pre * low_read_wc` in `verify()`. Against the
current rule this saves 4-10 % read power and 1-6 % read access at
>= 64 rows and costs 8-12 ps of restore (27-52 ps at 16 rows).

### 3.3 Wordline driver

Sweep of the wordline-driver inverter scale (NAND2 at the square root of
it unless noted) on read decks at SS / 125 C / 0.9 V. `TWLDRV` = `wl_en`
-> WL at VDD/2; slew = WL 10-90 %; "RWL - WL" = replica wordline rise
minus real wordline rise with the fixed AND2 replica driver.

| array | inverter scale (Wn / Wp um) | NAND2 scale | h_inv | h_nand | TWLDRV [ps] | slew [ps] | RWL - WL [ps] | read access [ps] | TCLK_DEC [ps] | read PAVG [uW] |
|---|---|---|---|---|---|---|---|---|---|---|
| 8x4 | 0.5 / 1.0 **rule** / 2.0 | 0.71 / 1 / 1.41 | 6 / 3 / 1.5 | 0.6 / 0.8 / 1.1 | 44 / 40 / 40 | 39 / 27 / 24 | -7 / -4 / -3 | 664 / 665 / 665 | 370 / 372 / 374 | 23.3 / 23.5 / 23.9 |
| 16x16 | 1 / 2 / 4 **rule** / 8 | 1 / 1.41 / 2 / 2.83 | 12 / 6 / 3 / 1.5 | 0.8 / 1.1 / 1.6 / 2.3 | 60 / 53 / 53 / 52 | 65 / 39 / 30 / 28 | -22 / -14 / -12 / -12 | 703 / 704 / 705 / 704 | 438 / 441 / 444 / 448 | 40.0 / 40.6 / 41.5 / 43.4 |
| 64x64 | 4 / 8 / 16 **rule** / 32 | 2 / 2.83 / 4 / 5.66 | 12 / 6 / 3 / 1.5 | 1.6 / 2.3 / 3.2 / 4.5 | 71 / 66 / 67 / 72 | 66 / 45 / 40 / 39 | -30 / -25 / -26 / -32 | 883 / 882 / 881 / 882 | 481 / 485 / 490 / 497 | 197 / 203 / 211 / 226 |
| 16x256 | 16 / 32 / 64 **rule** / 128 | 4 / 5.66 / 8 / 11.3 | 12 / 6 / 3 / 1.5 | 3.2 / 4.5 / 6.4 / 9.1 | 88 / 86 / 93 / 107 | 76 / 56 / 54 / 60 | -48 / -46 / -53 / -67 | 804 / 807 / 806 / 805 | 454 / 461 / 471 / 484 | 307 / 312 / 321 / 337 |
| 16x256, inverter 64 | NAND2 4 / 8 **rule** / 16 | 3 | 12.8 / 6.4 / 3.2 | 141 / 93 / **67** | 83 / 54 / **39** | -101 / -53 / -27 | 807 / 806 / 805 | 454 / 471 / 503 | 316 / 321 / 330 |

What the sweep shows:

1. **The wordline is fastest where the two stage efforts are equal,
   `h_nand ~ h_inv ~ 3`.** At 64 columns the square-root rule already gives
   that (3.2 / 3: 66-67 ps); at 256 columns it gives 6.4 for the NAND2
   (93 ps) and a NAND2 at 16x (3.2) is 25 ps faster with a 28 % steeper
   edge, at +3 % read power. A bigger inverter than `cols/4` never helps
   (`h_inv` < 3 slows the NAND2 stage). Balanced, the wordline path is
   65-70 ps at SS / 125 C / 0.9 V (~30 ps at TT) independent of the column
   count, which meets the 15 % requirement of section 2.2 at 256 columns
   (13 %; 18 % with today's rule).
2. **The read access does not depend on the wordline width at all** (+-3
   ps over 8x variation) because the replica timing waits for a full
   bitline swing; the wordline width only moves the bitline swing at
   `s_en`, which is why it becomes critical with the replica configurations
   of section 3.4.
3. **The fixed AND2 replica driver runs ahead of the real wordline** by 4-7
   ps at 4 columns, 12-22 ps at 16, 25-32 ps at 64 and 46-101 ps at 256
   columns — an early-sensing error that grows with the wordline load and
   with a weak NAND2 (the -101 ps point). Section 3.4 measures the matched
   driver.
4. **The decoder's fixed output inverter** sees `h_dec` = 1.25 x NAND2
   scale: 20 at 256 columns with the 16x NAND2 (`TCLK_DEC` +32 ps, still
   140 ps under the restore in the clock-high phase). The optional
   `dec_inv = max(1, nand / 4)` of section 4.1 keeps it at 5.
5. **Writes do not care either.** The three 16x256 write decks at
   SF / 125 C / 0.9 V with inverter 16 / 32 / 64 (NAND2 4 / 5.7 / 8) give a
   write access of 298 / 297 / 296 ps at a `TWLDRV` of 78 / 77 / 83 ps
   (slew 71 / 53 / 51 ps): the write is limited by the write driver and the
   bitline, not by the wordline width.

Rule: `inv = max(1, cols / 4)` (unchanged), `nand = max(1, cols / 15)`
(= `inv * 0.8 / 3`, replacing the square root): 1.07 at 16 columns
(half of today's 2 — the NAND2 input load on `wl_en` halves), 4.3 at 64
(unchanged), 17 at 256, 34 at 512.

### 3.4 Replica timing

Read decks with the replica hookup varied: `K` active replica cells on
RWL (of the `rows + 1` in the replica column), `N` delay-chain stages
(odd, polarity preserved), replica wordline driven by the fixed AND2
("AND2") or by the row wordline driver plus a dummy row of `cols` cells
("matched"). `s_en - WL` = sense enable rise after the wordline rise;
"BL at s_en" = target bitline level when the amplifier fires (the cell
stores 0, so the differential `dV` = BLB - BL); `TSA` = `s_en` -> `OUT`.

| array | corner | RWL | K, N | read access [ps] | s_en - WL [ps] | BL at s_en [V] | dV at s_en [V] | TSA [ps] | read PAVG [uW] |
|---|---|---|---|---|---|---|---|---|---|
| 16x16 | SS / 125 C / 0.9 V | AND2 | 1, 9 (today) | 705 | 573 | 0.004 | 0.894 | 79 | 41.5 |
| | | AND2 | 1, 5 | 507 | 375 | 0.008 | 0.890 | 79 | 39.8 |
| | | AND2 | 1, 1 | 317 | 176 | 0.061 | 0.830 | 88 | 38.1 |
| | | AND2 | 2, 9 | 682 | 550 | 0.004 | 0.892 | 79 | 41.5 |
| | | AND2 | 2, 5 | 483 | 350 | 0.009 | 0.887 | 79 | 39.9 |
| | | AND2 | 2, 1 | 298 | 152 | 0.122 | 0.770 | 93 | 38.1 |
| | | AND2 | 4, 5 | 474 | 341 | 0.009 | 0.887 | 79 | 40.0 |
| | | AND2 | 4, 1 | 292 | 143 | 0.155 | 0.735 | 96 | 38.2 |
| | | matched | 1, 9 | 725 | 592 | 0.004 | 0.895 | 79 | 42.4 |
| | | matched | 2, 1 | 311 | 168 | 0.078 | 0.814 | 89 | 38.9 |
| 16x16, slowest box cell (PU 0.108, PG 0.108, PD 0.164) | SS / 125 C / 0.9 V | AND2 | 1, 9 | 711 | 580 | 0.004 | 0.891 | 79 | 40.7 |
| | | AND2 | 2, 5 | 485 | 354 | 0.014 | 0.883 | 80 | 39.0 |
| | | AND2 | 4, 1 | 296 | 144 | 0.250 | 0.635 | 100 | 37.3 |
| | | matched | 2, 1 | 316 | 170 | 0.138 | 0.752 | 94 | 38.0 |
| 16x16 | TT / 25 C / 1.0 V | AND2 | 1, 9 (today) | 308 | 253 | 0.002 | 0.992 | 33 | 50.0 |
| | | AND2 | 1, 5 / 1, 1 | 218 / 130 | 163 / 73 | 0.005 / 0.094 | 0.988 / 0.898 | 32 / 34 | 47.8 / 45.4 |
| | | AND2 | 2, 5 / 2, 1 | 208 / 122 | 153 / 63 | 0.006 / 0.163 | 0.987 / 0.825 | 33 / 36 | 47.6 / 45.5 |
| | | AND2 | 4, 5 / 4, 1 | 204 / 119 | 149 / 59 | 0.004 / 0.156 | 0.983 / 0.826 | 33 / 38 | 47.8 / 45.5 |
| | | matched | 2, 1 | 127 | 70 | 0.093 | 0.893 | 35 | 46.3 |
| 16x16 | FF / -40 C / 1.0 V | AND2 | 1, 9 / 1, 5 / 1, 1 | 200 / 140 / 80 | 165 / 105 / 45 | 0.001 / 0.002 / 0.090 | 0.991 / 0.989 / 0.900 | 20 / 20 / 21 | 51.0 / 48.3 / 46.0 |
| | | AND2 | 2, 5 / 2, 1 | 133 / 75 | 98 / 38 | 0.001 / 0.072 | 0.987 / 0.915 | 20 / 22 | 48.4 / 45.9 |
| | | AND2 | 4, 1 | 74 | 36 | 0.249 | 0.739 | 23 | 45.9 |
| 8x4 | SS / 125 C / 0.9 V | matched | 2, 1 | 279 | 149 | 0.051 | 0.838 | 90 | 20.2 |
| 8x4 | TT / 25 C / 1.0 V | matched | 2, 1 | 113 | 62 | 0.073 | 0.918 | 34 | 24.9 |
| 64x16 | SS / 125 C / 0.9 V | matched | 2, 1 | 384 | 231 | 0.300 | 0.615 | 100 | 75.8 |
| 16x256 | SS / 125 C / 0.9 V | AND2 | 1, 9 (today) | 806 | 632 | 0.003 | 0.892 | 81 | 321 |
| | | matched | 1, 9 | 869 | 694 | 0.003 | 0.894 | 81 | 335 |
| | | AND2 | 2, 1 | 388 | 209 | 0.043 | 0.853 | 87 | 318 |
| | | matched | 2, 1 | 441 | 265 | 0.019 | 0.877 | 82 | 332 |
| 256x8 | SS / 125 C / 0.9 V | AND2 | 1, 9 (today) | 1210 | 1080 | 0.100 | 0.802 | 80 | 134 |
| | | AND2 | 2, 5 | 768 | 628 | 0.363 | 0.571 | 92 | 132 |
| | | AND2 | 4, 1 | 492 | 295 | 0.668 | 0.302 | 149 | 131 |
| | | matched | 2, 1 | 611 | 443 | 0.523 | 0.435 | 120 | 131 |
| 256x8 | FF / -40 C / 1.0 V | matched | 2, 1 | 165 | 118 | 0.578 | 0.463 | 34 | 143 |
| 512x4 | SS / 125 C / 0.9 V | AND2 | 1, 9 (today) | 1676 | 1560 | 0.211 | 0.701 | 76 | 156 |
| | | AND2 | 2, 5 | 1040 | 881 | 0.497 | 0.460 | 111 | 155 |
| | | AND2 | 4, 1 | 659 | 439 | 0.722 | 0.253 | 174 | 153 |
| | | matched | 2, 1 | 880 | 701 | 0.587 | 0.381 | 134 | 153 |

Period check for the recommended configuration, 16x16, `(2, 1)` matched,
SS / 125 C / 0.9 V: passes at 1.2 ns (`OUT` still settling at the end of
the wordline phase), fails at 1.0 ns (amplifier not fired); measured
`2 * (TCLK_WLEN + access)` = 1.15 ns against 1.97 ns with `(1, 9)`.
Ten seeded global-variation samples of the same deck (10 ns clock): read
access 297 ps mean / 338 ps worst, differential at `s_en` 0.79-0.81 V,
all samples pass; the limiting phase (`TCLK_WLEN` + access) is 7 % longer
at the worst sample than at the mean.

What the sweep shows:

1. **The delay chain is pure latency at <= 64 rows.** With one replica cell
   the replica bitline discharges at the rate of the real one (`rows + 1`
   cells of load, one active), so by the time `rbl_delay` crosses its
   threshold the target bitline is already at full swing; the eight extra
   stages add 180 ps at TT and 390 ps at the worst case with no gain in
   differential (0.89 -> 0.89 V). `N` = 1 alone brings the 16x16 read from
   308 to 130 ps at TT.
2. **`K` sets the bitline swing at which the amplifier fires, and the
   swing is nearly independent of the global corner** (K = 2, N = 1 at
   16x16: 0.77 V at SS / 125 C / 0.9 V, 0.83 V at TT, 0.92 V at
   FF / -40 C): replica and cell are the same devices and scale together.
   The margin has to be set against *local* mismatch (replica cells fast,
   target cell slow) and the amplifier offset, which the shared-model-card
   MC of the testbench cannot show; the slowest box cell costs 60-100 mV of
   differential at `(2, 1)` / `(4, 1)`.
3. **At large row counts the swing shrinks with K:** 256 rows 0.80 / 0.57 /
   0.30 V and 512 rows 0.70 / 0.46 / 0.25 V for `(1, 9)` / `(2, 5)` /
   `(4, 1)`, and the amplifier needs longer to resolve a small input
   (`TSA` 80 -> 150-175 ps). `(2, 1)` on the matched wordline leaves
   0.44 V at 256 rows (611 ps, half of today) and 0.38 V at 512 rows
   (880 ps); `(4, 1)` at 512 rows leaves 0.25 V, below the 0.3 V `dV_min`
   placeholder of section 5.
4. **The matched replica wordline removes the early-firing error** (RWL - WL
   = -12 ps -> +2 ps at 16x16, -53 -> 0 ps at 16x256) and adds the same
   time to the read access; with `(1, 9)` that is only cost (+20 / +63 ps),
   with a tuned `(K, N)` it is what makes the swing at `s_en`
   size-independent (16x16 `(2, 1)`: 0.77 V with the AND2, 0.81 V matched).
5. **Read-spec consequence.** With `(2, 1)` and the matched replica the read
   access is 113-127 ps at TT for 8-16 rows, ~170 ps at 64 rows and
   ~265 ps at 256 rows (worst case / 2.3): the 200 ps limit of
   `global.yaml` is met up to 64 rows by the replica configuration alone;
   no driver width does that.

## 4. Proposed method

### 4.1 Sizing rules

One scale per driver, `max(floor, load term)`, floors and coefficients
from section 3 (netlist-consistent, `p` = `parasitic_factor` = 1):

| driver | rule | floor | load term | changes against today |
|---|---|---|---|---|
| write driver, output stacks (M5-M12) | `wd_out = max(1.5, k_w * p * rows)`, `k_w` = 1/16 | 1.0 (box cell writes 10 / 10 at SF / 125 C / 0.9 V) x 1.5 strength margin | 100 ps TT write access (`global.yaml`); the period-driven alternative `k_w` = 1/32 halves the width at >= 48 rows for +35-40 ps write access at the worst case | floor 1.5 instead of 0.5 below 24 rows (8x4: 499 -> 148 ps worst-case write, 0 instead of 3 / 10 failing samples); unchanged from 24 rows up |
| write driver, DIN / EN inverters (M1-M4) | `wd_in = max(0.5, wd_out / 4)` | 0.5 | fan-out 4 into the stacks | new (one width pair today); -25 % `w_en` load per column, -6 % write access |
| precharge PMOS (x3) | `pre = max(0.5, p * rows / 32)`; check `TRESTORE_wc <= alpha_pre * low_read_wc` | 0.5 (0.135 um) | flat part of the restore-vs-`rows/scale` curve (knee at 32) | half of today's width at >= 16 rows: -4..-10 % read power, -1..-6 % read access, +8..+12 ps restore (+27..+52 ps at 16 rows), all inside the 0.8 budget |
| wordline inverter | `inv = max(1, cols / 4)` | 1 | `h_inv` = 3 | unchanged |
| wordline NAND2 | `nand = max(1, cols / 15)` | 1 | `h_nand` = 3 (balanced with the inverter) | square root -> linear: half at 16 columns, same at 64, 2x at 256 (-25 ps, -28 % slew), 4x at 512 |
| decoder last-stage inverter (optional) | `dec_inv = max(1, nand / 4)` | 1 | `h_dec` <= 5 | new; only matters above 64 columns (`TCLK_DEC` is in the clock-high phase) |
| replica wordline | row `WordlineDriver` + dummy row of `cols` cells | - | matches the real wordline within 2 ps (section 3.4) | replaces the fixed AND2 that runs 4-101 ps ahead |
| replica cells / delay stages | `(K, N)` from section 4.3 | - | sensing margin `dV_min` | default unchanged `(1, 9)`; `(2, 1)` recommended after the mismatch qualification |
| control buffers | fan-out 8, `n_stages = 2 * ceil(log8(F) / 2)`, loads from the resolved scales | - | - | stage count from the effort instead of 2 / 4 |

The rules are evaluated in the order write driver -> precharge -> wordline
driver -> replica -> control buffers, because the buffer loads depend on
the resolved widths. Every coefficient is stored with the hash of the
inputs it was fitted for (cell YAML values and bounds, periphery YAMLs,
PDK, `parasitic_factor`); a changed input re-runs the sweeps of section 7
item 1 (about 150 decks, 1 h on 30 cores) rather than silently reusing
the numbers.

### 4.2 Resolver

```
resolve_driver_sizes(cell, rows, cols, mux, yaml, pdk, cfg) -> DriverSizes:
    key = hash(cell yaml values + bounds, periphery yamls, cfg, pdk, rule version)
    if key in sizing_table:                      # characterised and verified (section 7)
        return sizing_table[key]
    p = cfg.parasitic_factor                     # 1.0 for the netlist-consistent flow
    # A. write driver: DC floor (cell / PDK, section 3.1) and load term (rows)
    wd_out  = max(cfg.wd_floor[cell][pdk] * cfg.wd_floor_margin,  cfg.k_w * p * rows)   # 1.0 * 1.5, 1/16
    wd_in   = max(0.5, wd_out / 4)                                 # input inverters: fan-out 4 into the stacks
    # B. precharge: flat part of the restore curve, then the budget check
    pre     = max(cfg.pre_min, p * rows / 32)                      # pre_min 0.5
    low_r   = timing_model(cell, 'read', mux, 'low')(rows, cols) * k_low_read      # TIMING_AUTOCONFIG
    assert t0_pre(cols) + k_pre * p * rows / pre <= cfg.alpha_pre * low_r        # fitted t0 / k, section 3.2
    # C. wordline driver: balanced NAND2 / inverter efforts (h = 3), decoder inverter bounded
    wl_inv  = max(1, cols / 4)                                     # h_inv = 3
    wl_nand = max(1, cols / 15)                                    # h_nand = 3 (0.45 um NAND2 input per 0.36 um unit)
    dec_inv = max(1, wl_nand / 4)                                  # h_dec <= 5 (optional, > 64 columns)
    # D. replica path
    rwl     = 'matched'                                            # row driver + dummy row of cols cells
    K, N    = cfg.replica_table.get((cell, rows_bucket(rows)), (1, 9))   # section 4.3
    # E. control buffers from the resolved loads (today hand-computed in the testbench)
    loads   = dict(pre_load=(cols + 1) * 3 * 0.27 * pre / 0.36,
                   wen_load=cols * (2 * 0.18 * wd_out + (0.18 + 0.36) * wd_in) / 0.36 + 4 * wenb_scale(cols),
                   wl_load=rows * wl_nand, num_sa=cols // mux_in)
    return DriverSizes(wd_out, wd_in, pre, wl_inv, wl_nand, dec_inv, rwl, K, N, loads, source='rule')
```

The load terms use `rows` and `cols` only; the floors and coefficients
(`wd_floor`, `k_w`, `k_pre`, `t0_pre`) are per (cell type, PDK) constants
fitted in section 3 and stored next to the rules with the hash of the
inputs they were fitted for. `timing_model` is the phase model of
`TIMING_AUTOCONFIG.md` (the two proposals share the worst-case phases).
Nothing in the loop simulates; a size outside the characterised range or a
changed YAML falls back to the rules and marks the result `source='rule'`,
and the verification of section 7 promotes it to the table.

### 4.3 Replica configuration (`K`, `N`)

Two-sided rule, evaluated on the matched replica wordline:

- early side: `dV_at_sen(rows, K, N)` at the worst case with the slowest
  box cell must exceed `dV_min` = SA offset (3 sigma, per-device MC of
  section 7 item 5) + replica / cell mismatch (3 sigma) + 100 mV; until
  those two are measured `dV_min` = 0.3 V;
- late side: `TCLK_WLEN + sen_minus_wl + TSA + latch` is the read phase
  the period is derived from (`TIMING_AUTOCONFIG.md`), so it is never
  violated, only paid for.

`N` = 1 always (the chain only adds latency, item 1 above); `K` is the
largest value whose swing at the worst case, slowest box cell, clears
`dV_min`. From section 3.4 (nominal cell unless noted):

| rows | `(K, N)` | dV at s_en, SS / 125 C / 0.9 V [V] | read access SS / TT [ps] | today `(1, 9)` SS / TT [ps] |
|---|---|---|---|---|
| <= 16 | 2, 1 | 0.81 (0.75 slowest box cell) | 279-311 / 113-127 | 705 / 308 |
| 64 | 2, 1 | 0.62 | 384 / ~170 | 822 / 358 |
| 256 | 2, 1 | 0.44 (0.46 at FF / -40 C) | 611 / ~265 | 1210 / 524 |
| 512 | 2, 1 (or 1, 1, not run) | 0.38 | 880 / ~380 | 1676 / 739 |

Recommendation: keep `(1, 9)` as the shipped default until the per-device
mismatch qualification has set `dV_min`; then switch the default to
`(2, 1)` up to 256 rows (0.44 V or more of differential at the worst case,
read access halved) and characterise `(1, 1)` for 512 rows, where `(2, 1)`
leaves 0.38 V before the slowest cell is taken into account (decision 5 of
section 9). The resolver exposes `replica: {K, N}` in the `sizing` block
and the table above is the seed of `sizing_rules.json`.

### 4.4 Control buffers

Keep the V2.0.2 fan-out rule (`TaperedBuffer`, effort ~8, 2 stages up to
16x, 4 above) and compute the loads in the resolver from the resolved
scales, not from the rules (today `create_time_circuit` re-derives
`pre_load` and `wen_load` from `PrechargeFactory.width_scale` and
`WriteDriverFactory.width_scale`; with a split write driver or a table entry
that goes stale). One change: choose the stage count from the effort
(`n = 2 * ceil(log8(F) / 2)`, even for polarity) instead of the fixed
2 / 4 split at 16x: identical up to an effort of 4096 (2 stages up to 64,
4 above), 6 stages beyond, so the rule stays valid for the `PRE` / `w_en`
buffers of the largest arrays instead of silently exceeding a fan-out of 8.

## 5. Margin policy

| source of uncertainty | how it is covered | size |
|---|---|---|
| global process / voltage / temperature | every speed term sized and checked at SS / 125 C / 0.9 V, write-ability at SF / 125 C / 0.9 V, leakage at FF and FS / 125 C (the corners `TIMING_AUTOCONFIG.md` section 3.4 measured as worst per phase) | 2.1-2.3x on every control phase, up to 3.8x on the write access, relative to TT |
| global variation beyond the corner | the MC testbench's `AGAUSS(5 %)` on `vth0 / u0 / voff` per model card is a random global shift on top of the corner; verification decks run 10 seeded samples and require 0 failures *and* the analog margins below (a marginal pass is not a pass) | worst of 10 samples over the mean: write access +16 % (8x4, 1.0x), +20 % (box cell, 1.5x), +48 % (box cell at the 1.0x floor); read phase +7 % (section 3.4) |
| DC write-ability | floor = smallest scale that writes the parameter-box cell (max PU, min PG, max PD) at SF / 125 C / 0.9 V with 10 / 10 samples, times `wd_floor_margin` = 1.5 (a driver whose NMOS is 33 % weaker than the model still writes); acceptance: driven bitline < 0.1 VDD, `q_written`, `q_retained` | section 3.1 |
| sensing margin | `dv_at_sen >= dV_min` for the slowest cell of the parameter box against the nominal replica, `dV_min` = SA offset (3 sigma, per-device flow) + replica / cell mismatch (3 sigma) + 100 mV; placeholder 0.3 V (100 + 100 + 100 mV) until measured; `K >= 2` halves the replica's own sigma | section 3.4: 0.75-0.84 V at <= 16 rows, 0.62 V at 64 rows with `(2, 1)` |
| phase budgets | `alpha` fractions of the worst-case read phase (0.8 restore / write, 0.15 wordline path) so the period stays array-limited; on top of that the 25 % margin of the period | section 2.2: worst measured 0.54 (restore, 16x16 at 0.25x), 0.13 (wordline, 16x256 with `cols/15`) |
| parasitics not in the netlist | `parasitic_factor` on the load terms (default 1.0 = netlist-consistent; 2-3 for the bitline of a laid-out array per section 1.3); a physical design re-characterises with `w_rc` / extraction | section 1.3 |
| local mismatch (cell vs replica, SA offset) | not covered by the MC testbench (shared model cards); the per-device flow (`per_device_mc/netlist.py`) is the tool; it is uncalibrated (relative perturbation, no area dependence), so `dV_min` keeps the 0.3 V placeholder until it is | open |

The margins are applied to the sizes at the worst case; a run at TT / 25 C
then shows the corresponding slack, which is physically right for a design
that must work at SS / 125 C / 0.9 V, and the yield flow that is meant to
see driver-related failures has to run at the same worst-case PVT (both
testbenches accept `corner`, `temperature` and `vdd`).

## 6. Integration

- New module `sram_compiler/sizing/driver_sizing.py`: `DriverSizes`
  (immutable), `resolve_driver_sizes()`, the coefficient / floor set
  (`sizing_rules.json`, with the YAML hash they were fitted for), the
  verified table (`sizing_table.json`) and `verify()` (section 7).
- `global.yaml`: a `sizing` block — `mode: auto | rules_only | fixed`,
  `alpha_pre / alpha_w / alpha_wl`, `wd_floor_margin`, `parasitic_factor`,
  `replica: {K, N}` (default `1, 9`), `table` path.
- `parameter_factor.py`: `PrechargeFactory`, `WriteDriverFactory`,
  `WordlineDriverFactory` take a `scale` (and the write driver an
  `out_scale`) from `DriverSizes` instead of computing it from `num_rows /
  num_cols`; the static rules stay as the `rules_only` fallback.
  `WriteDriver` gets separate input / output widths (12 transistors, two
  width pairs); `WordlineDriver` unchanged; `DecoderCascade` gets an
  optional last-stage inverter scale.
- `sram_6t_core_testbench.py`: `create_time_circuit` takes the buffer loads
  from `DriverSizes.loads`; `create_and2_for_rwl` becomes
  `create_replica_wordline` (row driver + dummy row); `create_replica_column`
  connects `K` cells to RWL; `TIME` takes `dc_stages`.
- The sweep paths (`sweep_precharge`, `sweep_writedriver`,
  `sweep_wordlinedriver`) emit the resolved scale as a SPICE expression on
  the swept parameter (as the wordline driver already does), so a swept base
  width keeps the array-dependent scale instead of bypassing it (the
  precharge and write-driver sweeps bypass it today).
- `size_optimization/exp_utils.py` and `main_estimation.py`: resolve the
  sizes once, before the loop, from the baseline cell; record `DriverSizes`
  with the experiment; the candidate cell does not re-size the periphery.
- `utils.estimate_scaled_array_area`: use the resolved precharge and
  wordline-driver widths (`prc_max_width`, `wld_max_width`) instead of the
  YAML base widths, so area follows the sizing.

## 7. Validation plan

1. **Floors and coefficients (done for this proposal, section 3):** write
   driver scale sweep at SF / 125 C / 0.9 V on 8x4 (default and box-corner
   cell, 10 samples), 16x16, 64x16, 256x8, 10T 8x4; precharge scale sweep at
   SS / 125 C / 0.9 V on 8x4, 16x16, 64x16, 256x8, read and write, plus the
   configured periods (2.5 / 3.8 ns); wordline-driver scale sweep on 8x4,
   16x16, 64x64, 16x256; replica wordline matched vs AND2 on 16x16,
   16x256, 256x8; `(K, N)` sweep on 16x16 at SS / 125 C / 0.9 V, TT and
   FF / -40 C, on 256x8 and 512x4 at SS; `(2, 1)` matched on 8x4 .. 512x4
   with a period sweep and 10 samples at 16x16.
2. **Rules against the sweeps:** for every characterised size the resolver's
   scales must give measured metrics inside the acceptance of section 2.2
   with the stated margins; where the rule is above the smallest passing
   scale, the ratio is the realised margin and is printed.
3. **Verification decks per array** (`verify()`): read at SS / 125 C /
   0.9 V, write at SS and SF / 125 C / 0.9 V, automatic period, 10 seeded
   samples, `next_row` hazard variant at >= 256 rows; all waveform checks
   and acceptance metrics pass; results stored with the hash. Run for the
   27 sizes of the V2.0.2 sweep, 6T and 10T, mux on and off.
4. **Parasitic sensitivity:** repeat item 3 on 16x16, 64x64 and 256x8 with
   `w_rc = True` (`pi_res` 100 Ohm, `pi_cap` 1 fF) and `parasitic_factor` 2;
   the drivers sized for factor 2 must pass with the RC deck.
5. **Local mismatch:** per-device MC on the target cell, the replica cell
   and the sense amplifier of a 16x16 and a 256x8 array (100 samples each)
   to measure the SA offset and the replica / cell skew that set `dV_min`;
   `(K, N)` accepted only if `dv_at_sen` at the 3-sigma skew exceeds `dV_min`.
6. **Optimiser smoke test:** `demo_*` on 32x1 with `sizing: auto`; a cell at
   the box corner (max PU / min PG) must still be written by the frozen
   periphery at SF / 125 C / 0.9 V; a cell outside the box must fail with
   `q_written = False`, not with a resized driver.

## 8. Effort

| step | size |
|---|---|
| `driver_sizing.py` (resolver, rules, table, verify wrapper) | ~300 lines, 1 day |
| write driver split, decoder inverter scale, matched replica wordline, `K` / `N` plumbing | ~150 lines, 1 day |
| factory / testbench / YAML / optimiser hooks, sweep-path expressions | ~100 lines, 0.5 day |
| validation runs (section 7, items 2-4) | ~1 day of machine time at 30 parallel decks |
| per-device mismatch qualification (item 5) | 1 day incl. harness |

## 9. Decisions needed

1. `alpha` defaults (0.8 restore / write, 0.15 wordline path),
   `wd_floor_margin` 1.5 and the write-access coefficient `k_w` (1/16 =
   100 ps at TT, or 1/32 = period-driven): project values or the proposal's.
2. Worst-case envelope: 125 C / 0.9 V with SS for speed and SF for
   write-ability (measured); if the product envelope differs, the floors
   and factors are re-read from the same sweeps at the other corners.
3. Whether the parameter box of the optimiser (cell YAML `upper / lower`)
   defines the write-ability floor (proposed) or only the default cell;
   the box raises the floor from 0.75 to 1.0 (a third more width below 24 rows).
4. `parasitic_factor` default for the flow: 1.0 (netlist-consistent,
   proposed) or a fixed derating (2.0) until extraction data exists.
5. Replica default `(K, N)`: keep `(1, 9)` (maximum margin, ~300 ps read at
   TT) or adopt `(2, 5)` (200 ps at TT, margin per section 3.4) after the
   mismatch qualification of section 7 item 5.

## 10. Relation to the previous draft of this file

Kept: the two-sided sensing constraint with local mismatch as its reason;
one physical width vector per array (never per PVT); the acceptance list
(restoration, equalisation, isolation ordering, write overlap and
retention, disturb, back-to-back patterns); the canonical-macro remark (the
read deck still omits the write drivers, section 6 keeps that as a
testbench item); the yield-driven failure budget and the caveats on the MC
testbench (shared model cards) and on FreePDK45 being predictive.

Replaced: the generic per-stage delay model with fitted `k_j / W_j` terms
and the constrained-Bayesian search over widths — the compiler's drivers
have one scale each and their loads are counts of rows and columns, so a
floor plus a proportional term per driver, calibrated by one sweep per
driver at its worst corner, is the whole model; the 20 % reserve against
the 200 / 100 ps `global.yaml` limits — the read limit is set by the replica
configuration, not by driver widths (section 2.1), so the drivers are
budgeted against the worst-case phases instead.
