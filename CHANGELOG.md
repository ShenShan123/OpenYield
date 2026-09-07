# Changelog

## V2.0.2 — 2026-09-05 — periphery fan-out, address hold, precharge control, timing configurations

Scope of this release: the open items of V2.0.1 (address-path hold hazard,
cycle-time dependence of the floating bitlines, `s_en` buffer sizing) plus a
review of every control buffer of the timing block and of the wordline driver
against its actual fan-out, over array sizes from 1x1 to 512x4 / 16x512, clock
periods from 0.6 ns to 100 ns, the five process corners with -40 C to 125 C,
and seeded Monte Carlo. Optimisation and yield-estimation algorithms were not
touched.

Every change was validated by running the generated netlists in Xyce 7.4 and
scoring the waveforms automatically (the V2.0.1 checks plus address-change
hold checks and the bitline levels at the start of every access). "Before"
numbers were measured on the V2.0.1 code (a detached `git worktree`), "after"
numbers on this release; the evidence tables are at the end of this entry.

### Summary of fixes

- **Address-path hold hazard confirmed and closed.** At 512 rows a changed
  address raised the *new* row's wordline to full VDD while the old one was
  still falling: the cell in the new row was overwritten by the bitline
  (read deck, stored 1 -> 0) or by the still-active write drivers (write
  deck, 0 -> 1). At 256 rows the second wordline reached 0.41-0.50 V and
  disturbed the neighbouring cell (Q dipped to 0.86 V). A transparent-low
  hold latch on the address register output (enabled by `wl_en_bar`, same
  scheme as the V2.0.1 write-data latch) keeps the decoder input constant
  while a wordline is on; the new decoder output now rises at least five
  gate delays after the old wordline is off, and no second wordline
  (< 10 mV) is seen at any size.
- **Control buffers sized for their fan-out.** `wl_en` (fixed output stage,
  600 ps fall at 512 rows), the address register output (650 ps edge at 512
  rows), `s_en` (190-230 ps edge and a 0.2-0.3 V precharge-coupling bump at
  >= 64 columns), `w_en` (130-180 ps edge from 64x16 up; the write waited for
  it) and `PRE` (fan-out ~49, only reached 0.05-0.08 V on the largest arrays)
  are now driven by geometrically tapered buffers with a fan-out of ~8 per
  stage; the wordline driver's NAND2 is scaled with the square root of its
  inverter scale instead of staying at the base width (fan-out ~100 and a
  349 ps stage delay at 512 columns).
- **Sense amplifier isolated during writes.** Its input pass gates were on
  whenever `s_en` was low, so the cross-coupled PMOS pair sat on the bitlines
  as a keeper during every write; the write succeeded only because `w_en`
  happened to rise before the wordline (60 ps margin at 2x128 in V2.0.1).
  With the faster wordline path of this release the order flipped on 2x128
  and the write deadlocked (BL 0.27 V, BLB 0.9 V, cell unchanged). The
  amplifier now has a separate `ISO` pin driven by `s_en | w_en`, so it is
  disconnected while the write drivers are on; reads are unchanged.
- **Precharge for the whole clock-high phase.** The ~300 ps self-timed pulse
  left the bitlines floating: with the default 10 ns clock the replica
  bitline had drooped to 0.74 V and the target bitline to 0.88 V by the next
  access at FF / 125 C (0.87 / 0.95 V at TT / 125 C), and to 0.76 V at TT /
  25 C with a 100 ns clock. `PRE = NAND3(clk_buf, cs, wl_en_bar)` now holds
  the bitlines at VDD until 40-70 ps before the wordline rises; the levels
  are 1.000 V at every access for 0.6-100 ns periods and at every corner.
- **Honest minimum-period estimate.** The printed `CLK(min)` was
  `2 * (access delay + 0.1 ns)` and claimed 0.48-0.80 ns for an 8x4 array
  whose read deck fails below 0.9 ns. Three new measures (clock -> `wl_en`,
  clock -> decoder output, clock -> bitline restored) give
  `T_min = 2 * max(clock-low work, clock-high work) * 1.1`, which matches the
  period sweeps within one step (see Evidence).
- **Testbench stimulus for the address path.** `next_row=<row>` makes the
  register capture another row at the clock edge that ends the access and
  prints that row's wordline and cell; the address was constant in every
  V2.0.1 deck, which is why the hazard was never exercised.
- **PSTC caveat.** A warning is printed for `t_period < 5 ns`, where the
  quiescent window overlaps the start-up precharge.

### Not completed / left open

- **128x128 and 256x64 arrays** were run only with a 4 ns clock (3 h per
  deck instead of ~11 h): 6T read 475 / 568 ps, 256x64 write 149 ps, all
  waveform checks pass; every 10T / mux / read&write deck of these two sizes
  was not run. In the size table only the two 10T 16x512 `read&write` decks
  are missing: they exceeded the 10 h job limit of the sweep harness (the 6T
  16x512 `read&write` decks took 7 h); the same flow runs them offline.
- **Energy cost of the fixes.** Reads cost more: 6T read PAVG at 100 MHz
  28.0 -> 29.3 uW (8x4, +5 %), 44.9 -> 50.0 (16x16, +11 %), 81.1 -> 94.6
  (32x32, +17 %), 202.9 -> 231.7 (64x64, +14 %); 2x128 is cheaper (158 ->
  144 uW). The largest single term is the wordline-driver NAND2 taper: the
  `wl_en` / decoder nets now drive `rows * sqrt(cols / 4)` base-size NAND2
  gates (115 fF instead of 29 fF at 64x64). The sharper `s_en`, the
  isolation buffer, the address latches and holding the bitlines (PSTC
  +20-30 %, the leakage is supplied through the precharge devices instead of
  being drawn from the floating bitline capacitance) make up the rest.
  Writes are cheaper on every array >= 32x32 (32x32 227.8 -> 201.4 uW,
  64x64 610.8 -> 504.8, 2x128 740.1 -> 560.3) because the write driver no
  longer fights the sense-amplifier keeper. Not tuned further: the buffers
  follow one fan-out rule (`nand_scale`, `TaperedBuffer` scale) that a
  power-optimised design would relax where the timing margin allows.
- **Delay reference at >= 256 rows.** `TREAD_TOTAL` / `TWRITE_TOTAL` are
  measured from the `wl_en` crossing. The re-sized `wl_en` buffer moves that
  crossing 70-150 ps earlier on 256-512-row arrays (clock -> `wl_en` 300 ->
  150 ps at 512x4), so the tabulated delays of those arrays grow (512x4 6T
  read 709 -> 739 ps, write 70 -> 95 ps) while the time from the clock edge
  to the output or to the written cell shrinks (512x4 6T read 1010 -> 890 ps,
  write 300 -> 180 ps; 256x8 read 740 -> 670 ps, write 220 -> 180 ps). The
  measures are kept as defined in V2.0.1; `TCLK_WLEN` is written next to
  them so the clock-referenced value is `TCLK_WLEN + TREAD_TOTAL`.
- **Design choices verified again and left as they are:** replica-timed,
  full-swing sensing (read delay ~300 ps up to 32 rows); `w_rc=True` default
  of `main_sram.py`; `read` reads a stored 0; all columns are written in a
  write (no half-selected column with the mux); the `s_en` / `w_en`
  precharge-coupling and hazard checks are only in the scratch harness, not
  in the flow's `.MEASURE` set.
- **Out of scope:** optimisation and yield-estimation algorithms, SNM beyond
  the V2.0.1 sanity run, `sweep_*` modes (the wordline-driver sweep mode
  carries the new NAND2 scale as a SPICE expression but was not simulated).

### Circuit topology

- **Address hold latch and buffer (`time_generate.py`).** `ADDR_DFF` now
  drives internal nodes `A_reg{i}`; per bit a `D_LATCH_ADDR` (EN =
  `wl_en_bar`) and a `TaperedBuffer` (`ABUF`, one 3-unit output per 8 gate
  inputs of the decoder: 5 inputs per last-level 3-to-8 decoder) produce the
  exported `A_dff{i}`. Timeline at the edge that ends an access (8x4): old
  wordline off at +190 ps, new decoder output at +300 ps (was +190 ps, zero
  margin). The `.IC` of the control path now also pins `XTIME:A_reg{i}`.
- **`wl_en` buffer (`wl_pdrive`)** scales both stages with
  `ceil(rows * nand_scale / 32)` (unchanged up to 16x16, 4x at 64x16, 16x at
  512x4). Edge at 512 rows: 280 / 600 ps -> see Evidence.
- **Wordline driver (`WordlineDriverFactory`)**: `nand_scale = sqrt(inv_scale)`
  with `inv_scale = max(cols, 4) / 4` as before; the sweep mode emits the same
  factor as a SPICE expression.
- **`s_en` / `sa_iso`**: `s_en` now drives only the amplifier footers and
  the output latch (4-unit inverter direct up to 32 unit loads, then a
  `TaperedBuffer` `SEN_BUF`); the input pass gates are driven by the new
  `sa_iso = NOR2(s_en, w_en)` + inverter (+ `TaperedBuffer` `ISO_BUF`, 4 unit
  loads per amplifier), exported by TIME and connected to the new `ISO` pin of
  `SENSEAMP` (`mux_and_sa.py`). `TS_EN` (20-80 %) is ~20 ps from 1 to 512
  columns (was 43 ps at 16, 138 ps at 64, 165 ps at 512 columns).
- **`w_en`**: `AND2_WEN` with a 4-unit inverter drives up to 32 unit loads
  directly, otherwise a `TaperedBuffer` (`WEN_BUF`) sized from the write
  driver's EN load (3 * NMOS + PMOS width, row-scaled) plus the testbench's
  `w_en_bar` inverter, which is itself scaled for the two latch-enable inputs
  per column (it would have had a fan-out of ~1300 at 512 columns). 64x16:
  w_en edge 140 -> 30 ps, driven bitline at VDD/2 120 -> 90 ps after
  `gated_clk_bar`, `TWRITE_TOTAL` 114 -> 94 ps.
- **Precharge control**: `PRE_UNBUF = NAND3(clk_buf, cs, wl_en_bar)` (was
  `NAND3(gated_clk_buf, rbl_delay, wl_en_bar)`), buffered by a
  `TaperedBuffer` (`PRE_BUF`) sized from `(cols + 1) * 3 * PMOS width`
  (row-scaled by `PrechargeFactory.width_scale`). PRE is off (90 %) 40 ps
  before the wordline rises at TT (70 ps at FF / 125 C), and turns on
  30-50 ps after the write driver has released the bitlines at the end of an
  access. `rbl_delay` is still the sense-enable timing and `rbl_delay_bar`
  stays exported. The bitline overshoot of 1.03-1.07 V after the old pulse is
  gone (1.000 V).
- `TIME` exports one more node, `sa_iso`; `TIME` / `TIMEFactory` take `num_sa`, `wl_load`, `pre_load`, `wen_load`
  (fan-out information computed by the testbench from the periphery
  factories); defaults reproduce the base YAML sizes.
- New helpers: `TaperedBuffer` (2 stages up to a scale of 16, 4 above;
  `name` must be unique per scope), `D_latch_addr` and `PNOR2`; static
  `WordlineDriverFactory.inv_scale / nand_scale`,
  `PrechargeFactory.width_scale`, `WriteDriverFactory.width_scale`.

### Measurements

- `TCLK_WLEN` (falling clock edge -> `wl_en` at VDD/2), `TCLK_DEC` (first
  capture edge -> target decoder output, `target_row != 0`) and `TRESTORE`
  (rising edge that ends the access -> the discharged bitline back at
  0.9 VDD; `TD=` on both events because the start-up precharge also crosses
  0.9 VDD) are added to the read and write decks.
- `_print_min_period()` replaces the `1/2CLK` print: clock-low work =
  `TCLK_WLEN + TREAD_TOTAL | TWRITE_TOTAL`, clock-high work =
  `max(TRESTORE, TCLK_DEC)`, `T_min = 2 * max(...) * 1.1`; mean + one
  standard deviation per term for `mc_runs > 1`.
- `_add_static_power_measures()` warns for `t_period < 5 ns`.

### Testbench

- `Sram6TCoreTestbench(next_row=...)` / `Sram6TCoreMcTestbench(next_row=...)`
  (read / write only; `read&write` raises). With it the address bits that
  differ between `target_row` and `next_row` alternate with a period of
  `2 * t_period`, so the register captures `target_row` at the edge that
  starts the access and `next_row` at the edge that ends it; `V(WL{next})`,
  `V(DEC_WL{next})` and the cell `(next_row, target_col)` are printed.
- `create_time_circuit()` passes the fan-out information to `TIMEFactory`;
  `create_write_periphery()` sizes the `w_en_bar` inverter with the column
  count (`_wenb_scale()`).

### Documentation

- `readme_compiler.md`: precharge description, `next_row`, the new switches,
  the minimum-period estimate and the PSTC caveat (sections 10, 11, 13.1,
  14.5). `CIRCUIT_REVIEW.md` Part III: method, defects D12-D18, sizing
  tables.

### Observations (verified, not changed)

- At TT / 125 C every control-path delay roughly doubles (16x16 6T read
  334 -> 573 ps, `TRESTORE` 256 -> 460 ps); the 10 ns clock still has > 8 ns
  of margin, the estimated minimum period is ~1.7 ns.
- SS / -40 C is the fastest condition in these models (8x4 6T read 226 ps
  vs 302 ps at TT / 25 C); SF is the slowest write corner (8x4 6T write
  215 ps vs 138 ps).
- The `read&write` sequence toggles `we` every cycle; `s_en` shows no glitch
  at the write-to-read transitions (< 1 mV in the first 180 ps after the
  edge) because `gated_clk_bar` falls before `we_bar` rises.
- The address change with the LSB flipped never produced a second wordline
  even on the V2.0.1 circuit: the LSB has the largest decoder fan-out and its
  path is slower than the `wl_en` fall. The middle address bit (fewest
  gates) is the critical one.
- Sizing rule used throughout: one unit (0.09 / 0.27 um) inverter per 8 unit
  loads, tapered geometrically; every buffered control edge is now
  20-40 ps (10-90 %) independent of the array size.
- With the isolation pin the sense amplifier's pass gates open ~50-70 ps
  after the footer fires (NOR + inverter + buffer), so the amplifier
  regenerates while still connected to the full-swing bitlines and the
  output latch flips before the isolation completes. Reads are 8-25 ps
  faster than in V2.0.1 (8x4 6T 301 -> 291 ps, 16x16 6T 337 -> 308 ps,
  2x128 6T 458 -> 322 ps together with the wordline-driver taper).

### Evidence

Filled in below from the final sweeps (nominal unless stated, all cells real,
`w_rc=False`, 25 C, TT, target cell = last row / last column). Values in
parentheses are the V2.0.1 results of the same configuration.

#### Size sweep (nominal, all cells real, no RC, mux off / on)

292 of 294 configurations completed in Xyce at release time, 292 pass every waveform check; failures: none.

Not completed (2): 10T_16x512_m0_rw, 10T_16x512_m1_rw -- stopped by the 10 h job limit of the sweep harness (the 6T decks of the same size took 7 h); the same flow runs them offline.

Delays in ps (read: `wl_en` rise -> `OUT`; write: `wl_en` rise -> Q at 90 %), V2.0.1 value in parentheses; `read&write` shows the measured period of `OUT` (40 ns = 4 clock cycles is the correct write-1 / read / write-0 / read sequence); PAVG at 100 MHz.

| array | cell | read delay [ps] mux off / on (V2.0.1) | write delay [ps] mux off / on (V2.0.1) | read&write OUT period mux off / on | PAVG read / write [uW] (V2.0.1) | waveform checks |
|---|---|---|---|---|---|---|
| 1x1 | 10T | 286 (289) / n/a | 119 (133) / n/a | 40 ns / n/a | 21.0 (20.1) / 22.5 (21.8) | PASS (3/3 runs) |
| 1x1 | 6T | 283 (286) / n/a | 119 (131) / n/a | 40 ns / n/a | 20.9 (20.1) / 22.6 (22.0) | PASS (3/3 runs) |
| 2x1 | 10T | 287 (290) / n/a | 120 (134) / n/a | 40 ns / n/a | 22.2 (21.5) / 23.6 (23.1) | PASS (3/3 runs) |
| 2x1 | 6T | 284 (287) / n/a | 121 (132) / n/a | 40 ns / n/a | 22.2 (21.4) / 23.8 (23.3) | PASS (3/3 runs) |
| 2x2 | 10T | 287 (293) / 287 (290) | 121 (134) / 138 (164) | 40 ns / 40 ns | 23.0 (22.2) / 26.9 (26.4) | PASS (6/6 runs) |
| 2x2 | 6T | 284 (290) / 284 (287) | 121 (132) / 140 (174) | 40 ns / 40 ns | 22.9 (22.2) / 27.3 (26.8) | PASS (6/6 runs) |
| 4x2 | 10T | 289 (295) / 289 (292) | 123 (136) / 139 (166) | 40 ns / 40 ns | 25.2 (24.1) / 29.2 (28.3) | PASS (6/6 runs) |
| 4x2 | 6T | 286 (292) / 285 (289) | 124 (134) / 142 (176) | 40 ns / 40 ns | 25.1 (24.0) / 29.6 (28.8) | PASS (6/6 runs) |
| 3x3 | 10T | 289 (297) / n/a | 124 (136) / n/a | 40 ns / n/a | 24.8 (23.7) / 31.5 (30.5) | PASS (3/3 runs) |
| 3x3 | 6T | 286 (294) / n/a | 123 (132) / n/a | 40 ns / n/a | 24.7 (23.6) / 31.8 (31.0) | PASS (3/3 runs) |
| 5x3 | 10T | 291 (299) / n/a | 125 (138) / n/a | 40 ns / n/a | 26.0 (24.5) / 32.6 (31.3) | PASS (3/3 runs) |
| 5x3 | 6T | 288 (296) / n/a | 125 (134) / n/a | 40 ns / n/a | 25.9 (24.4) / 33.1 (31.9) | PASS (3/3 runs) |
| 4x4 | 10T | 291 (301) / 289 (295) | 127 (138) / 143 (168) | 40 ns / 40 ns | 26.8 (25.7) / 36.1 (35.1) | PASS (6/6 runs) |
| 4x4 | 6T | 287 (298) / 286 (292) | 126 (133) / 145 (176) | 40 ns / 40 ns | 26.8 (25.7) / 36.8 (36.0) | PASS (6/6 runs) |
| 16x1 | 10T | 305 (308) / n/a | 83 (86) / n/a | 40 ns / n/a | 30.1 (28.3) / 31.5 (29.9) | PASS (3/3 runs) |
| 16x1 | 6T | 299 (302) / n/a | 70 (73) / n/a | 40 ns / n/a | 30.0 (28.3) / 31.5 (29.8) | PASS (3/3 runs) |
| 8x4 | 10T | 295 (305) / 294 (300) | 131 (142) / 146 (172) | 40 ns / 40 ns | 29.5 (28.0) / 39.0 (37.7) | PASS (6/6 runs) |
| 8x4 | 6T | 291 (301) / 290 (295) | 131 (138) / 149 (181) | 40 ns / 40 ns | 29.3 (28.0) / 39.6 (38.4) | PASS (6/6 runs) |
| 6x6 | 10T | 294 (308) / 292 (301) | 131 (142) / 146 (171) | 40 ns / 40 ns | 29.9 (28.4) / 44.7 (43.2) | PASS (6/6 runs) |
| 6x6 | 6T | 290 (305) / 288 (297) | 130 (137) / 148 (180) | 40 ns / 40 ns | 29.8 (28.3) / 45.7 (44.5) | PASS (6/6 runs) |
| 12x4 | 10T | 302 (311) / 301 (305) | 102 (106) / 110 (119) | 40 ns / 40 ns | 31.4 (29.8) / 40.8 (38.7) | PASS (6/6 runs) |
| 12x4 | 6T | 296 (306) / 295 (300) | 90 (93) / 100 (110) | 40 ns / 40 ns | 31.5 (29.5) / 40.9 (38.8) | PASS (6/6 runs) |
| 8x8 | 10T | 298 (315) / 296 (306) | 136 (147) / 150 (174) | 40 ns / 40 ns | 33.6 (31.6) / 54.0 (52.2) | PASS (6/6 runs) |
| 8x8 | 6T | 293 (311) / 291 (301) | 135 (141) / 152 (183) | 40 ns / 40 ns | 33.5 (31.4) / 55.3 (53.7) | PASS (6/6 runs) |
| 16x8 | 10T | 310 (327) / 307 (316) | 95 (98) / 100 (105) | 40 ns / 40 ns | 38.7 (36.2) / 59.0 (56.0) | PASS (6/6 runs) |
| 16x8 | 6T | 303 (320) / 301 (310) | 83 (85) / 88 (93) | 40 ns / 40 ns | 38.5 (35.8) / 58.9 (55.7) | PASS (6/6 runs) |
| 20x10 | 10T | 318 (336) / 315 (324) | 99 (95) / 100 (99) | 40 ns / 40 ns | 44.8 (40.1) / 71.6 (66.0) | PASS (6/6 runs) |
| 20x10 | 6T | 309 (329) / 306 (317) | 86 (83) / 88 (87) | 40 ns / 40 ns | 44.3 (39.8) / 71.5 (65.4) | PASS (6/6 runs) |
| 2x128 | 10T | 326 (463) / 320 (386) | 169 (202) / 184 (224) | 40 ns / 40 ns | 144.3 (158.8) / 528.8 (734.0) | PASS (6/6 runs) |
| 2x128 | 6T | 322 (458) / 316 (382) | 176 (191) / 192 (225) | 40 ns / 40 ns | 143.6 (158.4) / 560.3 (740.1) | PASS (6/6 runs) |
| 16x16 | 10T | 314 (344) / 310 (327) | 107 (111) / 111 (116) | 40 ns / 40 ns | 50.5 (45.3) / 94.3 (88.8) | PASS (6/6 runs) |
| 16x16 | 6T | 308 (337) / 304 (320) | 94 (98) / 99 (105) | 40 ns / 40 ns | 50.0 (44.9) / 93.7 (89.1) | PASS (6/6 runs) |
| 32x8 | 10T | 331 (347) / 329 (337) | 94 (84) / 96 (86) | 40 ns / 40 ns | 48.6 (43.9) / 72.5 (65.7) | PASS (6/6 runs) |
| 32x8 | 6T | 321 (336) / 318 (327) | 82 (72) / 84 (73) | 40 ns / 40 ns | 48.0 (43.5) / 71.7 (65.6) | PASS (6/6 runs) |
| 32x32 | 10T | 344 (396) / 337 (364) | 112 (148) / 114 (149) | 40 ns / 40 ns | 95.4 (81.9) / 202.0 (227.4) | PASS (6/6 runs) |
| 32x32 | 6T | 332 (386) / 325 (355) | 100 (137) / 102 (138) | 40 ns / 40 ns | 94.6 (81.1) / 201.4 (227.8) | PASS (6/6 runs) |
| 64x16 | 10T | 378 (401) / 373 (383) | 105 (125) / 106 (125) | 40 ns / 40 ns | 90.7 (76.3) / 156.8 (166.6) | PASS (6/6 runs) |
| 64x16 | 6T | 358 (382) / 353 (366) | 93 (114) / 94 (114) | 40 ns / 40 ns | 89.9 (75.1) / 155.0 (165.8) | PASS (6/6 runs) |
| 256x8 | 10T | 592 (579) / 591 (566) | 105 (78) / 107 (79) | 40 ns / 40 ns | 152.3 (124.9) / 225.1 (229.2) | PASS (6/6 runs) |
| 256x8 | 6T | 524 (526) / 521 (514) | 96 (68) / 96 (68) | 40 ns / 40 ns | 149.4 (122.2) / 223.3 (226.5) | PASS (6/6 runs) |
| 512x4 | 10T | 869 (791) / 861 (783) | 105 (80) / 105 (77) | 40 ns / 40 ns | 175.0 (153.3) / 238.9 (248.2) | PASS (6/6 runs) |
| 512x4 | 6T | 739 (709) / 734 (692) | 95 (70) / 94 (69) | 40 ns / 40 ns | 171.3 (148.3) / 234.9 (245.5) | PASS (6/6 runs) |
| 8x512 | 10T | 355 (642) / 349 (557) | 197 (512) / 206 (509) | 40 ns / 40 ns | 597.4 (1034.5) / 3144.5 (7386.7) | PASS (6/6 runs) |
| 8x512 | 6T | 350 (640) / 345 (550) | 202 (509) / 214 (509) | 40 ns / 40 ns | 590.5 (1078.5) / 3277.7 (7527.4) | PASS (6/6 runs) |
| 16x256 | 10T | 361 (542) / 347 (457) | 146 (273) / 150 (275) | 40 ns / 40 ns | 372.7 (444.6) / 1250.9 (2340.0) | PASS (6/6 runs) |
| 16x256 | 6T | 354 (535) / 339 (449) | 133 (261) / 139 (264) | 40 ns / 40 ns | 369.4 (441.7) / 1252.0 (2339.6) | PASS (6/6 runs) |
| 64x64 | 10T | 408 (505) / 390 (440) | 120 (173) / 120 (173) | 40 ns / 40 ns | 237.4 (210.2) / 513.9 (613.0) | PASS (6/6 runs) |
| 64x64 | 6T | 382 (492) / 366 (421) | 108 (162) / 108 (160) | 40 ns / 40 ns | 231.7 (202.9) / 504.8 (610.8) | PASS (6/6 runs) |
| 128x32 | 10T | 460 (510) / 453 (467) | 116 (146) / 116 (145) | 40 ns / 40 ns | 224.5 (181.1) / 418.8 (462.8) | PASS (6/6 runs) |
| 128x32 | 6T | 424 (468) / 419 (434) | 104 (135) / 105 (133) | 40 ns / 40 ns | 218.6 (174.8) / 416.7 (459.4) | PASS (6/6 runs) |
| 100x50 | 10T | 452 (517) / 423 (464) | 120 (168) / 121 (168) | 40 ns / 40 ns | 266.7 (212.7) / 529.3 (607.9) | PASS (6/6 runs) |
| 100x50 | 6T | 414 (488) / 396 (438) | 109 (157) / 110 (158) | 40 ns / 40 ns | 256.5 (205.4) / 525.6 (605.5) | PASS (6/6 runs) |
| 16x512 | 10T | 369 (656) / 363 (567) | 170 (472) / 173 (471) | n/a / n/a | 738.9 (1228.1) / 3349.4 (7729.3) | PASS (4/4 runs) |
| 16x512 | 6T | 360 (650) / 357 (559) | 158 (460) / 160 (460) | 40 ns / 40 ns | 729.8 (1286.9) / 3378.0 (7830.1) | PASS (6/6 runs) |

#### Address change (`next_row`, middle address bit flipped unless noted; read decks with the other cells storing 1, write decks storing 0)

`max V(WL_next)` is the highest level of the next row's wordline between the end of the access and the next access; the neighbour is the cell (next_row, target_col).

| array | op | next row (bit flipped) | V2.0.1 circuit: max V(WL_next) during hold [V], neighbour Q min/max [V] | V2.0.2 circuit: max V(WL_next), neighbour Q min/max | result |
|---|---|---|---|---|---|
| 256x8 10T | write | 239 (middle) | 0.463, Q -0.000/0.033 **second wordline** | 0.006, Q -0.000/0.000 | PASS |
| 128x32 6T | write | 119 (middle) | 0.006, Q -0.000/0.000 | 0.006, Q -0.000/0.000 | PASS |
| 256x8 6T | read | 254 (LSB) | 0.007, Q 1.000/1.000 | 0.007, Q 1.000/1.000 | PASS |
| 256x8 6T | read | 239 (middle) | 0.502, Q 0.856/1.002 **second wordline** | 0.007, Q 1.000/1.000 | PASS |
| 256x8 6T | write | 254 (LSB) | 0.006, Q -0.000/0.000 | 0.006, Q -0.000/0.000 | PASS |
| 256x8 6T | write | 239 (middle) | 0.406, Q -0.001/0.012 **second wordline** | 0.006, Q -0.000/0.000 | PASS |
| 512x4 6T | read | 495 (middle) | 1.002, Q -0.000/1.001 **second wordline** | 0.007, Q 1.000/1.000 | PASS |
| 512x4 6T | write | 495 (middle) | 1.009, Q 0.000/1.000 **second wordline** | 0.006, Q -0.000/0.000 | PASS |
| 64x16 6T | read | 62 (LSB) | 0.008, Q 1.000/1.000 | 0.008, Q 1.000/1.000 | PASS |
| 64x16 6T | read | 55 (middle) | 0.008, Q 1.000/1.000 | 0.009, Q 1.000/1.000 | PASS |
| 64x16 6T | write | 62 (LSB) | 0.007, Q -0.000/0.000 | 0.007, Q -0.000/0.000 | PASS |
| 64x16 6T | write | 55 (middle) | 0.007, Q -0.000/0.000 | 0.007, Q -0.000/0.000 | PASS |
| 8x4 6T | read | 5 (middle) | 0.020, Q 1.000/1.000 | 0.018, Q 1.000/1.000 | PASS |
| 8x4 6T | write | 5 (middle) | 0.023, Q -0.000/0.000 | 0.016, Q -0.000/0.000 | PASS |

#### Clock period sweep (nominal, TT, 25 C)

A period passes when every waveform check of the V2.0.1 harness passes (access completes inside the clock-low half, bitlines restored, cell data correct). `CLK(min)` is the estimate printed by the flow from the 10 ns run of the same configuration.

| array | cell | mux | op | periods that pass [ns] | periods that fail [ns] (failed check) | delay at 10 ns / at the shortest passing period [ps] | CLK(min) estimate [ns] |
|---|---|---|---|---|---|---|---|
| 16x16 | 10T | off | read | 1.5, 2, 3, 5, 50 | - | 314 / 315 | 0.98 |
| 16x16 | 10T | off | read&write | 2, 3, 5 | - | - / OUT period 8 ns | - |
| 16x16 | 10T | off | write | 1.5, 2, 3, 5, 50 | - | 107 / 108 | 0.55 |
| 8x4 | 10T | off | read | 1 | 0.8 (out_fell,sa_resolved) | 295 / 295 | 0.91 |
| 8x4 | 10T | off | write | 0.8, 1 | - | 131 / 131 | 0.56 |
| 16x16 | 6T | off | read | 0.9, 1, 1.5, 2, 3, 5, 20, 50, 100 | 0.6 (measure FAILED), 0.7 (sen_fired,out_fell,sa_resolved), 0.8 (sen_fired,out_fell,sa_resolved) | 308 / 307 | 0.97 |
| 16x16 | 6T | off | read&write | 1, 1.5, 2, 3, 5, 20, 50 | - | - / OUT period 4 ns | - |
| 16x16 | 6T | off | write | 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 3, 5, 20, 50, 100 | - | 94 / 94 | 0.54 |
| 16x16 | 6T | on | read | 2, 3, 5, 20 | - | 304 / 304 | 0.96 |
| 16x16 | 6T | on | read&write | 2, 3, 5, 20 | - | - / OUT period 8 ns | - |
| 16x16 | 6T | on | write | 2, 3, 5, 20 | - | 99 / 100 | 0.55 |
| 64x16 | 6T | off | read | 2, 3, 5 | - | 358 / 357 | 1.08 |
| 64x16 | 6T | off | read&write | 3, 5 | - | - / OUT period 12 ns | - |
| 64x16 | 6T | off | write | 2, 3, 5 | - | 93 / 94 | 0.57 |
| 8x4 | 6T | off | read | 0.9, 1, 1.5, 2, 3, 5, 20, 50, 100 | 0.6 (sen_fired,out_fell,sa_resolved), 0.7 (sen_fired,out_fell,sa_resolved), 0.8 (out_fell) | 291 / 291 | 0.90 |
| 8x4 | 6T | off | read&write | 1, 1.5, 2, 3, 5, 20, 50 | 0.8 (out_sequence_follows_reads) | - / OUT period 4 ns | - |
| 8x4 | 6T | off | write | 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 3, 5, 20, 50, 100 | - | 131 / 131 | 0.56 |

#### Process corners and temperature (default 10 ns clock, nominal devices)

| array | cell | op | condition | delay [ps] / OUT period | PAVG [uW] | PSTC [uW] | BL / BLB / RBL at next access [V] | checks |
|---|---|---|---|---|---|---|---|---|
| 16x16 | 10T | read | SS -40 C | 235 | 46.2 | 0.93 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | read | TT -40 C | 218 | 48.4 | 1.68 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | read | FF 25 C | 286 | 55.6 | 7.59 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | read | FS 25 C | 313 | 51.6 | 4.76 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | read | SF 25 C | 319 | 51.6 | 4.71 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | read | SS 25 C | 348 | 47.4 | 1.86 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | read | TT 85 C | 447 | 56.0 | 8.75 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | read | FF 125 C | 490 | 76.7 | 28.24 | 0.999 / 1.000 / 0.999 | PASS |
| 16x16 | 10T | read | TT 125 C | 550 | 62.1 | 14.87 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | read&write | SS -40 C | 40 ns | 74.2 | 1.23 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | read&write | TT -40 C | 40 ns | 77.1 | 2.23 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | read&write | FF 25 C | 40 ns | 87.6 | 10.39 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | read&write | FS 25 C | 40 ns | 80.8 | 6.03 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | read&write | SF 25 C | 40 ns | 81.7 | 6.95 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | read&write | SS 25 C | 40 ns | 75.6 | 2.57 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | read&write | TT 85 C | 40 ns | 87.6 | 12.02 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | read&write | FF 125 C | 40 ns | 116.2 | 38.90 | 0.999 / 1.000 / 0.999 | PASS |
| 16x16 | 10T | read&write | TT 125 C | 40 ns | 95.9 | 20.37 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | write | SS -40 C | 75 | 88.5 | 1.18 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | write | TT -40 C | 70 | 91.9 | 2.22 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | write | FF 25 C | 98 | 103.5 | 10.34 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | write | FS 25 C | 100 | 94.6 | 6.00 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | write | SF 25 C | 116 | 96.5 | 6.91 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | write | SS 25 C | 118 | 89.0 | 2.55 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | write | TT 85 C | 158 | 102.0 | 11.97 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 10T | write | FF 125 C | 176 | 133.5 | 38.76 | 0.999 / 1.000 / 0.999 | PASS |
| 16x16 | 10T | write | TT 125 C | 197 | 111.1 | 20.29 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | read | SS -40 C | 231 | 46.2 | 0.83 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | read | TT -40 C | 214 | 48.0 | 1.47 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | read | FF 25 C | 281 | 55.2 | 6.90 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | read | FS 25 C | 306 | 50.6 | 4.07 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | read | SF 25 C | 312 | 51.1 | 4.51 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | read | SS 25 C | 341 | 47.1 | 1.69 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | read | TT 85 C | 435 | 54.8 | 8.02 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | read | FF 125 C | 478 | 74.0 | 26.00 | 0.999 / 1.000 / 0.999 | PASS |
| 16x16 | 6T | read | TT 125 C | 536 | 60.7 | 13.61 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | read&write | SS -40 C | 40 ns | 73.8 | 1.07 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | read&write | TT -40 C | 40 ns | 76.9 | 2.10 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | read&write | FF 25 C | 40 ns | 86.9 | 9.80 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | read&write | FS 25 C | 40 ns | 80.1 | 5.35 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | read&write | SF 25 C | 40 ns | 81.6 | 6.82 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | read&write | SS 25 C | 40 ns | 75.5 | 2.36 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | read&write | TT 85 C | 40 ns | 86.7 | 11.30 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | read&write | FF 125 C | 40 ns | 114.2 | 36.68 | 0.999 / 1.000 / 0.999 | PASS |
| 16x16 | 6T | read&write | TT 125 C | 40 ns | 94.8 | 19.12 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | write | SS -40 C | 66 | 88.0 | 1.07 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | write | TT -40 C | 62 | 92.1 | 2.10 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | write | FF 25 C | 87 | 102.8 | 9.76 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | write | FS 25 C | 88 | 94.2 | 5.33 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | write | SF 25 C | 104 | 96.5 | 6.78 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | write | SS 25 C | 104 | 89.0 | 2.35 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | write | TT 85 C | 139 | 101.4 | 11.24 | 1.000 / 1.000 / 1.000 | PASS |
| 16x16 | 6T | write | FF 125 C | 156 | 131.7 | 36.53 | 0.999 / 1.000 / 0.999 | PASS |
| 16x16 | 6T | write | TT 125 C | 174 | 110.2 | 19.04 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | read | SS -40 C | 221 | 27.4 | 0.34 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | read | TT -40 C | 205 | 29.0 | 0.57 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | read | FF 25 C | 269 | 32.3 | 2.67 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | read | FS 25 C | 294 | 29.7 | 1.36 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | read | SF 25 C | 299 | 30.1 | 1.96 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | read | SS 25 C | 327 | 27.8 | 0.65 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | read | TT 85 C | 418 | 31.7 | 3.04 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | read | FF 125 C | 459 | 40.2 | 9.85 | 0.999 / 1.000 / 0.999 | PASS |
| 8x4 | 10T | read | TT 125 C | 515 | 34.2 | 5.12 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | read&write | SS -40 C | 40 ns | 33.6 | 0.37 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | read&write | TT -40 C | 40 ns | 35.2 | 0.73 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | read&write | FF 25 C | 40 ns | 39.1 | 3.35 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | read&write | FS 25 C | 40 ns | 36.0 | 1.65 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | read&write | SF 25 C | 40 ns | 36.9 | 2.49 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | read&write | SS 25 C | 40 ns | 34.0 | 0.80 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | read&write | TT 85 C | 40 ns | 38.5 | 3.80 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | read&write | FF 125 C | 40 ns | 48.6 | 12.34 | 0.999 / 1.000 / 0.999 | PASS |
| 8x4 | 10T | read&write | TT 125 C | 40 ns | 41.4 | 6.40 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | write | SS -40 C | 91 | 36.8 | 0.38 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | write | TT -40 C | 86 | 38.7 | 0.71 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | write | FF 25 C | 121 | 42.5 | 3.30 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | write | FS 25 C | 119 | 38.9 | 1.63 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | write | SF 25 C | 152 | 40.2 | 2.46 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | write | SS 25 C | 144 | 36.9 | 0.79 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | write | TT 85 C | 195 | 41.7 | 3.75 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 10T | write | FF 125 C | 222 | 51.9 | 12.20 | 0.999 / 1.000 / 0.999 | PASS |
| 8x4 | 10T | write | TT 125 C | 246 | 44.5 | 6.32 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | read | SS -40 C | 218 | 27.5 | 0.30 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | read | TT -40 C | 203 | 28.8 | 0.55 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | read | FF 25 C | 265 | 32.3 | 2.56 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | read | FS 25 C | 290 | 29.5 | 1.25 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | read | SF 25 C | 295 | 30.1 | 1.92 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | read | SS 25 C | 321 | 27.8 | 0.62 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | read | TT 85 C | 411 | 31.5 | 2.93 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | read | FF 125 C | 451 | 39.8 | 9.51 | 0.999 / 1.000 / 0.999 | PASS |
| 8x4 | 6T | read | TT 125 C | 507 | 33.9 | 4.93 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | read&write | SS -40 C | 40 ns | 33.8 | 0.38 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | read&write | TT -40 C | 40 ns | 35.3 | 0.69 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | read&write | FF 25 C | 40 ns | 39.4 | 3.25 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | read&write | FS 25 C | 40 ns | 36.1 | 1.55 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | read&write | SF 25 C | 40 ns | 37.7 | 2.47 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | read&write | SS 25 C | 40 ns | 34.3 | 0.78 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | read&write | TT 85 C | 40 ns | 38.8 | 3.69 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | read&write | FF 125 C | 40 ns | 48.8 | 12.01 | 0.999 / 1.000 / 0.999 | PASS |
| 8x4 | 6T | read&write | TT 125 C | 40 ns | 41.7 | 6.22 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | write | SS -40 C | 86 | 37.2 | 0.37 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | write | TT -40 C | 83 | 38.9 | 0.68 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | write | FF 25 C | 123 | 43.3 | 3.20 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | write | FS 25 C | 115 | 39.2 | 1.53 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | write | SF 25 C | 174 | 41.9 | 2.43 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | write | SS 25 C | 140 | 37.6 | 0.76 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | write | TT 85 C | 206 | 42.5 | 3.64 | 1.000 / 1.000 / 1.000 | PASS |
| 8x4 | 6T | write | FF 125 C | 244 | 52.8 | 11.86 | 0.999 / 1.000 / 0.999 | PASS |
| 8x4 | 6T | write | TT 125 C | 266 | 45.3 | 6.14 | 1.000 / 1.000 / 1.000 | PASS |

#### Monte Carlo (Xyce `.SAMPLING`, `vth_std = 0.05`, seed 2026)

| array | cell | mux | op | samples | delay mean +- sd [ps] | min / max [ps] | PAVG mean [uW] | waveform checks (all samples) |
|---|---|---|---|---|---|---|---|---|
| 16x16 | 10T | off | read | 5 | 318.2 +- 4.6 | 311.3 / 324.0 | 50.0 | PASS |
| 16x16 | 10T | off | read&write | 3 | OUT period 40.00 ns | - | 80.0 | PASS |
| 16x16 | 10T | off | write | 5 | 110.4 +- 2.3 | 108.4 / 114.7 | 93.4 | PASS |
| 16x16 | 10T | on | read | 5 | 314.7 +- 4.2 | 308.5 / 320.2 | 44.6 | PASS |
| 16x16 | 10T | on | read&write | 3 | OUT period 40.00 ns | - | 74.9 | PASS |
| 16x16 | 10T | on | write | 5 | 114.5 +- 2.8 | 111.3 / 119.6 | 88.7 | PASS |
| 32x8 | 10T | off | read | 5 | 336.3 +- 4.1 | 330.0 / 341.3 | 48.1 | PASS |
| 32x8 | 10T | off | write | 5 | 96.6 +- 2.0 | 94.7 / 100.2 | 71.3 | PASS |
| 64x16 | 10T | off | read | 5 | 382.6 +- 4.6 | 375.2 / 387.7 | 90.0 | PASS |
| 64x16 | 10T | off | read + address change | 5 | 382.8 +- 5.0 | 375.2 / 389.0 | 91.9 | PASS |
| 64x16 | 10T | off | write | 5 | 107.1 +- 1.7 | 105.1 / 109.9 | 154.6 | PASS |
| 64x16 | 10T | off | write + address change | 5 | 107.1 +- 1.7 | 105.1 / 109.9 | 156.4 | PASS |
| 64x16 | 10T | on | read | 5 | 378.3 +- 4.8 | 371.3 / 385.3 | 84.7 | PASS |
| 64x16 | 10T | on | write | 5 | 108.1 +- 1.8 | 105.9 / 111.0 | 150.9 | PASS |
| 8x4 | 10T | off | read | 5 | 299.0 +- 3.6 | 293.6 / 303.6 | 29.2 | PASS |
| 8x4 | 10T | off | read&write | 3 | OUT period 40.00 ns | - | 35.9 | PASS |
| 8x4 | 10T | off | write | 5 | 137.7 +- 5.1 | 132.1 / 147.2 | 38.8 | PASS |
| 8x4 | 10T | on | read | 5 | 297.9 +- 3.6 | 292.4 / 302.5 | 28.2 | PASS |
| 8x4 | 10T | on | read&write | 3 | OUT period 40.00 ns | - | 35.0 | PASS |
| 8x4 | 10T | on | write | 5 | 154.2 +- 6.2 | 147.3 / 165.7 | 38.0 | PASS |
| 16x16 | 6T | off | read | 5 | 311.7 +- 3.8 | 305.9 / 316.6 | 49.6 | PASS |
| 16x16 | 6T | off | read&write | 3 | OUT period 40.00 ns | - | 79.8 | PASS |
| 16x16 | 6T | off | write | 5 | 98.1 +- 2.3 | 95.5 / 102.1 | 93.2 | PASS |
| 16x16 | 6T | on | read | 5 | 307.9 +- 3.9 | 302.0 / 313.0 | 44.1 | PASS |
| 16x16 | 6T | on | read&write | 3 | OUT period 40.00 ns | - | 74.6 | PASS |
| 16x16 | 6T | on | write | 5 | 102.5 +- 2.7 | 100.5 / 107.7 | 88.4 | PASS |
| 32x8 | 6T | off | read | 5 | 324.5 +- 3.9 | 318.2 / 329.4 | 47.7 | PASS |
| 32x8 | 6T | off | write | 5 | 84.6 +- 1.8 | 82.6 / 87.8 | 71.1 | PASS |
| 64x16 | 6T | off | read | 5 | 362.2 +- 4.9 | 354.4 / 368.3 | 88.8 | PASS |
| 64x16 | 6T | off | read + address change | 5 | 362.2 +- 4.9 | 354.6 / 368.3 | 91.2 | PASS |
| 64x16 | 6T | off | write | 5 | 95.8 +- 1.6 | 94.0 / 98.2 | 153.4 | PASS |
| 64x16 | 6T | off | write + address change | 5 | 95.8 +- 1.6 | 94.0 / 98.2 | 155.2 | PASS |
| 64x16 | 6T | on | read | 5 | 358.1 +- 4.8 | 350.9 / 364.4 | 83.4 | PASS |
| 64x16 | 6T | on | write | 5 | 96.7 +- 1.7 | 94.9 / 99.3 | 149.5 | PASS |
| 8x4 | 6T | off | read | 5 | 294.6 +- 3.5 | 289.2 / 299.0 | 29.2 | PASS |
| 8x4 | 6T | off | read&write | 3 | OUT period 40.00 ns | - | 36.2 | PASS |
| 8x4 | 6T | off | write | 5 | 140.3 +- 9.2 | 132.7 / 158.0 | 39.6 | PASS |
| 8x4 | 6T | on | read | 5 | 293.4 +- 3.4 | 288.1 / 297.8 | 28.2 | PASS |
| 8x4 | 6T | on | read&write | 3 | OUT period 40.00 ns | - | 35.3 | PASS |
| 8x4 | 6T | on | write | 5 | 161.0 +- 11.5 | 150.5 / 182.8 | 38.8 | PASS |

#### 128x128 and 256x64 (4 ns clock, launched during the release)

- 6T_128x128_m0_read_T4: delay 475 ps, PAVG 1678.4 uW, waveform checks PASS (3.0 h)
- 6T_128x128_m0_write_T4: delay 155 ps, PAVG 3987.9 uW, waveform checks PASS (3.6 h)
- 6T_256x64_m0_read_T4: delay 568 ps, PAVG 1636.3 uW, waveform checks PASS (3.0 h)
- 6T_256x64_m0_write_T4: delay 149 ps, PAVG 3463.5 uW, waveform checks PASS (3.2 h)

## V2.0.1 — 2026-09-05 — transient / Monte Carlo circuit review

Scope of this release: the SRAM compiler circuits (6T and 10T cores, replica
column, timing generator, decoder, wordline driver, precharge, column mux,
sense amplifier, write driver, output latch), the transient testbenches
(`read`, `write`, `read&write`), their measurements, and the Xyce simulation /
result-parsing flow. Optimisation and yield-estimation algorithms were not
reviewed in this round.

Every change below was validated by running the generated netlists in Xyce
7.4 and scoring the waveforms automatically (wordline, bitlines, replica
bitline, precharge, sense enable, write enable, sense-amp outputs, output
latch, target-cell Q/QB) for every configuration. The final sweep covers
29 planned array sizes from 1x1 to 512x4, 16x512 and 100x50 (the 128x128 and
256x64 decks were stopped for run time, see the evidence section), both
cells, column mux on and off, all three operations, nominal and seeded Monte
Carlo. The evidence tables are at the end of this entry; the per-run
netlists, waveforms and Xyce logs were produced by the scratch sweep harness
described in `CIRCUIT_REVIEW.md`.

### Summary of fixes

- **Write pulse too short.** `w_en` was cut by the replica bitline (~250 ps),
  a cell-strength path, while the write path (row-scaled write driver, through
  the column mux) is weaker. Seeded 5 % sigma Monte Carlo samples left the
  bitline at 0.3-0.4 V and the cell kept its old data. `w_en` now spans the
  wordline phase; the hard-coded 16x512 `WenDelayChain` hack is removed.
- **Hold hazard introduced by that fix, caught by the sweep.** At the clock
  edge that ends a write the data register updates before the drivers
  release, so the next cycle's data was briefly written; at 64 rows this
  flipped the freshly written cell (`read&write` 64x16 failed). A per-column
  write-data hold latch fixes it.
- **Write testbench topology.** It had no bitline precharge and no sense-amp
  / mux load, so the stand-alone write delay was 30-40 % optimistic against
  the same write inside the `read&write` sequence. All transient decks now
  carry the full column periphery; `TWDRV` is measured on the driven bitline.
- **Nominal runs were random samples.** Every deck emitted `.SAMPLING` with
  a random seed, so identical calls returned different delays and
  occasionally failed. `mc_runs=1` is now deterministic; `mc_seed` makes
  Monte Carlo sweeps reproducible; the Xyce console log is kept per run.
- **Energy window** measured the start-up charging of the bitlines from 0 V
  (half of the "read energy" on 8x4) instead of a steady-state cycle; it is
  now one full clock period starting at the access. `read&write` averages
  over one 4-cycle pattern and its transient covers the 8th access.
- **Measurement details.** `TS_EN` was corrupted by a precharge-coupling bump
  on `s_en` (now measured from the access phase); the `w_en` buffer was not
  scaled with the row-dependent write-driver size (release lagged the
  wordline by 40-285 ps on 64-256-row arrays); the 10T core ignored the
  testbench RC parameters; a PySpice subcircuit-name collision (one
  definition per name and scope) is avoided with a dedicated `AND2_WEN`.
- **Xyce Newton stall on some 512-row decks** (residual 1e-12 A at every
  step size, not a circuit fault): the flow retries once with a 20 ps maximum
  time step, which keeps results of converging decks within 0.5 %;
  `t_max_step` and `xyce_options` are exposed on `Sram6TCoreMcTestbench`.
- **Static-review fixes carried into this release** (details in
  `CIRCUIT_REVIEW.md`): column-mux port mismatch that aborted every muxed
  read, free-running `SEL` pulse, wrong output-latch index and floating latch
  input on writes, replica column driven by the real wordlines, CS start-up
  clamp fighting the flip-flop, negative read delay and negative dynamic
  power from mis-placed measure thresholds and windows, write delay
  over-reported 2.2x by summing overlapping segments, `FAILED` measures
  silently becoming 0.0, `.prn` / SNM parsing that depended on `.PRINT`
  ordering, equivalent-circuit caps only inserted with `w_rc`, unused `regex`
  import, `python-graphviz` pip name, duplicated `config.py`, `mW` label.

### Not completed / left open

- **Array sizes not finished at release time.** 128x128 and 256x64 (all
  operations): the read transient reached 2.0 ns of 21 ns after 64 min
  (~11 h per read deck, ~45 h per `read&write` deck), so these 24 decks were
  stopped. The `read&write` decks of 8x512, 16x512, 16x256 and 10T 512x4
  (86 ns transients) were still running after 3-5 h and were not waited for;
  their `read` and `write` decks completed and are in the evidence table.
  All 283 completed decks pass every waveform check. The same flow runs them
  offline; use `t_max_step` / `xyce_options` if Xyce stalls.
- **Address-path hold hazard** (pre-existing, not exercised): the access
  window ends at the next capture edge, so a *changed* address would raise a
  second wordline for ~100-250 ps while the old one is still falling. The
  testbenches keep the address constant. A hold latch on `A_dff` (transparent
  while `wl_en` is low) or a delayed register clock would close it.
- **Design choices verified and left as they are** (see Observations below):
  sensing waits for a fully discharged replica bitline plus a 9-stage delay
  chain, so read delay is ~300 ps for every size up to 32 rows; the precharge
  is a ~300 ps self-timed pulse after which the bitlines float and droop to
  ~0.93 V; the `s_en` buffer keeps the columns/64 scaling; the `w_rc=True`
  default of `main_sram.py` puts 1 fF on every cell's Q/QB and triples the
  write delay; the fixed 10 ns clock leaves > 4 ns of margin for every size
  tested; `read` always reads a stored 0 (`read&write` covers both values).
- **Out of scope this round:** optimisation and yield-estimation algorithms,
  the SNM extraction beyond a sanity run (6T hold/read/write 0.325 / 0.182 /
  0.365 V, 10T 0.485 / 0.290 / 0.419 V), and parameter-sweep (`sweep_*`) modes.

### Circuit topology

- **Write testbench uses the full column periphery.** The `write` deck had
  only the write drivers and a precharge on the replica column: the bitlines
  started from the artificial `.IC` state (BL = 0 V, BLB = VDD), floated at
  the written values after the write pulse, and carried no sense-amplifier /
  column-mux load. The same write inside the `read&write` sequence took
  30-40 % longer (8x4 6T: 96 ps stand-alone vs 133 ps in the sequence).
  `create_testbench()` now instantiates precharge (all columns + replica
  column), column mux and sense amplifiers for every transient operation and
  adds the write drivers on top, so a write cycle is
  precharge -> write -> precharge with the real bitline load
  (`sram_6t_core_testbench.py`).
- **Write enable spans the whole wordline phase.** `w_en` was
  `rbl_delay_bar & gated_clk_bar & we`, i.e. the write pulse ended as soon as
  the replica cell had discharged the replica bitline (~250 ps). That path
  (cell pull-down through the pass gate) is stronger than the write path
  (row-scaled write driver, optionally through the column-mux transmission
  gate), so the pulse had ~30 % nominal margin and Monte Carlo samples with a
  weak NMOS left the bitline at 0.3-0.4 V when `w_en` ended: the cell kept its
  old data (4x2 6T with mux, 5 % sigma on vth0/u0/voff: 3 of 6 seeds contained
  failing samples, e.g. seed 11 sample 0 cycle 3: BL min 0.397 V, Q stayed 1).
  `w_en` is now `gated_clk_bar & we` (new `AND2_WEN` gate in
  `time_generate.py`, own subcircuit name so it does not replace the larger
  gated-clock `AND2`), the write drivers stay on as long as the wordline, and
  the hard-coded 16x512 `WenDelayChain` special case that lengthened the pulse
  for one array size is removed. With the fix the same seeds pass every
  sample; 12-sample seeded write sweeps at 4x2 and 8x4 pass with the slowest
  sample at ~250 ps.
- **Write data is held while the write drivers are enabled.** With `w_en`
  spanning the clock-low phase, the drivers are released 150-300 ps after
  the rising clock edge that ends the cycle, but the data register `DIN_dff`
  already changes 100-150 ps after that same edge, so the drivers briefly
  wrote the *next* cycle's data into the still-selected row. Small arrays
  survived because their row-scaled write driver is weak; at 64 rows the 4x
  driver flipped the freshly written cell back (`read&write` 64x16, both
  cells: Q = 1 at 12.9 ns, 0 at 13.4 ns, OUT never rose). A per-column
  transparent-low latch (`D_LATCH`, enable = `w_en_bar`) now feeds the write
  drivers: it is transparent while `w_en` is low and holds while the drivers
  are on, and it opens two gate delays after the drivers tristate, so the
  driver input cannot change while the driver is active
  (`create_write_periphery`). The analogous address-path hazard (the decoder
  output of a *new* address rises while the old wordline is still on) is not
  exercised by the testbenches, which keep the address fixed; see the
  observations below.
- **w_en buffer scaled with its real fan-out.** The `w_en` output inverter
  scaled with columns/64 only, while each write driver it drives scales with
  `max(8, rows)/16`; on 64-256-row arrays `w_en` was released 40-70 ps after
  the wordline (285 ps with RC at 64x64) and the drivers overlapped the start
  of the precharge. The row factor is now included; arrays up to 32 rows are
  unchanged.
- **Write-driver delay measured on the driven bitline.** `TWDRV` targeted
  `V(BL)` rising, which only existed because BL started at 0 V; with
  precharged bitlines a '1' is written by pulling BLB low, so `TWDRV` is now
  `w_en` rise -> `V(BLB) = VDD/2` falling.
- 10T core now receives the testbench `pi_res` / `pi_cap` in both the
  standard and the `custom_mc` path (it silently used the class defaults).
- Carried over from the static review (see `CIRCUIT_REVIEW.md`): column-mux
  instantiated with two extra `SELB` ports (aborted every muxed read in Xyce);
  `SEL` was a free-running pulse unrelated to the clock, now a static level;
  output D-latch used `SA_Q{target_col}` instead of `SA_Q{target_col // mux_in}`
  and was created for pure writes with a floating input; `choose_columnmux`
  with `num_cols % mux_in != 0` produced columns without a sense amplifier
  (now rejected); replica column tied its non-replica cells to the real
  wordlines so two cells discharged RBL on every access (now dummies tied to
  VSS); the CS start-up clamp fought the CS flip-flop for 200 ps after the
  first capture edge and re-engaged in cycle 3 (released before the edge,
  one-shot, slave node initialised); stimulus sources used a literal 1.0 V
  instead of `vdd`; `AND2` RC network was a dangling stub; sense-amp pass
  width and wordline-driver column scaling were dropped in parameter-sweep
  mode; `hold_snm` bitlines had no DC path.

### Measurements

- **Energy window is one full clock period starting at the access.**
  `EREAD` / `EWRITE` were integrated over 2 ns .. 2 ns + T, a window that
  contains the start-up precharge charging every bitline from its 0 V initial
  condition (156 fJ of the 298 fJ "read energy" on an 8x4 6T array) and cuts
  the access off 1 ns before the wordline falls, while the real post-access
  precharge (109 fJ) was excluded. The window is now
  `1 ns + 0.7 T` .. `1 ns + 1.7 T`: wordline access, sensing / writing, the
  self-timed precharge, and the idle time to the next access.
- `TS_EN` (s_en 20 %-80 % rise) is measured only from the start of the
  access phase (`TD=` on trigger and target). When the first precharge fires,
  all bitlines rise together and couple through the sense-amplifier pass
  gates into the weakly held `s_en` net; with 32-128 sense amplifiers that
  bump reaches 0.2-0.3 V and was taken as the 20 % crossing, so `TS_EN` read
  ~5.3 ns on 32x32 and 2x128 arrays.
- `read&write`: `PAVG` averages over one complete write-1 / read / write-0 /
  read pattern (4 T) starting at the first access, and the transient runs to
  `1 ns + 8.5 T` so the 8th access is no longer cut 1 ns after its wordline
  rises. `PSTC` / `PDYN` are now produced for this operation too.
- Carried over from the static review: read delay is `TREAD_TOTAL` (`wl_en`
  -> output latch) and write delay is `TWRITE_TOTAL` (`wl_en` -> Q at 90 %)
  instead of a sum of overlapping segments that over-reported the write delay
  2.2x; `TSA` / `TREAD_TOTAL` target the output latch instead of a 10 mV
  threshold on `SA_Q` that sat on the `.IC` parking level (read delay was
  -5.1 ns); `TS_EN` uses the same edge for trigger and target; `PSTC` is
  measured in a quiescent window (`1 ns + [0.4, 0.65] T`) instead of the
  start-up transient (PDYN was negative); `TDECODER` triggers on the lowest
  set address bit; a `FAILED` measure raises instead of contributing 0.0.

### Simulation control and reproducibility

- **`mc_runs = 1` is the nominal point.** Every deck carried
  `.SAMPLING useExpr=true` with a fresh random seed, so a single run was one
  random process sample: identical calls returned different delay / power
  numbers, `size_optimization` evaluated a noisy objective, and the random
  sample occasionally failed the write (the errors above were first seen as
  random single-run failures). `.SAMPLING` is now only emitted for
  `mc_runs > 1`; a single run evaluates every `AGAUSS(...)` at its mean and
  is bit-for-bit repeatable.
- New `mc_seed=<int>` argument of `Sram6TCoreMcTestbench` writes
  `.options samples numsamples=N seed=S`, so a Monte Carlo sweep is
  reproducible (verified: two seeded 6-sample sweeps give identical results).
- The Xyce console output of every run is kept as `<netlist>.log` next to the
  netlist (it was discarded on success, losing all netlist warnings and the
  random seed).
- Carried over: `sweep_senseamp` defaulted to `True` in the MC testbench
  (3-point `.STEP` that broke the block splitter); `.OPTIONS MEASURE MEASFAIL=1`;
  `1/2 CLK` margin adds the spread instead of discarding it.

### Parsing, configuration, documentation

- Carried over: `.prn` reader accepts both `Index` / no-`Index` layouts (the
  flow depended on `.PRINT` ordering); the SNM splitter handles
  `FORMAT=NOINDEX`; write-SNM uses the constriction beyond the eye of the
  remaining lobe instead of a global maximum pinned to the sweep bound;
  equivalent-circuit parasitic caps are inserted regardless of `w_rc`, the
  extraction is cached per process, static-power model failures propagate,
  `_PROJECT_ROOT` fixed; unused `regex` import removed; `python-graphviz`
  -> `graphviz` in `environment.yml`; `config.py` re-exports
  `sram_compiler/config_yaml/config.py`; `demo_run_a_testbench.py` paths and
  keyword; 10T `process_parameters.vars` made 2-D; temperature passed in
  `size_optimization/exp_utils.py`.
- `main_sram.py` prints power in µW (the value was µW, the label said mW).
- `readme_compiler.md` documents the energy window, the nominal single run,
  `mc_seed` and the write-testbench topology.

### Observations (verified, not changed)

- The read delay is ~300 ps almost independently of array size up to 32
  rows because the sense enable is derived from a fully discharged replica
  bitline plus a 9-stage delay chain: the sense amplifier fires ~250 ps after
  the wordline while the target bitline has already swung to ~0.02 V
  (`vswing` = 250 mV is reached after 10-35 ps). Sensing is therefore
  full-swing and robust but slow; the 10 ns clock leaves >4 ns of margin for
  every size tested.
- The precharge is a self-timed ~300 ps pulse; afterwards the bitlines float
  for the rest of the cycle. They overshoot to 1.03-1.07 V from the `PRE`
  gate coupling and then leak down (to 0.93-0.95 V by the next access on the
  10T 16x16 replica bitline and on the 2-row 128-column array). Functionally
  harmless in the sweep, but bitline levels at the start of an access depend
  on the cycle time.
- The `w_rc=True` default of `main_sram.py` places `pi_cap` (1 fF) on the
  internal storage nodes Q/QB of every cell as well as on BL/BLB/WL; this
  triples the write delay (16x16 6T: 98 ps -> 346 ps) and doubles the power.
  It is a parameter choice, not a code defect, but the default is heavy.
- `read` always reads a stored 0; `read&write` covers both data values.
- The equivalent-circuit model (`real_cell_mode=1`) tracks the all-real array
  with the same RC model within 6-12 % on delay and 5 % on power at 16x16
  (`w_rc=True`, 6T: read 501 vs 533 ps, write 300 vs 263 ps, 98 vs 97 µW;
  10T: read 541 vs 591 ps, write 275 vs 257 ps). The evidence table below
  lists the equivalent-circuit runs next to the no-RC all-real runs; most of
  that difference is the RC model itself (see the `w_rc` note above).
- The end of the access window coincides with the next capture edge, so any
  input that changes at that edge (address, data, chip select) races the
  release of the wordline. The data path is now held (see above); the address
  path is not exercised by the testbenches (the target address is constant),
  but a decoder output of a new address would rise ~250 ps after the edge
  while `WL_EN` is still falling, briefly selecting a second row. A hold
  latch on `A_dff` (transparent while `wl_en` is low) or a delayed register
  clock would close it.
- The `s_en` buffer keeps the authors' columns/64 scaling; it slows the sense
  enable by ~60 ps at 128 columns and lets the precharge-coupling bump on
  `s_en` reach 0.23 V. Not functional, but worth a fan-out-aware size like
  the wordline driver's.
- Xyce convergence at 512 rows: some nominal 512x4 decks stop with "time
  step too small" during the access (the Newton loop oscillates, 21
  iterations with a residual of only 1e-12 A, at every step size). Relaxing
  `ABSTOL`, `NLNEARCONV` or `MAXSTEP` does not help; `ERROPTION=1` completes
  the deck but shifts delays of converging decks by 2-15 % and power by up to
  2.4 %; a `.TRAN` maximum step of 20 ps completes it and keeps converging
  decks within 0.5 % (delay) / 0.4 % (energy) at ~1.8x the time steps.
  `run_mc_simulation()` therefore retries once with that maximum step when
  Xyce reports the failure (new `t_max_step` and `xyce_options` arguments of
  `Sram6TCoreMcTestbench` expose both knobs); details in the solver notes of
  `CIRCUIT_REVIEW.md`.
- A `.SAMPLING` run of the old netlist on 10T 64x16 with mux aborted with a
  Xyce "time step too small" at the start of the access; the nominal run with
  the corrected topology completes (139 ps).

### Evidence

Filled in below from the final sweep (nominal, all cells real, `w_rc=False`,
25 °C, TT, target cell = last row / last column).

#### Size sweep (nominal, all cells real, no RC, mux off / on)

283 of 318 configurations simulated (35 not finished at
release time), 283 completed in Xyce, 283 pass every waveform check.
Errors: none.
Not completed at release time (35): 10T_512x4_m1_rw, 10T_16x256_m0_rw, 10T_16x256_m1_rw, 6T_8x512_m0_rw, 6T_8x512_m1_rw, 10T_8x512_m0_rw, 10T_8x512_m1_rw, 6T_16x512_m0_rw, 6T_16x512_m1_rw, 10T_16x512_m0_rw, 10T_16x512_m1_rw, 6T_128x128_m0_read, 6T_128x128_m0_write, 6T_128x128_m0_rw, 6T_128x128_m1_read, 6T_128x128_m1_write, 6T_128x128_m1_rw, 10T_128x128_m0_read, 10T_128x128_m0_write, 10T_128x128_m0_rw, 10T_128x128_m1_read, 10T_128x128_m1_write, 10T_128x128_m1_rw, 6T_256x64_m0_read, 6T_256x64_m0_write, 6T_256x64_m0_rw, 6T_256x64_m1_read, 6T_256x64_m1_write, 6T_256x64_m1_rw, 10T_256x64_m0_read, 10T_256x64_m0_write, 10T_256x64_m0_rw, 10T_256x64_m1_read, 10T_256x64_m1_write, 10T_256x64_m1_rw.
The 128x128 and 256x64 decks (16k cells, ~100k transistors) were stopped after
64 min at 2.0 ns of the 21 ns read transient (~11 h per read deck, ~45 h per
read&write deck at that rate); the 8x512 / 16x512 / 16x256 / 512x4-10T
read&write decks (8.5 clock cycles, 86 ns) were still running at ~40-70 ns
after 3-5 h. Their read and write decks completed and are in the table.
Delays in ps (read: `wl_en` rise -> `OUT`; write: `wl_en` rise -> Q at 90 %);
`read&write` shows the measured period of `OUT` (40 ns = 4 clock cycles is the
correct write-1 / read / write-0 / read sequence); PAVG at 100 MHz.

| array | cell | read delay [ps] mux off / on | write delay [ps] mux off / on | read&write OUT period mux off / on | PAVG read / write [µW] | waveform checks |
|---|---|---|---|---|---|---|
| 1x1 | 10T | 289 / n/a | 133 / n/a | 40 ns / n/a | 20.1 / 21.8 | PASS (3 runs) |
| 1x1 | 6T | 286 / n/a | 131 / n/a | 40 ns / n/a | 20.1 / 22.0 | PASS (3 runs) |
| 2x1 | 10T | 290 / n/a | 134 / n/a | 40 ns / n/a | 21.5 / 23.1 | PASS (3 runs) |
| 2x1 | 6T | 287 / n/a | 132 / n/a | 40 ns / n/a | 21.4 / 23.3 | PASS (3 runs) |
| 2x2 | 10T | 293 / 290 | 134 / 164 | 40 ns / 40 ns | 22.2 / 26.4 | PASS (6 runs) |
| 2x2 | 6T | 290 / 287 | 132 / 174 | 40 ns / 40 ns | 22.2 / 26.8 | PASS (6 runs) |
| 4x2 | 10T | 295 / 292 | 136 / 166 | 40 ns / 40 ns | 24.1 / 28.3 | PASS (6 runs) |
| 4x2 | 6T | 292 / 289 | 134 / 176 | 40 ns / 40 ns | 24.0 / 28.8 | PASS (6 runs) |
| 3x3 | 10T | 297 / n/a | 136 / n/a | 40 ns / n/a | 23.7 / 30.5 | PASS (3 runs) |
| 3x3 | 6T | 294 / n/a | 132 / n/a | 40 ns / n/a | 23.6 / 31.0 | PASS (3 runs) |
| 5x3 | 10T | 299 / n/a | 138 / n/a | 40 ns / n/a | 24.5 / 31.3 | PASS (3 runs) |
| 5x3 | 6T | 296 / n/a | 134 / n/a | 40 ns / n/a | 24.4 / 31.9 | PASS (3 runs) |
| 4x4 | 10T | 301 / 295 | 138 / 168 | 40 ns / 40 ns | 25.7 / 35.1 | PASS (6 runs) |
| 4x4 | 6T | 298 / 292 | 133 / 176 | 40 ns / 40 ns | 25.7 / 36.0 | PASS (6 runs) |
| 16x1 | 10T | 308 / n/a | 86 / n/a | 40 ns / n/a | 28.3 / 29.9 | PASS (3 runs) |
| 16x1 | 6T | 302 / n/a | 73 / n/a | 40 ns / n/a | 28.3 / 29.8 | PASS (3 runs) |
| 8x4 | 10T | 305 / 300 | 142 / 172 | 40 ns / 40 ns | 28.0 / 37.7 | PASS (6 runs) |
| 8x4 | 6T | 301 / 295 | 138 / 181 | 40 ns / 40 ns | 28.0 / 38.4 | PASS (6 runs) |
| 6x6 | 10T | 308 / 301 | 142 / 171 | 40 ns / 40 ns | 28.4 / 43.2 | PASS (6 runs) |
| 6x6 | 6T | 305 / 297 | 137 / 180 | 40 ns / 40 ns | 28.3 / 44.5 | PASS (6 runs) |
| 12x4 | 10T | 311 / 305 | 106 / 119 | 40 ns / 40 ns | 29.8 / 38.7 | PASS (6 runs) |
| 12x4 | 6T | 306 / 300 | 93 / 110 | 40 ns / 40 ns | 29.5 / 38.8 | PASS (6 runs) |
| 8x8 | 10T | 315 / 306 | 147 / 174 | 40 ns / 40 ns | 31.6 / 52.2 | PASS (6 runs) |
| 8x8 | 6T | 311 / 301 | 141 / 183 | 40 ns / 40 ns | 31.4 / 53.7 | PASS (6 runs) |
| 16x8 | 10T | 327 / 316 | 98 / 105 | 40 ns / 40 ns | 36.2 / 56.0 | PASS (6 runs) |
| 16x8 | 6T | 320 / 310 | 85 / 93 | 40 ns / 40 ns | 35.8 / 55.7 | PASS (6 runs) |
| 20x10 | 10T | 336 / 324 | 95 / 99 | 40 ns / 40 ns | 40.1 / 66.0 | PASS (6 runs) |
| 20x10 | 6T | 329 / 317 | 83 / 87 | 40 ns / 40 ns | 39.8 / 65.4 | PASS (6 runs) |
| 2x128 | 10T | 463 / 386 | 202 / 224 | 40 ns / 40 ns | 158.8 / 734.0 | PASS (6 runs) |
| 2x128 | 6T | 458 / 382 | 191 / 225 | 40 ns / 40 ns | 158.4 / 740.1 | PASS (6 runs) |
| 16x16 | 10T | 344 / 327 | 111 / 116 | 40 ns / 40 ns | 45.3 / 88.8 | PASS (6 runs) |
| 16x16 | 6T | 337 / 320 | 98 / 105 | 40 ns / 40 ns | 44.9 / 89.1 | PASS (6 runs) |
| 32x8 | 10T | 347 / 337 | 84 / 86 | 40 ns / 40 ns | 43.9 / 65.7 | PASS (6 runs) |
| 32x8 | 6T | 336 / 327 | 72 / 73 | 40 ns / 40 ns | 43.5 / 65.6 | PASS (6 runs) |
| 32x32 | 10T | 396 / 364 | 148 / 149 | 40 ns / 40 ns | 81.9 / 227.4 | PASS (6 runs) |
| 32x32 | 6T | 386 / 355 | 137 / 138 | 40 ns / 40 ns | 81.1 / 227.8 | PASS (6 runs) |
| 64x16 | 10T | 401 / 383 | 125 / 125 | 40 ns / 40 ns | 76.3 / 166.6 | PASS (6 runs) |
| 64x16 | 6T | 382 / 366 | 114 / 114 | 40 ns / 40 ns | 75.1 / 165.8 | PASS (6 runs) |
| 256x8 | 10T | 579 / 566 | 78 / 79 | 40 ns / 40 ns | 124.9 / 229.2 | PASS (6 runs) |
| 256x8 | 6T | 526 / 514 | 68 / 68 | 40 ns / 40 ns | 122.2 / 226.5 | PASS (6 runs) |
| 512x4 | 10T | 791 / 783 | 80 / 77 | 40 ns / n/a | 153.3 / 248.2 | PASS (5 runs) |
| 512x4 | 6T | 709 / 692 | 70 / 69 | 40 ns / 40 ns | 148.3 / 245.5 | PASS (6 runs) |
| 8x512 | 10T | 642 / 557 | 512 / 509 | n/a / n/a | 1034.5 / 7386.7 | PASS (4 runs) |
| 8x512 | 6T | 640 / 550 | 509 / 509 | n/a / n/a | 1078.5 / 7527.4 | PASS (4 runs) |
| 16x256 | 10T | 542 / 457 | 273 / 275 | n/a / n/a | 444.6 / 2340.0 | PASS (4 runs) |
| 16x256 | 6T | 535 / 449 | 261 / 264 | 40 ns / 40 ns | 441.7 / 2339.6 | PASS (6 runs) |
| 64x64 | 10T | 505 / 440 | 173 / 173 | 40 ns / 40 ns | 210.2 / 613.0 | PASS (6 runs) |
| 64x64 | 6T | 492 / 421 | 162 / 160 | 40 ns / 40 ns | 202.9 / 610.8 | PASS (6 runs) |
| 128x32 | 10T | 510 / 467 | 146 / 145 | 40 ns / 40 ns | 181.1 / 462.8 | PASS (6 runs) |
| 128x32 | 6T | 468 / 434 | 135 / 133 | 40 ns / 40 ns | 174.8 / 459.4 | PASS (6 runs) |
| 100x50 | 10T | 517 / 464 | 168 / 168 | 40 ns / 40 ns | 212.7 / 607.9 | PASS (6 runs) |
| 100x50 | 6T | 488 / 438 | 157 / 158 | 40 ns / 40 ns | 205.4 / 605.5 | PASS (6 runs) |
| 16x512 | 10T | 656 / 567 | 472 / 471 | n/a / n/a | 1228.1 / 7729.3 | PASS (4 runs) |
| 16x512 | 6T | 650 / 559 | 460 / 460 | n/a / n/a | 1286.9 / 7830.1 | PASS (4 runs) |

#### Monte Carlo (Xyce `.SAMPLING`, `vth_std = 0.05`, seed 2026)

| array | cell | mux | op | samples | delay mean ± sd [ps] | min / max [ps] | PAVG mean [µW] | waveform checks (all samples) |
|---|---|---|---|---|---|---|---|---|
| 8x4 | 10T | off | read | 5 | 309.5 ± 3.7 | 303.9 / 314.1 | 27.8 | PASS |
| 8x4 | 10T | off | read&write | 3 | OUT period 40.00 ns | - | 34.9 | PASS |
| 8x4 | 10T | off | write | 5 | 152.0 ± 7.5 | 143.8 / 166.0 | 37.6 | PASS |
| 8x4 | 10T | on | read | 5 | 303.8 ± 3.6 | 298.3 / 308.4 | 26.8 | PASS |
| 8x4 | 10T | on | read&write | 3 | OUT period 40.00 ns | - | 34.2 | PASS |
| 8x4 | 10T | on | write | 5 | 184.6 ± 10.3 | 172.7 / 203.5 | 37.0 | PASS |
| 8x4 | 6T | off | read | 5 | 305.1 ± 3.6 | 299.6 / 309.7 | 27.8 | PASS |
| 8x4 | 6T | off | read&write | 3 | OUT period 40.00 ns | - | 35.3 | PASS |
| 8x4 | 6T | off | write | 5 | 153.3 ± 14.2 | 139.9 / 180.3 | 38.6 | PASS |
| 8x4 | 6T | on | read | 5 | 299.5 ± 3.5 | 294.2 / 303.9 | 26.8 | PASS |
| 8x4 | 6T | on | read&write | 3 | OUT period 40.00 ns | - | 34.6 | PASS |
| 8x4 | 6T | on | write | 5 | 202.8 ± 20.6 | 182.1 / 241.6 | 38.1 | PASS |
| 16x16 | 10T | off | read | 5 | 348.3 ± 3.8 | 342.9 / 353.3 | 44.8 | PASS |
| 32x8 | 10T | off | read | 5 | 351.3 ± 4.0 | 345.6 / 357.0 | 43.4 | PASS |
| 16x16 | 10T | off | write | 5 | 114.8 ± 2.8 | 111.8 / 119.9 | 88.4 | PASS |
| 32x8 | 10T | off | write | 5 | 86.6 ± 1.6 | 84.9 / 89.4 | 65.3 | PASS |
| 16x16 | 10T | on | read | 5 | 330.9 ± 3.8 | 325.1 / 335.3 | 40.8 | PASS |
| 32x8 | 10T | on | read | 5 | 342.3 ± 4.4 | 336.1 / 347.8 | 41.3 | PASS |
| 16x16 | 10T | on | write | 5 | 120.9 ± 3.4 | 117.4 / 127.4 | 85.8 | PASS |
| 32x8 | 10T | on | write | 5 | 88.3 ± 1.8 | 86.4 / 91.4 | 64.4 | PASS |
| 16x16 | 6T | off | read | 5 | 341.6 ± 4.0 | 335.4 / 346.4 | 44.3 | PASS |
| 32x8 | 6T | off | read | 5 | 341.6 ± 4.3 | 335.1 / 347.4 | 43.2 | PASS |
| 16x16 | 6T | off | write | 5 | 102.5 ± 3.0 | 99.5 / 108.3 | 88.5 | PASS |
| 32x8 | 6T | off | write | 5 | 74.8 ± 2.0 | 72.9 / 78.4 | 64.9 | PASS |
| 16x16 | 6T | on | read | 5 | 324.6 ± 3.5 | 319.1 / 329.0 | 40.5 | PASS |
| 32x8 | 6T | on | read | 5 | 331.5 ± 4.5 | 324.7 / 337.2 | 41.1 | PASS |
| 16x16 | 6T | on | write | 5 | 110.4 ± 3.9 | 106.7 / 118.0 | 85.4 | PASS |
| 32x8 | 6T | on | write | 5 | 75.8 ± 1.7 | 74.1 / 78.9 | 63.6 | PASS |

#### Equivalent-circuit model (`real_cell_mode=1`, `w_rc=True`, `main_sram.py` defaults)

| array | cell | op | all cells real, no RC: delay [ps] / PAVG [µW] | real_cell_mode=1, w_rc=True: delay [ps] / PAVG [µW] | waveform checks |
|---|---|---|---|---|---|
| 16x16 | 10T | read | 344 / 45.3 | 554 / 94.6 | PASS |
| 16x16 | 10T | write | 111 / 88.8 | 336 / 158.5 | PASS |
| 16x16 | 6T | read | 337 / 44.9 | 513 / 93.9 | PASS |
| 16x16 | 6T | write | 98 / 89.1 | 334 / 165.7 | PASS |
| 32x32 | 10T | read | 396 / 81.9 | 770 / 231.6 | PASS |
| 32x32 | 10T | write | 148 / 227.4 | 340 / 404.1 | PASS |
| 32x32 | 6T | read | 386 / 81.1 | 695 / 229.7 | PASS |
| 32x32 | 6T | write | 137 / 227.8 | 334 / 414.5 | PASS |
| 64x64 | 10T | read | 505 / 210.2 | 1188 / 701.6 | PASS |
| 64x64 | 10T | write | 173 / 613.0 | 305 / 1137.0 | PASS |
| 64x64 | 6T | read | 492 / 202.9 | 1051 / 696.4 | PASS |
| 64x64 | 6T | write | 162 / 610.8 | 297 / 1140.6 | PASS |
