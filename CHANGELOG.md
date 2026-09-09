# Changelog

The full evidence tables of V2.0.1 and V2.0.2 (294 size-sweep decks with
delay and power per configuration, 108 corner runs, the address-change and
clock-period sweeps, seeded Monte Carlo) were removed from this file in V2.0.3.
They are in the git history (`git show c3f6f44:CHANGELOG.md`) and in
`CIRCUIT_REVIEW.md` Parts II and III; the condensed numbers below are copied
from them unchanged.

## V2.0.5 — 2026-09-08 — full local mismatch and driver re-evaluation

In progress; no release qualification is claimed. The full working plan remains
in `DRIVER_SIZING_PROPOSAL.md`.

- The MC testbench defaults to independent per-MOS `vth0`, `u0`, and `voff`
  mismatch at a fixed global corner. Nominal and shared-card variation are
  explicit alternatives; the per-device CLI defaults to full transistor arrays.
- Canonical specialization preserves connectivity, widths, sweep expressions
  and `NF`, with immutable model artifacts and per-device CSV audits. Combined
  local sampling and legacy `.STEP` sweeps fail explicitly after a Xyce check
  showed that only the first geometry was executed.
- All five corners now have local read and write checks. Dedicated write-tail
  ensembles, fast-corner hazards, separate diagnostic screening, execution
  provenance and stricter qualification/report gates are included.
- The finalized 8x4 pilot passes all 12 nominal calibration decks and 120 local
  waveform samples for 6T/10T and mux on/off. Coefficients remain provisional.
- The three-sample representative-array screen (8x4, 16x16, 64x64, 256x8,
  16x256; both cells; mux on/off; 16x16 with RC) completed: 759 local samples
  scored, 723 pass every check, 36 fail only the wordline-phase budget on the
  16x16 explicit-RC reads (the reads themselves are functional; the wordline
  rule has no RC term). 28 four-rank cases and the 10T 64x64 / 16x256
  calibrations timed out at 1,800 s on a shared host, and three materialized
  MPI samples failed the DC operating point; these are unscored, not failed.
  See the Stage C outcome in `DRIVER_SIZING_PROPOSAL.md`.
- RC symmetry between the array and the replica column: with `w_rc` the
  fixed-mode replica wordline driver (AND2) now carries the same two RC
  segments on its inputs and RWL output as the real wordline drivers, and the
  replica bitline reaches the timing block through the same two segments a
  real bitline sees at its sense-amplifier input. Cells, dummy loads,
  precharge and write-driver stubs were already identical on both sides; the
  equivalent-cell modes aggregate the same stubs. Non-RC decks are unchanged.
  Measured effect (8x4 6T, TT, fixed mode, RC read): the replica wordline led
  the real wordline by 36 ps before and 13 ps after (rules_only: 1 ps), so
  the read access rises from 392 to 434 ps; the write access is unchanged.
  The wordline model with RC is the driver's two output segments plus one
  stub per cell pin (a star, no series segments between columns).
- Review fixes: the optimizer objective (`exp_utils.evaluate_sram`) requests
  the nominal corner explicitly instead of inheriting one unseeded per-device
  sample; `local_review` records calibration execution errors instead of
  labelling them waveform failures. Twenty fixed-mode decks are byte-identical
  to the previous release; default-mode 8x4 Xyce runs and the per-device CLI
  pass.
- Large-array runs use MPI cores with saved numeric local LHS model cards,
  bypassing a reproduced Xyce 7.4 MPI random-expression initialization crash.
  Four-core read/write checks pass; a 64x64 full-device waveform check passes
  in 854 seconds. MPI/serial timing and power measures match for an identical
  small-array sample. Worker/rank budgets and timeout cleanup are explicit.

## V2.0.4 — 2026-09-07 — baseline driver sizing and measured timing

Qualification is in progress. The full working proposal and completed/open
checklist are in `DRIVER_SIZING_PROPOSAL.md`; its original text and supplied
characterisation CSV are preserved.

### Changes

- Added immutable baseline driver sizing with separate write-input/output
  scales, load-based precharge and wordline rules, configurable replica K/N,
  decoder output scaling, and exact baseline/physical-context table lookup.
  `fixed` remains the default; `rules_only` and unmatched `auto` results are
  explicitly unverified.
- Matched real and replica wordline drivers and gate loads, froze replica
  devices across cell candidates, and included disabled write-stack loading
  on real and replica bitlines in canonical read decks.
- Corrected control-buffer loading and tapering. Wide inverter gates use
  BSIM4 `NF` with at most 2 um per finger, preserving total width and the PDK
  gate-resistance model. This closes measured wide-array control-edge failures.
- Included explicit RC enable-pin loads, removed a floating dummy-cell RC
  branch, and delayed RC precharge until the matched physical wordline is low.
- Added measured clock derivation from SS read and SS/SF write phases, with
  25% margin and upward 50 ps quantisation. The period remains frozen during
  candidate and variation evaluation.
- Added resumable Xyce qualification, waveform/phase checks, shared and local
  mismatch cases, eight-cycle read/write sequences, SA offset ensembles, and
  evidence-bound report/table export. Failed or incomplete evidence cannot
  produce qualified table records.
- Integrated baseline reuse and resolved-width area estimates into the
  optimizer adapter, corrected its actual array geometry, and report original
  access-limit violations. Per-device MC records sizing provenance; both
  simulation entry paths select Xyce explicitly.
- Added `AGENTS.md`, sizing documentation, and focused regression coverage.

### Qualification and remaining work

- The full campaign covers 112 physical configurations: both cell types,
  27 historical sizes and valid mux choices, RC sensitivity, and 32x1 optimizer
  cases. It schedules 1,786 decks and 6,538 waveform samples. Completion and
  table promotion are pending; see the proposal for current status.
- Folded-driver 16x256 and 16x512 TT read pilots pass all waveform checks.
  Independent gate-charge/fingering evidence is in `docs/qualification/`.
- Four 100-copy SA/latch offset ensembles pass at SS and cold FF, including
  mid-common-mode and precharged-bitline input profiles. Full local sensing
  qualification awaits the replica/cell ensembles.
- Original 200 ps read / 100 ps write constraints remain separate from phase
  qualification; default K=1/N=9 does not claim compliance with the read limit.
  Peripheral sweep ranges, equivalent-cell approximations, and the obsolete
  rare-event `main_estimation.py` backend/API migration remain open.

## V2.0.3 — 2026-09-06 — changelog compacted, automatic timing configuration proposed

Scope: documentation and characterisation only. No compiler, testbench or
algorithm code was changed.

### Changes

- **`CHANGELOG.md` compacted:** the V2.0.1 and V2.0.2 entries shrink from
  900 to 300 lines and keep their fixes, open items, code changes and observations; the
  evidence is condensed to representative sizes and one corner table. The
  full tables are in `git show c3f6f44:CHANGELOG.md`.
- **`TIMING_AUTOCONFIG.md`** (new): proposal for setting the SRAM timing
  automatically for every array size. Findings it rests on:
  - the only free timing knob of this architecture is the clock period (and
    duty): wordline enable and write enable are the clock-low phase,
    precharge the clock-high phase, sense enable is replica-timed and the
    output / data / address latch enables are derived from `s_en`, `w_en`
    and `wl_en`;
  - the phases the flow measures (`TCLK_WLEN + access` and
    `max(TRESTORE, TCLK_DEC)`) follow `a + b*rows + c*log2(cols)` for reads
    and `a + b*rows + c*cols + d*log2(rows)` for the write restore within
    5-15 ps rms / < 40 ps worst over the 27 sizes of the V2.0.2 sweep;
  - proposed rule `T = 2 * max(low_wc, high_wc) * (1 + margin)`, phases
    derated to the worst-case PVT with one measured factor per phase,
    default margin 25 %, quantised to 50 ps, anchored by a table of
    characterised sizes with a two-deck self-calibration for new sizes, and
    frozen per array size as the spec for optimisation and yield analysis.
  Not implemented; sections 5-8 of the document list the integration points,
  the validation plan, the effort and the decisions needed.

### Evidence (new, nominal devices unless stated, 6T, mux off, 10 ns clock)

- **Worst-case PVT characterisation** (scratch harness of V2.0.2 with a new
  `--vdd` switch): read and write decks at SS, SF, FS and FF x 125 C x
  1.0 V, SS and SF x 125 C x 0.9 V and TT / 25 C / 0.9 V on 8x4 and 16x16
  (6T and 10T), 64x16, 256x8, 16x256, 64x64 (SS, 1.0 V) and 512x4 (read);
  75 decks, all completed. Relative to TT / 25 C / 1.0 V every control phase and the read
  access are 2.13-2.31x slower at SS / 125 C / 0.9 V, with the same factor
  on every size run and both cells (the buffer taper and the bitline terms
  scale together); the supply step to 0.9 V costs 1.13x alone. The write
  access is worst at SF: 3.8x at 8x4 (0.5x write driver), 2.4x at 16x16.
  Every deck passes the waveform checks except the 256x8 and 64x16 reads at
  FF and FS / 125 C, where the un-selected bitline
  leaks from 1.00 to 0.69-0.73 V during the 5 ns wordline phase of the 10 ns
  clock (it is at 0.99 V when the amplifier fires; the read is correct).
  The same 256x8 FF / 125 C deck re-run with the period the proposal would
  assign (2.6 ns) keeps the bitline above 0.97 V and passes: the droop is a
  property of the oversized wordline phase, not of the array.
- **Period sweeps at the worst case** confirm the minimum-period formula
  within one 100 ps step: 8x4 read at SS / 125 C / 0.9 V passes at 1.9 ns,
  fails at 1.8 (measured `2 * low` = 1.81 ns); 8x4 write at SF / 125 C /
  0.9 V passes at 1.5, fails at 1.4 (1.44 ns); 16x16 read at SS / 125 C /
  0.9 V passes at 2.1, functionally at the limit at 2.0 (1.95 ns: `OUT`
  crosses VDD/2 10 ps after the clock edge).
- **Seeded Monte Carlo at the worst case** (`vth_std = 0.05`, 5 samples,
  seed 2026): 8x4 and 16x16 reads at the proposed periods (2.35 / 2.5 ns,
  25 % margin) pass every sample with 2.7 % sigma on the limiting phase and
  > 210 ps slack; the 8x4 write at SS / 125 C / 0.9 V and the 16x16 write at
  SF / 125 C / 0.9 V pass every sample (10 % and 6 % sigma on the write
  access).

### Left open

- **8-row write-ability at SF / 125 C / 0.9 V.** 2 of 5 Monte Carlo samples
  of the 8x4 write cannot write at all with the 10 ns clock (BLB stays at
  0.26 V, the cell keeps its data) and a third needs 2.5 ns: the 0.5x write
  driver that `WriteDriverFactory.width_scale` gives to arrays with <= 8
  rows has no margin against a strong-PMOS / weak-NMOS sample at that
  corner. Nominal decks pass (write access 499 ps). Not changed in this
  release; a yield analysis at that corner will report it.
- **Automatic timing is proposed, not implemented**; the 10 ns default period
  is unchanged.
- **Not characterised at the worst case:** 128x128 and 256x64, the column
  mux, 10T beyond 16x16, and the equivalent-circuit / `w_rc` variants; the
  phase model is fitted at TT / 25 C and derated, it has not been re-fitted
  on worst-case data (the table in `TIMING_AUTOCONFIG.md` section 3.4 is the
  anchor for the seven characterised sizes).
- **Testbench prerequisites for short periods** (unchanged, from V2.0.2):
  the PSTC window overlaps the start-up precharge for `t_period < 5 ns`;
  the scratch harness scores `OUT` inside the wordline phase, which is
  50 ps stricter than the functional limit at 16x16.
- Evidence of this entry: `TIMING_AUTOCONFIG_data.csv` (614 runs: the
  V2.0.2 sweeps with their phase measures, the worst-case decks and the
  validation decks).

## V2.0.2 — 2026-09-05 — periphery fan-out, address hold, precharge control, timing configurations

Scope: the open items of V2.0.1 (address-path hold hazard, cycle-time
dependence of the floating bitlines, `s_en` buffer sizing) plus a review of
every control buffer of the timing block and of the wordline driver against
its actual fan-out. Validated in Xyce 7.4 with automatic waveform scoring over
array sizes 1x1 to 512x4 / 16x512, clock periods 0.6-100 ns, the five process
corners at -40 to 125 C, and seeded Monte Carlo; "before" numbers come from a
detached worktree of V2.0.1. Optimisation and yield-estimation algorithms
were not touched.

### Fixes

- **Address-path hold hazard confirmed and closed.** With a changed address
  the new row's wordline reached 0.41-0.50 V at 256 rows (neighbouring cell
  Q dipped to 0.86 V) and full VDD at 512 rows, where the cell in the new row
  was overwritten (read deck: stored 1 -> 0; write deck: 0 -> 1). A
  transparent-low hold latch on the address register output (enabled by
  `wl_en_bar`, same scheme as the V2.0.1 write-data latch) keeps the decoder
  input constant while a wordline is on; the new decoder output now rises at
  least five gate delays after the old wordline is off and no second wordline
  (< 10 mV) is seen at any size. Flipping the LSB never glitched (largest
  decoder fan-out, slowest path); hazard tests must flip a *middle* bit.
- **Control buffers sized for their fan-out.** `wl_en` (600 ps fall at 512
  rows), the address register output (650 ps edge at 512 rows), `s_en`
  (190-230 ps edge and a 0.2-0.3 V precharge-coupling bump at >= 64
  columns), `w_en` (130-180 ps edge from 64x16 up) and `PRE` (fan-out ~49,
  only reached 0.05-0.08 V on the largest arrays) are driven by geometrically
  tapered buffers with a fan-out of ~8 per stage (`TaperedBuffer`: 2 stages
  up to a scale of 16, 4 above); the wordline driver's NAND2 scales with the
  square root of its inverter scale. Every buffered control edge is now
  20-40 ps (10-90 %) independent of the array size.
- **Sense amplifier isolated during writes.** Its input pass gates were on
  whenever `s_en` was low, so the cross-coupled PMOS pair was a bitline keeper
  during writes and a write only worked while `w_en` rose before the wordline
  (60 ps margin at 2x128 in V2.0.1; with the faster wordline path of this
  release the 2x128 write deadlocked at BL 0.27 V / BLB 0.9 V). `SENSEAMP`
  has a new `ISO` pin driven by `sa_iso = s_en | w_en` from TIME.
- **Precharge for the whole clock-high phase.** The ~300 ps self-timed pulse
  left the bitlines floating; with the 10 ns clock the replica bitline had
  drooped to 0.74 V at FF / 125 C and to 0.76 V at TT / 25 C with a 100 ns
  clock. `PRE = NAND3(clk_buf, cs, wl_en_bar)` holds the bitlines at VDD
  until 40-70 ps before the wordline rises; they are at 1.000 V at every
  access for 0.6-100 ns periods and at every corner.
- **Honest minimum-period estimate.** The printed `CLK(min)` was
  `2 * (access delay + 0.1 ns)` and claimed 0.48-0.80 ns for an 8x4 array
  whose read deck fails below 0.9 ns. Three new measures (`TCLK_WLEN`: clock
  -> `wl_en`; `TCLK_DEC`: capture edge -> decoder output; `TRESTORE`: end of
  access -> bitline back at 0.9 VDD) give
  `T_min = 2 * max(TCLK_WLEN + access, TRESTORE, TCLK_DEC) * 1.1`
  (`_print_min_period()`), which matches the period sweeps within one step.
- **Testbench stimulus for the address path.** `next_row=<row>` makes the
  register capture another row at the edge that ends the access and prints
  that row's wordline and cell (the address was constant in every V2.0.1
  deck, which is why the hazard was never exercised).
- **PSTC caveat.** A warning is printed for `t_period < 5 ns`, where the
  quiescent window overlaps the start-up precharge.

### Left open

- **128x128 and 256x64** were run only for 6T without mux with a 4 ns clock
  (read 475 / 568 ps, write 155 / 149 ps, ~3 h per deck, all checks pass);
  the 10T / mux / read&write decks of those sizes and the two 10T 16x512
  `read&write` decks (> 10 h) were not run.
- **Energy cost of the fixes.** 6T read PAVG at 100 MHz: 8x4 +5 %, 16x16
  +11 %, 32x32 +17 %, 64x64 +14 % (largest term: the wordline-driver NAND2
  taper, 115 fF instead of 29 fF on the `wl_en` / decoder nets at 64x64;
  PSTC +20-30 % because the bitline leakage is now supplied through the
  precharge devices). Writes are cheaper on every array >= 32x32 (-12..-24 %)
  because the driver no longer fights the sense-amplifier keeper. The buffers
  follow one fan-out rule and were not power-tuned.
- **Delay reference at >= 256 rows.** `TREAD_TOTAL` / `TWRITE_TOTAL` are
  measured from the `wl_en` crossing, which the re-sized buffer moves 70-150
  ps earlier at 256-512 rows, so those tabulated delays grow (512x4 6T read
  709 -> 739 ps) while clock-to-output shrinks (1010 -> 890 ps). The
  clock-referenced value is `TCLK_WLEN + TREAD_TOTAL`.
- **Design choices verified again and kept:** replica-timed full-swing
  sensing (read delay ~300 ps up to 32 rows); `w_rc=True` default of
  `main_sram.py`; `read` reads a stored 0; the hazard / coupling checks live
  only in the scratch harness, not in the flow's `.MEASURE` set.
- **Out of scope:** optimisation and yield-estimation algorithms, SNM beyond
  the V2.0.1 sanity run, `sweep_*` modes.

### Code changes

- `time_generate.py`: `ADDR_DFF` -> `D_LATCH_ADDR` (EN = `wl_en_bar`) ->
  `TaperedBuffer` `ABUF` per address bit; `wl_pdrive` scales both stages with
  `ceil(rows * nand_scale / 32)`; `s_en` drives only footers and output
  latch (`SEN_BUF` above 32 unit loads); new `sa_iso = NOR2(s_en, w_en)` +
  inverter (+ `ISO_BUF`); `w_en` = `AND2_WEN` (+ `WEN_BUF` above 32 unit
  loads); `PRE_UNBUF = NAND3(clk_buf, cs, wl_en_bar)` + `PRE_BUF`. New
  helpers `TaperedBuffer`, `D_latch_addr`, `PNOR2`; `TIME` / `TIMEFactory`
  take `num_sa`, `wl_load`, `pre_load`, `wen_load` (defaults reproduce the
  base YAML sizes) and export `sa_iso`.
- `wordline_driver.py`: static `WordlineDriverFactory.inv_scale / nand_scale`
  (`nand_scale = sqrt(inv_scale)`, also emitted in sweep mode);
  `PrechargeFactory.width_scale`, `WriteDriverFactory.width_scale`.
- `mux_and_sa.py`: `SENSEAMP` `ISO` pin.
- `sram_6t_core_MC_testbench.py`: `TCLK_WLEN`, `TCLK_DEC`, `TRESTORE`
  measures; `_print_min_period()` replaces the `1/2CLK` print (mean + one
  standard deviation per term for `mc_runs > 1`); PSTC warning.
- `sram_6t_core_testbench.py`: `next_row`; `create_time_circuit()` passes the
  fan-out information; `create_write_periphery()` sizes the `w_en_bar`
  inverter with the column count (`_wenb_scale()`).
- `readme_compiler.md` sections 10, 11, 13.1, 14.5; `CIRCUIT_REVIEW.md`
  Part III (defects D12-D20, sizing tables).

### Observations (verified, not changed)

- At TT / 125 C every control-path delay is 1.65-1.8x its 25 C value (16x16
  6T read 308 -> 536 ps, `TRESTORE` 257 -> 460 ps); SS / -40 C is the
  fastest condition in these models (8x4 6T read 226 ps vs 302 at TT / 25 C);
  SF is the slowest write corner (8x4 6T write 174 ps vs 131). The 10 ns
  clock leaves > 8 ns of margin everywhere; the estimated minimum period at
  TT / 125 C is ~1.7 ns for 16x16.
- With the isolation pin the amplifier's pass gates open 50-70 ps after the
  footer fires, so it regenerates while still connected to the full-swing
  bitlines; reads are 8-25 ps faster than in V2.0.1.
- `read&write` toggles `we` every cycle; `s_en` shows no glitch (< 1 mV) at
  the write-to-read transitions because `gated_clk_bar` falls before `we_bar`
  rises.

### Evidence (condensed)

Nominal, all cells real, `w_rc=False`, TT, 25 C, 10 ns clock, target cell =
last row / last column. Read delay = `wl_en` rise -> `OUT`; write delay =
`wl_en` rise -> Q at 90 %; PAVG at 100 MHz. V2.0.1 values in parentheses.

**Size sweep:** 292 of 294 configurations (27 sizes x {6T, 10T} x {mux off,
on} x {read, write, read&write}) completed and pass every waveform check;
only the two 10T 16x512 `read&write` decks are missing (10 h job limit).
Every `read&write` deck shows the correct 40 ns `OUT` period. Mux on: reads
0-16 ps faster, writes 0-19 ps slower than the values below.

| array (mux off) | 6T read [ps] | 6T write [ps] | 10T read [ps] | 10T write [ps] | 6T PAVG read / write [uW] |
|---|---|---|---|---|---|
| 1x1 | 283 (286) | 119 (131) | 286 (289) | 119 (133) | 20.9 / 22.6 |
| 4x4 | 287 (298) | 126 (133) | 291 (301) | 127 (138) | 26.8 / 36.8 |
| 8x4 | 291 (301) | 131 (138) | 295 (305) | 131 (142) | 29.3 / 39.6 |
| 16x16 | 308 (337) | 94 (98) | 314 (344) | 107 (111) | 50.0 / 93.7 |
| 32x32 | 332 (386) | 100 (137) | 344 (396) | 112 (148) | 94.6 / 201.4 |
| 64x16 | 358 (382) | 93 (114) | 378 (401) | 105 (125) | 89.9 / 155.0 |
| 64x64 | 382 (492) | 108 (162) | 408 (505) | 120 (173) | 231.7 / 504.8 |
| 2x128 | 322 (458) | 176 (191) | 326 (463) | 169 (202) | 143.6 / 560.3 |
| 128x32 | 424 (468) | 104 (135) | 460 (510) | 116 (146) | 218.6 / 416.7 |
| 16x256 | 354 (535) | 133 (261) | 361 (542) | 146 (273) | 369.4 / 1252.0 |
| 256x8 | 524 (526) | 96 (68) | 592 (579) | 105 (78) | 149.4 / 223.3 |
| 8x512 | 350 (640) | 202 (509) | 355 (642) | 197 (512) | 590.5 / 3277.7 |
| 16x512 | 360 (650) | 158 (460) | 369 (656) | 170 (472) | 729.8 / 3378.0 |
| 512x4 | 739 (709) | 95 (70) | 869 (791) | 105 (80) | 171.3 / 234.9 |
| 128x128 (4 ns clock) | 475 | 155 | - | - | 1678 / 3988 |
| 256x64 (4 ns clock) | 568 | 149 | - | - | 1636 / 3464 |

**Address change (`next_row`, middle bit flipped):** V2.0.1 circuit: next
row's wordline 0.41-0.50 V during the hold at 256x8 (read, write, 10T write)
and 1.00-1.01 V at 512x4 with the neighbouring cell overwritten; V2.0.2:
< 10 mV and the cell keeps its data at 8x4, 64x16, 128x32, 256x8 and 512x4,
read and write. LSB flips never glitched on either version.

**Clock period sweep (6T, mux off):** 8x4 read passes at 0.9-100 ns and fails
at 0.8 ns (estimate 0.90 ns); 16x16 read passes at 0.9 ns and fails at 0.8
(estimate 0.97); 8x4 / 16x16 write pass down to 0.6 ns (estimates 0.55);
64x16 read passes at 2 ns (estimate 1.08); 10T 8x4 read passes at 1.0, fails
at 0.8 (estimate 0.91). Delays are unchanged across periods; the bitlines are
at 1.000 V at every access for all periods.

**Process corners and temperature (6T, 10 ns clock, all checks pass, bitlines
1.000 V at every access):**

| condition | 8x4 read [ps] | 8x4 write [ps] | 16x16 read [ps] | 16x16 write [ps] | 16x16 PAVG read / PSTC [uW] |
|---|---|---|---|---|---|
| TT -40 C | 203 | 83 | 214 | 62 | 48.0 / 1.5 |
| SS -40 C | 218 | 86 | 231 | 66 | 46.2 / 0.8 |
| FF 25 C | 265 | 123 | 281 | 87 | 55.2 / 6.9 |
| FS 25 C | 290 | 115 | 306 | 88 | 50.6 / 4.1 |
| TT 25 C | 291 | 131 | 308 | 94 | 50.0 / 3.3 |
| SF 25 C | 295 | 174 | 312 | 104 | 51.1 / 4.5 |
| SS 25 C | 321 | 140 | 341 | 104 | 47.1 / 1.7 |
| TT 85 C | 411 | 206 | 435 | 139 | 54.8 / 8.0 |
| FF 125 C | 451 | 244 | 478 | 156 | 74.0 / 26.0 |
| TT 125 C | 507 | 266 | 536 | 174 | 60.7 / 13.6 |

10T values are within +4 to +20 ps of the 6T ones at every condition.

**Monte Carlo (Xyce `.SAMPLING`, `vth_std = 0.05`, seed 2026):** 5-sample
read / write decks at 8x4, 16x16, 32x8 and 64x16 (both cells, mux off / on)
and 3-sample `read&write` decks at 8x4 and 16x16 all pass. Read delay
standard deviation 3.4-5.0 ps (1.1-1.6 %); write 1.6-3.9 ps at >= 16 rows,
5-12 ps at 8x4 (up to 7 % with mux). The 64x16 address-change decks pass
for all samples.

## V2.0.1 — 2026-09-05 — transient / Monte Carlo circuit review

Scope: the SRAM compiler circuits (6T and 10T cores, replica column, timing
generator, decoder, wordline driver, precharge, column mux, sense amplifier,
write driver, output latch), the transient testbenches (`read`, `write`,
`read&write`), their measurements, and the Xyce simulation / result-parsing
flow. Every change was validated by running the generated netlists in Xyce
7.4 and scoring the waveforms automatically for 29 array sizes from 1x1 to
512x4 and 16x512, both cells, mux on and off, all three operations, nominal
and seeded Monte Carlo (283 completed decks, all pass). Optimisation and
yield-estimation algorithms were not reviewed.

### Fixes

- **Write pulse too short.** `w_en` was cut by the replica bitline (~250 ps),
  a cell-strength path, while the write path (row-scaled write driver, through
  the column mux) is weaker: seeded 5 % sigma Monte Carlo samples left the
  bitline at 0.3-0.4 V and the cell kept its old data. `w_en = gated_clk_bar
  & we` now spans the wordline phase (new `AND2_WEN` gate with its own
  subcircuit name); the hard-coded 16x512 `WenDelayChain` hack is removed.
- **Hold hazard introduced by that fix, caught by the sweep.** At the clock
  edge that ends a write the data register updates 100-150 ps before the
  drivers release, so the next cycle's data was briefly written; at 64 rows
  this flipped the freshly written cell (`read&write` 64x16 failed). A
  per-column transparent-low write-data hold latch (`D_LATCH`, EN =
  `w_en_bar`) fixes it.
- **Write testbench topology.** The `write` deck had no bitline precharge and
  no sense-amp / mux load (bitlines started from the artificial `.IC` state),
  so the stand-alone write delay was 30-40 % optimistic against the same write
  inside `read&write`. All transient decks now carry the full column
  periphery; a write cycle is precharge -> write -> precharge; `TWDRV` is
  measured on the driven bitline (`w_en` rise -> BLB at VDD/2).
- **Nominal runs were random samples.** Every deck emitted `.SAMPLING` with a
  random seed, so identical calls returned different delays and occasionally
  failed. `mc_runs=1` is now deterministic (no `.SAMPLING`, every `AGAUSS`
  at its mean); `mc_seed` makes Monte Carlo sweeps reproducible; the Xyce
  console log is kept per run as `<netlist>.log`.
- **Energy window** measured the start-up charging of the bitlines from 0 V
  (half of the "read energy" on 8x4) instead of a steady-state cycle; it is
  now one full clock period starting at the first access (`1 ns + 0.7 T` ..
  `1 ns + 1.7 T`). `read&write` averages over one 4-cycle pattern, its
  transient runs to `1 ns + 8.5 T`, and it now produces PSTC / PDYN too.
- **Measurement details.** `TS_EN` was corrupted by a precharge-coupling bump
  on `s_en` (now measured from the access phase); the `w_en` buffer was not
  scaled with the row-dependent write-driver size (release lagged the
  wordline by 40-285 ps on 64-256-row arrays); the 10T core ignored the
  testbench RC parameters; PySpice keeps one subcircuit definition per name
  and scope, so a gate class instantiated twice with different sizes silently
  kept the last one (avoided with dedicated class names).
- **Xyce Newton stall on some 512-row decks** (residual 1e-12 A at every step
  size, not a circuit fault): the flow retries once with a 20 ps maximum time
  step, which keeps results of converging decks within 0.5 %; `t_max_step`
  and `xyce_options` are exposed on `Sram6TCoreMcTestbench`.
- **Static-review fixes carried into this release** (details in
  `CIRCUIT_REVIEW.md` Part I): column-mux port mismatch that aborted every
  muxed read; free-running `SEL` pulse; wrong output-latch index and floating
  latch input on writes; replica column driven by the real wordlines (now one
  active replica cell, dummies tied to VSS); CS start-up clamp fighting the
  flip-flop; negative read delay and negative dynamic power from mis-placed
  measure thresholds and windows (read delay is `TREAD_TOTAL`, `wl_en` ->
  output latch; write delay `TWRITE_TOTAL`, `wl_en` -> Q at 90 %; PSTC in a
  quiescent window `1 ns + [0.4, 0.65] T`); write delay over-reported 2.2x by
  summing overlapping segments; `FAILED` measures silently becoming 0.0
  (now raise); `.prn` / SNM parsing that depended on `.PRINT` ordering;
  write-SNM taken as a global maximum; equivalent-circuit caps only inserted
  with `w_rc`; stimulus sources at a literal 1.0 V instead of `vdd`;
  `choose_columnmux` with `num_cols % mux_in != 0` rejected; `sweep_senseamp`
  no longer defaults to `True`; `python-graphviz` -> `graphviz`; duplicated
  `config.py`; µW label; `main_sram.py` paths.

### Left open

- **Array sizes not finished at release time:** 128x128 and 256x64 (~11 h per
  read deck at 10 ns) and the `read&write` decks of 8x512, 16x512, 16x256 and
  10T 512x4 (finished in V2.0.2).
- **Address-path hold hazard** (pre-existing, not exercised because the
  testbenches kept the address constant): fixed in V2.0.2.
- **Design choices verified and left as they are:** sensing waits for a fully
  discharged replica bitline plus a 9-stage delay chain, so the read delay is
  ~300 ps for every size up to 32 rows (the target bitline is at ~0.02 V when
  the amplifier fires; `vswing` = 250 mV is reached after 10-35 ps); the
  precharge was a ~300 ps self-timed pulse after which the bitlines floated
  and drooped to ~0.93 V (changed in V2.0.2); the `s_en` buffer kept the
  columns/64 scaling (changed in V2.0.2); the `w_rc=True` default of
  `main_sram.py` puts 1 fF on every cell's Q/QB and triples the write delay
  (16x16 6T: 98 -> 346 ps); the fixed 10 ns clock leaves > 4 ns of margin for
  every size tested; `read` always reads a stored 0.
- **Out of scope:** optimisation and yield-estimation algorithms, SNM beyond
  a sanity run (6T hold / read / write 0.325 / 0.182 / 0.365 V, 10T 0.485 /
  0.290 / 0.419 V), `sweep_*` modes.

### Evidence (condensed)

- **Size sweep:** 283 of 318 planned decks completed, all pass; the per-size
  delays are the values in parentheses of the V2.0.2 table above.
- **Monte Carlo (seed 2026, 5 samples):** 8x4, 16x16 and 32x8, both cells,
  mux off / on, read and write, all samples pass; read standard deviation
  3.5-4.4 ps, write 1.6-3.9 ps at 16-32 rows and 7.5-20.6 ps at 8x4.
- **Equivalent-circuit model** (`real_cell_mode=1`, `w_rc=True`,
  `main_sram.py` defaults) tracks the all-real array within 6-12 % on delay
  and 5 % on power at 16x16 with the same RC model; against the no-RC all-real
  runs the delays are 1.5-2.2x (6T read 337 / 386 / 492 ps -> 513 / 695 /
  1051 ps at 16x16 / 32x32 / 64x64; write 98 / 137 / 162 -> 334 / 334 / 297
  ps), most of which is the `w_rc` cell loading.
- **Solver:** a `.SAMPLING` run of the old 10T 64x16 mux netlist aborted with
  "time step too small"; the corrected topology completes (139 ps).
