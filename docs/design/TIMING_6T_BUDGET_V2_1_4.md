# V2.1.4 6T timing: write-request hold latch, 6T ladders and evidence run

Executed September 15, 2026 on the V2.1.4 sources (base commit
`9b785f8f` plus the TIME, table, test, tool and documentation changes recorded
in the [changelog](../CHANGELOG.md); every queue archives the exact sources it
ran). Illustrative wires, V2.0.9 driver classes, nominal seed 20260913,
maximum step 20 ps; per-device samples on one rank each with 5 % relative
sigma on `vth0`, `u0` and `voff` of every MOS. This closes the 6T items of the
V2.1.3 open items (`docs/plans/V2_1_3_OPEN_ITEMS.md` at `945815a`) (1 to 3); the 10T items
(4 and 5) are the next round. It is functional screening at SS 0.9 V / 125 °C
(SF for writes, FF 1.1 V / −40 °C for the control race), not PVT/mismatch or
extracted-metal qualification; no record is promoted to `sizing_table.json`.

**Outcome: on the final sources 92 of 92 attempts pass, 305,371 of 305,371 checks: 21 nominal class-bound cases (every 6T row and column bound with and without a mux, plus SF writes at 64x16 and 512x4), the 10-case write-waveform gate and 61 per-device seeds. Every nominal class bound keeps at least 250 ps between the read output and the 1.2 T deadline (257 ps at the 512x4 mux read, 298 ps at 128x8 mux, 299 ps at 256x4), every seed at least 264 ps, and every boundary write-enable peak is at most 17 mV. The run exposed a second bug, an analysis stop off the output grid at the new 4.75 ns class (section 6): 22 first-run attempts were rerun on the fixed sources. One seed needed the runtime's single DC-operating-point retry; no run hit a numerical failure, and 8 printed the non-fatal minimum-step warning. The probe run on the V2.1.3 table (11 of 14 attempts pass) is the evidence against the shared classes.**

"Request→OUT" is the local read output reaching 90 % (read 1) or 10 % (read 0)
of VDD, measured from the access request at 0.65 T, worst read cycle; "deadline
margin" is its distance to the 1.2 T data deadline. The validator samples the
output 0.015 T before the deadline, so a margin below about 60 to 70 ps fails
its output check. "Boundary W_EN / S_EN peak" is the largest local write
enable in a read cycle and sense enable in a write cycle, 0.65 T to 1.65 T,
at the driver and amplifier terminals (the new quiet checks). Measurements come
from the retained `.prn` traces through `dev/v213_sense_timing.py` and
`dev/v214_boundary_enable.py`.

## 1. Write-enable spike at the read-to-write boundary (open item 2)

**Cause.** TIME registers the write request (`we`, DFF_BUF on `clk_buf`) on
the rising clock edge that also ends the access, and gated both access
enables with the raw register output: `w_en = AND2(request, we)` and
`s_en = AND3(rbl_delay, request, we_bar)`. The register output moves before
the access request falls. From CLK_BUF 50 % in the retained traces:

| Corner, samples | WE 50 % | Access request 50 % | WL_EN 50 % | W_EN peak at the read→write edge |
|---|---:|---:|---:|---:|
| FF 1.1 V / −40 °C, 8x4 nominal (V2.1.1) | 21 ps | 37 ps | 59 ps | 0.41 V |
| FF 1.1 V / −40 °C, 8x4 seeds 20261001–20261010 (V2.1.2, V2.1.3) | 20–21 ps | 37–39 ps | 58–61 ps | 0.38–0.72 V |
| SS 0.9 V / 125 °C, 8x4 and 16x16 mux (V2.1.2, V2.1.3) | 70–88 ps | 132–141 ps | 208–219 ps | 1 mV |

For about 17 ps at FF −40 °C (50 ps at SS, where the slower gates filter the
pulse) both AND2 inputs were high while the read wordline was still on, so the
write drivers of the row being read were partly enabled. The same race exists
for `s_en` at the write-to-read edge; there the replica delay kept it at 1 to
2 mV. No check caught it because none looked at the write enable outside a
write.

**Fix.** `we` now passes a hold latch (the address-latch cell `Xwe_hold`:
D = `we`, EN = `wl_en_bar`, outputs `we_hold` / `we_hold_bar`) that is
transparent only while `wl_en` is low, the scheme the address bits and the
write data already use; `w_en` and `s_en` take the held request. `wl_en` is
derived from the request and falls after it, so neither enable can change
while a request is high. The `wl_en_bar` inverter is sized for the two extra
NAND2 inputs (one size step at 16, 128 and 512 rows). At FF −40 °C the held
request now rises 82 ps after the edge, 23 ps after WL_EN falls, and the
read-cycle write enable peaks at 6 mV. The access path is untouched: the
16x16 mux and 8x4 SS sequences reproduce their pre-fix request-to-output
times exactly (2.126 and 1.958 ns).

**New checks.** The waveform scorer adds `strict_cycle_N_write_enable_quiet`
(read cycles, every column's local write-enable terminal) and
`strict_cycle_N_sense_enable_quiet` (write cycles, local sense-enable
terminal), each at most 0.1 VDD from 0.65 T to 1.65 T. Scored read-only on the
retained traces, all eleven pre-fix FF −40 °C sequences fail it (0.34 to
0.66 VDD at the terminal) and the SS sequences pass.

Waveform evidence (local, ignored `outputs/validation/`):

- Pre-fix traces: `V2.1.1/small-queue/dist_8x4_FF_cold_sequence/attempt-001/dist_8x4_FF_cold_sequence/deck.sp.prn`;
  `V2.1.2-followup/phase5-P4-queue/p5_8x4_FF_cold_sequence_pd_s{1,2,3}/attempt-001/p5_8x4_FF_cold_sequence_pd_s{1,2,3}/deck.sp.prn`;
  `V2.1.3-10t-mux-budget/pd2-Q7-queue/p5x_8x4_FF_cold_sequence_pd_s{4..10}/attempt-001/p5x_8x4_FF_cold_sequence_pd_s{4..10}/deck.sp.prn`
  (worst: seed 20261004, 0.725 V).
- Edge table and plots: `V2.1.4-6t-timing/evidence/prefix_ff_cold_boundary.json`,
  `prefix_ff_cold_w_en_spike.png` (FF seed 20261004 against the SS 8x4
  sequence), `postfix_ff_cold_w_en.png`, `postfix_boundary.json`.
- Post-fix traces: `V2.1.4-6t-timing/probe-T-queue/t_8x4_FF_cold_sequence/`,
  `t_8x8_6t_mux_FF_cold_sequence/`, `probe-P{1,2,3}-queue/t_8x4_FF_cold_sequence_pd_s{1,2,3}/`,
  and the final write gate and pilot below.

## 2. Why both 6T ladders changed (open items 1 and 3)

The probe run (`probe-A/B/L/T`, four ranks per nominal case, eight for
512x4) ran the new TIME block on the unchanged V2.1.3 table and measured the
6T class bounds that had never been run, reading the target cell at each
bound:

| Case | Array | Op | PVT | Variation | Clock (ns), budget | Result | PRE90→WL50 min (ps) | WL10→PRE90 min (ps) | Request→OUT max (ns) | Deadline margin min (ps) | Boundary W_EN / S_EN peak (mV) |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| pr_32x16_6t_SS_read | 32x16 6T | read | SS 0.9 V 125 °C | nominal | 4 shared | passed (1,794 checks) | 539 | 359 | 2.091 | 109 |  |
| pr_32x16_6t_mux_SS_read | 32x16 6T mux | read | SS 0.9 V 125 °C | nominal | 4 shared | failed (1 of 1,794 checks) | 539 | 359 | 2.162 | 38 |  |
| pr_64x16_6t_SS_read | 64x16 6T | read | SS 0.9 V 125 °C | nominal | 4.5 shared | passed (3,394 checks) | 554 | 365 | 2.269 | 206 |  |
| pr_64x16_6t_mux_SS_read | 64x16 6T mux | read | SS 0.9 V 125 °C | nominal | 4.5 shared | passed (3,394 checks) | 554 | 365 | 2.362 | 113 |  |
| pr_128x8_6t_mux_SS_read | 128x8 6T mux | read | SS 0.9 V 125 °C | nominal | 5 shared | failed (1 of 3,434 checks) | 601 | 358 | 2.697 | 53 |  |
| pr_256x4_6t_mux_SS_read | 256x4 6T mux | read | SS 0.9 V 125 °C | nominal | 6 shared | failed (3 of 3,646 checks) | 694 | 351 | 3.329 | -29 |  |
| pr_512x4_6t_mux_SS_read | 512x4 6T mux | read | SS 0.9 V 125 °C | nominal | 9 shared | passed (1,116 checks) | 996 | 360 | 4.693 | 257 |  |
| t_8x4_FF_cold_sequence | 8x4 6T | read&write | FF 1.1 V -40 °C | nominal | 4 shared | passed (1,303 checks) | 151 | 79 | 0.727 | 1,473 | 6 / 1 |
| t_8x8_6t_mux_FF_cold_sequence | 8x8 6T mux | read&write | FF 1.1 V -40 °C | nominal | 4 shared | passed (2,435 checks) | 150 | 85 | 0.747 | 1,453 | 7 / 1 |
| t_16x16_6t_mux_SS_sequence | 16x16 6T mux | read&write | SS 0.9 V 125 °C | nominal | 4 shared | passed (6,939 checks) | 533 | 358 | 2.126 | 74 | 6 / 0 |
| t_8x4_SS_sequence | 8x4 6T | read&write | SS 0.9 V 125 °C | nominal | 4 shared | passed (1,303 checks) | 550 | 304 | 1.958 | 242 | 6 / 0 |
| t_8x4_FF_cold_sequence_pd_s1 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261001 | 4 shared | passed (1,303 checks) | 153 | 80 | 0.730 | 1,470 | 6 / 1 |
| t_8x4_FF_cold_sequence_pd_s2 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261002 | 4 shared | passed (1,303 checks) | 149 | 76 | 0.723 | 1,477 | 6 / 1 |
| t_8x4_FF_cold_sequence_pd_s3 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261003 | 4 shared | passed (1,303 checks) | 151 | 78 | 0.731 | 1,469 | 6 / 1 |

The shared classes left 109 ps at the 32x16 bound (4 ns), 206 ps at 64x16
(4.5 ns) and, from the retained V2.1.2 traces on the same access path, 203 ps
at 128x8 (5 ns) and 175 ps at 256x4 (6 ns); the 16x16 read seeds 4 to 10 of
V2.1.3 had 122 to 193 ps. A column mux delays the 6T output by about 70 ps at
32 rows, 95 ps at 64, 150 ps at 128, 200 ps at 256 and 275 ps at 512 rows
(10T showed the same growth). At the shared classes three mux reads failed
nominally, each with a healthy 0.75 to 0.88 V sense margin and a late output
only: 32x16 (38 ps before the deadline), 128x8 (53 ps) and 256x4 (29 ps
after it). With the V2.1.3 16x16 mux sequence (74 ps nominal, seeds 2 and 3
failing at 51 and 57 ps; `V2.1.3-10t-mux-budget/pd2-Q8-queue/c_16x16_6t_mux_SS_sequence_pd_s{2,3}/`)
these are the late-output failures the new ladders remove. Failing traces:
`V2.1.4-6t-timing/probe-A-queue/pr_32x16_6t_mux_SS_read/`,
`probe-B-queue/pr_128x8_6t_mux_SS_read/`, `probe-B-queue/pr_256x4_6t_mux_SS_read/`.

Across the reads the output margin follows `0.485 T − d(rows)` within a few ps
(the request sits 0.065 T after the nominal 0.65 T), with d = 1.83, 1.98,
2.22, 2.74 and 3.83 ns for 6T at 32, 64, 128, 256 and 512 rows and 1.90,
2.07, 2.37, 2.94 and 4.10 ns with a mux; sequences need about 50 ps more than
reads. The V2.1.3 rule (at least 250 ps between the local read output and the
1.2 T deadline at every class bound, nominal SS 0.9 V / 125 °C) then gives:

| Row bound | Shared (6T, no mux) budget | Period | 6T-mux budget | Period |
|---|---:|---:|---:|---:|
| 32 | 1800 ps (was 1600) | 4.5 ns | 1900 ps | 4.75 ns |
| 64 | 1900 ps (was 1800) | 4.75 ns | 2000 ps | 5 ns |
| 128 | 2100 ps (was 2000) | 5.25 ns | 2200 ps | 5.5 ns |
| 256 | 2500 ps (was 2400) | 6.25 ns | 2700 ps | 6.75 ns |
| 512 | 3600 ps | 9 ns | 3600 ps | 9 ns |

The column classes stay shared (read margins at the column bounds are large;
writes set them). `timing_lookup.json` is `v2.1.4-timing-3`; the 10T variant
is unchanged. `ArrayTiming.budget` reads `shared` or `SRAM_6T_CELL/mux`.

## 3. Nominal boundary run on the final table

Every 6T row and column class bound with and without a mux, the 8x8 mux
sequence, the 64x16 SF writes and the 512x4 SS read and SF write
(`final-N1..N4` at four ranks, `final-L` at eight):

| Case | Array | Op | PVT | Variation | Clock (ns), budget | Result | PRE90→WL50 min (ps) | WL10→PRE90 min (ps) | Request→OUT max (ns) | Deadline margin min (ps) | Boundary W_EN / S_EN peak (mV) |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| f_8x64_6t_SS_read | 8x64 6T | read | SS 0.9 V 125 °C | nominal | 5 shared | passed (2,274 checks) | 529 | 382 | 2.131 | 619 |  |
| f_8x128_6t_SS_read | 8x128 6T | read | SS 0.9 V 125 °C | nominal | 6 shared | passed (4,514 checks) | 569 | 392 | 2.308 | 992 |  |
| f_16x32_6t_SS_sequence | 16x32 6T | read&write | SS 0.9 V 125 °C | nominal | 4.5 shared | passed (13,643 checks) | 532 | 369 | 2.133 | 342 | 6 / 0 |
| f_32x16_6t_SS_sequence | 32x16 6T | read&write | SS 0.9 V 125 °C | nominal | 4.5 shared | passed (11,419 checks) | 534 | 359 | 2.154 | 321 | 5 / 0 |
| f_64x16_6t_SS_read | 64x16 6T | read | SS 0.9 V 125 °C | nominal | 4.75 shared | passed (3,394 checks) | 554 | 364 | 2.286 | 327 |  |
| f_64x16_6t_SF_write | 64x16 6T | write | SF 0.9 V 125 °C | nominal | 4.75 shared | passed (3,569 checks) | 492 | 318 |  |  |  |
| f_128x8_6t_SS_read | 128x8 6T | read | SS 0.9 V 125 °C | nominal | 5.25 shared | passed (3,434 checks) | 600 | 359 | 2.562 | 325 |  |
| f_256x4_6t_SS_read | 256x4 6T | read | SS 0.9 V 125 °C | nominal | 6.25 shared | passed (3,646 checks) | 694 | 349 | 3.139 | 299 |  |
| f_512x4_6t_SS_read | 512x4 6T | read | SS 0.9 V 125 °C | nominal | 9 shared | passed (1,116 checks) | 996 | 362 | 4.406 | 544 |  |
| f_512x4_6t_SF_write | 512x4 6T | write | SF 0.9 V 125 °C | nominal | 9 shared | passed (1,159 checks) | 934 | 309 |  |  |  |
| f_8x8_6t_mux_SS_sequence | 8x8 6T mux | read&write | SS 0.9 V 125 °C | nominal | 4.75 SRAM_6T_CELL/mux | passed (2,435 checks) | 545 | 322 | 2.108 | 505 | 6 / 0 |
| f_8x64_6t_mux_SS_read | 8x64 6T mux | read | SS 0.9 V 125 °C | nominal | 5 SRAM_6T_CELL/mux | passed (2,274 checks) | 534 | 380 | 2.208 | 542 |  |
| f_8x128_6t_mux_SS_read | 8x128 6T mux | read | SS 0.9 V 125 °C | nominal | 6 SRAM_6T_CELL/mux | passed (4,514 checks) | 569 | 390 | 2.369 | 931 |  |
| f_16x16_6t_mux_SS_sequence | 16x16 6T mux | read&write | SS 0.9 V 125 °C | nominal | 4.75 SRAM_6T_CELL/mux | passed (6,939 checks) | 533 | 358 | 2.176 | 436 | 6 / 0 |
| f_16x32_6t_mux_SS_sequence | 16x32 6T mux | read&write | SS 0.9 V 125 °C | nominal | 4.75 SRAM_6T_CELL/mux | passed (13,643 checks) | 529 | 368 | 2.229 | 383 | 6 / 0 |
| f_32x16_6t_mux_SS_sequence | 32x16 6T mux | read&write | SS 0.9 V 125 °C | nominal | 4.75 SRAM_6T_CELL/mux | passed (11,419 checks) | 538 | 358 | 2.240 | 372 | 4 / 0 |
| f_64x16_6t_mux_SS_read | 64x16 6T mux | read | SS 0.9 V 125 °C | nominal | 5 SRAM_6T_CELL/mux | passed (3,394 checks) | 554 | 365 | 2.393 | 357 |  |
| f_64x16_6t_mux_SF_write | 64x16 6T mux | write | SF 0.9 V 125 °C | nominal | 5 SRAM_6T_CELL/mux | passed (3,569 checks) | 492 | 318 |  |  |  |
| f_128x8_6t_mux_SS_read | 128x8 6T mux | read | SS 0.9 V 125 °C | nominal | 5.5 SRAM_6T_CELL/mux | passed (3,434 checks) | 601 | 360 | 2.727 | 298 |  |
| f_256x4_6t_mux_SS_read | 256x4 6T mux | read | SS 0.9 V 125 °C | nominal | 6.75 SRAM_6T_CELL/mux | passed (3,646 checks) | 695 | 352 | 3.380 | 333 |  |
| f_512x4_6t_mux_SS_read | 512x4 6T mux | read | SS 0.9 V 125 °C | nominal | 9 SRAM_6T_CELL/mux | passed (1,116 checks) | 996 | 360 | 4.693 | 257 |  |

Every case passes. The shared ladder keeps 321 ps at the 32x16 sequence bound
(4.5 ns), 327 ps at 64x16 (4.75 ns), 325 ps at 128x8 (5.25 ns), 299 ps at 256x4
(6.25 ns) and 544 ps at 512x4 (9 ns); with a mux 436 and 372 ps at the 16x16 and
32x16 sequences (4.75 ns), 357 ps at 64x16 (5 ns), 298 ps at 128x8 (5.5 ns),
333 ps at 256x4 (6.75 ns) and 257 ps at 512x4 (9 ns). The column bounds keep
342 and 383 ps (16x32 sequences), 619 and 542 ps (8x64) and 992 and 931 ps
(8x128). The reads land within 10 ps of the `0.485 T − d` model; the sequences
did better than its 50 ps sequence allowance. The 64x16 and 512x4 SF writes pass
their capture, drive, release and retention checks. The 4.75 ns mux sequences
in this table are the fix-group reruns. Minimum-step warning (non-fatal):
f_8x8_6t_mux_SS_sequence. Read-access waveforms at the tightest bounds, with the
rejected 256x4 mux read at 6 ns for comparison: `evidence/final_read_access_bounds.png`;
write-capture plots: `f_32x16_6t_SS_sequence-write-capture.png`,
`f_64x16_6t_SF_write-write-capture.png`.

## 4. Write-waveform gate on the new TIME block

The V2.1.1 Phase 2 gate cases (`final-G`, four ranks) plus an 8x8 6T mux
FF −40 °C sequence. The 8x4 10T mux sequence is a smoke check of the shared
TIME change only; the 10T class evidence is the next round:

| Case | Array | Op | PVT | Variation | Clock (ns), budget | Result | PRE90→WL50 min (ps) | WL10→PRE90 min (ps) | Request→OUT max (ns) | Deadline margin min (ps) | Boundary W_EN / S_EN peak (mV) |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| g_10x6_SF_write_hazard | 10x6 6T hazard | write | SF 0.9 V 125 °C | nominal | 4.5 shared | passed (349 checks) | 491 | 279 |  |  |  |
| g_16x16_SS_read | 16x16 6T | read | SS 0.9 V 125 °C | nominal | 4.5 shared | passed (994 checks) | 533 | 360 | 2.069 | 406 |  |
| g_64x4_TT_write_hazard | 64x4 6T hazard | write | TT 1.0 V 25 °C | nominal | 4.75 shared | passed (1,001 checks) | 255 | 136 |  |  |  |
| g_8x4_10t_mux_SS_sequence | 8x4 10T mux | read&write | SS 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL | passed (1,303 checks) | 550 | 304 | 2.202 | 548 | 6 / 0 |
| g_8x4_FF_cold_sequence | 8x4 6T | read&write | FF 1.1 V -40 °C | nominal | 4.5 shared | passed (1,303 checks) | 151 | 79 | 0.760 | 1,715 | 6 / 1 |
| g_8x4_SF_sequence | 8x4 6T | read&write | SF 0.9 V 125 °C | nominal | 4.5 shared | passed (1,303 checks) | 489 | 259 | 1.808 | 667 | 6 / 1 |
| g_8x4_SS_sequence | 8x4 6T | read&write | SS 0.9 V 125 °C | nominal | 4.5 shared | passed (1,303 checks) | 550 | 304 | 1.992 | 483 | 6 / 0 |
| g_8x4_TT_nostubs_sequence | 8x4 6T no stubs | read&write | TT 1.0 V 25 °C | nominal | 4.5 shared | passed (1,303 checks) | 215 | 138 | 0.931 | 1,544 | 17 / 4 |
| g_8x4_TT_write | 8x4 6T | write | TT 1.0 V 25 °C | nominal | 4.5 shared | passed (217 checks) | 244 | 132 |  |  |  |
| g_8x8_6t_mux_FF_cold_sequence | 8x8 6T mux | read&write | FF 1.1 V -40 °C | nominal | 4.75 SRAM_6T_CELL/mux | passed (2,435 checks) | 150 | 85 | 0.797 | 1,815 | 7 / 1 |

All ten pass on the new TIME block; the 8x8 mux FF −40 °C sequence is the
fix-group rerun. Boundary write-enable peaks stay at 6 to 7 mV in every
sequence except the 8x4 TT sequence without local stubs (17 mV, still under
the 0.1 VDD check). The 8x4 FF −40 °C sequence, the case behind open item 2,
keeps a 6 mV write-enable peak (0.41 V before the fix) and 1,715 ps of output
margin. Both address-change hazards and the 16x16 SS read (406 ps) pass. The
10T smoke sequence passes 1,303 checks with 548 ps (minimum-step warning,
non-fatal). Write-capture plot of the FF −40 °C sequence:
`g_8x4_FF_cold_sequence-write-capture.png`.

## 5. Mismatch at the class bounds and the ten-seed pilot

Single-rank per-device samples on the final table (`pd-Q1..Q8`): three seeds
(20261001 to 20261003) at the 32-, 64- and 128-row bounds with and without a
mux and at the 16x16 mux sequence that failed in V2.1.3; ten seeds (20261001
to 20261010) at the 8x8 mux sequence and the three 6T Phase 5 pilot cases
(16x16 SS read, 16x16 SF write, 8x4 FF −40 °C sequence), which the TIME change
required again:

| Case | Array | Op | PVT | Variation | Clock (ns), budget | Result | PRE90→WL50 min (ps) | WL10→PRE90 min (ps) | Request→OUT max (ns) | Deadline margin min (ps) | Boundary W_EN / S_EN peak (mV) |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| p_8x4_FF_cold_sequence_pd_s1 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261001 | 4.5 shared | passed (1,303 checks) | 153 | 80 | 0.763 | 1,712 | 6 / 1 |
| p_8x4_FF_cold_sequence_pd_s2 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261002 | 4.5 shared | passed (1,303 checks) | 149 | 76 | 0.756 | 1,719 | 6 / 1 |
| p_8x4_FF_cold_sequence_pd_s3 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261003 | 4.5 shared | passed (1,303 checks) | 151 | 78 | 0.764 | 1,711 | 6 / 1 |
| p_8x4_FF_cold_sequence_pd_s4 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261004 | 4.5 shared | passed (1,303 checks) | 151 | 79 | 0.761 | 1,714 | 6 / 1 |
| p_8x4_FF_cold_sequence_pd_s5 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261005 | 4.5 shared | passed (1,303 checks) | 150 | 80 | 0.757 | 1,718 | 6 / 1 |
| p_8x4_FF_cold_sequence_pd_s6 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261006 | 4.5 shared | passed (1,303 checks) | 152 | 78 | 0.763 | 1,712 | 6 / 1 |
| p_8x4_FF_cold_sequence_pd_s7 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261007 | 4.5 shared | passed (1,303 checks) | 149 | 80 | 0.759 | 1,716 | 6 / 1 |
| p_8x4_FF_cold_sequence_pd_s8 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261008 | 4.5 shared | passed (1,303 checks) | 149 | 80 | 0.763 | 1,712 | 6 / 1 |
| p_8x4_FF_cold_sequence_pd_s9 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261009 | 4.5 shared | passed (1,303 checks) | 152 | 83 | 0.763 | 1,712 | 6 / 1 |
| p_8x4_FF_cold_sequence_pd_s10 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261010 | 4.5 shared | passed (1,303 checks) | 150 | 79 | 0.762 | 1,713 | 6 / 1 |
| p_16x16_SS_read_pd_s1 | 16x16 6T | read | SS 0.9 V 125 °C | per-device 20261001 | 4.5 shared | passed (994 checks) | 521 | 356 | 2.024 | 451 |  |
| p_16x16_SS_read_pd_s2 | 16x16 6T | read | SS 0.9 V 125 °C | per-device 20261002 | 4.5 shared | passed (994 checks) | 541 | 375 | 2.078 | 397 |  |
| p_16x16_SS_read_pd_s3 | 16x16 6T | read | SS 0.9 V 125 °C | per-device 20261003 | 4.5 shared | passed (994 checks) | 520 | 354 | 2.053 | 422 |  |
| p_16x16_SS_read_pd_s4 | 16x16 6T | read | SS 0.9 V 125 °C | per-device 20261004 | 4.5 shared | passed (994 checks) | 551 | 324 | 2.126 | 349 |  |
| p_16x16_SS_read_pd_s5 | 16x16 6T | read | SS 0.9 V 125 °C | per-device 20261005 | 4.5 shared | passed (994 checks) | 539 | 352 | 2.115 | 360 |  |
| p_16x16_SS_read_pd_s6 | 16x16 6T | read | SS 0.9 V 125 °C | per-device 20261006 | 4.5 shared | passed (994 checks) | 512 | 347 | 2.060 | 415 |  |
| p_16x16_SS_read_pd_s7 | 16x16 6T | read | SS 0.9 V 125 °C | per-device 20261007 | 4.5 shared | passed (994 checks) | 529 | 363 | 2.050 | 425 |  |
| p_16x16_SS_read_pd_s8 | 16x16 6T | read | SS 0.9 V 125 °C | per-device 20261008 | 4.5 shared | passed (994 checks) | 531 | 371 | 2.075 | 400 |  |
| p_16x16_SS_read_pd_s9 | 16x16 6T | read | SS 0.9 V 125 °C | per-device 20261009 | 4.5 shared | passed (994 checks) | 521 | 341 | 2.094 | 381 |  |
| p_16x16_SS_read_pd_s10 | 16x16 6T | read | SS 0.9 V 125 °C | per-device 20261010 | 4.5 shared | passed (994 checks) | 529 | 347 | 2.054 | 421 |  |
| p_16x16_SF_write_pd_s1 | 16x16 6T | write | SF 0.9 V 125 °C | per-device 20261001 | 4.5 shared | passed (1,169 checks) | 455 | 312 |  |  |  |
| p_16x16_SF_write_pd_s2 | 16x16 6T | write | SF 0.9 V 125 °C | per-device 20261002 | 4.5 shared | passed (1,169 checks) | 467 | 311 |  |  |  |
| p_16x16_SF_write_pd_s3 | 16x16 6T | write | SF 0.9 V 125 °C | per-device 20261003 | 4.5 shared | passed (1,169 checks) | 480 | 314 |  |  |  |
| p_16x16_SF_write_pd_s4 | 16x16 6T | write | SF 0.9 V 125 °C | per-device 20261004 | 4.5 shared | passed (1,169 checks) | 496 | 293 |  |  |  |
| p_16x16_SF_write_pd_s5 | 16x16 6T | write | SF 0.9 V 125 °C | per-device 20261005 | 4.5 shared | passed (1,169 checks) | 464 | 303 |  |  |  |
| p_16x16_SF_write_pd_s6 | 16x16 6T | write | SF 0.9 V 125 °C | per-device 20261006 | 4.5 shared | passed (1,169 checks) | 504 | 326 |  |  |  |
| p_16x16_SF_write_pd_s7 | 16x16 6T | write | SF 0.9 V 125 °C | per-device 20261007 | 4.5 shared | passed (1,169 checks) | 506 | 330 |  |  |  |
| p_16x16_SF_write_pd_s8 | 16x16 6T | write | SF 0.9 V 125 °C | per-device 20261008 | 4.5 shared | passed (1,169 checks) | 474 | 304 |  |  |  |
| p_16x16_SF_write_pd_s9 | 16x16 6T | write | SF 0.9 V 125 °C | per-device 20261009 | 4.5 shared | passed (1,169 checks) | 462 | 308 |  |  |  |
| p_16x16_SF_write_pd_s10 | 16x16 6T | write | SF 0.9 V 125 °C | per-device 20261010 | 4.5 shared | passed (1,169 checks) | 480 | 317 |  |  |  |
| f_32x16_6t_SS_sequence_pd_s1 | 32x16 6T | read&write | SS 0.9 V 125 °C | per-device 20261001 | 4.5 shared | passed (11,419 checks) | 532 | 364 | 2.132 | 343 | 4 / 0 |
| f_32x16_6t_SS_sequence_pd_s2 | 32x16 6T | read&write | SS 0.9 V 125 °C | per-device 20261002 | 4.5 shared | passed (11,419 checks) | 544 | 346 | 2.106 | 369 | 5 / 0 |
| f_32x16_6t_SS_sequence_pd_s3 | 32x16 6T | read&write | SS 0.9 V 125 °C | per-device 20261003 | 4.5 shared | passed (11,419 checks) | 534 | 395 | 2.139 | 336 | 5 / 0 |
| f_64x16_6t_SS_read_pd_s1 | 64x16 6T | read | SS 0.9 V 125 °C | per-device 20261001 | 4.75 shared | passed (3,394 checks) | 547 | 362 | 2.271 | 342 |  |
| f_64x16_6t_SS_read_pd_s2 | 64x16 6T | read | SS 0.9 V 125 °C | per-device 20261002 | 4.75 shared | passed (3,394 checks) | 518 | 403 | 2.256 | 357 |  |
| f_64x16_6t_SS_read_pd_s3 | 64x16 6T | read | SS 0.9 V 125 °C | per-device 20261003 | 4.75 shared | passed (3,394 checks) | 536 | 359 | 2.283 | 329 |  |
| f_128x8_6t_SS_read_pd_s1 | 128x8 6T | read | SS 0.9 V 125 °C | per-device 20261001 | 5.25 shared | passed (3,434 checks) | 613 | 359 | 2.576 | 311 |  |
| f_128x8_6t_SS_read_pd_s2 | 128x8 6T | read | SS 0.9 V 125 °C | per-device 20261002 | 5.25 shared | passed (3,434 checks) | 625 | 358 | 2.571 | 316 |  |
| f_128x8_6t_SS_read_pd_s3 | 128x8 6T | read | SS 0.9 V 125 °C | per-device 20261003 | 5.25 shared | passed (3,434 checks) | 548 | 363 | 2.585 | 302 |  |
| f_8x8_6t_mux_SS_sequence_pd_s1 | 8x8 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261001 | 4.75 SRAM_6T_CELL/mux | passed (2,435 checks) | 563 | 350 | 2.161 | 451 | 6 / 0 |
| f_8x8_6t_mux_SS_sequence_pd_s2 | 8x8 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261002 | 4.75 SRAM_6T_CELL/mux | passed (2,435 checks) | 575 | 302 | 2.159 | 454 | 6 / 0 |
| f_8x8_6t_mux_SS_sequence_pd_s3 | 8x8 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261003 | 4.75 SRAM_6T_CELL/mux | passed (2,435 checks) | 555 | 345 | 2.142 | 470 | 6 / 0 |
| f_8x8_6t_mux_SS_sequence_pd_s4 | 8x8 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261004 | 4.75 SRAM_6T_CELL/mux | passed (2,435 checks) | 537 | 332 | 2.125 | 488 | 6 / 0 |
| f_8x8_6t_mux_SS_sequence_pd_s5 | 8x8 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261005 | 4.75 SRAM_6T_CELL/mux | passed (2,435 checks) | 550 | 336 | 2.120 | 492 | 6 / 0 |
| f_8x8_6t_mux_SS_sequence_pd_s6 | 8x8 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261006 | 4.75 SRAM_6T_CELL/mux | passed (2,435 checks) | 522 | 306 | 2.079 | 533 | 6 / 0 |
| f_8x8_6t_mux_SS_sequence_pd_s7 | 8x8 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261007 | 4.75 SRAM_6T_CELL/mux | passed (2,435 checks) | 527 | 372 | 2.084 | 528 | 5 / 0 |
| f_8x8_6t_mux_SS_sequence_pd_s8 | 8x8 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261008 | 4.75 SRAM_6T_CELL/mux | passed (2,435 checks) | 561 | 332 | 2.113 | 499 | 6 / 0 |
| f_8x8_6t_mux_SS_sequence_pd_s9 | 8x8 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261009 | 4.75 SRAM_6T_CELL/mux | passed (2,435 checks) | 552 | 327 | 2.146 | 466 | 6 / 0 |
| f_8x8_6t_mux_SS_sequence_pd_s10 | 8x8 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261010 | 4.75 SRAM_6T_CELL/mux | passed (2,435 checks) | 554 | 349 | 2.124 | 489 | 6 / 0 |
| f_16x16_6t_mux_SS_sequence_pd_s1 | 16x16 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261001 | 4.75 SRAM_6T_CELL/mux | passed (6,939 checks) | 550 | 359 | 2.195 | 418 | 6 / 0 |
| f_16x16_6t_mux_SS_sequence_pd_s2 | 16x16 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261002 | 4.75 SRAM_6T_CELL/mux | passed (6,939 checks) | 577 | 383 | 2.247 | 365 | 6 / 0 |
| f_16x16_6t_mux_SS_sequence_pd_s3 | 16x16 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261003 | 4.75 SRAM_6T_CELL/mux | passed (6,939 checks) | 529 | 362 | 2.139 | 474 | 6 / 0 |
| f_32x16_6t_mux_SS_sequence_pd_s1 | 32x16 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261001 | 4.75 SRAM_6T_CELL/mux | passed (11,419 checks) | 552 | 345 | 2.237 | 375 | 5 / 0 |
| f_32x16_6t_mux_SS_sequence_pd_s2 | 32x16 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261002 | 4.75 SRAM_6T_CELL/mux | passed (11,419 checks) | 529 | 354 | 2.237 | 376 | 5 / 0 |
| f_32x16_6t_mux_SS_sequence_pd_s3 | 32x16 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261003 | 4.75 SRAM_6T_CELL/mux | passed (11,419 checks) | 542 | 385 | 2.196 | 416 | 5 / 0 |
| f_64x16_6t_mux_SS_read_pd_s1 | 64x16 6T mux | read | SS 0.9 V 125 °C | per-device 20261001 | 5 SRAM_6T_CELL/mux | passed (3,394 checks) | 533 | 382 | 2.368 | 382 |  |
| f_64x16_6t_mux_SS_read_pd_s2 | 64x16 6T mux | read | SS 0.9 V 125 °C | per-device 20261002 | 5 SRAM_6T_CELL/mux | passed (3,394 checks) | 552 | 372 | 2.343 | 407 |  |
| f_64x16_6t_mux_SS_read_pd_s3 | 64x16 6T mux | read | SS 0.9 V 125 °C | per-device 20261003 | 5 SRAM_6T_CELL/mux | passed (3,394 checks) | 534 | 339 | 2.362 | 388 |  |
| f_128x8_6t_mux_SS_read_pd_s1 | 128x8 6T mux | read | SS 0.9 V 125 °C | per-device 20261001 | 5.5 SRAM_6T_CELL/mux | passed (3,434 checks) | 580 | 359 | 2.749 | 276 |  |
| f_128x8_6t_mux_SS_read_pd_s2 | 128x8 6T mux | read | SS 0.9 V 125 °C | per-device 20261002 | 5.5 SRAM_6T_CELL/mux | passed (3,434 checks) | 634 | 357 | 2.761 | 264 |  |
| f_128x8_6t_mux_SS_read_pd_s3 | 128x8 6T mux | read | SS 0.9 V 125 °C | per-device 20261003 | 5.5 SRAM_6T_CELL/mux | passed (3,434 checks) | 585 | 360 | 2.713 | 312 |  |

All 61 seeds pass. At the class bounds: 32x16 sequence 336–369 ps, 64x16
read 329–357 ps, 128x8 read 302–316 ps; with a mux 32x16 sequence 375–416 ps,
64x16 read 382–407 ps and 128x8 read 264–312 ps. The 16x16 mux sequence, whose
seeds 20261002 and 20261003 failed at 4 ns in V2.1.3, keeps 365–474 ps at
4.75 ns. Ten seeds: 8x8 mux sequence 451–533 ps, 16x16 SS read 349–451 ps,
16x16 SF write 10 of 10, 8x4 FF −40 °C sequence 1,711–1,719 ps with
write-enable peaks of at most 6 mV (0.38 to 0.72 V on these seeds before the
fix). Every 4.75 ns mux-sequence seed is the fix-group rerun.
`f_128x8_6t_mux_SS_read_pd_s3` needed the runtime's single DC-operating-point
retry and then passed. Minimum-step warnings (non-fatal): six mux-sequence
seeds. Sigma is 5 % relative; ten seeds are a pilot, not a yield statement
(0 of 10 failures bounds the rate only to 26 % at 95 % confidence).

## 6. Duplicated final sample at quarter-nanosecond clocks (found in this run)

In the first final run, 6T mux sequences at the new 4.75 ns class failed
`strict_finite_monotonic`: the scorer found two trace rows printed at the
final time, 42.325 ns, differing by at most 0.67 µV. The check fails closed,
so those attempts ran only the legacy and runtime checks (all passing; the
32x16 mux read output kept 372 ps). A single-rank retry of the same circuit and
seed (`final-R`) reproduced the duplicate row for row.

Cause: sequences stop at `1 ns + 8.7 T` and Xyce prints on the 2 ps
`INITIAL_INTERVAL` grid. With T = 4.75 ns the stop is 42.325 ns, off that
grid: passing traces end 42.324 ns, 42.325 ns, failing ones print 42.325 ns
twice. All earlier clocks were multiples of 0.5 ns, whose sequence stops lie
on the grid (40.150 ns at 4.5 ns), so no earlier record met it. Fix:
`_analysis_stop()` rounds the stop up to a multiple of the output interval
(42.326 ns here); the last-cycle measurement windows keep their 8.7 T bound.
A test checks every lookup class's sequence stop against the grid.

Failed attempts, kept as evidence (`V2.1.4-6t-timing/`, five of 93 in the
first final run, each `deck.sp.prn` ending with two rows at 4.23250000e-08):
`final-N3-queue/f_32x16_6t_mux_SS_sequence/attempt-001/` (nominal, four
ranks), `final-R-queue/f_32x16_6t_mux_SS_sequence/attempt-001/` (one rank),
`pd-Q2-queue/f_8x8_6t_mux_SS_sequence_pd_s2/attempt-001/`,
`pd-Q3-queue/f_8x8_6t_mux_SS_sequence_pd_s3/attempt-001/` and
`pd-Q7-queue/f_8x8_6t_mux_SS_sequence_pd_s8/attempt-001/`.
All 22 sequences with an off-grid stop (the 6T mux sequences at 4.75 ns,
nominal, write gate and seeds) were rerun on the fixed sources in the
`fix-*` queues, whose results replace those attempts in sections 3 to 5;
every other deck of the final run regenerates identical from the fixed
sources apart from the order of case-variant duplicate probes on the
`.PRINT` line and the per-device model-card path (`verify_decks.py`,
`evidence/verify_final_decks.{json,log}`): 70 of 93 decks and their model
cards are identical, the 22 off-grid-stop attempts differ as intended, and
one per-device read (`f_128x8_6t_mux_SS_read_pd_s3`) differs only by the
`.OPTIONS NONLIN SEARCHMETHOD=2` line that the runtime's single
DC-operating-point retry wrote into it (`evidence/verify_final_decks_notes.md`).

## 7. Deck comparison

Eight nominal decks generated without a simulator from a detached `9b785f8`
worktree and from the final V2.1.4 tree (`V2.1.4-6t-timing/deck-comparison/`):
the 10T decks (8x4 mux, 16x16, 256x4 mux) differ only in TIME (`Xwe_hold`, the
`w_en`/`s_en` inputs and, at 16 rows, the `wl_en_bar` inverter) with
unchanged clocks; the 6T decks differ in TIME and, where a class changed, in
stimulus, `.TRAN` and measurement times (8x4 at 4.5 ns, 16x16 mux at 4.75 ns
with its stop on the output grid, 32x16 at 4.5 ns); the 512x4 read and the
8x128 mux write keep 9 and 6 ns. Against the tree that ran the first final
run, the stop fix changes only the `.TRAN` stop of the off-grid sequence
(42.325 to 42.33 ns on the default 10 ps grid).

## 8. Evidence

Raw decks, waveforms, solver logs, per-attempt checkpoints and source archives
are local under ignored `outputs/validation/V2.1.4-6t-timing/` (`probe-*` on
the V2.1.3 table, `final-*` and `pd-Q*` on the final table; `chain.log`;
`evidence/`; `deck-comparison/`). The machine-readable
[record](TIMING_6T_BUDGET_V2_1_4.json) carries every case, identity, metric
and measurement of both runs.
