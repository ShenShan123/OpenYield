# V2.1.5 10T round: read-path resize, 512-row class and the re-run boundary matrix

Executed September 15 and 16, 2026 on the V2.1.5 sources (base commit
`acec3be0` plus the cell, table, equivalent-model, test and documentation
changes of this release; every queue archives the exact sources it ran).
Illustrative wires, V2.0.9 driver classes, the V2.1.4 TIME block with its
write-request hold latch and the V2.1.4 waveform checks throughout; nominal
seed 20260915, maximum step 20 ps, at most 34 simulator ranks at any time;
per-device samples on one rank each with 5 % relative sigma on `vth0`, `u0`
and `voff` of every MOS. This is functional screening at SS 0.9 V / 125 °C
(SF for writes, FF 1.1 V / −40 °C for the cold sequences), not PVT/mismatch or
extracted-metal qualification; no record is promoted to `sizing_table.json`.

It closes items 4, 5 and 7 of the [V2.1.3 open items](../plans/V2_1_3_OPEN_ITEMS.md).
Measurement definitions ("Request→OUT", "deadline margin", PRE90→WL50 and
WL10→PRE90) are as in the [V2.1.3 10T budget](TIMING_10T_BUDGET_V2_1_3.md),
measured from the retained `.prn` traces with `dev/v213_sense_timing.py`; the
read-disturb columns come from the new `dev/v215_read_disturb.py`.

**Outcome: on the released sources 45 of 45 attempts pass, 173,193 of 173,193 checks:
all 20 nominal boundary cases (including the 512-row bound at its new 10 ns class and two
FF −40 °C sequences) and all 25 mismatch seeds. No solver aborted (0 numerical events);
one per-device case took the runtime's single timestep retry and then passed, and the only
other solver message is the non-fatal minimum-step warning in 8 runs.**

## 1. The 10T read-disturb bump is a pull-down sizing problem (item 4)

V2.1.3 set the 512-row 10T class at 14 ns not from the sense path but from the
storage node: during a read the low node of the addressed 10T cell rises to
0.196 V (0.131 V for 6T at the same PVT) and decays only as the bitline
discharges, so a 512-row bitline left it at 0.108 V at the 1.2 T deadline
against the 0.1 VDD tolerance of `strict_cycle_N_hold_*` at 11 ns. The V2.1.5
probe reproduces that failure on the current TIME block and measures the same
node at 0.1001 V (`probe-L1`; the small difference from the V2.1.3 figure is
the sampling point, not the circuit).

The cause is the cell, not the clock. The 10T cell is a Schmitt-trigger cell:
each pull-down is a stack of two NMOS (`MNL1` in series with `MNL2`), where 6T
has one, and the feedback device on the disturbed side is off during the
disturb, so the bump is the read current through twice the pull-down
resistance of a 6T cell of the same width. The tracked cell used the 6T
pull-down width (205 nm) for both stacked devices.

Three candidates were screened with the probe-only `cell_widths` key of
`dev/validate_distributed_rc.py` (the tracked YAML untouched; the matched
replica cell follows the array cell through `resolve_driver_sizes`), at the
released classes and, for the 512-row bound, on a probe table:

| Pull-down width | Read-disturb peak (16x16, 5 ns) | 512x4 mux read at 11 ns: Q at the deadline | 512x4 output margin at 11 ns | 128x8 mux read margin at 6 ns | hold / read / write SNM (8x4) | Bitcell area | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| 205 nm (V2.1.3, the 6T width) | 0.190 V | **0.1001 V** | 560 ps | 282 ps | 422.7 / 240.9 / 370.1 mV | 1.194 µm² | fails two storage checks |
| **287 nm (1.4x, adopted)** | 0.155 V | 0.0556 V | 958 ps | 438 ps | 427.5 / 251.6 / 354.6 mV | 1.391 µm² (+16.5 %) | passes |
| 369 nm (1.8x, rejected) | 0.131 V | 0.0349 V | 1,172 ps | 522 ps | 428.9 / 261.0 / 343.4 mV | 1.588 µm² (+33 %) | passes |

The storage tolerance is 0.1 VDD = 0.090 V. Every candidate passes the 16x16
sequence, the 128x8 mux read, the 64x16 mux SF write and a 16x16 mux
FF 1.1 V / −40 °C sequence on the released classes, so the resize is not paid
for in function or write-ability; the SNM columns are DC sweeps of the same
cell at SS 0.9 V / 125 °C (`dev/v215_10t_snm.py`, `snm_candidates.json`).

Widening the stack also moves the read output earlier at every height, because
the array cell and the matched replica cell discharge their bitlines faster:
the 128-row bound, the smallest 10T margin in V2.1.3, goes from 282 to 438 ps.

The 1.8x candidate was rejected on area. It passes 512x4 at 10 ns and at 9.5 ns
(`probe-M2`, `probe-M1`), one class step below the adopted 10 ns, but it costs
another 16.5 % of bitcell area on *every* 10T array to buy 0.5 ns at 512 rows
only. The 1.4x cell reaches 10 ns with 314 ps of settling margin and 475 ps of
output margin on the released sources, which is what section 3 confirms.

The probe queues ran before the release edit and carry a `cell_widths` override,
so they are screening, not release evidence; `probe-L1`, `probe-L2` and
`probe-L3` additionally record `source_changed`, because the `VERSION` string
was bumped to V2.1.5 while they ran. That string reaches no netlist, and the
three 512-row probes above completed with `worker_exit 0` and full check sets
(1,116 checks each). The release evidence is sections 3 and 4, which ran on the
committed sources with no override.

## 2. Budget adopted

`sram_10t_cell.yaml` takes a 287 nm 10T pull-down (`nmos_width` index 0, with
the usual ±20 % optimizer bounds, 230 to 344 nm) and `timing_lookup.json`
`v2.1.5-timing-4` takes a 4000 ps 512-row row budget for the `SRAM_10T_CELL`
variant (10 ns with the 25 % margin, was 5600 ps / 14 ns). Everything else is
unchanged: the 10T row budgets 2000/2200/2400/3200 ps up to 256 rows, the 10T
column budgets, both 6T ladders, the driver classes, the TIME circuit and the
checks. Against a detached `acec3be` worktree the five 6T reference decks are
byte-identical and the four 10T decks differ only in the eleven pull-down
widths of the array and replica cells
(`deck-comparison/release_vs_acec3be.json`, `10t_*.diff`).

The pull-down width is the only cell change: the access, feedback and pull-up
widths, the length and every threshold model are unchanged, so write-ability
rests on the same pass-gate/pull-up ratio it did before.

One compiler fix came out of the new ladder. `timing_lookup.json`'s policy says
a variant "may never fall below the shared budget", and the loader enforces it
at every tabulated anchor, but beyond the last anchor each ladder extrapolates
its own final ratio. The 10T ladder now ends 3200 → 4000 ps (ratio 1.25) while
the shared one ends 2500 → 3600 ps (ratio 1.44), so a 513-row 10T array
resolved to 12.5 ns against 13 ns for the same 6T array. `resolve_timing()`
now floors every variant class at the shared class of the same size. Only
extrapolated sizes move, and only upwards.

## 3. Nominal boundary run on the final table (item 7)

Every 10T row and column class bound reachable with a full transistor array,
with and without a mux, plus the SF write gate and two FF −40 °C sequences that
the V2.1.3 round never ran (`final-N1..N4`, `final-L1/L2`; four ranks per case).
This is the rerun item 7 asked for: the same matrix as
[section 3 of the V2.1.3 record](TIMING_10T_BUDGET_V2_1_3.md), on the V2.1.4
TIME block with its write-request hold latch, scored by the V2.1.4 checks
including `strict_cycle_N_write_enable_quiet` and `_sense_enable_quiet`.

| Case | Array | Op | PVT | Variation | Clock (ns), budget | Result | PRE90→WL50 min (ps) | WL10→PRE90 min (ps) | Request→OUT max (ns) | Deadline margin min (ps) | Boundary W_EN / S_EN peak (mV) | Read-disturb peak / at deadline (mV) |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| b_8x4_10t_mux_SS_sequence | 8x4 10T mux | read&write | SS 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL | passed (1,303 checks) | 550 | 304 | 2.143 | 607 | 6 / 0 | 150 / 0 |
| b_16x16_10t_mux_SS_sequence | 16x16 10T mux | read&write | SS 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL | passed (6,939 checks) | 531 | 358 | 2.238 | 512 | 6 / 0 | 153 / 0 |
| b_16x16_10t_SS_sequence | 16x16 10T | read&write | SS 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL | passed (6,939 checks) | 533 | 358 | 2.171 | 579 | 5 / 0 | 155 / 0 |
| b_32x16_10t_SS_sequence | 32x16 10T | read&write | SS 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL | passed (11,419 checks) | 539 | 359 | 2.233 | 517 | 5 / 0 | 156 / 0 |
| b_32x16_10t_mux_SS_sequence | 32x16 10T mux | read&write | SS 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL | passed (11,419 checks) | 538 | 358 | 2.308 | 442 | 4 / 0 | 154 / 1 |
| b_16x32_10t_mux_SS_sequence | 16x32 10T mux | read&write | SS 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL | passed (13,643 checks) | 530 | 369 | 2.288 | 462 | 6 / 0 | 153 / 0 |
| b_8x4_10t_FF_cold_sequence | 8x4 10T | read&write | FF 1.1 V -40 °C | nominal | 5 SRAM_10T_CELL | passed (1,303 checks) | 151 | 79 | 0.798 | 1,952 | 6 / 1 | 175 / 0 |
| b_8x8_10t_mux_FF_cold_sequence | 8x8 10T mux | read&write | FF 1.1 V -40 °C | nominal | 5 SRAM_10T_CELL | passed (2,435 checks) | 150 | 85 | 0.819 | 1,931 | 7 / 1 | 173 / 0 |
| b_64x16_10t_mux_SS_read | 64x16 10T mux | read | SS 0.9 V 125 °C | nominal | 5.5 SRAM_10T_CELL | passed (3,394 checks) | 555 | 364 | 2.496 | 529 |  | 157 / 2 |
| b_64x16_10t_SS_read | 64x16 10T | read | SS 0.9 V 125 °C | nominal | 5.5 SRAM_10T_CELL | passed (3,394 checks) | 555 | 366 | 2.398 | 627 |  | 158 / 1 |
| b_128x8_10t_mux_SS_read | 128x8 10T mux | read | SS 0.9 V 125 °C | nominal | 6 SRAM_10T_CELL | passed (3,434 checks) | 600 | 360 | 2.862 | 438 |  | 159 / 12 |
| b_128x8_10t_SS_read | 128x8 10T | read | SS 0.9 V 125 °C | nominal | 6 SRAM_10T_CELL | passed (3,434 checks) | 600 | 359 | 2.704 | 596 |  | 159 / 10 |
| b_64x16_10t_mux_SF_write | 64x16 10T mux | write | SF 0.9 V 125 °C | nominal | 5.5 SRAM_10T_CELL | passed (3,569 checks) | 492 | 317 |  |  |  |  |
| b_256x4_10t_mux_SS_read | 256x4 10T mux | read | SS 0.9 V 125 °C | nominal | 8 SRAM_10T_CELL | passed (3,646 checks) | 695 | 351 | 3.626 | 774 |  | 160 / 25 |
| b_256x4_10t_SS_read | 256x4 10T | read | SS 0.9 V 125 °C | nominal | 8 SRAM_10T_CELL | passed (3,646 checks) | 695 | 351 | 3.404 | 996 |  | 161 / 23 |
| b_8x64_10t_mux_SS_read | 8x64 10T mux | read | SS 0.9 V 125 °C | nominal | 5.5 SRAM_10T_CELL | passed (2,274 checks) | 534 | 380 | 2.280 | 745 |  | 150 / 0 |
| b_8x64_10t_SS_read | 8x64 10T | read | SS 0.9 V 125 °C | nominal | 5.5 SRAM_10T_CELL | passed (2,274 checks) | 534 | 381 | 2.204 | 821 |  | 154 / 0 |
| b_8x128_10t_mux_SS_read | 8x128 10T mux | read | SS 0.9 V 125 °C | nominal | 6.5 SRAM_10T_CELL | passed (4,514 checks) | 570 | 391 | 2.444 | 1,131 |  | 148 / 0 |
| b_512x4_10t_mux_SS_read | 512x4 10T mux | read | SS 0.9 V 125 °C | nominal | 10 SRAM_10T_CELL | passed (1,116 checks) | 996 | 362 | 5.025 | 475 |  | 161 / 75 |
| b_512x4_10t_SS_read | 512x4 10T | read | SS 0.9 V 125 °C | nominal | 10 SRAM_10T_CELL | passed (1,116 checks) | 996 | 361 | 4.720 | 780 |  | 161 / 74 |

Every case passes. Read-output margins against the V2.1.3 record, same cases,
same clocks (V2.1.3 → V2.1.5): 8x4 mux sequence 548 → 607 ps, 16x16 mux
sequence 449 → 512 ps, 16x16 sequence 521 → 579 ps, 32x16 mux sequence
368 → 442 ps, 32x16 sequence 449 → 517 ps, 16x32 mux sequence 397 → 462 ps,
64x16 mux read 427 → 529 ps, 64x16 read 535 → 627 ps, 128x8 mux read
282 → 438 ps, 256x4 mux read 529 → 774 ps, 256x4 read 773 → 996 ps, 8x64 mux
read 683 → 745 ps, 8x64 read 767 → 821 ps, 8x128 mux read 1,073 → 1,131 ps.
No class bound is below 438 ps, so the 250 ps rule holds everywhere with room
to spare; a later round may spend that with its own evidence.

The 512-row bound is the class that moved. At the new 10 ns class the mux read
keeps 475 ps of read-output margin and settles its target storage node to
0.075 V at the deadline (0.090 V tolerance), 5.186 ns after the access request
against the 5.5 ns deadline; without a mux, 780 ps and 0.074 V. The V2.1.3 cell
at 11 ns left 0.100 V and failed two storage checks (`probe-L1`).

The two FF 1.1 V / −40 °C sequences (8x4 10T, 8x8 10T with a mux) are new in
this round and close the part of item 7 that the V2.1.4 TIME change created:
both pass every check, with boundary write-enable peaks of 6 and 7 mV and
sense-enable peaks of 1 mV — the same magnitudes the 6T record measured after
the hold latch, and far from the 0.38 to 0.72 V spikes the latch removed. The
64x16 mux SF write passes its capture, drive, release and retention checks at
5.5 ns.

Solver: no numerical event and no retry in the nominal run; the non-fatal
minimum-step warning appears in `b_8x4_10t_mux_SS_sequence` and
`b_512x4_10t_mux_SS_read`.

## 4. Mismatch at the class bounds (items 5 and 7)

Single-rank per-device samples on the released sources (`pd-Q1..Q10`), 5 %
relative sigma on `vth0`, `u0` and `voff` of every MOS. The V2.1.3 seeds are
repeated on the new cell (32- and 64-row bounds, the ten-seed 8x4 pilot), the
128-row bound gets the three seeds item 5 asked for with and without a mux, and
the new FF −40 °C sequence gets three.

| Case | Array | Op | PVT | Variation | Clock (ns), budget | Result | PRE90→WL50 min (ps) | WL10→PRE90 min (ps) | Request→OUT max (ns) | Deadline margin min (ps) | Boundary W_EN / S_EN peak (mV) | Read-disturb peak / at deadline (mV) |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| b_128x8_10t_SS_read_pd_s1 | 128x8 10T | read | SS 0.9 V 125 °C | per-device 20261001 | 6 SRAM_10T_CELL | passed (3,434 checks) | 601 | 357 | 2.826 | 474 |  | 150 / 12 |
| b_128x8_10t_SS_read_pd_s2 | 128x8 10T | read | SS 0.9 V 125 °C | per-device 20261002 | 6 SRAM_10T_CELL | passed (3,434 checks) | 559 | 362 | 2.652 | 648 |  | 152 / 13 |
| b_128x8_10t_SS_read_pd_s3 | 128x8 10T | read | SS 0.9 V 125 °C | per-device 20261003 | 6 SRAM_10T_CELL | passed (3,434 checks) | 609 | 381 | 2.805 | 495 |  | 159 / 12 |
| b_128x8_10t_mux_SS_read_pd_s1 | 128x8 10T mux | read | SS 0.9 V 125 °C | per-device 20261001 | 6 SRAM_10T_CELL | passed (3,434 checks) | 614 | 398 | 2.757 | 543 |  | 174 / 10 |
| b_128x8_10t_mux_SS_read_pd_s2 | 128x8 10T mux | read | SS 0.9 V 125 °C | per-device 20261002 | 6 SRAM_10T_CELL | passed (3,434 checks) | 602 | 405 | 3.034 | 266 |  | 155 / 14 |
| b_128x8_10t_mux_SS_read_pd_s3 | 128x8 10T mux | read | SS 0.9 V 125 °C | per-device 20261003 | 6 SRAM_10T_CELL | passed (3,434 checks) | 608 | 366 | 2.955 | 345 |  | 162 / 10 |
| b_32x16_10t_mux_SS_sequence_pd_s1 | 32x16 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261001 | 5 SRAM_10T_CELL | passed (11,419 checks) | 530 | 361 | 2.343 | 407 | 4 / 0 | 166 / 1 |
| b_32x16_10t_mux_SS_sequence_pd_s2 | 32x16 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261002 | 5 SRAM_10T_CELL | passed (11,419 checks) | 520 | 355 | 2.362 | 388 | 4 / 0 | 164 / 2 |
| b_32x16_10t_mux_SS_sequence_pd_s3 | 32x16 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261003 | 5 SRAM_10T_CELL | passed (11,419 checks) | 524 | 354 | 2.390 | 360 | 4 / 0 | 164 / 1 |
| b_64x16_10t_mux_SS_read_pd_s1 | 64x16 10T mux | read | SS 0.9 V 125 °C | per-device 20261001 | 5.5 SRAM_10T_CELL | passed (3,394 checks) | 553 | 383 | 2.516 | 509 |  | 150 / 2 |
| b_64x16_10t_mux_SS_read_pd_s2 | 64x16 10T mux | read | SS 0.9 V 125 °C | per-device 20261002 | 5.5 SRAM_10T_CELL | passed (3,394 checks) | 578 | 372 | 2.585 | 440 |  | 161 / 3 |
| b_64x16_10t_mux_SS_read_pd_s3 | 64x16 10T mux | read | SS 0.9 V 125 °C | per-device 20261003 | 5.5 SRAM_10T_CELL | passed (3,394 checks) | 567 | 363 | 2.483 | 542 |  | 182 / 2 |
| b_8x4_10t_FF_cold_sequence_pd_s1 | 8x4 10T | read&write | FF 1.1 V -40 °C | per-device 20261001 | 5 SRAM_10T_CELL | passed (1,303 checks) | 153 | 81 | 0.798 | 1,952 | 6 / 1 | 181 / 0 |
| b_8x4_10t_FF_cold_sequence_pd_s2 | 8x4 10T | read&write | FF 1.1 V -40 °C | per-device 20261002 | 5 SRAM_10T_CELL | passed (1,303 checks) | 148 | 80 | 0.796 | 1,954 | 6 / 1 | 183 / 0 |
| b_8x4_10t_FF_cold_sequence_pd_s3 | 8x4 10T | read&write | FF 1.1 V -40 °C | per-device 20261003 | 5 SRAM_10T_CELL | passed (1,303 checks) | 149 | 75 | 0.798 | 1,952 | 6 / 1 | 177 / 0 |
| b_8x4_10t_mux_SS_sequence_pd_s1 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261001 | 5 SRAM_10T_CELL | passed (1,303 checks) | 556 | 309 | 2.158 | 592 | 6 / 0 | 140 / 0 |
| b_8x4_10t_mux_SS_sequence_pd_s2 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261002 | 5 SRAM_10T_CELL | passed (1,303 checks) | 527 | 326 | 2.085 | 665 | 6 / 0 | 150 / 0 |
| b_8x4_10t_mux_SS_sequence_pd_s3 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261003 | 5 SRAM_10T_CELL | passed (1,303 checks) | 546 | 302 | 2.154 | 596 | 5 / 0 | 155 / 0 |
| b_8x4_10t_mux_SS_sequence_pd_s4 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261004 | 5 SRAM_10T_CELL | passed (1,303 checks) | 567 | 330 | 2.176 | 574 | 6 / 0 | 162 / 0 |
| b_8x4_10t_mux_SS_sequence_pd_s5 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261005 | 5 SRAM_10T_CELL | passed (1,303 checks) | 568 | 312 | 2.175 | 575 | 6 / 0 | 168 / 0 |
| b_8x4_10t_mux_SS_sequence_pd_s6 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261006 | 5 SRAM_10T_CELL | passed (1,303 checks) | 550 | 284 | 2.184 | 566 | 6 / 0 | 148 / 0 |
| b_8x4_10t_mux_SS_sequence_pd_s7 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261007 | 5 SRAM_10T_CELL | passed (1,303 checks) | 539 | 278 | 2.157 | 593 | 6 / 0 | 156 / 0 |
| b_8x4_10t_mux_SS_sequence_pd_s8 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261008 | 5 SRAM_10T_CELL | passed (1,303 checks) | 524 | 319 | 2.094 | 656 | 6 / 0 | 147 / 0 |
| b_8x4_10t_mux_SS_sequence_pd_s9 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261009 | 5 SRAM_10T_CELL | passed (1,303 checks) | 550 | 314 | 2.158 | 592 | 6 / 0 | 166 / 0 |
| b_8x4_10t_mux_SS_sequence_pd_s10 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261010 | 5 SRAM_10T_CELL | passed (1,303 checks) | 533 | 289 | 2.138 | 612 | 5 / 0 | 154 / 0 |

All 25 seeds pass. Read-output margin ranges: 360–407 ps at the 32-row bound
(three seeds), 440–542 ps at the 64-row bound, 266–543 ps at the 128-row bound
with a mux and 474–648 ps without one, and 566–665 ps over the ten 8x4 pilot
seeds. The 128x8 mux seed 20261002 is the tightest sample of the round at
266 ps, still above the 250 ps rule; on the V2.1.3 cell the same bound had
282 ps *nominally* and no seeds at all, which is what item 5 was about.
The three FF −40 °C seeds keep write-enable peaks at 6 mV and sense-enable at
1 mV.

Ten seeds are a pilot, not a yield statement: 0 of 10 failures bounds the rate
only to 26 % at 95 % confidence, and the statistical question belongs to the
yield-estimator brief of the [qualification scope](../plans/V2_1_2_QUALIFICATION_SCOPE.md).

Solver: one retry (`b_128x8_10t_mux_SS_read_pd_s2`, which then passed every
check) and six runs with the non-fatal minimum-step warning; no DC
operating-point failure and no `Time step too small` abort.

## 5. Evidence

Raw decks, waveforms, solver logs, per-attempt checkpoints and source archives
are local under ignored `outputs/validation/V2.1.5-10t/`:

| Directory | Contents |
|---|---|
| `probe-L1/L2/L3-queue` | 512x4 10T mux reads at the probe 11 ns class, one per pull-down candidate |
| `probe-M1/M2-queue` | The rejected 1.8x candidate at 9.5 ns and 10 ns |
| `probe-M4-queue` | The adopted 1.4x candidate at 10.5 ns (the fallback class, not needed) |
| `probe-S1/S2/S3-queue`, `probe-S1b/S2b/S3b-queue` | Candidate sanity: 16x16 sequence, 128x8 mux read, 64x16 mux SF write, 16x16 mux FF −40 °C sequence |
| `final-N1..N4-queue`, `final-L1/L2-queue` | The nominal boundary matrix of section 3 |
| `pd-Q1..Q10-queue` | The mismatch seeds of section 4 |
| `deck-comparison/` | Deck regeneration against a detached `acec3be` worktree |
| `timing_probe_10t_*.json` | The unreleased probe class tables |
| `snm_candidates.json` | The hold/read/write SNM sweep of the three candidates |
| `read_disturb_*.json` | Read-disturb measurements |
| `table_probe.md`, `table_final.md`, `record-summary.json` | The assembled record |
| `gen_probe_cases.py`, `gen_final_cases.py`, `run_probe*.sh`, `run_final.sh`, `apply_release.py`, `compare_decks.py`, `assemble_record.py` | The campaign scripts |

The machine-readable [record](TIMING_10T_BUDGET_V2_1_5.json) carries every
probe and final case with its identity, checks and measurements, the cell and
timing changes, the deck comparisons and the equivalent-model rows.

Measurements come from `dev/v213_sense_timing.py` (sense and output timing),
`dev/v215_read_disturb.py` (the storage-node bump) and `dev/v215_10t_snm.py`
(the DC margins), all read-only on the retained traces.
