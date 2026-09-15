# V2.1.3 timing budget for 10T cells: evidence run and ten-seed pilot

Executed September 14 and 15, 2026 on the V2.1.3 sources (base commit
`e26a7ece3e43120904ed74d0a8bc2b2323f4d5a1` plus the table, resolver, test and
documentation changes recorded in the [changelog](../CHANGELOG.md); every
queue archives the exact sources it ran). Illustrative wires, V2.0.9 driver
classes and the V2.1.3 lookup table throughout; nominal seed 20260913,
maximum step 20 ps, at most eight simulator ranks at any time; per-device
samples on one rank each with 5 % relative sigma on `vth0`, `u0` and `voff`
of every MOS. This is functional screening at SS 0.9 V / 125 °C (SF for the
write), not PVT/mismatch or extracted-metal qualification; no record is
promoted to `sizing_table.json`.

**Outcome: on the final table 56 of 58 attempts pass, 146,952 of 160,814 checks: all 18 nominal boundary cases, all 37 10T and 6T pilot and class-bound seeds, and 1 of 3 6T-mux control seeds at the unchanged shared class. Every failure in this record is electrical: no solver aborted (0 numerical events), no retry ran, and the only solver message is the non-fatal minimum-step warning in 11 runs.**

PRE90→WL50 and WL10→PRE90 are as in the [V2.1.2 follow-up](TIMING_FOLLOWUP_V2_1_2.md).
"Request→OUT" is the local read output reaching 90 % (read 1) or 10 %
(read 0) of VDD, measured from the access request at 0.65 T, worst read cycle
of a sequence; "deadline margin" is its distance to the 1.2 T data deadline.
The independent validator samples the output and the storage nodes 0.015 T
before the deadline, so a deadline margin below about 60 ps fails its output
check. The measurements come from the retained `.prn` traces (`S_EN`,
`XSENSEAMP_*:EN_end`, `OUT`, `WL_EN`, replica bitline).

## 1. Why the shared class was exhausted, and why two 10T ladders were rejected

The V2.1.2 follow-up left 10T cells with a column mux failing the shared
1600 ps class (4 ns) at 16 rows nominal and at 8 rows under mismatch. The
first V2.1.3 pass adopted the recorded proposal, one class step (200 ps) for
10T with a mux only, and ran two controls at the shared class. Both controls
and the first boundaries changed the design:

- The 6T sequence with a mux passes the shared class at 16x16, but with 74 ps
  at the deadline: the mux pass gate costs about 140 ps of the access window.
- The 10T sequence without a mux fails the shared class at 16x16 (39 ps at
  the deadline, past the validator's sample): the 10T read port alone costs
  about 170 ps at 16 rows, so the budget must cover 10T with and without a mux.
- At a flat 200 ps above the shared classes the 128x8 10T mux read fails at
  5.5 ns (40 ps at the deadline). The 10T replica discharge is about 1 ps per
  row slower than 6T (WL_EN→replica-50 % is 290 + 4.7·rows ps against
  230 + 3.7·rows ps), so the penalty grows with height: about 265 ps at
  16 rows and 440 ps at 128 rows.
- With row budgets 1800/2000/2400/3200/4400 ps every nominal case passes,
  but the 32-row class bound keeps only 125 ps at 32x16 and 4.5 ns, and the
  three mismatch seeds there leave 76, 81 and 15 ps: seed 3 fails all four
  read outputs. The 64-row bound keeps 183 ps nominal and 124 ps on one seed
  at 5 ns. Mismatch moves the output by up to about 110 ps at 5 % sigma.
- At 512 rows the 10T mux read at 11 ns sets its output 560 ps early but fails
  the storage-node checks of the target row: the 10T read-disturb bump on the
  low storage node peaks at 0.196 V (0.131 V for 6T) and decays only as the
  bitline discharges, and a 512-row bitline leaves Q at 0.108 V at the
  deadline against the 0.1 VDD tolerance (0.06 V at 256 rows, which passes).

Rule adopted for the 10T ladder: at least 250 ps between the local read
output and the 1.2 T deadline at every class bound, nominal SS 0.9 V / 125 °C,
and the storage nodes inside the 0.1 VDD tolerance at the deadline. Every
rejected attempt is preserved under `budget-A/B`, `final-A/B/L` and
`pd-Q1..Q4` (`why-summary.json`):

| Case | Array | Op | PVT | Variation | Clock (ns), budget | Result | PRE90→WL50 min (ps) | WL10→PRE90 min (ps) | Request→OUT max (ns) | Deadline margin min (ps) |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| c_16x16_6t_mux_SS_sequence | 16x16 6T mux | read&write | SS 0.9 V 125 °C | nominal | 4 shared | passed (6,931 checks) | 533 | 358 | 2.126 | 74 |
| c_16x16_10t_SS_sequence | 16x16 10T | read&write | SS 0.9 V 125 °C | nominal | 4 shared | failed (2 of 6,931 checks) | 533 | 358 | 2.161 | 39 |
| b_8x4_10t_mux_SS_sequence | 8x4 10T mux | read&write | SS 0.9 V 125 °C | nominal | 4.5 SRAM_10T_CELL/mux | passed (1,295 checks) | 550 | 304 | 2.168 | 307 |
| b_64x16_10t_mux_SS_read | 64x16 10T mux | read | SS 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL/mux | passed (388 checks) | 554 | 366 | 2.567 | 183 |
| b_128x8_10t_mux_SS_read | 128x8 10T mux | read | SS 0.9 V 125 °C | nominal | 5.5 SRAM_10T_CELL/mux | failed (1 of 404 checks) | 601 | 359 | 2.985 | 40 |
| c_16x16_6t_mux_SS_sequence | 16x16 6T mux | read&write | SS 0.9 V 125 °C | nominal | 4 shared | passed (6,931 checks) | 533 | 358 | 2.126 | 74 |
| b_8x4_10t_mux_SS_sequence | 8x4 10T mux | read&write | SS 0.9 V 125 °C | nominal | 4.5 SRAM_10T_CELL | passed (1,295 checks) | 550 | 304 | 2.168 | 307 |
| b_16x16_10t_mux_SS_sequence | 16x16 10T mux | read&write | SS 0.9 V 125 °C | nominal | 4.5 SRAM_10T_CELL | passed (6,931 checks) | 530 | 358 | 2.266 | 209 |
| b_16x16_10t_SS_sequence | 16x16 10T | read&write | SS 0.9 V 125 °C | nominal | 4.5 SRAM_10T_CELL | passed (6,931 checks) | 530 | 359 | 2.195 | 280 |
| b_32x16_10t_mux_SS_sequence | 32x16 10T mux | read&write | SS 0.9 V 125 °C | nominal | 4.5 SRAM_10T_CELL | passed (11,411 checks) | 539 | 358 | 2.350 | 125 |
| b_32x16_10t_SS_sequence | 32x16 10T | read&write | SS 0.9 V 125 °C | nominal | 4.5 SRAM_10T_CELL | passed (11,411 checks) | 539 | 358 | 2.266 | 209 |
| b_16x32_10t_mux_SS_sequence | 16x32 10T mux | read&write | SS 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL | passed (13,635 checks) | 530 | 369 | 2.353 | 397 |
| b_64x16_10t_mux_SS_read | 64x16 10T mux | read | SS 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL | passed (388 checks) | 554 | 366 | 2.567 | 183 |
| b_64x16_10t_SS_read | 64x16 10T | read | SS 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL | passed (388 checks) | 554 | 366 | 2.456 | 294 |
| b_128x8_10t_mux_SS_read | 128x8 10T mux | read | SS 0.9 V 125 °C | nominal | 6 SRAM_10T_CELL | passed (404 checks) | 600 | 360 | 3.018 | 282 |
| b_256x4_10t_mux_SS_read | 256x4 10T mux | read | SS 0.9 V 125 °C | nominal | 8 SRAM_10T_CELL | passed (604 checks) | 695 | 350 | 3.871 | 529 |
| b_256x4_10t_SS_read | 256x4 10T | read | SS 0.9 V 125 °C | nominal | 8 SRAM_10T_CELL | passed (604 checks) | 695 | 351 | 3.627 | 773 |
| b_8x64_10t_mux_SS_read | 8x64 10T mux | read | SS 0.9 V 125 °C | nominal | 5.5 SRAM_10T_CELL | passed (948 checks) | 534 | 380 | 2.342 | 683 |
| b_8x64_10t_SS_read | 8x64 10T | read | SS 0.9 V 125 °C | nominal | 5.5 SRAM_10T_CELL | passed (948 checks) | 534 | 380 | 2.258 | 767 |
| b_8x128_10t_mux_SS_read | 8x128 10T mux | read | SS 0.9 V 125 °C | nominal | 6.5 SRAM_10T_CELL | passed (1,844 checks) | 570 | 391 | 2.502 | 1,073 |
| b_64x16_10t_mux_SF_write | 64x16 10T mux | write | SF 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL | passed (563 checks) | 492 | 318 |  |  |
| b_512x4_10t_mux_SS_read | 512x4 10T mux | read | SS 0.9 V 125 °C | nominal | 11 SRAM_10T_CELL | failed (4 of 1,116 checks) | 996 | 360 | 5.490 | 560 |
| b_32x16_10t_mux_SS_sequence_pd_s1 | 32x16 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261001 | 4.5 SRAM_10T_CELL | passed (11,411 checks) | 560 | 344 | 2.399 | 76 |
| b_32x16_10t_mux_SS_sequence_pd_s2 | 32x16 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261002 | 4.5 SRAM_10T_CELL | passed (11,411 checks) | 561 | 373 | 2.394 | 81 |
| b_32x16_10t_mux_SS_sequence_pd_s3 | 32x16 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261003 | 4.5 SRAM_10T_CELL | failed (4 of 11,411 checks) | 545 | 347 | 2.460 | 15 |
| b_64x16_10t_mux_SS_read_pd_s1 | 64x16 10T mux | read | SS 0.9 V 125 °C | per-device 20261001 | 5 SRAM_10T_CELL | passed (388 checks) | 554 | 364 | 2.498 | 252 |
| b_64x16_10t_mux_SS_read_pd_s2 | 64x16 10T mux | read | SS 0.9 V 125 °C | per-device 20261002 | 5 SRAM_10T_CELL | passed (388 checks) | 567 | 363 | 2.626 | 124 |

## 2. Budget adopted

`timing_lookup.json` (`v2.1.3-timing-2`) carries one `variants` entry for
`SRAM_10T_CELL`, with or without a column mux: row budgets
2000/2200/2400/3200/5600 ps for ≤32/64/128/256/512 rows (5, 5.5, 6, 8 and
14 ns with the 25 % margin) and column budgets 200 ps above the shared ladder
(1800/1800/1800/2000/2200/2600/3000/3400 ps). The shared classes, every 6T
deck, the driver classes, the TIME circuit and the checks are unchanged
(6T decks compared byte-identical against a detached `e26a7ec` worktree; 10T
decks differ only in stimulus, `.TRAN` and measurement times). The 512-row
value is set by the storage-node tolerance, not by the output: 14 ns gives a
512-row bitline about 5.6 ns of wordline time. `ArrayTiming.budget` records
`SRAM_10T_CELL` or `shared` in every run record.

## 3. Nominal boundary run on the final table

Every 10T row and column class bound reachable with a full transistor array,
with and without a mux, plus the 6T-mux control at the shared class and one
SF write (`final2-A/B/L`, four ranks per case):

| Case | Array | Op | PVT | Variation | Clock (ns), budget | Result | PRE90→WL50 min (ps) | WL10→PRE90 min (ps) | Request→OUT max (ns) | Deadline margin min (ps) |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| c_16x16_6t_mux_SS_sequence | 16x16 6T mux | read&write | SS 0.9 V 125 °C | nominal | 4 shared | passed (6,931 checks) | 533 | 358 | 2.126 | 74 |
| b_8x4_10t_mux_SS_sequence | 8x4 10T mux | read&write | SS 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL | passed (1,295 checks) | 549 | 304 | 2.202 | 548 |
| b_16x16_10t_mux_SS_sequence | 16x16 10T mux | read&write | SS 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL | passed (6,931 checks) | 533 | 358 | 2.301 | 449 |
| b_16x16_10t_SS_sequence | 16x16 10T | read&write | SS 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL | passed (6,931 checks) | 529 | 358 | 2.229 | 521 |
| b_32x16_10t_mux_SS_sequence | 32x16 10T mux | read&write | SS 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL | passed (11,411 checks) | 538 | 358 | 2.382 | 368 |
| b_32x16_10t_SS_sequence | 32x16 10T | read&write | SS 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL | passed (11,411 checks) | 539 | 359 | 2.301 | 449 |
| b_16x32_10t_mux_SS_sequence | 16x32 10T mux | read&write | SS 0.9 V 125 °C | nominal | 5 SRAM_10T_CELL | passed (13,635 checks) | 530 | 369 | 2.353 | 397 |
| b_64x16_10t_mux_SS_read | 64x16 10T mux | read | SS 0.9 V 125 °C | nominal | 5.5 SRAM_10T_CELL | passed (388 checks) | 555 | 366 | 2.598 | 427 |
| b_64x16_10t_SS_read | 64x16 10T | read | SS 0.9 V 125 °C | nominal | 5.5 SRAM_10T_CELL | passed (388 checks) | 555 | 366 | 2.490 | 535 |
| b_128x8_10t_mux_SS_read | 128x8 10T mux | read | SS 0.9 V 125 °C | nominal | 6 SRAM_10T_CELL | passed (404 checks) | 600 | 360 | 3.018 | 282 |
| b_256x4_10t_mux_SS_read | 256x4 10T mux | read | SS 0.9 V 125 °C | nominal | 8 SRAM_10T_CELL | passed (604 checks) | 695 | 350 | 3.871 | 529 |
| b_256x4_10t_SS_read | 256x4 10T | read | SS 0.9 V 125 °C | nominal | 8 SRAM_10T_CELL | passed (604 checks) | 695 | 351 | 3.627 | 773 |
| b_8x64_10t_mux_SS_read | 8x64 10T mux | read | SS 0.9 V 125 °C | nominal | 5.5 SRAM_10T_CELL | passed (948 checks) | 534 | 380 | 2.342 | 683 |
| b_8x64_10t_SS_read | 8x64 10T | read | SS 0.9 V 125 °C | nominal | 5.5 SRAM_10T_CELL | passed (948 checks) | 534 | 380 | 2.258 | 767 |
| b_8x128_10t_mux_SS_read | 8x128 10T mux | read | SS 0.9 V 125 °C | nominal | 6.5 SRAM_10T_CELL | passed (1,844 checks) | 570 | 391 | 2.502 | 1,073 |
| b_64x16_10t_mux_SF_write | 64x16 10T mux | write | SF 0.9 V 125 °C | nominal | 5.5 SRAM_10T_CELL | passed (563 checks) | 492 | 317 |  |  |
| b_512x4_10t_mux_SS_read | 512x4 10T mux | read | SS 0.9 V 125 °C | nominal | 14 SRAM_10T_CELL | passed (1,116 checks) | 995 | 360 | 5.693 | 2,007 |
| b_512x4_10t_SS_read | 512x4 10T | read | SS 0.9 V 125 °C | nominal | 14 SRAM_10T_CELL | passed (1,116 checks) | 995 | 362 | 5.370 | 2,330 |

Every case passes. The 10T cases keep at least 282 ps at the 1.2 T deadline (the 128x8 mux read at 6 ns; every other 10T class bound keeps 368 ps or more), and the 6T-mux control keeps its 74 ps at the unchanged shared class. The 512-row reads keep 2.0 and 2.3 ns of output margin, and their target storage nodes settle to 0.051 V (mux) and 0.049 V (no mux) at the deadline, inside the 0.1 VDD tolerance that 11 ns missed (0.108 V). The 64x16 SF write passes its capture, drive, release and retention checks at 5.5 ns. Runs with the non-fatal minimum-step warning: b_32x16_10t_mux_SS_sequence, b_16x32_10t_mux_SS_sequence, b_512x4_10t_mux_SS_read.

## 4. Mismatch at the class bounds and the ten-seed pilot

Single-rank per-device samples on the final table (`pd2-Q1..Q8`). The
32-row and 64-row bounds carry three seeds each; the four Phase 5 pilot
cases of the V2.1.2 follow-up are expanded to ten seeds (20261001 to
20261010; the 6T cases keep their V2.1.2 seeds 1 to 3 and add 4 to 10, the
10T mux case runs all ten at its new clock); the 6T-mux 16x16 sequence gets
three control seeds at the shared class because of its 74 ps nominal margin.

| Case | Array | Op | PVT | Variation | Clock (ns), budget | Result | PRE90→WL50 min (ps) | WL10→PRE90 min (ps) | Request→OUT max (ns) | Deadline margin min (ps) |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| b_32x16_10t_mux_SS_sequence_pd_s1 | 32x16 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261001 | 5 SRAM_10T_CELL | passed (11,411 checks) | 559 | 345 | 2.431 | 319 |
| b_32x16_10t_mux_SS_sequence_pd_s2 | 32x16 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261002 | 5 SRAM_10T_CELL | passed (11,411 checks) | 561 | 372 | 2.427 | 323 |
| b_32x16_10t_mux_SS_sequence_pd_s3 | 32x16 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261003 | 5 SRAM_10T_CELL | passed (11,411 checks) | 545 | 347 | 2.491 | 259 |
| b_64x16_10t_mux_SS_read_pd_s1 | 64x16 10T mux | read | SS 0.9 V 125 °C | per-device 20261001 | 5.5 SRAM_10T_CELL | passed (388 checks) | 553 | 364 | 2.531 | 494 |
| b_64x16_10t_mux_SS_read_pd_s2 | 64x16 10T mux | read | SS 0.9 V 125 °C | per-device 20261002 | 5.5 SRAM_10T_CELL | passed (388 checks) | 567 | 362 | 2.659 | 366 |
| b_64x16_10t_mux_SS_read_pd_s3 | 64x16 10T mux | read | SS 0.9 V 125 °C | per-device 20261003 | 5.5 SRAM_10T_CELL | passed (388 checks) | 577 | 347 | 2.545 | 480 |
| p5x_8x4_10t_mux_SS_sequence_pd_s1 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261001 | 5 SRAM_10T_CELL | passed (1,295 checks) | 556 | 312 | 2.280 | 470 |
| p5x_8x4_10t_mux_SS_sequence_pd_s2 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261002 | 5 SRAM_10T_CELL | passed (1,295 checks) | 529 | 310 | 2.237 | 513 |
| p5x_8x4_10t_mux_SS_sequence_pd_s3 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261003 | 5 SRAM_10T_CELL | passed (1,295 checks) | 552 | 296 | 2.228 | 522 |
| p5x_8x4_10t_mux_SS_sequence_pd_s4 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261004 | 5 SRAM_10T_CELL | passed (1,295 checks) | 543 | 304 | 2.205 | 545 |
| p5x_8x4_10t_mux_SS_sequence_pd_s5 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261005 | 5 SRAM_10T_CELL | passed (1,295 checks) | 571 | 301 | 2.209 | 541 |
| p5x_8x4_10t_mux_SS_sequence_pd_s6 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261006 | 5 SRAM_10T_CELL | passed (1,295 checks) | 526 | 300 | 2.177 | 573 |
| p5x_8x4_10t_mux_SS_sequence_pd_s7 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261007 | 5 SRAM_10T_CELL | passed (1,295 checks) | 560 | 275 | 2.174 | 576 |
| p5x_8x4_10t_mux_SS_sequence_pd_s8 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261008 | 5 SRAM_10T_CELL | passed (1,295 checks) | 557 | 307 | 2.216 | 534 |
| p5x_8x4_10t_mux_SS_sequence_pd_s9 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261009 | 5 SRAM_10T_CELL | passed (1,295 checks) | 563 | 311 | 2.231 | 519 |
| p5x_8x4_10t_mux_SS_sequence_pd_s10 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | per-device 20261010 | 5 SRAM_10T_CELL | passed (1,295 checks) | 571 | 323 | 2.224 | 526 |
| p5x_16x16_SS_read_pd_s4 | 16x16 6T | read | SS 0.9 V 125 °C | per-device 20261004 | 4 shared | passed (994 checks) | 524 | 359 | 2.078 | 122 |
| p5x_16x16_SS_read_pd_s5 | 16x16 6T | read | SS 0.9 V 125 °C | per-device 20261005 | 4 shared | passed (994 checks) | 542 | 371 | 2.053 | 147 |
| p5x_16x16_SS_read_pd_s6 | 16x16 6T | read | SS 0.9 V 125 °C | per-device 20261006 | 4 shared | passed (994 checks) | 544 | 384 | 2.042 | 158 |
| p5x_16x16_SS_read_pd_s7 | 16x16 6T | read | SS 0.9 V 125 °C | per-device 20261007 | 4 shared | passed (994 checks) | 536 | 378 | 2.007 | 193 |
| p5x_16x16_SS_read_pd_s8 | 16x16 6T | read | SS 0.9 V 125 °C | per-device 20261008 | 4 shared | passed (994 checks) | 522 | 337 | 2.023 | 177 |
| p5x_16x16_SS_read_pd_s9 | 16x16 6T | read | SS 0.9 V 125 °C | per-device 20261009 | 4 shared | passed (994 checks) | 540 | 350 | 2.058 | 142 |
| p5x_16x16_SS_read_pd_s10 | 16x16 6T | read | SS 0.9 V 125 °C | per-device 20261010 | 4 shared | passed (994 checks) | 551 | 340 | 2.063 | 137 |
| p5x_16x16_SF_write_pd_s4 | 16x16 6T | write | SF 0.9 V 125 °C | per-device 20261004 | 4 shared | passed (1,169 checks) | 443 | 308 |  |  |
| p5x_16x16_SF_write_pd_s5 | 16x16 6T | write | SF 0.9 V 125 °C | per-device 20261005 | 4 shared | passed (1,169 checks) | 457 | 334 |  |  |
| p5x_16x16_SF_write_pd_s6 | 16x16 6T | write | SF 0.9 V 125 °C | per-device 20261006 | 4 shared | passed (1,169 checks) | 472 | 320 |  |  |
| p5x_16x16_SF_write_pd_s7 | 16x16 6T | write | SF 0.9 V 125 °C | per-device 20261007 | 4 shared | passed (1,169 checks) | 489 | 327 |  |  |
| p5x_16x16_SF_write_pd_s8 | 16x16 6T | write | SF 0.9 V 125 °C | per-device 20261008 | 4 shared | passed (1,169 checks) | 443 | 314 |  |  |
| p5x_16x16_SF_write_pd_s9 | 16x16 6T | write | SF 0.9 V 125 °C | per-device 20261009 | 4 shared | passed (1,169 checks) | 470 | 304 |  |  |
| p5x_16x16_SF_write_pd_s10 | 16x16 6T | write | SF 0.9 V 125 °C | per-device 20261010 | 4 shared | passed (1,169 checks) | 457 | 306 |  |  |
| p5x_8x4_FF_cold_sequence_pd_s4 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261004 | 4 shared | passed (1,295 checks) | 149 | 80 | 0.723 | 1,477 |
| p5x_8x4_FF_cold_sequence_pd_s5 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261005 | 4 shared | passed (1,295 checks) | 148 | 79 | 0.726 | 1,474 |
| p5x_8x4_FF_cold_sequence_pd_s6 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261006 | 4 shared | passed (1,295 checks) | 151 | 77 | 0.726 | 1,474 |
| p5x_8x4_FF_cold_sequence_pd_s7 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261007 | 4 shared | passed (1,295 checks) | 151 | 80 | 0.727 | 1,473 |
| p5x_8x4_FF_cold_sequence_pd_s8 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261008 | 4 shared | passed (1,295 checks) | 148 | 80 | 0.724 | 1,476 |
| p5x_8x4_FF_cold_sequence_pd_s9 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261009 | 4 shared | passed (1,295 checks) | 153 | 79 | 0.731 | 1,469 |
| p5x_8x4_FF_cold_sequence_pd_s10 | 8x4 6T | read&write | FF 1.1 V -40 °C | per-device 20261010 | 4 shared | passed (1,295 checks) | 149 | 81 | 0.733 | 1,467 |
| c_16x16_6t_mux_SS_sequence_pd_s1 | 16x16 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261001 | 4 shared | passed (6,931 checks) | 529 | 348 | 2.103 | 97 |
| c_16x16_6t_mux_SS_sequence_pd_s2 | 16x16 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261002 | 4 shared | failed (2 of 6,931 checks) | 552 | 339 | 2.149 | 51 |
| c_16x16_6t_mux_SS_sequence_pd_s3 | 16x16 6T mux | read&write | SS 0.9 V 125 °C | per-device 20261003 | 4 shared | failed (2 of 6,931 checks) | 552 | 362 | 2.143 | 57 |

All 40 scheduled seeds are accounted for. 10T: 3 of 3 at the 32-row bound (259–323 ps), 3 of 3 at the 64-row bound (366–494 ps), 10 of 10 at 8x4 with a mux (470–576 ps), all at their lookup clocks. 6T pilot expansion: 16x16 SS read 7 of 7 (122–193 ps), 16x16 SF write 7 of 7, 8x4 FF cold sequence 7 of 7; with the V2.1.2 seeds 1 to 3 these three cases stand at 10 of 10, and the 8x4 10T mux case at 10 of 10 on its new clock (its V2.1.2 seeds 1 to 3 had failed at 4 ns). The 6T-mux 16x16 control at the shared 4 ns class fails seeds 2 and 3 on the read outputs (51 and 57 ps at the deadline; seed 1 passes with 97 ps): 6T with a column mux needs its own variant and evidence run, which this release does not attempt. The FF −40 °C write-enable spike of the V2.1.2 follow-up recurs at 0.38 to 0.72 V across seeds 4 to 10 with no check failing; it remains an open TIME item. Sigma is 5 % relative; ten seeds are a pilot, not a yield statement (0 of 10 failures bounds the rate only to 26 % at 95 % confidence). Per-device runs with the minimum-step warning: b_32x16_10t_mux_SS_sequence_pd_s3, p5x_8x4_10t_mux_SS_sequence_pd_s4, p5x_8x4_10t_mux_SS_sequence_pd_s5, p5x_8x4_10t_mux_SS_sequence_pd_s6, p5x_8x4_10t_mux_SS_sequence_pd_s7, p5x_8x4_10t_mux_SS_sequence_pd_s8, p5x_8x4_10t_mux_SS_sequence_pd_s10, c_16x16_6t_mux_SS_sequence_pd_s2.

## 5. Evidence

Raw decks, waveforms, solver logs, per-attempt checkpoints and source
archives are local under ignored `outputs/validation/V2.1.3-10t-mux-budget/`
(`final2-A/B/L-queue`, `pd2-Q1..Q8-queue` for the final table; `budget-A/B`,
`final-A/B/L` and `pd-Q1..Q4` for the rejected ladders; `chain.log`;
`*-write-capture.png`). The machine-readable [record](TIMING_10T_BUDGET_V2_1_3.json)
carries every final-table case, identity, metric and measurement, and the
`why` block the rejected attempts. Representative write-capture plots of the
16x16 10T mux sequence, the 32x16 10T sequence and the 64x16 10T mux SF
write were inspected: register capture precedes write enable, precharge is
off before the wordline, the driver pulls the bitline, Q meets the deadline,
release precedes precharge and data holds through the window.
The `rejected_ladders` block of the JSON holds the three rejected passes (23 of 27 attempts, 97,448 of 117,310 checks) with the same identities and measurements.
