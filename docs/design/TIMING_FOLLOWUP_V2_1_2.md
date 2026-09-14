# V2.1.2 timing follow-up: Phase 4 boundaries and Phase 5 mismatch pilot

Executed September 14, 2026 on the V2.1.2 sources (base commit `61d01a7ee00284ff58d2b9d8d6dd535a0ca9411e`
plus the uncommitted post-release cleanup and CLI repairs recorded in the
[changelog](../CHANGELOG.md)) for Phases 4 and 5 of the
[evaluation plan](../plans/V2_1_1_TIMING_FOLLOWUP.md). Illustrative wires,
frozen V2.1.0 lookup clocks and V2.0.9 driver classes throughout; nominal seed
20260913, maximum step 20 ps; at most eight simulator ranks at any time. This
is functional screening, not PVT/mismatch or extracted-metal qualification;
no record is promoted to `sizing_table.json`.

**Outcome: 30 of 35 attempts pass; 70,450 of 82,224 checks pass.**
The Phase 4 lookup-clock matrix passes 17 of 19 cases; the Phase 5 pilot passes
9 of 12 per-device seeds. All five failures are one mechanism at the frozen
4 ns class, diagnosed below, and every failure is electrical: no solver
aborted, no retry ran, and the only solver message is one non-fatal
minimum-step warning in the 8x8 mux TT sequence.

## Results

Runtime measures and the independent all-column waveform checks of
`dev/v210_waveform_checks.py` both apply. PRE90→WL50 is precharge rising
through 90 % VDD to the local wordline rising through 50 %; WL10→PRE90 is the
wordline falling through 10 % to local precharge falling through 90 %. Fixed
clocks mark diagnostics, not lookup-class results.

| Case | Array | Op | PVT | Wires | Variation | Clock (ns) | Result | PRE90→WL50 min (ps) | WL10→PRE90 min (ps) |
|---|---|---|---|---|---|---|---|---:|---:|
| p4_8x64_SS_write | 8x64 | write | SS 0.9 V 125 °C | stubs | nominal | 5 lookup | passed (2,977 checks) | 533 | 382 |
| p4_8x64_SF_write | 8x64 | write | SF 0.9 V 125 °C | stubs | nominal | 5 lookup | passed (2,977 checks) | 473 | 332 |
| p4_8x128_SF_write | 8x128 | write | SF 0.9 V 125 °C | stubs | nominal | 6 lookup | passed (3,251 checks) | 506 | 340 |
| p4_8x128_SF_write_nostubs | 8x128 | write | SF 0.9 V 125 °C | no stubs | nominal | 6 lookup | passed (3,251 checks) | 479 | 335 |
| p4_8x64_SS_write_refine2 | 8x64 | write | SS 0.9 V 125 °C | stubs ref. 2 | nominal | 5 lookup | passed (2,977 checks) | 532 | 381 |
| p4_8x64_SS_write_stress3 | 8x64 | write | SS 0.9 V 125 °C | stubs ×3 R/C | nominal | 5 lookup | passed (2,977 checks) | 564 | 372 |
| p4_8x4_SS_sequence_cellpin | 8x4 | read&write | SS 0.9 V 125 °C | stubs + pin RC | nominal | 4 lookup | passed (1,295 checks) | 579 | 256 |
| p4_8x8_mux_TT_sequence | 8x8 mux | read&write | TT 1.0 V 25 °C | stubs | nominal | 4 lookup | passed (2,427 checks) | 243 | 141 |
| p4_16x16_SS_nostubs_sequence | 16x16 | read&write | SS 0.9 V 125 °C | no stubs | nominal | 4 lookup | passed (6,931 checks) | 499 | 330 |
| p4_48x20_SS_sequence | 48x20 | read&write | SS 0.9 V 125 °C | stubs | nominal | 4.5 lookup | passed (3,865 checks) | 542 | 373 |
| p4_128x8_SS_read | 128x8 | read | SS 0.9 V 125 °C | stubs | nominal | 5 lookup | passed (404 checks) | 601 | 359 |
| p4_256x4_SS_read | 256x4 | read | SS 0.9 V 125 °C | stubs | nominal | 6 lookup | passed (604 checks) | 693 | 351 |
| p4_256x4_SS_read_nostubs | 256x4 | read | SS 0.9 V 125 °C | no stubs | nominal | 6 lookup | passed (604 checks) | 528 | 368 |
| p4_64x16_FF_cold_write_hazard | 64x16 hazard→5 | write | FF 1.1 V -40 °C | stubs | nominal | 4.5 lookup | passed (572 checks) | 156 | 97 |
| p4_32x8_FF_cold_read_hazard | 32x8 hazard→3 | read | FF 1.1 V -40 °C | stubs | nominal | 4 lookup | passed (938 checks) | 154 | 85 |
| p4_16x16_10t_mux_SS_sequence | 16x16 10T mux | read&write | SS 0.9 V 125 °C | stubs | nominal | 4 lookup | failed (9 of 6931 checks) | 531 | 359 |
| p4_64x4_SS_read_stress3 | 64x4 | read | SS 0.9 V 125 °C | stubs ×3 R/C | nominal | 4.5 lookup | failed (1 of 958 checks) | 592 | 302 |
| p4_512x4_SS_read | 512x4 | read | SS 0.9 V 125 °C | stubs | nominal | 9 lookup | passed (1,116 checks) | 996 | 362 |
| p4_512x4_SF_write | 512x4 | write | SF 0.9 V 125 °C | stubs | nominal | 9 lookup | passed (1,159 checks) | 933 | 309 |
| p5_16x16_SS_read_pd_s1 | 16x16 | read | SS 0.9 V 125 °C | stubs | per-device 20261001 | 4 lookup | passed (994 checks) | 487 | 358 |
| p5_16x16_SS_read_pd_s2 | 16x16 | read | SS 0.9 V 125 °C | stubs | per-device 20261002 | 4 lookup | passed (994 checks) | 538 | 319 |
| p5_16x16_SS_read_pd_s3 | 16x16 | read | SS 0.9 V 125 °C | stubs | per-device 20261003 | 4 lookup | passed (994 checks) | 524 | 367 |
| p5_16x16_SF_write_pd_s1 | 16x16 | write | SF 0.9 V 125 °C | stubs | per-device 20261001 | 4 lookup | passed (1,169 checks) | 466 | 309 |
| p5_16x16_SF_write_pd_s2 | 16x16 | write | SF 0.9 V 125 °C | stubs | per-device 20261002 | 4 lookup | passed (1,169 checks) | 495 | 314 |
| p5_16x16_SF_write_pd_s3 | 16x16 | write | SF 0.9 V 125 °C | stubs | per-device 20261003 | 4 lookup | passed (1,169 checks) | 484 | 342 |
| p5_8x4_10t_mux_SS_sequence_pd_s1 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | stubs | per-device 20261001 | 4 lookup | failed (7 of 1295 checks) | 556 | 312 |
| p5_8x4_10t_mux_SS_sequence_pd_s2 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | stubs | per-device 20261002 | 4 lookup | failed (4 of 1295 checks) | 529 | 310 |
| p5_8x4_10t_mux_SS_sequence_pd_s3 | 8x4 10T mux | read&write | SS 0.9 V 125 °C | stubs | per-device 20261003 | 4 lookup | failed (2 of 1295 checks) | 552 | 296 |
| p5_8x4_FF_cold_sequence_pd_s1 | 8x4 | read&write | FF 1.1 V -40 °C | stubs | per-device 20261001 | 4 lookup | passed (1,295 checks) | 151 | 84 |
| p5_8x4_FF_cold_sequence_pd_s2 | 8x4 | read&write | FF 1.1 V -40 °C | stubs | per-device 20261002 | 4 lookup | passed (1,295 checks) | 149 | 78 |
| p5_8x4_FF_cold_sequence_pd_s3 | 8x4 | read&write | FF 1.1 V -40 °C | stubs | per-device 20261003 | 4 lookup | passed (1,295 checks) | 148 | 78 |
| p4d_16x16_10t_mux_SS_sequence_T4500 | 16x16 10T mux | read&write | SS 0.9 V 125 °C | stubs | nominal | 4.5 fixed | passed (6,931 checks) | 530 | 358 |
| p4d_16x16_10t_mux_SS_sequence_T5000 | 16x16 10T mux | read&write | SS 0.9 V 125 °C | stubs | nominal | 5 fixed | passed (6,931 checks) | 533 | 358 |
| p4d_16x16_10t_mux_TT_sequence_lookup | 16x16 10T mux | read&write | TT 1.0 V 25 °C | stubs | nominal | 4 lookup | passed (6,931 checks) | 237 | 157 |
| p4d_64x4_SS_read_stress3_T5000 | 64x4 | read | SS 0.9 V 125 °C | stubs ×3 R/C | nominal | 5 fixed | passed (958 checks) | 592 | 303 |

Phase 4 queues A (width, wires) and B (height, hazards, mux) ran at four ranks
each in parallel, the 512-row boundary at eight ranks, and the twelve
per-device samples on one rank each in four parallel queues. The 512x4 cases
took 19 and 21 minutes; every other case took under 20 minutes.

## Finding 1: the shared 4 ns class is exhausted for 10T with a column mux

The 16x16 10T column-mux SS sequence fails every read cycle's output check
(validator, independent scorer and the runtime access measure agree) while
its cells, sense margin (0.87 V at the amplifier input), precharge exclusion,
restore and retention all pass. The sense enable is late:

| Case | Clock | WL_EN 50 % | Replica BL 50 % | S_EN 50 % rise | Output at 1.18 T |
|---|---|---|---|---|---|
| 16x16 10T mux SS sequence (failed) | 4.0 ns lookup | 0.896 T | 0.983 T | 1.184 T | stale |
| 8x4 10T mux SS sequence (V2.1.1 pass) | 4.0 ns lookup | 0.880 T | 0.965 T | 1.161 T | 0.86 V, marginal |
| 8x4 10T mux SS, per-device seeds 1/2/3 (failed) | 4.0 ns lookup | 0.878–0.883 T | 0.968–0.976 T | 1.167–1.182 T | 0.00 / 0.31 / 0.71 V |
| 16x16 6T SS read (V2.1.1 pass) | 4.0 ns lookup | 0.890 T | 0.967 T | 1.136 T | correct |
| 16x16 10T mux SS sequence, diagnostic | 4.5 ns fixed | 0.877 T | 0.954 T | 1.133 T | correct, 6,931 checks |
| 16x16 10T mux SS sequence, diagnostic | 5.0 ns fixed | | | | correct, 6,931 checks |
| 16x16 10T mux TT sequence | 4.0 ns lookup | | | | correct, 6,931 checks |

From the access request at 0.65 T the 10T-mux path needs about 2.15 ns to
sense enable at SS 0.9 V / 125 °C in every case above, plus about 0.1 ns for
the amplifier and output latch; the 4 ns class allows 2.2 ns to the 1.2 T data
deadline. The 6T path needs about 2.0 ns. The difference is the replica mux
pass gate in front of the replica sense input that TIME observes and the
slower 10T replica discharge, both of which the shared class did not budget.
Consequences:

- At the reference wires the class holds only to 8 rows for 10T with a mux,
  and with 5 % per-device mismatch even 8x4 fails on all three seeds.
- The array passes the same checks at 4.5 ns (190 ps of margin) and at 5 ns,
  and at TT on the 4 ns lookup clock. The failure is corner and architecture
  bound, not a connectivity or data fault.
- Proposal, not applied: give 10T-with-mux its own row/column budget one class
  up (1800 ps, 4.5 ns at 32 rows / 16 columns) or an earlier replica
  observation point. Either is a table or TIME change that needs its own
  evidence; the V2.1.0 table stays frozen in this release.

## Finding 2: explicit wire stress

With three times the per-pitch R and C, the 64x4 SS read at its 4.5 ns lookup
clock fails only the validator's output sample at 1.18 T (sense enable at
1.171 T; the runtime measure at 1.2 T passes with 0.023 V error) and passes at
5 ns. The 8x64 SS write passes the same stress, and refining the ladder to two
sections per half pitch changes its margins by under 2 ps. The stress geometry
is not the reference; the result bounds how much extracted wire the class can
absorb and is reported, not fitted.

## Observation: write-enable spike at the read-to-write boundary at FF −40 °C

In the FF 1.1 V / −40 °C sequences the local write enable spikes at the end of
each read access, while the wordline is still high: 0.41 V nominal, 0.50 to
0.63 V across the three mismatch seeds, 2 to 4 ps above half VDD, with no
bitline or data disturbance and every check passing. It is absent at TT and
SS. It belongs to the request gating in TIME and is left as an open item; a
TIME change invalidates every waveform record above.

## Phase 5 accounting

Every scheduled seed is accounted for: 16x16 SS read 3/3, 16x16 SF write 3/3,
8x4 FF cold sequence 3/3, 8x4 10T mux SS sequence 0/3 (Finding 1). Samples
ran on one rank through the plain runtime retry path, never the materialized
MPI fallback; `dev/v210_followup_queue.py` accepts single-rank per-device
cases with explicit seeds since this release. Sigma is 5 % relative on `vth0`,
`u0` and `voff` of every MOS. Three seeds are a pilot, not a yield statement.

## Evidence

Raw decks, waveforms, solver logs, per-attempt checkpoints, source archives
and plots are local under ignored `outputs/validation/V2.1.2-followup/`
(`phase4-A/B/L-queue`, `phase5-P1..P4-queue`, `phase4-D` for the direct
validator diagnostics, `*-write-capture.png`). The machine-readable
[record](TIMING_FOLLOWUP_V2_1_2.json) carries every case, identity, metric
and measurement. Representative write-capture plots of the 512x4 SF write,
the 16x16 SS sequence without stubs and the per-device FF cold sequence were
inspected: register capture precedes write enable, precharge is off before
the wordline, the driver pulls the bitline, Q meets the deadline, release
precedes precharge and data holds through the window. Phase 6 scope is in the
[qualification scope](../plans/V2_1_2_QUALIFICATION_SCOPE.md).
