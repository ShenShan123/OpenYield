# V2.1.3 open items — status after V2.1.4

Written September 15, 2026 after the V2.1.3 release (`6c619ab`): everything the
[10T timing budget record](../design/TIMING_10T_BUDGET_V2_1_3.md) found but
did not resolve, with the evidence behind each item, the proposed next step
and its cost. Each item needs its own evidence run before a table or circuit
change, as the [V2.1.1 plan](V2_1_1_TIMING_FOLLOWUP.md) and the
[qualification scope](V2_1_2_QUALIFICATION_SCOPE.md) require.

**V2.1.4 (September 15, 2026) closed the 6T items 1 to 3** with the
[6T budget record](../design/TIMING_6T_BUDGET_V2_1_4.md); the 10T items 4
and 5 are the next round, joined by item 7, which the V2.1.4 TIME change
created. Waveform paths below are relative to the ignored
`outputs/validation/` directory; each case directory holds `deck.sp`,
`deck.sp.prn`, `result.json` and `xyce.log`.

| Item | Status | Where |
|---|---|---|
| 1. 6T with a column mux needs its own budget | Closed in V2.1.4 | `SRAM_6T_CELL` mux variant, section 2 of the 6T record |
| 2. FF −40 °C write-enable spike | Closed in V2.1.4 (bug fixed) | TIME write-request hold latch, section 1 |
| 3. Shared 6T ladder margins | Closed in V2.1.4 (classes raised) | Shared row classes, sections 2 and 3 |
| 4. 10T read-disturb bump | Open (10T round) | below |
| 5. 10T classes without mismatch evidence | Open (10T round) | below |
| 6. Phase 6 scope | Open, unchanged | below |
| 7. 10T evidence predates the TIME latch | Open (10T round, new) | below |
| Found during V2.1.4: duplicated final waveform sample | Fixed in V2.1.4 (bug) | below, and section 6 of the 6T record |

## 1. 6T with a column mux needs its own timing budget — closed

Original evidence (SS 0.9 V / 125 °C, 16x16 6T sequence with a 2:1 mux at the
shared 4 ns class): nominal read output 74 ps before the 1.2 T deadline;
per-device seeds 20261001 to 20261003 left 97, 51 and 57 ps, and seeds 2
and 3 failed the validator's read-output checks.

Found in V2.1.4 (probe reads on the new TIME block, V2.1.3 table): the mux
penalty grows with rows (about 70, 95, 150, 200 and 275 ps at 32, 64, 128,
256 and 512 rows), and the shared classes fail nominally, output late with a
healthy sense margin:

| Case | Clock | Read output vs 1.2 T deadline | Waveforms |
|---|---|---|---|
| 16x16 mux sequence, seed 20261002 (V2.1.3) | 4 ns | 51 ps before, output check failed | `V2.1.3-10t-mux-budget/pd2-Q8-queue/c_16x16_6t_mux_SS_sequence_pd_s2/attempt-001/c_16x16_6t_mux_SS_sequence_pd_s2/` |
| 16x16 mux sequence, seed 20261003 (V2.1.3) | 4 ns | 57 ps before, output check failed | `V2.1.3-10t-mux-budget/pd2-Q8-queue/c_16x16_6t_mux_SS_sequence_pd_s3/attempt-001/c_16x16_6t_mux_SS_sequence_pd_s3/` |
| 32x16 mux read, nominal | 4 ns | 38 ps before, `output_data` failed | `V2.1.4-6t-timing/probe-A-queue/pr_32x16_6t_mux_SS_read/attempt-001/pr_32x16_6t_mux_SS_read/` |
| 128x8 mux read, nominal | 5 ns | 53 ps before, `output_data` failed | `V2.1.4-6t-timing/probe-B-queue/pr_128x8_6t_mux_SS_read/attempt-001/pr_128x8_6t_mux_SS_read/` |
| 256x4 mux read, nominal | 6 ns | 29 ps after, output and retention checks failed | `V2.1.4-6t-timing/probe-B-queue/pr_256x4_6t_mux_SS_read/attempt-001/pr_256x4_6t_mux_SS_read/` |

Resolution: `timing_lookup.json` `v2.1.4-timing-3` adds
`{"cell_type": "SRAM_6T_CELL", "mux": true}` with rows 1900/2000/2200/2700/3600 ps
and the shared columns (4.75, 5, 5.5, 6.75 and 9 ns), set by the V2.1.3 rule
(at least 250 ps at every class bound, nominal; sequences about 50 ps slower
than reads). Final evidence (nominal SS 0.9 V / 125 °C, read output to the
1.2 T deadline): 436 ps at the 16x16 mux sequence, 372 ps at 32x16, 357 ps at
64x16, 298 ps at 128x8, 333 ps at 256x4 and 257 ps at 512x4; 383, 542 and
931 ps at the 16x32, 8x64 and 8x128 column bounds; the 64x16 mux SF write
passes. Mismatch: 3 of 3 seeds at the 16x16 mux sequence (365–474 ps; the
two seeds that failed at 4 ns pass), the 32x16 sequence (375–416 ps), the
64x16 read (382–407 ps) and the 128x8 read (264–312 ps), and 10 of 10 at the
8x8 mux sequence (451–533 ps). Traces: `V2.1.4-6t-timing/final-N3-queue/` and
`final-N4-queue/` (mux reads and writes), `fix-N1-queue/`, `fix-N2-queue/`
and `fix-Q1..Q4-queue/` (4.75 ns mux sequences), `pd-Q*-queue/` (mux read
seeds); `evidence/final_read_access_bounds.png`.

## 2. FF −40 °C write-enable spike at the read-to-write boundary — closed

Original evidence: in FF 1.1 V / −40 °C 8x4 6T sequences the local write
enable spiked at the end of each read access while the wordline was still
high: 0.41 V nominal, 0.50 to 0.63 V on seeds 1 to 3 (V2.1.2), 0.38 to 0.72 V
on seeds 4 to 10 (V2.1.3); every check passed.

Root cause (a bug): TIME gated `w_en = AND2(request, we)` with the raw write
request register, which updates on the same clock edge that ends the access.
From CLK_BUF 50 %, WE moves at 20 to 21 ps but the access request falls only
at 37 to 39 ps (WL_EN at 58 to 61 ps), so both AND inputs are high for about
17 ps while the read wordline is on (at SS 82 against 133 ps; the slow gates
filter it to 1 mV). `s_en` had the same race with `we_bar` at the
write-to-read edge (1 to 2 mV). No check looked at the write enable outside a
write.

Pre-fix waveforms (all FF 1.1 V / −40 °C 8x4 6T sequences; the new check
fails every one at 0.34 to 0.66 VDD):

- `V2.1.1/small-queue/dist_8x4_FF_cold_sequence/attempt-001/dist_8x4_FF_cold_sequence/deck.sp.prn` (nominal, 0.41 V)
- `V2.1.2-followup/phase5-P4-queue/p5_8x4_FF_cold_sequence_pd_s{1,2,3}/attempt-001/p5_8x4_FF_cold_sequence_pd_s{1,2,3}/deck.sp.prn` (0.63, 0.60, 0.52 V)
- `V2.1.3-10t-mux-budget/pd2-Q7-queue/p5x_8x4_FF_cold_sequence_pd_s{4..10}/attempt-001/p5x_8x4_FF_cold_sequence_pd_s{4..10}/deck.sp.prn` (worst seed 20261004, 0.725 V)
- Edge table and plot: `V2.1.4-6t-timing/evidence/prefix_ff_cold_boundary.json`, `prefix_ff_cold_w_en_spike.png`

Resolution: the write request passes a hold latch (`Xwe_hold`, enabled by
`wl_en_bar`) and both enables take the held request, so neither can change
while a request is high. The waveform scorer gains
`strict_cycle_N_write_enable_quiet` and `strict_cycle_N_sense_enable_quiet`
(≤ 0.1 VDD from 0.65 T to 1.65 T). Post-fix: the held request rises 23 ps after
WL_EN falls and the read-cycle write enable peaks at 6 mV
(`V2.1.4-6t-timing/probe-T-queue/t_8x4_FF_cold_sequence/`,
`t_8x8_6t_mux_FF_cold_sequence/`, `probe-P{1,2,3}-queue/t_8x4_FF_cold_sequence_pd_s{1,2,3}/`,
`evidence/postfix_ff_cold_w_en.png`, `postfix_boundary.json`). Final evidence:
the 8x4 FF −40 °C sequence of the write gate
(`V2.1.4-6t-timing/final-G-queue/g_8x4_FF_cold_sequence/`, plot
`g_8x4_FF_cold_sequence-write-capture.png`) and all ten pilot seeds
(`pd-Q*-queue/p_8x4_FF_cold_sequence_pd_s*/`) pass with write-enable peaks of
at most 6 mV; the 8x8 mux FF −40 °C sequence
(`fix-N1-queue/g_8x8_6t_mux_FF_cold_sequence/`) peaks at 7 mV. Every sequence
of the final record passes the new quiet checks (largest peak 17 mV, the 8x4
TT sequence without local stubs).

## 3. Shared 6T ladder margins at its class bounds — closed

Original evidence: the 6T clocks were set in V2.1.0 from 16x16 (4 ns) and
512x4 (9 ns); nominal read-output margins 203 ps at 128x8 (5 ns) and 175 ps
at 256x4 (6 ns) were below the 250 ps held for 10T.

Found in V2.1.4: the never-run 32-row bound was worse, 109 ps at 32x16 at
4 ns (`V2.1.4-6t-timing/probe-A-queue/pr_32x16_6t_SS_read/attempt-001/pr_32x16_6t_SS_read/`),
and 64x16 kept 206 ps at 4.5 ns (`probe-A-queue/pr_64x16_6t_SS_read/`). The
128x8 and 256x4 traces are `V2.1.2-followup/phase4-B-queue/p4_128x8_SS_read/`
and `p4_256x4_SS_read/`.

Resolution: shared row budgets 1800/1900/2100/2500/3600 ps (4.5, 4.75, 5.25,
6.25 and 9 ns), by the same rule. Every 6T array up to 32 rows now clocks at
4.5 ns instead of 4 ns. Final evidence (nominal SS 0.9 V / 125 °C, read output
to 1.2 T deadline): 321 ps at the 32x16 sequence bound, 327 ps at 64x16,
325 ps at 128x8, 299 ps at 256x4 and 544 ps at 512x4; the column bounds keep
342 ps (16x32 sequence), 619 ps (8x64) and 992 ps (8x128); the 64x16 and
512x4 SF writes pass. Mismatch: 3 of 3 seeds at the 32x16 sequence
(336–369 ps), 64x16 read (329–357 ps) and 128x8 read (302–316 ps); the
16x16 SS read pilot 10 of 10 (349–451 ps) and the 16x16 SF write pilot 10 of 10.

## Found during V2.1.4: duplicated final waveform sample — fixed

Evidence: in the V2.1.4 final run, 6T mux sequences at the new 4.75 ns class
failed the scorer's `strict_finite_monotonic` check (which fails closed and
skips the rest of the strict checks) although every legacy and runtime check
passed and the read output kept 372 ps: the trace ends with two rows printed
at the same time, 42.325 ns, differing by at most 0.67 µV. A single-rank
retry of the same circuit and seed reproduced it row for row, so it is not
a solver-partition effect.

Root cause (a bug): a sequence runs to `1 ns + 8.7 T` and Xyce prints on the
`.OPTIONS OUTPUT INITIAL_INTERVAL` grid (2 ps in the validator). The
quarter-nanosecond clocks that V2.1.4's 50 ps rounding produces put that stop
off the grid (1 ns + 8.7 × 4.75 ns = 42.325 ns), so the trace ends with an
off-grid sample after 42.324 ns, and in some runs Xyce prints that final time
twice. Every earlier clock was a multiple of 0.5 ns, whose stops lie on the
grid (for example 40.150 ns at 4.5 ns), so no earlier record shows it.

Waveforms (ignored `outputs/validation/`), each `deck.sp.prn` ending with two
rows at 4.23250000e-08:

- `V2.1.4-6t-timing/final-N3-queue/f_32x16_6t_mux_SS_sequence/attempt-001/f_32x16_6t_mux_SS_sequence/` (nominal, four ranks)
- `V2.1.4-6t-timing/final-R-queue/f_32x16_6t_mux_SS_sequence/attempt-001/f_32x16_6t_mux_SS_sequence/` (the same case on one rank)
- `V2.1.4-6t-timing/pd-Q2-queue/f_8x8_6t_mux_SS_sequence_pd_s2/attempt-001/f_8x8_6t_mux_SS_sequence_pd_s2/`
- `V2.1.4-6t-timing/pd-Q3-queue/f_8x8_6t_mux_SS_sequence_pd_s3/attempt-001/f_8x8_6t_mux_SS_sequence_pd_s3/`
- `V2.1.4-6t-timing/pd-Q7-queue/f_8x8_6t_mux_SS_sequence_pd_s8/attempt-001/f_8x8_6t_mux_SS_sequence_pd_s8/`

Resolution: `_analysis_stop()` rounds the stop up onto the output interval,
which changes only decks whose stop was off the grid (the 6T mux sequences
at 4.75 ns in this release); a test checks every lookup clock's sequence stop
lies on the grid and never ends before 8.7 T. All 22 affected sequences were
rerun on the fixed sources (`fix-*` queues); every other campaign deck
regenerates identical from the final sources (`verify_decks.py`; one
per-device read differs only by the runtime's DC-operating-point retry option).
Final evidence: all 21 reruns pass (116,295 of 116,295 checks); their decks
stop at 42.326 ns and their traces end at 42.324 and 42.326 ns
(`V2.1.4-6t-timing/fix-*-queue/`).

## 4. 10T read-disturb bump (cell-level observation) — open

Evidence: during a read the 10T cell's low storage node rises to 0.196 V
(0.131 V for 6T) and decays only as the bitline discharges. At 512 rows
this, not the sense path, sets the 14 ns class (Q at 0.108 V at the deadline
at 11 ns against the 0.1 VDD tolerance; 0.051 V at 14 ns). At 256 rows Q is
0.06 V at 8 ns. For comparison, the 512x4 6T mux read of V2.1.4 settles Q to
0.063 V at 9 ns.

Proposal: this is a property of the 10T cell as sized in
`sram_compiler/config_yaml/` (read-port and pull-down widths), not of the
timing. Either accept the 14 ns class for 512-row 10T arrays or size the
10T read path in a separate, evidenced cell change; a cell width change
alters every 10T deck and needs the 10T boundary run again.

## 5. 10T classes without mismatch evidence — open

Evidence: mismatch seeds cover the 32- and 64-row 10T bounds and 8x4. The
128-row bound keeps 282 ps nominal at 6 ns (the smallest 10T margin), the
256-row bound 529 ps at 8 ns, the column bounds 397 ps or more; none has
seeds. Ten seeds bound a failure rate only to 26 % at 95 % confidence.

Proposal: three seeds at 128x8 (about 15 minutes each on one rank) in the
10T round, together with item 7; the full statistical question belongs to
the yield estimator brief.

## 6. Carried from the Phase 6 scope — open

Unchanged from the [qualification scope](V2_1_2_QUALIFICATION_SCOPE.md):
extracted-metal inputs before any timing qualification; the half-select
write architecture; the yield-estimator repairs. The qualification matrix's
array classes are per architecture: shared (6T), 6T with a mux, and 10T.

## 7. 10T evidence predates the V2.1.4 TIME latch — open (new)

The write-request hold latch is in the TIME block that 10T arrays share. The
10T clocks are unchanged and the 10T decks differ only in TIME
(`V2.1.4-6t-timing/deck-comparison/10t_*.diff`), but every V2.1.3 10T
waveform record ran the old TIME block. V2.1.4 ran one 10T smoke case, the
8x4 10T mux SS sequence of the write gate (passed 1,303 checks, 548 ps of
read-output margin at 5 ns, boundary write-enable peak 6 mV).

Proposal: in the 10T round, rerun the V2.1.3 10T boundary matrix and seeds
(`V2.1.3-10t-mux-budget/final2-*-cases.json`, `pd2-Q1..Q5`) on the current
sources with the new quiet checks, add an FF −40 °C 10T sequence, and
combine it with items 4 and 5. About 3.5 hours on eight ranks.

## Tooling left local

`dev/v213_sense_timing.py` measures request-to-sense-enable, request-to-
output and the deadline margin per read cycle from a validator case
directory; `dev/v214_boundary_enable.py` measures the WE, held-request,
access-request and WL_EN edges and the write/sense-enable peaks at every
access-ending clock edge (with a plot option); `dev/v212_followup_report.py`
and `dev/v212_followup_plots.py` take `--base`/`--queues` and
`--base`/`--select` for another campaign root. All stay under ignored
`dev/`, listed in the [development guide](../DEVELOPMENT.md). The V2.1.4
campaign scripts (`run_probe.sh`, `run_final.sh`, `gen_final_cases.py`,
`assemble_record.py`) are in `V2.1.4-6t-timing/`.
