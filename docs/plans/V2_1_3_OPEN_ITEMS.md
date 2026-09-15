# V2.1.3 open items

Written September 15, 2026 after the V2.1.3 release (`6c619ab`): everything the
[10T timing budget record](../design/TIMING_10T_BUDGET_V2_1_3.md) found but
did not resolve, with the evidence behind each item, the proposed next step
and its cost. Nothing here changes a clock, a driver class or a check; each
item needs its own evidence run before a table or circuit change, as the
[V2.1.1 plan](V2_1_1_TIMING_FOLLOWUP.md) and the
[qualification scope](V2_1_2_QUALIFICATION_SCOPE.md) require. Items are in
the order they should be taken.

## 1. 6T with a column mux needs its own timing budget

Evidence (SS 0.9 V / 125 °C, illustrative wires, 16x16 6T sequence with a
2:1 mux at the shared 4 ns class): nominal read output 74 ps before the 1.2 T
deadline; per-device seeds 20261001 to 20261003 leave 97, 51 and 57 ps, and
seeds 2 and 3 fail the validator's read-output checks (it samples 0.015 T
before the deadline). The mux pass gate in the array and replica sense paths
costs about 140 ps of the access window, which the shared class absorbs
nominally but not under mismatch. The V2.1.2 follow-up screened 6T with a
mux only at TT (8x8 sequence, 4 ns).

Proposal: a second `variants` entry in `timing_lookup.json`,
`{"cell_type": "SRAM_6T_CELL", "mux": true}`, which the V2.1.3 loader and
resolver already support (`ArrayTiming.budget` would read
`SRAM_6T_CELL/mux`). Start from the shared ladder plus 200 ps at every
anchor (4.5 ns up to 32 rows / 16 columns) and apply the V2.1.3 rule: at
least 250 ps between the read output and the 1.2 T deadline at every class
bound, nominal, plus three mismatch seeds at the two smallest bounds.

Evidence run, by the V2.1.3 recipe (`outputs/validation/V2.1.3-10t-mux-budget/run_final2.sh`
as the template): 8x4, 16x16, 32x16, 16x32 sequences; 64x16, 128x8, 256x4,
8x64, 8x128 reads; 64x16 SF write; 512x4 read; three seeds at the 32- and
64-row bounds; ten seeds at 8x8. About 3.5 hours on eight ranks. A 6T-mux
penalty that does not grow with rows (unlike 10T) may allow a flat offset;
report the margins rather than fitting them.

## 2. FF −40 °C write-enable spike at the read-to-write boundary

Evidence: in FF 1.1 V / −40 °C sequences the local write enable spikes at
the end of each read access while the wordline is still high. 0.41 V
nominal (V2.1.2), 0.50 to 0.63 V on seeds 1 to 3 (V2.1.2) and 0.38 to 0.72 V
on seeds 4 to 10 (V2.1.3), 8x4 6T; no bitline or data disturbance and every
check passes. Absent at TT and SS.

Proposal: diagnose the request gating in TIME (the write-enable NAND sees
the read's wordline-enable fall and the next write's request within the
same gate delay at the fast corner). Any TIME change invalidates every
waveform record since V2.1.1 and needs the Phase 2 write gate, the V2.1.3 10T
boundary run and the pilot seeds again. Cost of the diagnosis alone: a few
FF sequences with the TIME internals probed, under an hour.

## 3. Shared 6T ladder margins at its class bounds

Evidence: the 6T clocks were set in V2.1.0 from 16x16 (4 ns) and 512x4
(9 ns) and never measured at the 32-, 64-, 128- and 256-row bounds under
mismatch. Nominal read-output margins from the retained traces: 128x8 at
5 ns 203 ps, 256x4 at 6 ns 175 ps, 512x4 at 9 ns 544 ps; 16x16 SS read seeds
4 to 10 at 4 ns 122 to 193 ps (16 rows, not the 32-row bound). The 10T
ladder was held to 250 ps at every bound; the 6T ladder is below that at
128 and 256 rows.

Proposal: run the 6T class bounds nominally (32x16 sequence, 64x16, 128x8,
256x4 reads at SS 0.9 V / 125 °C) and three seeds at each bound that is
below 250 ps; raise a class only on evidence, as a new table version.
About two hours on eight ranks.

## 4. 10T read-disturb bump (cell-level observation)

Evidence: during a read the 10T cell's low storage node rises to 0.196 V
(0.131 V for 6T) and decays only as the bitline discharges. At 512 rows
this, not the sense path, sets the 14 ns class (Q at 0.108 V at the deadline
at 11 ns against the 0.1 VDD tolerance; 0.051 V at 14 ns). At 256 rows Q is
0.06 V at 8 ns.

Proposal: this is a property of the 10T cell as sized in
`sram_compiler/config_yaml/` (read-port and pull-down widths), not of the
timing. Either accept the 14 ns class for 512-row 10T arrays or size the
10T read path in a separate, evidenced cell change; a cell width change
alters every 10T deck and needs the 10T boundary run again.

## 5. 10T classes without mismatch evidence

Evidence: mismatch seeds cover the 32- and 64-row 10T bounds and 8x4. The
128-row bound keeps 282 ps nominal at 6 ns (the smallest 10T margin), the
256-row bound 529 ps at 8 ns, the column bounds 397 ps or more; none has
seeds. Ten seeds bound a failure rate only to 26 % at 95 % confidence.

Proposal: three seeds at 128x8 (about 15 minutes each on one rank) when the
next campaign runs; the full statistical question belongs to the yield
estimator brief.

## 6. Carried from the Phase 6 scope

Unchanged from the [qualification scope](V2_1_2_QUALIFICATION_SCOPE.md):
extracted-metal inputs before any timing qualification; the half-select
write architecture; the yield-estimator repairs. The 10T clocks now differ
from 6T, so the qualification matrix's array classes are per cell type.

## Tooling left local

`dev/v213_sense_timing.py` measures request-to-sense-enable, request-to-
output and the deadline margin per read cycle from a validator case
directory (it produced the sense-timing columns of the V2.1.3 record);
`dev/v212_followup_report.py` and `dev/v212_followup_plots.py` take
`--base`/`--queues` and `--base`/`--select` for another campaign root. All
three stay under ignored `dev/`, listed in the
[development guide](../DEVELOPMENT.md).
