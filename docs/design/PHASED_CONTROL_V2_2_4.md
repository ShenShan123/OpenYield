# V2.2.4: replica load cells that do not leak, and the V2.2.3 review fixes

> **No V2.2.4 screen has been run. There is no V2.2.4 evidence record.** The
> replica column changed, so every generated deck differs from V2.2.3 and
> neither the V2.2.3 screen nor [`docs/data/PHASED_CONTROL_V2_2_2.json`](../data/PHASED_CONTROL_V2_2_2.json)
> certifies this tree. What was simulated on it is the targeted evidence
> below: the seven tall draws that failed on V2.2.3 (commit `a1c4a5b`), ten
> A/B cases on the first version of the fix (`1441c7d`, with the static short
> described below; the rails are ideal, so its waveforms are not affected), a
> 64x4 power check and the operating-point probes. **The 256x256 case still has no
> working operating point** ([below](#the-operating-point-of-the-largest-arrays)),
> so the screen cannot complete until that is solved.

V2.2.4 fixes the sense-margin failure that a per-device Monte Carlo run exposed
at the tall edge of the envelope, and the defects found while reviewing V2.2.3
against its own screen and a 1,487-case discovery campaign. The controller
(`time_generate.py`), the array cells, the wires, the driver classes and every
clock are unchanged. The replica column is not: every generated deck differs
from V2.2.3, so no V2.2.3 result carries over.

## The tall-array sense failure

**Symptom.** The V2.2.3 tall Monte Carlo run (`dev/v223mc/tall.json`: 52
per-device draws at 512x4 and 256x4, 6T / 6T+mux / 10T / 10T+mux, FS at
0.9 V / 125 C and FF at 1.1 V / 125 C) was stopped after 14 cases, all 512x4
6T. **7 of those 14 failed**, all on the same checks,
`cycle1/cycle5_read_cX_sense_margin`. The same seeds, re-run on the fixed tree
with the same mismatch draws, all pass:

| Case | Bar (0.28 VDD) | V2.2.3 worst sense margin | V2.2.4 | Read-output margin, V2.2.3 → V2.2.4 |
|---|---:|---:|---:|---:|
| tall_512x4_6t_FShot_s00 | 0.252 V | 0.248 V fail | 0.425 V | 5.90 → 5.27 ns |
| tall_512x4_6t_FShot_s01 | 0.252 V | 0.218 V fail | 0.404 V | 5.65 → 5.07 ns |
| tall_512x4_6t_FShot_s03 | 0.252 V | 0.177 V fail | 0.328 V | 5.94 → 5.44 ns |
| tall_512x4_6t_FShot_s08 | 0.252 V | 0.250 V fail | 0.443 V | 5.61 → 4.98 ns |
| tall_512x4_6t_FShot_s09 | 0.252 V | 0.187 V fail | 0.369 V | 5.25 → 4.61 ns |
| tall_512x4_6t_FFhot_s00 | 0.308 V | 0.279 V fail | 0.539 V | 7.39 → 6.91 ns |
| tall_512x4_6t_FFhot_s03 | 0.308 V | 0.251 V fail | 0.494 V | 7.58 → 7.17 ns |

(`outputs/validation/V2.2.3-mc/tall` and `outputs/validation/V2.2.4-ab/tall`,
the latter from commit `a1c4a5b`. All 52 draws are in the V2.2.4 screen.)

Cycles 1 and 5 are the reads of a 1 against a column whose other 511 cells
store 0. The same traces read a 0 (cycles 3 and 7) with 0.68-0.80 V.

**Cause.** The replica column has one cell per row on the same bitline
geometry as a real column. Only the last `K` cells (`replica_k`, 1 by
default) are on the replica wordline. Through V2.2.3 the other `rows - K`
cells, whose wordlines are tied to ground, were built as the same cell storing
the same 0. At a fast NMOS corner and 125 C, the subthreshold leakage of 511
off pass gates, each against a stored 0, discharged RBL in parallel with the
active cell, so the sense enable fired early. It fired early exactly when the
array's own leakage was worst. A read of 1 against a column of 0s has its
reference bitline pulled down by the same leakage, so that read had the least
differential to spare.

Measured on one seed (`mc_512x4_6t_FShot_s0`, FS 0.9 V 125 C, per-device), same
draws, before and after:

| | RWL far → RBL 50 % | WL far → SA_ISO | read-1 ΔBL at SA_ISO (4 columns) | read-0 ΔBL |
|---|---:|---:|---|---|
| V2.2.3 (511 leaking load cells) | 811 ps | 1,699 ps | 0.335 / 0.343 / 0.292 / 0.256 V | 0.68-0.71 V |
| V2.2.4 (passive load cells) | 1,466 ps | 2,365 ps | 0.548 / 0.579 / 0.498 / 0.457 V | 0.78-0.80 V |

The load cells supplied about 45 % of the replica discharge current at this
corner. At SS the leakage is small and the effect is too: the same array at
SS 125 C moves from 0.455 to 0.531 V.

**Fix.** `ReplicaColumn(active_rows=replica_k)` builds the cells off the
replica wordline as *passive* load cells, `Replica_CELL_PASSIVE`. Both
storage nodes are held at VDD, so each pass gate still loads RBL and RBLB with
the same junction and overlap capacitance but has no drain-source voltage
across it and no leakage. The driven cells are unchanged. Instance names
(`XReplica_CELL_{row}`) and every node name are unchanged, so no probe moves.

- 6T: the left inverter's gates go to VSS (Q held at VDD by the pull-up) and so
  do the right inverter's, whose output is tied to VDD.
- 10T: both cross-coupled gate inputs go to VSS. Both pull-ups hold Q and QB at
  VDD, every pull-down is off, and the feedback devices only lift the internal
  nodes, so there is no static path.

**A defect in the first version of the fix.** The first passive 6T cell left
the right inverter's gates on Q. With Q at VDD, its pull-down conducts from the
tied-high output straight to VSS in every load cell. Static power at 512x4 FS
125 C went from 0.156 mW to 52.4 mW. Waveforms were unaffected because the
rails are ideal, so the checker could not see it; `PSTC` in the `.mt0` did.
It was fixed (`a1c4a5b`) before the tall re-runs; the A/B cases above predate the fix. At 64x4 FS 125 C the static power
is now 95.7 µW for 6T and 87.7 µW for 10T, against V2.2.3's 95.5 and 88.3 µW.
`tests/test_driver_paths.py::test_replica_load_cells_neither_leak_nor_short`
propagates static levels through every replica cell and fails on that
version.

**Cost.** The sense enable now fires later wherever leakage used to hurry it,
so the read-output margin shrinks: by 0.35 ns at 512x4 SS 125 C (3.55 to
3.19 ns), 0.66 ns at FS and 0.44 ns at FF. No ordering or recovery margin
moves by more than 1.5 ps, because clock-low recovery does not depend on when
the sense enable fired.

## The other V2.2.4 fixes

Found reviewing V2.2.3 against its executed screen (284 of 285 cases, 6
negative controls failing as required; the 256x256 case never left DCOP) and
a 1,487-case discovery campaign run on V2.2.3 (1,474 passed, 6,837,032
checks, about 880 solver-hours). The campaign's 13 failures were 12 one-row
checker errors and one DCOP failure, both fixed below.

- **Write-register startup.** Decks set `we` to VDD at t = 0 against a free
  slave node. The operating point put that node at VDD, so `we`, `we_bar` and
  `write_ready` each swung rail to rail in the first nanosecond of every
  deck, before any stimulus. The register now starts having latched "no
  write": `we = 0`, `we_bar = VDD` and the slave `Xdff_buf1:qint = VDD`, as
  the chip-select register already did. The largest startup excursion is
  0.3 mV.
- **Variant floor.** V2.2.3 removed the resolver's variant floor on the grounds
  that it only mattered under extrapolation. Under the joint-dimension rule a
  variant that raises only its early classes has a smaller joint excess, so it
  would get a shorter clock than the shared ladder; for example 128x128 would
  get 19.5 ns against 20.5 ns. The floor is restored for every size. The
  shipped variants never fell below it, so no period changes.
- **Envelope with an injected clock.** `resolve_timing` enforced the envelope,
  but a testbench given its own `timing_config` skipped `resolve_timing` and
  would build 1024x8 or 8x512. The testbench now checks the envelope in both
  paths.
- **Column-mux input 0 was never read.** The runner always targeted the last
  column, so `SEL0` was never selected in any screen. The runner and the
  checker now take `case['col']`, and the manifest reads and writes column 0 at
  8x4, 16x8, 128x8, 8x256 and 64x64. A column-0 trace re-scored as the last
  column fails, so the checker tells the two apart.
- **One-row arrays.** The checker decodes the row from the external address,
  and the compiler's `.PRINT` omits `A0` for a one-row array, so every one-row
  case raised `KeyError`. The runner now probes the address bits itself.
- **0.7 T leftovers.** The checker's skip rule and `access_cycles` still used
  0.7 T after V2.2.3 made `ACCESS_DEADLINE = 0.68` the single deadline. Both
  now use the constant. No deck changes.
- **DCOP retry on the multi-rank path.** A nominal multi-rank case had no retry
  after a failed operating point, while the one-rank path and the per-device
  ladder both did. A 257x4 10T+mux FF deck failed plain Newton after 23 minutes
  and converged with a line search. The runner now retries once with
  `SEARCHMETHOD=2` and keeps the failed attempt beside the deck.
- **Fallback options were discarded.** Xyce keeps only the last `.OPTIONS`
  line of a package, and `deck_with_option` inserted its fallback before the
  deck's own lines, so a retry could repeat the attempt that failed. The
  option now goes last, before `.END`.

## The operating point of the largest arrays

The V2.2.3 screen's 256x256 case spent 91.8 hours in Xyce's DC operating point
and was killed without converging. Plain Newton fails at these sizes and falls
back to GMIN stepping, and each step refactors a matrix of 1.2 to 2.4 million
unknowns. 256x128 converged, but only after about 13 hours of its 42.8.

MOSFET homotopy (`.OPTIONS NONLIN CONTINUATION=2`) was the first V2.2.4
answer. It is erratic. It converged a V2.2.3 256x128 deck in 16 minutes and a
V2.2.3 256x256 deck in 65, but on the V2.2.4 decks it converged one 128x128
`.IC` variant out of five in 6 minutes and failed or stalled for over an hour
on every other variant at 128x128 and 256x128. That includes the V2.2.3 `.IC`
set that had converged before. No pattern in the initial conditions explains
it, so it is not a method a screen can depend on.

**The seeded operating point.** A case marked `operating_point: seeded` runs
in two Xyce passes (`tests/spice/execution.py`):

1. A **settle** pass: the same deck, with `.PRINT` and `.MEASURE` removed, runs
   a `UIC` transient to 0.9 ns, before the first cycle at 1 ns and so before
   any stimulus moves. At its end it writes every node with
   `.SAVE TYPE=NODESET`.
2. The deck itself, with that file included as its `.NODESET` guess. Xyce
   refuses `.IC` together with `.NODESET`, so the `.IC` values replace the
   settled values of their nodes in the guess and the `.IC` cards are
   removed. The deck saves its own operating point at t = 0, and the runner
   fails the case unless every `.IC` node is within 10 mV of its value.

A settle alone is **not** an operating point. After 0.9 ns, 906 of 49,695
nodes of a 64x64 deck were still more than 1 mV from the DC solution, up to
0.4 V on the floating internal nodes of NAND stacks. So the settle is only the
starting guess, and the DC solve still decides the result. That result is the
deck's own operating point:

| Array | Settle | Operating point + 10 ps | `.IC` nodes | Worst `.IC` deviation | Before |
|---|---:|---:|---:|---:|---|
| 64x64 | 263 s | 154 s | — | all 49,695 nodes within 0.4 mV of plain DCOP | 1,239 s plain |
| 128x128 | 886 s | 142 s | 34,454 | 0.42 mV; all 181,151 node voltages within 0.42 mV of plain DCOP | 9,932 s plain, 2,223 Jacobians |
| 256x128 | 1,819 s | 246 s | 67,224 | 0.51 mV | ~13 h plain (V2.2.3) |
| 256x256 | 4,128 s | **not converged after 5.6 h** (4.6 h with line search) | | | 91.8 h, no convergence (V2.2.3) |

(4 MPI ranks, SS 0.9 V 125 C, nominal. The 64x64 row compares the saved
solution of both runs on every node.)

**256x256 is not solved.** From the same kind of guess, Newton at 256x256 was
still iterating (not yet in GMIN stepping) after 5.6 hours, where 256x128
needed about 35 Jacobians in 4 minutes. A second run with Newton line search
from the same guess had not converged after 4.6 hours either. Both were still
running at release; the likely suspects are a settle window too short for the
256-column wordlines and fill-in of the 2.35-million-unknown factorization,
and neither has been checked.

The seeded path applies only to nominal decks of 16,384 cells or more. The
per-device decks keep plain Newton, which converged every 128x128 draw of
V2.2.3 (19.9 and 23.2 hours in total).

## The screen

**Not run.** The manifests are ready and every case in them resolves:

| Manifest | Cases | What it adds |
|---|---:|---|
| `tests/spice/v224_cases.json` | 330 | the 285 V2.2.3 cases, plus 45: mux column 0 at five sizes, FF cold and hot on 128- to 512-row arrays (V2.2.3 had no FF case at 128 rows or more), sizes that are not powers of two including 1x1, 1x4, 2x1, 257x4, 511x4 and 200x100, and partial address-bit flips between neighbouring rows on tall arrays |
| `tests/spice/v224_mc_cases.json` | 1,444 | per-device draws at each failure mechanism's worst global corner (timing SS hot, write SF cold, read stability FS hot, races FF cold, leakage FF hot), 20 per cell/mux configuration at 8x4; corner-case patterns under mismatch (read after write, write after write, adjacent bit flips, idle mixes, walking patterns); larger-array draws; and the 52 tall draws, with the seeds of their V2.2.3 run so each before/after pair shares its draw |
| `tests/spice/v224_negative_cases.json` | 6 | the V2.2.3 controls, each at a period the lookup would not grant; must exit non-zero |

Estimated cost about 1,400 solver-hours (the V2.2.3 cost model, which runs
about 5 % optimistic), of which the eight arrays of 16,384 cells or more are
about 410. Run each manifest from a worktree pinned at the release commit so
that no later edit trips the runner's source-hash guard:

```bash
python3 -m tests.spice.phased_access --cases tests/spice/v224_cases.json \
  --output outputs/validation/V2.2.4-phased/main --xyce "$(command -v Xyce)" --workers 4
python3 -m tests.spice.phased_access --cases tests/spice/v224_mc_cases.json \
  --output outputs/validation/V2.2.4-phased/mc --xyce "$(command -v Xyce)" --workers 4
python3 -m tests.spice.phased_access --cases tests/spice/v224_negative_cases.json \
  --output outputs/validation/V2.2.4-phased/negative --xyce "$(command -v Xyce)" --workers 4   # must exit non-zero
```

**What V2.2.3's own screen showed**, for the record, since the V2.2.3 design
record was written before it ran: launched 2026-09-20 from the clean V2.2.3
commit, 284 of 285 cases passed (2,514,655 checks) and all 6 negative controls
failed as required. `256x256_6t_SS_read_write` was killed after 91.8 hours in
its operating point. That screen was never assembled into an evidence record,
and it did not include the per-device tall draws that exposed the sense
failure.

## Still open

- **The 256x256 operating point** (above). Until it is solved, the V2.2.4
  screen cannot complete and 256x256 has no evidence under any release.
- **The V2.2.4 screen itself** (above).

- **Dynamic column select.** The compiler has no column decoder: `SEL` is a
  DC level for the whole deck, so switching between mux inputs from one access
  to the next has never been simulated. V2.2.4 closes only the static half
  (input 0 is now exercised).
- **A stopped or gated clock**, unchanged from V2.2.3.
- **Sense amplifier offset.** The 0.28 VDD bar is still a screening floor, not
  a measured offset.
- Extracted metal, half-selected writes and yield estimation
  (`docs/README.md`).
