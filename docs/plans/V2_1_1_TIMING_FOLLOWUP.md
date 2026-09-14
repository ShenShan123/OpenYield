# V2.1.1 distributed-only implementation and evaluation plan

Updated September 14, 2026 with the Phase 4 and 5 outcome. Written September 13,
2026 for the user's explicit instruction: remove every star-RC implementation, keep only distributed RC as
the default, alter the evaluation, and release the change as **V2.1.1**.
Write correctness remains the first functional priority. Maximum compute: eight ranks total, one large case at a
time, one BLAS thread per rank.

Read `AGENTS.md` and `docs/design/DISTRIBUTED_ONLY_V2_1_1.md` before resuming.
The input star-RC screen is in git history
(`git show 61d01a7:docs/design/STAR_RC_SCREEN_V2_1_0.md`).

## Current checkpoint

- The topology change is implemented: distributed-only defaults and rejection
  of star options; tapped array/replica/dummy/equivalent loads; BL/RBL peripheral
  ladders; decoder address/true/complement/enable routing; DATA_DFF clocks;
  mux-select lines. Local series stubs remain independently selectable.
- Lookup transistor classes and clocks are fixed. The initial wire-only stage
  matched 64 incoming sizing vectors; the access-start guard now adds TIME
  devices, observer/access loads and frozen RC settling, changing full
  fingerprints. Historical comparisons are not relabeled as final vectors.
- The initial distributed-only snapshot passed 112 compiler tests, 51
  development tests and six optimizer tests; these retain their historical identity.
  Numeric/sweep, 6T/10T, default/custom geometry, RC conservation, decoding and
  rejection paths are covered. The local scoring-source manifest has been reviewed and refreshed.
- The seven fresh small-array cases completed under
  `outputs/validation/distributed-only-V2.1.0/small-queue/`: all pass 6,463
  checks. The distributed-default CLI also passed a two-sample per-device write.
  These pre-release artifacts keep their original V2.1.0 labels and sources.
- The initial Phase 3 queue has finished: **64x64 TT sequence passed 10,185
  checks**; **8x512 SS write completed but failed 478 checks**, consisting of
  477 per-column precharge/write overlaps and the aggregate failure. Preserve
  both results. The race is repaired and fresh V2.1.1 waveforms pass at the
  frozen clocks, as recorded below. Inspect active processes before another
  launch; the total limit is eight simulator ranks.
- The race repair is implemented: common WL/write/sense assertion waits for
  the far PRE observation and baseline-derived RC settling plus four stages.
  Runtime and all-column waveform checks reject access/precharge overlap.
  Final-source checks pass 124 compiler tests (Python 3.11 and 3.9), 54
  development tests and six optimizer tests. Nine new small cases passed
  under `outputs/validation/V2.1.1/small-queue/`, including the tight 16x16 SS
  read at the unchanged 4 ns clock and the fast/cold sequence.
- The final-source large queue completed under
  `outputs/validation/V2.1.1/large-queue/`: 8x512 SS write passes 12,851 checks
  and 64x64 TT sequence passes 11,209 checks. All eleven final-source cases
  pass **33,096 checks**. The functional V2.1.1 release gate is complete;
  commit/push to `main` is authorized after final documentation/diff review.
- The previous queue (PID 692597) was gracefully interrupted, including its
  MPI ranks. Its 8x512 write remains partial evidence under the old baseline.
  It must not resume against the changed sources.
- Phases 4 and 5 ran on September 14 on the V2.1.2 sources: 30 of 35 attempts
  and 70,450 of 82,224 checks pass. All five failures are the 10T column-mux
  SS sequence at the 4 ns class (nominal 16x16 and three per-device 8x4 seeds)
  plus the three-times wire-stress read; both pass at the next class. The
  clock table and driver classes are unchanged. Phase 6 is scoped in the
  [qualification scope](V2_1_2_QUALIFICATION_SCOPE.md).

## Preserve historical baselines and evidence labels

`outputs/validation/V2.1.0-review-20260912/` contains the first review's
retention/CLI fixes, seven pre-removal write cases and the interrupted large
attempt. Those results retain their original identities and limits.

`outputs/validation/distributed-only-V2.1.0/incoming-source.tar.gz`,
`incoming.json` and `incoming.patch` preserve the complete tree immediately
before star removal, including untracked release files and ignored development
sources. Five old generation scripts were retired from active `dev/`; their
names are recorded in `retired-tools.json` and their bytes remain in that
archive. Historical CSVs, qualification JSON and the original star screen
must remain unchanged. Read-only plots of historical traces keep their labels.
The initial small/large distributed-only results also retain their original
V2.1.0 evidence paths; they preceded the V2.1.1 release designation. New fixes
use a fresh V2.1.1 output root and record all changed source identities.

## Schedule and dependency gates

Dates are work windows, not convergence promises. Restart the functional
sequence for the changed physical topology; previous distributed-array passes
cannot bypass these gates because peripheral and control routes changed too.

| Phase | Window | Dependency | Exit criterion |
|---|---|---|---|
| 1. Remove star paths and close A–E | Complete | User instruction | Implementation, software/connectivity tests and reviewed source manifest complete |
| 2. Revalidate small writes and control capture | Complete | 1 | Nine final-source cases pass independent waveforms and runtime measures |
| 3. Resolve large-array write race | Complete | 2 | Both large cases pass at 8 ns / 5 ns; diagnosed failure preserved |
| 4. Clock-class/control-path coverage | Complete (September 14) | 3 | Scored: 17 of 19 pass; 10T-mux class and wire-stress limits recorded |
| 5. PVT and per-device pilot | Complete (September 14) | 4 | All 12 seeds accounted for: 9 pass, 3 electrical 10T-mux failures, no numerical event |
| 6. Qualification and architecture backlog | Scope written (September 14); execution later | 5 reviewed | [Scope and briefs](V2_1_2_QUALIFICATION_SCOPE.md) written; extracted metal, half-select and yield work remain |

### Phase 1 — topology and compatibility (implemented)

Only `mode: distributed` is accepted. The omitted/default geometry matches
`interconnect_example.yaml` (illustrative 1 ohm / 0.1 fF per pitch). Do not
claim extracted metal. Keep `w_rc` storage/peripheral series RC and optional
`cell_pin_rc`; physical wires remain when these switches are off.

Verify individual consumer taps, not just resistor totals: periphery A/B,
decoder C, clock D and mux-select E from the input screen. Check decoder binary
polarity for small and non-power-of-two arrays, mirrored replica geometry and
sense connection, full/equivalent modes, all 6T/10T numeric/sweep paths, and
baseline reuse under candidates/PVT. Keep transistor classes and fixed clock
budgets unchanged unless a separately evidenced change is intended.

```bash
MPLBACKEND=Agg python3 -m unittest discover -s tests -v
python3 -m unittest discover -s dev/tests -v
python3 -m unittest discover -s size_optimization/openyield_v2/tests -v
python3 -m compileall -q sram_compiler utils tests size_optimization/exp_utils.py yield_estimation/model_lib
git diff --check
```

### Phase 2 — write-waveform gate (complete after repair)

Final-source matrix: `outputs/validation/V2.1.1/small-cases.json`.
Full arrays, lookup timing, nominal variation, seed 20260913, maximum step
20 ps, four MPI ranks per case, one case at a time, 900–1,200 seconds per solver run:

1. 8x4 6T TT write, local stubs on.
2. 8x4 6T SS eight-cycle sequence, local stubs on.
3. 8x4 6T SF eight-cycle sequence, local stubs on.
4. 8x4 6T TT eight-cycle sequence, local stubs off.
5. 8x4 muxed 10T SS eight-cycle sequence, local stubs on.
6. 10x6 6T SF write with row 9 to row 1 address-change hazard.
7. 64x4 6T TT write with row 37 to row 5 address-change hazard.
8. 16x16 6T SS read, run first to check the tight 4 ns deadline.
9. 8x4 6T FF / 1.1 V / −40 C eight-cycle sequence.

The sequence writes/reads both polarities twice. Probe every selected-row Q/QB,
unselected storage, all wordlines, actual local PRE/write enable, input at
local clock capture, DIN_dff, held data/complement, driver data, bitlines and
replica peripheral taps. Verify register capture precedes write enable, data
holds through the local drive interval, real bitline transitions perform the
write, Q/QB meet the frozen deadline and full retention window, and restoration
and release exclusion hold through the last cycle. Read phases require local
sense margin and correct retained OUT. Visually inspect representative write-1
and write-0 plots as well as independent crossings and runtime measures.

A late access, disturbed neighbor, missing edge, bad capture, precharge overlap,
incomplete restore or partial simulation is a failure to investigate. Do not
lengthen the clock or guard silently. A new numerical strategy uses the same
seed/circuit in a new preserved attempt. All nine cases passed before the
final large queue started.

### Phase 3 — large-array write and sequence (complete)

The initial `large-cases.json` is under the preserved evidence root. It scheduled
8x512 SS write first (8 ns), then 64x64 TT sequence (5 ns), full arrays, eight
ranks, original nominal seed 20260909, maximum step 20 ps, up to six hours per
solver run and fourteen hours per batch. Use sampled unselected storage probes
but every selected-row Q/QB and per-column control/data path.

The initial 64x64 sequence passed 10,185 checks. The initial 8x512 write failed
precharge/write exclusion at 477 columns (478 failures including the
aggregate). Investigate local PRE, write enable/complement, driver-current
path and Q/QB before changing circuit controls. A final stored value alone is
insufficient. Preserve the failed run and its fixed 8 ns clock.

After the repair, Phase 2 and both large cases passed in fresh directories.
The repaired wide write has minimum PRE90-to-WL50 margin 1,038.988 ps and
minimum PRE90-to-write50 margin 1,264.433 ps across all columns. The final
64x64 sequence has minimum PRE90-to-WL50 margin 268.154 ps. Both original
clocks and all thresholds remain unchanged. See the [final evidence record](../design/DISTRIBUTED_ONLY_V2_1_1.json).

Independent large cases may both execute even if one fails, but no failed or
partial result permits the subsequent statistical campaign. Record exact
runtime/scorer/model/simulator identity and all attempt artifacts. Neither the
old 1,200-second timeouts nor the interrupted pre-change run is current
passing evidence.

### Phase 4 — class boundaries and physical sensitivity

After reviewing the large results, archive a compact matrix before launching:
8x64/8x128 writes at final 5 ns/6 ns clocks; 48x20 unseen geometry; tall
SS reads and fast-corner address changes; 6T/10T mux alternatives; local series
stubs on/off; optional cell-pin RC; same-R/C wire refinement and explicit wire
stress. All cases use distributed wiring. There is no star comparison axis.

Changed decoder local stubs and assumed predecode placement require dedicated
height sensitivity checks. The three-pitch periphery and clock routes require
width sensitivity and local capture/skew inspection. Preserve thresholds and
clocks across physical variants; report failures rather than fitting them away.

Result (September 14, 2026): 17 of 19 lookup-clock cases pass under
`outputs/validation/V2.1.2-followup/` (queues A and B at four ranks each,
the 512x4 SS read and SF write at eight ranks and 9 ns). Widths to 128
columns, heights to 512 rows, the unseen 48x20 geometry, both fast-corner
address hazards, 6T mux, local stubs off, cell-pin RC and the same-R/C
refinement pass. Two failures, both a late sense enable at the frozen
class and both passing at the next class in fixed-clock diagnostics: the
16x16 10T column-mux SS sequence at 4 ns (passes at 4.5 and 5 ns and at TT)
and the 64x4 SS read with three times the wire R and C at 4.5 ns (passes at
5 ns). The table is unchanged; a 10T-with-mux budget one class up is a
recorded proposal. See the [follow-up record](../design/TIMING_FOLLOWUP_V2_1_2.md).

### Phase 5 — PVT/local mismatch

The supplied [first-500-run write failure inventory](../issue_reports/write_failure_cases_first_500.md)
is a historical triage input. Match each deck, source version, solver and seed
before interpreting its numerical failures; do not relabel it as post-fix
V2.1.1 evidence or count missing results as successful writes.
The V2.1.2 audit appended to that report found its source CSV, logs and
collection script absent, its step-size summary a non-fatal solver warning,
and one Open MPI (`ORTE`) failure from a different solver stack. Seven of its
configurations pass nominal V2.1.1 runs. Use the CLI `--vdd`/`--temperature`
options and keep the recorded `xyce` installation for every pilot sample.

Repair or disable the legacy materialized-MPI numerical-fallback path before
using it: it still deletes failed outputs and can consume multiple full
timeouts. The queue accepts nominal cases and, since September 14, per-device
cases on one rank with an explicit seed, which use the plain runtime retry
and never that fallback path. First use
three explicit per-device seeds for passing distributed SS read, SF write and
muxed 10T sequences; expand to ten only after review. Freeze timing/driver
baseline through all seeds and PVT samples. Keep electrical, numerical and
incomplete outcomes separate; no NaN is success. These are pilots, not yield
estimates or qualification records.

Result (September 14, 2026): twelve single-rank per-device samples, seeds
20261001 to 20261003, 5 % relative sigma: 16x16 SS read 3 of 3, 16x16 SF write
3 of 3, 8x4 FF cold sequence 3 of 3, 8x4 10T mux SS sequence 0 of 3 (the
Phase 4 class finding; sense enable at 1.167 to 1.182 cycles). No sample
needed a retry or printed a solver warning. The FF cold sequences show a
write-enable spike of 0.50 to 0.63 V (0.41 V nominal) at the read-to-write
boundary while the wordline is still high; no check fails and it is an open
TIME item. Expansion to ten seeds waits for review of the 10T-mux budget.

### Phase 6 — remaining scope

The [qualification scope](V2_1_2_QUALIFICATION_SCOPE.md) (September 14, 2026)
defines the extracted-metal inputs, the PVT/sample matrix and the two separate
briefs below; execution follows the reviewed Phase 5.
Define extracted-metal inputs and supported PVT/sample matrix before a full
qualification. The legacy campaign recalibrates timing; adapt it to frozen
lookup clocks before using it for this policy. Never promote partial/failed
screens to `sizing_table.json`.

Half-select needs an explicit write-mask/column-select architecture; current
writes drive all columns of a row. Legacy yield estimators still require path,
dependency and numerical validity work beyond their repaired return contract.
These remain separate briefs after current SRAM write/read/hold correctness.

## Status, resume and stop

The Phase 4/5 queues under `outputs/validation/V2.1.2-followup/` are complete
(`chain.log` ends with `ALL_DONE`); do not resume them. Their fixed-clock
diagnostics ran through the direct validator into `phase4-D/`.

Read-only status for the completed final-source Phase 3 queue:

```bash
python3 -m dev.v210_followup_queue --output outputs/validation/V2.1.1/large-queue --status
```

Do not resume this completed queue after source changes. Create a fresh
V2.1.1 output root for repaired sources and preserve the old checkpoints.

The runner is `dev/v210_followup_queue.py`; pass `--cases`, `--output`, `--xyce`
and `--budget-hours`. The small queue uses three hours, the large queue fourteen.
Obtain the exact Xyce path and command from the queue's `manifest.json` and
per-attempt `launch.json`; no new machine-local path belongs in source code.

A lock prevents duplicate writers to the same queue, but inspect all active
Xyce PIDs before starting another output root. Source/case/solver identities
must match on `--resume`. Interrupted cases get fresh attempt directories;
completed failures remain terminal until a new diagnosed case/queue is created.
The direct validator also requires a fresh case directory. Its historical
`--score-only` option rejects in-place regeneration; use read-only waveform
scoring functions for retained traces. Rejection/summary sidecars never replace
historical results. The small queue preserves its source snapshot before the
duplicate-column parser repair; generation and independent scoring are identical
to the large queue.
Create the queue's `STOP` file to stop after the current case; remove it before
intentional resume. SIGINT to the recorded supervisor PID immediately interrupts
its worker and cleans up owned MPI ranks while preserving partial evidence.

A background process schedules bounded simulations while the machine remains
available. It does not schedule future agent reasoning. Resume this thread
with the plan and live checkpoint to inspect outcomes and advance the gates.
