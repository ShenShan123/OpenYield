# Historical V2.1.0-labeled distributed-only evaluation plan

This is the pre-release schedule snapshot. The user designated the
distributed-only change **V2.1.1**; use
[V2_1_1_TIMING_FOLLOWUP.md](V2_1_1_TIMING_FOLLOWUP.md) for the current
checkpoint, completed large-array outcomes and resume instructions. Running
PIDs and pending states below describe this historical snapshot only.

Updated September 12, 2026 (America/Los_Angeles) for the user's explicit
instruction: remove every star-RC implementation, keep only distributed RC as
the default, and alter the evaluation. Write correctness remains the first
functional priority. Maximum compute: eight ranks total, one large case at a
time, one BLAS thread per rank.

Read `AGENTS.md`, `docs/design/STAR_RC_SCREEN_V2_1_0.md` (the preserved input
screen) and `docs/design/DISTRIBUTED_ONLY_V2_1_0.md` before resuming.

## Current checkpoint

- The topology change is implemented: distributed-only defaults and rejection
  of star options; tapped array/replica/dummy/equivalent loads; BL/RBL peripheral
  ladders; decoder address/true/complement/enable routing; DATA_DFF clocks;
  mux-select lines. Local series stubs remain independently selectable.
- Driver transistor classes and lookup clocks are fixed. All 64 compared
  serialized sizing vectors match the incoming explicit-distributed baseline.
- 112 compiler tests, 51 development tests and six optimizer tests pass. Numeric/sweep, 6T/10T,
  default/custom geometry, RC conservation, decoding and rejection paths are
  covered. The local scoring-source manifest has been reviewed and refreshed.
- The seven fresh small-array cases completed under
  `outputs/validation/distributed-only-V2.1.0/small-queue/`: all pass 6,463
  checks. The distributed-default CLI also passed a two-sample per-device write.
- Phase 3 is running under `large-queue/`, supervisor **759485**, launched
  **2026-09-12 21:17 PDT** (2026-09-13 04:17 UTC): 8x512 SS write first, then
  64x64 TT sequence. Check live state before another launch; the total limit
  is eight active simulator ranks.
- The previous queue (PID 692597) was gracefully interrupted, including its
  MPI ranks. Its 8x512 write remains partial evidence under the old baseline.
  It must not resume against the changed sources.

## Preserve the two historical baselines

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

## Schedule and dependency gates

Dates are work windows, not convergence promises. Restart the functional
sequence for the changed physical topology; previous distributed-array passes
cannot bypass these gates because peripheral and control routes changed too.

| Phase | Window | Dependency | Exit criterion |
|---|---|---|---|
| 1. Remove star paths and close A–E | Day 1 | User instruction | Complete implementation, software/connectivity tests, reviewed source manifest |
| 2. Revalidate small writes and control capture | Day 1–2 | 1 | Entire fresh small matrix passes independent waveforms and runtime measures |
| 3. Retry unfinished large arrays | Day 2 onward | 2 | 8x512 SS write at 8 ns and 64x64 TT sequence at 5 ns have complete scored traces, or preserved diagnosed failures |
| 4. Clock-class/control-path coverage | After large failures reviewed | 3 | Final-clock width/height boundaries, address hazards and wire refinements scored |
| 5. PVT and per-device pilot | After deterministic functional gate | 4 | Every scheduled seed accounted for, no invalid sample counted as success |
| 6. Qualification and architecture backlog | Later | 5 reviewed | Explicit physical/statistical qualification scope; separate half-select/yield briefs |

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

### Phase 2 — fresh write-waveform gate (passed)

Exact matrix: `outputs/validation/distributed-only-V2.1.0/small-cases.json`.
Full arrays, lookup timing, nominal variation, seed 20260913, maximum step
20 ps, four MPI ranks per case, one case at a time, 900 seconds per solver run:

1. 8x4 6T TT write, local stubs on.
2. 8x4 6T SS eight-cycle sequence, local stubs on.
3. 8x4 6T SF eight-cycle sequence, local stubs on.
4. 8x4 6T TT eight-cycle sequence, local stubs off.
5. 8x4 muxed 10T SS eight-cycle sequence, local stubs on.
6. 10x6 6T SF write with row 9 to row 1 address-change hazard.
7. 64x4 6T TT write with row 37 to row 5 address-change hazard.

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
seed/circuit in a new preserved attempt. All seven cases must pass before the
large queue is started.

### Phase 3 — large-array write and sequence (running)

The fresh `large-cases.json` is under the new evidence root. It schedules
8x512 SS write first (8 ns), then 64x64 TT sequence (5 ns), full arrays, eight
ranks, original nominal seed 20260909, maximum step 20 ps, up to six hours per
solver run and fourteen hours per batch. Use sampled unselected storage probes
but every selected-row Q/QB and per-column control/data path.

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

### Phase 5 — PVT/local mismatch

Repair or disable the legacy materialized-MPI numerical-fallback path before
using it: it still deletes failed outputs and can consume multiple full
timeouts. The new queue intentionally accepts nominal cases only. First use
three explicit per-device seeds for passing distributed SS read, SF write and
muxed 10T sequences; expand to ten only after review. Freeze timing/driver
baseline through all seeds and PVT samples. Keep electrical, numerical and
incomplete outcomes separate; no NaN is success. These are pilots, not yield
estimates or qualification records.

### Phase 6 — remaining scope

Define extracted-metal inputs and supported PVT/sample matrix before a full
qualification. The legacy campaign recalibrates timing; adapt it to frozen
lookup clocks before using it for this policy. Never promote partial/failed
screens to `sizing_table.json`.

Half-select needs an explicit write-mask/column-select architecture; current
writes drive all columns of a row. Legacy yield estimators still require path,
dependency and numerical validity work beyond their repaired return contract.
These remain separate briefs after current SRAM write/read/hold correctness.

## Status, resume and stop

Read-only status for the active Phase 3 queue:

```bash
python3 -m dev.v210_followup_queue --output outputs/validation/distributed-only-V2.1.0/large-queue --status
```

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
historical results. The small queue preserves its preflight-fix source snapshot;
the large queue uses the hardened runner, with identical generation/scoring.
Create the queue's `STOP` file to stop after the current case; remove it before
intentional resume. SIGINT to the recorded supervisor PID immediately interrupts
its worker and cleans up owned MPI ranks while preserving partial evidence.

A background process schedules bounded simulations while the machine remains
available. It does not schedule future agent reasoning. Resume this thread
with the plan and live checkpoint to inspect outcomes and advance the gates.
