# V2.1.0 follow-up review and write validation

Historical V2.1.0 evidence record. The distributed-only implementation and
subsequent retention/CLI repairs are released as **V2.1.1**; see the
[current report](DISTRIBUTED_ONLY_V2_1_1.md) and
[evaluation schedule](../../plans/V2_1_1_TIMING_FOLLOWUP.md). Measurements,
source identities and report-time queue states below keep their original
labels and do not describe current qualification.

Review date: September 12, 2026 (America/Los_Angeles). The incoming V2.1.0
working tree is based on V2.0.11 `0433072`. User priorities are timing
correctness, the unfinished large arrays, and explicit write-waveform checks,
followed by PVT/local mismatch. The [resumable schedule](../../plans/V2_1_0_TIMING_FOLLOWUP.md)
contains dependencies, commands, limits and handoff instructions.

Two runtime validation defects and several independent-scorer coverage gaps
were repaired. All seven new write/sequence cases pass **3,153 checks** at
the unchanged 4 ns lookup clock. The 8x512 SS write has been launched with a
six-hour limit; the 64x64 TT sequence is queued next. Neither large case is
counted as passing evidence while running or pending.

Exact source identities, case specifications, measurements and launch state
are recorded in [the follow-up JSON](TIMING_LOOKUP_V2_1_0_FOLLOWUP.json).
Raw artifacts are under `outputs/validation/V2.1.0-review-20260912/`.

## Review findings and repairs

### P1: retention checking left two unobserved intervals

`_add_access_checks()` checked the access at `1 ns + (cycle + 1.2) T`, but
checked retention only over `[1.5, 1.65] T`. A disturbance after the access
deadline and before that interval, or immediately before the next access,
could escape the runtime check.

`VHOLD_ERROR_n` now covers `[1.2, 1.7] T`, including the complete interval
from the deadline to the next-access boundary. Q/QB are checked on writes;
read OUT is also checked on reads. The final eight-cycle interval ends at
the existing `1 ns + 8.7 T` analysis stop.

An actual Xyce negative control with a known PWL disturbance reproduced the
blind spot: at a 4 ns clock, a 0.5 V Q/QB error at 6.1 ns produced
`OLD_HOLD_ERROR = 0`, `VACCESS_ERROR_0 = 0`, and the repaired
`VHOLD_ERROR_0 = 0.5 V`. This tests measurement rejection, not an observed
failure of a transistor SRAM. Its deck, trace and measures are in
`hold-negative/`.

A read-only audit of the 14 historical final-release traces checked target
Q/QB and read OUT over all **49 complete retention windows**; all pass the
extended window. This audit used waveform samples and interpolated boundary
values, not new simulations. It does not add unrecorded local-control probes
to the historical traces. See `retention-audit.json`.

### P2: CLI success did not require usable primary metrics

The CLI could return success when voltage checks passed but `TREAD_TOTAL`
was `FAILED`. The same gap applied to write/sequence delay and power metrics.
A regression first reproduced that successful return with a failed delay.

The CLI now requires every requested sample to have a finite parsed operation
delay, `PAVG`, `PSTC` and `PDYN`, and records `metrics_checked`. A missing or
failed metric rejects the ensemble and preserves `summary.json` and the
measurements. The existing voltage/release checks still apply independently.

### P1: independent write evidence did not cover its stated scope

The local validator skipped the final sequence cycle's independent
retention/restoration/precharge-order checks. Single writes did not check
every selected-row cell's complementary data. Sequences also lacked quiet
unselected-wordline checks and write-drive/precharge-exclusion checks, and
sense timing was sampled at the near enable rather than its local terminal.

The additional scorer in `dev/v210_waveform_checks.py` reads `.prn` values
independently of `.MEASURE`. Its required probes and checks now include:

- Every selected-row Q/QB pair, with transitions for both write polarities
  and retention from deadline through the next-access boundary, on all cycles.
- Probed unselected-cell Q/QB and every unselected wordline, including sequences.
- Per-column cell WL/BL/BLB terminals, local precharge `ENB_end` and write-driver
  `EN_end` with RC enabled, plus actual local sense-amplifier `EN_end`.
- Bitlines restored before a write, an observed driven bitline transition,
  correct low/high levels by the deadline, precharge off during the write
  interval, and wordline release before local precharge with no rebound.
- Complete finite traces, final restoration, read OUT retention and the
  existing 0.3 V sense-differential threshold sampled at local enable.

Four local test methods exercise valid fixtures and corrupt traces: wrong
complements, late retained-data errors, incomplete final restore, unselected
wordline pulses, local precharge/write overlap, undriven bitlines, missing
probes, nonfinite data and truncated traces. Missing inputs fail the scorer.
These checks participate in the validator's overall pass/fail result.

## Fresh write waveform results

All arrays below are full 4x4 transistor arrays with local RC enabled. TT is
1.0 V / 25 C; SS and SF are 0.9 V / 125 C. Distributed wiring remains the
illustrative 1 ohm / 0.1 fF per pitch model. Every cell is probed in these
small cases. Runs use nominal variation, seed 20260912, one Xyce rank, a
20 ps maximum step and the unchanged four-stage precharge guard.

Clock-to-Q is measured at 50% crossings; it is reported separately from the
90% data-valid deadline checks. Release margin is the minimum time from
WL falling through 10% VDD to local PRE falling through 90% VDD.

| Array / operation | Checks | Maximum write clock-to-Q (ps) | Minimum release-to-precharge (ps) | Result |
|---|---:|---:|---:|---|
| 6T star TT write 1 | 99 | 244.34 | 109.99 | Pass |
| 6T star SS eight-cycle sequence | 591 | 549.11 | 259.97 | Pass |
| 6T star SF eight-cycle sequence | 591 | 508.87 | 207.58 | Pass |
| 6T distributed TT write 1 | 99 | 223.63 | 132.07 | Pass |
| 6T distributed SS eight-cycle sequence | 591 | 502.86 | 303.85 | Pass |
| 6T distributed SF eight-cycle sequence | 591 | 467.04 | 258.65 | Pass |
| 10T muxed distributed SS eight-cycle sequence | 591 | 512.20 | 304.01 | Pass |

Each sequence writes 1, reads 1, writes 0 and reads 0, twice. Thus these
results include both write polarities and the final read/retention interval.
The check total comprises 3,139 waveform checks and 14 aggregate runtime
checks; individual runtime measures are preserved in each `.mt0`.

The following plots were generated and visually inspected, showing local
write/precharge enable, first/last wordlines, bitlines, and Q/QB for write 1
and write 0. The shaded interval is the extended retention check:

- [SS star writes](../../outputs/validation/V2.1.0-review-20260912/star_4x4_SS_read_write-write-waveforms.png).
- [SF distributed writes](../../outputs/validation/V2.1.0-review-20260912/distributed_4x4_SF_read_write-write-waveforms.png).
- [SS muxed 10T writes](../../outputs/validation/V2.1.0-review-20260912/distributed_4x4_10t_mux_SS_sequence-write-waveforms.png).

The CLI additionally completed a real two-sample, seed-20260912 per-device
2x2 write with finite primary metrics and passing access/hold/restore/release
checks. It preserves its waveform and summary under `cli-write/`. That CLI
smoke is not included in the seven independently scored cases above.

## Scheduling and remaining work

The serial queue was launched at **20:25 PDT on September 12** (03:25 UTC on
September 13), supervisor PID **692597**. Its first case is the full 8x512
distributed SS write at **8 ns**, followed by the full 64x64 distributed TT
eight-cycle sequence at **5 ns**. Each uses eight ranks, a 20 ps maximum step,
seed 20260909 and a six-hour solver limit. Total batch budget is fourteen
hours; only one simulation case runs at a time. All selected-row cells and
sampled unselected cells are probed. Neither case uses equivalent cells.

The live `large-queue/checkpoint.json` is authoritative; the JSON adjacent to
this document is a report-time snapshot. The queue survives the current chat
turn while this machine remains available. It schedules simulations only;
further agent reviews are resumed from the recorded plan, not automatically
triggered by the queue.

The queue checks source/case/model/simulator identities, retains an executable
source archive, records launches before starting Xyce, preserves each attempt,
checks artifact hashes on resume and rejects a duplicate worker. A failure or
timeout in one large case does not suppress the other independent case.
Further class/PVT/mismatch expansion requires review of both results.

Open issues remain: large-array completion, final-clock boundary/control-path
coverage, wider PVT and local mismatch, qualified metal, half-select
architecture, and numerical validation of the legacy yield estimators.
Before MPI mismatch work, repair or disable the legacy materialized runner's
fallback path, which deletes failed outputs and can consume repeated full
timeouts. The new queue accepts nominal cases only until that is addressed.

## Software and provenance checks

- **97 tracked compiler tests**, **33 local development tests**, and **six
  offline optimizer tests** pass on Python 3.11.7. The optimizer tests passed
  at the incoming baseline; this follow-up changes no optimizer code.
- Compilation and `git diff --check` pass. No lint/type-suite claim is made.
- All 37 recorded release source hashes match the incoming tree; the timing
  lookup hash matches the archived final table, and transistor classes match
  V2.0.11. The existing 32-deck comparison remains historical evidence; it was
  not rerun in this follow-up. No transistor width, timing budget or sizing
  formula was changed here.
- The incoming tracked diff and untracked/ignored source contents were
  preserved before follow-up edits. The independent plan review verified
  all 210 archived files having initial recorded hashes.

Historical CSVs and `docs/qualification/*.json` remain untouched.
`sizing_table.json` remains empty. These results are functional diagnostics,
not full PVT/mismatch, extracted-metal, half-select or yield qualification.
