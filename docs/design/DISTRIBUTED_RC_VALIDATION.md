# V2.0.7 RC validation — 2026-09-09

This validates the implementation against the [plan](DISTRIBUTED_RC_PLAN.md).
The [model guide](DISTRIBUTED_RC_MODEL.md) defines the supported topology and
configuration. Metal values are illustrative (1 ohm / 0.1 fF per 0.6 um pitch),
not foundry extraction. No qualification record or sizing coefficient is promoted.

## Software checks

- Before implementation: 46 compiler/optimizer tests and 62 subtests passed.
- RC regressions first reproduced dropped values, extraction-context/cache
  mismatches and frozen-baseline/load errors. Checkpoint `397df3f` records RED;
  `4f7849c` records the initial fix and 31 passing focused tests / 54 subtests.
- Distributed invariants first failed on the absent implementation
  (`f127b5b`). The final tracked suite passes 65 tests / 112 subtests, including
  numeric and sweep dimensions, both cell types, mux paths, equivalent modes
  1–4, physical fingerprints, CLI run identities and per-device specialization.
- All 59 compiler tests pass under the project's Python 3.9 environment.
- The interconnect unit/integration tests cover 109 of 119 executable lines
  (91.6%) using Python's `trace` module. This is coverage of the new wire module,
  not a coverage claim for the entire legacy compiler.
- The local development suite checks scoring/reporting, manifest validation,
  retry behavior and distributed probe mapping. Compiler tests and CLI
  generation are also exercised from an exported runtime tree without `dev/`.
- `git diff --check`, compilation and local documentation links are checked.

## Electrical checks

Xyce `DEVELOPMENT-202312200606-(Public_Release-7.4.0-36-gb7bb12d8)`, KLU,
maximum internal time step 10 ps, waveform output interval 2 ps. All arrays are
4x4 with fixed driver scales and 100-ohm / 1-fF local storage/peripheral RC.
Nominal cases explicitly disable sampling. Per-device cases use independent
5% `vth0/u0/voff` variation with seed `20260909`, one sample per cell type.

| Cases | Count | Conditions | Outcome |
|---|---:|---|---|
| Full-real 6T and 10T sequences | 2 | TT, 25 C, 1 V, 5 ns | Pass |
| Mux-enabled 6T and 10T sequences | 2 | TT, 25 C, 1 V, 2.5 ns | Pass |
| Near and middle 6T access positions | 2 | TT, 25 C, 1 V, 2.5 ns | Pass |
| Fourfold wire subdivision, 6T | 1 | Same physical R/C as full-real reference | Pass |
| Equivalent mode 4, 6T and 10T | 2 | TT, 25 C, 1 V, 5 ns; actual extraction | Pass |
| Per-device 6T and 10T sequences | 2 | TT, 25 C, 1 V, 5 ns | Pass |
| SS and SF sequences, both cell types | 4 | 125 C, 0.9 V, 5 ns | Pass |
| Read/write with next-row address change, both cell types | 4 | TT, 25 C, 1 V, 5 ns | Pass |
| Additional SS/SF 6T clock stress | 2 | 125 C, 0.9 V, 2.5 ns | Fail; retained |

Sequence cases execute write 1 / read 1 / write 0 / read 0 twice. The scorer
checks finite complete waveforms, target Q/QB and output data, sense-input
differential at enable, retention after release, unselected-row disturbance,
far-end bitline restoration and precharge after wordline release. Hazard cases
check the neighboring row during the address-change/release interval.

The two short-clock failures are substantive: SS output settles after the
capture check; SF writing has not completed at capture. Both pass the same
checks at 5 ns. This limits the clock claim; passing simulator measures alone
would not establish correctness. No failed case is counted among the 19 passes.

At TT, mode-4 equivalent versus full-real arrays differ in local WL delay by at
most 0.210 ps (6T) / 0.131 ps (10T), and sense differential by at most
0.196 mV / 0.331 mV across the eight accesses. Fourfold subdivision changes the
6T WL delay by at most 0.015 ps and sense differential by 0.417 mV. These are
small-array comparisons, not accuracy bounds for other geometries or activity.

Generated decks, logs, `.prn` traces, results, model/deck hashes and summary
metadata are retained locally in ignored `outputs/validation/V2.0.7/`.
The diagnostic runner is `dev/validate_distributed_rc.py`; its `*-cases.json`
inputs are retained with those outputs. The Xyce binary SHA-256 is
`13971579e5902562364d9b5bb53ad23ab19990907b44fe93a887d71cd9b7480e`.

## Compatibility and remaining qualification

A detached worktree at V2.0.6 commit `029626b` supplied twelve comparison
decks. Eight RC-off/explicit-100-ohm star decks are byte-for-byte identical.
The four former-default decks change only the ten replica-path resistors from
10 to 100 ohm. Custom R/C and equivalent extraction intentionally change.

Full-real arrays remain the reference. Equivalent modes do not simulate
retention or mismatch in omitted devices. One per-device sample per cell type
does not establish a yield distribution. Larger arrays, extracted metal and
coupling, the complete intended PVT/mismatch population and operating-clock
qualification are required before recalibrating driver rules or publishing
yield claims. Historical V2.0.5 evidence remains unchanged.
