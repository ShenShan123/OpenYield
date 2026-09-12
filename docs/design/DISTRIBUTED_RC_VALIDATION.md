# Distributed RC validation — V2.0.7 record and V2.0.8 audit

The V2.0.7 record below is retained unchanged as evidence; the
[V2.0.8 audit](#v208-audit--2026-09-09) follows it.
The separate [V2.0.9 review](DISTRIBUTED_RC_V210_REVIEW.md) extends the checks
to larger arrays and records subsequent fixes and remaining timing limits.

## V2.0.7 RC validation — 2026-09-09

This validates the implementation against the [plan](DISTRIBUTED_RC_PLAN.md).
The [model guide](DISTRIBUTED_RC_MODEL.md) defines the supported topology and
configuration. Metal values are illustrative (1 ohm / 0.1 fF per 0.6 um pitch),
not foundry extraction. No qualification record or sizing coefficient is promoted.

### Software checks

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

### Electrical checks

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

### Compatibility and remaining qualification

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

## V2.0.8 audit — 2026-09-09

### Audit of the V2.0.7 record and design documents

- The quoted equivalent-mode and subdivision deltas reproduce from the retained
  `result.json` files: mode 4 versus full-real differs by at most 0.209 ps (6T)
  / 0.131 ps (10T) in local WL delay and 0.195 mV / 0.331 mV in sense
  differential; fourfold subdivision changes the 6T WL delay by 0.015 ps and the
  sense differential by 0.417 mV. The two retained 2.5 ns SS/SF failures are
  output-capture and write-completion timing failures; both 5 ns reruns pass.
- The "sense differential at enable" of about 0.99 V is not a sensing margin.
  With the K=1 matched replica, sense enable fires one delay chain after the
  replica bitline crosses the TIME threshold, by which time a real bitline
  discharging at the same rate has swung fully. The 0.3 V check is therefore
  a functional check; a margin needs weak cells or a larger K.
- Plan item 3 formerly asked for periphery at "specified" locations; the
  implementation fixes WL drivers at column zero and bitline periphery at row
  zero, and the [plan](DISTRIBUTED_RC_PLAN.md) and
  [model guide](DISTRIBUTED_RC_MODEL.md) now state that. Plan item 5 lists
  half-select disturbance; no sequence deck exercises a half-selected column
  because every column carries a write driver and the same data, so that item
  remains open.
- The model guide did not state that TIME's replica input is wired to the
  replica sense amplifier's internal node `XREPLICA_SENSEAMP:IN_end` (a Xyce
  hierarchical reference used as a connection). It does now.
- Code findings fixed in V2.0.8: `InterconnectConfig` constructed directly
  defaulted `cell_pin_rc` to true even for distributed mode, contradicting the
  documented default that `resolve_interconnect()` applied; `main_sram.py` had
  no in-memory way to select an interconnect YAML although it is the main
  entrance. `interconnect.load_interconnect()` now serves the CLI and the new
  `INTERCONNECT_CONFIG` setting. The non-square replica tap mapping (K active
  cells on the last K RWL taps, dummies on the rest) was checked for 8x4, 4x8
  with K=2, and 2x4.

### Software checks

- 61 tracked compiler tests pass under Python 3.11 and under the project's
  Python 3.9 environment (59 from V2.0.7 plus the `cell_pin_rc` default, YAML
  loader and main-entrance regressions); 6 offline optimizer tests and 21 local
  development tests pass; `compileall` and `git diff --check` are clean.
- Deck compatibility: 72 star decks (6T/10T; read, write, read&write; mux on
  and off; RC off, default and custom 777 ohm / 7 fF; `fixed` and `rules_only`)
  generated from a detached worktree at V2.0.7 commit `315333b` and from
  V2.0.8 are byte-for-byte identical apart from the checkout path inside
  `.include`. V2.0.8 changes no generated netlist.

### Electrical checks

Same Xyce binary and settings as the V2.0.7 record (KLU, 10 ps maximum step,
2 ps output interval, fixed driver scales, 100 ohm / 1 fF local RC unless
noted, illustrative 1 ohm / 0.1 fF per pitch unless noted). All cases run the
write 1 / read 1 / write 0 / read 0 sequence twice at 5 ns and the same scorer
as V2.0.7, including retention of every real unselected cell and far-end
restoration of every bitline (287 checks at 16x16). Per-device cases use
independent 5% `vth0/u0/voff` variation with seed `20260909`, one sample each.

| Cases | Count | Conditions | Outcome |
|---|---:|---|---|
| Non-square full-real 6T, 8x4 | 1 | TT, 25 C, 1 V | Pass |
| 4x8 6T with K=2 replica cells and column mux | 1 | TT, 25 C, 1 V | Pass |
| 8x4 10T, column mux, per-device sample | 1 | TT, 25 C, 1 V | Pass |
| Distributed metal without `w_rc`: 4x4 6T, 4x4 10T mux | 2 | TT, 25 C, 1 V | Pass |
| 4x4 6T with `cell_pin_rc: true` (per-cell stubs plus wires) | 1 | TT, 25 C, 1 V | Pass |
| 8x4 equivalent modes 1 and 3 (6T) and 2 (10T), actual extraction | 3 | TT, 25 C, 1 V | Pass |
| 8x4 6T SS and SF | 2 | 125 C, 0.9 V | Pass |
| 16x16 6T full-real | 1 | TT, 25 C, 1 V | Pass |
| 16x16 6T, 10 ohm / 1 fF per pitch, far (15,15) and near (0,0) cell | 2 | TT, 25 C, 1 V | Pass |
| 16x16 10T, column mux, per-device sample | 1 | TT, 25 C, 1 V | Pass |

Fifteen of fifteen cases pass; no case is retained as failed. Measured local
wordline delay (wl_en to the selected cell's tap, 50%): about 43 ps at 4x4
and 8x4, 35 ps at 16x16 (the fixed-mode wordline scale grows with columns),
100 ps (SS) and 89 ps (SF) at 8x4, 125 C, 0.9 V; 18 ps without `w_rc`
(no driver stubs) and 57.5 ps with `cell_pin_rc: true` (the probe sits behind
the per-cell stub). At 16x16 with tenfold wires the far cell's wordline
arrives 1.5 ps after the near cell's and 14.4 ps later than with the 1x wires,
and the sense differential drops from 0.99 V to 0.96 V. Equivalent modes 1 and
3 differ from the full-real 8x4 array by at most 0.225 ps in WL delay and
0.536 mV in sense differential. Decks, `.prn` traces, results, per-device model
cards, audits and summaries are retained locally under ignored
`outputs/validation/V2.0.8/`; the runner is `dev/validate_distributed_rc.py`
with the case files listed in `summary_*.json`.

### Remaining qualification

Unchanged from V2.0.7: metal values are illustrative; wire coupling is absent;
one per-device sample per case is not a yield distribution; half-select column
disturbance is not exercised; larger arrays, extracted geometry, the intended
PVT/mismatch population and operating-clock qualification are required before
recalibrating driver rules or publishing yield claims. No sizing-table record
is promoted. Historical V2.0.5 evidence remains unchanged.
