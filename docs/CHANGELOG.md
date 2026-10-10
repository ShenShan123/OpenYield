# Changelog

One entry per release, newest first, each naming its design record. Entries
were condensed after V2.2.6; the previous full text is at
`git show df3fab4:docs/CHANGELOG.md`, the text before the V2.1.10 condensation at
`git show 945815a:docs/CHANGELOG.md`, and the V2.0.1 / V2.0.2 evidence tables at
`git show c3f6f44:CHANGELOG.md` and in Parts II and III of
`git show b4987df:sram_compiler/CIRCUIT_REVIEW.md`. The historical sizing and timing
proposals and that circuit review were purged after V2.2.6; retrieve them with
`git show b4987df:<path>`.

## V2.2.7 development — 2026-10-10 — independent signal and power RC calibration

[Complete macro record](design/FULL_RC_CALIBRATION_V2_2_7.md), preserving the
[initial record](design/RC_CALIBRATION_V2_2_7.md). Not a completed release.

- A reproducible 8x8 6T FreePDK45 macro and standalone array tile use KLayout
  geometry and official Calibre DRC/LVS and coupled xRC extraction.
- Complete controller/register/output-latch layout passes official Calibre DRC
  with zero violations and LVS `CORRECT`; xRC retains R, C and coupling.
- TIME_CONTROL sizes, model defaults, guard/delay settings and fallback sizing
  policy move to `config_yaml/time_control.yaml`. Authorized controller sizing
  legalization and delay-driver changes pass scoped native waveform checks.
- Defaults bind 709 independently configurable per-net networks, including
  separate VDD/VSS resistive meshes for IR drop. Critical route classes have
  separate fallback settings. Extracted junction geometry is back-annotated.
- At nominal 8×8 TT, 1 V, 25 °C, 9 ns, four complete W1/R1/W0/R0 accesses pass
  1,241 waveform checks for each of three models. Mean edge error against xRC
  falls from 1,334.66 ps to 3.40 ps in the calibrated compiler.
- Automatic defaults apply only to this reference. Other configurations retain
  illustrative routing and ideal supplies unless an explicit accepted profile
  is supplied. No full V2.2.7 screen is complete.
- [Additional-size calibration](design/MULTI_SIZE_RC_CALIBRATION_V2_2_7.md)
  adds complete 8×16, 16×8 and 16×16 physical references, SNM coupons and
  explicit per-net profiles. At nominal TT / 1 V / 25 °C and a fixed 30 ns
  clock, all 22,881 checks pass; mean edge error falls to 4.05–6.00 ps.
  Physical supply feeds correct small-macro wordline glitches with schematic
  sizing frozen. `.IC` parsing excludes measurement thresholds; cap-only
  dummy-net export preserves initialization. Automatic 8×8 selection stays
  scoped; larger unseen sizes and the full release screen remain unqualified.

## V2.2.6 — 2026-10-01 — configuration review fixes

[Review record](design/CONFIGURATION_REVIEW_V2_2_6.md).

- Custom 6T/10T SNM construction passes valid factory keywords.
- The audit and the shared/custom model-card builder use one active-statement
  parser. Comments and later directives no longer count as parameters.
- Sample counts are validated before the main entrance can truncate them.
- One coverage rule: single-cell SNM decks do not claim full-array mismatch
  coverage.

Models, circuits, driver sizes and timing tables are unchanged. No V2.2.6
waveform screen is complete; 256x256 remains unresolved.

## V2.2.5 — 2026-09-28 — macro entrance and generation checks

[Design record](design/COMPILER_ENTRY_V2_2_5.md). `main_sram.py` is the documented
macro entrance. It generates decks without Xyce, applies local variation by
default, and saves configuration provenance. Invalid targets, operations, sample
controls and dimensions fail before a deck or result is reported.

- Model cards: SS, FF, FS and SF set `NMOS_VTH u0` twice. The earlier `0.05` is
  commented out, so each card keeps `0.049`, as TT already did. A compiler
  guard rejects future duplicates
  ([follow-up](design/COMPILER_ENTRY_V2_2_5.md#nmos_vth-model-card-follow-up)).
- Diagnostics, not a screen:
  - Eight new short-clock negative controls and the six preserved ones failed
    as intended ([evaluation](V2_2_5_NEGATIVE_CONTROL_EVALUATION.md)).
  - On the corrected model hashes, 22 of 22
    [corner cases](V2_2_5_CORRECTED_CORNER_CASES.md) (20,960 checks) and 4 of 4
    class-boundary cases (87,975 checks) passed.
  - The positive screen ran on the old hashes and stopped incomplete; the
    [simulation summary](V2_2_5_SIMULATION_SUMMARY.md) reconciles it with the
    diagnostics.
- Docs: the resolved first-500 write-failure inventory is retrievable with
  `git show 39ccae8:docs/issue_reports/write_failure_cases_first_500.md`.

## V2.2.4 — 2026-09-26 — passive replica loads and V2.2.3 review fixes

[Design record](design/PHASED_CONTROL_V2_2_4.md). **Not screened; 256x256 has no
working operating point.** The replica column changed, so no V2.2.3 result
carries over.

- Replica load cells are passive (`ReplicaColumn(active_rows=replica_k)`).
  - In V2.2.3, leakage from 511 tied-off load cells fired the 512-row sense
    enable early. Seven of 14 512x4 per-device draws failed at FS 0.9 V and
    FF 1.1 V, 125 C (worst 0.177 V against 0.252 V).
  - The same draws now pass at 0.328-0.539 V.
  - The first passive cell shorted VDD to VSS; `PSTC` exposed it, and it was
    fixed before the re-runs.
- Other fixes:
  - The write register starts in a consistent state (`we = 0`).
  - The variant clock floor is restored.
  - An injected `timing_config` no longer bypasses the 512x256 envelope.
- Screen tooling: the runner and checker take the target column, probe
  one-row address bits, and use `ACCESS_DEADLINE`. Large arrays get a seeded
  operating point: 128x128 in 17 min instead of 2.8 h, 256x128 in 34 min.
  256x256 had not converged after 5.6 h.
- Manifests and tests:
  - `tests/spice/v224_cases.json` has 330 cases; later 8x4 mux-read cases that
    switch select input bring it to 334.
  - `v224_mc_cases.json` has 1,444 cases and `v224_negative_cases.json` has 6.
  - 188 tracked tests.
  - Prior evidence: the V2.2.3 screen ran 284 of 285 cases, and a discovery
    campaign passed 1,474 of 1,487; both failure causes are fixed here.
- Repository cleanup:
  - Unused entry points and helpers were removed.
  - Python regression and waveform scripts are now ignored and stay local;
    JSON manifests and evidence remain tracked.
  - `AGENTS.md` holds the working conventions.

## V2.2.3 — 2026-09-20 — supported envelope, joint-dimension clocks, one access deadline

[Design record](design/PHASED_CONTROL_V2_2_3.md). **Not screened; the V2.2.2
evidence does not certify it.**

- Envelope: `supported_envelope` in `timing_lookup.json` sets the limit at
  512x256. Every timing mode enforces it.
  - The 512-column class is removed.
  - The fifteen 8x512 V2.2.2 cases are now historical only.
- Joint clock rule: `half = max(row, column) + min(row_excess, column_excess)`.
  - Every screened size keeps its period.
  - 128x128 goes from 16.5 to 20.5 ns, 256x256 from 20.25 to 26.25 ns, and
    512x256 from 27.75 to 33.75 ns.
- The runtime cards and the checker share `ACCESS_DEADLINE = 0.68`, replacing
  `k + 0.7 T`. The checker's sense bar is 0.28 VDD. One `probed_rows` rule
  serves both.
- `tests/spice/v223_cases.json` has 285 cases and `v223_negative_cases.json`
  has 6. 180 tracked tests.

## V2.2.2 — 2026-09-20 — precharged startup, reported margins, re-screen

[Design and validation record](design/PHASED_CONTROL_V2_2_2.md). The controller
netlist is unchanged.

- Decks start from the precharged clock-low state (`_init_precharged_columns`).
  This removes V2.2.1's hand-patched FF 10T mux read.
- The checker reports read-output, write-flip, restore and precharge margins.
  `time_generate.py` carries a measured timing diagram.
- Twelve per-device read&write cases were added, from 32x32 to 8x512.
- Evidence: 254 of 254 cases, 2,164,669 checks, in
  [`docs/data/PHASED_CONTROL_V2_2_2.json`](data/PHASED_CONTROL_V2_2_2.json).
  - The recorded hashes equal the tree.
  - The 2 ns negative control fails.
  - 170 tracked tests.
- A post-release pass recorded
  [coverage gaps](design/PHASED_CONTROL_V2_2_2.md#what-the-screen-does-not-cover):
  - The clock policy takes the maximum of the row and column budgets, and no
    screened array is large in both.
  - Sense margin falls from 0.90 V at 32 rows to 0.47 V at 512 rows.
  - Each corner is screened at one voltage and temperature.
  - The runtime read deadline is 0.7 T, but the checker uses 0.68 T.

## V2.2.1 — 2026-09-20 — clock-high access and clock-low recovery

[Design and validation record](design/PHASED_CONTROL_V2_2_1.md).

- Access happens while the clock is high. Clock-low releases the array and
  precharges it after every operation.
- The secondary hold latches and slot logic are removed. The 16x16
  control/data path uses 1,076 MOS, against 1,384.
- Reads isolate the sense inputs and release WL before regeneration. Writes
  wait for the far-end WEN.
  - New `TIME_CONTROL` inputs: `iso_far`, `sen_far`, `wen_far`.
- `v2.2.0-timing-3` sets the budgets. Capture-boundary, enable and isolation
  checks reject invalid accesses.
- Opt-in Xyce tests are tracked under `tests/spice/`.
- Identities hashed by the evidence keep their V2.2.0 labels; only
  `version.py` changes.
- Evidence: 242 of 242 cases, 1,767,615 checks, in
  [`docs/data/PHASED_CONTROL_V2_2_1.json`](data/PHASED_CONTROL_V2_2_1.json).
  The screen used a pinned overlay; the tracked sources regenerate every
  screened deck byte for byte.

## V2.1.10 — 2026-09-19 — registered write data

[Record](design/WRITE_LATCH_V2_1_10.md).

- Write data and the write request are rising-edge DFF outputs, so the
  drivers stay on between writes.
- The write window holds a write's busy term. The slot-arm latch is removed.
- Fixed:
  - `score_waveform` raised `TypeError` on every write deck since V2.1.6.
  - V2.1.9 misstated what `restore_budget` measures.
  - V2.1.9 write→write cycles missed the restore check at 32x16 and 32x32.
- `v2.1.10-timing-9` sets the 10T 512-row class to 4300 ps.
- Evidence: 340 of 340 cases, 753,452 checks, 262 per-device samples.
  148 tracked tests.
- Cleanup: unused factories and helpers removed; all 1,331 netlists are
  identical before and after. Open items are listed in the
  [documentation index](README.md#open-items).

## V2.1.9 — 2026-09-18 — write drivers held until the wordline is off

[Record](design/WRITE_HOLD_V2_1_9.md).

- `wordline_busy` holds `w_en` until the wordline is observed off. Drivers had
  released with up to 0.90 V still on the wordline at 512x4 SS.
- `VWEN_ACCESS_ERROR` rejects a release while the wordline is open.
- `v2.1.9-timing-8` re-derives the read clocks under mismatch.
- Evidence: 313 of 313 cases, 638,438 checks.

## V2.1.8 — 2026-09-17 — no enable overlaps

[Record](design/ENABLE_OVERLAP_V2_1_8.md).

- The read wordline is released at the sense trigger.
- The precharge waits for the previous access's enables (`enables_off`).
- `TRESTORE` falls 30-40 %. Evidence: 91 of 91 cases, 251,553 checks.

## V2.1.7 — 2026-09-17 — select gate; TIME_CONTROL rename

[Record](design/SELECT_GATE_V2_1_7.md).

- `cs_pre = cs & cs_delayed` gates the precharge and write enable. Four
  boundary fixes.
- `TIME` is renamed `TIME_CONTROL` ([map](design/TIME_CONTROL_PATH.md)).
  1,351 netlists are identical. Evidence: 74 of 74 cases, 236,570 checks.

## V2.1.6 — 2026-09-16 — write drivers in the precharge slot

[Record](design/WRITE_SLOT_V2_1_6.md).

- The bitlines reach their rails before the write wordline rises.
- Write decks measure `TWSLOT`, and `select_every` probes idle→write.
- `time_generate.py` is rebuilt; 1,331 netlists are identical. Evidence:
  58 of 58 cases, 213,020 checks.

## V2.1.5 — 2026-09-16 — equivalent model as an input; 10T read-path resize

[10T record](design/TIMING_10T_BUDGET_V2_1_5.md), [equivalent-model record](design/EQUIVALENT_MODEL_V2_1_5.md).

- Equivalent modes 0-4 (`equivalent:` in `global.yaml`, `compare.py`): delays
  are within 0.5 % of mode 0.
- The 10T pull-down widens to 287 nm, so the 10T 512-row class drops from 14
  to 10 ns.
- Evidence: 45 of 45 attempts, 173,193 checks.

## V2.1.4 — 2026-09-15 — write-request hold latch and 6T ladders

[Record](design/TIMING_6T_BUDGET_V2_1_4.md). `Xwe_hold` holds the write request
while a wordline is on. New shared 6T and 6T-mux ladders. Evidence: 92 of 92
attempts, 305,371 checks.

## V2.1.3 — 2026-09-14 — separate 10T timing budget

[Record](design/TIMING_10T_BUDGET_V2_1_3.md). `timing_lookup.json` gains cell-type
`variants`; 10T runs about 200 ps above the shared ladder. 56 of 58 attempts.

## V2.1.2 — 2026-09-13 — run traceability and write-failure inventory audit

- `--vdd` and `--temperature` enter the run identity, and the Xyce
  installation is recorded.
- The supplied CSVs move to `docs/data/` ([audit](data/README.md)).
- The first-500 write-failure inventory showed no electrical write failure.
- [Phase 4/5 follow-up](design/TIMING_FOLLOWUP_V2_1_2.md); see
  [open items](README.md#open-items).

## V2.1.1 — 2026-09-13 — distributed-only wiring and access exclusion

[Record](design/DISTRIBUTED_ONLY_V2_1_1.md). Star wiring is removed. WL, write and
sense assertion waits for the settled far-end precharge. 11 cases, 33,096 checks.

## V2.1.0 — 2026-09-12 — fixed timing classes

[Record](design/TIMING_LOOKUP_V2_1_0.md). The clock follows a row/column lookup
table, frozen with the baseline. Use `timing.mode: fixed` or `--period` for
diagnostics.

## V2.0.11 — 2026-09-12 — distributed control lines

[Record](design/WRITE_VALIDATION_V211.md). `PRE`, `w_en`, `w_en_bar`, `s_en`,
`sa_iso` and `wl_en` are tapped RC lines. 58 of 65 cases pass.

## V2.0.10 — 2026-09-12 — distributed timing and simulation correctness

[Record](design/DISTRIBUTED_RC_V210_REVIEW.md). The precharge guard gains a
replica settling delay. Failed runtime DC solves retry with line search.

## V2.0.9 — 2026-09-10 — fixed driver size classes

`sizing.mode: lookup` with `sizing_lookup.json` integer classes; the legacy
`fixed` mode is removed. 1,989 of 2,025 waveform samples pass.

## V2.0.8 — 2026-09-09 — audit of the distributed interconnect

[Record](design/DISTRIBUTED_RC_VALIDATION.md). The `cell_pin_rc` default is fixed.
Adds `interconnect.load_interconnect()` and `INTERCONNECT_CONFIG`.

## V2.0.7 — 2026-09-09 — RC corrections and distributed interconnect

[Guide](design/DISTRIBUTED_RC_MODEL.md). RC reaches every factory. Distributed
WL/BL ladders get matched replica paths.

## V2.0.6 — 2026-09-08 — compiler integration and documentation

Per-device mismatch moves into `sram_compiler/per_device_mc/`, and the `utils/`
package is created.

## V2.0.5 — 2026-09-08 — full local mismatch

Per-device `vth0` / `u0` / `voff` mismatch is the default.

## V2.0.4 — 2026-09-07 — baseline driver sizing and measured timing

Proposal: `git show b4987df:docs/DRIVER_SIZING_PROPOSAL.md`. Baseline driver sizing is immutable, with
a matched replica and tapered control buffers.

## V2.0.3 — 2026-09-06 — timing configuration proposal

`git show b4987df:docs/TIMING_AUTOCONFIG.md` gives worst-case PVT factors
(2.1-2.3x at SS 125 C 0.9 V) and the phase model.

## V2.0.2 — 2026-09-05 — periphery fan-out, address hold, precharge control

Adds an address hold latch, tapered buffers, sense isolation during writes, and
full-phase precharge.

## V2.0.1 — 2026-09-05 — transient / Monte Carlo circuit review

The write pulse spans the wordline phase, and nominal runs are deterministic
(`git show b4987df:sram_compiler/CIRCUIT_REVIEW.md`).
