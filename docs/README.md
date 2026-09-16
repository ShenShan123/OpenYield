# OpenYield V2.1.6 documentation

V2.1.1 removes star wiring, distributes the remaining signal fan-out, and
extends write-capture, retention and CLI validation. See the
[current release report](design/DISTRIBUTED_ONLY_V2_1_1.md) and
[evaluation schedule](plans/V2_1_1_TIMING_FOLLOWUP.md). The unchanged timing
table was introduced in the [V2.1.0 review](design/TIMING_LOOKUP_V2_1_0.md).

V2.1.2 keeps that circuit and adds run traceability (CLI VDD/temperature,
recorded Xyce installation, bounded-step retry for explicit `.TRAN` fields).
Its audit of the supplied [first-500 write-failure inventory](issue_reports/write_failure_cases_first_500.md)
is appended to that report; see the [changelog](CHANGELOG.md). Its Phase 4/5
[timing follow-up](design/TIMING_FOLLOWUP_V2_1_2.md) screens class boundaries
and a per-device pilot and records the 10T column-mux class limit.

V2.1.3 adopts a separate timing budget for 10T cells (with or without a
column mux) after its own [evidence run](design/TIMING_10T_BUDGET_V2_1_3.md):
row classes 2000/2200/2400/3200/5600 ps and column classes 200 ps above the
shared ladder, with at least 250 ps of read-output margin at every class
bound; the ten-seed per-device pilot is recorded there too. 6T decks and the
shared classes are unchanged. What it left open, with evidence and next
steps, is in the [V2.1.3 open items](plans/V2_1_3_OPEN_ITEMS.md).

V2.1.4 closes the 6T open items there with its own
[evidence run](design/TIMING_6T_BUDGET_V2_1_4.md): a write-request hold latch
in TIME removes the FF −40 °C write-enable spike, the shared 6T row classes
become 1800/1900/2100/2500/3600 ps and 6T with a column mux gets a variant
(rows 1900/2000/2200/2700/3600 ps), both keeping 250 ps of read-output margin
at every class bound.

V2.1.5 closes the remaining 10T items with its own
[10T round](design/TIMING_10T_BUDGET_V2_1_5.md) and merges the equivalent array
model into the compiler as a simulation input
([accuracy record](design/EQUIVALENT_MODEL_V2_1_5.md)). The 10T pull-down is
widened to 287 nm because the read-disturb bump that set the 512-row class is
the read current through two stacked pull-downs, which drops that class from
14 ns to 10 ns and gains read-output margin at every height; the whole 10T
boundary matrix and its mismatch seeds were rerun on the current TIME block.
Only the carried Phase 6 scope is still open.

V2.1.6 makes the write drivers take the precharge slot after an audit of the
TIME block ([write-slot record](design/WRITE_SLOT_V2_1_6.md)): in a write
cycle the precharge is inhibited and the write enable turns the drivers on in
the clock-high phase, so BL/BLB are at their write rails before the wordline
rises. The checks, the write decks (`TWSLOT`, write 1 then 0) and an
idle → write probe (`select_every`) follow; every timing class is kept and
re-evidenced. The [TIME control-path reference](design/TIME_CONTROL_PATH.md)
holds the signal table, the history of every stage and the naming contract
of the refactored `time_generate.py`.

Working plans (`plans/`), design and validation records (`design/`) and the
release history live here. The [qualification scope](plans/V2_1_2_QUALIFICATION_SCOPE.md)
defines the extracted-metal inputs, the PVT/sample matrix and the separate
half-select and yield briefs (plan Phase 6). Superseded pre-release snapshots were removed on
2026-09-13; the [changelog](CHANGELOG.md) names them and their git-history
paths. Usage guides live beside the code
they describe. Commands and code paths in the documents are relative to the
repository root unless stated otherwise.

V2.0.11 audits the V2.0.10 release and adds the first star-topology waveform
screen; V2.0.6 integrates the default per-device local mismatch package into the
compiler and establishes this documentation layout. Earlier release entries, design
snapshots, and qualification evidence retain their original version labels.

| Document | Purpose |
|---|---|
| [Driver sizing proposal](DRIVER_SIZING_PROPOSAL.md) | Full working proposal, implementation progress, and current qualification status |
| [Automatic timing proposal](TIMING_AUTOCONFIG.md) | Original timing design and measured basis; implemented timing lives in `sram_compiler/sizing/timing.py` |
| [Changelog](CHANGELOG.md) | Release history and validation results |
| [Development guide](DEVELOPMENT.md) | Tracked regression tests and ignored local experiments/qualification runners |
| [Original V2.0.4 design](design/DRIVER_SIZING_PROPOSAL_V2.0.4.md) | Preserved design snapshot, including its historical source paths |

The supplied evidence CSVs are kept unchanged in `docs/data/`:
[driver sizing data](data/DRIVER_SIZING_data.csv) and
[timing data](data/TIMING_AUTOCONFIG_data.csv). See the [data audit](data/README.md).

| Directory guide | Contents |
|---|---|
| [SRAM compiler](../sram_compiler/README.md) | Configuration, circuit generation, simulations, and waveform outputs |
| [Shared utilities](../utils/README.md) | Measurement and waveform parsing, plots, SPICE models, and area estimates |
| [Compiler regression tests](../tests/README.md) | Simulator-free checks that run without local development scripts |
| [Per-device mismatch](../sram_compiler/per_device_mc/README.md) | Default local mismatch, CLI, and in-memory configuration |
| [Driver sizing](../sram_compiler/sizing/README.md) | Fixed driver size classes (V2.0.9 lookup table), timing, and qualification workflows |
| [Circuit review](../sram_compiler/CIRCUIT_REVIEW.md) | Circuit findings and verification evidence |
| [Equivalent modeling](../sram_compiler/equivalent_modeling/README.md) | Equivalent circuit modes, usage, and accuracy boundaries |
| [Sizing optimization](../size_optimization/README.md) | Circuit-backed algorithms and offline optimizers |
| [Yield estimation](../yield_estimation/README.md) | Monte Carlo and importance-sampling algorithms |

- [Distributed RC model/configuration guide](design/DISTRIBUTED_RC_MODEL.md) and
  [validation record](design/DISTRIBUTED_RC_VALIDATION.md) (V2.0.7 implementation, V2.0.8 audit).
- [V2.0.10 distributed-RC review](design/DISTRIBUTED_RC_V210_REVIEW.md): the
  precharge settling guard, TIME load correction and the wire/timing limits.
- [V2.0.11 audit and write screen](design/WRITE_VALIDATION_V211.md): failure
  handling and measurement-window fixes, the star-topology write and read
  waveform screen, and the open release-margin and yield-path items.
- [V2.1.1 distributed-only wiring](design/DISTRIBUTED_ONLY_V2_1_1.md): removal
  of star paths, peripheral/decoder/clock/select ladders, and fresh evaluation.
- [V2.1.2 timing follow-up](design/TIMING_FOLLOWUP_V2_1_2.md): Phase 4 class
  boundaries, wire refinement and stress, Phase 5 per-device pilot, and the
  10T column-mux finding at the frozen 4 ns class.
- [V2.1.3 10T timing budget](design/TIMING_10T_BUDGET_V2_1_3.md): why the
  shared class and two candidate 10T ladders were rejected, the adopted 10T
  ladder with its nominal boundary run, and the mismatch seeds at the class
  bounds plus the ten-seed pilot.
- [V2.1.4 6T timing budget](design/TIMING_6T_BUDGET_V2_1_4.md): the TIME
  write-enable race and its hold latch, the probe reads that rejected the
  shared 6T classes, the adopted shared and 6T-mux ladders with their
  boundary run, mismatch seeds, pilot and write-waveform gate.
- [V2.1.5 10T round](design/TIMING_10T_BUDGET_V2_1_5.md): why the read-disturb
  bump is a pull-down sizing problem, the three cell candidates and why the
  1.8x one was rejected, the new 512-row class, and the rerun boundary matrix
  and seeds on the V2.1.4 TIME block.
- [V2.1.5 equivalent model](design/EQUIVALENT_MODEL_V2_1_5.md): what the merge
  into `sram_compiler/equivalent_modeling/` changed, and the measured delay,
  power and runtime error of each mode against the full transistor array.
