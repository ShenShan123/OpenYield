# OpenYield V2.1.2 documentation

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
| [Equivalent modeling](../equivalent_modeling/README.md) | Equivalent circuit modes, usage, and accuracy boundaries |
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
