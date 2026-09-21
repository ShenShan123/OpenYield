# OpenYield V2.2.2 documentation

Design and validation records (`design/`), the supplied evidence (`data/`,
`qualification/`, `issue_reports/`) and the release history live here; usage
guides live beside the code they describe. Paths are relative to the
repository root unless stated otherwise. Every release entry of the
[changelog](CHANGELOG.md) names its record; earlier entries, snapshots and
evidence keep their original version labels.

## Open items

The carried Phase 6 scope. The working plans that defined it
(`V2_1_1_TIMING_FOLLOWUP.md`, `V2_1_2_QUALIFICATION_SCOPE.md`,
`V2_1_3_OPEN_ITEMS.md`) were removed in the V2.1.10 cleanup and remain at
`git show 945815a:docs/plans/<name>`.

- **Extracted metal.** The `interconnect` mapping (`global.yaml`,
  `INTERCONNECT_CONFIG` in `main_sram.py`, `--interconnect-config`) is the whole
  wire model: `wl.layer` / `bl.layer`, `pitch_m`, `width_m`,
  `sheet_resistance_ohm`, `capacitance_f_per_m` (coupling folded to ground),
  `sections_per_half_pitch`, `cell_pin_rc`. The default 1 ohm / 0.1 fF per pitch
  is illustrative and must not appear in a qualification record. Layout
  distances must replace the fixed periphery (three pitches beyond each array
  port), the control-line and decoder-trunk pitches and the local `w_rc` /
  `pi_res` / `pi_cap` stubs. Not modelled: signal coupling between neighbouring
  lines, metal-resistance temperature dependence, via resistance outside
  `cell_pin_rc`, supply IR drop. A changed geometry re-runs the functional gates
  at the frozen clocks and driver classes; the clock table changes only through
  a new evidenced version.
- **Half-select writes.** Every column of the selected row is written today, so
  no cell is half-selected in a write. Needed: a per-column write enable from
  the column select and a write mask; checks that half-selected cells retain
  their data and every column still restores and releases; cases at mux ratios
  2 and 4, SS / SF 0.9 V / 125 C and FF 1.1 V / -40 C with address hazards, then
  per-device seeds. Read-modify-write is out of scope.
- **Yield estimators.** `yield_estimation/model_lib/{MC,MNIS,AIS,ACS,HSCS}.py`
  are not usable as shipped: machine-local paths and an import-time deletion,
  dependencies absent from `environment.yml` (`torch`, `gpytorch`, `mpmath`,
  `prettytable`), sampling through the `custom` variation mode and a delay
  threshold instead of the release checks. Needed: configurable paths, a sample
  mapped to a per-device seed (or an explicit vector in the per-device layout)
  with the frozen timing and drivers, failure defined by the release checks
  (numerical non-completion counted as incomplete), and MC against the
  importance samplers on a small array at inflated sigma before any full-sigma
  estimate.

## Documents

| Document | Purpose |
|---|---|
| [Changelog](CHANGELOG.md) | Release history, one entry per release with its record |
| [Development guide](DEVELOPMENT.md) | Tracked regression tests and the ignored local tools |
| [Driver sizing proposal](DRIVER_SIZING_PROPOSAL.md) | Working proposal and qualification status of the driver classes |
| [Automatic timing proposal](TIMING_AUTOCONFIG.md) | Timing design, measured basis and the per-signal phase table |
| [V2.2.2 review and screen](design/PHASED_CONTROL_V2_2_2.md) | Current review findings, margins, and validation |
| [V2.2.1 control path](design/PHASED_CONTROL_V2_2_1.md) | Phase ordering and timing contract |
| [V2.1.10 control path](design/TIME_CONTROL_PATH.md) | Preserved previous architecture and naming contract |
| [Original V2.0.4 design](design/DRIVER_SIZING_PROPOSAL_V2.0.4.md) | Preserved design snapshot |

## Release records

| Release | Record |
|---|---|
| V2.2.2 | [production review, precharged startup, margins](design/PHASED_CONTROL_V2_2_2.md) |
| V2.2.1 | [capture/access/recovery phase control](design/PHASED_CONTROL_V2_2_1.md) |
| V2.1.10 | [write-data latch on the registered write](design/WRITE_LATCH_V2_1_10.md) |
| V2.1.9 | [write hold and read clocks under mismatch](design/WRITE_HOLD_V2_1_9.md) |
| V2.1.8 | [enable overlaps](design/ENABLE_OVERLAP_V2_1_8.md) |
| V2.1.7 | [select gate](design/SELECT_GATE_V2_1_7.md) |
| V2.1.6 | [write slot](design/WRITE_SLOT_V2_1_6.md) |
| V2.1.5 | [10T round](design/TIMING_10T_BUDGET_V2_1_5.md), [equivalent model](design/EQUIVALENT_MODEL_V2_1_5.md) |
| V2.1.4 | [6T timing budget](design/TIMING_6T_BUDGET_V2_1_4.md) |
| V2.1.3 | [10T timing budget](design/TIMING_10T_BUDGET_V2_1_3.md) |
| V2.1.2 | [timing follow-up](design/TIMING_FOLLOWUP_V2_1_2.md), [first-500 write-failure inventory](issue_reports/write_failure_cases_first_500.md) |
| V2.1.1 | [distributed-only wiring](design/DISTRIBUTED_ONLY_V2_1_1.md) |
| V2.1.0 | [timing lookup](design/TIMING_LOOKUP_V2_1_0.md) |
| V2.0.11 | [audit and write screen](design/WRITE_VALIDATION_V211.md) |
| V2.0.10 | [distributed-RC review](design/DISTRIBUTED_RC_V210_REVIEW.md) |
| V2.0.7 / V2.0.8 | [distributed RC model](design/DISTRIBUTED_RC_MODEL.md), [validation](design/DISTRIBUTED_RC_VALIDATION.md) |

Raw simulation outputs of releases before V2.1.9 were purged from the ignored
`outputs/` directory in the V2.1.10 cleanup; the records above, `data/` and
`qualification/` keep their results.

The supplied evidence CSVs are kept unchanged in `data/`:
[driver sizing data](data/DRIVER_SIZING_data.csv) and
[timing data](data/TIMING_AUTOCONFIG_data.csv) ([audit](data/README.md)).

| Directory guide | Contents |
|---|---|
| [SRAM compiler](../sram_compiler/README.md) | Configuration, circuit generation, simulations, and waveform outputs |
| [Shared utilities](../utils/README.md) | Measurement and waveform parsing, plots, SPICE models, and area estimates |
| [Compiler regression tests](../tests/README.md) | Simulator-free checks that run without local development scripts |
| [Per-device mismatch](../sram_compiler/per_device_mc/README.md) | Default local mismatch, CLI, and in-memory configuration |
| [Driver sizing](../sram_compiler/sizing/README.md) | Driver size classes, clock classes, and qualification workflows |
| [Circuit review](../sram_compiler/CIRCUIT_REVIEW.md) | Circuit findings and verification evidence |
| [Equivalent modeling](../sram_compiler/equivalent_modeling/README.md) | Equivalent circuit modes, usage, and accuracy boundaries |
| [Sizing optimization](../size_optimization/README.md) | Circuit-backed algorithms and offline optimizers |
| [Yield estimation](../yield_estimation/README.md) | Monte Carlo and importance-sampling algorithms |
