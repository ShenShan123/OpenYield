# Design records

Records are grouped by subject. Version labels and validation limits remain
those of the original record. Historical JSON and waveform images stay beside
their reports; diagnostic datasets remain in [`../data/`](../data/README.md).

For execution instructions, use the [compiler tutorial](../../sram_compiler/README.md)
or [layout guide](../../sram_compiler/layout/README.md). The [documentation index](../README.md)
tracks release status and open work.

## Layout

Component, bank and TIME_CONTROL physical design and verification.

- [Compact point-symmetric bitcell and pitch-matched bank — V2.2.8](layout/COMPACT_CELL_V2_2_8.md)
- [V2.2.8 component layout development](layout/COMPONENT_LAYOUT_V2_2_8.md)
- [Single-bank layout scaling, driver candidates and centre floorplan — V2.2.8](layout/LAYOUT_SCALING_V2_2_8.md)
- [OpenRAM-pitch bank layout, V2.2.8](layout/OPENRAM_PITCH_V2_2_8.md)
- [OpenRAM 32×32 FreePDK45 reference run](layout/OPENRAM_REFERENCE_V2_2_8.md)

Superseded data-path, folded-bank, row-pitch and controller reports were purged
on 2026-10-24. Their [physical evidence](../data/README.md#historical-layout-evidence)
retains its original scope; the reports are available at `git show 8dc77df:docs/design/layout/<name>`.

## Compiler

Compiler organization, configuration reviews and experiment entry.

- [Compiler entrance and local variation — V2.2.5](compiler/COMPILER_ENTRY_V2_2_5.md)
- [V2.2.5 configuration review and V2.2.6 fixes](compiler/CONFIGURATION_REVIEW_V2_2_6.md)
- [V2.2.7 compiler organization and unified experiments](compiler/ORGANIZATION_V2_2_7.md)

## Timing and control

Clock budgets, control-path changes and version-scoped waveform screens.

- [V2.1.8: no enable overlaps: the read wordline ends at the sense trigger, the clock-high enables wait for the previous access](timing/ENABLE_OVERLAP_V2_1_8.md)
- [V2.2.1: capture, access, and recovery in separate phases](timing/PHASED_CONTROL_V2_2_1.md)
- [V2.2.2: production review of the phased controller](timing/PHASED_CONTROL_V2_2_2.md)
- [V2.2.3: supported envelope, joint-dimension clocks, one access deadline](timing/PHASED_CONTROL_V2_2_3.md)
- [V2.2.4: replica load cells that do not leak, and the V2.2.3 review fixes](timing/PHASED_CONTROL_V2_2_4.md)
- [V2.1.7: the select gate of the clock-high enables, and the TIME_CONTROL rename](timing/SELECT_GATE_V2_1_7.md)
- [TIME_CONTROL control path: signals, stages and their history](timing/TIME_CONTROL_PATH.md)
- [V2.1.3 timing budget for 10T cells: evidence run and ten-seed pilot](timing/TIMING_10T_BUDGET_V2_1_3.md)
- [V2.1.5 10T round: read-path resize, 512-row class and the re-run boundary matrix](timing/TIMING_10T_BUDGET_V2_1_5.md)
- [V2.1.4 6T timing: write-request hold latch, 6T ladders and evidence run](timing/TIMING_6T_BUDGET_V2_1_4.md)
- [V2.1.2 timing follow-up: Phase 4 boundaries and Phase 5 mismatch pilot](timing/TIMING_FOLLOWUP_V2_1_2.md)
- [V2.1.0: timing lookup and V2.0.11 review](timing/TIMING_LOOKUP_V2_1_0.md)
- [V2.1.9: the write drivers stay on until the wordline is off; read clocks re-derived under mismatch](timing/WRITE_HOLD_V2_1_9.md)
- [V2.1.10: the write data is held by the registered write; the drivers stay on between writes](timing/WRITE_LATCH_V2_1_10.md)
- [V2.1.6: the write drivers take the precharge slot](timing/WRITE_SLOT_V2_1_6.md)
- [V2.0.11: audit of V2.0.10, distributed control lines, star-topology screen](timing/WRITE_VALIDATION_V211.md)

## Interconnect

Distributed RC topology, models and validation.

- [V2.1.1 distributed-only signal wiring](interconnect/DISTRIBUTED_ONLY_V2_1_1.md)
- [Distributed signal-wire RC — V2.1.2](interconnect/DISTRIBUTED_RC_MODEL.md)
- [V2.0.10: SRAM distributed-RC review](interconnect/DISTRIBUTED_RC_V210_REVIEW.md)
- [Distributed RC validation — V2.0.7 record and V2.0.8 audit](interconnect/DISTRIBUTED_RC_VALIDATION.md)

## Equivalent modeling

Approximate array models and their accuracy boundaries.

- [V2.1.5 equivalent array model: merge into the compiler and measured accuracy](equivalent/EQUIVALENT_MODEL_V2_1_5.md)
