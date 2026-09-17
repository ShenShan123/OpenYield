# Compiler regression tests — V2.1.8

V2.1.0 adds timing lookup boundaries, extrapolation, invalid tables, baseline reuse, numeric/sweep clocks, access/retention rejection, preserved CLI attempts and timestep retries. Yield contract tests cover all 15 four-value call sites and nonfinite failure indicators.

V2.1.1 adds distributed defaults and explicit star rejection, consumer-tap
connectivity, peripheral/decoder/clock/select fan-out, complete retention and
finite CLI metric rejection. Software checks do not establish waveform
correctness; see the [current release report](../docs/design/DISTRIBUTED_ONLY_V2_1_1.md).

V2.1.2 adds bounded-step retry cases for explicit `.TRAN` fields, CLI
VDD/temperature overrides in the deck and run identity, and the recorded Xyce
installation on a failed run. The post-release cleanup adds the interrupted
CLI run, whose summary and solver identity must survive a `KeyboardInterrupt`,
and the failed waveform plot, which records `waveform_error` without
rejecting passing measures.

V2.1.3 adds the timing-table variant cases: the 10T budget above the shared
ladder at every anchor and beyond it, with and without a mux, the shared
budget for 6T under PVT and RC changes, `ArrayTiming.budget`, a mux-restricted
variant leaving the other mux setting on the shared class, and rejection of
variants with an unsupported cell, non-boolean mux, duplicate cell, shifted
anchors, a missing class or a budget below the shared one.

V2.1.4 adds the 6T ladders at every class bound with and without a mux (a
mux never gets a shorter clock), the `SRAM_6T_CELL/mux` budget record, and
the TIME write-request hold latch: every access gate, buffered or not, sees
the write request only through the latch enabled by `wl_en_bar`; and the
transient stop, whose printed `.TRAN` value must lie on the output interval
and never end before 8.7 T (2 T for single operations) for every lookup
clock and a fixed diagnostic clock.

The access-start regressions also cover far-PRE guard connectivity, physical
RC settling frozen across candidates/PVT, stale-baseline rejection, and
`VPRE_ACCESS_ERROR` rejection despite correct retained data. Waveform-parser
tests preserve duplicate-probe values/sample boundaries and reject conflicting
duplicates.

These reusable tests are tracked outside the circuit generator. They cover
driver sizing and frozen loads, MOS connectivity and widths, per-device
mismatch, numeric and swept geometry, sampling, optimizer integration, and
qualification-table validation. They use full transistor arrays and require
neither Xyce nor the ignored local development scripts.

V2.0.10 also checks sense-control RC accounting, the distributed precharge
guard, access-window measurements, write-feedback initialization, preserved
sample indices, and the shared DC retry/seed/evidence contract. V2.0.11 adds
the retry's stale-output contract, the rejected sample's retained provenance,
the precharge windows' clamp to the analysis stop, and the parser's
missing-value and all-missing-frame behaviour. Actual waveform validation is
recorded in the [V2.0.10 review](../docs/design/DISTRIBUTED_RC_V210_REVIEW.md)
and the [V2.0.11 write screen](../docs/design/WRITE_VALIDATION_V211.md).

Keep these tests in Git: they document expected behavior and let every checkout
verify compiler and utility changes. `test_utils.py` also checks measurement
parsing, transient/DC sample splitting, plot files, and existing import paths.
`test_main_entrance.py` checks that `main_sram.py` defaults to seeded per-device
mismatch over the full array and applies its settings in memory without
changing the tracked YAML files.

Run from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

The separate offline optimizer checks are:

```bash
python3 -m unittest discover -s size_optimization/openyield_v2/tests -v
```

Tests of local experiments and qualification runners live under ignored
`dev/tests/`. See the [development guide](../docs/DEVELOPMENT.md) for those
optional tools. Keep runtime testbenches in `sram_compiler/testbenches/`;
they construct the simulator circuits used by the project.

V2.0.7 adds `test_rc_configuration.py` and `test_interconnect.py` for RC-value
propagation, extraction context/cache identity, distributed pi-section R/C
conservation, cell/equivalent taps, matched replica paths and local MC probes.
V2.0.8 adds the `cell_pin_rc` default for directly constructed configurations,
the project-root YAML loader, and the `main_sram.py` interconnect setting.
V2.0.9 adds the `LookupTests` in `test_driver_sizing.py`: fixed size classes,
their rule lower bound, independence from cell/mux/RC/wires, ladder
interpolation and extrapolation, invalid or alternative tables, and the removed
legacy mode; the RC and driver-path tests now exercise `lookup` and `rules_only`.

V2.1.6 updates the TIME netlist expectations to the write slot (`AND3_WEN`,
`write_slot`, `write_window`, `pre_gate`, `cs_pre`, the guard's exported
`pre_off_ready`, the enabled replica write driver in `wen_load`), the
plan-aware runtime measures (`VWL_WEN_*` before a write, `TWSLOT` in write
decks) and the renamed helper classes (`ReplicaDelayChain`, `DataRegister`,
`ReplicaColumn`); `dev/tests/test_v210_waveform_checks.py` models the slot in
its synthetic traces and rejects a bitline that is not at its rail when the
wordline starts.

V2.1.8 adds
`test_read_wordline_ends_at_the_sense_enable_and_the_clock_high_enables_wait_for_the_enables`
(the wordline request is gated by `s_en_bar`, the precharge gate by
`enables_off`, the write slot by `s_en_bar` and never by `w_en`, the write
enable never by `cs_pre` directly, the observers read the buffered enables)
and updates the wiring and load tests for the AND3 precharge gate, the
selected slot, the AND2 write enable and the new observer loads
(`access_load = 1.25 * wl_en_scale + 2.5`, `sen_load` and `wen_load`).
`dev/tests/test_v210_waveform_checks.py` models the sense-timed read
wordline and a corruption for every new ordering check.

V2.1.7 renames the control block (`TIME_CONTROL`, `XTIME_CONTROL`,
`TimeControlFactory`, inner subcircuits such as `PRECHARGE_WRITE_AND`,
`DATA_REGISTER`, `SELECT_DELAY`) and adds one test per review finding, each
checked to fail with its fix reverted:
`test_select_delay_holds_back_only_the_rising_edge_of_both_clock_high_enables`
(`cs_pre = cs & cs_delayed` feeds only the precharge and write-enable gates),
`test_write_enable_follows_the_wordline_enable_without_the_replica_guard`
(it replaces the V2.1.6 expectation of a `wl_en_bar` write window, which was
the bug), `test_passed_transistor_models_reach_every_control_device`,
`test_write_slot_checks_do_not_require_replica_settling_stages` and
`test_idle_probe_registers_new_write_data_at_the_selected_edge`.
`dev/tests/test_v210_waveform_checks.py` adds idle cycles to its synthetic
traces and rejects precharge or enable activity there, and new write data
that reaches the driver latch after the enable.
