# Compiler regression tests — V2.2.2

V2.2.2 adds `test_generated_decks_start_from_the_precharged_clock_low_state`
(every deck's `.IC` starts bitlines, RBL/RBLB and the sense nodes at VDD) and
`test_access_and_recovery_margins_are_reported_and_cannot_be_negative`
(read output, write wordline dwell, restore and precharge-on margins; a late
output fails both the data and the margin check).

V2.2.1 replaces the old hold-latch/slot topology expectations with the phased
controller contract. Regressions cover new pin routing in numeric and sweep
decks, release guards, boundary/role/isolation rejection, recovery after writes,
and stale control-architecture snapshots. The versioned V2.1.x descriptions
below are historical. Actual SPICE transition checks are an explicit opt-in
under [`spice/`](spice/README.md); unittest discovery does not launch them, and
that README's coverage table says which sizes, corners and data patterns the
screen does and does not reach.


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

V2.1.10 rewrites `test_write_drivers_stay_on_until_the_wordline_is_observed_off`
for the write-data latch hold: across write -> write the drivers never turn
off and the latches pass the new data only after the previous wordline is
observed off, also when the slot arrives three gate delays after the
wordline is observed off (the slow NAND3 path at SS, which collapsed the window
under mismatch before the busy hold); across write -> read and write -> idle
the drivers release and the latches keep the written data until they are off,
then open; across read -> write the enable stays off until the slot (a held
read wordline pulsed it in the second evidence pass), and at a write's own
wordline the held busy term takes over from the slot. The wiring tests follow
`din_hold` (last port, `DIN_HOLD_NAND` x 3 reading `we`, `cs`, `w_en` and
`wordline_idle`), the `din_en` column line, the `SELECTED_SLOT_NAND` and the
`WRITE_WINDOW_NAND` of the held write-busy term and the slot; `test_slot_arm_latch_is_seeded_in_its_start_state` is removed with the
latch, and the clock tests take the 10T 512-row class at 10.75 ns.

V2.1.9 adds `test_write_drivers_stay_on_until_the_wordline_is_observed_off`
(a unit-delay evaluation of the block's own gates across write -> write and
write -> read boundaries: `w_en` and the held request stay on while the
wordline is busy, and between two writes the drivers are released for at
least the four settling stages plus the slot path before the slot reopens)
and `test_drivers_released_under_an_open_wordline_reject_even_correct_written_data`
(`VWEN_ACCESS_ERROR_<cycle>` in every write cycle and only there, and
`access_validity` rejects a sample above 0.1 VDD or without the measure),
and `test_slot_arm_latch_is_seeded_in_its_start_state` (every deck seeds the
latch's t = 0 state; unseeded, mux read decks lost their DC operating point),
and updates the wiring tests for the busy wordline, the `wordline_idle`
request latch and the slot-arm latch; each new test fails with its part of
the fix reverted. The clock tests follow the re-derived read classes
(`test_6t_budgets_keep_read_output_margin_at_every_class_bound_under_mismatch`,
formerly `..._keep_250ps_...`, and the 10T and boundary tests). `dev/tests/test_v210_waveform_checks.py` rejects a driver
release under an open wordline and a dip of the enable inside the access.

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
