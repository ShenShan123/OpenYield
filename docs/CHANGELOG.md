# Changelog

The full evidence tables of V2.0.1 and V2.0.2 (294 size-sweep decks with
delay and power per configuration, 108 corner runs, the address-change and
clock-period sweeps, seeded Monte Carlo) were removed from this file in V2.0.3.
They are in the git history (`git show c3f6f44:CHANGELOG.md`) and in
`sram_compiler/CIRCUIT_REVIEW.md` Parts II and III; the condensed numbers below are copied
from them unchanged.

## V2.1.10 — 2026-09-18 — write data held by the registered write; drivers stay on between writes

V2.1.10 reviews V2.1.9 ([record](design/WRITE_LATCH_V2_1_10.md)). The write
data and the write request are DFF outputs of the rising clock edge, so
between two writes only the data changes; the drivers no longer turn off and
on again.

- Found: the scorer's `restore_budget` compares `TWSLOT` with 0.8 x the
  *measured* read phase (its callers derive the timing from measurements),
  not with a class budget as V2.1.9 stated, so no clock change could meet it;
  `score_waveform` raised `TypeError` on every write deck since V2.1.6 (the
  drive-rail loop rebound the waveform array); V2.1.9's write -> write slot
  (drivers off, four settling stages, slot-arm latch, drivers on: 1.6 to
  1.8 ns at SS 0.9 V / 125 C) left the driven bitline outside 0.02 VDD of its
  rail at the runtime restore check at 32x16 and 32x32, with and without a mux
  (`VRESTORE_ERROR` 76 and 124 mV at 4.5 ns), bounds V2.1.9 never ran as
  write -> write decks; the 10T 512x4 read missed its rule by 7 ps.
- Write-data latches: `din_en = wordline_idle & ((we & cs) | !w_en)`
  (`din_hold`, three `DIN_HOLD_NAND` + `DIN_HOLD_BUFFER`, a new last port of
  write-capable `TIME_CONTROL`; the testbench line `din_en` replaces
  `w_en_bar`). Between two writes the new data passes once the previous
  wordline is observed off, with the drivers on; before a read or idle cycle
  the written data is held until the drivers are off; after a read or idle
  cycle the latches are already open.
- Write window: `held(wordline_busy & we_hold) | selected_slot`
  (`BUSY_WRITE_NAND`, `BUSY_HOLD_DELAY`, `WRITE_WINDOW_NAND`; V2.1.9
  `wordline_busy | selected_slot`). The busy wordline and the next slot follow
  the same observer and the slot's path is longer, so under mismatch at SS / SF
  the window collapsed between two writes (`w_en` down to 0.21 V of 0.9 V at
  8x4 10T SS, 16 Monte-Carlo samples of the first evidence pass); the end of a
  write's busy term is held for four unit stages, so the slot takes over first.
  Holding a read's busy wordline too kept the window open for the next write
  request (`w_en` pulsed to 0.30 V at 16x16 SS, second pass), so a read's busy
  wordline is not in the window.
- Removed: the slot-arm latch (`SLOT_ARM_NOR` x 2), its settling chain
  (`SLOT_ARM_DELAY`), the `slot_armed` input of the selected slot (now
  `SELECTED_SLOT_NAND`) and the latch's `.IC` seeding.
- Clocks: `timing_lookup.json` `v2.1.10-timing-9` keeps the V2.1.9 classes and
  moves the 10T 512-row class to 4300 ps (10.75 ns). A raised-class
  alternative (2100 to 2200 ps up to 64 rows and columns) was stopped in
  favour of the circuit fix.
- Checks: the local checker scores a write followed by a write with
  `release_before_new_data` and `write_enable_held_between_writes`; the
  validator keeps such a write's drive window; the scorer's write entry is the
  far bitline leaving its rail, `restore_budget` scores the drivers' swing and
  `write_slot_budget` the whole slot against the clock-high phase.
- Evidence: 340 of 340 cases, 753,452 checks, 262 per-device samples:
  the V2.1.9 campaign plus 27 class-bound write -> write slots; every slot
  meets the runtime restore check with at least 229 ps after 1.1 x its rail
  time and is 115 to 517 ps shorter than in V2.1.9; five mux reads after a DC
  operating-point retry (one with `GMIN=1e-10`). Per-device seeds are new
  samples, not V2.1.9's.
- Tests: 148 tracked, 64 local; the unit-delay model of the write window
  covers write -> write, write -> read, write -> idle, read -> write and a
  write's own wordline, and fails for the earlier holds.

## V2.1.9 — 2026-09-18 — write drivers stay on until the wordline is off; read clocks re-derived under mismatch

V2.1.9 audits V2.1.8 against one requirement: the write drivers are fully on
for the whole time the wordline is asserted and are never turned off while it
is ([record](design/WRITE_HOLD_V2_1_9.md)), then checks reads and writes at
the worst PVT combinations under mismatch. V2.1.8 broke the requirement at
every write boundary; the sweep found the read clocks of the 128- to
512-row classes too short under mismatch. Driver sizes, cells and the read
path are unchanged.

- Found: `w_en = we_hold & (wl_en | selected_slot)` ended with the wordline
  enable, and at a write -> read boundary with the held request (its latch
  reopened on `wl_en_bar`), while the physical wordline outlives the enable
  by the row driver and the wire. The drivers started to release (0.9 VDD)
  with the local wordline at 0.51 V of 1.1 V (8x4 FF -40 C), 0.27 V (8x4
  SS), about half VDD (256x4) and 0.90 V (512x4 SS, off 235 ps before the
  wordline was at half VDD). The V2.1.8 record had made this tail a metric
  and no check or runtime measure looked at it.
- Busy wordline: `wordline_busy = !(wl_en_bar & wordline_off)`
  (`WORDLINE_BUSY_NAND`) from the enable until the replica wordline is
  observed off with its settling stages; `write_window = wordline_busy |
  selected_slot`, and the write-request latch opens on `wordline_idle`. The
  drivers now start to release 106 ps (8x4 FF) to 483 ps (16x64 mux SS) after
  the local wordline is below 0.1 VDD (76 ps with zero settling stages, TT);
  V2.1.8 did so up to 322 ps before it.
- Slot-arm latch: the release and the next write slot follow the same
  observer, so `selected_slot = cs_pre & write_slot & slot_armed`
  (`SELECTED_SLOT_AND`, an AND3), `slot_armed` a NOR latch (`SLOT_ARM_NOR`)
  set by `enables_off` through four unit stages (`SLOT_ARM_DELAY`) and reset
  by `wordline_busy`. Between two writes the drivers drop and the write-data
  hold latch takes the new data before the slot reopens: at least 113 ps at
  8x4 FF -40 C under mismatch (58 ps without the stages; 106 ps in V2.1.8),
  421 ps at SS.
- Start state: the testbench seeds the slot-arm latch (`slot_armed`,
  `slot_armed_bar`, `enables_off_settled`) with `.IC` at its t = 0 state; the
  unseeded cross-coupled pair made the DC operating point of mux read decks
  fail (8x128 at four ranks even with Newton line search) or take minutes.
  One per-device 10T sample (16x16 mux read, FS 1.1 V / -40 C, seed 1) fails
  its operating point with every solver on the V2.1.8 sources as well and
  runs with gmin stepping.
- Checks: every write cycle carries `VWEN_ACCESS_ERROR_<cycle>` (the highest
  local or far wordline level while the target driver's enable is below
  0.9 VDD), limited to 0.1 VDD by `access_validity`, so every run rejects
  such a sample; the local checker adds `_write_enable_on_during_wordline`
  per write cycle and column, the scorer `write_enable_covers_wordline`.
  The testbench prints `XTIME_CONTROL:wordline_busy` and `slot_armed`.
- Naming: new `WORDLINE_BUSY_NAND`, `PINV_wordline_idle`, `SLOT_ARM_DELAY`,
  `SLOT_ARM_NOR`, nodes `wordline_busy`, `wordline_idle`, `slot_armed`,
  `slot_armed_bar`, `enables_off_settled`
  ([map](design/TIME_CONTROL_PATH.md), section 8).
- Read clocks: at SS 0.9 V / 125 C per-device mismatch moved the local read
  output by up to 214 ps at 256 rows and 332 ps at 512 rows (up to 8 % of
  the access; the single replica cell sets the sense trigger), and two of
  eight 256x4 6T seeds and one 512x4 6T-mux seed had the data inside the
  checker's 0.02 T guard at the V2.1.8 clocks. New rule: at every class
  bound the nominal read output leads the 1.2 T deadline by 0.02 T plus
  10 % of the access time (and at least 250 ps). `timing_lookup.json`
  `v2.1.9-timing-8`: shared rows 1800/1900/2200/2700/3700 ps (5.5, 6.75,
  9.25 ns at 128, 256, 512 rows), 6T-mux rows 1900/2000/2300/2900/4000 ps
  (5.75, 7.25, 10 ns), 10T rows 2000/2200/2400/3200/4200 ps (10.5 ns at
  512 rows); the smaller classes and the columns are unchanged; a 513-row
  6T array extrapolates to 12.7 ns instead of 13 ns (unqualified).
- Evidence: the V2.1.8 matrix, a per-device Monte-Carlo sweep at SS 0.9 V /
  125 C, FF 1.1 V / -40 C, SF 0.9 V / 125 C and FS 1.1 V / -40 C (6T and
  10T, 8x4 to 512x4) and the class-bound reads, on the release sources
  (313 of 313 cases, 638,438 of 638,438 checks, 218 of them Monte-Carlo
  samples; two four-rank DC operating points retried with Newton line
  search); a first pass on the first V2.1.9 sources found the latch and
  read-clock items above. Every seed has the read data out at least 97 ps
  before the checker's guard; the 10T 512x4 bound keeps 662 ps against the
  rule's 669 ps at the new clock (reported, not raised).
- Open: the write -> write slot now uses up to 73 % of the clock-high phase
  at SS (`TWSLOT` 1136 -> 1625 ps at 8x4, 1191 -> 1727 ps at 16x16 with a
  mux), within the documented period rule but above the qualification
  scorer's `restore_budget` (0.8 x the class budget) at seven write decks
  up to 128 rows; the scorer attributes that budget to
  the precharge / write drivers, which cannot shorten the wordline tail.
  Left for a decision (record, section 8).
- Tests: `test_write_drivers_stay_on_until_the_wordline_is_observed_off`
  (unit-delay evaluation of the block's gates across write -> write and
  write -> read) and
  `test_drivers_released_under_an_open_wordline_reject_even_correct_written_data`,
  each shown to fail with its part of the fix reverted, and
  `test_slot_arm_latch_is_seeded_in_its_start_state`; the clock tests follow
  the new classes; 149 tracked tests pass; the local checker tests reject a
  release under an open wordline and a dip of the enable (`dev/tests`).

## V2.1.8 — 2026-09-17 — no enable overlaps: sense-timed read wordline; precharge and write slot wait for the enables

V2.1.8 reviews the enable pulses of `TIME_CONTROL` for overlaps inside an
access and across every access boundary (read -> read, write -> read, read ->
write, write -> write, idle), for 6T and 10T cells, and removes the ones it
found ([record](design/ENABLE_OVERLAP_V2_1_8.md)). Driver sizes, cells and
the timing classes are unchanged.

- Read wordline released at the sense trigger: `wl_en =
  WORDLINE_ENABLE_BUFFER(access_clk_bar, s_en_bar)`, whose first stage is a
  NAND2 (`WORDLINE_REQUEST_NAND`) with the drive of the inverter it replaces.
  Before, the read wordline stayed on for the whole clock-low phase, 1.1 to
  2.2 ns (25 to 44 % of the period) after the sense enable had fired, with
  the amplifier already isolated from the bitlines and the bitline at its
  rail; the pulse was 1.8 to 9 times longer than the sensing needs. It now
  starts to release 45 ps (FF) to 192 ps (16x16 10T mux SS) after the
  amplifier's terminal sees the sense enable; the sense margin at the
  amplifier is unchanged. A write wordline lasts the whole access as before.
- Precharge and write slot wait for the previous enables: `pre_gate =
  wordline_off & !we_hold & enables_off` with `enables_off = !(s_en | w_en)`
  (`PRECHARGE_GATE_AND`, `ENABLES_OFF_NOR`), and `write_slot = wordline_off &
  pre_off_ready & !s_en` (`WRITE_SLOT_AND`, now an AND3; the precharge input
  tied high without the precharge-off guard). Before, these orders were set
  by path length only (27 ps from write enable off to precharge and 15 ps
  from isolation off to precharge with zero settling stages); they are now
  gate relations: 42 ps (FF) / 174 ps (SS) from sense enable off to
  precharge, 86 / 321 ps from write enable off to precharge, 67 / 262 ps
  from sense enable off to write enable at 8x4, 70 ps at zero stages.
- Write enable ends with the wordline enable: `selected_slot = cs_pre &
  write_slot`, `write_window = wl_en | selected_slot`, `w_en = we_hold &
  write_window` (`SELECTED_SLOT_AND`; `WRITE_ENABLE_AND` is an AND2). The
  V2.1.7 deselect dropped the drivers with the local wordline at 0.50 V (TT)
  / 0.43 V (SS) at a write -> idle boundary; they now release 30 ps after
  the wordline is below 0.1 VDD, as at a selected boundary.
- Column-mux select: a DC level in this testbench (the column address never
  changes), so it cannot pulse or overlap; the amplifier is isolated by
  `sa_iso` while it fires and while the drivers are on. Recorded, unchanged.
- Loads: `access_load = 1.25 * wl_en_scale + 2.5`; `sen_load` and
  `wen_load` count the new observer inputs; `validate_for` expects the new
  access load. The validator prints `XTIME_CONTROL:selected_slot`, `s_en_bar`
  and `enables_off`.
- Checks: the local checker, the validator and the qualification scorer
  search the wordline release from its rise; new checks
  `_sense_before_wordline_release`, `_sense_off_before_precharge`,
  `_write_enable_off_before_precharge`, `_sense_off_before_write_enable` and
  `_write_enable_off_before_write_enable`, each with a metric, and the metrics
  `_wordline_release_to_deadline_ps` (the read release relative to the edge
  that ends the access: 7 ps before it at 256x4 SS, 309 / 90 ps after it at
  512x4 6T / 10T, where the trigger sits close to the deadline and the
  release path is 547 ps) and `_wordline_at_write_enable_off_v` /
  `_write_enable_off_after_wordline_ps` (the wordline tail when a write's
  drivers release: at 256 rows the release and the local wordline's 50 % fall
  coincide in both releases, at 512 rows the drivers release 295 ps before
  the wordline is off, as before); a read cycle's write-enable quiet window
  ends at the later of the wordline release and the sense enable off. Tools in `dev/v218_overlap/` (`overlap.py`,
  `summarize.py`, `compare.py`, the probe launchers).
- Naming: `PRECHARGE_WRITE_AND` / `Xpre_write_gate` -> `PRECHARGE_GATE_AND` /
  `Xpre_gate`; new nodes `s_en_bar`, `enables_off`, `selected_slot`;
  `WORDLINE_ENABLE_BUFFER` has the ports `A`, `B`, `Z`
  ([map](design/TIME_CONTROL_PATH.md), section 6).
- `timing_lookup.json` `v2.1.8-timing-7`: every class kept and re-evidenced
  (91 of 91 cases, 251,553 of 251,553 checks) on the V2.1.7 matrix plus read -> read and write -> write
  decks at 8x4 and 16x16 (6T and 10T, FF and SS) with mismatch seeds; two
  four-rank DC operating points and the five cases that failed only the two
  provisional checks above (before they became metrics) were rerun in a fix
  pass on the release checker. `TRESTORE` falls by 30 to 40 % at 64 to 512
  rows because the restore no longer waits for a wordline released at the
  edge; the per-cycle energy of the shared read and write decks rises 1 to
  3 % (the release work moves into the access, plus the observer gates): with
  `K = 1`, `N = 9` the bitline is at its rail when the amplifier fires, so
  the bitline energy needs an earlier trigger, which the sense-timed wordline
  now makes safe (record, section 4).
- Tests: `test_read_wordline_ends_at_the_sense_enable_and_the_clock_high_enables_wait_for_the_enables`
  and the updated wiring and load tests (145 tracked tests pass); the local
  checker tests model the sense-timed release and a corruption for every
  new check (`dev/tests`).

## V2.1.7 — 2026-09-17 — select gate of the clock-high enables; TIME_CONTROL rename

V2.1.7 reviews the V2.1.6 write slot for abnormal functions and boundary
bugs, fixes the five it found, renames the control block `TIME` to
`TIME_CONTROL` and re-evidences the timing table on the new sources
([record](design/SELECT_GATE_V2_1_7.md)). Driver sizes, cells and the timing
classes are unchanged; steady-state read and write cycles are unchanged.

- Select gate (`TIME_CONTROL`): `cs_pre = cs & cs_delayed` delays only the
  rising edge of the select (the eight V2.1.6 stages plus an AND2) and now
  gates both clock-high enables, `PRE = NAND3(clk_buf, cs_pre, pre_gate)` and
  `w_en = we_hold & cs_pre & write_window`. With the raw select, `w_en` rose at
  an idle -> write edge as soon as the select did, and the testbench's
  write-data hold latch closed only 24 to 29 ps after new data had settled at
  FF 1.1 V / -40 C (4x4 and 8x4, 6T and 10T, twenty mismatch samples); it is
  now 109 to 112 ps there (180 ps TT, 404 ps SS). The symmetric V2.1.6 delay also
  kept the select high for eight stages into an unselected cycle while the
  wordline-off guard re-opened the precharge gate (58 ps apart at 2x4 FF);
  the gap is now 114 ps. No V2.1.6 deck failed on either race. An idle -> read
  precharge starts one gate later (+10 to +41 ps) and an idle -> write slot
  opens after the select delay (TWSLOT 206 / 329 / 738 ps at FF / TT / SS,
  8x4).
- Fix: without the replica guard (the factory defaults) the V2.1.6 write
  window was `wl_en | wl_en_bar`, constantly high, so `w_en` stayed on for
  every phase of consecutive writes (1.00 V in a stand-alone deck): the
  write-data hold latch never reopened and the drivers were on while the
  previous wordline fell. Such a block now starts its drivers with `wl_en`.
  The testbench always builds both guards.
- Fix: a deck with the documented `sizing.precharge_guard_stages: 0` raised in
  the V2.1.6 safety measures; they now require the two guards only.
- Fix: five builders ignored the block's transistor models (the replica
  wordline observer was hard-coded; the replica delay chain, guard delay,
  wordline-enable buffer and every tapered buffer used the defaults), contrary
  to the V2.1.6 entry. Netlists with the default models are unchanged.
- Fix (testbench): a `select_every` write deck holds its data through the idle
  cycles and changes it 0.1 T before the next selected edge. The V2.1.6 decks
  changed it inside the idle cycle, so the V2.1.6 idle -> write probes never
  registered new data at that edge.
- Checks: the local waveform checker adds `strict_idle_<k>_precharge_off` /
  `_enables_off` for unselected cycles and
  `strict_cycle_<k>_col_<c>_data_before_write_enable` with
  `_data_to_write_enable_ps`.
- Rename: `TIME` -> `TIME_CONTROL` (class, subcircuit, top-level instance
  `XTIME_CONTROL`, so probe paths read `XTIME_CONTROL:<node>`),
  `TIMEFactory` -> `TimeControlFactory`, `create_time_circuit` ->
  `create_time_control_circuit`, and every inner subcircuit named after its
  class (`pdrive` -> `CLOCK_BUFFER`, `wl_pdrive` -> `WORDLINE_ENABLE_BUFFER`,
  `DFF_BUF` -> `DFF_BUFFER`, `D_LATCH_ADDR` -> `HOLD_LATCH`, `delay_chain` ->
  `REPLICA_DELAY_CHAIN`, `wen_delay_chain` -> `UNIT_DELAY_CHAIN`, `ADDR_DFF` /
  `DATA_DFF` -> `ADDRESS_REGISTER` / `DATA_REGISTER`, `AND3_WEN` ->
  `WRITE_ENABLE_AND`, `AND2_PRE_*` -> `PRECHARGE_*_AND`, the tapered-buffer
  roles `ABUF` / `WEN_BUF` / `SEN_BUF` / `ISO_BUF` / `PRE_BUF` ->
  `*_BUFFER`; full map in
  [`TIME_CONTROL_PATH.md`](design/TIME_CONTROL_PATH.md), section 6). Instance
  and node names inside the block are kept. Before the circuit change the
  rename was proven on 1351 of 1351 netlists identical after the name map
  (`dev/v217_boundary/`).
- `timing_lookup.json` `v2.1.7-timing-6`: every class kept and re-evidenced
  (74 of 74 cases, 236,570 of 236,570 checks; the 64x16 mux read idle probe
  after a four-rank DC operating-point failure, rerun with Newton line search
  as V2.1.6 did for the 8x128 read).
- Tests: one per finding, each shown to fail with its fix reverted; the V2.1.6
  expectation of a `wl_en_bar` write window for guard-less blocks is removed.

## V2.1.6 — 2026-09-16 — write drivers in the precharge slot; TIME audit and refactor

V2.1.6 changes the write timing of the TIME block and re-evidences the
timing table on the new sources ([record](design/WRITE_SLOT_V2_1_6.md)).
It follows an audit of `time_generate.py` and `replica_column.py` (static
connectivity over 630 TIME configurations, eight Xyce decks checked against
every Boolean relation of the block; tools in `dev/v216_time_audit/`) that
found no functional defect but one forbidden ordering: in a write cycle the
local wordline rose 23 to 189 ps *before* the write drivers had the bitlines
at their rails, because `wl_en` and `w_en` fanned out from the same access
request. Reads are unchanged within 10 ps on the wordline path; driver size
classes are unchanged.

- Write slot (`TIME`): in a write cycle the precharge is inhibited by the held
  write request (`pre_gate = pre_ready & we_hold_bar`) and
  `w_en = we_hold & cs & (wl_en | (pre_ready & pre_off_ready))` turns the
  drivers on in the clock-high phase, once the previous wordline (replica
  observer plus settling) and the physical precharge (far-PRE observer plus
  settling, now exported by the guard as `pre_off_ready`) are off. BL/BLB sit
  at their write rails 1.9 to 2.3 ns before the wordline starts in every
  measured case; the cell flips 131 ps (8x4 TT) and 333 ps (64x16 SS) after
  the clock falls instead of 287 and 826 ps. The select is delayed by eight
  unit stages into the precharge gate (`cs_pre`): the request reaches that
  gate five gate delays after the select, so the first selected write after an
  idle cycle otherwise precharged for those five gates under the rising write
  enable. The replica column's write driver, an inert load since V2.0.5, is
  enabled from the far `w_en` tap and writes a 0; `wen_load` counts it. The
  testbench's write-data hold latch scales with the write-driver input class
  (`wd_in`) and `wenb_scale` counts that scale: the first evidence run found a
  unit latch driving the 8x input of a 512-row array still slewing (about
  700 ps at SS 125 °C) when the slot turned the drivers on; decks up to 64
  rows are unchanged by this.
- Checks: bitlines are expected restored to VDD before a read and at the write
  rails before a write. `VRESTORE_ERROR_k` takes its targets from the next
  selected cycle; before a write `VWL_WEN_*_k` (wordline low when
  `XTIME:write_slot` rises) replaces `VWL_PRE_*_k`; write decks write 1 then 0
  and measure `TWSLOT` (capture edge of the next write to its driven bitline at
  0.1 VDD) instead of `TRESTORE`, which `timing_from_measurements` and the
  minimum-period estimate use as the clock-high work of a write cycle. The
  local checks add `bitlines_driven_before_wordline`,
  `release_before_write_enable` and `WL_during_write_slot`, end the
  write-enable quiet window at the wordline release, and the scorer checks
  `all_bitlines_driven` and `wl_off_before_write_enable` for write cases.
  `scoring_sources.json` is refreshed.
- Boundary probes: the testbench option `select_every` (single read/write
  decks; validator case key of the same name) selects one cycle in N, so the
  idle → write boundary is simulated; the toggled write data covers
  write → write with new data.
- `timing_lookup.json` `v2.1.6-timing-5`: every class kept and re-evidenced
  at its bound (58 of 58 cases, 213,020 of 213,020 checks); the write side only gained margin.
- Readability: `time_generate.py` is rebuilt as fifteen builders in signal
  order with the sizing policy in `ControlSizing`, named constants for the
  unit devices, the passed models honoured everywhere, type hints, the history
  moved to [`TIME_CONTROL_PATH.md`](design/TIME_CONTROL_PATH.md) and the
  unused `pdrive2_for_pre`, dead `disconnect` path and unused imports removed;
  classes are renamed (`pdrive` → `ClockBuffer`, `wl_pdrive` →
  `WordlineEnableBuffer`, `dff` → `Dff`, `DFF_BUF` → `DffBuffer`,
  `DelayChain` → `ReplicaDelayChain`, `WenDelayChain` → `UnitDelayChain`,
  `ADDR_DFF` → `AddressRegister`, `DATA_DFF` → `DataRegister`,
  `D_latch_addr` → `HoldLatch`, `Replica_Cell` / `Replica_Column` →
  `ReplicaCell` / `ReplicaColumn`) while every subcircuit `NAME`, instance and
  node name is kept. The refactor was accepted on 1331 of 1331 byte-identical
  netlists (`dev/v216_time_audit/snapshot_decks.py`).

## V2.1.5 — 2026-09-16 — equivalent model as an input; 10T read-path resize

V2.1.5 merges the top-level `equivalent_modeling/` directory into
`sram_compiler/` and makes the equivalent array model a first-class simulation
input, and closes the 10T items (4, 5 and 7) of the
[V2.1.3 open items](plans/V2_1_3_OPEN_ITEMS.md) with its own
[evidence run](design/TIMING_10T_BUDGET_V2_1_5.md). It changes the 10T cell and
the 512-row 10T clock; 6T decks, driver size classes and the TIME circuit are
unchanged. Against a detached `acec3be` worktree, the merge alone leaves all
nine reference decks byte-identical; with the cell resize the five 6T decks
stay identical and the four 10T decks differ only in the eleven pull-down
widths of the array and replica cells (22 lines; the 512-row clock is the only
timing change). See `outputs/validation/V2.1.5-10t/deck-comparison/`.

- Equivalent model (`sram_compiler/equivalent_modeling/`): the package holds
  `EquivalentConfig` / `resolve_equivalent` (modes 0-4, validated once and
  recorded), the accuracy and runtime entrance `compare.py`, and the guide that
  was `equivalent_modeling/README.md`. `global.yaml` gains an `equivalent:`
  block; `--real-cell-mode`, the `real_cell_mode` testbench argument and
  `REAL_CELL_MODE` in `main_sram.py` default to `None` and take that block,
  exactly as `interconnect` already worked. Run records (`summary.json`,
  `*.variation.json`, the validator's `result.json`) carry the resolved
  `equivalent` block, and `full_device_coverage` still requires mode 0.
  The old `equivalent_modeling/main_sram.py` (which called the removed
  `enable_mc=` API) and its `run.sh` and notebook are replaced by
  `python3 -m sram_compiler.equivalent_modeling.compare`.
  Measured against mode 0 at TT 1.0 V / 25 °C
  ([record](design/EQUIVALENT_MODEL_V2_1_5.md)): read and write delay stay
  within 0.21 % and 0.48 % in every mode; average power stays within 0.4 % in
  the modes that keep the whole target row (1 and 2) but falls 7 to 8 % in the
  modes that keep only the target column or cell (3 and 4); the static-power
  term is unusable (−27 % in mode 1, sign-flipped in mode 2). The runtime gain
  grows with the array — for the cross mode, +31 % (slower) at 8x8, −16 % at
  16x16 and −37 % at 32x32, where the target-cell mode reaches −51 % — because
  the netlist-time Xyce extraction cost is roughly fixed while the removed
  transistors are not. That is not the order-of-magnitude figure the old
  documentation carried, which came from a topology that aggregated the omitted
  cells' wire loads; V2.1.1 keeps every wire segment.
- 10T cell (`sram_compiler/config_yaml/sram_10t_cell.yaml`): the pull-down
  width becomes 287 nm (was 205 nm, the 6T single-pull-down width), with the
  usual ±20 % optimizer bounds. Reason: the 10T cell pulls its storage node
  down through two stacked NMOS, so the read-disturb bump is the read current
  through twice a 6T pull-down resistance. At SS 0.9 V / 125 °C the bump peaked
  at 0.196 V (0.131 V for 6T) and a 512-row bitline left the node at 0.100 V at
  the 11 ns deadline against the 0.1 VDD storage tolerance. The wider stack
  drops the peak to 0.161 V and the 512-row deadline value to 0.056 V, moves
  the read output about 400 ps earlier at every height (the 128x8 bound gains
  156 ps, from 282 to 438 ps), and raises read SNM by 11 mV while costing
  16 mV of write SNM and 16.5 % of bitcell area (1.194 → 1.391 µm²).
- `timing_lookup.json` `v2.1.5-timing-4`: the `SRAM_10T_CELL` 512-row row
  budget becomes 4000 ps (10 ns, was 5600 ps / 14 ns). Every other class is
  unchanged; the resize leaves them with more margin than the 250 ps rule
  needs, which a later round can spend with its own evidence.
- Fix (`sram_compiler/sizing/timing.py`): a variant class is floored at the
  shared class at every size, not only at the tabulated anchors. Beyond the
  last anchor each ladder extrapolates its own final ratio, and the V2.1.5 10T
  ladder ends 3200 → 4000 ps against the shared 2500 → 3600 ps, so a 513-row
  10T array resolved to 12.5 ns while the same 6T array resolved to 13 ns —
  a variant asking for less time than the architecture it was separated from.
  The same anomaly existed for 6T-with-mux (12 ns against 13 ns) and a test had
  worked around it. Only extrapolated (already flagged, unevidenced) sizes move.
- Known staleness surfaced by the resize: `size_optimization/openyield_v2/datasets/train_10t.csv`
  was sampled over the old 10T bounds (`pd_width` 164-246 nm) and is not
  regenerated here, so the offline 10T surrogate now describes a design space
  that barely overlaps the tracked cell. Recorded in the README; the 6T dataset
  is unaffected.
- Tests: the equivalent option from YAML, CLI and API with its provenance
  (`tests/test_equivalent_modeling.py`); the new 10T classes and table version;
  the variant floor beyond the table; the entrance default.
- Evidence ([record](design/TIMING_10T_BUDGET_V2_1_5.md) and its tables): on
  the released sources **45 of 45 attempts and 173,193 of 173,193 checks pass**
  — 20 nominal 10T class-bound cases at SS 0.9 V / 125 °C (SF for the write,
  FF 1.1 V / −40 °C for two sequences the V2.1.3 round never ran), including
  the 512-row bound at its new 10 ns class, and 25 mismatch seeds (the V2.1.3
  32- and 64-row seeds and ten-seed 8x4 pilot repeated on the new cell, plus
  three seeds at the 128-row bound and three on the FF −40 °C sequence). No
  class bound keeps less than 438 ps of read-output margin nominally, and the
  tightest sample of the round is 266 ps. One per-device case took the
  runtime's single timestep retry and then passed; no solver aborted.

## V2.1.4 — 2026-09-15 — 6T timing: write-request hold latch and 6T ladders

V2.1.4 closes the 6T items of the [V2.1.3 open items](plans/V2_1_3_OPEN_ITEMS.md)
(1 to 3) and leaves the 10T items for the next round. It changes the TIME
block of every array and the clock of 6T arrays; driver size classes and the
10T clocks are unchanged.

- Fix (TIME, `sram_compiler/subcircuits/time_generate.py`): the write request
  registered on the clock edge that ends an access gated `w_en` and `s_en`
  directly. At FF 1.1 V / −40 °C the register output moved 20 ps after
  CLK_BUF while the access request fell at 38 ps, so the write enable pulsed
  to 0.38 to 0.72 V (8x4 6T, eleven retained V2.1.1 to V2.1.3 sequences) while
  the read wordline was still on. The request now passes a hold latch
  (`Xwe_hold`, the address-latch cell enabled by `wl_en_bar`) and both enables
  use the held request; the `wl_en_bar` inverter counts the two extra NAND2
  inputs. The read-cycle write enable now peaks at 6 mV and the access path is
  unchanged (16x16 mux and 8x4 SS sequences reproduce their request-to-output
  times exactly).
- Fix (testbench, `_analysis_stop()`): the transient stop `1 ns + 8.7 T` is
  rounded up onto the output interval. The new quarter-nanosecond clocks put
  it off the 2 ps print grid (42.325 ns at 4.75 ns), and some traces then
  ended with the final time printed twice, which the waveform scorer rejects
  (`strict_finite_monotonic`; five of 93 attempts of the first final run).
  Only decks whose stop was off the grid change.
- `timing_lookup.json` `v2.1.4-timing-3`: shared row budgets
  1800/1900/2100/2500/3600 ps (was 1600/1800/2000/2400/3600; 4.5 ns up to
  32 rows, 4.75, 5.25, 6.25 and 9 ns), which now apply to 6T without a mux,
  and a `SRAM_6T_CELL` variant with `"mux": true`, rows
  1900/2000/2200/2700/3600 ps and shared columns (4.75, 5, 5.5, 6.75 and
  9 ns); `ArrayTiming.budget` reads `SRAM_6T_CELL/mux`. Reason: V2.1.4 probe
  reads at SS 0.9 V / 125 °C on the V2.1.3 table left 109 ps at the 32x16 6T
  bound (4 ns) and 206 ps at 64x16 (4.5 ns), and a mux delays the 6T output
  by about 70 to 275 ps from 32 to 512 rows, so the 32x16 (38 ps), 128x8
  (53 ps) and 256x4 (29 ps late) mux reads failed their output checks
  nominally. Both 6T ladders now follow the V2.1.3 rule of at least 250 ps
  between the read output and the 1.2 T deadline at every class bound.
- Tests: the TIME latch connectivity for buffered and unbuffered enables
  (no access gate sees the raw request), the 6T ladders with and without a
  mux at every bound (a mux never gets a shorter clock), the updated shared
  classes and `SRAM_6T_CELL/mux`. Local scorer: `write_enable_quiet` and
  `sense_enable_quiet` checks with tests; the validator prints `WE` and
  `XTIME:we_hold`; new `dev/v214_boundary_enable.py`.
- Evidence ([record](design/TIMING_6T_BUDGET_V2_1_4.md) and JSON): on the
  final sources **92 of 92 attempts and 305,371 of 305,371 checks pass**:
  21 nominal class-bound cases at SS 0.9 V / 125 °C (SF for writes) with at
  least 257 ps of read-output margin (298 ps below 512 rows); the V2.1.1
  write-waveform gate (10 cases, including an 8x8 mux FF −40 °C sequence and
  one 8x4 10T mux smoke case); three mismatch seeds at the 32-, 64- and
  128-row bounds with and without a mux and at the 16x16 mux sequence (all
  pass with 264 ps or more); and ten seeds each at the 8x8 mux sequence and
  the 16x16 SS read, 16x16 SF write and 8x4 FF −40 °C pilot cases. Every
  boundary write-enable peak is at most 17 mV. The probe run on the V2.1.3
  table (11 of 14 attempts) and the 22 superseded first-run attempts (17
  passed, 5 failed on the duplicated sample) are preserved in the same record.
- Known limits: the V2.1.3 10T boundary run, seeds and pilot ran the old TIME
  block and are not repeated (one 8x4 10T mux sequence smoke case); the 10T
  read-disturb bump and the 10T classes without seeds stay open, now with the
  TIME re-run as open item 7.
- Validation: 132 compiler tests on Python 3.11 and 3.9, 55 development and
  six optimizer tests, compileall and `git diff --check`. Eight decks compared
  against a detached `9b785f8` worktree: 10T decks differ only in TIME, 6T
  decks in TIME and, where a class changed, in stimulus, `.TRAN` and
  measurement times. Every deck of the first final run was regenerated from
  the final sources without a simulator: 70 of 93 identical (probe order and
  per-device model paths normalized, model cards compared), the 22
  off-grid-stop attempts changed as intended and were rerun, and one
  per-device read differed only by the runtime's single DC-operating-point
  retry option. Artifacts are
  local under ignored `outputs/validation/V2.1.4-6t-timing/`.

## V2.1.3 — 2026-09-14 — separate 10T timing budget

V2.1.3 changes the clock of every 10T array and nothing else: no 6T deck,
driver size class, TIME circuit or check changes. The V2.1.2 follow-up left
the shared 4 ns class failing 10T cells with a column mux at SS 0.9 V /
125 °C; this release adopts a separate, evidenced 10T budget and expands the
per-device pilot to ten seeds.

- `timing_lookup.json` (`v2.1.3-timing-2`) gains `variants`: a separate row
  and column ladder keyed by `cell_type` and, optionally, `mux`. The loader
  requires the shared anchors, one entry per cell type and never a budget
  below the shared one. `resolve_timing()` selects the variant for the
  baseline's cell type and mux; `ArrayTiming.budget` records `shared` or
  `SRAM_10T_CELL` (a mux-restricted variant would read `.../mux` or
  `.../nomux`) in every run record.
- The `SRAM_10T_CELL` variant applies with or without a column mux: row
  budgets 2000/2200/2400/3200/5600 ps for ≤32/64/128/256/512 rows (5, 5.5, 6,
  8 and 14 ns) and column budgets 200 ps above the shared ladder
  (1800/1800/1800/2000/2200/2600/3000/3400 ps). Reason, from the retained
  V2.1.2 traces and the V2.1.3 evidence run: at SS 0.9 V / 125 °C the 10T read
  port discharges the replica bitline about 1 ps per row slower than 6T and
  its sense path adds a near-constant 200 ps, so the penalty grows with height
  (about 265 ps at 16 rows, 440 ps at 128 rows); a column mux adds a pass-gate
  delay of about 140 ps that the shared class absorbs for 6T (74 ps of margin
  at 16x16) but not for 10T. Rule applied to the 10T ladder: at least 250 ps
  between the local read output and the 1.2 T deadline at every class bound,
  nominal SS 0.9 V / 125 °C, because the validator samples the output 60 ps
  before the deadline and the mismatch seeds move it by up to about 110 ps.
  The 512-row class is set by the storage node: the 10T read-disturb bump
  (0.196 V against 0.131 V for 6T) decays only as the bitline discharges, and
  at 11 ns a 512-row array still held Q at 0.108 V at the deadline against
  the 0.1 VDD tolerance, so that class gives the bitline about 5.6 ns.
- Tests: the 10T budget at every anchor and beyond it with and without a mux,
  the shared budget for 6T under PVT and RC changes, `ArrayTiming.budget`, a
  mux-restricted variant, and rejection of malformed variants (129 compiler
  tests).
- Local tooling: `dev/v212_followup_report.py` accepts `--base`, `--queues`,
  `--diagnostics`, `--version` and `--plan` so it summarizes the V2.1.3 queues.
- Evidence ([record](design/TIMING_10T_BUDGET_V2_1_3.md) and JSON): on the
  final table **56 of 58 attempts and 146,952 of 160,814 checks pass**:
  all 18 nominal class-bound cases at SS 0.9 V / 125 °C with and without a
  mux (8x4 to 512x4 and 8x128, plus the 64x16 SF write), at least 282 ps
  of read-output margin at every 10T bound; 3 of 3 mismatch seeds at the 32-row
  and 64-row bounds; the Phase 5 pilot expanded to ten seeds (8x4 10T mux 10
  of 10 at 5 ns, 16x16 SS read, 16x16 SF write and 8x4 FF cold 10 of 10
  each with the V2.1.2 seeds). The three rejected ladders (23 of 27 attempts)
  are preserved in the same record: the shared class fails 16x16 10T without
  a mux, a flat 200 ps fails the 128x8 mux read at 5.5 ns, and the 4.5 ns
  32-row class fails one of three mismatch seeds at 32x16.
- Known limit: the 6T sequence with a column mux passes the shared 4 ns
  class at 16x16 with 74 ps nominal and fails 2 of 3 mismatch seeds
  (51 and 57 ps at the deadline); a 6T-with-mux variant needs its own
  evidence run. The FF −40 °C write-enable spike recurs at 0.38 to 0.72 V
  over seven new seeds with no check failing (open TIME item). The 6T ladder
  is unchanged and not re-evidenced under mismatch at its 128- and 256-row
  bounds (203 and 175 ps nominal). These and the other open items are
  collected in the [V2.1.3 open items](plans/V2_1_3_OPEN_ITEMS.md).
- Validation: 129 compiler tests on Python 3.11 and 3.9, 54 development and
  six optimizer tests, compileall and `git diff --check`. Eight decks compared
  against a detached `e26a7ec` worktree: the four 6T decks are byte-identical
  and the four 10T decks differ only in stimulus, `.TRAN` and measurement
  times. Artifacts are local under ignored `outputs/validation/V2.1.3-10t-mux-budget/`.

## V2.1.2 — 2026-09-13 — run traceability and write-failure inventory audit

V2.1.2 changes no generated circuit, timing budget or driver size class. It
audits the V2.1.1 release record and the supplied
[first-500 write-failure inventory](issue_reports/write_failure_cases_first_500.md),
and repairs the traceability gaps the audit exposed.

- Apply the single bounded-step Xyce retry to `.TRAN` lines with a start time
  or a step ceiling coarser than 20 ps (SPICE suffixes accepted). Decks already
  bounded at 20 ps are not rerun; two-field decks are retried as before.
- Add `--vdd` and `--temperature` to the per-device CLI. They override
  `global.yaml`, reach the deck and summary, and enter the run identity, so a
  PVT point is never filed as another point's attempt. Run names change for
  every configuration.
- Record the resolved Xyce installation as `xyce` in CLI summaries, including
  failed runs, and in `Sram6TCoreMcTestbench` `.variation.json` sidecars.
- Remove the unreferenced `docs/DRIVER_SIZING_data.csv` and
  `docs/TIMING_AUTOCONFIG_data.csv` copies added in V2.1.1. The supplied root
  CSVs are unchanged and remain the linked, hash-pinned sources.
- Append the audit to the inventory without changing its table. Its source CSV,
  logs and collection script are absent; `Step size reached minimum step size
  bound` is a non-fatal solver warning (all 49 local logs containing it
  completed); its `ORTE` failure is Open MPI, while the `openyield` Xyce links
  MPICH. The inventory records no electrical write failure.

Audit of V2.1.1: its 124 compiler / 54 development / 6 optimizer test counts,
33,096 waveform checks across eleven cases, 73 evidence source hashes and the
precharge-off tau (584.5864 ps for 8x512, 0.1451 ps for 8x4) were reproduced.

Validation: **126 compiler tests pass on Python 3.9 and 3.11**, alongside 54
development and six optimizer tests; compileall and `git diff --check` pass.
Four default CLI decks (8x4 write, 16x16 per-device read, 8x4 SF read&write,
64x4 write without waveforms) match a detached `6001a48` worktree after
run-directory normalization. The per-device model file name differs only
because its digest includes the absolute PDK path; the model and audit files
are byte-identical. Seven 16x16 write configurations from the inventory (six
reported failures and one reported pass) pass nominal V2.1.1 runs with runtime
release, access, hold, restore and metric checks, with no solver warning or
retry. A V2.1.2 CLI run with `--vdd 0.8 --temperature 125` (SS) reproduces the
wrapper-generated deck and its `.mt0` measurements byte-for-byte and records
`xyce`. No independent waveform scoring, PVT/mismatch campaign or qualification
record is added.

### Post-release review — 2026-09-13 (compiler identity unchanged)

- Move the supplied CSVs, byte-identical, to `docs/data/` with an
  [audit record](data/README.md). The local `dev/sizing/campaign.py` reads the
  new path; `scoring_sources.json` changes only that file's hash, which
  supersedes earlier local scoring identities (no qualification record exists).
  `sizing_rules.json` keeps the bare file name because it is part of the driver
  baseline digest. Correction: no tracked file pinned the CSV hashes; the
  manifest pins `campaign.py`, not the data.
- Both CSVs are sound; the timing CSV has caveats (duplicate rows, unflagged
  superseded `v202_corners`/`v202_tsweep` runs, no sample index). Values the
  documents misprinted are corrected in place from the CSVs.
  `DRIVER_SIZING_PROPOSAL.md`: the 512x4 K1N9 read access is 1683 ps (printed
  1676) and eight other cells are re-rounded. `TIMING_AUTOCONFIG.md`: the
  section 3.4 PVT factor table was built from the superseded corner runs and is
  recomputed from `v202_corners_f`. Also corrected there: eight worst-case factor
  cells, two fit cells, the V2.0.2 row count (494, not 526), the `TCLK_WLEN` and
  mux ranges, the write-phase exception and the Monte Carlo sigma wording. This
  file: V2.0.3 row count (615, not 614), write sigma and factor range; V2.0.2 mux,
  10T and fastest-corner statements. "108 corner runs" is correct: the 72 final
  rows plus 36 `read&write` decks the CSV does not hold. The V2.0.4 snapshot keeps
  its original numbers; no deck, table or rule changes.
- Waveform rerun on these sources: all eleven V2.1.1 release cases pass
  **33,096 checks** with unchanged per-case counts, and every `.prn` waveform and
  `.mt0` measurement is byte-identical to V2.1.1. Decks differ only in the order
  of three case-insensitive `.PRINT` names. The seven 16x16 inventory write
  points, now scored by the independent waveform checks, pass **8,183 checks**
  (1,169 each); `TWRITE_TOTAL` is within 0.4 ps and `VPRE_ACCESS_ERROR` within
  0.0001 V of the CLI audit, with minimum PRE90-to-WL50 188.9 ps and WL10-to-PRE90
  restore 122.6 ps (SF 1.1 V 0 °C). None of the 18 runs printed a solver warning or
  needed a retry. Two-sample per-device 2x2 CLI writes (TT, and SS 0.9 V 125 °C
  through `--vdd`/`--temperature`) pass and record `xyce`. 126 compiler, 54
  development and six optimizer tests pass. Artifacts are local under ignored
  `outputs/validation/V2.1.2-rerun/`; this is functional screening, not
  PVT/mismatch qualification.

### Post-release cleanup — 2026-09-13 (generated circuits unchanged)

- Move the working plans from `plans/` to `docs/plans/` and drop the
  superseded V2.1.0-labeled snapshot of the schedule. Remove the other
  superseded records: the pre-release `DISTRIBUTED_ONLY_V2_1_0` report and
  JSON, the `TIMING_LOOKUP_V2_1_0_FOLLOWUP` review and JSON (its repairs are
  released in V2.1.1), the star-RC screen of the removed topology and the
  completed V2.0.7 `DISTRIBUTED_RC_PLAN`. Their exact contents remain at
  `git show 61d01a7:docs/design/<name>`; the remaining documents cite them
  that way, and every relative link in the tracked Markdown resolves.
- Remove unreferenced code: `GlobalConfig.get_metric()`, `get_metric_names()`,
  `get_objective_formula()` and `get_constraints()`; the CLI's never-called
  `clean_generated_outputs()` (attempts are preserved, not cleaned);
  `Sram6TCoreMcTestbench.get_table_head()`; the never-instantiated `Pbuff`
  standard cell; the equivalent-cell tester's actual-versus-model comparison
  and static-power plotting helpers; and `testbenches/yaml_change.py`, which
  nothing imported, needed the absent `ruamel.yaml` and called its own CSV
  summary with a missing argument. The empty
  `yield_estimation/demo_6tstamTestbench.py` is deleted. `dev/`-only helpers
  (`timing_from_measurements`, `provisional_period`, `materialize_decks`) stay.
- Review of the V2.1.1/V2.1.2 runtime diff: the bounded-step retry, the
  PVT overrides, the access-guard probes and the duplicate-probe parser
  behave as recorded. Two repairs: an interrupted CLI solver run
  (`KeyboardInterrupt`, not an `Exception`) left no `summary.json`, so the
  recorded `xyce` and seed were lost; the summary is now written before the
  interrupt propagates, with a regression test. The read&write power-window
  comment cited the old 8.5-cycle analysis stop; it now cites the 8.7-cycle
  `_analysis_stop()`. A waveform plotting failure no longer rejects a CLI run
  whose measures passed: the `.prn` stays on disk, the summary records
  `waveform_error` with `waveform_png: null` and is written to `summary.json`,
  and the run exits 0, with a regression test. The preserved V2.1.1 attempt
  that a duplicate-probe plot failure had rejected keeps its historical label.
- Validation: 127 compiler tests pass on Python 3.11 and 3.9, alongside 54
  development tests, six optimizer tests and compileall; `git diff --check` is
  clean. Three CLI decks (8x4 nominal write, 16x16
  two-sample per-device read, 8x8 SF 0.9 V / 125 °C read&write) match a
  detached `61d01a7` worktree; the per-device model file again differs only in
  the name digest that includes the absolute PDK path, with byte-identical
  content. No generated circuit, timing budget, driver class or run identity
  changes. `current_scoring_version()` hashes every `sram_compiler/*.py`, so
  its digest changes as for any source edit; `sizing_table.json` holds no
  record, so nothing is invalidated. No simulation was run.

### Phase 4–6 follow-up — 2026-09-14 (generated circuits unchanged)

- Run Phases 4 and 5 of the [evaluation plan](plans/V2_1_1_TIMING_FOLLOWUP.md)
  on these sources: 19 nominal lookup-clock cases (widths to 128 columns,
  heights to 512 rows at 9 ns, the unseen 48x20 geometry, FF −40 °C address
  hazards, 6T and 10T mux, local stubs off, cell-pin RC, same-R/C refinement
  and three-times wire stress) and 12 single-rank per-device seeds. **30 of 35
  attempts pass, 70,450 of 82,224 checks**, no solver abort or retry. The
  [follow-up record](design/TIMING_FOLLOWUP_V2_1_2.md) and its JSON hold every
  identity, metric and measurement.
- Finding: the shared 4 ns class is exhausted for 10T with a column mux at
  SS 0.9 V / 125 °C. The 16x16 sequence fails only its read outputs (sense
  enable at 1.184 cycles against the 1.2-cycle deadline) and all three 8x4
  mismatch seeds fail the same way; the array passes at 4.5 and 5 ns and at
  TT on 4 ns. The three-times wire-stress 64x4 read fails its 1.18-cycle
  output sample at 4.5 ns and passes at 5 ns. The clock table and driver
  classes are unchanged; a 10T-with-mux budget one class up is recorded as a
  proposal. A write-enable spike at the read-to-write boundary at FF −40 °C
  (up to 0.63 V under mismatch, no check failing) is an open TIME item.
- Phase 6: the [qualification scope](plans/V2_1_2_QUALIFICATION_SCOPE.md)
  defines extracted-metal inputs, the PVT/sample matrix and the separate
  half-select and yield-estimator briefs.
- Local tooling: the follow-up queue accepts single-rank per-device cases with
  explicit seeds (never the materialized MPI fallback); `dev/v212_followup_report.py`
  and `dev/v212_followup_plots.py` summarize and plot the queues.

## V2.1.1 — 2026-09-13 — distributed-only signal wiring and access exclusion

The [V2.1.1 change](design/DISTRIBUTED_ONLY_V2_1_1.md) removes the star
implementation and uses distributed reference geometry by default. Explicit
star settings fail. Local storage/peripheral series stubs remain selectable;
physical wire ladders remain present when local stubs are disabled.

- Separate array/replica precharge, write-driver and mux/sense connections with
  peripheral wire ladders. Distribute decoder fan-out, DATA_DFF clocks and mux
  selects, and retain local omitted-cell loads on every equivalent-mode wire.
- Extend retention checks through the complete interval after the frozen
  access deadline; reject missing or nonfinite primary CLI measurements.
- Gate WL/write/sense assertion from settled far-end PRE, using a baseline
  wire/load RC delay and four stages. Retain prompt request deassertion and
  the existing replica-WL precharge-on guard. Reject precharge overlap during
  target access at runtime and every column's access in the waveform scorer.
- Check local write-data capture, held data, write enable, driven bitlines,
  selected-row Q/QB, neighbors, precharge exclusion and final retention directly
  on waveforms. Preserve rejected attempts and their source/scorer identities.
- Parse repeated waveform probe names by position and retain them only after
  exact finite-vector equality checks, preserving all sample boundaries and
  rejecting conflicting duplicates.
- Retain V2.0.9 transistor classes and the V2.1.0 timing table. The illustrative
  1-ohm / 0.1-fF wire pitch is not extracted metal; no qualification record is
  promoted and the V2.0.5 rule/artifact identities remain unchanged.

Validation: **eleven final-source waveform cases pass 33,096 checks**, including
the repaired 8x512 SS write at 8 ns and 64x64 TT eight-cycle sequence at 5 ns.
All 124 compiler tests pass on Python 3.11 and 3.9, alongside 54 local development
tests, six optimizer tests and a two-sample per-device CLI write with plotting.
Compilation, isolated generation without `dev/`, and `git diff --check` pass.
The initial wide-write overlap failure and earlier V2.1.0 measurements retain
their original labels and source identities. Remaining class/PVT/mismatch and
extracted-metal work is recorded in the [evaluation plan](plans/V2_1_1_TIMING_FOLLOWUP.md).

## V2.1.0 — 2026-09-12 — fixed timing classes and V2.0.11 review

The clock now follows a lookup table like driver sizing. Fixed row/column
budgets select 4–9 ns within the supplied anchors at the default 25% margin;
unseen dimensions round up, larger arrays extrapolate with an explicit flag.
The period is bound to the baseline and frozen across cell candidates and PVT.
`timing.mode: fixed` and CLI `--period` provide an explicit diagnostic override.
No simulation or fitted historical model runs inside the resolver.

- Extend precharge settling to local-RC star arrays; check physical wordline
  release in both topologies. Add an explicit even-stage override for validated
  wire settings, and connect the distributed output latch at its sense-amp tap.
- Complete data-register and hold-latch startup initialization, addressing the
  reported 8x32 distributed write operating-point failure.
- Enforce access deadlines, retention and restore checks. Move static-power
  measurement after the first restore and include the full last sequence phase.
- Preserve decks, logs, sample identities and stale-output cleanup for both DC
  and timestep retries. Keep CLI attempts and failure summaries, and record the
  actual version, PVT, model and timing configuration.
- Repair all 15 yield call sites for the four-value compiler result and count
  nonfinite delays as failures. Driver transistor classes and the historical
  qualification format remain unchanged.

The initial 3 ns timing budget failed the SS 16x16 read sequence and was
increased after waveform inspection. The [release review](design/TIMING_LOOKUP_V2_1_0.md)
records measured results, test coverage, deck comparisons and remaining limits.
Lookup budgets are design settings, not full PVT/local-mismatch qualification;
no qualification record is promoted.

## V2.0.11 — 2026-09-12 — audit of V2.0.10, distributed control lines, star screen

Audit of the V2.0.10 release commit `9773f29`. No driver size class changes and
every star-topology deck is byte-identical to V2.0.10; distributed decks change
because the control lines that span an array dimension are now modelled as
wires. The [review record](design/WRITE_VALIDATION_V211.md) has the measured
results, the exact coverage and the remaining limits.

- Model distributed RC on every control line that spans an array dimension.
  `PRE`, `w_en`, `w_en_bar`, `s_en` and `sa_iso` run the array width on the
  wordline pitch; `wl_en` runs its height on the bitline pitch. Each is a
  tapped pi ladder whose consumers connect at their own column or row
  (`<net>_line_tap<i>`), with the replica precharge, sense amplifier and
  wordline driver past the last one (`<net>_line_far`). Previously a
  512-column `PRE` or `s_en` reached the first and last column at the same
  instant. The driver output keeps the original net name, so existing measures
  and prints still observe the driver; `control_tap()` returns the plain net in
  star topology. Control-line wire RC is not added to `DriverLoads`, exactly as
  WL/BL wire RC is not, so the lookup classes stay independent of the
  interconnect.
- Delete a failed operating point's per-sample outputs after copying them to
  `dcop_attempt/`. `execute_xyce()` left them beside the deck, so a retry that
  wrote fewer `.mt`/`.ms` files than the failed attempt reported the failed
  attempt's value for the missing sample indices.
- Keep the provenance of a rejected distributed release. `run.py` raised before
  writing `summary.json`, discarding the deck, seed, model hashes and driver
  sizes of the sample that has to be investigated. The summary (with
  `precharge_release_checked: false` and the rejection text) and the waveform
  are written first; the run still fails.
- Clamp the per-cycle precharge windows to the analysis stop. Cycle 7 of a
  `read&write` deck ran 0.2 `t_period` past the `.TRAN` end, so `VWL_PRE_PEAK_7`
  claimed a rebound interval longer than was simulated. `add_analysis()` and
  the safety measures now share `_analysis_stop()`.
- Measurement parsing: a value below the numeric floor is a missing value for
  that run instead of removing the measurement, and `generate_mc_statistics()`
  rejects an all-missing frame instead of publishing NaN statistics.
- Documentation: the V2.0.10 deck comparison said the 32 star netlists without
  local RC "are identical"; their circuit is identical but their analysis and
  initialization cards change. Reworded.

Validation: 65 Xyce 7.4 waveform cases (58 pass, 23,815 independent checks) across 6T/10T, mux on/off, five corners, star and distributed wiring, 4x4 to 64x64 and the re-run 8x512, 64x64 and 512x4 V2.0.10 sequences. 85 tracked compiler tests pass under Python 3.9 and 3.11, 24 local development tests pass, and `git diff --check` is clean. Star decks were compared against V2.0.10 in a detached worktree: 24 of 24 identical.

## V2.0.10 — 2026-09-12 — distributed timing and simulation correctness

- Count both sense-amplifier EN/ISO RC sections in TIME loads. Lookup classes
  and continuous sizing coefficients are unchanged; RC-enabled sense and
  isolation buffers intentionally change.
- Restrict read `TWL`/`TBL` measurements to the access phase, preventing startup
  crossings from producing negative `TSWING` on an electrically correct read.
- Preserve requested MC sample indices when files or measurements are missing;
  retain valid zeros and mark negative read-timing/nonfinite values as missing.
- Add cycle-specific distributed wordline-at-precharge measurements. The Python
  simulation API and CLI retain raw evidence and reject missing release events
  or precharge overlap instead of returning ordinary SRAM metrics.
- Add a four-stage replica settling delay with an immediate inhibit to the
  distributed precharge guard, addressing the 512-column release-criterion
  miss. The stage count is frozen with the baseline; star guards add no stages.
- Initialize write-data hold nodes and register slave feedback consistently
  with the existing zero-data startup state, addressing large-write DC failures.
- Retry runtime DC failures once with Newton line search and the same native
  sampling seed, preserving the first attempt. Exit-zero DC failures and
  unsuccessful retries remain failures; electrical misses are never retried.

The [review record](design/DISTRIBUTED_RC_V210_REVIEW.md) gives the waveform
results, exact coverage and remaining wire/timing limits. Historical evidence
remains unchanged; no sizing-table record is promoted.

## V2.0.9 — 2026-09-10 — fixed driver size classes for layout generation

Driver sizing is simplified to fixed values: every array configuration now takes
its precharge, split write-driver, wordline NAND2/inverter and decoder output
inverter scales from a lookup table of integer size classes, so a future layout
library of fixed transistors can implement every array. The legacy `fixed`
rule mode is removed at the user's request, and unseen array sizes are resolved
by interpolation on the class ladder.

- New default `sizing.mode: lookup` reading `sram_compiler/sizing/sizing_lookup.json`:
  row classes ≤32/64/128/256/512 give precharge 1/2/4/8/16, write input 1/1/2/4/8
  and write output 2/4/8/16/32; column classes ≤4/8/16/32/64/128/256/512 give
  wordline inverter 1…128 (doubling), wordline NAND2 1/1/2/3/5/9/18/35 and decoder
  output inverter 1/1/1/1/2/3/5/9. Each entry is the V2.0.5 `rules_only` rule at
  the class upper bound rounded up to an integer, so no array gets a weaker
  driver than the screened rule (the tracked test checks every class against the
  rule). Classes are independent of cell type, mux, RC stubs and interconnect
  mode; the replica stays `(1, 9)` and matched, with canonical read loading,
  decoder scaling and effort buffers always on. TIME control buffers are still
  sized from the resolved class loads and are not tabulated.
- `interpolate_class()`: inside the table an unseen size rounds up to the next
  anchor (48x20 uses the ≤64-row and ≤32-column classes); beyond the last anchor
  the ladder continues geometrically with the ratio of the last two anchors,
  rounded up to integers, and `DriverSizes.extrapolated` is set with the
  synthetic bound in `size_class` (1024x4 gives precharge 32, write 16/64, flagged).
  Extrapolated sizes carry no waveform evidence. `sizing.lookup` selects an
  alternative table (for example a layout library with different classes).
- Removed the legacy `fixed` mode (original `rows/16` and `sqrt` array rules,
  with the 8-row write weakness of V2.0.3) and `fixed_scales`; the resolver's
  mode-dependent replica/load/buffer defaults collapsed accordingly. `rules_only`
  (the derivation basis) and `auto` (qualified records, otherwise rules) remain.
  The optimizer adapter always freezes a baseline now and its `w_rc` default is
  false, as it already was outside the removed mode.
- Deck compatibility: 105 `rules_only`/legacy decks (8x4…16x256, 6T/10T, mux
  on/off, RC on/off, shared MC) generated from V2.0.8 (`c6ce403`) and from this
  tree are byte-for-byte identical, so the rule path the classes derive from is
  unchanged. Lookup decks for 256x8 and 512x4 are identical to their `rules_only`
  decks (classes coincide at ≤8 columns and ≥256 rows); 8x4, 16x16, 64x64 and
  16x256 differ only in the tabulated driver widths and the resulting buffer loads.
- Qualification tools: local `Case` objects carry `sizing_mode`, and
  `dev/sizing/campaign.py` takes `--sizing-mode`, `--samples`, `--seed`, `--cells`,
  `--mux` and `--no-rc` (`local_review.py` takes `--sizing-mode`); the scoring
  manifest was refreshed. `rules_only` case names and cached results are unchanged.
- Runner retry for a failed Xyce operating point (the Stage D open item): Xyce
  7.4 with the runner's KLU direct solver fails `DC Operating Point` for
  occasional mismatch samples, after which the later samples of the same
  native-sampling run start from a corrupt state (dead wordline, negative node
  voltages) and the materialized MPI path aborts. Reproduced on the 16x16 6T SF
  write-box sample 1 of seed 82026 with a 200 ps transient: GMIN stepping and
  source stepping still fail, `CONTINUATION=1/2/33` abort, skipping the operating
  point (`NOOP`) runs but invalidates the measures, Xyce's default linear solver
  converges those three samples but fails a different sample of the 32x32 FF
  write ensemble, and Newton with line search (`.OPTIONS NONLIN SEARCHMETHOD=2`,
  no circuit change) converges every sample of both reproduced decks, as does
  `GMIN=1e-10`, while `GMIN=1e-9` fails the 32x32 sample and `MAXSTEP=1000`
  alone fails the 16x16 sample. On the four-rank materialized path the operating point also
  failed for 64-row and 16x256 write samples (the 64x16 TT sample 0 with KLU on
  four and on one rank), so `execute_local_ensemble` reruns only the failed
  sample through a ladder of line search on the same ranks, the default solver
  on the same ranks, then line search and the default solver on one rank,
  recorded per sample in `execution.operating_point_fallbacks`, and a
  materialized sample that stops with `Time step too small` is rerun alone
  with a 5 ps maximum step (`execution.timestep_retries`; the 1024x4 FS read
  hazard deck passed this way after two of its samples stopped); `run_case`
  additionally reruns a native sampling case once in `retry_dcop/` with the
  line-search option (`xyce_options` in the result) and keeps the original
  attempt; an ensemble with an electrical failure is never retried.
- Validation: 69 tracked compiler tests pass under Python 3.11 and 3.9 (eight new
  lookup tests), 6 optimizer and 21 local development tests pass, `git diff --check`
  passes. Xyce 7.4 waveform screen of the fixed classes (per-device mismatch, three
  samples per deck, seed 82026, full transistor arrays, clock derived from the
  nominal SS read and SS/SF write phases):
  - Coverage (ignored `outputs/qualification/V2.0.9/`): screens (primary decks,
    all five corners for read and write, SF read, read-box and write-box cells,
    eight-cycle read/write sequences) at 8x4, 16x16 and 32x32 for both cells and
    both mux choices with explicit-RC 16x16 variants; at 64x16 for both cells
    and mux choices; at 64x64 and 16x256 (6T); interpolated non-power-of-two
    sizes 3x3, 5x3, 6x6, 12x4 and 20x10 for both cells and mux choices; the
    functional schedule (primary decks, FF/FS 125 C and FF -40 C reads and
    writes with address-change hazards, SS read/write sequence) at 48x20,
    100x50 and 128x32 and at the extrapolated 1024x4, 2048x2 and 8x1024
    classes; pilots with hazards at 256x8, 512x4 and 16x512 and at 128x128 and
    256x64. Hold is checked inside every deck (data retained after wordline and
    write-enable release, no read disturb, quiet unselected wordlines, every
    bitline restored and equalized, the neighbor row's cell retained across the
    address change), plus the 32x1 `unwritable` rejection of the frozen periphery.
  - Result at the time of writing (2026-09-11, 21:45; the last 8x1024 FS write
    deck of the functional campaign was still running, see
    `dev/summarize_qualification.py`):
    58 architectures, 174 of 174 calibration decks, 665 of 686 verification
    decks and 1,989 of 2,025 waveform samples pass, including the 32x1
    rejections behaving as intended, every deck of 48x20, 128x32 and 256x64,
    the 128x128 SS read and SS write-box decks (sense differential 0.86 to
    0.89 V, write phase 0.45 of the read phase), the first five 2048x2 decks
    and thirteen decks recovered by the operating-point and time-step retries;
    every calibration deck passes; every electrical check passes on every array without explicit RC,
    including the extrapolated classes; measured clocks run from 2.3 ns (3x3)
    to 7.4 ns (1024x4). Margins at the frozen clocks on the completed screens:
    write phase at most 0.57 and restore at most 0.58 of the read phase (budget
    0.8), wordline path at most 0.13 of the access phase without RC (budget
    0.15), sense differential at least 0.62 V (512x4) and 0.88 V below 256 rows
    (limit 0.3 V), written data retained at 0.9 V of 0.9 V and driven bitlines
    below 1 mV, wl_en/PRE/w_en/s_en edges 28 to 35 ps at TT without RC.
  - Remaining misses: (1) the explicit-RC 16x16 reads at SS miss only the 0.15
    wordline-phase budget on some samples (186 to 209 ps against about 190 ps,
    both cells, both mux choices; 40 ps better than the V2.0.5 rules but the
    classes carry no RC term by design, and the generic 100 ohm / 1 fF stubs are
    not extracted values), and one TT mismatch sample of the 16x16 mux RC read
    has a 40.9 ps wl_en fall edge against the 40 ps target; all functional
    checks of those decks pass. The extrapolated 8x1024 class (wordline inverter
    256, NAND2 69) misses the same budget by 6 ps on one of three SS read
    samples (138 ps against 131 ps; the 8-row read phase is short while the
    1024-column wordline is long), again with every functional check passing:
    a measured anchor at 1024 columns should replace the extrapolation.
    (2) Xyce operating-point execution failures on
    about 1% of the mismatch samples, concentrated on the 16x256 and 64x64
    write decks on four ranks: sixteen decks hit it, the four native-sampling
    ones (6T/10T 16x16 SF write-box, 32x32 FF write, 10T 32x32 mux FF read) were
    recovered by the ensemble retry, eight materialized ones (6T/10T 64x16 TT,
    64x64 SF/TT/FF, 16x256 SS and SF read, 48x20 FS hazard) by the per-sample
    ladder, three 16x256 write samples (TT and FF sample 0, FS sample 1) fail
    every rung, and the 16x512 SS write-box sample 0 failed plain and
    line-search Newton on four ranks and then exhausted the six-hour limit on
    the default-solver rung, as did the extrapolated 8x1024 SS/SF write-box
    and FF 125 C write samples on the line-search or default-solver rungs;
    those stay unscored execution failures (ten at the time of writing: the
    16x256 TT/FF/FS writes, the 16x512 SS write-box, the 8x1024 SS/SF write-box
    and FF 125 C/-40 C writes, the 100x50 FF -40 C write hazard and the 128x128
    SF write-box, with the 8x1024 FS write still running), while the 16x256 SS/SF write-box and
    SS/SF read decks, the 16x512 SS read and SF write-box decks, the 8x1024
    reads and the other eight 100x50 decks pass. Write decks, above all the
    wide ones (256 to 1024 columns of write drivers and hold latches), are
    where the operating point is fragile; a deck-level fix such as initial
    conditions on the column latches is the open item, not a driver size. The
    recovered decks pass every electrical check, and a sample solved by
    another solver reproduces the KLU measures to 0.1 ps where both converge.
    (3) One runtime limit: the 2048x2 eight-cycle SS read/write sequence
    (about 100 ns of transient on 4,096 cells at the 11.45 ns clock) exceeded
    the six-hour sample limit on four ranks and is unscored; the other 2048x2
    decks, including the hazard reads and writes, pass.
  A screen is not tail or yield qualification: `sizing_table.json` stays empty,
  half-select waveform qualification remains open, and the extrapolated classes
  have three-sample screening evidence only.

## V2.0.8 — 2026-09-09 — audit of the distributed interconnect release

- Reviewed the V2.0.7 code and the three `docs/design/DISTRIBUTED_RC_*.md`
  documents against the implementation. Replica tap mapping, wire R/C
  conservation, probe names, extraction context and identity handling were
  confirmed; the quoted V2.0.7 deltas reproduce from the retained results.
- Fixed `InterconnectConfig`: a directly constructed distributed configuration
  defaulted `cell_pin_rc` to true, contradicting the documented default that
  only `resolve_interconnect()` applied. `None` now resolves to the documented
  default on every construction path.
- Added `interconnect.load_interconnect()` (project-root YAML loader shared by
  the CLI's `--interconnect-config`) and the `INTERCONNECT_CONFIG` setting of
  `main_sram.py`, so the main entrance selects wires in memory without editing
  tracked YAML. CLI run names and summaries stamp `V2.0.8`.
- Documentation corrections: periphery positions are fixed (WL drivers at
  column zero, bitline periphery at row zero), half-select column disturbance
  is not exercised by any sequence deck, TIME's distributed replica input is a
  Xyce hierarchical node connection, and the ~0.99 V "sense differential" is a
  replica-timed full-discharge functional check rather than a margin. The
  equivalent-model guide now states that distributed mode replaces the
  aggregated `pi_res/N`, `pi_cap*N` branches with per-tap loads.
- Validation: 61 tracked compiler tests pass under Python 3.9 and 3.11
  (two new regressions), 6 optimizer and 21 local development tests pass.
  72 star decks (6T/10T, three operations, mux, RC off/default/custom, fixed
  and rules_only) are byte-for-byte identical to V2.0.7 commit `315333b`.
  Fifteen additional Xyce waveform cases pass: 8x4 and 4x8 (K=2) arrays,
  distributed metal without `w_rc`, `cell_pin_rc: true`, equivalent modes 1-3
  with actual extraction, 8x4 SS/SF at 125 C / 0.9 V, 16x16 6T with 1x and 10x
  wires at far and near cells, and 8x4/16x16 10T per-device mux samples. See
  the [validation record](design/DISTRIBUTED_RC_VALIDATION.md). Metal values
  remain illustrative; no sizing-table record or coefficient is promoted.

## V2.0.7 — 2026-09-09 — RC corrections and distributed interconnect

- Corrected RC propagation through factories, real/replica/dummy cells and
  peripheral circuits. The common local default is now 100 ohm / 1 fF;
  testbenches previously defaulted to 10 ohm while most circuits silently
  retained 100 ohm. Sizing loads now consume configured peripheral capacitance.
- Equivalent extraction uses the effective corner, VDD and temperature, with
  the main simulator's 27 C nominal model temperature. Its cache fingerprints
  selected PDK contents, storage-node RC and extraction settings. Unsupported
  equivalent-cell dimension sweeps fail explicitly; full-real sweeps remain
  supported.
- Added opt-in distributed WL/BL/BLB pi ladders, independent metal geometry,
  centered cell taps and subdivision that conserves total wire R/C. Equivalent
  cells retain the wire geometry and load individual taps. Local cell-pin
  parasitics are selectable separately from storage and peripheral parasitics.
- Distributed replica paths match array wire lengths and tap loads, including
  mux/sense input circuits. TIME observes the far replica wordline before
  precharge; transient probes use local cell terminals, sense inputs and far
  bitline endpoints. Star remains the default topology.
- Frozen baselines and run/qualification identities include the physical wire
  model. Runtime source fingerprints cover compiler topology; local scoring
  and report tools use distributed probes and physical identities. Historical
  V2.0.5 measurements and rule labels are preserved; no new sizing-table
  qualification or coefficient recalibration is claimed.
- Added a [configuration guide](design/DISTRIBUTED_RC_MODEL.md) and an
  illustrative YAML example, selectable with `--interconnect-config` or through
  the existing global YAML/Python configuration path.
- Validation: 65 tracked compiler/optimizer tests and 112 subtests pass under
  Python 3.11; all 59 compiler tests pass under Python 3.9. The new interconnect
  module has 91.6% statement coverage from its tracked unit/integration checks.
  Local development tests, CLI generation and a runtime export without `dev/`
  are also checked. See the [validation record](design/DISTRIBUTED_RC_VALIDATION.md).
- Xyce waveform diagnostics: 19 passing cases cover full-real 6T/10T read/write
  sequences, mux paths, near/middle/far positions, subdivision, equivalent
  mode 4, seeded per-device samples, SS/SF at 125 C / 0.9 V, and next-row read/write
  hazards. Two additional 6T SS/SF stress cases fail at 2.5 ns; both pass at
  5 ns. Failed stress evidence is retained. These small-array checks use
  illustrative metal parameters and do not establish extracted-array yield.
- Compatibility: against a detached V2.0.6 worktree (`029626b`), eight 4x4
  6T/10T read/write star decks with RC off or explicit 100 ohm are byte-for-byte
  identical. Four decks using the previous default differ only at ten replica
  resistors corrected from 10 to 100 ohm.

## V2.0.6 — 2026-09-08 — compiler integration and documentation organization

This release reorganizes the compiler and documentation. Driver and timing
qualification remain in progress; the inherited rule identity and qualification
artifact format remain V2.0.5. See the [working proposal](DRIVER_SIZING_PROPOSAL.md).

- Integrated the local mismatch runner, netlist specialization, and sampling
  into `sram_compiler/per_device_mc/`, retaining per-device mismatch as the default.
  The CLI is now `python -m sram_compiler.per_device_mc.run`.
- Moved working proposals and release history into `docs/`, and placed compiler,
  equivalent-model, and optimization documentation beside their code with README
  links. The original design snapshot and supplied CSV evidence are preserved.
- Updated imports, repository-relative data lookup, qualification source
  fingerprints, directory trees, and current-version documentation. The compiler
  guide now describes a single per-device run as one random local sample.
- Moved nine development experiment/qualification helpers into ignored
  `dev/sizing/`, and their tests into `dev/tests/`. Reusable compiler regression
  tests live in tracked top-level `tests/`. Runtime simulation testbenches remain
  part of the compiler. `.gitignore` excludes future local development scripts.
- Runtime qualified-table lookup uses a tracked scoring-source hash manifest
  instead of opening development scripts. Local qualification tools check that
  manifest before emitting evidence; stale source identities remain rejected.
- Validation: 34 compiler tests, 20 local development tests, and 6 offline
  optimizer tests pass; module and
  direct-script CLI read/write generation reproduce all eight baseline deck,
  model, audit, and summary files byte for byte, including execution outside the
  repository. Nominal/shared CLI generation, Python 3.9/3.11 compilation,
  qualification scheduling dry-run, local Markdown links, and `git diff --check`
  pass. The tracked-file export passes compiler/optimizer tests and CLI generation
  with `dev/` absent. No new Xyce simulation or electrical qualification was run
  for this move.
- 2026-09-09 utility follow-up: replaced root `utils.py` with a `utils/` package
  for measurements, waveform loading, plots, area estimates, and SPICE models.
  Existing `from utils import ...` imports and optimizer plot imports remain valid.
  Reusable comparison plots from `plot_data.py` now live in `utils.plotting`;
  their hardcoded experiment is retained locally in ignored `dev/plot_data_demo.py`.
- Comparison plots create their output directory, save under `outputs/plots/`
  by default, scope their style changes, and close figures. Display is optional
  with `show=True`. Fixed an existing transient/DC splitting compatibility bug:
  DataFrame slices preserve the signal labels that NumPy splitting discarded.
- Four tracked utility regressions cover failed measurements, indexed and
  index-free transient/DC waveforms, headless comparison plots, and optimizer
  import compatibility. Reusable `tests/` remain versioned alongside the code.
- Utility follow-up validation: 38 compiler/utility tests, 20 local development
  tests, and 6 offline optimizer tests pass; the four utility tests also pass
  under Python 3.9. A tracked-file export passes tests and direct CLI generation
  without `dev/`. All eight read/write artifacts still match the original
  baseline byte for byte; SPICE model output and 6T/10T area estimates are unchanged.
- 2026-09-09 entrance follow-up: `main_sram.py` is the main entrance again. It
  loads the YAML files in memory through `load_config()`, applies its `ARRAY`,
  `CORNER` and `CELL_6T` settings to the loaded configuration, and no longer
  rewrites `global.yaml` or `sram_6t_cell.yaml`. It passes
  `variation_mode='per-device'`, `mc_seed=20260711` and `real_cell_mode=0`
  explicitly; before, `mc=True` with the equivalent-cell cross and no seed gave
  one unseeded local sample over the retained devices only. The previous script
  could not start at all: it imported `ruamel.yaml`, which neither the conda
  environment nor the workspace interpreter installs. `VARIATION_MODE` selects
  `nominal`, `shared` or `custom`; custom tables pass through `get_custom_vars()`
  and the sample count through `resolve_mc_runs()`, so a table/sample mismatch
  fails before generation. The 10T area estimate now uses the 10T cell widths.
- Removed dead root files: the empty `conda` and `refreshenv`, and
  `main_estimation.py`, which imported a `sram_yield_estimation` package and a
  positional testbench API that no longer exist. `demo_run_a_testbench.py`
  remains the yield-estimation entrance (explicit custom tables). `main_opt.py`
  stays: its `config_sram.yaml` resolves inside `size_optimization/`.
- Documentation: the root, compiler, per-device, sizing, tests, yield-estimation
  and equivalent-model guides describe the entrance and its defaults. The
  compiler guide's `use_equivalent=True` switch, which the testbench never had,
  is replaced by `real_cell_mode`; its output-file list shows the per-device
  model cards, audit and variation summary.
- Entrance validation: `tests/test_main_entrance.py` (two tests) checks the
  seeded per-device full-array default and that script settings reach the deck
  while the tracked YAML digests stay unchanged. 40 compiler/utility tests pass
  under Python 3.11; the entrance and packaging tests also pass under 3.9.
  Relative Markdown links resolve and `git diff --check` passes.
  The default `python main_sram.py` run (16x16 6T, TT, write on cell 15/15,
  RC, one seeded local sample, full array: 3,479 unique per-device model cards)
  completed with Xyce 7.4 in 5 min 13 s single-threaded: all 13 write measures
  valid, write access 287.5 ps (`TWRITE_TOTAL`), 169.4 uW, minimum-clock
  estimate 1.01 ns; the target cell's Q/QB flip is visible in the `.prn`. The
  tracked YAML files were unchanged after the run. Nominal, shared and custom
  modes were exercised only through the testbench-level regression tests.

## V2.0.5 — 2026-09-08 — full local mismatch and driver re-evaluation

Electrical qualification remained in progress at this checkpoint. The full
working plan continues in `docs/DRIVER_SIZING_PROPOSAL.md`.

- The MC testbench defaults to independent per-MOS `vth0`, `u0`, and `voff`
  mismatch at a fixed global corner. Nominal and shared-card variation are
  explicit alternatives; the per-device CLI defaults to full transistor arrays.
- Canonical specialization preserves connectivity, widths, sweep expressions
  and `NF`, with immutable model artifacts and per-device CSV audits. Combined
  local sampling and legacy `.STEP` sweeps fail explicitly after a Xyce check
  showed that only the first geometry was executed.
- All five corners now have local read and write checks. Dedicated write-tail
  ensembles, fast-corner hazards, separate diagnostic screening, execution
  provenance and stricter qualification/report gates are included.
- The finalized 8x4 pilot passes all 12 nominal calibration decks and 120 local
  waveform samples for 6T/10T and mux on/off. Coefficients remain provisional.
- The three-sample representative-array screen (8x4, 16x16, 64x64, 256x8,
  16x256; both cells; mux on/off; 16x16 with RC) completed: 759 local samples
  scored, 723 pass every check, 36 fail only the wordline-phase budget on the
  16x16 explicit-RC reads (the reads themselves are functional; the wordline
  rule has no RC term). 28 four-rank cases and the 10T 64x64 / 16x256
  calibrations timed out at 1,800 s on a shared host, and three materialized
  MPI samples failed the DC operating point; these are unscored, not failed.
  See the Stage C outcome in `docs/DRIVER_SIZING_PROPOSAL.md`.
- RC symmetry between the array and the replica column: with `w_rc` the
  fixed-mode replica wordline driver (AND2) now carries the same two RC
  segments on its inputs and RWL output as the real wordline drivers, and the
  replica bitline reaches the timing block through the same two segments a
  real bitline sees at its sense-amplifier input. Cells, dummy loads,
  precharge and write-driver stubs were already identical on both sides; the
  equivalent-cell modes aggregate the same stubs. Non-RC decks are unchanged.
  Measured effect (8x4 6T, TT, fixed mode, RC read): the replica wordline led
  the real wordline by 36 ps before and 13 ps after (rules_only: 1 ps), so
  the read access rises from 392 to 434 ps; the write access is unchanged.
  The wordline model with RC is the driver's two output segments plus one
  stub per cell pin (a star, no series segments between columns).
- Added `sizing/wordline_model.py`, a reproducible star-versus-distributed
  wordline check (evidence in `docs/qualification/V2.0.5_wordline_model.json`):
  at 256 columns a 1 ohm / 0.1 fF-per-column line delays the last cell by
  36 ps and widens its slew by 78 ps relative to the compiler's star model; at
  16 columns the difference is 1 ps. The inherited star model is kept for the
  wordline sizing rule by decision. `local_review --rc-only` reruns only the
  explicit-RC 16x16 architectures.
- Review fixes: the optimizer objective (`exp_utils.evaluate_sram`) requests
  the nominal corner explicitly instead of inheriting one unseeded per-device
  sample; `local_review` records calibration execution errors instead of
  labelling them waveform failures. Twenty fixed-mode decks are byte-identical
  to the previous release; default-mode 8x4 Xyce runs and the per-device CLI
  pass.
- Large-array runs use MPI cores with saved numeric local LHS model cards,
  bypassing a reproduced Xyce 7.4 MPI random-expression initialization crash.
  Four-core read/write checks pass; a 64x64 full-device waveform check passes
  in 854 seconds. MPI/serial timing and power measures match for an identical
  small-array sample. Worker/rank budgets and timeout cleanup are explicit.

## V2.0.4 — 2026-09-07 — baseline driver sizing and measured timing

Qualification is in progress. The full working proposal and completed/open
checklist are in `docs/DRIVER_SIZING_PROPOSAL.md`; its original text and supplied
characterisation CSV are preserved.

### Changes

- Added immutable baseline driver sizing with separate write-input/output
  scales, load-based precharge and wordline rules, configurable replica K/N,
  decoder output scaling, and exact baseline/physical-context table lookup.
  `fixed` remains the default; `rules_only` and unmatched `auto` results are
  explicitly unverified.
- Matched real and replica wordline drivers and gate loads, froze replica
  devices across cell candidates, and included disabled write-stack loading
  on real and replica bitlines in canonical read decks.
- Corrected control-buffer loading and tapering. Wide inverter gates use
  BSIM4 `NF` with at most 2 um per finger, preserving total width and the PDK
  gate-resistance model. This closes measured wide-array control-edge failures.
- Included explicit RC enable-pin loads, removed a floating dummy-cell RC
  branch, and delayed RC precharge until the matched physical wordline is low.
- Added measured clock derivation from SS read and SS/SF write phases, with
  25% margin and upward 50 ps quantisation. The period remains frozen during
  candidate and variation evaluation.
- Added resumable Xyce qualification, waveform/phase checks, shared and local
  mismatch cases, eight-cycle read/write sequences, SA offset ensembles, and
  evidence-bound report/table export. Failed or incomplete evidence cannot
  produce qualified table records.
- Integrated baseline reuse and resolved-width area estimates into the
  optimizer adapter, corrected its actual array geometry, and report original
  access-limit violations. Per-device MC records sizing provenance; both
  simulation entry paths select Xyce explicitly.
- Added `AGENTS.md`, sizing documentation, and focused regression coverage.

### Qualification and remaining work

- The full campaign covers 112 physical configurations: both cell types,
  27 historical sizes and valid mux choices, RC sensitivity, and 32x1 optimizer
  cases. It schedules 1,786 decks and 6,538 waveform samples. Completion and
  table promotion are pending; see the proposal for current status.
- Folded-driver 16x256 and 16x512 TT read pilots pass all waveform checks.
  Independent gate-charge/fingering evidence is in `docs/qualification/`.
- Four 100-copy SA/latch offset ensembles pass at SS and cold FF, including
  mid-common-mode and precharged-bitline input profiles. Full local sensing
  qualification awaits the replica/cell ensembles.
- Original 200 ps read / 100 ps write constraints remain separate from phase
  qualification; default K=1/N=9 does not claim compliance with the read limit.
  Peripheral sweep ranges, equivalent-cell approximations, and the obsolete
  rare-event `main_estimation.py` backend/API migration remain open.

## V2.0.3 — 2026-09-06 — changelog compacted, automatic timing configuration proposed

Scope: documentation and characterisation only. No compiler, testbench or
algorithm code was changed.

### Changes

- **`docs/CHANGELOG.md` compacted:** the V2.0.1 and V2.0.2 entries shrink from
  900 to 300 lines and keep their fixes, open items, code changes and observations; the
  evidence is condensed to representative sizes and one corner table. The
  full tables are in `git show c3f6f44:CHANGELOG.md`.
- **`docs/TIMING_AUTOCONFIG.md`** (new): proposal for setting the SRAM timing
  automatically for every array size. Findings it rests on:
  - the only free timing knob of this architecture is the clock period (and
    duty): wordline enable and write enable are the clock-low phase,
    precharge the clock-high phase, sense enable is replica-timed and the
    output / data / address latch enables are derived from `s_en`, `w_en`
    and `wl_en`;
  - the phases the flow measures (`TCLK_WLEN + access` and
    `max(TRESTORE, TCLK_DEC)`) follow `a + b*rows + c*log2(cols)` for reads
    and `a + b*rows + c*cols + d*log2(rows)` for the write restore within
    5-15 ps rms / < 40 ps worst over the 27 sizes of the V2.0.2 sweep;
  - proposed rule `T = 2 * max(low_wc, high_wc) * (1 + margin)`, phases
    derated to the worst-case PVT with one measured factor per phase,
    default margin 25 %, quantised to 50 ps, anchored by a table of
    characterised sizes with a two-deck self-calibration for new sizes, and
    frozen per array size as the spec for optimisation and yield analysis.
  Not implemented; sections 5-8 of the document list the integration points,
  the validation plan, the effort and the decisions needed.

### Evidence (new, nominal devices unless stated, 6T, mux off, 10 ns clock)

- **Worst-case PVT characterisation** (scratch harness of V2.0.2 with a new
  `--vdd` switch): read and write decks at SS, SF, FS and FF x 125 C x
  1.0 V, SS and SF x 125 C x 0.9 V and TT / 25 C / 0.9 V on 8x4 and 16x16
  (6T and 10T), 64x16, 256x8, 16x256, 64x64 (SS, 1.0 V) and 512x4 (read);
  75 decks, all completed. Relative to TT / 25 C / 1.0 V every control phase and the read
  access are 2.12-2.31x slower at SS / 125 C / 0.9 V, with the same factor
  on every size run and both cells (the buffer taper and the bitline terms
  scale together); the supply step to 0.9 V costs 1.13x alone. The write
  access is worst at SF: 3.8x at 8x4 (0.5x write driver), 2.4x at 16x16.
  Every deck passes the waveform checks except the 256x8 and 64x16 reads at
  FF and FS / 125 C, where the un-selected bitline
  leaks from 1.00 to 0.69-0.73 V during the 5 ns wordline phase of the 10 ns
  clock (it is at 0.99 V when the amplifier fires; the read is correct).
  The same 256x8 FF / 125 C deck re-run with the period the proposal would
  assign (2.6 ns) keeps the bitline above 0.97 V and passes: the droop is a
  property of the oversized wordline phase, not of the array.
- **Period sweeps at the worst case** confirm the minimum-period formula
  within one 100 ps step: 8x4 read at SS / 125 C / 0.9 V passes at 1.9 ns,
  fails at 1.8 (measured `2 * low` = 1.81 ns); 8x4 write at SF / 125 C /
  0.9 V passes at 1.5, fails at 1.4 (1.44 ns); 16x16 read at SS / 125 C /
  0.9 V passes at 2.1, functionally at the limit at 2.0 (1.95 ns: `OUT`
  crosses VDD/2 10 ps after the clock edge).
- **Seeded Monte Carlo at the worst case** (`vth_std = 0.05`, 5 samples,
  seed 2026): 8x4 and 16x16 reads at the proposed periods (2.35 / 2.5 ns,
  25 % margin) pass every sample with 2.7 % sigma on the limiting phase and
  > 210 ps slack; the 8x4 write at SS / 125 C / 0.9 V and the 16x16 write at
  SF / 125 C / 0.9 V pass every sample (16 % and 6 % sigma on the write
  access; 10 % and 4 % on the `low` phase).

### Left open

- **8-row write-ability at SF / 125 C / 0.9 V.** 2 of 5 Monte Carlo samples
  of the 8x4 write cannot write at all with the 10 ns clock (BLB stays at
  0.26 V, the cell keeps its data) and a third needs 2.5 ns: the 0.5x write
  driver that `WriteDriverFactory.width_scale` gives to arrays with <= 8
  rows has no margin against a strong-PMOS / weak-NMOS sample at that
  corner. Nominal decks pass (write access 499 ps). Not changed in this
  release; a yield analysis at that corner will report it.
- **Automatic timing is proposed, not implemented**; the 10 ns default period
  is unchanged.
- **Not characterised at the worst case:** 128x128 and 256x64, the column
  mux, 10T beyond 16x16, and the equivalent-circuit / `w_rc` variants; the
  phase model is fitted at TT / 25 C and derated, it has not been re-fitted
  on worst-case data (the table in `docs/TIMING_AUTOCONFIG.md` section 3.4 is the
  anchor for the seven characterised sizes).
- **Testbench prerequisites for short periods** (unchanged, from V2.0.2):
  the PSTC window overlaps the start-up precharge for `t_period < 5 ns`;
  the scratch harness scores `OUT` inside the wordline phase, which is
  50 ps stricter than the functional limit at 16x16.
- Evidence of this entry: `TIMING_AUTOCONFIG_data.csv` (615 rows: the
  V2.0.2 sweeps with their phase measures, the worst-case decks and the
  validation decks).

## V2.0.2 — 2026-09-05 — periphery fan-out, address hold, precharge control, timing configurations

Scope: the open items of V2.0.1 (address-path hold hazard, cycle-time
dependence of the floating bitlines, `s_en` buffer sizing) plus a review of
every control buffer of the timing block and of the wordline driver against
its actual fan-out. Validated in Xyce 7.4 with automatic waveform scoring over
array sizes 1x1 to 512x4 / 16x512, clock periods 0.6-100 ns, the five process
corners at -40 to 125 C, and seeded Monte Carlo; "before" numbers come from a
detached worktree of V2.0.1. Optimisation and yield-estimation algorithms
were not touched.

### Fixes

- **Address-path hold hazard confirmed and closed.** With a changed address
  the new row's wordline reached 0.41-0.50 V at 256 rows (neighbouring cell
  Q dipped to 0.86 V) and full VDD at 512 rows, where the cell in the new row
  was overwritten (read deck: stored 1 -> 0; write deck: 0 -> 1). A
  transparent-low hold latch on the address register output (enabled by
  `wl_en_bar`, same scheme as the V2.0.1 write-data latch) keeps the decoder
  input constant while a wordline is on; the new decoder output now rises at
  least five gate delays after the old wordline is off and no second wordline
  (< 10 mV) is seen at any size. Flipping the LSB never glitched (largest
  decoder fan-out, slowest path); hazard tests must flip a *middle* bit.
- **Control buffers sized for their fan-out.** `wl_en` (600 ps fall at 512
  rows), the address register output (650 ps edge at 512 rows), `s_en`
  (190-230 ps edge and a 0.2-0.3 V precharge-coupling bump at >= 64
  columns), `w_en` (130-180 ps edge from 64x16 up) and `PRE` (fan-out ~49,
  only reached 0.05-0.08 V on the largest arrays) are driven by geometrically
  tapered buffers with a fan-out of ~8 per stage (`TaperedBuffer`: 2 stages
  up to a scale of 16, 4 above); the wordline driver's NAND2 scales with the
  square root of its inverter scale. Every buffered control edge is now
  20-40 ps (10-90 %) independent of the array size.
- **Sense amplifier isolated during writes.** Its input pass gates were on
  whenever `s_en` was low, so the cross-coupled PMOS pair was a bitline keeper
  during writes and a write only worked while `w_en` rose before the wordline
  (60 ps margin at 2x128 in V2.0.1; with the faster wordline path of this
  release the 2x128 write deadlocked at BL 0.27 V / BLB 0.9 V). `SENSEAMP`
  has a new `ISO` pin driven by `sa_iso = s_en | w_en` from TIME.
- **Precharge for the whole clock-high phase.** The ~300 ps self-timed pulse
  left the bitlines floating; with the 10 ns clock the replica bitline had
  drooped to 0.74 V at FF / 125 C and to 0.76 V at TT / 25 C with a 100 ns
  clock. `PRE = NAND3(clk_buf, cs, wl_en_bar)` holds the bitlines at VDD
  until 40-70 ps before the wordline rises; they are at 1.000 V at every
  access for 0.6-100 ns periods and at every corner.
- **Honest minimum-period estimate.** The printed `CLK(min)` was
  `2 * (access delay + 0.1 ns)` and claimed 0.48-0.80 ns for an 8x4 array
  whose read deck fails below 0.9 ns. Three new measures (`TCLK_WLEN`: clock
  -> `wl_en`; `TCLK_DEC`: capture edge -> decoder output; `TRESTORE`: end of
  access -> bitline back at 0.9 VDD) give
  `T_min = 2 * max(TCLK_WLEN + access, TRESTORE, TCLK_DEC) * 1.1`
  (`_print_min_period()`), which matches the period sweeps within one step.
- **Testbench stimulus for the address path.** `next_row=<row>` makes the
  register capture another row at the edge that ends the access and prints
  that row's wordline and cell (the address was constant in every V2.0.1
  deck, which is why the hazard was never exercised).
- **PSTC caveat.** A warning is printed for `t_period < 5 ns`, where the
  quiescent window overlaps the start-up precharge.

### Left open

- **128x128 and 256x64** were run only for 6T without mux with a 4 ns clock
  (read 475 / 568 ps, write 155 / 149 ps, ~3 h per deck, all checks pass);
  the 10T / mux / read&write decks of those sizes and the two 10T 16x512
  `read&write` decks (> 10 h) were not run.
- **Energy cost of the fixes.** 6T read PAVG at 100 MHz: 8x4 +5 %, 16x16
  +11 %, 32x32 +17 %, 64x64 +14 % (largest term: the wordline-driver NAND2
  taper, 115 fF instead of 29 fF on the `wl_en` / decoder nets at 64x64;
  PSTC +20-30 % because the bitline leakage is now supplied through the
  precharge devices). Writes are cheaper on every array >= 32x32 (-12..-24 %)
  because the driver no longer fights the sense-amplifier keeper. The buffers
  follow one fan-out rule and were not power-tuned.
- **Delay reference at >= 256 rows.** `TREAD_TOTAL` / `TWRITE_TOTAL` are
  measured from the `wl_en` crossing, which the re-sized buffer moves 70-150
  ps earlier at 256-512 rows, so those tabulated delays grow (512x4 6T read
  709 -> 739 ps) while clock-to-output shrinks (1010 -> 890 ps). The
  clock-referenced value is `TCLK_WLEN + TREAD_TOTAL`.
- **Design choices verified again and kept:** replica-timed full-swing
  sensing (read delay ~300 ps up to 32 rows); `w_rc=True` default of
  `main_sram.py`; `read` reads a stored 0; the hazard / coupling checks live
  only in the scratch harness, not in the flow's `.MEASURE` set.
- **Out of scope:** optimisation and yield-estimation algorithms, SNM beyond
  the V2.0.1 sanity run, `sweep_*` modes.

### Code changes

- `time_generate.py`: `ADDR_DFF` -> `D_LATCH_ADDR` (EN = `wl_en_bar`) ->
  `TaperedBuffer` `ABUF` per address bit; `wl_pdrive` scales both stages with
  `ceil(rows * nand_scale / 32)`; `s_en` drives only footers and output
  latch (`SEN_BUF` above 32 unit loads); new `sa_iso = NOR2(s_en, w_en)` +
  inverter (+ `ISO_BUF`); `w_en` = `AND2_WEN` (+ `WEN_BUF` above 32 unit
  loads); `PRE_UNBUF = NAND3(clk_buf, cs, wl_en_bar)` + `PRE_BUF`. New
  helpers `TaperedBuffer`, `D_latch_addr`, `PNOR2`; `TIME` / `TIMEFactory`
  take `num_sa`, `wl_load`, `pre_load`, `wen_load` (defaults reproduce the
  base YAML sizes) and export `sa_iso`.
- `wordline_driver.py`: static `WordlineDriverFactory.inv_scale / nand_scale`
  (`nand_scale = sqrt(inv_scale)`, also emitted in sweep mode);
  `PrechargeFactory.width_scale`, `WriteDriverFactory.width_scale`.
- `mux_and_sa.py`: `SENSEAMP` `ISO` pin.
- `sram_6t_core_MC_testbench.py`: `TCLK_WLEN`, `TCLK_DEC`, `TRESTORE`
  measures; `_print_min_period()` replaces the `1/2CLK` print (mean + one
  standard deviation per term for `mc_runs > 1`); PSTC warning.
- `sram_6t_core_testbench.py`: `next_row`; `create_time_circuit()` passes the
  fan-out information; `create_write_periphery()` sizes the `w_en_bar`
  inverter with the column count (`_wenb_scale()`).
- `sram_compiler/README.md` sections 10, 11, 13.1, 14.5; `sram_compiler/CIRCUIT_REVIEW.md`
  Part III (defects D12-D20, sizing tables).

### Observations (verified, not changed)

- At TT / 125 C every control-path delay is 1.64-1.74x its 25 C value (16x16
  6T read 308 -> 536 ps, `TRESTORE` 258 -> 437 ps); TT / -40 C is the
  fastest condition in these models (8x4 6T read 203 ps vs 291 at TT / 25 C);
  SF is the slowest write corner (8x4 6T write 174 ps vs 131). The 10 ns
  clock leaves > 8 ns of margin everywhere; the estimated minimum period at
  TT / 125 C is ~1.7 ns for 16x16.
- With the isolation pin the amplifier's pass gates open 50-70 ps after the
  footer fires, so it regenerates while still connected to the full-swing
  bitlines; reads are 8-25 ps faster than in V2.0.1.
- `read&write` toggles `we` every cycle; `s_en` shows no glitch (< 1 mV) at
  the write-to-read transitions because `gated_clk_bar` falls before `we_bar`
  rises.

### Evidence (condensed)

Nominal, all cells real, `w_rc=False`, TT, 25 C, 10 ns clock, target cell =
last row / last column. Read delay = `wl_en` rise -> `OUT`; write delay =
`wl_en` rise -> Q at 90 %; PAVG at 100 MHz. V2.0.1 values in parentheses.

**Size sweep:** 292 of 294 configurations (27 sizes x {6T, 10T} x {mux off,
on} x {read, write, read&write}) completed and pass every waveform check;
only the two 10T 16x512 `read&write` decks are missing (10 h job limit).
Every `read&write` deck shows the correct 40 ns `OUT` period. Mux on: reads
0-29 ps faster, writes 1 ps faster to 19 ps slower than the values below.

| array (mux off) | 6T read [ps] | 6T write [ps] | 10T read [ps] | 10T write [ps] | 6T PAVG read / write [uW] |
|---|---|---|---|---|---|
| 1x1 | 283 (286) | 119 (131) | 286 (289) | 119 (133) | 20.9 / 22.6 |
| 4x4 | 287 (298) | 126 (133) | 291 (301) | 127 (138) | 26.8 / 36.8 |
| 8x4 | 291 (301) | 131 (138) | 295 (305) | 131 (142) | 29.3 / 39.6 |
| 16x16 | 308 (337) | 94 (98) | 314 (344) | 107 (111) | 50.0 / 93.7 |
| 32x32 | 332 (386) | 100 (137) | 344 (396) | 112 (148) | 94.6 / 201.4 |
| 64x16 | 358 (382) | 93 (114) | 378 (401) | 105 (125) | 89.9 / 155.0 |
| 64x64 | 382 (492) | 108 (162) | 408 (505) | 120 (173) | 231.7 / 504.8 |
| 2x128 | 322 (458) | 176 (191) | 326 (463) | 169 (202) | 143.6 / 560.3 |
| 128x32 | 424 (468) | 104 (135) | 460 (510) | 116 (146) | 218.6 / 416.7 |
| 16x256 | 354 (535) | 133 (261) | 361 (542) | 146 (273) | 369.4 / 1252.0 |
| 256x8 | 524 (526) | 96 (68) | 592 (579) | 105 (78) | 149.4 / 223.3 |
| 8x512 | 350 (640) | 202 (509) | 355 (642) | 197 (512) | 590.5 / 3277.7 |
| 16x512 | 360 (650) | 158 (460) | 369 (656) | 170 (472) | 729.8 / 3378.0 |
| 512x4 | 739 (709) | 95 (70) | 869 (791) | 105 (80) | 171.3 / 234.9 |
| 128x128 (4 ns clock) | 475 | 155 | - | - | 1678 / 3988 |
| 256x64 (4 ns clock) | 568 | 149 | - | - | 1636 / 3464 |

**Address change (`next_row`, middle bit flipped):** V2.0.1 circuit: next
row's wordline 0.41-0.50 V during the hold at 256x8 (read, write, 10T write)
and 1.00-1.01 V at 512x4 with the neighbouring cell overwritten; V2.0.2:
< 10 mV and the cell keeps its data at 8x4, 64x16, 128x32, 256x8 and 512x4,
read and write. LSB flips never glitched on either version.

**Clock period sweep (6T, mux off):** 8x4 read passes at 0.9-100 ns and fails
at 0.8 ns (estimate 0.90 ns); 16x16 read passes at 0.9 ns and fails at 0.8
(estimate 0.97); 8x4 / 16x16 write pass down to 0.6 ns (estimates 0.55);
64x16 read passes at 2 ns (estimate 1.08); 10T 8x4 read passes at 1.0, fails
at 0.8 (estimate 0.91). Delays are unchanged across periods; the bitlines are
at 1.000 V at every access for all periods.

**Process corners and temperature (6T, 10 ns clock, all checks pass, bitlines
1.000 V at every access):**

| condition | 8x4 read [ps] | 8x4 write [ps] | 16x16 read [ps] | 16x16 write [ps] | 16x16 PAVG read / PSTC [uW] |
|---|---|---|---|---|---|
| TT -40 C | 203 | 83 | 214 | 62 | 48.0 / 1.5 |
| SS -40 C | 218 | 86 | 231 | 66 | 46.2 / 0.8 |
| FF 25 C | 265 | 123 | 281 | 87 | 55.2 / 6.9 |
| FS 25 C | 290 | 115 | 306 | 88 | 50.6 / 4.1 |
| TT 25 C | 291 | 131 | 308 | 94 | 50.0 / 3.3 |
| SF 25 C | 295 | 174 | 312 | 104 | 51.1 / 4.5 |
| SS 25 C | 321 | 140 | 341 | 104 | 47.1 / 1.7 |
| TT 85 C | 411 | 206 | 435 | 139 | 54.8 / 8.0 |
| FF 125 C | 451 | 244 | 478 | 156 | 74.0 / 26.0 |
| TT 125 C | 507 | 266 | 536 | 174 | 60.7 / 13.6 |

10T reads are 2-14 ps slower than the 6T ones at every condition; 10T writes
are 8-23 ps slower at 16x16 and between 22 ps faster and 5 ps slower at 8x4.

**Monte Carlo (Xyce `.SAMPLING`, `vth_std = 0.05`, seed 2026):** 5-sample
read / write decks at 8x4, 16x16, 32x8 and 64x16 (both cells, mux off / on)
and 3-sample `read&write` decks at 8x4 and 16x16 all pass. Read delay
standard deviation 3.4-5.0 ps (1.1-1.6 %); write 1.6-3.9 ps at >= 16 rows,
5-12 ps at 8x4 (up to 7 % with mux). The 64x16 address-change decks pass
for all samples.

## V2.0.1 — 2026-09-05 — transient / Monte Carlo circuit review

Scope: the SRAM compiler circuits (6T and 10T cores, replica column, timing
generator, decoder, wordline driver, precharge, column mux, sense amplifier,
write driver, output latch), the transient testbenches (`read`, `write`,
`read&write`), their measurements, and the Xyce simulation / result-parsing
flow. Every change was validated by running the generated netlists in Xyce
7.4 and scoring the waveforms automatically for 29 array sizes from 1x1 to
512x4 and 16x512, both cells, mux on and off, all three operations, nominal
and seeded Monte Carlo (283 completed decks, all pass). Optimisation and
yield-estimation algorithms were not reviewed.

### Fixes

- **Write pulse too short.** `w_en` was cut by the replica bitline (~250 ps),
  a cell-strength path, while the write path (row-scaled write driver, through
  the column mux) is weaker: seeded 5 % sigma Monte Carlo samples left the
  bitline at 0.3-0.4 V and the cell kept its old data. `w_en = gated_clk_bar
  & we` now spans the wordline phase (new `AND2_WEN` gate with its own
  subcircuit name); the hard-coded 16x512 `WenDelayChain` hack is removed.
- **Hold hazard introduced by that fix, caught by the sweep.** At the clock
  edge that ends a write the data register updates 100-150 ps before the
  drivers release, so the next cycle's data was briefly written; at 64 rows
  this flipped the freshly written cell (`read&write` 64x16 failed). A
  per-column transparent-low write-data hold latch (`D_LATCH`, EN =
  `w_en_bar`) fixes it.
- **Write testbench topology.** The `write` deck had no bitline precharge and
  no sense-amp / mux load (bitlines started from the artificial `.IC` state),
  so the stand-alone write delay was 30-40 % optimistic against the same write
  inside `read&write`. All transient decks now carry the full column
  periphery; a write cycle is precharge -> write -> precharge; `TWDRV` is
  measured on the driven bitline (`w_en` rise -> BLB at VDD/2).
- **Nominal runs were random samples.** Every deck emitted `.SAMPLING` with a
  random seed, so identical calls returned different delays and occasionally
  failed. `mc_runs=1` is now deterministic (no `.SAMPLING`, every `AGAUSS`
  at its mean); `mc_seed` makes Monte Carlo sweeps reproducible; the Xyce
  console log is kept per run as `<netlist>.log`.
- **Energy window** measured the start-up charging of the bitlines from 0 V
  (half of the "read energy" on 8x4) instead of a steady-state cycle; it is
  now one full clock period starting at the first access (`1 ns + 0.7 T` ..
  `1 ns + 1.7 T`). `read&write` averages over one 4-cycle pattern, its
  transient runs to `1 ns + 8.5 T`, and it now produces PSTC / PDYN too.
- **Measurement details.** `TS_EN` was corrupted by a precharge-coupling bump
  on `s_en` (now measured from the access phase); the `w_en` buffer was not
  scaled with the row-dependent write-driver size (release lagged the
  wordline by 40-285 ps on 64-256-row arrays); the 10T core ignored the
  testbench RC parameters; PySpice keeps one subcircuit definition per name
  and scope, so a gate class instantiated twice with different sizes silently
  kept the last one (avoided with dedicated class names).
- **Xyce Newton stall on some 512-row decks** (residual 1e-12 A at every step
  size, not a circuit fault): the flow retries once with a 20 ps maximum time
  step, which keeps results of converging decks within 0.5 %; `t_max_step`
  and `xyce_options` are exposed on `Sram6TCoreMcTestbench`.
- **Static-review fixes carried into this release** (details in
  `sram_compiler/CIRCUIT_REVIEW.md` Part I): column-mux port mismatch that aborted every
  muxed read; free-running `SEL` pulse; wrong output-latch index and floating
  latch input on writes; replica column driven by the real wordlines (now one
  active replica cell, dummies tied to VSS); CS start-up clamp fighting the
  flip-flop; negative read delay and negative dynamic power from mis-placed
  measure thresholds and windows (read delay is `TREAD_TOTAL`, `wl_en` ->
  output latch; write delay `TWRITE_TOTAL`, `wl_en` -> Q at 90 %; PSTC in a
  quiescent window `1 ns + [0.4, 0.65] T`); write delay over-reported 2.2x by
  summing overlapping segments; `FAILED` measures silently becoming 0.0
  (now raise); `.prn` / SNM parsing that depended on `.PRINT` ordering;
  write-SNM taken as a global maximum; equivalent-circuit caps only inserted
  with `w_rc`; stimulus sources at a literal 1.0 V instead of `vdd`;
  `choose_columnmux` with `num_cols % mux_in != 0` rejected; `sweep_senseamp`
  no longer defaults to `True`; `python-graphviz` -> `graphviz`; duplicated
  `config.py`; µW label; `main_sram.py` paths.

### Left open

- **Array sizes not finished at release time:** 128x128 and 256x64 (~11 h per
  read deck at 10 ns) and the `read&write` decks of 8x512, 16x512, 16x256 and
  10T 512x4 (finished in V2.0.2).
- **Address-path hold hazard** (pre-existing, not exercised because the
  testbenches kept the address constant): fixed in V2.0.2.
- **Design choices verified and left as they are:** sensing waits for a fully
  discharged replica bitline plus a 9-stage delay chain, so the read delay is
  ~300 ps for every size up to 32 rows (the target bitline is at ~0.02 V when
  the amplifier fires; `vswing` = 250 mV is reached after 10-35 ps); the
  precharge was a ~300 ps self-timed pulse after which the bitlines floated
  and drooped to ~0.93 V (changed in V2.0.2); the `s_en` buffer kept the
  columns/64 scaling (changed in V2.0.2); the `w_rc=True` default of
  `main_sram.py` puts 1 fF on every cell's Q/QB and triples the write delay
  (16x16 6T: 98 -> 346 ps); the fixed 10 ns clock leaves > 4 ns of margin for
  every size tested; `read` always reads a stored 0.
- **Out of scope:** optimisation and yield-estimation algorithms, SNM beyond
  a sanity run (6T hold / read / write 0.325 / 0.182 / 0.365 V, 10T 0.485 /
  0.290 / 0.419 V), `sweep_*` modes.

### Evidence (condensed)

- **Size sweep:** 283 of 318 planned decks completed, all pass; the per-size
  delays are the values in parentheses of the V2.0.2 table above.
- **Monte Carlo (seed 2026, 5 samples):** 8x4, 16x16 and 32x8, both cells,
  mux off / on, read and write, all samples pass; read standard deviation
  3.5-4.4 ps, write 1.6-3.9 ps at 16-32 rows and 7.5-20.6 ps at 8x4.
- **Equivalent-circuit model** (`real_cell_mode=1`, `w_rc=True`,
  `main_sram.py` defaults) tracks the all-real array within 6-12 % on delay
  and 5 % on power at 16x16 with the same RC model; against the no-RC all-real
  runs the delays are 1.5-2.2x (6T read 337 / 386 / 492 ps -> 513 / 695 /
  1051 ps at 16x16 / 32x32 / 64x64; write 98 / 137 / 162 -> 334 / 334 / 297
  ps), most of which is the `w_rc` cell loading.
- **Solver:** a `.SAMPLING` run of the old 10T 64x16 mux netlist aborted with
  "time step too small"; the corrected topology completes (139 ps).
