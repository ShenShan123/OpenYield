# TIME_CONTROL control path: signals, stages and their history

Reference for the `TIME_CONTROL` block in
`sram_compiler/subcircuits/time_generate.py` (V2.1.10; the block was called
`TIME` up to V2.1.6, section 6 maps the old names, sections 7, 8 and 9 the
V2.1.8, V2.1.9 and V2.1.10 changes). The module docstring
lists the signals; this document holds what the code comments used to carry:
why each stage exists, the measurements that led to it, the sizing policy,
and the naming contract a refactor must keep. Release numbers refer to
`CHANGELOG.md`; the D-numbered items are the review findings in
`sram_compiler/CIRCUIT_REVIEW.md`.

## 1. Ports

| Port | Direction | Meaning |
|---|---|---|
| `clk`, `csb`, `web` | in | external clock, chip select (active low), write request (active low) |
| `A{i}` | in | address bits, valid 0.1 T before to 0.1 T after the capture edge |
| `DIN{i}` | in | write data (write operations only) |
| `rbl` | in | replica bitline at its sense-position tap |
| `rwl` | in | far end of the replica wordline (with the replica precharge guard) |
| `pre_far` | in | far end of the PRE line, at the replica precharge (with the precharge-off guard) |
| `clk_buf`, `clk_bar` | out | buffered internal clock and its complement |
| `cs`, `cs_bar`, `we`, `we_bar` | out | registered select and write request (active high on `cs` / `we`) |
| `gated_clk_bar`, `gated_clk_buf` | out | `cs & clk_bar` (the access phase) and `cs & clk_buf` |
| `A_dff{i}` | out | held and buffered address for the decoder |
| `DIN_dff{i}` | out | registered write data for the write-data hold latches |
| `wl_en` | out | wordline enable, one NAND2 input per row driver plus the replica driver; a read wordline ends at the sense trigger (V2.1.8) |
| `rbl_delay`, `rbl_delay_bar` | out | replica bitline through the inverting delay chain |
| `s_en` | out | sense enable (footers and output latch) |
| `w_en` | out | write enable (write drivers; the write-data hold latches until V2.1.9) |
| `sa_iso` | out | sense-amplifier input isolation, `s_en \| w_en` |
| `PRE` | out | precharge, active low |
| `din_hold` | out | hold of the column write-data latches (write operations with the replica guard, last port, V2.1.10) |

The clock has a 50 % duty cycle: the capture edge at `1 ns + 0.2 T + k T`,
the access (falling) edge at `1 ns + 0.7 T + k T`. Clock-high is the
precharge phase of a read cycle and the write slot of a write cycle;
clock-low is the access. Since V2.1.8 no two enables of different roles
overlap (`docs/design/ENABLE_OVERLAP_V2_1_8.md`): the read wordline ends
when the sense enable fires, the precharge waits for the sense and write
enables of the previous access, the write slot for its sense enable, and
the write enable ends with the wordline enable rather than with the
deselect. The sense enable and the output latch overlap by design.

## 2. Stages in signal order

The builders of `TIME_CONTROL` follow this order, which is also the order of
the subcircuit definitions and instances in the netlist.

### 2.1 Address path (`_add_address_path`)

`A{i}` -> `ADDRESS_REGISTER` (`A_reg{i}`) -> `HOLD_LATCH` (`A_lat{i}`,
enable `wl_en_bar`) -> `ADDRESS_BUFFER` (tapered) -> `A_dff{i}`.

The register updates about 100 to 150 ps after the clock edge that ends an
access, while `wl_en` and the old wordline are still falling; before V2.0.2 a
changed address raised the *new* row's wordline for the tail of the old
access and wrote the old bitline data into that row (measured at 64 to 512
rows, D15). The latch is transparent while `wl_en` is low and holds the
address while a wordline is on, so a new decoder output can only rise after
the wordline driver has been disabled. The buffer covers the decoder input
load, which grows with the row count: five gate inputs per last-level 3-to-8
decoder for the low address bits, 320 gates at 512 rows and a 650 ps register
edge before the buffer (D14).

### 2.2 Data register (`_add_data_register`)

`DIN{i}` -> `DATA_REGISTER` -> `DIN_dff{i}`, write operations only. Since
V2.1.1 the register clock crosses the array width on the wordline wire
geometry with one tap per column. The testbench's write-data hold latch
follows this register. Until V2.1.9 its enable was `w_en_bar`: the latch
closed when `w_en` rose, so new data had to be through it first (the select
delay of section 2.11 guaranteed that at an idle -> write edge, V2.1.7, and
the slot-arm latch between two writes, V2.1.9). Since V2.1.10 its enable is
`din_en = !din_hold` (section 2.10): the latch holds while a wordline is open,
and while the drivers are on unless the new cycle is a selected write.

### 2.3 Clock tree (`_add_clock_tree`)

`clk` -> `CLOCK_BUFFER` (four stages, 1 : 3 : 9 : 27 unit widths) ->
`clk_buf` -> unit inverter -> `clk_bar`. The buffer scale is the flip-flop
count relative to a 16x16 write deck (`ControlSizing.clk_drive_scale`).

### 2.4 Select and write registers (`_add_select_and_write_registers`)

`csb` and `web` are registered by `DFF_BUFFER` on the rising `clk_buf` edge.
`DFF_BUFFER` buffers Q (the registered input, so `cs_bar` and `we_bar`) and QB
(`cs` and `we`). The testbench parks the select flip-flop inactive with a
start-up clamp and an `.IC` on `XTIME_CONTROL:Xdff_buf:qint`.

### 2.5 Gated clocks (`_add_gated_clocks`)

`gated_clk_bar = cs & clk_bar` is the raw access request, `gated_clk_buf =
cs & clk_buf` the raw precharge phase (six-unit AND2 output inverters).

### 2.6 Access request (`_add_access_request`)

`access_clk_bar = gated_clk_bar & pre_off_ready` (V2.1.1). The
`PRECHARGE_OFF_GUARD` observes the far end of the physical PRE line with a
half-unit inverter, adds the frozen wire/load settling constant
(`precharge_off_tau`, from `driver_sizing`) and an even delay chain, and only
then lets the request through, so no wordline, write driver or sense amplifier
starts before the precharge has really released. The raw request inhibits the
final gate directly, so the settling delay never extends an access. Since
V2.1.6 the settled `pre_off_ready` is also a port of the guard: the write slot
needs it.

### 2.7 Wordline enable (`_add_wordline_enable`, V2.1.8)

`wl_en = WORDLINE_ENABLE_BUFFER(access_clk_bar, s_en_bar)`, both stages
scaled with the row-driver load (`ceil(wl_load / 32)`, 24 with effort
buffers). The first stage is a NAND2 (`WORDLINE_REQUEST_NAND`, the drive of
the inverter it replaced, folded like it): the wordline request ends when
the sense enable fires, so a read wordline is released at the sense trigger
while a write wordline (no sense enable) lasts the whole access. Before
V2.1.8 the read wordline stayed on until the clock edge, 1.1 to 2.2 ns after
the amplifier had fired and been isolated from the bitlines, with the
bitline already at its rail (F1 of the V2.1.8 record); it now starts to
release 45 ps (FF) to 192 ps (16x16 10T mux SS) after the amplifier's
terminal sees the sense enable, with the sense margin unchanged; the release
path grows with the row count (547 ps at 512x4 SS, where the local wordline
is below 0.1 VDD only after the edge that ends the access; the precharge and
the write slot still wait for the wordline observer). `s_en` stays high
until the access request ends, so the request cannot re-arm within the
access. With the fixed 1.35 / 0.45 um output stage the edge
took 280 ps (rise) and 600 ps (fall) at 512 rows, which is also what opened
the address-change hazard (D12).
`wl_en_bar` enables the address and write-request hold latches (two NAND2
inputs each) and, before V2.1.6, the precharge NAND3; it is sized for that
fan-out (`wlb_scale`). Since V2.1.8 the latches reopen mid-access in a read;
the registers they follow only change at the next edge.

### 2.8 Replica delay (`_add_replica_delay`)

`rbl` -> `REPLICA_DELAY_CHAIN` (`N` unit inverters, four unit loads each;
odd, so `rbl_delay` rises once the replica bitline has discharged) ->
`rbl_delay_bar`.
`K` active replica cells and `N` stages set the sensing margin; `K = 1`,
`N = 9` deliberately keeps margin at the cost of the 200 ps YAML access limit
(`docs/DRIVER_SIZING_PROPOSAL.md`).

### 2.9 Write-request hold (`_add_write_request_hold`)

`we_hold` / `we_hold_bar`: the write request through a `HOLD_LATCH` with
enable `wl_en_bar` (V2.1.4). `we` is registered on the edge that also ends
an access. At FF 1.1 V / -40 C the register output changed 20 ps after
`clk_buf` but the access request fell only after 38 ps (82 and 133 ps at SS),
so at a read-to-write boundary `w_en = request & we` pulsed to 0.38 to
0.72 V while the read wordline was still on. Like the address bits, the
request is held while a wordline is on; `wl_en` falls only after the request,
so `w_en` and `s_en` end with the request alone.

V2.1.9: with the replica guard the enable is `wordline_idle` (section 2.10),
so the request stays held until the wordline is observed off. With
`wl_en_bar` the latch reopened when the wordline enable fell, and at a write ->
read boundary the new (read) request dropped `w_en` while the wordline was
still falling, like the write -> write release of section 2.13.

V2.1.10: the column write-data latches follow the same rule through
`din_hold` (section 2.10).

### 2.10 Wordline-off guard (`_add_wordline_off_guard`)

`pre_ready`: the previous wordline is off, as seen on the far replica
wordline. With terminal RC the physical wordline outlives `wl_en`, so the
matched replica wordline is observed with a half-unit inverter
(`rwl_pre_bar`, V2.1.0) and, because the observer switches near mid-rail and
a distributed wordline can still exceed 10 % VDD when the logic delay expires
(observed at 512 columns), its falling tail gets `precharge_guard_stages` of
settling (`PRECHARGE_GUARD_DELAY` + `PRECHARGE_GUARD_AND`, V2.1.1; zero
stages leave `rwl_pre_bar` itself). Without the replica guard the signal is
`wl_en_bar`, which only says that the enable is off, not the wordline.

V2.1.9 (`_add_wordline_busy`): `wordline_busy = !(wl_en_bar & pre_ready)`
(`WORDLINE_BUSY_NAND`) is high from the wordline enable until the replica
wordline has been observed off with its settling stages, and `wordline_idle`
is its complement (a unit inverter). The write enable holds the drivers on
while a write's wordline is busy (section 2.12) and the write-request latch opens
only when it is idle (section 2.9). Without the replica guard neither exists.

V2.1.10, write operations with the replica guard: `din_hold =
!(wordline_idle & ((we & cs) | !w_en))` (three `DIN_HOLD_NAND`:
`Xselected_write_nand` `we`, `cs` -> `selected_write_bar`; `Xdin_open_nand`
`selected_write_bar`, `w_en` -> `din_open`; `Xdin_hold_nand` `wordline_idle`,
`din_open` -> `din_hold_unbuf`; then `DIN_HOLD_BUFFER`, sized for
`din_hold_load`) holds the column write-data latches. The data, the request
and the select are registered on the rising edge that also ends the previous
access, so the latches pass the new data only once that wordline is observed
off; with the drivers on only for a selected write (between two writes the
drivers stay on and only their data changes), and otherwise only after the
drivers are off (a write -> read or write -> idle boundary releases the data
it wrote). In a read or idle cycle the latches are open, so the data of a
following write is at the drivers long before its slot. `w_en` is the block's
output node, as for `enables_off`; nothing in the block reads `din_hold`.

### 2.11 Select delay (`_add_select_delay`, V2.1.6, V2.1.7)

`cs_pre = cs & cs_delayed`, with `cs_delayed` the select through eight unit
stages (`SELECT_DELAY`, two loads per stage) and the AND in
`SELECT_DELAY_AND`: the select of the two clock-high enables, delayed on its
rising edge only. It feeds the precharge NAND3 and the write-enable AND3.

* V2.1.6 introduced the delay for the precharge. The select and the write
  request are registered on the same edge, but the request reaches the
  precharge gate through the hold latch and the AND2 (five gate delays)
  while the select is a register output; at the first selected write after
  an idle cycle the precharge fired for those five gates and overlapped the
  write drivers (seen at the start of every write deck before the delay).
* V2.1.7 gates the write enable with it too. With the raw select, `w_en` rose
  at an idle -> write edge as soon as the select did (the write slot is
  already open after an idle cycle), and the write-data hold latch closed
  24 to 29 ps after new data had settled at FF 1.1 V / -40 C (4x4 and 8x4,
  6T and 10T, twelve mismatch seeds; 47 ps at 8x64). The V2.1.6 idle probes
  changed the data inside the idle cycle, so that edge was never simulated
  with new data. Through `cs_pre` the margin is 110 to 112 ps at FF, 180 ps
  at TT and 404 ps at SS (8x4).
* V2.1.7 also stops delaying the falling edge. A symmetric delay kept
  `cs_pre` high for eight stages into an unselected cycle, while the
  wordline-off guard re-opened `pre_gate`: the precharge stayed off by 58 ps
  at 2x4 FF -40 C (the `PRE_UNBUF` node dipped 50 mV). The AND lets `cs_pre`
  fall one gate after `cs`; the gap is now 114 ps there.

In steady state (read -> read, write -> write, read <-> write) the select is
constant and the delay changes nothing. An idle -> read precharge starts one
AND2 later (+10 ps at FF, +17 ps at TT, +41 ps at SS, 8x4) and an idle ->
write slot opens after the delay (TWSLOT 206 / 329 / 738 ps at FF / TT / SS,
8x4), both far inside the clock-high phase.

### 2.12 Write slot (`_add_write_slot`, V2.1.6, V2.1.8)

In a write cycle the write drivers take the precharge slot:

* `pre_gate = pre_ready & we_hold_bar & enables_off` (`PRECHARGE_GATE_AND`)
  is the third input of the precharge NAND3, so the precharge is inhibited
  for the whole write cycle and, since V2.1.8, until the sense and write
  enables of the previous access are off (`enables_off = !(s_en | w_en)`,
  section 2.17);
* `write_slot = pre_ready & pre_off_ready & s_en_bar` (`WRITE_SLOT_AND`)
  opens as soon as the previous wordline, the physical precharge and, since
  V2.1.8, the previous sense enable are observed off, in the clock-high
  phase (the precharge input is tied high without the precharge-off guard).
  The write enable is not part of the slot's gate: the slot feeds it, and the
  loop would oscillate;
* `selected_slot_bar = !(cs_pre & write_slot)` (`SELECTED_SLOT_NAND`; an AND2
  `selected_slot` in V2.1.8 and V2.1.9, now an inverter for the probes only)
  is the slot of a selected cycle;
* `write_window = held(wordline_busy & we_hold) | selected_slot` keeps the
  drivers on until the wordline of a write is observed off (V2.1.9; until the
  wordline enable ended before, section 2.13): `busy_write_bar =
  !(wordline_busy & we_hold)` (`BUSY_WRITE_NAND`) through four unit stages
  (`BUSY_HOLD_DELAY`) to `busy_write_held_bar`, and `write_window =
  !(busy_write_held_bar & selected_slot_bar)` (`WRITE_WINDOW_NAND`). Between
  two writes the write's busy term hands over to the next slot and the drivers
  stay on (V2.1.10; the local checker requires the enable at every driver to
  stay at or above 0.9 VDD across the boundary); the new data reaches them
  through the write-data latches (section 2.10). The busy wordline and the
  slot follow the same observer, and the slot's path is longer (the NAND3 of
  `write_slot` and the selected-slot gate): at SS 0.9 V / 125 C under mismatch
  it arrived 61 ps after the busy wordline fell, and the window, and `w_en`
  with it, dropped to 0.2 V between two writes (the first V2.1.10 evidence
  pass). The busy term's end is therefore held for four unit stages, so the
  slot always takes over first; a write -> read release follows the request
  latch, a write -> idle release comes four stages later. A read's busy
  wordline is not in the window (V2.1.9 had `wordline_busy | selected_slot`):
  a first version held the busy wordline itself, and at the end of a read the
  window was still open when the request latch passed the next write request,
  so `w_en` pulsed to 0.30 V as the read's sense enable fell (16x16 SS,
  second evidence pass). At a write's own wordline the held term arrives six stages
  after `wl_en_bar` falls, while the slot it replaces closes only once the
  replica wordline is observed on.

V2.1.9 made the slot wait for the drivers to be observed off between two
writes (`slot_armed`, a NOR latch `SLOT_ARM_NOR` x 2 set by `enables_off`
through four unit stages `SLOT_ARM_DELAY`, reset by `wordline_busy`, with an
AND3 selected slot), because the write-data latches then took new data only
while `w_en` was low. That round trip made the write -> write slot 1.6 to
1.8 ns at SS 0.9 V / 125 C and failed the runtime restore check at the 32x16
and 32x32 class bounds; V2.1.10 removed it (`WRITE_LATCH_V2_1_10.md`).

Before V2.1.8 the boundary orders were set by path length alone (the next
precharge 125 ps at FF / 460 ps at SS after the previous sense enable was
off, 83 / 304 ps after the previous write enable; with zero settling stages
27 ps and 15 ps before the isolation dropped); they are gate relations now
(42 / 174 ps, 86 / 321 ps, 70 ps at zero stages, F2 of the V2.1.8 record).

The slot needs the replica guard. Without it the wordline-off signal is
`wl_en_bar`, and `wl_en | wl_en_bar` is constantly high: V2.1.6 held `w_en`
on for every phase of consecutive selected writes (1.00 V in a stand-alone
deck of the factory-default block), so the write-data hold latch never
reopened and the drivers stayed on while the previous wordline fell. Since
V2.1.7 the window of such a block is `wl_en` itself: the drivers start with
the wordline enable, as before V2.1.6. The testbench always builds both
guards.

See `WRITE_SLOT_V2_1_6.md`, `SELECT_GATE_V2_1_7.md`,
`ENABLE_OVERLAP_V2_1_8.md`, `WRITE_HOLD_V2_1_9.md` and `WRITE_LATCH_V2_1_10.md`
for the measurements.

### 2.13 Write enable (`_add_write_enable`, V2.1.8, V2.1.9)

`w_en = WRITE_ENABLE_AND(we_hold, write_window)`, buffered by
`WRITE_ENABLE_BUFFER` above 32 unit loads (24 with effort buffers). The
select reaches the drivers through the selected slot (clock-high) and
through a write's busy wordline (access), never directly: in V2.1.7 (`w_en =
we_hold & cs_pre & write_window`) a deselect dropped the drivers with the
local wordline still at 0.50 V (TT) / 0.43 V (SS) at a write -> idle
boundary (F3 of the V2.1.8 record). V2.1.8 ended the drivers with the
wordline enable, and with the held request at a write -> read boundary;
the physical wordline outlives the enable by the row driver and the wire, so
the drivers started to release (0.9 VDD) with the local wordline at 0.51 V of
1.1 V at 8x4 FF -40 C, 0.27 V at 8x4 SS, about half VDD at 256x4 and 0.90 V
(the wordline not yet falling) at 512x4 SS. Since V2.1.9 they stay fully on
until the wordline is observed off (`wordline_busy`; V2.1.10 holds only a
write's busy term, and its end for four unit stages), 106 ps (8x4 FF under
mismatch) to 483 ps (16x64 with a mux, SS) after the local wordline is below
0.1 VDD. Since V2.1.10 they stay on from one write to the next, and only their
data changes (section 2.10). History:
V2.0.1 scaled one inverter with columns/64 and rows/16, which left a fan-out
of 40 to 60, a 130 to 180 ps edge from 64x16 to 16x512, and the driven
bitline reached VDD/2 only 120 to 240 ps after `gated_clk_bar` (D18). Before
V2.0.2 `w_en` was also gated by `rbl_delay_bar`, so the write pulse ended
when the replica cell had discharged the replica bitline (about 250 ps);
that path is stronger than the row-scaled write driver through the mux, the
pulse had about 30 % margin nominally, and Monte Carlo samples with a weak
NMOS left the bitline at 0.3 to 0.4 V when `w_en` ended. Before V2.1.6
`w_en = access_clk_bar & we_hold` started with the wordline, so the wordline
rose 23 to 189 ps before the driven bitline reached its rail (see
`WRITE_SLOT_V2_1_6.md`). V2.1.6 used the raw select `cs`; V2.1.7 used
`cs_pre` (section 2.11) directly, V2.1.8 through the selected slot.

### 2.14 Sense enable (`_add_sense_enable`)

`s_en = AND3(rbl_delay, access_clk_bar, we_hold_bar)`, buffered by
`SENSE_ENABLE_BUFFER` above 32 unit loads. It enables the sense-amplifier
footers (one 0.27 um NMOS gate, about 0.75 unit loads per amplifier) and the
output latch, and since V2.1.8 the observers of section 2.17 (which end the
read wordline and hold the next precharge and write slot). It stays high
until the access request ends: the amplifier is isolated from the bitlines
(`sa_iso`) for that time and the output latch is transparent. V2.0.1 scaled one four-unit inverter per 64 columns for the
footers *and* the pass gates (fan-out about 75): a 190 to 230 ps edge at 64
columns and more, and a 0.2 to 0.3 V precharge-coupling bump on `s_en` (D16).

### 2.15 Sense isolation (`_add_sense_isolation`)

`sa_iso = s_en | w_en` (NOR2 + inverter, `SENSE_ISOLATION_BUFFER` above 32
unit loads) drives the amplifier's input pass gates. Before V2.0.2 they were
driven by `s_en` alone, so the cross-coupled PMOS pair stayed connected to
the bitlines during a write and acted as a keeper: the write only succeeded
while `w_en` rose before the wordline (60 ps margin at 2x128 in V2.0.1; with
the faster wordline path of V2.0.2 the order flipped and the 2x128 write
deadlocked at BL 0.27 V / BLB 0.9 V, D20).

### 2.16 Precharge (`_add_precharge`)

`PRE = NAND3(clk_buf, cs_pre, pre_gate)` through `PRECHARGE_BUFFER`, sized
for three precharge PMOS gates per column plus the replica column (the
previous two-stage buffer had a fan-out of about 49 and PRE reached only 0.05
to 0.08 V on the largest arrays, D17). The bitlines are precharged for the
whole clock-high phase of a selected read cycle, from the previous wordline
and both enables observed off (section 2.12), and released when the clock
falls. Before V2.0.2 `PRE = NAND3(gated_clk_buf, rbl_delay, wl_en_bar)` was a
self-timed pulse of about 300 ps; afterwards every bitline floated and leaked
through the off pass gates of the cells storing a 0 on that side: RBL 0.89 V
at the next access with a 50 ns clock and 0.76 V with 100 ns at TT 25 C,
0.74 to 0.77 V at FF 125 C with the then-default 10 ns clock, an 80 to
120 mV BL/BLB offset before the read (D17). Holding the bitlines costs no
dynamic energy; the static-power window includes the leakage supplied through
the precharge devices.

### 2.17 Enable observers (`_add_enable_observers`, V2.1.8)

`s_en_bar = !s_en` (an inverter of `senb_scale` units) and `enables_off =
!(s_en | w_en)` (`ENABLES_OFF_NOR`). `s_en_bar` ends the wordline request of
a read at the sense trigger (section 2.7) and holds the write slot until the
sense enable of a preceding read is off; `enables_off` holds the precharge
until both enables are off (in V2.1.9 it also set the slot-arm latch,
section 2.12). Both read the block's output
nodes, the buffered enables, so a wordline, slot or precharge only follows
what the periphery has seen. They are built last (after the enables they
observe); the earlier builders reference them by name.

## 3. Sizing policy

`ControlSizing` in `time_generate.py` turns the loads supplied by
`driver_sizing.py` into buffer scales. Every enable drives up to `effort`
unit loads per unit of drive directly; above `4 * effort` a `TaperedBuffer`
follows the gate (two stages up to a scale of 16, four above, or an
effort-based even chain with 2 um gate fingers).

| Enable | Load (unit inverters) | Direct limit | Effort | Buffer |
|---|---|---|---|---|
| `wl_en` | `wl_load` NAND2 units, from `driver_sizing` | one unit per 32 (24) loads, both stages | - | `WORDLINE_ENABLE_BUFFER` scale |
| `w_en` | `wen_load` (write drivers incl. the replica, hold latches, the enables-off NOR input since V2.1.8) | 32 (24) | 8 (6) | `WRITE_ENABLE_BUFFER` |
| `s_en` | `sen_load` (footers + 3.5 + `senb_scale` + 1.75 since V2.1.8) | 32 (16, 12 with RC) | 8 (4, 3 with RC) | `SENSE_ENABLE_BUFFER` |
| `sa_iso` | `iso_load` (two pass gates per amplifier) | 32 (16) | 8 (3) | `SENSE_ISOLATION_BUFFER`, fall strength 2 with effort buffers |
| `PRE` | `pre_load` (3 PMOS per column + replica + observer) | `ceil(load / 8)` (2 + `ceil(load / 4)`) | - | `PRECHARGE_BUFFER` |
| `A_dff` | 5 (6) gate inputs per 8 rows | `ceil(units / 24)` (`/ 8`) | - | `ADDRESS_BUFFER` |
| `wl_en_bar` | 2 latch inputs per address bit + write request + 1 (with the replica guard the request latch reads `wordline_idle` and `wl_en_bar` the busy NAND2 instead since V2.1.9; the count is kept as an upper bound) | fan-out 5 | - | inverter scale |
| `s_en_bar` (V2.1.8) | the wordline-request NAND2 input (1.25 units per unit of `wl_en` scale) + the write-slot NAND3 input | fan-out 4 (`senb_scale`) | - | inverter scale |
| access request | `access_load` = 1.25 x `wl_en` scale + 2.5 (the NAND2 input since V2.1.8, the sense NAND3 input) | `max(6, ceil(load / 6))` | - | the guard's `PRECHARGE_ACCESS_AND` output |
| `clk_buf` | flip-flop count | relative to 16x16 | - | `CLOCK_BUFFER` scale |
| write-data hold latch (testbench) | one write-driver input of class `wd_in` | scaled with `wd_in` (V2.1.6) | - | `D_LATCH` widths; `wenb_scale` counts two enables per column |

Values in parentheses apply with `effort_buffers` (the lookup sizing default).
The fallbacks used when a load is not supplied reproduce the V2.0.2 estimates
for direct `TIME_CONTROL` callers; the testbench always supplies the loads.

## 4. Naming contract

The testbench, the per-device CLI, the validator and the qualification scorer
address nodes inside the block by instance path. A topology or readability
change must keep these names:

* the top-level instance `XTIME_CONTROL` (V2.1.7; `XTIME` before), so probe
  paths read `XTIME_CONTROL:<node>`;
* instances `Xdff_buf` (`qint` is initialised), `Xdff_buf1`, `Xdff_buf_addr`,
  `Xdff_buf_data` with `Xdff_{col}` and the `DFF` internal nodes `D_b`, `z1`
  .. `z5`, `QB`, and the clock taps `CLK_line_tap{col}` / `CLK_line_far`;
* nodes `A_reg{i}`, `A_lat{i}`, `we_hold`, `we_hold_bar`, `wl_en_bar`,
  `access_clk_bar`, `pre_off_ready`, `pre_ready`, `rwl_pre_bar`, `pre_gate`,
  `cs_delayed`, `cs_pre`, `write_slot`, `selected_slot_bar`, `selected_slot`,
  `write_window`, `s_en_bar`, `enables_off`, `wordline_busy`, `wordline_idle`,
  `busy_write_bar`, `busy_write_held_bar`, `selected_write_bar`, `din_open`,
  `din_hold_unbuf`, `PRE_UNBUF`,
  `sa_iso_bar`, and the `*_unbuf` nodes of buffered enables;
* the guard instance `Xaccess_guard` with its internal `pre_on`,
  `pre_on_filtered`, `pre_on_delayed`;
* subcircuit names, which are also how the deck-comparison tools find
  blocks: `TIME_CONTROL`, `PRECHARGE_OFF_GUARD`, `WRITE_ENABLE_AND`,
  `SELECT_DELAY`, `SELECT_DELAY_AND`, `PRECHARGE_GATE_AND`, `WRITE_SLOT_AND`,
  `SELECTED_SLOT_NAND`, `ENABLES_OFF_NOR`, `WORDLINE_REQUEST_NAND`,
  `WORDLINE_BUSY_NAND`, `BUSY_WRITE_NAND`, `BUSY_HOLD_DELAY`,
  `WRITE_WINDOW_NAND`, `DIN_HOLD_NAND`, `DIN_HOLD_BUFFER` and the others of
  section 6. Every
  subcircuit defined in `time_generate.py` is named after its class in upper
  snake case; `TaperedBuffer` takes its role name (`*_BUFFER`) and the
  standard cells keep theirs (`AND2`, `AND3`, `PNOR2`, `PNAND3`, `PINV*`).

## 5. Proving a refactor netlist-identical

`dev/v216_time_audit/snapshot_decks.py OUT_DIR all` writes 1296 control-block
netlists (rows 2 to 512, columns 4 to 512, mux on and off, all three
operations, default sizing, without effort buffers and without the guard
delay), 24 replica columns (6T and 10T, with and without RC) and 11 full-array
decks; `compare_snapshots.py BEFORE AFTER` diffs them. The V2.1.6 readability
refactor of `time_generate.py` and `replica_column.py` was accepted on
1331 of 1331 identical netlists. `topo_check.py` in the same directory is the
static connectivity audit (no floating input, no multi-driven node, no port
against its direction, no same-name subcircuit collision) that the V2.1.5
audit ran before the change; V2.1.7 ran it again on 630 configurations (only
the intentional dummy loads and the unused `A_latb{i}` outputs are reported)
and the collision audit on 72.

V2.1.7 renamed names only in a first step and proved it with
`dev/v217_boundary/snapshot.py` (the same matrix plus blocks without guards,
idle probes and a zero-settling deck; it runs against either tree) and
`compare_renamed.py BEFORE AFTER`, which applies the section 6 map to the
V2.1.6 text: 1351 of 1351 netlists identical. `structural_diff.py` then
compared the renamed tree with the fixed one per subcircuit scope; the only
changes are the select gate, the `w_en` input, the no-guard window and the
`select_every` write data.

## 6. Name map V2.1.6 -> V2.1.7

| V2.1.6 | V2.1.7 |
|---|---|
| class `TIME`, subcircuit `TIME`, instance `XTIME` | `TIME_CONTROL`, `TIME_CONTROL`, `XTIME_CONTROL` |
| `TIMEFactory` | `TimeControlFactory` |
| `Sram6TCoreTestbench.create_time_circuit` | `create_time_control_circuit` |
| `pdrive` / `wl_pdrive` | `CLOCK_BUFFER` / `WORDLINE_ENABLE_BUFFER` |
| `DFF_BUF` | `DFF_BUFFER` |
| `D_LATCH_ADDR` | `HOLD_LATCH` |
| `delay_chain` / `wen_delay_chain` (`PINV_wen_delay`) | `REPLICA_DELAY_CHAIN` / `UNIT_DELAY_CHAIN` (`PINV_unit_delay`) |
| `ADDR_DFF` / `DATA_DFF` | `ADDRESS_REGISTER` / `DATA_REGISTER` |
| class `AND3_WEN`, subcircuit `AND3_WEN` | `WriteEnableAnd`, `WRITE_ENABLE_AND` |
| `AND2_PRE_WRITE` / `AND2_WRITE_SLOT` / `PNOR2_WRITE_WINDOW` | `PRECHARGE_WRITE_AND` / `WRITE_SLOT_AND` / `WRITE_WINDOW_NOR` |
| `AND2_PRE_GUARD` / `AND2_PRE_ACCESS` | `PRECHARGE_GUARD_AND` / `PRECHARGE_ACCESS_AND` |
| class `PrechargeSelectDelay`, `PRECHARGE_SELECT_DELAY`, instance `Xprecharge_select_delay` (cs -> cs_pre) | `SelectDelay`, `SELECT_DELAY`, `Xselect_delay` (cs -> cs_delayed), plus `SelectDelayAnd` / `SELECT_DELAY_AND` / `Xselect_gate` (cs & cs_delayed -> cs_pre) |
| `TaperedBuffer` roles `ABUF`, `WEN_BUF`, `SEN_BUF`, `ISO_BUF`, `PRE_BUF` (inverters `PINV_<role>_<k>`) | `ADDRESS_BUFFER`, `WRITE_ENABLE_BUFFER`, `SENSE_ENABLE_BUFFER`, `SENSE_ISOLATION_BUFFER`, `PRECHARGE_BUFFER` |

## 7. Changes V2.1.7 -> V2.1.8

| V2.1.7 | V2.1.8 |
|---|---|
| `WORDLINE_ENABLE_BUFFER` ports `A`, `Z`; first stage `PINV1` (`Xbuf_inv1`) | ports `A`, `B`, `Z`; first stage `WORDLINE_REQUEST_NAND` (`Xbuf_nand1`), `B` = `s_en_bar` |
| `PrechargeWriteAnd` / `PRECHARGE_WRITE_AND` (AND2), instance `Xpre_write_gate` (`wordline_off`, `we_hold_bar`) | `PrechargeGateAnd` / `PRECHARGE_GATE_AND` (AND3), instance `Xpre_gate` (`wordline_off`, `we_hold_bar`, `enables_off`) |
| `WRITE_SLOT_AND` (AND2: `wordline_off`, `pre_off_ready`), only with the precharge-off guard | AND3 (`wordline_off`, `pre_off_ready` or `VDD`, `s_en_bar`), with the replica guard |
| `WRITE_WINDOW_NOR` (`wl_en`, `write_slot` / `pre_ready` / `rwl_pre_bar`) | (`wl_en`, `selected_slot`); new `SelectedSlotAnd` / `SELECTED_SLOT_AND`, instance `Xselected_slot` (`cs_pre`, `write_slot`) |
| `WRITE_ENABLE_AND` (AND3: `we_hold`, `cs_pre`, window) | AND2 (`we_hold`, window) |
| - | `PINV_s_en_bar` (`Xinv_s_en_bar`: `s_en` -> `s_en_bar`), `EnablesOffNor` / `ENABLES_OFF_NOR` (`Xenables_off_nor`: `s_en`, `w_en` -> `enables_off`) |

## 8. Changes V2.1.8 -> V2.1.9

| V2.1.8 | V2.1.9 |
|---|---|
| - | `WordlineBusyNand` / `WORDLINE_BUSY_NAND`, instance `Xwordline_busy_nand` (`wl_en_bar`, `pre_ready` or `rwl_pre_bar` -> `wordline_busy`); `PINV_wordline_idle` (`Xinv_wordline_idle`: `wordline_busy` -> `wordline_idle`); replica guard only |
| `Xwe_hold` enable `wl_en_bar` | `wordline_idle` with the replica guard (`wl_en_bar` without) |
| `WRITE_WINDOW_NOR` (`wl_en`, `selected_slot`) | (`wordline_busy`, `selected_slot`) |
| `SELECTED_SLOT_AND` (AND2: `cs_pre`, `write_slot`) | AND3 (`cs_pre`, `write_slot`, `slot_armed`) |
| - | `SlotArmDelay` / `SLOT_ARM_DELAY` (four unit stages, `Xslot_arm_delay`: `enables_off` -> `enables_off_settled`); `SlotArmNor` / `SLOT_ARM_NOR` x 2 (`Xslot_arm_set`: `enables_off_settled`, `slot_armed` -> `slot_armed_bar`; `Xslot_arm_reset`: `wordline_busy`, `slot_armed_bar` -> `slot_armed`) |

## 9. Changes V2.1.9 -> V2.1.10

| V2.1.9 | V2.1.10 |
|---|---|
| `SelectedSlotAnd` / `SELECTED_SLOT_AND` (AND3: `cs_pre`, `write_slot`, `slot_armed`), instance `Xselected_slot` | `SelectedSlotNand` / `SELECTED_SLOT_NAND` (`Xselected_slot_nand`: `cs_pre`, `write_slot` -> `selected_slot_bar`); `PINV_selected_slot` (`Xselected_slot` -> `selected_slot`, probes only) |
| `SlotArmDelay` / `SLOT_ARM_DELAY` (`Xslot_arm_delay`), `SlotArmNor` / `SLOT_ARM_NOR` x 2 (`Xslot_arm_set`, `Xslot_arm_reset`), nodes `slot_armed`, `slot_armed_bar`, `enables_off_settled` and their testbench `.IC` | removed |
| - | `DinHoldNand` / `DIN_HOLD_NAND` x 3 (`Xselected_write_nand`: `we`, `cs` -> `selected_write_bar`; `Xdin_open_nand`: `selected_write_bar`, `w_en` -> `din_open`; `Xdin_hold_nand`: `wordline_idle`, `din_open` -> `din_hold_unbuf`), `DIN_HOLD_BUFFER` (`Xdin_hold_buf` -> `din_hold`); port `din_hold` (last; writes with the replica guard); `TIME_CONTROL(din_hold_load=...)` |
| `WriteWindowNor` / `WRITE_WINDOW_NOR` (`Xwrite_window_nor`: `wordline_busy`, `selected_slot` -> `write_window_bar`), `PINV_write_window` (`Xwrite_window_inv` -> `write_window`) | `BusyWriteNand` / `BUSY_WRITE_NAND` (`Xbusy_write_nand`: `wordline_busy`, `we_hold` -> `busy_write_bar`), `BusyHoldDelay` / `BUSY_HOLD_DELAY` (four unit stages, `Xbusy_hold_delay`: `busy_write_bar` -> `busy_write_held_bar`), `WriteWindowNand` / `WRITE_WINDOW_NAND` (`Xwrite_window_nand`: `busy_write_held_bar`, `selected_slot_bar` -> `write_window`) |
| testbench `WEN_BAR` (`PINV_wen_bar`: `w_en` -> `w_en_bar`), column line `w_en_bar` | `DIN_EN` (`PINV_din_en`: `din_hold` -> `din_en`, `w_en` without the replica guard), column line `din_en` |
