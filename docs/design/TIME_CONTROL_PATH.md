# TIME control path: signals, stages and their history

Reference for `sram_compiler/subcircuits/time_generate.py` (V2.1.6). The
module docstring lists the signals; this document holds what the code
comments used to carry: why each stage exists, the measurements that led to
it, the sizing policy, and the naming contract a refactor must keep. Release
numbers refer to `CHANGELOG.md`; the D-numbered items are the review findings
in `sram_compiler/CIRCUIT_REVIEW.md`.

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
| `wl_en` | out | wordline enable, one NAND2 input per row driver plus the replica driver |
| `rbl_delay`, `rbl_delay_bar` | out | replica bitline through the inverting delay chain |
| `s_en` | out | sense enable (footers and output latch) |
| `w_en` | out | write enable (write drivers, write-data hold latches) |
| `sa_iso` | out | sense-amplifier input isolation, `s_en \| w_en` |
| `PRE` | out | precharge, active low |

The clock has a 50 % duty cycle: the capture edge at `1 ns + 0.2 T + k T`,
the access (falling) edge at `1 ns + 0.7 T + k T`. Clock-high is the
precharge phase of a read cycle and the write slot of a write cycle;
clock-low is the access.

## 2. Stages in signal order

The builders of `TIME` follow this order, which is also the order of the
subcircuit definitions and instances in the netlist.

### 2.1 Address path (`_add_address_path`)

`A{i}` -> `ADDR_DFF` register (`A_reg{i}`) -> `D_LATCH_ADDR` hold latch
(`A_lat{i}`, enable `wl_en_bar`) -> `ABUF` tapered buffer -> `A_dff{i}`.

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

`DIN{i}` -> `DATA_DFF` -> `DIN_dff{i}`, write operations only. Since V2.1.1
the register clock crosses the array width on the wordline wire geometry with
one tap per column. The testbench's write-data hold latch (enable `w_en_bar`)
follows this register, so the driver input cannot change while the drivers
are on.

### 2.3 Clock tree (`_add_clock_tree`)

`clk` -> `pdrive` (four stages, 1 : 3 : 9 : 27 unit widths) -> `clk_buf` ->
unit inverter -> `clk_bar`. The buffer scale is the flip-flop count relative
to a 16x16 write deck (`ControlSizing.clk_drive_scale`).

### 2.4 Select and write registers (`_add_select_and_write_registers`)

`csb` and `web` are registered by `DFF_BUF` on the rising `clk_buf` edge.
`DFF_BUF` buffers Q (the registered input, so `cs_bar` and `we_bar`) and QB
(`cs` and `we`). The testbench parks the select flip-flop inactive with a
start-up clamp and an `.IC` on `XTIME:Xdff_buf:qint`.

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

### 2.7 Wordline enable (`_add_wordline_enable`)

`wl_en = wl_pdrive(access_clk_bar)`, both stages scaled with the row-driver
load (`ceil(wl_load / 32)`, 24 with effort buffers). With the fixed 1.35 /
0.45 um output stage the edge took 280 ps (rise) and 600 ps (fall) at 512
rows, which is also what opened the address-change hazard (D12).
`wl_en_bar` enables the address and write-request hold latches (two NAND2
inputs each) and, before V2.1.6, the precharge NAND3; it is sized for that
fan-out (`wlb_scale`).

### 2.8 Replica delay (`_add_replica_delay`)

`rbl` -> `delay_chain` (`N` unit inverters, four unit loads each; odd, so
`rbl_delay` rises once the replica bitline has discharged) -> `rbl_delay_bar`.
`K` active replica cells and `N` stages set the sensing margin; `K = 1`,
`N = 9` deliberately keeps margin at the cost of the 200 ps YAML access limit
(`docs/DRIVER_SIZING_PROPOSAL.md`).

### 2.9 Write-request hold (`_add_write_request_hold`)

`we_hold` / `we_hold_bar`: the write request through a `D_LATCH_ADDR` with
enable `wl_en_bar` (V2.1.4). `we` is registered on the edge that also ends
an access. At FF 1.1 V / -40 C the register output changed 20 ps after
`clk_buf` but the access request fell only after 38 ps (82 and 133 ps at SS),
so at a read-to-write boundary `w_en = request & we` pulsed to 0.38 to
0.72 V while the read wordline was still on. Like the address bits, the
request is held while a wordline is on; `wl_en` falls only after the request,
so `w_en` and `s_en` end with the request alone.

### 2.10 Wordline-off guard (`_add_wordline_off_guard`)

`pre_ready`: the previous wordline is off, as seen on the far replica
wordline. With terminal RC the physical wordline outlives `wl_en`, so the
matched replica wordline is observed with a half-unit inverter
(`rwl_pre_bar`, V2.1.0) and, because the observer switches near mid-rail and
a distributed wordline can still exceed 10 % VDD when the logic delay expires
(observed at 512 columns), its falling tail gets `precharge_guard_stages` of
settling (`PRECHARGE_GUARD_DELAY` + `AND2_PRE_GUARD`, V2.1.1). Without the
replica guard the signal is `wl_en_bar`.

### 2.11 Write slot (`_add_write_slot`, V2.1.6)

In a write cycle the write drivers take the precharge slot:

* `pre_gate = pre_ready & we_hold_bar` replaces `pre_ready` as the third input
  of the precharge NAND3, so the precharge is inhibited for the whole write
  cycle;
* `write_slot = pre_ready & pre_off_ready` opens as soon as the previous
  wordline and the physical precharge are observed off, in the clock-high
  phase;
* `write_window = wl_en | write_slot` keeps the drivers on until the wordline
  request ends;
* `cs_pre`: the select delayed by eight unit stages (`PRECHARGE_SELECT_DELAY`)
  into the precharge NAND3. The select and the write request are registered
  on the same edge, but the request reaches the precharge gate through the
  hold latch and the AND2 (five gate delays) while the select is a register
  output; at the first selected write after an idle cycle the precharge fired
  for those five gates and overlapped the write drivers (seen at the start of
  every write deck before the delay). A read cycle's precharge waits for
  `pre_ready` anyway, so the delay never limits it.

See `WRITE_SLOT_V2_1_6.md` for the measurements.

### 2.12 Write enable (`_add_write_enable`)

`w_en = AND3_WEN(we_hold, cs, write_window)`, buffered by `WEN_BUF` above
32 unit loads (24 with effort buffers). History: V2.0.1 scaled one inverter
with columns/64 and rows/16, which left a fan-out of 40 to 60, a 130 to
180 ps edge from 64x16 to 16x512, and the driven bitline reached VDD/2 only
120 to 240 ps after `gated_clk_bar` (D18). Before V2.0.2 `w_en` was also
gated by `rbl_delay_bar`, so the write pulse ended when the replica cell had
discharged the replica bitline (about 250 ps); that path is stronger than the
row-scaled write driver through the mux, the pulse had about 30 % margin
nominally, and Monte Carlo samples with a weak NMOS left the bitline at 0.3
to 0.4 V when `w_en` ended. Before V2.1.6 `w_en = access_clk_bar & we_hold`
started with the wordline, so the wordline rose 23 to 189 ps before the
driven bitline reached its rail (see `WRITE_SLOT_V2_1_6.md`).

### 2.13 Sense enable (`_add_sense_enable`)

`s_en = AND3(rbl_delay, access_clk_bar, we_hold_bar)`, buffered by `SEN_BUF`
above 32 unit loads. It enables the sense-amplifier footers (one 0.27 um
NMOS gate, about 0.75 unit loads per amplifier) and the output latch. V2.0.1
scaled one four-unit inverter per 64 columns for the footers *and* the pass
gates (fan-out about 75): a 190 to 230 ps edge at 64 columns and more, and a
0.2 to 0.3 V precharge-coupling bump on `s_en` (D16).

### 2.14 Sense isolation (`_add_sense_isolation`)

`sa_iso = s_en | w_en` (NOR2 + inverter, `ISO_BUF` above 32 unit loads)
drives the amplifier's input pass gates. Before V2.0.2 they were driven by
`s_en` alone, so the cross-coupled PMOS pair stayed connected to the bitlines
during a write and acted as a keeper: the write only succeeded while `w_en`
rose before the wordline (60 ps margin at 2x128 in V2.0.1; with the faster
wordline path of V2.0.2 the order flipped and the 2x128 write deadlocked at
BL 0.27 V / BLB 0.9 V, D20).

### 2.15 Precharge (`_add_precharge`)

`PRE = NAND3(clk_buf, cs_pre, pre_gate)` through `PRE_BUF`, sized for three
precharge PMOS gates per column plus the replica column (the previous
two-stage buffer had a fan-out of about 49 and PRE reached only 0.05 to
0.08 V on the largest arrays, D17). The bitlines are precharged for the whole
clock-high phase of a selected read cycle and released when the clock falls.
Before V2.0.2 `PRE = NAND3(gated_clk_buf, rbl_delay, wl_en_bar)` was a
self-timed pulse of about 300 ps; afterwards every bitline floated and leaked
through the off pass gates of the cells storing a 0 on that side: RBL 0.89 V
at the next access with a 50 ns clock and 0.76 V with 100 ns at TT 25 C,
0.74 to 0.77 V at FF 125 C with the then-default 10 ns clock, an 80 to
120 mV BL/BLB offset before the read (D17). Holding the bitlines costs no
dynamic energy; the static-power window includes the leakage supplied through
the precharge devices.

## 3. Sizing policy

`ControlSizing` in `time_generate.py` turns the loads supplied by
`driver_sizing.py` into buffer scales. Every enable drives up to `effort`
unit loads per unit of drive directly; above `4 * effort` a `TaperedBuffer`
follows the gate (two stages up to a scale of 16, four above, or an
effort-based even chain with 2 um gate fingers).

| Enable | Load (unit inverters) | Direct limit | Effort | Buffer |
|---|---|---|---|---|
| `wl_en` | `wl_load` NAND2 units, from `driver_sizing` | one unit per 32 (24) loads, both stages | - | `wl_pdrive` scale |
| `w_en` | `wen_load` (write drivers incl. the replica, hold latches) | 32 (24) | 8 (6) | `WEN_BUF` |
| `s_en` | `sen_load` (footers + 3.5) | 32 (16) | 8 (4, 3 with RC) | `SEN_BUF` |
| `sa_iso` | `iso_load` (two pass gates per amplifier) | 32 (16) | 8 (3) | `ISO_BUF`, fall strength 2 with effort buffers |
| `PRE` | `pre_load` (3 PMOS per column + replica + observer) | `ceil(load / 8)` (2 + `ceil(load / 4)`) | - | `PRE_BUF` |
| `A_dff` | 5 (6) gate inputs per 8 rows | `ceil(units / 24)` (`/ 8`) | - | `ABUF` |
| `wl_en_bar` | 2 latch inputs per address bit + write request + 1 | fan-out 5 | - | inverter scale |
| `clk_buf` | flip-flop count | relative to 16x16 | - | `pdrive` scale |
| write-data hold latch (testbench) | one write-driver input of class `wd_in` | scaled with `wd_in` (V2.1.6) | - | `D_LATCH` widths; `wenb_scale` counts two enables per column |

Values in parentheses apply with `effort_buffers` (the lookup sizing default).
The fallbacks used when a load is not supplied reproduce the V2.0.2 estimates
for direct `TIME` callers; the testbench always supplies the loads.

## 4. Naming contract

The testbench, the per-device CLI, the validator and the qualification scorer
address nodes inside TIME by instance path. A topology or readability change
must keep these names:

* instances `Xdff_buf` (`qint` is initialised), `Xdff_buf1`, `Xdff_buf_addr`,
  `Xdff_buf_data` with `Xdff_{col}` and the `DFF` internal nodes `D_b`, `z1`
  .. `z5`, `QB`, and the clock taps `CLK_line_tap{col}` / `CLK_line_far`;
* nodes `A_reg{i}`, `A_lat{i}`, `we_hold`, `we_hold_bar`, `wl_en_bar`,
  `access_clk_bar`, `pre_off_ready`, `pre_ready`, `rwl_pre_bar`, `pre_gate`,
  `cs_pre`, `write_slot`, `write_window`, `PRE_UNBUF`, `sa_iso_bar`, and the
  `*_unbuf` nodes of buffered enables;
* the guard instance `Xaccess_guard` with its internal `pre_on`,
  `pre_on_filtered`, `pre_on_delayed`;
* subcircuit names (`TIME`, `PRECHARGE_OFF_GUARD`, `AND2_WEN` .. ), which are
  also how the deck-comparison tools find blocks.

## 5. Proving a refactor netlist-identical

`dev/v216_time_audit/snapshot_decks.py OUT_DIR all` writes 1296 TIME netlists
(rows 2 to 512, columns 4 to 512, mux on and off, all three operations, default
sizing, without effort buffers and without the guard delay), 24 replica
columns (6T and 10T, with and without RC) and 11 full-array decks;
`compare_snapshots.py BEFORE AFTER` diffs them. The V2.1.6 readability
refactor of `time_generate.py` and `replica_column.py` was accepted on
1331 of 1331 identical netlists. `topo_check.py` in the same directory is the
static connectivity audit (no floating input, no multi-driven node, no port
against its direction, no same-name subcircuit collision) that the V2.1.5
audit ran before the change.
