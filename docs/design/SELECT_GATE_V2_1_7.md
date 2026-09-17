# V2.1.7: the select gate of the clock-high enables, and the TIME_CONTROL rename

Executed September 16 and 17, 2026 on the V2.1.7 sources (base commit
`59fc2e3` plus the changes of this release). A review of the
[V2.1.6 write slot](WRITE_SLOT_V2_1_6.md) looked for abnormal functions and
boundary bugs in the control block, the testbench and the checks, probed the
boundaries the V2.1.6 matrix did not simulate, fixed five defects, renamed
the block `TIME` to `TIME_CONTROL` together with its inner subcircuits, and
re-ran the evidence matrix. Illustrative wires, V2.0.9 driver classes and the
unchanged timing classes throughout; per-device samples use 5 % relative
sigma on `vth0`, `u0` and `voff` of every MOS. This is functional screening,
not PVT/mismatch or extracted-metal qualification; no record is promoted to
`sizing_table.json`.

**Outcome: on the release sources 74 of 74 cases pass, 236,570 of 236,570 checks: all 47 nominal boundary cases and all 27 mismatch seeds (p_64x16_6t_mux_SS_read_idle2 after a DC operating-point retry with Newton line search). New write data reached every driver's hold latch at least 94 ps before the column's write enable rose (p_8x4_6t_FF_cold_write_idle2_pd_s3); in the 23 decks with unselected cycles the precharge stayed off (its terminal never below 99.8% of VDD) and no write or sense enable rose above 1 mV there; the driven bitline reached its rail at least 1495 ps before the local wordline started; with the settling stages the write slot or precharge never opened sooner than 70 ps after the wordline release; the longest write slot is 1642 ps (w_512x4_6t_mux_SS_write).**

## 1. What the review found

| | Finding | Where | Consequence |
|---|---|---|---|
| F1 | A deck with the documented `sizing.precharge_guard_stages: 0` raised `The write-slot safety measures need the replica and precharge-off guards` | `_add_precharge_safety_measures` (V2.1.6) | every transient deck refused; the write slot exists with both guards regardless of the settling stages |
| F2 | Without the replica guard the write window was `wl_en \| wl_en_bar`, constantly high | `_add_write_slot` (V2.1.6) | `w_en = we_hold & cs` for the whole selected write: the drivers never released between consecutive writes, the write-data hold latch never reopened (the second write drives old data) and the drivers were on while the previous wordline fell. The factory defaults build this block |
| F3 | Five builders ignored the passed transistor models (a hard-coded replica-wordline observer; default-model replica delay chain, guard delay, wordline-enable buffer and every tapered buffer) although the V2.1.6 changelog said they were honoured | `time_generate.py` | a block built for other models still instantiated `NMOS_VTG` / `PMOS_VTG` |
| F4 | The select delay `cs_pre` was symmetric and the write enable used the raw select `cs` | `_add_write_slot`, `_add_write_enable` (V2.1.6) | at an idle -> write edge with new data the write-data hold latch closed 24 to 29 ps after the data had settled at FF 1.1 V / -40 C; in an unselected cycle the delayed select raced the wordline-off guard (58 ps at 2x4 FF) |
| F5 | The `select_every` write decks changed the data inside the idle cycle | testbench `DIN` sources (V2.1.6) | the idle -> write probes of the V2.1.6 record never registered new data at the idle -> write edge, so F4 was not simulated |

F4 is the only one that touched production decks, and none of them failed:
the latch always closed on the new data, with a margin of two to three gate
delays that did not move under mismatch. It is fixed because the ordering is
a race between two paths from the same clock edge, not a relation the circuit
enforces.

### 1.1 How the boundaries were probed

`dev/v217_boundary/probe.py` builds the release testbench with extra
control-block probes and two stimulus overrides (`--din idle_change`: the
write data changes only at the idle -> write edge; `--web idle_read`: a read
request at the unselected edges); `analyze.py` tabulates every enable edge
per cycle, the hold-latch timing per column, the idle-cycle levels and the
Boolean relations of the clock-high enables. The probe decks have no local
RC (`w_rc` off), nominal models unless noted.

| Boundary (8x4 6T unless noted) | Quantity | V2.1.6 | V2.1.7 |
|---|---|---:|---:|
| idle -> write, new data, FF 1.1 V / -40 C | hold-latch output settled (50 %) to latch closing (`w_en_bar` 50 %) | 29 ps | 111 ps |
| same, twelve / three mismatch seeds | same | 24 to 29 ps | 109 to 112 ps |
| same, 4x4 (four seeds in V2.1.6) | same | 26 to 28 ps | 111 ps |
| same, 8x4 10T (four seeds in V2.1.6) | same | 26 to 28 ps | 110 ps |
| same, 8x64 with mux | same | 47 ps | 130 ps |
| same, TT 1.0 V / 25 C | same | 47 ps | 180 ps |
| same, SS 0.9 V / 125 C | same | 107 ps | 404 ps |
| read -> idle, 2x4 FF | `cs_pre` falling to `pre_gate` rising (PRE stays off) | 58 ps | 114 ps |
| read -> idle, 8x4 FF / TT / SS | same | 62 / 101 / 226 ps | 118 / 191 / 426 ps |
| idle -> write after a read request, FF / TT | `pre_gate` falling to `cs_pre` rising | 40 / 60 ps | 51 / 80 ps |
| idle -> read, FF / TT / SS | PRE (50 %) after the capture edge | 178 / 270 / 568 ps | 188 / 287 / 609 ps |
| idle -> read, FF / TT / SS | TRESTORE | 181 / 295 / 662 ps | 192 / 313 / 703 ps |
| idle -> write, FF / TT / SS | TWSLOT | 124 / 197 / 441 ps | 206 / 329 / 738 ps |
| write -> write, FF | hold-latch output settled to latch closing | not probed | 84 ps (the select is constant: unchanged) |

Without the replica guard (F2), a stand-alone block from the factory defaults
driven through three selected writes (`noguard_block.py`) held `w_en` at
1.00 V in every clock-high and clock-low phase in V2.1.6; in V2.1.7 it is
0.00 V in the clock-high phases and 1.00 V in the clock-low phases. With zero
settling stages (F1) V2.1.6 generated no deck; the V2.1.7 8x4 TT sequence
passes all 56 of its runtime checks.

On the V2.1.7 sources every probe deck (21 runs: reads and writes with idle
cycles at FF, TT and SS, 2x4 to 8x64, 6T and 10T, three mismatch seeds,
local RC, the sequences at FF and SS, a 16x16 mux sequence with mismatch and
the zero-settling sequence) passed its runtime checks; `cs_pre == cs &
cs_delayed`, `w_en == we_hold & cs_pre & write_window` and
`PRE_UNBUF == !(clk_buf & cs_pre & pre_gate)` held everywhere up to the
propagation delay (at most 136 ps, 8x4 SS), and PRE and `w_en` were never
active together.

## 2. The changes

* **Select gate (F4).** `cs_pre = cs & cs_delayed`, where `cs_delayed` is the
  select through the eight unit stages that V2.1.6 used for `cs_pre` itself.
  `cs_pre` still feeds the precharge NAND3 and now also the write-enable AND3
  (`w_en = we_hold & cs_pre & write_window`). The rising edge keeps the
  V2.1.6 delay (plus one AND2), so the held write request inhibits the
  precharge first and new data passes the driver's hold latch before `w_en`
  closes it; the falling edge follows the select, so an unselected cycle
  stops both enables at once. Steady-state cycles see a constant select and
  are unchanged. `docs/design/TIME_CONTROL_PATH.md`, section 2.11.
* **No write slot without the replica guard (F2).** Without the observer the
  window is `wl_en`: the drivers start with the wordline enable, as before
  V2.1.6. The precharge stays inhibited in a write cycle.
* **Models (F3).** Every builder passes the block's models; for the default
  models the netlists are unchanged.
* **Safety measures (F1).** They require both guards (which own the
  `write_slot` node), not a nonzero settling-stage count.
* **Idle probes (F5).** A `select_every` write deck holds its data through the
  idle cycles and changes it 0.1 T before the next selected edge (the setup
  of the address and select stimuli). `select_every = 1` decks are unchanged.
* **Checks.** The local waveform checker adds `strict_idle_<k>_precharge_off`
  and `strict_idle_<k>_enables_off` for the unselected cycles (PRE off from
  the idle edge, write and sense enables at most 0.1 VDD from 0.3 T after it)
  and `strict_cycle_<k>_col_<c>_data_before_write_enable` with the metric
  `_data_to_write_enable_ps` (new data at the driver's hold latch output
  before the column's write enable rises).
* **Rename.** `TIME` -> `TIME_CONTROL` (class, subcircuit, instance
  `XTIME_CONTROL`), `TIMEFactory` -> `TimeControlFactory`,
  `create_time_circuit` -> `create_time_control_circuit`, and every inner
  subcircuit named after its class (map in `TIME_CONTROL_PATH.md`, section 6).
  Instance and node names inside the block are kept.

## 3. Proof of the rename and scope of the circuit change

The rename was made first, with the model fix, and proven on 1351 netlists
(1296 control blocks over rows 2 to 512, columns 4 to 512, mux, three
operations and three sizing variants; 18 blocks without guards; 24 replica
columns; 13 full decks including a `select_every` read and the zero-settling
sequence): after mapping the old names every netlist is identical to V2.1.6
(`dev/v217_boundary/snapshot.py`, `compare_renamed.py`). A per-scope
structural comparison of the renamed and the fixed trees
(`structural_diff.py`) shows only: `SELECT_DELAY` (the former
`PRECHARGE_SELECT_DELAY`, same body) now drives `cs_delayed`, the new
`SELECT_DELAY_AND` drives `cs_pre`, `Xw_en` reads `cs_pre`; in the 18 blocks
without guards the window gates are gone and `Xw_en` reads `wl_en`; the
`select_every = 2` write deck changes its `DIN` pulse width; the zero-settling
deck exists.

The static connectivity audit (`dev/v216_time_audit/topo_check.py`) over 630
configurations reports only the intentional dummy loads and the unused
`A_latb{i}` outputs; the subcircuit collision audit finds no same-name,
different-body definition in 72 configurations. Tracked tests: 145 pass,
with new tests for each finding that fail when the fix is reverted (checked
by mutation); local tool tests: 58 pass.

## 4. Timing table

`timing_lookup.json` becomes `v2.1.7-timing-6` with every class kept. The
select gate changes only the first selected cycle after an unselected one:
an idle -> read precharge starts one gate later and an idle -> write slot
opens after the select delay, both far inside clock-high phases of 1800 ps
and more (section 1.1).

## 5. Evidence

The matrix (`outputs/validation/V2.1.7-select-gate/gen_cases.py`, 74 cases
in 20 queues at up to 56 simulator ranks) is the V2.1.6 matrix with the
V2.1.6 fix-pass settings built in (7200 s for the 64x16 10T writes, Newton
line search for the 8x128 mux read) and three probe queues: idle -> write
with new data at FF -40 C (4x4, 8x4 6T and 10T, 8x64 mux, and three mismatch
seeds), read -> idle -> read at FF (2x4, 8x4 6T and 10T, two seeds) and at
the SS bounds (16x16, 64x16, 16x64 with mux), and the 8x4 TT sequence with
zero settling stages. The idle -> write probes of the V2.1.6 matrix now
register new data at the idle -> write edge (F5).

One attempt failed before any waveform: the four-rank DC operating point of
the 64x16 mux read idle probe did not converge (the validator retries the
operating point only on one rank). It was rerun on the same sources with
Newton line search (`run_fix.sh`, queue `fix-A`), the setting the V2.1.6 fix
pass used for the 8x128 read, and passed. Every queue archived its sources,
and all 21 manifests match the release tree. With zero settling stages the
next precharge or write slot follows the replica-wordline observer alone,
13 ps after the wordline release at TT (70 ps and more with the four default
stages at FF); that option remains for explicitly validated wire
configurations (`sram_compiler/sizing/README.md`).

Every case, latest attempt (`outputs/validation/V2.1.7-select-gate/`, `assemble_record.py`; the JSON record is `SELECT_GATE_V2_1_7.json` next to this file). Rail before WL: smallest distance from the driven bitline reaching 10 % VDD to the local wordline reaching 10 % VDD over the write cycles; entry after release: smallest distance from the wordline release (10 %) to the next write slot or precharge; data before w_en: smallest distance from the new data at a driver's hold-latch output (50 %) to that column's write enable (50 %); idle: lowest precharge terminal and highest write or sense enable over the unselected cycles; TWSLOT (write decks) or TRESTORE (read decks) from the deck's `.mt0`; clock to Q: largest clock-fall-to-cell-flip time. Seeds `sN` are `20261700 + N`.

| Case | Cell | Size | Mux | Operation | Corner | Variation | T [ns] | Checks | Rail before WL [ps] | Entry after release [ps] | Data before w_en [ps] | Idle PRE min / enable peak [V] | TWSLOT / TRESTORE [ps] | Clock to Q [ps] | Result |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| p_16x16_10t_SS_write_idle2 | 10T | 16x16 | no | write (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 1941 | 2052.5 | 286.4 | 398.1 | 0.898 / 0.0 | 906.6 | 529.8 | pass |
| p_16x16_6t_mux_SS_read_idle2 | 6T_ | 16x16 | yes | read (every 2) | SS 0.9 V / 125 C | nominal | 4.75 | 1560 | - | 4933.6 | - | 0.9 / 0.0 | 793.9 | - | pass |
| p_16x16_6t_mux_SS_write_idle2 | 6T_ | 16x16 | yes | write (every 2) | SS 0.9 V / 125 C | nominal | 4.75 | 1941 | 1918.6 | 285.6 | 397.0 | 0.9 / 0.0 | 913.8 | 510.0 | pass |
| p_16x64_6t_mux_SS_read_idle2 | 6T_ | 16x64 | yes | read (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 6024 | - | 5210.4 | - | 0.9 / 0.0 | 826.1 | - | pass |
| p_2x4_6t_FF_cold_read_idle2 | 6T_ | 2x4 | no | read (every 2) | FF 1.1 V / -40 C | nominal | 4.5 | 122 | - | 4520.8 | - | 1.099 / 0.0 | 214.6 | - | pass |
| p_4x4_6t_FF_cold_write_idle2 | 6T_ | 4x4 | no | write (every 2) | FF 1.1 V / -40 C | nominal | 4.5 | 261 | 2175.8 | 72.0 | 95.6 | 1.099 / 0.0 | 228.8 | 142.3 | pass |
| p_64x16_6t_mux_SS_read_idle2 | 6T_ | 64x16 | yes | read (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 5544 | - | 5167.5 | - | 0.9 / 0.0 | 798.8 | - | pass |
| p_64x16_6t_mux_SS_write_idle2 | 6T_ | 64x16 | yes | write (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 5925 | 2078.7 | 287.0 | 385.4 | 0.9 / 0.0 | 901.9 | 527.2 | pass |
| p_8x4_10t_FF_cold_read_idle2 | 10T | 8x4 | no | read (every 2) | FF 1.1 V / -40 C | nominal | 5.0 | 260 | - | 5016.9 | - | 1.099 / 0.0 | 218.6 | - | pass |
| p_8x4_10t_FF_cold_write_idle2 | 10T | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | nominal | 5.0 | 353 | 2426.8 | 71.9 | 95.6 | 1.099 / 0.0 | 232.2 | 148.5 | pass |
| p_8x4_6t_FF_cold_read_idle2 | 6T_ | 8x4 | no | read (every 2) | FF 1.1 V / -40 C | nominal | 4.5 | 260 | - | 4516.0 | - | 1.099 / 0.0 | 217.5 | - | pass |
| p_8x4_6t_FF_cold_read_idle2_pd_s1 | 6T_ | 8x4 | no | read (every 2) | FF 1.1 V / -40 C | per-device s1 | 4.5 | 260 | - | 4515.1 | - | 1.099 / 0.0 | 219.0 | - | pass |
| p_8x4_6t_FF_cold_read_idle2_pd_s2 | 6T_ | 8x4 | no | read (every 2) | FF 1.1 V / -40 C | per-device s2 | 4.5 | 260 | - | 4516.4 | - | 1.099 / 0.0 | 217.0 | - | pass |
| p_8x4_6t_FF_cold_write_idle2 | 6T_ | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | nominal | 4.5 | 353 | 2171.8 | 72.0 | 95.7 | 1.099 / 0.0 | 232.3 | 141.9 | pass |
| p_8x4_6t_FF_cold_write_idle2_pd_s1 | 6T_ | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | per-device s1 | 4.5 | 353 | 2172.5 | 70.5 | 93.8 | 1.099 / 0.0 | 230.7 | 143.4 | pass |
| p_8x4_6t_FF_cold_write_idle2_pd_s2 | 6T_ | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | per-device s2 | 4.5 | 353 | 2173.2 | 72.1 | 95.2 | 1.099 / 0.0 | 231.2 | 143.1 | pass |
| p_8x4_6t_FF_cold_write_idle2_pd_s3 | 6T_ | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | per-device s3 | 4.5 | 353 | 2172.9 | 72.7 | 93.6 | 1.099 / 0.0 | 231.8 | 143.7 | pass |
| p_8x4_6t_FF_cold_write_idle3 | 6T_ | 8x4 | no | write (every 3) | FF 1.1 V / -40 C | nominal | 4.5 | 355 | 2171.8 | 72.0 | 95.7 | 1.099 / 0.0 | 232.3 | 141.9 | pass |
| p_8x4_6t_SS_write_idle2_pd_s1 | 6T_ | 8x4 | no | write (every 2) | SS 0.9 V / 125 C | per-device s1 | 4.5 | 353 | 1848.9 | 253.7 | 350.3 | 0.898 / 0.0 | 827.6 | 529.1 | pass |
| p_8x4_6t_SS_write_idle2_pd_s2 | 6T_ | 8x4 | no | write (every 2) | SS 0.9 V / 125 C | per-device s2 | 4.5 | 353 | 1861.8 | 280.1 | 362.4 | 0.898 / 0.0 | 834.4 | 531.6 | pass |
| p_8x4_6t_SS_write_idle2_pd_s3 | 6T_ | 8x4 | no | write (every 2) | SS 0.9 V / 125 C | per-device s3 | 4.5 | 353 | 1866.4 | 280.2 | 343.9 | 0.898 / 0.0 | 843.9 | 540.8 | pass |
| p_8x4_6t_TT_sequence_settle0 | 6T_ | 8x4 | no | read&write (0 settling stages) | TT 1.0 V / 25 C | nominal | 4.5 | 1311 | 2074.0 | 12.8 | 184.8 | - | - | 229.7 | pass |
| p_8x4_6t_TT_write_idle2 | 6T_ | 8x4 | no | write (every 2) | TT 1.0 V / 25 C | nominal | 4.5 | 353 | 2102.2 | 119.3 | 156.4 | 0.998 / 0.0 | 370.1 | 229.4 | pass |
| p_8x64_6t_mux_FF_cold_write_idle2 | 6T_ | 8x64 | yes | write (every 2) | FF 1.1 V / -40 C | nominal | 5.0 | 4973 | 2405.2 | 74.3 | 111.6 | 1.099 / 0.0 | 263.0 | 150.7 | pass |
| r_128x8_10t_mux_SS_read | 10T | 128x8 | yes | read | SS 0.9 V / 125 C | nominal | 6.0 | 3435 | - | 407.3 | - | - | 1084.5 | - | pass |
| r_128x8_6t_mux_SS_read | 6T_ | 128x8 | yes | read | SS 0.9 V / 125 C | nominal | 5.5 | 3435 | - | 404.6 | - | - | 1080.8 | - | pass |
| r_256x4_10t_SS_read | 10T | 256x4 | no | read | SS 0.9 V / 125 C | nominal | 8.0 | 3647 | - | 395.4 | - | - | 1207.7 | - | pass |
| r_256x4_6t_mux_SS_read | 6T_ | 256x4 | yes | read | SS 0.9 V / 125 C | nominal | 6.75 | 3647 | - | 396.2 | - | - | 1189.9 | - | pass |
| r_512x4_10t_mux_SS_read | 10T | 512x4 | yes | read | SS 0.9 V / 125 C | nominal | 10.0 | 1117 | - | 407.2 | - | - | 1526.6 | - | pass |
| r_512x4_6t_mux_SS_read | 6T_ | 512x4 | yes | read | SS 0.9 V / 125 C | nominal | 9.0 | 1117 | - | 405.8 | - | - | 1523.3 | - | pass |
| r_64x16_10t_mux_SS_read | 10T | 64x16 | yes | read | SS 0.9 V / 125 C | nominal | 5.5 | 3395 | - | 409.8 | - | - | 1044.0 | - | pass |
| r_64x16_6t_mux_SS_read | 6T_ | 64x16 | yes | read | SS 0.9 V / 125 C | nominal | 5.0 | 3395 | - | 410.5 | - | - | 1041.9 | - | pass |
| r_8x128_6t_mux_SS_read | 6T_ | 8x128 | yes | read | SS 0.9 V / 125 C | nominal | 6.0 | 4515 | - | 443.7 | - | - | 1058.5 | - | pass |
| s_16x16_10t_SS_sequence | 10T | 16x16 | no | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 6971 | 1791.2 | 286.8 | 464.3 | - | - | 532.1 | pass |
| s_16x16_10t_SS_sequence_pd_s1 | 10T | 16x16 | no | read&write | SS 0.9 V / 125 C | per-device s1 | 5.0 | 6971 | 1753.0 | 266.6 | 441.1 | - | - | 538.5 | pass |
| s_16x16_10t_SS_sequence_pd_s2 | 10T | 16x16 | no | read&write | SS 0.9 V / 125 C | per-device s2 | 5.0 | 6971 | 1753.5 | 294.3 | 454.0 | - | - | 548.0 | pass |
| s_16x16_10t_mux_SS_sequence | 10T | 16x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 6971 | 1785.0 | 286.6 | 463.4 | - | - | 534.4 | pass |
| s_16x16_6t_SS_sequence | 6T_ | 16x16 | no | read&write | SS 0.9 V / 125 C | nominal | 4.5 | 6971 | 1536.8 | 286.4 | 464.0 | - | - | 509.7 | pass |
| s_16x16_6t_mux_SS_sequence | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 4.75 | 6971 | 1658.8 | 285.3 | 464.6 | - | - | 511.8 | pass |
| s_16x16_6t_mux_SS_sequence_pd_s1 | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | per-device s1 | 4.75 | 6971 | 1631.9 | 285.4 | 489.9 | - | - | 509.1 | pass |
| s_16x16_6t_mux_SS_sequence_pd_s2 | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | per-device s2 | 4.75 | 6971 | 1607.1 | 321.7 | 460.1 | - | - | 525.6 | pass |
| s_16x16_6t_mux_SS_sequence_pd_s3 | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | per-device s3 | 4.75 | 6971 | 1666.8 | 280.5 | 445.2 | - | - | 520.2 | pass |
| s_16x32_6t_mux_SS_sequence | 6T_ | 16x32 | yes | read&write | SS 0.9 V / 125 C | nominal | 4.75 | 13707 | 1647.7 | 287.7 | 474.2 | - | - | 517.2 | pass |
| s_32x16_10t_mux_SS_sequence | 10T | 32x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 11451 | 1744.5 | 286.9 | 463.4 | - | - | 537.6 | pass |
| s_32x16_6t_SS_sequence | 6T_ | 32x16 | no | read&write | SS 0.9 V / 125 C | nominal | 4.5 | 11451 | 1494.6 | 286.1 | 465.0 | - | - | 514.7 | pass |
| s_32x16_6t_mux_SS_sequence | 6T_ | 32x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 4.75 | 11451 | 1615.6 | 286.3 | 463.9 | - | - | 516.2 | pass |
| s_8x4_10t_FF_cold_sequence | 10T | 8x4 | no | read&write | FF 1.1 V / -40 C | nominal | 5.0 | 1311 | 2341.4 | 72.6 | 112.6 | - | - | 148.6 | pass |
| s_8x4_10t_mux_SS_sequence | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 1311 | 1818.0 | 273.0 | 421.2 | - | - | 541.9 | pass |
| s_8x4_10t_mux_SS_sequence_pd_s1 | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | per-device s1 | 5.0 | 1311 | 1832.1 | 264.5 | 429.3 | - | - | 538.6 | pass |
| s_8x4_10t_mux_SS_sequence_pd_s2 | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | per-device s2 | 5.0 | 1311 | 1805.4 | 293.2 | 425.0 | - | - | 581.0 | pass |
| s_8x4_10t_mux_SS_sequence_pd_s3 | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | per-device s3 | 5.0 | 1311 | 1780.9 | 267.2 | 404.1 | - | - | 559.1 | pass |
| s_8x4_6t_FF_cold_sequence | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | nominal | 4.5 | 1311 | 2085.6 | 72.5 | 112.6 | - | - | 142.1 | pass |
| s_8x4_6t_FF_cold_sequence_pd_s1 | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | per-device s1 | 4.5 | 1311 | 2085.2 | 71.7 | 109.6 | - | - | 143.3 | pass |
| s_8x4_6t_FF_cold_sequence_pd_s2 | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | per-device s2 | 4.5 | 1311 | 2086.6 | 70.2 | 110.8 | - | - | 144.1 | pass |
| s_8x4_6t_FF_cold_sequence_pd_s3 | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | per-device s3 | 4.5 | 1311 | 2085.5 | 73.5 | 111.5 | - | - | 143.0 | pass |
| s_8x4_6t_SS_sequence | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | nominal | 4.5 | 1311 | 1569.2 | 273.0 | 421.2 | - | - | 516.0 | pass |
| s_8x4_6t_SS_sequence_pd_s1 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s1 | 4.5 | 1311 | 1552.6 | 263.5 | 402.6 | - | - | 534.1 | pass |
| s_8x4_6t_SS_sequence_pd_s2 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s2 | 4.5 | 1311 | 1569.8 | 234.9 | 410.7 | - | - | 527.3 | pass |
| s_8x4_6t_SS_sequence_pd_s3 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s3 | 4.5 | 1311 | 1542.6 | 295.1 | 421.6 | - | - | 524.2 | pass |
| s_8x4_6t_SS_sequence_pd_s4 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s4 | 4.5 | 1311 | 1590.6 | 258.9 | 411.4 | - | - | 518.9 | pass |
| s_8x4_6t_SS_sequence_pd_s5 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s5 | 4.5 | 1311 | 1554.2 | 269.0 | 412.2 | - | - | 514.1 | pass |
| s_8x8_6t_mux_FF_cold_sequence | 6T_ | 8x8 | yes | read&write | FF 1.1 V / -40 C | nominal | 4.75 | 2451 | 2218.0 | 75.4 | 115.3 | - | - | 143.4 | pass |
| w_128x8_6t_SS_write | 6T_ | 128x8 | no | write | SS 0.9 V / 125 C | nominal | 5.25 | 3530 | 2279.8 | 282.5 | 451.5 | - | 1199.3 | 570.5 | pass |
| w_16x64_6t_mux_SS_write | 6T_ | 16x64 | yes | write | SS 0.9 V / 125 C | nominal | 5.0 | 4594 | 2017.2 | 286.2 | 482.8 | - | 1220.6 | 534.1 | pass |
| w_256x4_6t_SF_write | 6T_ | 256x4 | no | write | SF 0.9 V / 125 C | nominal | 6.25 | 3694 | 2899.6 | 230.3 | 393.8 | - | 1170.7 | 604.5 | pass |
| w_256x4_6t_SS_write | 6T_ | 256x4 | no | write | SS 0.9 V / 125 C | nominal | 6.25 | 3694 | 2875.9 | 273.9 | 439.2 | - | 1302.2 | 666.2 | pass |
| w_512x4_6t_mux_SS_write | 6T_ | 512x4 | yes | write | SS 0.9 V / 125 C | nominal | 9.0 | 1164 | 4512.2 | 275.4 | 426.3 | - | 1641.9 | 980.0 | pass |
| w_64x16_10t_mux_SF_write | 10T | 64x16 | yes | write | SF 0.9 V / 125 C | nominal | 5.5 | 3586 | 2368.6 | 246.9 | 408.4 | - | 1070.6 | 491.7 | pass |
| w_64x16_10t_mux_SS_write | 10T | 64x16 | yes | write | SS 0.9 V / 125 C | nominal | 5.5 | 3586 | 2336.0 | 285.9 | 457.5 | - | 1185.6 | 548.7 | pass |
| w_64x16_6t_mux_SF_write | 6T_ | 64x16 | yes | write | SF 0.9 V / 125 C | nominal | 5.0 | 3586 | 2115.1 | 246.5 | 407.6 | - | 1068.0 | 473.4 | pass |
| w_64x16_6t_mux_SS_write | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | nominal | 5.0 | 3586 | 2082.4 | 286.1 | 457.3 | - | 1183.1 | 528.8 | pass |
| w_64x16_6t_mux_SS_write_pd_s1 | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | per-device s1 | 5.0 | 3586 | 2071.4 | 290.4 | 431.6 | - | 1189.3 | 543.9 | pass |
| w_64x16_6t_mux_SS_write_pd_s2 | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | per-device s2 | 5.0 | 3586 | 2047.5 | 300.8 | 447.8 | - | 1216.6 | 534.8 | pass |
| w_64x16_6t_mux_SS_write_pd_s3 | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | per-device s3 | 5.0 | 3586 | 2083.9 | 299.2 | 446.7 | - | 1201.8 | 571.9 | pass |

## 6. Limits

* Screening at the class bounds with illustrative wires; extracted metal,
  half-select writes and the yield estimator remain the carried Phase 6
  scope.
* The data setup at the driver is ordered by construction (the enable path
  contains the data path's register plus the select delay), but it is still
  a relation between two paths from one clock edge, checked here at FF
  -40 C, TT and SS; a different write-data latch or driver input class needs
  the same probe.
* The factory-default block without guards is a legacy configuration: its
  drivers start with the wordline enable, so it does not provide the V2.1.6
  rails-before-wordline order. The testbench never builds it.
