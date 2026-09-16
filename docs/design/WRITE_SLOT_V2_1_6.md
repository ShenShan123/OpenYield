# V2.1.6: the write drivers take the precharge slot

Executed September 16, 2026 on the V2.1.6 sources (base commit `6c45a3b`
plus the changes of this release; every queue archives the sources it ran).
Illustrative wires, V2.0.9 driver classes, the V2.1.5 timing classes, the
V2.1.4 TIME block with the write-request hold latch, and the redefined
V2.1.6 checks throughout; nominal seed 20260916, maximum step 20 ps; per-device
samples on one rank each with 5 % relative sigma on `vth0`, `u0` and `voff`
of every MOS. This is functional screening at SS 0.9 V / 125 °C (SF for
write-ability, FF 1.1 V / −40 °C for the cold sequences and probes, TT for
one idle probe), not PVT/mismatch or extracted-metal qualification; no record
is promoted to `sizing_table.json`.

**Outcome: on the release sources 58 of 58 attempts pass, 213,020 of 213,020 checks: all 36 nominal boundary cases (sequences, single writes with toggled data, idle → write probes and reads at the class bounds) and all 22 mismatch seeds. In every write cycle the driven bitline reached its rail at least 1494 ps before the local wordline started; the write slot never opened before the previous wordline was released (smallest gap 72 ps); the longest write slot (TWSLOT, capture edge to the driven far bitline at 0.1 VDD) is 1642 ps at w_512x4_6t_mux_SS_write, against clock-high phases of 1800 ps and more.**

## 1. What the audit found

The V2.1.5 audit of `time_generate.py` and `replica_column.py` (static
connectivity over 630 TIME configurations, eight Xyce decks with a
Boolean-relation checker; tools in `dev/v216_time_audit/`) found every control
relation correct and no functional defect, but one ordering that the design
intent forbids: in a write cycle the local wordline started to rise before the
write drivers had driven the bitlines to their rails. Both `wl_en` and `w_en`
fanned out from the same access request, and the write driver, its buffer and
the bitline slew are slower than the two-stage wordline buffer.

| Case (V2.1.5) | `w_en` 50 % after the clock fall | driven bitline at 10 % VDD | local WL 10 % / 50 % | bitline at WL 50 % | margin (WL start − rail) |
|---|---:|---:|---:|---:|---:|
| 8x4 6T TT nominal | 237 ps | 291 ps | 252 / 258 ps | 0.57 V | −39 ps |
| 8x4 6T FF −40 °C per-device | 149 | 181 | 158 / 162 | 0.59 V | −23 |
| 8x4 6T TT with local RC | 260 | 361 | 277 / 291 | 0.67 V | −84 |
| 16x16 6T mux TT per-device | 289 | 349 | 277 / 284 | 1.04 V | −72 |
| 16x16 10T TT nominal | 288 | 356 | 273 / 280 | 1.04 V | −83 |
| 64x16 6T mux SS 125 °C 0.9 V | 685 | 840 | 651 / 667 | 0.93 V | −189 |

The cell therefore saw its wordline with both bitlines high (a read
condition) and was overpowered 30 to 190 ps later. Every write completed, so
the access checks passed; the sequence was simply not enforced.

## 2. The change

The write enable now references the precharge phase, the way the precharge
references it for a read: in a write cycle the drivers take the precharge
slot. In `TIME` (`docs/design/TIME_CONTROL_PATH.md`, section 2.11):

* `pre_gate = pre_ready & we_hold_bar` feeds the precharge NAND3, so the
  precharge is inhibited for the whole write cycle;
* `write_slot = pre_ready & pre_off_ready` opens once the previous wordline
  (replica-wordline observer plus settling) and the physical precharge (far
  PRE observer plus settling) are off, in the clock-high phase;
* `w_en = we_hold & cs & (wl_en | write_slot)` turns the drivers on from the
  slot and keeps them on until the wordline request ends;
* the replica column's write driver, previously an inert load, is enabled
  from the far `w_en` tap and writes a 0, so the replica bitline sees what a
  real column sees (`wen_load` counts it);
* the wordline path is untouched. In a write cycle it no longer waits for the
  precharge-off guard, because the precharge never turned on.

### 2.1 The start-up race and the select delay

The first prototype passed every sequence but failed the "precharge off
during write" check at the first selected write of every write deck: the
select and the write request are registered on the same edge, but the request
reaches the precharge qualifier through the hold latch and the AND2 (five
gate delays) while the select is a register output, so at the first selected
write after an idle cycle the precharge fired for those five gates
(PRE at 0.03 V at 2.10 ns with `w_en` at 0.79 V in the 8x4 TT write deck)
and overlapped the rising write enable. In steady state the replica-wordline
guard hides this, because `pre_ready` arrives about 240 ps after the edge and
`we_hold` at about 130 ps. The race-free fix is `cs_pre`: the select delayed
by eight unit stages (`PRECHARGE_SELECT_DELAY`, two loads per stage) into the
precharge NAND3 only. A read cycle's precharge waits for `pre_ready` anyway,
so the delay never limits it (TRESTORE +18 ps at 16x16 TT, +43 ps at 64x16 SS).

### 2.2 The write-data hold latch must be sized for the driver input

The first evidence run failed the 512x4 6T mux SS write on twelve of 1160
checks: `hold_data_during_write`, `hold_complement_during_write` and
`driver_data_during_write` of every column. The write itself completed (the
rails were reached 4.7 ns before the wordline) but the testbench's write-data
hold latch, a unit `D_LATCH` driving the 8x-scaled write-driver input of a
512-row array, took about 700 ps at SS 125 °C to bring `DIN_hold` from 0.1 to
0.9 V, and the slot turned the drivers on about 500 ps after the capture edge
while that data was still mid-rail. Before V2.1.6 the enable came a nanosecond
after the data and hid the under-sized latch. The latch now scales with the
write-driver input class (`wd_in`: 1 up to 64 rows, 2 at 128, 4 at 256, 8 at
512) and `wenb_scale`, the `w_en_bar` inverter that drives two latch enables
per column, counts that scale. Decks up to 64 rows are byte-identical to the
first run; the 128-, 256- and 512-row cases were rerun on the released sources.

### 2.3 Prototype measurements (worktree, before the checks were redefined)

| Case | bitline at rail before WL starts: V2.1.5 → V2.1.6 | cell flip after the clock fall: V2.1.5 → V2.1.6 |
|---|---:|---:|
| 8x4 6T TT nominal | −39 ps → +2.0 ns | 287 → 131 ps |
| 8x4 6T FF −40 °C per-device | −23 ps → +2.1 ns | 179 → 78 ps |
| 16x16 6T mux TT per-device | −72 ps → +2.1 ns | 342 → 142 ps |
| 16x16 10T TT nominal | −83 ps → +2.3 ns | 353 → 150 ps |
| 64x16 6T mux SS 125 °C 0.9 V | −189 ps → +2.1 ns | 826 → 333 ps |

In every write cycle the driven bitline sat at 1 mV and the other at VDD when
the wordline started, with `w_en` fully high; PRE and `w_en` were never
active together; reads changed by 3 to 9 ps on the wordline enable.

## 3. Checks redefined

"Bitlines restored to VDD in every selected clock-high phase" was the V2.0.2
policy; it now reads "restored to VDD before a read, at the write rails before
a write". The testbench, the local validator and the qualification scorer
derive the expected state of every cycle from a cycle plan
(`cycle_plan` / `access_cycles` in `sram_6t_core_MC_testbench.py`):

| Where | Before | V2.1.6 |
|---|---|---|
| `VRESTORE_ERROR_k` (runtime) | BL/BLB/RBL far ends at VDD 0.6 T into the next cycle | same before a read; BL = data, BLB = !data, RBL = 0 before a write |
| `VWL_PRE_*_k` (runtime) | wordline low when PRE falls in the next cycle | kept before a read; `VWL_WEN_*_k` before a write: wordline low when `XTIME:write_slot` rises, and while it is high |
| `TRESTORE` (write decks) | next-cycle precharge restores the written bitline | `TWSLOT`: capture edge of the next write to its driven bitline at 0.1 VDD, the clock-high work of a write cycle; `timing_from_measurements` and the minimum-period estimate use it |
| write decks | write 1 twice | write 1 then write 0 (write → write with new data) |
| `strict_*` checks (`dev/v210_waveform_checks.py`) | restore at VDD, `w_en` rises after the clock fall | rails per the next cycle, `w_en` rises after the capture edge, `bitlines_driven_before_wordline` (both rails at WL 10 %), `release_before_write_enable` / `WL_during_write_slot`, the write-enable quiet window ends at the wordline release |
| scorer (`dev/sizing/qualification.py`) | `all_bitlines_restored` / `equalized` for every case | for write cases `all_bitlines_driven`; `wl_off_before_write_enable`; `TWSLOT` required |

A new testbench option `select_every` (single read/write decks) selects one
cycle in N, so the idle → write boundary (first write after an unselected
cycle) is simulated; the validator case key of the same name drives it.

## 4. Timing table

`timing_lookup.json` becomes `v2.1.6-timing-5` with every class kept.
Reads are unchanged within 10 ps on the wordline path. The clock-high work
of a write cycle is now the write slot: wordline-off guard, write enable and
bitline drive, which is far shorter than the restore it replaces
(section 5), and the write access no longer waits for the precharge-off
guard nor for the bitline drive, so the write side only gained margin. The
restore after a write happens in the following read cycle's clock-high phase
from a full swing, which the read classes already budgeted (the sense
amplifier fires after a full bitline swing in this design).

## 5. Evidence

The matrix (`outputs/validation/V2.1.6-write-slot/gen_cases.py`, 58 cases in
18 queues at up to 48 simulator ranks) ran in three passes on the same
sources for every deck it reports: the main run; `fix2` reruns of every 128-,
256- and 512-row case after the hold-latch sizing of section 2.2 (decks up to
64 rows are byte-identical before and after it, `dev/v216_time_audit/
snapshot_decks.py`); and `fix3` reruns of the four attempts the queue tool
had marked `timeout` (the 64x16 10T writes, 30 min under full machine load;
the 6T twin took 16 min), `failed` (the 8x128 6T mux read, whose 4-rank DC
operating point failed before any waveform; rerun with Newton line search,
the qualification runner's own retry) or `source_changed` (attempts in flight
when the hold-latch edit landed). Superseded attempts are retained in their
queue directories; the record keeps the latest attempt of every case.

Every case, latest attempt (`outputs/validation/V2.1.6-write-slot/`, `assemble_record.py`; the JSON record is `WRITE_SLOT_V2_1_6.json` next to this file). Rail before WL: smallest distance from the driven bitline reaching 10 % VDD to the local wordline reaching 10 % VDD over the write cycles; slot after release: smallest distance from the wordline release (10 %) to the write slot opening; clock to Q: largest clock-fall-to-cell-flip time; TWSLOT from the deck's `.mt0`.

| Case | Cell | Size | Mux | Operation | Corner | Variation | T [ns] | Checks | Rail before WL [ps] | Slot after release [ps] | Clock to Q [ps] | TWSLOT [ps] | Result |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| w_128x8_10t_mux_SS_read | 10T | 128x8 | yes | read | SS 0.9 V / 125 C | nominal | 6.0 | 3435 | - | - | - | - | pass |
| w_128x8_6t_SS_write | 6T_ | 128x8 | no | write | SS 0.9 V / 125 C | nominal | 5.25 | 3522 | 2541.9 | 282.9 | 570.5 | 1201.5 | pass |
| w_128x8_6t_mux_SS_read | 6T_ | 128x8 | yes | read | SS 0.9 V / 125 C | nominal | 5.5 | 3435 | - | - | - | - | pass |
| w_16x16_10t_SS_sequence | 10T | 16x16 | no | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 6939 | 1791.3 | 286.8 | 532.7 | - | pass |
| w_16x16_10t_SS_sequence_pd_s1 | 10T | 16x16 | no | read&write | SS 0.9 V / 125 C | per-device s1 | 5.0 | 6939 | 1766.9 | 283.0 | 549.5 | - | pass |
| w_16x16_10t_SS_sequence_pd_s2 | 10T | 16x16 | no | read&write | SS 0.9 V / 125 C | per-device s2 | 5.0 | 6939 | 1805.1 | 277.0 | 557.7 | - | pass |
| w_16x16_10t_SS_write_idle2 | 10T | 16x16 | no | write (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 1905 | 2316.3 | 286.1 | 532.4 | 609.2 | pass |
| w_16x16_10t_mux_SS_sequence | 10T | 16x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 6939 | 1785.2 | 286.5 | 534.4 | - | pass |
| w_16x16_6t_SS_sequence | 6T_ | 16x16 | no | read&write | SS 0.9 V / 125 C | nominal | 4.5 | 6939 | 1537.0 | 286.4 | 509.7 | - | pass |
| w_16x16_6t_mux_SS_sequence | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 4.75 | 6939 | 1659.6 | 285.4 | 511.8 | - | pass |
| w_16x16_6t_mux_SS_sequence_pd_s1 | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | per-device s1 | 4.75 | 6939 | 1619.7 | 277.3 | 512.8 | - | pass |
| w_16x16_6t_mux_SS_sequence_pd_s2 | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | per-device s2 | 4.75 | 6939 | 1635.3 | 288.2 | 527.5 | - | pass |
| w_16x16_6t_mux_SS_sequence_pd_s3 | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | per-device s3 | 4.75 | 6939 | 1663.3 | 290.1 | 519.4 | - | pass |
| w_16x16_6t_mux_SS_write_idle2 | 6T_ | 16x16 | yes | write (every 2) | SS 0.9 V / 125 C | nominal | 4.75 | 1905 | 2183.0 | 286.0 | 512.1 | 617.3 | pass |
| w_16x32_6t_mux_SS_sequence | 6T_ | 16x32 | yes | read&write | SS 0.9 V / 125 C | nominal | 4.75 | 13643 | 1648.0 | 287.0 | 517.2 | - | pass |
| w_16x64_6t_mux_SS_write | 6T_ | 16x64 | yes | write | SS 0.9 V / 125 C | nominal | 5.0 | 4530 | 2280.1 | 286.2 | 534.1 | 1220.4 | pass |
| w_256x4_10t_SS_read | 10T | 256x4 | no | read | SS 0.9 V / 125 C | nominal | 8.0 | 3647 | - | - | - | - | pass |
| w_256x4_6t_SF_write | 6T_ | 256x4 | no | write | SF 0.9 V / 125 C | nominal | 6.25 | 3690 | 3132.6 | 230.3 | 604.5 | 1171.0 | pass |
| w_256x4_6t_SS_write | 6T_ | 256x4 | no | write | SS 0.9 V / 125 C | nominal | 6.25 | 3690 | 3137.6 | 273.9 | 666.2 | 1302.0 | pass |
| w_256x4_6t_mux_SS_read | 6T_ | 256x4 | yes | read | SS 0.9 V / 125 C | nominal | 6.75 | 3647 | - | - | - | - | pass |
| w_32x16_10t_mux_SS_sequence | 10T | 32x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 11419 | 1745.0 | 286.8 | 537.4 | - | pass |
| w_32x16_6t_SS_sequence | 6T_ | 32x16 | no | read&write | SS 0.9 V / 125 C | nominal | 4.5 | 11419 | 1494.2 | 286.7 | 514.7 | - | pass |
| w_32x16_6t_mux_SS_sequence | 6T_ | 32x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 4.75 | 11419 | 1616.0 | 286.3 | 516.2 | - | pass |
| w_512x4_10t_mux_SS_read | 10T | 512x4 | yes | read | SS 0.9 V / 125 C | nominal | 10.0 | 1117 | - | - | - | - | pass |
| w_512x4_6t_mux_SS_read | 6T_ | 512x4 | yes | read | SS 0.9 V / 125 C | nominal | 9.0 | 1117 | - | - | - | - | pass |
| w_512x4_6t_mux_SS_write | 6T_ | 512x4 | yes | write | SS 0.9 V / 125 C | nominal | 9.0 | 1160 | 4775.5 | 275.4 | 980.0 | 1641.7 | pass |
| w_64x16_10t_mux_SF_write | 10T | 64x16 | yes | write | SF 0.9 V / 125 C | nominal | 5.5 | 3570 | 2601.9 | 246.9 | 491.7 | 1070.4 | pass |
| w_64x16_10t_mux_SS_read | 10T | 64x16 | yes | read | SS 0.9 V / 125 C | nominal | 5.5 | 3395 | - | - | - | - | pass |
| w_64x16_10t_mux_SS_write | 10T | 64x16 | yes | write | SS 0.9 V / 125 C | nominal | 5.5 | 3570 | 2598.4 | 285.9 | 548.7 | 1185.4 | pass |
| w_64x16_6t_mux_SF_write | 6T_ | 64x16 | yes | write | SF 0.9 V / 125 C | nominal | 5.0 | 3570 | 2347.4 | 246.5 | 473.4 | 1067.8 | pass |
| w_64x16_6t_mux_SS_read | 6T_ | 64x16 | yes | read | SS 0.9 V / 125 C | nominal | 5.0 | 3395 | - | - | - | - | pass |
| w_64x16_6t_mux_SS_write | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | nominal | 5.0 | 3570 | 2344.9 | 286.1 | 528.8 | 1182.7 | pass |
| w_64x16_6t_mux_SS_write_idle2 | 6T_ | 64x16 | yes | write (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 5889 | 2345.0 | 285.6 | 528.8 | 603.7 | pass |
| w_64x16_6t_mux_SS_write_pd_s1 | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | per-device s1 | 5.0 | 3570 | 2358.0 | 289.6 | 538.6 | 1202.9 | pass |
| w_64x16_6t_mux_SS_write_pd_s2 | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | per-device s2 | 5.0 | 3570 | 2352.5 | 291.0 | 559.1 | 1167.0 | pass |
| w_64x16_6t_mux_SS_write_pd_s3 | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | per-device s3 | 5.0 | 3570 | 2341.7 | 287.7 | 543.6 | 1206.8 | pass |
| w_8x128_6t_mux_SS_read | 6T_ | 8x128 | yes | read | SS 0.9 V / 125 C | nominal | 6.0 | 4515 | - | - | - | - | pass |
| w_8x4_10t_FF_cold_sequence | 10T | 8x4 | no | read&write | FF 1.1 V / -40 C | nominal | 5.0 | 1303 | 2340.6 | 72.5 | 148.6 | - | pass |
| w_8x4_10t_mux_SS_sequence | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 1303 | 1818.6 | 272.8 | 541.9 | - | pass |
| w_8x4_10t_mux_SS_sequence_pd_s1 | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | per-device s1 | 5.0 | 1303 | 1772.6 | 280.5 | 554.2 | - | pass |
| w_8x4_10t_mux_SS_sequence_pd_s2 | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | per-device s2 | 5.0 | 1303 | 1814.1 | 254.5 | 543.2 | - | pass |
| w_8x4_10t_mux_SS_sequence_pd_s3 | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | per-device s3 | 5.0 | 1303 | 1795.3 | 274.3 | 561.6 | - | pass |
| w_8x4_6t_FF_cold_sequence | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | nominal | 4.5 | 1303 | 2086.5 | 72.5 | 142.2 | - | pass |
| w_8x4_6t_FF_cold_sequence_pd_s1 | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | per-device s1 | 4.5 | 1303 | 2081.1 | 71.6 | 139.3 | - | pass |
| w_8x4_6t_FF_cold_sequence_pd_s2 | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | per-device s2 | 4.5 | 1303 | 2080.5 | 75.2 | 142.7 | - | pass |
| w_8x4_6t_FF_cold_sequence_pd_s3 | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | per-device s3 | 4.5 | 1303 | 2089.2 | 71.8 | 144.3 | - | pass |
| w_8x4_6t_FF_cold_write_idle3 | 6T_ | 8x4 | no | write (every 3) | FF 1.1 V / -40 C | nominal | 4.5 | 341 | 2246.4 | 72.3 | 142.2 | 150.2 | pass |
| w_8x4_6t_SS_sequence | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | nominal | 4.5 | 1303 | 1569.7 | 273.0 | 516.0 | - | pass |
| w_8x4_6t_SS_sequence_pd_s1 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s1 | 4.5 | 1303 | 1510.1 | 268.7 | 502.8 | - | pass |
| w_8x4_6t_SS_sequence_pd_s2 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s2 | 4.5 | 1303 | 1499.0 | 306.2 | 537.6 | - | pass |
| w_8x4_6t_SS_sequence_pd_s3 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s3 | 4.5 | 1303 | 1575.3 | 272.6 | 531.0 | - | pass |
| w_8x4_6t_SS_sequence_pd_s4 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s4 | 4.5 | 1303 | 1564.7 | 240.2 | 529.9 | - | pass |
| w_8x4_6t_SS_sequence_pd_s5 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s5 | 4.5 | 1303 | 1554.3 | 279.3 | 533.3 | - | pass |
| w_8x4_6t_SS_write_idle2_pd_s1 | 6T_ | 8x4 | no | write (every 2) | SS 0.9 V / 125 C | per-device s1 | 4.5 | 341 | 2107.5 | 265.5 | 514.7 | 545.9 | pass |
| w_8x4_6t_SS_write_idle2_pd_s2 | 6T_ | 8x4 | no | write (every 2) | SS 0.9 V / 125 C | per-device s2 | 4.5 | 341 | 2107.6 | 301.3 | 511.5 | 530.9 | pass |
| w_8x4_6t_SS_write_idle2_pd_s3 | 6T_ | 8x4 | no | write (every 2) | SS 0.9 V / 125 C | per-device s3 | 4.5 | 341 | 2140.6 | 310.7 | 533.2 | 530.1 | pass |
| w_8x4_6t_TT_write_idle2 | 6T_ | 8x4 | no | write (every 2) | TT 1.0 V / 25 C | nominal | 4.5 | 341 | 2220.2 | 119.5 | 229.7 | 237.2 | pass |
| w_8x8_6t_mux_FF_cold_sequence | 6T_ | 8x8 | yes | read&write | FF 1.1 V / -40 C | nominal | 4.75 | 2435 | 2217.2 | 75.5 | 143.4 | - | pass |

## 6. Limits

* Screening at the class bounds with illustrative wires; extracted metal,
  half-select writes (a column mask does not exist in this architecture: every
  column is written) and the yield estimator remain the carried Phase 6 scope.
* The idle → write probe uses `select_every` with the write data toggling
  between selected cycles; an idle cycle leaves the bitlines floating at their
  previous levels, as in every earlier release (the precharge needs `cs`).
* `precharge_off_tau` still describes the PRE line; the write slot reuses the
  same settled `pre_off_ready`, so a wire change that moves the guard moves the
  slot with it.
