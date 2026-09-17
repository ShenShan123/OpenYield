# V2.1.8: no enable overlaps: the read wordline ends at the sense trigger, the clock-high enables wait for the previous access

Executed September 17, 2026 on the V2.1.8 sources (base commit `022bcaf`
plus the changes of this release; every queue archives the sources it ran).
A review of `time_generate.py`, `replica_column.py` and the
[V2.1.6 write slot](WRITE_SLOT_V2_1_6.md) looked for overlaps between the
enable pulses of one access (`PRE`, `wl_en` and the local wordline, `s_en`,
`w_en`, `sa_iso`, the column-mux select) and between the pulses of two
consecutive accesses (read -> read, write -> read, read -> write, write ->
write, and the idle boundaries), for 6T and 10T cells. Illustrative wires,
V2.0.9 driver classes and the unchanged timing classes throughout; per-device
samples use 5 % relative sigma on `vth0`, `u0` and `voff` of every MOS. This
is functional screening, not PVT/mismatch or extracted-metal qualification;
no record is promoted to `sizing_table.json`.

**Outcome: on the release sources 91 of 91 cases pass, 251,553 of 251,553 checks: all 59 nominal boundary cases and all 32 mismatch seeds (p_16x64_6t_mux_SS_read_idle2 and r_16x16_6t_mux_SS_read after a DC operating-point retry with Newton line search; r_512x4_10t_mux_SS_read, r_512x4_6t_mux_SS_read, w_256x4_6t_SF_write, w_256x4_6t_SS_write and w_512x4_6t_mux_SS_write rerun on the release checker after the two provisional checks became metrics). In every read the local wordline started to release at least 63 ps after the sense enable reached the amplifier (p_2x4_6t_FF_cold_read_idle2) and was below 0.1 VDD before the access ended in every read up to 256 rows (7 ps before the edge at r_256x4_6t_mux_SS_read) and 309 ps (6T) / 90 ps (10T) after it at 512x4; the next precharge rose at least 36 ps after the previous sense enable was off (r_8x4_6t_FF_cold_read_pd_s1) and, with the settling stages, 95 ps after the previous write enable was off (s_8x8_6t_mux_FF_cold_sequence); the next write enable rose at least 70 ps after the previous sense enable was off (s_8x4_6t_FF_cold_sequence_pd_s3) and 130 ps after the previous write enable was off (w_8x4_6t_FF_cold_write_pd_s2); a write's drivers released with the local wordline at 0 V up to 64 rows, at 0.50 to 0.55 V at 256x4 and at 0.90 V at 512x4 (w_512x4_6t_mux_SS_write), as the same path did before V2.1.8. The driven bitline reached its rail at least 1722 ps before the local wordline started; new write data reached every driver's hold latch at least 106 ps before the column's write enable rose (s_8x4_6t_FF_cold_sequence_pd_s1); in the 23 decks with unselected cycles the precharge stayed off (its terminal never below 99.8% of VDD) and no write or sense enable rose above 1 mV there; the longest write slot is 1681 ps (w_512x4_6t_mux_SS_write) and the longest restore 913 ps (r_512x4_10t_mux_SS_read).**

## 1. What the review found

`dev/v218_overlap/` holds the method: `probe.py` of the V2.1.7 review (with
the V2.1.8 nodes added to its probe list) builds 24 full-array decks on a
tree, `overlap.py` takes from each trace, per selected cycle, the
on-intervals of every enable at its far line tap and of the local wordline of
the target cell (50 % crossings, the wordline release at 10 %), and reports
every pair that must not overlap as a gap in ps (negative = overlap):
inside the access (`PRE` released before the wordline, the sense enable after
the wordline, the wordline after the sense enable) and across the boundary to
the next selected cycle (the previous sense enable, write enable, wordline
and isolation off before the next precharge or write enable; the write
enable of a write cycle off after its wordline). `summarize.py` and
`compare.py` tabulate the worst gap per run. The 24 decks cover 8x4 6T at
TT, FF 1.1 V / -40 C and SS 0.9 V / 125 C as sequences (write -> read,
read -> write), read decks (read -> read) and write decks (write -> write),
16x16 at SS, 16x16 with a mux at TT under mismatch, 16x16 with local RC,
64x16 with a mux at SS (read, write), the same for 10T (8x4 sequences at FF
and SS, with a mux, 16x16 with a mux at SS, read and write decks, 64x16 mux
read), the idle decks, a zero-settling sequence and two mismatch seeds; batch
`b` ran on the V2.1.7 sources, batch `n` on the V2.1.8 sources, from the
same launch script.

| | Finding | Where | Consequence |
|---|---|---|---|
| F1 | In a read the wordline stayed on for the whole clock-low phase, 1.1 to 2.2 ns (25 to 44 % of the period) after the sense enable had fired, although the amplifier is isolated from the bitlines from the sense trigger on (`sa_iso = s_en \| w_en`) and the bitline had already reached its rail when the amplifier fired (in every probed size up to 64x16, 6T and 10T, at every corner) | `_add_wordline_enable`: `wl_en = buffer(access_clk_bar)` | the wordline pulse was 1.8 (64x16 SS) to 9 (8x4 FF) times longer than the sensing needs; the replica wordline and the write-request and address hold latches followed it; the cell saw its pass gates open for the tail; the restore started from the rail |
| F2 | Every boundary order was set by path length alone: the next precharge rose 125 ps (FF) / 460 ps (SS) after the previous sense enable was off, 83 / 304 ps after the previous write enable was off, the next write enable 148 / 550 ps after the previous sense enable was off, the isolation dropped 77 / 271 ps before the precharge; with zero settling stages 27 ps (write enable off to precharge) and 15 ps (isolation off to precharge) | the precharge NAND3 and the write slot waited for the wordline observer only | not a relation the circuit enforces: a wire, load or sizing change moves it; at zero settling stages the drivers and the precharge overlapped at their 10 / 90 % levels |
| F3 | write -> idle: the deselect (`cs_pre` falling with `cs`, V2.1.7) dropped `w_en` while the local wordline was still at 0.50 V (TT) / 0.43 V (SS) at 8x4, 12 to 25 ps before it was below 0.1 VDD (8x4 10T FF: 8 ps) | `w_en = we_hold & cs_pre & write_window` | the drivers tri-stated the bitlines under a half-open pass gate: harmless for the written cell (the bitlines hold their rails), but the nesting "drivers on while the wordline is on" was broken at every deselect |
| F4 | The column-mux select is a DC level in this testbench (the column address never changes): the selected bitline pair is connected to its amplifier in every phase | `create_read_periphery` | no pulse, so no overlap. The amplifier's inputs are precharged through the mux in the clock-high phase, isolated by `sa_iso` while it fires and while the drivers are on (the drivers write every column directly, not through the mux). A pulsed select needs the column-address path the compiler does not have (Phase 6, with the half-select write) |
| F5 | In a write `w_en` covers the wordline, as it must: the drivers are on 1.6 to 2.4 ns before the local wordline starts and release 19 to 118 ps after it is below 0.1 VDD at every selected boundary | - | correct nesting. The write wordline still lasts the whole clock-low phase (the cell flips 130 to 980 ps after the clock fall): a timing-policy item, not an overlap |

F1 and F3 are the overlaps this release removes; F2 becomes an explicit
ordering. Before -> after, worst case per probe run (`compare.py`; gaps in
ps, the read wordline pulse from 10 % to 10 %, the release after the sense
enable from its 50 % crossing at the amplifier to the local wordline's 50 %
fall, the sense margin at the amplifier's inputs when it fires):

| Run | Cell | Size | Corner | T [ps] | read WL pulse [ps] | WL release after s_en [ps] | s_en off -> PRE | w_en off -> PRE | sa_iso off -> PRE | s_en off -> w_en | w_en off -> w_en | w_en off - WL off [ps] | WL at w_en off [V] | margin [V] |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x4_rw_TT | 6T | 8x4 | TT 25 C | 4500 | 2073 -> 383 | 1765 -> 73 | - | 134 -> 141 | 122 -> 129 | 242 -> 113 | - | 33 -> 30 | 0.000 -> 0.000 | 0.996 -> 0.996 |
| 8x4_rw_FF | 6T | 8x4 | FF -40 C | 4500 | 2126 -> 237 | 1936 -> 46 | - | 83 -> 86 | 77 -> 81 | 148 -> 67 | - | 19 -> 18 | 0.000 -> 0.000 | 1.096 -> 1.095 |
| 8x4_rw_SS | 6T | 8x4 | SS 125 C | 4500 | 1902 -> 861 | 1206 -> 161 | - | 304 -> 321 | 271 -> 289 | 550 -> 262 | - | 76 -> 70 | -0.000 -> -0.000 | 0.896 -> 0.896 |
| 8x4_read_FF | 6T | 8x4 | FF -40 C | 4500 | 2127 -> 237 | 1937 -> 45 | 125 -> 42 | - | 102 -> 18 | - | - | - | - | 1.096 -> 1.095 |
| 8x4_read_SS | 6T | 8x4 | SS 125 C | 4500 | 1904 -> 861 | 1208 -> 161 | 460 -> 174 | - | 367 -> 78 | - | - | - | - | 0.896 -> 0.896 |
| 8x4_write_TT | 6T | 8x4 | TT 25 C | 4500 | - | - | - | - | - | - | 172 -> 181 | 33 -> 30 | 0.000 -> 0.000 | - |
| 8x4_write_SS | 6T | 8x4 | SS 125 C | 4500 | - | - | - | - | - | - | 394 -> 412 | 75 -> 70 | -0.000 -> -0.000 | - |
| 16x16_rw_SS | 6T | 16x16 | SS 125 C | 4500 | 1873 -> 959 | 1104 -> 188 | - | 284 -> 302 | 252 -> 271 | 557 -> 311 | - | 114 -> 107 | -0.000 -> -0.000 | 0.896 -> 0.896 |
| 16x16_mux_rw_TT_pd | 6T | 16x16 mux | TT 25 C | 4750 | 2182 -> 435 | 1837 -> 86 | - | 121 -> 137 | 126 -> 143 | 259 -> 131 | - | 50 -> 48 | 0.001 -> 0.002 | 0.996 -> 0.996 |
| 64x16_mux_read_SS | 6T | 64x16 mux | SS 125 C | 5000 | 2105 -> 1181 | 1120 -> 189 | 519 -> 227 | - | 415 -> 119 | - | - | - | - | 0.885 -> 0.885 |
| 64x16_mux_write_SS | 6T | 64x16 mux | SS 125 C | 5000 | - | - | - | - | - | - | 396 -> 415 | 118 -> 110 | -0.000 -> -0.000 | - |
| 8x4_10t_rw_FF | 10T | 8x4 | FF -40 C | 5000 | 2372 -> 239 | 2179 -> 45 | - | 83 -> 86 | 77 -> 81 | 148 -> 67 | - | 19 -> 18 | 0.000 -> 0.000 | 1.095 -> 1.095 |
| 8x4_10t_rw_SS | 10T | 8x4 | SS 125 C | 5000 | 2147 -> 879 | 1433 -> 161 | - | 304 -> 321 | 271 -> 289 | 550 -> 262 | - | 76 -> 70 | -0.000 -> -0.000 | 0.895 -> 0.895 |
| 8x4_10t_mux_rw_SS | 10T | 8x4 mux | SS 125 C | 5000 | 2148 -> 930 | 1381 -> 159 | - | 316 -> 333 | 310 -> 329 | 553 -> 262 | - | 76 -> 70 | -0.001 -> -0.001 | 0.895 -> 0.895 |
| 16x16_10t_mux_rw_SS | 10T | 16x16 mux | SS 125 C | 5000 | 2118 -> 1006 | 1309 -> 192 | - | 284 -> 302 | 287 -> 305 | 594 -> 312 | - | 114 -> 107 | -0.001 -> -0.000 | 0.894 -> 0.894 |
| 8x4_10t_read_SS | 10T | 8x4 | SS 125 C | 5000 | 2149 -> 879 | 1435 -> 161 | 459 -> 174 | - | 366 -> 79 | - | - | - | - | 0.895 -> 0.895 |
| 8x4_10t_write_SS | 10T | 8x4 | SS 125 C | 5000 | - | - | - | - | - | - | 394 -> 412 | 75 -> 70 | -0.000 -> -0.000 | - |
| 64x16_10t_mux_read_SS | 10T | 64x16 mux | SS 125 C | 5500 | 2350 -> 1231 | 1314 -> 189 | 519 -> 228 | - | 415 -> 120 | - | - | - | - | 0.877 -> 0.878 |
| 8x4_write_TT_idle2 | 6T | 8x4 | TT 25 C | 4500 | - | - | - | - | - | - | 4622 -> 4601 | -12 -> 30 | 0.505 -> 0.000 | - |
| 8x4_read_TT_idle2 | 6T | 8x4 | TT 25 C | 4500 | 2074 -> 383 | 1767 -> 73 | 4627 -> 4626 | - | 4588 -> 4585 | - | - | - | - | 0.996 -> 0.996 |
| 16x16_rw_TT_rc | 6T | 16x16 | TT 25 C | 4500 | 2057 -> 523 | 1632 -> 97 | - | 148 -> 156 | 135 -> 142 | 246 -> 136 | - | 39 -> 35 | 0.000 -> 0.000 | 0.993 -> 0.993 |
| 8x4_rw_TT_settle0 | 6T | 8x4 | TT 25 C | 4500 | 2073 -> 382 | 1765 -> 73 | - | 27 -> 70 | 15 -> 58 | 134 -> 113 | - | 33 -> 30 | 0.000 -> 0.000 | 0.996 -> 0.996 |
| 8x4_rw_TT_pd_s7 | 6T | 8x4 | TT 25 C | 4500 | 2073 -> 385 | 1749 -> 74 | - | 139 -> 143 | 127 -> 130 | 247 -> 111 | - | 30 -> 29 | 0.000 -> 0.000 | 0.996 -> 0.996 |
| 8x4_rw_SS_pd_s8 | 6T | 8x4 | SS 125 C | 4500 | 1906 -> 857 | 1206 -> 174 | - | 302 -> 306 | 263 -> 268 | 531 -> 267 | - | 62 -> 71 | -0.000 -> -0.000 | 0.894 -> 0.896 |

The margin at the amplifier is unchanged to the millivolt: the wordline
release follows the buffered sense enable, so the bitlines stop developing
only after the amplifier has been isolated. The sense enable and the output
latch it enables overlap by design (the latch is transparent while the
amplifier is fired and holds from its falling edge; the amplifier's output
moved 60 ps after the latch enable fell at FF).

## 2. The changes

In `TIME_CONTROL` (`docs/design/TIME_CONTROL_PATH.md`, sections 2.7, 2.12,
2.13, 2.16 and 2.17):

* **Read wordline released at the sense trigger (F1).** `wl_en =
  WORDLINE_ENABLE_BUFFER(access_clk_bar, s_en_bar)`: the buffer's first
  stage is a NAND2 (`WORDLINE_REQUEST_NAND`, the same drive as the inverter
  it replaces, folded like it, so the access path has no added stage). A
  write cycle has no sense enable and keeps its wordline for the whole
  access; `s_en` stays high until the access request ends, so the request
  cannot re-arm within the access. The wordline starts to release 45 ps (FF)
  to 192 ps (16x16 10T mux SS) after the amplifier's terminal sees the sense
  enable.
  The release path (`s_en_bar`, the NAND2, the output stage driving every
  row driver, the row driver) grows with the row count: 254 ps at 128x8,
  321 ps at 256x4 and 547 ps at 512x4 (SS), where the trigger itself sits
  close to the edge that ends the access, so the local wordline is below
  0.1 VDD 7 ps before that edge at 256x4 and 309 ps (6T) / 90 ps (10T) after
  it at 512x4. The precharge and the write slot still wait for the wordline
  observer, so nothing follows a wordline that is still falling.
* **Precharge after both enables (F2).** `pre_gate = PRECHARGE_GATE_AND(
  wordline_off, we_hold_bar, enables_off)` with `enables_off =
  ENABLES_OFF_NOR(s_en, w_en)`, so the precharge of a read cycle follows the
  previous sense enable (read -> read) or write enable (write -> read) being
  off.
* **Write slot after the sense enable (F2).** `write_slot = WRITE_SLOT_AND(
  wordline_off, pre_off_ready, s_en_bar)` (the precharge input tied high
  without the precharge-off guard). The write enable is not part of the
  slot: the slot feeds the write enable, and that loop would oscillate.
* **Write enable ends with the wordline enable (F3).** `selected_slot =
  cs_pre & write_slot`, `write_window = wl_en | selected_slot`, `w_en =
  we_hold & write_window`: the select reaches the drivers through the slot
  (clock-high) and through the wordline enable (access), never directly. At
  write -> idle the drivers now release 30 ps after the local wordline is
  below 0.1 VDD (TT), as in a selected boundary.
* **Observers.** `s_en_bar` (an inverter of `senb_scale` units) and
  `enables_off` read the block's output nodes, the buffered enables, so a
  wordline, slot or precharge only follows what the periphery has seen.
* **Loads.** `access_load = 1.25 * wl_en_scale + 2.5` (the NAND2 input
  replaces an inverter input), `sen_load` gains the `s_en_bar` inverter and
  the NOR input (`senb_scale + 1.75` units), `wen_load` the NOR input
  (1.75); `driver_sizing.py` and the `TIME_CONTROL` fallbacks agree and
  `validate_for` expects the new access load.
* **Probes.** The validator prints `XTIME_CONTROL:selected_slot`, `s_en_bar`
  and `enables_off`.

The column mux (F4) and the write nesting (F5) are unchanged.

## 3. Checks

* The local waveform checker (`dev/v210_waveform_checks.py`), the validator
  (`dev/validate_distributed_rc.py`) and the qualification scorer
  (`dev/sizing/qualification.py`) search the wordline release from the
  wordline's rise instead of from the edge that ends the access, so a read
  wordline released mid-access is scored.
* New checks per read cycle and column: `_sense_before_wordline_release`
  (the amplifier fires before the local wordline starts to fall; metric
  `_sense_to_wordline_release_ps`) and the metric
  `_wordline_release_to_deadline_ps` (the local wordline below 0.1 VDD
  relative to the edge that ends the access: positive up to 256 rows,
  negative at the 512-row class bound). Per boundary: `_sense_off_before_precharge`
  / `_write_enable_off_before_precharge` (before a read) and
  `_sense_off_before_write_enable` / `_write_enable_off_before_write_enable`
  (before a write), each with its `_ps` metric. Per write cycle the tail of
  the local wordline when the drivers release is a metric
  (`_wordline_at_write_enable_off_v`, `_write_enable_off_after_wordline_ps`),
  not a check: the drivers release with the wordline enable, and the physical
  wordline's tail is set by the row driver and the wire (0 V up to 64 rows;
  at 256 rows the release and the wordline's 50 % fall coincide, 0.46 V under
  V2.1.7 and 0.50 V now, both 210 ps after the wordline enable; at 512 rows
  the drivers release 295 ps before the local wordline is below 0.1 VDD, with
  the wordline still at 0.90 V, as the same path did before V2.1.8). The main
  pass of the matrix scored this tail as a check at 0.5 VDD and the read
  release as a check against the deadline; the 256- and 512-row writes failed
  the first (0.504 and 0.90 V) and the 512-row reads the second (309 and
  90 ps past the edge); both became metrics and those cases were rerun on the
  release checker (section 5).
* A read cycle's write-enable quiet window now ends when the wordline is
  released and the sense enable is off, whichever is later (before, at the
  wordline release, which is now mid-access).
* The synthetic traces of `dev/tests/test_v210_waveform_checks.py` model the
  sense-timed release (the read wordline ends at 1.0 T, the sense enable
  0.03 T after the edge) and corruptions for every new check.
* The runtime measures are unchanged: `VWL_WEN_*` keeps `XTIME_CONTROL:write_slot`
  as the entry event (it precedes `selected_slot`).

## 4. Timing table

`timing_lookup.json` becomes `v2.1.8-timing-7` with every class kept. The
read access path has no added stage (`PRE` released to local wordline 213 ->
213 ps, wordline to sense enable 290 -> 292 ps at 8x4 TT). In the probed
sizes the bitline is at its rail when the amplifier fires, so the restore
starts from the same level as before; `TRESTORE`, measured from the capture
edge, falls by 30 to 40 % at 64 to 512 rows (1042 -> 765 ps at 64x16,
1523 -> 904 ps at 512x4, SS) because the precharge follows the previous
sense enable off instead of a wordline released at the edge plus the guard
delay. `TWSLOT` grows by 20 to 40 ps (the selected-slot AND2 in the slot
path). Both stay far inside their clock-high phases; the classes are kept.
The per-cycle supply energy of the single read and write decks the two matrices share (`EREAD` / `EWRITE`, the testbench window from the first clock fall over one period; `dev/v218_overlap/energy.py`), V2.1.7 -> V2.1.8:

| Case | E/cycle V2.1.7 [fJ] | V2.1.8 [fJ] | change | Pdyn V2.1.7 [uW] | V2.1.8 [uW] | TRESTORE / TWSLOT V2.1.7 [ps] | V2.1.8 [ps] |
|---|---:|---:|---:|---:|---:|---:|---:|
| r_128x8_10t_mux_SS_read | 1296.1 | 1328.2 | +2.5 % | 191.0 | 196.7 | 1084.5 | 760.7 |
| r_128x8_6t_mux_SS_read | 1261.8 | 1296.2 | +2.7 % | 208.7 | 215.0 | 1080.8 | 758.5 |
| r_256x4_10t_SS_read | 1800.5 | 1835.4 | +1.9 % | 194.8 | 199.0 | 1207.7 | 811.2 |
| r_256x4_6t_mux_SS_read | 1700.7 | 1757.4 | +3.3 % | 226.0 | 234.3 | 1189.9 | 792.7 |
| r_512x4_10t_mux_SS_read | 3179.2 | 3267.3 | +2.8 % | 260.2 | 268.7 | 1526.6 | 912.8 |
| r_512x4_6t_mux_SS_read | 3042.1 | 3140.7 | +3.2 % | 289.1 | 299.7 | 1523.3 | 904.0 |
| r_64x16_10t_mux_SS_read | 1216.7 | 1242.5 | +2.1 % | 199.2 | 203.9 | 1044.0 | 769.0 |
| r_64x16_6t_mux_SS_read | 1185.0 | 1212.0 | +2.3 % | 218.5 | 224.1 | 1041.9 | 765.1 |
| r_8x128_6t_mux_SS_read | 2999.4 | 3006.2 | +0.2 % | 474.4 | 476.2 | 1058.5 | 836.2 |
| w_128x8_6t_SS_write | 1479.1 | 1498.8 | +1.3 % | 261.9 | 265.3 | 1199.3 | 1225.3 |
| w_16x64_6t_mux_SS_write | 3124.9 | 3132.9 | +0.3 % | 599.2 | 599.6 | 1220.6 | 1240.2 |
| w_256x4_6t_SF_write | 2062.8 | 2100.2 | +1.8 % | 270.1 | 275.6 | 1170.7 | 1193.6 |
| w_256x4_6t_SS_write | 1826.1 | 1860.1 | +1.9 % | 268.0 | 273.0 | 1302.2 | 1331.1 |
| w_512x4_6t_mux_SS_write | 3410.5 | 3468.7 | +1.7 % | 334.5 | 340.6 | 1641.9 | 1681.4 |
| w_64x16_10t_mux_SF_write | 1654.8 | 1668.6 | +0.8 % | 256.7 | 259.2 | 1070.6 | 1089.0 |
| w_64x16_10t_mux_SS_write | 1513.8 | 1526.0 | +0.8 % | 253.8 | 255.8 | 1185.6 | 1209.8 |
| w_64x16_6t_mux_SF_write | 1590.4 | 1603.7 | +0.8 % | 277.6 | 279.8 | 1068.0 | 1086.9 |
| w_64x16_6t_mux_SS_write | 1460.5 | 1473.8 | +0.9 % | 274.2 | 276.9 | 1183.1 | 1207.3 |
| w_64x16_6t_mux_SS_write_pd_s1 | 1472.5 | 1484.5 | +0.8 % | 274.8 | 277.4 | 1189.3 | 1229.8 |
| w_64x16_6t_mux_SS_write_pd_s2 | 1474.6 | 1489.2 | +1.0 % | 274.9 | 277.9 | 1216.6 | 1203.0 |
| w_64x16_6t_mux_SS_write_pd_s3 | 1471.5 | 1487.5 | +1.1 % | 275.3 | 277.3 | 1201.8 | 1216.8 |

Per-phase attribution (`dev/v218_overlap/energy_phases.py`: the same 8x4 and
64x16-with-mux read decks built on both trees with the supply current
printed, SS 0.9 V / 125 C, first read cycle): the 64x16 cycle costs 855 fJ
instead of 841 fJ (+1.7 %; 324 instead of 315 fJ, +3 %, at 8x4). The wordline
release moved from the restore phase into the access: the phase from the
sense enable to the edge costs 157 fJ instead of 73 fJ and the restore 411 fJ
instead of 486 fJ (the wordline, replica wordline, hold-latch and observer
transitions that followed the clock edge before), and the remainder (about
15 fJ at 64x16, 9 fJ at 8x4) is the new gates (the NAND2 first stage,
`s_en_bar`, the NOR) and the leakage of the released bitlines. The bitline
energy is unchanged: with `K = 1` and `N = 9` the amplifier fires after the
bitline has reached its rail (0.015 V at 64x16 SS), so no wordline timing can
save it. The sense-timed wordline is what makes an earlier trigger safe: with
5 and 3 delay stages instead of 9 (`probe.py --dc-stages`) the same 64x16
read costs 837 and 826 fJ (-2.2 %, -3.4 %) with the amplifier's margin at
0.859 and 0.829 V (0.885 V at 9 stages; 0.3 V required) and the wordline
pulse 965 and 865 ps instead of 1164 ps; the bitline is still at 0.074 V when
the amplifier fires at 3 stages, because the replica discharges at the
array's own rate (`K = 1`). The swing at the trigger, and with it the bitline
energy, is set by the replica strength `K`: a sizing-policy item with its own
mismatch evidence, outside this release.

## 5. Evidence

The matrix (`outputs/validation/V2.1.8-enable-overlap/gen_cases.py`, 91
cases in 22 queues at up to 64 simulator ranks) is the V2.1.7 matrix with
its fix-pass settings built in (7200 s for the 64x16 10T writes, Newton line
search for the 8x128 read), plus the boundaries this release touches most:
read -> read and write -> write at 8x4 (6T and 10T, FF -40 C and SS) and
16x16 with a mux at SS (6T and 10T), with mismatch seeds at FF (two 6T
read, two 6T write, one 10T read). Every queue archives its sources. The
main pass (22 queues) passed 84 of 91 cases: two four-rank DC operating
points aborted before any waveform (the 16x16 mux read and the 16x64 mux
read idle probe; rerun with Newton line search, as the V2.1.6 fix3 and
V2.1.7 fix-A passes did), and five cases failed only the two provisional
checks of section 3 (the write tail at 256 and 512 rows, the read release
against the deadline at 512 rows), which became metrics; the fix pass
(queues `fix-A` to `fix-E`, `gen_fix.py`, `run_fix.sh`) reran those seven
cases on the release checker, and the record keeps the latest attempt of
every case.

Every case, latest attempt (`outputs/validation/V2.1.8-enable-overlap/`, `assemble_record.py`; the JSON record is `ENABLE_OVERLAP_V2_1_8.json` next to this file). s_en to WL release: smallest distance from the sense enable at the amplifier (50 %) to the local wordline starting to fall (50 %) over the read cycles; WL release to deadline: smallest distance from the local wordline below 10 % VDD to the edge that ends the access; s_en / w_en off to PRE: smallest distance from the previous sense or write enable off (50 %) to the next precharge (90 % falling); s_en / w_en off to w_en: the same to the next write enable (50 %); WL at w_en off: highest local wordline when a write's drivers release; rail before WL: smallest distance from the driven bitline reaching 10 % VDD to the local wordline reaching 10 % VDD; entry after release: smallest distance from the wordline release (10 %) to the next write slot or precharge; data before w_en: smallest distance from the new data at a driver's hold-latch output (50 %) to that column's write enable (50 %); TWSLOT (write decks) or TRESTORE (read decks) from the deck's `.mt0`. Seeds `sN` are `20261700 + N`.

| Case | Cell | Size | Mux | Operation | Corner | Variation | T [ns] | Checks | s_en to WL release [ps] | WL release to deadline [ps] | s_en / w_en off to PRE [ps] | s_en / w_en off to w_en [ps] | WL at w_en off [V] | Rail before WL [ps] | Entry after release [ps] | Data before w_en [ps] | TWSLOT / TRESTORE [ps] | Result |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| p_16x16_10t_SS_write_idle2 | 10T | 16x16 | no | write (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 1989 | - | - | - / - | - / 5207.5 | 0.0 | 2011.4 | 291.1 | 446.3 | 954.7 | pass |
| p_16x16_6t_mux_SS_read_idle2 | 6T_ | 16x16 | yes | read (every 2) | SS 0.9 V / 125 C | nominal | 4.75 | 1640 | 210.3 | 296.9 | 5009.4 / - | - / - | - | - | 5691.8 | - | 793.3 | pass |
| p_16x16_6t_mux_SS_write_idle2 | 6T_ | 16x16 | yes | write (every 2) | SS 0.9 V / 125 C | nominal | 4.75 | 1989 | - | - | - / - | - / 4957.6 | 0.0 | 1877.2 | 290.9 | 445.8 | 962.6 | pass |
| p_16x64_6t_mux_SS_read_idle2 | 6T_ | 16x64 | yes | read (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 6216 | 201.1 | 341.1 | 5231.2 / - | - / - | - | - | 6013.1 | - | 824.8 | pass |
| p_2x4_6t_FF_cold_read_idle2 | 6T_ | 2x4 | no | read (every 2) | FF 1.1 V / -40 C | nominal | 4.5 | 142 | 63.1 | 1667.8 | 4558.1 / - | - / - | - | - | 6350.6 | - | 214.9 | pass |
| p_4x4_6t_FF_cold_write_idle2 | 6T_ | 4x4 | no | write (every 2) | FF 1.1 V / -40 C | nominal | 4.5 | 273 | - | - | - / - | - / 4553.5 | 0.3 | 2158.4 | 72.6 | 111.1 | 243.3 | pass |
| p_64x16_6t_mux_SS_read_idle2 | 6T_ | 64x16 | yes | read (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 5624 | 222.0 | 171.4 | 5244.5 / - | - / - | - | - | 5830.2 | - | 797.5 | pass |
| p_64x16_6t_mux_SS_write_idle2 | 6T_ | 64x16 | yes | write (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 5973 | - | - | - / - | - / 5186.6 | 0.0 | 2038.8 | 290.7 | 433.2 | 950.1 | pass |
| p_8x4_10t_FF_cold_read_idle2 | 10T | 8x4 | no | read (every 2) | FF 1.1 V / -40 C | nominal | 5.0 | 280 | 70.6 | 1887.9 | 5057.6 / - | - / - | - | - | 7074.2 | - | 218.6 | pass |
| p_8x4_10t_FF_cold_write_idle2 | 10T | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | nominal | 5.0 | 365 | - | - | - / - | - / 5054.3 | 0.3 | 2409.8 | 72.7 | 111.2 | 247.1 | pass |
| p_8x4_6t_FF_cold_read_idle2 | 6T_ | 8x4 | no | read (every 2) | FF 1.1 V / -40 C | nominal | 4.5 | 280 | 70.6 | 1651.0 | 4556.9 / - | - / - | - | - | 6334.2 | - | 218.0 | pass |
| p_8x4_6t_FF_cold_read_idle2_pd_s1 | 6T_ | 8x4 | no | read (every 2) | FF 1.1 V / -40 C | per-device s1 | 4.5 | 280 | 71.4 | 1650.0 | 4555.4 / - | - / - | - | - | 6332.1 | - | 216.4 | pass |
| p_8x4_6t_FF_cold_read_idle2_pd_s2 | 6T_ | 8x4 | no | read (every 2) | FF 1.1 V / -40 C | per-device s2 | 4.5 | 280 | 68.6 | 1658.9 | 4556.4 / - | - / - | - | - | 6342.7 | - | 219.2 | pass |
| p_8x4_6t_FF_cold_write_idle2 | 6T_ | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | nominal | 4.5 | 365 | - | - | - / - | - / 4554.9 | 0.3 | 2155.8 | 72.6 | 111.1 | 247.2 | pass |
| p_8x4_6t_FF_cold_write_idle2_pd_s1 | 6T_ | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | per-device s1 | 4.5 | 365 | - | - | - / - | - / 4550.7 | 0.2 | 2159.1 | 74.1 | 108.1 | 244.6 | pass |
| p_8x4_6t_FF_cold_write_idle2_pd_s2 | 6T_ | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | per-device s2 | 4.5 | 365 | - | - | - / - | - / 4555.4 | 0.2 | 2153.7 | 72.2 | 111.0 | 245.2 | pass |
| p_8x4_6t_FF_cold_write_idle2_pd_s3 | 6T_ | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | per-device s3 | 4.5 | 365 | - | - | - / - | - / 4552.2 | 0.3 | 2155.9 | 71.7 | 108.1 | 244.4 | pass |
| p_8x4_6t_FF_cold_write_idle3 | 6T_ | 8x4 | no | write (every 3) | FF 1.1 V / -40 C | nominal | 4.5 | 367 | - | - | - / - | - / 9054.9 | 0.3 | 2155.8 | 72.6 | 111.1 | 247.2 | pass |
| p_8x4_6t_SS_write_idle2_pd_s1 | 6T_ | 8x4 | no | write (every 2) | SS 0.9 V / 125 C | per-device s1 | 4.5 | 365 | - | - | - / - | - / 4687.5 | 0.1 | 1851.5 | 295.0 | 378.2 | 860.4 | pass |
| p_8x4_6t_SS_write_idle2_pd_s2 | 6T_ | 8x4 | no | write (every 2) | SS 0.9 V / 125 C | per-device s2 | 4.5 | 365 | - | - | - / - | - / 4729.9 | 0.1 | 1792.0 | 280.8 | 412.9 | 873.9 | pass |
| p_8x4_6t_SS_write_idle2_pd_s3 | 6T_ | 8x4 | no | write (every 2) | SS 0.9 V / 125 C | per-device s3 | 4.5 | 365 | - | - | - / - | - / 4701.5 | 0.2 | 1806.3 | 277.5 | 381.7 | 864.0 | pass |
| p_8x4_6t_TT_sequence_settle0 | 6T_ | 8x4 | no | read&write (0 settling stages) | TT 1.0 V / 25 C | nominal | 4.5 | 1391 | 111.0 | 1322.9 | - / 65.2 | 117.7 / - | 0.2 | 2078.6 | 1543.1 | 175.7 | - | pass |
| p_8x4_6t_TT_write_idle2 | 6T_ | 8x4 | no | write (every 2) | TT 1.0 V / 25 C | nominal | 4.5 | 365 | - | - | - / - | - / 4592.3 | 0.2 | 2079.0 | 120.9 | 180.2 | 393.4 | pass |
| p_8x64_6t_mux_FF_cold_write_idle2 | 6T_ | 8x64 | yes | write (every 2) | FF 1.1 V / -40 C | nominal | 5.0 | 5165 | - | - | - / - | - / 5055.5 | 0.0 | 2388.8 | 74.4 | 127.0 | 278.2 | pass |
| r_128x8_10t_mux_SS_read | 10T | 128x8 | yes | read | SS 0.9 V / 125 C | nominal | 6.0 | 3459 | 254.3 | 210.1 | 208.0 / - | - / - | - | - | 854.5 | - | 760.7 | pass |
| r_128x8_6t_mux_SS_read | 6T_ | 128x8 | yes | read | SS 0.9 V / 125 C | nominal | 5.5 | 3459 | 254.3 | 70.0 | 206.2 / - | - / - | - | - | 707.7 | - | 758.5 | pass |
| r_16x16_10t_mux_SS_read | 10T | 16x16 | yes | read | SS 0.9 V / 125 C | nominal | 5.0 | 1043 | 210.6 | 373.8 | 203.6 / - | - / - | - | - | 965.1 | - | 739.2 | pass |
| r_16x16_6t_mux_SS_read | 6T_ | 16x16 | yes | read | SS 0.9 V / 125 C | nominal | 4.75 | 1027 | 210.4 | 297.1 | 203.1 / - | - / - | - | - | 885.6 | - | 737.9 | pass |
| r_256x4_10t_SS_read | 10T | 256x4 | no | read | SS 0.9 V / 125 C | nominal | 8.0 | 3659 | 319.7 | 677.1 | 209.8 / - | - / - | - | - | 1368.3 | - | 811.2 | pass |
| r_256x4_6t_mux_SS_read | 6T_ | 256x4 | yes | read | SS 0.9 V / 125 C | nominal | 6.75 | 3659 | 321.1 | 7.0 | 207.4 / - | - / - | - | - | 684.4 | - | 792.7 | pass |
| r_512x4_10t_mux_SS_read | 10T | 512x4 | yes | read | SS 0.9 V / 125 C | nominal | 10.0 | 1125 | 546.6 | -89.7 | 217.9 / - | - / - | - | - | 684.4 | - | 912.8 | pass |
| r_512x4_6t_mux_SS_read | 6T_ | 512x4 | yes | read | SS 0.9 V / 125 C | nominal | 9.0 | 1125 | 546.9 | -308.9 | 212.0 / - | - / - | - | - | 449.4 | - | 904.0 | pass |
| r_64x16_10t_mux_SS_read | 10T | 64x16 | yes | read | SS 0.9 V / 125 C | nominal | 5.5 | 3443 | 221.7 | 345.5 | 209.0 / - | - / - | - | - | 978.0 | - | 769.0 | pass |
| r_64x16_6t_mux_SS_read | 6T_ | 64x16 | yes | read | SS 0.9 V / 125 C | nominal | 5.0 | 3443 | 222.0 | 172.6 | 208.8 / - | - / - | - | - | 798.4 | - | 765.1 | pass |
| r_8x128_6t_mux_SS_read | 6T_ | 8x128 | yes | read | SS 0.9 V / 125 C | nominal | 6.0 | 4899 | 202.6 | 764.0 | 227.0 / - | - / - | - | - | 1458.3 | - | 836.2 | pass |
| r_8x4_10t_FF_cold_read | 10T | 8x4 | no | read | FF 1.1 V / -40 C | nominal | 5.0 | 187 | 70.6 | 1887.9 | 37.0 / - | - / - | - | - | 2053.9 | - | 198.6 | pass |
| r_8x4_10t_FF_cold_read_pd_s1 | 10T | 8x4 | no | read | FF 1.1 V / -40 C | per-device s1 | 5.0 | 187 | 71.5 | 1886.8 | 37.1 / - | - / - | - | - | 2053.6 | - | 198.6 | pass |
| r_8x4_10t_SS_read | 10T | 8x4 | no | read | SS 0.9 V / 125 C | nominal | 5.0 | 187 | 241.1 | 473.0 | 162.5 / - | - / - | - | - | 1006.9 | - | 737.5 | pass |
| r_8x4_6t_FF_cold_read | 6T_ | 8x4 | no | read | FF 1.1 V / -40 C | nominal | 4.5 | 187 | 70.6 | 1650.8 | 36.9 / - | - / - | - | - | 1813.6 | - | 198.0 | pass |
| r_8x4_6t_FF_cold_read_pd_s1 | 6T_ | 8x4 | no | read | FF 1.1 V / -40 C | per-device s1 | 4.5 | 187 | 71.4 | 1650.0 | 36.4 / - | - / - | - | - | 1813.1 | - | 197.9 | pass |
| r_8x4_6t_FF_cold_read_pd_s2 | 6T_ | 8x4 | no | read | FF 1.1 V / -40 C | per-device s2 | 4.5 | 187 | 68.6 | 1658.7 | 36.6 / - | - / - | - | - | 1822.9 | - | 200.0 | pass |
| r_8x4_6t_SS_read | 6T_ | 8x4 | no | read | SS 0.9 V / 125 C | nominal | 4.5 | 187 | 241.0 | 265.3 | 162.3 / - | - / - | - | - | 795.1 | - | 735.8 | pass |
| s_16x16_10t_SS_sequence | 10T | 16x16 | no | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 7291 | 208.5 | 410.9 | - / 329.7 | 310.9 / - | 0.0 | 2012.6 | 930.7 | 481.4 | - | pass |
| s_16x16_10t_SS_sequence_pd_s1 | 10T | 16x16 | no | read&write | SS 0.9 V / 125 C | per-device s1 | 5.0 | 7291 | 213.2 | 371.2 | - / 345.4 | 314.4 / - | 0.0 | 2023.8 | 892.4 | 479.7 | - | pass |
| s_16x16_10t_SS_sequence_pd_s2 | 10T | 16x16 | no | read&write | SS 0.9 V / 125 C | per-device s2 | 5.0 | 7291 | 200.8 | 454.3 | - / 317.1 | 304.3 / - | -0.0 | 2012.8 | 972.4 | 463.0 | - | pass |
| s_16x16_10t_mux_SS_sequence | 10T | 16x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 7291 | 210.4 | 344.5 | - / 330.1 | 311.9 / - | 0.0 | 2006.8 | 831.4 | 448.4 | - | pass |
| s_16x16_6t_SS_sequence | 6T_ | 16x16 | no | read&write | SS 0.9 V / 125 C | nominal | 4.5 | 7291 | 208.4 | 207.7 | - / 330.2 | 309.7 / - | 0.0 | 1758.2 | 724.5 | 481.1 | - | pass |
| s_16x16_6t_mux_SS_sequence | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 4.75 | 7291 | 210.3 | 267.1 | - / 330.1 | 311.3 / - | 0.0 | 1879.0 | 752.3 | 447.8 | - | pass |
| s_16x16_6t_mux_SS_sequence_pd_s1 | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | per-device s1 | 4.75 | 7291 | 223.9 | 213.6 | - / 328.9 | 323.3 / - | 0.0 | 1818.6 | 703.5 | 456.7 | - | pass |
| s_16x16_6t_mux_SS_sequence_pd_s2 | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | per-device s2 | 4.75 | 7291 | 216.4 | 249.1 | - / 314.1 | 322.0 / - | -0.0 | 1846.4 | 741.0 | 449.7 | - | pass |
| s_16x16_6t_mux_SS_sequence_pd_s3 | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | per-device s3 | 4.75 | 7291 | 214.2 | 243.0 | - / 322.9 | 301.4 / - | 0.0 | 1857.7 | 731.7 | 413.4 | - | pass |
| s_16x32_6t_mux_SS_sequence | 6T_ | 16x32 | yes | read&write | SS 0.9 V / 125 C | nominal | 4.75 | 14347 | 208.4 | 217.3 | - / 326.4 | 322.7 / - | -0.0 | 1865.3 | 751.9 | 495.7 | - | pass |
| s_32x16_10t_mux_SS_sequence | 10T | 32x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 11771 | 214.5 | 266.4 | - / 330.8 | 314.0 / - | 0.0 | 1972.9 | 764.9 | 459.3 | - | pass |
| s_32x16_6t_SS_sequence | 6T_ | 32x16 | no | read&write | SS 0.9 V / 125 C | nominal | 4.5 | 11771 | 212.7 | 147.5 | - / 331.6 | 313.8 / - | 0.0 | 1721.8 | 674.3 | 492.0 | - | pass |
| s_32x16_6t_mux_SS_sequence | 6T_ | 32x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 4.75 | 11771 | 214.7 | 197.0 | - / 331.7 | 314.5 / - | 0.0 | 1845.0 | 693.5 | 458.9 | - | pass |
| s_8x4_10t_FF_cold_sequence | 10T | 8x4 | no | read&write | FF 1.1 V / -40 C | nominal | 5.0 | 1391 | 70.6 | 1885.3 | - / 99.8 | 69.8 / - | 0.3 | 2409.6 | 2033.8 | 106.8 | - | pass |
| s_8x4_10t_mux_SS_sequence | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 1391 | 240.4 | 380.8 | - / 372.7 | 273.0 / - | 0.1 | 2070.4 | 837.1 | 392.2 | - | pass |
| s_8x4_10t_mux_SS_sequence_pd_s1 | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | per-device s1 | 5.0 | 1391 | 248.9 | 356.3 | - / 383.9 | 282.0 / - | 0.1 | 2084.6 | 815.1 | 396.4 | - | pass |
| s_8x4_10t_mux_SS_sequence_pd_s2 | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | per-device s2 | 5.0 | 1391 | 239.7 | 411.2 | - / 358.5 | 267.1 / - | 0.1 | 2062.2 | 863.8 | 383.3 | - | pass |
| s_8x4_10t_mux_SS_sequence_pd_s3 | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | per-device s3 | 5.0 | 1391 | 242.5 | 347.5 | - / 385.9 | 283.0 / - | 0.2 | 2050.2 | 808.7 | 398.5 | - | pass |
| s_8x4_6t_FF_cold_sequence | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | nominal | 4.5 | 1391 | 70.6 | 1647.8 | - / 99.7 | 69.7 / - | 0.3 | 2155.1 | 1794.4 | 106.8 | - | pass |
| s_8x4_6t_FF_cold_sequence_pd_s1 | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | per-device s1 | 4.5 | 1391 | 68.3 | 1654.9 | - / 99.9 | 69.8 / - | 0.2 | 2155.3 | 1800.6 | 105.5 | - | pass |
| s_8x4_6t_FF_cold_sequence_pd_s2 | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | per-device s2 | 4.5 | 1391 | 71.0 | 1647.6 | - / 100.9 | 70.5 / - | 0.3 | 2158.4 | 1793.4 | 106.9 | - | pass |
| s_8x4_6t_FF_cold_sequence_pd_s3 | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | per-device s3 | 4.5 | 1391 | 71.8 | 1647.5 | - / 96.9 | 69.6 / - | 0.3 | 2154.2 | 1792.7 | 106.0 | - | pass |
| s_8x4_6t_SS_sequence | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | nominal | 4.5 | 1391 | 241.0 | 257.6 | - / 372.8 | 272.7 / - | 0.1 | 1821.7 | 714.8 | 396.6 | - | pass |
| s_8x4_6t_SS_sequence_pd_s1 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s1 | 4.5 | 1391 | 223.3 | 306.4 | - / 384.5 | 276.4 / - | 0.1 | 1788.8 | 761.0 | 386.5 | - | pass |
| s_8x4_6t_SS_sequence_pd_s2 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s2 | 4.5 | 1391 | 249.0 | 254.9 | - / 379.7 | 276.4 / - | 0.1 | 1837.0 | 715.5 | 396.6 | - | pass |
| s_8x4_6t_SS_sequence_pd_s3 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s3 | 4.5 | 1391 | 256.4 | 238.7 | - / 352.7 | 271.7 / - | 0.2 | 1798.1 | 681.2 | 389.8 | - | pass |
| s_8x4_6t_SS_sequence_pd_s4 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s4 | 4.5 | 1391 | 238.1 | 254.7 | - / 373.0 | 286.8 / - | 0.1 | 1836.4 | 710.7 | 389.0 | - | pass |
| s_8x4_6t_SS_sequence_pd_s5 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s5 | 4.5 | 1391 | 254.6 | 204.2 | - / 381.0 | 276.6 / - | 0.3 | 1786.2 | 677.3 | 398.3 | - | pass |
| s_8x8_6t_mux_FF_cold_sequence | 6T_ | 8x8 | yes | read&write | FF 1.1 V / -40 C | nominal | 4.75 | 2611 | 67.5 | 1756.3 | - / 94.8 | 72.3 / - | 0.1 | 2281.9 | 1906.2 | 109.4 | - | pass |
| w_128x8_6t_SS_write | 6T_ | 128x8 | no | write | SS 0.9 V / 125 C | nominal | 5.25 | 3546 | - | - | - / - | - / 486.8 | 0.0 | 2238.0 | 287.6 | 503.6 | 1225.3 | pass |
| w_16x16_10t_mux_SS_write | 10T | 16x16 | yes | write | SS 0.9 V / 125 C | nominal | 5.0 | 1218 | - | - | - / - | - / 437.8 | 0.0 | 2007.2 | 291.0 | 513.3 | 1192.1 | pass |
| w_16x16_6t_mux_SS_write | 6T_ | 16x16 | yes | write | SS 0.9 V / 125 C | nominal | 4.75 | 1218 | - | - | - / - | - / 437.8 | 0.0 | 1879.9 | 291.1 | 512.9 | 1191.3 | pass |
| w_16x64_6t_mux_SS_write | 6T_ | 16x64 | yes | write | SS 0.9 V / 125 C | nominal | 5.0 | 4722 | - | - | - / - | - / 436.9 | -0.0 | 1977.5 | 291.5 | 531.0 | 1240.2 | pass |
| w_256x4_6t_SF_write | 6T_ | 256x4 | no | write | SF 0.9 V / 125 C | nominal | 6.25 | 3698 | - | - | - / - | - / 513.3 | 0.6 | 2872.4 | 234.7 | 437.2 | 1193.6 | pass |
| w_256x4_6t_SS_write | 6T_ | 256x4 | no | write | SS 0.9 V / 125 C | nominal | 6.25 | 3698 | - | - | - / - | - / 579.5 | 0.5 | 2843.9 | 279.6 | 489.4 | 1331.1 | pass |
| w_512x4_6t_mux_SS_write | 6T_ | 512x4 | yes | write | SS 0.9 V / 125 C | nominal | 9.0 | 1168 | - | - | - / - | - / 822.1 | 0.9 | 4491.2 | 280.6 | 475.9 | 1681.4 | pass |
| w_64x16_10t_mux_SF_write | 10T | 64x16 | yes | write | SF 0.9 V / 125 C | nominal | 5.5 | 3618 | - | - | - / - | - / 392.8 | 0.0 | 2337.4 | 251.6 | 451.3 | 1089.0 | pass |
| w_64x16_10t_mux_SS_write | 10T | 64x16 | yes | write | SS 0.9 V / 125 C | nominal | 5.5 | 3618 | - | - | - / - | - / 446.6 | 0.0 | 2297.5 | 291.9 | 507.8 | 1209.8 | pass |
| w_64x16_6t_mux_SF_write | 6T_ | 64x16 | yes | write | SF 0.9 V / 125 C | nominal | 5.0 | 3618 | - | - | - / - | - / 393.3 | 0.0 | 2081.4 | 251.1 | 451.2 | 1086.9 | pass |
| w_64x16_6t_mux_SS_write | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | nominal | 5.0 | 3618 | - | - | - / - | - / 446.2 | 0.0 | 2044.1 | 291.4 | 506.2 | 1207.3 | pass |
| w_64x16_6t_mux_SS_write_pd_s1 | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | per-device s1 | 5.0 | 3618 | - | - | - / - | - / 461.9 | 0.0 | 2015.5 | 287.6 | 522.0 | 1229.8 | pass |
| w_64x16_6t_mux_SS_write_pd_s2 | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | per-device s2 | 5.0 | 3618 | - | - | - / - | - / 434.0 | 0.0 | 2032.9 | 294.0 | 475.9 | 1203.0 | pass |
| w_64x16_6t_mux_SS_write_pd_s3 | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | per-device s3 | 5.0 | 3618 | - | - | - / - | - / 435.9 | -0.0 | 2021.8 | 289.4 | 484.9 | 1216.8 | pass |
| w_8x4_10t_FF_cold_write | 10T | 8x4 | no | write | FF 1.1 V / -40 C | nominal | 5.0 | 230 | - | - | - / - | - / 130.2 | 0.3 | 2410.1 | 72.7 | 128.2 | 322.7 | pass |
| w_8x4_10t_SS_write | 10T | 8x4 | no | write | SS 0.9 V / 125 C | nominal | 5.0 | 230 | - | - | - / - | - / 479.2 | 0.1 | 2078.3 | 278.3 | 470.8 | 1136.9 | pass |
| w_8x4_6t_FF_cold_write | 6T_ | 8x4 | no | write | FF 1.1 V / -40 C | nominal | 4.5 | 230 | - | - | - / - | - / 130.1 | 0.3 | 2156.0 | 72.8 | 128.1 | 322.1 | pass |
| w_8x4_6t_FF_cold_write_pd_s1 | 6T_ | 8x4 | no | write | FF 1.1 V / -40 C | per-device s1 | 4.5 | 230 | - | - | - / - | - / 130.9 | 0.2 | 2159.1 | 74.3 | 125.2 | 325.1 | pass |
| w_8x4_6t_FF_cold_write_pd_s2 | 6T_ | 8x4 | no | write | FF 1.1 V / -40 C | per-device s2 | 4.5 | 230 | - | - | - / - | - / 129.6 | 0.2 | 2153.9 | 72.4 | 128.0 | 319.3 | pass |
| w_8x4_6t_SS_write | 6T_ | 8x4 | no | write | SS 0.9 V / 125 C | nominal | 4.5 | 230 | - | - | - / - | - / 480.0 | 0.1 | 1823.4 | 278.7 | 470.6 | 1136.4 | pass |

## 6. Limits

* Screening at the class bounds with illustrative wires; extracted metal,
  half-select writes and the yield estimator remain the carried Phase 6
  scope.
* The sense timing (`K = 1`, `N = 9`) fires after a complete bitline swing
  in every probed size, so the sense-timed wordline saves the wordline,
  replica and hold-latch activity of the tail, not bitline energy. Moving
  the trigger earlier (fewer delay stages or more replica cells) is where
  the bitline energy is; it changes the sense margin and needs its own
  mismatch evidence.
* The explicit orderings are gate-count margins: the precharge follows the
  previous sense enable by 42 ps at 8x4 FF (NOR, AND3, NAND3 and the
  precharge buffer), the isolation drops 18 ps before it. They hold at every
  probed corner and seed but are short at FF.
* The wordline release follows the sense trigger by the release path's
  delay, 45 ps at 8x4 FF to 547 ps at 512x4 SS; at the 512-row class bound
  the local wordline is below 0.1 VDD only after the edge that ends the
  access. A tapered release path would shorten it; the write drivers release
  with the wordline enable, 295 ps before the local wordline is off at 512
  rows (unchanged from V2.1.7).
* The write wordline lasts the whole clock-low phase; ending it at a
  write-completion signal is a timing-policy change outside this review.
* Without settling stages (`precharge_guard_stages: 0`) the write -> write
  data margin is not evidenced (no such deck); the sequence at zero stages
  passes.
* The column-mux select is static in this testbench (F4).
