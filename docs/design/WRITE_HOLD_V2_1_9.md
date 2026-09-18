# V2.1.9: the write drivers stay on until the wordline is off; read clocks re-derived under mismatch

Executed September 17 and 18, 2026 (US Pacific time) on the V2.1.9 sources (base commit
`a14578f` plus the changes of this release; every queue archives the sources
it ran). An audit of the [V2.1.8 release](ENABLE_OVERLAP_V2_1_8.md) checked
one requirement on the write path: the write drivers must be fully on for the
whole time the wordline is asserted, and must never be turned off while it
is. It then checked that reads and writes still work at the worst PVT
combinations under device mismatch; that sweep found the read clocks of the
128- to 512-row classes too short under mismatch at SS 0.9 V / 125 C, and a
DC operating-point weakness of the new latch, both fixed in this release
(sections 5 and 6). Illustrative wires, V2.0.9 driver classes and the
V2.1.9 timing classes throughout; per-device samples use
5 % relative sigma on `vth0`, `u0` and `voff` of every MOS (cells, periphery
and control logic). This is functional screening, not yield qualification or
extracted-metal evidence; no record is promoted to `sizing_table.json`.

**Outcome: on the release sources 313 of 313 cases pass, 638,438 of 638,438 checks: the 91 cases of the V2.1.8 matrix (59 nominal, 32 mismatch seeds), the 4 nominal reads at the re-derived class bounds and 218 Monte-Carlo samples (the worst-PVT sweep and the class-bound reads); r_128x8_10t_mux_SS_read and r_128x8_6t_mux_SS_read after a four-rank DC operating-point retry with Newton line search. In every write cycle and column the drivers' enable stayed at or above 99.5% of VDD while the local wordline was above 0.1 VDD (p_8x64_6t_mux_FF_cold_write_idle2); the drivers started to release at least 106 ps after the local wordline was below 0.1 VDD (mc_8x4_6t_FF_cold_write_pd_s1; 76 ps without settling stages, p_8x4_6t_TT_sequence_settle0), and the runtime `VWEN_ACCESS_ERROR` never exceeded 1.2% of VDD (mc_16x16_10t_mux_SF_sequence_pd_s1). In V2.1.8 the same cases released the drivers up to 322 ps before the wordline was off (w_512x4_6t_mux_SS_write), with the wordline at up to 100% of VDD (w_512x4_6t_mux_SS_write). Between two writes new data passed every driver's hold latch at least 113 ps before it closed again (mc_8x4_6t_FF_cold_write_pd_s5); at an idle -> write or read -> write edge at least 110 ps before the write enable (p_8x4_6t_FF_cold_write_idle2_pd_s2). The previous write enable was off at least 36 ps before the next precharge (mc_8x4_10t_FF_cold_sequence_pd_s2) and 137 ps before the next write enable (mc_8x4_10t_FF_cold_write_pd_s3); the previous sense enable 37 ps before the next precharge (mc_8x4_6t_FF_cold_read_pd_s2) and 138 ps before the next write enable (s_8x4_6t_FF_cold_sequence_pd_s1). The driven bitline reached its rail at least 1490 ps before the local wordline started; the smallest sense margin was 0.494 V (mc_512x4_6t_mux_SS_read_pd_s2); the longest write slot 2230 ps (w_512x4_6t_mux_SS_write) and the longest restore 943 ps (mc_512x4_6t_mux_SS_read_pd_s4).**

## 1. What the audit found

The release write enable was `w_en = we_hold & (wl_en | selected_slot)`,
with the write request held in a latch that reopened on `wl_en_bar`. Both
terms end with the wordline *enable*, but the physical wordline outlives it
by the row driver and the wordline wire, and at 256 and 512 rows by the
`wl_en` line across the rows. The traces of the V2.1.8 evidence campaign
(`outputs/validation/V2.1.8-enable-overlap`, the far column's driver
terminal against the target cell's local wordline):

| | Finding | Where | Consequence |
|---|---|---|---|
| F1 | At a write -> write boundary the drivers started to release (`w_en` at 90 % of VDD) with the local wordline at 0.51 V of 1.1 V (8x4 FF -40 C, 16 ps before it was below 0.1 VDD), 0.27 V of 0.9 V (8x4 SS), about half VDD at 256x4 SS and SF, and 0.90 V at 512x4 SS, where they were off (50 %) 235 ps before the wordline had fallen to half VDD and began to release 321 ps before it was below 0.1 VDD | `write_window = wl_en \| selected_slot` | the drivers tri-stated the bitlines under an open pass gate. The written cell survived in every deck (the bitlines float at their rails), but the requirement is broken at every write boundary, most at the largest arrays |
| F2 | At a write -> read boundary the held request reopened on `wl_en_bar` and the next (read) request dropped `w_en` at the same time, independently of the window | `we_hold` latch enable `wl_en_bar` | the same release under an open wordline; a window fix alone does not cover it |
| F3 | The V2.1.8 record stated that the drivers release 19 to 118 ps after the wordline is below 0.1 VDD (its F5, from the probe tool) and downgraded the write tail to a metric; its own campaign column `WL at w_en off` shows 0.2 to 0.3 V at 8x4 FF, 0.5 to 0.6 V at 256x4 and 0.9 V at 512x4 | `ENABLE_OVERLAP_V2_1_8.md` sections 1, 3 and 5 | the violation passed every check; nothing in the runtime measures looked at it |

Holding the drivers until the wordline is observed off puts their release
and the opening of the next write slot on the same event (the replica
wordline observer). Between two writes that is a race: if the slot opens
before the drivers have dropped, `w_en` never falls, and the write-data hold
latch in front of every driver (transparent only while `w_en` is low) never
takes the next cycle's data. The release therefore also needs an explicit
handshake to the next slot (section 2).

## 2. The change

In `TIME_CONTROL` (`docs/design/TIME_CONTROL_PATH.md`, sections 2.9, 2.10,
2.12, 2.13 and 8):

* **Busy wordline.** `wordline_busy = !(wl_en_bar & wordline_off)`
  (`WORDLINE_BUSY_NAND`), high from the wordline enable until the replica
  wordline has been observed off with its settling stages (`pre_ready`, or
  `rwl_pre_bar` without them); `wordline_idle` is its complement.
* **Drivers held while the wordline is busy (F1).** `write_window =
  wordline_busy | selected_slot`, `w_en = we_hold & write_window`.
* **Request held while the wordline is busy (F2).** The write-request hold
  latch opens on `wordline_idle` instead of `wl_en_bar`.
* **Slot armed by the drivers' release.** `selected_slot = cs_pre &
  write_slot & slot_armed` (`SELECTED_SLOT_AND`, now an AND3). `slot_armed`
  is a NOR latch (`SLOT_ARM_NOR` x 2), set by `enables_off` through four unit
  stages (`SLOT_ARM_DELAY`) and reset by `wordline_busy`: the slot reopens
  only after the drivers are observed off, and it stays open once `w_en` is
  back on (a combinational `& enables_off` would oscillate). The four stages
  are the write-data latch's window: without them it was 58 ps at 8x4 FF
  -40 C (V2.1.8: 106 ps, with the drivers released early); with them at
  least 113 ps there under mismatch and 421 ps at SS.
* **Start state.** The testbench seeds the slot-arm latch with `.IC`
  (`slot_armed` = VDD, `slot_armed_bar` = 0, `enables_off_settled` = VDD), the
  state it must have at t = 0 (select clamped off, wordline idle, both enables
  off), as it already seeds the select flip-flop and the data latches. The
  first evidence pass (section 6) found the unseeded latch made the DC
  operating point of mux read decks fail or take minutes: the 8x128 mux read
  deck failed at four ranks even with Newton line search, converged slowly at
  one rank, and converged at four ranks once seeded (the V2.1.8 deck also
  converged).
* **Unchanged.** The read path, the sense timing, the wordline enable, the
  precharge gate, the driver classes and every load (`enables_off` gains one
  unit-inverter input, `wl_en_bar` loses the request latch; its scale is kept
  as an upper bound). Without the replica guard (factory defaults only; the
  testbench always builds it) nothing observes the wordline and the drivers
  still end with `wl_en`.

## 3. Checks

* **Runtime measure.** Every write cycle of a deck carries
  `VWEN_ACCESS_ERROR_<cycle>`: the highest level of the target cell's local
  wordline or of the far end of its row while the target driver's enable
  is below 0.9 VDD, from 0.65 T into the cycle to 1.7 T.
  `Sram6TCoreMcTestbench.access_validity` limits it to 0.1 VDD like the other
  access checks, so `run_mc_simulation`, the per-device CLI and the yield
  callers reject such a sample even when the written data is right.
* **Local checker** (`dev/v210_waveform_checks.py`, run by the validator):
  `_write_enable_on_during_wordline` per write cycle and column, the
  driver's enable at least 0.9 VDD from the local wordline's 10 % rise to its
  10 % fall (so a dip inside the access fails too), with the metrics
  `_min_write_enable_during_wordline_v` and
  `_write_enable_release_after_wordline_ps` (the enable's 90 % fall after the
  wordline's 10 % fall). `_wordline_at_write_enable_off_v` stays a metric.
* **Qualification scorer** (`dev/sizing/qualification.py`):
  `write_enable_covers_wordline` in write decks and per write cycle of a
  sequence, the block's `W_EN` (released first) against the far wordline
  (released last).
* The validator prints `XTIME_CONTROL:rwl_pre_bar` and `wordline_idle`; the
  testbench prints `XTIME_CONTROL:wordline_busy` and `slot_armed`.

## 4. Before -> after

The V2.1.9 matrix repeats the V2.1.8 cases under the same names and settings.
`compare_v218.py` measures the traces of both campaigns the same way: at every
edge that ends a write, per column, the driver's enable at its terminal
falling through 0.9 VDD against the target row's local wordline at that
column falling through 0.1 VDD (release after WL off; negative means the
drivers began to release while the wordline was on) and the wordline level at
that moment; the write -> write data-latch window of the single write decks;
the checker's write-enable orderings; `TWSLOT` and the per-cycle write energy
`EWRITE` (one period from the first clock fall) from the `.mt0`. Read decks
keep the V2.1.8 read path and are left out (their clocks are section 6); the
256- and 512-row writes run at the longer V2.1.9 clocks, which adds leakage
to their one-period `EWRITE`.

| Case | release after WL off [ps] | WL at release [V] | W->W latch window [ps] | w_en off -> w_en [ps] | w_en off -> PRE [ps] | TWSLOT [ps] | EWRITE [fJ] |
|---|---:|---:|---:|---:|---:|---:|---:|
| p_16x16_10t_SS_write_idle2 | 57.2 -> 467.1 | 0.002 -> -0.0 | - -> - | 5207.5 -> 4808.6 | - -> - | 954.7 -> 965.0 | 611.3 -> 628.6 |
| p_16x16_6t_mux_SS_write_idle2 | 57.1 -> 466.6 | 0.002 -> -0.0 | - -> - | 4957.6 -> 4559.6 | - -> - | 962.6 -> 973.3 | 528.4 -> 545.5 |
| p_4x4_6t_FF_cold_write_idle2 | -16.5 -> 108.4 | 0.533 -> -0.0 | - -> - | 4553.5 -> 4430.9 | - -> - | 243.3 -> 244.7 | 435.0 -> 464.1 |
| p_64x16_6t_mux_SS_write_idle2 | 53.6 -> 470.8 | 0.003 -> -0.0 | - -> - | 5186.6 -> 4779.9 | - -> - | 950.1 -> 960.6 | 831.9 -> 850.7 |
| p_8x4_10t_FF_cold_write_idle2 | -15.9 -> 108.5 | 0.515 -> -0.0 | - -> - | 5054.3 -> 4932.6 | - -> - | 247.1 -> 249.1 | 496.7 -> 524.7 |
| p_8x4_6t_FF_cold_write_idle2 | -15.9 -> 108.6 | 0.514 -> -0.0 | - -> - | 4554.9 -> 4432.5 | - -> - | 247.2 -> 248.8 | 482.3 -> 512.4 |
| p_8x4_6t_FF_cold_write_idle2_pd_s1 | -15.1 -> 112.4 | 0.493 -> -0.0 | - -> - | 4550.7 -> 4429.7 | - -> - | 244.6 -> 250.1 | 484.2 -> 511.6 |
| p_8x4_6t_FF_cold_write_idle2_pd_s2 | -15.3 -> 107.8 | 0.501 -> -0.0 | - -> - | 4555.4 -> 4430.2 | - -> - | 245.2 -> 247.7 | 485.9 -> 513.0 |
| p_8x4_6t_FF_cold_write_idle2_pd_s3 | -18.0 -> 114.8 | 0.567 -> -0.0 | - -> - | 4552.2 -> 4429.7 | - -> - | 244.4 -> 250.7 | 483.9 -> 515.1 |
| p_8x4_6t_FF_cold_write_idle3 | -15.9 -> 108.6 | 0.514 -> -0.0 | - -> - | 9054.9 -> 8932.5 | - -> - | 247.2 -> 248.8 | 482.3 -> 512.4 |
| p_8x4_6t_SS_write_idle2_pd_s1 | -26.4 -> 443.3 | 0.22 -> -0.0 | - -> - | 4687.5 -> 4252.8 | - -> - | 860.4 -> 903.8 | 291.1 -> 308.7 |
| p_8x4_6t_SS_write_idle2_pd_s2 | -20.1 -> 411.6 | 0.181 -> -0.0 | - -> - | 4729.9 -> 4257.7 | - -> - | 873.9 -> 888.8 | 293.1 -> 308.2 |
| p_8x4_6t_SS_write_idle2_pd_s3 | -53.3 -> 474.2 | 0.391 -> -0.0 | - -> - | 4701.5 -> 4258.2 | - -> - | 864.0 -> 907.9 | 294.0 -> 310.1 |
| p_8x4_6t_TT_sequence_settle0 | -20.1 -> 76.5 | 0.379 -> -0.0 | - -> - | - -> - | 65.2 -> 66.0 | - -> - | - -> - |
| p_8x4_6t_TT_write_idle2 | -20.3 -> 179.9 | 0.381 -> -0.0 | - -> - | 4592.3 -> 4396.3 | - -> - | 393.4 -> 397.1 | 367.4 -> 388.9 |
| p_8x64_6t_mux_FF_cold_write_idle2 | 11.6 -> 126.9 | 0.021 -> -0.0 | - -> - | 5055.5 -> 4942.1 | - -> - | 278.2 -> 279.5 | 2083.3 -> 2112.7 |
| s_16x16_10t_SS_sequence | 57.6 -> 467.0 | 0.002 -> -0.0 | - -> - | - -> - | 329.7 -> 204.5 | - -> - | - -> - |
| s_16x16_10t_SS_sequence_pd_s1 | 45.2 -> 468.0 | 0.007 -> -0.0 | - -> - | - -> - | 345.4 -> 203.0 | - -> - | - -> - |
| s_16x16_10t_SS_sequence_pd_s2 | 73.4 -> 462.4 | 0.0 -> -0.0 | - -> - | - -> - | 317.1 -> 198.6 | - -> - | - -> - |
| s_16x16_10t_mux_SS_sequence | 57.4 -> 466.7 | 0.002 -> -0.0 | - -> - | - -> - | 330.1 -> 204.5 | - -> - | - -> - |
| s_16x16_6t_SS_sequence | 57.1 -> 466.3 | 0.002 -> -0.0 | - -> - | - -> - | 330.2 -> 204.4 | - -> - | - -> - |
| s_16x16_6t_mux_SS_sequence | 57.4 -> 466.7 | 0.002 -> -0.0 | - -> - | - -> - | 330.1 -> 204.2 | - -> - | - -> - |
| s_16x16_6t_mux_SS_sequence_pd_s1 | 52.7 -> 463.4 | 0.003 -> -0.0 | - -> - | - -> - | 328.9 -> 203.6 | - -> - | - -> - |
| s_16x16_6t_mux_SS_sequence_pd_s2 | 65.0 -> 478.6 | 0.001 -> -0.0 | - -> - | - -> - | 314.1 -> 201.1 | - -> - | - -> - |
| s_16x16_6t_mux_SS_sequence_pd_s3 | 45.1 -> 467.1 | 0.005 -> -0.0 | - -> - | - -> - | 322.9 -> 200.2 | - -> - | - -> - |
| s_16x32_6t_mux_SS_sequence | 70.5 -> 475.9 | 0.0 -> -0.0 | - -> - | - -> - | 326.4 -> 214.4 | - -> - | - -> - |
| s_32x16_10t_mux_SS_sequence | 56.9 -> 467.4 | 0.002 -> -0.0 | - -> - | - -> - | 330.8 -> 204.5 | - -> - | - -> - |
| s_32x16_6t_SS_sequence | 56.3 -> 467.2 | 0.002 -> -0.0 | - -> - | - -> - | 331.6 -> 204.4 | - -> - | - -> - |
| s_32x16_6t_mux_SS_sequence | 56.5 -> 466.9 | 0.002 -> -0.0 | - -> - | - -> - | 331.7 -> 204.6 | - -> - | - -> - |
| s_8x4_10t_FF_cold_sequence | -15.8 -> 108.3 | 0.511 -> -0.0 | - -> - | - -> - | 99.8 -> 36.5 | - -> - | - -> - |
| s_8x4_10t_mux_SS_sequence | -36.0 -> 411.9 | 0.27 -> -0.0 | - -> - | - -> - | 372.7 -> 163.1 | - -> - | - -> - |
| s_8x4_10t_mux_SS_sequence_pd_s1 | -37.5 -> 416.3 | 0.269 -> -0.0 | - -> - | - -> - | 383.9 -> 166.9 | - -> - | - -> - |
| s_8x4_10t_mux_SS_sequence_pd_s2 | -30.2 -> 405.0 | 0.25 -> -0.0 | - -> - | - -> - | 358.5 -> 166.4 | - -> - | - -> - |
| s_8x4_10t_mux_SS_sequence_pd_s3 | -39.5 -> 396.1 | 0.286 -> -0.0 | - -> - | - -> - | 385.9 -> 166.7 | - -> - | - -> - |
| s_8x4_6t_FF_cold_sequence | -15.9 -> 108.5 | 0.515 -> -0.0 | - -> - | - -> - | 99.7 -> 36.5 | - -> - | - -> - |
| s_8x4_6t_FF_cold_sequence_pd_s1 | -12.7 -> 107.2 | 0.43 -> -0.0 | - -> - | - -> - | 99.9 -> 36.4 | - -> - | - -> - |
| s_8x4_6t_FF_cold_sequence_pd_s2 | -15.9 -> 106.6 | 0.521 -> -0.0 | - -> - | - -> - | 100.9 -> 37.0 | - -> - | - -> - |
| s_8x4_6t_FF_cold_sequence_pd_s3 | -17.6 -> 107.8 | 0.537 -> -0.0 | - -> - | - -> - | 96.9 -> 37.0 | - -> - | - -> - |
| s_8x4_6t_SS_sequence | -36.0 -> 411.5 | 0.271 -> -0.0 | - -> - | - -> - | 372.8 -> 163.1 | - -> - | - -> - |
| s_8x4_6t_SS_sequence_pd_s1 | -13.0 -> 396.7 | 0.143 -> -0.0 | - -> - | - -> - | 384.5 -> 167.9 | - -> - | - -> - |
| s_8x4_6t_SS_sequence_pd_s2 | -36.6 -> 391.5 | 0.285 -> -0.0 | - -> - | - -> - | 379.7 -> 170.0 | - -> - | - -> - |
| s_8x4_6t_SS_sequence_pd_s3 | -63.4 -> 406.8 | 0.377 -> -0.0 | - -> - | - -> - | 352.7 -> 165.9 | - -> - | - -> - |
| s_8x4_6t_SS_sequence_pd_s4 | -33.6 -> 394.0 | 0.251 -> -0.0 | - -> - | - -> - | 373.0 -> 162.2 | - -> - | - -> - |
| s_8x4_6t_SS_sequence_pd_s5 | -56.7 -> 438.3 | 0.405 -> -0.0 | - -> - | - -> - | 381.0 -> 165.1 | - -> - | - -> - |
| s_8x8_6t_mux_FF_cold_sequence | -6.7 -> 112.8 | 0.308 -> -0.0 | - -> - | - -> - | 94.8 -> 38.9 | - -> - | - -> - |
| w_128x8_6t_SS_write | 7.6 -> 464.9 | 0.06 -> -0.0 | 378.8 -> 459.4 | 486.8 -> 567.7 | - -> - | 1225.3 -> 1766.2 | 1498.8 -> 1532.9 |
| w_16x16_10t_mux_SS_write | 57.7 -> 466.9 | 0.002 -> -0.0 | 337.2 -> 463.5 | 437.8 -> 564.1 | - -> - | 1192.1 -> 1728.1 | 1001.3 -> 1027.8 |
| w_16x16_6t_mux_SS_write | 57.5 -> 466.5 | 0.002 -> -0.0 | 337.3 -> 463.3 | 437.8 -> 563.8 | - -> - | 1191.3 -> 1726.8 | 973.0 -> 999.4 |
| w_16x64_6t_mux_SS_write | 76.8 -> 483.2 | 0.0 -> -0.0 | 336.0 -> 484.2 | 436.9 -> 585.3 | - -> - | 1240.2 -> 1796.5 | 3132.9 -> 3162.2 |
| w_256x4_6t_SF_write | -97.3 -> 392.8 | 0.764 -> 0.0 | 421.9 -> 411.7 | 513.3 -> 503.5 | - -> - | 1193.6 -> 1673.9 | 2100.2 -> 2162.9 |
| w_256x4_6t_SS_write | -91.9 -> 458.5 | 0.719 -> -0.0 | 475.0 -> 461.6 | 579.5 -> 565.8 | - -> - | 1331.1 -> 1871.2 | 1860.1 -> 1901.9 |
| w_512x4_6t_mux_SS_write | -321.9 -> 466.2 | 0.9 -> 0.0 | 719.7 -> 475.8 | 822.1 -> 577.1 | - -> - | 1681.4 -> 2229.8 | 3468.7 -> 3558.5 |
| w_64x16_10t_mux_SF_write | 39.7 -> 408.7 | 0.007 -> -0.0 | 289.2 -> 400.0 | 392.8 -> 504.0 | - -> - | 1089.0 -> 1569.1 | 1668.6 -> 1698.9 |
| w_64x16_10t_mux_SS_write | 55.2 -> 473.2 | 0.002 -> -0.0 | 330.9 -> 454.7 | 446.6 -> 570.8 | - -> - | 1209.8 -> 1753.8 | 1526.0 -> 1555.4 |
| w_64x16_6t_mux_SF_write | 39.2 -> 409.8 | 0.007 -> -0.0 | 289.8 -> 397.5 | 393.3 -> 501.4 | - -> - | 1086.9 -> 1566.1 | 1603.7 -> 1633.6 |
| w_64x16_6t_mux_SS_write | 54.3 -> 470.4 | 0.002 -> -0.0 | 330.5 -> 450.9 | 446.2 -> 566.7 | - -> - | 1207.3 -> 1744.8 | 1473.8 -> 1502.3 |
| w_64x16_6t_mux_SS_write_pd_s1 | 46.5 -> 456.9 | 0.003 -> -0.0 | 329.0 -> 422.5 | 461.9 -> 552.9 | - -> - | 1229.8 -> 1712.0 | 1484.5 -> 1511.7 |
| w_64x16_6t_mux_SS_write_pd_s2 | 57.9 -> 464.5 | 0.002 -> -0.0 | 298.3 -> 459.0 | 434.0 -> 575.9 | - -> - | 1203.0 -> 1774.6 | 1489.2 -> 1514.8 |
| w_64x16_6t_mux_SS_write_pd_s3 | 58.3 -> 471.0 | 0.002 -> -0.0 | 298.4 -> 464.2 | 435.9 -> 593.2 | - -> - | 1216.8 -> 1800.1 | 1487.5 -> 1516.2 |
| w_8x4_10t_FF_cold_write | -15.8 -> 108.5 | 0.51 -> -0.0 | 106.4 -> 114.6 | 130.2 -> 138.4 | - -> - | 322.7 -> 455.2 | 679.6 -> 726.3 |
| w_8x4_10t_SS_write | -36.2 -> 411.3 | 0.273 -> -0.0 | 379.1 -> 421.3 | 479.2 -> 521.4 | - -> - | 1136.9 -> 1626.5 | 414.9 -> 440.7 |
| w_8x4_6t_FF_cold_write | -15.8 -> 108.5 | 0.51 -> -0.0 | 106.5 -> 114.7 | 130.1 -> 138.4 | - -> - | 322.1 -> 453.7 | 666.0 -> 710.9 |
| w_8x4_6t_FF_cold_write_pd_s1 | -15.0 -> 112.3 | 0.487 -> -0.0 | 105.1 -> 113.4 | 130.9 -> 138.2 | - -> - | 325.1 -> 458.7 | 667.9 -> 712.1 |
| w_8x4_6t_FF_cold_write_pd_s2 | -15.3 -> 107.9 | 0.5 -> -0.0 | 105.1 -> 112.7 | 129.6 -> 137.2 | - -> - | 319.3 -> 454.9 | 666.7 -> 713.2 |
| w_8x4_6t_SS_write | -36.1 -> 411.5 | 0.272 -> -0.0 | 380.0 -> 421.4 | 480.0 -> 521.5 | - -> - | 1136.4 -> 1625.2 | 406.5 -> 432.1 |

## 5. Worst-PVT Monte-Carlo sweep

The corner screen at 8x4 (nominal, `probes/probe3`) set the four corners:
SS 0.9 V / 125 C is the slowest (the read wordline released 260 ps before
the deadline; SS at -40 C left 1453 ps, so there is no temperature inversion
at 0.9 V), FF 1.1 V / -40 C and FS 1.1 V / -40 C carry the shortest
gate-count orderings (36 and 39 ps from the write enable off to the next
precharge), SF 0.9 V / 125 C is the write-ability corner (slow NMOS pass
gate against a fast PMOS pull-up) and FS the read-stability corner. Per
corner, 6T and 10T: 8x4 sequences, writes and reads with five seeds each,
16x16 with a mux with two; 64x16 with a mux (writes at SS and SF, reads at
SS and FS, two seeds); 256x4 (SS write, SF write, SS read) and 512x4 with a
mux (SS write, SS read), one seed each. FF 1.1 V / 125 C, the leakage corner
of a tall array, was probed at 256x4 (read, nominal and two seeds,
`probes/rt19b`): 1.3 ns of margin, 0.81 V at the amplifier.

The first pass of the sweep (`pass1/`, on the first V2.1.9 sources) found two
things the write hold did not cause directly:

* **DC operating point.** Five mux read decks (8x128 at four ranks with
  Newton line search, 16x16 10T at four ranks, three 16x16 10T per-device
  samples) aborted before any waveform. With the slot-arm latch seeded
  (section 2) the 8x128 deck and four of the five converge and pass
  (`probes/dcop`, `probes/dcop2`). The fifth, the 10T 16x16 mux read at FS
  1.1 V / -40 C with seed 1, fails on the V2.1.8 sources too and with every
  solver setting (KLU or the default solver, with and without line search);
  gmin stepping (`.OPTIONS NONLIN CONTINUATION=3`) converges it, and pass 2
  runs that one sample with it (`probes/dcop3`, `probes/dcop18`).
* **Read clocks.** The 6T 256x4 read at SS 0.9 V / 125 C with seed 1 had the
  data at the output 58 ps before the capture edge, inside the checker's
  0.02 T guard (section 6).

Pass 2 runs the same sweep and the class-bound reads of section 6 on the
release sources. All 218 samples pass, 376,503 checks: the 189 of the
sweep (54 6T and 29 10T at SS, 24 / 23 at SF, 23 / 23 at FS, 21 / 21 at FF)
and the 29 class-bound read seeds of section 6. Worst values over the
samples: the drivers' enable never below 99.8 % of VDD while a wordline was
on (16x16 10T mux, FF), released at least 106 ps after the local wordline
was off (8x4 6T FF, seed 1), `VWEN_ACCESS_ERROR` at most 12.9 mV; between
two writes new data through every hold latch at least 113 ps before it
closed again (8x4 6T FF, seed 5), at an idle -> write or read -> write edge
126 ps before the write enable (8x4 10T FF); the previous write enable off
36 ps before the next precharge and the previous sense enable off 37 ps
before it (8x4 FF, the gate-count orderings), 138 ps before the next write
enable; the driven bitline at its rail at least 1599 ps before the wordline
started (8x4 6T SS); the smallest sense margin 0.494 V (512x4 6T mux SS,
seed 2; 0.3 V required); the slowest write flips the cell 997 ps after the
clock fall (512x4 6T mux SS). At FS 1.1 V / -40 C the reads keep 1.6 ns
before the deadline and the orderings 39 ps; SF 0.9 V / 125 C writes flip the
cell sooner than SS ones (8x4: 495 against 550 ps after the clock fall, 6T),
so with the row-scaled drivers write ability is not the limiting corner.

Per group (size, cell, corner, operation), worst value over its seeds (`record-mc-table.md`; every sample is in `record-mc-cases.md` and in the JSON record). Seeds are `20261900 + N`.

| Size | Cell | Mux | Corner | Operation | Seeds | Pass | Checks | w_en min during WL [VDD] | w_en release after WL off [ps] | VWEN_ACCESS [V] | W->W latch window [ps] | Data before w_en [ps] | Rail before WL [ps] | WL release to deadline [ps] | s_en / w_en off to PRE [ps] | s_en off to w_en [ps] | Sense margin [V] | Clock to Q [ps] |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128x8 | 10T | yes | SS 0.9 V / 125 C | read | 2 | 2 | 6902 | - | - | - | - | - | - | 204.2 | 202.3 / - | - | 0.8 | - |
| 128x8 | 6T_ | no | SS 0.9 V / 125 C | read | 4 | 4 | 13804 | - | - | - | - | - | - | 220.0 | 209.8 / - | - | 0.777 | - |
| 128x8 | 6T_ | yes | SS 0.9 V / 125 C | read | 4 | 4 | 13804 | - | - | - | - | - | - | 35.2 | 208.8 / - | - | 0.806 | - |
| 16x16 | 10T | yes | FF 1.1 V / -40 C | read | 2 | 2 | 2054 | - | - | - | - | - | - | 1864.0 | 49.7 / - | - | 1.084 | - |
| 16x16 | 10T | yes | FF 1.1 V / -40 C | read&write | 2 | 2 | 14454 | 0.998 | 122.3 | 0.0129 | - | 140.6 | 2344.9 | 1856.9 | - / 47.7 | 150.0 | 1.084 | 154.2 |
| 16x16 | 10T | yes | FF 1.1 V / -40 C | write | 2 | 2 | 2436 | 0.998 | 124.7 | 0.0061 | 125.4 | 138.8 | 2393.9 | - | - / - | - | - | 152.3 |
| 16x16 | 10T | yes | FS 1.1 V / -40 C | read | 2 | 2 | 2054 | - | - | - | - | - | - | 1833.2 | 53.5 / - | - | 1.087 | - |
| 16x16 | 10T | yes | FS 1.1 V / -40 C | read&write | 2 | 2 | 14454 | 0.999 | 132.2 | 0.0126 | - | 148.7 | 2335.7 | 1824.3 | - / 51.6 | 159.8 | 1.087 | 163.0 |
| 16x16 | 10T | yes | FS 1.1 V / -40 C | write | 2 | 2 | 2436 | 0.998 | 134.2 | 0.0064 | 132.2 | 147.0 | 2389.6 | - | - / - | - | - | 160.6 |
| 16x16 | 10T | yes | SF 0.9 V / 125 C | read | 2 | 2 | 2054 | - | - | - | - | - | - | 528.8 | 183.2 / - | - | 0.872 | - |
| 16x16 | 10T | yes | SF 0.9 V / 125 C | read&write | 2 | 2 | 14454 | 0.999 | 397.7 | 0.0106 | - | 462.4 | 1859.2 | 542.6 | - / 175.5 | 495.7 | 0.872 | 539.1 |
| 16x16 | 10T | yes | SF 0.9 V / 125 C | write | 2 | 2 | 2436 | 0.999 | 407.3 | 0.0037 | 386.2 | 451.6 | 2000.6 | - | - / - | - | - | 504.4 |
| 16x16 | 10T | yes | SS 0.9 V / 125 C | read | 2 | 2 | 2054 | - | - | - | - | - | - | 317.5 | 208.7 / - | - | 0.881 | - |
| 16x16 | 10T | yes | SS 0.9 V / 125 C | read&write | 2 | 2 | 14454 | 0.999 | 457.4 | 0.0095 | - | 519.2 | 1796.7 | 329.5 | - / 199.8 | 556.8 | 0.882 | 594.9 |
| 16x16 | 10T | yes | SS 0.9 V / 125 C | write | 2 | 2 | 2436 | 0.999 | 465.3 | 0.0044 | 432.1 | 503.8 | 1958.4 | - | - / - | - | - | 554.2 |
| 16x16 | 6T_ | yes | FF 1.1 V / -40 C | read | 2 | 2 | 2054 | - | - | - | - | - | - | 1755.9 | 49.3 / - | - | 1.088 | - |
| 16x16 | 6T_ | yes | FF 1.1 V / -40 C | read&write | 2 | 2 | 14454 | 0.998 | 124.7 | 0.0129 | - | 138.9 | 2215.0 | 1737.6 | - / 47.4 | 152.2 | 1.088 | 144.6 |
| 16x16 | 6T_ | yes | FF 1.1 V / -40 C | write | 2 | 2 | 2436 | 0.998 | 124.8 | 0.0064 | 125.0 | 140.3 | 2266.1 | - | - / - | - | - | 143.6 |
| 16x16 | 6T_ | yes | FS 1.1 V / -40 C | read | 2 | 2 | 2054 | - | - | - | - | - | - | 1722.2 | 53.1 / - | - | 1.089 | - |
| 16x16 | 6T_ | yes | FS 1.1 V / -40 C | read&write | 2 | 2 | 14454 | 0.998 | 134.5 | 0.0127 | - | 147.2 | 2207.2 | 1703.3 | - / 51.2 | 161.9 | 1.09 | 154.3 |
| 16x16 | 6T_ | yes | FS 1.1 V / -40 C | write | 2 | 2 | 2436 | 0.998 | 134.2 | 0.0066 | 132.4 | 148.7 | 2263.1 | - | - / - | - | - | 152.9 |
| 16x16 | 6T_ | yes | SF 0.9 V / 125 C | read | 2 | 2 | 2054 | - | - | - | - | - | - | 545.0 | 183.1 / - | - | 0.875 | - |
| 16x16 | 6T_ | yes | SF 0.9 V / 125 C | read&write | 2 | 2 | 14454 | 0.998 | 402.7 | 0.0099 | - | 452.9 | 1717.6 | 405.4 | - / 175.2 | 509.8 | 0.874 | 483.5 |
| 16x16 | 6T_ | yes | SF 0.9 V / 125 C | write | 2 | 2 | 2436 | 0.999 | 405.2 | 0.004 | 393.2 | 453.9 | 1887.1 | - | - / - | - | - | 480.6 |
| 16x16 | 6T_ | yes | SS 0.9 V / 125 C | read | 2 | 2 | 2054 | - | - | - | - | - | - | 340.3 | 207.5 / - | - | 0.882 | - |
| 16x16 | 6T_ | yes | SS 0.9 V / 125 C | read&write | 2 | 2 | 14454 | 0.999 | 461.8 | 0.009 | - | 509.1 | 1652.3 | 192.3 | - / 199.3 | 575.6 | 0.882 | 540.5 |
| 16x16 | 6T_ | yes | SS 0.9 V / 125 C | write | 2 | 2 | 2436 | 0.999 | 467.6 | 0.0047 | 438.9 | 511.8 | 1843.8 | - | - / - | - | - | 532.3 |
| 256x4 | 6T_ | no | SF 0.9 V / 125 C | write | 1 | 1 | 3702 | 1.0 | 380.2 | 0.0024 | 406.9 | 432.7 | 3150.4 | - | - / - | - | - | 653.0 |
| 256x4 | 6T_ | no | SS 0.9 V / 125 C | read | 8 | 8 | 29240 | - | - | - | - | - | - | -3.4 | 203.8 / - | - | 0.641 | - |
| 256x4 | 6T_ | no | SS 0.9 V / 125 C | write | 1 | 1 | 3702 | 1.0 | 440.6 | 0.0032 | 468.4 | 482.0 | 3126.6 | - | - / - | - | - | 722.5 |
| 256x4 | 6T_ | yes | SS 0.9 V / 125 C | read | 4 | 4 | 14620 | - | - | - | - | - | - | 132.5 | 206.6 / - | - | 0.731 | - |
| 512x4 | 10T | yes | SS 0.9 V / 125 C | read | 2 | 2 | 2250 | - | - | - | - | - | - | -162.8 | 226.8 / - | - | 0.67 | - |
| 512x4 | 6T_ | no | SS 0.9 V / 125 C | read | 3 | 3 | 3375 | - | - | - | - | - | - | -235.0 | 215.2 / - | - | 0.612 | - |
| 512x4 | 6T_ | yes | SS 0.9 V / 125 C | read | 4 | 4 | 4500 | - | - | - | - | - | - | 50.2 | 213.7 / - | - | 0.494 | - |
| 512x4 | 6T_ | yes | SS 0.9 V / 125 C | write | 1 | 1 | 1172 | 1.0 | 465.3 | 0.0022 | 458.2 | 480.2 | 4961.4 | - | - / - | - | - | 996.6 |
| 64x16 | 10T | yes | FS 1.1 V / -40 C | read | 2 | 2 | 6854 | - | - | - | - | - | - | 1994.7 | 54.8 / - | - | 1.065 | - |
| 64x16 | 10T | yes | SF 0.9 V / 125 C | write | 2 | 2 | 7236 | 0.999 | 420.3 | 0.0039 | 383.3 | 434.9 | 2307.1 | - | - / - | - | - | 499.8 |
| 64x16 | 10T | yes | SS 0.9 V / 125 C | read | 2 | 2 | 6854 | - | - | - | - | - | - | 270.3 | 215.4 / - | - | 0.868 | - |
| 64x16 | 10T | yes | SS 0.9 V / 125 C | write | 2 | 2 | 7236 | 0.999 | 484.1 | 0.0043 | 430.3 | 489.2 | 2265.7 | - | - / - | - | - | 558.5 |
| 64x16 | 6T_ | yes | FS 1.1 V / -40 C | read | 2 | 2 | 6854 | - | - | - | - | - | - | 1759.6 | 55.1 / - | - | 1.074 | - |
| 64x16 | 6T_ | yes | SF 0.9 V / 125 C | write | 2 | 2 | 7236 | 0.999 | 397.0 | 0.0041 | 389.5 | 463.0 | 2040.8 | - | - / - | - | - | 523.5 |
| 64x16 | 6T_ | yes | SS 0.9 V / 125 C | read | 2 | 2 | 6854 | - | - | - | - | - | - | 81.2 | 214.2 / - | - | 0.871 | - |
| 64x16 | 6T_ | yes | SS 0.9 V / 125 C | write | 2 | 2 | 7236 | 0.999 | 457.0 | 0.0046 | 442.2 | 519.0 | 1996.3 | - | - / - | - | - | 582.4 |
| 8x4 | 10T | no | FF 1.1 V / -40 C | read | 5 | 5 | 915 | - | - | - | - | - | - | 1885.9 | 37.5 / - | - | 1.091 | - |
| 8x4 | 10T | no | FF 1.1 V / -40 C | read&write | 5 | 5 | 6875 | 0.999 | 107.7 | 0.0071 | - | 127.7 | 2357.0 | 1883.6 | - / 36.1 | 138.0 | 1.091 | 149.7 |
| 8x4 | 10T | no | FF 1.1 V / -40 C | write | 5 | 5 | 1150 | 0.999 | 106.2 | 0.0031 | 113.1 | 126.0 | 2403.2 | - | - / - | - | - | 151.6 |
| 8x4 | 10T | no | FS 1.1 V / -40 C | read | 5 | 5 | 915 | - | - | - | - | - | - | 1852.9 | 40.5 / - | - | 1.093 | - |
| 8x4 | 10T | no | FS 1.1 V / -40 C | read&write | 5 | 5 | 6875 | 0.999 | 117.1 | 0.0071 | - | 135.0 | 2348.7 | 1850.2 | - / 39.2 | 146.6 | 1.092 | 159.1 |
| 8x4 | 10T | no | FS 1.1 V / -40 C | write | 5 | 5 | 1150 | 0.999 | 115.8 | 0.0031 | 120.3 | 133.4 | 2399.1 | - | - / - | - | - | 161.0 |
| 8x4 | 10T | no | SF 0.9 V / 125 C | read | 5 | 5 | 915 | - | - | - | - | - | - | 647.2 | 140.9 / - | - | 0.877 | - |
| 8x4 | 10T | no | SF 0.9 V / 125 C | read&write | 5 | 5 | 6875 | 0.999 | 351.2 | 0.0077 | - | 415.5 | 1913.2 | 634.9 | - / 138.2 | 449.5 | 0.878 | 511.1 |
| 8x4 | 10T | no | SF 0.9 V / 125 C | write | 5 | 5 | 1150 | 0.999 | 337.7 | 0.0024 | 349.7 | 409.0 | 2063.2 | - | - / - | - | - | 538.4 |
| 8x4 | 10T | no | SS 0.9 V / 125 C | read | 5 | 5 | 915 | - | - | - | - | - | - | 439.3 | 160.8 / - | - | 0.886 | - |
| 8x4 | 10T | no | SS 0.9 V / 125 C | read&write | 5 | 5 | 6875 | 0.999 | 410.9 | 0.0075 | - | 467.0 | 1853.1 | 426.8 | - / 157.8 | 506.8 | 0.887 | 568.5 |
| 8x4 | 10T | no | SS 0.9 V / 125 C | write | 5 | 5 | 1150 | 0.999 | 398.4 | 0.0024 | 394.4 | 458.4 | 2023.1 | - | - / - | - | - | 602.3 |
| 8x4 | 6T_ | no | FF 1.1 V / -40 C | read | 5 | 5 | 915 | - | - | - | - | - | - | 1648.4 | 37.3 / - | - | 1.093 | - |
| 8x4 | 6T_ | no | FF 1.1 V / -40 C | read&write | 5 | 5 | 6875 | 0.999 | 106.9 | 0.0071 | - | 127.5 | 2101.7 | 1645.2 | - / 36.4 | 139.0 | 1.093 | 141.6 |
| 8x4 | 6T_ | no | FF 1.1 V / -40 C | write | 5 | 5 | 1150 | 0.999 | 106.1 | 0.0031 | 112.7 | 127.5 | 2152.7 | - | - / - | - | - | 141.2 |
| 8x4 | 6T_ | no | FS 1.1 V / -40 C | read | 5 | 5 | 915 | - | - | - | - | - | - | 1615.9 | 40.4 / - | - | 1.094 | - |
| 8x4 | 6T_ | no | FS 1.1 V / -40 C | read&write | 5 | 5 | 6875 | 0.999 | 116.0 | 0.0071 | - | 134.7 | 2094.4 | 1612.1 | - / 39.5 | 148.5 | 1.094 | 150.8 |
| 8x4 | 6T_ | no | FS 1.1 V / -40 C | write | 5 | 5 | 1150 | 0.999 | 115.5 | 0.0031 | 119.7 | 134.9 | 2149.4 | - | - / - | - | - | 150.6 |
| 8x4 | 6T_ | no | SF 0.9 V / 125 C | read | 5 | 5 | 915 | - | - | - | - | - | - | 439.3 | 142.1 / - | - | 0.885 | - |
| 8x4 | 6T_ | no | SF 0.9 V / 125 C | read&write | 5 | 5 | 6875 | 0.999 | 342.0 | 0.0075 | - | 420.6 | 1659.9 | 417.7 | - / 144.5 | 465.5 | 0.886 | 481.6 |
| 8x4 | 6T_ | no | SF 0.9 V / 125 C | write | 5 | 5 | 1150 | 0.999 | 328.9 | 0.0024 | 351.9 | 408.1 | 1839.5 | - | - / - | - | - | 494.8 |
| 8x4 | 6T_ | no | SS 0.9 V / 125 C | read | 5 | 5 | 915 | - | - | - | - | - | - | 231.0 | 161.4 / - | - | 0.889 | - |
| 8x4 | 6T_ | no | SS 0.9 V / 125 C | read&write | 5 | 5 | 6875 | 0.999 | 398.9 | 0.0074 | - | 471.1 | 1599.3 | 206.6 | - / 164.6 | 526.8 | 0.89 | 536.5 |
| 8x4 | 6T_ | no | SS 0.9 V / 125 C | write | 5 | 5 | 1150 | 0.999 | 388.1 | 0.0025 | 394.8 | 460.0 | 1804.7 | - | - / - | - | - | 549.8 |

## 6. Read clocks under mismatch

The read path is the V2.1.8 one (the nominal 256x4 6T read at SS on the
V2.1.8 and V2.1.9 sources differs by at most 1 ps at every event,
`probes/rt18`). The sense trigger is replica timed with a single replica
cell (`K = 1`) and nine delay stages, so the mismatch of that cell and of the
chain moves the read output against the clock. At the V2.1.8 clocks, SS
0.9 V / 125 C, the margin from the local read output below 0.1 VDD to the
1.2 T deadline at each class bound (`probes/class_margins.py` over `pass1/`
and `probes/rt19*`; the same seed number draws a different sample on the two
trees, because the added control transistors shift the per-device stream):

| Cell | Mux | Bound | T [ns] | access [ps] | nominal margin [ps] | seed margins [ps] | worst shift [ps] | 0.02 T + 10 % access [ps] |
|---|---|---|---:|---:|---:|---|---:|---:|
| 10T | no | 8x4 | 5.00 | 1820 | 680 | - | - | 282 |
| 10T | no | 256x4 | 8.00 | 3040 | 960 | - | - | 464 |
| 10T | yes | 64x16 | 5.50 | 2244 | 506 | 444, 494 | -62 | 334 |
| 10T | yes | 128x8 | 6.00 | 2588 | 412 | - | - | 379 |
| 10T | yes | 512x4 | 10.00 | 4580 | 420 | - | - | 658 |
| 6T | no | 8x4 | 4.50 | 1776 | 474 | - | - | 268 |
| 6T | no | 32x16 | 4.50 | 1920 | 330 | 246, 274, 288, 290 | -84 | 282 |
| 6T | no | 64x16 | 4.75 | 2067 | 308 | 238, 334, 348, 356 | -70 | 302 |
| 6T | no | 128x8 | 5.25 | 2323 | 302 | 312, 344, 444 | 10 | 337 |
| 6T | no | 256x4 | 6.25 | 2863 | 262 | 48, 58, 172, 230, 240, 254, 344, 354 | -214 | 411 |
| 6T | no | 512x4 | 9.00 | 4010 | 490 | 158, 326, 466 | -332 | 581 |
| 6T | yes | 16x16 | 4.75 | 1927 | 448 | 490, 502 | 42 | 288 |
| 6T | yes | 64x16 | 5.00 | 2166 | 334 | 246, 304 | -88 | 317 |
| 6T | yes | 128x8 | 5.50 | 2478 | 272 | 112, 258, 288 | -160 | 358 |
| 6T | yes | 256x4 | 6.75 | 3081 | 294 | 188, 342, 362 | -106 | 443 |
| 6T | yes | 512x4 | 9.00 | 4298 | 202 | 66, 240 | -136 | 610 |

Mismatch moved the output by up to 84 ps at 32 rows, 88 ps at 64, 160 ps at
128, 214 ps at 256 and 332 ps at 512 rows, up to 8 % of the nominal access
time. Two of eight 256x4 6T seeds and one 512x4 6T-mux seed had the data
inside the checker's guard; a 128x8 6T-mux seed kept 2 ps over it. The
V2.1.3 / V2.1.4 rule (250 ps nominal at every bound) does not cover that at
128 rows and more.

**Rule (V2.1.9):** at every class bound, SS 0.9 V / 125 C nominal, the local
read output leads the 1.2 T deadline by at least 0.02 T plus 10 % of the
nominal access time (clock fall to the output), and by at least 250 ps.
Every raised budget is the smallest 100 ps step that meets it (the next
lower one fails it):

| Ladder | 128 rows | 256 rows | 512 rows | Unchanged |
|---|---|---|---|---|
| shared (6T without a mux) | 2100 -> 2200 ps (5.25 -> 5.5 ns) | 2500 -> 2700 ps (6.25 -> 6.75 ns) | 3600 -> 3700 ps (9 -> 9.25 ns) | 32, 64 rows |
| 6T with a mux | 2200 -> 2300 ps (5.5 -> 5.75 ns) | 2700 -> 2900 ps (6.75 -> 7.25 ns) | 3600 -> 4000 ps (9 -> 10 ns) | 32, 64 rows |
| 10T | 2400 ps (6 ns, 412 ps against 379 ps) | 3200 ps (8 ns) | 4000 -> 4200 ps (10 -> 10.5 ns) | 32, 64, 128, 256 rows |

The column classes are unchanged (the wide arrays keep 480 ps at 16x64 and
912 ps at 8x128 with a mux). Beyond 512 rows each ladder extrapolates its own
final ratio, which the raised 256-row budget flattens for the shared ladder:
a 513-row 6T array without a mux now extrapolates to 12.7 ns instead of 13 ns
(with a mux 13.8 ns, 10T 13.8 ns); extrapolated clocks remain unqualified.

At the new clocks (pass 2, part 3; margins from `record-summary.json`):

| Cell | Mux | Bound | T [ns] | access [ps] | nominal margin [ps] | 0.02 T + 10 % access [ps] | seed margins [ps] | smallest seed margin - 0.02 T [ps] |
|---|---|---|---:|---:|---:|---:|---|---:|
| 10T | no | 8x4 | 5.0 | 1820 | 680 | 282 | 658, 664, 664, 666, 700 | 558 |
| 10T | no | 256x4 | 8.0 | 3040 | 960 | 464 | - | - |
| 10T | yes | 16x16 | 5.0 | 1972 | 528 | 297 | 474, 520 | 374 |
| 10T | yes | 64x16 | 5.5 | 2244 | 506 | 334 | 444, 494 | 334 |
| 10T | yes | 128x8 | 6.0 | 2588 | 412 | 379 | 424, 426 | 304 |
| 10T | yes | 256x4 | 8.0 | 3264 | 736 | 486 | - | - |
| 10T | yes | 512x4 | 10.5 | 4588 | 662 | 669 | 316, 610 | 106 |
| 6T | no | 8x4 | 4.5 | 1776 | 474 | 268 | 430, 460, 466, 482, 496 | 340 |
| 6T | no | 128x8 | 5.5 | 2328 | 422 | 343 | 432, 460, 464, 564 | 322 |
| 6T | no | 256x4 | 6.75 | 2869 | 506 | 422 | 290, 298, 416, 474, 484, 496, 586, 596 | 155 |
| 6T | no | 512x4 | 9.25 | 4013 | 612 | 586 | 282, 446, 590 | 97 |
| 6T | yes | 8x128 | 6.0 | 2088 | 912 | 329 | - | - |
| 6T | yes | 16x16 | 4.75 | 1927 | 448 | 288 | 490, 502 | 395 |
| 6T | yes | 64x16 | 5.0 | 2166 | 334 | 317 | 246, 304 | 146 |
| 6T | yes | 128x8 | 5.75 | 2481 | 394 | 363 | 232, 352, 380, 410 | 117 |
| 6T | yes | 256x4 | 7.25 | 3087 | 538 | 454 | 430, 580, 584, 606 | 285 |
| 6T | yes | 512x4 | 10.0 | 4314 | 686 | 631 | 552, 728, 814, 952 | 352 |

Every seed now has the data at the output at least 97 ps before the
checker's guard (512x4 6T, seed 2, 330 ps behind its nominal). One bound
misses the rule by a hair: the 10T 512x4 read keeps 662 ps against 669 ps at
10.5 ns, because its access is 8 ps longer at the longer clock than at the
10 ns clock the budget was derived from (4588 against 4580 ps); its seeds keep
106 and 400 ps over the guard. 4300 ps (10.75 ns) would close the 7 ps; the
class is left at 4200 ps and reported here.

## 7. Evidence

`gen_cases.py` builds 313 cases in 45 queues (15 four-rank, 30 single-rank):
part 1 is the V2.1.8 matrix with the same names and settings (the V2.1.8
fix-pass settings built in: Newton line search for the 8x128 mux read, the
16x16 mux read and the 16x64 mux read idle probe); part 2 is the worst-PVT
sweep of section 5 (seeds `20261900 + N`; the FS -40 C 10T sample 1 with
gmin stepping); part 3 the class-bound reads of section 6. Every queue
archives its sources. Pass 1 (`pass1/`) ran parts 1 and 2 on the first
V2.1.9 sources, before the latch seed and the new clocks, and was stopped
once it had found them (sections 5, 6); pass 2 ran all three parts on the
release sources, and the fix pass `fix-A` reran its two four-rank DC
operating-point aborts (the 128x8 mux reads, 6T and 10T; both had converged
in pass 1) with Newton line search. The record keeps the latest attempt of
every case.

Every case of the V2.1.8 matrix and the four nominal class-bound reads, latest attempt (`outputs/validation/V2.1.9-write-hold/`, `assemble_record.py`; the JSON record, with the Monte-Carlo samples, is `WRITE_HOLD_V2_1_9.json` next to this file). w_en min during WL: the lowest driver enable, as a fraction of VDD, while the local wordline of the same column is above 0.1 VDD (every write cycle and column); w_en release after WL off: the smallest distance from the local wordline below 0.1 VDD to the driver's enable falling through 0.9 VDD; VWEN_ACCESS: the runtime measure (highest wordline level while the target driver's enable is below 0.9 VDD); W->W latch window: in single write decks, the smallest distance from the next data at a driver's hold-latch output (50 %) to the latch closing again (its enable at the column falling through 50 %); the ordering columns as in the V2.1.8 record; data before w_en: new data through the hold latch before the column's enable rises, for the checked write cycles (idle -> write, read -> write and the first write of a deck). Seeds `sN` are `20261700 + N`.

| Case | Cell | Size | Mux | Operation | Corner | Variation | T [ns] | Checks | w_en min during WL [VDD] | w_en release after WL off [ps] | VWEN_ACCESS [V] | W->W latch window [ps] | s_en / w_en off to PRE [ps] | s_en / w_en off to w_en [ps] | WL release to deadline [ps] | Rail before WL [ps] | Data before w_en [ps] | Sense margin [V] | TWSLOT / TRESTORE [ps] | Result |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| p_16x16_10t_SS_write_idle2 | 10T | 16x16 | no | write (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 1989 | 0.999 | 467.1 | 0.0011 | - | - / - | - / 4808.6 | - | 2002.2 | 456.7 | - | 965.0 | pass |
| p_16x16_6t_mux_SS_read_idle2 | 6T_ | 16x16 | yes | read (every 2) | SS 0.9 V / 125 C | nominal | 4.75 | 1608 | - | - | - | - | 5009.1 / - | - / - | 298.1 | - | - | 0.885 | 793.0 | pass |
| p_16x16_6t_mux_SS_write_idle2 | 6T_ | 16x16 | yes | write (every 2) | SS 0.9 V / 125 C | nominal | 4.75 | 1989 | 0.999 | 466.6 | 0.0016 | - | - / - | - / 4559.6 | - | 1866.6 | 456.8 | - | 973.3 | pass |
| p_16x64_6t_mux_SS_read_idle2 | 6T_ | 16x64 | yes | read (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 6216 | - | - | - | - | 5230.4 / - | - / - | 340.9 | - | - | 0.887 | 825.2 | pass |
| p_2x4_6t_FF_cold_read_idle2 | 6T_ | 2x4 | no | read (every 2) | FF 1.1 V / -40 C | nominal | 4.5 | 134 | - | - | - | - | 4557.9 / - | - / - | 1669.3 | - | - | 1.094 | 214.8 | pass |
| p_4x4_6t_FF_cold_write_idle2 | 6T_ | 4x4 | no | write (every 2) | FF 1.1 V / -40 C | nominal | 4.5 | 273 | 0.999 | 108.4 | 0.0033 | - | - / - | - / 4430.9 | - | 2156.4 | 112.6 | - | 244.7 | pass |
| p_64x16_6t_mux_SS_read_idle2 | 6T_ | 64x16 | yes | read (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 5592 | - | - | - | - | 5243.4 / - | - / - | 172.1 | - | - | 0.867 | 797.1 | pass |
| p_64x16_6t_mux_SS_write_idle2 | 6T_ | 64x16 | yes | write (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 5973 | 0.999 | 470.8 | 0.0014 | - | - / - | - / 4779.9 | - | 2031.4 | 443.8 | - | 960.6 | pass |
| p_8x4_10t_FF_cold_read_idle2 | 10T | 8x4 | no | read (every 2) | FF 1.1 V / -40 C | nominal | 5.0 | 272 | - | - | - | - | 5057.4 / - | - / - | 1888.8 | - | - | 1.091 | 218.4 | pass |
| p_8x4_10t_FF_cold_write_idle2 | 10T | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | nominal | 5.0 | 365 | 0.999 | 108.5 | 0.003 | - | - / - | - / 4932.6 | - | 2407.6 | 112.8 | - | 249.1 | pass |
| p_8x4_6t_FF_cold_read_idle2 | 6T_ | 8x4 | no | read (every 2) | FF 1.1 V / -40 C | nominal | 4.5 | 272 | - | - | - | - | 4557.3 / - | - / - | 1651.1 | - | - | 1.094 | 218.5 | pass |
| p_8x4_6t_FF_cold_read_idle2_pd_s1 | 6T_ | 8x4 | no | read (every 2) | FF 1.1 V / -40 C | per-device s1 | 4.5 | 272 | - | - | - | - | 4556.8 / - | - / - | 1653.6 | - | - | 1.094 | 217.2 | pass |
| p_8x4_6t_FF_cold_read_idle2_pd_s2 | 6T_ | 8x4 | no | read (every 2) | FF 1.1 V / -40 C | per-device s2 | 4.5 | 272 | - | - | - | - | 4557.9 / - | - / - | 1652.0 | - | - | 1.093 | 217.6 | pass |
| p_8x4_6t_FF_cold_write_idle2 | 6T_ | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | nominal | 4.5 | 365 | 0.999 | 108.6 | 0.003 | - | - / - | - / 4432.5 | - | 2153.6 | 112.8 | - | 248.8 | pass |
| p_8x4_6t_FF_cold_write_idle2_pd_s1 | 6T_ | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | per-device s1 | 4.5 | 365 | 0.999 | 112.4 | 0.003 | - | - / - | - / 4429.7 | - | 2152.3 | 112.6 | - | 250.1 | pass |
| p_8x4_6t_FF_cold_write_idle2_pd_s2 | 6T_ | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | per-device s2 | 4.5 | 365 | 0.999 | 107.8 | 0.003 | - | - / - | - / 4430.2 | - | 2154.7 | 110.2 | - | 247.7 | pass |
| p_8x4_6t_FF_cold_write_idle2_pd_s3 | 6T_ | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | per-device s3 | 4.5 | 365 | 0.999 | 114.8 | 0.0029 | - | - / - | - / 4429.7 | - | 2151.6 | 113.4 | - | 250.7 | pass |
| p_8x4_6t_FF_cold_write_idle3 | 6T_ | 8x4 | no | write (every 3) | FF 1.1 V / -40 C | nominal | 4.5 | 367 | 0.999 | 108.6 | 0.003 | - | - / - | - / 8932.5 | - | 2153.6 | 112.8 | - | 248.8 | pass |
| p_8x4_6t_SS_write_idle2_pd_s1 | 6T_ | 8x4 | no | write (every 2) | SS 0.9 V / 125 C | per-device s1 | 4.5 | 365 | 0.999 | 443.3 | 0.0012 | - | - / - | - / 4252.8 | - | 1786.3 | 419.3 | - | 903.8 | pass |
| p_8x4_6t_SS_write_idle2_pd_s2 | 6T_ | 8x4 | no | write (every 2) | SS 0.9 V / 125 C | per-device s2 | 4.5 | 365 | 0.999 | 411.6 | 0.0013 | - | - / - | - / 4257.7 | - | 1801.1 | 393.4 | - | 888.8 | pass |
| p_8x4_6t_SS_write_idle2_pd_s3 | 6T_ | 8x4 | no | write (every 2) | SS 0.9 V / 125 C | per-device s3 | 4.5 | 365 | 0.999 | 474.2 | 0.001 | - | - / - | - / 4258.2 | - | 1775.3 | 424.2 | - | 907.9 | pass |
| p_8x4_6t_TT_sequence_settle0 | 6T_ | 8x4 | no | read&write (0 settling stages) | TT 1.0 V / 25 C | nominal | 4.5 | 1375 | 0.999 | 76.5 | 0.007 | - | - / 66.0 | 230.0 / - | 1324.3 | 1998.7 | 212.1 | 0.993 | - | pass |
| p_8x4_6t_TT_write_idle2 | 6T_ | 8x4 | no | write (every 2) | TT 1.0 V / 25 C | nominal | 4.5 | 365 | 0.999 | 179.9 | 0.0017 | - | - / - | - / 4396.3 | - | 2075.0 | 184.0 | - | 397.1 | pass |
| p_8x64_6t_mux_FF_cold_write_idle2 | 6T_ | 8x64 | yes | write (every 2) | FF 1.1 V / -40 C | nominal | 5.0 | 5165 | 0.995 | 126.9 | 0.006 | - | - / - | - / 4942.1 | - | 2387.4 | 128.3 | - | 279.5 | pass |
| r_128x8_10t_mux_SS_read | 10T | 128x8 | yes | read | SS 0.9 V / 125 C | nominal | 6.0 | 3451 | - | - | - | - | 211.4 / - | - / - | 210.9 | - | - | 0.807 | 763.1 | pass |
| r_128x8_6t_SS_read | 6T_ | 128x8 | no | read | SS 0.9 V / 125 C | nominal | 5.5 | 3451 | - | - | - | - | 210.8 / - | - / - | 219.5 | - | - | 0.811 | 796.4 | pass |
| r_128x8_6t_mux_SS_read | 6T_ | 128x8 | yes | read | SS 0.9 V / 125 C | nominal | 5.75 | 3451 | - | - | - | - | 211.5 / - | - / - | 191.9 | - | - | 0.826 | 763.6 | pass |
| r_16x16_10t_mux_SS_read | 10T | 16x16 | yes | read | SS 0.9 V / 125 C | nominal | 5.0 | 1027 | - | - | - | - | 207.4 / - | - / - | 377.8 | - | - | 0.881 | 743.0 | pass |
| r_16x16_6t_mux_SS_read | 6T_ | 16x16 | yes | read | SS 0.9 V / 125 C | nominal | 4.75 | 1027 | - | - | - | - | 207.1 / - | - / - | 297.8 | - | - | 0.885 | 742.2 | pass |
| r_256x4_10t_SS_read | 10T | 256x4 | no | read | SS 0.9 V / 125 C | nominal | 8.0 | 3655 | - | - | - | - | 213.1 / - | - / - | 676.8 | - | - | 0.692 | 815.2 | pass |
| r_256x4_10t_mux_SS_read | 10T | 256x4 | yes | read | SS 0.9 V / 125 C | nominal | 8.0 | 3655 | - | - | - | - | 212.9 / - | - / - | 451.9 | - | - | 0.727 | 798.5 | pass |
| r_256x4_6t_SS_read | 6T_ | 256x4 | no | read | SS 0.9 V / 125 C | nominal | 6.75 | 3655 | - | - | - | - | 211.8 / - | - / - | 220.7 | - | - | 0.716 | 814.4 | pass |
| r_256x4_6t_mux_SS_read | 6T_ | 256x4 | yes | read | SS 0.9 V / 125 C | nominal | 7.25 | 3655 | - | - | - | - | 212.8 / - | - / - | 252.5 | - | - | 0.75 | 797.8 | pass |
| r_512x4_10t_mux_SS_read | 10T | 512x4 | yes | read | SS 0.9 V / 125 C | nominal | 10.5 | 1125 | - | - | - | - | 224.3 / - | - / - | 150.3 | - | - | 0.637 | 919.8 | pass |
| r_512x4_6t_SS_read | 6T_ | 512x4 | no | read | SS 0.9 V / 125 C | nominal | 9.25 | 1125 | - | - | - | - | 224.6 / - | - / - | 99.1 | - | - | 0.621 | 925.7 | pass |
| r_512x4_6t_mux_SS_read | 6T_ | 512x4 | yes | read | SS 0.9 V / 125 C | nominal | 10.0 | 1125 | - | - | - | - | 223.0 / - | - / - | 174.3 | - | - | 0.657 | 920.6 | pass |
| r_64x16_10t_mux_SS_read | 10T | 64x16 | yes | read | SS 0.9 V / 125 C | nominal | 5.5 | 3427 | - | - | - | - | 213.5 / - | - / - | 344.9 | - | - | 0.856 | 769.8 | pass |
| r_64x16_6t_mux_SS_read | 6T_ | 64x16 | yes | read | SS 0.9 V / 125 C | nominal | 5.0 | 3427 | - | - | - | - | 212.6 / - | - / - | 173.2 | - | - | 0.867 | 769.1 | pass |
| r_8x128_6t_mux_SS_read | 6T_ | 8x128 | yes | read | SS 0.9 V / 125 C | nominal | 6.0 | 4771 | - | - | - | - | 231.0 / - | - / - | 765.3 | - | - | 0.889 | 840.5 | pass |
| r_8x4_10t_FF_cold_read | 10T | 8x4 | no | read | FF 1.1 V / -40 C | nominal | 5.0 | 183 | - | - | - | - | 37.9 / - | - / - | 1888.7 | - | - | 1.091 | 199.7 | pass |
| r_8x4_10t_FF_cold_read_pd_s1 | 10T | 8x4 | no | read | FF 1.1 V / -40 C | per-device s1 | 5.0 | 183 | - | - | - | - | 38.4 / - | - / - | 1889.9 | - | - | 1.091 | 200.2 | pass |
| r_8x4_10t_SS_read | 10T | 8x4 | no | read | SS 0.9 V / 125 C | nominal | 5.0 | 183 | - | - | - | - | 167.0 / - | - / - | 475.2 | - | - | 0.889 | 741.9 | pass |
| r_8x4_6t_FF_cold_read | 6T_ | 8x4 | no | read | FF 1.1 V / -40 C | nominal | 4.5 | 183 | - | - | - | - | 38.0 / - | - / - | 1651.1 | - | - | 1.094 | 199.6 | pass |
| r_8x4_6t_FF_cold_read_pd_s1 | 6T_ | 8x4 | no | read | FF 1.1 V / -40 C | per-device s1 | 4.5 | 183 | - | - | - | - | 38.3 / - | - / - | 1653.6 | - | - | 1.094 | 199.1 | pass |
| r_8x4_6t_FF_cold_read_pd_s2 | 6T_ | 8x4 | no | read | FF 1.1 V / -40 C | per-device s2 | 4.5 | 183 | - | - | - | - | 39.1 / - | - / - | 1651.9 | - | - | 1.093 | 199.4 | pass |
| r_8x4_6t_SS_read | 6T_ | 8x4 | no | read | SS 0.9 V / 125 C | nominal | 4.5 | 183 | - | - | - | - | 166.5 / - | - / - | 268.4 | - | - | 0.891 | 740.8 | pass |
| s_16x16_10t_SS_sequence | 10T | 16x16 | no | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 7227 | 0.999 | 467.0 | 0.0077 | - | - / 204.5 | 565.4 / - | 411.7 | 1789.1 | 524.0 | 0.888 | - | pass |
| s_16x16_10t_SS_sequence_pd_s1 | 10T | 16x16 | no | read&write | SS 0.9 V / 125 C | per-device s1 | 5.0 | 7227 | 0.999 | 468.0 | 0.0075 | - | - / 203.0 | 580.2 / - | 445.4 | 1736.1 | 537.8 | 0.886 | - | pass |
| s_16x16_10t_SS_sequence_pd_s2 | 10T | 16x16 | no | read&write | SS 0.9 V / 125 C | per-device s2 | 5.0 | 7227 | 0.999 | 462.4 | 0.0074 | - | - / 198.6 | 570.1 / - | 447.1 | 1722.6 | 502.9 | 0.889 | - | pass |
| s_16x16_10t_mux_SS_sequence | 10T | 16x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 7227 | 0.999 | 466.7 | 0.0092 | - | - / 204.5 | 566.0 / - | 345.9 | 1813.7 | 523.1 | 0.881 | - | pass |
| s_16x16_6t_SS_sequence | 6T_ | 16x16 | no | read&write | SS 0.9 V / 125 C | nominal | 4.5 | 7227 | 0.999 | 466.3 | 0.0077 | - | - / 204.4 | 565.0 / - | 209.5 | 1532.7 | 523.5 | 0.891 | - | pass |
| s_16x16_6t_mux_SS_sequence | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 4.75 | 7227 | 0.999 | 466.7 | 0.0092 | - | - / 204.2 | 565.4 / - | 267.0 | 1689.4 | 523.7 | 0.885 | - | pass |
| s_16x16_6t_mux_SS_sequence_pd_s1 | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | per-device s1 | 4.75 | 7227 | 0.999 | 463.4 | 0.0094 | - | - / 203.6 | 592.3 / - | 250.7 | 1643.6 | 513.9 | 0.883 | - | pass |
| s_16x16_6t_mux_SS_sequence_pd_s2 | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | per-device s2 | 4.75 | 7227 | 0.999 | 478.6 | 0.009 | - | - / 201.1 | 571.2 / - | 218.2 | 1673.2 | 520.1 | 0.886 | - | pass |
| s_16x16_6t_mux_SS_sequence_pd_s3 | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | per-device s3 | 4.75 | 7227 | 0.999 | 467.1 | 0.009 | - | - / 200.2 | 576.0 / - | 244.1 | 1663.1 | 512.6 | 0.882 | - | pass |
| s_16x32_6t_mux_SS_sequence | 6T_ | 16x32 | yes | read&write | SS 0.9 V / 125 C | nominal | 4.75 | 14219 | 0.998 | 475.9 | 0.0102 | - | - / 214.4 | 574.1 / - | 219.1 | 1638.7 | 532.7 | 0.887 | - | pass |
| s_32x16_10t_mux_SS_sequence | 10T | 32x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 11707 | 0.999 | 467.4 | 0.0075 | - | - / 204.5 | 566.1 / - | 268.3 | 1774.6 | 523.8 | 0.875 | - | pass |
| s_32x16_6t_SS_sequence | 6T_ | 32x16 | no | read&write | SS 0.9 V / 125 C | nominal | 4.5 | 11707 | 0.999 | 467.2 | 0.007 | - | - / 204.4 | 565.8 / - | 148.5 | 1490.2 | 525.0 | 0.888 | - | pass |
| s_32x16_6t_mux_SS_sequence | 6T_ | 32x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 4.75 | 11707 | 0.999 | 466.9 | 0.0075 | - | - / 204.6 | 566.7 / - | 196.8 | 1647.1 | 524.6 | 0.881 | - | pass |
| s_8x4_10t_FF_cold_sequence | 10T | 8x4 | no | read&write | FF 1.1 V / -40 C | nominal | 5.0 | 1375 | 0.999 | 108.3 | 0.007 | - | - / 36.5 | 140.0 / - | 1886.2 | 2359.4 | 129.8 | 1.091 | - | pass |
| s_8x4_10t_mux_SS_sequence | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 1375 | 0.999 | 411.9 | 0.0089 | - | - / 163.1 | 524.4 / - | 383.7 | 1896.7 | 481.2 | 0.882 | - | pass |
| s_8x4_10t_mux_SS_sequence_pd_s1 | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | per-device s1 | 5.0 | 1375 | 0.999 | 416.3 | 0.0084 | - | - / 166.9 | 507.3 / - | 370.9 | 1896.3 | 477.0 | 0.883 | - | pass |
| s_8x4_10t_mux_SS_sequence_pd_s2 | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | per-device s2 | 5.0 | 1375 | 0.999 | 405.0 | 0.0089 | - | - / 166.4 | 527.4 / - | 399.1 | 1910.3 | 485.3 | 0.88 | - | pass |
| s_8x4_10t_mux_SS_sequence_pd_s3 | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | per-device s3 | 5.0 | 1375 | 0.999 | 396.1 | 0.01 | - | - / 166.7 | 538.4 / - | 304.0 | 1878.2 | 496.2 | 0.886 | - | pass |
| s_8x4_6t_FF_cold_sequence | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | nominal | 4.5 | 1375 | 0.999 | 108.5 | 0.007 | - | - / 36.5 | 139.9 / - | 1648.7 | 2105.7 | 129.8 | 1.094 | - | pass |
| s_8x4_6t_FF_cold_sequence_pd_s1 | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | per-device s1 | 4.5 | 1375 | 0.999 | 107.2 | 0.007 | - | - / 36.4 | 137.8 / - | 1652.3 | 2102.8 | 127.1 | 1.094 | - | pass |
| s_8x4_6t_FF_cold_sequence_pd_s2 | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | per-device s2 | 4.5 | 1375 | 0.999 | 106.6 | 0.0071 | - | - / 37.0 | 141.0 / - | 1645.0 | 2105.3 | 128.6 | 1.094 | - | pass |
| s_8x4_6t_FF_cold_sequence_pd_s3 | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | per-device s3 | 4.5 | 1375 | 0.999 | 107.8 | 0.007 | - | - / 37.0 | 140.5 / - | 1639.9 | 2104.3 | 127.1 | 1.094 | - | pass |
| s_8x4_6t_SS_sequence | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | nominal | 4.5 | 1375 | 0.999 | 411.5 | 0.0071 | - | - / 163.1 | 524.1 / - | 260.4 | 1642.9 | 481.4 | 0.891 | - | pass |
| s_8x4_6t_SS_sequence_pd_s1 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s1 | 4.5 | 1375 | 0.999 | 396.7 | 0.0073 | - | - / 167.9 | 509.2 / - | 262.9 | 1610.7 | 469.0 | 0.891 | - | pass |
| s_8x4_6t_SS_sequence_pd_s2 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s2 | 4.5 | 1375 | 0.999 | 391.5 | 0.0079 | - | - / 170.0 | 542.6 / - | 212.0 | 1630.9 | 481.9 | 0.89 | - | pass |
| s_8x4_6t_SS_sequence_pd_s3 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s3 | 4.5 | 1375 | 0.999 | 406.8 | 0.0071 | - | - / 165.9 | 527.0 / - | 167.2 | 1641.5 | 468.4 | 0.891 | - | pass |
| s_8x4_6t_SS_sequence_pd_s4 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s4 | 4.5 | 1375 | 0.999 | 394.0 | 0.0074 | - | - / 162.2 | 535.4 / - | 334.7 | 1611.2 | 479.1 | 0.892 | - | pass |
| s_8x4_6t_SS_sequence_pd_s5 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s5 | 4.5 | 1375 | 0.999 | 438.3 | 0.0065 | - | - / 165.1 | 481.6 / - | 294.2 | 1665.8 | 465.5 | 0.891 | - | pass |
| s_8x8_6t_mux_FF_cold_sequence | 6T_ | 8x8 | yes | read&write | FF 1.1 V / -40 C | nominal | 4.75 | 2579 | 0.998 | 112.8 | 0.0121 | - | - / 38.9 | 142.5 / - | 1757.0 | 2233.5 | 132.2 | 1.09 | - | pass |
| w_128x8_6t_SS_write | 6T_ | 128x8 | no | write | SS 0.9 V / 125 C | nominal | 5.5 | 3546 | 0.999 | 464.9 | 0.0034 | 459.4 | - / - | - / 567.7 | - | 2357.6 | 511.8 | - | 1766.2 | pass |
| w_16x16_10t_mux_SS_write | 10T | 16x16 | yes | write | SS 0.9 V / 125 C | nominal | 5.0 | 1218 | 0.999 | 466.9 | 0.0045 | 463.5 | - / - | - / 564.1 | - | 1995.7 | 523.8 | - | 1728.1 | pass |
| w_16x16_6t_mux_SS_write | 6T_ | 16x16 | yes | write | SS 0.9 V / 125 C | nominal | 4.75 | 1218 | 0.999 | 466.5 | 0.0047 | 463.3 | - / - | - / 563.8 | - | 1869.1 | 523.3 | - | 1726.8 | pass |
| w_16x64_6t_mux_SS_write | 6T_ | 16x64 | yes | write | SS 0.9 V / 125 C | nominal | 5.0 | 4722 | 0.998 | 483.2 | 0.0065 | 484.2 | - / - | - / 585.3 | - | 1967.0 | 541.6 | - | 1796.5 | pass |
| w_256x4_6t_SF_write | 6T_ | 256x4 | no | write | SF 0.9 V / 125 C | nominal | 6.75 | 3702 | 1.0 | 392.8 | 0.002 | 411.7 | - / - | - / 503.5 | - | 3121.2 | 446.3 | - | 1673.9 | pass |
| w_256x4_6t_SS_write | 6T_ | 256x4 | no | write | SS 0.9 V / 125 C | nominal | 6.75 | 3702 | 1.0 | 458.5 | 0.0027 | 461.6 | - / - | - / 565.8 | - | 3089.7 | 500.0 | - | 1871.2 | pass |
| w_512x4_6t_mux_SS_write | 6T_ | 512x4 | yes | write | SS 0.9 V / 125 C | nominal | 10.0 | 1172 | 1.0 | 466.2 | 0.002 | 475.8 | - / - | - / 577.1 | - | 4990.6 | 485.9 | - | 2229.8 | pass |
| w_64x16_10t_mux_SF_write | 10T | 64x16 | yes | write | SF 0.9 V / 125 C | nominal | 5.5 | 3618 | 0.999 | 408.7 | 0.0039 | 400.0 | - / - | - / 504.0 | - | 2327.7 | 461.0 | - | 1569.1 | pass |
| w_64x16_10t_mux_SS_write | 10T | 64x16 | yes | write | SS 0.9 V / 125 C | nominal | 5.5 | 3618 | 0.999 | 473.2 | 0.0043 | 454.7 | - / - | - / 570.8 | - | 2287.7 | 517.7 | - | 1753.8 | pass |
| w_64x16_6t_mux_SF_write | 6T_ | 64x16 | yes | write | SF 0.9 V / 125 C | nominal | 5.0 | 3618 | 0.999 | 409.8 | 0.0039 | 397.5 | - / - | - / 501.4 | - | 2072.6 | 459.7 | - | 1566.1 | pass |
| w_64x16_6t_mux_SS_write | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | nominal | 5.0 | 3618 | 0.999 | 470.4 | 0.0045 | 450.9 | - / - | - / 566.7 | - | 2033.0 | 517.5 | - | 1744.8 | pass |
| w_64x16_6t_mux_SS_write_pd_s1 | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | per-device s1 | 5.0 | 3618 | 0.999 | 456.9 | 0.0047 | 422.5 | - / - | - / 552.9 | - | 2013.5 | 501.4 | - | 1712.0 | pass |
| w_64x16_6t_mux_SS_write_pd_s2 | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | per-device s2 | 5.0 | 3618 | 0.999 | 464.5 | 0.0043 | 459.0 | - / - | - / 575.9 | - | 1984.8 | 504.8 | - | 1774.6 | pass |
| w_64x16_6t_mux_SS_write_pd_s3 | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | per-device s3 | 5.0 | 3618 | 0.999 | 471.0 | 0.0042 | 464.2 | - / - | - / 593.2 | - | 2002.0 | 527.8 | - | 1800.1 | pass |
| w_8x4_10t_FF_cold_write | 10T | 8x4 | no | write | FF 1.1 V / -40 C | nominal | 5.0 | 230 | 0.999 | 108.5 | 0.0031 | 114.6 | - / - | - / 138.4 | - | 2407.9 | 129.8 | - | 455.2 | pass |
| w_8x4_10t_SS_write | 10T | 8x4 | no | write | SS 0.9 V / 125 C | nominal | 5.0 | 230 | 0.999 | 411.3 | 0.0023 | 421.3 | - / - | - / 521.4 | - | 2066.7 | 481.3 | - | 1626.5 | pass |
| w_8x4_6t_FF_cold_write | 6T_ | 8x4 | no | write | FF 1.1 V / -40 C | nominal | 4.5 | 230 | 0.999 | 108.5 | 0.0031 | 114.7 | - / - | - / 138.4 | - | 2153.7 | 129.6 | - | 453.7 | pass |
| w_8x4_6t_FF_cold_write_pd_s1 | 6T_ | 8x4 | no | write | FF 1.1 V / -40 C | per-device s1 | 4.5 | 230 | 0.999 | 112.3 | 0.003 | 113.4 | - / - | - / 138.2 | - | 2153.1 | 129.8 | - | 458.7 | pass |
| w_8x4_6t_FF_cold_write_pd_s2 | 6T_ | 8x4 | no | write | FF 1.1 V / -40 C | per-device s2 | 4.5 | 230 | 0.999 | 107.9 | 0.0031 | 112.7 | - / - | - / 137.2 | - | 2154.9 | 127.5 | - | 454.9 | pass |
| w_8x4_6t_SS_write | 6T_ | 8x4 | no | write | SS 0.9 V / 125 C | nominal | 4.5 | 230 | 0.999 | 411.5 | 0.0023 | 421.4 | - / - | - / 521.5 | - | 1812.0 | 481.1 | - | 1625.2 | pass |

## 8. Cost of the write hold

Holding the drivers through the wordline tail and arming the next slot
after their release moves the write -> write slot later in the clock-high
phase. `TWSLOT` (the edge that starts the second write's clock-high phase to
its driven bitline at 0.1 VDD, single write decks, SS 0.9 V / 125 C) grows by
490 to 560 ps (section 4: 1136 -> 1625 ps at 8x4, 1191 -> 1727 ps at 16x16
with a mux, 1331 -> 1871 ps at 256x4, 1681 -> 2230 ps at 512x4 with a mux)
and now takes 45 to 73 % of the clock-high phase:

| Write deck (SS) | T/2 [ps] | TWSLOT [ps] | share of T/2 | class budget [ps] | 0.8 x budget [ps] |
|---|---:|---:|---:|---:|---:|
| w_8x4_6t_SS_write | 2250 | 1625 | 72 % | 1800 | 1440 |
| w_8x4_10t_SS_write | 2500 | 1626 | 65 % | 2000 | 1600 |
| w_16x16_6t_mux_SS_write | 2375 | 1727 | 73 % | 1900 | 1520 |
| w_16x16_10t_mux_SS_write | 2500 | 1728 | 69 % | 2000 | 1600 |
| w_16x64_6t_mux_SS_write | 2500 | 1796 | 72 % | 2000 | 1600 |
| w_64x16_6t_mux_SS_write | 2500 | 1745 | 70 % | 2000 | 1600 |
| w_64x16_10t_mux_SS_write | 2750 | 1754 | 64 % | 2200 | 1760 |
| w_128x8_6t_SS_write | 2750 | 1766 | 64 % | 2200 | 1760 |
| w_256x4_6t_SS_write | 3375 | 1871 | 55 % | 2700 | 2160 |
| w_512x4_6t_mux_SS_write | 5000 | 2230 | 45 % | 4000 | 3200 |

Every slot fits its class budget, which is the documented period rule
(`T/2 >= TWSLOT x (1 + margin)`, `docs/TIMING_AUTOCONFIG.md`), and the driven
bitline still reaches its rail at least 1.6 ns before the wordline rises.
The qualification scorer's `restore_budget` (`dev/sizing/qualification.py`)
is stricter, `TWSLOT <= 0.8 x budget`, and seven of these decks exceed it
(by 6 ps at 128x8 up to 207 ps at 16x16 with a mux). That check belongs to
the precharge / write-driver sizing review (`dev/sizing/local_review.py`),
but the added time is the wordline tail, the observer's settling and the
slot-arm handshake, which no driver size shortens; a sizing campaign at SS
would flag these classes and try to upsize the wrong devices. **Open for a
decision, not changed in this release:** score only the drivers' swing (slot
open to rail) against 0.8 x budget and the whole slot against the budget, or
grow the classes up to 128 rows by about 10 %.

The per-cycle write energy (`EWRITE`, section 4) rises 2 to 7 %: most at 8x4
FF (666 -> 711 fJ; the new gates, the settling chain and the longer driver-on
time are a larger share of a small array), 2 % at 64x16 and 16x64 with a
mux; the 256- and 512-row decks also integrate their longer V2.1.9 period.
The write access itself is unchanged: the drivers are on and the bitlines at
their rails long before the wordline rises, so the cell flips as before
(`clock_to_Q` 141 ps at 8x4 FF, 550 ps at 8x4 SS).

## 9. Limits

* Screening at the class bounds with illustrative wires; extracted metal,
  half-select writes and the yield estimator remain the carried Phase 6
  scope. The Monte-Carlo sweep has one to eight seeds per group: it shows
  that the worst corners work under mismatch and how much margin is left,
  not a failure probability. The 10 % allowance of the read-clock rule
  covers the largest shift of 34 seeds at the class bounds (8.3 %), not a
  sigma target; a yield-grade clock needs the estimator.
* The read margin under mismatch is set by the single replica cell. A
  stronger replica (`K > 1`) would narrow the spread instead of paying for it
  in the clock; that is a sizing-policy change with its own evidence.
* The drivers now stay on for the wordline's tail plus the observer's
  settling (0.1 to 0.5 ns after the wordline is off); a write -> read
  precharge follows later by the same amount, the next write slot by the
  same amount plus the slot-arm stages. Both stay inside the clock-high
  phase at every class bound (section 8).
* The orderings after a write are gate-count margins, as in V2.1.8: the
  precharge of a write -> read boundary follows the write enable off by the
  NOR, the precharge gate and the precharge NAND3 (36 ps at 8x4 FF -40 C).
* The write-data latch window is a delay-line margin (four unit stages plus
  the slot path), not an observation of the latch at the far column; it
  scales with the gate corner, not with the `w_en_bar` wire.
