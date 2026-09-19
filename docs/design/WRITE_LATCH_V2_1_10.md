# V2.1.10: the write data is held by the registered write; the drivers stay on between writes

Executed September 18 and 19, 2026 (US Pacific time) on the V2.1.10 sources (base commit
`81ecfa0` plus the changes of this release; every queue archives the sources it
ran). A review of the [V2.1.9 release](WRITE_HOLD_V2_1_9.md) set out to close its
two open items: the write -> write slot against the qualification scorer's
budget, and the 10T 512-row read class 7 ps short of its rule. The review found
the scorer item misdiagnosed, the V2.1.9 slot failing a runtime check at four
class bounds, and the slot itself avoidable: the write data and the write
request are registered on the rising clock edge, so between two writes only the
data changes, and the drivers do not have to turn off and on again. Illustrative
wires, V2.0.9 driver classes and the V2.1.9 clock classes (the 10T 512-row class
at 4300 ps) throughout; per-device samples use 5 % relative sigma on `vth0`,
`u0` and `voff` of every MOS. This is functional screening, not yield
qualification or extracted-metal evidence; no record is promoted to
`sizing_table.json`.

**Outcome: on the release sources 340 of 340 cases pass, 753,452 of 753,452 checks: the 313 cases of the V2.1.9 campaign (its V2.1.8 matrix, the class-bound reads and the worst-PVT Monte-Carlo sweep) and 27 class-bound write -> write slots (15 nominal, 12 per-device seeds; 32x16 to 128x16 and 32x128, 6T, 6T with a mux and 10T, SS and SF 0.9 V / 125 C); 262 of the 340 are per-device samples. Five mux reads lost their DC operating point before any waveform and pass on a retry: two with Newton line search, two with gmin stepping on top, and one sample (mc_64x16_10t_mux_FS_cold_read_pd_s2) only with GMIN raised to 1e-10 as well (section 7.4). Between two writes the drivers' enable stayed at or above 98.1 % of VDD in every column (mc_512x4_6t_mux_SS_write_pd_s1; V2.1.9 turned the drivers off), and the new data left the write-data latch at least 106 ps after the local wordline was below 0.1 VDD (mc_8x4_10t_FF_cold_write_pd_s1). Every write -> write slot met the runtime restore check with at least 229 ps to spare after 1.1 x the bitline's settling time (w_32x32_6t_SS_write_pd_s2), where V2.1.9's slot failed the check itself at 32x16 and 32x32; on the 87 single write decks of both campaigns the slot is 115 to 517 ps shorter. While a wordline was above 0.1 VDD the drivers' enable stayed at or above 99.5 % of VDD (p_8x64_6t_mux_FF_cold_write_idle2) and `VWEN_ACCESS_ERROR` never exceeded 1.2 % of VDD (mc_16x16_6t_mux_FF_cold_sequence_pd_s1); the drivers released at least 108 ps after the wordline was off (p_8x4_6t_TT_sequence_settle0). At a read -> write or idle -> write edge the new data led the write enable by at least 97 ps (mc_8x4_6t_FF_cold_sequence_pd_s3) and the read's sense enable was off at least 61 ps before it (mc_8x4_10t_FF_cold_sequence_pd_s4). The 10T 512x4 read at 4300 ps leads its deadline by 784 ps, 110 ps above its rule. The smallest sense margin was 0.562 V (mc_512x4_6t_SS_read_pd_s2), the longest write slot 1786 ps (mc_512x4_6t_mux_SS_write_pd_s1) and the longest restore 953 ps (mc_512x4_6t_SS_read_pd_s1).**

## 1. What the review found

| | Finding | Where | Consequence |
|---|---|---|---|
| F1 | The scorer's `restore_budget` compares `TWSLOT` with 0.8 x `timing.low_read`, and both callers (`dev/sizing/campaign.py`, `dev/sizing/local_review.py`) build that timing with `timing_from_measurements`: `low_read` is the measured nominal SS read phase (`TCLK_WLEN + TREAD_TOTAL`), never a lookup class. The V2.1.9 record (section 8) and commit said "0.8 x class budget" | `dev/sizing/qualification.py` | A longer clock cannot satisfy it. On the V2.1.9 traces the slot exceeded 0.8 x the read phase at every write deck up to 64 rows (8x4 by 274 ps, 16x16 with a mux by 258 ps, 64x16 with a mux by 89 ps); 128x8 passed |
| F2 | `score_waveform` rebound its waveform array: the V2.1.6 drive-rail loop `for when, data in ...` shadowed `data`, which `signal()` reads | `dev/sizing/qualification.py` | Every write deck raised `TypeError`, which the runner does not catch: since V2.1.6 the scorer could not score a single write. The evidence campaigns use the validator and the local checker, so nothing noticed |
| F3 | V2.1.9 never ran a write -> write deck at the 32x16 and 32x32 bounds. There its slot (the previous wordline's tail, the observer, the drivers off, four settling stages, the slot-arm latch, the drivers on and their swing) put the driven bitline within 0.02 VDD of its rail only after the runtime restore check samples it, 0.6 T into the cycle (0.395 T after the 0.205 T clock rise) | V2.1.9 sources, `outputs/validation/V2.1.10-slot-budget/probes/tw1`, `tw4`, `tw6` | `VRESTORE_ERROR` 76 mV (32x16, 4.5 ns), 124 mV (32x32, 4.5 ns) and over 18 mV at 32x16 and 32x32 with a mux (4.75 ns): `access_validity` rejects every such write sample. The V2.1.9 8x4 and 16x16 write decks had passed with 3 to 8 mV |
| F4 | The 10T 512x4 read kept 662 ps against the 669 ps its rule requires at 10.5 ns | V2.1.9 record, section 6 | Reported but not fixed in V2.1.9 |
| F5 | The write -> write handshake existed only because the column write-data latches were transparent while `w_en` was low: new data could reach the drivers only if they turned off in between | `create_write_periphery`, `TIME_CONTROL._add_write_slot` | The data and `we` are registered on the rising edge (`DIN_dff`, `dff_buf1`), so between two writes nothing but the data changes; the drivers' off/on round trip (about 520 ps at SS) is what made the slot 1.6 to 1.8 ns |

The V2.1.9 write -> write slot at the class bounds, on the V2.1.9 sources and
clocks (single write decks, SS 0.9 V / 125 C nominal; the rail is the driven far
bitline within 0.02 VDD, the check samples 0.395 T after the clock rise, and the
runtime limit is 18 mV):

| Bound | Clock [ns] | TWSLOT [ps] | Rail [ps] | Check after rise [ps] | `VRESTORE_ERROR` [mV] |
|---|---:|---:|---:|---:|---:|
| 32x16 6T | 4.5 | 1767 | 1867 | 1778 | 76.1 (fails) |
| 32x32 6T | 4.5 | 1798 | 1898 | 1778 | 124.0 (fails) |
| 32x16 6T mux | 4.75 | 1776 | 1891 | 1876 | 22.0 (fails) |
| 32x32 6T mux | 4.75 | 1806 | 1920 | 1876 | 33.7 (fails) |
| 64x16 6T | 4.75 | 1744 | 1826 | 1876 | 7.1 |
| 32x64 6T | 5.0 | 1843 | 1943 | 1975 | 11.0 |
| 32x64 6T mux | 5.0 | 1851 | 1965 | 1975 | 15.8 |
| 32x16 10T | 5.0 | 1773 | 1873 | 1975 | 3.9 |
| 32x16 10T mux | 5.0 | 1778 | 1892 | 1975 | 5.7 |
| 128x16 6T | 5.5 | 1778 | 1854 | 2172 | 0.6 |
| 128x16 6T mux | 5.75 | 1786 | 1870 | 2271 | 0.5 |

(`outputs/validation/V2.1.10-slot-budget/probes/tw1`, `tw4`, `tw6`, analysed
with `slot_rail.py`.)

## 2. The decision

Raising the classes up to 64 rows and 64 columns by the slot (2100 to 2200 ps,
+10 to 17 %) would have met the runtime check; a campaign on those clocks was
started and stopped after 72 passing cases (diagnostic only; its queues were
removed, the probes are in `outputs/validation/V2.1.10-slot-budget/probes`). It would have paid for F5 with every access of every small
array and left the scorer's check (F1) failing anyway. Two structures remove the
handshake instead:

* **A (chosen):** keep `DIN_dff` and a latch per column, but hold the latch with
  the registered write rather than with `w_en`: it is transparent while no
  wordline is open, and while the drivers are on only for a selected write.
* **B (not now):** end the write wordline before the rising edge with a
  write-completion timer, so the DFFs drive the columns directly and no latch is
  needed. Shortest slot, but a new timing element whose write ability at SF 0.9 V
  / 125 C and under mismatch needs its own evidence.

## 3. The change

In `TIME_CONTROL` (`docs/design/TIME_CONTROL_PATH.md`, sections 2.10, 2.12 and 9):

* **Write-data hold.** `din_hold = !(wordline_idle & ((we & cs) | !w_en))`,
  three NAND2 (`DIN_HOLD_NAND`: `selected_write_bar = !(we & cs)`,
  `din_open = !(selected_write_bar & w_en)`, `din_hold = !(wordline_idle &
  din_open)`) and a tapered buffer (`DIN_HOLD_BUFFER`) sized for the testbench's
  latch-enable inverter (`din_hold_load`, appended optional argument, default two
  NAND2 inputs per column at a fan-out of 8). `din_hold` is a new port, last,
  for write operations with the replica guard.
* **Testbench.** `DIN_EN = !din_hold` drives the column latch enables through the
  `din_en` control line (V2.1.9: `WEN_BAR = !w_en` on `w_en_bar`); without the
  replica guard the inverter still reads `w_en`. `wen_load` keeps the former
  inverter as an upper bound, so the write-enable buffer is unchanged.
* **Removed.** The slot-arm NOR latch (`SLOT_ARM_NOR` x 2), its four settling
  stages (`SLOT_ARM_DELAY`), the `slot_armed` input of the selected slot
  (`selected_slot_bar = !(cs_pre & write_slot)`, `SELECTED_SLOT_NAND`) and the
  `.IC` seeding of the latch, which V2.1.9 had added because the latch broke the
  DC operating point of mux decks.
* **Write window.** `write_window = held(wordline_busy & we_hold) |
  selected_slot`: `BUSY_WRITE_NAND` (`busy_write_bar`), four unit stages
  (`BUSY_HOLD_DELAY`, `busy_write_held_bar`) and `WRITE_WINDOW_NAND` with
  `selected_slot_bar`. V2.1.9 had `wordline_busy | selected_slot`. The busy
  wordline and the next slot follow the same observer, but the slot's path is
  longer (the NAND3 of `write_slot` and the selected-slot gate): the first
  evidence pass found the window collapsing between two writes under mismatch
  at SS and SF (below). Holding the end of the write's busy term lets the slot
  take over first; a write -> idle release comes four stages later, a
  write -> read release with the request latch. A read's busy wordline is not
  in the window, so the next write's request never meets an open window
  (second pass, below).
* **Unchanged.** `w_en = we_hold & write_window`: the drivers stay fully on
  while a wordline is on (V2.1.9); the read path, the wordline enable, the
  precharge gate, the select delay, the driver classes and every load.

What each boundary does now:

| Boundary | Latch (column data) | Drivers (`w_en`) |
|---|---|---|
| write -> write | holds the written data until the previous wordline is observed off, then passes the new data | stay on: the write's held busy term hands over to the next slot |
| write -> read, write -> idle | holds the written data until the drivers are off (`w_en` low), then opens | released once the wordline is observed off (V2.1.9): with the request latch (read), four stages later (idle) |
| read -> write, idle -> write | already open (no wordline, drivers off): the new data passes at the registers' clock-to-Q | on at the slot, after the select delay and the previous enables |
| during an access | held (a wordline is on) | on for a write, off for a read |

The first evidence pass on these sources without the busy hold (`pass1/`, 340
cases, 320 passing) found the drivers' enable dipping between two writes in 16
Monte-Carlo samples at SS 0.9 V / 125 C and SF 0.9 V / 125 C (8x4, 16x16 and
64x16 with a mux, 6T and 10T), down to 0.21 V of 0.9 V at 8x4 10T SS: the busy
wordline fell at +769 ps after the clock rise, the slot rose at +830 ps, and the
window was low for about 50 ps. The new data still left the latches at least
325 ps after the wordline was off, so nothing was mis-written, but the drivers
did turn off; nominal decks had kept the enable above 99 % of VDD. Four mux
read decks lost their DC operating point in that pass (the fix pass retries
them).

A first busy hold kept the busy wordline itself four stages longer
(`write_window = wordline_busy | held(wordline_busy) | selected_slot`). The
second pass on it (`pass2/`, stopped after 31 cases, 28 passing, two 10T mux
reads without a DC operating point) failed `s_16x16_6t_SS_sequence`: at the
read -> write edge the held read wordline kept the window open while the
request latch passed the new write request. `w_en` pulsed to 0.26 to 0.30 V
at the block output while the read's sense enable was still falling (0.21 V),
and the column enables reached 0.19 to 0.21 V (the local checker's
`write_enable_quiet`, at most 0.1 VDD, failed in read cycles 1, 5 and 7).
Only a write's busy term is held now; with it no read leaves the window open. A
unit-delay model of the block's gates in `tests/test_precharge_exclusion.py`
covers write -> write, write -> read, write -> idle, read -> write and a
write's own wordline with a slot three stages late, and fails for the first
hold and for a hold of the request alone. A probe of the final window before
the third pass (`V2.1.10-slot-budget/probes/c1`, 38 cases: the pass-2 failure
and the other 16x16 sequences, the 8x4 / 16x16 / 64x16 SS and SF write seeds
of the first pass, FF sequences, an idle probe) passed 37; the 38th, a 16x16
mux read, lost its four-rank DC operating point and passed on one rank and
with line search. The drivers' enable stayed at or above 0.994 VDD between
writes, and at or below 0.003 VDD at the columns during reads;
`s_16x16_6t_SS_sequence` has no pulse above 20 mV at the block output.

A plainer hold (`din_hold = wordline_busy`) was probed first: between two writes
it behaves the same, but at a write -> read edge it let the next cycle's data
reach the drivers while they were releasing (the wordline was off, so nothing
was written, but the local checker's hold checks caught the bitlines moving to
data no cycle writes). Gating it with `we & cs` alone fixed that and cost the
read -> write edge half its setup lead (61 ps instead of 107 ps before the write
enable at 8x4 FF -40 C: the latch then opened only when the registered request
arrived); the `!w_en` term restores it (`probes/a2`, `a3`, `a4`).

## 4. Checks

* **Local checker** (`dev/v210_waveform_checks.py`): at a write -> write
  boundary, per column, `release_before_new_data` (the data at the latch
  output leaves its value by 0.05 VDD only after the local wordline is below
  0.1 VDD, metric `release_to_new_data_ps`), `WL_during_new_data` and
  `write_enable_held_between_writes` (the driver's enable at least 0.9 VDD from
  the capture edge to the next access, metric
  `min_write_enable_between_writes_v`); such a write's drive ends when its data
  leaves the latch, not at the enable's release. Every other boundary keeps its
  V2.1.8 / V2.1.9 checks.
* **Validator** (`dev/validate_distributed_rc.py`): a write followed by a write
  keeps its drive window to the next entry.
* **Qualification scorer** (`dev/sizing/qualification.py`): the drive-rail loop
  no longer shadows the waveform (F2); for write decks the entry event is the
  far bitline leaving its rail with the next data (`wl_off_before_new_data`,
  `write_enable_held_between_writes`); `restore_budget` scores the drivers'
  swing (the far bitline 0.9 -> 0.1 VDD) against 0.8 x the read phase, and the
  whole slot must fit the clock-high phase with the period margin
  (`write_slot_budget`, `TWSLOT <= T/2 / (1 + margin)`), since the observer and
  the wordline tail are no driver's work (F1). `scoring_sources.json` refreshed.

## 5. Before -> after

`compare_v219.py` measures both campaigns with the same code
(`assemble_record.summarize`) on the 188 write-bearing cases they share, under
the same names and settings (V2.1.9's latest attempts in
`outputs/validation/V2.1.9-write-hold`). Nominal cases are paired; per-device
seeds are not. The one-rank runs use Xyce's native sampling, which draws the
cards in deck order, and this release removes and adds control-block devices,
so the same seed is a new sample: the decoder of mc_512x4_6t_SS_read_pd_s2,
whose devices did not change, is 62 ps faster.

* **Write -> write slot** (`TWSLOT`, 87 single write decks): 115 to 517 ps
  shorter (mc_8x4_10t_FF_cold_write_pd_s3, w_64x16_6t_mux_SS_write_pd_s3); the
  slack to the runtime restore check after 1.1 x the rail time grows by 128 to
  571 ps. In V2.1.9, 19 of these decks were below that 1.1 x rule (down to
  -135 ps, mc_16x16_6t_mux_SS_write_pd_s1) while still passing the check
  itself, and the slot failed the check at the 32x16 and 32x32 bounds, which
  its campaign never ran.
* **Drivers between two writes:** V2.1.9 turned them off and on; now the
  lowest enable at a driver is 0.883 V of 0.9 V.
* **Release after the wordline is off:** 16 to 98 ps later at write -> read
  (86 sequences; the release now follows the request latch on
  `wordline_idle`) and 49 to 185 ps later at write -> idle (15 idle probes; the
  four hold stages). The next precharge waits for it (V2.1.8) and still ends
  inside the clock-high phase (every restore check passes).
* **Data before the write enable** at read -> write and idle -> write: between
  164 ps less and 39 ps more. The slot path is two gates shorter (a NAND2
  selected slot and the NAND2 window instead of an AND3, a NOR2 and an
  inverter), so `w_en` rises earlier; the smallest lead is 97 ps at FF -40 C.
* **`VWEN_ACCESS_ERROR`:** unchanged, at most 1.2 % of VDD.
* Every shared case passes in both campaigns.

Nominal cases (the per-device rows are in `compare-v219.md` and `compare-v219.json`
of the campaign directory; restore-check slack: after 1.1 x the rail time, write
decks only):

| Case | T [ns] | TWSLOT [ps] | Restore-check slack [ps] | W->W w_en min [V] | w_en release after WL off [ps] | VWEN_ACCESS [V] | Data before w_en [ps] | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| p_16x16_10t_SS_write_idle2 | 5.0 -> 5.0 | 965.0 -> 932.5 | - -> - | - -> - | 467.1 -> 650.2 | 0.0011 -> 0.0011 | 456.7 -> 424.0 | pass -> pass |
| p_16x16_6t_mux_SS_write_idle2 | 4.75 -> 4.75 | 973.3 -> 939.8 | - -> - | - -> - | 466.6 -> 651.2 | 0.0016 -> 0.0016 | 456.8 -> 423.2 | pass -> pass |
| p_4x4_6t_FF_cold_write_idle2 | 4.5 -> 4.5 | 244.7 -> 236.3 | - -> - | - -> - | 108.4 -> 161.9 | 0.0033 -> 0.0033 | 112.6 -> 104.2 | pass -> pass |
| p_64x16_6t_mux_SS_write_idle2 | 5.0 -> 5.0 | 960.6 -> 927.5 | - -> - | - -> - | 470.8 -> 656.1 | 0.0014 -> 0.0013 | 443.8 -> 410.5 | pass -> pass |
| p_8x4_10t_FF_cold_write_idle2 | 5.0 -> 5.0 | 249.1 -> 240.7 | - -> - | - -> - | 108.5 -> 161.9 | 0.003 -> 0.003 | 112.8 -> 104.2 | pass -> pass |
| p_8x4_6t_FF_cold_write_idle2 | 4.5 -> 4.5 | 248.8 -> 240.2 | - -> - | - -> - | 108.6 -> 162.0 | 0.003 -> 0.003 | 112.8 -> 104.2 | pass -> pass |
| p_8x4_6t_FF_cold_write_idle3 | 4.5 -> 4.5 | 248.8 -> 240.2 | - -> - | - -> - | 108.6 -> 162.0 | 0.003 -> 0.003 | 112.8 -> 104.2 | pass -> pass |
| p_8x4_6t_TT_sequence_settle0 | 4.5 -> 4.5 | - -> - | - -> - | - -> - | 76.5 -> 107.7 | 0.007 -> 0.0071 | 212.1 -> 163.6 | pass -> pass |
| p_8x4_6t_TT_write_idle2 | 4.5 -> 4.5 | 397.1 -> 383.1 | - -> - | - -> - | 179.9 -> 264.0 | 0.0017 -> 0.0017 | 184.0 -> 169.4 | pass -> pass |
| p_8x64_6t_mux_FF_cold_write_idle2 | 5.0 -> 5.0 | 279.5 -> 272.6 | - -> - | - -> - | 126.9 -> 180.2 | 0.006 -> 0.0058 | 128.3 -> 120.1 | pass -> pass |
| s_16x16_10t_SS_sequence | 5.0 -> 5.0 | - -> - | - -> - | - -> - | 467.0 -> 539.3 | 0.0077 -> 0.0078 | 524.0 -> 455.3 | pass -> pass |
| s_16x16_10t_mux_SS_sequence | 5.0 -> 5.0 | - -> - | - -> - | - -> - | 466.7 -> 539.4 | 0.0092 -> 0.0092 | 523.1 -> 420.9 | pass -> pass |
| s_16x16_6t_SS_sequence | 4.5 -> 4.5 | - -> - | - -> - | - -> - | 466.3 -> 538.8 | 0.0077 -> 0.0078 | 523.5 -> 438.6 | pass -> pass |
| s_16x16_6t_mux_SS_sequence | 4.75 -> 4.75 | - -> - | - -> - | - -> - | 466.7 -> 539.1 | 0.0092 -> 0.0092 | 523.7 -> 421.1 | pass -> pass |
| s_16x32_6t_mux_SS_sequence | 4.75 -> 4.75 | - -> - | - -> - | - -> - | 475.9 -> 548.4 | 0.0102 -> 0.0101 | 532.7 -> 458.8 | pass -> pass |
| s_32x16_10t_mux_SS_sequence | 5.0 -> 5.0 | - -> - | - -> - | - -> - | 467.4 -> 539.5 | 0.0075 -> 0.0075 | 523.8 -> 432.1 | pass -> pass |
| s_32x16_6t_SS_sequence | 4.5 -> 4.5 | - -> - | - -> - | - -> - | 467.2 -> 539.5 | 0.007 -> 0.007 | 525.0 -> 392.7 | pass -> pass |
| s_32x16_6t_mux_SS_sequence | 4.75 -> 4.75 | - -> - | - -> - | - -> - | 466.9 -> 539.6 | 0.0075 -> 0.0075 | 524.6 -> 409.6 | pass -> pass |
| s_8x4_10t_FF_cold_sequence | 5.0 -> 5.0 | - -> - | - -> - | - -> - | 108.3 -> 127.3 | 0.007 -> 0.0071 | 129.8 -> 99.4 | pass -> pass |
| s_8x4_10t_mux_SS_sequence | 5.0 -> 5.0 | - -> - | - -> - | - -> - | 411.9 -> 483.7 | 0.0089 -> 0.0089 | 481.2 -> 365.2 | pass -> pass |
| s_8x4_6t_FF_cold_sequence | 4.5 -> 4.5 | - -> - | - -> - | - -> - | 108.5 -> 127.3 | 0.007 -> 0.0071 | 129.8 -> 99.4 | pass -> pass |
| s_8x4_6t_SS_sequence | 4.5 -> 4.5 | - -> - | - -> - | - -> - | 411.5 -> 484.3 | 0.0071 -> 0.0071 | 481.4 -> 369.8 | pass -> pass |
| s_8x8_6t_mux_FF_cold_sequence | 4.75 -> 4.75 | - -> - | - -> - | - -> - | 112.8 -> 131.7 | 0.0121 -> 0.0122 | 132.2 -> 102.0 | pass -> pass |
| w_128x8_6t_SS_write | 5.5 -> 5.5 | 1766.2 -> 1306.5 | 145.9 -> 651.8 | - -> 0.894 | 464.9 -> - | 0.0034 -> 0.0 | 511.8 -> 477.0 | pass -> pass |
| w_16x16_10t_mux_SS_write | 5.0 -> 5.0 | 1728.1 -> 1243.1 | -32.0 -> 502.5 | - -> 0.893 | 466.9 -> - | 0.0045 -> 0.0 | 523.8 -> 490.3 | pass -> pass |
| w_16x16_6t_mux_SS_write | 4.75 -> 4.75 | 1726.8 -> 1242.0 | -128.7 -> 405.9 | - -> 0.893 | 466.5 -> - | 0.0047 -> 0.0 | 523.3 -> 490.4 | pass -> pass |
| w_16x64_6t_mux_SS_write | 5.0 -> 5.0 | 1796.5 -> 1311.0 | -106.6 -> 428.4 | - -> 0.891 | 483.2 -> - | 0.0065 -> 0.0 | 541.6 -> 507.8 | pass -> pass |
| w_256x4_6t_SF_write | 6.75 -> 6.75 | 1673.9 -> 1263.0 | 731.7 -> 1185.2 | - -> 0.893 | 392.8 -> - | 0.002 -> 0.0 | 446.3 -> 416.4 | pass -> pass |
| w_256x4_6t_SS_write | 6.75 -> 6.75 | 1871.2 -> 1407.9 | 515.1 -> 1025.9 | - -> 0.892 | 458.5 -> - | 0.0027 -> 0.0 | 500.0 -> 465.8 | pass -> pass |
| w_512x4_6t_mux_SS_write | 10.0 -> 10.0 | 2229.8 -> 1755.6 | 1361.9 -> 1884.7 | - -> 0.889 | 466.2 -> - | 0.002 -> 0.0 | 485.9 -> 451.8 | pass -> pass |
| w_64x16_10t_mux_SF_write | 5.5 -> 5.5 | 1569.1 -> 1171.6 | 343.9 -> 782.8 | - -> 0.895 | 408.7 -> - | 0.0039 -> 0.0 | 461.0 -> 429.9 | pass -> pass |
| w_64x16_10t_mux_SS_write | 5.5 -> 5.5 | 1753.8 -> 1304.4 | 140.9 -> 636.3 | - -> 0.895 | 473.2 -> - | 0.0043 -> 0.0 | 517.7 -> 484.3 | pass -> pass |
| w_64x16_6t_mux_SF_write | 5.0 -> 5.0 | 1566.1 -> 1170.9 | 150.2 -> 585.6 | - -> 0.895 | 409.8 -> - | 0.0039 -> 0.0 | 459.7 -> 429.5 | pass -> pass |
| w_64x16_6t_mux_SS_write | 5.0 -> 5.0 | 1744.8 -> 1300.3 | -46.6 -> 444.0 | - -> 0.895 | 470.4 -> - | 0.0045 -> 0.0 | 517.5 -> 483.9 | pass -> pass |
| w_8x4_10t_FF_cold_write | 5.0 -> 5.0 | 455.2 -> 332.6 | 1454.1 -> 1588.0 | - -> 1.097 | 108.5 -> - | 0.0031 -> 0.0 | 129.8 -> 121.1 | pass -> pass |
| w_8x4_10t_SS_write | 5.0 -> 5.0 | 1626.5 -> 1195.1 | 111.0 -> 586.0 | - -> 0.895 | 411.3 -> - | 0.0023 -> 0.0 | 481.3 -> 447.8 | pass -> pass |
| w_8x4_6t_FF_cold_write | 4.5 -> 4.5 | 453.7 -> 332.1 | 1258.3 -> 1392.0 | - -> 1.097 | 108.5 -> - | 0.0031 -> 0.0 | 129.6 -> 121.1 | pass -> pass |
| w_8x4_6t_SS_write | 4.5 -> 4.5 | 1625.2 -> 1193.4 | -85.0 -> 390.8 | - -> 0.895 | 411.5 -> - | 0.0023 -> 0.0 | 481.1 -> 447.5 | pass -> pass |

## 6. The 10T 512-row class

`timing_lookup.json` `v2.1.10-timing-9` moves the 10T 512-row class from
4200 to 4300 ps (10.5 -> 10.75 ns); every other class is the V2.1.9 one.

At 4300 ps the nominal 10T 512x4 mux read (r_512x4_10t_mux_SS_read) has its
local read output below 0.1 VDD 4591 ps after the clock fall and 784 ps before
the 1.2 T deadline; the rule (0.02 T plus 10 % of the access) asks for 674 ps,
so it holds with 110 ps to spare (V2.1.9 at 4200 ps: 662 against 669 ps). Its
two seeds lead by 990 and 1066 ps. The other 512-row classes are unchanged
(6T 612 ps against 586 ps, 6T mux 686 ps against 631 ps); their seeds lead by
398 to 824 ps, above the checker's 0.02 T guard. As in V2.1.9 (down to
-235 ps), the local wordline of some 512-row seeds is still above 0.1 VDD at
the deadline edge (down to -60 ps, mc_512x4_6t_mux_SS_read_pd_s3); the next
precharge waits for the replica wordline, and every hold and restore check
passes.

## 7. Evidence

Every case, latest attempt (`outputs/validation/V2.1.10-write-latch/`,
`assemble_record.py`; the JSON record is `WRITE_LATCH_V2_1_10.json` next to
this file). Seeds `sN` are `20261900 + N` for the V2.1.9 sweep (`mc_`),
`20261000 + N` for the class-bound slots and `20261700 + N` for the V2.1.8
matrix. Queues ran with four ranks (nominal) or one (per-device) on 96 cores;
every queue archives the sources it ran.

### 7.1 Write -> write slots against the runtime restore check

Every single write deck: `TWSLOT` from the `.mt0`; rail 2 %: from the clock rise
to the driven bitline within 0.02 VDD of its rail (`slot_rail.py`); check - rise:
the runtime `VRESTORE` check (0.6 T) after the clock rise; slack: check - rise
minus 1.1 x rail; w_en min: the lowest driver enable between the two writes.

| Case | Cell | Size | Mux | Corner | Variation | T [ns] | TWSLOT [ps] | Rail 2 % [ps] | Check - rise [ps] | Slack after 1.1 x rail [ps] | w_en min between writes [V] | Result |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| w_32x32_6t_SS_write_pd_s2 | 6T_ | 32x32 | no | SS 0.9 V / 125 C | per-device s2 | 4.5 | 1310.1 | 1407.9 | 1777.5 | 228.8 | 0.893 | pass |
| w_32x16_6t_SS_write_pd_s3 | 6T_ | 32x16 | no | SS 0.9 V / 125 C | per-device s3 | 4.5 | 1297.4 | 1400.9 | 1777.5 | 236.5 | 0.895 | pass |
| w_32x32_6t_SS_write_pd_s1 | 6T_ | 32x32 | no | SS 0.9 V / 125 C | per-device s1 | 4.5 | 1300.6 | 1398.2 | 1777.5 | 239.5 | 0.893 | pass |
| w_32x32_6t_SS_write | 6T_ | 32x32 | no | SS 0.9 V / 125 C | nominal | 4.5 | 1292.9 | 1392.3 | 1777.5 | 245.9 | 0.89 | pass |
| w_32x16_6t_SS_write_pd_s2 | 6T_ | 32x16 | no | SS 0.9 V / 125 C | per-device s2 | 4.5 | 1293.0 | 1388.9 | 1777.5 | 249.7 | 0.895 | pass |
| w_32x16_6t_SS_write_pd_s1 | 6T_ | 32x16 | no | SS 0.9 V / 125 C | per-device s1 | 4.5 | 1284.6 | 1384.4 | 1777.5 | 254.7 | 0.894 | pass |
| w_32x16_6t_SS_write | 6T_ | 32x16 | no | SS 0.9 V / 125 C | nominal | 4.5 | 1271.7 | 1371.0 | 1777.5 | 269.4 | 0.891 | pass |
| w_32x32_6t_mux_SS_write | 6T_ | 32x32 | yes | SS 0.9 V / 125 C | nominal | 4.75 | 1313.1 | 1426.2 | 1876.2 | 307.4 | 0.891 | pass |
| w_32x16_6t_mux_SS_write_pd_s1 | 6T_ | 32x16 | yes | SS 0.9 V / 125 C | per-device s1 | 4.75 | 1298.3 | 1410.6 | 1876.2 | 324.6 | 0.893 | pass |
| w_32x16_6t_mux_SS_write | 6T_ | 32x16 | yes | SS 0.9 V / 125 C | nominal | 4.75 | 1291.9 | 1405.9 | 1876.2 | 329.8 | 0.89 | pass |
| w_32x16_6t_mux_SS_write_pd_s3 | 6T_ | 32x16 | yes | SS 0.9 V / 125 C | per-device s3 | 4.75 | 1280.4 | 1395.7 | 1876.2 | 341.0 | 0.894 | pass |
| mc_8x4_6t_SS_write_pd_s2 | 6T_ | 8x4 | no | SS 0.9 V / 125 C | per-device s2 | 4.5 | 1230.7 | 1302.8 | 1777.5 | 344.4 | 0.896 | pass |
| w_32x64_6t_mux_SS_write | 6T_ | 32x64 | yes | SS 0.9 V / 125 C | nominal | 5.0 | 1367.2 | 1480.5 | 1975.0 | 346.4 | 0.887 | pass |
| w_32x16_6t_mux_SS_write_pd_s2 | 6T_ | 32x16 | yes | SS 0.9 V / 125 C | per-device s2 | 4.75 | 1272.9 | 1387.7 | 1876.2 | 349.8 | 0.895 | pass |
| mc_8x4_6t_SS_write_pd_s4 | 6T_ | 8x4 | no | SS 0.9 V / 125 C | per-device s4 | 4.5 | 1230.5 | 1296.0 | 1777.5 | 351.9 | 0.895 | pass |
| w_32x64_6t_mux_SS_write_pd_s1 | 6T_ | 32x64 | yes | SS 0.9 V / 125 C | per-device s1 | 5.0 | 1362.2 | 1474.5 | 1975.0 | 353.0 | 0.893 | pass |
| mc_8x4_6t_SS_write_pd_s5 | 6T_ | 8x4 | no | SS 0.9 V / 125 C | per-device s5 | 4.5 | 1219.2 | 1283.9 | 1777.5 | 365.2 | 0.897 | pass |
| w_64x16_6t_SS_write | 6T_ | 64x16 | no | SS 0.9 V / 125 C | nominal | 4.75 | 1289.4 | 1370.7 | 1876.2 | 368.5 | 0.895 | pass |
| mc_16x16_6t_mux_SS_write_pd_s2 | 6T_ | 16x16 | yes | SS 0.9 V / 125 C | per-device s2 | 4.75 | 1263.3 | 1365.2 | 1876.2 | 374.5 | 0.896 | pass |
| w_32x64_6t_mux_SS_write_pd_s2 | 6T_ | 32x64 | yes | SS 0.9 V / 125 C | per-device s2 | 5.0 | 1339.4 | 1450.3 | 1975.0 | 379.6 | 0.893 | pass |
| w_32x64_6t_SS_write | 6T_ | 32x64 | no | SS 0.9 V / 125 C | nominal | 5.0 | 1347.6 | 1447.7 | 1975.0 | 382.5 | 0.887 | pass |
| mc_8x4_6t_SS_write_pd_s3 | 6T_ | 8x4 | no | SS 0.9 V / 125 C | per-device s3 | 4.5 | 1194.5 | 1263.0 | 1777.5 | 388.2 | 0.896 | pass |
| w_8x4_6t_SS_write | 6T_ | 8x4 | no | SS 0.9 V / 125 C | nominal | 4.5 | 1193.4 | 1260.6 | 1777.5 | 390.8 | 0.895 | pass |
| w_32x16_6t_SF_write | 6T_ | 32x16 | no | SF 0.9 V / 125 C | nominal | 4.5 | 1149.2 | 1249.2 | 1777.5 | 403.3 | 0.892 | pass |
| w_16x16_6t_mux_SS_write | 6T_ | 16x16 | yes | SS 0.9 V / 125 C | nominal | 4.75 | 1242.0 | 1336.7 | 1876.2 | 405.9 | 0.893 | pass |
| mc_8x4_6t_SS_write_pd_s1 | 6T_ | 8x4 | no | SS 0.9 V / 125 C | per-device s1 | 4.5 | 1179.4 | 1246.7 | 1777.5 | 406.1 | 0.897 | pass |
| w_64x16_6t_mux_SS_write_pd_s1 | 6T_ | 64x16 | yes | SS 0.9 V / 125 C | per-device s1 | 5.0 | 1326.3 | 1421.2 | 1975.0 | 411.7 | 0.896 | pass |
| w_32x16_10t_mux_SS_write_pd_s1 | 10T | 32x16 | yes | SS 0.9 V / 125 C | per-device s1 | 5.0 | 1302.8 | 1416.5 | 1975.0 | 416.9 | 0.894 | pass |
| mc_16x16_6t_mux_SS_write_pd_s1 | 6T_ | 16x16 | yes | SS 0.9 V / 125 C | per-device s1 | 4.75 | 1229.7 | 1326.5 | 1876.2 | 417.1 | 0.897 | pass |
| w_32x16_10t_mux_SS_write | 10T | 32x16 | yes | SS 0.9 V / 125 C | nominal | 5.0 | 1295.1 | 1409.0 | 1975.0 | 425.1 | 0.891 | pass |
| w_16x64_6t_mux_SS_write | 6T_ | 16x64 | yes | SS 0.9 V / 125 C | nominal | 5.0 | 1311.0 | 1406.0 | 1975.0 | 428.4 | 0.891 | pass |
| w_64x16_6t_mux_SS_write | 6T_ | 64x16 | yes | SS 0.9 V / 125 C | nominal | 5.0 | 1300.3 | 1391.8 | 1975.0 | 444.0 | 0.895 | pass |
| w_64x16_6t_mux_SS_write_pd_s2 | 6T_ | 64x16 | yes | SS 0.9 V / 125 C | per-device s2 | 5.0 | 1290.8 | 1384.2 | 1975.0 | 452.4 | 0.895 | pass |
| mc_64x16_6t_mux_SS_write_pd_s2 | 6T_ | 64x16 | yes | SS 0.9 V / 125 C | per-device s2 | 5.0 | 1283.6 | 1376.1 | 1975.0 | 461.3 | 0.895 | pass |
| w_32x16_10t_mux_SS_write_pd_s2 | 10T | 32x16 | yes | SS 0.9 V / 125 C | per-device s2 | 5.0 | 1263.8 | 1375.6 | 1975.0 | 461.9 | 0.894 | pass |
| w_32x16_10t_SS_write | 10T | 32x16 | no | SS 0.9 V / 125 C | nominal | 5.0 | 1275.0 | 1374.4 | 1975.0 | 463.2 | 0.891 | pass |
| w_32x16_6t_mux_SF_write | 6T_ | 32x16 | yes | SF 0.9 V / 125 C | nominal | 4.75 | 1168.3 | 1282.4 | 1876.2 | 465.6 | 0.892 | pass |
| w_64x16_6t_mux_SS_write_pd_s3 | 6T_ | 64x16 | yes | SS 0.9 V / 125 C | per-device s3 | 5.0 | 1283.2 | 1371.7 | 1975.0 | 466.1 | 0.895 | pass |
| mc_64x16_6t_mux_SS_write_pd_s1 | 6T_ | 64x16 | yes | SS 0.9 V / 125 C | per-device s1 | 5.0 | 1271.7 | 1363.5 | 1975.0 | 475.1 | 0.895 | pass |
| mc_8x4_6t_SF_write_pd_s2 | 6T_ | 8x4 | no | SF 0.9 V / 125 C | per-device s2 | 4.5 | 1105.2 | 1177.6 | 1777.5 | 482.1 | 0.897 | pass |
| mc_8x4_6t_SF_write_pd_s4 | 6T_ | 8x4 | no | SF 0.9 V / 125 C | per-device s4 | 4.5 | 1103.2 | 1169.6 | 1777.5 | 490.9 | 0.895 | pass |
| w_16x16_10t_mux_SS_write | 10T | 16x16 | yes | SS 0.9 V / 125 C | nominal | 5.0 | 1243.1 | 1338.7 | 1975.0 | 502.5 | 0.893 | pass |
| mc_8x4_6t_SF_write_pd_s5 | 6T_ | 8x4 | no | SF 0.9 V / 125 C | per-device s5 | 4.5 | 1093.6 | 1158.1 | 1777.5 | 503.6 | 0.897 | pass |
| mc_16x16_10t_mux_SS_write_pd_s1 | 10T | 16x16 | yes | SS 0.9 V / 125 C | per-device s1 | 5.0 | 1237.3 | 1336.6 | 1975.0 | 504.8 | 0.896 | pass |
| mc_16x16_10t_mux_SS_write_pd_s2 | 10T | 16x16 | yes | SS 0.9 V / 125 C | per-device s2 | 5.0 | 1234.7 | 1331.5 | 1975.0 | 510.3 | 0.896 | pass |
| mc_16x16_6t_mux_SF_write_pd_s2 | 6T_ | 16x16 | yes | SF 0.9 V / 125 C | per-device s2 | 4.75 | 1139.8 | 1241.7 | 1876.2 | 510.4 | 0.897 | pass |
| mc_8x4_6t_SF_write_pd_s3 | 6T_ | 8x4 | no | SF 0.9 V / 125 C | per-device s3 | 4.5 | 1072.4 | 1143.1 | 1777.5 | 520.1 | 0.897 | pass |
| mc_8x4_6t_SF_write_pd_s1 | 6T_ | 8x4 | no | SF 0.9 V / 125 C | per-device s1 | 4.5 | 1061.9 | 1130.2 | 1777.5 | 534.3 | 0.897 | pass |
| mc_16x16_6t_mux_SF_write_pd_s1 | 6T_ | 16x16 | yes | SF 0.9 V / 125 C | per-device s1 | 4.75 | 1111.6 | 1208.6 | 1876.2 | 546.8 | 0.897 | pass |
| mc_8x4_10t_SS_write_pd_s3 | 10T | 8x4 | no | SS 0.9 V / 125 C | per-device s3 | 5.0 | 1216.2 | 1285.7 | 1975.0 | 560.7 | 0.896 | pass |
| mc_64x16_10t_mux_SS_write_pd_s1 | 10T | 64x16 | yes | SS 0.9 V / 125 C | per-device s1 | 5.5 | 1350.1 | 1445.2 | 2172.5 | 582.8 | 0.895 | pass |
| mc_8x4_10t_SS_write_pd_s5 | 10T | 8x4 | no | SS 0.9 V / 125 C | per-device s5 | 5.0 | 1197.2 | 1263.8 | 1975.0 | 584.8 | 0.896 | pass |
| w_64x16_6t_mux_SF_write | 6T_ | 64x16 | yes | SF 0.9 V / 125 C | nominal | 5.0 | 1170.9 | 1263.1 | 1975.0 | 585.6 | 0.895 | pass |
| w_8x4_10t_SS_write | 10T | 8x4 | no | SS 0.9 V / 125 C | nominal | 5.0 | 1195.1 | 1262.7 | 1975.0 | 586.0 | 0.895 | pass |
| mc_8x4_10t_SS_write_pd_s4 | 10T | 8x4 | no | SS 0.9 V / 125 C | per-device s4 | 5.0 | 1186.1 | 1254.6 | 1975.0 | 594.9 | 0.895 | pass |
| mc_64x16_6t_mux_SF_write_pd_s2 | 6T_ | 64x16 | yes | SF 0.9 V / 125 C | per-device s2 | 5.0 | 1158.9 | 1251.3 | 1975.0 | 598.6 | 0.896 | pass |
| mc_8x4_10t_SS_write_pd_s2 | 10T | 8x4 | no | SS 0.9 V / 125 C | per-device s2 | 5.0 | 1180.1 | 1249.0 | 1975.0 | 601.1 | 0.897 | pass |
| mc_64x16_6t_mux_SF_write_pd_s1 | 6T_ | 64x16 | yes | SF 0.9 V / 125 C | per-device s1 | 5.0 | 1147.3 | 1239.8 | 1975.0 | 611.2 | 0.895 | pass |
| mc_8x4_10t_SS_write_pd_s1 | 10T | 8x4 | no | SS 0.9 V / 125 C | per-device s1 | 5.0 | 1157.6 | 1224.9 | 1975.0 | 627.6 | 0.895 | pass |
| mc_64x16_10t_mux_SS_write_pd_s2 | 10T | 64x16 | yes | SS 0.9 V / 125 C | per-device s2 | 5.5 | 1307.1 | 1401.3 | 2172.5 | 631.1 | 0.895 | pass |
| w_64x16_10t_mux_SS_write | 10T | 64x16 | yes | SS 0.9 V / 125 C | nominal | 5.5 | 1304.4 | 1396.5 | 2172.5 | 636.3 | 0.895 | pass |
| mc_16x16_10t_mux_SF_write_pd_s1 | 10T | 16x16 | yes | SF 0.9 V / 125 C | per-device s1 | 5.0 | 1113.8 | 1212.8 | 1975.0 | 640.9 | 0.897 | pass |
| mc_16x16_10t_mux_SF_write_pd_s2 | 10T | 16x16 | yes | SF 0.9 V / 125 C | per-device s2 | 5.0 | 1112.2 | 1209.9 | 1975.0 | 644.1 | 0.897 | pass |
| w_128x16_6t_SS_write | 6T_ | 128x16 | no | SS 0.9 V / 125 C | nominal | 5.5 | 1310.6 | 1386.9 | 2172.5 | 646.9 | 0.893 | pass |
| w_128x8_6t_SS_write | 6T_ | 128x8 | no | SS 0.9 V / 125 C | nominal | 5.5 | 1306.5 | 1382.5 | 2172.5 | 651.8 | 0.894 | pass |
| w_32x128_6t_mux_SS_write | 6T_ | 32x128 | yes | SS 0.9 V / 125 C | nominal | 6.0 | 1412.2 | 1526.3 | 2370.0 | 691.1 | 0.885 | pass |
| mc_8x4_10t_SF_write_pd_s3 | 10T | 8x4 | no | SF 0.9 V / 125 C | per-device s3 | 5.0 | 1091.7 | 1161.7 | 1975.0 | 697.1 | 0.896 | pass |
| mc_8x4_10t_SF_write_pd_s5 | 10T | 8x4 | no | SF 0.9 V / 125 C | per-device s5 | 5.0 | 1076.5 | 1143.9 | 1975.0 | 716.7 | 0.897 | pass |
| mc_8x4_10t_SF_write_pd_s4 | 10T | 8x4 | no | SF 0.9 V / 125 C | per-device s4 | 5.0 | 1067.1 | 1136.4 | 1975.0 | 725.0 | 0.896 | pass |
| mc_8x4_10t_SF_write_pd_s2 | 10T | 8x4 | no | SF 0.9 V / 125 C | per-device s2 | 5.0 | 1065.3 | 1133.9 | 1975.0 | 727.7 | 0.897 | pass |
| w_128x16_6t_mux_SS_write | 6T_ | 128x16 | yes | SS 0.9 V / 125 C | nominal | 5.75 | 1317.0 | 1401.2 | 2271.3 | 729.9 | 0.893 | pass |
| mc_64x16_10t_mux_SF_write_pd_s1 | 10T | 64x16 | yes | SF 0.9 V / 125 C | per-device s1 | 5.5 | 1211.5 | 1307.5 | 2172.5 | 734.3 | 0.895 | pass |
| mc_8x4_10t_SF_write_pd_s1 | 10T | 8x4 | no | SF 0.9 V / 125 C | per-device s1 | 5.0 | 1042.2 | 1110.2 | 1975.0 | 753.8 | 0.896 | pass |
| mc_64x16_10t_mux_SF_write_pd_s2 | 10T | 64x16 | yes | SF 0.9 V / 125 C | per-device s2 | 5.5 | 1176.0 | 1270.1 | 2172.5 | 775.3 | 0.895 | pass |
| w_64x16_10t_mux_SF_write | 10T | 64x16 | yes | SF 0.9 V / 125 C | nominal | 5.5 | 1171.6 | 1263.3 | 2172.5 | 782.8 | 0.895 | pass |
| w_128x16_10t_mux_SS_write | 10T | 128x16 | yes | SS 0.9 V / 125 C | nominal | 6.0 | 1320.4 | 1404.6 | 2370.0 | 824.9 | 0.893 | pass |
| w_256x4_6t_SS_write | 6T_ | 256x4 | no | SS 0.9 V / 125 C | nominal | 6.75 | 1407.9 | 1491.2 | 2666.2 | 1025.9 | 0.892 | pass |
| mc_256x4_6t_SS_write_pd_s1 | 6T_ | 256x4 | no | SS 0.9 V / 125 C | per-device s1 | 6.75 | 1407.4 | 1488.1 | 2666.2 | 1029.3 | 0.891 | pass |
| w_256x4_6t_SF_write | 6T_ | 256x4 | no | SF 0.9 V / 125 C | nominal | 6.75 | 1263.0 | 1346.4 | 2666.2 | 1185.2 | 0.893 | pass |
| mc_256x4_6t_SF_write_pd_s1 | 6T_ | 256x4 | no | SF 0.9 V / 125 C | per-device s1 | 6.75 | 1262.3 | 1343.4 | 2666.2 | 1188.6 | 0.892 | pass |
| mc_8x4_6t_FS_cold_write_pd_s4 | 6T_ | 8x4 | no | FS 1.1 V / -40 C | per-device s4 | 4.5 | 356.3 | 374.6 | 1777.5 | 1365.5 | 1.097 | pass |
| mc_8x4_6t_FS_cold_write_pd_s5 | 6T_ | 8x4 | no | FS 1.1 V / -40 C | per-device s5 | 4.5 | 354.5 | 373.5 | 1777.5 | 1366.6 | 1.097 | pass |
| mc_8x4_6t_FS_cold_write_pd_s2 | 6T_ | 8x4 | no | FS 1.1 V / -40 C | per-device s2 | 4.5 | 353.0 | 371.9 | 1777.5 | 1368.4 | 1.097 | pass |
| mc_8x4_6t_FS_cold_write_pd_s3 | 6T_ | 8x4 | no | FS 1.1 V / -40 C | per-device s3 | 4.5 | 349.7 | 368.6 | 1777.5 | 1372.0 | 1.097 | pass |
| mc_8x4_6t_FS_cold_write_pd_s1 | 6T_ | 8x4 | no | FS 1.1 V / -40 C | per-device s1 | 4.5 | 347.7 | 366.8 | 1777.5 | 1374.1 | 1.097 | pass |
| mc_8x4_6t_FF_cold_write_pd_s4 | 6T_ | 8x4 | no | FF 1.1 V / -40 C | per-device s4 | 4.5 | 336.9 | 354.6 | 1777.5 | 1387.4 | 1.097 | pass |
| mc_8x4_6t_FF_cold_write_pd_s5 | 6T_ | 8x4 | no | FF 1.1 V / -40 C | per-device s5 | 4.5 | 335.7 | 354.2 | 1777.5 | 1387.9 | 1.097 | pass |
| w_8x4_6t_FF_cold_write_pd_s1 | 6T_ | 8x4 | no | FF 1.1 V / -40 C | per-device s1 | 4.5 | 333.7 | 352.7 | 1777.5 | 1389.6 | 1.097 | pass |
| mc_8x4_6t_FF_cold_write_pd_s2 | 6T_ | 8x4 | no | FF 1.1 V / -40 C | per-device s2 | 4.5 | 333.9 | 352.0 | 1777.5 | 1390.3 | 1.097 | pass |
| w_8x4_6t_FF_cold_write | 6T_ | 8x4 | no | FF 1.1 V / -40 C | nominal | 4.5 | 332.1 | 350.5 | 1777.5 | 1392.0 | 1.097 | pass |
| w_8x4_6t_FF_cold_write_pd_s2 | 6T_ | 8x4 | no | FF 1.1 V / -40 C | per-device s2 | 4.5 | 331.8 | 350.4 | 1777.5 | 1392.1 | 1.097 | pass |
| mc_8x4_6t_FF_cold_write_pd_s3 | 6T_ | 8x4 | no | FF 1.1 V / -40 C | per-device s3 | 4.5 | 331.2 | 349.4 | 1777.5 | 1393.1 | 1.097 | pass |
| mc_8x4_6t_FF_cold_write_pd_s1 | 6T_ | 8x4 | no | FF 1.1 V / -40 C | per-device s1 | 4.5 | 329.4 | 348.4 | 1777.5 | 1394.2 | 1.097 | pass |
| mc_16x16_6t_mux_FS_cold_write_pd_s2 | 6T_ | 16x16 | yes | FS 1.1 V / -40 C | per-device s2 | 4.75 | 363.7 | 389.9 | 1876.3 | 1447.3 | 1.096 | pass |
| mc_16x16_6t_mux_FS_cold_write_pd_s1 | 6T_ | 16x16 | yes | FS 1.1 V / -40 C | per-device s1 | 4.75 | 359.5 | 386.1 | 1876.3 | 1451.6 | 1.096 | pass |
| mc_16x16_6t_mux_FF_cold_write_pd_s2 | 6T_ | 16x16 | yes | FF 1.1 V / -40 C | per-device s2 | 4.75 | 344.6 | 369.7 | 1876.3 | 1469.6 | 1.096 | pass |
| mc_16x16_6t_mux_FF_cold_write_pd_s1 | 6T_ | 16x16 | yes | FF 1.1 V / -40 C | per-device s1 | 4.75 | 340.9 | 365.7 | 1876.3 | 1473.9 | 1.096 | pass |
| mc_16x16_10t_mux_FS_cold_write_pd_s2 | 10T | 16x16 | yes | FS 1.1 V / -40 C | per-device s2 | 5.0 | 361.0 | 387.6 | 1975.0 | 1548.6 | 1.096 | pass |
| mc_16x16_10t_mux_FS_cold_write_pd_s1 | 10T | 16x16 | yes | FS 1.1 V / -40 C | per-device s1 | 5.0 | 361.8 | 387.2 | 1975.0 | 1549.1 | 1.096 | pass |
| mc_8x4_10t_FS_cold_write_pd_s3 | 10T | 8x4 | no | FS 1.1 V / -40 C | per-device s3 | 5.0 | 354.1 | 373.6 | 1975.0 | 1564.1 | 1.097 | pass |
| mc_8x4_10t_FS_cold_write_pd_s5 | 10T | 8x4 | no | FS 1.1 V / -40 C | per-device s5 | 5.0 | 352.8 | 370.9 | 1975.0 | 1567.0 | 1.097 | pass |
| mc_16x16_10t_mux_FF_cold_write_pd_s2 | 10T | 16x16 | yes | FF 1.1 V / -40 C | per-device s2 | 5.0 | 344.8 | 370.1 | 1975.0 | 1567.9 | 1.096 | pass |
| mc_8x4_10t_FS_cold_write_pd_s4 | 10T | 8x4 | no | FS 1.1 V / -40 C | per-device s4 | 5.0 | 349.5 | 368.6 | 1975.0 | 1569.6 | 1.097 | pass |
| mc_16x16_10t_mux_FF_cold_write_pd_s1 | 10T | 16x16 | yes | FF 1.1 V / -40 C | per-device s1 | 5.0 | 342.9 | 368.2 | 1975.0 | 1570.0 | 1.096 | pass |
| mc_8x4_10t_FS_cold_write_pd_s2 | 10T | 8x4 | no | FS 1.1 V / -40 C | per-device s2 | 5.0 | 348.7 | 367.7 | 1975.0 | 1570.5 | 1.097 | pass |
| mc_8x4_10t_FS_cold_write_pd_s1 | 10T | 8x4 | no | FS 1.1 V / -40 C | per-device s1 | 5.0 | 346.4 | 365.6 | 1975.0 | 1572.8 | 1.097 | pass |
| mc_8x4_10t_FF_cold_write_pd_s3 | 10T | 8x4 | no | FF 1.1 V / -40 C | per-device s3 | 5.0 | 335.2 | 352.8 | 1975.0 | 1587.0 | 1.097 | pass |
| mc_8x4_10t_FF_cold_write_pd_s5 | 10T | 8x4 | no | FF 1.1 V / -40 C | per-device s5 | 5.0 | 334.2 | 352.7 | 1975.0 | 1587.1 | 1.097 | pass |
| w_8x4_10t_FF_cold_write | 10T | 8x4 | no | FF 1.1 V / -40 C | nominal | 5.0 | 332.6 | 351.8 | 1975.0 | 1588.0 | 1.097 | pass |
| mc_8x4_10t_FF_cold_write_pd_s4 | 10T | 8x4 | no | FF 1.1 V / -40 C | per-device s4 | 5.0 | 331.8 | 349.6 | 1975.0 | 1590.5 | 1.097 | pass |
| mc_8x4_10t_FF_cold_write_pd_s2 | 10T | 8x4 | no | FF 1.1 V / -40 C | per-device s2 | 5.0 | 328.8 | 346.7 | 1975.0 | 1593.7 | 1.097 | pass |
| mc_8x4_10t_FF_cold_write_pd_s1 | 10T | 8x4 | no | FF 1.1 V / -40 C | per-device s1 | 5.0 | 327.6 | 346.5 | 1975.0 | 1593.9 | 1.097 | pass |
| mc_512x4_6t_mux_SS_write_pd_s1 | 6T_ | 512x4 | yes | SS 0.9 V / 125 C | per-device s1 | 10.0 | 1786.3 | 1909.0 | 3950.0 | 1850.1 | 0.883 | pass |
| w_512x4_6t_mux_SS_write | 6T_ | 512x4 | yes | SS 0.9 V / 125 C | nominal | 10.0 | 1755.6 | 1877.6 | 3950.0 | 1884.7 | 0.889 | pass |

### 7.2 Monte-Carlo groups

| Size | Cell | Mux | Corner | Operation | Seeds | Pass | Checks | w_en min during WL [VDD] | w_en release after WL off [ps] | VWEN_ACCESS [V] | W->W w_en min [V] / new data after WL off [ps] | Data before w_en [ps] | Rail before WL [ps] | WL release to deadline [ps] | s_en / w_en off to PRE [ps] | s_en off to w_en [ps] | Sense margin [V] | Clock to Q [ps] |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128x8 | 10T | yes | SS 0.9 V / 125 C | read | 2 | 2 | 6902 | - | - | - | - / - | - | - | 196.5 | 208.3 / - | - | 0.813 | - |
| 128x8 | 6T_ | no | SS 0.9 V / 125 C | read | 4 | 4 | 13804 | - | - | - | - / - | - | - | 251.0 | 205.3 / - | - | 0.79 | - |
| 128x8 | 6T_ | yes | SS 0.9 V / 125 C | read | 4 | 4 | 13804 | - | - | - | - / - | - | - | 68.1 | 202.3 / - | - | 0.783 | - |
| 16x16 | 10T | yes | FF 1.1 V / -40 C | read | 2 | 2 | 2054 | - | - | - | - / - | - | - | 1861.9 | 48.3 / - | - | 1.084 | - |
| 16x16 | 10T | yes | FF 1.1 V / -40 C | read&write | 2 | 2 | 14454 | 0.998 | 141.7 | 0.0129 | - / - | 111.2 | 2404.2 | 1855.4 | - / 46.5 | 71.4 | 1.084 | 152.5 |
| 16x16 | 10T | yes | FF 1.1 V / -40 C | write | 2 | 2 | 2436 | 0.998 | - | 0.0 | 1.096 / 117.0 | 129.6 | 2405.1 | - | - / - | - | - | 152.2 |
| 16x16 | 10T | yes | FS 1.1 V / -40 C | read | 2 | 2 | 2054 | - | - | - | - / - | - | - | 1826.5 | 52.2 / - | - | 1.087 | - |
| 16x16 | 10T | yes | FS 1.1 V / -40 C | read&write | 2 | 2 | 14454 | 0.998 | 152.9 | 0.0126 | - / - | 118.1 | 2399.1 | 1822.2 | - / 50.3 | 76.6 | 1.087 | 161.0 |
| 16x16 | 10T | yes | FS 1.1 V / -40 C | write | 2 | 2 | 2436 | 0.998 | - | 0.0 | 1.096 / 126.6 | 137.3 | 2399.8 | - | - / - | - | - | 160.3 |
| 16x16 | 10T | yes | SF 0.9 V / 125 C | read | 2 | 2 | 2054 | - | - | - | - / - | - | - | 530.3 | 181.3 / - | - | 0.863 | - |
| 16x16 | 10T | yes | SF 0.9 V / 125 C | read&write | 2 | 2 | 14454 | 0.999 | 454.3 | 0.0103 | - / - | 345.0 | 2049.5 | 537.5 | - / 169.1 | 244.0 | 0.866 | 509.8 |
| 16x16 | 10T | yes | SF 0.9 V / 125 C | write | 2 | 2 | 2436 | 0.999 | - | 0.0 | 0.897 / 369.4 | 408.6 | 2045.4 | - | - / - | - | - | 513.7 |
| 16x16 | 10T | yes | SS 0.9 V / 125 C | read | 2 | 2 | 2054 | - | - | - | - / - | - | - | 300.0 | 205.3 / - | - | 0.877 | - |
| 16x16 | 10T | yes | SS 0.9 V / 125 C | read&write | 2 | 2 | 14454 | 0.999 | 523.6 | 0.0094 | - / - | 387.0 | 2008.4 | 321.6 | - / 192.9 | 276.1 | 0.877 | 562.3 |
| 16x16 | 10T | yes | SS 0.9 V / 125 C | write | 2 | 2 | 2436 | 0.999 | - | 0.0 | 0.896 / 429.8 | 457.5 | 2009.9 | - | - / - | - | - | 574.6 |
| 16x16 | 6T_ | yes | FF 1.1 V / -40 C | read | 2 | 2 | 2054 | - | - | - | - / - | - | - | 1754.9 | 48.3 / - | - | 1.088 | - |
| 16x16 | 6T_ | yes | FF 1.1 V / -40 C | read&write | 2 | 2 | 14454 | 0.998 | 143.6 | 0.013 | - / - | 111.6 | 2277.0 | 1735.3 | - / 45.5 | 71.8 | 1.089 | 143.6 |
| 16x16 | 6T_ | yes | FF 1.1 V / -40 C | write | 2 | 2 | 2436 | 0.998 | - | 0.0 | 1.096 / 114.3 | 129.8 | 2274.5 | - | - / - | - | - | 141.6 |
| 16x16 | 6T_ | yes | FS 1.1 V / -40 C | read | 2 | 2 | 2054 | - | - | - | - / - | - | - | 1720.6 | 52.0 / - | - | 1.089 | - |
| 16x16 | 6T_ | yes | FS 1.1 V / -40 C | read&write | 2 | 2 | 14454 | 0.998 | 154.2 | 0.0127 | - / - | 119.0 | 2275.0 | 1700.5 | - / 49.3 | 77.2 | 1.09 | 153.8 |
| 16x16 | 6T_ | yes | FS 1.1 V / -40 C | write | 2 | 2 | 2436 | 0.998 | - | 0.0 | 1.096 / 123.1 | 137.2 | 2271.8 | - | - / - | - | - | 152.6 |
| 16x16 | 6T_ | yes | SF 0.9 V / 125 C | read | 2 | 2 | 2054 | - | - | - | - / - | - | - | 507.6 | 175.7 / - | - | 0.873 | - |
| 16x16 | 6T_ | yes | SF 0.9 V / 125 C | read&write | 2 | 2 | 14454 | 0.999 | 473.9 | 0.0103 | - / - | 352.7 | 1939.4 | 392.9 | - / 164.6 | 240.6 | 0.882 | 494.0 |
| 16x16 | 6T_ | yes | SF 0.9 V / 125 C | write | 2 | 2 | 2436 | 0.999 | - | 0.0 | 0.897 / 363.1 | 419.2 | 1911.0 | - | - / - | - | - | 471.5 |
| 16x16 | 6T_ | yes | SS 0.9 V / 125 C | read | 2 | 2 | 2054 | - | - | - | - / - | - | - | 297.0 | 199.8 / - | - | 0.882 | - |
| 16x16 | 6T_ | yes | SS 0.9 V / 125 C | read&write | 2 | 2 | 14454 | 0.999 | 542.9 | 0.0093 | - / - | 344.8 | 1903.5 | 172.9 | - / 187.8 | 271.9 | 0.887 | 550.6 |
| 16x16 | 6T_ | yes | SS 0.9 V / 125 C | write | 2 | 2 | 2436 | 0.999 | - | 0.0 | 0.896 / 414.8 | 469.7 | 1872.3 | - | - / - | - | - | 524.8 |
| 256x4 | 6T_ | no | SF 0.9 V / 125 C | write | 1 | 1 | 3702 | 1.0 | - | 0.0 | 0.892 / 360.4 | 417.2 | 3119.1 | - | - / - | - | - | 605.0 |
| 256x4 | 6T_ | no | SS 0.9 V / 125 C | read | 8 | 8 | 29240 | - | - | - | - / - | - | - | 172.6 | 201.4 / - | - | 0.664 | - |
| 256x4 | 6T_ | no | SS 0.9 V / 125 C | write | 1 | 1 | 3702 | 1.0 | - | 0.0 | 0.891 / 422.1 | 466.8 | 3089.9 | - | - / - | - | - | 666.5 |
| 256x4 | 6T_ | yes | SS 0.9 V / 125 C | read | 4 | 4 | 14620 | - | - | - | - / - | - | - | 208.5 | 211.0 / - | - | 0.73 | - |
| 512x4 | 10T | yes | SS 0.9 V / 125 C | read | 2 | 2 | 2250 | - | - | - | - / - | - | - | 480.0 | 215.5 / - | - | 0.566 | - |
| 512x4 | 6T_ | no | SS 0.9 V / 125 C | read | 3 | 3 | 3375 | - | - | - | - / - | - | - | -39.5 | 221.3 / - | - | 0.562 | - |
| 512x4 | 6T_ | yes | SS 0.9 V / 125 C | read | 4 | 4 | 4500 | - | - | - | - / - | - | - | -60.0 | 220.1 / - | - | 0.616 | - |
| 512x4 | 6T_ | yes | SS 0.9 V / 125 C | write | 1 | 1 | 1172 | 1.0 | - | 0.0 | 0.883 / 486.3 | 429.4 | 5047.3 | - | - / - | - | - | 1032.0 |
| 64x16 | 10T | yes | FS 1.1 V / -40 C | read | 2 | 2 | 6854 | - | - | - | - / - | - | - | 2001.5 | 52.6 / - | - | 1.057 | - |
| 64x16 | 10T | yes | SF 0.9 V / 125 C | write | 2 | 2 | 7236 | 0.999 | - | 0.0 | 0.895 / 384.1 | 426.6 | 2298.1 | - | - / - | - | - | 516.2 |
| 64x16 | 10T | yes | SS 0.9 V / 125 C | read | 2 | 2 | 6854 | - | - | - | - / - | - | - | 376.3 | 203.6 / - | - | 0.835 | - |
| 64x16 | 10T | yes | SS 0.9 V / 125 C | write | 2 | 2 | 7236 | 0.999 | - | 0.0 | 0.895 / 444.5 | 478.6 | 2250.8 | - | - / - | - | - | 575.2 |
| 64x16 | 6T_ | yes | FS 1.1 V / -40 C | read | 2 | 2 | 6854 | - | - | - | - / - | - | - | 1769.6 | 53.4 / - | - | 1.074 | - |
| 64x16 | 6T_ | yes | SF 0.9 V / 125 C | write | 2 | 2 | 7236 | 0.999 | - | 0.0 | 0.895 / 369.7 | 401.1 | 2104.3 | - | - / - | - | - | 504.7 |
| 64x16 | 6T_ | yes | SS 0.9 V / 125 C | read | 2 | 2 | 6854 | - | - | - | - / - | - | - | 147.0 | 203.2 / - | - | 0.868 | - |
| 64x16 | 6T_ | yes | SS 0.9 V / 125 C | write | 2 | 2 | 7236 | 0.999 | - | 0.0 | 0.895 / 426.1 | 453.4 | 2070.6 | - | - / - | - | - | 563.2 |
| 8x4 | 10T | no | FF 1.1 V / -40 C | read | 5 | 5 | 915 | - | - | - | - / - | - | - | 1887.6 | 36.2 / - | - | 1.091 | - |
| 8x4 | 10T | no | FF 1.1 V / -40 C | read&write | 5 | 5 | 6875 | 0.997 | 125.0 | 0.0072 | - / - | 98.1 | 2414.9 | 1884.1 | - / 35.1 | 60.9 | 1.091 | 150.0 |
| 8x4 | 10T | no | FF 1.1 V / -40 C | write | 5 | 5 | 1150 | 0.999 | - | 0.0 | 1.097 / 105.8 | 119.2 | 2414.8 | - | - / - | - | - | 148.9 |
| 8x4 | 10T | no | FS 1.1 V / -40 C | read | 5 | 5 | 915 | - | - | - | - / - | - | - | 1854.7 | 39.0 / - | - | 1.092 | - |
| 8x4 | 10T | no | FS 1.1 V / -40 C | read&write | 5 | 5 | 6875 | 0.997 | 135.6 | 0.0073 | - / - | 104.7 | 2411.9 | 1850.6 | - / 37.9 | 65.4 | 1.093 | 159.3 |
| 8x4 | 10T | no | FS 1.1 V / -40 C | write | 5 | 5 | 1150 | 0.999 | - | 0.0 | 1.097 / 115.6 | 126.0 | 2411.0 | - | - / - | - | - | 158.3 |
| 8x4 | 10T | no | SF 0.9 V / 125 C | read | 5 | 5 | 915 | - | - | - | - / - | - | - | 669.4 | 135.7 / - | - | 0.877 | - |
| 8x4 | 10T | no | SF 0.9 V / 125 C | read&write | 5 | 5 | 6875 | 0.997 | 400.8 | 0.0076 | - / - | 309.6 | 2103.5 | 642.2 | - / 138.0 | 212.7 | 0.883 | 516.7 |
| 8x4 | 10T | no | SF 0.9 V / 125 C | write | 5 | 5 | 1150 | 0.999 | - | 0.0 | 0.896 / 336.0 | 387.7 | 2095.1 | - | - / - | - | - | 512.0 |
| 8x4 | 10T | no | SS 0.9 V / 125 C | read | 5 | 5 | 915 | - | - | - | - / - | - | - | 465.5 | 153.9 / - | - | 0.885 | - |
| 8x4 | 10T | no | SS 0.9 V / 125 C | read&write | 5 | 5 | 6875 | 0.997 | 465.2 | 0.0073 | - / - | 350.8 | 2074.7 | 430.6 | - / 156.4 | 240.9 | 0.889 | 571.2 |
| 8x4 | 10T | no | SS 0.9 V / 125 C | write | 5 | 5 | 1150 | 0.999 | - | 0.0 | 0.895 / 393.9 | 435.1 | 2061.7 | - | - / - | - | - | 567.7 |
| 8x4 | 6T_ | no | FF 1.1 V / -40 C | read | 5 | 5 | 915 | - | - | - | - / - | - | - | 1649.1 | 36.6 / - | - | 1.094 | - |
| 8x4 | 6T_ | no | FF 1.1 V / -40 C | read&write | 5 | 5 | 6875 | 0.997 | 124.8 | 0.0071 | - / - | 97.2 | 2161.4 | 1642.1 | - / 34.9 | 61.2 | 1.093 | 142.8 |
| 8x4 | 6T_ | no | FF 1.1 V / -40 C | write | 5 | 5 | 1150 | 0.999 | - | 0.0 | 1.097 / 106.9 | 118.6 | 2159.5 | - | - / - | - | - | 142.3 |
| 8x4 | 6T_ | no | FS 1.1 V / -40 C | read | 5 | 5 | 915 | - | - | - | - / - | - | - | 1616.8 | 39.5 / - | - | 1.094 | - |
| 8x4 | 6T_ | no | FS 1.1 V / -40 C | read&write | 5 | 5 | 6875 | 0.997 | 135.5 | 0.0072 | - / - | 103.2 | 2158.5 | 1608.5 | - / 38.0 | 65.7 | 1.094 | 151.6 |
| 8x4 | 6T_ | no | FS 1.1 V / -40 C | write | 5 | 5 | 1150 | 0.999 | - | 0.0 | 1.097 / 116.6 | 125.2 | 2156.9 | - | - / - | - | - | 151.5 |
| 8x4 | 6T_ | no | SF 0.9 V / 125 C | read | 5 | 5 | 915 | - | - | - | - / - | - | - | 452.3 | 139.9 / - | - | 0.886 | - |
| 8x4 | 6T_ | no | SF 0.9 V / 125 C | read&write | 5 | 5 | 6875 | 0.997 | 403.9 | 0.0079 | - / - | 306.0 | 1860.7 | 366.8 | - / 135.2 | 209.4 | 0.883 | 499.6 |
| 8x4 | 6T_ | no | SF 0.9 V / 125 C | write | 5 | 5 | 1150 | 0.999 | - | 0.0 | 0.895 / 345.4 | 386.4 | 1852.3 | - | - / - | - | - | 489.8 |
| 8x4 | 6T_ | no | SS 0.9 V / 125 C | read | 5 | 5 | 915 | - | - | - | - / - | - | - | 245.8 | 159.1 / - | - | 0.889 | - |
| 8x4 | 6T_ | no | SS 0.9 V / 125 C | read&write | 5 | 5 | 6875 | 0.996 | 468.4 | 0.0077 | - / - | 331.0 | 1830.6 | 157.3 | - / 153.8 | 237.6 | 0.888 | 554.4 |
| 8x4 | 6T_ | no | SS 0.9 V / 125 C | write | 5 | 5 | 1150 | 0.999 | - | 0.0 | 0.895 / 404.9 | 431.4 | 1820.0 | - | - / - | - | - | 546.5 |

### 7.3 Nominal and matrix cases

| Case | Cell | Size | Mux | Operation | Corner | Variation | T [ns] | Checks | w_en min during WL [VDD] | w_en release after WL off [ps] | VWEN_ACCESS [V] | W->W w_en min / new data after WL off [V / ps] | s_en / w_en off to PRE [ps] | s_en / w_en off to w_en [ps] | WL release to deadline [ps] | Rail before WL [ps] | Data before w_en [ps] | Sense margin [V] | TWSLOT / TRESTORE [ps] | Result |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| p_16x16_10t_SS_write_idle2 | 10T | 16x16 | no | write (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 1989 | 0.999 | 650.2 | 0.0011 | - / - | - / - | - / 4592.1 | - | 2034.3 | 424.0 | - | 932.5 | pass |
| p_16x16_6t_mux_SS_read_idle2 | 6T_ | 16x16 | yes | read (every 2) | SS 0.9 V / 125 C | nominal | 4.75 | 1608 | - | - | - | - / - | 5009.5 / - | - / - | 297.3 | - | - | 0.885 | 793.4 | pass |
| p_16x16_6t_mux_SS_write_idle2 | 6T_ | 16x16 | yes | write (every 2) | SS 0.9 V / 125 C | nominal | 4.75 | 1989 | 0.999 | 651.2 | 0.0016 | - / - | - / - | - / 4341.6 | - | 1899.6 | 423.2 | - | 939.8 | pass |
| p_16x64_6t_mux_SS_read_idle2 | 6T_ | 16x64 | yes | read (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 6216 | - | - | - | - / - | 5231.8 / - | - / - | 341.1 | - | - | 0.887 | 825.8 | pass |
| p_2x4_6t_FF_cold_read_idle2 | 6T_ | 2x4 | no | read (every 2) | FF 1.1 V / -40 C | nominal | 4.5 | 134 | - | - | - | - / - | 4558.1 / - | - / - | 1669.1 | - | - | 1.094 | 214.9 | pass |
| p_4x4_6t_FF_cold_write_idle2 | 6T_ | 4x4 | no | write (every 2) | FF 1.1 V / -40 C | nominal | 4.5 | 273 | 0.997 | 161.9 | 0.0033 | - / - | - / - | - / 4369.1 | - | 2164.9 | 104.2 | - | 236.3 | pass |
| p_64x16_6t_mux_SS_read_idle2 | 6T_ | 64x16 | yes | read (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 5592 | - | - | - | - / - | 5244.4 / - | - / - | 171.7 | - | - | 0.867 | 797.7 | pass |
| p_64x16_6t_mux_SS_write_idle2 | 6T_ | 64x16 | yes | write (every 2) | SS 0.9 V / 125 C | nominal | 5.0 | 5973 | 0.999 | 656.1 | 0.0013 | - / - | - / - | - / 4559.9 | - | 2064.5 | 410.5 | - | 927.5 | pass |
| p_8x4_10t_FF_cold_read_idle2 | 10T | 8x4 | no | read (every 2) | FF 1.1 V / -40 C | nominal | 5.0 | 272 | - | - | - | - / - | 5057.6 / - | - / - | 1888.6 | - | - | 1.091 | 218.6 | pass |
| p_8x4_10t_FF_cold_write_idle2 | 10T | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | nominal | 5.0 | 365 | 0.997 | 161.9 | 0.003 | - / - | - / - | - / 4870.6 | - | 2416.0 | 104.2 | - | 240.7 | pass |
| p_8x4_6t_FF_cold_read_idle2 | 6T_ | 8x4 | no | read (every 2) | FF 1.1 V / -40 C | nominal | 4.5 | 272 | - | - | - | - / - | 4557.4 / - | - / - | 1651.0 | - | - | 1.094 | 218.5 | pass |
| p_8x4_6t_FF_cold_read_idle2_pd_s1 | 6T_ | 8x4 | no | read (every 2) | FF 1.1 V / -40 C | per-device s1 | 4.5 | 272 | - | - | - | - / - | 4557.9 / - | - / - | 1644.7 | - | - | 1.094 | 219.2 | pass |
| p_8x4_6t_FF_cold_read_idle2_pd_s2 | 6T_ | 8x4 | no | read (every 2) | FF 1.1 V / -40 C | per-device s2 | 4.5 | 272 | - | - | - | - / - | 4558.1 / - | - / - | 1648.8 | - | - | 1.094 | 218.0 | pass |
| p_8x4_6t_FF_cold_write_idle2 | 6T_ | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | nominal | 4.5 | 365 | 0.997 | 162.0 | 0.003 | - / - | - / - | - / 4370.7 | - | 2160.8 | 104.2 | - | 240.2 | pass |
| p_8x4_6t_FF_cold_write_idle2_pd_s1 | 6T_ | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | per-device s1 | 4.5 | 365 | 0.997 | 161.5 | 0.0032 | - / - | - / - | - / 4367.7 | - | 2163.2 | 102.2 | - | 238.3 | pass |
| p_8x4_6t_FF_cold_write_idle2_pd_s2 | 6T_ | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | per-device s2 | 4.5 | 365 | 0.997 | 159.5 | 0.0031 | - / - | - / - | - / 4370.3 | - | 2160.5 | 101.5 | - | 238.8 | pass |
| p_8x4_6t_FF_cold_write_idle2_pd_s3 | 6T_ | 8x4 | no | write (every 2) | FF 1.1 V / -40 C | per-device s3 | 4.5 | 365 | 0.998 | 164.7 | 0.0031 | - / - | - / - | - / 4367.9 | - | 2162.4 | 102.5 | - | 240.7 | pass |
| p_8x4_6t_FF_cold_write_idle3 | 6T_ | 8x4 | no | write (every 3) | FF 1.1 V / -40 C | nominal | 4.5 | 367 | 0.997 | 162.0 | 0.003 | - / - | - / - | - / 8870.7 | - | 2160.8 | 104.2 | - | 240.2 | pass |
| p_8x4_6t_SS_write_idle2_pd_s1 | 6T_ | 8x4 | no | write (every 2) | SS 0.9 V / 125 C | per-device s1 | 4.5 | 365 | 0.997 | 594.4 | 0.0014 | - / - | - / - | - / 4037.7 | - | 1847.4 | 369.5 | - | 842.2 | pass |
| p_8x4_6t_SS_write_idle2_pd_s2 | 6T_ | 8x4 | no | write (every 2) | SS 0.9 V / 125 C | per-device s2 | 4.5 | 365 | 0.997 | 576.7 | 0.0013 | - / - | - / - | - / 4055.2 | - | 1821.4 | 358.4 | - | 839.8 | pass |
| p_8x4_6t_SS_write_idle2_pd_s3 | 6T_ | 8x4 | no | write (every 2) | SS 0.9 V / 125 C | per-device s3 | 4.5 | 365 | 0.997 | 625.1 | 0.0013 | - / - | - / - | - / 4042.8 | - | 1823.0 | 373.2 | - | 871.5 | pass |
| p_8x4_6t_TT_sequence_settle0 | 6T_ | 8x4 | no | read&write (0 settling stages) | TT 1.0 V / 25 C | nominal | 4.5 | 1375 | 0.997 | 107.7 | 0.0071 | - / - | - / 63.8 | 105.2 / - | 1324.0 | 2088.7 | 163.6 | 0.993 | - | pass |
| p_8x4_6t_TT_write_idle2 | 6T_ | 8x4 | no | write (every 2) | TT 1.0 V / 25 C | nominal | 4.5 | 365 | 0.997 | 264.0 | 0.0017 | - / - | - / - | - / 4298.6 | - | 2089.5 | 169.4 | - | 383.1 | pass |
| p_8x64_6t_mux_FF_cold_write_idle2 | 6T_ | 8x64 | yes | write (every 2) | FF 1.1 V / -40 C | nominal | 5.0 | 5165 | 0.995 | 180.2 | 0.0058 | - / - | - / - | - / 4882.4 | - | 2395.0 | 120.1 | - | 272.6 | pass |
| r_128x8_10t_mux_SS_read | 10T | 128x8 | yes | read | SS 0.9 V / 125 C | nominal | 6.0 | 3451 | - | - | - | - / - | 207.9 / - | - / - | 210.6 | - | - | 0.807 | 759.3 | pass |
| r_128x8_6t_SS_read | 6T_ | 128x8 | no | read | SS 0.9 V / 125 C | nominal | 5.5 | 3451 | - | - | - | - / - | 207.1 / - | - / - | 219.1 | - | - | 0.811 | 791.9 | pass |
| r_128x8_6t_mux_SS_read | 6T_ | 128x8 | yes | read | SS 0.9 V / 125 C | nominal | 5.75 | 3451 | - | - | - | - / - | 206.8 / - | - / - | 191.4 | - | - | 0.826 | 757.8 | pass |
| r_16x16_10t_mux_SS_read | 10T | 16x16 | yes | read | SS 0.9 V / 125 C | nominal | 5.0 | 1027 | - | - | - | - / - | 203.4 / - | - / - | 376.7 | - | - | 0.881 | 739.0 | pass |
| r_16x16_6t_mux_SS_read | 6T_ | 16x16 | yes | read | SS 0.9 V / 125 C | nominal | 4.75 | 1027 | - | - | - | - / - | 202.9 / - | - / - | 297.7 | - | - | 0.885 | 738.6 | pass |
| r_256x4_10t_SS_read | 10T | 256x4 | no | read | SS 0.9 V / 125 C | nominal | 8.0 | 3655 | - | - | - | - / - | 209.6 / - | - / - | 677.1 | - | - | 0.692 | 811.0 | pass |
| r_256x4_10t_mux_SS_read | 10T | 256x4 | yes | read | SS 0.9 V / 125 C | nominal | 8.0 | 3655 | - | - | - | - / - | 208.9 / - | - / - | 451.1 | - | - | 0.727 | 794.4 | pass |
| r_256x4_6t_SS_read | 6T_ | 256x4 | no | read | SS 0.9 V / 125 C | nominal | 6.75 | 3655 | - | - | - | - / - | 207.7 / - | - / - | 220.4 | - | - | 0.716 | 810.2 | pass |
| r_256x4_6t_mux_SS_read | 6T_ | 256x4 | yes | read | SS 0.9 V / 125 C | nominal | 7.25 | 3655 | - | - | - | - / - | 207.3 / - | - / - | 251.1 | - | - | 0.75 | 792.7 | pass |
| r_512x4_10t_mux_SS_read | 10T | 512x4 | yes | read | SS 0.9 V / 125 C | nominal | 10.75 | 1125 | - | - | - | - / - | 220.6 / - | - / - | 269.8 | - | - | 0.637 | 915.6 | pass |
| r_512x4_6t_SS_read | 6T_ | 512x4 | no | read | SS 0.9 V / 125 C | nominal | 9.25 | 1125 | - | - | - | - / - | 218.0 / - | - / - | 99.4 | - | - | 0.621 | 921.0 | pass |
| r_512x4_6t_mux_SS_read | 6T_ | 512x4 | yes | read | SS 0.9 V / 125 C | nominal | 10.0 | 1125 | - | - | - | - / - | 219.2 / - | - / - | 174.5 | - | - | 0.657 | 915.3 | pass |
| r_64x16_10t_mux_SS_read | 10T | 64x16 | yes | read | SS 0.9 V / 125 C | nominal | 5.5 | 3427 | - | - | - | - / - | 209.3 / - | - / - | 345.4 | - | - | 0.855 | 766.5 | pass |
| r_64x16_6t_mux_SS_read | 6T_ | 64x16 | yes | read | SS 0.9 V / 125 C | nominal | 5.0 | 3427 | - | - | - | - / - | 208.1 / - | - / - | 172.5 | - | - | 0.867 | 765.0 | pass |
| r_8x128_6t_mux_SS_read | 6T_ | 8x128 | yes | read | SS 0.9 V / 125 C | nominal | 6.0 | 4771 | - | - | - | - / - | 226.7 / - | - / - | 765.1 | - | - | 0.889 | 835.8 | pass |
| r_8x4_10t_FF_cold_read | 10T | 8x4 | no | read | FF 1.1 V / -40 C | nominal | 5.0 | 183 | - | - | - | - / - | 37.0 / - | - / - | 1888.6 | - | - | 1.091 | 198.7 | pass |
| r_8x4_10t_FF_cold_read_pd_s1 | 10T | 8x4 | no | read | FF 1.1 V / -40 C | per-device s1 | 5.0 | 183 | - | - | - | - / - | 37.2 / - | - / - | 1878.8 | - | - | 1.092 | 199.7 | pass |
| r_8x4_10t_SS_read | 10T | 8x4 | no | read | SS 0.9 V / 125 C | nominal | 5.0 | 183 | - | - | - | - / - | 162.6 / - | - / - | 475.4 | - | - | 0.888 | 737.4 | pass |
| r_8x4_6t_FF_cold_read | 6T_ | 8x4 | no | read | FF 1.1 V / -40 C | nominal | 4.5 | 183 | - | - | - | - / - | 36.9 / - | - / - | 1651.1 | - | - | 1.094 | 198.7 | pass |
| r_8x4_6t_FF_cold_read_pd_s1 | 6T_ | 8x4 | no | read | FF 1.1 V / -40 C | per-device s1 | 4.5 | 183 | - | - | - | - / - | 37.0 / - | - / - | 1644.6 | - | - | 1.094 | 198.9 | pass |
| r_8x4_6t_FF_cold_read_pd_s2 | 6T_ | 8x4 | no | read | FF 1.1 V / -40 C | per-device s2 | 4.5 | 183 | - | - | - | - / - | 36.5 / - | - / - | 1648.6 | - | - | 1.094 | 197.1 | pass |
| r_8x4_6t_SS_read | 6T_ | 8x4 | no | read | SS 0.9 V / 125 C | nominal | 4.5 | 183 | - | - | - | - / - | 162.0 / - | - / - | 268.1 | - | - | 0.891 | 736.3 | pass |
| s_16x16_10t_SS_sequence | 10T | 16x16 | no | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 7227 | 0.999 | 539.3 | 0.0078 | - / - | - / 198.6 | 284.1 / - | 410.8 | 2035.1 | 455.3 | 0.888 | - | pass |
| s_16x16_10t_SS_sequence_pd_s1 | 10T | 16x16 | no | read&write | SS 0.9 V / 125 C | per-device s1 | 5.0 | 7227 | 0.999 | 529.3 | 0.0076 | - / - | - / 203.8 | 305.4 / - | 428.7 | 2015.1 | 450.7 | 0.888 | - | pass |
| s_16x16_10t_SS_sequence_pd_s2 | 10T | 16x16 | no | read&write | SS 0.9 V / 125 C | per-device s2 | 5.0 | 7227 | 0.999 | 521.1 | 0.0078 | - / - | - / 198.1 | 281.2 / - | 406.6 | 1990.5 | 445.8 | 0.888 | - | pass |
| s_16x16_10t_mux_SS_sequence | 10T | 16x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 7227 | 0.999 | 539.4 | 0.0092 | - / - | - / 199.0 | 283.8 / - | 345.8 | 2029.4 | 420.9 | 0.881 | - | pass |
| s_16x16_6t_SS_sequence | 6T_ | 16x16 | no | read&write | SS 0.9 V / 125 C | nominal | 4.5 | 7227 | 0.999 | 538.8 | 0.0078 | - / - | - / 198.6 | 283.7 / - | 208.0 | 1781.3 | 438.6 | 0.891 | - | pass |
| s_16x16_6t_mux_SS_sequence | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 4.75 | 7227 | 0.999 | 539.1 | 0.0092 | - / - | - / 198.8 | 284.0 / - | 267.1 | 1901.3 | 421.1 | 0.885 | - | pass |
| s_16x16_6t_mux_SS_sequence_pd_s1 | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | per-device s1 | 4.75 | 7227 | 0.999 | 557.3 | 0.0086 | - / - | - / 198.9 | 292.8 / - | 301.8 | 1892.9 | 415.5 | 0.883 | - | pass |
| s_16x16_6t_mux_SS_sequence_pd_s2 | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | per-device s2 | 4.75 | 7227 | 0.999 | 539.1 | 0.0093 | - / - | - / 196.9 | 276.4 / - | 239.3 | 1889.2 | 401.5 | 0.887 | - | pass |
| s_16x16_6t_mux_SS_sequence_pd_s3 | 6T_ | 16x16 | yes | read&write | SS 0.9 V / 125 C | per-device s3 | 4.75 | 7227 | 0.999 | 526.2 | 0.0095 | - / - | - / 207.3 | 280.8 / - | 263.8 | 1881.0 | 423.8 | 0.883 | - | pass |
| s_16x32_6t_mux_SS_sequence | 6T_ | 16x32 | yes | read&write | SS 0.9 V / 125 C | nominal | 4.75 | 14219 | 0.998 | 548.4 | 0.0101 | - / - | - / 208.6 | 295.4 / - | 218.6 | 1889.6 | 458.8 | 0.887 | - | pass |
| s_32x16_10t_mux_SS_sequence | 10T | 32x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 11707 | 0.999 | 539.5 | 0.0075 | - / - | - / 198.8 | 287.2 / - | 267.2 | 1996.9 | 432.1 | 0.875 | - | pass |
| s_32x16_6t_SS_sequence | 6T_ | 32x16 | no | read&write | SS 0.9 V / 125 C | nominal | 4.5 | 11707 | 0.999 | 539.5 | 0.007 | - / - | - / 198.8 | 287.2 / - | 147.5 | 1746.4 | 392.7 | 0.888 | - | pass |
| s_32x16_6t_mux_SS_sequence | 6T_ | 32x16 | yes | read&write | SS 0.9 V / 125 C | nominal | 4.75 | 11707 | 0.999 | 539.6 | 0.0075 | - / - | - / 199.3 | 287.9 / - | 198.0 | 1868.4 | 409.6 | 0.881 | - | pass |
| s_8x4_10t_FF_cold_sequence | 10T | 8x4 | no | read&write | FF 1.1 V / -40 C | nominal | 5.0 | 1375 | 0.997 | 127.3 | 0.0071 | - / - | - / 35.3 | 61.8 / - | 1886.1 | 2415.9 | 99.4 | 1.091 | - | pass |
| s_8x4_10t_mux_SS_sequence | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | nominal | 5.0 | 1375 | 0.997 | 483.7 | 0.0089 | - / - | - / 157.4 | 246.0 / - | 383.4 | 2091.9 | 365.2 | 0.882 | - | pass |
| s_8x4_10t_mux_SS_sequence_pd_s1 | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | per-device s1 | 5.0 | 1375 | 0.997 | 480.0 | 0.0092 | - / - | - / 154.5 | 241.3 / - | 306.2 | 2067.1 | 348.1 | 0.886 | - | pass |
| s_8x4_10t_mux_SS_sequence_pd_s2 | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | per-device s2 | 5.0 | 1375 | 0.997 | 468.4 | 0.009 | - / - | - / 159.2 | 242.6 / - | 369.3 | 2094.1 | 351.2 | 0.879 | - | pass |
| s_8x4_10t_mux_SS_sequence_pd_s3 | 10T | 8x4 | yes | read&write | SS 0.9 V / 125 C | per-device s3 | 5.0 | 1375 | 0.997 | 453.8 | 0.0092 | - / - | - / 148.6 | 248.1 / - | 370.9 | 2071.3 | 358.3 | 0.886 | - | pass |
| s_8x4_6t_FF_cold_sequence | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | nominal | 4.5 | 1375 | 0.997 | 127.3 | 0.0071 | - / - | - / 35.2 | 61.8 / - | 1648.5 | 2161.5 | 99.4 | 1.094 | - | pass |
| s_8x4_6t_FF_cold_sequence_pd_s1 | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | per-device s1 | 4.5 | 1375 | 0.997 | 124.8 | 0.0071 | - / - | - / 34.3 | 62.6 / - | 1644.0 | 2157.6 | 100.1 | 1.094 | - | pass |
| s_8x4_6t_FF_cold_sequence_pd_s2 | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | per-device s2 | 4.5 | 1375 | 0.997 | 127.6 | 0.0073 | - / - | - / 35.0 | 61.7 / - | 1645.4 | 2162.8 | 98.5 | 1.094 | - | pass |
| s_8x4_6t_FF_cold_sequence_pd_s3 | 6T_ | 8x4 | no | read&write | FF 1.1 V / -40 C | per-device s3 | 4.5 | 1375 | 0.997 | 126.5 | 0.0071 | - / - | - / 36.0 | 61.7 / - | 1645.1 | 2159.6 | 99.1 | 1.094 | - | pass |
| s_8x4_6t_SS_sequence | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | nominal | 4.5 | 1375 | 0.997 | 484.3 | 0.0071 | - / - | - / 157.6 | 245.8 / - | 260.0 | 1843.4 | 369.8 | 0.891 | - | pass |
| s_8x4_6t_SS_sequence_pd_s1 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s1 | 4.5 | 1375 | 0.997 | 462.2 | 0.0075 | - / - | - / 151.9 | 253.1 / - | 212.2 | 1798.3 | 375.7 | 0.891 | - | pass |
| s_8x4_6t_SS_sequence_pd_s2 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s2 | 4.5 | 1375 | 0.997 | 483.5 | 0.0073 | - / - | - / 153.2 | 252.0 / - | 213.8 | 1842.0 | 365.2 | 0.892 | - | pass |
| s_8x4_6t_SS_sequence_pd_s3 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s3 | 4.5 | 1375 | 0.997 | 462.4 | 0.0068 | - / - | - / 163.3 | 245.8 / - | 203.6 | 1818.6 | 362.8 | 0.892 | - | pass |
| s_8x4_6t_SS_sequence_pd_s4 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s4 | 4.5 | 1375 | 0.997 | 471.4 | 0.0074 | - / - | - / 150.1 | 236.0 / - | 225.9 | 1805.0 | 361.8 | 0.891 | - | pass |
| s_8x4_6t_SS_sequence_pd_s5 | 6T_ | 8x4 | no | read&write | SS 0.9 V / 125 C | per-device s5 | 4.5 | 1375 | 0.997 | 484.3 | 0.0073 | - / - | - / 158.0 | 234.0 / - | 211.4 | 1841.4 | 352.0 | 0.892 | - | pass |
| s_8x8_6t_mux_FF_cold_sequence | 6T_ | 8x8 | yes | read&write | FF 1.1 V / -40 C | nominal | 4.75 | 2579 | 0.998 | 131.7 | 0.0122 | - / - | - / 37.6 | 64.3 / - | 1756.8 | 2288.4 | 102.0 | 1.09 | - | pass |
| w_128x16_10t_mux_SS_write | 10T | 128x16 | yes | write | SS 0.9 V / 125 C | nominal | 6.0 | 6818 | 0.999 | - | 0.0 | 0.893 / 459.8 | - / - | - / - | - | 2629.4 | 487.5 | - | 1320.4 | pass |
| w_128x16_6t_SS_write | 6T_ | 128x16 | no | write | SS 0.9 V / 125 C | nominal | 5.5 | 6818 | 0.999 | - | 0.0 | 0.893 / 458.3 | - / - | - / - | - | 2377.0 | 488.8 | - | 1310.6 | pass |
| w_128x16_6t_mux_SS_write | 6T_ | 128x16 | yes | write | SS 0.9 V / 125 C | nominal | 5.75 | 6818 | 0.999 | - | 0.0 | 0.893 / 458.4 | - / - | - / - | - | 2505.1 | 487.1 | - | 1317.0 | pass |
| w_128x8_6t_SS_write | 6T_ | 128x8 | no | write | SS 0.9 V / 125 C | nominal | 5.5 | 3546 | 0.999 | - | 0.0 | 0.894 / 442.6 | - / - | - / - | - | 2392.4 | 477.0 | - | 1306.5 | pass |
| w_16x16_10t_mux_SS_write | 10T | 16x16 | yes | write | SS 0.9 V / 125 C | nominal | 5.0 | 1218 | 0.999 | - | 0.0 | 0.893 / 448.2 | - / - | - / - | - | 2029.3 | 490.3 | - | 1243.1 | pass |
| w_16x16_6t_mux_SS_write | 6T_ | 16x16 | yes | write | SS 0.9 V / 125 C | nominal | 4.75 | 1218 | 0.999 | - | 0.0 | 0.893 / 448.6 | - / - | - / - | - | 1902.1 | 490.4 | - | 1242.0 | pass |
| w_16x64_6t_mux_SS_write | 6T_ | 16x64 | yes | write | SS 0.9 V / 125 C | nominal | 5.0 | 4722 | 0.998 | - | 0.0 | 0.891 / 493.4 | - / - | - / - | - | 2000.6 | 507.8 | - | 1311.0 | pass |
| w_256x4_6t_SF_write | 6T_ | 256x4 | no | write | SF 0.9 V / 125 C | nominal | 6.75 | 3702 | 1.0 | - | 0.0 | 0.893 / 372.1 | - / - | - / - | - | 3151.1 | 416.4 | - | 1263.0 | pass |
| w_256x4_6t_SS_write | 6T_ | 256x4 | no | write | SS 0.9 V / 125 C | nominal | 6.75 | 3702 | 1.0 | - | 0.0 | 0.892 / 434.6 | - / - | - / - | - | 3124.1 | 465.8 | - | 1407.9 | pass |
| w_32x128_6t_mux_SS_write | 6T_ | 32x128 | yes | write | SS 0.9 V / 125 C | nominal | 6.0 | 15570 | 0.997 | - | 0.0 | 0.885 / 498.4 | - / - | - / - | - | 2455.4 | 510.5 | - | 1412.2 | pass |
| w_32x16_10t_SS_write | 10T | 32x16 | no | write | SS 0.9 V / 125 C | nominal | 5.0 | 2018 | 0.999 | - | 0.0 | 0.891 / 448.6 | - / - | - / - | - | 2002.5 | 490.8 | - | 1275.0 | pass |
| w_32x16_10t_mux_SS_write | 10T | 32x16 | yes | write | SS 0.9 V / 125 C | nominal | 5.0 | 2018 | 0.999 | - | 0.0 | 0.891 / 448.9 | - / - | - / - | - | 1997.3 | 490.1 | - | 1295.1 | pass |
| w_32x16_10t_mux_SS_write_pd_s1 | 10T | 32x16 | yes | write | SS 0.9 V / 125 C | per-device s1 | 5.0 | 2018 | 0.999 | - | 0.0 | 0.894 / 433.1 | - / - | - / - | - | 1973.2 | 460.2 | - | 1302.8 | pass |
| w_32x16_10t_mux_SS_write_pd_s2 | 10T | 32x16 | yes | write | SS 0.9 V / 125 C | per-device s2 | 5.0 | 2018 | 0.999 | - | 0.0 | 0.894 / 436.9 | - / - | - / - | - | 2002.0 | 484.9 | - | 1263.8 | pass |
| w_32x16_6t_SF_write | 6T_ | 32x16 | no | write | SF 0.9 V / 125 C | nominal | 4.5 | 2018 | 0.999 | - | 0.0 | 0.892 / 388.5 | - / - | - / - | - | 1783.9 | 435.4 | - | 1149.2 | pass |
| w_32x16_6t_SS_write | 6T_ | 32x16 | no | write | SS 0.9 V / 125 C | nominal | 4.5 | 2018 | 0.999 | - | 0.0 | 0.891 / 448.7 | - / - | - / - | - | 1747.7 | 489.7 | - | 1271.7 | pass |
| w_32x16_6t_SS_write_pd_s1 | 6T_ | 32x16 | no | write | SS 0.9 V / 125 C | per-device s1 | 4.5 | 2018 | 0.999 | - | 0.0 | 0.894 / 436.1 | - / - | - / - | - | 1727.2 | 480.7 | - | 1284.6 | pass |
| w_32x16_6t_SS_write_pd_s2 | 6T_ | 32x16 | no | write | SS 0.9 V / 125 C | per-device s2 | 4.5 | 2018 | 0.999 | - | 0.0 | 0.895 / 458.7 | - / - | - / - | - | 1740.1 | 469.2 | - | 1293.0 | pass |
| w_32x16_6t_SS_write_pd_s3 | 6T_ | 32x16 | no | write | SS 0.9 V / 125 C | per-device s3 | 4.5 | 2018 | 0.999 | - | 0.0 | 0.895 / 432.7 | - / - | - / - | - | 1719.1 | 490.5 | - | 1297.4 | pass |
| w_32x16_6t_mux_SF_write | 6T_ | 32x16 | yes | write | SF 0.9 V / 125 C | nominal | 4.75 | 2018 | 0.999 | - | 0.0 | 0.892 / 388.4 | - / - | - / - | - | 1906.7 | 435.8 | - | 1168.3 | pass |
| w_32x16_6t_mux_SS_write | 6T_ | 32x16 | yes | write | SS 0.9 V / 125 C | nominal | 4.75 | 2018 | 0.999 | - | 0.0 | 0.89 / 448.4 | - / - | - / - | - | 1868.3 | 490.6 | - | 1291.9 | pass |
| w_32x16_6t_mux_SS_write_pd_s1 | 6T_ | 32x16 | yes | write | SS 0.9 V / 125 C | per-device s1 | 4.75 | 2018 | 0.999 | - | 0.0 | 0.893 / 451.4 | - / - | - / - | - | 1881.5 | 464.8 | - | 1298.3 | pass |
| w_32x16_6t_mux_SS_write_pd_s2 | 6T_ | 32x16 | yes | write | SS 0.9 V / 125 C | per-device s2 | 4.75 | 2018 | 0.999 | - | 0.0 | 0.895 / 438.7 | - / - | - / - | - | 1848.8 | 479.7 | - | 1272.9 | pass |
| w_32x16_6t_mux_SS_write_pd_s3 | 6T_ | 32x16 | yes | write | SS 0.9 V / 125 C | per-device s3 | 4.75 | 2018 | 0.999 | - | 0.0 | 0.894 / 462.6 | - / - | - / - | - | 1875.8 | 484.7 | - | 1280.4 | pass |
| w_32x32_6t_SS_write | 6T_ | 32x32 | no | write | SS 0.9 V / 125 C | nominal | 4.5 | 3954 | 0.999 | - | 0.0 | 0.89 / 460.4 | - / - | - / - | - | 1733.8 | 499.8 | - | 1292.9 | pass |
| w_32x32_6t_SS_write_pd_s1 | 6T_ | 32x32 | no | write | SS 0.9 V / 125 C | per-device s1 | 4.5 | 3954 | 0.999 | - | 0.0 | 0.893 / 460.7 | - / - | - / - | - | 1695.4 | 490.0 | - | 1300.6 | pass |
| w_32x32_6t_SS_write_pd_s2 | 6T_ | 32x32 | no | write | SS 0.9 V / 125 C | per-device s2 | 4.5 | 3954 | 0.999 | - | 0.0 | 0.893 / 448.7 | - / - | - / - | - | 1700.3 | 490.4 | - | 1310.1 | pass |
| w_32x32_6t_mux_SS_write | 6T_ | 32x32 | yes | write | SS 0.9 V / 125 C | nominal | 4.75 | 3954 | 0.999 | - | 0.0 | 0.891 / 460.6 | - / - | - / - | - | 1857.9 | 498.4 | - | 1313.1 | pass |
| w_32x64_6t_SS_write | 6T_ | 32x64 | no | write | SS 0.9 V / 125 C | nominal | 5.0 | 7826 | 0.998 | - | 0.0 | 0.887 / 493.8 | - / - | - / - | - | 1973.1 | 508.0 | - | 1347.6 | pass |
| w_32x64_6t_mux_SS_write | 6T_ | 32x64 | yes | write | SS 0.9 V / 125 C | nominal | 5.0 | 7826 | 0.998 | - | 0.0 | 0.887 / 493.8 | - / - | - / - | - | 1970.2 | 507.7 | - | 1367.2 | pass |
| w_32x64_6t_mux_SS_write_pd_s1 | 6T_ | 32x64 | yes | write | SS 0.9 V / 125 C | per-device s1 | 5.0 | 7826 | 0.998 | - | 0.0 | 0.893 / 484.4 | - / - | - / - | - | 1957.8 | 498.7 | - | 1362.2 | pass |
| w_32x64_6t_mux_SS_write_pd_s2 | 6T_ | 32x64 | yes | write | SS 0.9 V / 125 C | per-device s2 | 5.0 | 7826 | 0.998 | - | 0.0 | 0.893 / 470.1 | - / - | - / - | - | 1953.1 | 501.6 | - | 1339.4 | pass |
| w_512x4_6t_mux_SS_write | 6T_ | 512x4 | yes | write | SS 0.9 V / 125 C | nominal | 10.0 | 1172 | 1.0 | - | 0.0 | 0.889 / 443.5 | - / - | - / - | - | 5024.7 | 451.8 | - | 1755.6 | pass |
| w_64x16_10t_mux_SF_write | 10T | 64x16 | yes | write | SF 0.9 V / 125 C | nominal | 5.5 | 3618 | 0.999 | - | 0.0 | 0.895 / 388.6 | - / - | - / - | - | 2358.6 | 429.9 | - | 1171.6 | pass |
| w_64x16_10t_mux_SS_write | 10T | 64x16 | yes | write | SS 0.9 V / 125 C | nominal | 5.5 | 3618 | 0.999 | - | 0.0 | 0.895 / 450.3 | - / - | - / - | - | 2321.0 | 484.3 | - | 1304.4 | pass |
| w_64x16_6t_SS_write | 6T_ | 64x16 | no | write | SS 0.9 V / 125 C | nominal | 4.75 | 3618 | 0.999 | - | 0.0 | 0.895 / 449.0 | - / - | - / - | - | 1942.6 | 483.7 | - | 1289.4 | pass |
| w_64x16_6t_mux_SF_write | 6T_ | 64x16 | yes | write | SF 0.9 V / 125 C | nominal | 5.0 | 3618 | 0.999 | - | 0.0 | 0.895 / 389.0 | - / - | - / - | - | 2103.0 | 429.5 | - | 1170.9 | pass |
| w_64x16_6t_mux_SS_write | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | nominal | 5.0 | 3618 | 0.999 | - | 0.0 | 0.895 / 448.8 | - / - | - / - | - | 2065.6 | 483.9 | - | 1300.3 | pass |
| w_64x16_6t_mux_SS_write_pd_s1 | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | per-device s1 | 5.0 | 3618 | 0.999 | - | 0.0 | 0.896 / 479.4 | - / - | - / - | - | 2059.7 | 473.3 | - | 1326.3 | pass |
| w_64x16_6t_mux_SS_write_pd_s2 | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | per-device s2 | 5.0 | 3618 | 0.999 | - | 0.0 | 0.895 / 440.3 | - / - | - / - | - | 2055.0 | 484.0 | - | 1290.8 | pass |
| w_64x16_6t_mux_SS_write_pd_s3 | 6T_ | 64x16 | yes | write | SS 0.9 V / 125 C | per-device s3 | 5.0 | 3618 | 0.999 | - | 0.0 | 0.895 / 423.3 | - / - | - / - | - | 2075.9 | 481.7 | - | 1283.2 | pass |
| w_8x4_10t_FF_cold_write | 10T | 8x4 | no | write | FF 1.1 V / -40 C | nominal | 5.0 | 230 | 0.999 | - | 0.0 | 1.097 / 108.5 | - / - | - / - | - | 2416.3 | 121.1 | - | 332.6 | pass |
| w_8x4_10t_SS_write | 10T | 8x4 | no | write | SS 0.9 V / 125 C | nominal | 5.0 | 230 | 0.999 | - | 0.0 | 0.895 / 415.7 | - / - | - / - | - | 2100.3 | 447.8 | - | 1195.1 | pass |
| w_8x4_6t_FF_cold_write | 6T_ | 8x4 | no | write | FF 1.1 V / -40 C | nominal | 4.5 | 230 | 0.999 | - | 0.0 | 1.097 / 108.5 | - / - | - / - | - | 2162.1 | 121.1 | - | 332.1 | pass |
| w_8x4_6t_FF_cold_write_pd_s1 | 6T_ | 8x4 | no | write | FF 1.1 V / -40 C | per-device s1 | 4.5 | 230 | 0.999 | - | 0.0 | 1.097 / 107.9 | - / - | - / - | - | 2163.4 | 119.5 | - | 333.7 | pass |
| w_8x4_6t_FF_cold_write_pd_s2 | 6T_ | 8x4 | no | write | FF 1.1 V / -40 C | per-device s2 | 4.5 | 230 | 0.999 | - | 0.0 | 1.097 / 106.9 | - / - | - / - | - | 2160.7 | 118.8 | - | 331.8 | pass |
| w_8x4_6t_SS_write | 6T_ | 8x4 | no | write | SS 0.9 V / 125 C | nominal | 4.5 | 230 | 0.999 | - | 0.0 | 0.895 / 415.8 | - / - | - / - | - | 1845.1 | 447.5 | - | 1193.4 | pass |

### 7.4 DC operating point

| Case | Ranks | Failed with | Passed with |
|---|---:|---|---|
| p_16x16_6t_mux_SS_read_idle2 | 4 | default | Newton line search (`fix-B4`) |
| r_16x16_10t_mux_SS_read | 4 | default | Newton line search (`fix-B5`) |
| mc_16x16_10t_mux_FS_cold_read_pd_s2 | 1 | default; line search (`fix-A`) | line search + gmin stepping (`fix-B1`) |
| mc_64x16_10t_mux_FS_cold_read_pd_s1 | 1 | default; line search (`fix-B2`) | line search + gmin stepping (`fix-C1`) |
| mc_64x16_10t_mux_FS_cold_read_pd_s2 | 1 | default; line search (`fix-B3`); + gmin stepping (`fix-C2`); line search + `GMIN=1e-10` (probe) | line search + gmin stepping + `.OPTIONS DEVICE GMIN=1e-10` (`fix-D1`) |

Line search is `.OPTIONS NONLIN SEARCHMETHOD=2`, gmin stepping
`CONTINUATION=3`. The last sample converges only with the device GMIN at 100x
its default, a conductance of 0.1 nS across every junction for the whole run;
its 3427 checks pass with it, so it is evidence of a perturbed circuit, not of
the exact one. At four ranks the same case converges without it, but four ranks
materialize path-keyed draws, a different sample, so that run stays a probe
(`V2.1.10-slot-budget/probes/dcop`). As in V2.1.9 (one 10T 16x16 mux sample at
FS 1.1 V / -40 C that failed on the V2.1.8 sources too), the aborts are 10T and
mux reads, mostly at FS -40 C; none is a write or a sequence, and the same
decks converge for other samples. `gen_fix.py` now escalates from the options
of the failed attempt itself (it read the original case, so a second failure
never reached gmin stepping), and a fix queue has a 12-hour budget (six hours
stopped `fix-A` before its second case, a 6-hour timeout).

## 8. Limits

* Screening at the class bounds with illustrative wires; extracted metal,
  half-select writes and the yield estimator remain the carried Phase 6 scope.
  One to eight seeds per group show that the worst corners work under mismatch,
  not a failure probability.
* The new data waits for the replica wordline observer and its settling stages,
  the same observer that V2.1.9 used for the drivers' release: its margin to the
  local wordline is a structural one (gate counts and the replica's matched
  wire), not an observation of every row.
* The read -> write and idle -> write setup lead of the data over the write
  enable is a delay-line margin (the select delay and the slot path against the
  registers' clock-to-Q and the latch), as in V2.1.7.
* Option B (a self-timed write wordline, no latches) is not evaluated.
