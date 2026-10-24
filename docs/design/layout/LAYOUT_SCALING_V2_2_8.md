# Single-bank layout scaling and driver candidates — V2.2.8

The layout generator builds single banks up to 512×512 at the 705 nm column
and 1365 nm row pitch, with TIME_CONTROL and the data registers in the
floorplan. Beyond the automatic 512×256 timing envelope an explicit
diagnostic clock is required; no new operating point or release
qualification is implied.

## Controller

The current source deletes the wordline-setup RC, the three observer RC
filters and the tall-decoder settling capacitors. Bank interconnect stays
distributed RC; the far-end PRE, isolation, enable and replica-wordline
observers and the short digital delay chains remain. Legacy tau arguments are
accepted but insert nothing. The archived automatic 8×8 RC reference (earlier
controller) is declined when source topology differs, with the reason in
`variation_summary.calibrated_rc`; an explicitly requested stale profile
raises an error.

`compact_control=True` factors isolation completion into two NAND gates and
drops the unused sense-enable complement inverter and the obsolete hold-latch
and complement loads from sizing. The external port contract is unchanged;
driver snapshots from the previous RC architecture are rejected.

Large gates may split into parallel finger slices with unchanged source W/L.
The router measures distance to the actual routing tree (a bounding-box
estimate explored tall trees for minutes), routes scarce address accesses
first and retries congested nets with priority. Full source MOS, finger and
gate-area checks remain. For large banks the Python connectivity audit skips
the core interior but keeps its boundary and every interface endpoint;
Calibre checks the complete GDS and source.

## Shared decoder and controller floorplan

Retired on 2026-10-16 and replaced by the
[split-array centre floorplan](#split-arrays-with-a-centre-periphery); kept as
its area baseline.

The corner plan fused the row decoder and TIME_CONTROL into one
`ROW_DECODER_TIME_CONTROL` module whose lower edge was level with the
data-register band. Controller cells filled the corner below the decoder,
then a lane beside the decoder gates (4 µm wide at 32 rows, 5 µm for 33–256,
20 µm above 256, 10 µm below 32), 1 µm from the decoder because their row
heights differ (1500 nm against the decoder's 1365 nm).
Oversized gates split into W/L-conserving slices. The router kept to the two
regions; address and `wl_en` pins stayed under their decoder spines, and a
1 µm band kept the address-pin antenna taps clear of logic. For tall frames
the routing heuristic adds the minimum layer-change cost of wrong-direction
travel, and long feedback nets route first over the whole region. Folded NAND
output metal clears its straps, same-net M4–M6 notches are closed, and the
replica-wordline and far sense-enable receivers use reserved diffusion taps
because their drivers join only through M7/M8. `layout preview` also writes
`decoder_control.png`.

Evidence is in
[`DECODER_CONTROL_FUSION_V2_2_8.json`](../../data/DECODER_CONTROL_FUSION_V2_2_8.json):

| Bank | Width × height (µm) | Macro area (µm²) | Area reduction from archived layout |
| --- | ---: | ---: | ---: |
| 32×32 | 42.7825 × 73.0975 | 3127.294 | 11.62% |
| 128×128 | 115.4825 × 204.1375 | 23574.309 | 7.27% |
| 512×512 | 404.2825 × 732.6975 | 296216.777 | 8.58% |

The source netlists are byte-identical to the archived ones. All three pass
Calibre DRC 0 and strict LVS CORRECT (as does an 8×8 smoke layout);
R+C+CC extraction is complete for 8×8, 32×32 and 128×128, with the 512×512
status in the evidence file.

## Split arrays with a centre periphery

`layout bank --with-controller` (16 rows or more) uses this floorplan by
default. The core splits into lower and upper sub-arrays around the column
periphery. The split keeps whole predecode groups in one half (64-row groups
for three decoder levels, otherwise 8-row slices); only level-0 enables,
address buses and `wl_en` reach both halves. Each bitline is one M5 net from
the lower array through the register, write, sense and precharge bands to the
upper array. SPICE, sizing, timing and pitch are unchanged. Banks below 16
rows have no integrated controller floorplan.

Each decoder half places its own rows, with predecode slices at the group end
farthest from the band. A net used by both halves is bridged over the array
(staggered M5–M8 stacks, an M8 jog over the array, one M10 track across the
band), and its upper trunk gets a diffusion tap. Bank-only builds use straight
M5 tracks.

TIME_CONTROL starts at the decoder's left edge:

- **Block.** A full-width block between the decoder halves, 1 µm above the
  register band, holds the DFFs, transmission gates, NOR gates, tapped
  receivers and as many other gates as fit. The address buffers and `wl_en`
  driver take its bottom rows under the DFFs, so no buffer's internal net
  crosses them; at 512 rows the DFFs go first.
- **Lanes.** The remaining gates use spare predecode rows on the decoder's
  1365 nm rows and M1 rails. A lane ends 500 nm before the row's first slice,
  and gates slide right until every access and tap is 300 nm clear of the
  M5 trunks crossing it. Wells and rails run through each row; a gate's well
  ends at least 200 nm past, or 225 nm before, any decoder well on the shared
  edge. Fingers stay at or below 475 nm; oversized gates split into slices
  stacked on consecutive rows and joined by M4.
- **Antenna taps.** A lane input gets a diffusion tap unless a nearby lane
  gate drives it; long-route inputs (block outputs, `iso_far`, drivers more
  than 15 µm away) reached 2700 times their gate area at 512 rows without
  one. Stacked slices and M4-bar inputs rely on the router check below.
- **Decoder drives.** Each decoder net is driven once, from a tapped M5 pin
  on the block's edge up a planned M5 riser; no route comes within 285 nm of
  the off-grid tap landing. The replica-wordline return drops on M9 beside
  the lanes.
- **Routing.** M4–M9 over the full frame width, with every bank conductor
  and planned wire a square-clearance obstacle. Nets with planned pins route
  first, then shortest first. A blocked net rips up routed nets on its
  cheapest path (never fixed metal or the clock tree); ripped nets cost more
  to cross, and a net avoids ripping the net that ripped it (one 128-row
  build otherwise alternated over a hundred times). A full restart with a new
  priority is the fallback. Afterwards same-net points within two grid steps
  are joined, M7–M9 ends facing across a sub-minimum gap are extended level,
  router cuts keep via spacing from fixed cuts such as the register via7
  lifts, and offset via landings get their neck widened.
- **Router antenna check.** As each layer from M5 to M9 is etched, a gate
  group not yet joined to a driver, tap or planned pin may carry at most 250
  times its gate area (Calibre allows 300). An offending net reroutes with
  M4–M7 steps costing ten times more. It found the one error Calibre reported
  on the first tapped 32×32 build (`wl_en` at M8); Calibre remains the final
  check.
- **Spare column.** The first decoder column widens in 100 nm steps until the
  lanes hold every gate within 75 % of the available rows.

The data registers keep one frozen transmission-gate DFF per column, two per
1410 nm slot above the lower array; a searched track set shrinks the slot from
9750 nm to 7730 nm. NanGate45 `DFF_X1` was studied only as a reference (a
different 28-transistor circuit); a single-row layout of the frozen DFF is
blocked by its M3 columns.

The centre layouts use the corner layouts' source decks.
[`CENTRE_FLOORPLAN_V2_2_8.json`](../../data/CENTRE_FLOORPLAN_V2_2_8.json)
binds each GDS, source deck and Calibre receipt by hash; after the corner
retirement, default-floorplan rebuilds are XOR-identical to the recorded
layouts.

| Square bank | Centre macro (µm) | Centre area (µm²) | Corner area (µm²) | Change | Spare column | Lane rows | Rip-ups | Calibre |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 40.92 × 71.79 | 2937.37 | 3127.29 | −6.07% | 2.7 µm | 24 | 5 | DRC 0, LVS correct, R+C+CC extracted |
| 128 | 110.62 × 202.83 | 22436.27 | 23574.31 | −4.83% | 0.7 µm | 88 | 2 | DRC 0, LVS correct, R+C+CC extracted |
| 512 | 384.82 × 731.39 | 281450.71 | 296216.78 | −4.98% | 1.1 µm | 363 | 6 | DRC 0, LVS correct, R+C+CC extracted |

At 512 rows the decoder trunks leave narrow lane rows (median 4.0 µm), so the
controller spans 363 rows in a 17.0 × 530.6 µm frame (corner frame
36.9 × 90.4 µm), and some buffer stages split across the band (the precharge
buffer's fifth stage has slices about 400 µm apart). The centre metal is not a
calibrated RC baseline. Failed attempts were purged on 2026-10-15
([development notes](../../DEVELOPMENT.md)).

## Driver search

The explicit candidate profile is
[`layout_sizing_lookup.json`](../../../sram_compiler/sizing/layout_sizing_lookup.json);
the default sizing table stays a separate baseline. Candidate 1 rests on the
complete 32×32 bank's verified DRC/LVS and R+C+CC extraction
(`outputs/layout_scaling/baseline_32x32_verify`). The load audit counts
coupling on either DSPF net, separates metal capacitance from gate charge and
solves terminal resistance including parallel paths. Four-section coupons
include BSIM gate and junction loads; larger banks extrapolate that bank's
metal geometrically.

375 coupons cover five sizes, candidate strengths and three corners (TT
1 V/25 °C, SS 0.9 V/125 °C, FF 1 V/−40 °C). The chosen classes meet coupon
targets of 2 ns wordline, 2.5 ns write driver, 3 ns precharge and 0.5 ns
decoder at every corner; these are not clock periods. Metal resistance stays
nominal (no temperature coefficient was measured).

| Square bank | PRE | Write output | Write input | WL inverter | WL NAND | Decoder inverter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.5 | 1.5 | 0.5 | 1 | 1 | 1 |
| 64 | 0.5 | 1.5 | 0.5 | 1 | 1 | 1 |
| 128 | 1 | 1.5 | 0.5 | 2 | 1 | 1 |
| 256 | 2 | 4 | 1 | 4 | 1 | 1 |
| 512 | 8 | 8 | 2 | 16 | 4 | 1 |

Values are scale factors on the YAML unit widths. At 512 the previous classes
were 16/32/8/128/35/9. Controller output buffers follow the resized loads;
clock, register, sense-amplifier and storage transistors were not optimized.
Candidate 2 below replaces these classes.

## Performance pass (candidate 2)

The 2026-10-16 pass re-derived the driver classes and controller delays for
the five square banks, performance first and density second. Evidence is in
[`LAYOUT_PERFORMANCE_V2_2_8.json`](../../data/LAYOUT_PERFORMANCE_V2_2_8.json).
Numbers are SS (0.9 V, 125 °C) coupon results on extracted metal unless
stated: design estimates, not post-layout simulation.

### Driver classes

Candidate 1 met loose absolute targets (2–3 ns), so its drivers were slow
relative to their clock. Candidate 2 takes the smallest class whose
worst-corner edge stays within 10 % of the frozen half-period of its size
class (row class for bitline drivers, column class for wordline and decoder
drivers; 512 columns extrapolate the column ladder by its last step). Bitline
coupons model the centre plan: two half-array branches behind their band
wires, with measured 128×128 metal above 64. Where no class meets the
fraction (the 512-column wordline), the ladder's knee is taken: the last
doubling that still gains 15 %.

| Square bank | WL inverter c1 → c2 | WL edge (ps) | PRE c1 → c2 | restore edge (ps) | write output c1 → c2 | write edge (ps) | half-period (ps) |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| 32 | 1 → 4 | 936 → 265 | 0.5 → 4 | 1526 → 215 | 1.5 → 4 | 701 → 290 | 3600 |
| 64 | 1 → 8 | 1839 → 271 | 0.5 → 4 | 2573 → 358 | 1.5 → 8 | 1156 → 266 | 3800 / 4000 |
| 128 | 2 → 16 | 1769 → 296 | 1 → 4 | 1923 → 543 | 1.5 → 8 | 1726 → 388 | 6600 / 4800 |
| 256 | 4 → 32 | 1809 → 395 | 2 → 4 | 1913 → 1057 | 4 → 8 | 1294 → 726 | 8100 / 5600 |
| 512 | 16 → 64 | 1279 → 799 | 8 → 4 | 1398 → 2186 | 8 → 8 | 1461 → 1461 | 11100 / 6400 |

The wordline NAND is a quarter of the inverter class and the decoder inverter
is 1/1/1/2/2. The row ladder alone would give PRE 4/4/4/8/16 and write output
4/8/8/8/16 (the 512 write output still misses its 1110 ps allocation at the
knee); the table shows the capped values explained next.

### Control bars cap the bitline drivers

Every column hangs its precharge and write-driver gates on one M4 bar per
control net. Solved from the verified 32×32 and 128×128 extractions, the
driving-point resistance to the bar's near end is 90–120 Ω (PRE, s_en),
140 Ω (sa_iso) and 300–360 Ω (w_en), plus 1.2–2.3 Ω/µm along the bar (the
audit's earlier 600 Ω "far gate" figure included a 250–450 Ω stub per slot).
The buffered PRE line reaches 90 % at the far end 170 ps after the controller
edge at 32 columns, 423 ps at 128 and 1.9 ns at 256; w_en needs 0.3, 1.2 and
2.8 ns. Above 128 columns the bars, not the bitline drivers, limit restore
and write completion.

Sweeping far-end line plus bitline edge over the ladder gives the column caps
`pre_cap` 8/8/4/4/4 and `wd_cap` 8, applied as the minimum of row class and
cap; the write input follows the capped output. At 512 columns neither bar is
fast at any class (PRE never reaches 90 % at class 16; w_en needs 7.8 ns at
class 8): that needs repeaters or a wide upper-metal bar, not implemented.

### TIME_CONTROL audit

One delay stage (0.72/2.16 µm driver, four unit loads) takes 8.1 ps at TT,
17.6 ps at SS and 5.5 ps at FF. The fixed four-stage guards add 67 ps at SS
against far-end tails of 42–184 ps at 32 columns, 146–830 ps at 128 and
0.4–2 ns at 256: robustness comes from the far-end observers, not the guards.
The decoder settles in 286/275/408/423/463 ps for 32–512 rows, against a
138 ps eight-stage setup chain. At 32×32 the controller holds 1382 MOS with
615 µm of gate width (registers 51 %, enable buffers 14 %, other logic 10 %,
guards 10 %, setup chain 6 %, replica delay 4 %); at 512×512 the enable
buffers are 46 % of 9393 µm.

The class table now sets these delays per class instead of the fixed
8/4/4/2/4 (setup, precharge-off, isolation, enables-off, replica-wordline-off):

| Class | Setup chain (row class) | Guards (column class) |
| --- | ---: | ---: |
| 32 | 2 | 4 |
| 64 | 2 | 8 |
| 128 | 2 | 16 |
| 256 | 2 | 44 |
| 512 | 2 | 64 (capped) |

Guards are the smallest even count covering 1.25 × the worst read-critical
far-end tail (PRE off or ISO on) at the capped precharge class; the
enables-off guard uses the same count. From 256 columns no practical chain
covers the PRE tail (1.5 ns at 256, 2 ns at 512), so 512 is capped at 64 and
flagged. Two setup stages suffice for every row class: in mini-bank path
coupons the registered request already trails the clock by 540–730 ps, so
`wl_en` rises 1.6–2.7 × after the decoder settles at every corner (margins
112–162 ps FF, 164–299 ps TT, 359–537 ps SS for 32–512 rows with 32 columns;
3.2–6.5 ×, 563 ps FF to 2.4 ns SS, at 256×256 and 512×512). The fixed eight
stages only delayed every read by 108 ps at SS. Tables without these fields
keep the fixed counts, so the default compiler path is unchanged.

### Layouts

Three generator rules came out of the candidate-2 rebuilds:

- **Tie-offs.** The first 128×128 build failed one `Antenna.metal8` check: an
  isolation-guard input tied to `VDD` was routed as part of the supply over a
  65 µm M7 spine, and the router's antenna model skipped supplies. Constant
  inputs now tie to their own gate's rail on M3, and the model checks supply
  nets that carry gates.
- **Via clearance.** 512×512 failed `Via7.2` twice: router via7 cuts landed
  425 nm from a decoder bridge stack's via7 (rule 440 nm). Bridge and
  replica-wordline stacks now block 40 nm beyond their M7/M8 metal and 80 nm
  on M9 (Via9.2, 880 nm).
- **Frame width.** The 11.3 and 11.7 µm frames of 32×32 and 64×64 did not
  route with candidate-2 drivers (1769 rip-ups over seven hours at 32×32),
  although their lane fill matched the larger frames: they ran out of
  vertical tracks. The spare column now keeps every frame at least
  `MIN_FRAME_WIDTH` = 13 µm wide (spare 3.9 µm at 32, 2.9 µm at 64), and both
  route in 7 minutes (frames 13.0 × 45.7 and 13.0 × 72.2 µm). A 15 µm trial
  also routed but costs more width.

| Bank | Macro (µm × µm) | Area (µm²) | vs centre tree 15 | Controller block (µm²) | Spare column | Build | DRC | LVS | Extraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 32×32 | 42.4 × 73.1 | 3,102 | +5.6 % | 336 | 3.9 µm | 7 min, 26 rip-ups | 0 | CORRECT | distributed R+C+CC |
| 64×64 | 64.9 × 118.7 | 7,706 | | 361 | 2.9 µm | 7 min, 10 rip-ups | 0 | CORRECT | distributed R+C+CC |
| 128×128 | 113.8 × 206.1 | 23,454 | +4.5 % | 463 | 0.8 µm | 17 min, 2 rip-ups | 0 | CORRECT | distributed R+C+CC |
| 256×256 | 209.3 × 380.8 | 79,704 | | 610 | 2.1 µm | 37 min, 5 rip-ups | 0 | CORRECT | distributed R+C+CC |
| 512×512 | 396.8 × 730.2 | 289,765 | +3.0 % | 804 | 1.5 µm | 70 min, 0 rip-ups | 0 | CORRECT | distributed R+C+CC |

Rip-ups are the TIME_CONTROL rip-ups numbered in each build log. The macro
grows 3–6 % over the candidate-1 centre layouts (tree 15) for the larger
drivers and, at 32 columns, the wider spare column. 512×512 remains outside
the timing envelope and is built with the 80 ns diagnostic clock.

Native full-mode waveform checks (`dev/layout_scaling/waveforms.py`: read,
write, hold, sense and recovery on illustrative distributed wiring, no
extraction or yield) pass with the candidate-2 table at 9.0 ns for 8×8 at TT
1.0 V 25 °C, SS 0.9 V 125 °C and FF 1.0 V −40 °C, and for 32×32 at TT and SS.
Each run is bound by its result hash in the performance record.

## Macro edge pins and right edge

The 2026-10-21 pass moves the macro pins to the edges and the output latch
inside the bank, so the bank's VSS trunk is the macro's right edge. SPICE,
sizing, timing and pitch are unchanged. Evidence (pin coordinates, receipts,
generator hashes) is in
[`MACRO_PINS_V2_2_8.json`](../../data/MACRO_PINS_V2_2_8.json).

| Ports | Edge | Layer | From |
| --- | --- | --- | --- |
| `clk`, `csb`, `web`, `A*` | left, bottom or top (per layout) | M10 | the input's capture access |
| `OUT`, `OUT_B` | right | M4 | output-latch `Q`/`QB` tracks |
| `VDD`, `VSS` | bottom and top | M6 | supply trunks |
| `DIN*`, `SA_Q*`, `SA_QB*` | internal | M3 | register `D`, sense outputs |

- **Inputs.** Before routing, each input's capture access (its register's
  `D` M8 landing; for `clk` the clock buffer's M4 gate access) continues as a
  via stack to M10. The router sees it only as fixed metal of the input's own
  net, not as a new target; candidate 2 had tried edge pads and landings as
  targets and got 6500 rip-ups over 17 attempts in 17 hours at 32×32
  (`dev/layout_scaling/macro_pins_wip.patch`). Attaching pins only after
  routing fails too, because router M9 lies over the access of 5 of the 8
  inputs at 32×32 and of every input at 128×128 and 512×512. A crowded stack
  slides along an access-layer stub (at most 3 µm), or else stands alone
  within 8 µm as one more router access (`clk` at 256×256). After routing each
  input runs on M10, which the router never uses: straight left if free, else
  straight down or up, else after one jog, keeping 800 nm from other M10. The
  first register column exits left; the other columns exit down from the lower
  register rows and up from the upper ones. The FreePDK45 antenna rule at each metal counts only lower layers, so M10 adds
  nothing to any ratio.
- **Right edge.** The 2.91 µm single-row latch sat past the supply trunks and
  the far-end return stacks reached 465 nm past the VSS trunk, making the
  macro 3.46–3.54 µm wider than the bank. The latch now folds into two
  1365 nm rows (1.67 × 2.73 µm): the cross-coupled pair is mirrored above the
  input gates onto a shared VDD rail and shifted 390 nm, so `INT1`/`INT2` are
  straight M3. It sits in the empty replica strip at register height, 300 nm
  from the last register slot, with its VDD rail on a free register VDD bar.
  `D` and `EN` descend on M5 between the last column's and the replica
  bitlines, the band's only M5-free channel. The return stacks sit 300 nm left
  of the VDD trunk's centre.
- **Data pins.** `DIN*`, `SA_Q*` and `SA_QB*` stay internal. The M5 bitlines
  leave 190–225 nm gaps through every band (a landing needs 420 nm), the
  M3-to-M5 landing row at each array edge leaves 195–230 nm, and the array's
  full-width M3 straps close the M3 gaps below it. Edge pins would need a
  different band layer plan.

All five banks pass Calibre DRC (0), LVS (CORRECT) and distributed R+C+CC
extraction. `OUT`/`OUT_B` are on the right edge and `VDD`/`VSS` on the bottom
and top edges at every size; the inputs fall per register placement:

| Bank | Macro (µm) | Area (µm²) | vs candidate 2 | Left | Bottom | Top | Build (candidate 2) |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 32×32 | 38.95 × 73.12 | 2,848 | −8.2 % | `web`, `A0`, `A2`, `A4` | `A1`, `A3` | `clk`, `csb` | 16 min, 61 rip-ups (7 min, 26) |
| 64×64 | 61.46 × 118.70 | 7,295 | −5.3 % | `csb`, `A0`, `A2`, `A4` | `A1`, `A3` | `clk`, `web`, `A5` | 24 min, 46 rip-ups (7 min, 10) |
| 128×128 | 110.28 × 206.06 | 22,724 | −3.1 % | `A0`, `A3`, `A6` | `A1`, `A2`, `A4`, `A5` | `clk`, `csb`, `web` | 15 min, 2 rip-ups (17 min, 2) |
| 256×256 | 205.81 × 380.78 | 78,367 | −1.7 % | `clk`, `web`, `A0`, `A3`, `A6` | `A1`, `A2`, `A4`, `A5` | `csb`, `A7` | 41 min, 4 rip-ups (37 min, 5) |
| 512×512 | 393.29 × 730.22 | 287,186 | −0.9 % | `web`, `A0`, `A5` | `A1`, `A2`, `A3`, `A4` | `clk`, `csb`, `A6`, `A7`, `A8` | 73 min, 0 rip-ups (70 min, 0) |

The M10 lift stacks slow routing in the narrow 32×32 and 64×64 frames, where
the fixed width saving is also largest in relative terms.

## Evidence and limits

The no-RC compact controller with candidate sizing passes 2546 waveform
checks per corner (7638 in total) on native full-mode 8×8 arrays at a fixed
9 ns period, covering read, write, hold, sense and recovery over eight
cycles. These runs use illustrative distributed wiring, nominal devices and
seed 20260711; they are not post-layout or Monte Carlo yield evidence. Solver
logs and waveforms are under
`outputs/layout_scaling/waveforms/final_8x8_{tt,ss,ff}`.

Layout verification, dimensions, source hashes and image paths are recorded
in [`LAYOUT_SCALING_V2_2_8.json`](../../data/LAYOUT_SCALING_V2_2_8.json).
Failed and superseded attempts were purged on 2026-10-15. The earlier
selected 512×512 GDS carries a 45.7 µm M6 connection from an isolated
clock-bar input branch to its driver diffusion, resolving its recorded M6
antenna violation without changing the source MOS, other layers or placement;
the patch script hash and coordinates are in its manifest.

No layout in this record has extracted waveform, operating-point or yield
qualification; the 512×512 operating point remains unqualified. The V2.2.2
release screen certifies only its own tree.

## Reproduction

Run from the repository root in the `openyield` environment, with `PDK_DIR`
set to a FreePDK45 installation and Calibre on PATH.

```bash
python main_sram.py simulate --rows 512 --cols 512 --operation 'read&write' \
  --variation-mode nominal --real-cell-mode 0 --period 80e-9 --no-run-xyce \
  --sizing-lookup sram_compiler/sizing/layout_sizing_lookup.json \
  --compact-control --no-w-rc --no-column-mux --output-dir outputs/layout/scaling_source
python main_sram.py layout bank --with-controller --deck "$DECK" \
  --output-dir outputs/layout/scaling_512
python main_sram.py layout verify-full --layout-dir outputs/layout/scaling_512 \
  --pdk-dir "$PDK_DIR" --output-dir outputs/layout/scaling_512_verify \
  --extraction-workers 2
python main_sram.py layout preview --layout-dir outputs/layout/scaling_512
python main_sram.py layout driver-audit --layout-dir outputs/layout/scaling_512 \
  --extraction-dir outputs/layout/scaling_512_verify \
  --output-dir outputs/layout/scaling_512_loads
```

`DECK` is the generated run directory's `deck.sp`; output directories must be
new. Use 9 ns for a 32×32 source and 30 ns for 128×128 (explicit diagnostic
periods). The archived corner layouts came from `corner_control.py`
(`git show 545c6c3:sram_compiler/layout/freepdk45/corner_control.py`). The
tracked case manifest is
[`v228_scaling_cases.json`](../../../tests/spice/v228_scaling_cases.json).
