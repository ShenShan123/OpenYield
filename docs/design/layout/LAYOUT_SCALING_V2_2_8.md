# Single-bank layout scaling and driver candidates — V2.2.8

The layout generator now accepts a single 512×512 bank at the existing
705 nm column and 1365 nm row pitch. TIME_CONTROL and the data registers are
included in the generated corner floorplan. An explicit diagnostic clock is
required beyond the automatic 512×256 timing envelope; no new operating point
or release qualification is implied.

## Controller

The current source deletes the wordline-setup resistor/capacitor, the three
observer RC filters, and the tall-decoder settling capacitors. Physical bank
interconnect remains distributed RC. Far-end PRE, isolation, enable and replica
wordline observers remain, together with the existing short digital delay chains.
Legacy tau arguments remain accepted for call compatibility and do not insert
electrical elements or additional stages.

The archived automatic 8×8 RC reference contains the earlier controller.
It is declined when source topology differs, with the reason recorded in
`variation_summary.calibrated_rc`; an explicitly requested stale profile
still raises an error. Native wiring is illustrative in that case.

`compact_control=True` factors isolation completion into two NAND gates,
removes the unused sense-enable complement inverter, and removes obsolete
hold-latch and complement-inverter loads from sizing. The external controller
port contract remains unchanged. Driver snapshots from the previous RC
architecture are rejected instead of silently reused.

Large logic gates can be divided into parallel finger slices without changing
source W/L. The controller router uses distance to the actual routing tree;
the previous bounding-box estimate could spend minutes exploring the interior
of a tall tree. Scarce address accesses route first; congested nets retry with
priority after a bounded search. Full source MOS, finger and total gate-area checks remain.
For large banks, the Python interface connectivity audit excludes the core
interior and retains its boundary and all interface endpoints. Calibre checks
the complete GDS and source.

## Shared decoder and controller floorplan

The default corner plan now fuses the row decoder and TIME_CONTROL logic into
one physical `ROW_DECODER_TIME_CONTROL` module. The controller's lower frame
edge equals the data-register band's lower edge. It fills the broad corner
below the decoder first; excess standard cells continue upward in a narrow
lane alongside the decoder gates. The lane extends the shared decoder block
to the left, and its height must fit within the decoder's upper boundary.
It replaces the previous policy of extending TIME_CONTROL downward.

Both blocks use CMOS standard-cell families, but their existing row heights
are 1365 nm and 1500 nm. A nominal 1 µm separation keeps their wells and supply
rails clear. The overflow lane is nominally 4 µm wide at 32 rows, 5 µm for
33–256 rows and 20 µm above 256 rows. Banks below 32 rows use a 10 µm lane
to compensate for their short decoder. Oversized controller
gates split into parallel slices that conserve each source device's total
W/L. Data registers, source logic, sizing, timing and array pitch are preserved.
This is physical integration; no electrical gate factoring is introduced.
Narrower 512-row lanes failed to route or were superseded; the wider lane
reserves enough tracks for its large buffer groups.

The controller router respects the two placement regions instead of routing
through the decoder. Address and wordline-enable pins stay under their decoder
spines, while replica-wordline feedback reaches the top of the overflow lane.
Controller supply rails follow the changing row width. The manifest records
both regions, their actual area, the shared module, controller placements and
the lower-edge offset. Placement frames define the aligned boundary; individual
cell masks retain their existing insets. The shared module's GDS bounding box
is recorded separately from its floorplan envelope. A rectangular controller bounding box alone is no
longer its occupied frame area. `layout preview` also writes
`decoder_control.png` for the combined module.
A 1 µm band under the corner's address pins keeps their antenna taps clear
of the last logic row, including when the cells continue into the upper lane.

For tall shared frames, the routing heuristic includes the minimum cost of
changing layers to travel against a metal's preferred direction. Equal-cost
searches favor progress toward the existing routing tree. Local searches remain
bounded; long feedback and interface nets can search the complete reserved
region and route before the local groups. Supply access uses the existing
lower trunk instead of a redundant tall route. Retries and physical connectivity
checks are retained. Folded NAND
output metal stays clear of its internal straps, and same-net M4–M6 notches
are closed without changing transistor sizing.
Replica-wordline and far sense-enable receivers use the existing reserved
diffusion-tap filler, like the precharge observer. Their far-end drivers join
through M7/M8, so these taps protect gate-only branches during earlier metal
steps while preserving the feedback path. Connectivity through the final
metal stack alone cannot establish antenna safety.

New generation and verification evidence is recorded separately in
[`DECODER_CONTROL_FUSION_V2_2_8.json`](../../data/DECODER_CONTROL_FUSION_V2_2_8.json).
The earlier scaling evidence remains archived below. Its physical verification
receipts do not qualify the changed placement or its interconnect.

| Bank | Width × height (µm) | Macro area (µm²) | Area reduction from archived layout |
| --- | ---: | ---: | ---: |
| 32×32 | 42.7825 × 73.0975 | 3127.294 | 11.62% |
| 128×128 | 115.4825 × 204.1375 | 23574.309 | 7.27% |
| 512×512 | 404.2825 × 732.6975 | 296216.777 | 8.58% |

All three source netlists are byte-identical to their archived counterparts,
and their controller frame bottoms have zero offset from the data-register
band. The evidence file binds each physical verification status to the exact
GDS and source hashes. All three pass zero Calibre DRC violations and strict
LVS CORRECT; an additional default-source 8×8 smoke layout also passes.
Distributed R+C+CC extraction is complete for 8×8, 32×32 and 128×128. The
512×512 extraction status is recorded separately in the same evidence file.
No new extracted waveform or operating-point qualification is implied.

## Split arrays with a centre periphery

`layout bank --with-controller --controller-floorplan center` (16 rows or more)
splits the core into a lower and an upper sub-array with the column periphery
between them. The split row keeps whole predecode groups in one half: 64-row
groups for three decoder levels, 8-row final slices otherwise. Only level-0
enables, address buses and `wl_en` reach both halves. Every bitline stays one
net: M5 runs from the lower array's top edge through the register, write,
sense and precharge bands to the upper array. Source SPICE, sizing, timing and
the array pitch are unchanged; `corner` remains the default floorplan.

Each decoder half places its own rows. Predecode slices sit at the end of their
group farthest from the band, so rows next to the band have no slices in their
predecode columns. Trunks end at the half's band edge. A net used by both
halves is bridged over the array: staggered M5–M8 stacks on both trunk ends, M8
jogs to a column over the array and one M10 track across the band. The upper
trunk gets a diffusion tap in the strip below it. Bank-only builds bridge the
halves with straight M5 tracks instead.

TIME_CONTROL starts at the decoder's left edge:

- A full-width block between the decoder halves holds the DFFs, transmission
  gates, NOR gates, antenna-tapped receivers and as many other gates as fit.
  Its bottom edge is 1 µm above the data-register band's, leaving room for the
  decoder pins' taps. The address buffers and `wl_en` driver take its bottom
  rows, under the DFF rows, together with any of their stages too wide for a
  lane. Otherwise each address buffer's first stage (a 505 nm P finger at 128
  rows) landed above the DFF rows, and every buffer's internal net crossed
  them. At 512 rows the drivers are too large, and the DFFs go first.
- Remaining gates sit in spare predecode rows above and below the block, on
  the decoder's 1365 nm rows and M1 rails. A lane ends 500 nm before the row's
  first predecode slice. Trunks of other slices still cross it on M5, so lane
  gates slide right until every pin access and tap wire is 300 nm clear of
  them; a pin under a trunk would otherwise meet the trunk with the via above
  its access. Each gate's wells and rails continue to the next gate in its
  row, and a row's wells start at the lane's left end. A gate's well ends at
  least 200 nm past, or 225 nm before, any decoder well along the edge it
  shares with the neighbouring row, so the merged wells keep their minimum
  width and spacing. Lane gates keep fingers at or below 475 nm and split their
  wells mid-gap, so neither diffusion crosses a well edge. Oversized gates
  split into slices that fit the median row's widest clear stretch; slices of
  one gate stack on consecutive rows at one x where possible, joined by M4
  straps.
- A lane input gets a diffusion antenna tap unless a lane gate close in source
  order drives it. Block outputs (DFF `Q` included), bank nets such as
  `iso_far`, and lane drivers placed more than 15 µm away (across the band or
  past a stack) reach it over long routes; at 512 rows such inputs carried up
  to 2700 times their gate area in lower metal. Stacked slices need no tap:
  their combined gate area keeps the ratio low. Inputs on M4 bars (folded
  multi-input gates) have no tap position; the routing antenna check below
  covers them, as it does block gates.
- The controller drives each decoder net once, from a tapped M5 pin on the
  block's lower edge (upper edge for an upper-only net) up a planned M5 riser.
  The tap's M4/M5 landing sits 350 nm from the pin, off the routing grid; no
  route, its own net's included, comes within 285 nm of it, because a wire
  grazing it leaves a neck narrower than the 140 nm minimum width.
  The replica-wordline return drops on M9 beside the lanes into the block's top
  edge, so the bridges' M8 jogs pass under it.
- Routing spans the frame width on M4–M9, over the decoder as well as the
  lanes. Every bank conductor and planned interface wire there is an obstacle
  with a square clearance, because a wire ending beside an obstacle corner
  extends half its width past its last grid point. Nets with planned edge pins
  route first, then the rest shortest first. A net that cannot connect rips up
  the routed nets on its cheapest path through their metal (never fixed metal
  or the clock tree). That path then costs other nets more, and a net costs
  more to cross each time it is ripped. A net that would rip a net which
  ripped it before first looks for a path past other nets, so two nets
  contending for one passage do not take turns (one 128-row build did for
  over a hundred rip-ups). A full restart with a new priority remains the
  fallback. After routing, two points
  of one net that come within two grid steps without touching are joined by
  free metal, and two cuts the drawing merges claim the metal between them;
  router clearance separates only different nets. On M7–M9, wires are as wide
  as their spacing, so two ends of one net can meet within spacing, for
  example where an M6 jog joins two M7 runs. They can also face each other
  across a gap that the closing would bridge over less than the minimum width
  (a 5 nm sliver at 128 rows). One end then extends level with the other, and
  the closing fills the straight gap at full width. A net whose gap cannot be
  closed this way is rerouted with that spot costing it more. Router cuts
  also keep via spacing from cuts drawn outside the router, such as the
  register pins' via7 lifts. Same-net closing on M7–M10 bridges two via
  landings across their overlap, which is narrower than the minimum width
  when the landings are offset. If the usual repair box would come within
  spacing of another wire, only the neck itself is widened.
- The router then repeats Calibre's cumulative antenna check on its own
  metal. As each layer from M5 to M9 is etched, a group of gate inputs not
  yet joined to a driver, tap or planned pin may carry at most 250 times
  its gate area in lower metal; Calibre allows 300, and the margin covers
  cell-internal metal. A net that exceeds this reroutes with M4–M7 steps
  costing ten times more, so its long runs move to M8/M9. Those count only
  once the whole net, driver included, is joined. On the first 32×32 build
  with the new taps, the check found the one error Calibre reported
  (`wl_en` at M8). Planned pins are assumed to reach a diffusion in the
  bank; Calibre remains the final check.
- The first decoder column widens in 100 nm steps until the lanes hold every
  gate within 75% of the available rows and no unsliced decoder driver leaves
  the block. The left edges stay aligned.

The data registers keep one frozen transmission-gate DFF per column, two per
1410 nm slot, directly above the lower array. A searched track set with the
fewest legal gate and strip levels reduces the slot from 9750 nm to 7730 nm.
D inputs are internal band pins; Q keeps the write slot's buffer-input track.
NanGate45 `DFF_X1` (3.23 × 1.4 µm) was studied as a layout reference only: it
is a different 28-transistor low-Vt circuit. A single-row layout of the frozen
DFF in that style is blocked here by M3 columns: seven contact columns cannot
hold the P–N joins, three cross-row tracks and the lower register's Q exit.

The centre layouts use the corner layouts' source decks, so SPICE, sizing and
timing are the same. [`CENTRE_FLOORPLAN_V2_2_8.json`](../../data/CENTRE_FLOORPLAN_V2_2_8.json)
binds each GDS, source deck and Calibre receipt by hash. All three come from
one generator tree. On that tree, a corner 32×32 rebuild is XOR-identical to
its recorded layout.

| Square bank | Centre macro (µm) | Centre area (µm²) | Corner area (µm²) | Change | Spare column | Lane rows | Rip-ups | Calibre |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 40.92 × 71.79 | 2937.37 | 3127.29 | −6.07% | 2.7 µm | 24 | 5 | DRC 0, LVS correct, R+C+CC extracted |
| 128 | 110.62 × 202.83 | 22436.27 | 23574.31 | −4.83% | 0.7 µm | 88 | 2 | DRC 0, LVS correct, R+C+CC extracted |
| 512 | 384.82 × 731.39 | 281450.71 | 296216.78 | −4.98% | 1.1 µm | 363 | 6 | DRC 0, LVS correct, R+C+CC extracted |

At 512 rows the decoder trunks leave only a narrow strip of each spare row
(the median lane row is 4.0 µm wide). The controller therefore spreads over
363 lane rows in a 17.0 × 530.6 µm frame (the corner frame was 36.9 × 90.4
µm). Some buffer stages are split across the band: the precharge buffer's
fifth stage has slices on both sides, about 400 µm apart. A wider spare column
would shorten the frame but give back most of the area saving. The centre
metal is not a calibrated RC baseline, and no waveform, operating-point or
yield qualification has been run on these layouts. Failed and superseded
attempts were purged on 2026-10-15; see the [development
notes](../../DEVELOPMENT.md).

## Driver search

The explicit candidate profile is
[`layout_sizing_lookup.json`](../../../sram_compiler/sizing/layout_sizing_lookup.json).
The default sizing table remains a separate baseline. The candidate is based on
the complete 32×32 bank's hash-bound, passing Calibre DRC/LVS and distributed
R+C+CC extraction under `outputs/layout_scaling/baseline_32x32_verify`.

The load audit counts coupling incident on either DSPF net, separates metal
capacitance from model-dependent gate charge, and solves terminal resistance
including parallel paths. Four-section coupons include actual BSIM gate and
junction loads. Larger-bank metal loads are geometric extrapolations of that
measured bank, with the extracted endpoint contact resistance retained.

375 coupons cover five sizes, candidate strengths, and TT/SS/FF corners:
TT at 1 V/25 °C, SS at 0.9 V/125 °C, and FF at 1 V/−40 °C. The chosen classes
meet the 2 ns wordline, 2.5 ns write-driver, 3 ns precharge and 0.5 ns decoder
coupon targets at all three corners. These targets are not SRAM clock periods.
MOS models follow the stated corner and temperature; extracted metal resistance
remains at its nominal value because no metal temperature coefficient was measured.

| Square bank | PRE | Write output | Write input | WL inverter | WL NAND | Decoder inverter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.5 | 1.5 | 0.5 | 1 | 1 | 1 |
| 64 | 0.5 | 1.5 | 0.5 | 1 | 1 | 1 |
| 128 | 1 | 1.5 | 0.5 | 2 | 1 | 1 |
| 256 | 2 | 4 | 1 | 4 | 1 | 1 |
| 512 | 8 | 8 | 2 | 16 | 4 | 1 |

These are scale factors on the YAML unit widths, not widths in micrometres.
At 512, the previous PRE/write-output/write-input/WL-inverter/WL-NAND/decoder
classes were 16/32/8/128/35/9. Controller output buffers are recomputed from
the actual resized device loads. Clock, register, sense-amplifier and storage
transistors have not been independently optimized by this coupon search.

## Evidence and limits

The final no-RC compact controller with candidate sizing passes 2546 waveform
checks per corner on native full-mode 8×8 arrays at a fixed 9 ns period
(7638 checks total). Checks cover read, write, hold, sense and recovery over
eight cycles. These runs use illustrative distributed wiring, nominal devices
and seed 20260711; they are not post-layout simulations or Monte Carlo yield
evidence. The retained solver logs and waveform files are under
`outputs/layout_scaling/waveforms/final_8x8_{tt,ss,ff}`.

Layout verification, dimensions, source hashes and image paths are recorded
in [`LAYOUT_SCALING_V2_2_8.json`](../../data/LAYOUT_SCALING_V2_2_8.json).
Only the recorded layouts, their verification and audit directories, and the
source decks remain under `outputs/layout_scaling`. Failed and superseded
generation and verification attempts were purged on 2026-10-15.
The 512×512 operating point and full extracted read/write/recovery validation
remain unqualified. The previous V2.2.2 release screen certifies its own tree.

The earlier selected 512×512 GDS includes a 45.7 µm M6 connection from an isolated
clock-bar input branch to its existing driver diffusion. This resolves the
recorded M6 antenna violation while preserving the source MOS, all other
layers and the original controller placement. The patch script hash and path
coordinates are recorded in the layout manifest.

## Reproduction

Run from the repository root in the `openyield` environment. Set `PDK_DIR`
to a FreePDK45 installation and put Calibre on PATH.

```bash
python main_sram.py simulate --rows 512 --cols 512 --operation 'read&write' \
  --variation-mode nominal --real-cell-mode 0 --period 80e-9 --no-run-xyce \
  --sizing-lookup sram_compiler/sizing/layout_sizing_lookup.json \
  --compact-control --no-w-rc --output-dir outputs/layout/scaling_source
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

`DECK` is the generated run directory's `deck.sp`. Output directories must be
new. Use 9 ns for the 32×32 layout source and 30 ns for 128×128; these are
explicit diagnostic periods. The tracked case manifest is
[`v228_scaling_cases.json`](../../../tests/spice/v228_scaling_cases.json).
