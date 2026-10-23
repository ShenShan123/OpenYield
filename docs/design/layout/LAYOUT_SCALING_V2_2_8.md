# Single-bank layout scaling and driver candidates — V2.2.8

The layout generator now accepts a single 512×512 bank at the existing
705 nm column and 1365 nm row pitch. TIME_CONTROL and the data registers are
included in the generated floorplan. An explicit diagnostic clock is
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

The corner floorplan described here was retired on 2026-10-16; the
[split-array centre floorplan](#split-arrays-with-a-centre-periphery) replaces
it. This section and its evidence remain the area baseline for that change.

The corner plan fused the row decoder and TIME_CONTROL logic into
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

`layout bank --with-controller` (16 rows or more) uses this floorplan by
default (`--controller-floorplan center`). It splits the core into a lower and
an upper sub-array with the column periphery between them. The split row keeps whole predecode groups in one half: 64-row
groups for three decoder levels, 8-row final slices otherwise. Only level-0
enables, address buses and `wl_en` reach both halves. Every bitline stays one
net: M5 runs from the lower array's top edge through the register, write,
sense and precharge bands to the upper array. Source SPICE, sizing, timing and
the array pitch are unchanged. Banks below 16 rows cannot split and have no
integrated controller floorplan since the corner plan was retired.

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
its recorded layout. After the corner plan was retired, default-floorplan
rebuilds of all three centre layouts are XOR-identical to the recorded ones,
with equal manifests apart from the changed generator hashes.

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

## Performance pass (candidate 2)

The 2026-10-16 pass re-derived the driver classes, the controller's fixed
delays and the macro's pin positions for the five square banks, performance
first and density second. Its evidence is bound in
[`LAYOUT_PERFORMANCE_V2_2_8.json`](../../data/LAYOUT_PERFORMANCE_V2_2_8.json);
every number below is an SS (0.9 V, 125 °C) coupon result on extracted metal
unless stated otherwise. Coupons are design estimates, not post-layout
simulation.

### Driver classes

Candidate 1 picked the smallest class that met loose absolute targets
(2–3 ns), so its drivers were slow relative to the clock they serve.
Candidate 2 picks the smallest class whose worst-corner edge stays within 10 %
of the frozen half-period of its size class (row class for bitline drivers,
column class for wordline and decoder drivers; 512 columns extrapolate the
column ladder by its last step because they lie outside the timing envelope).
The bitline coupons now model the centre plan: the precharge and write drivers
sit in the band and drive two half-array branches, each behind its band wire,
instead of one full-height bitline, and sizes above 64 extrapolate the
measured 128×128 metal instead of the 32×32 bank. The decoder inverter faces
the selected wordline NAND class. Where no class meets the fraction (the
512-column wordline), the ladder's knee is taken: the last doubling that still
gains 15 %.

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
knee). The precharge and write-driver classes in the table are the capped
values explained next.

### Control bars cap the bitline drivers

Every column hangs its precharge and write-driver gates on one M4 control
bar, so the gate load of a wide array is set by the bitline class. The
driving-point resistance of each control net was solved in the verified
32×32 and 128×128 extractions from all driver fingers together: to the bar's
near end it is about 90–120 Ω (PRE, s_en), 140 Ω (sa_iso) and 300–360 Ω
(w_en), with bar gradients of 1.2–2.3 Ω/µm. The audit's earlier "far gate
resistance" of about 600 Ω includes a 250–450 Ω stub inside each slot, which
sits in series with one gate only. With these measured series terms the
buffered PRE line reaches 90 % at the far end 170 ps after the controller
edge at 32 columns, 423 ps at 128 and 1.9 ns at 256 (precharge class 8),
while w_en needs 0.3, 1.2 and 2.8 ns. Above 128 columns the bars, not the
bitline drivers, limit restore and write completion.

The joint metric (far-end line at 90 % plus the bitline edge) was swept over
the class ladder for each square size. It is minimal at precharge 8 for 32
and 64 columns and at precharge 4 from 128 columns up, and at write output 8
for 32–64 columns with 4–8 equivalent above; a stronger class only lengthens
the line. The class table therefore carries column caps, `pre_cap`
8/8/4/4/4 and `wd_cap` 8, applied as the minimum of the row class and the
cap, and the write input stage follows the capped output stage. At 512
columns the PRE bar never reaches 90 % within the coupon window at precharge
16 and w_en needs 7.8 ns at write output 8: the bars need repeaters or a wide
upper-metal bar before 512 columns can be fast, which this pass does not
implement.

### TIME_CONTROL audit

Measured on the compiler's own cells: one delay stage (0.72/2.16 µm driver
with four unit loads) takes 8.1 ps at TT, 17.6 ps at SS and 5.5 ps at FF.
The fixed four-stage guards therefore add 67 ps at SS, while the far-end
tails they were meant to cover (50 % to 90 % or 10 %) are 42–184 ps at 32
columns, 146–830 ps at 128 and 0.4–2 ns at 256: the guards contribute almost
nothing, and robustness comes from the far-end observers themselves. The
decoder settle (address buffer, half trunk with predecode loads, real 3-to-8
blocks of every level) is 286/275/408/423/463 ps for 32–512 rows, against a
138 ps eight-stage setup chain.

At 32×32 the controller holds 1382 MOS with 615 µm of gate width: registers
51 %, enable buffers 14 %, other logic 10 %, guard delays 10 %, setup chain 6 %,
replica delay 4 %. At 512×512 the enable buffers are 46 % of 9393 µm.

The class table now sets these delays per class instead of the fixed
8/4/4/2/4 (setup, precharge-off, isolation, enables-off, replica-wordline-off):

| Class | Setup chain (row class) | Guards (column class) |
| --- | ---: | ---: |
| 32 | 2 | 4 |
| 64 | 2 | 8 |
| 128 | 2 | 16 |
| 256 | 2 | 44 |
| 512 | 2 | 64 (capped) |

Guards: the smallest even count whose chain covers 1.25 × the worst
read-critical far-end tail (PRE off or ISO on) at the capped precharge class;
the enables-off guard follows the same count. From 256 columns the digital
guard can no longer cover the PRE tail (1.5 ns at 256, 2 ns at 512 columns)
without more than a hundred stages; 512 is capped at 64 and flagged. Setup
chain: two stages for every row class. In the mini-bank path coupons (the
real TIME_CONTROL driving extracted-metal line models and the real decoder)
the registered access request already follows the clock by 540–730 ps, so
with the minimum chain `wl_en` rises 1.6–2.7 × later than the decoder
settles at every corner: margins of 112–162 ps at FF, 164–299 ps at TT and
359–537 ps at SS for 32–512 rows with 32 columns, and 3.2–6.5 × (563 ps FF to
2.4 ns SS) at 256×256 and 512×512, whose later precharge release delays the
access request further. The fixed eight stages
only delayed every read by 108 ps at SS. Smaller arrays therefore lose most
of their fixed delay; larger arrays gain settling margin where a digital
chain can provide it. Tables without the fields keep the fixed counts, so the default compiler
path is unchanged.

### Macro pin positions

Implemented on 2026-10-21 with a different access scheme; see
[Macro edge pins and right edge](#macro-edge-pins-and-right-edge).
The centre layout labels every port at an interior point: the controller's
gate access for `clk`, `csb`, `web` and `A*` on M4, the register D tracks for
`DIN*`, the sense tracks for `SA_Q*`/`SA_QB*`, the latch for `OUT`/`OUT_B` and
the trunks for the supplies. Candidate 2 examined moving them to the
boundary; the allocation itself is deferred:

- `clk`, `csb`, `web`, `A*` (left side): every variant that gave the router
  a planned access for these nets (M6 pads on the block's left edge, M8 pads
  at the router border, M8 or M9 landings beside the capture gates) set off
  rip-up cascades, 6500 rip-ups over 17 attempts in 17 hours at 32×32
  against 0 rip-ups in 3 attempts for the baseline block. The inputs are
  captured on the block's dense register side, and a reserved access there
  blocks the enable and supply nets. The work is kept as
  `dev/layout_scaling/macro_pins_wip.patch`; a workable scheme attaches the
  pins after routing, on M10 which the router never uses, from a free via
  column found in the routed geometry.
- `OUT`, `OUT_B`: the latch's M4 tracks can be extended to the right side;
  `VDD`, `VSS`: the M6 trunks to the top and bottom sides. Both are
  post-route additions and stay out of this pass so the candidate-2 builds
  keep the verified baseline router behaviour.
- `DIN*`, `SA_Q*`, `SA_QB*`: internal in any scheme. Every periphery band
  carries full-width M4 bars (supplies, enables, the precharge bar), M5 is
  the bitline layer through the band and the arrays, and M3 is the cell
  bitline layer, so no 705 nm column offers a via column from the band to
  an edge; the M4 landing row at each array edge leaves 195–230 nm gaps
  where a 140 nm wire needs 420 nm. Per-column edge pins would need a
  different band layer plan.

### Layouts

The first candidate-2 build of 128×128 failed Calibre with one `Antenna.metal8`
violation (0 DRC violations in every other check). The gate at the end of a
65 µm M7 spine was an input of the isolation guard tied to `VDD` in the
netlist: the router carried it as part of the supply net over M7, M8 and an
M9 hop, and the router's own antenna model skipped the supplies as
diffusion-rich. At the M8 etch step that branch has a gate and no diffusion.
The fix is in the controller builder: a constant input now ties to its own
gate's rail on M3 (via stack onto the lane's M1 rail or the row's M2 rail)
and never joins the routed supply, and the antenna model checks supply nets
whenever a gate hangs on them.

Two more generator rules came out of the rebuilds. The 512×512 layout failed
`Via7.2` twice: router via7 cuts 425 nm from the fixed via7 of a decoder
bridge stack, where the metal clearance the router keeps (400 nm) is 40 nm
short of the cut rule (440 nm). The bridge and replica-wordline stacks now
block 40 nm more than their M7/M8 metal and 80 nm more on M9 (Via9.2, 880 nm).
The 32×32 and 64×64 frames did not route at all with candidate-2 drivers:
1769 rip-ups over five attempts in seven hours at 32×32, against none for
candidate 1, although their lane fill (0.67 and 0.71 of the available rows)
matched the 128–512 frames (0.65–0.72) that converge in minutes. What
differs is width: 11.3 and 11.7 µm against 16.7 µm and more, so the narrow
frames run out of vertical tracks. The spare decoder column now also makes
the frame at least `MIN_FRAME_WIDTH` = 13 µm wide (spare 3.9 µm at 32, 2.9 µm
at 64; 128–512 unchanged). Both then route in 7 minutes (28 and 13 rip-ups,
frames 13.0 × 45.7 µm and 13.0 × 72.2 µm); a 15 µm trial routed with 9 and 8
rip-ups and was not kept, since the narrower frame suffices and costs less
macro width.

All five candidate-2 banks, built from the tracked generator with the three
rules above, verified with the FreePDK45 Calibre decks (DRC, LVS, xRC
extraction; receipts and hashes in `docs/data/LAYOUT_PERFORMANCE_V2_2_8.json`):

| Bank | Macro (µm × µm) | Area (µm²) | vs centre tree 15 | Controller block (µm²) | Spare column | Build | DRC | LVS | Extraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 32×32 | 42.4 × 73.1 | 3,102 | +5.6 % | 336 | 3.9 µm | 7 min, 28 rip-ups | 0 | CORRECT | distributed R+C+CC |
| 64×64 | 64.9 × 118.7 | 7,706 | | 361 | 2.9 µm | 7 min, 13 rip-ups | 0 | CORRECT | distributed R+C+CC |
| 128×128 | 113.8 × 206.1 | 23,454 | +4.5 % | 463 | 0.8 µm | 17 min, 8 rip-ups | 0 | CORRECT | distributed R+C+CC |
| 256×256 | 209.3 × 380.8 | 79,704 | | 610 | 2.1 µm | 37 min, 15 rip-ups | 0 | CORRECT | distributed R+C+CC |
| 512×512 | 396.8 × 730.2 | 289,765 | +3.0 % | 804 | 1.5 µm | 70 min, 14 rip-ups | 0 | CORRECT | distributed R+C+CC |

The macro grows 3–6 % against the candidate-1 centre layouts (tree 15) for
the larger drivers and, at 32 columns, the wider spare column; the controller
block itself is 336–804 µm². Density was the second objective and this is
its cost. 512×512 remains outside the timing envelope and is built with the
80 ns diagnostic clock.

Native full-mode waveform checks (`dev/layout_scaling/waveforms.py`, read, write,
hold, sense and recovery on the compiler's illustrative distributed wiring, no
post-layout extraction or yield) with the candidate-2 table:

| Case | Corner | Period | Result |
| --- | --- | ---: | --- |
| 8×8 | TT 1.0 V 25 °C | 9.0 ns | pass |
| 8×8 | SS 0.9 V 125 °C | 9.0 ns | pass |
| 8×8 | FF 1.0 V −40 °C | 9.0 ns | pass |
| 32×32 | TT 1.0 V 25 °C | 9.0 ns | pass |
| 32×32 | SS 0.9 V 125 °C | 9.0 ns | pass |

Each run is bound by its result hash in `docs/data/LAYOUT_PERFORMANCE_V2_2_8.json`.

## Macro edge pins and right edge

The 2026-10-21 pass puts the centre macro's pins on all four edges instead of
labelling them inside, and moves the output latch inside the bank so that the
bank's VSS trunk is the macro's right edge. Source SPICE, sizing, timing and
the array pitch are unchanged. The evidence is bound in
[`MACRO_PINS_V2_2_8.json`](../../data/MACRO_PINS_V2_2_8.json).

| Ports | Edge | Layer | Connection |
| --- | --- | --- | --- |
| `clk`, `csb`, `web`, `A*` | left, bottom or top (per layout, table below) | M10 | from the input's capture access |
| `OUT`, `OUT_B` | right | M4 | output-latch `Q`/`QB` tracks extended |
| `VDD`, `VSS` | bottom and top | M6 | supply trunks extended |
| `DIN*`, `SA_Q*`, `SA_QB*` | internal, unchanged | M3 | register `D`, sense outputs |

### Inputs

Before routing, each input's capture access continues as a via stack to M10:
the M8 landing of its register's `D` pin, or for `clk` the M4 access of the
clock buffer's gate. The router sees the stack only as fixed metal of the
input's own net. The candidate-2 variants instead gave the router edge pads or
landings beside the gates as new targets and set off rip-up cascades; here an
input needs no extra connection. Attaching pins only after routing does not
work either: on the candidate-2 layouts, router M9 lies within spacing above
the access of 5 of the 8 inputs at 32×32 and of every input at 128×128 and
512×512.

A stack that would come within spacing of another stack, an access or a bank
conductor slides along a stub on its access layer, at most 3 µm. If no stub is
clear, the stack stands alone at the nearest clear point within 8 µm and the
router joins it as one more access of the net. At 256×256 the clock buffer's
access is boxed in on M4 and beside a register's `D` landing, and `clk` takes
this path.

After routing, each input runs on M10, which the router never uses: straight
to the left edge if that run is free, otherwise straight down or up (first
toward the half of the inputs it belongs to), otherwise after one horizontal
jog to the nearest free column. Runs keep 800 nm from the decoder bridges, the
other stacks and the runs already drawn. The register rows decide the outcome:
the first register column reaches the left edge, and the other columns leave
downward from the lower rows and upward from the upper rows. The FreePDK45
antenna rule at each metal step counts only the layers below that metal, so the
M10 runs add nothing to any ratio.

### Right edge

The latch was a 2.91 µm single row placed past the supply trunks, and the
far-end return stacks reached 465 nm past the VSS trunk. Together they made the
macro 3.46–3.54 µm wider than the bank.

- The latch folds into two 1365 nm rows (1.67 × 2.73 µm). The inverter and the
  input NANDs form the lower row. The cross-coupled pair is mirrored above onto a
  shared VDD rail and shifted 390 nm, so each pair input sits over the gate that
  drives it and `INT1`/`INT2` are straight M3 lines. No M4 track comes within
  495 nm of the row boundary, so a supply bar can cross it there.
- It sits in the replica strip at register-band height, where neither the
  register band nor the write band places devices. It is 300 nm from the last
  register slot (pwell–nwell spacing is 225 nm), with its row boundary on the
  highest register VDD bar whose window holds no other M4. The bar supplies the
  shared rail through an M1–M4 stack, and both VSS rails stack into the VSS
  trunk. `D` (the last column's sense output) and `EN` (from the `s_en` bar)
  descend on M5 between the last data column's bitlines and the replica
  bitlines, the only M5-free channel in the band.
- The return stacks sit 300 nm left of the VDD trunk's centre, between the
  replica bitline and the trunk, with 140 nm M5/M6 spacing to both.

### Data pins

`DIN*`, `SA_Q*` and `SA_QB*` stay internal. On the 32×32 layout, two M5
bitlines per 705 nm column leave 190–225 nm gaps through every band, where an
M5 landing between them needs 420 nm. At each array edge the bitlines change
from M3 to M5 through an M4 landing row with 195–230 nm gaps. Below that row
the array's full-width M3 straps close the 265–300 nm M3 gaps between the
bitline stubs. No column has a via path to an edge; per-column edge pins need a
different band layer plan.

### Layouts

All five banks were built from one generator tree (hashes in
`dev/layout_pins/final_tree.sha256`) and verified with the FreePDK45 Calibre
decks (receipts, pin coordinates and hashes in
`docs/data/MACRO_PINS_V2_2_8.json`). OUT/OUT_B are on the right edge and
VDD/VSS on the bottom and top edges at every size:

| Bank | Macro (µm) | Area (µm²) | vs candidate 2 | Left | Bottom | Top | Build (candidate 2) | DRC | LVS | Extraction |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| 32×32 | 38.95 × 73.12 | 2,848 | -8.2 % | `web`, `A0`, `A2`, `A4` | `A1`, `A3` | `clk`, `csb` | 16 min, 61 rip-ups (7 min, 26) | 0 | CORRECT | distributed R+C+CC |
| 64×64 | 61.46 × 118.70 | 7,295 | -5.3 % | `csb`, `A0`, `A2`, `A4` | `A1`, `A3` | `clk`, `web`, `A5` | 24 min, 46 rip-ups (7 min, 10) | 0 | CORRECT | distributed R+C+CC |
| 128×128 | 110.28 × 206.06 | 22,724 | -3.1 % | `A0`, `A3`, `A6` | `A1`, `A2`, `A4`, `A5` | `clk`, `csb`, `web` | 15 min, 2 rip-ups (17 min, 2) | 0 | CORRECT | distributed R+C+CC |
| 256×256 | 205.81 × 380.78 | 78,367 | -1.7 % | `clk`, `web`, `A0`, `A3`, `A6` | `A1`, `A2`, `A4`, `A5` | `csb`, `A7` | 41 min, 4 rip-ups (37 min, 5) | 0 | CORRECT | distributed R+C+CC |
| 512×512 | 393.29 × 730.22 | 287,186 | -0.9 % | `web`, `A0`, `A5` | `A1`, `A2`, `A3`, `A4` | `clk`, `csb`, `A6`, `A7`, `A8` | 73 min, 0 rip-ups (70 min, 0) | 0 | CORRECT | distributed R+C+CC |

Rip-ups here are the TIME_CONTROL rip-ups numbered in each build log, counted
the same way for both builds. The 28/13/8/15/14 in the candidate-2 table above
count every log line that mentions a rip-up, including per-block summary
lines. At 256×256 the `clk` lift found no clear stack within 3 µm of its
access and was placed as a standalone stack the router reaches (`escape` mode
`routed`). The escape pads cost routing at the narrow frames: 32×32 and 64×64
take 16 and 24 minutes against 7. Widths fall 3.46–3.54 µm at every size, so
the saving is largest in relative terms at 32×32.

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

`layout bank --with-controller` now builds the centre floorplan. The archived
corner layouts came from `corner_control.py`, retrievable with
`git show 545c6c3:sram_compiler/layout/freepdk45/corner_control.py`.

`DECK` is the generated run directory's `deck.sp`. Output directories must be
new. Use 9 ns for the 32×32 layout source and 30 ns for 128×128; these are
explicit diagnostic periods. The tracked case manifest is
[`v228_scaling_cases.json`](../../../tests/spice/v228_scaling_cases.json).
