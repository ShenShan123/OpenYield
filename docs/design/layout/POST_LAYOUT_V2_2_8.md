# Post-layout access verification — V2.2.8

Post-layout transients of the complete compact macros at the worst global
PVT corners, without local mismatch. They replaced every generated device
and wire of the compiler deck with the Calibre R+C+CC extraction of the
DRC/LVS-clean layout and scored the waveforms with the compiler's phased
access checks. The first pass found a missed read deadline and supply
bounce on quiet control lines. This record documents the fixes: a shorter
replica delay chain, a 12 ns 64-row clock in the layout timing table, an
upper-metal supply grid with right-side controller exits and extra supply
pins, controller rows that keep each enable driver chain apart, and
low-resistance decoder band rails. It also records an open failure: the
bitcell VSS rails are fed from one end of each half-array.

Status (layout g14): 32×32 and 64×64 pass all ten worst-PVT cases with the
scorer's limits unchanged (worst quiet line 0.096 V of 0.110 V). 128×128
passes nine of ten; at FF 1.1 V −40 °C the `seq` read flips far-end cells
of the read row because their ground rises about 0.35 V (open; the fix is
left for the next round by decision of 2026-10-31). 256×256 and 512×512
carry physical verification only. Evidence:
[`POST_LAYOUT_V2_2_8.json`](../../data/POST_LAYOUT_V2_2_8.json).

## Method

**Netlist.** The compiler builds the phased read&write deck for the case's
size, corner, VDD and temperature (`tests/spice/phased_access.py`). Every
schematic MOS must exist once in the extraction with its model and total W,
and the deck's device list (names, nets, W/L, NF) must equal the layout's
LVS source deck. All generated hardware is removed; stimulus sources, the
start-up clamp, the analysis and the print list stay. The DSPF subcircuit
replaces the hardware. Each probe the scorer reads becomes a 1 Ω alias of
one extracted device terminal: the same device and terminal for nodes with
devices, the nearest device through schematic wires for wire nodes, and, for
`*_far` probes, the load terminal of the same role with the largest
driving-point resistance from its peripheral driver.

**Initial state.** The schematic deck's own `.IC` operating point is solved
first; every extracted node is then held at its schematic net's value for
the DC solution and released for the transient. A `.NODESET`-seeded, unheld
operating point converged to the metastable state of every cell
(Q = QB = 0.40 V at 32×32 SS) and inverted the data registers; it is not
used. The released state holds every printed `.IC` node within 0.7 mV at
0.9 ns, before the first clock edge.

**Network model.** Calibre DSPF, reduced without changing physics that the
checks can see:

| Option | Default | Effect |
| --- | --- | --- |
| degree-two series merge | ≤ 20 Ω | existing `reduce_rc.reduce`; terminals kept |
| short resistors | < 1 Ω | node merge; ≤ 1 Ω series error |
| ground coupling | < 10 aF | Miller factor 1 on both nets; total C kept |
| supply | extracted mesh | `--ideal-supply` merges VDD/VSS meshes into the ports |
| frozen cells (≥ 64 rows) | off | cells outside the probed rows and the edge column pairs keep both access transistors; their storage nets join the background rails |

A 32×32 SS comparison with coupling kept down to 1 aF changed the read
margin by 23 ps (220 → 197 ps). The DSPF has one port node per supply net.
Through g10, transients fed each supply at that node; from g11 on, the
extracted node nearest every macro supply pin of the net joins the port
(`--tie-supply-pins`; all ten pins within 1.9 µm of a node at 32×32). For
the two g10 pins this changes the supply resistances by less than 2 Ω.

**Extraction rules.** The FreePDK45 `calibrexRC.rul` defines no resistance
temperature coefficients, so one extraction serves every corner. Its
`RESISTANCE CONNECTION` values (metal1–poly 0.37752, metal1–metal2 and
metal2–metal3 0.43264) are applied per unit cut area: a 70 nm via1/via2 is
88.3 Ω, a 65 nm via1 102.4 Ω and a 65 nm poly contact 89.4 Ω; via3–via9
have no resistance. A 32×32 macro has about 11,000 such resistors. The kit
deck is the signoff model (decision of 2026-10-29), and every pass/fail
result uses it unchanged. A sensitivity run with these cuts
scaled to about 3 Ω (`--via-scale 0.034`) moved the 32×32 SS read margin
from 220 to 366 ps; the post-layout slowdown is mainly capacitive.

**Clock.** The layout timing table
`sram_compiler/sizing/layout_timing_lookup.json` (6T column-mux variant):
9.5 ns at 32×32, 12.0 ns at 64×64, 21.25 ns at 128×128. It is the shared
`timing_lookup.json` except the 64-row class, which this verification
raised (below).

**Corners.** The per-mechanism worst global corners of the V2.2.4 screen:
SS 0.9 V 125 °C (timing), SF 0.9 V −40 °C (write), FS 0.9 V 125 °C (read
stability), FF 1.1 V −40 °C (races), FF 1.1 V 125 °C (leakage). Nominal
devices; no local mismatch.

**Patterns.** `seq`: the compiler's W1/R1/W0/R0 ×2 on the last row and
column (mux input 1). `mixed`: checkerboards written to the first and last
rows, both rows read on both mux inputs, rewritten and re-read; every
address bit toggles and DIN carries the opposite value during reads. The
target stays in the last column pair because the output latch observes it.
Arrays up to 1024 cells probe every cell; larger arrays probe the quarter,
addressed, neighbour and last rows (the scorer's `probed_rows`).

**Pass criteria.** `tests/spice/phased_waveforms.py` unchanged: exclusive
enables at 0.1/0.9 VDD, one WL pulse on the addressed row and none
elsewhere, retained data within 0.1 VDD, restored bitlines within
0.02 VDD, read data on its rail by 0.68 T, and non-negative phase margins.

## First pass: findings

The first pass used the existing layouts (`macro_density/f3_macroN`,
replica N = 9).

| Run (original layout) | Supply model | Result |
| --- | --- | --- |
| 32×32, 5 corners × 2 patterns | ideal VDD/VSS | 10/10 pass, 228,040 checks; SS read margin 220 ps (schematic 2,312 ps) |
| 32×32 SS `seq` | extracted mesh | **fail**: 825 checks; read data 206 ps late |
| 64×64, 5 corners × 2 patterns | ideal VDD/VSS | 8/10 pass; **SS fails both patterns**, read data 100 ps late |
| 128×128 SS `seq` (frozen cells, 4 accesses) | ideal VDD/VSS | pass, 16,697 checks; read margin 3.17 ns |

1. **Read deadline at 64×64 SS.** Every self-timed control stage runs about
   twice as slow as in the schematic decks that set the clock budgets
   (CLK→OUT 4.19 ns post-layout against 2.15 ns at 32×32 SS). Cut resistance
   is not the main cause: with the kit's cuts scaled to about 3 Ω the read
   moves only 146 ps earlier. The 64-row clock class has the least slack;
   the 128-row class has about 3 ns.
2. **Supply network.** Every supply wire was minimum width; the decoder,
   controller and left half reached the pins through single feeds and two
   140 nm M4 bars about 45 µm long. With the extracted mesh, switching
   current lifted quiet wordlines and the sense enable to 0.09 V (the 0.1 VDD
   limit at 0.9 V) and delayed the read by 0.43 ns.

Guard slack does not offer an alternative: the ordering guards keep only
96–233 ps at FF −40 °C (64×64), against 523–875 ps at SS.

## Fixes

**Replica delay chain, N = 9 → 3** (`layout_sizing_lookup.json`,
`v2.2.8-layout-candidate-3`). Post-layout reads fire the sense amplifier
after a complete bitline swing: the 0.28 VDD sense bar is reached 315 ps
(FF −40 °C) to 1470 ps (SS) before isolation. The first chain stage, the
RBL observer, dominates the chain delay, so N = 3 keeps at least 200 ps of
that lead at every corner for the target column at 32×32 and 64×64. At
128×128 the far-end columns lead by only 159 ps at SF −40 °C, because of the
bitcell VSS feed (open failure, below). The lookup's `replica_rule` text
predates that result and is left unchanged: the lookup file's hash is part of
every driver-sizes key. The schematic matrix passes all ten 32×32 cases
and the 64×64 SS and FF −40 °C cases with N = 3. Source decks change by the
six removed chain inverters; the area changes at 64 and 128 rows only
(5,143.1 → 5,118.5 and 13,996.8 → 13,959.6 µm²).

**Supply grid** (`sram_compiler/layout/freepdk45/power_grid.py`), added
after all routing, so it never constrains the controller router:

* VDD straps on M8 and VSS straps on M9 beside the right root trunks and the
  left trunks, joined to the M6 trunks and to each core's M4 supply trunks
  by stacks every 3.2 µm per net;
* a top strap pair (VDD on M8; VSS on M7, because an M9 riser crosses the
  decoder) across the macro;
* M3 ties from the decoder's M2 supply spines to the left core's M4 trunks
  in the gaps between rows (the left core trunks lie under the M6
  wordline crossings, which leave no room for stacks);
* a direct M2 bridge from every controller row rail to its left M6 trunk,
  with two via2 cuts where they fit, VSS rails first;
* extra cuts where rails meet their spines and edge trunks: up to two via2
  per controller rail and four via1 per decoder band VSS rail.

Every via, pad and strap is checked against all existing metal and cut
shapes; a blocked site is skipped and counted in `layout.json`
(`power_grid`). The macro boundary does not change.

The grid reached this form in steps, each found by the extracted waveforms
and the per-block supply resistance (`dev/postlayout/supply_hotspots.py`
locates each block's dominant resistors by coordinate):

| Revision | Added | Remaining failure at 32×32 (extracted mesh) |
| --- | --- | --- |
| g2–g4 | straps, trunk stacks, decoder spine ties, controller trunk bridges | SS/FS 125 °C: `sa_iso` dips below 0.9 VDD when `w_en` rises (both buffers share one 88 Ω via2); FF/SF −40 °C: `s_en` lifts to 0.17/0.12 V at precharge-on |
| g5–g6 | bridges for every controller VDD rail, two via2 per bridge | FF −40 °C `s_en` lift unchanged (VSS side) |
| g8–g9 | extra rail via2, VSS rail bridges first | FF −40 °C unselected wordlines 0.118 V (decoder VSS) |
| g10 | four extra via1 per decoder VSS band rail | none: 10/10 pass (64×64: `s_en` lifts 0.099–0.205 V when `w_en` falls) |
| g11 | M3 gap straps with a right exit for every controller rail; supply pins at the left strap ends and the gap | 64×64 FF −40 °C `s_en` still 0.119 V (limit 0.110 V) |
| g12 | g11 plus enable driver chains on separate controller rows | none at 32×32 and 64×64; 128×128 top and bottom wordline drivers share 144/166 Ω of decoder VSS |
| g13 | decoder band VSS rails merged; top band rail under an M2 bus | none at 32×32 and 64×64; 128×128 bottom band still 166 Ω |
| g14 | the bottom band rail treated like the top one | none at 32×32 and 64×64; 128×128 FF −40 °C `seq`: far-end cells flip in the first read (bitcell VSS feed, open) |

**Right controller exits and supply pins (g11).** At 64×64 and above the
`s_en` buffer and the `w_en` buffer's final stage shared one controller row
rail that left the frame only at its left M3 trunk, and the macro returned
its supply over the top strap to two right-edge pins. g11 adds:

* two 480 nm M3 straps (VDD, then VSS) in the 2.2 µm gap between the
  controller frame and the right half, from the macro bottom to the core;
  M2 and M3 are free there, while M4, M5 stubs, M7 and M10 escapes cross it.
  The straps keep the 0.27 µm wide-metal spacing;
* a right exit for every controller row rail: its M2 continues into the gap
  and joins its strap through a 3 × 2 via2 array (15 Ω of kit cuts);
* macro VDD/VSS pins at the bottom of both gap straps (M3) and at both ends
  of the left M8/M9 straps.

g11 lowered the 64×64 FF −40 °C `s_en` lift from 0.205 to 0.119 V, but the M2 rail between the two
buffers and the well node that Calibre extracts as one node per row kept
31.6 Ω shared within that row; from another row the `w_en` final stage
shares 1.4 Ω. Cutting the rail between them raised the shared resistance,
because current then crosses through the body taps.

**Enable chains on separate rows (g12;
`sram_compiler/layout/freepdk45/time_control.py`, `ENABLE_CHAINS`).** The
controller band places cells in source order. A cell of one enable-line
driver chain (`w_en`, `s_en`, `sa_iso`, the precharge gate and buffer,
`wl_en`) now starts a new row when the row already holds another chain.
Later single-row logic cells back-fill the row ends this leaves, so no band
rows are added: every macro keeps its g10 bounds and area. Controllers
are re-placed and re-routed; the 128×128 controller needed one routing
retry.

**Decoder band rails (g13–g14).** Each decoder band of eight rows has VSS M1
rails along both gate rows, 70 nm wide, and its wordline drivers sit at the
decoder's right end, 20–38 µm from the VSS spine at the left edge. At
128×128 the four drivers on the decoder's top-edge rail shared 143.8 Ω
(g12) and those on its bottom-edge rail 165.5 Ω (g13); both edge rails are
single, while inner bands face a second rail 200 nm away. With g10 at
128×128, WL125 rose to 0.165 V at FF −40 °C as WL127 fell. M6 wordline crossings cover every
decoder row and the right array is 225 nm away, so the grid stays at M1/M2:

* facing VSS rails 200 nm apart at band boundaries merge into one 270 nm M1
  bus wherever no foreign M1 comes within 90 nm (the 0.09 µm wide-metal
  spacing); the 128×128 decoder has no such M1 anywhere along them;
* each edge rail widens outward to 270 nm under a 270 nm M2 bus. The bus
  keeps 90 nm from the band's own M2 over the rail, joins the VSS spine (a
  neck at the top; directly, below the decoder, where the spine continues) and
  is stitched to the widened rail by via1 every 400 nm.

A bus that misses the spine only shunts the rail: g14's first build left the
128×128 bottom bus 0.54 µm short of the spine, and that band still shared
126.5 Ω through one 88 Ω via1. The grid now reports whether each bus reaches
the spine.

Supply resistance per block (kit cuts; body taps excluded; every macro
supply pin tied), original → g10 → g14, worst source/drain terminal:

| Block | 32×32 VDD | 32×32 VSS | 64×64 VDD | 64×64 VSS | 128×128 VDD | 128×128 VSS |
| --- | --- | --- | --- | --- | --- | --- |
| Wordline drivers | 264 → 88 → 81 | 348 → 111 → 42 | 350 → 112 → 101 | 432 → 129 → 57 | 508 → 143 → 117 | 627 → 192 → 67 |
| RWL driver | 772 → 596 → 589 | 307 → 70 → 34 | 859 → 615 → 605 | 393 → 89 → 48 | 1037 → 672 → 640 | 564 → 113 → 50 |
| Precharge / sense / write slots | 120–138 → 45–54 → 38–47 | 123–139 → 46–54 → 39–43 | 151–171 → 43–52 → 34–42 | 157–177 → 49–50 → 41–43 | 225–240 → 49–51 → 33–34 | 229–233 → 53–55 → 35 |
| Bitcells (worst) | 406 → 336 → 282 | 360 → 293 → 282 | 510 → 406 → 348 | 477 → 376 → 364 | 704 → 530 → 463 | 677 → 511 → 482 |

The bounce depends on the resistance two blocks share, which the per-block
table does not show (`dev/postlayout/supply_mutual.py`: unit current into
one block's supply terminals, voltage at the other's, every pin tied):

| Shared path | 32×32 g10 → g12 → g14 | 64×64 g10 → g12 → g14 | 128×128 g10 → g12 → g14 |
| --- | ---: | ---: | ---: |
| `w_en` buffer → `s_en` buffer, VSS | 50.4 → 1.3 → 1.3 Ω | 54.4 → 3.6 → 3.6 Ω | 70.6 → 1.4 → 1.4 Ω |
| precharge buffer → `sa_iso` buffer, VSS | 11.1 → 2.0 → 2.0 Ω | 13.6 → 5.4 → 5.4 Ω | 30.9 → 1.9 → 1.9 Ω |
| `w_en` buffer → precharge buffer, VDD | 10.8 → 1.6 → 1.6 Ω | 13.7 → 1.6 → 1.6 Ω | 19.1 → 2.1 → 2.1 Ω |
| top wordline drivers (rows N−4 … N−1), VSS | — → 97.1 → 34.5 Ω | — → 114.6 → 47.6 Ω | 171.8 → 143.8 → 53.0 Ω |
| bottom wordline drivers (rows 0 … 3), VSS | — → 75.7* → 29.6 Ω | — → 82.2* → 38.8 Ω | — → 165.5* → 37.2 Ω |

\* measured on g13, whose bottom rail is unchanged from g12.

A bitcell reaches its supply through one or two minimum cuts (88–102 Ω in
the kit deck), which no grid outside the cell can remove; with cuts near
3 Ω the same 32×32 cells are within 91 Ω.

**64-row clock, 10.5 → 12.0 ns** (`sram_compiler/sizing/layout_timing_lookup.json`,
`v2.2.8-layout-timing-1`). The extracted 64×64 SS read missed the 0.68 T
deadline by 153 ps at the shared 10.5 ns class (below). The layout table
raises the 6T column-mux 64-row budget from 4000 to 4300 ps; arrays of
33–64 rows with a column mux get the longer clock, and every other class is
unchanged. Device sizes and the netlist do not depend on the clock, so the
layouts, receipts and extractions stay as they are. A 64×64 deck resolved
from the table equals the 12 ns diagnostic deck. The default
`timing_lookup.json` keeps its schematic-screened periods.

## Results after the fixes

**Physical verification (g14).** Every complete macro with replica N = 3,
the g12 controller placement and the g14 grid passes Calibre DRC with zero
results and LVS `CORRECT`, followed by distributed R+C+CC extraction. The grid leaves every
boundary unchanged; the areas differ from the macro-density baseline only
where N = 3 changed the controller placement.

| Macro | Bounds (µm) | Area (µm²) | Baseline area | Extracted nets | Grid stacks / skipped sites | Merged decoder M1 (µm) | Edge buses (µm) / via1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32×32 | 64.78 × 34.42 | 2,229.5 | 2,229.5 | 3,568 | 169 / 17 | 70.8 | 2 × 23.6 / 116 |
| 64×64 | 104.08 × 49.18 | 5,118.5 | 5,143.1 | 10,952 | 251 / 29 | 188.4 | 2 × 26.9 / 132 |
| 128×128 | 187.48 × 74.46 | 13,959.6 | 13,996.8 | 38,128 | 391 / 48 | 574.7 | 2 × 38.3 / 190 |
| 256×256 | 351.12 × 125.02 | 43,896.7 | 43,896.7 | 141,718 | 731 / 88 | 1,796.6 | 58.0 + 57.9 / 288 |
| 512×512 | 669.08 × 226.14 | 151,305.2 | 151,305.2 | 544,772 | 1,349 / 171 | 5,538.6 | 2 × 87.9 / 438 |

Each macro has six more supply pins than before and a right exit on every
controller row rail (22–26 per macro). 32×32 uses the register gap and
decoder pad of the baseline (400 nm, 1.7 µm); the area-optimal N = 3
candidate (no gap, 0.7 µm pad) routes only after about 150 controller
rip-ups.

**32×32 waveforms (g14)**, extracted VDD/VSS mesh, coupling ≥ 10 aF, kit cut
resistance, every supply pin tied, 8 accesses: **10/10 pass, 228,020
checks**.

| Corner | Read output margin (ps) | Sense margin (V) | WL off → sense (ps) | Isolation → enable (ps) | Enables off → precharge (ps) | Driver on → WL (ps) | Restore → capture (ps) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SS 0.9 V 125 °C | 302 | 0.882 | 690 | 653 | 735 | 632 | 1,739 |
| FS 0.9 V 125 °C | 886 | 0.886 | 598 | 554 | 635 | 521 | 2,081 |
| FF 1.1 V 125 °C | 1,882 | 1.027 | 406 | 393 | 430 | 364 | 2,754 |
| SF 0.9 V −40 °C | 2,936 | 0.884 | 236 | 233 | 271 | 217 | 3,465 |
| FF 1.1 V −40 °C | 3,277 | 1.051 | 181 | 179 | 204 | 154 | 3,686 |

Minimum of both patterns per corner. g12 and g13 also passed all ten
32×32 cases (228,020 checks each); g12's margins are within 4 ps of these. With
g10 the matrix passed with an SS read margin of 277 ps and FF −40 °C quiet
lines at 0.101 V (unselected wordlines) and 0.092 V (`s_en`) of the 0.110 V
limit.

**64×64 and 128×128** use frozen cells and the first four accesses
(W1/R1/W0/R0 or both rows written and read), with the start state from a
stimulus-free settle. A full-cell, eight-access 64×64 SS run is the
reference for the frozen-cell model: through the first read its control
edges agree within 7 ps (`s_en` at 7,085 ps frozen, 7,092 ps full). It
stopped at postsim's 10 h limit at 29.6 ns, during the third access.

**64×64 history.** With g10 at the shared 10.5 ns clock all ten cases failed
(8,313–8,905 checks per case):

| Corner | Failed checks | Mechanism |
| --- | --- | --- |
| SS 0.9 V 125 °C | read data 153 ps late; bitlines 0.862 V at 1.1 T after a write (0.882 V required) | access time |
| FS 0.9 V 125 °C | `s_en` 0.099 V (limit 0.090 V) | enable ground bounce |
| FF 1.1 V 125 °C | `s_en` 0.141 V (limit 0.110 V) | enable ground bounce |
| SF 0.9 V −40 °C | `s_en` 0.141 V (limit 0.090 V); the far `sa_iso` taps just above 0.090 V when precharge turns on after a read | enable ground bounce |
| FF 1.1 V −40 °C | `s_en` 0.205 V (limit 0.110 V); unselected WL61 0.115–0.121 V when the row address changes to 63; precharge 0.974 V (limit 0.990 V) when `w_en` rises | enable, decoder and supply bounce |

At SS, `s_en` rose 7,085 ps into the cycle against the 7,140 ps deadline.
It waits for the 8-stage wordline-off guard (650 ps after the far RWL
falls), 295 ps after isolation is ready. With g10 at the 12 ns layout clock,
SS passed both patterns, and every failure of the other eight cases was
bounce. `dev/postlayout/bump_rescore.py` zeroes only sub-0.3 V bumps on
`s_en`, `sa_iso`, the wordlines and droops of the precharge line, then rescores
the copy with the unchanged scorer: all ten 12 ns cases pass that way. The
12 ns clock therefore leaves no timing or logic failure. The `s_en` lift
came when `w_en` fell at the end of each write, through 54 of the `s_en`
buffer's 73 Ω to the pins shared with the `w_en` buffer; the precharge line
drooped through 14 Ω of VDD shared with the `w_en` buffer.

**64×64 waveforms (g14)** at the 12 ns layout clock: **10/10 pass, 86,410
checks**. g12 and g13 also passed all ten cases.

| Corner | Read output margin (ps) | Sense margin (V) | WL off → sense (ps) | Isolation → enable (ps) | Enables off → precharge (ps) | Driver on → WL (ps) | Restore → capture (ps) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SS 0.9 V 125 °C | 559 | 0.805 | 888 | 835 | 957 | 651 | 1,941 |
| FS 0.9 V 125 °C | 1,265 | 0.798 | 773 | 720 | 815 | 511 | 2,363 |
| FF 1.1 V 125 °C | 2,446 | 0.878 | 536 | 522 | 555 | 320 | 3,209 |
| SF 0.9 V −40 °C | 3,671 | 0.723 | 315 | 331 | 329 | 156 | 4,084 |
| FF 1.1 V −40 °C | 4,087 | 0.831 | 242 | 262 | 227 | 90 | 4,368 |

Worst quiet-line levels over all accesses with g14 (`dev/postlayout/quiet_margins.py`,
the scorer's cycle windows; limits 0.1 VDD and 0.9 VDD):

| Corner | `s_en` in writes | `w_en` in reads | Unselected wordlines | Precharge while a WL or enable is active |
| --- | ---: | ---: | ---: | ---: |
| 32×32 SS / FS / FF 125 °C / SF / FF −40 °C | 0.067 / 0.064 / 0.082 / 0.057 / 0.071 V | 0.025 / 0.025 / 0.035 / 0.030 / 0.040 V | 0.035 / 0.035 / 0.045 / 0.037 / 0.064 V | 0.865 / 0.861 / 1.047 / 0.844 / 1.021 V |
| 64×64 SS / FS / FF 125 °C / SF / FF −40 °C | 0.056 / 0.053 / 0.069 / 0.047 / 0.058 V | 0.029 / 0.033 / 0.040 / 0.037 / 0.046 V | 0.049 / 0.050 / 0.066 / 0.059 / 0.096 V | 0.856 / 0.852 / 1.036 / 0.841 / 1.018 V |
| 128×128 SS / FS / FF 125 °C / SF / FF −40 °C | 0.049 / 0.047 / 0.057 / 0.040 / 0.045 V | 0.022 / 0.024 / 0.030 / 0.026 / 0.033 V | 0.042 / 0.041 / 0.054 / 0.055 / 0.089 V | 0.862 / 0.860 / 1.046 / 0.850 / 1.031 V |

The closest is the 64×64 FF −40 °C unselected wordline, 0.096 V of
0.110 V (g12: 0.104 V; 32×32: 0.077 → 0.064 V). It lifts about 120 ps after
the address register switches, while `wl_en` is still at 0.007 V, so it is
not a decoder race; it comes from the decoder's own switching and coupling,
which the band rails reduce only in part.

**128×128 history.** With g10, the first write lifted `s_en`
at every corner (0.119 V at SS to 0.247 V at FF −40 °C) and WL125 to
0.165 V at FF −40 °C (0.112 V at FF 125 °C, 0.114 V at SF −40 °C) as WL127
fell. Those runs stopped at postsim's 10 h limit; the g14 runs had 48 h
and took 13.5–18 h. g14 lowers the `s_en` path (70.6 → 1.4 Ω) and the top
and bottom wordline-driver bands (171.8 → 53.0 Ω and 165.5 → 37.2 Ω).

**128×128 waveforms (g14)** at the 21.25 ns clock: **9/10 pass, 172,170
checks; FF 1.1 V −40 °C `seq` fails** (109 of 16,697 checks, see the open
failure below). Every quiet line stays inside its limit in all ten cases
(table above), so the enable and decoder bounce fixes hold at this size. At
SS the read data settle 2.9 ns before the 0.68 T deadline (`s_en` fires
11.28 ns into the cycle).

| Corner | Read output margin (ps) | Sense margin (V) | WL off → sense (ps) | Isolation → enable (ps) | Enables off → precharge (ps) | Driver on → WL (ps) | Restore → capture (ps) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SS 0.9 V 125 °C | 2,925 | 0.512 | 1,128 | 1,331 | 1,441 | 804 | 4,509 |
| FS 0.9 V 125 °C | 3,848 | 0.484 | 981 | 1,151 | 1,215 | 613 | 5,058 |
| FF 1.1 V 125 °C | 5,367 | 0.502 | 688 | 828 | 819 | 357 | 6,111 |
| SF 0.9 V −40 °C | 6,868 | 0.323 | 410 | 531 | 533 | 187 | 7,218 |
| FF 1.1 V −40 °C (`mixed` only) | 7,433 | 0.637 | 321 | 411 | 337 | 84 | 7,589 |

The sense margins fall from 0.88–1.07 V (32×32) and 0.72–0.93 V (64×64)
to 0.32–0.72 V. The lowest are not at the target column: they come from the
far-end columns of each half-array (open failure, below).

**256×256 and 512×512** are not simulated. With frozen cells, the 256×256
network still holds 1.20 M resistors, 1.31 M capacitors and 187,321 MOS
(3.4 times 128×128), and its schematic start state alone ran for more than
68 minutes; a transient would take about a week at the measured rates.
512×512 is also outside the automatic timing envelope. Both sizes carry the
physical verification above, the same grid construction and enable rows,
and the bitcell VSS feed below projects read failures at both.

## Open failure: bitcell VSS feed

**Failure.** At 128×128 FF 1.1 V −40 °C `seq`, cells of the read row 127
at the far end of each half-array (columns 0–11 and 64–73) flip 5,985–6,055 ps
into the first read, while the wordline is on and before isolation
(6,445 ps); more cells of the same columns flip in the second read. The
sense groups over them (0–5 and 32–36) then read the reversed data. The 109
failed checks are 60 read retention, 34 sense data and hold, 14 write-cycle
retention of the flipped cells and the read output margin (−5 ps). The low storage node of well-fed cells (columns 62, 63, 126,
127) peaks at 0.20 V; far-end cells reach the 0.55 V reversal point (column
2 peaked at 0.549 V and held). The `mixed` pattern at the same corner passes
with a 0.144 V storage margin; why it holds has not been analysed.

At SF 0.9 V −40 °C `seq` the same columns pass with less margin: their
sense differential at isolation is 0.323 V (limit 0.28 VDD = 0.252 V)
against 0.83 V at the target column, and their low storage node peaks at
0.318 V against 0.132 V next to the feed. The 0.28 VDD bar is reached
456 ps later at column 0 than at column 63 (the left half's near end); the
same spread is 61 ps at 64×64 (columns 0 and 31) and 8 ps at 32×32
(columns 0 and 15).

**Mechanism.** The bitcell has VSS on 70 nm M3 rails along both row edges,
shared with the mirrored neighbour row (`compact_core.py`). Each half-array
joins these rails only to the M4 VSS trunk in the empty column slot at one
end (`compact_array.py`): the left array at its centre side (x ≈ 70 µm at
128×128), the right array at the macro's right edge. Nothing ties the
rails together inside the array, and M4–M7 are empty over both arrays. A
read turns on all cells of one row, so the whole row's current flows along
that row's two rails to one end. Neither the wordline nor isolation causes
the spread: the wordline reaches 0.89 V at both ends by 6.2 ns at SF −40 °C,
and isolation samples every column within 7 ps.

Transfer resistance from the read row's cell VSS terminals (unit current
shared by the row) to one cell's VSS, every supply pin tied, kit cuts
(`dev/postlayout/supply_mutual.py`; evidence key `bitcell_row_vss`):

| Macro (read row) | Left half far / near | Right half far / near | Far − near in each half |
| --- | ---: | ---: | ---: |
| 32×32 (31) | col 0: 21.5 Ω / col 15: 13.9 Ω | col 16: 17.0 Ω / col 31: 9.4 Ω | 7.6 Ω |
| 64×64 (63) | col 0: 33.1 Ω / col 31: 17.5 Ω | col 32: 22.9 Ω / col 63: 7.3 Ω | 15.6 Ω |
| 128×128 (127) | col 0: 48.6 Ω / col 63: 16.9 Ω | col 64: 38.1 Ω / col 127: 6.4 Ω | 31.7 Ω |
| 256×256 (255) | col 0: 82.6 Ω / col 127: 18.8 Ω | col 128: 70.3 Ω / col 255: 6.5 Ω | 63.8 Ω |

The far − near resistance doubles with each size and the row current
doubles too, so the far-end ground rise grows about four times per size
step. At 128×128 it is 0.19 V at SF −40 °C and at least 0.35 V at
FF −40 °C. A linear estimate for 256×256 is four times that; the cell
current falls as its ground rises, so the true rise is lower, but the
128×128 FF −40 °C read already flips. 256×256 and 512×512 reads are
therefore not expected to pass until the feed changes.

**Planned fix (next round).** Vertical M4 VSS straps over each array at
column boundaries, with a via3 on every row-edge M3 rail, would tie all
rails into one mesh. A read row's current would then reach the trunk
through every row's rails in parallel instead of its own two. A second trunk
at the far end of each half-array is an alternative. Either change needs a
rebuild, Calibre at every size and the 32/64/128 worst-PVT matrix again.
Decision of 2026-10-31: record the failure and commit g14; the fix is left
for the next round.
