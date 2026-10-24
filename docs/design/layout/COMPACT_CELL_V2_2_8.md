# Compact point-symmetric bitcell and its periphery — V2.2.8

The opt-in compact bank replaces the OpenRAM-derived 6T cell (705 × 1365 nm)
with a point-symmetric thin cell of **1125 × 395 nm (0.444 µm², −53.8 %)**,
modelled on the TSMC N28 MC2 cell, and re-pitches the whole bank around it
with a 2:1 column mux. Device W/L, models and topology are the OpenRAM and
OpenYield 6T sizes (PD 205, PG 135, PU 90 nm, L 50 nm). The OpenRAM pitch
remains the default layout; sizing and timing tables are unchanged. Since
2026-10-23 the compiler's default deck has the 2:1 mux the compact bank needs.
The same day the decoder was rebuilt with two gate rows per band, so the
complete macro is smaller than the OpenRAM-pitch one at every size from
32 × 32 to 512 × 512.

## Reference cell

The reference macro is `ts1n28hpcplvtb512x32m4s_120a.gds` (512 words × 32
bits, mux 4). Its bitcell `S1BLVTS200W20_032_D156_V1D0_X1` is two
`_HALF` cells, the second rotated 180° about the cell centre: 0.58 × 0.27 µm
(0.157 µm²). Each half has one NMOS strip carrying the pass gate and the
pull-down in series, and one PMOS strip; the two PMOS strips share the
centre n-well. Poly runs in two horizontal rows: the pass-gate row continues
across the column edge to a WL contact shared with the mirrored neighbour,
and the inverter row ends in a butted contact on the other half's PMOS drain.
Cut poly (layer 17/12) separates the two gate nets on each row. Bitlines and
VDD run on vertical M2, the wordline and VSS on horizontal M3. Rows and
columns mirror, so every diffusion contact on a cell edge is shared.

## FreePDK45 cell

![Compact bitcell with all six MOS instances and their W/L annotated](COMPACT_CELL_V2_2_8.png)

The PNG renders the actual `CMP_CELL` polygons from the verified 32×32 bank
GDS, with translucent metal and labels at each active/poly gate intersection.
Its geometry was XOR-checked against `compact_core.draw`; the six drawn gate
widths and lengths match the source deck. The dashed frame is the tiling
boundary; shared contacts and rails extend beyond it.

FreePDK45 has neither butted contacts nor cut poly, and its rules are
coarser, so the topology is kept and the dimensions follow the Calibre deck:

- **Height 395 nm.** Gates keep 140 nm (Poly.2) to the next row's gates
  across a shared edge contact, so each gate row sits 70 nm from its edge;
  pass gate and pull-down are 140 nm apart with the storage contact between.
  A PMOS drain sits 67.5 nm below its gate; the other inverter's poly pad
  keeps 50 nm (Poly.5) above that strip end. A 387.5 nm variant needs a poly
  jog that leaves 42.5 nm of width (Poly.1).
- **Width 1125 nm** = 2 × (40 PMOS half-gap + 90 PU + 140 N–P gate spacing
  + 205 PD + 87.5 to the column edge). The edge margin is the shared WL pad
  (37.5 + 50 to active), equal to the pull-down endcap and half the 75 nm
  field-poly gap.
- **Cross-couple.** Each inverter gate ends in a 75 nm poly pad beyond the
  other PMOS strip's drain end. The pad contact sits 35 nm outward and
  7.5 nm higher than the strip so it keeps 75 nm (Contact.2) to the drain
  contact; the M1 storage node joins pad, PMOS drain and NMOS node.
- **Routing.** BL/BLB and two VDD lines on M2, WL on M3 across the row centre,
  VSS on M3 rails along both row edges; via stacks on the column edges.

Variants for the replica column and the RWL loads: `replica` ties QB to
VDD; `passive` also joins both inverter gates to the wordline poly, which
the replica column ties to VSS; `dummy` leaves the pass-gate drains
uncontacted. A 460 nm well-strap row (n-tap on an M1 VDD rail, edge p-taps
on an M3 VSS rail, bitlines passing through) separates every array edge from
the next block. Because bitline contacts sit on the row edge, the RWL load
row sits one strap above the array instead of abutting it.

Calibre (official FreePDK45 decks): 4 × 4 and 8 × 6 arrays, a replica column
(passive + replica), a mixed array and dummy rows pass DRC 0 and LVS
CORRECT; the 32 × 32 core with replica column, RWL load row, straps and
trunks passes DRC 0, LVS CORRECT and R+C+CC extraction.

## Periphery at the compact pitch

| Block | Plan |
| --- | --- |
| Column pair slot | 2250 nm, columns c (L) and c+1 (U); mirrored columns put the bitlines at 172.5, 952.5, 1297.5 and 2077.5 nm; mirror-symmetric M3 track tables |
| Write | one slot row each of data buffers, both columns' input inverters, both BL tristates and both BLB tristates (two fingers per device), shared EN; 7.39 µm at 32 rows (10.95 µm as three rows per column) |
| Mux and sense | one 2:1 `COLUMNMUX2` and one `SENSEAMP` per pair: SEL inverters, BL transmission gates, BLB transmission gates, folded amplifier (tail device in two fingers); 6.65 µm |
| Precharge, registers | `dense_slots` builders with the compact geometry (registers 7.73 µm per pair; tracks from a conflict-repair search) |
| Decoder | one 3-to-8 slice per 3160 nm band (8 rows): two 1365 nm gate rows of four chains, the upper mirrored; the slice's seven bus lines as M2 between the rows; `wl_en` on M2 lines along the band edges; every WL on its own M4 line |
| Bank | M5 bitlines lifted in the facing strap of each array, one M6 trunk per control net below the decoder's right edge, supply trunks right of the core, SEL0/SEL1 as bank inputs |

Two FreePDK45 limits shape the decoder: a 140 nm M4 landing cannot sit
between M4 lines 395 nm apart, and the dense gates are 1365 nm tall. Every
decoder gate keeps its M2 between 220 and 1055 nm of its own rail side
(checked for each gate), so one band holds two gate rows, row B mirrored
above row A with the VDD sides facing. The slice's seven bus lines (three
addresses T, their complements C, the enable) fit as M2 at a 140 nm pitch
between the rows, 70 nm from the M2 of both. Column j stacks chains j and
j + 4, so the pins of both rows sit on one x. The two chains share every
line except the first address, and the line order (C0, C2, T0, EN, T1, C1,
T2 from the bottom) keeps a row-A riser two pitches below the row-B riser
wherever their nets differ. Input inverters 1 and 2 stack the same way;
inverter 3 has nothing above it.

The driver NAND2s take `wl_en` from M2 lines on the band boundaries (one
line serves the two rows that face it), so no M4 landing sits among the
wordlines and each WL leaves its driver on its own M4 line. A folded driver
NAND2 joins its A fingers on M3 above B, so B is taken down in row A and up
in row B. `wl_en` has one M5 trunk; the eight-chain decoder had an M5 track
per chain column. Each level reserves only the trunk slots it uses, with
the `wl_en` slot last, so the centre plan's off-grid riser jogs stay at
three per run. Both halves of a split bank reserve every decoder level's
region, so a net has one trunk x in both. A predecode slice spreads its
chains over up to four bands of its span.

`dense_slots` takes a `geometry` argument; with the default OpenRAM geometry
its write, precharge, replica and register slots are XOR-identical to the
previous source at 32 and 128 rows, and the 94 local layout regression tests
pass.

## Centre floorplan with TIME_CONTROL

`layout bank --compact-cell --with-controller` follows the
[centre plan](LAYOUT_SCALING_V2_2_8.md#split-arrays-with-a-centre-periphery):
split arrays around the column periphery with continuous M5 bitlines, data
registers folded into the 2250 nm slots below the write band (7.73 µm), M10
bridges for decoder nets of both halves and inputs escaping on M10 to the
left, bottom or top edge. The decoder halves leave a block as tall as the
periphery, and TIME_CONTROL needs no decoder lanes. The block is as wide as
the decoder unless TIME_CONTROL needs more; then the core moves right to the
narrowest block width (to 1 µm) whose band holds every controller gate. That
happens from 256 × 256: TIME_CONTROL needs 69.5 µm there against a 58.8 µm
decoder, and 141.1 µm at 512 × 512 against 90.4 µm. The array then starts
12.6 and 52.6 µm further right; that strip carries only the M4 wordlines and
the control trunks (3.5 % and 7.3 % of the macro). The interface differs
from the OpenRAM-pitch plan in these points:

- Each control net (`PRE`, `w_en`, `s_en`, `sa_iso`) has its own M6 trunk
  and takes one M6 drive. Its far-end return leaves the end of the net's
  longest bar (in the replica slot) beside the VDD trunk. It climbs on M5 to
  its own pin height before lifting to M7, because bars of different nets can
  sit closer than one M7 pad pitch (400 + 400 nm).
- The replica sense return is the mux output `RBL_MUX`. The replica wordline
  returns from the core's M4 riser at the RWL load row (M7 over the upper
  array, M9 down over the decoder). The RWL driver sits on the upper
  decoder, and the output latch folds into the replica slot's register position.
- `SEL0`/`SEL1` are macro inputs: their M6 trunks stack to M10 and escape like
  the TIME_CONTROL inputs.
- Decoder trunks off the router's 100 nm grid jog to it. Neighbours 300 nm
  apart jog the same way, so their stubs are staggered 300 nm in height.
- Sense-slot outputs land on M4 at the point of their M3 track farthest from
  every slot-wide M4 bar (at least 280 nm).

## Results

All compact layouts come from the candidate-2 decks with the 2:1 column mux
(`--column-mux`, nominal TT, explicit diagnostic period). They pass Calibre
DRC 0, LVS CORRECT and distributed R+C+CC extraction
([evidence](../../data/COMPACT_CELL_V2_2_8.json)). The OpenRAM-pitch
references use the same driver table without a mux. Bank-only references
are generator bounding boxes; centre references are the verified
[macro-pin layouts](LAYOUT_SCALING_V2_2_8.md#macro-edge-pins-and-right-edge).
The superseded eight-chain decoder's raw outputs were purged on 2026-10-24;
their historical dimensions and hashes remain in the evidence JSON.

| Layout | Compact (µm) | Compact (µm²) | OpenRAM pitch (µm²) | Change |
| --- | --- | ---: | ---: | ---: |
| bank 32 × 32 | 63.58 × 32.14 | 2,044 | 2,339 | −12.6 % |
| bank 64 × 64 | 104.58 × 45.74 | 4,783 | — | — |
| bank 128 × 128 | 189.83 × 71.02 | 13,481 | 21,863 | −38.3 % |
| bank 256 × 256 | 350.39 × 121.58 | 42,600 | — | — |
| bank 512 × 512 | 669.99 × 222.70 | 149,206 | — | — |
| centre 32 × 32 | 65.87 × 42.14 | 2,776 | 2,848 | −2.5 % |
| centre 64 × 64 | 104.58 × 55.74 | 5,829 | 7,295 | −20.1 % |
| centre 128 × 128 | 189.83 × 81.02 | 15,380 | 22,724 | −32.3 % |
| centre 256 × 256 | 362.99 × 131.58 | 47,762 | 78,367 | −39.1 % |
| centre 512 × 512 | 722.59 × 232.70 | 168,145 | 287,186 | −41.5 % |

The 64 × 64 and 256 × 256 decks use the OpenRAM-pitch references' periods
(10.5 and 26.25 ns). Every layout was generated from the final source tree,
or regenerated from it XOR-identical to the verified GDS (bank 128, 256 and
512 and centre 128, built before the controller-width search; recorded per
layout). The 512 × 512 centre build took 36 min (12.3 GB) and the 256 × 256
centre build 16 min.

The decoder had set the small-array result. Its eight chains per band sat
side by side in one gate row with the bus in a channel above them: 49.2 µm
of decoder at 32 rows against a 36 µm array, so the 32 × 32 macro was 34.1 %
larger than the OpenRAM-pitch one. TIME_CONTROL's gates filled only the
lowest 9.05 µm of its 49.1 × 24.7 µm block. Two gate rows per band halve the
chain columns. Together with the used-slots-only trunks, tighter region gaps
and the control trunks moved from a 2.2 µm strip between decoder and core
to below the decoder's edge, the 32 × 32 decoder is 26.2 µm wide (split
halves) and the macro 2.5 % smaller than the OpenRAM-pitch one. Its
TIME_CONTROL fills nine rows of a 24.3 µm block. The decoder is 42.2 µm at
128 rows (74.1 µm before) and 90.4 µm at 512 rows (166.8 µm). From
256 × 256, TIME_CONTROL rather than the decoder sets the block width.

## Limits

- Driver classes are the candidate-2 table, derived from OpenRAM-pitch loads.
  The compact wordline is 1.6 × longer per cell and the bitline 3.5 × shorter,
  so the classes need re-derivation from compact extractions; none was run.
- No waveform, extracted-timing, operating-point or yield validation ran for
  any compact layout. The decks use explicit diagnostic periods (9, 10.5, 30,
  26.25 and 80 ns at 32, 64, 128, 256 and 512 rows); 512 × 512 is outside
  the automatic timing envelope. `wl_en` now reaches the drivers over M2 band
  lines and one M5 trunk; it was not re-timed.
- The 2:1 column mux became the compiler default on 2026-10-23
  (`choose_columnmux: true` in `global.yaml`, `ARRAY = [16, 16, True]` in
  `main_sram.py`), without simulation re-validation. The default clock
  class follows the mux timing ladder (16 × 16: 9.5 ns instead of 9.0 ns).
  The OpenRAM-pitch `layout bank` still needs an unmuxed deck
  (`--no-column-mux`).
- The `beside` and legacy floorplans have no compact variant; banks below 16
  rows have no integrated controller.

## Reproduction

Decks, layouts and receipts are under the ignored `outputs/layout/compact/`;
the [layout README](../../../sram_compiler/layout/README.md#compact-point-symmetric-cell-opt-in)
has the commands. `dev/compact_cell/collect_evidence.py` rebuilds the evidence
record from the current receipts, retaining the purged eight-chain summaries
from the existing evidence JSON;
`tests/test_compact_cell.py` (local) checks the cell's point symmetry, its
gate W/L against the source deck, the pitch constants, the decoder bus and
stacking rules, the mux default, the recorded receipts and that every
recorded compact macro is smaller than its OpenRAM-pitch reference.
`layout preview --region=X0,Y0,X1,Y1` renders a window such as one decoder
band.
`dev/compact_cell/render_bitcell.py` reproduces the annotated bitcell PNG
and writes its GDS/deck/image hash receipt under `outputs/layout/compact/`.
