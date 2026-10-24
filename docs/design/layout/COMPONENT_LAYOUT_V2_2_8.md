# V2.2.8 component layout development

**Current bank implementation:** the user-selected OpenRAM pitch is now
**705 nm per column and 1365 nm per row**. The decoder remains on the left,
with one output per physical row; the replica column is inside the core.
Physical refingering preserves every MOS’s total W/L and topology. See
[the current implementation and verification record](OPENRAM_PITCH_V2_2_8.md).
The stages below retain their original geometry and evidence.

The user requested a bottom-up physical implementation: build small cells
first, then place and route their instances; peripheral geometry must follow
the SRAM core's row/column pitch. The new opt-in `layout cells` entrance
implements that component stage. It does not replace the historical compact
or calibration macro generators. Sizing, timing, variation and calibrated RC
defaults are unchanged.

## Source inspection

The comparison inspected these public trees, rather than only their READMEs:

| Project | Pinned commit | Relevant implementation |
|---|---|---|
| OpenRAM | `b2b069ce119d1488cbe6883b2240bceb5c7ce29a` | Shared schematic/layout hierarchy, parameterized inverter/NAND cells, module factory, bank/address/data ports and repeated peripheral arrays |
| SRAM22 | `37eca15a9f69348d3cee5ba69b2b1c918a969050` | Component parameter types, gate schematic/layout views, composite gates, column tiling, array boundary cells, SRAM planning and per-component verification |

OpenRAM's [hierarchy_design](https://github.com/VLSIDA/OpenRAM/blob/b2b069ce119d1488cbe6883b2240bceb5c7ce29a/compiler/base/hierarchy_design.py)
combines schematic and layout hierarchy. Its
[inverter](https://github.com/VLSIDA/OpenRAM/blob/b2b069ce119d1488cbe6883b2240bceb5c7ce29a/compiler/modules/pinv.py)
owns transistor placement, well contacts, gate/output wiring and supply rails.
The [factory](https://github.com/VLSIDA/OpenRAM/blob/b2b069ce119d1488cbe6883b2240bceb5c7ce29a/compiler/sram_factory.py)
reuses modules with matching constructor parameters. Bank/data-port placement
passes actual bitcell column offsets into repeated peripheral arrays.

SRAM22's [gate types](https://github.com/ucb-substrate/sram22/blob/37eca15a9f69348d3cee5ba69b2b1c918a969050/src/blocks/gate/mod.rs)
associate parameters with schematic and layout implementations. Its
[AND layouts](https://github.com/ucb-substrate/sram22/blob/37eca15a9f69348d3cee5ba69b2b1c918a969050/src/blocks/gate/layout.rs)
instantiate NAND/inverter components, join their local ports and expose the
parent interface. Its
[column layout](https://github.com/ucb-substrate/sram22/blob/37eca15a9f69348d3cee5ba69b2b1c918a969050/src/blocks/columns/layout.rs)
tiles precharge, mux, sense, write, latch and register components with explicit
port exposure. Separate physical design scripts determine layout constraints.

The useful common pattern is a circuit-owned physical component with an
ordered interface and local connections. OpenYield's earlier compact flow
reused groups of transistor geometry but still sent every MOS terminal to the
macro router. Geometry grouping alone does not establish a reusable cell.

The original component stage kept both device sizes and finger counts and did
not import technology cells. The current bank preserves logical sizes and
topology, permits the subsequently authorized physical refingering, and uses
attributed OpenRAM core/peripheral assets where their source devices match.
Historical generators retain their earlier policy.

## Implementation

`components.py` resolves the selected native compiler instance into immutable
component, child-instance and MOS specifications. It retains the ordered
formal ports before upstream ties, shorts only known illustrative wiring R,
and drops only known wiring C. Intentional passives, symbolic sizing and
unsupported parameters require another physical view. Cache identity includes
connectivity, scoped child structure, models, actual W/L and NF, with the
physical axis, core pitch and array pin policy held in the physical context.

Leaf cells partition each source MOS into its explicit fingers. Connected
source/drain terminals can share a diffusion strip only with matching model,
finger width, length and body. Diffusion breaks keep every remaining finger.
Every strip has a body tap. Local routing owns M1/M2/M3 and reserves the
boundary pin's upper-metal access before routing other signals. Row cells
include physical M3 supply rails. Active/poly intersection area provides an
independent local check of total source gate width and finger count.

`cell_flow.py` places those library cells and routes named ports. Column
composites now reuse at most four M5 tracks inside one column, leaving
clearance at mirrored boundaries. Row
components use M5 pin legs and M3 channels spaced by the core row pitch;
short nets use their own physical span and disjoint spans can reuse channels.
The root source retains the original schematic hierarchy even where physical
placement descends directly to its leaf cells. Every placed leaf and all four
terminals of every source finger remain in the abstract.

The macro router's defaults remain the original M3/M4/M5, 280 nm grid.
Optional routing-stack arguments support the leaf router's M1/M2/M3,
160 nm grid with actual via-layer numbering and enclosure clearance.

## Pitch contract and limits

The current technology uses the unchanged 6T core frame: 1365 nm by 400 nm.
Column peripheral roles cannot choose the row axis. Their cell frame is one
column wide, and their bitline access follows BL x=155 nm and BLB x=1210 nm.
Those access points are on M5 to avoid the cell's local wiring; the bank data
path now connects them explicitly to core M3 through M4. Odd columns mirror the
cell, interface and instance terminal map together.

Row leaf geometry is three core rows high, with widths in whole core columns.
Composite channels have explicit additional row spans. Wordline output
interfaces retain the physical row phase. These spans preserve frozen W/NF;
they are not a density result or proof of direct single-row rail abutment.

The generic decoder assembly exercises cell reuse and port routing. Its
initial long global connections failed antenna checks. Reusing local channels,
restricting short connections to their physical span and using 70 nm M3 signal
buses allowed the final standalone decoder to pass. Nothing generated by this
entrance is accepted merely because it routes. Failed builds and physical
checks retain separate attempts and remain unqualified.

This stage supports the nominal FreePDK45 VTG devices with legal 5 nm W/NF
and length increments. There is no complete new bank integration, multi-column
mux view, TIME_CONTROL passive/controller view, cell characterization,
compact waveform qualification or compiler-wide RC/default promotion. The
component bank, bitcell views and their evidence remain available. The older
fixed/scaled/compact macro generators and calibration artifacts have been purged.

## Original component evidence

See [the layout tutorial](../../../sram_compiler/layout/README.md#small-cells-and-component-placement).
Use native nominal 8x8 mode-0 read/write source with explicit illustrative
interconnect, TT, 1 V, 25 C and a 9 ns period. Each component and its leaf
bundles have separate official DRC/LVS/R+C+CC receipts. The source deck and
schematic, GDS, generator, Calibre executable and external rule hashes bind
the checks to the tested artifacts.

All nine final component cases pass **zero Calibre DRC violations**, LVS
**`CORRECT`** and complete distributed **R+C+CC** extraction:

| Component | Axis | Physical scope |
|---|---|---|
| Precharge | Column | Shared diffusion and core-aligned BL/BLB access |
| Sense amplifier | Column | All source MOS, controls, outputs and bitline access |
| Write driver | Column | All source devices and bitline access |
| Write-data buffer | Column | Two placed instances of the same inverter view |
| Wordline driver | Row | Placed NAND2 and inverter, with core wordline phase |
| 8-row decoder | Row | 35 placed leaf instances and core row output access |
| NAND3 | Row | Source-matched six-device logic cell |
| NOR2 | Row | Source-matched four-device logic cell |
| Folded clock inverter | Row | Four PMOS and two NMOS fingers, unchanged source NF |

The **ten exported leaf variants** also pass independent DRC/LVS/R+C+CC
checks. This includes the additional inverter/NAND2 variants reused by the
assemblies. These checks do not establish standard-cell characterization or
full-bank abutment. The final **48** local component, compact-layout, routing,
calibrated-RC, controller-configuration and unified-entrance regressions pass.
Compilation and `git diff --check` pass.

The tracked [case protocol](../../../tests/spice/v228_component_cases.json) and
[physical evidence](../../data/COMPONENT_LAYOUT_V2_2_8.json) retain these exact
scopes and hashes. Raw artifacts and independent receipts remain under
`outputs/layout/cells/campaign_07/`; the
[native GDS preview](../../../outputs/layout/cells/campaign_07/preview/components.png)
shows the cells and assembled driver without modifying their geometry.
Python regression and campaign scripts stay under ignored `tests/` and `dev/`.
No waveform comparison ran in this component stage and no yield or full-array
qualification is claimed.

## Density and bank integration review

The follow-up review found three physical issues. Each diffusion strip carried
two unnecessary empty routing rows in addition to its tap/access clearance.
The parent M2 via landing was not reserved in both orientations before a row
leaf rotated. Finally, the original five-trunk column assembly put metal and
vias too close to neighbouring mirrored cells. Standalone checks could not
prove that those interfaces were safe to tile.

The generator removes the duplicate strip gap, reserves both M2 enclosure
orientations, and prevents M1 via landings from consuming the mirrored edge
clearance. Column assemblies allocate tracks by vertical span: disjoint nets
reuse a track with a 300 nm separation. The data buffer needs three tracks;
four are available with legal boundary margins. No W/L/NF, schematic topology,
clock, model or RC default changes accompany these edits.

| Component | Original area (µm²) | Revised area (µm²) | Reduction |
|---|---:|---:|---:|
| Precharge | 4.914 | 4.368 | 11.11% |
| Sense amplifier | 20.748 | 18.018 | 13.16% |
| Write driver | 27.846 | 23.478 | 15.69% |
| Write-data buffer | 13.104 | 12.012 | 8.33% |
| 8-row decoder | 2186.184 | 1993.992 | 8.79% |
| NAND3 | 13.104 | 9.828 | 25.00% |
| Folded clock inverter | 13.104 | 11.466 | 12.50% |
| Wordline driver | 54.600 | 54.600 | 0% |
| NOR2 | 6.552 | 6.552 | 0% |

Areas include the complete standalone frame and routing. Whole-column width
quantization prevents an area reduction in two cases. The decoder still uses
a long row of leaf instances; its area and aspect ratio remain the main row
integration limitation.

All nine revised component cases and ten leaf variants pass independent
Calibre DRC, LVS and distributed R+C+CC extraction. The
[new density and bank evidence](../../data/BANK_LAYOUT_V2_2_8.json) retains separate
hashes and receipts; the original component evidence above is preserved.

The initial `layout bank` entrance assembled and verified the **bank data path**:
the complete data array plus precharge, sense, write drivers and data buffers.
It exposed decoded wordlines and control signals. That stage did not integrate
the decoder, replica, TIME_CONTROL or output latch. The
[bank evidence](../../data/BANK_LAYOUT_V2_2_8.json) records the physical scope
and verification at that stage. The subsequent
[integrated bank](../../data/BANK_INTEGRATION_V2_2_8.json) adds row logic, replica circuitry
inside the core and the output latch, keeping TIME_CONTROL external. The
original data-path boundary is available with `--data-path-only`. No bank
waveform validation ran in either physical integration stage.

The [row-pitch correction](../../data/BANK_ROW_PITCH_V2_2_8.json) subsequently places the
decoder to the left of the core with exact 400 nm WL alignment. The
[executed OpenRAM reference](OPENRAM_REFERENCE_V2_2_8.md) records the generated
32×32 reference bank and its separate verification limits.

The [OpenRAM-pitch implementation](OPENRAM_PITCH_V2_2_8.md) adopts its
705 nm column / 1365 nm row pitch. Historical
[beside-controller evidence](../../data/TIME_CONTROL_PLANNED_INTERFACE_V2_2_8.json)
retains the earlier standard-cell integration. The current
[centre floorplan](LAYOUT_SCALING_V2_2_8.md#split-arrays-with-a-centre-periphery)
places TIME_CONTROL between the decoder halves; the opt-in
[compact cell](COMPACT_CELL_V2_2_8.md) re-pitches the bank around its thinner cell.
