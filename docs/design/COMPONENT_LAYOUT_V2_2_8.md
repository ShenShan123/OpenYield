# V2.2.8 component layout development

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

OpenYield keeps its own frozen device sizes and finger counts. In particular,
OpenRAM's height-driven refingering and SRAM22's sizing policies are not adopted
as replacements for the existing schematic. Their technology cells and source
code are not imported into this generator.

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
composites use at most five separate M5 trunks inside one column. Row
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
Those access points are on M5 to avoid the cell's local wiring; a future bank
must connect them to the core metals explicitly. Odd columns must mirror the
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
and length increments. There is no full new bank integration, multi-column
mux view, TIME_CONTROL passive/controller view, cell characterization,
compact waveform qualification or compiler-wide RC/default promotion. The
existing reference bank, bitcells and evidence remain separate reproducible
references while the new cell library is developed.

## Reproduction and evidence

See [the layout tutorial](../../sram_compiler/layout/README.md#small-cells-and-component-placement).
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

The tracked [case protocol](../../tests/spice/v228_component_cases.json) and
[physical evidence](../data/COMPONENT_LAYOUT_V2_2_8.json) retain these exact
scopes and hashes. Raw artifacts and independent receipts remain under
`outputs/layout/cells/campaign_07/`; the
[native GDS preview](../../outputs/layout/cells/campaign_07/preview/components.png)
shows the cells and assembled driver without modifying their geometry.
Python regression and campaign scripts stay under ignored `tests/` and `dev/`.
No waveform comparison ran in this component stage and no yield or full-array
qualification is claimed.
