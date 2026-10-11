# V2.2.8 compact layout generation

V2.2.8 adds compact FreePDK45 macro placement while keeping the V2.2.7
transistor topology, sizing, fingers, intentional passives and timing policy.
The dense mirrored 6T bitcell remains unchanged. Historical RC profiles and
their physical/waveform receipts keep their original geometry and scope.

## Implementation references

The implementation scan used these pinned public trees:

| Reference | Commit | Inspected surface |
|---|---|---|
| [OpenRAM](https://github.com/VLSIDA/OpenRAM/tree/b2b069ce119d1488cbe6883b2240bceb5c7ce29a) | `b2b069ce119d1488cbe6883b2240bceb5c7ce29a` | Bank/port placement, bitcell arrays, precharge, sense/write arrays, wordline drivers and parameterized gates |
| [Nangate45 Synopsys enablement](https://github.com/ABKGroup/NanGate45-Synopsys-Enablement/tree/fe37a4a1387f319ce110450c0d13109a0e3a1fea) | `fe37a4a1387f319ce110450c0d13109a0e3a1fea` | Technology LEF and standard-cell LEF pins, sites and power rails |
| [Nangate45 GDS platform](https://github.com/rbarzic/platform_nangate45/tree/992137b98949d875622155730d63227c7e68dac1) | `992137b98949d875622155730d63227c7e68dac1` | Actual inverter, NAND2, NOR2, buffer and DFF diffusion/poly/contact/M1 geometry |

OpenRAM's [data port](https://github.com/VLSIDA/OpenRAM/blob/b2b069ce119d1488cbe6883b2240bceb5c7ce29a/compiler/modules/port_data.py)
stacks repeated peripheral arrays by their physical dimensions. Its
[precharge array](https://github.com/VLSIDA/OpenRAM/blob/b2b069ce119d1488cbe6883b2240bceb5c7ce29a/compiler/modules/precharge_array.py)
uses explicit column offsets and bitcell-compatible mirroring; sense/write
arrays follow the same pattern. The address port places the decoder beside
the wordline-driver array. These suggest preserving subcircuit boundaries,
reusing matching geometry and avoiding a separate oversized global channel
for each repeated column.

Nangate's [standard-cell LEF](https://github.com/ABKGroup/NanGate45-Synopsys-Enablement/blob/fe37a4a1387f319ce110450c0d13109a0e3a1fea/NanGate45/lef/NangateOpenCellLibrary.macro.mod.lef)
uses a 1.4 µm cell height, mirrored placement and abutting M1 supply rails.
Selected nominal frame areas are 0.532 µm² for INV_X1, 0.798 µm² for
BUF_X1/NAND2_X1/NOR2_X1 and 4.522 µm² for DFF_X1. These are library frame
areas, not OpenYield area estimates. Well/implant geometry extends outside
those frames; measuring all-layer GDS bounding boxes gives larger numbers.

Inspection of the [actual GDS](https://github.com/rbarzic/platform_nangate45/blob/992137b98949d875622155730d63227c7e68dac1/gds/NangateOpenCellLibrary.gds)
confirms two diffusion strips and shared diffusion between adjacent gates in
NAND2_X1/NOR2_X1, with local M1 connections. Their 50 nm gate lengths match
this reference, but INV_X1/NAND2_X1 gate intersections give 415 nm NMOS and
630 nm PMOS widths. OpenYield's actual compiler sizes differ. A direct
library-cell substitution would therefore change the frozen circuit.
The repositories are inspection references; their code and library cells
are not bundled or imported into the runtime generator.

## Changes

`placement.py` keeps complete common subcircuits together, including DFF and
driver child gates, and groups the remaining direct MOS by leaf subcircuit.
It caches a GDS template
for each matching model/width/length/finger-layout sequence. This covers
inverters, transmission gates, NAND/NOR gates, precharge, sense amplifiers,
write drivers and their component gates. Every source finger retains its
individual placement and four-terminal map. Electrical connectivity is
routed from the current source, independently of geometry reuse.

Decreasing-height first-fit shelves reduce space lost when one tall MOS sets
the height of a whole flat placement row. The body pin shares the source
access row on its independent M3 track; gate and drain retain staggered
access. Active, wells, poly, contacts and lower-metal device straps are
unchanged. Explicit body contacts and 560 nm row clearance remain.

The generic compact macro halves the left placement margin and reuses eight
staggered bitline escape tracks across columns. Adjacent columns use different
tracks. The channel no longer grows by two escape rows for every added
column. The fixed 8×8 compact variant retains its existing pitch-aligned
row/column periphery and uses twelve repeated data/sense escape tracks.

M4 pin escapes are reserved before routing. Width-aware clearance protects
190 nm supply trunks near the actual M3 pin-access stubs while retaining
space for narrower signals. Short local nets route early.
Same-net diagonal landings are joined to satisfy square corner-spacing rules;
fixed off-grid array/feedback landings retain explicit clearance and avoid
nearby duplicate via3 cuts.
When an earlier net encloses a terminal, bounded route-order negotiation
restores all physical obstacles and reservations and routes the enclosed net
before its blockers. No DRC spacing, via clearance or circuit constraint is
relaxed. Successful retry history is recorded in the layout manifest.

The original `layout reference`, `layout full` and `layout scaled` defaults
remain reproducible. `layout compact` and `layout scaled --floorplan compact`
select generic compaction; `layout full --floorplan compact` compacts the
fixed 8×8 reference. Optional positional arguments are appended.

The manifest records actual macro area, bounding box, stored bits per µm²,
source and generator hashes, reused subcircuit placements, and routing retries.
Area includes physical routes, passives and supply buses. Generation alone
is marked unverified; official verification uses a separate receipt.

## Validation scope

The final standalone fixed 8×8 generator uses 32 placement columns and the
existing pitch-aligned array/row/column periphery. All 1,692 source MOS and
854 controller/latch fingers remain. Its 239 control subcircuit instances
reuse 49 geometry templates. The rendered source matches the historical
`controller/full_build_11` source byte for byte; fresh V2.2.8 CLI generation
also matches its complete MOS and intentional-passive inventory.

| Actual complete macro | Reference | Compact |
|---|---:|---:|
| Bounding box, µm | 62.975 × 126.350 | 63.2325 × 103.950 |
| Area including routes, passives and supply buses, µm² | 7,956.89125 | 6,573.018375 |
| Stored bits per µm² | 0.00804334 | 0.00973677 |

Area falls **17.3921%** and density rises **21.0538%**. These are matched
8×8 physical-layout measurements, not an OpenRAM/Nangate performance benchmark.
The [native GDS comparison](../../outputs/layout/V2.2.8/preview_final/comparison.png)
uses the same spatial scale and preserves the geometry.

The final `full_compact_13` standalone build passes official Calibre DRC with
**zero violations**, LVS **`CORRECT`**, and complete distributed **R+C+CC**
extraction of 709 nets. Source/GDS/DSPF, rule, generator and tool hashes,
precise scope and area values are preserved in the
[physical evidence](../data/LAYOUT_COMPACTION_V2_2_8.json).
The intermediate renderer replay matches the final geometry on every layer;
the accepted receipt is bound to the standalone CLI build.

The generic compact 8×8/16×16 campaigns have separate routing attempts and
are not certified by this fixed-macro receipt. Failed/interrupted attempts
remain unqualified. Every additional size or placement setting needs its own
complete physical and waveform evidence.

Local Python regression checks cover geometry reuse, width-dependent cache
identity, complete folded-device coverage, grid alignment, nonoverlapping
placement, unchanged transistor/contact geometry, blocker identification,
restoration of routing obstacles, existing supply-stack protection and RC
invariants. Python runners remain ignored; the tracked
[`v228_layout_cases.json`](../../tests/spice/v228_layout_cases.json) preserves
the physical comparison protocol.

All **40** local layout, routing-clearance, calibrated-RC, configuration and
unified-entrance checks pass. No compact waveform comparison ran, and no
new RC profile or compiler-wide default was promoted.

New compact geometry requires its own official DRC/LVS and extraction receipts.
Existing calibrated compiler defaults continue to represent their historical
layouts. No compact waveform, full-release or yield qualification is implied.
Shared-diffusion parameterized logic cells, full standard-cell rail abutment,
and pitch-aligned arbitrary-size periphery remain further density work.
