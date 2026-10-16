# FreePDK45 component, bank and TIME_CONTROL layouts

The component-based flow generates source-matched cells, banks and controllers.
Circuit sizing and timing stay fixed. Each physical boundary requires its own
DRC/LVS/extraction and waveform evidence.

## Small cells and component placement

`layout cells` is the first stage of the component-based FreePDK45 flow. It
reads a selected compiler subcircuit with its original ordered ports and
numeric W/L/NF, builds reusable small cells, then places and routes those cells
through their ports. Each small cell owns diffusion, body taps, contacts,
local M1/M2/M3 wiring and a physical pin/obstruction abstract. Adjacent fingers
share diffusion only when their geometry, body and connected terminal match.
No transistor is resized or refingered to meet a physical frame.

Column peripherals require the column axis and occupy exactly one **1365 nm**
core column. Precharge, sense and write-driver bitline access pins follow the
core's BL/BLB x coordinates; mirror their view with the corresponding array
column. Row logic uses a **three-row, 1200 nm** leaf frame, with explicit larger
row spans for component routing. Widths and origins use the core column grid;
row channels use the **400 nm** core row pitch. Wordline and decoder output
access points preserve the core wordline phase, `200 + row * 400` nm. The
electrical terminal map follows every physical transform.

Start with a small NAND2, then assemble its wordline driver from that NAND2
and an inverter using the same native nominal mode-0 `DECK`:

```bash
python main_sram.py layout cells --deck "$DECK" --axis row \
  --instance XWL_DRV_0/XPNAND2 --output-dir outputs/layout/cells/nand2
python main_sram.py layout cells --deck "$DECK" --axis row \
  --instance XWL_DRV_0 --output-dir outputs/layout/cells/wordline
python main_sram.py layout cells --deck "$DECK" --axis column \
  --instance XPRECHARGE_0 --output-dir outputs/layout/cells/precharge
python main_sram.py layout verify-full \
  --layout-dir outputs/layout/cells/wordline --pdk-dir "$PDK_DIR" \
  --output-dir outputs/layout/cells/wordline_verification
```

Other native column instances are `XSENSEAMP_0`, `XWRITEDRIVER_0` and
`XDIN_BUF_0`. Every output must be new. `cells/` contains independently
verifiable leaf GDS/schematic/abstract bundles; `library.json` catalogs them.
The root bundle contains the placed cell instances and their connections.
`attempt.json` preserves failed and interrupted builds. Geometry and library
entries stay `generated_unverified`; physical verification writes separate,
scope-specific receipts. Source, model/configuration provenance and generator
hashes are retained. Intentional passives and unsupported physical views fail
explicitly. A one-column composite reuses at most four upper-metal tracks;
disjoint vertical spans share a track while retaining mirrored edge clearance.

Standalone components have their own verification scope. Larger row spans
do not claim that cells abut a one-row array slice; bank and controller
integration are described below. Physical checks do not qualify a new RC/default
profile or establish waveform correctness. See the
[component design record](../../docs/design/layout/COMPONENT_LAYOUT_V2_2_8.md) for the
OpenRAM/SRAM22 source comparison, actual checks and physical limitations.

## OpenRAM-pitch bank

`layout bank` now defaults to the source-preserving OpenRAM-pitch implementation:
**705 nm columns and 1365 nm rows**, a decoder on the left, and a replica column
inside the core. Physical generation accepts up to a single 512×512 bank;
verified artifacts and their limits are recorded separately. Every source
MOS except TIME_CONTROL is retained. Physical fingers may change while total
W/L, topology, sizing policy and timing remain unchanged.

```bash
python main_sram.py layout bank --deck "$DECK" --output-dir outputs/layout/openram_pitch/generated
python main_sram.py layout verify-full --layout-dir outputs/layout/openram_pitch/generated \
  --pdk-dir "$PDK_DIR" --output-dir outputs/layout/openram_pitch/verification
```

The generator checks actual GDS bounding-box areas against recorded OpenRAM
cell and 32×32 block ceilings. It also checks the complete physical finger and
terminal inventory. The [current record](../../docs/design/layout/OPENRAM_PITCH_V2_2_8.md)
contains the measured areas, strict DRC/LVS/xRC evidence, asset attribution and
comparison limits. TIME_CONTROL and its input registers remain external in
this default flow; no waveform or RC-default qualification is implied.

### Optional TIME_CONTROL integration

`layout bank --with-controller` adds TIME_CONTROL and its input registers.
The default `--controller-floorplan center` is described under the single-bank
section below. For the 32×32 physical plan, `--controller-floorplan beside`
follows OpenRAM's floorplan:

- Control logic sits left of the decoder, with all bank-facing pins on its
  right edge.
- The 32 data registers form a pitch-planned array below the column periphery.
- Every interface net is an explicit one- to three-segment track; there is no
  maze search. Drives are straight M6 lines into the bank's control trunks,
  far-end returns are straight M7 lines, and the data bus is a left-edge
  M3/M4 channel.

The historical design used reusable OpenYield standard-cell templates with frozen source
W/L and topology, plus four custom RC guards. The recorded export
(`planned_02`) is 4050.6731 µm², 29.7% smaller than the maze-routed `full_40`.
It passes zero Calibre DRC violations, strict LVS `CORRECT` and distributed
R+C+CC extraction; all four guard resistances are independently checked. See
the [controller layout record](../../docs/design/layout/TIME_CONTROL_LAYOUT_V2_2_8.md)
for the floorplan, routing comparison, verification and limitations.

```bash
python main_sram.py layout bank --with-controller --controller-floorplan beside \
  --deck "$DECK" --output-dir outputs/layout/time_control/generated
python main_sram.py layout verify-full --layout-dir outputs/layout/time_control/generated \
  --pdk-dir "$PDK_DIR" --output-dir outputs/layout/time_control/generated_verify
```

`layout time-control` generates the standalone controller; the historical
RC-based export was 838.7843625 µm². `layout controlled-bank` is the direct complete-bank command.
Output directories must be new.

### Single 512×512 bank and compact digital control

Current TIME_CONTROL source contains no intentional RC guards or decoder
settling capacitors. Far-end feedback and the existing short digital delays
remain. `simulate --compact-control` also factors isolation logic and removes
the unused sense-enable complement inverter and obsolete sizing loads.

Use `--sizing-lookup sram_compiler/sizing/layout_sizing_lookup.json` for the
explicit driver candidate derived from the verified 32×32 bank's coupled RC
extraction. Automatic timing retains the 512×256 envelope; an explicit
`--period` permits diagnostic source generation through 512×512. No operating
point or release qualification is implied. See the
[scaling record](../../docs/design/layout/LAYOUT_SCALING_V2_2_8.md) for commands, sizes,
waveform evidence and actual layout verification.

`layout bank --with-controller` (16 rows or more; `--controller-floorplan
center` is the default) splits the core into lower and upper sub-arrays around
the column periphery. Bitlines stay continuous on M5 through the periphery, and decoder
nets used by both halves are bridged on M10. TIME_CONTROL starts at the decoder's
left edge, fills the block between the decoder halves and continues in spare
decoder rows. The data registers fold into a 7.73 µm band directly above the
lower array. The earlier `corner` floorplan was retired on 2026-10-16, so banks
below 16 rows have no integrated controller layout; `beside` remains a
32×32-only plan. See the
[scaling record](../../docs/design/layout/LAYOUT_SCALING_V2_2_8.md) for areas,
verification and limits.

`layout preview --layout-dir PATH` produces `layout.png`, `placement.png` and
`time_control.png` directly from the GDS. `layout driver-audit` measures metal
loads and gate widths only after hash-bound, passing DRC/LVS/coupled extraction.

## Integrated bank with external control (historical pitch)

`layout bank --legacy-pitch` includes the decoder, data and replica wordline drivers,
output latch, and all replica circuitry. The replica column and dummy load row
are inside the dense core hierarchy, with matching row pitch, taps and physical
supply connections. The decoder and data wordline drivers sit entirely to the
left of the core, span its data-row height, and join every WL directly on M2
at 400 nm pitch. Unchanged leaf cells occupy horizontal eight-row bands; this
integrated floorplan requires at least eight rows and a multiple of eight.
RWL/output-latch auxiliary logic remains below the decoder.
TIME_CONTROL and its input registers remain external by user choice; every
other native circuit instance is included.

```bash
python main_sram.py layout bank --legacy-pitch --deck "$DECK" \
  --output-dir outputs/layout/bank_integrated/generated
python main_sram.py layout verify-full \
  --layout-dir outputs/layout/bank_integrated/generated --pdk-dir "$PDK_DIR" \
  --output-dir outputs/layout/bank_integrated/verification
```

The 8×8 and 32×32 integrated banks pass DRC/LVS/R+C+CC. These are physical
checks, with no new waveform qualification or default promotion. See the
[row-pitch record](../../docs/design/layout/BANK_ROW_PITCH_V2_2_8.md) for geometry,
source coverage, dimensions, receipts and native previews. The generic leaves
currently produce a wide decoder strip; the measured area increase is recorded.
The [executed OpenRAM reference](../../docs/design/layout/OPENRAM_REFERENCE_V2_2_8.md)
provides a separate 32×32 bank GDS and physical-check scope for comparison.

## Component-based bank data path

`layout bank --data-path-only` retains the earlier stage: it tiles the complete
dense 6T data array with reusable precharge,
sense, write-driver and write-data-buffer components. Odd columns mirror both
geometry and terminal maps. M5 bitlines connect to core M3 through explicit
M4 landings; M4 control/supply buses and M6 buffer supply buses join the cells.
The sizing and timing from the supplied compiler source remain fixed.

```bash
python main_sram.py layout bank --data-path-only --deck "$DECK" \
  --output-dir outputs/layout/bank/generated
python main_sram.py layout verify-full \
  --layout-dir outputs/layout/bank/generated --pdk-dir "$PDK_DIR" \
  --output-dir outputs/layout/bank/verification
```

Use a native nominal, mode-0 6T deck with explicit illustrative wiring using
`main_sram.py simulate --variation-mode nominal --real-cell-mode 0 --operation
read&write --no-run-xyce --interconnect-config
sram_compiler/config_yaml/interconnect_illustrative.yaml`. The dense tile requires even dimensions of
at least 2×2 and its exact device geometry/connectivity. The generator rejects
changed bitcells, muxed columns, mismatched peripheral variants, and unsupported
connections. Each size and source requires its own physical verification.

This entrance implements the **bank data path**, with decoded `WL*`, `PRE`,
`s_en`, `sa_iso`, `w_en`, and registered `DIN_dff*` as inputs, and `SA_Q*` /
`SA_QB*` as outputs. Decoder, replica, TIME_CONTROL and output latch are recorded
as excluded instances. The hierarchy and verification receipt preserve this
boundary; they do not describe a complete autonomous SRAM bank.

`layout.json` records every source MOS and finger terminal, physical gate
count/area, transformed instance pins, full bounding-box area, density, and
source/generator provenance. `cells/` exports independently verifiable masters.
Generated geometry stays unverified until a separate receipt completes DRC,
LVS and R+C+CC extraction. Failed attempts remain in their own directories.
See the [bank record](../../docs/design/layout/BANK_LAYOUT_V2_2_8.md) and
[density review](../../docs/design/layout/COMPONENT_LAYOUT_V2_2_8.md#density-and-bank-integration-review)
for measurements and exact validation scope. No bank waveform comparison or
RC-default promotion is implied by these physical checks.

## Generate a source deck

Run from the repository root in the project Conda environment. Layout generation
requires the KLayout Python library; verification requires installed Calibre
and an external FreePDK45 kit selected by `PDK_DIR`.

```bash
python main_sram.py simulate --rows 8 --cols 8 --operation 'read&write' \
  --variation-mode nominal --real-cell-mode 0 --no-run-xyce \
  --interconnect-config sram_compiler/config_yaml/interconnect_illustrative.yaml \
  --output-dir outputs/layout/source_8x8
```

Set `DECK` to the emitted `deck.sp`. Use fresh output directories for generation
and verification. The retained `verify-full`, `reduce-rc`, `split-rc`,
`compare-full`, `compare-snm` and `waveform-metrics` tools support physical
verification and RC analysis. `layout --help` lists the current commands.

The older fixed reference, full, scaled and compact macro generators, their
calibration campaigns and their outputs have been removed. The compiler's
scoped reference RC data remains under `config_yaml/rc_profiles/`; it describes
its original geometry and must not be applied to a newly generated bank.
