# FreePDK45 compact bank, component and TIME_CONTROL layouts

`layout bank` defaults to the compact **1125 nm column / 395 nm row** 6T
cell, with two full-row column halves beside one shared decoder and a 2:1
column mux. `--compact-cell --split-columns` remain accepted compatibility
flags. This is the only bank floorplan. Source sizing, timing and simulation
RC profiles remain frozen; physical checks do not establish waveform
correctness or release qualification.

## Compact column-split bank

Use the project's `openyield` Conda environment. Generate a nominal mode-0
source deck with the column mux enabled, then set `DECK` to its `deck.sp`:

```bash
python main_sram.py simulate --rows 128 --cols 128 --operation 'read&write' \
  --variation-mode nominal --real-cell-mode 0 --period 30e-9 --no-run-xyce \
  --compact-control --no-w-rc --column-mux \
  --sizing-lookup sram_compiler/sizing/layout_sizing_lookup.json \
  --output-dir outputs/layout/compact/source_128x128
python main_sram.py layout bank --deck "$DECK" \
  --output-dir outputs/layout/compact/bank_128x128
python main_sram.py layout bank --with-controller --deck "$DECK" \
  --output-dir outputs/layout/compact/macro_128x128
python main_sram.py layout verify-full --layout-dir outputs/layout/compact/macro_128x128 \
  --pdk-dir "$PDK_DIR" --output-dir outputs/layout/compact/macro_128x128_verify
python main_sram.py layout preview --layout-dir outputs/layout/compact/macro_128x128
```

Output directories must be new. Generation requires KLayout; verification
requires installed Calibre and an external FreePDK45 kit at `PDK_DIR`.
Banks accept 8–512 rows in multiples of eight and 4–512 columns in multiples
of four. Controller integration requires at least 16 rows. Automatic timing
retains the 512×256 envelope; 512×512 needs an explicit diagnostic period.

Each half has its own 2250 nm column-pair periphery below it: three-row
write and read/mux/sense cells, followed by one-row mirrored precharge.
The right half owns the replica column; all RWL loads and source devices
are retained. The external-control bank uses M1–M6 and excludes TIME_CONTROL,
the RWL driver and output latch. `SEL0`/`SEL1` are bank inputs.

`--with-controller` includes those circuits and one data register per
physical column. TIME_CONTROL sits below the decoder, aligned with the
register-band bottoms. Large controllers use spare decoder rows before
widening the central gap. Controller routing and input escapes reach M10;
OUT/OUT_B leave right on M4 and supply trunks reach top/bottom on M6.

Compact decoder leaves abut like implants; wide WL NAND drivers use fewer,
wider fingers at unchanged total W/L. Minimum decoder-to-array well clearance
is 225 nm. The controller fit search uses the 100 nm routing grid, reserves
500 nm for escapes and keeps a 20.6 µm minimum frame. Integrated controllers
can require a larger right-side gap than the external-control bank.

Controller logic leaves are 55 nm narrower at unchanged source W/L and gate
pitch. Their compact DFFs shrink from 5.710 to 5.325 µm; data-column register
slots retain their geometry. Last-input escapes lift at the native pin
position before jogging onto the routing grid on M4. Smaller controller cells
do not necessarily reduce macro width when the decoder sets that width.

The [compact design record](../../docs/design/layout/COMPACT_CELL_V2_2_8.md)
contains cell rules, module responsibilities, current area measurements,
physical verification scope and limits. The latest five complete macros
pass DRC 0/LVS CORRECT; the latest 256×256 and 512×512 geometries have no
R+C+CC extraction. No compact waveform or operating point is qualified.
The additional 16×4 controller routing screen did not complete; it has no
DRC/LVS receipt. The five-size physical evidence does not qualify that case.

The latest [read/write periphery](../../docs/design/layout/PERIPHERY_HEIGHT_V2_2_8.md)
uses two write rows, one shared select-inverter output-bar level, and abutting
write/read VSS rails. Complete macros are 1.290 µm shorter. Controller frame
widening can offset the area saving; the linked record reports both bounds
and area. Decoder antenna taps are checked for well clearance before routing.

Alternative bank implementations and their CLI selectors have been removed:
`--data-path-only`, `--legacy-pitch`, `--controller-floorplan`,
`--controller-layout-dir`, and `layout controlled-bank`.

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
profile or establish waveform correctness. See the [compact bank record](../../docs/design/layout/COMPACT_CELL_V2_2_8.md)
for current bank geometry and verification limits.

## Standalone TIME_CONTROL and analysis tools

`layout time-control` generates the standalone source controller.
`simulate --compact-control` selects the existing factored digital controller;
its source devices and timing are independent of the bank CLI defaults.

`layout preview --layout-dir PATH` renders actual GDS routing, placement and
TIME_CONTROL views. `--region=X0,Y0,X1,Y1` (µm) and `--name FILE.png` select a
window. `layout driver-audit` measures metal loads and gate widths after
matching DRC/LVS/coupled-extraction receipts.

The retained `verify-full`, `reduce-rc`, `split-rc`, `compare-full`,
`compare-snm` and `waveform-metrics` commands support physical verification
and RC analysis. `layout --help` lists available commands. Compiler reference
RC under `config_yaml/rc_profiles/` retains its original scope and must not
be applied to newly generated compact geometry.
