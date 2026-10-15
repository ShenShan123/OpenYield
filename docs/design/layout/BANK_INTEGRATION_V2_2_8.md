# V2.2.8 bank integration with external TIME_CONTROL

This preserves the earlier below-core folded placement and its physical
evidence. The current default uses the
[corrected left-side, row-pitch decoder](BANK_ROW_PITCH_V2_2_8.md). The commands
below now generate that current floorplan; the old measured geometry remains
in the archived artifact directories and receipts identified here.

`layout bank` now includes the decoder, data and replica wordline drivers,
replica column and dummy loads, replica column periphery, and output latch.
TIME_CONTROL and its address/data input registers remain deferred by user
choice. Every other native circuit instance is included. Source topology,
models, W/L/NF and timing settings are unchanged.

This extends the independently recorded
[bank data-path stage](BANK_LAYOUT_V2_2_8.md). That earlier scope remains available
with `layout bank --data-path-only`; its evidence is preserved.

## Replica inside the core

`bank_core.py` builds one physical `CORE_WITH_REPLICA_<rows>X<cols>` hierarchy
containing the data cells, replica cells, tap strips, dummy load row and edges.
The replica column sits immediately to the right of the data columns, separated
by a 400 nm well spacer. It shares the 400 nm row pitch and row mirrors.
Every source passive replica row is retained, followed by the source's active
replica cell in the final row. The dummy load row contains exactly `cols - 1`
source cells above the top tap strip; its RWL joins the active replica row.

The first candidate's 155 nm separator violated opposite-well spacing at
peripheral strip corners. The final 400 nm spacer clears those corners without
changing transistor geometry. Both replica supply stripes receive physical
connections. Replica precharge, sense and write-load cells reuse the matching
column masters. The replica write-load DIN ties to VSS locally; a supply bus
across that entire DIN row would short the data-column inputs.

The manifest records each replica cell and dummy load with its source instance,
row/column position and mirror transform. Floating dummy PG drains are marked
as active diffusion access, not fictitious metal contacts. Physical gate count,
gate area and the complete four-terminal inventory are checked independently.

## Folded row logic and bank connections

`bank_rows.py` places unchanged row leaf cells on shelves. Each data wordline
driver follows the decoder leaf that supplies its input, keeping those local
connections short. Shelf channels reuse tracks when horizontal spans do not
overlap. Only connections between shelves and bank interfaces consume side
trunks, which also reuse tracks when their vertical spans do not overlap.
The cell bodies remain three core rows high.

`bank_complete.py` joins the folded row logic to the data/replica block through
named source nets. It connects every decoded wordline, RWL, supplies, sense
enable and the selected sense output to the output latch. Wordlines and RWL
are internal to the bank boundary. Registered `A_dff*` and `DIN_dff*`, `wl_en`,
`PRE`, `s_en`, `sa_iso` and `w_en` remain externally driven by TIME_CONTROL.
`RBL` is exposed for replica feedback; the output latch provides `OUT`/`OUT_B`.
The generated receipt explicitly records `controller_layout: false`.

## Physical verification

| Data array | Source MOS / physical gates | Terminal locations | Bounding box (µm) | Area (µm²) |
|---|---:|---:|---:|---:|
| 8×8 | 894 | 3,576 | 50.905 × 110.425 | 5,621.184625 |
| 32×32 | 8,182 | 32,728 | 63.590 × 306.425 | 19,485.565750 |

Both cases pass **zero Calibre DRC violations**, **LVS `CORRECT`**, and
**distributed R+C+CC extraction** against their independently generated source
schematics. The 32×32 row logic uses 41 shelves, 45 side tracks, and a
59.095 × 245.2 µm frame. The complete bank area includes those channels and
all replica/row/latch geometry; it is not directly comparable to the earlier
array/column-only footprint. Row-channel height remains a density limitation.

The 59-check local suite passes, including asymmetric 2×4 integrated generation,
real GDS gate area, exact terminal coverage, core hierarchy, unsupported replica
geometry rejection and external-control scope. After final metadata changes,
the 11 bank checks pass again. The exported row leaf variants receive independent
DRC/LVS/xRC receipts. Compilation and `git diff --check` pass.

[Tracked evidence](../../data/BANK_INTEGRATION_V2_2_8.json) binds each result to the
source deck/model, GDS, schematic, generator files, Calibre binary and rule
hashes. [The case protocol](../../../tests/spice/v228_bank_integration_cases.json)
remains tracked; regression/campaign Python stays local and ignored. Earlier
failed candidates and the prior data-path evidence remain preserved.

Native previews: [32×32 core](../../../outputs/layout/bank_integrated/final_32x32/core.png)
and [whole bank](../../../outputs/layout/bank_integrated/final_32x32/bank.png).
The core preview shows the data array, the replica column at the right, and
the dummy load row above the data cells.

## Reproduction and limits

Use the nominal mode-0 8×8 or 32×32 read/write source generation command in the
[layout tutorial](../../../sram_compiler/layout/README.md), with explicit illustrative
wiring, TT, 1 V, 25 °C and a 9 ns period. Set `DECK` to the resulting deck and
`PDK_DIR` to the installed FreePDK45 directory, then use new output directories:

```bash
python main_sram.py layout bank --deck "$DECK" \
  --output-dir outputs/layout/bank_integrated/generated
python main_sram.py layout verify-full \
  --layout-dir outputs/layout/bank_integrated/generated --pdk-dir "$PDK_DIR" \
  --output-dir outputs/layout/bank_integrated/verification
```

The new default includes every circuit instance except TIME_CONTROL. It requires
an unmuxed native 6T read/write source with the supported data/replica/dummy
geometry. Other sizes and source variants need independent physical checks.
No bank transient, read/write/hold/sense/recovery waveform comparison, yield
screen or RC-default promotion ran. A physical LVS/extraction pass does not
qualify dynamic bank operation or an autonomous controller.
