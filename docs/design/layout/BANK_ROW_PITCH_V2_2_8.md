# V2.2.8 left-side decoder and core row-pitch correction

The decoder and data wordline drivers now occupy a strip **entirely to the
left of the core**, with the same nominal data-row height. Every wordline
leaves that strip on M2 at `array_y + 200 + row * 400` nm and connects directly
to the matching core M2 wordline, without a vertical jog. The earlier folded
placement below the array did not meet this floorplan requirement and is
superseded. Its [evidence](BANK_INTEGRATION_V2_2_8.md) remains preserved.

The replica column, dummy loads, taps and replica feedback connections remain
inside the core hierarchy. The RWL driver and output latch occupy the auxiliary
region below the decoder. TIME_CONTROL and its input registers remain external,
as requested. The two new bank schematics have exactly the same SHA256 values
as the preceding integrated-bank stage: no device, W/L/NF, model or topology
change accompanies the floorplan correction.

## Placement and routing contract

The existing row leaf cells remain three core rows high. `bank_decoder.py`
packs these unchanged views horizontally into **eight-row bands**; the complete
decoder strip is exactly `rows * 400` nm high. This is a matched strip interface,
not a claim that each leaf cell fits a single 400 nm row. The current integrated
entrance requires a row count divisible by eight and at least eight. The older
`--data-path-only` entrance retains its earlier even-dimension support.

Leaf ownership follows actual wordline fanout. Native cascaded decoders also
contain unused predecode branches; all sixteen such leaf instances in the
32-row case remain physically present. No dead-device removal is used to fit
the row envelope. Placement records identify each leaf's consumed core rows
and band, with an empty row list for a retained unused branch.

An 80 nm routing grid with a 40 nm vertical phase accommodates the existing
leaf ports and exact 400 nm WL access. M3 obstructions preserve leaf-local
wiring; upper-metal access is reserved before routing global nets. Shared
address/control buses preferentially use M6 to keep lower-metal antenna areas
bounded. Clearance follows real metal/via widths, not the smaller search grid.
Nearest-tree Manhattan distance bounds the maze search without changing route
clearance or device geometry. Auxiliary connections use M6; each WL uses only
a short horizontal M2 link across the decoder/core boundary.

## Verified geometry

| Data array | Decoder strip (µm) | Complete bank (µm) | Area (µm²) | Source MOS / gates |
|---|---:|---:|---:|---:|
| 8×8 | 360.240 × 3.200 | 377.025 × 50.025 | 18,860.675625 | 894 |
| 32×32 | 457.920 × 12.800 | 507.465 × 59.625 | 30,257.600625 | 8,182 |

Both cases pass **zero Calibre DRC violations**, **LVS `CORRECT`**, and
**distributed R+C+CC extraction**. The active/poly gate inventory and all four
source terminal locations are retained. The 59-check local suite passes,
including an asymmetric 8×4 integrated build with explicit checks for decoder
side, total height, and every WL's layer and row phase. Compilation and
`git diff --check` pass.

The corrected pitch constraint makes the current generic leaf-based strip wide.
Bank area increases from 5,621.18 to 18,860.68 µm² at 8×8, and from 19,485.57 to
30,257.60 µm² at 32×32. This is a placement/pitch correction, not a density gain
over the superseded folded floorplan. Further density work needs more compact
row-cell physical views and local routing within the frozen source constraints.
The measured widths and full native GDS are exposed rather than hidden by a
cropped floorplan image.

The [recorded evidence](../../data/BANK_ROW_PITCH_V2_2_8.json) binds each receipt to
the actual source, GDS, generator, model, Calibre executable and rules. The
[case protocol](../../../tests/spice/v228_bank_row_pitch_cases.json) remains tracked;
local campaign and rendering helpers stay under ignored `dev/bank_pitch/`.
All failed and interrupted routing candidates remain unqualified.

## Review and reproduce

- [32×32 full native layout](../../../outputs/layout/bank_pitch/bank_32x32_04/bank.png)
- [32×32 direct wordline interface](../../../outputs/layout/bank_pitch/bank_32x32_04/wordline_interface.png)
- [32×32 GDS](../../../outputs/layout/bank_pitch/bank_32x32_04/full_macro.gds)
- [Generated OpenRAM bank reference](OPENRAM_REFERENCE_V2_2_8.md)

Run the existing nominal mode-0 source-generation command in the layout tutorial
with explicit illustrative wiring. Use new directories for the corrected bank:

```bash
python main_sram.py layout bank --deck "$DECK" \
  --output-dir outputs/layout/bank_pitch/generated
python main_sram.py layout verify-full \
  --layout-dir outputs/layout/bank_pitch/generated --pdk-dir "$PDK_DIR" \
  --output-dir outputs/layout/bank_pitch/verification
```

No transient, read/write/hold/sense/recovery waveform comparison or yield screen
ran for this floorplan. Extracted capacitance is not a timing or operating-point
qualification. No prior calibrated RC profile or compiler default is promoted
to this geometry.
