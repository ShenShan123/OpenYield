# V2.2.8 component-based bank data path

This is the preserved data-path-stage record. The current `layout bank` default
adds the missing components described in [the integration record](BANK_INTEGRATION_V2_2_8.md).
Use `--data-path-only` to reproduce this earlier physical boundary.

`main_sram.py layout bank` integrates the full dense 6T data array with repeated
precharge, sense-amplifier, write-driver and write-data-buffer components.
This is the next physical integration stage after the
[component density review](COMPONENT_LAYOUT_V2_2_8.md#density-and-bank-integration-review).
It retains source sizing, device fingers, topology, timing and model parameters.

## Physical boundary

The generated schematic exposes VDD/VSS, decoded `WL*`, `PRE`, `s_en`,
`sa_iso`, `w_en`, registered `DIN_dff*` and `SA_Q*` / `SA_QB*`. Decoder,
replica column and loads, TIME_CONTROL and output latch remain outside this
boundary, with their exact compiler instance names recorded as exclusions.
The current standalone decoder is still 226.59 µm wide: folding and integrating
that row logic is a subsequent density task. Full controller layout remains
deferred. The data-path result must not be described as a complete SRAM bank.

The source contract checks the actual bitcell topology, placement connectivity,
models, W/L/NF, full cell inventory, and each column's BL/BLB and buffer-to-driver
connections. It rejects muxed columns, changed bitcells, nonuniform peripheral
masters, tied internal bitlines, and intentional passives. It derives the bank
schematic from the selected source hierarchy rather than a hard-coded transistor
list. RC reduction removes only the compiler's recognized illustrative wiring.

The physical implementation preserves the 1365 nm column and 400 nm row pitches.
Columns stack data buffers, write drivers, sense amplifiers and precharge below
the core. Odd columns reflect the cell geometry, boundary pins and all device
terminals together. M5 bitline spines meet explicit M3/M4/M5 landings at the core
tap strip. M4 buses connect column control and supply interfaces; buffer supply
buses use M6 above the composite's existing routing. Side M5 trunks join buses
across stages. Every core VDD/VSS stripe receives a physical metal connection.

The first assembled attempt exposed mirrored buffer-trunk and local-M1 spacing
violations, plus a gap at alternating core ground tap edges. The final cell
library reuses nonoverlapping trunk spans, reserves boundary landing clearance,
and the bank extends each supply feed into the continuous core stripe. The
failed attempt remains separate and unqualified.

## Verified layouts

| Data array | Source MOS / physical gates | Terminal locations | Bounding box (µm) | Area (µm²) | Bits/µm² |
|---|---:|---:|---:|---:|---:|
| 8×8 | 592 | 2,368 | 14.05 × 49.60 | 696.880 | 0.09184 |
| 32×32 | 6,976 | 27,904 | 46.81 × 59.20 | 2,771.152 | 0.36952 |

Area includes the entire bank frame, routing and supply buses. The core stores
64 and 1,024 bits respectively. Peripheral component masters are identical
between the two sizes; the larger array amortizes their height. These numbers
are not area comparisons against a complete macro containing decoder/controller
logic, nor predictions for other array sizes.

Both sizes pass **zero Calibre DRC violations**, **LVS `CORRECT`**, and
**distributed R+C+CC extraction**. The independent active/poly intersection
count and area match the exact selected source devices; all four terminals of
every finger are recorded. Nine revised component cases and ten leaf variants
also pass DRC/LVS/xRC independently. The local 54-check suite passes, including
an asymmetric 2×4 generation case; that 2×4 case has no official physical receipt.
After the final source-contract guards, the six bank regressions pass again.
Compilation and `git diff --check` pass.

The tracked [physical evidence](../../data/BANK_LAYOUT_V2_2_8.json) binds source,
GDS, generator, model, Calibre binary and external rules to the results. The
[case protocol](../../../tests/spice/v228_bank_cases.json) remains usable in a fresh
clone; local Python regression/campaign scripts remain ignored. Original
component evidence is preserved unchanged.

Native GDS previews and raw artifacts remain under
`outputs/layout/bank/final/` and `outputs/layout/bank/large_32x32_01/`.
The [32×32 preview](../../../outputs/layout/bank/large_32x32_01/bank.png) shows actual
geometry, with the data array above the repeated column periphery.

## Reproduction

Generate a fresh native source for each size; use the same explicit nominal
corner, voltage, temperature, period and illustrative wiring configuration:

```bash
python main_sram.py simulate --no-run-xyce --rows 32 --cols 32 \
  --operation 'read&write' --variation-mode nominal --real-cell-mode 0 \
  --corner TT --vdd 1 --temperature 25 --period 9e-9 --no-waveform \
  --interconnect-config sram_compiler/config_yaml/interconnect_illustrative.yaml \
  --output-dir outputs/layout/bank/source_32x32
```

Set `DECK` to the resulting `deck.sp` and `PDK_DIR` to the external FreePDK45
installation. Use new output directories for every attempt:

```bash
python main_sram.py layout bank --data-path-only --deck "$DECK" \
  --output-dir outputs/layout/bank/generated_32x32
python main_sram.py layout verify-full \
  --layout-dir outputs/layout/bank/generated_32x32 --pdk-dir "$PDK_DIR" \
  --output-dir outputs/layout/bank/verified_32x32
```

No Xyce transient, read/write/hold/sense/recovery waveform comparison, or yield
screen ran in this work. Physical LVS does not establish dynamic correctness.
No calibrated RC profile, compiler-wide default or release qualification is
promoted. Decoder and replica integration, followed by bank waveform validation,
remain necessary before claiming a complete operational bank.
