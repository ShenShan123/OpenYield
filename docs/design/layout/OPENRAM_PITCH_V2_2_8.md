# OpenRAM-pitch bank layout, V2.2.8

The current `layout bank` implementation uses **705 nm column pitch and
1365 nm row pitch**, as requested after generating the local OpenRAM reference.
The decoder sits to the left of the core, and each wordline connects directly
on M3 at `array_y + (row + 0.5) × 1365 nm`. The replica column and the source
replica load row are integrated into the core. TIME_CONTROL and its input
registers remain deferred.

## Implementation review and changes

The previous decoder used generic three-row leaves packed horizontally into
bands. Its 32-row strip was 457.920 µm wide at the previous 400 nm row pitch.
The replacement uses shared diffusion, exact physical refingering and local
row connections. Each final decode chain and wordline driver occupies one
1365 nm row. Local M3 buses and upper-metal address distribution retain every
source branch, including the unused predecode outputs.

OpenYield retains each transistor's total W, L, model and circuit connections.
Physical fingering changes only where needed to fit the frozen drive sizes.
The generator checks per-device width sums, physical gate count/area and all
four terminal accesses. Imported fixed views reject mismatched topology or
W/L before generating geometry.

The core derives from attributed, immutable OpenRAM FreePDK45 masters whose
six MOS sizes match OpenYield. Derived replica/passive/dummy wiring implements
the OpenYield source. The sense view separates ISO and EN and removes the
reference's output buffers to match the seven-MOS OpenYield circuit. The write
driver is source-owned geometry with its stronger original devices retained.
Its physical bounding box includes the contact overhang; the final area gate
caught and corrected an initially oversized landing.

Precharge and sense cells repeat on the 705 nm column pitch. Write drivers
and rotated data buffers use two-column tiles. This preserves the array and
bitline pitch without pretending that every peripheral leaf is one column
wide. Physical supply routing joins the blocks; verification uses strict
ports and does not substitute virtual supply connections.

## Measured area ceilings

Areas include recursive GDS geometry, contacts, wells, rails and routing;
nominal placement frames are not substituted for physical bounding boxes.
The reference is the recorded **32×32, one-port, one-bank OpenRAM run**.

| OpenYield view / region | OpenYield µm² | Reference ceiling µm² |
|---|---:|---:|
| Base inverter | 0.586950 | 1.142419 |
| NAND2 | 0.846300 | 1.450406 |
| NAND3 | 1.105650 | 1.944619 |
| Strong wordline NAND2 | 1.365000 | 1.450406 |
| Strong wordline inverter | 1.883700 | 3.112106 |
| Precharge | 0.990375 | 0.990375 |
| Sense amplifier | 3.475875 | 4.657750 |
| Write driver | 3.504638 | 3.507000 |
| Two-inverter data buffer | 1.173900 | 2.284838 |
| Replica driver, 32×32 source | 3.248700 | 4.404938 |
| Output latch | 4.019925 | 7.636200 |
| 32×32 data/replica/dummy core | 1093.119200 | 1277.478125 |
| Decoder region | 354.681600 | 389.372963 |
| Wordline-driver region | 105.487200 | 176.856550 |
| Complete 32×32 bank, external controller | **2106.700888** | **2457.493100** |

The complete bank is **14.27% smaller** than the reference bank. Decoder and
wordline-driver areas partition the complete row-logic bounding box at their
placement boundary, including routing overhead. The buffer ceiling is the sum
of two reference inverter bounding boxes. The latch uses the reference DFF as
an area ceiling, not an assertion of identical storage topology. Peripheral
rows use different packing and circuits; the table compares the named views
and regions, not a transistor-for-transistor replacement of every OpenRAM
hierarchy block. No matched OpenRAM 8×8 bank reference was generated.

The machine-readable bounds and reference GDS hash are in
[`area_reference.json`](../../../sram_compiler/layout/freepdk45/assets/openram/area_reference.json).
The generator rejects an oversized named view or 32×32 core/decoder/bank.
The upstream BSD-3-Clause license and asset hashes remain with the assets.

## Verification and artifacts

Both **8×8** and **32×32** banks pass Calibre DRC with zero violations, strict
LVS `CORRECT`, and distributed R+C+CC extraction. They retain respectively
**894 / 8182 logical MOS** and **912 / 8644 physical fingers**. The 8×8 physical
area is **369.782538 µm²**. The extra fingers preserve the total source widths.

- [32×32 GDS](../../../outputs/layout/openram_pitch/bank_32x32_01/full_macro.gds)
- [32×32 preview](../../../outputs/layout/openram_pitch/bank_32x32_01/bank.png)
- [Wordline interface close-up](../../../outputs/layout/openram_pitch/bank_32x32_01/wordline_interface.png)
- [32×32 physical receipt](../../../outputs/layout/openram_pitch/bank_32x32_01_verify/receipt.json)
- [8×8 GDS](../../../outputs/layout/openram_pitch/bank_8x8_01/full_macro.gds)
- [8×8 physical receipt](../../../outputs/layout/openram_pitch/bank_8x8_01_verify/receipt.json)
- [Tracked evidence summary](../../data/OPENRAM_PITCH_V2_2_8.json)
- [Tracked case protocol](../../../tests/spice/v228_openram_pitch_cases.json)

Twenty-nine local regression checks pass: 22 existing component/bank checks
and seven new pitch, source-conservation, area and receipt checks.

Generated artifacts remain under ignored `outputs/`; local regression scripts
remain under ignored `tests/` and `dev/`. The default CLI exercises the new
physical plans. `layout bank --legacy-pitch` retains the earlier integrated
bank, and `--data-path-only` retains the earlier data-path-only flow.

The physical scope is limited to the 8×8 and 32×32 plans. No waveform validation
ran for the new pitch or refingering. Frozen compiler sizing/timing, illustrative
source wiring, calibrated RC defaults and release qualification are unchanged.
The OpenRAM reference's bank check uses its documented adapter policy; its full
macro did not pass LVS. See [the reference record](OPENRAM_REFERENCE_V2_2_8.md)
for those separate verification limits.
