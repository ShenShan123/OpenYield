# Single-bank layout scaling and driver candidates — V2.2.8

The layout generator now accepts a single 512×512 bank at the existing
705 nm column and 1365 nm row pitch. TIME_CONTROL and the data registers are
included in the generated corner floorplan. An explicit diagnostic clock is
required beyond the automatic 512×256 timing envelope; no new operating point
or release qualification is implied.

## Controller

The current source deletes the wordline-setup resistor/capacitor, the three
observer RC filters, and the tall-decoder settling capacitors. Physical bank
interconnect remains distributed RC. Far-end PRE, isolation, enable and replica
wordline observers remain, together with the existing short digital delay chains.
Legacy tau arguments remain accepted for call compatibility and do not insert
electrical elements or additional stages.

The archived automatic 8×8 RC reference contains the earlier controller.
It is declined when source topology differs, with the reason recorded in
`variation_summary.calibrated_rc`; an explicitly requested stale profile
still raises an error. Native wiring is illustrative in that case.

`compact_control=True` factors isolation completion into two NAND gates,
removes the unused sense-enable complement inverter, and removes obsolete
hold-latch and complement-inverter loads from sizing. The external controller
port contract remains unchanged. Driver snapshots from the previous RC
architecture are rejected instead of silently reused.

Large logic gates can be divided into parallel finger slices without changing
source W/L. The controller router uses distance to the actual routing tree;
the previous bounding-box estimate could spend minutes exploring the interior
of a tall tree. Scarce address accesses route first; congested nets retry with
priority after a bounded search. Full source MOS, finger and total gate-area checks remain.
For large banks, the Python interface connectivity audit excludes the core
interior and retains its boundary and all interface endpoints. Calibre checks
the complete GDS and source.

## Shared decoder and controller floorplan

The default corner plan now fuses the row decoder and TIME_CONTROL logic into
one physical `ROW_DECODER_TIME_CONTROL` module. The controller's lower frame
edge equals the data-register band's lower edge. It fills the broad corner
below the decoder first; excess standard cells continue upward in a narrow
lane alongside the decoder gates. The lane extends the shared decoder block
to the left, and its height must fit within the decoder's upper boundary.
It replaces the previous policy of extending TIME_CONTROL downward.

Both blocks use CMOS standard-cell families, but their existing row heights
are 1365 nm and 1500 nm. A nominal 1 µm separation keeps their wells and supply
rails clear. The overflow lane is nominally 4 µm wide at 32 rows, 5 µm for
33–256 rows and 20 µm above 256 rows. Banks below 32 rows use a 10 µm lane
to compensate for their short decoder. Oversized controller
gates split into parallel slices that conserve each source device's total
W/L. Data registers, source logic, sizing, timing and array pitch are preserved.
This is physical integration; no electrical gate factoring is introduced.
The narrower 512-row lanes remain failed or superseded routing experiments;
the wider lane reserves enough tracks for its large buffer groups.

The controller router respects the two placement regions instead of routing
through the decoder. Address and wordline-enable pins stay under their decoder
spines, while replica-wordline feedback reaches the top of the overflow lane.
Controller supply rails follow the changing row width. The manifest records
both regions, their actual area, the shared module, controller placements and
the lower-edge offset. Placement frames define the aligned boundary; individual
cell masks retain their existing insets. The shared module's GDS bounding box
is recorded separately from its floorplan envelope. A rectangular controller bounding box alone is no
longer its occupied frame area. `layout preview` also writes
`decoder_control.png` for the combined module.
A 1 µm band under the corner's address pins keeps their antenna taps clear
of the last logic row, including when the cells continue into the upper lane.

For tall shared frames, the routing heuristic includes the minimum cost of
changing layers to travel against a metal's preferred direction. Equal-cost
searches favor progress toward the existing routing tree. Local searches remain
bounded; long feedback and interface nets can search the complete reserved
region and route before the local groups. Supply access uses the existing
lower trunk instead of a redundant tall route. Retries and physical connectivity
checks are retained. Folded NAND
output metal stays clear of its internal straps, and same-net M4–M6 notches
are closed without changing transistor sizing.
Replica-wordline and far sense-enable receivers use the existing reserved
diffusion-tap filler, like the precharge observer. Their far-end drivers join
through M7/M8, so these taps protect gate-only branches during earlier metal
steps while preserving the feedback path. Connectivity through the final
metal stack alone cannot establish antenna safety.

New generation and verification evidence is recorded separately in
[`DECODER_CONTROL_FUSION_V2_2_8.json`](../../data/DECODER_CONTROL_FUSION_V2_2_8.json).
The earlier scaling evidence remains archived below. Its physical verification
receipts do not qualify the changed placement or its interconnect.

| Bank | Width × height (µm) | Macro area (µm²) | Area reduction from archived layout |
| --- | ---: | ---: | ---: |
| 32×32 | 42.7825 × 73.0975 | 3127.294 | 11.62% |
| 128×128 | 115.4825 × 204.1375 | 23574.309 | 7.27% |
| 512×512 | 404.2825 × 732.6975 | 296216.777 | 8.58% |

All three source netlists are byte-identical to their archived counterparts,
and their controller frame bottoms have zero offset from the data-register
band. The evidence file binds each physical verification status to the exact
GDS and source hashes. All three pass zero Calibre DRC violations and strict
LVS CORRECT; an additional default-source 8×8 smoke layout also passes.
Distributed R+C+CC extraction is complete for 8×8, 32×32 and 128×128. The
512×512 extraction status is recorded separately in the same evidence file.
No new extracted waveform or operating-point qualification is implied.

## Driver search

The explicit candidate profile is
[`layout_sizing_lookup.json`](../../../sram_compiler/sizing/layout_sizing_lookup.json).
The default sizing table remains a separate baseline. The candidate is based on
the complete 32×32 bank's hash-bound, passing Calibre DRC/LVS and distributed
R+C+CC extraction under `outputs/layout_scaling/baseline_32x32_verify`.

The load audit counts coupling incident on either DSPF net, separates metal
capacitance from model-dependent gate charge, and solves terminal resistance
including parallel paths. Four-section coupons include actual BSIM gate and
junction loads. Larger-bank metal loads are geometric extrapolations of that
measured bank, with the extracted endpoint contact resistance retained.

375 coupons cover five sizes, candidate strengths, and TT/SS/FF corners:
TT at 1 V/25 °C, SS at 0.9 V/125 °C, and FF at 1 V/−40 °C. The chosen classes
meet the 2 ns wordline, 2.5 ns write-driver, 3 ns precharge and 0.5 ns decoder
coupon targets at all three corners. These targets are not SRAM clock periods.
MOS models follow the stated corner and temperature; extracted metal resistance
remains at its nominal value because no metal temperature coefficient was measured.

| Square bank | PRE | Write output | Write input | WL inverter | WL NAND | Decoder inverter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.5 | 1.5 | 0.5 | 1 | 1 | 1 |
| 64 | 0.5 | 1.5 | 0.5 | 1 | 1 | 1 |
| 128 | 1 | 1.5 | 0.5 | 2 | 1 | 1 |
| 256 | 2 | 4 | 1 | 4 | 1 | 1 |
| 512 | 8 | 8 | 2 | 16 | 4 | 1 |

These are scale factors on the YAML unit widths, not widths in micrometres.
At 512, the previous PRE/write-output/write-input/WL-inverter/WL-NAND/decoder
classes were 16/32/8/128/35/9. Controller output buffers are recomputed from
the actual resized device loads. Clock, register, sense-amplifier and storage
transistors have not been independently optimized by this coupon search.

## Evidence and limits

The final no-RC compact controller with candidate sizing passes 2546 waveform
checks per corner on native full-mode 8×8 arrays at a fixed 9 ns period
(7638 checks total). Checks cover read, write, hold, sense and recovery over
eight cycles. These runs use illustrative distributed wiring, nominal devices
and seed 20260711; they are not post-layout simulations or Monte Carlo yield
evidence. The retained solver logs and waveform files are under
`outputs/layout_scaling/waveforms/final_8x8_{tt,ss,ff}`.

Layout verification, dimensions, source hashes and image paths are recorded
in [`LAYOUT_SCALING_V2_2_8.json`](../../data/LAYOUT_SCALING_V2_2_8.json).
Failed and superseded generation/verification attempts remain under `outputs/`.
The 512×512 operating point and full extracted read/write/recovery validation
remain unqualified. The previous V2.2.2 release screen certifies its own tree.

The earlier selected 512×512 GDS includes a 45.7 µm M6 connection from an isolated
clock-bar input branch to its existing driver diffusion. This resolves the
recorded M6 antenna violation while preserving the source MOS, all other
layers and the original controller placement. The patch script hash and path
coordinates are recorded in the layout manifest. Wider controller floorplans
are retained as separate generation experiments.

## Reproduction

Run from the repository root in the `openyield` environment. Set `PDK_DIR`
to a FreePDK45 installation and put Calibre on PATH.

```bash
python main_sram.py simulate --rows 512 --cols 512 --operation 'read&write' \
  --variation-mode nominal --real-cell-mode 0 --period 80e-9 --no-run-xyce \
  --sizing-lookup sram_compiler/sizing/layout_sizing_lookup.json \
  --compact-control --no-w-rc --output-dir outputs/layout/scaling_source
python main_sram.py layout bank --with-controller --deck "$DECK" \
  --output-dir outputs/layout/scaling_512
python main_sram.py layout verify-full --layout-dir outputs/layout/scaling_512 \
  --pdk-dir "$PDK_DIR" --output-dir outputs/layout/scaling_512_verify \
  --extraction-workers 2
python main_sram.py layout preview --layout-dir outputs/layout/scaling_512
python main_sram.py layout driver-audit --layout-dir outputs/layout/scaling_512 \
  --extraction-dir outputs/layout/scaling_512_verify \
  --output-dir outputs/layout/scaling_512_loads
```

`DECK` is the generated run directory's `deck.sp`. Output directories must be
new. Use 9 ns for the 32×32 layout source and 30 ns for 128×128; these are
explicit diagnostic periods. The tracked case manifest is
[`v228_scaling_cases.json`](../../../tests/spice/v228_scaling_cases.json).
