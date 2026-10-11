# V2.2.7 additional macro RC calibration

Development diagnostic, 2026-10-10. This extends the
[complete 8×8 reference](FULL_RC_CALIBRATION_V2_2_7.md) to the user-selected
8×16, 16×8 and 16×16 6T macros. All three complete paired waveform comparisons
pass. The accepted cases total 22,881 waveform checks across illustrative,
back-annotated and full coupled models; earlier failed attempts remain
preserved and unqualified.

All cases use FreePDK45, TT, 1 V, 25 °C, nominal variation, full transistor
mode 0, no column mux, and the last row/column. The clock is fixed at 30 ns
within every comparison. Compiler sizing classes are resolved independently
for each size and then frozen between its variants; no transistor width or
length is retuned to close the gap. This clock establishes diagnostic
functionality and does not measure the maximum operating frequency.

The new calibration generator uses the proven mirrored dense bitcell tile,
with 205/135/90 nm pull-down/access/pull-up widths and 50 nm channel length.
It places peripheral, replica, decoder, register and controller MOS directly
from each compiler source, including explicit folded fingers, body taps and
contacts. This is a spacious calibration placement rather than an
area-optimized physical SRAM compiler. Each macro includes the four physical
RC filters and the output latch. Calibre LVS retains the existing passive DC
reduction policy; coupled xRC includes the realized passive structures.

The two rectangular 128-bit cases exercise different array aspect ratios;
16×16 doubles both dimensions relative to the original reference. Periphery
and routing also change with size, so these cases do not isolate a pure wire
length scaling law. A separate extraction supplies the ground-truth
parasitics for every accepted physical layout.

All three selected physical references pass official Calibre DRC with zero
violations and LVS `CORRECT`, followed by coupled R+C+CC extraction. The
twenty-feed 16×8 reference required a fresh one-worker xRC retry after a
parallel capacitance-extraction crash; both attempts are preserved and use
the same GDS, source and rule contents.

| Macro | Schematic MOS | Extracted MOS | Reduced R | Reduced C |
|---|---:|---:|---:|---:|
| 8×16 | 2,520 | 2,571 | 15,463 | 96,489 |
| 16×8 | 2,434 | 2,473 | 14,686 | 88,956 |
| 16×16 | 3,646 | 3,699 | 21,142 | 127,158 |

Parallel physical fingers account for the MOS-count differences. Reduction
discards zero capacitance in all three models.

The compiler comparison retains all generated hardware and external stimuli.
The estimated variant replaces illustrative wiring with terminal-preserving
distributed networks, extracted junction geometry and realized passives.
Degree-two wire reduction preserves series resistance and all coupling, and
moves capacitance to the nearer surviving endpoint with a 500 Ω elimination
ceiling. The full coupled network remains the separate simulation reference.
These are size-specific reduced-network estimates derived from extraction;
they do not demonstrate geometry-based prediction for an unseen size.

Each transient contains W1/R1/W0/R0 and final recovery, ending at 126.995 ns,
5 ps before the fifth external clock starts. Output spacing is 5 ps and the
integration ceiling is 20 ps. Every cell, row and periphery column is probed.
The original 0.1/0.9 VDD ordering thresholds, 0.1 VDD retained-data tolerance,
and 0.02 VDD restored-bitline tolerance remain unchanged. Xyce uses KLU on one
process and common complete schematic initial conditions with UIC, released
throughout the transient. Testbench startup clamps remain outside the macro.

The comparison preparer previously scanned voltage equalities across the
whole deck, importing measurement thresholds `OUT=0.5` and `PRE=0.9` into its
initial-condition dictionary. It now reads `.IC` cards exclusively, including
continuations and signed numeric values. Initial external request inputs
match the actual compiler stimuli (`csb=1`, `web=1`, `clk=0`). Earlier attempts
using the incorrect initialization remain preserved and unqualified.

The initial 16×8 physical mesh also produces an unselected row-6 wordline peak
of 0.11345783 V at 9.69 ns. Ten upper-metal controller feeds per supply reduce
the observed peak to about 0.1016 V, which still exceeds the unchanged 0.1 V
limit. Both attempts remain unqualified. The accepted layout uses twenty feeds
per supply without changing schematic device sizing. Its first-write peak
falls to 0.0909 V, and the complete comparison subsequently passes. The initial
8×16 layout also produces an extra selected-wordline pulse before write
enable, peaking at 0.104203146 V. Twenty physical feeds per supply reduce
that early peak to 0.026245251 V and remove the extra pulse. The accepted
16×16 reference retains four controller feeds per supply. All schematic
widths, lengths and intentional passive targets remain fixed.

DC comparisons use the compiler's ordinary single-cell hold/read/write SNM
benches and extracted last-cell coupons. Each coupon retains the six physical
MOS, complete resistor components incident on their terminals, remote WL/BL
bias access, and VDD/VSS resistance. Other MOS are removed, so this is
embedded-cell DC characterization rather than full-macro operating current.
Capacitances are omitted because they carry no steady-state DC current.
The four inverter gates are opened with the compiler's rotated butterfly
stimulus. Each sweep has 1,421 finite points from −0.71 to +0.71 V in 1 mV
increments. Hold/read SNM use bounded classical lobes without artificial
endpoint fallback; write margin uses the existing constriction definition.
Matching DC curves test resistance and geometry, while the transient
comparisons separately test capacitance placement and dynamic coupling.

The DC comparisons are complete. The unannotated compiler cell gives
347.9096 mV hold SNM, 173.4065 mV read SNM and 347.8337 mV write margin for
all three sizes. The final extracted coupons give:

| Macro | Hold SNM (mV) | Read SNM (mV) | Write margin (mV) |
|---|---:|---:|---:|
| 8×16 | 346.0117 | 165.1790 | 329.4052 |
| 16×8 | 346.0571 | 164.6432 | 332.7638 |
| 16×16 | 345.9436 | 164.7952 | 330.0156 |

Reduced and full extracted DC curves agree at the recorded precision, as
expected from preserving resistance. The complete transient comparisons
separately test capacitance placement and dynamic coupling.

Each transient model passes 2,297 checks for 8×16, 1,881 for 16×8 and 3,449
for 16×16. Across eighteen clock/control, far-WL/BL and storage/output signals,
each size has 99 matched 50% edges with no edge-count differences. Errors use
absolute time, without shifting or fitted delay compensation.

| Macro | Illustrative mean edge error (ps) | Annotated mean / max edge error (ps) | Timing gap closed | Annotated mean signal RMSE (V) | Voltage gap closed |
|---|---:|---:|---:|---:|---:|
| 8×16 | 1,444.59 | 6.00 / 13.64 | 99.58% | 0.003763 | 98.25% |
| 16×8 | 1,332.06 | 4.67 / 9.09 | 99.65% | 0.002712 | 98.68% |
| 16×16 | 1,522.22 | 4.05 / 10.27 | 99.73% | 0.002407 | 98.90% |

Clock-to-data-rail timing uses the external clock's 50% rising edge and the
written cell or latched read output reaching 0.9/0.1 V, according to the datum.
The full extracted reference gives:

| Macro | Write 1 / write 0 (ns) | Read 1 / read 0 (ns) | Maximum annotated access-delay error (ps) |
|---|---:|---:|---:|
| 8×16 | 3.552 / 3.480 | 4.286 / 4.365 | 13.50 |
| 16×8 | 3.245 / 3.212 | 4.248 / 4.326 | 9.56 |
| 16×16 | 3.625 / 3.572 | 4.619 / 4.684 | 9.78 |

These metrics use the stated 5 ps output grid and 20 ps integration ceiling;
they are not a sub-picosecond convergence qualification. The improvement
includes junction geometry, realized passives and supply meshes, so it does
not establish the accuracy of wire scalar changes alone.

Validated exports expose 1,098 independent networks for 8×16, 1,044 for 16×8
and 1,514 for 16×16, with separate VDD/VSS meshes and explicit coupling.
Capacitance-only floating dummy drains obtain their network identity from
the exact schematic alias; export preserves the existing initialization map
and every electrical element. Normal compiler loading, without a mock or
validation bypass, reproduces each simulated physical element, stimulus and
complete initial-condition dictionary. Open observational probe branches
are excluded from the hardware comparison. The compiler-wide automatic
8×8 reference selection is unchanged; these new profiles are explicit,
size-specific diagnostics.

The workspace's ordinary nominal compiler generation can use the exported
configuration, for example:

```bash
python -m sram_compiler.per_device_mc.run --rows 16 --cols 8 \
  --target-row 15 --target-col 7 --operation 'read&write' \
  --variation-mode nominal --mc-runs 1 --real-cell-mode 0 \
  --corner TT --vdd 1 --temperature 25 --period 30e-9 \
  --interconnect-config outputs/rc_calibration/V2.2.7/multi_size/16x8/actual_compiler_07/interconnect_calibrated.yaml \
  --output-dir outputs/rc_calibration/annotated_16x8
```

The default generated sequence is longer than this four-access diagnostic;
hardware reproduction does not qualify the additional accesses. A fresh
clone requires the external PDK/extractor and new physical evidence to
recreate these ignored local profiles.

The relevant local regression suite passes all 46 tests, including asymmetric
probe axes, measurement-threshold exclusion from initial conditions and
capacitance-only net export. Compilation and `git diff --check` pass. Thirteen
legacy distributed-fanout expectations fail unchanged on the pre-change
`2521ccf` tree; the broad local suite is not reported as passing.

The protocol is preserved in
[`v227_multi_size_rc_cases.json`](../../tests/spice/v227_multi_size_rc_cases.json).
Generated GDS, source/DSPF, networks, decks, waveforms, overlays and all earlier
attempts remain under ignored `outputs/rc_calibration/V2.2.7/multi_size/`.
Python waveform/regression scripts remain local under repository convention.
The tracked [layout tutorial](../../sram_compiler/layout/README.md) describes
generation, verification, back-annotation and comparison tools.

The accepted receipts and artifact hashes are recorded in
[`V2_2_7_MULTI_SIZE_RC_CALIBRATION.json`](../data/V2_2_7_MULTI_SIZE_RC_CALIBRATION.json).
Waveform and SNM PNG/PDF overlays are in each case's `comparison_08` /
`snm_08` (8×16) or `comparison_07` / `snm_07` (16×8 and 16×16) directories.

| Macro | Timing/voltage overlay | DC curves | Compiler configuration |
|---|---|---|---|
| 8×16 | [PNG](../../outputs/rc_calibration/V2.2.7/multi_size/8x16/comparison_08/comparison.png), [PDF](../../outputs/rc_calibration/V2.2.7/multi_size/8x16/comparison_08/comparison.pdf) | [PNG](../../outputs/rc_calibration/V2.2.7/multi_size/8x16/snm_08/snm_comparison.png) | [YAML](../../outputs/rc_calibration/V2.2.7/multi_size/8x16/actual_compiler_08/interconnect_calibrated.yaml) |
| 16×8 | [PNG](../../outputs/rc_calibration/V2.2.7/multi_size/16x8/comparison_07/comparison.png), [PDF](../../outputs/rc_calibration/V2.2.7/multi_size/16x8/comparison_07/comparison.pdf) | [PNG](../../outputs/rc_calibration/V2.2.7/multi_size/16x8/snm_07/snm_comparison.png) | [YAML](../../outputs/rc_calibration/V2.2.7/multi_size/16x8/actual_compiler_07/interconnect_calibrated.yaml) |
| 16×16 | [PNG](../../outputs/rc_calibration/V2.2.7/multi_size/16x16/comparison_07/comparison.png), [PDF](../../outputs/rc_calibration/V2.2.7/multi_size/16x16/comparison_07/comparison.pdf) | [PNG](../../outputs/rc_calibration/V2.2.7/multi_size/16x16/snm_07/snm_comparison.png) | [YAML](../../outputs/rc_calibration/V2.2.7/multi_size/16x16/actual_compiler_07/interconnect_calibrated.yaml) |

No full V2.2.7 release screen, mux/PVT/mismatch qualification, larger-size
prediction, or resolved 256×256 operating point is claimed. The original
8×8 evidence and historical V2.2.2 qualification remain preserved.

The [extended investigation](EXTENDED_RC_VALIDATION_V2_2_7.md) adds a 32×32
physical layout, five-size critical-route inventories, and fixed-geometry
8×8 MOS PVT and per-device mismatch comparisons. Its scope and failures
are separate from the passing references above.
