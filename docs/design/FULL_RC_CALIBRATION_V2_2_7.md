# V2.2.7 complete macro RC calibration

The subsequent [additional-size comparison](MULTI_SIZE_RC_CALIBRATION_V2_2_7.md)
extends the workflow to 8×16, 16×8 and 16×16 with explicit profiles, DC curves
and complete four-access waveforms. The evidence below retains its original
8×8 scope and measurements.

Date: 2026-10-10. Development reference: FreePDK45, 8×8 6T, TT,
1 V, 25 °C, full transistor mode, no column mux, row/column 7.
This work supersedes the controller deferral in the
[initial calibration record](RC_CALIBRATION_V2_2_7.md). That record and its
original evidence remain preserved. The user subsequently authorized controller
legalization, complete layout, and compiler RC calibration for the whole macro.

## Controller and physical implementation

All TIME_CONTROL sizes and policy remain in
[`time_control.yaml`](../../sram_compiler/config_yaml/time_control.yaml).
The observer inverter is now 90/270 nm and the far-enable NOR is 90/540 nm.
Inverter widths round upwards on a 5 nm width grid with a 90 nm minimum;
folded fingers individually fit that grid. Unit-delay driving inverters use
720/2160 nm and replica-delay drivers use 360/1080 nm; dummy loads remain
90/270 nm. Stage and load counts are
unchanged. The stronger drivers address the initial physical write deadline failure
and the later 1.1 ns read deadline overrun. Actual observer widths now determine the compiler's
load and settling calculations.

The complete KLayout reference contains 876 memory/peripheral MOS, 798
TIME_CONTROL MOS and 18 output-latch MOS. Controller devices use explicit
body taps, gate/source/drain contacts, and M3 access pins. A deterministic
M3/M4/M5 router uses 190 nm M3 supply rails and joins all devices, input registers, clock distribution,
controller feedback, decoder, periphery and the array. This is a calibration
placement with generous routing space, not an area-optimized SRAM design.

Thin initial power routing produced a 0.102 V common-mode wordline glitch,
which failed the unchanged 0.1 V threshold. Widening the M3 supply rails from
70 nm to 190 nm reduces the first-clock unselected-wordline peak to 0.068 V.
Complete traces subsequently exposed read-recovery supply droop to 0.87 V
and a roughly 40 ps read-zero output overrun. Real 2 µm M7 buses and 280 nm M6
feeds now supply the controller and array through twelve physical tap stacks.
Average pad-to-cell VDD resistance falls from 408.5 to 232.8 Ω and VSS from
382.6 to 196.7 Ω; local array wiring/contact resistance remains extracted.
The previous failing traces remain preserved and unqualified.

Official FreePDK45 Calibre verification of the revised layout reports zero DRC violations and
LVS `CORRECT`. The coupled xRC result has 709 logical nets and 1,757 physical
MOS elements. Folding explains the difference from the 1,692 schematic MOS.
Rule, GDS, source and DSPF hashes are recorded with the results.

The four intentional filters use field-poly resistors and M4/M5 overlap
capacitors. The installed kit does not recognize these as explicit LVS passive
devices, so LVS uses `R SHORT SOURCE` and `C OPEN SOURCE` reductions. It proves
MOS connectivity under that passive DC reduction. Independent xRC passive
coupons measure the realized values:

| Filter | Intended R | Extracted R | Incident extracted C |
|---|---:|---:|---:|
| Precharge off | 263.4 Ω | 278.741 Ω | 1.241 fF |
| Isolation | 666.4 Ω | 681.114 Ω | 1.484 fF |
| Enables off | 698.4 Ω | 713.228 Ω | 1.518 fF |
| Read done | 666.4 Ω | 681.114 Ω | 1.484 fF |

Each intended capacitor is 1 fF; the incident xRC values include wire, poly and
contact parasitics. These are measured differences, not exact passive matching.
The complete macro network includes realized passives and their surroundings.

## Compiler back-annotation

The old WL/BL pair also supplied clocks, control buses and peripheral stubs.
Those physical routes have different lengths, fanout, access resistance and
coupling. The new model retains an explicit distributed network for every routed
net, including internal TIME_CONTROL nets and supplies. It replaces the generated
schematic's wiring at simulation assembly, preserves the complete schematic
hardware and stimulus, and back-annotates the extracted junction areas and
perimeters. Parallel layout fingers remain explicit MOS elements.

`global.yaml` now loads `interconnect_example.yaml`. Its 707 signal entries
and separate `power_models.VDD` / `power_models.VSS` entries each bind their
own network file, SHA-256, resistance scale and capacitance scale. The schema-2
profile composes these 709 independent networks. Cross-net capacitors remain
explicit coupling edges, with a symmetric geometric-mean capacitance scale
when endpoint settings change. VDD contains 2,708 resistors and VSS contains
2,609; MOS supplies connect at physical taps, producing local IR drop and
ground bounce.

Outside the validated reference, the compiler has independent illustrative
geometry settings for array WL/BL and thirteen named critical route classes:
replica WL/BL, decoder address, data-register clock, PRE, isolation, sense,
write and wordline enables, peripheral BL/BLB, write data and mux select.
These fallback geometries remain unqualified, and their supplies remain ideal.
The extracted power meshes apply only to the qualified reference. Changing a
network or scale records different physical context and marks the reference
settings unvalidated; a changed bound file also requires a new hash.

The estimated network combines parallel resistor edges and removes only
unprotected degree-two wire nodes with at most 500 Ω of combined series
resistance. Transistor and port terminals are protected. Series resistance is
preserved exactly; capacitances move to the nearer surviving endpoint. All
coupling capacitance is retained. The model has 10,185 R and 56,288 C elements,
eliminates 23,157 wire nodes, and discards zero capacitance. This approximates
capacitance placement; the full xRC network remains the independent reference.

Physical observation names map to transistor terminals. Array taps use the
corresponding cell gate/drain; control `line_far` uses the replica endpoint,
control `line_tapN` uses the addressed load, and peripheral taps distinguish
precharge, write and sense terminals. Existing compiler measurement names remain
available through open-ended 1 Ω observational branches.

A profile requires passing physical and waveform evidence. Explicit profiles
reject different architecture, operation, target, PVT, clock, controller settings,
model contents, MOS topology/geometry, and intentional passive values. Profile
hashes participate in physical context and run identity, so stale baselines fail.
The optional `reference_profile` setting selects the model only in its validated
reference scope and reports an unapplied profile outside that scope.

## Validation protocol

Before changing sizes, the full compiler passed nominal TT read, write and
read/write waveforms at 9 ns. After legalization and stronger delay drivers,
all three original checks passed again: 641 read, 624 write and 2,546 sequence
checks. Software regression checks cover configuration, wire reduction, physical
probe mapping, grid/finger rounding, sizing and stale-profile rejection.
The final local suite passes all 78 tests; compilation and `git diff --check`
also pass. Python regression scripts remain ignored under repository policy.

The final physical comparison uses identical compiler stimuli at the same fixed
9 ns clock. It contains four complete accesses, W1/R1/W0/R0, followed by final
bitline restore and hold. Simulation ends at 38.795 ns, 5 ps before the fifth
external capture. The output grid is 5 ps and the maximum integration step is
20 ps. The checker retains its existing 0.1/0.9 VDD ordering thresholds,
0.1 VDD retained-data tolerance, and 0.02 VDD restored-bitline tolerance.
All 64 cells, all rows and all eight periphery columns are observed.

Both physical networks receive the same complete quiescent schematic logic
state as transient initial conditions, with UIC. Conditions are released for the
entire transient; no operational signal is clamped. This avoids spurious startup
clock captures from zero-initialized internal wire nodes. Testbench startup
clamps remain stimulus infrastructure and are excluded from the physical macro.
Xyce 7.4 uses explicit KLU in both cases. An additional deck exercises the
compiler's own profile application path.

All three final models pass 1,241 waveform checks each, for 3,723 checks total.
Across twelve control, far-wordline and output signals, 71 matched 50% edges
have no edge-count differences:

Metrics use the stated 5 ps output grid and 20 ps integration ceiling; they
are not a sub-picosecond convergence qualification.

| Compared with full coupled xRC | Mean absolute edge error | Maximum edge error | Mean signal RMSE |
|---|---:|---:|---:|
| Illustrative pre-layout RC | 1,334.66 ps | 2,989.90 ps | 0.38740 V |
| Calibrated compiler default | 3.40 ps | 7.62 ps | 0.00452 V |

The mean timing gap falls by 99.75% and voltage RMSE by 98.83%. The calibrated
result includes extracted junction geometry and realized passives as well as
independent signal and power RC networks. These errors do not establish the
accuracy of wire scalar changes alone. The reduced-network preparation path
separately gives 3.37 ps mean edge error. Ordinary default CLI generation
reproduces every qualified compiler MOS/R/C/source element and initial
condition. Rebuilding the physical generator reproduces the source and all
layer geometries exactly (zero per-layer XOR difference).

A separate DC diagnostic applies a known 100 µA load between the far cell's
VDD and VSS source terminals through the shipped meshes. Xyce gives
0.992639343 V on local VDD and 0.008228088 V on local VSS: 7.360657 mV supply
drop and 8.228088 mV ground rise. This verifies the mesh IR-drop mechanism;
it is not a measurement of the macro's operating current. Regression checks
also double VDD resistance alone and confirm that VSS and signal resistance
remain unchanged.

The standalone eight-cycle pre-layout screen and the four-cycle physical
comparison have different stated scopes. Neither certifies the full V2.2.7
release screen, other sizes/topologies, mux, PVT or mismatch yield. The historical
V2.2.2 qualification and the unresolved 256×256 operating point remain unchanged.

The passing reference and artifact hashes are recorded in
[`V2_2_7_FULL_RC_CALIBRATION.json`](../data/V2_2_7_FULL_RC_CALIBRATION.json).
Raw layout, extraction, waveforms, overlays and unqualified earlier attempts
remain under ignored `outputs/rc_calibration/V2.2.7/controller/`.
