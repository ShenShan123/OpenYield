# TIME_CONTROL compaction and larger array readiness report

> **Update (2026-10-12):** The maze-routed bank integration reported here
> (`full_40`) is superseded by the [planned controller-to-core
> interface](TIME_CONTROL_LAYOUT_V2_2_8.md). `planned_02` is 4050.6731 µm²,
> passes DRC/LVS/xRC, and routes every interface net on explicit tracks.
> The standalone-controller results below still hold.

Date: 2026-10-12. Compiler: V2.2.8. Physical scope: one complete 32×32 6T
bank with TIME_CONTROL, input registers, decoder, replica circuitry, column
periphery and output latch, using FreePDK45.

The compact controller and assembled bank now pass zero Calibre DRC violations,
strict LVS `CORRECT`, and distributed resistance, capacitance and coupling
extraction. The latch-input short is repaired. This gives a verified physical
baseline for deciding on a larger layout experiment. Dynamic read/write timing,
retention and recovery for this layout still require waveform validation.

## Area and circuit preservation

| Physical region | Previous area µm² | Verified area µm² | Reduction |
| --- | ---: | ---: | ---: |
| TIME_CONTROL including 39 input/control registers | 867.222113 | 838.784363 | 3.28% |
| Complete 32×32 bank including routing | 6667.883863 | 5760.713113 | 13.61% |

The controller measures 32.445 × 25.8525 µm; the bank measures
70.405 × 81.8225 µm. Areas include recursive GDS geometry, wells, contacts,
passives and routing. The previous controller passed physical verification;
the previous complete bank passed DRC but failed LVS, and remains unqualified.

Both source schematics are byte-identical to their previous exports. The
controller retains 1,390 logical MOS and 2,058 physical fingers; the complete
bank retains 9,572 logical MOS and 10,702 physical fingers. Device models,
connections, total widths and channel lengths remain unchanged. The data array
keeps 705 nm column pitch and 1365 nm row pitch. Sizing, timing tables and
compiler RC defaults are unchanged. [Exact measurements and receipts](../../data/TIME_CONTROL_COMPACTION_V2_2_8.json).

## Placement and routing changes

OpenRAM supplied the physical organization and area reference. Shared register
wells and supply rails allow the horizontal register gap to fall from 290 to
190 nm, narrowing the controller frame from 33.6 to 32.5 µm. Vertical register
spacing remains at its verified value: a tighter candidate violated clock-rail
spacing. The OpenYield DFF topology and device sizes remain intact.

The latch D escape now sits beyond both bank power spines and clear of RBL.
Metal/poly connectivity audits reject shorts before routing and require real
endpoint connectivity after geometry legalization. Official LVS independently
checks the complete transistor circuit.

Feedback routes reach the actual replica-side taps without the former forced
perimeter loops. All observer returns remain on M9 over the controller to
preserve antenna jumpers. Data-register outputs use their native M8 access;
bank data inputs retain M10 antenna jumpers. Routing stays inside a bounded
outer channel, penalizes the full clearance around congestion, and periodically
reconsiders surrounding routes. Final via geometry receives independent DRC.

The upper-metal routing still occupies substantial space. This compaction does
not establish a regular routing plan for arbitrary array dimensions. The
[OpenRAM reference](OPENRAM_REFERENCE_V2_2_8.md) has different circuitry; its
872.20185 µm² controller/register and 8811.4131 µm² full-macro areas are reference
ceilings, not equal-circuit comparisons. Its full macro failed LVS.

## Verification results

| Check | Controller | Complete bank |
| --- | --- | --- |
| Official FreePDK45 Calibre DRC | 0 violations | 0 violations |
| Strict source versus layout LVS | CORRECT | CORRECT |
| Distributed R+C+CC extraction | 665 nets | 3,600 nets |
| All four intentional guard resistances | Within 1% | Within 1% |

The three unique resistor targets, 2462.4, 4102.4 and 4742.4 Ω, extract as
2473.7000, 4083.6297 and 4724.0990 Ω in both layouts. The largest absolute
error is 0.459%. The fourth guard repeats 4102.4 Ω. The unchanged passive
generator also retains six independently verified filter/blank coupons:
matched subtraction gives 1.02011 fF for each intended 1 fF plate, within the
3% plate tolerance. This does not measure the complete loaded node capacitance.
LVS uses resistor-short/capacitor-open reductions, so passive values are checked
separately from transistor connectivity.

All 33 local checks pass: 12 controller/bank acceptance checks and 21 bank,
pitch and conductor-audit regressions. They cover actual GDS area and pitch,
complete device/finger terminals, unchanged source schematics, guard values,
and hash-bound physical receipts. Python compilation and `git diff --check`
pass. Failed routing, spacing and antenna candidates remain preserved.

The source is nominal TT, 1 V, 25 °C, full transistor mode 0, with recorded
seed 20260711, illustrative source wiring and a fixed diagnostic 9 ns clock.
The extraction uses the installed FreePDK45 metal rules. This is physical
verification, not a PVT, mismatch, yield or full-release screen.

## Review artifacts

- [Complete bank GDS](../../../outputs/layout/time_control/full_40/full_macro.gds)
- [Complete bank PNG](../../../outputs/layout/time_control/full_40/layout.png)
- [Placement PNG with upper metals hidden](../../../outputs/layout/time_control/full_40/placement.png)
- [Controller detail in the bank](../../../outputs/layout/time_control/full_40/time_control.png)
- [Standalone controller PNG](../../../outputs/layout/time_control/control_20/layout.png)
- [Bank physical receipt](../../../outputs/layout/time_control/full_40_verify/receipt.json)
- [Controller physical receipt](../../../outputs/layout/time_control/control_20_verify/receipt.json)
- [Machine-readable evidence and provenance](../../data/TIME_CONTROL_COMPACTION_V2_2_8.json)
- [Design details and reproduction commands](TIME_CONTROL_LAYOUT_V2_2_8.md)

The [original stopped-work evidence](../../data/TIME_CONTROL_LAYOUT_V2_2_8.json)
remains unchanged and records the earlier LVS failure.

## Decision on larger arrays

The current physical generators explicitly support only 8×8 and 32×32 bank
plans; integrated TIME_CONTROL supports only 32×32 with last-column readout.
A larger array needs new decoder placement, register/interface planning,
size-dependent driver geometry and routing. The present 32×32 RC extraction
and area ceilings cannot qualify or predict that larger layout.

| Possible next step | What it establishes |
| --- | --- |
| Validate the current 32×32 extracted waveforms | Read/write/hold/sense/recovery behavior and physical control ordering at the frozen baseline |
| Extend to 64×32 | Row scaling with the existing 32-column data bus; requires a new decoder and address-register plan |
| Extend to 32×64 | Column scaling, twice the data registers, wider data routing and different wordline/control loads |
| Extend to 64×64 | Both scaling mechanisms together; four times the stored bits and a broader integration task |

For a larger **physical-layout** experiment, 64×32 is the preferred first step
because it changes one axis while retaining the current column count and
readout column. It still requires a new matched OpenRAM reference, fresh
DRC/LVS/extraction, source-device conservation checks, and independent guard
checks. Resolve the new size's sizing/timing baseline once and keep it fixed
through comparisons.

Before treating either the current or a larger bank as operational, validate
controller pulses, far-end crossings, data retention, sensing and recovery
against complete coupled waveforms. The
older 32×32 waveform attempts (whose legacy artifacts have been purged) were
computationally incomplete; runtime for this compact extraction is not yet
established. No larger-array generation or simulation has been started in this
task. The next step is reserved for the user's decision.
