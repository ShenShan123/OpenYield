# V2.2.7 larger-array, PVT and mismatch RC validation

Development diagnostic, 2026-10-11. This continues the
[complete 8×8 calibration](FULL_RC_CALIBRATION_V2_2_7.md) and the
[8×16, 16×8 and 16×16 comparisons](MULTI_SIZE_RC_CALIBRATION_V2_2_7.md).
Sizing, controller policy and clock periods remain fixed within each size.
The original evidence and automatic 8×8 reference scope are preserved.

## Protocol and scope

The new size is a complete 32×32 6T macro, including all memory cells,
decoder, replica, registers, TIME_CONTROL, four realized passive filters,
periphery and output latch. The compiler resolves its ordinary sizing class
once; illustrative, reduced-network and full coupled variants retain those
same devices. The period is 30 ns, the target is row/column 31, and twenty
upper-metal controller feeds per supply are part of the new physical layout.
Horizontal device placement uses a 3.36 µm pitch; all widths and lengths stay fixed.
Independent Calibre DRC/LVS/R+C+CC extraction is required before comparison.
A reduction of this new extraction tests estimation for this layout; it
does not establish prediction of parasitics from an unseen geometry.

The 8×8 campaign retains its existing physical geometry, junction areas and
perimeters, realized filters, VDD/VSS meshes and metal RC. Its clock remains
9 ns across all PVT and mismatch cases. The nine nominal points are TT
1 V/25 °C, SS 0.9 V/125 °C and −40 °C, FF 1.1 V/−40 °C and 125 °C,
FS 0.9 V/125 °C, SF 0.9 V/−40 °C, and TT 0.9 V/−40 °C and 1.1 V/125 °C.
These points change the transistor corner, supply and device temperature.
They are a selected operating matrix, rather than a complete Cartesian grid.

Five TT samples use seeds 2026101011–2026101015. One SS 0.9 V/125 °C sample
uses seed 2026101016. Every schematic MOS receives independent Gaussian
`vth0`, `u0` and `voff` draws with relative sigma 0.05 and no area scaling,
matching the repository's default mismatch parameter policy. Folded physical
fingers share their schematic MOS's draw. Numeric cards, per-device audits,
sampler identity and hashes are saved. Both RC variants use exactly the same
model file within each sample. The 1,692 macro devices plus two startup-fixture
MOS use 1,694 cards; 1,757 physical macro fingers plus two fixtures give 1,759
simulated MOS. Original parser-aborted MC decks omitted the fixture cards.
All six seeds were repeated with byte-identical macro cards and two additional
fixture cards. Setup failures are excluded from scientific counts. These six
samples do not certify yield.

The comparison protocol contains W1/R1/W0/R0 and final bitline restore and hold.
Its planned stop is 5 ps before the fifth capture: 38.795 ns for 8×8 and 126.995 ns
for 32×32. Every cell, row and periphery column is probed. Xyce 7.4 family
uses KLU on one process, 5 ps output and a 20 ps integration ceiling.
The existing 0.1/0.9 VDD ordering thresholds, 0.1 VDD retained-data tolerance
and 0.02 VDD restored-bitline tolerance remain unchanged.

Physical initial conditions use common complete quiescent schematic logic,
UIC and the actual initial pulse values CSB=0, WEB=VDD and CLK=0.
All conditions are released throughout the transient. The comparison
preparer now passes the profile VDD to wire-node initialization; waveform
metrics likewise use 50%, 90% and 10% of that VDD instead of fixed voltages.
The default 1 V behavior is preserved.

Accuracy is evaluated in absolute time, without alignment or fitted delay
compensation. A passing diagnostic requires both complete functional checks
and matching edge counts, maximum edge error at most 50 ps, and mean signal
RMSE at most 0.02 VDD. The 5 ps output and 20 ps integration ceiling limit
the precision of the comparison. Critical-route endpoints are evaluated
separately: replica WL/BL, decoder addresses, data-register clock, PRE,
isolation, sense/write/wordline enables, peripheral BL/BLB and write data.
Mux-select routing is absent from these no-mux macros.

## Results

The complete 8×8 investigation has twenty-two paired cases, 42 distinct
physical transients and 52,008 waveform checks on those distinct runs:
eighteen operating comparisons, two finer reductions and two numeric
reproduction references. The finer reductions reuse two completed full-network
references; counting their check results again gives 54,490 paired results.
Five of nine nominal 9 ns points pass both functional and RC-accuracy gates.
All five TT mismatch samples pass 1,241 checks per model; the SS-hot sample
fails. The direct physical-probe TT comparison also passes. Failed points
remain unqualified. The two finer-reduction comparisons have also completed.
The 32×32 physical waveform attempts were interrupted for computational cost
and remain unqualified; no partial waveform is promoted.

| Nominal point, 9 ns | Functional checks, both models | Main mean / max edge error (ps) | Main mean RMSE (V) | Critical max edge error (ps) | All edge counts match |
|---|---|---:|---:|---:|---|
| FF, 1.1 V, -40 °C | pass | 2.40 / 7.10 | 0.004228 | 7.06 | yes |
| FF, 1.1 V, 125 °C | fail | 15.43 / 1114.74 | 0.017400 | 20.47 | no |
| FS, 0.9 V, 125 °C | fail | 2.88 / 8.18 | 0.002473 | 6.54 | yes |
| SF, 0.9 V, -40 °C | pass | 3.54 / 8.55 | 0.004670 | 8.46 | yes |
| SS, 0.9 V, -40 °C | pass | 2.75 / 7.40 | 0.003519 | 7.29 | yes |
| SS, 0.9 V, 125 °C | fail | 2.04 / 5.13 | 0.001250 | 3.80 | yes |
| TT, 0.9 V, -40 °C | pass | 2.90 / 6.76 | 0.003833 | 6.71 | yes |
| TT, 1.1 V, 125 °C | fail | 3.66 / 64.03 | 0.009115 | 63.74 | no |
| TT, 1 V, 25 °C | pass | 3.10 / 7.59 | 0.003922 | 7.36 | yes |

SS 0.9 V/125 °C has 144 failures among 1,221 checks per model; FS at the
same voltage/temperature has 86 among 1,224. Their close RC matches do not
qualify the failed operations. Both separate SS/FS hot 30 ns diagnostics
pass 1,241 checks per model; maximum critical-route errors are 7.56 ps and
7.37 ps. Ordinary compiler stimuli also change their input ramps from
90 ps at 9 ns to 300 ps at 30 ns, so this changes the full clock/stimulus
context and does not isolate frequency alone. Defaults stay unchanged.

TT 1.1 V/125 °C and FF 1.1 V/125 °C fail continuous isolation and
precharge/sense-exclusion checks. The last TT-hot column's full trace has
about 508 ps maximum precharge/sense overlap and 44 ps isolation/sense
overlap. Its estimate also has isolation edge-count differences. FF-hot
has replica-bitline edge-count differences and over 1 ns maximum main-signal
error. The 500 Ω reduction therefore fails its accuracy gate at these hot
points. Separate 20 Ω reductions retain more capacitance locations without
changing transistor dimensions, geometry, stimuli or timing policy. They use
28,396 R and 87,368 C elements and remove 4,946 wire nodes. Their full-network
reference waveforms are the identical completed original PVT runs.

| Hot point, 20 Ω reduction | Main mean / max edge error (ps) | Main mean RMSE (V) | Critical max edge error (ps) | All edge counts match | Functional result |
|---|---:|---:|---:|---|---|
| TT, 1.1 V, 125 °C | 0.53 / 20.98 | 0.001273 | 21.01 | yes | fail |
| FF, 1.1 V, 125 °C | 0.37 / 23.55 | 0.000304 | 23.80 | yes | fail |

Both finer models meet the independent RC-accuracy gate. The full physical
waveforms still fail control ordering, so neither point becomes qualified.
The 500 Ω estimate's unchanged terminal resistance is insufficient to predict
the narrow recovery pulses at these corners; capacitance relocation matters.
No experimental profile or reduction threshold becomes a compiler default.

The illustrative 32×32 compiler passes all 10,937 checks at 30 ns. Initial
32-column and 64-column placements, and a same-net-via retry, fail routing
of sense-amplifier local nets. Widening horizontal spacing permits a complete
fourth layout. Its DRC is clean but LVS catches a VDD/VSS short: WL24's M5
route crosses both fixed array supply landings before the M3–M6 stacks are
added. DRC alone does not establish connectivity.

The generator now reserves those pad/array M4/M5 footprints before routing.
The fifth layout reuses 1,554 unaffected routes and reroutes two conflicting
nets; extraction and waveform qualification are never reused. Fresh official
verification reports DRC zero, LVS `CORRECT`, and coupled extraction of
3,662 nets. The estimate has 51,262 R and 276,108 C elements and eliminates
107,822 wire nodes. Full physical waveforms remain separate from layout proof.
The optional same-net-via routing policy preserves legacy routing defaults;
its new geometry receives independent physical verification.

Sparse conductance solves compare full/reduced terminal resistance on fifteen
critical routes and power endpoints for each extracted size, including 32×32.
They preserve parallel paths and agree at numerical solve precision. These
unloaded two-terminal network results are not operating driver resistance or
macro current. The maximum relative resistance discrepancy over these five
sizes is 1.46×10⁻⁹. DSPF NET capacitance totals include coupling counted on both nets.

| Size | Selected WL resistance / incident C | Selected BL resistance / incident C | Replica WL resistance / incident C | Replica BL resistance / incident C |
|---|---:|---:|---:|---:|
| 8×8 | 330.75 Ω / 6.60 fF | 371.13 Ω / 2.30 fF | 515.40 Ω / 14.49 fF | 371.13 Ω / 11.18 fF |
| 8×16 | 891.06 Ω / 19.17 fF | 696.35 Ω / 13.97 fF | 542.02 Ω / 36.01 fF | 910.64 Ω / 29.34 fF |
| 16×8 | 804.07 Ω / 16.56 fF | 540.57 Ω / 11.79 fF | 548.86 Ω / 22.21 fF | 797.36 Ω / 34.31 fF |
| 16×16 | 924.62 Ω / 21.78 fF | 687.53 Ω / 14.31 fF | 547.83 Ω / 34.69 fF | 928.59 Ω / 40.49 fF |
| 32×32 | 1032.37 Ω / 51.72 fF | 865.35 Ω / 31.90 fF | 628.53 Ω / 70.37 fF | 1592.17 Ω / 68.36 fF |

The receipt records each driver/load terminal and the separate VDD/VSS
source endpoints. A net's incident capacitance is not the effective load
capacitance of a two-terminal resistor. These physically different placements
also do not establish a single array-size scaling law.

Sixty-five passive wire coupons (thirteen signal routes per size) also solve
the complex nodal system `G + j·2πf·C` at 101 frequencies from 1 MHz to
100 GHz. Each uses a fixed unit voltage at the recorded driver terminal.
All endpoints outside the selected resistor-connected component, including
floating conductors, are held at AC ground. Transistor intrinsic capacitance
and driver output resistance are excluded. This explicit boundary condition
tests the wire representation; it does not certify full-array operation,
aggressor switching or metal PVT.

At 100 MHz, the 32×32 selected WL's full/reduced phase delays are
27.85/30.27 ps, and the replica BL's are 61.21/58.79 ps. CLK_BUF and PRE
agree within 0.09 ps at that frequency. These phase delays are not
clock-to-data delays. The maximum absolute complex voltage-transfer
difference over the entire frequency sweep is 0.073 on the 32×32 BLB
coupon; no SRAM timing-accuracy gate is applied to these coupons.
The local implementation passes independent single-pole and bridge-capacitor
closed-form checks, plus disconnected-terminal rejection.

The [passive response PNG](../../outputs/rc_calibration/V2.2.7/extended/critical_rc_frequency.png)
and [PDF](../../outputs/rc_calibration/V2.2.7/extended/critical_rc_frequency.pdf)
show the actual 32×32 WL, replica BL, register-clock and PRE curves.

The old replica `RBL_FAR` / `RBLB_FAR` aliases observe precharge terminals.
Supplemental TT and later queued PVT/MC cases add actual last-replica-cell
drain nodes and selected-cell left/right VDD and VSS sources. Original aliases
and historical evidence remain preserved; metadata identifies direct-terminal
coverage separately. Critical metrics also include the last peripheral
sense-enable, isolation and write-enable gates.

## Numerical verification and larger-array runtime limit

The complete 8×8 BTF ordering reference retains the original physical
elements, sources and initial states. Both models pass 1,241 checks and the
RC-accuracy gate; maximum critical-route edge error is 7.36 ps. All 362 printed
voltages were independently compared with the original solver results after
1 ns, in absolute time. Edge counts match; the largest voltage difference is
1.60 µV. This is numerical reproduction, rather than a claim of physical
accuracy finer than the 5 ps output / 20 ps integration resolution.

Observation aliases also need consistent initial values. In the original
32×32 deck, 1,250 printed aliases start at zero while their physical terminal
has a high initial value. The preparer now initializes only unloaded aliases
to that terminal value. Loaded startup-fixture aliases keep their existing
policy. Every physical element and physical initial condition is preserved.
The complete corrected 8×8 reference passes both functional and RC-accuracy
gates. Against the original results, all 362 voltage-channel edge counts match
after 1 ns and the largest voltage difference is below 0.1 mV. This corrects
the initial observation state; it did not resolve the larger solver bottleneck.

The original 32×32 KLU comparisons were interrupted after about three hours:
the reduced model reached 0.950 ns and the full model 32.525 ns of the required
126.995 ns. A BTF reduced-model trial reached 21.400 ns after about 71 minutes.
Observed throughput could not complete the four-access protocol within the
planned four-hour limits. These attempts have zero qualified waveform checks,
recorded interruption receipts and preserved partial waveforms. They are
computationally incomplete, rather than evidence of SRAM functional failure.

Startup-only trials explored KLU repivot/order/filter options, KSparse and
parallel iterative solves. Completed startup pilots are excluded from array
qualification. The installed binary rejects optional SuperLU, SuperLU_DIST and
ShyLU solvers; the tested iterative configurations did not complete startup.
DC-start trials retain floating-conductor charge anchors and use logical
NODESET hints. The compiler's source DC values can differ from initial PULSE
levels, so corrected trials match those initial levels explicitly while
preserving every transient pulse. No DC-start trial reached a qualified full
sequence, and no initialization or solver experiment promotes a profile. The
[Xyce reference guide](https://xyce.sandia.gov/files/xyce/Xyce_Reference_Guide_7.7.pdf)
documents the linear-system transformations and the limits of serial direct
solves during parallel execution; actual availability was checked on this binary.

The scoped software checks pass all 61 distinct tests: 58 relevant compiler
and layout checks plus three independent passive-frequency checks. Compilation
and `git diff --check` pass. This is not a claim that the broad legacy local
suite passes; its previously recorded distributed-fanout failures are outside
this follow-up.

## Limits and remaining work

The installed kit supplies one fixed metal resistance/capacitance dataset.
The emitted DSPF and comparison R/C elements carry no calibrated metal
temperature coefficients. Changing MOS corner or `.OPTIONS TEMP` therefore
does not prove metal process or temperature dependence. Verification of
metal corners needs independent qualified extraction decks or measured
coefficients; this campaign does not invent them.

The existing explicit compiler profiles remain restricted to their original
validated PVT and timing scope. Experimental comparison decks are assembled
separately; compiler profile-loading guards are preserved. A functional or
accuracy failure remains unqualified even if the solver completes.
Earlier setup and interrupted attempts remain under ignored `outputs/`.

No full V2.2.7 release screen, full PVT grid, metal-corner qualification,
statistical yield certification, mux/10T coverage, geometry-based scaling
law or resolved 256×256 operating point is claimed.

The tracked [case manifest](../../tests/spice/v227_extended_rc_cases.json)
preserves the operating matrix and criteria. Local runners and Python
waveform/regression scripts remain ignored under repository policy.
The compact [results and artifact hashes](../data/V2_2_7_EXTENDED_RC_VALIDATION.json)
preserve every completed case's failures, solver/model/seed/RC provenance,
numerical reproduction, passive coupons, layout receipts and interrupted runs.

The next checks are a scalable coupled-network solver for the complete 32×32
sequence, qualified metal process/temperature extraction data, a wider PVT
matrix and a statistically meaningful mismatch campaign. The failed hot 9 ns
control ordering remains an independent issue under the frozen sizing policy.

## Layout PNGs

Native KLayout renders the actual GDS polygons, with labels and axes added
for interpretation. The overview separates the dense memory tile from the
spacious controller/peripheral placement, supply buses and fanout routes.

- [Labeled macro overview](../../outputs/rc_calibration/V2.2.7/extended/32x32/layout_05/preview/layout_overview.png)
- [Full 32×32 array detail](../../outputs/rc_calibration/V2.2.7/extended/32x32/layout_05/preview/array_detail.png)
- [Protected array supply landings](../../outputs/rc_calibration/V2.2.7/extended/32x32/layout_05/preview/power_landings.png)

The PNG receipt binds the GDS SHA-256 and records native KLayout rendering
with 2× oversampling. Images remain ignored local artifacts; a fresh clone
needs the external layout environment to recreate them.
