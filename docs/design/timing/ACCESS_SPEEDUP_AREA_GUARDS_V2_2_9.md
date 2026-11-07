# V2.2.9 area-candidate guard characterization

The [completed area revision](ACCESS_SPEEDUP_AREA_V2_2_9.md) records the final
macros and native 64×64 coupons. This pilot record retains its original scope.

The normalized-delay **pilot** layouts pass 16 isolated ISO/replica-wordline
checks and five all-row wordline-release checks. The
[evidence record](../../data/ACCESS_SPEEDUP_AREA_GUARDS_V2_2_9.json) binds the
source, extraction, models, solver, decks and waveforms. These results do not
promote a sizing profile or clock and do not qualify the final resized macros.

The pilots use 180/540 nm delay inverters with one 90/270 nm dummy load per
stage, retaining the 16/44-stage guards at 128/256 columns. Their PRE and write
drivers retain the earlier classes. The final area candidate changes those
peripheral classes and some timing references, so its physical receipts and
setup validation remain separate requirements.

| All-row release cases | Minimum ordering margin | Minimum 1.5-tail margin |
|---|---:|---:|
| 128×128, SS hot, read and write | 434.669 ps | 361.100 ps |
| 128×128, FF cold, read and write | 108.876 ps | 79.615 ps |
| 256×256, FF cold, write | 336.546 ps | 284.579 ps |

The table uses the latest real-wordline 50% crossing as the tail reference.
Checks using the actual replica-wordline reference also pass. Every real
pass-gate terminal is included: 32,768 gates at 128×128 and 131,072 at
256×256. The slowest rows were 121 at 128 SS, 116 at 128 FF, and 244 at
256 FF. Neither the last row nor the replica was assumed to be slowest.

The all-row circuits retain the extracted wordline-enable buffer and
distribution, every row driver, the replica driver, and the actual load MOS.
Read and write release exercise different inputs of the control NAND. All
decoder inputs are held high as a loading stress; this does not test decoder
selection or storage retention. Consistent idle initial conditions are released
at transient startup, and all gate envelopes must settle before the command.

The retained signal-network contact, via and metal resistances are unchanged.
Supplies and load drain/source/body terminals use the explicit ideal boundaries
recorded per case. Quiet coupling uses equal shunts; the adverse case uses
twice the external bar coupling as shunt capacitance and removes external
guard coupling. These are tested loading assumptions, not a bound on arbitrary
aggressors. External ready loads are omitted to test an early-ready boundary.

The earlier coarse model combined 32×32-derived bar scaling with schematic
guards and reported failures even for known passing historical layouts. The
new results use each pilot's actual signal R/C and guard devices. The earlier
attempts remain available; they are not reclassified as physical failures.

The first 256 transient ended before the release completed. It is recorded as
an **incomplete window**. A fresh run extending only the stop time from 2.5 to
3.5 ns completed and passed; its overlapping waveform samples match exactly.
Unsuccessful initialization attempts and the corrected early load-boundary
attempt are preserved separately.

SS 256 write ordering has a separately labeled transfer estimate, not a direct
simulation. Decoder setup, additional corners and final resized-layout timing
are outside this record. No guard-stage reduction is supported by this study.
The broader validation protocol is
[v229_area_speedup_cases.json](../../../tests/spice/v229_area_speedup_cases.json).

Local reproduction uses `dev/area_speedup/guard_*` and
`archive_guard_evidence.py`; these Python tools and raw outputs remain ignored
under the repository's working conventions.
