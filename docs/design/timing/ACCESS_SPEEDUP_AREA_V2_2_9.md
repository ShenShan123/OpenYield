# V2.2.9 area-constrained frequency improvement

**Completed V2.2.9 area revision.** The compact defaults use the validated
source profiles below. The 32×32 clock is measured; larger clocks are
conditional estimates, as authorized. This is a finite screen, not release
or yield qualification.

This round continues the [access-speed plan](ACCESS_SPEEDUP_PLAN_V2_2_8.md)
on `fix/postlayout-timing-pdn`, with driver resizing authorized by the user.
The release remains V2.2.9. The required verification sizes are 32×32,
64×64, 128×128 and 256×256. The hard ceiling is each complete V2.2.8 g15
macro's actual GDS bounding-box area, including its controller, registers,
periphery and supply grid. The area baseline is commit `58bf5cb`; the prior
V2.2.9 timing comparison is commit `8a04c76`.

## Circuit changes

The compact self-timed 6T profile uses two-unit delay drivers with one dummy
load per stage. The previous eight-unit drivers used four loads. The smaller
implementation retains nearly the same steady delay per stage while reducing
physical area. Measured fixed overhead and physical completion windows are
accounted for separately.

| Array | PRE scale | WD input/output scales | Setup stages | PRE-off guard | Read/write completion stages |
|---|---:|---|---:|---:|---|
| 32×32 | 4 | 1 / 4 | 2 | 6 | 8 / 8 |
| 64×64 | 2 | 1 / 4 | 28 | 8 | 8 / 8 |
| 128×128 | 2 | 1 / 4 | 60 | 16 | 8 / 14 |
| 256×256 | 2 | 1 / 4 | 74 | 44 | 8 / 54 |

The setup and completion references preserve the independent 1.5× timing
reserve. An earlier candidate passed functional first-read checks but missed
the setup reserve at cold corners. The final candidate adds the required
setup/PRE-off stages. A longer external clock cannot repair early admission
or early completion.

The override applies to four rounded joint classes of compact self-timed
6T arrays. Legacy/10T profiles and other joint classes retain their existing
sizing. Optional arguments are appended to existing APIs. Frozen sizing and
timing objects reject changed lookup identities.

## Routed area

| Array | V2.2.8 ceiling (µm²) | Final actual area (µm²) | Physical status |
|---|---:|---:|---|
| 32×32 | 2229.47960625 | 2216.92721250 | DRC 0, LVS CORRECT, R+C+CC |
| 64×64 | 5118.53145000 | 5105.64176875 | DRC 0, LVS CORRECT, R+C+CC |
| 128×128 | 13959.57465000 | 13782.03206875 | DRC 0, LVS CORRECT, R+C+CC |
| 256×256 | 43896.70985000 | 43857.04171875 | DRC 0, LVS CORRECT, R+C+CC |

Low-current read/write completion logic can share a placement row in the
small classes. The five high-current control groups remain separate. DFF rows
align to the routing grid. Output-latch data routing checks the assembled
enable and supply metal, including connections placed before the data route.

The production generator selects characterized frames by controller circuit
identity. Changed controllers use the general search. All four recipes reproduce
the verified bank geometry and controller placements. The 64×64 controller
uses local M3/M4 wiring for its unchanged four/eight-stage delay chains,
reducing global routing demand from 300 units / 216 nets to 181 units / 161 nets.
Primitive supply vias and source devices are retained. Controller rerouting
can retry a priority net after another net has ripped its completed route;
this fixes an early-abort path. No M10 signal-routing extension is adopted. Physical receipts alone do not establish waveform correctness.

## Measured 32×32 evidence

All 9,632 source MOS remain in the extracted network. Every macro supply pin
is fed, with the extracted supply mesh and kit contact/via resistances.
The g15 reduction remains 20 Ω series reduction, shorts below 1 Ω and grounded
coupling below 10 aF. The solver is Xyce 7.4 with KLU and the recorded
FreePDK45 corner models.

The completed screen contains ten nominal four-access seq/mixed cases at
SS, FS-hot, FF-hot, SF-cold and FF-cold; a matched SS nominal/5% per-device
sample pair at 8 ns, seed 20261106; and two cold first-read cases at 7.4 ns
and 70% duty. All fourteen coupled cases pass. Four matched equal-shunt
coupling sensitivity cases also pass.

The maximum observed rail compression is 49.9934834 mV; the maximum matched
coupling sensitivity is 7.2198382 mV. Their sum sets **Vsys = 57.2133216 mV**.
All fourteen coupled traces pass rescoring at this system term, preserving
the original scores and all electrical predicates. Nominal cases require
1.5× the offset/noise bar; the mismatch sample requires 1×. An impossible
2 V system-term negative control fails 129 electrical checks.

Two probe-only first-window replays retain the exact electrical network and
initial state. They observe all 64 actual row-driver input gate fingers.
The 1.5× decoder setup reserve is **+72.555725 ps at FF-cold** and
**+87.478930 ps at SF-cold**. The decoder-drain aliases are distinct from
these gate nodes; their scores are recorded separately.

The worst nominal access/full-recovery times are 3629.882251 / 6123.168183 ps.
The matched SS sample gives separate factors 1.017274529 / 1.017642402.
Rounding the adjusted budgets upward gives **3693 ps access / 6232 ps cycle**.
One 10% margin and upward 50 ps rounding produce the **6.90 ns selected
period and 4.10 ns deadline**. All five physical confirmations at those settings
pass: SS seq/mixed, the matched SS sample, and FF/SF cold first reads with
actual gate probes. All five also pass the final Vsys-based scorer. The cold
gate-side reserves are +72.592273 / +87.641837 ps. Maximum selected-period rail
compression is 49.8893797 mV, below the frozen 49.9934834 mV rail bound.

The schematic screen additionally passes thirteen nominal/mismatch array
cases, twenty corner setup checks across the four sizes, and two complete
cold first-read array cases.

## Larger-array timing estimates

The user authorized measured anchors with explicitly labeled larger-array
estimates. These are conditional event-model results, with the historical
physical scaling, measured peripheral coupons, final five-corner 32×32
completion windows and unchanged rectangular-class floors retained.

| Array | V2.2.8 period (ns) | Prior V2.2.9 period (ns) | Selected period (ns) | Deadline (ns) | Evidence |
|---|---:|---:|---:|---:|---|
| 32×32 | 9.50 | 7.40 | 6.90 | 4.10 | Measured budget and five period confirmations |
| 64×64 | 12.00 | 10.70 | 10.60 | 6.00 | Conditional estimate |
| 128×128 | 21.25 | 15.80 | 15.85 | 9.35 | Conditional estimate |
| 256×256 | 27.75 | 27.10 | 27.00 | 14.80 | Conditional estimate with separable PRE-tail assumption |

The 128×128 estimate is 0.05 ns slower than the prior V2.2.9 setting because
the corrected decoder reserve is charged. It remains faster than V2.2.8.
This round does not claim improvement over the prior V2.2.9 clock at every size.

For 256×256, the strict no-peripheral-credit estimate is **28.60 ns** and
misses the old 27.75 ns clock. Incremental peripheral transfer gives 27.25 ns;
common WEN transfer gives 27.35 ns. The 27.00 ns estimate additionally reduces
the original model's explicit additive PRE-tail allowance. A physical
PRE-limiting recovery branch has not been established. These alternatives
remain visible; no independent metal-strap timing credit is added.

The final-source conditional completion checks pass with read8 and
write8/14/54. Minimum read margins at 64/128/256 are 39.400 / 57.593 / 23.419 ps;
minimum write margins are 41.597 / 8.095 / 14.920 ps. At 64×64, twenty
extracted completion-subtree transients (read/write, five PVT corners, two
input slews) replace the earlier 32×32 transfer bound. They retain ideal
supplies and a fast input/output boundary, so they are conditional lower-bound
sensitivities rather than full-array qualification. Eight additional ISO and
all-real-WL guard coupons pass; the all-WL coupon retains all 8,192 real PG
gate loads. Minimum 1.5× margins are 24.210 ps for all-WL and 57.122 ps for ISO.
Native eight/32-stage component sensitivities also pass at five PVT corners;
no extra clock credit is taken from local wiring. The checks retain the
20% lower-delay reserve, 1.5× data-resolution requirement, 25%/40% engineering
uncertainty and 1.10 mismatch factor. The external clock adds its single
10% margin afterward. New physical upper costs can raise the prior model;
the model does not subtract a whole-macro speedup from the larger arrays.

Large-array data differential at sensing must satisfy the final Vsys-based
bar. Full large-array waveforms, supply-noise transfer and unconditional macro
timer lower bounds remain unqualified. The finite sample screen establishes
neither statistical yield nor foundry signoff.

## Reproduction and evidence

The tracked [protocol](../../../tests/spice/v229_area_speedup_cases.json)
defines the four area ceilings and measurement policy. The
[guard audit](ACCESS_SPEEDUP_AREA_GUARDS_V2_2_9.md) preserves the completed
pilot ordering checks and their limits.

Local helpers and raw results are under ignored `dev/area_speedup/` and
`outputs/area_speedup/`. `integration_collect.py` checks source, model,
solver, seed, GDS, DSPF, waveform and result identities, and suppresses draft
tables while any required gate is incomplete. The earlier V2.2.9 records and
failed intermediate candidates remain preserved. The [final archival record](../../data/ACCESS_SPEEDUP_AREA_V2_2_9.json)
contains the completed gate results and provenance. The exact characterized
sizing JSON is promoted without changing its bytes; its historical candidate
identifier is retained so the source-deck fingerprints remain valid. Timing
metadata names the area revision. Baseline input hashes identify files before
promotion at `8a04c76`; they do not describe the updated default tables.
Production checks cover source-network equivalence, 35 dimension combinations
spanning all 20 joint classes, both timing axes, stale-baseline rejection,
and legacy/10T/schematic policy. All four macro recompositions match the
verified source and polygon geometry on every layer when importing the
bound controller route. A fresh 64×64 production route independently matches
its verified source and all-layer geometry as well. These equivalence checks
reuse the identical geometry's physical receipts; they are not new waveform runs.
The raw local generators, trials and waveforms remain preserved.

The final software screen passes 28 focused area/layout/sizing/guard tests.
The broader 89-case timing/driver/configuration/baseline screen passes 88
cases and identifies one stale CLI expectation for the old table version
and clock. Updating that expectation to the promoted profile gives four
passing CLI rechecks. No selected cases are skipped. Compilation, changed
Markdown links, whitespace checks, and independent code review also pass.
The archived software receipt retains the original failure and its rerun.
