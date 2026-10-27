# Larger macro-area verification — V2.2.8

This follows the [initial area exploration](MACRO_AREA_EXPLORATION_V2_2_8.md)
with native complete-macro builds at 64, 128, 256 and 512 rows. The comparison
baseline remains the [periphery-height revision](PERIPHERY_HEIGHT_V2_2_8.md).
Source topology, models, every transistor's total W/L, sizing/timing tables,
bitcell pitch and simulation RC profiles remain frozen.

Verified area changes among completed candidates:

- **64 × 64: -0.273%**, `pitch285_macro64`.
- **128 × 128: -0.215%**, `pitch285_macro128`.
- **256 × 256: -0.530%**, `wide_macro256_m6_repaired`.
- **512 × 512: -2.256%**, `gap3300_macro512`.

These are physical results for the recorded settings. The default floorplan
selection remains unchanged; electrical qualification is outside this study.

## Register connections for taller controller bands

The production change is limited to `compact_center.py`: when a data
register's Q access and its write-buffer access no longer overlap, add an
aligned M3 bridge between their pin centres. Both existing accesses have
70 nm end landings. The native planner checks the new metal for width and
spacing and includes both endpoints in its interface-connectivity audit.

No bridge is drawn when the existing tracks already overlap. Before/after
polygon comparisons across every layer at the default 300 nm register gap
and the earlier 800 nm gap are identical up to controller routing. The new
code does not select a different floorplan, change a default constant or
expose a new CLI flag. Each experiment records its process-local gap, trunk
pitch and controller-order overrides separately from source-file hashes.

At gaps of 1300 and 3300 nm, the unmodified geometry fails the same all-column
connectivity check that the bridged geometry passes. Those local tests cover
all 32 data columns and M3 minimum width/spacing before controller routing.
The large completed macros separately check every source device and pass
through the native interface-conductor audit, which includes the data paths
outside the excluded bitcell interiors.

## Complete-macro results

Each completed bound below comes from actual GDS, including both array halves, shared
decoder, periphery, data registers, TIME_CONTROL, replica column, RWL driver,
output latch, routing and pins. Changes are against the same-size current
baseline, not the older pre-compaction records.

| Complete macro / candidate | Actual W × H (µm) | Area (µm²) | Area change | Calibre DRC / LVS | Extraction |
| --- | --- | ---: | ---: | --- | --- |
| 64 × 64 / `pitch285_macro64` | 104.2925 × 50.1300 | 5,228.2 | -0.273% | 0 / CORRECT | distributed R+C+CC |
| 128 × 128 / `pitch285_macro128` | 187.5725 × 75.4100 | 14,144.8 | -0.215% | 0 / CORRECT | distributed R+C+CC |
| 256 × 256 / `gap1300_macro256` | — | — | — | not run (generation time limit) | not run |
| 256 × 256 / `wide_macro256` | 356.8175 × 125.9700 | 44,948.3 | -0.530% | 1 / not run | not run |
| 512 × 512 / `gap3300_macro512` | 672.1775 × 230.0900 | 154,661.3 | -2.256% | 0 / CORRECT | not run |
| 64 × 64 / `pitch290_macro64` | — | — | — | not run (generation failed) | not run |
| 64 × 64 / `pitch295_macro64` | — | — | — | not run (preferred candidate verified) | not run |
| 256 × 256 / `gap1300_wide200_macro256` | 355.6175 × 126.9700 | 45,152.8 | -0.077% | 0 / CORRECT | not run |
| 256 × 256 / `wide_macro256_m6_repaired` | 356.8175 × 125.9700 | 44,948.3 | -0.530% | 0 / CORRECT | not run |

Rejected or unfinished attempts:

- `gap1300_macro256`: 5400 second native generation limit; no completed macro verification. No completed macro or physical pass.
- `wide_macro256`: physical verification failed: DRC violations; inspect results and summary.
- `pitch290_macro64`: upper decoder pins closer than 300 nm. No completed macro or physical pass.
- `pitch295_macro64`: The smaller 285nm 64x64 candidate completed DRC/LVS and extraction; fallback no longer needed. No completed macro or physical pass.

The 64/128 cases use 285 nm decoder trunk pitch and the default register gap.
The 256 height case uses 1300 nm gap (1000 nm extra); its separate ordering
case sorts controller groups by decreasing summed width at the default gap.
The 512 height case uses 3300 nm gap (3000 nm extra). Both height cases keep
the original 300 nm decoder trunk pitch and source order, so their results
isolate the height/width tradeoff. A separate 256 fallback adds another
200 nm of routing width to its 1300 nm gap case. These percentages must
not be added to one another.

## Local M6 repair of the reordered 256×256 candidate

The native reordered candidate fails `Metal6.2` once: two M6 shapes on `w_en`
leave a 25 nm gap. Via-aware conductor ownership confirms that they already
belong to the same electrical net. The original failing artifact and receipt
remain intact.

A separate repaired artifact adds the rectangle from (152612.5, 540) to
(166357.5, 565) nm: **0.343625 µm² of M6**. The repair checks 140 nm clearance
to foreign M6, preserves every original M6 polygon, and leaves every other
layer unchanged. It changes no transistor, source SPICE, pin position or
macro bound. Full DRC/LVS is rerun on the repaired GDS.

This is an explicitly recorded postprocessing step, not a new automatic
repair policy in the production generator. The case manifest and evidence
bind its coordinates, net-ownership proof, parent GDS and runner hash.
The smaller saving from the wider-frame 256 fallback is obtained directly
from the native generator without this postprocessing.

## Physical verification and provenance

The initial DRC and independent LVS probes failed because the commands had
no license-server address configured. Read-only inspection found an existing
local Siemens license service. Pointing the verification processes to that
configured service restored checkout; no license files or services were
changed. The failed probes and successful retry remain separately recorded.

The preceding +0.5 µm 32×32 reference now has **zero Calibre DRC violations,
CORRECT LVS and completed distributed R+C+CC extraction**, bound to its
original GDS. This resolves its earlier license-blocked verification
without rewriting the initial exploration record.

The table lists the completed physical scope for each larger candidate.
Extraction was requested for successful 64/128 candidates; 256/512 checks
stop after DRC/LVS. The original rule decks and strict source comparison
remain in force; no spacing guard or DRC rule was relaxed.

The [evidence JSON](../../data/MACRO_AREA_LARGE_V2_2_8.json) binds each
attempt, source deck, generator and override settings to its GDS/manifest,
source SPICE, tool/rule identities and verification receipts. Previous
exploration evidence and historical physical receipts are preserved.

Aggregate extracted capacitance for the 32-row height case and 285 nm
64/128-row pitch cases versus their matching baselines:

| Macro | Shunt C before → after (fF) | Coupling C before → after (fF) | Coupling change |
| --- | --- | --- | ---: |
| 32 × 32 | 531.879 → 530.626 | 1719.112 → 1712.139 | -0.406% |
| 64 × 64 | 1352.634 → 1352.085 | 4759.384 → 4773.501 | +0.297% |
| 128 × 128 | 3938.407 → 3935.214 | 15008.892 → 14997.904 | -0.073% |

Each capacitor edge is counted once. Extractor NET annotations remain
separate in the JSON. These aggregate changes do not establish delay,
power or read/write margins; individual critical nets still need waveform analysis.

6 targeted regression tests pass with no skips. The log records bridge geometry/connectivity checks and receipt
identity checks. Completed candidates also receive independent source-inventory,
drawn-gate/bounds, M1–M10 minimum-width/spacing and opposite-well checks.

The first 128×128 inventory screen incorrectly required identical physical
finger IDs. Its narrower placement adds one finger to each MOS in one
inverter while preserving total W. That failed screen is retained. The
corrected checker requires equal logical MOS terminals and complete terminals
for every actual finger, and passes alongside strict LVS.

The source-inventory checks compare recorded device/model/node/L/summed-W
information, while drawn gates are independently counted from active/poly
intersections. These checks supplement LVS; they are not a replacement for
it. Refingering can change physical gate count while total drawn gate area
stays fixed. The evidence records both counts and the exact M3 bridge segments.

No waveform, extracted timing, operating-point, read/write/hold/recovery,
yield, IR-drop or electromigration validation ran. Physical results and
extraction do not qualify sizing tables or replace the compiler's scoped
simulation RC profiles. The 512×512 deck retains its explicit diagnostic
period; automatic timing remains limited to 512×256.

## Reproduction

Use the existing `openyield` Conda environment and nominal mode-0 source
decks from the baseline. Run from the repository root into fresh directories.
The verification runners set `MGLS_LICENSE_FILE` and `SALT_LICENSE_SERVER`
for the existing local license service; no shell profile or service was
modified.

The [case manifest](../../../tests/spice/v228_macro_area_large_cases.json)
records the matrix, physical scope and generation limit.

A large height trial uses the new conditional bridges automatically:

```python
from pathlib import Path
from sram_compiler.layout.freepdk45 import compact_center

compact_center.REG_GAP = 3300.0
compact_center.generate(Path("path/to/512x512/deck.sp"),
                        Path("outputs/layout/compact/macro_area_large/new_512"))
```

For the 64/128 pitch trials, leave `REG_GAP` at 300 nm and set
`compact_decoder.SLOT = 285.0` in a fresh process before generation. These
internal overrides remain experimental. The controller-order variant is
implemented only by the ignored local runner.

`dev/macro_area_large/` contains the build, geometry and Calibre runners;
`tests/test_macro_area_large.py` holds local regression tests. These Python
files remain ignored by project policy. The tracked case manifest and
receipts record their scope and hashes; they are not shipped as a test runner
in a fresh clone. Raw outputs remain under
`outputs/layout/compact/macro_area_large/`.
