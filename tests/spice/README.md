# Phased SRAM access integration checks

These opt-in tests exercise real Xyce waveforms for V2.2.4. They are separate
from `python3 -m unittest discover -s tests`, which does not launch this screen.
**The V2.2.4 manifests have not been run**, and `256x256_6t_SS_read_write`
has no working operating point yet; the last assembled screen is V2.2.2's, and
it certifies the V2.2.2 tree, not this one.
Run from the repository root with the normal OpenYield Python dependencies and
Xyce 7.4 available.

```bash
python3 -m tests.spice.phased_access \
  --output outputs/validation/my-phased-screen \
  --xyce "$(command -v Xyce)" --workers 4
# The per-device sweep and the negative controls (which must exit non-zero).
python3 -m tests.spice.phased_access \
  --cases tests/spice/v224_mc_cases.json \
  --output outputs/validation/my-phased-screen-mc \
  --xyce "$(command -v Xyce)" --workers 4
python3 -m tests.spice.phased_access \
  --cases tests/spice/v224_negative_cases.json \
  --output outputs/validation/my-phased-screen-negative \
  --xyce "$(command -v Xyce)" --workers 4
```

Each case directory must be new. A repeated run needs a fresh output directory;
failed attempts are evidence and must not be overwritten. `--name NAME` selects
a case and can be repeated; `--cases FILE` selects another JSON manifest.

`v224_cases.json` covers consecutive reads, consecutive writes, alternating
reads/writes, different row addresses and column patterns, idle cycles, 6T/10T,
mux/no-mux, every corner at more than one voltage and temperature, arrays large
in both dimensions up to 256x256, and seeded per-device mismatch. Large nominal cases
request four MPI ranks; the matching launcher is resolved beside Xyce. Worker
count is the number of cases, not the total rank count. These runs can take
hours and generate substantial waveforms.

`phased_waveforms.py` checks the physical endpoints and terminals: exactly one
wordline pulse on the selected row, no other row pulse, full write enable while
WL is open, isolation before write/sense enable, read-WL/sense and write/sense
exclusion, PRE exclusion, quiet capture boundaries, data/output correctness,
retention of every probed cell, every sense group's read data, and bitline/replica
restoration. Exclusion checks cover the complete trace after first capture,
including all cycle boundaries; pulse windows meet at capture. Missing or
nonfinite data, missing output-grid samples, and negative interpolated ordering
margins cannot pass. Threshold-interval intersections are checked between
samples using linear interpolation. The recorded spacing is 5 ps; unrecorded
sub-sample excursions remain a limit of this evidence. Grid validation allows
1% timestamp jitter (50 fs) with exact sample count and contiguous indices.
Write-input stability and storage polarity checks include partial final
accesses. Reads, idle cycles, and unselected writes must preserve logical cell
state throughout; selected writes may change state during their physical WL
window. Normal read disturb is allowed below the 0.5 VDD polarity boundary,
with the stricter 0.1 VDD rail-error check after completion. Sense-group data
must remain valid from completion until physical sense-enable release.
Small arrays probe every cell; larger arrays probe
every cell on selected and sentinel rows, with every row's WL endpoints.

Ordering checks also report their worst margin, so a passing screen shows how
close it came: sense margin, WL off before capture, driver release after WL,
WL off before sense, isolation before enable, driver on before WL, both
physical enables off before precharge turns on, and stored polarity. The
enable-off margin is the one the superseded root-only observer reduced to
20.55 ps at 8x256 SS; a yes/no exclusion alone would not have shown that.

V2.2.2 adds four budget margins: `min_read_output_margin_ps` (the latched
output and every sense group settled before the `k + 0.68 T` checker
deadline; negative if OUT settles after it), `min_write_wl_after_flip_ps`
(the earliest physical wordline endpoint leaves 0.1 VDD this long after the
last written cell crossed mid-rail), `min_restore_before_capture_ps` (the
last far bitline or RBL back above 0.9 VDD before the next capture) and
`min_precharge_on_before_capture_ps`.

V2.2.3 takes the access deadline from the compiler (`ACCESS_DEADLINE`, so a
runtime `.MEASURE` pass and a screen pass accept the same traces), scales the
sense bar to 0.28 of VDD instead of a fixed 0.25 V, and shares one
`probed_rows` between the runner and the checker so arrays too large to probe
whole also probe the quarter points and each addressed row's neighbours.

V2.2.4: a case may set `col` (the target column; default the last), and the
checker reads it back, so column-mux input 0 is exercised. The runner probes
the address bits itself (the compiler omits `A0` for a one-row array), retries
a failed nominal multi-rank operating point once with Newton line search
(keeping the failed attempt), and runs cases marked `operating_point: seeded`
in two passes: a stimulus-free UIC settle to 0.9 ns that saves every node, then
the deck's own DC solve from that `.NODESET` guess. The `.IC` values join the
guess (Xyce refuses both), the solution at t = 0 is saved, and the case fails
if any `.IC` node is more than 10 mV from its value
(`execution['seeded_operating_point']` in the metadata records the deviation).

The arbitrary-pattern cases replace only the external PWL sources. Their data
expectations come from the pattern, not the compiler's fixed WRWR measurement
schedule. Solver failures, source changes during a campaign, and Python/checker errors
cause a nonzero runner exit. Actual PVT model selection is regression-tested. Simulation/model/seed metadata and source hashes stay beside the deck.

## What the manifests cover, and what they do not

`v224_cases.json` is 334 cases: the 285 of V2.2.3 plus 49 V2.2.4 additions
(`added_in`), namely mux column 0, FF cold and hot at 128 to 512 rows, sizes
that are not powers of two including one-row and one-column arrays, partial
address-bit flips between neighbouring rows on tall arrays, and four
post-release alternating mux-input reads. A `column_sequence` starts at `col`
and stays in its mux group. The runner changes the external one-hot selects in
clock-low before the next capture; the scorer checks each select at the root
and the target mux tap through clock-high and follows the selected cell per
cycle. This exercises dynamic mux input selection, while the compiler still
has no physical column decoder.
`v224_mc_cases.json` is 1,444 per-device draws at each failure mechanism's
worst global corner, corner-case patterns under mismatch, and the 52 tall
draws that exposed the V2.2.3 replica-leakage sense failure (with their V2.2.3
seeds). The 285 V2.2.3 cases are the 239 V2.2.2 cases inside the supported
envelope, unchanged, plus 46 that close the gaps the V2.2.2 review recorded. `ScreenManifestTests` in `tests/test_phased_waveforms.py`
holds each property below, so removing a case to save runtime is a deliberate
act rather than a silent one.

| Dimension | Covered | Not covered |
|---|---|---|
| Array size | every row-class and column-class bound, and arrays large in both dimensions to 256x256 (65,536 cells): 64x64, 64x128, 128x64, 128x128, 256x128, 256x256 | the envelope corner 512x256 itself; a per-device draw above 128x128 (a 256x256 model card would be about 7 GB) |
| Array weighting | 134 of 285 cases at 8x4, so the small arrays still carry the fast-corner ordering margins | mismatch at 256x128 and above |
| PVT | every corner at two operating points: FF at 1.1 V/-40 C and 0.9 V/125 C, SS at 0.9 V/125 C and 1.1 V/-40 C, SF at 0.9 V/125 C and 0.9 V/-40 C, FS at 1.1 V/-40 C and 0.9 V/125 C, TT at 1.0 V/25 C over six array sizes | a full PVT grid; each new point is one or a few array sizes |
| Column data | complementary words at 8, 64, 128 and 256 columns, and at 64x64 | line-to-line coupling, which the wire model folds to ground and no case can exercise |
| Idle | 8 cases with one idle cycle between accesses (`select_every: 2`) | a multi-cycle stall, and a stopped clock in either state |
| Cell probes | every cell when `rows * cols <= 1024`; otherwise the quarter points, every addressed row and each addressed row's neighbours, plus every row's WL endpoints always | electrical disturb on a row that is neither addressed, adjacent nor a quarter point |
| Wiring / cells | the default illustrative wires, equivalent mode 0 | a custom `interconnect` (no case sets one), equivalent modes 1-4 |

`v224_negative_cases.json` holds the same six controls, at 8x4, 32x32, 64x64, 128x8,
8x256 and 512x4, each at a period the lookup would not grant, so different
margin families are the ones that break. V2.2.2 had a single 2 ns 8x4 control:
it failed 212 of 1,442 checks including the data families and two of the twelve
reported margins, leaving the other ten asserted rather than demonstrated. A
metric's non-negativity check is emitted only when the metric is derivable, so
a case whose crossing search finds nothing contributes no check rather than a
failure; in the V2.2.2 screen every record carried every metric its operation
admits.

The screen uses illustrative distributed wires and equivalent mode 0. Passing
these cases is functional screening, not a yield estimate, extracted-metal
qualification, or proof for arbitrary sizes/PVT. It never promotes a record to
`sizing_table.json`. The V2.2.2 coverage this manifest answers is
[what the screen does not cover](../../docs/design/PHASED_CONTROL_V2_2_2.md#what-the-screen-does-not-cover),
and the mapping from each gap to each new case is in
[the V2.2.3 record](../../docs/design/PHASED_CONTROL_V2_2_3.md) and
[the V2.2.4 record](../../docs/design/PHASED_CONTROL_V2_2_4.md).
