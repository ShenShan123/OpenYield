# Phased SRAM access integration checks

These opt-in tests exercise real Xyce waveforms for V2.2.2. They are separate
from `python3 -m unittest discover -s tests`, which does not launch this screen.
Run from the repository root with the normal OpenYield Python dependencies and
Xyce 7.4 available.

```bash
python3 -m tests.spice.phased_access \
  --output outputs/validation/my-phased-screen \
  --xyce "$(command -v Xyce)" --workers 4
```

Each case directory must be new. A repeated run needs a fresh output directory;
failed attempts are evidence and must not be overwritten. `--name NAME` selects
a case and can be repeated; `--cases FILE` selects another JSON manifest.

`v220_cases.json` covers consecutive reads, consecutive writes, alternating
reads/writes, different row addresses and column patterns, idle cycles, 6T/10T,
mux/no-mux, SS/SF and FF/FS, and seeded per-device mismatch. Large nominal cases
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
`min_precharge_on_before_capture_ps`. The manifest also carries twelve
per-device read&write cases at 32x32 to 8x512 across all four corners.

The arbitrary-pattern cases replace only the external PWL sources. Their data
expectations come from the pattern, not the compiler's fixed WRWR measurement
schedule. Solver failures, source changes during a campaign, and Python/checker errors
cause a nonzero runner exit. Actual PVT model selection is regression-tested. Simulation/model/seed metadata and source hashes stay beside the deck.

## What the manifest covers, and what it does not

`v220_cases.json` is 254 cases over 12 array sizes. The shape of that coverage
matters more than its size:

| Dimension | Covered | Not covered |
|---|---|---|
| Array size | every row-class and column-class bound (512x4, 8x512, 256x4, 8x256, 128x8, 8x128, 64x16, 32x32) | any array in a high row class and a high column class at once; every case satisfies `rows * cols <= 4096` |
| Array weighting | 172 of 254 cases at 16x8 or smaller; 52 of 72 per-device cases at 8x4 | one mismatch sample each at 256 rows, 512 rows, 256 columns and 512 columns |
| PVT | SS and SF at 0.9 V / 125 C, FF and FS at 1.1 V / -40 C, TT at 1.0 V / 25 C | any other voltage/temperature per corner; SF cold, FS hot; TT above 8x4; seven non-SS cases in total above 16x8 |
| Column data | complementary words (`10101010`, `01010101`) at 16x8 | every wider case writes one value to all columns, so data dependence above 8 columns is unscreened |
| Idle | 8 cases with one idle cycle between accesses (`select_every: 2`) | a multi-cycle stall, and a stopped clock in either state |
| Cell probes | every cell when `rows * cols <= 1024`; otherwise `{0, rows // 2, rows - 1, target, next_row}`, plus every row's WL endpoints always | electrical disturb on an unprobed row of a large array |
| Wiring / cells | the default illustrative wires, equivalent mode 0 | a custom `interconnect` (no case sets one), equivalent modes 1-4 |

The negative control is a single 2 ns 8x4 SS case. It fails 212 of 1,442
checks, which does include the data families and two of the twelve reported
margins, but the other ten margins have never been observed to fail, so their
sensitivity is asserted rather than demonstrated. A metric's non-negativity
check is also emitted only when the metric is derivable, so a case whose
crossing search finds nothing would contribute no check rather than a failure;
in the V2.2.2 screen every record carries every metric its operation admits.

The screen uses illustrative distributed wires and equivalent mode 0. Passing
these cases is functional screening, not a yield estimate, extracted-metal
qualification, or proof for arbitrary sizes/PVT. It never promotes a record to
`sizing_table.json`. The consequences of the table above, with the numbers, are
[what the screen does not cover](../../docs/design/PHASED_CONTROL_V2_2_2.md#what-the-screen-does-not-cover).
