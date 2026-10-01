# V2.2.5 corrected-model corner cases

**Status: both corrected-model tranches complete; diagnostic only.** The first tranche passed 22 of 22 small and irregular positives and eight matched negative controls; the medium class-boundary tranche passed 4 of 4. These runs are separate from the stopped [full-coverage campaign](V2_2_5_FULL_COVERAGE_EVALUATION.md), whose results used the original model files and cannot certify this tree. The [first-tranche evidence](data/V2_2_5_CORRECTED_CORNER_CASES.json) and [medium-tranche evidence](data/V2_2_5_CORRECTED_MEDIUM_CASES.json) record per-case results and provenance.

## Frozen inputs and acceptance

- Source commit: `8a8bd770fec157368eeb825c297aea1d22d52514` (the V2.2.5 model-card correction).
- Case source: 22 entries copied unchanged from tracked `tests/spice/v224_cases.json` into ignored `outputs/validation/V2.2.5-corrected-corners/cases.json`. Frozen subset SHA-256: `c7594695840d48ca16b3f44042fa817a2df2b137493f8a124c5e67c33ef10929`.
- Solver: Conda Xyce 7.4, eight one-thread workers. Results and source/model hashes are saved per case under ignored `outputs/validation/V2.2.5-corrected-corners/results/`; `campaign.json` stores the selected model hashes.
- A case passes only if Xyce completes and the independent waveform scorer passes read/write data, retention, sensing, physical far-end access timing, precharge, isolation, and recovery checks. A generated deck or passing `.MEASURE` alone is insufficient.

| Coverage group | Cases | Configuration edges |
|---|---:|---|
| Degenerate and odd geometry | 10 | 1×1 6T/10T, 1×4 mux, 2×1, 3×3, 5×7 changed-row read, 6×2 mux, 12×5, 17×3 |
| Static mux input 0 | 2 | SS 6T read against background 1, FF 10T read/write |
| Dynamic mux selection | 4 | 6T/10T at SS and FF, both mux inputs between cycles |
| Seeded local mismatch | 4 | 8×4 read/write and 16×8 muxed column-0 sequences at SS/FS/SF |
| TT without local stubs | 2 | Read and write with distributed wires retained |

The 22 matched cases passed in the stopped campaign on the original model hashes. On the corrected hashes, all 22 again passed, with 20,960 independent waveform checks and no solver or scorer error. For each pair, the reported check count and metric dictionary are identical; the model hashes differ for SS, FF, FS, and SF. This comparison does not assert that every waveform sample is byte-identical and does not transfer qualification from the old run.

The smallest reported control-ordering margins in this selected set were 83.49 ps between enable-off and precharge-on (6×2 6T/mux FF) and 99.98 ps between the physical wordline turning off and sense enabling (8×4 6T/mux FF dynamic-column read). The minimum read-output deadline margin was 1,312.53 ps (8×4 SS per-device read); the minimum far-end bitline restore-before-capture margin was 2,882.17 ps (3×3 SS read/write). These are minima over the selected cases only.

This diagnostic does not cover the full 512×256 envelope, all PVT combinations, or a yield distribution; 256×256 still has no working operating point.

The eight added controls in [`tests/spice/v225_negative_cases.json`](../tests/spice/v225_negative_cases.json) were rerun under `outputs/validation/V2.2.5-corrected-corners/negative-new/`. All eight had Xyce return code 0, were rejected by waveform scoring, and triggered both named `expected_failure_checks`. Each of their eight positive references passed on corrected model hashes with the same driver baseline. The six larger preserved negative controls have not been rerun on corrected hashes.

## Progress

| Checkpoint | Scored passes | Failures or errors | Note |
|---|---:|---:|---|
| 2026-09-29, launch | 0 / 22 | 0 | Corrected-model cases running from the fresh output root. |
| 2026-09-29 14:40 UTC | 8 / 22 | 0 | Degenerate and odd small-array cases are passing; 16 cases have started. No solver or scorer error is recorded. |
| 2026-09-29 14:43 UTC | 17 / 22 | 0 | All 22 cases started. Passing results now include static and dynamic mux selection, TT no-stub read, SS local-mismatch read, SF 10T mux write, and 12×5 10T read/write. |
| 2026-09-29 14:51 UTC | 21 / 22 | 0 | The 16×8 6T SS per-device muxed column-0 sequence passed. Eight of eight added negative controls produced waveform rejections; two of three extra positive references passed. The 16×8 10T FS per-device case and one reference remain active. |
| 2026-09-29, completed diagnostic | 22 / 22 | 0 | The 16×8 10T FS per-device sequence and the last matched reference passed. Eight of eight corrected-model negative controls met their expected failure checks. Evidence assembled; no full-screen qualification claimed. |

## Medium class-boundary tranche

Four unchanged tracked cases ran from a source worktree pinned at commit `744c6a41fc2d5251aa9a0f908e3677ee54f96d87`, after the model-card correction. The frozen subset is `outputs/validation/V2.2.5-corrected-medium/cases.json` (SHA-256 `c6f508b5e2a79d1b67458701b74c9a1688ae642e4deaba67265124bad13bac0a`); per-case outputs are under `outputs/validation/V2.2.5-corrected-medium/results/`. Conda Xyce 7.4 ran four workers and exited 0. Every saved result passed waveform scoring with the corrected model hash.

| Case | Boundary under test | Waveform checks | Xyce elapsed |
|---|---|---:|---:|
| `33x33_6t_SShot_read_write` | One past the 32-row and 32-column classes | 9,049 passed | 2,001 s |
| `65x9_10t_SShot_read_write` | One past the 64-row and 8-column classes | 13,476 passed | 2,347 s |
| `8x256_6t_mux_SS_read_write_col0` | Supported maximum column count, mux input 0 | 55,674 passed | 7,176 s |
| `257x4_10t_mux_FFcold_read_write` | One past the 256-row class at FF cold | 9,776 passed | 7,462 s |

The four medium cases passed 87,975 waveform checks with no solver or scorer failures. Their smallest reported enable-off-before-precharge margin was 102.82 ps, and the smallest wordline-off-before-sense margin was 102.84 ps, both at 257×4 FF cold. The 257×4 case's reported metrics and check count match its old-model run, but the old result is not reused as evidence for the corrected model. These four cases extend the diagnostic toward the size-class edges; they do not include 512×256 or resolve the 256×256 operating point.

The medium worktree's source-hash inventory omits the ignored, unused `sram_compiler/equivalent_modeling/compare.py` file present in the root workspace. This makes its aggregate source hash differ from the first tranche's, although the tracked runtime and model hashes match. Both distinct source hashes are recorded with the respective cases. Counts may be summed as separate diagnostic runs, never as one completed screen.

| Checkpoint | Scored passes | Failures or errors | Note |
|---|---:|---:|---|
| 2026-09-29 14:59 UTC, launch | 0 / 4 | 0 | All four cases started; results pending. |
| 2026-10-01 08:39 UTC, completed | 4 / 4 | 0 | Runner exited 0; 87,975 waveform checks passed on corrected SS/FF hashes. Results and provenance saved in the medium-tranche evidence. |
