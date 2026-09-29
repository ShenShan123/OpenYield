# V2.2.5 corrected-model corner cases

**Status: 22 of 22 selected positives passed; diagnostic only.** This run exercises boundary and unusual SRAM configurations on the corrected V2.2.5 model-card hashes. It is separate from the stopped [full-coverage campaign](V2_2_5_FULL_COVERAGE_EVALUATION.md), whose results used the original model files and cannot certify this tree. [Machine-readable case evidence](data/V2_2_5_CORRECTED_CORNER_CASES.json) records result and source hashes, model hashes, seeds, and the eight corrected-model negative controls.

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
