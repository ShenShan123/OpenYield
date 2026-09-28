# V2.2.5 full-coverage evaluation — live record

**Status: in progress, not a qualification record.** This record tracks an Xyce waveform campaign on the V2.2.5 source. Update the counts and failures from saved case results; never count a generated deck, a running case, or a failed solver as a pass. The last assembled screen remains V2.2.2, and 256x256 has no working operating point.

## Scope and acceptance

The campaign uses the preserved V2.2.4 case definitions against compiler commit `206fbd8f73786d546203dedbdd6638707d1df21a` (V2.2.5). The manifests contain 334 main cases, 1,444 per-device draws, and six negative controls. The `dev/v224/{hero,big,mid,small}.json` launch queues cover 1,774 positive cases; four later dynamic mux-column reads are absent from those queues and will run as an additional queue. A full positive screen requires 1,778 scored passing cases. All six negative controls must fail for their intended waveform reasons. Source changes, missing or invalid measurements, and numerical non-completion count separately and prevent qualification.

Waveform scoring checks read and write data, retention, sensing, physical far-end wordline and bitline timing, precharge, isolation, and clock-low recovery. The runner records per-case model, seed, corner, VDD, temperature, RC, equivalent mode, source hashes, solver output, measures, and raw waveform. A passing campaign would still use illustrative distributed wire geometry and would not establish extracted-metal timing or a yield probability.

## Execution and provenance

- Solver: `/proj/workarea/user5/miniconda3/envs/openyield/bin/Xyce`, `Xyce DEVELOPMENT-202312200606-(Public_Release-7.4.0-36-gb7bb12d8)-opensource`.
- Conda environment: `openyield`, PySpice 1.5. Worker processes use one OMP, Kokkos, OpenBLAS, and MKL thread each.
- Source snapshot: ignored worktree `outputs/validation/v225-source` at the commit above, with the local ignored `tests/spice/*.py`, `tests/__init__.py`, and `sram_compiler/per_device_mc/sampling.py` copied in. Case metadata records source hashes.
- Results: ignored `outputs/validation/V2.2.5-phased/`. The `pilot-*` directories are diagnostics and do not count toward the full queue totals. Frozen launch manifests, logs, and queue progress belong under this root.
- Frozen launch manifests: `manifests/{hero,big,mid,small,dynamic,negative}.json`; `campaign.json` records their SHA-256 hashes and case counts. `dynamic.json` contains the four names omitted by the older queue split. The first background launch exited with no cases and preserved empty logs as `*.aborted-startup.log`; the active launch is held by a persistent shell session and recorded in `pids.txt`.
- Case runner: `python3 -m tests.spice.phased_access` from the source snapshot. Queue outputs are separate and must not overwrite prior attempts.

## Progress

| Checkpoint | Positive cases | Negative controls | Result |
|---|---:|---:|---|
| 2026-09-28, Xyce located | 0 / 1,778 | 0 / 6 | Conda Xyce 7.4 and PySpice 1.5 verified. |
| 2026-09-28, nominal pilot | 0 / 1,778 | 0 / 6 | `8x4_6t_SS_read` passed 369 waveform checks in 51.4 s; saved in `pilot-main`, outside campaign totals. |
| 2026-09-28 08:34 UTC, local-variation pilot | 0 / 1,778 | 0 / 6 | `mc_8x4_10t_SShot_s00` passed in 280.7 s; saved in `pilot-mc`, outside campaign totals. |
| 2026-09-28 08:34 UTC, full launch | 0 / 1,778 | 0 / 6 | Six persistent queues active: hero 2 workers, big 8, mid 8, small 20, dynamic 2, negative 2. Cases are running; no queue result had finished at this checkpoint. |
| 2026-09-28 08:37 UTC, first scored results | 2 / 1,778 | 1 / 6 | Two dynamic mux-column reads passed. `negative_8x4_SS_2ns` failed waveform checks as intended (1,442 checks, including isolation and precharge exclusion failures); Xyce exited 0. Other queues remain active. |
| 2026-09-28 08:39 UTC, dynamic queue complete | 4 / 1,778 | 1 / 6 | All four dynamic mux-column reads passed. The other positive queues have active Xyce cases and no scored failures yet. The negative queue has five controls pending or running. |

## Live queue snapshot

The local monitor reads saved `result.json` and `metadata.json` files and refreshes this table while the campaign runs. It does not treat a started case as a pass.

<!-- v225-progress-start -->
Updated 2026-09-28 08:44:41 UTC.

| Queue | Expected | Passed | Expected negative rejections | Failed | Solver/checker errors | Active | Pending |
|---|---:|---:|---:|---:|---:|---:|---:|
| hero | 8 | 0 | 0 | 0 | 0 | 2 | 6 |
| big | 164 | 0 | 0 | 0 | 0 | 8 | 156 |
| mid | 157 | 0 | 0 | 0 | 0 | 8 | 149 |
| small | 1445 | 0 | 0 | 0 | 0 | 20 | 1425 |
| dynamic | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| negative | 6 | 0 | 1 | 0 | 0 | 2 | 3 |
<!-- v225-progress-end -->

## Open outcomes

- The dynamic positive queue has completed; the hero, big, mid, small, and negative queues remain active. The 256x256 seeded operating point remains an explicit risk. Its preserved manifest timeout is 1,697,233 s (about 19.6 days), not evidence that the operating point will converge. Report it as incomplete if it cannot converge within a bounded diagnostic window.
- Do not assemble or promote a V2.2.5 evidence record until every required case finishes and the negative controls show their intended failures.
