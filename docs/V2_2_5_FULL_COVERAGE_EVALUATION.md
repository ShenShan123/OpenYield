# V2.2.5 full-coverage evaluation — partial record

**Status: stopped and incomplete; not a qualification record.** This record tracks an Xyce waveform campaign on the original V2.2.5 model files. Count only saved case results as passes. The last assembled screen remains V2.2.2, and 256x256 has no working operating point. A later V2.2.5 model-card correction changed four PDK hashes, so this partial campaign cannot certify the corrected tree.

## Scope and acceptance

The campaign uses the preserved V2.2.4 case definitions against compiler commit `206fbd8f73786d546203dedbdd6638707d1df21a` (V2.2.5). The original manifests contain 334 main cases, 1,444 per-device draws, and six negative controls. The `dev/v224/{hero,big,mid,small}.json` launch queues cover 1,774 positive cases; four later dynamic mux-column reads are absent from those queues and run as an additional queue. A full positive screen requires 1,778 scored passing cases. All six original negative controls must fail for their intended waveform reasons; a [V2.2.5 supplement](V2_2_5_NEGATIVE_CONTROL_EVALUATION.md) adds eight matched scenarios under a separate result root. Source changes, missing or invalid measurements, and numerical non-completion count separately and prevent qualification.

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
| 2026-09-28 12:32 UTC, four-hour check | 178 / 1,778 | 5 / 6 | Small 171, mid 3, dynamic 4 passed; no positive failures or solver/checker errors. Five controls produced waveform failures with Xyce return code 0. The last control, `negative_512x4_SS_14ns`, is active. Hero and big have no completed cases yet. |
| 2026-09-29 06:33 UTC, 22-hour check | 1,556 / 1,778 | 6 / 6 | Small 1,445/1,445 and dynamic 4/4 complete; mid 83/157 and big 24/164 pass, with no positive failures or solver/checker errors. Every negative control failed waveform checks with Xyce return code 0. Hero 0/8; both first cases remain active. |
| 2026-09-29, expanded negative controls | — | 14 / 14 total | Eight additional V2.2.5 short-clock scenarios passed their diagnostic rejection criteria in a separate run. See the [negative-control evaluation](V2_2_5_NEGATIVE_CONTROL_EVALUATION.md); the main positive screen is still running. |
| 2026-09-29 14:22 UTC, interrupted screen audit | 1,594 / 1,778 | 6 / 6 | Small 1,445, dynamic 4, mid 113, and big 32 passed; hero 0. No runner or monitor process remained. No scored positive failures or recorded solver/checker errors; 184 positives have no result. Treat this as incomplete, not a pass. |

## Frozen partial queue snapshot

The local monitor updated this section while the campaign ran. It and the campaign workers had stopped by the 2026-09-29 14:22 UTC audit; the table below reflects saved results only. Cases without a `result.json` are incomplete, including previously started solves.

<!-- v225-progress-start -->
Frozen 2026-09-29 14:22 UTC. No campaign worker or monitor process was running.

| Queue | Expected | Passed | Expected negative rejections | Failed | Solver/checker errors | Incomplete |
|---|---:|---:|---:|---:|---:|---:|
| hero | 8 | 0 | 0 | 0 | 0 | 8 |
| big | 164 | 32 | 0 | 0 | 0 | 132 |
| mid | 157 | 113 | 0 | 0 | 0 | 44 |
| small | 1445 | 1445 | 0 | 0 | 0 | 0 |
| dynamic | 4 | 4 | 0 | 0 | 0 | 0 |
| negative | 6 | 0 | 6 | 0 | 0 | 0 |
<!-- v225-progress-end -->

## Open outcomes

- Model-card audit (2026-09-29): the as-run SS/FF/FS/SF `NMOS_VTH` cards had duplicate active `u0=0.05` and `u0=0.049` assignments. Saved campaign `model_sha256` values match those old files. Direct Xyce later rejected each old card when an `NMOS_VTH` device was instantiated; the corrected files now pass that device test. The [correction record](design/COMPILER_ENTRY_V2_2_5.md#nmos_vth-model-card-follow-up) lists the new hashes and targeted checks. This campaign ran on the old model hashes and cannot qualify the corrected tree.
- At the last live check, 256x256 had spent about 20 h 25 min in its DC operating-point calculation without a result. The 256x128 transient last reported 63.2% complete at 2026-09-28 23:31 UTC. Neither hero case produced a scored result before the campaign stopped. The preserved 256x256 manifest timeout is 1,697,233 s (about 19.6 days), not evidence that the operating point will converge.
- Do not assemble or promote a V2.2.5 evidence record from this partial campaign. A fresh screen on the corrected model hashes is required before qualification.
