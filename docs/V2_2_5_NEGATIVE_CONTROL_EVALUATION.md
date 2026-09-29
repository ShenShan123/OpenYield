# V2.2.5 negative-control evaluation

**Status: final manifest rerun pending.** This diagnostic expands the six preserved V2.2.4 short-clock controls with eight V2.2.5 cases in [`tests/spice/v225_negative_cases.json`](../tests/spice/v225_negative_cases.json). It does not qualify the unfinished positive waveform screen.

## Acceptance rule

Each added case starts from a V2.2.5 positive case that passed in the active full screen. Only its clock period is shortened; cell type, mux, target, PVT, stimulus, mismatch seed, and physical RC remain the same. The negative case counts as a valid control only when Xyce exits successfully, waveform scoring returns `passed: false`, and **every** `expected_failure_checks` entry in the manifest appears in `result.json`. A solver timeout, missing waveform, or checker exception is inconclusive. The original six controls retain their V2.2.4 case definitions and must fail waveform checks with a successful solver exit.

| New control | Passing reference | Clock | Expected waveform failure |
|---|---|---:|---|
| 6T SS read | `8x4_6t_SS_read` | 4 ns | Sense event and output deadline |
| 6T SS write | `8x4_6t_SS_write` | 4 ns | Written-cell retention and bitline restore |
| 6T SS changed-row write | `8x4_6t_SS_write_nextrow` | 4 ns | Next-row retention and bitline restore |
| 6T mux input-0 read, background 1 | `8x4_6t_mux_SS_read_col0_bg1` | 4 ns | Isolation pulse and output deadline |
| 6T SF write | `8x4_6t_SF_write` | 2.5 ns | Logical retention and wordline pulse |
| 10T mux SF write | `8x4_10t_mux_SF_write` | 2.5 ns | Logical retention and wordline pulse |
| 6T SS seeded local-mismatch read | `8x4_6t_SS_read_pd202609191` | 4 ns | Sense event and output deadline |
| 6T SS alternating write/read | `8x4_6t_SS_read_write` | 4 ns | Write retention and next read's replica reset |

## Probe evidence and limits

The cases were chosen from two exploratory runs of 18 variants on compiler commit `206fbd8f73786d546203dedbdd6638707d1df21a` with Conda Xyce 7.4. Eight variants failed waveform scoring with Xyce return code 0 and became the new controls. Ten trial variants still passed at their shortened periods and were excluded; in particular, fast-corner reads and an idle write were not mislabeled as negatives. The probe output is under ignored `outputs/validation/V2.2.5-negative-probes*`; final-case output will be kept separately.

The new cases exercise more failure families and contexts, but all use clock compression. They do not inject faults into the circuit or prove that every checker family is sensitive to an independent defect. The original six controls cover larger geometries, including 512x4; no new large-array control is needed to establish that the small-array scenarios fail.

## Progress

| Checkpoint | Result |
|---|---|
| 2026-09-29, candidate selection | All eight positive references passed in the ongoing V2.2.5 full screen. Probe outputs showed the expected failures with successful Xyce exits. Final manifest rerun remains pending. |
