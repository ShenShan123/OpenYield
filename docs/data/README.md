# Supplied characterisation data

`PHASED_CONTROL_V2_2_2.json` is the assembled V2.2.2 functional screen: one
record per case with its clock, corner, seed, solver, deck, waveform and
scorer hashes and the worst value of every ordering and budget margin, the
tracked source hashes every trace was generated from, and the negative
control. It is written by the ignored assembler (`dev/v222/assemble_release.py`),
which refuses an incomplete, failing or drifted screen, and it is described in
[the V2.2.2 record](../design/PHASED_CONTROL_V2_2_2.md).
`PHASED_CONTROL_V2_2_1.json` is the preceding V2.2.1 screen with its
deck-reproduction and equivalence proofs, described in
[the phase-control record](../design/PHASED_CONTROL_V2_2_1.md).
Like the CSVs below, keep it as written; corrections belong in the documents
that cite it.

Read the V2.2.2 record's `coverage` block before quoting a minimum from it. The
254 records are 131 SS, 49 FF, 32 SF, 30 FS and 12 TT; each corner appears at
exactly one voltage and temperature; 172 of 254 cases are at 16x8 or smaller;
and every array satisfies `rows * cols <= 4096`. The `minimum_metrics` block is
therefore the worst value over that population, not over the compiler's input
space, and several of its minima are single per-device draws. The population's
gaps are enumerated in
[what the screen does not cover](../design/PHASED_CONTROL_V2_2_2.md#what-the-screen-does-not-cover).

These two CSVs are the only numeric record of the V2.0.2–V2.0.4 timing and
driver-sizing campaigns; their raw decks and waveforms were not retained. They
moved here from the repository root on 2026-09-13 (after the V2.1.2 release)
with their bytes unchanged. Preserve them as supplied; corrections belong in
the documents that cite them, not in the files.

| File | SHA-256 | Rows × columns | First commit | Content |
|---|---|---|---|---|
| `DRIVER_SIZING_data.csv` | `9c8c0671ce24eb8f2d201a132429023f09eea7f1b0dd870dd924987a352eb990` | 163 × 37 | `d845c42` | V2.0.4 driver-sizing decks, one row per deck |
| `TIMING_AUTOCONFIG_data.csv` | `c1fe8083664edfb51acd797a88883653ef788abc97697a03155975b839345c84` | 615 × 19 | `d30a23b` | V2.0.2 sweeps and V2.0.3 worst-case/validation decks, one row per deck or MC sample |

Both files are byte-identical in every commit that contains them. No tracked
file pinned these hashes before this record; the V2.1.2 changelog's
"hash-pinned" wording referred to `dev/sizing/campaign.py`, which the scoring
manifest pins, not to the CSVs.

## Why they are kept

- `DRIVER_SIZING_data.csv` is the evidence named by
  `sram_compiler/sizing/sizing_rules.json` (the V2.0.5 `rules_only` rules from
  which the V2.0.9 lookup classes were derived) and by section 3 of
  [the driver sizing proposal](../DRIVER_SIZING_PROPOSAL.md) and its
  [V2.0.4 snapshot](../design/DRIVER_SIZING_PROPOSAL_V2.0.4.md). The rule file
  keeps the bare file name: it is part of the frozen driver baseline digest.
- `TIMING_AUTOCONFIG_data.csv` backs section 3 of
  [the timing proposal](../TIMING_AUTOCONFIG.md) and the V2.0.2/V2.0.3
  changelog entries. The local `dev/sizing/campaign.py` reads its 27
  `v202_sweep` array sizes as the default campaign matrix. The compiler's
  `timing_lookup.json` is **not** derived from it (see
  [the V2.1.0 timing review](../design/TIMING_LOOKUP_V2_1_0.md));
  `provisional_period()` in `sram_compiler/sizing/timing.py` hard-codes the
  proposal's section 3.3 fit coefficients for the calibration clock only.

## Columns

`DRIVER_SIZING_data.csv`: `grp` (A_wd write driver 50, B_pre precharge 40,
C_wl wordline 20, D_rep replica 53), `tag` (array, operation, corner and swept
knob; `mc10` = ten samples), `status` (the run finished, `ok`, or was
`rescored`; it is not a pass flag), `n`/`nfail`/`failed` (samples, failing
samples, failed checks), times in ps, `pavg_uW` in µW, levels in V, `t` runtime
in s. Corner, VDD, temperature, seed, clock and solver exist only in the tag
and the proposal text.

`TIMING_AUTOCONFIG_data.csv`: `source`, `cell`, `rows`, `cols`, `mux`, `op`,
`corner`, `temp` (°C), `vdd` (V), `T` (clock period, s), `mc`, phase measures in
ps, `wave` (waveform checks passed) and `failed` (reason). `low = tclk_wlen +
access` within 0.1 ps (559 complete rows) and `high = trestore` exactly (564
rows).

## Audit, 2026-09-13

Scripts parsed the documents' tables and compared them with the CSVs at the
printed precision; no simulation was run. The scripts and outputs are local
(ignored) under `outputs/validation/V2.1.2-rerun/csv-audit/`. The counts, the
discrepancy tables and the cited deck values were re-checked separately; the
numbers of matching table cells are script results.

### `DRIVER_SIZING_data.csv` — sound

- 163 decks as claimed, no duplicate tags, no NaN/inf, consistent
  `n`/`nfail`/`failed`, all levels within 0–1.2 V. Five baseline decks appear
  under several sweep tags (13 rows) and carry identical measures.
- Two `rescored` box-cell writes (`8x4_w_SF_box_wd0.5`, `wd0.75`) fail
  `q_written`; `16x16_r_SS_K2N1_matched_T1.0` has `status=ok` but fails five
  sensing checks. The proposal reports all three as failures.
- 681 table cells in proposal sections 3.1–3.4 were compared; the working copy's
  tables equalled the V2.0.4 snapshot. Nine cells differed. They are corrected
  in `DRIVER_SIZING_PROPOSAL.md`, including the section 4.3 repeats; the
  snapshot keeps the printed values:

| Deck | Quantity | Printed | CSV | Corrected | Cause |
|---|---|---:|---:|---:|---|
| `512x4_r_SS_K1N9` | read access (ps) | 1676 | 1682.57 | 1683 | Transcription error |
| `512x4_r_SS_K1N9` | s_en − WL (ps) | 1560 | 1562.12 | 1562 | Three significant figures |
| `256x8_r_SS_K1N9` | read access / s_en − WL (ps) | 1210 / 1080 | 1208.60 / 1081.09 | 1209 / 1081 | Three significant figures |
| `512x4_r_SS_K2N5` | read access (ps) | 1040 | 1036.85 | 1037 | Three significant figures |
| `16x16_r_TT_K4N5`, `16x16_r_FFm40_K1N1`, `8x4_r_SS_K2N1_matched` | TSA / s_en − WL (ps) | 33 / 45 / 90 | 32.49 / 44.48 / 89.48 | 32 / 44 / 89 | Rounded twice |
| `16x16_w_SF_wd1.0_mc10` | driven BLB minimum (V) | 0.000 | 0.00085 | 0.001 | Rounded down |

- `sizing_rules.json` coefficients match proposal section 4.1. The data support
  the write-driver floor (box cell fails at 0.5 and 0.75 nominally and writes
  10/10 at 1.0; the default cell fails 3 of 10 MC samples at 0.5). `k_w`, the
  floor margin and the decoder divisor are policy choices without CSV rows;
  the NAND2 `cols/15` rule was measured only at 16x256; no 10T floor sweep exists.

### `TIMING_AUTOCONFIG_data.csv` — sound with caveats; citing documents corrected

Caveats in the file:

- Four `v202_after` rows are exact duplicates (6T 64x16 and 256x8, read and write).
- `v202_corners` (62 rows) and `v202_tsweep` (71 rows) are earlier runs of decks
  repeated in `v202_corners_f` (72) and `v202_tsweep_f` (75), with no
  superseded flag. Use the `_f` sources.
- MC rows (`mc` > 1) carry no sample index, and none of the 31 failing
  (`wave=False`) v202 rows records a `failed` reason.
- Missing measures: 39 `v202_tsweep` and 12 `v202_after` rows record only
  `access`; four `v202_sweep` rows lack `tclk_dec`; five `v203_validation` MC
  write samples lack `access` and `low`.
- No topology, RC or solver column exists, so statements about the wiring of
  these decks (for example in `design/WRITE_VALIDATION_V211.md`) cannot be
  checked against the file.

Document tables reproduced: section 3.2 phases 126/126; section 3.1 period
sweeps; section 3.5 MC 21/21; section 3.4 worst-case phases and periods 47/47;
the V2.0.2 changelog size table 60/60 and corner table 36/36 (`_f` rows).
Section 3.3 fits and the section 3.4 worst-case factor table reproduced except
two fit cells (6T read low `c` 9.053 printed 9.06; 10T write high rms 8.5
printed 9) and eight factor cells printed 0.01 off (for example 1.875 as 1.87);
those cells are corrected.

Discrepancies, corrected in the citing documents on 2026-09-13:

| Document claim | CSV | Correction |
|---|---|---|
| Changelog V2.0.3: "614 runs" | 615 rows | 615 rows |
| Timing proposal: "526" V2.0.2 runs | 494 `v202_*` rows (490 unique) | Row count cited |
| Proposal section 3.4 PVT factor table | Not reproducible from the final corner runs (39 of 72 cells differ from `v202_corners_f`); its "1.64 at 8x4 6T" SF write-access note uses superseded rows (215.0/130.8 ps; re-run 174.0/130.8 = 1.33) | Recomputed from `v202_corners_f`: mean of per-deck ratios to `v202_sweep` TT / 25 C, control phases from the read decks; parentheses 1.33 and 2.03 |
| TCLK_WLEN 112–150 ps, 290 ps for 512-column writes | 112.6–189.1 ps; 287.3–290.4 ps for 512-column writes | 112-190 ps (287-290 ps) |
| Mux changes phases by −16..+20 ps; changelog V2.0.2 "reads 0-16 ps faster" | access −28.7..+18.9 ps, high −22.3..+24.1 ps | −29..+24 ps; reads 0-29 ps faster |
| Worst-case restore is the longer write phase except 8x4 | 10T 16x16 write: low 566.6 ps, high 537.4 ps | Exception added |
| Write access at SS from 16 rows 2.31–2.45x | 2.31–2.41x (2.45x is 8x4) | 2.31-2.41x |
| Changelog V2.0.3: 10 % sigma on the 8x4 SS write access; proposal sections 3.5 and 4.3 ~30 % three-sigma "write access" | access 15.9 %, `low` phase 9.9 % (5 samples) | Sigma attributed to `low`; access 16 % |
| Changelog V2.0.3: control phases and read access 2.13–2.31x | 2.12–2.31x | 2.12-2.31x |
| Changelog V2.0.2: 10T within +4 to +20 ps of 6T | reads +2.4..+13.9 ps; writes −22.6..+23.5 ps (the V2.0.2 table agrees) | Stated per operation |
| Changelog V2.0.2: SS / −40 C fastest (8x4 read 226 ps), TT / 125 C control 1.65–1.8x, `TRESTORE` 460 ps | Superseded rows; final: TT / −40 C 202.9 ps, 1.64–1.74x, 436.6 ps | Final values |

The changelog's "108 corner runs" is correct: 72 final rows plus the 36
`read&write` decks of the V2.0.2 corner table, which the CSV does not hold.

`v202_sweep` holds 196 rows: 27 sizes × two cells × mux × read/write, minus the
20 mux-on decks of the five odd-column sizes. `v202_big` adds 128x128 and
256x64, which the campaign size list excludes.
