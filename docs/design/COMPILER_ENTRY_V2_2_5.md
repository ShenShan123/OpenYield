# Compiler entrance and local variation — V2.2.5

V2.2.5 makes `main_sram.py` the documented entrance for generating one SRAM macro. `RUN_XYCE=False` writes a sampled deck and `summary.json` under ignored `outputs/main_sram/`; `RUN_XYCE=True` runs the same configuration with Xyce. The default process mode remains independent `vth0`, `u0`, and `voff` variation for every retained MOS. Nominal mode remains available for a fixed-corner diagnostic. The batch runner now passes `mc=True` for its per-device default and saves `summary.json` after successful deck-only generation.

The compiler rejects out-of-range row or column targets instead of silently clamping them. It also rejects unsupported operations, invalid sample counts and seeds, and nonfinite or negative mismatch sigma before emitting simulation directives. The main entrance validates its array dimensions and 6T widths. Failed samples cannot be reported as a successful mean by `main_sram.py`.

The cell, replica, peripheral, wire, driver, and clock circuits are unchanged. `timing_lookup.json` keeps its V2.2.4 version because its budgets were not changed; `sizing_lookup.json` likewise keeps its original artifact version. Existing V2.2.4 case manifests and evidence retain their original labels. The compiler identity written to new results is V2.2.5.

Generation checks exercised 144 small configurations across 6T/10T, mux choices, and all six operations (108 valid decks; 36 intentionally rejected odd-column mux cases); 16 tall, wide, and non-power-of-two configurations; 20 configurations across all five corners; and 12 complete deck exports. Focused local regression checks passed 44 tests and 567 subtests. `git diff --check` and Python compilation passed. Xyce was not available in this environment, so no V2.2.5 transient or DC waveform validation ran. The V2.2.4 manifests have not been executed as a V2.2.5 screen. The V2.2.2 assembled screen certifies only that earlier tree, and 256x256 still has no working operating point.

A later run located Xyce in the Conda environment and started a [V2.2.5 full-coverage evaluation](../V2_2_5_FULL_COVERAGE_EVALUATION.md). This design record retains the release-time validation statement above; the evaluation records subsequent results without promoting an incomplete screen.

## NMOS_VTH model-card follow-up

All five corner libraries were audited: each has six unique `.model` cards. TT already had one active `NMOS_VTH u0=0.049`; SS, FF, FS, and SF still had an earlier active `u0=0.05` in the same card. The earlier assignment is now commented out in all four files, matching TT, while the later `0.049` and every other parameter value remain unchanged. No duplicate parameter or model name remains in the 30 shipped cards.

Direct Xyce 7.4 DC probes that instantiate `NMOS_VTH` aborted on each old SS/FF/FS/SF file with `Duplicate specification of parameter U0`; the corrected files all solved. The compiler's per-device parser now rejects duplicate model parameters or model names, and nominal as well as per-device deck generation audits the selected corner before building a circuit. A test with the old SS file confirms both paths reject it early. The [diagnostic evidence](../data/V2_2_5_NMOS_VTH_U0.json) records old and new model hashes, solver outcomes, and eight read/write waveform cases across the four corrected corners; all eight passed 2,916 independent waveform checks on the final hashes. This targeted check does not replace the full screen.

The prior V2.2.5 campaign stopped at 1,594 of 1,778 positive passes, with its source snapshot and old model hashes preserved. No result from that partial run is promoted or silently reused for the corrected libraries. A fresh waveform screen is required for qualification. The release identifier remains V2.2.5.

A subsequent [corrected-model corner-case diagnostic](../V2_2_5_CORRECTED_CORNER_CASES.md) passed 22 selected positive cases with 20,960 waveform checks and confirmed eight matched negative rejections. It covers unusual small geometry, all five corners, mux selection, local mismatch, and TT no-stub wiring. It does not fill the remaining full-screen or 256×256 gaps.

Four additional medium class-boundary cases passed 87,975 waveform checks on
corrected SS/FF model hashes. The [V2.2.5 simulation summary](../V2_2_5_SIMULATION_SUMMARY.md)
separates those results from the stopped old-hash screen and other targeted
diagnostics; no full-screen qualification is claimed.
