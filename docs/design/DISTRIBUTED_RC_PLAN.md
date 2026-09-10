# Parasitic RC correction and distributed interconnect plan

Status: implemented for V2.0.7 after the 2026-09-09 review of V2.0.6; audited
in V2.0.8 (2026-09-09), see the [audit status](#v208-audit-status) below.
See [the model guide](DISTRIBUTED_RC_MODEL.md) for configuration and compatibility,
and [the validation record](DISTRIBUTED_RC_VALIDATION.md) for the evidence.
Extracted-array sizing and yield qualification remain separate from this
implementation.

Correct SRAM read, write and hold behavior before recalibrating driver sizing
or estimating yield. The V2.0.6 model used ideal shared row/column nets with
local RC branches, so it cannot represent distance-dependent wire delay.
The [V2.0.5 wordline comparison](../qualification/V2.0.5_wordline_model.json)
provides reference evidence; its segment values are illustrative, not extracted.

1. **Fix the existing RC configuration bugs.** Forward `pi_res` and `pi_cap`
   through every factory and nested real-cell, replica, dummy and peripheral
   constructor. Resolve the testbench's 10-ohm versus subcircuit's 100-ohm
   default explicitly. Derive sizing loads from actual configured capacitance.
   Give equivalent extraction the same corner, voltage and temperature as the
   simulated array, and include PDK content and extraction settings in its cache
   identity. Add regression checks using distinct nondefault R/C values.

2. **Define an explicit distributed mode.** Retain the star topology for
   historical reproducibility and introduce distributed wiring as opt-in.
   Configure WL and BL/BLB separately using cell pitch, metal geometry and
   technology-derived resistance/capacitance; keep geometry in SI metres.
   Separate wire parasitics from local pin, storage-node and peripheral
   parasitics to avoid double counting. Append optional API arguments.

3. **Build physical wire segments and taps.** Start with full-real 6T/10T
   arrays: connect each cell to its own WL and BL/BLB taps through shared series
   wire segments. Use pi sections with each segment's total capacitance split
   between its endpoints. Connect WL drivers at column zero and the bitline
   precharge, write, mux and sense circuits at row zero. Extend equivalent modes by attaching
   omitted-cell loads at local taps; preserve wire length and never replace a
   wire ladder with the current parallel-branch `R/N` aggregate.

4. **Match timing and qualification identities.** Give replica WL/RBL paths
   matching physical lengths, tap locations and loads. Measure local cell
   wordlines and bitlines, including far-end wordline release before precharge
   and sense-input differential at enable. Include topology, geometry and R/C
   settings in baseline fingerprints and qualification records. Keep baseline
   driver sizes frozen across cell candidates and PVT samples.

5. **Validate functionality, then sizing and yield.** Check connectivity and
   total wire R/C for both cell types, mux settings, numeric/sweep dimensions
   and real/equivalent modes. Compare near, middle and far positions using
   Xyce waveforms for read/write 0 and 1, hold, half-select disturbance,
   precharge restoration and next-row hazards. Check segment refinement and
   equivalent-model accuracy against full-real arrays. After nominal behavior
   passes, qualify PVT and per-device mismatch before recalibrating sizing.

## V2.0.8 audit status

| Item | Status |
|---|---|
| 1. RC configuration bugs | Done. Regressions in `tests/test_rc_configuration.py`; 72 star decks unchanged between V2.0.7 and V2.0.8. |
| 2. Explicit distributed mode | Done. V2.0.8 fixes the `cell_pin_rc` default of a directly constructed `InterconnectConfig` and adds `main_sram.py`'s `INTERCONNECT_CONFIG`. |
| 3. Wire segments and taps | Done for full-real and equivalent arrays. Periphery positions are fixed (WL drivers at column zero, bitline periphery at row zero); no wire coupling. |
| 4. Timing and qualification identities | Done. Replica lengths, taps and far-wordline guard are tested; identities include the wire model. No qualified record exists for any distributed configuration. |
| 5. Validation | Nominal function checked on 4x4 (V2.0.7) plus 8x4, 4x8 and 16x16 arrays, illustrative metal only (V2.0.8). Half-select column disturbance is not exercised: the sequence decks write every column. PVT/mismatch qualification and sizing recalibration remain open. |

Completion requires passing waveform checks, not just generated decks or
successful measures. Compare any changed fixed-mode default decks against the
previous commit in a detached worktree and document intended differences.
Preserve existing CSV/JSON evidence and V2.0.5 labels; use new evidence identities
for changed models and never promote partial or failed runs. Keep generated
results under ignored `outputs/`, ad hoc tools under `dev/`, and reusable
regressions under tracked `tests/`.
