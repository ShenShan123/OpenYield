# V2.2.5 configuration review and V2.2.6 fixes

Reviewed `5f5cbf6..ae53050`, including the compiler entrance, generation
validation, and subsequent model-card audit. Findings were reported before
production edits. The following configurations reproduce on V2.2.5:

| Priority | Configuration | Failure | V2.2.6 correction |
|---|---|---|---|
| P1 | 6T or 10T, `variation_mode='custom'`, any of `hold_snm`, `read_snm`, `write_snm` | Single-cell construction raises `TypeError` for `pmos_modle_choices`; `nmos_modle_choices` is also invalid. | Pass the existing factory arguments `pmos_choices` and `nmos_choices`. |
| P2 | A PDK library with a commented `.model` of the same name, a semicolon comment containing an old parameter value, or a following `.param` with the same parameter name | The V2.2.5 audit invents a duplicate and rejects a valid library, including nominal generation. Direct Xyce DC probes accept all three examples. | Parse only active `.model` statements and their continuation lines; ignore full-line and semicolon comments and stop at the next directive. Real case-insensitive duplicates still fail. |
| P2 | `main_sram.py` with `MC_RUNS=1.5`, `True`, or `'1'` | `int()` silently changes the requested count to one before validation. | Validate positive integral counts before conversion in the shared resolver; pass the original setting from the entrance. |
| P2 | 2×2 configuration, per-device mode, any SNM operation | The deck contains one cell but metadata says `full_device_coverage=true`. | Only full-transistor transient arrays can claim full-array mismatch coverage; correct deck exports, CLI summaries, and simulation variation records. |

The custom SNM argument defect and single-cell coverage issue predate V2.2.5;
the promoted entrance/export path exposes them. The model-audit rejection is a
V2.2.5 regression. The sample-count issue bypasses that release's new validation.

The code release is V2.2.6. Circuit topology, model cards, sizing and timing
tables are unchanged. All 30 shipped model cards parse to exactly the same
names, types and parameter values before and after the parser change. Existing
evidence and frozen artifact version labels retain their original identities.

## Reproduction and validation

Local regression scripts remain ignored under `tests/`; a fresh clone does not
contain them. `tests/test_v226_review.py` checks the four failures, active
duplicate rejection, both cell types, all SNM operations, invalid counts in all
four variation modes, and the retained transient coverage claim. Before the
fix it reported 19 assertion failures and 11 errors across six test methods.

Commands used from the repository root in the supplied `openyield` environment:

```bash
command -v Xyce
python -m unittest tests.test_v226_review
python -m unittest discover -s tests
python -m dev.v226_snm_review
python -m tests.spice.phased_access \
  --cases tests/spice/v226_review_cases.json \
  --output outputs/validation/V2.2.6-review-transient --workers 4
git diff --check
```

The DC diagnostics use TT, 1 V, 25 °C, mode 0, local RC enabled, the supplied
custom parameter rows, and seed 20261001 for per-device sampling. The configured
geometry is 2×2, but each SNM deck contains one cell. These diagnostics verify
successful execution and complete finite DC sweeps, not an SRAM array's
transient behavior or yield. Raw decks, logs and metadata live under ignored
`outputs/validation/V2.2.6-review-*`.

All checks passed with Xyce 7.4 (build
`DEVELOPMENT-202312200606-(Public_Release-7.4.0-36-gb7bb12d8)-opensource`):

- 195 compiler/local regression tests (including the six new methods) and six
  offline optimizer tests. The environment lacks `pytest`; the suites were run
  through their `unittest` interface. Python compilation and `git diff --check`
  passed.
- 18 DC cases / 27 complete finite sweeps across both cells, all three SNM
  operations, and nominal/custom/per-device modes. Each sweep contains 1,421
  points. Three additional simulation-API calls returned finite hold SNM for
  6T custom (two rows), 10T custom (one row), and 6T per-device (two samples).
- Deck-export, CLI, and simulation variation records reject the SNM full-array
  coverage claim; mode 0 per-device transient exports retain it.
- Four transient cases passed 1,467 independent waveform checks: 8×4 6T SS
  read and write, 8×4 6T mux SS dynamic-column read, and 8×4 10T mux FF
  dynamic-column read. These nominal mode 0 cases use the frozen lookup clocks,
  default illustrative distributed wiring and local 100-ohm/1-fF stubs. The
  checker examines control timing and physical endpoints, write/read data,
  retention, isolation, sense enable, and recovery at 5 ps output spacing.

The [diagnostic evidence](../data/V2_2_6_CONFIGURATION_REVIEW.json) records
model/source/scorer hashes, solver, seed, RC, equivalent mode, cases and margins.
The tracked `tests/spice/v226_review_cases.json` contains the four-case manifest.

## Follow-up: model-card builder and coverage rule

A post-commit review of `38312cb` found that the parser fix reached only the
audit. `utils.parse_spice_models` still built the cards for shared-mode
`AGAUSS` libraries and custom-mode `model_dict` cells. It read the commented
model, the semicolon comment, and the following `.param` from the P2 examples
as parameters, and the last value won. It also truncated `{0.4*1.0}` at `*`.
V2.2.5 rejected these libraries in the audit, but V2.2.6 accepted them, so
shared and custom decks could simulate values that Xyce does not read. Xyce 7.4
reads the active value: a semicolon probe gave −97 µA at `vth0=0.40`, compared
with 5e-11 A when 0.99 was active. Nominal and per-device decks were correct.

`utils.parse_spice_models` now delegates to the audit parser. Duplicate active
models or parameters therefore raise `SpiceParseError` instead of keeping the
last value. Xyce 7.4 rejects `$` inline comments, and the audit does too. The
unused `size_optimization/model_lib/models.spice` sets `toxref` twice in
`PMOS_VTL`, so the strict parser rejects it. The optimizer simulates
`tran_models/models_TT.spice`, and the compiler audit already rejected that
library. The file is unchanged.

Before this change, `full_device_coverage` was written three ways: a
transient-operation tuple in `main_sram.py`, `not in SNM_OPERATIONS` in the CLI,
and a literal in the simulation record. All three now call
`full_device_coverage()` and use `TRANSIENT_OPERATIONS` from
`sram_6t_core_MC_testbench.py`. The positive transient list fails closed if a
new operation is added.

Checks on the follow-up, using Xyce 7.4 in the same environment:

- The two parsers return identical results for all 30 shipped corner cards.
  Deck generation was repeated for 48 configurations: 2×2 and 8×4; all six
  operations; and nominal, shared (3 runs), custom, and per-device (seed 7)
  modes. The 144 generated decks, model files, summaries and data files are
  byte-identical to `38312cb`'s once the output root is normalized. Coverage
  flags are true only for the six per-device transient decks.
- `tests/test_v226_review.py` has three new methods. They cover parser
  agreement on the examples, a shared-mode library with a trailing commented
  `NMOS_VTG` card, and the coverage rule across all modes, equivalent modes
  and operations. With the old `utils` parser they fail with five assertion
  failures. All 198 local regression tests and six offline optimizer tests
  pass, and Python compilation succeeds.
- `run_mc_simulation` returned finite hold SNM with
  `full_device_coverage=false` for 6T shared (2 runs), 6T custom (2 rows),
  10T custom (1 row) and 6T per-device (2 samples, seed 20261001). A 2×2
  per-device read recorded `true` and passed its runtime access checks.

The evidence JSON's source hashes identify the `38312cb` tree. This follow-up
changes `main_sram.py`, `utils/spice.py`, `per_device_mc/run.py` and
`sram_6t_core_MC_testbench.py`. The generated decks for the shipped models are
identical, so the recorded diagnostics still describe the decks this tree
generates. The version label remains V2.2.6.

## Remaining limits

No full V2.2.6 waveform screen has completed. The targeted checks do not resolve
the 256×256 operating point, qualify arbitrary PVT/RC or equivalent modes, or
establish yield. The stopped V2.2.5 campaign and corrected-model diagnostics
retain their [original scope](../V2_2_5_SIMULATION_SUMMARY.md); the last assembled
screen remains V2.2.2 and certifies only that earlier tree.
