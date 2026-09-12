# Automatic driver sizing for all array sizes — proposal (V2.0.6)

Status (2026-09-10): **V2.0.9** replaces the continuous sizing rules of this
proposal, as the shipped strategy, with a lookup table of fixed integer size
classes (`sram_compiler/sizing/sizing_lookup.json`, `sizing.mode: lookup`,
see the [sizing guide](../sram_compiler/sizing/README.md)). The motivation is
layout generation from fixed transistors: every array configuration now maps to
one row class (≤32/64/128/256/512 rows for the precharge and the split write
driver) and one column class (≤4…512 columns for the wordline NAND2/inverter and
the decoder output inverter), independent of cell type, mux, RC and wires. Each
class is the V2.0.5 rule of section 4.1 (pre-review column, `k_w` = 1/16,
`rows/32`, `cols/4`, `cols/15`, `wl_nand/4`) evaluated at the class upper bound
and rounded up to an integer, so no array receives a weaker driver than the
screened rule; unseen sizes round up to the next anchor, and sizes beyond the
table extrapolate the ladder geometrically and are flagged. The legacy `fixed`
mode (the original `rows/16` and `sqrt` array rules of section 1.1, with the
8-row write weakness) was removed; `rules_only` remains as the derivation basis
and the qualification tools take `--sizing-mode lookup`. The function-first
load terms (`rows/64`, section 4.1) and the `(2, 1)` replica remain proposals;
Stages D-F below now apply to class candidates rather than coefficient
candidates, and any changed class needs new waveform evidence before it enters
the table. The V2.0.9 changelog records the three-sample fixed-class screen.

Status (2026-09-08): **V2.0.6** integrates the default local mismatch flow into
`sram_compiler/per_device_mc/` and relocates the working plans into `docs/`.
The V2.0.5 full-local driver and timing re-evaluation remains in progress.
The integration checks passed 51 compiler/development tests and 6 offline
optimizer tests; read/write CLI artifacts matched the previous layout byte for byte.
Developer experiments and qualification helpers now live in ignored `dev/sizing/`,
with their tests in `dev/tests/`; reusable compiler tests live in tracked `tests/`.
The separated suites pass 34 compiler tests, 20 local development tests, and
6 offline optimizer tests. Commands below using `dev.sizing` require those local
files; see [DEVELOPMENT.md](DEVELOPMENT.md). Runtime table lookup uses a tracked
source-hash manifest and does not require the ignored scripts.
No new Xyce simulation or electrical qualification is claimed for V2.0.6.

The following campaign checkpoints retain their original evidence versions.
The previous campaign was terminated and is historical evidence only (684
completed cases, 2,663 waveform samples, two unresolved simulator underflows).
No release qualification is claimed. The new campaign must independently vary
every instantiated MOS in both read and write paths, including the full array,
replica, precharge, write stacks, mux, sense amplifier, latches, decoder and
timing/control buffers. The implementation and measurements below are being
updated at each execution checkpoint; the 2026-09-08 review checkpoint below is
the first committed V2.0.5 state.
The full original design and measured basis remain below. Source revision
`d30a23b` plus the uncommitted V2.0.3 documents (`docs/TIMING_AUTOCONFIG.md`).
The numbers in section 3 come from a characterisation campaign run for this
proposal (163 Xyce decks, `DRIVER_SIZING_data.csv`, harness in the session
scratchpad, see section 3.0);
everything else is derived from the code. This revision replaces the earlier
draft of this file; section 10 lists what was kept from it.

Review 2026-09-07 (function-first revision, documentation only, no code
changed): the priority order is now (1) functional correctness at every
corner, (2) driver terms bounded relative to the array's own bitline-limited
phase, which grows with the array, (3) the absolute `global.yaml` access
limits as information only. The load terms of section 4.1 are re-derived from
that order (sections 0, 2, 4, 5, 7, 9 updated in place); section 11 lists the
blind spots found and where each is handled. The code still implements the
pre-review rules (`k_w` = 1/16, `rows/32`). That stopped campaign remains
historical nominal/shared evidence; the full-local re-evaluation supersedes
its qualification schedule.

## Full-local-mismatch execution schedule — 2026-09-08

Dependencies determine the schedule; measured pilot runtimes determine worker
count and wall-clock estimates. Do not restart the old broad campaign.

| Stage | Work and exit evidence | Status |
|---|---|---|
| A — default and coverage | One canonical MC path defaults to independent per-MOS `vth0`, `u0`, `voff`; full arrays by default; explicit nominal/shared comparisons; device/model audit for 6T/10T, read/write and mux on/off; numeric and swept-width regression checks | Complete: 48 driver tests and 6 offline optimizer tests pass |
| B — small pilot | 8x4, both cells/mux choices: nominal baseline calibration, then full-local read and PU/PD/PG width-box SS/SF write at a frozen clock; retain seeds, model/solver hashes, every waveform failure and elapsed time | Complete: 12 calibrations and 120 local samples pass; 2.35 ns for 6T, 2.40 ns for 10T |
| C — rule diagnosis | Small/floor, square, tall and wide representatives plus explicit RC. Re-evaluate write output/input, precharge, WL inverter/NAND, decoder and control buffers, and replica K/N. Measure retention, sensing/settling, restoration, races and phase budgets before changing coefficients | Screen complete 2026-09-08 (three samples per case, seed 82026): 24 architectures, 20 calibrated; 759 of 852 scheduled local samples scored, 723 pass every check; 36 fail only the wordline-phase budget (all 16x16 explicit-RC reads); 28 four-rank cases timed out at 1,800 s, 3 aborted at the DC operating point; the four 10T 64x64 / 16x256 architectures did not calibrate (four-rank nominal decks timed out). Details in the screen outcome below |
| D — coefficient candidates | Compare one rule family at a time with the same seeded device ensemble; rederive loads and clock once per baseline; freeze across cell/PVT samples; reject functional failures before area/timing ranking | Pending C |
| E — independent verification | Full historical size/cell/mux matrix and RC sensitivity, all PVT read/write checks with local mismatch; 100-sample read and write tail ensembles; sequences/hazards/half-select; independent validation seeds | Pending D |
| F — evidence and disposition | Export rule-by-rule findings, unresolved failures and exact coverage; promote only complete passing records with matching mismatch model, scope, solver, baseline and scoring provenance | Pending E |

Version contract: the user designated this revision **V2.0.6**. This is the
compiler/package and documentation release; the active sizing rule identity
remains `v2.0.5-local-1`, and coefficients remain unqualified starting values
from V2.0.4. The existing qualification tools retain their V2.0.5 artifact
format and default `outputs/qualification/V2.0.5/` directory. The package move
updates the scoring fingerprint to use runtime sources plus the tracked local-tool
source manifest; it does not promote earlier
records or relabel measurements. The pre-label integration and
finalized pilots retain their original `V2.0.4-local/` paths so their absolute
model includes and evidence remain reproducible. Original V2.0.4 measurements,
stopped campaign and archived proposal remain historical evidence.

Mismatch contract: use the existing independent Gaussian relative standard
deviation of 5% for each of `vth0`, `u0`, `voff` on each MOS. Corner selection
sets the nominal model; this local model does not also add shared/global random
variation. It has no area dependence or calibrated parameter correlations and
therefore supports model-based sensitivity conclusions, not a silicon-yield
claim. Devices represented by `NF` remain one MOS with one local draw; equivalent
cell modes are explicit approximations and cannot qualify full-device coverage.
Nominal baseline calibration remains useful, but every acceptance ensemble runs
under local mismatch; neither sizes nor clocks may adapt to individual samples.

Run policy: separate `outputs/qualification/V2.0.5/` from the stopped
campaign, checkpoint each case atomically, bound the executor queue, and inspect
pilot failures before spending on large arrays. Ten samples are screening;
100 samples and a Gaussian three-sigma calculation are model diagnostics, not
evidence of rare-event yield. Preserve the supplied CSV and original proposal.

Execution checkpoint: `pilot/` is the initial integration run; `pilot-v2/`
uses the finalized serializer, which retains the original model numeric
precision and immutable model files for each exported circuit. Do not combine
these ensembles. The canonical testbench specializes in `create_testbench()`;
both the CLI and sizing runner now use it. An 8x4 read audits 893 independent
MOS models. Every generated circuit includes a device-to-model CSV audit;
qualification also records the simulator binary hash. One local sample is a
random sample; deterministic calibration explicitly requests `nominal`.

The Xyce 7.4 `.STEP` + `.SAMPLING` compatibility check ran only the first
geometry (two random samples instead of the requested two-by-two grid).
Evidence: `outputs/qualification/V2.0.4-local/sampling_step_check.*`.
The local-MC API now fails explicitly for combined legacy sweep flags; run
one deck per geometric candidate with frozen sizing. Width expressions still
survive specialization, and deterministic sweep paths remain available.

The full revised schedule is **112 physical configurations, 336 nominal
calibration decks, 1,682 local verification decks, 17,156 waveform samples**.
All acceptance PVT read/write and sequence decks use ten local samples.
both read and write cover all five global corners, including the previously
missing SF read. Fast-corner cases also exercise next-row hazards on tall arrays.
dedicated read ensembles use seeds 3026-3035 and write ensembles 4026-4035.
The diagnostic `local_review` harness uses three samples with a separate seed
82026 to expose failures early; it cannot publish sizing-table records. It
covers 8x4, 16x16, 64x64, 256x8 and 16x256, both cell types and mux choices,
plus 16x16 RC sensitivity. Reports distinguish failed path checks from proven
component causes. The full ten-sample screening stage is also available as
`campaign --screen`; `campaign --dry-run` prints counts without simulations.

### Multicore execution checkpoint

The user requested multiple cores for large SRAM simulations. The installed
Xyce 7.4 binary reports `Parallel with MPI`; use its adjacent Conda `mpiexec`,
not the unrelated EDA launcher on the shell PATH. Arrays with at least 1,024
cells use four MPI ranks. Each rank uses one BLAS thread; the current 20-job
limit admits at most 80 cores against the 96 available CPU affinity slots.
The runner kills the full MPI process group on timeout and records rank count,
launcher hash, solver and sampler provenance. The diagnostic limit is currently
1,800 seconds per sample; an execution timeout is not a circuit failure or a
passing qualification result.

Native `.SAMPLING`/`AGAUSS` under MPI crashed at both four and eight ranks on
the 64x64 full-local deck (`double free or corruption` during initialization).
V2.0.5 therefore materializes independent Gaussian Latin-hypercube draws into
numeric model cards before MPI transients. Mean models, 5% relative sigma,
geometry, timing, solver and acceptance thresholds are unchanged. Parameter
streams are keyed by ensemble seed, device model identity and parameter name,
so inserting unrelated devices or changing widths does not reassign existing
devices' normalized draws. The sampler and every numeric model file are hashed;
these draws are distinct from the historical Xyce-generated seed ensemble.
Small serial runs retain native Xyce LHS; the backend is explicit in results.

Validation: materialized four-core 8x4 read and SF write-box runs each pass two
samples. Re-running the identical first read sample on one core gives identical
printed `TREAD_TOTAL`, `TCLK_WLEN`, `TRESTORE`, `TWLDRV` and `PAVG`, and both
waveforms pass. Evidence: `outputs/qualification/V2.0.5/mpi-integration/`.
A 64x64 benchmark with 28,729 independently perturbed MOS devices (86,187
parameters) completes on four cores in 854.3 seconds and passes the waveform
checks; its matched serial run exceeds the 900-second benchmark limit, so no
precise speedup ratio is claimed. It is a solver/mismatch execution check at
an explicit 3 ns clock, not frozen-period rule qualification. Evidence is under
the pre-label `outputs/qualification/V2.0.4-local/mpi-benchmark/` directory.

Resume/inspect the scheduled diagnostic screen from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m dev.sizing.local_review --workers 20 --mpi-ranks 4 --samples 3 --timeout 1800 --xyce /path/to/openyield/bin/Xyce
python3 -m dev.sizing.local_review --summarize
```

`review/schedule.json` records the admitted matrix; each architecture has an
atomic checkpoint, and `review/review.json` summarizes observed rule checks.
The supplied CSV, the old stopped campaign and pre-label pilots are retained.

### Stage C screen outcome — 2026-09-08

The diagnostic screen (`local_review`, 20 workers, four MPI ranks above 1,024
cells, 1,800 s per sample, seed 82026) finished at 18:35. `review/review.json`
holds the ledger; the observed rule families are:

| Rule family | Checks observed | Failed | Where |
|---|---:|---:|---|
| Write output / input and hold | 1,500 | 0 | — |
| Precharge | 2,133 | 0 | — |
| Wordline inverter / NAND | 1,122 | 36 | 16x16 with explicit RC (`parasitic_factor` 2), reads at SS, SF and FS 125 C, 6T and 10T, mux on and off, baseline and read-box cells |
| Decoder and address | 2,127 | 0 | — |
| Control buffers | 2,286 | 0 | — |
| Replica and sensing | 1,233 | 0 | — |

- **Wordline budget with RC.** The failing samples measure `TWLDRV` 130-141 ps
  plus a 103-109 ps wordline slew against 0.15 of the 1.53 ns RC read phase at
  the calibrated 3.85 ns clock; the same arrays without RC sit at 0.135 of a
  1.00 ns phase and pass. Every other check of those reads passes (sensing
  differential, output settling, restore, retention, hazards), so the read is
  functional and the finding is a sizing-rule gap: the wordline inverter/NAND
  rule has no parasitic term (`parasitic_factor` only multiplies the precharge
  and write loads). Stage D must add the RC wordline load to the rule or
  restate the wordline fraction for RC decks before any coefficient changes.
  RC audit after the screen: with `w_rc` every wordline pin (real, replica
  and dummy cells, both cell types, equivalent-cell aggregates) carries one
  100 ohm / 1 fF stub and every wordline driver output two segments; the
  fixed-mode replica driver (AND2) lacked them and now matches, and the
  replica bitline now enters TIME through the same two segments the real
  bitlines see at the sense-amplifier input. At 8x4/TT the fixed-mode RC
  replica wordline led the real one by 36 ps before the change and 13 ps
  after (the legacy AND2 stays unscaled and lightly loaded); the matched
  driver of rules_only mode tracks within 1 ps. The RC screen above predates
  that change; rerun the 16x16 RC cases before Stage D uses them.
- **Wordline wire model: a star, not a distributed line.** The compiler
  inserts RC per subcircuit pin: every cell's WL pin gets one 100 ohm / 1 fF
  stub and the driver output two segments, while the row net between the
  cells is ideal. N stubs in parallel are 100/N ohm and N fF, so the wordline
  is a lumped capacitance, the driver width alone sets its delay, and every
  cell of the row sees the same edge. That is why the wordline rule is a
  fan-out rule (`cols/4`, `cols/15`) without a wire term and why the wordline
  check reads the row net. The model was inherited from the cell-level RC
  insertion (the bitlines use the same star); it was not chosen for the rule.
  `python3 -m dev.sizing.wordline_model` compares it with a
  distributed line built from the same driver and cells (TT, 25 C, fixed-mode
  driver scale; `docs/qualification/V2.0.5_wordline_model.json`):

  | Columns | Row model | First cell arrival / slew (ps) | Last cell arrival / slew (ps) |
  |---|---|---:|---:|
  | 16 | star (compiler) | 46 / 44 | 46 / 44 |
  | 16 | distributed, 100 ohm + 1 fF per column | 46 / 81 | 73 / 100 |
  | 16 | distributed, 1 ohm + 0.1 fF per column | 47 / 47 | 48 / 47 |
  | 256 | star (compiler) | 92 / 159 | 92 / 159 |
  | 256 | distributed, 100 ohm + 1 fF per column | 42 / 48 | no 50 % crossing in the 2 ns pulse |
  | 256 | distributed, 1 ohm + 0.1 fF per column | 79 / 211 | 128 / 237 |

  At 16 columns a metal-pitch line (about 1 ohm and 0.1 fF per 0.6 um cell
  pitch in this node) is indistinguishable from the star. At 256 columns it
  delays the last cell by 36 ps, widens its slew by 78 ps and delays its fall
  by 37 ps while the first cell gets faster, because the line shields the
  driver. Chaining the compiler's generic 100 ohm / 1 fF stub per column is
  not a wire model: 25.6 kohm and 256 fF at 256 columns never reach 50 %. A
  distributed wordline therefore needs pitch-based segment values, the
  wordline check moved to the last cell, the replica wordline routed through
  the same line length, and a wire term in the rule that no driver width
  removes (Elmore 0.5 N^2 R C). Decision (user, 2026-09-08): keep the
  inherited star model for the wordline sizing rule; the distributed-line
  numbers stay as reference only and are not a Stage D item.
- **Execution, not circuit, losses.** 28 four-rank cases (6T 64x64, 256x8 and
  16x256 sequences, fast-corner reads and writes, write boxes) hit the 1,800 s
  diagnostic limit and the twelve 10T 64x64 / 16x256 nominal calibration decks
  did the same, while the host was shared with unrelated jobs (load average
  185 on 96 cores; the idle 64x64 benchmark needed 854 s). These 93 samples
  are unscored, not failed. The screen originally labelled those calibration
  timeouts as "failed waveform acceptance"; `local_review` now records the
  execution error itself.
- **DC operating point on materialized MPI samples.** Three materialized
  samples (6T 64x64 write at TT and FF/-40 C, 6T 16x256 read at FF/-40 C)
  stopped with `DC Operating Point Failed` after about 5,000 Newton iterations
  on four ranks; the sibling samples of the same cases converged and the
  serial native-sampling decks of the same arrays never showed this. The
  tighter-timestep retry does not cover a DCOP failure. Open item for Stage D:
  a DCOP continuation or serial fallback for the materialized MPI path.
- **Independent review of the code state (2026-09-08).** Twenty fixed-mode
  decks (8x4 to 256x8, 6T and 10T, mux on and off, with and without RC, plus
  the legacy shared-card MC deck) are byte-identical to the `d30a23b` tree, so
  the shipped default is unchanged. Default-mode Xyce runs at 8x4 (nominal,
  per-device and shared, read, write and read&write, both cells and mux
  choices) and the per-device CLI complete with valid measures. One
  integration regression was corrected: the optimizer objective in
  `size_optimization/exp_utils.py` inherited the new per-device default with a
  single unseeded sample, which made every evaluation random; it now requests
  the nominal corner explicitly. `main_sram.py` keeps `mc=True`, which under
  V2.0.5 means one unseeded local-mismatch sample when `monte_carlo_runs` is 1;
  pass `variation_mode='nominal'` or `mc_seed` there for a reproducible run.
  (Resolved 2026-09-09: `main_sram.py` now passes `variation_mode='per-device'`,
  `mc_seed=20260711` and `real_cell_mode=0` explicitly and reads the YAML in
  memory; see the V2.0.6 changelog entry.)

### Rule disposition under the new mismatch contract

Historical nominal/shared sweeps justify starting values only. None of the
following coefficients is yet requalified for full local mismatch.

| Rule family | Current starting rule | Required full-local evidence / decision |
|---|---|---|
| Write output | `max(1.5, rows/16)` | Recheck 6T and 10T strongest-box SS/SF write/retention separately. Diagnose floor failures on small arrays before increasing a row-load term; retain local tails on both stack devices and all cell devices |
| Write input / data hold | `max(0.5, wd_out/4)` | Independently vary the input stage with fixed output scale; measure data arrival and hold through enable release. Whole-path write success alone does not calibrate this floor |
| Precharge | `max(0.5, rows/32)` | Check every bitline's restore/equalization at the frozen clock with local precharge, cell leakage and control mismatch. Evaluate the proposed `rows/64` reduction only after the existing rule is screened |
| WL inverter / NAND | `max(1, cols/4)` / `max(1, cols/15)` | Check both local real-WL and replica-WL drivers, slew and the 0.15 phase fraction; matching drawn sizes does not imply zero random skew |
| Decoder / address | Current final-inverter and address-buffer fan-out rules | Check address edges, selected/unselected WLs and next-row hazards at fast and slow corners; retain frozen actual loads |
| PRE/enable/isolation/clock buffers | Current effort targets and finger limit | Check both rise and fall edges, PRE/WL ordering, data-hold ordering and sensing glitches. Attribute failures with internal waveform probes before changing an entire buffer family |
| Replica K/N and sensing | `(1,9)` | Use full-local read-box ensembles and SA offset evidence for early sensing; also require output settling at the frozen period. Nominal ±10 ps systematic alignment is not a random-skew acceptance bound |

The muxed write topology currently writes every column. It does **not** exercise
half-selected cells. Section 7 item 10 therefore remains an explicit separate
topology/probe task before release qualification; it cannot be closed by more
samples of the existing write deck. Independent coefficient-validation seeds
and the proposed load-range extension remain subsequent work.

Initial full-local observations (8x4, no explicit RC, `(K,N)=(1,9)`, 30 samples
per physical configuration: ten SS reads and ten write-box samples at each of
SS and SF). Values below are the worst observed ratios at the frozen clock;
their limits are 0.8 for write/restore and 0.15 for wordline. These samples
support retaining the starting rules while the larger-array screen runs.

| Cell / mux | Period (ns) | Write / read phase | Restore / read phase | WL driver + slew / access phase | Minimum dV at sensing (V) |
|---|---:|---:|---:|---:|---:|
| 6T / off | 2.35 | 0.474 | 0.568 | 0.100 | 0.8942 |
| 6T / on | 2.35 | 0.491 | 0.644 | 0.103 | 0.8938 |
| 10T / off | 2.40 | 0.489 | 0.568 | 0.101 | 0.8918 |
| 10T / on | 2.40 | 0.509 | 0.629 | 0.101 | 0.8887 |

Write-output scale is 1.5 and precharge scale 0.5 in these pilots. The minimum
retained written Q is 0.8983 V at 0.9 V supply. The observed margin does not
justify reducing either floor before tail and size coverage. Shortening the
replica delay also shortens the reference read phase, so every driver budget
must be rechecked when K/N changes; margins measured with N=9 cannot be carried
unchanged into N=1. Numeric evidence and the original case link are in
`outputs/qualification/V2.0.5/pilot_rule_observations.json`.

Cell-box scope correction: the existing `write_box`/`read_box` cases vary
PU/PD/PG widths only. They retain nominal channel length and model selections;
the 10T feed-forward (`fd`) width also stays nominal. Thus the pilots do not
qualify the entire configured YAML domain (including FD, length bounds and
VTL/VTG/VTH choices). Stage D must explicitly sweep those independent inputs,
or restrict the qualified domain to the measured baseline family. Do not label
this three-width stress point the strongest cell over every configured choice.

## Implementation status — historical checkpoint 2026-09-07

The checklist below preserves the V2.0.4 checkpoint. Sections 0-10 retain the full
proposal; their proposed behavior is not a claim that every item is implemented.
The original document is also preserved at
`docs/design/DRIVER_SIZING_PROPOSAL_V2.0.4.md`. The 163-row
`DRIVER_SIZING_data.csv` is unchanged.

### Completed

- [x] V2.0.4 physical-path follow-up: matched real/replica wordline drivers,
  dummy gate loading that accounts for K active replica cells, frozen replica
  device parameters, odd configurable delay-chain stages, optional decoder final
  output-inverter scaling, and effort-based tapered buffers.
- [x] Canonical read loading: disabled write-driver stacks on real bitlines and
  matching disabled stack loading on RBL/RBLB. The replica must track the newly
  included real-bitline drain load as well as the wordline gate load.
- [x] Measured timing support (`sizing/timing.py`): derive the frozen period from
  SS read plus SS/SF write phases, apply 25% margin, and round upward to 50 ps.
  Historical fit coefficients only choose a long calibration period.
- [x] Reproducible qualification runner and campaign (`qualification.py`,
  `campaign.py`), including per-sample waveform failures, all-bitline restoration,
  retention, next-row hazards, local MC, and back-to-back read/write scoring.
- [x] Exact-match table reader (`table.py`), `auto` rule fallback, and
  `sizing_rules.json` provenance. The table remains empty pending qualification.
- [x] Optimizer adapter baseline reuse, resolved-width area estimates, and
  reporting of the original access-limit violations. Per-device runner metadata
  now records its frozen driver result; both simulation entry paths select Xyce
  explicitly instead of relying on an installed ngspice shared library.

- [x] Scanned the repository and wrote root `AGENTS.md` with architecture,
  configuration ownership, circuit/sweep conventions, and verification commands.
- [x] Added `sram_compiler/sizing/driver_sizing.py`: immutable `DriverSizes` and
  `DriverLoads`, `resolve_driver_sizes()`, and JSON-ready `to_dict()` metadata.
- [x] Implemented the proposed write-output floor/load rule, separate write-input
  rule, precharge rule, and balanced wordline inverter/NAND rule (section 4.1).
  Settings expose `parasitic_factor`, `wd_floor_margin`, `k_w`, and `pre_min`.
- [x] Added deterministic baseline fingerprints covering cell values/bounds,
  peripheral parameters, all five PDK model-file contents, geometry, cell type,
  mux, sizing settings, and rule version. These identify inputs, not qualification.
- [x] Added explicit `driver_sizes=` reuse to both normal and MC testbenches.
  Cell candidates and run PVT can change; geometry, cell type, mux, peripheral
  configuration, or PDK changes reject reuse. Construction resolves once when
  no result is supplied; netlist generation checks for stale peripheral inputs.
- [x] Split `WriteDriver` widths: M1-M4 use the input scale; M5-M12 use the output
  scale. Existing direct calls retain equal input/output widths by default.
- [x] Added optional scale arguments to precharge, write, and wordline factories.
  Normal testbench generation passes resolved scales to all three factories.
- [x] Replaced testbench load-rule duplication with `DriverSizes.loads` for PRE,
  write enable, wordline enable, sense-amp count, and the data-hold inverter scale.
  New rules count the fixed replica NAND input and configured WL NAND widths.
- [x] Preserved array-dependent scales in precharge/write SPICE sweep expressions;
  wordline sweep expressions now consume the resolved scales as well.
- [x] Added `global.yaml`/loader sizing settings and usage/status documentation in
  `sram_compiler/sizing/README.md`. Invalid options and invalid mux geometry fail
  explicitly.

### Current behavior and deliberate limits

- `sizing.mode: fixed` remains the shipped default. For this increment it means
  the legacy array-dependent numeric scales, optionally overridden by
  `fixed_scales: {pre, wd_in, wd_out, wl_inv, wl_nand}`. It retains the legacy
  replica-load omission for comparisons and the known small-array write weakness.
- `rules_only` opts into the new formulas and returns `source='rule'`; it is
  **unverified**. `auto` uses an exact baseline/physical-context table match and
  otherwise also returns unverified rules. Incomplete or stale records cannot
  promote a rule prediction to `source='table'`.
- Rule coefficients use the proposal defaults (write floor 1.0 times margin 1.5,
  `k_w=1/16`, `parasitic_factor=1`, `pre_min=0.5`). The 6T floor is provisionally
  shared with 10T. They are not recalibrated for changed PDKs or YAMLs yet.
- Precharge/write sweep scaling is corrected even in fixed mode. TIME loads stay
  at the resolved baseline during peripheral sweeps; complete sweep-range
  electrical qualification remains open.
- Fixed mode retains the original replica AND2 and taper rules. Rules/auto mode
  uses the matched replica path; `(K, N)=(1, 9)` remains the default. A supplied
  measured `TimingConfig` or qualified-table timing sets the clock; otherwise it
  stays at the existing 10 ns. YAML 200/100 ps access violations are reported
  separately and are never silently marked feasible.

### Full qualification — stopped before completion

The first broad runs exposed control-edge and RC handoff failures. Focused
pilots pass with the corrections below. The stopped campaign used internal rule
revision `v2.0.4-qualification-10`. Earlier passing runs are
historical evidence, not qualification for changed circuits.

The wide-buffer investigation identified the underlying limit: the PDK enables
BSIM4 gate resistance (`RGATEMOD=1`), while very large output devices were still
single-finger gates. Increasing those widths increased gate RC. New-mode timing
buffers, clock drivers and wordline-enable drivers now partition wide inverter
devices into fingers of at most 2 um using `NF`, preserving their total `W`.
The PDK resistance model remains enabled. A reference with unchanged total
92.34 um PMOS / 61.56 um NMOS widths reduced its edges from approximately
44/42 ps to 14/14 ps with multiple fingers; DC current stayed approximately
constant. Evidence: `docs/qualification/V2.0.4_gate_finger_check.json` and
`V2.0.4_gate_finger_dc.json`; `V2.0.4_gate_finger_reference.json` records the
conditions and model/simulator hashes. Reproduce both checks with
`python3 -m dev.sizing.gate_fingers --xyce /path/to/Xyce`.
The folded-driver 16x256 and 16x512 read pilots
pass every check (16x256 isolation edges about 20/16 ps). The release remains
V2.0.4.

Progress snapshot (2026-09-07, 10.3 h into the 80-worker run, `campaign.json`):
656 of 1,786 decks complete, 90 of 112 physical configurations calibrated
(measured periods 2.25 ns at 1x1 to 5.1 ns at 512x4), 384 verification decks
passed, 1 failed (128x32 mux SF write-box, the time-step retry above), 1
calibration deck with a Xyce exit (10T 100x50 read SS, above). Phase ratios
measured at the frozen periods (6T, mux off, SS / 125 C / 0.9 V, nominal):

| array | `wd_out` / `pre` today | write phase / read phase (SS, SF) | restore / read phase | wordline (`TWLDRV` + slew) / access | dV at `s_en` `(1, 9)` |
|---|---|---|---|---|---|
| 8x4 | 1.5 / 0.5 | 0.43, 0.39 | 0.56 | 0.10 | 0.89 V |
| 16x16 | 1.5 / 0.5 | 0.48, 0.44 | 0.57 | 0.14 | 0.89 V |
| 64x16 | 4 / 2 | 0.44, 0.40 | 0.48 | 0.12 | 0.89 V |
| 64x64 | 4 / 2 | 0.47, 0.42 | 0.49 | 0.11 | 0.89 V |
| 256x8 | 16 / 8 | 0.33, 0.30 | 0.36 | 0.07 | 0.78 V |
| 512x4 | 32 / 16 | 0.25, 0.23 | 0.29 | 0.04 | 0.67 V |
| 16x256 | 1.5 / 0.5 | 0.56, 0.51 | 0.60 | 0.13 | 0.90 V |

The budgets are 0.8 (write, restore) and 0.15 (wordline). Every ratio
*falls* as the rows grow: the row-proportional load terms give the large
arrays 2-4x more driver than the budget needs, and those drivers sit on the
bitline that sets the period (section 11, item 1).

- [x] Corrected address-buffer sizing: the old formula divided by three although
  `TaperedBuffer` had a one-unit output per scale. Use six inverter-input units
  per decoder group and size the actual output for its fan-out.
- [x] Corrected PRE and isolation falling-edge failures against the unchanged
  40 ps target. PRE uses effort four and a two-unit floor; buffered isolation
  uses effort three and twice the final NMOS strength. Gate fingering closes
  the remaining wide-array failures. Small, tall and wide TT pilots pass.
- [x] Removed an unused dummy-cell QB RC branch that formed a floating DC island.
  The RC decks now converge without changing model or solver tolerances.
- [x] Accounted for explicit peripheral enable-pin capacitors in control loads.
  A measured reference inverter has 0.554-0.564 fF input charge capacitance across
  the sampled corners; the rule uses 0.5 fF for load margin. The reproducible
  experiment is `dev/sizing/gate_cap.py` and its data is
  `docs/qualification/V2.0.4_unit_gate_cap.json`.
- [x] Validate the RC precharge handoff guard in the 16x16 pilot. RC waveforms showed precharge
  beginning 49-86 ps before the physical WL reached its low threshold. The new
  guard observes the matched RWL, and the sense/wordline-enable buffers have
  stronger effort targets. Full RC sensitivity coverage remains in progress.
- [x] Added per-case process locks and atomic campaign checkpoints so concurrent
  resumes cannot corrupt generated files or progress records.
- [x] Detect incomplete individual transients even when the Xyce sampling
  command returns success. One 128x32 muxed SF write-box sample stopped at
  6.556 ns with `Time step too small`; all its scalar measures were present.
  The same seeded ensemble is being retried at a tighter 5 ps maximum step.
  Both attempts are retained. Electrical check failures and nonfinite signals
  are ineligible for this automatic numerical retry.
- [x] Preserve nonzero Xyce exit codes with their input signatures and allow
  the same tighter-step retry for a single nominal transient underflow. The
  10T 100x50 read calibration exposed this exit path. Its recovery and subsequent
  verification are running against the canonical case cache.

- [ ] Complete the 27-size historical matrix, both cell types and valid mux
  choices, plus RC sensitivity and 32x1 frozen-optimizer cases (112 physical
  configurations). The campaign schedules 336 calibration decks and 1,450
  verification decks: 6,538 waveform samples in total, including representative
  eight-cycle back-to-back sequences.
- [x] Characterize 100 independent SA/latch copies at SS/125 C/0.9 V and
  FF/-40 C/1.0 V, retaining each copy's mismatch over its differential sweep.
  Each corner includes two 100-copy ensembles: 0.75 VDD common mode and a
  precharged-bitline profile with one input at VDD. Conservative offset bounds
  (absolute mean + 3 sigma + half the 25 mV sweep step) reach 170.1 mV at SS and
  169.8 mV at FF. Combine with the still-running replica/cell mismatch campaign
  before setting `dV_min` or promoting table rows.
- [ ] Finish per-device replica/cell mismatch, close waveform/phase-budget
  failures, export the V2.0.4 report and qualified table, and commit/push to main.

Qualification artifacts are under `outputs/qualification/V2.0.4/`. Restoration
is sampled at the end of clock-high, before intentional write predrive and PRE
release charge injection. This implements the restore-phase requirement without
mistaking access activity for a precharge failure. Individual checks and their
numeric evidence are retained in every case's `result.json`.
The execution allowance is six hours per sample, so a ten-sample deck can run
for up to 60 hours. This addresses measured large-array runtimes; waveform
thresholds, timestep limits, models, and required sample counts are unchanged.

### Validation completed / in progress

- [x] 35 simulator-free regression tests pass:
  `python3 -m unittest discover -s tests -v`.
  Coverage includes floors/load crossovers, actual gate loads, modes/overrides,
  invalid inputs, immutable reuse and fingerprints, split MOS widths, numeric
  and swept factory paths, TIME integration, and full-array read/write generation
  for 6T/10T with mux on/off.
- [x] Existing offline optimizer suite: 6 tests pass.
- [x] Six fixed-mode full-array decks are byte-for-byte identical to the original
  testbench's numeric output: 8x4, 16x16, 64x64, each read and write. This caught
  and corrected unit-conversion roundoff that could over-size a WL buffer at a
  `ceil()` boundary.
- [x] Nominal Xyce smoke decks at 8x4, 125 C / 0.9 V, no RC, full transistor
  array, 10 ns clock: the strongest box cell fails to write at SF with legacy
  sizing (`TWRITE_TOTAL=FAILED`), but writes with the new frozen periphery
  (`TWRITE_TOTAL` about 167 ps). Default-cell SS write/read decks also complete;
  write access is about 141 ps and the read output settles low. These are smoke
  results, not the full acceptance checks of section 2.2.
- [x] Ten seeded shared-model MC samples of the strongest 8x4 box cell at SF /
  125 C / 0.9 V with the new frozen periphery all pass the focused write/retention
  checks (seed 2026, std 0.05): write access 130-193 ps, retained Q at least
  0.89984 V during 14.5-17 ns while WL and write enable are low. The driven
  bitline reaches below 0.1 VDD in every sample. This does not qualify local
  mismatch or the automatic-period budgets.
  Decks, waveform/measure outputs, baseline metadata and `smoke_summary.json`
  are under ignored `outputs/driver_sizing/`; `check_waveforms.py` there records
  the focused scorer. These new results are separate from the original CSV.
- [x] Python compilation, Python 3.9 syntax parsing of the new code/tests, and
  `git diff --check` pass.

### Still open — next implementation order

1. [x] Complete the physical path: optional decoder last-stage inverter scaling;
   matched replica wordline and dummy-row loading; K active replica cells;
   configurable odd N delay stages; effort-based control-buffer stage count.
   Keep `(1, 9)` until local mismatch is qualified. Include disabled write-driver
   loading in read decks to represent the canonical macro.
2. [ ] Finish integration before optimizer/yield candidate loops in
   `size_optimization/exp_utils.py` and `main_estimation.py`; serialize the result
   with experiments; use resolved widths in area estimation. The optimizer adapter
   and per-device MC path are updated. The legacy `main_estimation.py` still imports
   an absent `spice` backend and uses obsolete algorithm/testbench APIs; porting
   those rare-event algorithms is separate from the current compiler campaign.
3. [ ] Implement the automatic timing dependency in `docs/TIMING_AUTOCONFIG.md`, then
   add reproducible `verify()` waveform/phase-budget checks, coefficient provenance
   in `sizing_rules.json`, hash-keyed `sizing_table.json`, and `auto` lookup with
   explicitly unverified rule fallback. The measured-period path, runner, rules,
   and table reader now exist; full verification and table promotion remain open.
4. [ ] Execute section 7's full size/cell/mux/corner campaign: 10 samples per
   verification deck, automatic periods, next-row hazards, and RC/parasitic
   sensitivity. Add the missing 10T parameter-box qualification.
5. [ ] Qualify SA offset and replica/cell local mismatch using per-device MC,
   set `dV_min`, and evaluate faster K/N choices. Shared-model MC cannot close
   this item. Complete the frozen-periphery optimizer smoke and failure checks.
6. [ ] Function-first revision (section 11, after the current campaign has
   finished so its evidence stays attributable to one rule version): move the
   write-driver and precharge load terms to the characterised-range clamp
   `rows / 64` (section 4.1), extend the sweeps to `rows / scale` = 64-256
   (section 7 item 7), add the fast-corner race, per-device write-box,
   wordline-phase upper-bound and half-select decks (section 7 items 8-11),
   and make the wordline speed check relative to the phase (section 2.2).
   `k_w` stays available as an explicit speed override only.

## 0. Recommendation in short

1. **Function first, then the array's own timing, never the absolute YAML
   limits.** Every driver has a functional requirement that holds for any
   clock (the write driver flips the strongest cell of the parameter box, the
   precharge restores and equalises, the wordline reaches VDD and falls before
   the precharge, the amplifier fires only with `dV_min` of differential, the
   cell keeps its data) and a timing requirement that is a *fraction of the
   array's bitline-limited read phase*. That phase grows with the rows
   (934 ps at 8 rows, 2029 ps at 512 rows at SS / 125 C / 0.9 V), so the
   timing requirement on the drivers relaxes as the load grows; the 200 /
   100 ps `global.yaml` access limits are absolute, are met by no
   configuration above 64 rows whatever the drivers are, and are therefore
   reported but not used for sizing (section 2.1).
2. **Each driver is `scale = max(floor, load term)`, with the floor set by
   function and the load term by the array-relative budget — not by a fixed
   absolute speed.** The one functional failure found so far — an 8-row array
   that cannot be written at SF / 125 C / 0.9 V — is a *floor* problem: the
   strength a write driver needs to flip a cell does not depend on the row
   count, so a rule that only scales with rows gives small arrays a driver
   weaker than the cell. The opposite error is in the load terms: `rows/16`
   for the write driver was chosen for a 100 ps TT write access (the YAML
   limit), and at 512 rows it gives a 32x driver whose drains add a third of
   the cell load to the bitline that sets the period, while the write phase
   uses 25 % of the read phase against a budget of 80 % (status table). The
   revised load terms are the smallest widths inside the characterised range
   that meet the budget, `rows / 64` for both the write driver and the
   precharge (section 4.1), and the sweeps of section 7 item 7 decide whether
   the range can be extended further. Floors and speed terms are measured at
   the driver's own worst corner: write-ability at SF / 125 C / 0.9 V, speed
   at SS / 125 C / 0.9 V, leakage and hold at FF and FS / 125 C, races and
   ordering at the fast corner as well, sensing margin against local mismatch.
3. **One resolver, frozen per array.** `resolve_driver_sizes(cell, rows,
   cols, mux)` computes all scales from the YAML base widths, then the TIME
   buffer loads *from those scales* (today the testbench recomputes the
   loads from the rules by hand), and the factories consume the immutable
   result. Like the clock period of `docs/TIMING_AUTOCONFIG.md`, the sizes are
   derived once from the baseline cell and frozen as the spec for size
   optimisation and yield analysis; a candidate cell that the frozen
   periphery cannot write or read fails, the periphery is not re-sized around it.
4. **Replica-timing drivers become a copy of the real path:** the replica
   wordline is driven by the same NAND2 + inverter wordline driver as every
   row, loaded by a dummy row of `cols` cells, instead of a fixed AND2 that
   drives one cell (today the replica wordline rises 46-101 ps before the real one at
   256 columns, section 3.3). The number of active replica cells
   `K` and delay-chain stages `N` stay `(1, 9)` by default; the table of
   section 4.3 gives the sensing margin for the alternatives.
5. **Verification is three primary decks per array** (read at SS, write at
   SS and SF, all 125 C / 0.9 V, with the automatic period) whose waveform
   checks and per-driver acceptance metrics (section 2.2) must pass with 10
   local-mismatch samples at each fixed global corner, plus the hazard, sequence and tail
   decks of section 7 (12-32 decks per array in the campaign); accepted
   vectors go into a table keyed by a hash of the cell and periphery YAMLs,
   so the optimiser never simulates for sizing.

## 1. What is sized today

### 1.1 The drivers and their rules

All widths in um, channel length 0.05 um. "unit" = one 0.09 / 0.27 um
inverter input (0.36 um of gate). R = rows, C = columns of the subarray.

| driver | rule (scale on the YAML base width) | base widths | drives | is limited by / worst corner | code |
|---|---|---|---|---|---|
| precharge PMOS x3 per column (+ replica column) | `max(0.5, R/16)` | 0.27 | one bitline each (R pass-gate drains); the equaliser bridges BL/BLB | restore time in the clock-high phase, SS / 125 C / 0.9 V | `PrechargeFactory.width_scale`, `Precharge` |
| write driver, 12 transistors per column, one width pair | `max(8, R)/16` | Wn 0.18 / Wp 0.36 | two series stacks per bitline (2 PMOS up, 2 NMOS down) against the cell; the DIN / EN input inverters have the same width | DC write-ability against the cell PMOS through the pass gate, SF / 125 C / 0.9 V; speed at SS | `WriteDriverFactory.width_scale`, `WriteDriver` |
| wordline driver (NAND2 + inverter), one per row | inverter `max(C, 4)/4`, NAND2 `sqrt(inverter scale)` | inv 0.09 / 0.27, NAND 0.18 / 0.27 | C cells x 2 pass-gate gates; the NAND2 is driven by the decoder's fixed 0.09 / 0.27 output inverter and by `wl_en` | wordline rise (read / write start) and fall (address hold), SS / 125 C / 0.9 V | `WordlineDriverFactory.inv_scale / nand_scale` |
| replica wordline driver | fixed AND2 (0.18 / 0.27 NAND2 + 0.09 / 0.27 inverter) | - | one replica cell (the other R replica cells are tied to VSS) | must track the real wordline | `create_and2_for_rwl` |
| replica bitline -> `rbl_delay` | 9 inverter stages, 4 unit loads each, fixed size | 0.09 / 0.27 | AND3 of `s_en` | ~250 ps at TT, ~2.3x at the worst case; sets the sensing margin | `DelayChain` |
| `wl_en` buffer | `wl_pdrive`, scale `ceil(R * nand_scale / 32)` | 0.45 / 1.35 output | R NAND2 inputs + replica AND2 | fan-out <= 8 | `TIME` |
| `PRE`, `w_en`, `s_en`, `sa_iso`, address buffers | `TaperedBuffer`, scale `ceil(load / 8)`, 2 stages up to 16, 4 above | 0.09 / 0.27 x scale | 3 precharge gates per column, write-driver EN gates, SA footers, SA pass gates, decoder inputs | fan-out <= 8 | `TIME` |
| clock buffer | `pdrive`, scale = DFF count / reference count | 4 stages to 2.43 / 7.29 | address, data, CS, WE flip-flops | - | `TIME` |

### 1.2 What the rules produce

Computed from the code for representative sizes (mux off). `h` is the
electrical effort of a stage (gate width driven / gate width of the driving
inverter input); `h_dec` is the effort of the decoder's fixed output
inverter into the wordline-driver NAND2.

| R x C | precharge W (x3/col) | PRE load [units] -> buffer scale | write driver Wn/Wp | w_en load -> buffer | WL inverter Wn/Wp | WL NAND2 Wn/Wp | h_inv | h_nand | h_dec | wl_en load -> scale |
|---|---|---|---|---|---|---|---|---|---|---|
| 8x4 | 0.14 | 5.6 -> 1 | 0.09 / 0.18 | 13 -> direct | 0.09 / 0.27 | 0.18 / 0.27 | 3.0 | 0.8 | 1.2 | 8 -> 1 |
| 16x16 | 0.27 | 38 -> 5 | 0.18 / 0.36 | 60 -> 8 | 0.36 / 1.08 | 0.36 / 0.54 | 3.0 | 1.6 | 2.5 | 32 -> 1 |
| 64x16 | 1.08 | 153 -> 20 | 0.72 / 1.44 | 180 -> 23 | 0.36 / 1.08 | 0.36 / 0.54 | 3.0 | 1.6 | 2.5 | 128 -> 4 |
| 64x64 | 1.08 | 585 -> 74 | 0.72 / 1.44 | 720 -> 90 | 1.44 / 4.32 | 0.72 / 1.08 | 3.0 | 3.2 | 5.0 | 256 -> 8 |
| 256x8 | 4.32 | 324 -> 41 | 2.88 / 5.76 | 332 -> 42 | 0.18 / 0.54 | 0.25 / 0.38 | 3.0 | 1.1 | 1.8 | 362 -> 12 |
| 512x4 | 8.64 | 360 -> 46 | 5.76 / 11.52 | 328 -> 41 | 0.09 / 0.27 | 0.18 / 0.27 | 3.0 | 0.8 | 1.2 | 512 -> 16 |
| 16x256 | 0.27 | 578 -> 73 | 0.18 / 0.36 | 960 -> 120 | 5.76 / 17.28 | 1.44 / 2.16 | 3.0 | 6.4 | 10.0 | 128 -> 4 |
| 16x512 | 0.27 | 1154 -> 145 | 0.18 / 0.36 | 1920 -> 240 | 11.52 / 34.56 | 2.04 / 3.05 | 3.0 | 9.1 | 14.1 | 181 -> 6 |
| 128x128 | 2.16 | 2322 -> 291 | 1.44 / 2.88 | 2720 -> 340 | 2.88 / 8.64 | 1.02 / 1.53 | 3.0 | 4.5 | 7.1 | 724 -> 23 |

Three things stand out:

- **The write driver has no floor.** At 8 rows it is 0.09 / 0.18 um, and
  its pull-down is two of those NMOS in series (EN-gated M7 / M11 above the
  data-gated M8 / M12), i.e. ~0.045 um of effective NMOS against the cell's
  0.09 um PMOS pull-up through a 0.135 um pass gate. At SF (weak NMOS,
  strong PMOS), 125 C and 0.9 V that is not enough (section 3.1).
- **The wordline path is unbalanced at wide arrays.** The inverter keeps
  `h = 3` by construction, but the NAND2 and the decoder output stage see
  `h = 0.8 sqrt(C/4)` and `1.25 sqrt(C/4)`: 9 and 14 at 512 columns. The
  decoder's 0.09 / 0.27 output inverter (fixed in `decoder.yaml`) is then
  the slowest stage of the row path, and the wordline inverter itself is
  larger than a balanced taper would make it.
- **The precharge and write-driver widths feed straight into the largest
  buffers of the timing block** (`PRE` 291x, `w_en` 340x at 128x128):
  the precharge PMOS is scaled by rows *and* instantiated per column, so its
  gate load grows with `R * C`. Where the restore has slack the width can
  come down and the buffer with it.

### 1.3 What the netlist actually loads the drivers with

The transistor instances carry no `AS / AD / PS / PD`, so the bitline sees
per row only the gate-edge sidewall junction (`cjswgd` 0.5 fF/um x 0.135 um
= 0.07 fF) and the overlap / fringe capacitance (`cgdo`, `cgdl`: 0.02-0.05
fF) of one pass-gate drain: **0.06-0.12 fF per row**, against 0.2-0.3 fF
per cell (junction area + wire) in a laid-out 45 nm array. The wordline
sees two pass-gate gates per column (~0.45 fF). Wire RC exists only as the
optional `w_rc` lumped branch per cell (100 Ohm / 1 fF), not as a
distributed line, so there is no far-end wordline or bitline delay.

Consequences for this proposal: the *method* (floors, load terms, worst
corners, margins, resolver) does not depend on the absolute loads, but the
*coefficients* of the load terms fitted in section 3 are optimistic by
2-3x for the bitline and unknown for the wordline wire. Section 5 therefore
carries a `parasitic_factor` on the load terms; a physical design has to
set it from extraction (or run with `w_rc` and calibrated `pi_res / pi_cap`)
and re-run the characterisation of section 7.

## 2. Requirements

### 2.1 What "timing requirement" means for the drivers

`docs/TIMING_AUTOCONFIG.md` sets the clock period per array from the two
clock phases measured at the worst case, `T = 2 * max(low, high) * (1 +
margin)`, with `low = TCLK_WLEN + access` (wordline phase) and `high =
max(TRESTORE, TCLK_DEC)` (precharge / decode phase). The read access is
bitline-limited (one cell discharging `R` rows of bitline, 0.95 ps per row
nominal, 2.3x at the worst case) — the physical floor that no periphery
driver changes. The requirement on the drivers is therefore:

The requirements are ordered; a lower item never overrides a higher one.

1. **Function first, at every corner and at the frozen period.** Independently
   of the clock, every driver has a DC / functional requirement: the precharge
   restores and equalises every bitline (also after a full write swing, also
   against the leakage of `rows - 1` off cells at FF / FS / 125 C), the write
   driver flips the strongest cell of the parameter box and the cell keeps
   the data, the wordline reaches VDD and falls before the precharge turns on
   and before the next decoder output, the amplifier fires only with
   `dV_min` of differential, no unselected wordline glitches, no cell is
   disturbed. These are checked at the slow corner (SS / SF, 125 C, 0.9 V)
   *and* at the fast corner (FF, -40 C and 125 C, high VDD): a race or an
   ordering violation is a fast-corner failure, a leakage droop is a hot
   fast-corner failure, and neither is visible at SS. A configuration that
   fails any of these is not sized, whatever its timing.
2. **No driver term may become the limiting phase, relative to this array.**
   `TRESTORE` (precharge), `TCLK_WLEN + TWLDRV` (control + wordline path),
   the write access (write driver) and `TCLK_DEC` (decoder) each stay below
   a fraction `alpha` of the bitline-limited read phase *of the same array*
   at the worst case. The read phase is `TCLK_WLEN + access`, 0.95 ps per
   row nominal and 2.3x at the worst case (934 ps at 8 rows, 1533 ps at 256,
   2029 ps at 512, status table), so the budget a driver has grows with the
   rows: the timing requirement is relaxed by the array itself, and a
   driver that meets `alpha` at 8 rows with a given `rows / scale` meets it
   with a larger `rows / scale` at 512. The period of `docs/TIMING_AUTOCONFIG.md`
   is then set by the array, not by the periphery, and its 25 % margin
   covers the drivers as well. Within the budget, *smaller is better*: every
   unit of driver width on a bitline adds drain capacitance to the phase
   that sets the period (section 3.1 item 4, section 3.2 item 2), gate load
   to the `PRE` / `w_en` buffers, and power.
3. **An upper bound on the relaxed timing.** Relaxing the period is not free
   at the fast corners: the wordline phase (`T/2`, 1.2-2.6 ns at the frozen
   periods) is the time the unselected bitline of the target column leaks
   towards the cell level and the time every half-selected cell of the row
   is exposed to read disturb. V2.0.3 measured the leak at 0.69-0.73 V after
   a 5 ns phase at FF / FS / 125 C (64x16, 256x8) and > 0.97 V at 2.6 ns.
   The requirement is therefore two-sided: the frozen period must also pass
   the read (`sense_differential`, restoration) and retention checks at
   FF / 125 C and FS / 125 C, which the campaign runs at the frozen period
   (section 7 item 3); the largest arrays (512x4, 5.1 ns) are the ones to
   watch (section 11 item 4).
4. **The `global.yaml` limits (`delay.upper` = 200 ps read, 100 ps write)
   are informational.** They are access limits measured from `wl_en`, absolute
   in picoseconds, and independent of the array. The baseline read is
   290-310 ps at TT for every size up to 32 rows because the replica timing
   waits for a full bitline swing plus nine delay stages (~250 ps); at 256
   rows the bitline alone takes 240 ps nominal; no driver width changes
   either. The read limit is a replica-configuration decision (section 4.3:
   `(K, N) = (2, 1)` gives 113-127 ps at 8-16 rows and ~265 ps at 256 rows at
   TT) and above 64 rows it is unreachable. The resolver therefore never
   sizes a driver to the YAML limits; the campaign reports them as a separate
   metric (`yaml_access_spec_pass`), and the optimiser's feasibility
   constraint should be the array-relative budget of item 2 (decision 1 of
   section 9), not the absolute limits, which would mark every array above
   64 rows infeasible regardless of its drivers.

### 2.2 Requirement per driver

| driver | functional requirement (any clock) | timing requirement (worst case) | sized at | checked at | acceptance metric (existing measure / waveform check) |
|---|---|---|---|---|---|
| precharge | BL, BLB and RBL back above 0.98 VDD and within 5 mV of each other before the next wordline rises; after a full write swing (BLB at 0) as well as after a read | `TRESTORE_wc <= alpha_pre * low_read_wc` (default `alpha_pre` = 0.8) so the restore is never the limiting phase | SS / 125 C / 0.9 V | SS (speed), FF / 125 C (leakage: bitline held at VDD, unselected bitline droop only during the wordline phase) | `TRESTORE`, `TPRCH`, `bl_at_acc0` / `blb_at_acc0`, `bl_recovered` |
| write driver | flips the strongest cell of the parameter box (max PU, min PG, max PD) at fixed SF and SS / 125 C / 0.9 V with independent local mismatch throughout the cell, driver, hold and control paths (section 7 item 8); the driven bitline reaches < 0.1 VDD; the cell keeps the new data after `w_en` / WL release | `TCLK_WLEN + TWRITE_TOTAL_wc <= alpha_w * low_read_wc` (default 0.8): a write never sets the period; the load term is the smallest characterised `rows / scale` that meets this (section 4.1), not an absolute access time | SF / 125 C / 0.9 V (floor), SS / 125 C / 0.9 V (speed) | SF and SS, 10 full-local samples each, plus dedicated 100-sample tail ensembles | `TWRITE_TOTAL`, `TWDRV`, `blb_driven_low`, `q_written`, `q_retained` |
| wordline driver | WL reaches > 0.9 VDD; WL falls before the precharge turns on (`wl_en_bar` gating) and before the next decoder output (address latch), at the slow *and* the fast corner | `TWLDRV_wc + slew_wc <= alpha_wl * (low_read_wc - TCLK_WLEN_wc)` (default 0.15, relative to the phase; with a fast replica configuration the *access* is 280-310 ps at 8-16 rows and the balanced wordline path, 67-83 ps of `TWLDRV` + slew, would fail a fraction of the access while being functionally right); the decoder stage effort bounded (`h_dec <= 8`) | SS / 125 C / 0.9 V | SS and FF / -40 C; `next_row` hazard deck at 256 / 512 rows at both corners | `TWLDRV`, `wl_slew`, `wl_off_before_precharge`, `unselected_wordlines_quiet`, `neighbor_retained` |
| replica wordline | rises within +-10 ps of the real wordline at every size and corner (same driver, same load) | - | - | SS, FF -40 C | `rwl_minus_wl` |
| replica bitline + delay chain (`K`, `N`) | amplifier fired only when the target bitline differential exceeds `dV_min` (0.3 V placeholder: SA offset 3 sigma + replica / cell mismatch 3 sigma + 100 mV) for the slowest cell / fastest replica; and early enough that `OUT` settles before the clock edge at the configured period | `sen_minus_wl + TSA + latch <= low_read_wc - TCLK_WLEN` | SS / 125 C / 0.9 V (late side), FF / -40 C + local mismatch (early side) | both, per-device MC for the mismatch | `dv_at_sen`, `sen_minus_wl`, `TSA`, `out_end_acc` |
| control buffers (`wl_en`, `PRE`, `w_en`, `s_en`, `sa_iso`, address) | edge 10-90 % <= 40 ps at TT regardless of size (V2.0.2 rule), no coupling glitch on `s_en` | included in `TCLK_WLEN`, `TRESTORE` | fan-out rule, checked at SS | SS | `TCLK_WLEN`, `TS_EN`, `sen_no_glitch_before` |
| whole array at the frozen period (upper bound of section 2.1 item 3) | unselected bitline of the target column stays above `VDD - dV_min` margin at `s_en`; every bitline restored at the end of the phase; target and half-selected cells retain | - | - | FF / 125 C, FS / 125 C, FF / -40 C at high VDD, at the frozen period | `sense_differential`, `all_bitlines_restored`, `cell_retained`, `read_no_disturb`, half-select deck (section 7 item 10) |

`alpha` values are proposal defaults (section 9); they are fractions of the
worst-case phases of the same array, so the drivers inherit the same PVT
derating as the period and the same growth with the row count. `alpha_w`
and `alpha_pre` are budgets, not targets: the resolver picks the smallest
width that stays inside them.

## 3. Measured basis

### 3.0 Harness

The V2.0.3 sweep harness (`run_one.py`) was extended with per-run overrides
of the sizing rules and of the replica hookup, applied by monkeypatching the
factories before the deck is built:

| switch | overrides |
|---|---|
| `--wd-scale s` | `WriteDriverFactory.width_scale` -> `s` (all 12 transistors) |
| `--wd-out-scale o` | the two tristate stacks at `o`, the DIN / EN inverters at `--wd-scale` |
| `--pre-scale s` | `PrechargeFactory.width_scale` -> `s` |
| `--wl-inv-scale s`, `--wl-nand-scale n` | `WordlineDriverFactory.inv_scale / nand_scale` |
| `--rwl matched` | replica wordline driven by the row `WordlineDriver` plus a dummy row of `cols` cells (bitlines tied to VDD) |
| `--rwl-k K` | `K` of the `rows + 1` replica cells on RWL |
| `--dc-stages N` | delay chain of `N` (odd) stages |
| `--cell-pu / --cell-pg / --cell-pd` | cell widths (parameter-box corners) |

The waveform scorer records in addition the wordline 10-90 % slew, the
replica-wordline / wordline skew, the bitline differential and replica
level at `s_en`, and for writes the driven-bitline level and the time from
the wordline to the cell flip. The TIME buffer loads follow the overridden
scales automatically (the testbench derives them from the factory rules).
All decks: 6T unless noted, mux off, nominal devices unless `mc10` (10
seeded global-variation samples, `vth_std` 0.05, seed 2026), 10 ns clock
unless noted, target cell = last row / last column.
`DRIVER_SIZING_data.csv` holds one row per deck (group, tag, waveform
result, the measures and the scorer quantities used in the tables below);
the raw decks and waveforms stay in the scratchpad `runs/` tree.

### 3.1 Write driver

Sweep of the write-driver scale at SF / 125 C / 0.9 V (the write-ability
corner). "rule" marks the value `max(8, rows)/16` gives today. Write access
= `TWRITE_TOTAL` (`wl_en` -> Q at 90 %); MC = 10 seeded global-variation
samples, mean / worst; the driven-bitline level is the worst sample.

| array, cell | scale (Wn / Wp um) | write access nominal [ps] | MC mean / worst [ps] | TWDRV [ps] | driven BLB min [V] | result |
|---|---|---|---|---|---|---|
| 8x4 default | 0.5 (0.09 / 0.18) **rule** | 499 | 364 / 420 (7 samples) | 119 | 0.269 | **3 of 10 samples do not write** |
| | 0.625 | 259 | - | 101 | 0.000 | pass |
| | 0.75 | 215 | 204 / 245 | 89 | 0.001 | 10 / 10 |
| | 0.875 | 192 | - | 80 | 0.000 | pass |
| | 1.0 (0.18 / 0.36) | 177 | 168 / 195 | 72 | 0.000 | 10 / 10 |
| | 1.25 | 160 | - | 62 | 0.000 | pass |
| | 1.5 (0.27 / 0.54) | 148 | 141 / 160 | 55 | 0.000 | 10 / 10 |
| | 2.0 (0.36 / 0.72) | 136 | - | 47 | 0.000 | pass |
| 8x4 box cell (PU 0.108, PG 0.108, PD 0.246) | 0.5 **rule** | - | - | - | 0.253 | **fails with nominal devices** |
| | 0.75 | - | - | - | 0.204 | **fails with nominal devices** |
| | 1.0 | 236 | 232 / 343 | 70 | 0.000 | 10 / 10 |
| | 1.5 | 176 | 168 / 201 | 53 | 0.000 | 10 / 10 |
| | 2.0 | 158 | 151 / 176 | 46 | 0.000 | 10 / 10 |
| | 3.0 | 162 | - | 34 | 0.000 | pass (slower than 2.0: self-loading) |
| 8x4 default, split: inputs 0.5, stacks o | o = 1.0 | 173 | - | 70 | 0.000 | pass |
| | o = 1.5 | 139 | 132 / 151 | 51 | 0.000 | 10 / 10 |
| | o = 2.0 | 124 | - | 40 | 0.000 | pass |
| 8x4 default, SS / 125 C / 0.9 V | 0.5 / 0.75 / 1.0 / 1.5 / 2.0 | 321 / 208 / 177 / 151 / 141 | - | 121 / 90 / 72 / 55 / 48 | 0.000 | pass |
| 8x4 10T | 0.5 **rule** | 344 | 327 / 411 | 120 | 0.001 | 10 / 10 |
| | 1.0 | 204 | - | 72 | 0.000 | pass |
| 16x16 default | 0.5 | 609 | - | 147 | 0.001 | pass (at the DC limit) |
| | 0.75 | 271 | 258 / 308 | 106 | 0.001 | 10 / 10 |
| | 1.0 **rule** | 226 | 214 / 247 | 86 | 0.000 | 10 / 10 |
| | 1.5 / 2.0 | 189 / 173 | - | 66 / 55 | 0.000 | pass |
| 16x16 box cell | 1.0 / 1.5 / 2.0 | 286 / 216 / 194 | - | 82 / 63 / 53 | 0.000 | pass |
| 64x16 default | 1.0 / 2.0 / 4.0 **rule** / 8.0 | 366 / 245 / 211 / 189 | - | 165 / 95 / 58 / 40 | 0.002 | pass; write PAVG 145 / 145 / 157 / 181 uW |
| 256x8 default | 4.0 / 8.0 / 16.0 **rule** | 313 / 247 / 210 | - | 138 / 80 / 52 | 0.002 | pass; write PAVG 207 / 219 / 244 uW |

What the sweep shows:

1. **The floor is set by the cell, not by the rows.** Below a certain scale
   the write access diverges and then the cell does not flip at all: the
   default cell fails 3 of 10 samples at 0.5x (bitline stuck at 0.27 V) and
   writes every sample at 0.75x; the parameter-box cell (strongest PMOS,
   weakest pass gate the optimiser may choose) fails at 0.75x with nominal
   devices and writes every sample at 1.0x (worst 343 ps). The 16-row array
   with the same 0.5x driver takes 609 ps, the 8-row array 499 ps: the row
   count changes the time, the cell decides whether it flips. **Floor
   `wd_floor` = 1.0** (box cell, 10 / 10); with the 1.5x strength margin of
   section 5 the resolved floor is **1.5** (worst box-cell sample 201 ps,
   worst default-cell sample 160 ps). The 10T cell writes at 0.5x
   (10 / 10, worst 411 ps); the 6T floor is kept for it as the
   conservative choice until it has its own box-cell sweep.
2. **Above the floor the load term is a budget choice, and the budget grows
   with the rows.** Fitted on 64 and 256 rows, `write access = 170-190 ps +
   2.1-3.2 ps * rows / scale` at SF / 125 C / 0.9 V; the pre-review
   `rows/16` (`rows/scale` = 16) gives 210-226 ps at every size from 16
   rows up, i.e. the ~100 ps TT write access the `global.yaml` limit asks
   for. Half of it (`rows/32`) costs +35-40 ps (245-247 ps), a quarter
   (`rows/64`, the largest measured `rows / scale`) 313-366 ps; all of them
   are far inside the write budget of section 2.2 (`0.8 * low_read_wc` =
   0.9-1.6 ns at these sizes: measured write phase / read phase = 0.44 at
   64 rows and 0.33 at 256 rows with the `rows/16` driver, status table).
   The read phase grows by ~2.1 ps per row at the worst case and the write
   access by 2.1-3.2 ps per unit of `rows / scale`, so the budget-derived
   scale is asymptotically *constant* (about 1.5-2x), not proportional to the
   rows. The fit is only measured up to `rows / scale` = 64; below that
   line the write is nowhere near bitline-limited, above it the transient
   condition (bitline discharged and cell flipped inside the wordline phase)
   is unmeasured. Section 4.1 therefore clamps the load term at
   `rows / scale` = 64 until the sweep of section 7 item 7 extends the
   range. Beyond 2x at 8 rows the driver gets slower again (3.0x: 162 ps vs
   158 ps): the DIN / EN inverters and the `w_en` buffer load grow with the
   same width.
3. **Only the output stacks need the floor.** Keeping the DIN / EN inverters
   at 0.5x and the stacks at 1.5x writes 6 % faster than 1.5x everywhere
   (139 vs 148 ps; 10 / 10 samples, worst 151 ps) with 25 % less `w_en`
   load per column (`2 Wn_out + Wn_in + Wp_in` = 0.81 um instead of
   `3 Wn + Wp` = 1.08 um).
4. **The driver drains load the bitline.** The two stack transistors on each
   bitline add `cjswg * (Wn + Wp)` of gate-edge junction: 0.27 fF per column
   at 1.0x, 8.6 fF at the 32x the rule gives 512 rows — a third of the
   ~26 fF of cells on that bitline (section 1.3). The read deck of the
   characterisation had no write drivers (the canonical-macro point of the
   previous draft), so it under-estimated the read access of a macro with
   row-scaled write drivers at large row counts; the campaign decks now
   include the disabled stacks (`canonical_read`), and the 512x4 read phase
   at the worst case is 2029 ps against 1986 ps without them and with the
   old 32x precharge. The split of item 3 and the smallest coefficient that
   meets the budget in item 2 limit that load: at `rows / 64` the 512-row
   driver drains fall from 8.6 fF to 2.2 fF per column.

### 3.2 Precharge

Sweep of the precharge PMOS scale at SS / 125 C / 0.9 V, read and write
decks (10 ns clock; `T` marks the period `docs/TIMING_AUTOCONFIG.md` would
assign). `TRESTORE` = rising clock edge -> discharged bitline back at
0.9 VDD; `TPRCH` = `PRE` fall -> bitline at 0.9 VDD from 0 V (start-up
precharge, the raw PMOS charging time); the bitline levels are read at the
next wordline rise (VDD = 0.9 V here).

| array | scale (W um) | rows / scale | TRESTORE read / write [ps] | TPRCH [ps] | BL, BLB at next WL [V] | read access [ps] | read PAVG [uW] |
|---|---|---|---|---|---|---|---|
| 8x4 | 0.25 (0.07) | 32 | 652 / 528 | 269 | 0.900 / 0.900 | 661 | 23.3 |
| | 0.5 (0.14) **rule** | 16 | 548 / 483 | 162 | 0.900 / 0.900 | 665 | 23.5 |
| | 1.0 (0.27) | 8 | 492 / 455 | 103 | 0.900 / 0.900 | 671 | 24.0 |
| | 2.0 (0.54) | 4 | 478 / 456 | 77 | 0.900 / 0.900 | 683 | 24.9 |
| 16x16 | 0.25 | 64 | 743 / 637 (623 at T = 2.5 ns) | 309 | 0.899 / 0.900 | 696 | 39.1 |
| | 0.5 | 32 | 615 / 564 (549) | 180 | 0.900 / 0.900 | 699 | 39.9 |
| | 1.0 **rule** | 16 | 563 / 537 (523) | 120 | 0.900 / 0.900 | 705 | 41.5 |
| | 2.0 | 8 | 536 / 526 | 82 | 0.900 / 0.900 | 717 | 44.7 |
| | 4.0 | 4 | 534 / 540 | 53 | 0.900 / 0.900 | 740 | 52.0 |
| 64x16 | 1.0 | 64 | 609 / 592 | 163 | 0.899 / 0.900 | 790 | 67.8 |
| | 2.0 | 32 | 562 / 558 | 106 | 0.900 / 0.900 | 800 | 71.0 |
| | 4.0 **rule** | 16 | 550 / 559 | 68 | 0.900 / 0.900 | 822 | 78.4 |
| | 8.0 | 8 | 542 / 557 | 49 | 0.900 / 0.900 | 863 | 91.1 |
| 256x8 | 4.0 | 64 | 608 / 616 (604 at T = 3.8 ns) | 120 | 0.899 / 0.900 | 1081 | 112.5 |
| | 8.0 | 32 | 593 / 600 (586) | 79 | 0.900 / 0.900 | 1129 | 120.3 |
| | 16.0 **rule** | 16 | 585 / 590 (578) | 58 | 0.900 / 0.900 | 1209 | 133.7 |
| | 32.0 | 8 | 588 / 594 | 45 | 0.900 / 0.900 | 1364 | 160.5 |

What the sweep shows:

1. **The restore is control-path limited.** `TRESTORE` = 440-590 ps of
   fixed delay (clock buffer, `NAND3`, `PRE` buffer; growing with the
   columns through the buffer taper) plus an RC term that is flat for
   `rows / scale <= 16` (< 15 ps) and adds 30-110 ps at `rows / scale` =
   32-64, most at the small arrays where the 0.25x device is 0.07 um wide.
   Every point is inside the budget `0.8 * low_read_wc` (740 / 790 / 889 /
   1206 ps for the four arrays); the tightest is 8x4 at 0.25x (652 ps).
2. **Width costs read access and power.** The precharge drains sit on the
   bitline: the read access grows 3 % (64x16, 1x -> 4x), 12 % (256x8,
   4x -> 16x) and 26 % (256x8, 32x), and the read power 16-19 % over the
   same steps (the `PRE` buffer scales with the load). Today's rule at 256
   rows (16x, 4.32 um x 3 per column) therefore costs 12 % of the read
   access for a restore that is 23 ps faster than 4x.
3. **Equalisation and restore are complete at every width**: BL and BLB
   are at VDD within 1 mV at the next wordline in every deck, also at the
   automatic periods (1.25 ns and 1.9 ns clock-high phases).

Pre-review rule: the smallest width whose RC term stays on the flat part,
`pre = max(0.5, p * rows / 32)` — half of the legacy rule in the netlist
(`p` = 1), the legacy rule at `p` = 2 — plus the budget check
`TRESTORE_wc <= alpha_pre * low_read_wc` in `verify()`. Against the legacy
rule this saves 4-10 % read power and 1-6 % read access at >= 64 rows and
costs 8-12 ps of restore (27-52 ps at 16 rows).

Function-first revision: the knee is a *speed* choice. The restore budget
is `0.8 * low_read_wc` = 0.9-1.6 ns while the restore at `rows / scale` =
64 is 608-743 ps (ratios 0.74 at 16 rows, 0.54 at 64, 0.40 at 256), so the
largest measured `rows / scale` is inside the budget everywhere, and every
halving of the width buys read access and power on the phase that sets the
period (256x8: 16x -> 4x is -11 % read access, -16 % read power). The
revised rule is `pre = max(0.5, p * rows / 64)` (section 4.1): unchanged up
to 16 rows (the 0.5 floor), half of the pre-review width above. Two things
gate going lower still: the FF / FS / 125 C hold of a long bitline by a
narrow precharge against `rows - 1` leaking cells (never measured below
the knee), and the equalisation within 5 mV; section 7 item 7 measures
`rows / scale` = 64-256 for both.

### 3.3 Wordline driver

Sweep of the wordline-driver inverter scale (NAND2 at the square root of
it unless noted) on read decks at SS / 125 C / 0.9 V. `TWLDRV` = `wl_en`
-> WL at VDD/2; slew = WL 10-90 %; "RWL - WL" = replica wordline rise
minus real wordline rise with the fixed AND2 replica driver.

| array | inverter scale (Wn / Wp um) | NAND2 scale | h_inv | h_nand | TWLDRV [ps] | slew [ps] | RWL - WL [ps] | read access [ps] | TCLK_DEC [ps] | read PAVG [uW] |
|---|---|---|---|---|---|---|---|---|---|---|
| 8x4 | 0.5 / 1.0 **rule** / 2.0 | 0.71 / 1 / 1.41 | 6 / 3 / 1.5 | 0.6 / 0.8 / 1.1 | 44 / 40 / 40 | 39 / 27 / 24 | -7 / -4 / -3 | 664 / 665 / 665 | 370 / 372 / 374 | 23.3 / 23.5 / 23.9 |
| 16x16 | 1 / 2 / 4 **rule** / 8 | 1 / 1.41 / 2 / 2.83 | 12 / 6 / 3 / 1.5 | 0.8 / 1.1 / 1.6 / 2.3 | 60 / 53 / 53 / 52 | 65 / 39 / 30 / 28 | -22 / -14 / -12 / -12 | 703 / 704 / 705 / 704 | 438 / 441 / 444 / 448 | 40.0 / 40.6 / 41.5 / 43.4 |
| 64x64 | 4 / 8 / 16 **rule** / 32 | 2 / 2.83 / 4 / 5.66 | 12 / 6 / 3 / 1.5 | 1.6 / 2.3 / 3.2 / 4.5 | 71 / 66 / 67 / 72 | 66 / 45 / 40 / 39 | -30 / -25 / -26 / -32 | 883 / 882 / 881 / 882 | 481 / 485 / 490 / 497 | 197 / 203 / 211 / 226 |
| 16x256 | 16 / 32 / 64 **rule** / 128 | 4 / 5.66 / 8 / 11.3 | 12 / 6 / 3 / 1.5 | 3.2 / 4.5 / 6.4 / 9.1 | 88 / 86 / 93 / 107 | 76 / 56 / 54 / 60 | -48 / -46 / -53 / -67 | 804 / 807 / 806 / 805 | 454 / 461 / 471 / 484 | 307 / 312 / 321 / 337 |
| 16x256, inverter 64 | NAND2 4 / 8 **rule** / 16 | 3 | 12.8 / 6.4 / 3.2 | 141 / 93 / **67** | 83 / 54 / **39** | -101 / -53 / -27 | 807 / 806 / 805 | 454 / 471 / 503 | 316 / 321 / 330 |

What the sweep shows:

1. **The wordline is fastest where the two stage efforts are equal,
   `h_nand ~ h_inv ~ 3`.** At 64 columns the square-root rule already gives
   that (3.2 / 3: 66-67 ps); at 256 columns it gives 6.4 for the NAND2
   (93 ps) and a NAND2 at 16x (3.2) is 25 ps faster with a 28 % steeper
   edge, at +3 % read power. A bigger inverter than `cols/4` never helps
   (`h_inv` < 3 slows the NAND2 stage). Balanced, the wordline path is
   65-70 ps at SS / 125 C / 0.9 V (~30 ps at TT) independent of the column
   count, which meets the 15 % requirement of section 2.2 at 256 columns
   (13 %; 18 % with today's rule).
2. **The read access does not depend on the wordline width at all** (+-3
   ps over 8x variation) because the replica timing waits for a full
   bitline swing; the wordline width only moves the bitline swing at
   `s_en`, which is why it becomes critical with the replica configurations
   of section 3.4.
3. **The fixed AND2 replica driver runs ahead of the real wordline** by 4-7
   ps at 4 columns, 12-22 ps at 16, 25-32 ps at 64 and 46-101 ps at 256
   columns — an early-sensing error that grows with the wordline load and
   with a weak NAND2 (the -101 ps point). Section 3.4 measures the matched
   driver.
4. **The decoder's fixed output inverter** sees `h_dec` = 1.25 x NAND2
   scale: 20 at 256 columns with the 16x NAND2 (`TCLK_DEC` +32 ps, still
   140 ps under the restore in the clock-high phase). The optional
   `dec_inv = max(1, nand / 4)` of section 4.1 keeps it at 5.
5. **Writes do not care either.** The three 16x256 write decks at
   SF / 125 C / 0.9 V with inverter 16 / 32 / 64 (NAND2 4 / 5.7 / 8) give a
   write access of 298 / 297 / 296 ps at a `TWLDRV` of 78 / 77 / 83 ps
   (slew 71 / 53 / 51 ps): the write is limited by the write driver and the
   bitline, not by the wordline width.

Rule: `inv = max(1, cols / 4)` (unchanged), `nand = max(1, cols / 15)`
(= `inv * 0.8 / 3`, replacing the square root): 1.07 at 16 columns
(half of today's 2 — the NAND2 input load on `wl_en` halves), 4.3 at 64
(unchanged), 17 at 256, 34 at 512.

### 3.4 Replica timing

Read decks with the replica hookup varied: `K` active replica cells on
RWL (of the `rows + 1` in the replica column), `N` delay-chain stages
(odd, polarity preserved), replica wordline driven by the fixed AND2
("AND2") or by the row wordline driver plus a dummy row of `cols` cells
("matched"). `s_en - WL` = sense enable rise after the wordline rise;
"BL at s_en" = target bitline level when the amplifier fires (the cell
stores 0, so the differential `dV` = BLB - BL); `TSA` = `s_en` -> `OUT`.

| array | corner | RWL | K, N | read access [ps] | s_en - WL [ps] | BL at s_en [V] | dV at s_en [V] | TSA [ps] | read PAVG [uW] |
|---|---|---|---|---|---|---|---|---|---|
| 16x16 | SS / 125 C / 0.9 V | AND2 | 1, 9 (today) | 705 | 573 | 0.004 | 0.894 | 79 | 41.5 |
| | | AND2 | 1, 5 | 507 | 375 | 0.008 | 0.890 | 79 | 39.8 |
| | | AND2 | 1, 1 | 317 | 176 | 0.061 | 0.830 | 88 | 38.1 |
| | | AND2 | 2, 9 | 682 | 550 | 0.004 | 0.892 | 79 | 41.5 |
| | | AND2 | 2, 5 | 483 | 350 | 0.009 | 0.887 | 79 | 39.9 |
| | | AND2 | 2, 1 | 298 | 152 | 0.122 | 0.770 | 93 | 38.1 |
| | | AND2 | 4, 5 | 474 | 341 | 0.009 | 0.887 | 79 | 40.0 |
| | | AND2 | 4, 1 | 292 | 143 | 0.155 | 0.735 | 96 | 38.2 |
| | | matched | 1, 9 | 725 | 592 | 0.004 | 0.895 | 79 | 42.4 |
| | | matched | 2, 1 | 311 | 168 | 0.078 | 0.814 | 89 | 38.9 |
| 16x16, slowest box cell (PU 0.108, PG 0.108, PD 0.164) | SS / 125 C / 0.9 V | AND2 | 1, 9 | 711 | 580 | 0.004 | 0.891 | 79 | 40.7 |
| | | AND2 | 2, 5 | 485 | 354 | 0.014 | 0.883 | 80 | 39.0 |
| | | AND2 | 4, 1 | 296 | 144 | 0.250 | 0.635 | 100 | 37.3 |
| | | matched | 2, 1 | 316 | 170 | 0.138 | 0.752 | 94 | 38.0 |
| 16x16 | TT / 25 C / 1.0 V | AND2 | 1, 9 (today) | 308 | 253 | 0.002 | 0.992 | 33 | 50.0 |
| | | AND2 | 1, 5 / 1, 1 | 218 / 130 | 163 / 73 | 0.005 / 0.094 | 0.988 / 0.898 | 32 / 34 | 47.8 / 45.4 |
| | | AND2 | 2, 5 / 2, 1 | 208 / 122 | 153 / 63 | 0.006 / 0.163 | 0.987 / 0.825 | 33 / 36 | 47.6 / 45.5 |
| | | AND2 | 4, 5 / 4, 1 | 204 / 119 | 149 / 59 | 0.004 / 0.156 | 0.983 / 0.826 | 33 / 38 | 47.8 / 45.5 |
| | | matched | 2, 1 | 127 | 70 | 0.093 | 0.893 | 35 | 46.3 |
| 16x16 | FF / -40 C / 1.0 V | AND2 | 1, 9 / 1, 5 / 1, 1 | 200 / 140 / 80 | 165 / 105 / 45 | 0.001 / 0.002 / 0.090 | 0.991 / 0.989 / 0.900 | 20 / 20 / 21 | 51.0 / 48.3 / 46.0 |
| | | AND2 | 2, 5 / 2, 1 | 133 / 75 | 98 / 38 | 0.001 / 0.072 | 0.987 / 0.915 | 20 / 22 | 48.4 / 45.9 |
| | | AND2 | 4, 1 | 74 | 36 | 0.249 | 0.739 | 23 | 45.9 |
| 8x4 | SS / 125 C / 0.9 V | matched | 2, 1 | 279 | 149 | 0.051 | 0.838 | 90 | 20.2 |
| 8x4 | TT / 25 C / 1.0 V | matched | 2, 1 | 113 | 62 | 0.073 | 0.918 | 34 | 24.9 |
| 64x16 | SS / 125 C / 0.9 V | matched | 2, 1 | 384 | 231 | 0.300 | 0.615 | 100 | 75.8 |
| 16x256 | SS / 125 C / 0.9 V | AND2 | 1, 9 (today) | 806 | 632 | 0.003 | 0.892 | 81 | 321 |
| | | matched | 1, 9 | 869 | 694 | 0.003 | 0.894 | 81 | 335 |
| | | AND2 | 2, 1 | 388 | 209 | 0.043 | 0.853 | 87 | 318 |
| | | matched | 2, 1 | 441 | 265 | 0.019 | 0.877 | 82 | 332 |
| 256x8 | SS / 125 C / 0.9 V | AND2 | 1, 9 (today) | 1210 | 1080 | 0.100 | 0.802 | 80 | 134 |
| | | AND2 | 2, 5 | 768 | 628 | 0.363 | 0.571 | 92 | 132 |
| | | AND2 | 4, 1 | 492 | 295 | 0.668 | 0.302 | 149 | 131 |
| | | matched | 2, 1 | 611 | 443 | 0.523 | 0.435 | 120 | 131 |
| 256x8 | FF / -40 C / 1.0 V | matched | 2, 1 | 165 | 118 | 0.578 | 0.463 | 34 | 143 |
| 512x4 | SS / 125 C / 0.9 V | AND2 | 1, 9 (today) | 1676 | 1560 | 0.211 | 0.701 | 76 | 156 |
| | | AND2 | 2, 5 | 1040 | 881 | 0.497 | 0.460 | 111 | 155 |
| | | AND2 | 4, 1 | 659 | 439 | 0.722 | 0.253 | 174 | 153 |
| | | matched | 2, 1 | 880 | 701 | 0.587 | 0.381 | 134 | 153 |

Period check for the recommended configuration, 16x16, `(2, 1)` matched,
SS / 125 C / 0.9 V: passes at 1.2 ns (`OUT` still settling at the end of
the wordline phase), fails at 1.0 ns (amplifier not fired); measured
`2 * (TCLK_WLEN + access)` = 1.15 ns against 1.97 ns with `(1, 9)`.
Ten seeded global-variation samples of the same deck (10 ns clock): read
access 297 ps mean / 338 ps worst, differential at `s_en` 0.79-0.81 V,
all samples pass; the limiting phase (`TCLK_WLEN` + access) is 7 % longer
at the worst sample than at the mean.

What the sweep shows:

1. **The delay chain is pure latency at <= 64 rows.** With one replica cell
   the replica bitline discharges at the rate of the real one (`rows + 1`
   cells of load, one active), so by the time `rbl_delay` crosses its
   threshold the target bitline is already at full swing; the eight extra
   stages add 180 ps at TT and 390 ps at the worst case with no gain in
   differential (0.89 -> 0.89 V). `N` = 1 alone brings the 16x16 read from
   308 to 130 ps at TT.
2. **`K` sets the bitline swing at which the amplifier fires, and the
   swing is nearly independent of the global corner** (K = 2, N = 1 at
   16x16: 0.77 V at SS / 125 C / 0.9 V, 0.83 V at TT, 0.92 V at
   FF / -40 C): replica and cell are the same devices and scale together.
   The margin has to be set against *local* mismatch (replica cells fast,
   target cell slow) and the amplifier offset, which the shared-model-card
   MC of the testbench cannot show; the slowest box cell costs 60-100 mV of
   differential at `(2, 1)` / `(4, 1)`.
3. **At large row counts the swing shrinks with K:** 256 rows 0.80 / 0.57 /
   0.30 V and 512 rows 0.70 / 0.46 / 0.25 V for `(1, 9)` / `(2, 5)` /
   `(4, 1)`, and the amplifier needs longer to resolve a small input
   (`TSA` 80 -> 150-175 ps). `(2, 1)` on the matched wordline leaves
   0.44 V at 256 rows (611 ps, half of today) and 0.38 V at 512 rows
   (880 ps); `(4, 1)` at 512 rows leaves 0.25 V, below the 0.3 V `dV_min`
   placeholder of section 5.
4. **The matched replica wordline removes the early-firing error** (RWL - WL
   = -12 ps -> +2 ps at 16x16, -53 -> 0 ps at 16x256) and adds the same
   time to the read access; with `(1, 9)` that is only cost (+20 / +63 ps),
   with a tuned `(K, N)` it is what makes the swing at `s_en`
   size-independent (16x16 `(2, 1)`: 0.77 V with the AND2, 0.81 V matched).
5. **Read-spec consequence.** With `(2, 1)` and the matched replica the read
   access is 113-127 ps at TT for 8-16 rows, ~170 ps at 64 rows and
   ~265 ps at 256 rows (worst case / 2.3): the 200 ps limit of
   `global.yaml` is met up to 64 rows by the replica configuration alone;
   no driver width does that.

## 4. Proposed method

### 4.1 Sizing rules

2026-09-08 qualification amendment: the function-first formulas below remain
proposals. Keep the implemented coefficients as the initial full-local baseline,
then test each rule family under the fixed-corner plus per-device contract at
the top of this document. Neither `rows/64` nor a smaller floor is accepted from
the old shared-model results alone. Recompute dependent loads for each proposed
baseline, then freeze its sizes and measured clock throughout candidate/PVT MC.

One scale per driver, `max(floor, load term)`, floors and coefficients
from section 3 (netlist-consistent, `p` = `parasitic_factor` = 1). The
floor is functional (any clock); the load term is the smallest scale, inside
the characterised range, that keeps the driver under its `alpha` budget of
the array's own read phase (section 2.1 item 2). `R_w` and `R_pre` are the
largest `rows / scale` values measured to pass the budget at every
characterised size (64 today, section 3.1 item 2 and section 3.2); they
move only with the sweep of section 7 item 7. The code implements the
pre-review column (`k_w` = 1/16, `rows / 32`) until item 6 of the status
list is done.

| driver | rule (function-first) | floor | load term | pre-review rule (implemented) | changes against the pre-review rule |
|---|---|---|---|---|---|
| write driver, output stacks (M5-M12) | `wd_out = max(1.5, p * rows / R_w)`, `R_w` = 64 | 1.0 (box cell writes 10 / 10 at SF / 125 C / 0.9 V) x 1.5 strength margin | budget `alpha_w * low_read_wc`; write access at `rows / scale` = 64 is 313-366 ps at SF / 125 C / 0.9 V, i.e. a write phase of 0.37-0.54 of the read phase at 64-256 rows | `max(1.5, k_w * p * rows)`, `k_w` = 1/16 (100 ps TT write access, the `global.yaml` limit) | identical up to 24 rows (both at the 1.5 floor); the floor holds to 96 rows where the pre-review rule reaches 6x; 4x smaller above (256x8: 16 -> 4, 512x4: 32 -> 8; driver drains on the bitline 8.6 -> 2.2 fF per column at 512 rows, `w_en` gate load 184 -> 52 units); +100-150 ps of write access at the worst case, inside the budget. `k_w` remains as an explicit speed override (`sizing.speed.k_w`) |
| write driver, DIN / EN inverters (M1-M4) | `wd_in = max(0.5, wd_out / 4)` | 0.5 | fan-out 4 into the stacks | same | none; -25 % `w_en` load per column, -6 % write access against one width pair |
| precharge PMOS (x3) | `pre = max(0.5, p * rows / R_pre)`, `R_pre` = 64; check `TRESTORE_wc <= alpha_pre * low_read_wc` | 0.5 (0.135 um) | budget `alpha_pre * low_read_wc`; restore at `rows / scale` = 64 is 0.40-0.74 of the read phase | `max(0.5, p * rows / 32)` (knee of the restore curve) | identical up to 16 rows (both at the 0.5 floor); half above (256x8: 8 -> 4 = -11 % read access, -16 % read power; 512x4: 16 -> 8, `PRE` gate load 180 -> 90 units); +30-110 ps restore, inside the budget |
| wordline inverter | `inv = max(1, cols / 4)` | 1 | `h_inv` = 3 | unchanged |
| wordline NAND2 | `nand = max(1, cols / 15)` | 1 | `h_nand` = 3 (balanced with the inverter) | square root -> linear: half at 16 columns, same at 64, 2x at 256 (-25 ps, -28 % slew), 4x at 512 |
| decoder last-stage inverter (optional) | `dec_inv = max(1, nand / 4)` | 1 | `h_dec` <= 5 | new; only matters above 64 columns (`TCLK_DEC` is in the clock-high phase) |
| replica wordline | row `WordlineDriver` + dummy row of `cols` cells | - | matches the real wordline within 2 ps (section 3.4) | replaces the fixed AND2 that runs 4-101 ps ahead |
| replica cells / delay stages | `(K, N)` from section 4.3 | - | sensing margin `dV_min` | default unchanged `(1, 9)`; `(2, 1)` recommended after the mismatch qualification |
| control buffers | fan-out 8, `n_stages = 2 * ceil(log8(F) / 2)`, loads from the resolved scales | - | - | stage count from the effort instead of 2 / 4 |

The rules are evaluated in the order write driver -> precharge -> wordline
driver -> replica -> control buffers, because the buffer loads depend on
the resolved widths. Every coefficient is stored with the hash of the
inputs it was fitted for (cell YAML values and bounds, periphery YAMLs,
PDK, `parasitic_factor`); a changed input re-runs the sweeps of section 7
item 1 (about 150 decks, 1 h on 30 cores) rather than silently reusing
the numbers.

### 4.2 Resolver

```
resolve_driver_sizes(cell, rows, cols, mux, yaml, pdk, cfg) -> DriverSizes:
    key = hash(cell yaml values + bounds, periphery yamls, cfg, pdk, rule version)
    if key in sizing_table:                      # characterised and verified (section 7)
        return sizing_table[key]
    p = cfg.parasitic_factor                     # 1.0 for the netlist-consistent flow
    low_r   = timing_model(cell, 'read', mux, 'low')(rows, cols) * k_low_read      # TIMING_AUTOCONFIG; grows with rows
    # A. write driver: DC floor (cell / PDK, section 3.1) and the smallest characterised load term
    #    inside the write budget.  R_w = 64 is the largest measured rows/scale; the budget-derived
    #    value  k_fit * p * rows / (alpha_w * low_r - wlen_wc - t0_w)  is asymptotically constant
    #    and replaces R_w once section 7 item 7 has measured that range.
    wd_out  = max(cfg.wd_floor[cell][pdk] * cfg.wd_floor_margin,  p * rows / cfg.R_w)   # 1.0 * 1.5, R_w 64
    if cfg.speed: wd_out = max(wd_out, cfg.speed.k_w * p * rows)  # optional absolute-speed override only
    wd_in   = max(0.5, wd_out / 4)                                 # input inverters: fan-out 4 into the stacks
    assert wlen_wc + t0_w + k_w_fit * p * rows / wd_out <= cfg.alpha_w * low_r         # fitted, section 3.1
    # B. precharge: smallest characterised load term inside the restore budget (R_pre = 64), then the check
    pre     = max(cfg.pre_min, p * rows / cfg.R_pre)               # pre_min 0.5
    assert t0_pre(cols) + k_pre * p * rows / pre <= cfg.alpha_pre * low_r        # fitted t0 / k, section 3.2
    # C. wordline driver: balanced NAND2 / inverter efforts (h = 3), decoder inverter bounded
    wl_inv  = max(1, cols / 4)                                     # h_inv = 3
    wl_nand = max(1, cols / 15)                                    # h_nand = 3 (0.45 um NAND2 input per 0.36 um unit)
    dec_inv = max(1, wl_nand / 4)                                  # h_dec <= 5 (optional, > 64 columns)
    # D. replica path
    rwl     = 'matched'                                            # row driver + dummy row of cols cells
    K, N    = cfg.replica_table.get((cell, rows_bucket(rows)), (1, 9))   # section 4.3
    # E. control buffers from the resolved loads (today hand-computed in the testbench)
    loads   = dict(pre_load=(cols + 1) * 3 * 0.27 * pre / 0.36,
                   wen_load=cols * (2 * 0.18 * wd_out + (0.18 + 0.36) * wd_in) / 0.36 + 4 * wenb_scale(cols),
                   wl_load=rows * wl_nand, num_sa=cols // mux_in)
    return DriverSizes(wd_out, wd_in, pre, wl_inv, wl_nand, dec_inv, rwl, K, N, loads, source='rule')
```

The load terms use `rows` and `cols` only; the floors and coefficients
(`wd_floor`, `R_w`, `R_pre`, `k_w_fit`, `t0_w`, `k_pre`, `t0_pre`) are per
(cell type, PDK) constants fitted in section 3 and stored next to the rules
with the hash of the inputs they were fitted for. `timing_model` is the
phase model of `docs/TIMING_AUTOCONFIG.md` (the two proposals share the
worst-case phases); it is used only for the budget *asserts*, never to size
a driver, so there is no loop between the period and the sizes: the read
phase that sets the budgets is the bitline-limited phase, which the driver
widths move only through their drain loading (a few per cent), and the
measured period of `timing.py` is taken after the sizes are frozen.
Nothing in the resolver simulates; a size outside the characterised range or
a changed YAML falls back to the rules and marks the result `source='rule'`,
and the verification of section 7 promotes it to the table.

### 4.3 Replica configuration (`K`, `N`)

Two-sided rule, evaluated on the matched replica wordline:

- early side: `dV_at_sen(rows, K, N)` at the worst case with the slowest
  box cell must exceed `dV_min` = SA offset (3 sigma, per-device MC of
  section 7 item 5) + replica / cell mismatch (3 sigma) + 100 mV; until
  those two are measured `dV_min` = 0.3 V;
- late side: `TCLK_WLEN + sen_minus_wl + TSA + latch` is the read phase
  the period is derived from (`docs/TIMING_AUTOCONFIG.md`), so it is never
  violated, only paid for.

`N` = 1 always (the chain only adds latency, item 1 above); `K` is the
largest value whose swing at the worst case, slowest box cell, clears
`dV_min`. From section 3.4 (nominal cell unless noted):

| rows | `(K, N)` | dV at s_en, SS / 125 C / 0.9 V [V] | read access SS / TT [ps] | today `(1, 9)` SS / TT [ps] |
|---|---|---|---|---|
| <= 16 | 2, 1 | 0.81 (0.75 slowest box cell) | 279-311 / 113-127 | 705 / 308 |
| 64 | 2, 1 | 0.62 | 384 / ~170 | 822 / 358 |
| 256 | 2, 1 | 0.44 (0.46 at FF / -40 C) | 611 / ~265 | 1210 / 524 |
| 512 | 2, 1 (or 1, 1, not run) | 0.38 | 880 / ~380 | 1676 / 739 |

Measured since: the SA / latch offset ensembles (100 copies, section 7
item 5) give a conservative 3-sigma bound of 170 mV at both SS / 125 C /
0.9 V and FF / -40 C / 1.0 V, so `dV_min` is at least 0.27 V + 3 sigma of the
replica / cell skew, i.e. above the 0.3 V placeholder that the scorer still
hard-codes (`sense_differential >= 0.3`); `report.py` computes the qualified
value as `offset + 3 sigma + 0.1 V` once the local ensembles are in. With the
matched replica and `(1, 9)` the campaign measures 0.89 V at <= 64 rows,
0.78 V at 256 and 0.67 V at 512 rows (status table): the default clears any
plausible `dV_min` at every size. `(2, 1)` leaves 0.62 V at 64 rows but only
0.44 / 0.38 V at 256 / 512 rows, which a 0.35-0.45 V `dV_min` does not accept.

Recommendation (function first): keep `(1, 9)` as the shipped default; after
the local-mismatch qualification has set `dV_min`, offer `(2, 1)` up to 64
rows (0.62 V or more at the worst case, read access halved) as the
speed option, and characterise `(1, 1)` / `(1, 3)` for 256-512 rows before
any faster default there (decision 5 of section 9). The absolute read limit
of `global.yaml` is not a reason to change the default (section 2.1 item 4).
The resolver exposes `replica: {K, N}` in the `sizing` block and the table
above is the seed of `sizing_rules.json`.

### 4.4 Control buffers

Proposed: keep the V2.0.2 fan-out rule (`TaperedBuffer`, effort ~8, 2
stages up to 16x, 4 above) and compute the loads in the resolver from the
resolved scales, not from the rules (before V2.0.4 `create_time_circuit`
re-derived `pre_load` and `wen_load` from `PrechargeFactory.width_scale`
and `WriteDriverFactory.width_scale`; with a split write driver or a table
entry that goes stale). One change: choose the stage count from the effort
(`n = 2 * ceil(log8(F) / 2)`, even for polarity) instead of the fixed
2 / 4 split at 16x, so the rule stays valid for the `PRE` / `w_en` buffers
of the largest arrays instead of silently exceeding a fan-out of 8.

Implemented (supersedes the fan-out-8 rule for these buffers, status
section): the loads come from `DriverSizes.loads` with the measured 0.5 fF
per enable pin; the per-stage effort is 6 for `w_en`, 4 for `s_en` and
`PRE` (two-unit floor), 3 for the isolation buffer (with twice the final
NMOS strength), and 3 for `s_en` with RC decks, because the 40 ps TT edge
budget of section 2.2 was missed at effort 8 on the wide arrays; the stage
count is `2 * ceil(log2(F) / 6)`; wide inverters are folded into `NF`
fingers of <= 2 um because the PDK's gate resistance (`RGATEMOD=1`) made
single-finger 60-90 um devices 3x slower than their width promised (44 ->
14 ps edges at constant total width). The row of section 1.1 describes the
pre-V2.0.4 buffers. The stricter efforts cost `PRE` / `w_en` buffer area and
power on the wide arrays (16x512 write 3.3 mW at SS / 125 C / 0.9 V, mostly
buffers); with the smaller precharge and write drivers of section 4.1 the
loads, and with them the buffers, shrink 2-4x on the tall arrays.

## 5. Margin policy

| source of uncertainty | how it is covered | size |
|---|---|---|
| global process / voltage / temperature | every speed term sized and checked at SS / 125 C / 0.9 V, write-ability at SF / 125 C / 0.9 V, leakage at FF and FS / 125 C (the corners `docs/TIMING_AUTOCONFIG.md` section 3.4 measured as worst per phase) | 2.1-2.3x on every control phase, up to 3.8x on the write access, relative to TT |
| global variation beyond the corner | the MC testbench's `AGAUSS(5 %)` on `vth0 / u0 / voff` per model card is a random global shift on top of the corner; verification decks run 10 seeded samples and require 0 failures *and* the analog margins below (a marginal pass is not a pass) | worst of 10 samples over the mean: write access +16 % (8x4, 1.0x), +20 % (box cell, 1.5x), +48 % (box cell at the 1.0x floor); read phase +7 % (section 3.4) |
| DC write-ability | floor = smallest scale that writes the parameter-box cell (max PU, min PG, max PD) at SF / 125 C / 0.9 V with 10 / 10 samples, times `wd_floor_margin` = 1.5 (a driver whose NMOS is 33 % weaker than the model still writes); acceptance: driven bitline < 0.1 VDD, `q_written`, `q_retained` | section 3.1 |
| sensing margin | `dv_at_sen >= dV_min` for the slowest cell of the parameter box against the nominal replica, `dV_min` = SA offset (3 sigma, per-device flow) + replica / cell mismatch (3 sigma) + 100 mV; placeholder 0.3 V (100 + 100 + 100 mV) until measured; `K >= 2` halves the replica's own sigma | section 3.4: 0.75-0.84 V at <= 16 rows, 0.62 V at 64 rows with `(2, 1)` |
| phase budgets | `alpha` fractions of the worst-case read phase *of the same array* (0.8 restore / write, 0.15 wordline path) so the period stays array-limited; on top of that the 25 % margin of the period; the budgets grow with the rows, the drivers are the smallest inside them | section 2.2 and status table: worst measured 0.74 (restore, 16x16 at 0.25x), 0.61 (restore, 16x256), 0.56 (write, 16x256), 0.14 (wordline, 16x16 `(1, 9)`) |
| fast-corner races and ordering | WL off before `PRE` on, `wl_en_bar` gating, address hold (`next_row`), `s_en` after isolation, no premature `s_en`: checked at FF / -40 C (and at the high end of the VDD envelope, decision 6 of section 9) as well as at SS, because a race margin is a difference of two paths and is smallest where both are fastest | section 7 item 9; today only the replica skew is checked at FF |
| upper bound of the relaxed period | at the frozen period the FF / 125 C and FS / 125 C read and write decks must pass `sense_differential`, restoration and retention (unselected-bitline leak over a 1.2-2.6 ns wordline phase; V2.0.3: 0.69-0.73 V after 5 ns); half-selected cells retain (section 7 item 10) | section 2.1 item 3; the campaign runs these decks at the frozen period (single nominal sample) |
| parasitics not in the netlist | `parasitic_factor` on the row (bitline) load terms (default 1.0 = netlist-consistent; 2-3 for the bitline of a laid-out array per section 1.3); the column (wordline) terms have no factor and no distributed RC, so a wide laid-out array needs `w_rc` plus extraction to re-characterise the wordline rules | section 1.3 |
| local mismatch (cell vs replica, SA offset) | now the default MC testbench mode, covering every instantiated MOS in array and periphery at a fixed global corner; 5% relative perturbation has no area dependence or calibrated correlations, so measured margins inherit that uncertainty; historical SA offset bound 170 mV, new full-path ensembles pending | section 4.3 and the 2026-09-08 schedule |
| local mismatch on the write path | both series driver devices, input/hold logic, enables, pass gates and cell devices vary independently. The 1.5 floor is a starting value, not a bound on independent tails; require SS and SF full-local write-box ensembles and retention after release | section 7 item 8; new pilot running |

The margins are applied to the sizes at the worst case; a run at TT / 25 C
then shows the corresponding slack, which is physically right for a design
that must work at SS / 125 C / 0.9 V, and the yield flow that is meant to
see driver-related failures has to run at the same worst-case PVT (both
testbenches accept `corner`, `temperature` and `vdd`).

## 6. Integration

- New module `sram_compiler/sizing/driver_sizing.py`: `DriverSizes`
  (immutable), `resolve_driver_sizes()`, the coefficient / floor set
  (`sizing_rules.json`, with the YAML hash they were fitted for), the
  verified table (`sizing_table.json`) and `verify()` (section 7).
- `global.yaml`: a `sizing` block — `mode: auto | rules_only | fixed`,
  `alpha_pre / alpha_w / alpha_wl`, `wd_floor_margin`, `parasitic_factor`,
  `replica: {K, N}` (default `1, 9`), `table` path.
- `parameter_factor.py`: `PrechargeFactory`, `WriteDriverFactory`,
  `WordlineDriverFactory` take a `scale` (and the write driver an
  `out_scale`) from `DriverSizes` instead of computing it from `num_rows /
  num_cols`; the static rules stay as the `rules_only` fallback.
  `WriteDriver` gets separate input / output widths (12 transistors, two
  width pairs); `WordlineDriver` unchanged; `DecoderCascade` gets an
  optional last-stage inverter scale.
- `sram_6t_core_testbench.py`: `create_time_circuit` takes the buffer loads
  from `DriverSizes.loads`; `create_and2_for_rwl` becomes
  `create_replica_wordline` (row driver + dummy row); `create_replica_column`
  connects `K` cells to RWL; `TIME` takes `dc_stages`.
- The sweep paths (`sweep_precharge`, `sweep_writedriver`,
  `sweep_wordlinedriver`) emit the resolved scale as a SPICE expression on
  the swept parameter (as the wordline driver already does), so a swept base
  width keeps the array-dependent scale instead of bypassing it (the
  precharge and write-driver sweeps bypass it today).
- `size_optimization/exp_utils.py` and `main_estimation.py`: resolve the
  sizes once, before the loop, from the baseline cell; record `DriverSizes`
  with the experiment; the candidate cell does not re-size the periphery.
- `utils.estimate_scaled_array_area`: use the resolved precharge and
  wordline-driver widths (`prc_max_width`, `wld_max_width`) instead of the
  YAML base widths, so area follows the sizing.

## 7. Validation plan

1. **Floors and coefficients (done for this proposal, section 3):** write
   driver scale sweep at SF / 125 C / 0.9 V on 8x4 (default and box-corner
   cell, 10 samples), 16x16, 64x16, 256x8, 10T 8x4; precharge scale sweep at
   SS / 125 C / 0.9 V on 8x4, 16x16, 64x16, 256x8, read and write, plus the
   configured periods (2.5 / 3.8 ns); wordline-driver scale sweep on 8x4,
   16x16, 64x64, 16x256; replica wordline matched vs AND2 on 16x16,
   16x256, 256x8; `(K, N)` sweep on 16x16 at SS / 125 C / 0.9 V, TT and
   FF / -40 C, on 256x8 and 512x4 at SS; `(2, 1)` matched on 8x4 .. 512x4
   with a period sweep and 10 samples at 16x16.
2. **Rules against the sweeps:** for every characterised size the resolver's
   scales must give measured metrics inside the acceptance of section 2.2
   with the stated margins; where the rule is above the smallest passing
   scale, the ratio is the realised margin and is printed.
3. **Verification decks per array** (`verify()`): read at SS / 125 C /
   0.9 V, write at SS and SF / 125 C / 0.9 V, automatic period, 10 seeded
   full-local samples at a fixed global corner (including every array and
   peripheral MOS), `next_row` hazard variant at >= 256 rows; all waveform checks
   and acceptance metrics pass; results stored with the hash. Run for the
   27 sizes of the V2.0.2 sweep, 6T and 10T, mux on and off.
4. **Parasitic sensitivity:** repeat item 3 on 16x16, 64x64 and 256x8 with
   `w_rc = True` (`pi_res` 100 Ohm, `pi_cap` 1 fF) and `parasitic_factor` 2;
   the drivers sized for factor 2 must pass with the RC deck.
5. **Local mismatch tails:** local mismatch is now present in every acceptance
   deck, including write/control paths. In addition, dedicated 16x16 and 256x8
   full-array read-box ensembles (100 samples each at SS/hot and FF/cold)
   measure replica / cell skew alongside separate SA offset characterization;
   keep mux/RC/operation/cell-box groups and independent batch seeds distinct.
   `(K, N)` accepted only if `dv_at_sen` at the 3-sigma skew exceeds `dV_min`.
6. **Optimiser smoke test:** `demo_*` on 32x1 with `sizing: auto`; a cell at
   the box corner (max PU / min PG) must still be written by the frozen
   periphery at SF / 125 C / 0.9 V; a cell outside the box must fail with
   `q_written = False`, not with a resized driver.

Added by the 2026-09-07 review; now scheduled after the full-local baseline
screen rather than after the terminated shared-model campaign. Preserve the
historical `v2.0.4-qualification-10` evidence and distinguish every new model,
rule vector, seed ensemble and scoring revision:

7. **Characterised-range extension for the load terms (gates `R_w`,
   `R_pre`):** write-driver scale sweep at `rows / scale` = 64, 128, 256 on
   128x32, 256x8 and 512x4 (6T and 10T, mux on and off) at SF and SS /
   125 C / 0.9 V with the box cell and 10 samples, at the frozen period:
   driven bitline < 0.1 VDD, `cell_written`, `cell_retained`, write phase
   <= 0.8 of the read phase; precharge scale sweep over the same
   `rows / scale` on the same arrays, read and write decks at SS / 125 C /
   0.9 V *and* the hold / leakage decks at FF and FS / 125 C: restoration
   >= 0.98 VDD and equalisation <= 5 mV at the end of every clock-high
   phase, restore <= 0.8 of the read phase. Record the read access and read
   power against the scale (the gain that motivates the change). The
   largest `rows / scale` that passes everything becomes the new `R_w` /
   `R_pre`; the fitted `k_w_fit`, `t0_w`, `k_pre`, `t0_pre` go into
   `sizing_rules.json`.
8. **Per-device write-ability:** the write-box decks at SS and SF / 125 C / 0.9 V
   with per-device mismatch (100 samples in ten seeded batches, as for the
   read box) on 8x4, 16x16 and 256x8, 6T and 10T: no sample may fail
   `cell_written` / `cell_retained`; the worst driven-bitline level and
   write access are the realised floor margin. If a sample fails, the
   failure must first be localized to the driver, cell or control/hold path.
   Raise the floor only when the current output strength is responsible; a
   control-edge failure must not be hidden by blindly increasing the driver.
9. **Fast-corner races:** the read and write decks and the `next_row` hazard
   variant (>= 256 rows) at FF / -40 C at the high end of the VDD envelope
   (1.0 V today, 1.1 V if decision 6 of section 9 adopts +10 %):
   `wl_off_before_precharge`, `unselected_wordlines_quiet`,
   `neighbor_retained`, `no_premature_sense`, `replica_skew`, and the
   `s_en` coupling glitch check, at the frozen period.
10. **Half-select and the wordline-phase upper bound:** a write deck variant
    with the mux on in which only the selected column of each mux group is
    driven (the write drivers of the other column tristated, `DIN` held),
    at FF / 125 C, FS / 125 C and FF / -40 C at the frozen period, with the
    read-box cell (min PD, min PG): every half-selected cell retains, and
    the unselected bitlines are restored at the end of the phase. The
    compiler's write deck drives every column today, so a muxed macro's
    half-select exposure over a relaxed wordline phase has never been
    exercised (section 11 item 6). The same decks bound the relaxed period
    from above (section 2.1 item 3); 512x4 (5.1 ns) and 16x512 are the
    first cases.
11. **Wordline budget with the fast replica:** re-run the 8x4 and 16x16
    `(2, 1)` read decks against the phase-relative wordline check of section
    2.2 (`0.15 * (low_read - TCLK_WLEN)`), which the access-relative form
    fails at 14-17 % while the wordline is functionally right; confirm that
    the phase-relative form passes and that the balanced `cols / 15` NAND2
    is the smallest that does.

## 8. Effort

| step | size | state |
|---|---|---|
| `driver_sizing.py` (resolver, rules, table, verify wrapper) | ~300 lines, 1 day | done (status section) |
| write driver split, decoder inverter scale, matched replica wordline, `K` / `N` plumbing | ~150 lines, 1 day | done |
| factory / testbench / YAML / optimiser hooks, sweep-path expressions | ~100 lines, 0.5 day | done except `main_estimation.py` |
| validation runs (section 7, items 2-4) | 1,786 decks; 656 done after 10.3 h on 80 workers, the tall / wide arrays and the per-device batches are the long tail (six-hour allowance per sample) | running |
| full-local qualification (items 3-5, 8) | historical SA offset evidence retained; V2.0.5 default integration and pilot complete; full-path rule diagnosis running; tail and half-select qualification pending | running |
| function-first revision: `R_w` / `R_pre` clamp, `speed` override, phase-relative wordline check, `dV_min` from the report instead of the 0.3 V constant | ~60 lines in `driver_sizing.py`, `sizing_rules.json`, `qualification.py`, `global.yaml` | after the campaign |
| section 7 items 7-11 (range extension, per-device write box, fast-corner races, half-select deck, `(2, 1)` wordline check) | ~150 decks for item 7, ~60 for items 8-9, a testbench variant (~40 lines) for item 10; ~1 day of machine time at 80 workers, dominated by the 512x4 / 256x8 per-device batches | after the campaign |

## 9. Decisions needed

1. **What the optimiser's timing constraint is.** Proposed: the
   array-relative budget of section 2.1 item 2 (a candidate cell is
   infeasible when its read or write phase at the worst case exceeds the
   frozen period's phase, i.e. `TCLK_WLEN + access > T / 2 / 1.25`), with
   the absolute `delay.upper` of `global.yaml` reported but not enforced
   (today `exp_utils.py` enforces the absolute limits and marks every array
   above 64 rows infeasible). Alternatives: scale `delay.upper` with the
   array from the phase model, or keep the absolute limits as a hard
   constraint and accept that only the `(2, 1)` replica at <= 64 rows can
   meet them.
2. `alpha` defaults (0.8 restore / write, 0.15 wordline path, relative to
   the phase), `wd_floor_margin` 1.5, and the characterised-range clamps
   `R_w` = `R_pre` = 64 as the interim load terms (identical to the
   pre-review rules up to 24 / 16 rows, where both sit on the floors); whether an explicit `speed.k_w`
   override (1/16 = 100 ps at TT) is wanted at all.
3. Worst-case envelope: 125 C / 0.9 V with SS for speed and SF for
   write-ability (measured); if the product envelope differs, the floors
   and factors are re-read from the same sweeps at the other corners.
4. Whether the parameter box of the optimiser (cell YAML `upper / lower`)
   defines the write-ability floor (proposed) or only the default cell;
   the box raises the floor from 0.75 to 1.0 (a third more width below 24 rows).
5. `parasitic_factor` default for the flow: 1.0 (netlist-consistent,
   proposed) or a fixed derating (2.0) until extraction data exists; and
   whether the wordline (column) terms get their own factor.
6. Fast-corner VDD: the campaign's fast corner is FF / -40 C / 1.0 V; a
   0.9-1.1 V envelope puts the races and the leakage decks at 1.1 V (section
   7 items 9-10).
7. Replica default `(K, N)`: keep `(1, 9)` (proposed: 0.67 V of differential
   at 512 rows, ~300 ps read at TT) with `(2, 1)` as a speed option at <= 64
   rows after the mismatch qualification of section 7 item 5; the absolute
   read limit is not a reason to change the default (section 2.1 item 4).
8. Order of work: finish the running campaign on the pre-review rules
   (its floors, budget checks and corner evidence carry over), then apply
   the revision and run section 7 items 7-11 as a second campaign, rather
   than restarting now.

## 10. Relation to the previous draft of this file

Kept: the two-sided sensing constraint with local mismatch as its reason;
one physical width vector per array (never per PVT); the acceptance list
(restoration, equalisation, isolation ordering, write overlap and
retention, disturb, back-to-back patterns); the canonical-macro remark (the
read deck still omits the write drivers, section 6 keeps that as a
testbench item); the yield-driven failure budget and the caveats on the MC
historical shared-card testbench and on FreePDK45 being predictive. V2.0.5
adds full local mismatch by default; the predictive-model limitation remains.

Replaced: the generic per-stage delay model with fitted `k_j / W_j` terms
and the constrained-Bayesian search over widths — the compiler's drivers
have one scale each and their loads are counts of rows and columns, so a
floor plus a proportional term per driver, calibrated by one sweep per
driver at its worst corner, is the whole model; the 20 % reserve against
the 200 / 100 ps `global.yaml` limits — the read limit is set by the replica
configuration, not by driver widths (section 2.1), so the drivers are
budgeted against the worst-case phases instead.

## 11. Review 2026-09-07 — blind spots and where they are handled

Premise of the review: the timing requirement on a driver may relax as the
array grows, because the load grows with the array and so does the
bitline-limited phase the driver is budgeted against; functional
correctness comes first, the absolute `global.yaml` limits last. Measured
against that premise (status table, campaign at 656 of 1,786 decks):

1. **The load terms were absolute-speed terms, not budget terms.** `k_w` =
   1/16 was chosen for a 100 ps TT write access (the YAML limit) and the
   precharge `rows / 32` for the knee of the restore curve; at 256-512 rows
   they leave the write phase at 25-33 % and the restore at 29-36 % of the
   read phase against 80 % budgets, with 16-32x drivers whose drains load
   the bitline that sets the period (8.6 fF per column at 512 rows) and
   whose gates set the `PRE` / `w_en` buffers (180-184 units of load at
   512x4). Handled: sections
   0, 2.1, 4.1, 4.2 (`R_w` = `R_pre` = 64 as the characterised-range clamp,
   budget-derived asymptote, `k_w` demoted to an explicit speed override);
   section 7 item 7 measures `rows / scale` = 64-256 before the clamp moves.
2. **The write and restore fits stop at `rows / scale` = 64.** Below the
   floor the write is bitline-limited in a regime nobody measured (the
   transient condition, not the DC one), and a narrow precharge on a long
   leaky bitline at FF / FS / 125 C was never measured either. Handled:
   the clamp of item 1 and section 7 item 7.
3. **The optimiser enforces the absolute YAML limits.** `exp_utils.py`
   marks a candidate infeasible against 200 / 100 ps, which no array above
   64 rows can meet with any driver or replica setting; the campaign
   correctly reports the same limits as a separate metric. Handled: section
   2.1 item 4 and decision 1 of section 9 (constraint = array-relative
   budget; YAML limits informational).
4. **Relaxing the period has an upper bound that nothing in the plan
   stated.** A long wordline phase at the fast, hot corners lets the
   unselected bitline leak (V2.0.3: 0.69-0.73 V after 5 ns at FF / FS /
   125 C) and exposes half-selected cells; the 512x4 period is 5.1 ns. The
   campaign runs FF / 125 C and FS / 125 C at the frozen period (one
   nominal sample) which covers the selected column; the half-selected
   columns are not covered (item 6). Handled: section 2.1 item 3, section
   2.2 last row, section 5, section 7 item 10.
5. **Races and ordering are checked at the slow corner only.** `WL` off
   before `PRE` on, address hold (`next_row`), `s_en` coupling: the campaign
   runs the fast corner (FF / -40 C / 1.0 V) as one nominal deck without
   the hazard variant, and the VDD envelope has no high end. Handled:
   section 2.2 (wordline row), section 5, section 7 item 9, decision 6.
6. **The write deck drives every column, so a muxed macro's half-select
   is never exercised.** With the mux on the compiler still instantiates
   one write driver per column and writes all of them; a real 2:1-muxed
   macro writes one column per group and half-selects the other for the
   whole wordline phase. Handled: section 7 item 10 (testbench variant,
   not yet written).
7. **Local mismatch is planned for sensing only.** The per-device batches
   use the read-box cell; the write floor margin (1.5) is a global-shift
   margin and the four series NMOS of a write path (two stack transistors,
   pass gate) plus the cell PMOS have no per-device qualification. Handled:
   section 2.2 (write row), section 5, section 7 item 8.
8. **The `dV_min` placeholder is already below the measured SA offset.**
   170 mV (3 sigma bound) + 100 mV margin + the replica / cell skew puts
   `dV_min` at 0.35-0.45 V; the scorer hard-codes 0.3 V and section 4.3
   recommended `(2, 1)` to 256 rows (0.44 V) on the placeholder. Handled:
   section 4.3 (`(1, 9)` default confirmed, `(2, 1)` only <= 64 rows as a
   speed option, `dV_min` from `report.py`).
9. **The wordline speed check is access-relative and breaks with a fast
   replica.** `0.15 * access` is 42-47 ps at 8-16 rows with `(2, 1)` while
   the balanced wordline path is 67-83 ps (`TWLDRV` + slew) at SS / 125 C /
   0.9 V; the code
   already uses the phase-relative form (`0.15 * (low_read - TCLK_WLEN)`),
   the proposal text did not. Handled: section 2.2, section 7 item 11.
10. **Section 4.4 no longer described the implemented buffers.** The
    implementation moved to per-buffer efforts of 3-6, a `log2(F) / 6`
    stage rule and `NF` gate fingering after the 40 ps edge budget failed on
    wide arrays; the proposal still said fan-out 8 / 2-4 stages. Handled:
    section 4.4 (implemented rule and its reason recorded; section 1.1 marked
    as pre-V2.0.4).
11. **`parasitic_factor` derates the row terms only.** The wordline (column)
    terms have no factor and the netlist has no distributed wordline RC, so
    a laid-out wide array cannot be represented by `p` alone. Handled:
    section 5, decision 5 (open; needs extraction data).
12. **The campaign's long tail.** 1,786 decks with a six-hour allowance per
    sample and ten-sample per-device batches on full transistor arrays; the
    tall / wide arrays and the local ensembles dominate. Not a correctness
    issue; recorded in section 8 so that the second campaign of section 7
    items 7-11 is planned as a separate run (decision 8).

Not changed by the review: the write-ability floor (1.0 x 1.5, the one
measured functional failure and its fix), the split write driver, the
balanced wordline driver, the matched replica wordline, the `(1, 9)`
default, the frozen-periphery principle for the optimiser, and the
qualification structure (hash-keyed table, unverified rule fallback).
