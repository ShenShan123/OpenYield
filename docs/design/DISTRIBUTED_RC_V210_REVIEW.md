# V2.0.10: SRAM distributed-RC review

Review of V2.0.9 release commit `47bdd1e`, started 2026-09-11; the corrections
are released as V2.0.10. This is a functional
diagnostic with illustrative wire geometry, not a sizing-table or yield
qualification. Historical V2.0.5–V2.0.8 evidence is unchanged.

## Findings and corrections

1. **TIME undercounted sense-control capacitance.** `SenseAmp` has two local RC
   sections on each of EN and ISO, but `resolve_driver_sizes()` counted one.
   Both sections now contribute to the sense-enable and isolation buffer
   loads, including the distributed replica amplifier. At 16 unmuxed sense
   amplifiers and 1-fF sections, the missing load was 16 fF on each signal.
   The lookup classes and continuous transistor-sizing coefficients are
   unchanged. Buffer widths, and occasionally stage counts, change with RC.

2. **Read swing could measure startup instead of the access.** On the full-real
   16×64 10T muxed FF read (125 °C, 1.1 V, 5-ns clock), `TBL` reported a
   2.264-ps initialization crossing while the wordline rose at 4.794 ns;
   `TSWING` consequently reported −4.792 ns. `TWL` and `TBL` now have a window
   confined to the first access. Xyce remeasurement of the same waveform
   gives `TBL = 4.931542 ns` and `TSWING = 137.374 ps`. The actual read passes
   the waveform checks; this was a measurement error.

3. **The result parser removed samples.** Missing measurement files and runs
   with negative `TSA`, `TS_EN` or `TSWING` disappeared from multi-run output.
   Every requested run now retains its index. Invalid measurements become
   missing values; valid zero values are preserved and nonfinite values are
   rejected. Valid measurements in the same sample remain available.

4. **An unsafe wire configuration could return ordinary SRAM metrics.** The
   replica precharge observer switches near mid-rail, so its fixed logic delay
   does not guarantee that a slow distributed wordline has finished falling.
   New `VWL_PRE_FAR_n` and `VWL_PRE_LOCAL_n` measurements sample the far wire and
   selected access pin as PRE falls through 90% VDD, in a separate window for
   every access. `VWL_PRE_PEAK_n` checks the subsequent precharge interval for
   wordline rebound caused by bitline restoration. `run_mc_simulation()` preserves raw measurements and rejects
   any sample with a missing event or wordline magnitude above 10% VDD.
   This prevents unsafe timing results from reaching optimization or yield
   consumers. It does **not** make arbitrary metal geometry electrically safe.

   The 8×512 sequence also exposed an 11-ps release-criterion miss with the
   ordinary 1-Ω / 0.1-fF pitches. Distributed TIME now adds a four-stage
   non-inverting settling delay after the replica observer, ANDed with the
   immediate observer output. The immediate inhibit is retained, and the
   delayed acknowledgment postpones precharge. The fixed count is frozen in
   `DriverSizes.precharge_guard_stages`; star topology retains zero stages.
   This changes internal timing, not the tabulated transistor classes. The
   clock remains the external timing knob; its restore phase must include the
   added delay. Final guarded waveform results are recorded separately below.

   On the first 8×512 write, the final guard changes the release margin from
   −11.16 ps to +105.24 ps and the wordline at precharge onset from 0.118 V to
   0.020 V. The restoration-induced rebound stays below the 10%-VDD limit.

5. **Large write startup left feedback states unspecified.** The write decks
   already initialized `DIN_dff` to zero, but left the write-data hold pair
   and the register slave's complementary node to the DC solver. They now
   start consistently at zero/one and one respectively. On the exact archived
   16×256 TT star/no-local-RC materialized sample 0 of seed 82026, the original short deck
   reproduces the DC failure; adding those states converges. The full original
   2.85-ns-clock write, with the same model cards and only the initialization
   additions, also passes the waveform scorer: retained Q ≥ 0.999865 V,
   restored bitlines ≥ 0.999928 V and release-before-precharge margin 46.58 ps.
   This is a recovery of one previously failing sample, not requalification of
   its three-sample campaign or a claim that the 100-ps YAML access target passes.

6. **Runtime DC failures needed the same numerical recovery as local tools.**
   A seeded 8×16 10T mux control run converges with the old initialization but
   needs Newton line search with the added feedback states. Initialization
   alone therefore does not make every KLU operating point converge. The
   Python and CLI execution paths
   now retry a DC failure once with `SEARCHMETHOD=2`, retaining the original
   deck, log and outputs in `dcop_attempt/`. The native sampling seed is
   preserved; an unseeded run is retried only if its actual seed can be recovered
   from the log. A DC failure reported with exit zero is still failure, and an
   unsuccessful retry remains failure. Electrical check failures are not retried.

The fourth finding is reproduced with a 4×64 6T write/address-change test at
TT, 1 V, 25 °C, 10-ns clock and **30 Ω / 3 fF per pitch**. In the release,
precharge begins 302.3 ps before the selected wordline reaches 10% VDD;
the far wordline is 0.352 V at precharge onset. Correcting the independent
sense-control load issue does not resolve this overlap (305.5 ps, 0.357 V).
The final four-stage guard reduces this stress-case miss to 186.3 ps but does
not eliminate it.
Data and retention pass, but timing fails, so none of these attempts is qualified.
Xyce remeasurement and the runtime rejection path both reproduce the failure.
The failure remains a limit on supported wire/timing combinations, rather than
being hidden by a larger clock or a passing data sample.

## Verification

All 80 compiler regressions pass with Python 3.11 and the project's Python 3.9;
24 local development and six optimizer tests pass. Compiler regressions cover
numeric and sweep paths, both cell types, mux
choices, configured capacitance, preserved MC sample indices, bounded read
measurements, and rejection of unsafe or missing precharge measurements.
Circuit comparisons use a detached worktree at `47bdd1e`: 128 read/write
netlists spanning 16×16 and 8×512, 6T/10T, mux on/off, star/distributed,
RC on/off and lookup/rules_only. The 32 star netlists without local RC have an
identical circuit; their analysis and initialization cards change as described
above (a plain `diff` of those decks shows the read access window and the write
initialization, and nothing else). The other 96 also change the TIME
sense/isolation buffers, the distributed precharge guard, or both. All
tabulated precharge, write, wordline and decoder classes match.

All eight final ordinary-wire cases below pass both the independent waveform
checks and the final Xyce release/peak measurements. They use distributed
1-Ω / 0.1-fF pitches and 100-Ω / 1-fF local RC. The last row deliberately
uses 30× wire R and C and remains failed. All arrays are full transistor models;
nominal variation is explicit except for the last 8×16 TT row, which uses one
5% per-device `vth0/u0/voff` sample with seed `20260911`.

| Array | Cell / mux | PVT | Operation / clock | Minimum release margin (ps) | Minimum sense differential (V) | Result |
|---|---|---|---|---:|---:|---|
| 8×512 | 6T | TT / 1 V / 25 °C | read&write / 5 ns | 105.2 | 0.995 | Pass |
| 64×64 | 6T | TT / 1 V / 25 °C | read&write / 5 ns | 168.7 | 0.974 | Pass |
| 512×4 | 6T | TT / 1 V / 25 °C | read&write / 5 ns | 156.3 | 0.661 | Pass |
| 64×16 | 6T | SF / 0.9 V / 125 °C | write / 8 ns | 317.9 | — | Pass |
| 16×64 | 10T mux | FF / 1.1 V / 125 °C | read / 5 ns | 230.9 | 1.048 | Pass |
| 8×16 | 6T mux | SS / 0.9 V / 125 °C | read&write / 8 ns | 357.4 | 0.886 | Pass |
| 8×16 | 10T mux | SF / 0.9 V / 125 °C | read&write / 8 ns | 311.1 | 0.866 | Pass |
| 8×16 | 10T mux | TT / 1 V / 25 °C | read&write / 5 ns | 166.1 | 0.980 | Pass |
| 4×64 | 6T | TT / 1 V / 25 °C | write / 10 ns | -186.3 | — | Fail (wire stress) |

The large sequences run write 1 / read 1 / write 0 / read 0 once (four
accesses); the 8×16 sequences repeat that pattern twice (eight accesses).
The single read/write cases change to a neighboring address at release.
Checks include selected-row Q/QB, retention, read disturbance, output data,
sense-input differential, near/far unselected wordlines, every far bitline's
restoration/equalization, precharge onset and subsequent wordline rebound.
The 8×512 case passes 5,153 checks; its maximum observed wordline rebound
while precharge is active is 0.08375 V at 1 V VDD.

Storage probes cover every selected-row cell plus near/middle/far unselected
cells on large arrays (518 storage cells at 8×512, 70 at 64×64, 10 at 512×4).
The SS/SF 8×16 runs probe all 128 storage cells. Omitted probes do not omit
transistors. This is finite coverage, not exhaustive state/pattern coverage.

Xyce 7.4 development build `7.4.0-36-gb7bb12d8`, KLU, 2-ps waveform output;
maximum steps are 50 ps for large cases/hazards, 20 ps for SS/SF 8×16 and
10 ps for the seeded 8×16. Large nominal runs use four or eight MPI ranks;
others use one. The SS 8×16 case needs the preserved Newton line-search retry;
the seeded final 8×16 uses line search explicitly. The runtime retry was also
exercised end to end on a same-card 8×16 mismatch failure and passed all eight
accesses. No electrical failure is relabeled as a numerical retry success.

The initial, longer 128×8 / 10×-wire diagnostic exceeded its 600-s execution
limit and is unscored. Three initial large runs were stopped when replaced
by shorter diagnostics, and one redundant 8×512 solver comparison was stopped
only after the identical nominal circuit completed and passed in the primary
run. Their partial artifacts remain. Earlier failed wire-stress attempts and
DC failures also remain; none is promoted to qualification.

Final decks, model hashes, solver binary hash, source hashes, cases, check
counts and metrics are collected in the local
`outputs/validation/V2.0.9-review/V2.0.10-summary.json`. The directory retains
the original review label because the work began against V2.0.9. The solver
binary SHA-256 is `13971579e5902562364d9b5bb53ad23ab19990907b44fe93a887d71cd9b7480e`.

Waveform figures retained locally:
[512-column release before/after](../../outputs/validation/V2.0.9-review/V2.0.10-precharge-guard.png)
and [final 512-row sequence](../../outputs/validation/V2.0.9-review/V2.0.10-512x4-sequence.png),
with PDF versions alongside them. No incomplete or failed run is promoted to
qualification, and `sizing_table.json` remains empty.


The 10-ps versus 50-ps maximum-step comparison on the 16×64 10T FF muxed read
changes local wordline delay by 0.0031 ps, release margin by 0.0735 ps and
sense differential by 0.369 mV. This comparison predates the additional
precharge guard and is a numerical-resolution check, not a universal error
bound. Four-cycle diagnostics use explicit per-access waveform checks;
`TVOUT_PERIOD` (which needs two output rising edges) and release measures for
cycles beyond the shortened trace are inapplicable, rather than qualification
passes. Normal compiler `read&write` decks retain all eight cycles.

## Reproduction and limits

Local case JSON files, original decks, Xyce logs, `.mt0` measures, `.prn`
waveforms, independent `audit.json` results and plots are retained in ignored
`outputs/validation/V2.0.9-review/`. Local runners are
`dev/validate_distributed_rc.py` and `dev/review_score_v209.py`.
The comparison files are under `deck-comparison/` and `deck-comparison-write/`;
`remeasure/` retains the corrected swing measurement, the overlap rejection
and a small Xyce test proving that a missing event cannot borrow a later cycle.

Distributed wire values are illustrative, not extracted metal. Coupling is
absent. Large cases instantiate every transistor but probe every selected-row
cell and a stated subset of unselected storage nodes, plus every wordline and
far bitline. A single seeded mismatch sample is not a yield estimate. Column
half-select qualification remains open because the compiler writes all columns
of the selected row. Passing cases establish only the stated operating points.
