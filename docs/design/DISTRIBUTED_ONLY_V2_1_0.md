# Distributed-only signal wiring — pre-release V2.1.0-labeled record

This is the initial September 12 snapshot, created before the user designated
the change **V2.1.1**. Its measurements, artifact paths and report-time queue
state are preserved below. Use the
[V2.1.1 release report](DISTRIBUTED_ONLY_V2_1_1.md) and
[current evaluation schedule](../../plans/V2_1_1_TIMING_FOLLOWUP.md) for
current implementation and completed large-array outcomes.

The [star-RC screen](STAR_RC_SCREEN_V2_1_0.md) identified both the selectable
star topology and shared junctions remaining in distributed mode. The user's
subsequent instruction removes that topology and makes distributed wiring the
only implementation. The screen itself is preserved as a pre-change record.

## Configuration and removal

`InterconnectConfig()`, `resolve_interconnect(None)`, omitted YAML settings,
the main entrance, CLI and optimizer now resolve distributed wiring. Explicit
`mode: star` is rejected; it is not silently reinterpreted. The default matches
`interconnect_example.yaml`: 1 ohm and 0.1 fF per pitch, with 0.6 µm pitch and
0.1 µm wire width. These values are illustrative and are not foundry extraction.

The compiler no longer contains shared-node array/replica alternatives,
aggregate omitted-cell signal capacitances, the extra star replica row, or
the duplicate lumped RBL sensing branch. Dummy row/column factories also use
distributed taps. Unused top-level dummy insertion helpers and the unmatched
AND2 replica path were removed. Explicit unmatched or noncanonical diagnostics
are rejected by the sizing resolver.

`w_rc` remains the independent switch for local storage/peripheral **series**
RC. `cell_pin_rc` optionally adds series cell-pin sections and defaults to
false, including direct cell constructors. Disabling local stubs leaves all
physical signal-wire ladders present. Supply leakage aggregation is retained;
it is not a signal-wire RC topology.

## Closing the screen's A–E junctions

| Screen item | Current implementation | Geometry assumption |
|---|---|---|
| A: BL/BLB periphery | Three-pitch ladder from each array port; distinct precharge, write and mux/sense taps | Reuse bitline geometry; taps at 0.5, 1.5 and 2.5 pitches, far endpoint at 3 pitches |
| B: replica periphery | Same BL/BLB ladder and peripheral load order as the array | Same three pitches; TIME observes actual replica sense input |
| C: decoder fan-out | Per-bit address trunks plus true/complement/enable ladders in each 3-to-8 block | Trunks span row height; each level partitions that height among its blocks; eight outputs span each block, including unused gates |
| D: write-data clock | DATA_DFF clock pi ladder, one centered tap per column | Wordline pitch across column count |
| E: mux selects | Separate SEL ladders with each group at its own column tap | Wordline pitch across column count |

Decoder predecode outputs feed downstream block enable ladders. Existing
gate logic and binary address polarity are retained, including non-power-of-two
row counts. Address registers remain clustered inside TIME; they do not have
an assumed array-height clock route. This model does not attempt to replace
ordinary shared transistor nodes inside an individual logic gate with wires.

`create_decoder()` now propagates local `w_rc/pi_res/pi_cap` as well as physical
geometry. With `w_rc=True`, this intentionally adds local decoder gate stubs
in addition to the address ladders; fresh timing validation covers both local
stub settings. These routing lengths and placement assumptions must be replaced
or checked against actual layout before an extracted-metal claim.

Peripheral nodes are `BL{c}_periph_tap0/1/2`, `BLB{c}_periph_tap0/1/2`, and
`RBL_periph_tap0/1/2` / `RBLB_periph_tap0/1/2`.
`periphery_tap(pin, col, role)` resolves them; `col=None` denotes the replica.
Without local RC, un-muxed sense input and TIME observe tap 2, not the array
port. With RC they observe the existing sense-amplifier internal input node.

## Sizing, timing and evidence identity

Driver transistor classes and fixed timing budgets are unchanged. A comparison
against the archived incoming explicit-distributed baseline matches **all 64
serialized sizing vectors**, including fingerprints, across four geometries,
6T/10T, mux on/off, local stubs on/off, lookup/rules_only. The geometric clock
table remains 4–9 ns within its anchors. This does not establish that every
new route meets that clock.

Sixteen full circuit decks were also generated from this tree and a detached
V2.0.11 `0433072` worktree, with an explicit 10 ns clock: 8x4/64x4, 6T/10T,
mux on/off and local stubs on/off. Recursively expanded MOS instance/model/
width/length signatures match in all sixteen pairs. Every new deck contains
additional wire resistance; the decks are intentionally different. Artifacts:
`deck-comparison/summary.json` and both source-tree deck directories.

New topology needs fresh waveform evidence even for cases that previously
used distributed array wires. The earlier queue was interrupted and retains
its partial attempt under `outputs/validation/V2.1.0-review-20260912/large-queue/`.
It must not be resumed on the new sources. Its old star and distributed
measurements remain historical and are not promoted to current coverage.

The reviewed local scoring-source manifest was refreshed. New qualification
cases include explicit distributed geometry in their identity, and exports
reject historical absent/star physical contexts. `sizing_table.json` remains
empty; this change makes no qualification or yield claim.

## Verification and evaluation

The [revised evaluation plan](../../plans/V2_1_0_TIMING_FOLLOWUP.md) makes
topology and small write correctness the first gate, before large-array timing
and PVT/mismatch. Resources remain at most eight ranks and one large run at a
time. New artifacts are under `outputs/validation/distributed-only-V2.1.0/`.

Software checks: **112 compiler tests**, **51 local development tests** and
**six offline optimizer tests pass**. Tests cover distinct consumer connections, conserved wire R/C,
default/rejection paths, replica and equivalent modes, decoder truth tables,
6T/10T numeric and sweep decks, and frozen sizing. Connectivity checks examine
actual consumers, not only resistor counts.

Runtime waveforms now expose near/target/far DATA_DFF clocks, register/held
data, decoder-address far endpoints and peripheral taps. The independent
scorer checks input at local clock capture, registered data before write enable,
held data/complement and driver input through the local write window, driven
bitlines, every selected-row Q/QB, unselected-cell/wordline quietness, read
sensing, release before local precharge, and complete final retention/restore.
Missing probes or partial intervals fail. Prior corruption tests were migrated
to distributed fixtures, including late/missing clock capture and changing
held data during write.

All seven fresh full-array waveform cases pass **6,463 checks**, including
6,449 independent waveform checks and 14 aggregate runtime measurement checks.
TT uses 1.0 V / 25 C; SS and SF use 0.9 V / 125 C. Runs used four MPI ranks,
nominal variation, seed 20260913, a 20 ps maximum step and the unmodified lookup
clock. All storage cells were probed in this matrix.

| Case | Period (ns) | Checks | Worst write clock-to-Q (ps) | Minimum local release margin (ps) |
|---|---:|---:|---:|---:|
| 8x4 6T TT write | 4.0 | 209 | 228.77 | 131.85 |
| 8x4 6T SS sequence | 4.0 | 1,231 | 514.94 | 303.61 |
| 8x4 6T SF sequence | 4.0 | 1,231 | 479.96 | 258.48 |
| 8x4 6T TT sequence, local stubs off | 4.0 | 1,231 | 166.22 | 138.35 |
| 8x4 muxed 10T SS sequence | 4.0 | 1,231 | 521.85 | 303.45 |
| 10x6 6T SF write, row 9→1 hazard | 4.0 | 337 | 493.03 | 278.76 |
| 64x4 6T TT write, row 37→5 hazard | 4.5 | 993 | 238.41 | 135.76 |

Clock-to-Q uses 50% crossings. Release margin is local WL falling through 10%
VDD to local precharge falling through 90% VDD. Separate checks require Q/QB
within 10% of expected data at the deadline and throughout retention. Every
sequence writes/reads both polarities twice. A real two-sample per-device 2x2
CLI write also passes with the default distributed configuration and finite
primary metrics; it is not included in the independently scored matrix.

Representative write-capture plots show the new local register clocks, input
capture, held data, driver enable, bitline action, Q/QB and retention:

- [SS 6T write 1 and write 0](../../outputs/validation/distributed-only-V2.1.0/dist_8x4_SS_sequence-write-capture.png).
- [TT 6T with local stubs off](../../outputs/validation/distributed-only-V2.1.0/dist_8x4_TT_nostubs_sequence-write-capture.png).
- [SS muxed 10T](../../outputs/validation/distributed-only-V2.1.0/dist_8x4_10t_mux_SS_sequence-write-capture.png).

The full [machine-readable record](DISTRIBUTED_ONLY_V2_1_0.json) retains case,
source, deck, model and solver identities. Compilation, `git diff --check`
and default deck generation in an isolated tree without `dev/` pass.

After the matrix completed, a reviewed diagnostic-runner fix made rejected
historical configurations and attempted in-place rescoring preserve existing
results and summaries. New runs require a fresh case directory; rejected
invocations create unique failure sidecars. `--score-only` explicitly rejects
unsafe in-place regeneration; read-only scoring functions and archived sources
remain available. Seven added regressions cover these paths and fresh-case
timeout recording. This changes no circuit or waveform scoring; the small
matrix retains its original runner snapshot.

The new large queue launched at **21:17 PDT on September 12** (04:17 UTC on
September 13), supervisor **759485**. It runs the full 8x512 SS write at 8 ns,
then the 64x64 TT sequence at 5 ns, with eight ranks, one case at a time,
six-hour solver limits and a fourteen-hour batch budget. Check the live
`large-queue/checkpoint.json`; these running/pending cases are not passing
evidence. Class/PVT/mismatch expansion remains gated on their review.
