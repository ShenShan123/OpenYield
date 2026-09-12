# V2.0.11: audit of V2.0.10, distributed control lines, star-topology screen

Audit of the V2.0.10 release commit `9773f29`, started 2026-09-12; the
corrections are released as V2.0.11. This is a functional diagnostic with
illustrative wire geometry, not a sizing-table or yield qualification.
Historical V2.0.5–V2.0.10 evidence is unchanged, `sizing_table.json` stays
empty and no record is promoted.

No driver size class changes, and every star-topology deck is byte-identical
to V2.0.10 (24 of 24 compared across 8x8 and 16x16, 6T, mux on/off, RC on/off,
read/write/read&write). The lookup classes, the `rules_only` derivation, the
TIME sense/isolation load accounting and the distributed precharge guard are
exactly as released in V2.0.10. Distributed decks do change: the control lines
that span an array dimension are now modelled as wires (finding 6 below).

## Findings and corrections

1. **A failed operating point could lend its samples to the retry.**
   `execute_xyce()` copied the first attempt into `dcop_attempt/` but left the
   originals next to the deck. Parsers index `.mt`/`.ms` files by sample
   number, so a retry that wrote fewer per-sample files than the failed attempt
   would report the failed attempt's value for the missing indices — the
   corrupt post-failure state the retry exists to discard. The solver outputs
   are now deleted after they are copied. Caller-owned files beside the deck
   (for example `*.variation.json`) are not solver outputs and are kept.

2. **A rejected distributed release lost its provenance.** `run.py` raised the
   release rejection before writing `summary.json`, discarding the deck path,
   seed, model hashes, variation summary and driver sizes of exactly the sample
   that has to be investigated. The summary — carrying
   `precharge_release_checked: false` and the rejection text — and the waveform
   are now written first; the run then fails as before. The rejection is still
   a failure, and `.data.csv` is still written before the check.

3. **The last sequence cycle's precharge window claimed unsimulated time.**
   `read&write` stops at 1 ns + 8.5 `t_period`, but cycle 7's
   `VWL_PRE_FAR_7`/`LOCAL_7`/`PEAK_7` window ran to 1 ns + 8.7 `t_period`. The
   rebound check therefore reported an interval 0.2 `t_period` longer than the
   analysis produced. `add_analysis()` and `_add_precharge_safety_measures()`
   now share `_analysis_stop()`, and every window is clamped to it. Cycle 7's
   precharge interval is still only partly simulated; the window no longer
   claims otherwise.

4. **Measurement parsing contradicted its own contract.** A value below the
   numeric floor removed the measurement from that run instead of marking it
   missing, unlike every other reject path. It is now a missing value.
   `generate_mc_statistics()` additionally rejects a frame in which no run
   produced any measurement: since V2.0.10 keeps one row per requested run,
   such a frame is a simulation failure, and describing it published a
   statistics file made entirely of NaN.

5. **Not a defect: the precharge safety window's lead-in.** The window opens
   0.04 `t_period` before the clock edge that starts precharge, inside the
   access phase. PRE cannot fall before that edge, so the lead-in only guards
   edge placement. Measured on retained decks, V(PRE) in that interval is flat:
   0.89999 V on the 1024-column SS write (the largest write-driver count in the
   campaign), 0.89975–0.89997 V across all eight cycles of the 1024x4 SS
   read&write, and exactly 1.00000 V on the distributed guarded 8x512 sequence
   and on the failing 30x-wire 4x64 stress case. The largest PRE disturbance
   found anywhere is 45.1 mV (5.0% of VDD) at 6.620 ns of the 8x1024 SS write,
   2.73 ns before that deck's window would open: the write drivers switch
   immediately after the precharge release, at the start of the access, while
   the window is at its end with every bitline static. The lead-in is retained.

6. **Control lines that span the array were lumped star nets.** Distributed
   wiring modelled WL, BL and BLB as tapped pi ladders but left every global
   control net as a single node, so a 512-column `PRE` or `s_en` reached the
   first and the last column at the same instant — the one thing the
   distributed model exists to disprove. Six nets span an array dimension and
   drive one load per row or column; each is now a tapped line whose consumers
   connect at their own position:

   | Net | Direction | Pitch | Consumers |
   |---|---|---|---|
   | `PRE` | columns | `wl` | precharge cell per column, replica at `_far` |
   | `w_en` | columns | `wl` | write driver per column |
   | `w_en_bar` | columns | `wl` | write-data hold latch per column (write decks) |
   | `s_en` | columns | `wl` | sense amplifier per mux group, replica at `_far` |
   | `sa_iso` | columns | `wl` | as `s_en` |
   | `wl_en` | rows | `bl` | wordline driver per row, replica driver at `_far` |

   The driver output keeps the original net name as the line's near end, so
   `.MEASURE` and `.PRINT` cards that observe `V(PRE)`, `V(s_en)` or `V(wl_en)`
   still measure the driver. `control_tap()` returns the plain net in star
   topology, and star decks are byte-identical to V2.0.10.

   Wire RC on these lines is deliberately **not** added to `DriverLoads`,
   exactly as WL/BL wire RC is not: the size classes stay independent of the
   interconnect, and the wire appears as delay and skew rather than as a bigger
   buffer. Not modelled, with reasons: `SEL{i}`/`SELB{i}` also span the width
   but are ideal static stimulus sources in the testbench, so a ladder on them
   would add nodes and no timing; the decoder address lines `A_dff{i}` span the
   decoder height, but the decoder is a block without a per-row physical model
   here.

## Write-operation screen

V2.0.10 changed the TIME sense-enable and isolation buffers on **every**
`w_rc=True` array, star topology included (`rc_sa_units` depends on `w_rc`, not
on the interconnect mode), but all nine of its waveform cases were distributed.
A 16x16 6T unmuxed RC array moves `sen_load` 47.5 to 79.5 and `iso_load` 96 to
128; at 16x256 it is 707.5 to 1219.5 and 1536 to 2048. This release screens the
star topology directly, with writes as the primary operation and reads included
because the sense path is what those buffers drive.

Checks are independent of the `.MEASURE` cards: the scorer reads the `.prn`
trace and verifies, per case, the target cell's Q and QB after the access, data
retained after wordline release, every unselected wordline quiet, every probed
unselected cell undisturbed, every bitline restored to 0.98 VDD before the next
access, precharge onset after wordline release, and for writes two checks added
here — **precharge held off for the whole write-enable window**, and the driven
bitline actually pulled below 0.1 VDD by the write driver rather than by its
initial condition. Reads additionally check the sense-input differential and
the output datum.

Local runner: `dev/validate_distributed_rc.py`, extended in this release with
an `interconnect: star` case option and the two write checks. Results, decks,
Xyce logs and `result.json` files are retained in ignored
`outputs/validation/V2.0.11-write/`; `V2.0.11-summary.json` there collects the
case list, check counts, metrics, deck and model hashes, the solver binary hash
and the runtime source hashes.

56 screen cases (50 star, 6 distributed) and 9 re-run V2.0.10 cases ran; 58 of 65 pass, over 23,815 independent waveform checks.

| Group | Cases | Pass | Waveform checks |
|---|---:|---:|---:|
| Star topology (write and read) | 50 | 44 | 18,714 |
| Distributed wires (write) | 6 | 6 | 1,808 |
| V2.0.10 cases re-run with distributed control lines | 9 | 8 | 3,293 |

Across every write case, V(PRE) over the whole write-enable window stays at 0.9705–0.9989 of VDD, so precharge is fully off for the entire write, and every driven bitline is pulled to or below 0 V during the write (worst case -0.0029 of VDD, against a 0.1 limit) by the write driver rather than by its initial condition.
Read cases resolve a sense-input differential of 0.73–1.00 V against the 0.3 V limit.

The 9 V2.0.10 cases re-run with distributed control lines keep their V2.0.10 verdicts, including the deliberate 30x-wire failure:

| Case | Array | V2.0.10 margin (ps) | V2.0.11 margin (ps) | V2.0.10 sense (V) | V2.0.11 sense (V) | Verdict |
|---|---|---:|---:|---:|---:|---|
| `16x64_10t_mux_read_hazard` | 16x64 | 230.9 | — | 1.048 | 1.047 | pass |
| `4x64_wire30x` | 4x64 | -186.3 | — | — | — | fail (expected) |
| `512x4_tt_sequence` | 512x4 | 156.3 | 151.6 | 0.661 | 0.660 | pass |
| `64x16_6t_sf_write_hazard` | 64x16 | 317.9 | — | — | — | pass |
| `64x64_tt_sequence` | 64x64 | 168.7 | 167.7 | 0.974 | 0.974 | pass |
| `8x16_10t_mux_local` | 8x16 | 166.1 | 165.9 | 0.980 | 0.981 | pass |
| `8x16_SRAM_10T_CELL_mux_SF` | 8x16 | 311.1 | 311.5 | 0.866 | 0.866 | pass |
| `8x16_SRAM_6T_CELL_mux_SS__retry` | 8x16 | 357.4 | 357.3 | 0.886 | 0.886 | pass |
| `8x512_tt_sequence` | 8x512 | 105.2 | 100.9 | 0.995 | 0.997 | pass |

The cost of the control-line wire tracks the dimension it spans, which is the behaviour the model is meant to produce: 4.7 ps of release margin at 512 rows of `wl_en` (512x4), 1.0 ps at 64 columns (64x64), and under 0.5 ps at 16 columns (the three 8x16 cases), with sense differentials unchanged to within 1 mV everywhere. A lumped node would have shown none of it.

A dash in the V2.0.11 margin column is a single-access case, which records the release as a pass/fail check rather than a per-cycle margin. The two scorers also measure the margin differently (the V2.0.10 column comes from `dev/review_score_v209.py`, the V2.0.11 column from the per-cycle metric of `dev/validate_distributed_rc.py`), so compare verdicts and trends rather than the last digit.

Failing cases, all of them expected and discussed under Open items:

| Case | Array | Wires | PVT | Failed check |
|---|---|---|---|---|
| `b_16x4_rc1` | 16x4 | star | TT / 1.0 V / 25 C | precharge_after_release |
| `b_32x4_rc1` | 32x4 | star | TT / 1.0 V / 25 C | precharge_after_release |
| `b_4x4_rc1` | 4x4 | star | TT / 1.0 V / 25 C | precharge_after_release |
| `b_8x4_rc1` | 8x4 | star | TT / 1.0 V / 25 C | precharge_after_release |
| `v211_4x64_wire30x` | 4x64 | distributed | TT | precharge_after_release |
| `w_star_rc_8x4_6t_SF` | 8x4 | star | SF / 0.9 V / 125 C | precharge_after_release |
| `w_star_rc_8x4_6t_TT` | 8x4 | star | TT / 1.0 V / 25 C | precharge_after_release |

## Open items

- **Star + local RC precharge release on narrow arrays.** With the default
  100 ohm / 1 fF stubs, a four-column star array asserts precharge before the
  selected wordline has fallen to 10% VDD. The miss is a property of the column
  count, not the row count, and it disappears without local RC:

  | Array | `w_rc` | Release margin | V(WL) at PRE 90% |
  |---|---|---:|---:|
  | 4x4 | on | -3.3 ps | 0.1136 V |
  | 8x4 | on | -3.4 ps | 0.1136 V |
  | 16x4 | on | -3.4 ps | 0.1136 V |
  | 32x4 | on | -3.6 ps | 0.1147 V |
  | 8x8 | on | +6.9 ps | 0.0737 V |
  | 16x16 | on | +23.2 ps | 0.0343 V |
  | 4x4 | off | +17.4 ps | -0.0046 V |
  | 8x4 | off | +14.5 ps | -0.0027 V |

  At SF/0.9 V/125 C the 8x4 margin widens to -15.9 ps. The cause is the
  opposite of the 512-column case that motivated the V2.0.10 guard: the fewer
  the columns, the lighter `pre_load`, the faster the PRE buffer, and the more
  easily it beats the wordline's RC tail, while the star topology carries
  `precharge_guard_stages = 0`. **This is pre-existing, not a V2.0.10 or
  V2.0.11 regression** — it reproduces on the V2.0.9 tree (`47bdd1e`) with the
  same deck and solver. Every data, retention, disturbance and restore check of
  those cases passes; only the release criterion misses. It is not caught
  automatically because `VWL_PRE_*` is emitted for distributed arrays only.
  A fix belongs with a timing change, not a fix release: either a skewed
  low-trip-point replica observer, which tracks the wordline RC that a fixed
  delay chain cannot, or extending `precharge_guard_stages` to RC star arrays.
  Both change every RC deck's TIME block and need their own screen.
- **The distributed guard is a fixed delay.** `precharge_guard_stages = 4`
  regardless of array size, wire R/C or corner, while the residual wordline
  tail it covers scales with RC. The V2.0.10 30 ohm / 3 fF stress case still
  misses by 186.3 ps, and a small distributed array pays roughly four fanout-5
  inverter delays out of its restore phase for nothing.
- **Derived clocks predate the guard.** `TIMING_AUTOCONFIG_data.csv` and
  `sizing/timing.py` records were derived from star decks without the settling
  delay. A distributed array reusing one gets a shorter effective restore
  phase than the derivation assumed; no distributed timing record exists.
- **`yield_estimation/` cannot consume the compiler's return value.** All 15
  `run_mc_simulation()` call sites in `model_lib/{AIS,ACS,MC,MNIS,HSCS}.py`
  unpack two values from the four-value result, so every one raises
  `ValueError` before the new release rejection can reach a yield estimate.
  This is pre-existing and systemic; it is reported rather than half-fixed
  here because the package has no regression coverage to validate a change.
- **Narrow-tall, wide-column distributed write decks can fail the operating
  point.** An 8x32 6T distributed write deck fails `DC Operating Point` under
  KLU and still fails after the Newton line-search retry. The identical deck
  built from the V2.0.10 tree (`9773f29`) fails the same way with the same exit
  code and the same retry, so this is **pre-existing and not caused by the
  control lines**. It is the shape V2.0.9 already flagged — few rows give a
  light write-driver class that is then spread across many columns — and it
  sits in a corner none of the passing cases covers (8x16, 32x32 and 8x512 all
  converge). The deck-level fix V2.0.9 proposed, initial conditions on the
  column latches, is still the open item; V2.0.11 initialises the hold pair and
  register slave but evidently not enough for this shape.
- **No direct control-line skew measurement.** The near/far skew of the newly
  tapped lines is demonstrated only indirectly, through release margins that
  scale with the spanned dimension (see the re-run table). Dedicated probes at
  8x32, 8x64 and 8x128 were abandoned because they land on the operating-point
  shape above; their partial artifacts were removed rather than retained.
- Half-select column qualification remains open: the compiler writes all
  columns of the selected row.

## Reproduction and limits

```
python dev/validate_distributed_rc.py --xyce "$(command -v Xyce)" \
    --cases <cases.json> --output outputs/validation/V2.0.11-write
```

`dev/` and `outputs/` are untracked, as for every earlier review: the runner and
the retained decks, logs, `.prn` traces and `result.json` files live only in the
working tree that produced them. A case is a JSON object with `name`, `cell`,
`rows`, `cols`, `operation`, `corner`, `vdd`, `temperature`, `period`,
`max_step`, `w_rc` and — new in this release — `interconnect: star`.

Xyce 7.4 development build `7.4.0-36-gb7bb12d8`, KLU, nominal variation (no
mismatch), 2-ps waveform output. The screen cases run on one rank with a 20-ps
maximum step; the re-run V2.0.10 cases keep their original ranks (one, four or
eight) and step settings so they stay comparable to their baselines. The solver
binary SHA-256 is
`13971579e5902562364d9b5bb53ad23ab19990907b44fe93a887d71cd9b7480e`, the same
binary as the V2.0.10 review.

Nominal variation is not a yield statement, and a passing operating point is
not a timing qualification. Distributed wire values are illustrative, not
extracted metal, and coupling is absent. Arrays instantiate every transistor;
storage probes cover the selected row plus near/middle/far unselected cells on
the larger arrays. The screen establishes only the stated operating points.
