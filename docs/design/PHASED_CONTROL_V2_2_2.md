# V2.2.2: production review of the phased controller

**Completed functional screen. The assembled record is**
[`docs/data/PHASED_CONTROL_V2_2_2.json`](../data/PHASED_CONTROL_V2_2_2.json)**.
It is not yield, extracted-metal or half-select qualification.** The coverage
the screen does not reach is enumerated in
[what the screen does not cover](#what-the-screen-does-not-cover).

V2.2.2 is the production review of the V2.2.1 phased controller
([record](PHASED_CONTROL_V2_2_1.md)): every control gate and observer chain
was re-read against the four operation transitions and the idle boundaries,
the generated decks were re-examined, the waveform checker was extended to
report access and recovery margins, the startup bias that V2.2.1 carried as
open scope was fixed, and the whole screen was re-run from the tracked tree
with additional per-device mismatch cases at the larger arrays and the skewed
corners. The 6T/10T cells, the matched replica column, the distributed wires,
the driver size classes, the clock budgets and the TIME_CONTROL topology are
unchanged; no generated controller netlist differs from V2.2.1.

## What was reviewed and what was found

The controller was reviewed gate by gate in signal order
(`sram_compiler/subcircuits/time_generate.py`, whose module docstring now
carries the measured timing diagram and the causal chain of both operations).
The questions asked of each boundary were: can a delayed path prolong an
access past clock fall, can a stale term re-arm an enable in the next cycle,
and is every forbidden overlap excluded by an observed physical terminal
rather than by path length.

No functional defect was found in the controller. The cases examined, and
why each is safe, are recorded here so that a later change can be checked
against them:

- **Capture into an idle cycle.** When `cs` is captured low, the raw request
  `gated_clk_buf = cs & clk_buf` pulses for the flip-flop's clock-to-Q while
  the old `cs` is still high. The access request is `gated_clk_buf &
  pre_off_ready`, and `pre_off_ready` falls the moment the far precharge
  terminal turns on in the previous recovery, so the pulse cannot reach
  `write_prepare`, the setup chain or the write window. The screen's idle
  cases (`select_every: 2`) cover the access → idle → access boundary.
- **Idle clock-high.** PRE is `!(clk_bar & wordline_off & enables_off)`, so
  an idle clock-high phase leaves the bitlines floating at VDD with the sense
  pass gates open; they are precharged again in the following clock-low.
  The checker's restore and quiet-enable checks cover idle cycles.
- **Write → read and read → write captures.** `we` changes at clock-to-Q
  after capture. `w_en = we & (A | B) & iso_ready` cannot glitch because `A`
  waits for the far PRE release, `B` (the observed wordline) is low after a
  complete recovery and `iso_ready` is low once isolation has released;
  `read_done = rbl_delay & A & !we` is masked the same way, and the checker
  requires `A`, `B`, `iso_ready`, `access_settled`, `read_done` and
  `rbl_delay` low and `enables_off` high at every capture.
- **Read completion is latched by the discharged replica.** RBL cannot be
  precharged during clock-high, so `read_done` stays high until clock fall
  and the read wordline cannot re-arm within its own access. During a
  write the replica write driver discharges RBL too; `!we` masks the term.
- **Forced release.** Clock fall clears `A`, which clears `wordline_request`,
  `wl_en`, `read_done` and `write_prepare` through raw gates; every delayed
  path (setup chain, RC filters, observer chains) only delays assertion.
- **Write drivers cover the wordline tail.** `B = wl_en | !wordline_off` is
  already high through `wl_en` before `A` falls, so the window `A | B` has no
  gap at clock fall, and it closes only after the far replica wordline has
  been observed low and settled.
- **Nested observer subcircuits.** The precharge-off, isolation and
  enable-off guards reuse one Python class with different stage counts. In
  the generated deck each guard's delay chain is defined inside its own
  parent subcircuit (`PRECHARGE_OFF_GUARD`, `ISOLATION_OFF_GUARD`,
  `ENABLE_OFF_GUARD`), so the 2-stage enable chain and the 4-stage chains do
  not collide; this was checked on the generated 8x4 deck, not assumed.

Two production defects were found outside the controller logic:

- **Startup bias.** Generated decks forced every bitline, RBL and the sense
  nodes to 0 V at t = 0 while the controller, with the clock low, held PRE
  active and the sense pass gates open. That operating point does not exist
  in the circuit; Xyce 7.4 could not solve it for the per-device FF 1.1 V /
  −40 °C 10T mux read `8x4_10t_mux_FF_read_pd`, which V2.2.1 had to carry as
  a hand-patched exception. Every deck now starts from the precharged
  clock-low state (bitlines, RBL/RBLB and SA_Q/SA_QB at VDD); the output
  latch presets are unchanged. No measurement window moves, because every
  window already started at or after the first capture. The formerly failing
  case runs from the default deck in this screen.
- **Blind margins.** V2.2.1 reported only ordering margins. The 8x512 read
  that met the runtime `k + 0.7 T` check while OUT was still on the wrong
  rail at the checker's `k + 0.68 T` deadline was caught only because the
  checker failed it; a passing screen gave no number for how close reads
  came. The checker now reports `min_read_output_margin_ps` (the latched
  output and every sense group settled this long before the checker
  deadline; the sense nodes are only followed to the deadline because they
  are precharged again after isolation releases), `min_write_wl_after_flip_ps`
  (the earliest physical wordline endpoint leaves 0.1 VDD this long after the
  last written cell crossed mid-rail), `min_restore_before_capture_ps` (the
  last far bitline or RBL back above 0.9 VDD before the next capture) and
  `min_precharge_on_before_capture_ps`. All four are checked non-negative,
  and a read whose output settles after the deadline fails both the output
  check and the margin check.

Documentation drift was corrected: the controller source and the clock table
cited `PHASED_CONTROL_V2_2_0.md`, a record that never existed under that
name; both now cite this record, and the clock table's `version` is V2.2.2
with its `v2.2.0-timing-3` ladder unchanged.

## Timing diagram

The module docstring of `sram_compiler/subcircuits/time_generate.py` holds
the to-scale diagram of both operations, measured at 0.5 VDD on the 8x4 6T
array at SS, 0.9 V, 125 °C with a 9 ns clock, and the numbered causal chain
from capture through recovery. The order below is the contract; the spacing
is one array at one corner.

Write: capture → far PRE off (observed, settled) → access request →
isolation → far ISO observed → drivers on → far WEN observed → setup chain →
WL → clock fall clears WL → far replica WL observed off → drivers off → far
enables observed off → ISO release and PRE on → restore → next capture.

Read: capture → far PRE off → access request → setup chain → WL → replica
discharge → `read_done` isolates the sense inputs and clears WL → far replica
WL observed off and far ISO observed high → `s_en` → output latch → clock
fall clears `read_done` and `s_en` → far enables observed off → ISO release
and PRE on → restore → next capture.

The following completed 16x8 6T per-device trace at SS, 0.9 V, 125 °C from
this screen changes rows and column data and exercises all four transitions;
enable, ISO and PRE traces are physical terminals at the last column and WL
is the maximum over all probed row endpoints.

![Eight accesses with recovery between captures](PHASED_CONTROL_V2_2_2.svg)

## Screen

The screen re-runs every V2.2.1 case from the tracked tree with the tracked
runner (`python3 -m tests.spice.phased_access`), without the controller
overlay and injected clocks that the V2.2.1 campaign needed, so each trace's
recorded source hashes equal the tracked tree and no deck-reproduction proof
is required. Twelve per-device read → write → read → write cases were added
where V2.2.1 had mismatch only at 16 rows or fewer and one 128-row read:
32x32 (SF, FS with a mux), 64x16 (FF with a mux, SS 10T), 128x8 (SF with a
mux), 256x4 (SS 6T, SS 10T with a mux), 8x128 (FS, SS 10T with a mux), 8x256
(SS with a mux), 512x4 and 8x512 (SS 6T). The 2 ns negative control is run
with the same checker and must fail.

All 254 cases have a complete trace and pass the tracked checker:
2,164,669 checks, no failure, no missing or duplicated case. The screen ran
72 per-device mismatch cases and 182 nominal cases over 12 array sizes at
SS (131 cases), SF (32), FF (49), FS (30) and TT (12), with 24 custom
mixed-row patterns, 8 idle patterns and 32 changed-address reads. The 2 ns
negative control fails 212 of its 1,442 checks under the same checker. No
case needed a numerical retry, and the formerly excepted
`8x4_10t_mux_FF_read_pd` converged from the default deck.

The worst value of every ordering and budget margin over the whole screen is:

| Margin | Worst | Case |
|---|---:|---|
| Sense margin | 0.467 V | `512x4_6t_SS_read_write_pd` |
| Stored polarity | 0.260 V | `128x8_10t_mux_SS_read_changed_pd` |
| WL off before capture | 3817.8 ps | `16x8_6t_SS_mixed_rows_pd` |
| Driver on before WL | 184.1 ps | `2x2_6t_mux_FF_read_write` |
| Isolation before enable | 102.0 ps | `2x2_10t_mux_FF_write` |
| WL off before sense | 98.8 ps | `8x4_10t_mux_FF_read_pd` |
| Driver release after WL | 94.1 ps | `2x2_6t_mux_FF_read_write` |
| Enables off before precharge | 83.2 ps | `2x2_10t_mux_FF_read_write` |
| Read output before the k + 0.68 T deadline | 1131.8 ps | `32x32_6t_SS_read_write` |
| WL dwell after the written cell flips | 2793.8 ps | `32x32_6t_SS_read_write` |
| Bitlines and RBL restored (0.9 VDD) before capture | 2678.6 ps | `32x32_6t_SS_read_write` |
| Precharge on before capture | 2896.6 ps | `32x32_6t_SS_read_write` |

The four smallest ordering margins remain gate-delay limited in the observer
chains at the fast corner of the smallest arrays, as in V2.2.1, and agree
with the V2.2.1 values to within a few picoseconds, which is the expected
effect of the changed startup bias on an access that begins several cycles
later. The tightest budget margins all fall on the nominal 32x32 6T SS
sequence, the upper bound of both the 32-row and the 32-column class at
9 ns: its read output leads the checker deadline by 1.13 ns, so that class
uses about 3.2 ns of its 4.3 ns access budget at SS 0.9 V / 125 °C, and its
precharge turns on 2.9 ns before the next capture. The per-device cases at
larger arrays keep more than 1.4 ns of read-output margin (the smallest is
the 32x32 6T SF sample). A change to the clock table, the observer chains
or the buffer sizes must be checked against these numbers.

Runtime: 174.5 solver-hours (641 core-hours) on Xyce 7.4, the longest case
being `8x512_10t_mux_SS_read_write` at 9.2 h on four ranks; Python 3.9.19,
NumPy 1.26.4, PySpice 1.5. The tracked test suites pass (170 tracked, 64
local development-tool tests, 6 optimizer tests), with `compileall` and
`git diff --check`.

## What the screen does not cover

The screen passes every case it runs. This section records what it does not
run. It was re-derived from the assembled record after the release, so that a
complete screen is not read as a complete qualification, and so that the next
person to extend the ladder knows which number moves first. Every figure below
comes from [`docs/data/PHASED_CONTROL_V2_2_2.json`](../data/PHASED_CONTROL_V2_2_2.json).

**1. No screened array is large in both dimensions.** Every row-class and
every column-class bound of `timing_lookup.json` is screened, but each one only
against a small value of the other dimension: 512 rows only with 4 columns, 512
columns only with 8 rows, and 32x32 is the largest array that sits in a high
class on both axes. Every screened size satisfies `rows * cols <= 4096`, so the
largest macro with waveform evidence is 4 kb. The clock policy is
`T = ceil_to_50ps(2 * max(row_budget, column_budget) * (1 + margin))` — a
maximum, not a sum — so a 128x128 or a 256x256 array is given the same period
as 128x8 or 256x4 while carrying a tall bitline and a wide wordline at once.
The compiler will still emit those decks, and the `max()` rule has no evidence
at any point where both terms are large.

**2. The sense margin falls with row count; 512 rows is the practical end of
the ladder.** The worst bitline differential at the isolation sampling instant,
over the SS cases of all four architectures at each row count, against the fixed
0.25 V bar, with the shared 6T class period for scale:

| Rows | 2-32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| Worst sense margin (V) | 0.904 | 0.857 | 0.799 | 0.669 | 0.467 |
| Shared 6T row-class period (ns) | 9 | 9.5 | 16.5 | 20.25 | 27.75 |

The decrement per row doubling is itself growing (58, 130 and 202 mV over the
last three doublings) while the period triples, so the limiter is the
replica-to-array ratio, not the clock budget: a longer cycle does not buy the
differential back. Extrapolating the last decrement puts a 1024-row class near
0.27 V, and the observed acceleration puts it below the bar. The margin table
above reports 0.467 V as a worst value; it is the trend, not the value, that
bounds the ladder, and a 1024-row class needs sense evidence before a clock.

**3. Process corner is confounded with voltage and temperature.** Each corner
is screened at exactly one operating point: FF and FS only at 1.1 V / -40 C,
SS and SF only at 0.9 V / 125 C, TT only at 1.0 V / 25 C. There is no
fast-process low-voltage point, no slow cold point and no hot fast point. SF,
the write-ability corner, is screened only hot, where writing is easiest; FS,
the read-stability corner, is screened only cold at 1.1 V, where the cell is
strongest. The screen therefore bounds the delay extremes (SS 0.9 V / 125 C
slowest, FF 1.1 V / -40 C fastest) but not the stability extremes.

**4. Corner coverage collapses above 16x8, and the nominal corner is never
screened at a production size.** 131 of 254 cases are SS, and 172 of 254 are at
16x8 or smaller. Above 16x8 there are seven non-SS cases in total: 64x16 FF,
32x32 SF, 32x32 FS, 128x8 SF, 8x128 FS and 8x512 SF twice. TT at 1.0 V / 25 C —
the default of `global.yaml`, and so the condition a user runs first — exists
only at 8x4. Every array size a user would actually build is therefore screened
at essentially one PVT point, and it is not the one the compiler runs by
default.

**5. The screen is weighted towards arrays too small to exercise the wires.**
152 of 254 cases (60 %) are 2x2 or 8x4, and 52 of the 72 per-device cases are
8x4. Local mismatch at 256 rows, 512 rows, 256 columns and 512 columns is one
seeded sample each. At 512x4 that single draw already cost 13 % of the nominal
sense margin (0.540 V nominal, 0.467 V with mismatch); one sample bounds no
tail, and the margin table's worst sense value is such a sample.

**6. The compiler's own runtime acceptance is looser than this checker.** The
generated `.MEASURE` cards check read data at `1 ns + (cycle + 0.7) T`
(`sram_6t_core_MC_testbench.py`), while the screen's checker uses
`k + 0.68 T`. The gap is 0.02 T: 180 ps at a 9 ns clock and 555 ps at 27.75 ns.
The V2.2.1 escape described above — an 8x512 read still on the wrong rail at
0.68 T and correct by 0.7 T — is exactly that gap. V2.2.2 closed the detection
side by reporting `min_read_output_margin_ps`, but the shipped measures still
accept the looser deadline, so a deck that this screen would reject can pass
`main_sram.py`. Tightening the runtime cards changes the acceptance of every
existing flow and is deliberately not done here; until it is, the runtime
`.mt0` verdict is the weaker of the two.

**7. Column data dependence is screened only at 8 columns.** The complementary
column words `10101010` and `01010101` appear only in the 16x8 pattern cases.
Every other case, including every 64-, 128-, 256- and 512-column case, writes
the same value to every column of the selected row. Adjacent-bitline data
dependence and full-width simultaneous switching above 8 columns are unscreened,
and line-to-line coupling is not modelled at all (the wire model folds coupling
to ground), so the two gaps compound.

**8. There is no defined behaviour for a stopped or gated clock.** PRE is
`!(clk_bar & wordline_off & enables_off)`, so precharge is a clock-low-only
function. Holding the clock high parks the array with the bitlines and RBL
floating and the sense pass gates open; holding it low parks it precharged and
is the safe idle state. The screen's idle coverage is 8 cases with a single
idle cycle between accesses (`select_every: 2`), so a multi-cycle stall is
unscreened in either state. The 50 % duty-cycle assumption recorded under
Limits is about phase length; this is about the clock stopping at all.

**9. Cell probing is sparse above 1024 cells.** When `rows * cols > 1024` the
checker probes cells only on `{0, rows // 2, rows - 1, target, next_row}`: at
512x4 that is 4 of 512 rows, 16 of 2048 cells. Every row's wordline endpoints
are still probed, so a spurious wordline on an unprobed row is caught, but a
purely electrical disturb on one is not.

**10. Two reported margins are not the quantities their names suggest.** The
0.25 V sense bar is a fixed voltage rather than a VDD fraction (28 % of the rail
at 0.9 V, 23 % at 1.1 V) and it is not tied to the sense amplifier that has to
resolve the differential: the amplifier's own input offset under per-device
mismatch is never measured anywhere in this screen. `min_storage_polarity_margin_v`
measures distance to 0.5 VDD, which is not the trip point of the 10T cell:
`sram_10t_core.py` is a Schmitt-trigger cell — stacked pull-downs NL1/NL2 and
NR1/NR2 with NFL/NFR feedback — read differentially through the same BL/BLB
access transistors as the 6T, so a larger storage-node excursion at a higher
trip point is the expected behaviour. Reported against a fixed boundary the 10T
therefore looks worse than the 6T (0.260 V against 0.291 V); that ordering is an
artifact of the metric, not a stability deficit.

**11. One negative control bounds only part of the checker.** The single 2 ns
8x4 SS case fails 212 of 1,442 checks, and those do include the data families
(cell retention, logical retention, `output`, sense-group data) and two of the
twelve reported margins (`min_enable_off_before_precharge_ps` and
`min_read_output_margin_ps`). The other ten margins — including every
recovery margin, the sense margin and all four smallest ordering margins — have
never been observed to fail, so their sensitivity is asserted rather than
demonstrated. A related construction detail: the non-negativity checks are
emitted only when their metric is derivable (`if values:`), so a case whose
crossing search finds nothing contributes no check rather than a failure. In
this screen no such case occurred — all 254 records carry every metric their
operation admits — but the construction fails open.

### What was confirmed

The provenance claim holds exactly: the 63 source hashes recorded in the
campaign metadata equal the tracked tree at the release commit, with no file in
one set and not the other, so no deck-reproduction proof is needed. 254 of 254
cases passed, the per-record check counts sum to the reported 2,164,669, and no
record carries a numerical retry or a preserved failed attempt.

The claim that the four smallest ordering margins are bounded by the fast corner
of the smallest arrays is consistent with the data: at every corner those
margins grow monotonically with array size, and the 64x16 FF case added in this
screen continues the trend upwards. One margin is the exception.
`min_driver_release_after_wl_ps` turns around along the column axis: at SS it
rises from 344 ps at 8x4 to 460 ps at 8x128 and then falls to 426 ps at 8x512,
and at SF it is lower at 8x512 (357 ps) than at 32x32 (385 ps). It is the only
reported margin that shrinks as the array widens, so it is the one to watch if
the column ladder is ever extended past 512.

## Limits

What the screen runs, and what it leaves untouched, is enumerated in
[what the screen does not cover](#what-the-screen-does-not-cover) above; the
items below are the boundaries of the whole flow rather than of this campaign.

All evidence uses full cells (equivalent mode 0), the supplied FreePDK45
models and illustrative distributed metal (1 ohm / 0.1 fF per pitch).
Statistical screening cannot establish a zero failure probability; the
per-device cases are single seeded samples per configuration. An untested
size, a custom RC or PVT point, or altered decoder sizing needs its own
waveform validation at the frozen clocks; an array that is in a high row class
and a high column class at the same time is such a size, because the clock
policy takes the maximum of the two budgets and no screened case has both.
Extracted metal, half-selected writes and yield estimation remain open
(`docs/README.md`). The clock contract assumes the 50 % duty cycle of the
stimulus: a shorter high phase shortens the access budget and a shorter low
phase the recovery budget. It also assumes the clock keeps running, because
precharge is a clock-low function: a clock stopped high parks the array with
floating bitlines, and a clock stopped low parks it precharged.
