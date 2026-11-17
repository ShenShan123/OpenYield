# V2.2.9 overlapped-recovery TIME_CONTROL (cycle ≈ access time)

**Status (2026-11-14): default controller for compact 2:1-mux 6T layouts.**
Every joint timing class from 32×32 to 512×256 has a post-layout clock measured
on an extracted macro (FineSim `spicexd`, nominal devices, finite corner
screen). `sizing.recovery_overlap: false` keeps the V2.2.9
recovery-before-capture controller. The result is a functional/timing screen
of the stated scope, not foundry signoff or yield qualification.

The user directed that the compact 6T clock follow the actual access time: an
access completes in one cycle and the clock's low phase is no longer spent idle
on recovery. The V2.2.9 self-timed controller completes an access independently
of clock fall, but its clock budget still requires the bitlines, enables and
every completion timer to be recovered **before the next capture**. The
overlapped-recovery controller lets the next capture, decode and setup proceed
while the previous access recovers, and admits the next access to the array
only once it has recovered.

Reads still limit the cycle: a read restores its bitlines only after sensing,
so the clock is 1.3–1.6× the read deadline (V2.2.9: 1.7–2.0×). Against the
V2.2.9 table the clock is 4% shorter at 32×32, 26% at 64×64, 37% at 128×128,
37% at 256×256 and 30% at 512×256.

## Architecture

| Signal | Role |
|---|---|
| `access_phase` | Register captured on rising `clk_buf`, cleared by `access_done`. Decoder setup (`access_settled`) starts from it, so it overlaps the previous access's recovery. |
| `recovered` | NOR set/reset latch. Reset by the `access_done` pulse; set by `recovery_ready`. |
| `recovery_ready` | `restore_ready & pre_on_settled & iso_released`: the replica bitline back high (`rbl_delay` low) for `restore_stages`, far PRE on for the PRE-off guard's stages, and far ISO released. |
| `access_admit` | `access_phase & recovered`. PRE release, write isolation and the request start here. |
| `access_request` | The PRE-off guard of `access_admit` (far PRE observed off). It falls when PRE turns on again mid-access, so state that must outlive it is latched. |
| `read_done`, `read_end`, `write_done`, `read_fired_held` | NOR latches. `read_done` (replica trigger) stops the read WL and asserts ISO; `read_end` (far `s_en` resolution timer) ends `s_en`; `write_done` stops the write WL; `read_fired_held` keeps the far `s_en` firing until completion. They clear with the admitted phase. |
| `PRE` | On when `!(access_phase & recovered & !write_done) & pre_ready & sense_released`, with `sense_released = !read_done & !sen_far`. A write's PRE turns on once its WL is physically off; a read's once far `s_en` is off. PRE turns off at the next admission. |
| `W_EN` | `(we & request & pre_off_ready) | (write_tail & iso_ready)`: drive from far PRE off, then through the observed WL tail. |
| `access_done` | `(read_complete | write_complete) & access_admit`. A read completes from the held far `s_en` firing once its WL is off; a write once `write_done` and its WL are off. |

Validation in `TIME_CONTROL`: overlap requires compact self-timed control;
`restore_stages` is a positive even integer; the isolation guard may not be
longer than the PRE-off guard; the write-drive guard may not exceed twice the
PRE-off guard.

**Read-completion latch.** The first 256×256 FF 1.1 V −40 °C sequence hung on
both reads: `read_complete` was AND(read_fired, pre_ready, we_bar), and the far
`s_en` pulse (345 ps) ended before the 44-stage WL-off guard released. The
fired state is now latched. The shipped V2.2.9 controller had the same gate on
`read_resolved`; it carries the equivalent `read_held` latch.

## Hazards and what holds them

* **Lost or merged accesses.** The completion pulse ends before the next capture
  can be accepted; the scorer reports a capture during an unfinished access.
* **Address, data or request change during the previous access.** Both
  wordlines are physically off at completion, and write drive lives inside the
  admitted request.
* **Guards reused before reset.** The next admission waits for the PRE-off
  guard's settled-on node; the isolation and write-drive guards are bounded by
  the PRE-off settling.
* **Sensing during restore.** PRE never overlaps an on WL or `s_en`
  (`sense_released`). An earlier netlist turned PRE on mid-sense; at 128×128 SS
  the precharge current compressed the sense amplifiers' supply by 155 mV and
  read completion margin fell to −115 ps.
* **Restore before the next access — held by the clock from 256 rows.**
  `recovered` follows the replica bitline. Through 128 rows it implies restored
  data bitlines (stalled read streams pass every restore-level check). At 256
  rows a stalled admission found the far bitlines at 0.952–0.959 VDD: the
  replica detector (`rbl_delay` plus 16 stages) fires before the RC tail ends.
  Reads still sense correctly and the 0.95 VDD contract holds there
  (0.002–0.009 VDD above it); the clock budget adds the measured restore to
  0.96 VDD (below). Longer `restore_stages` for tall classes are an open item.

## Waveform contract

`tests/spice/phased_waveforms.py` scores every trace twice.

* **Strict pass** (0.1 / 0.9 VDD), unchanged. Its verdict is kept under
  `strict`, and its metrics (margins, crossings) are the published metrics.
* **Contract pass** (the verdict). Control-signal ordering, settlement and pulse
  counts use the device-off level `CONTROL_OFF` = 0.2 VDD (0.8 VDD for
  PMOS-gated controls). At FF 1.1 V 125 °C a 180 nm/50 nm device conducts 1% of
  its on-current at 0.20 VDD overdrive (NMOS) and 0.18 VDD (PMOS); the strict
  0.1 VDD level is 0.1%. Small overlaps and unsettled tails on slow edges
  therefore pass. Examples: ISO releasing while a far `W_EN` tail is still at
  0.11 V; a 0.107 VDD coupling bump on a far ISO tap; PRE rippling about
  0.9 VDD.
* **Strict in both passes:** unselected wordlines (0.1 VDD), `W_EN`/`s_en`
  exclusion, write-completion WL edges, cell retention, OUT and sense data,
  and bitline restore (far bitlines within 0.05 VDD at the next admission;
  replica reset 0.95 VDD).
* **Edge hysteresis.** A repeat crossing counts only after the waveform returns
  0.05 VDD past the level, so a tail rippling a few mV about a threshold is one
  edge. Example: `access_phase` clearing at 512×64 SS (0.091 → 0.122 VDD).
* **Restore level.** The precharged far-bitline plateau at 125 °C is
  leakage-limited to 0.980–0.985 VDD at 256 rows and 0.964–0.970 VDD at 512
  rows, and a coupling dip of about 4 mV lasting about 100 ps reaches the next
  admission (512×256 FF 125 °C mixed: 0.9599 VDD there, 0.9640 VDD 0.2 ns
  before). The precharge is kept as is, and restore is checked at 0.95 VDD
  (user decisions; the cycle budget keeps the 0.96 VDD crossing).

This follows the user's direction (2026-11-13): *"loose some guards that are
not critical, which we allow to small overlap or unstable settlement. We hope
the TIME_CONTROL design should be as simple as possible."* No controller logic
was added for the contract change; only per-class stage counts and sizing.

Negative controls still fail under the contract: overlap 32×256 with the
square-anchor sizing (unselected WL glitch), the 512×256 controller with a
22-stage enable-off guard (ISO release 25 ps before far `W_EN` off at
FF −40 °C) and overlap 64×256 / 128×256 with row-class sizing (ISO release
170 / 95 ps early). V2.2.9 traces simulated before the
`read_held` latch (and the ov11 overlap traces) lack that probe and keep their
original scorer verdicts.

## Sizing references

`layout_sizing_lookup.json` carries `overlapped_recovery_references`; an array
takes the smallest covering class:

| Class | Read / write completion | Restore | WL-off / enable-off guards |
|---|---|---|---|
| 32×32 | 12 / 4 | 8 | 2 / 2 |
| 64×64 | 12 / 4 | 10 | 4 / 4 |
| 128×128 | 12 / 4 | 12 | 8 / 8 |
| 256×256 | 12 / 30 | 16 | 22 / 22 |
| 512×32, 512×64, 512×128 | 12 / 30 | 16 | 22 / 22 |
| 512×256 | 12 / 30 | 16 | 22 / **30** |
| 512×512 (outside the envelope) | 12 / 30 | 16 | 32 / 32 |

The explicit 512×32/64/128 entries keep those classes at their former
(512×256-derived) values. The 512×256 enable-off guard grew from 22 to 30
stages for its ISO-release margin (above); it also covers 264×132, 384×192 and
504×252.

Peripheral sizing (`self_timed_joint_sizing`) is shared with the V2.2.9
controller. 64×256 and 128×256 now use the 256×256 square's PRE2,
write-driver input 1 / output 4 and two-unit delay drivers. With the row-class
PRE4 / 2 / 8 their 256-column PRE and `W_EN` edges were slow enough that ISO
released 95–170 ps before far `W_EN` was off at FF −40 °C. The override also
shortens read occupancy (20% in a 32×256 test). Short wide classes keep row-class sizes:
applied to 32×256, the faster delay chain let unselected wordlines glitch with
only two decoder setup stages.

## Layout

The overlap controller has more latches and settled guards than V2.2.9. Its
whole feedback graph stays in the band below the array through 512 rows (lane
spill did not route). Band cells keep 80 nm spacing through 256 rows (130 nm at
16 rows, 30 nm at 512 rows). The frame is at least 24.5 µm wide: narrower
16–32-row frames (22.2–23.6 µm) needed 400–500 rip-ups and did not finish
within 90 min, while at 24.5 µm they route in under 60. If a frame candidate
fails to route, generation tries the next candidate by estimated area. A
256×64 frame failed its first two candidates on one `clk_buf` terminal and
routed in the third.

The size sweep ([LAYOUT_SIZE_SWEEP_V2_2_9.json](../../data/LAYOUT_SIZE_SWEEP_V2_2_9.json))
covers 47 sizes per controller: rows 16–512, columns 4–512, every power-of-two
class and off-grid sizes. All 94 complete macros pass DRC 0 / LVS CORRECT with
R+C+CC extraction in their first routing attempt (at most 90 rip-ups). Square
anchors:

| Array | V2.2.9 area (µm²) | Overlap area (µm²) | Change |
|---|---:|---:|---:|
| 32×32 | 2,216.93 | 2,295.25 | +3.5% |
| 64×64 | 5,105.64 | 5,081.58 | −0.5% |
| 128×128 | 13,782.03 | 14,157.62 | +2.7% |
| 256×256 | 43,857.04 | 44,630.51 | +1.8% |
| 512×256 | 81,750.6 | 88,502.2 | +8.3% |

The V2.2.9 areas are the shipped macros. Both controllers' current macros also
carry the array VDD strap and, for V2.2.9, the `read_held` latch.

## Simulation method

FineSim V-2023.12-SP1 `spicexd` (8 threads, 20 ps maximum step) runs the
translated extracted decks. On the 32×32 and 128×128 SS cases it reproduces
every HSPICE verdict and check count, with timing within 10 ps and sense
margin within 0.2 mV
([FINESIM_VALIDATION_V2_2_9.json](../../data/FINESIM_VALIDATION_V2_2_9.json)).
The reduction is 20 Ω / 1 Ω / 10 aF, with every supply pin fed. The schematic
`.IC` operating point is solved by HSPICE. Each case's `metadata.json` names its
solver, model, lookup and RC identities.

## Clock policy

An access is admitted only after the previous one has recovered. If the clock
is shorter than occupancy plus the recovered → admit delay, admission falls
behind the capture by the difference on every access. Four-access patterns
hide that drift; a read stream (two writes, six back-to-back reads) exposes it.

* **cycle budget** = the larger of
  * the longest SS 0.9 V 125 °C read occupancy (admission → `recovered`) plus
    the recovered → admit delay measured in a stalled read stream, and
  * the restore-limited spacing: admission to the last far bitline (and RBL)
    back at 0.96 VDD, at the worst measured corner (SS, FS 125 °C or FF 125 °C).
* **access budget** = the worst CLK → OUT at that spacing.
* **one 10% margin**, rounded up to 50 ps, as in V2.2.9. The grid may not
  decrease with either dimension: 64×256 (13.05 ns measured) and 128×256
  (14.00 ns) take 32×256's budget, and each class's confirmation ran at its
  published clock.

Restore binds at 256×256 and at 512×128 / 512×256. At 512×256 it binds at
FS 125 °C (slow precharge PMOS). The other classes are recovered-limited.

| Class | SS read occupancy | Stalled recovered → admit | Restore limit | Binds | Cycle / access budget | Clock / read deadline | V2.2.9 clock / deadline |
|---|---:|---:|---:|---|---|---|---|
| 32×32 | 5.73 | 0.23 | 4.81 | recovered | 5.96 / 3.75 | **6.60 / 4.15** | 6.90 / 4.10 (postlayout) |
| 32×64 | 6.48 | 0.22 | 5.51 | recovered | 6.70 / 4.33 | **7.40 / 4.80** | 10.60 / 5.50 (extrapolated) |
| 32×128 | 7.66 | 0.21 | 6.70 | recovered | 7.87 / 5.21 | **8.70 / 5.75** | 15.45 / 7.40 (extrapolated) |
| 32×256 | 13.04 | 0.33 | 12.29 | recovered | 13.36 / 9.02 | **14.75 / 9.95** | 26.20 / 12.10 (extrapolated) |
| 64×32 | 6.23 | 0.25 | 5.35 | recovered | 6.48 / 4.28 | **7.15 / 4.75** | 9.55 / 5.55 (extrapolated) |
| 64×64 | 6.74 | 0.33 | 5.94 | recovered | 7.07 / 4.53 | **7.80 / 5.00** | 10.60 / 6.00 (extrapolated) |
| 64×128 | 8.56 | 0.24 | 7.73 | recovered | 8.80 / 6.01 | **9.70 / 6.65** | 15.50 / 7.80 (extrapolated) |
| 64×256 | 11.49 | 0.34 | 10.89 | recovered | 11.84 / 8.72 | **14.75 / 9.95** (own 13.05 / 9.60) | 26.25 / 12.50 (extrapolated) |
| 128×32 | 7.77 | 0.28 | 6.93 | recovered | 8.05 / 5.61 | **8.90 / 6.20** | 12.75 / 7.85 (extrapolated) |
| 128×64 | 8.09 | 0.26 | 7.37 | recovered | 8.34 / 5.84 | **9.20 / 6.45** | 13.10 / 8.15 (extrapolated) |
| 128×128 | 8.90 | 0.23 | 8.50 | recovered | 9.13 / 6.46 | **10.05 / 7.15** | 15.85 / 9.35 (extrapolated) |
| 128×256 | 12.44 | 0.24 | 12.14 | recovered | 12.69 / 9.32 | **14.75 / 10.30** (own 14.00 / 10.30) | 26.40 / 13.15 (extrapolated) |
| 256×32 | 9.84 | 0.27 | 8.90 | recovered | 10.11 / 7.12 | **11.15 / 7.85** | 16.65 / 10.55 (extrapolated) |
| 256×64 | 10.40 | 0.31 | 9.66 | recovered | 10.71 / 7.44 | **11.80 / 8.20** | 16.90 / 10.80 (extrapolated) |
| 256×128 | 11.54 | 0.27 | 11.13 | recovered | 11.81 / 8.33 | **13.00 / 9.20** | 18.75 / 11.70 (extrapolated) |
| 256×256 | 14.89 | 0.27 | 15.34 | restore | 15.34 / 11.02 | **16.90 / 12.15** | 27.00 / 14.80 (extrapolated) |
| 512×32 | 13.05 | 0.29 | 12.63 | recovered | 13.34 / 10.11 | **14.70 / 11.15** | 16.65 / 11.05 (postlayout) |
| 512×64 | 14.26 | 0.33 | 14.07 | recovered | 14.58 / 10.61 | **16.05 / 11.70** | 17.25 / 11.55 (postlayout) |
| 512×128 | 16.96 | — | 17.67 (FS 125 °C 16.47) | restore | 17.69 / 12.20 | **19.50 / 13.45** | 24.30 / 15.10 (postlayout) |
| 512×256 | 20.66 | — | 22.77 (FS 125 °C 22.91) | restore (FS 125 °C) | 22.91 / 14.69 | **25.20 / 16.20** | 35.95 / 17.75 (postlayout) |

Times in ns. The clock column is the published table entry; 64×256 and
128×256 take 32×256's 13.36 / 9.02 ns budget where it is larger. 512×128
and 512×256 have no stalled run: restore binds there by more than 0.5 ns.

## Evidence

Each class has the following runs:

* a relaxed round 1: SS seq, SS read stream, FF/FS 125 °C and FF −40 °C seq,
  at the V2.2.9 table clock;
* a stalled SS read stream, unless restore binds by more than 0.5 ns. It runs
  below the sustained cycle on purpose, so admission lag accumulates. Five of
  them (32×32, 64×32, 64×64, 32×256, 64×256) fail only the per-clock-window
  PRE edge count (`restore_once`; at 64×256 also one `access_done` reset)
  once the lag moves a restore edge out of its window; restore levels, sensing
  and data pass;
* a confirmation at the published clock.

Confirmations use the full 14-case matrix for the squares and 512×256
(5 corners × seq/mixed, FF/SF −40 °C first reads, SS/FF −40 °C read streams).
For the other classes they use SS seq and read stream plus FF −40 °C,
FF 125 °C and FS 125 °C seq.

| Class | Confirmation runs (published clock unless noted) |
|---|---|
| 32×32, 64×64, 128×128, 256×256 | 14/14 each |
| 512×256 | 14/14 |
| 32×64, 32×128, 64×32, 64×128, 128×32, 128×64, 256×32, 256×64, 256×128, 512×32, 512×64 | 5/5 each |
| 32×256 | 5/5 (also 5/5 at 14.70 ns) |
| 64×256 | 5/5 (also 5/5 at its own 13.05 / 9.60 ns) |
| 128×256 | 5/5 (also 5/5 at its own 14.00 / 10.30 ns) |
| 512×128 | 5/5 (also 5/5 at 19.45 ns, before the restore recheck) |

All confirmations are scored by the final contract scorer (0.95 VDD restore).
The 256×256 matrix ran with a 12.20 ns read deadline. The deadline sets only
measurement times, and every read output settles at least 1,181 ps before it,
so the published 12.15 ns holds.

At 256×256 the unmargined budget (15.35 ns) was bracketed with an SS read
stream. It passes every 0.96 VDD restore check with 13 ps to spare and a flat
admission lag (1,204–1,237 ps). At 15.05 ns the far bitlines are at
0.952–0.959 VDD at admission: below the 0.96 VDD budget crossing, above the
0.95 VDD contract level. ISO-release margin at the
strict levels (signed, negative = overlap) is 90–100 ps at 32–128 rows
(FF −40 °C) and 55–65 ps at 256×256; at the contract level it is
≥ +230 ps at 256×256.

Protocol: [v229_overlapped_recovery_cases.json](../../../tests/spice/v229_overlapped_recovery_cases.json).
Records: [OVERLAPPED_RECOVERY_V2_2_9.json](../../data/OVERLAPPED_RECOVERY_V2_2_9.json)
(case verdicts both ways, margins, per-access admission/occupancy/CLK→OUT, macro
receipts, solver/model/RC provenance).

## Open items

1. **Restore detection at ≥ 256 rows.** `recovered` precedes the data-bitline
   restore there (above). Longer `restore_stages` for tall classes would let
   the interlock hold the 0.96 VDD budget level on its own (512×128 and
   512×256 have no stalled run). The clock holds it now.
2. **Read recovery still sets the cycle.** A read restores only after
   sensing; restoring during sensing collapsed the 128-column sense supply.
   A restore path that does not share that supply is design work.
3. **32×256 speed.** The square-anchor sizing would make 32×256 about 20%
   faster (and with it 64×256/128×256, which the monotonic grid holds at
   32×256's clock). It needs more decoder setup stages for that class.
4. **Guard simplification.** Under the relaxed contract some settle guards may
   be shorter than needed today (the margins above). Shortening them changes
   every netlist and needs the full re-verification.
5. **512×512** is outside the supported envelope: its macros are built and
   verified physically, and V2.2.9 passes SS at 62.75 ns, but it has no table
   entry.
