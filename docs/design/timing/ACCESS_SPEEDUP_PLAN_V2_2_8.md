# Access speed-up plan from the V2.2.8 baseline: earlier sensing, a self-timed cycle and measured clocks

**Implementation plan. Nothing in this document is release evidence.**
Baseline: branch `fix/postlayout-timing-pdn` at `58bf5cb`, compact macros with
layout g15 ([post-layout record](../layout/POST_LAYOUT_V2_2_8.md)). The
implementation release is **V2.2.9**, as assigned by the user on 2026-11-05.

Implementation and verification are recorded in the
[V2.2.9 result](ACCESS_SPEEDUP_V2_2_9.md). A1, the permitted A2 fallback,
B1, C1 and D are implemented; A3 detector replacement is deferred on the
measured feasibility evidence below. The result's scope and conditional
larger-array bounds take precedence over the original estimates in this plan.

## Execution amendments (2026-11-05)

- The supported architecture has a **2:1 column mux as a standard component**,
  as confirmed by the user. Mux-free layouts and timing tables are outside this
  implementation. The existing compact generator requires a 2:1-mux deck.
- The g15 segment measurements are now reproducible; see
  [baseline evidence](../../data/ACCESS_SPEEDUP_BASELINE.json). Actual CLK
  50% crossings and the old scorer's nominal falling-edge recovery convention
  are recorded separately. Historical metadata naming a schematic timing
  table is preserved rather than relabeled as a layout-table run.
- Retain the existing settling guards. None of the 29 screened detector
  candidates qualified across six corners, and 1,800 mismatch samples of
  extreme candidates did not establish the proposed trip bounds; see
  [detector feasibility](../../data/ACCESS_SPEEDUP_DETECTORS.json). This is a
  finite feasibility screen, not proof that every possible detector fails.
  Threshold-detector replacement is deferred until a viable cell is measured.
- Use the permitted sense-enable fallback: far `iso_ready` remains the
  interlock, while wordline release no longer gates sense assertion. Keep a
  settled far-write observation for the parallel preamble. Driver classes
  and physical supply/via models remain frozen.
- Self-timed read completion also waits for `wordline_off`. This permits
  sensing during the wordline tail while preventing the earlier completion
  from releasing ISO before the cell is disconnected. The asynchronous clear
  resets both master and slave so releasing clear during clock-high cannot
  launch another access.
- A self-timed sense amplifier must be checked at completion and while its
  footer remains enabled; it may already be precharged at the external read
  deadline. The output latch and stored data still have to retain the correct
  value through that deadline and to the next capture. Isolation release is
  checked at its falling 90% crossings, avoiding floating-point artifacts
  from starting an overlap window at an interpolated rising crossing.
- Larger-array verification may use estimation and extrapolation, as the
  user authorized. The new full-macro post-layout anchor is 32×32; larger
  controller classes, including 64 and 128, must be labeled extrapolated
  unless a new-controller post-layout transient is actually completed.
  Historical V2.2.8 traces supply scaling evidence, not new-controller
  waveform qualification. Decoder setup is checked against every output,
  including unselected rows; its revised stage counts replace the original
  two-stage assumption.
- Completion timing references can grow with both row and column class.
  Each read/write timer uses the larger of its row and column stage counts;
  the measured 32×32 circuit keeps eight stages. A longer clock cannot
  correct a timer that ends an access before its data has settled.
- The new recovery check is at the next capture, rather than the legacy
  0.1-period early check. Keeping that early reserve would stack another
  margin on D's single 10%. Cropped diagnostic traces must already be fully
  recovered at their end; single-operation traces extend to observe their
  final recovery before another capture.
- Use explicit joint row/column budgets for the new controller. The retained
  separable excess rule double-counts shared paths in the conservative event
  model: a large shape can receive about 43.6 ns from that rule against
  about 30.1 ns from the complete joint path estimate. The full class grid
  stores the bounded joint result and its evidence status; dimensions still
  round up. Legacy-controller tables keep their original rule.
- Resolve settling guards separately for each physical control path in the
  schematic profile. Large frozen write/precharge bars need longer references
  than isolation or wordline release. Their row/column references and reset
  costs belong in the joint budget. Preserve the measured 32×32 circuit's
  references; use actual bar coupons and explicit bounds for larger classes.

These amendments alone do not qualify the controller or shorter default
clocks. The linked implementation record supplies the waveform, mismatch and
completion-budget evidence used for promotion.

The V2.2.x controller makes every ordering a chain: drive a line, observe its
far terminal, wait a fixed number of settling stages, then start the next
step. Those chains, together with a sense amplifier that fires on a fully
swung bitline, set the access time. The clock contract then doubles the
slower of access and recovery into the period, because access runs only
while the clock is high and recovery only while it is low. This plan removes
all three sources of delay in five circuit work items. A sixth (D) then
derives the clock tables from measured completion times instead of stacked
margins. The plan keeps the recovered-state invariant of
[V2.2.1](PHASED_CONTROL_V2_2_1.md).

## Decisions (2026-11-02)

| Item | Decision | Consequence |
|---|---|---|
| A2 | The read wordline may overlap the sense enable; input isolation, not wordline release, protects regeneration | Scorer check `continuous_c{col}_sense_after_wl` is retired (A2) |
| B1 | A measured sense-amplifier offset replaces the 0.28 VDD floor, with margin | Margin rule in B1 |
| C1 | Self-timed cycle | Access and recovery no longer follow the clock levels |
| D | Work item | New 6T clock and access-time tables from measured completion times (D) |
| E | Deferred | Every driver class stays frozen |

Steps 0–5 run at the V2.2.8 table periods, so their gains show up as
read-output margin, phase slack and measured completion times. Step 6 (D)
turns those measurements into the default periods and read deadlines.

## Scope and invariants

- **Cells.** 6T cells with the standard 2:1 column mux, in the compact layout.
  TIME_CONTROL is shared with 10T. An appended optional controller argument
  selects the V2.2.8 controller or the new one; the compiler selects the new
  one for 6T, and 10T keeps the V2.2.8 controller until it has its own
  screen. Positional factory and testbench arguments are unchanged.
- **Frozen.** Every driver and wire class in `layout_sizing_lookup.json`, the
  device models and every scorer limit other than the rules replaced below.
  The clock tables stay at their V2.2.8 values until D. The replica fields
  (`replica.K`, `replica.N`) and the stage fields (`setup_stages`,
  `guard_stages`) are timing references, not driver sizes. B1 and A3 change
  them, and with them the lookup version.
- **Kept.**
  - Address, data, select and write request are captured on the rising edge.
  - Every access ends in the same recovered state before the next capture:
    WL, `w_en`, `s_en`, `sa_iso` and `rbl_delay` low, bitlines high, PRE on.
  - The forbidden overlaps stay forbidden: PRE with WL, write driver or
    sense footer; `w_en` with `s_en`; any enable before input isolation.
  - The write drivers stay on through the physical wordline tail.
- **Replaced.** Read WL off before sense (A2), the 0.28 VDD sense floor (B1),
  clock-high access with clock-low recovery (C1), and the 2 × half-period
  clock policy with its 0.68 T read deadline (D).
- **Identity.** Each item changes the control-architecture identity in the
  sizing fingerprints or the lookup version. No V2.2.8 frozen object or
  evidence is reused for it.

## Baseline: where the time goes

**Provenance.**
- *Runs:* g15 extracted macros (Calibre xRC R+C+CC, reduced as in the
  post-layout record), with the extracted supply mesh, every supply pin tied
  and kit via resistance.
- *Simulator and devices:* Xyce 7.4 (`linsol type=KLU`), FreePDK45
  `NMOS_VTG`/`PMOS_VTG` nominal devices. There is no local mismatch, so
  there is no seed.
- *Corner and clocks:* SS 0.9 V 125 °C. Clocks of 9.5, 12.0 and 21.25 ns
  from `layout_timing_lookup.json` `v2.2.8-layout-timing-1`; sizing
  `v2.2.8-layout-candidate-3` (replica K = 1, N = 3).
- *Measurement:* 50 % VDD crossings, measured from the CLK rising crossing of
  the same cycle, `seq` pattern. 32×32 uses cycle 1 (read of 1). 64×64 and
  128×128 use the fourth access (read of 0), with frozen cells outside the
  probed rows.
- *Files:* the run directories are
  `outputs/postlayout/g15/c{32,64,128}/c*_SS_seq` (ignored local outputs);
  their scored results are in
  [`POST_LAYOUT_V2_2_8.json`](../../data/POST_LAYOUT_V2_2_8.json).
- *Status of the tables:* they come from an ad hoc extraction and are not yet
  part of the assembled evidence. Step 0 makes that extraction a tool.

**Read, CLK → OUT (ns).**

| Segment | 32×32 | 64×64 | 128×128 | Mechanism |
|---|---:|---:|---:|---|
| CLK → far PRE off | 0.67 | 0.87 | 1.32 | clock buffer, gating, PRE buffer and bar |
| far PRE off → `access_request` | 0.49 | 0.62 | 1.03 | far observer, 4 settling stages, AND |
| `access_request` → far WL | 0.57 | 0.69 | 0.84 | 2-stage setup chain, `wl_en` buffer, row driver |
| far WL → `read_done` | 1.05 | 1.36 | 1.89 | replica K = 1 full discharge, N = 3 chain |
| `read_done` → far `s_en` | 1.23 | 1.40 | 1.86 | ISO; far ISO + 4 stages; far RWL off + 4/8/16 guard stages; `s_en` buffer |
| far `s_en` → OUT | 0.11 | 0.16 | 0.17 | amplifier and output latch |
| **CLK → OUT** | **4.11** | **5.10** | **7.12** | read output margin 0.30 / 0.56 / 2.93 ns at 0.68 T |

- **Full-swing sensing.** At 32×32 the selected amplifier input crosses
  0.5 VDD 0.58 ns after the wordline, and `s_en` fires 1.7 ns after that.
  The differential at isolation is 0.882 / 0.854 / 0.826 V, essentially the
  rail. At 128×128 the worst column reaches the 0.28 VDD bar 1.58 ns before
  isolation.
- **What `s_en` waits for.** Today `s_en = read_done & !B & iso_ready`.
  - Far ISO to `iso_ready` takes 0.35 / 0.52 / 0.90 ns.
  - Far RWL low to `wordline_busy` low takes 0.47 / 0.68 / 0.91 ns.
  - `s_en` follows the later of the two.
- **SS pays for fast-corner guards.** The settling stages are sized for the
  far-end tail, so they cost far more at SS than at FF: at 32×32 the
  WL-off-to-sense interval is 690 ps at SS against 179 ps at FF −40 °C.
- **The decoder is not critical at 32×32.** In the address-toggling `mixed`
  run, the far decoder address lines settle 0.34 ns after `clk_buf`, while
  `wl_en` comes 1.37 ns after it.

**Write, CLK → event (ns).**

| Event | 32×32 | 64×64 | 128×128 |
|---|---:|---:|---:|
| far PRE off | 0.66 | 0.86 | 1.31 |
| `iso_ready` | 1.82 | 2.35 | 3.70 |
| far `w_en` | 2.21 | 2.79 | 4.37 |
| `access_settled` | 2.57 | 3.26 | 4.95 |
| far WL | 2.96 | 3.69 | 5.60 |
| for comparison: far WL of a read | 1.72 | 2.17 | 3.20 |

The write preparation is serial: precharge off, then isolation, then drivers,
then setup. At 32×32 the cell's storage nodes cross mid-rail 3.02–3.06 ns
after CLK. The wordline stays on until the clock falls and is released only at
5.73 ns, 2.7 ns after the cell has flipped.

**Recovery.** The low phase is 0.5 T. Recovery uses 3.01 / 4.06 / 6.14 ns of
it, leaving 1.74 / 1.94 / 4.49 ns of restore-to-capture slack. At 128×128,
a write's recovery from clock fall to RBL at 50 % breaks down as follows:

| Step | ns |
|---|---:|
| clock fall → `wl_en` low | 1.12 |
| → `wordline_busy` low | 1.18 |
| → far `w_en` off | 0.84 |
| → `enables_off` | 1.19 |
| → PRE on | 0.73 |
| → RBL at 50 % | 0.30 |
| **Total** | **5.36** |

Each link in that chain has the same structure: observe, then settle.

## Work items

### A1. Parallel access preamble

**Today:**
- `A = gated_clk_buf & pre_off_ready`
- `write_prepare = we & A`
- `w_en = we & write_window & iso_ready`
- `wordline_start = A & write_ready` starts the setup chain
- `wordline_request = A & access_settled`

A write's isolation, drive and setup each wait for the step before. A read's
setup chain starts only after the precharge is observed off.

**Change:**
- **Setup chain.** It starts from the raw request `gated_clk_buf`, so the
  decoder allowance runs while PRE releases.
- **Wordline request.** It becomes
  `wordline_request = A & access_settled & write_drive_ready`.
- **Isolation.** `write_prepare = we & gated_clk_buf`. Isolating the sense
  inputs does not conflict with an active precharge, so ISO rises while PRE
  releases.
- **Write enable.** `w_en = we & write_window & iso_ready & pre_off_ready`.
  The conditions are the same as today; only their start points move.
- **Drive-ready condition.** `write_drive_ready` (`!we`, or far `w_en`
  observed high through an A3 observer) replaces the setup interval as the
  "drivers fully on" condition. The setup chain then covers decoder settling
  only. The V2.2.1 reason for starting the setup interval at the far write
  enable was that the WL must open with stable data and fully enabled
  drivers. The explicit AND keeps that guarantee.

**Stage rule.** `setup_stages` (2 for every row class) was chosen so the
`wl_en` rise is at least 1.5 × the decoder settle in the path coupons. Those
coupons include the precharge-off path that A1 takes off the critical path.
The rule is re-measured with the new start point, and the stage count rises
if it fails. The 512-row first read, where V2.2.1 found false wordline
pulses at SS, is a required schematic case.

**Files.** `sram_compiler/subcircuits/time_generate.py` (access request,
setup, write enable builders), [TIME_CONTROL_PATH.md](TIME_CONTROL_PATH.md)
(naming contract), and the controller layout's `ENABLE_CHAINS` in
`sram_compiler/layout/freepdk45/time_control.py` if a chain cell moves. No
array or slot change.

**Estimate.** Read −0.2 to −0.5 ns; it grows once A3 shortens
`pre_off_ready`. Write CLK → WL −1.1 ns at 32×32 and −1.9 ns at 128×128.

### A2. Sense enable by launch skew

**Today:** `s_en = read_done & !B & iso_ready`.

**Change.** `s_en = read_done & iso_launched`, where `iso_launched` is the
ISO driver's own output delayed by a short skew chain. The wordline still
releases at `read_done`. `B` leaves `s_en` but stays in the write window and
in PRE.

**Required ordering.** At every amplifier, the isolation pass gates must be
at 0.9 VDD before the footer reaches 0.1 VDD. The scorer's
`continuous_c{col}_isolation` and `min_isolation_before_enable_ps` checks
stay unchanged.

**Skew sizing.**
- *Why a launch skew works:* the ISO and `s_en` bars leave the same
  controller on matched tapped geometry, so a skew introduced at the
  launch point arrives at every column.
- *Why it still needs margin:* ISO drives two pass gates per amplifier and
  `s_en` drives one footer and the latch, so their edges differ.
- *How it is sized:* launch both lines together and measure, over every
  column tap and corner, the worst case of (ISO at 0.9 VDD) minus (`s_en`
  at 0.1 VDD). The skew chain is 1.5 × that worst case. It is measured in a
  schematic line coupon and confirmed post-layout.

**Scorer.**
- Retire `continuous_c{col}_sense_after_wl`.
- `min_wl_off_before_sense_ps` stays as an informational metric. It becomes
  negative.
- Add `iso_release_after_wl_off`: ISO falls through 0.9 VDD only after the
  read wordline is below 0.1 VDD. Once sensing overlaps the wordline, the
  amplifier's regenerated rails must not reconnect to a bitline whose cell
  is still on. The release path already orders this through `enables_off`;
  the check makes it explicit.
- Cell retention checks are unchanged. Energy per cycle keeps being reported;
  the bitline keeps discharging only for the wordline fall time.

**Fallback.** If the far-tap skew cannot be bounded with that margin, keep
`iso_ready` and drop only `!B`. The gain is about 0.3 ns at 32×32 and 64×64
and about 0.2 ns at 128×128, where the two observations finish within
0.2 ns of each other.

**Estimate.** `read_done` → `s_en` drops from 1.23 / 1.40 / 1.86 ns to
≈ 0.55 / 0.65 / 0.85 ns.

### A3. Threshold observers instead of observe-and-settle

**Today.** Each guard is a half-unit inverter that trips near mid-rail,
followed by a settling chain:

| Guard | Settling today |
|---|---|
| `pre_off_ready` | 4 stages + `precharge_off_tau` |
| `iso_ready` | 4 stages + `isolation_tau` |
| `wordline_off` | `guard_stages` 4 / 8 / 16 / 44 / 64 by column class |
| `enables_off` | 2 stages + `enable_off_tau` |

The stages exist because a mid-rail trip says nothing about a tail that is
still above 0.1 VDD.

**Change.** Observe each far terminal with a detector that trips near the
scorer's own thresholds: low-threshold for a falling line, high-threshold
for a rising one. Leave a residual settle of 0–2 stages.
- *Candidate cells:* a ratioed skewed inverter, a Schmitt trigger, or a
  stacked detector, built from the deck's existing devices (no other
  threshold-voltage flavour is assumed).
- *Selection:* across the five corners and per-device MC, the trip point is
  ≤ 0.15 VDD for falling detection and ≥ 0.85 VDD for rising detection. The
  residual settle is 1.5 × the measured tail time from the trip level to
  the scorer threshold.
- *Reuse:* C1 uses the same detectors as completion signals.

**Risk.** Ordering slack is smallest at FF: the first-pass record kept only
96–233 ps at 64×64 FF −40 °C. A detector whose trip point drifts toward
mid-rail at a corner eats that slack. FF −40 °C and FF 125 °C are required
corners for every observer, and the trip point is screened under MC.

**Files.** Guard classes in `time_generate.py`; stage defaults in
`sram_compiler/config_yaml/time_control.yaml`; `guard_stages` in
`layout_sizing_lookup.json`, which becomes a residual count (new lookup
version); the detector cell in the controller layout library.

**Estimate.** The largest gains are at 128 columns and up, where the
wordline-off guard is 16 to 64 stages. A3 shortens both access
(`pre_off_ready`) and recovery (every link of the recovery chain above).

### B1. Small-swing sensing with a measured offset

**Step 1 — offset.**
- *Setup:* Monte Carlo of `SENSEAMP` in its compact sense slot, with the
  extracted slot parasitics, the A2 enable order, and the default
  per-device `vth0/u0/voff` mismatch.
- *Samples:* at least 1000 per corner at the five worst-PVT corners and TT,
  with seeds recorded.
- *Offset per sample:* found by bisection on the input differential at the
  isolation instant. Both inputs start precharged; one is lowered by ΔV
  before ISO closes.
- *Recorded:* μ, σ and the extreme sample. The same runs record the
  resolution time from far `s_en` to the output latch, which C1 needs.

**Step 2 — bar (the margin decided on 2026-11-02).**

    V_bar(corner) = |μ_os| + 6 σ_os + V_sys,   floored at 0.10 VDD

- *`V_sys`:* the bitline coupling from neighbouring columns plus the supply
  bounce at the sense slot during a read. It is 30 mV until step 3 of the
  rollout measures it post-layout.
- *Scorer:* the bar is read per corner from a tracked JSON (proposed
  `docs/data/SENSE_OFFSET_<version>.json`), replacing
  `SENSE_MARGIN_FRACTION`. The 0.28 VDD comparison remains as an
  informational metric.

**Step 3 — replica.** For each row class, choose the replica (K = 1–4 active
cells, N = 1 or 3) so that:

1. the nominal worst-PVT post-layout differential at the isolation sample,
   worst column, is at least **1.5 × V_bar**;
2. under per-device Monte Carlo of the array, replica, delay chain and
   amplifier, every sample is at least **V_bar**, with every column still
   reaching it before isolation (the `sense_lead` analysis). Required
   corners are SS, FS 125 °C and FF 125 °C (where V2.2.4 found replica
   leakage); the record states the sample count, because it bounds only
   what it samples;
3. the V2.2.4 passive replica load cells stay
   ([PHASED_CONTROL_V2_2_4.md](PHASED_CONTROL_V2_2_4.md)).

**Expected bar.** With σ_os of 15–20 mV, V_bar is about 0.14–0.18 VDD and
the nominal target about 0.21–0.27 VDD. Most of the gain therefore comes from
not waiting for the full swing (0.83–0.88 V today), not from a very small
differential. A read's bitline restore also shortens, because the swing is
smaller. Writes are unchanged.

**Files.**
- *Lookup:* replica K and N per row class in `layout_sizing_lookup.json`. It
  is one global `replica` object today, so per-class fields are appended
  with the global value as their default.
- *Sizing:* the corresponding validation in
  `sram_compiler/sizing/driver_sizing.py`.
- *Layout:* `sram_compiler/layout/freepdk45/compact_array.py` places the
  only active replica cell in the last row (line 82). It must place K active
  cells as the netlist does.
- *Scorer:* local `tests/spice/phased_waveforms.py`, plus a new tracked case
  manifest.

**Estimate.** Far WL → `read_done` drops from 1.05 / 1.36 / 1.89 ns to
≈ 0.45 / 0.55 / 0.70 ns.

### C1. Self-timed cycle

**Contract.** The rising edge captures and starts the operation. Access and
recovery then run on internal completion signals; the clock-low level is not
used. The period must cover access plus recovery plus register setup, and
the clock needs only minimum high and low pulse widths set by the capture
registers, not a 50 % duty cycle.

**Signals.** Names are proposals.

- **`access_phase`** replaces clock-high as the access window. It is set by
  the rising `clk_buf` edge of a selected cycle and cleared by
  `access_done`. The default candidate is a flop clocked by `clk_buf`,
  D = the select request, with an asynchronous clear from `access_done`. It
  is edge-triggered by construction, so a long clock-high cannot re-arm it.
  Its clear must release before the earliest next edge (FF −40 °C at the
  measured minimum period). The alternative is a pulse generator with an SR
  latch.
- **`A = access_phase & pre_off_ready`.** Today it is
  `gated_clk_buf & pre_off_ready`; `gated_clk_buf` keeps its name and
  function.
- **`read_complete`** = far `s_en` observed high, plus a settle of 1.5 × the
  Monte Carlo worst amplifier resolution and output-latch time from B1
  step 1. `read_done` stays the replica trigger.
- **`write_done`** is the write-completion timer. Two candidates are compared
  in step 4:
  - **W1, replica write cell.** A dedicated cell in its own replica column.
    A replica driver at the far end of the `w_en` line writes it with
    toggling data on every write, and an A3 observer reads its storage
    node; `write_done` = flip observed + settle. It tracks cell, driver and
    line at the far end, and costs one column pitch plus a slot. The
    existing replica column cannot do this, because its write driver pulls
    RBL low during writes (V2.2.1).
  - **W2, matched delay** from the far wordline rise, sized from the
    per-device Monte Carlo worst flip time at SF 0.9 V −40 °C. It needs no
    layout component, but it tracks corners only through its own delay
    chain.
  - **Write margin (same 1.5 rule).** The wordline-on time, from WL at
    0.9 VDD at the far tap to WL release, is at least 1.5 × the Monte Carlo
    worst time from WL on until both storage nodes pass 0.9 / 0.1 VDD.
    Every write sample must pass, and the scorer's write dwell (WL off after
    the last flip) stays positive.
- **`access_done = read_complete | write_done`.** An unselected cycle never
  sets `access_phase`.
- **`PRE = !(!access_phase & wordline_off & enables_off)`.** Precharge
  starts once the access is complete, not at clock fall.
- **Unchanged structure.** The write window `A | B`, `w_en`, `s_en` release,
  ISO release and `enables_off` are as today, but follow `access_phase`
  instead of the clock.

**Side benefit.** A clock stopped high now parks the array precharged after
its access. Today it parks with floating bitlines and open sense pass gates
(open item of [V2.2.3](PHASED_CONTROL_V2_2_3.md)).

**Testbench and scorer.**
- **Clock source.** Duty becomes an appended optional testbench parameter
  (default 50 %). C1 cases run at 30 %, 50 % and 70 % high, plus a clock
  stopped high for several periods.
- **Checks.**
  - The scorer's cycle windows already run from capture to capture.
  - The 0.68 T read deadline stays until D replaces it with an absolute
    access time.
  - New: `access_phase` rises once per selected capture.
  - New: `access_phase` is cleared and every recovered-state condition is
    met before the next capture, at every duty.
  - New: PRE turns on after `access_done` regardless of the clock level.
  - `access_phase` joins the `capture_reset` nodes.
- **Measures.** `TRESTORE` starts at `access_done` instead of the falling
  edge. The testbench's 50 %-duty CLK(min) diagnostic
  (`sram_6t_core_MC_testbench.py`) becomes access plus recovery.
- **Minimum period.** Completion times and a period sweep of the same decks
  give T_min per size and corner; D turns them into table values.
- **Documentation.** New nodes are added to the naming contract in
  [TIME_CONTROL_PATH.md](TIME_CONTROL_PATH.md) §4.

### D. Measured clock and access-time tables

**Today.**
- *Period:* `T = ceil_to_50ps(2 × half × 1.25)`, where `half` comes from the
  row and column ladders and the joint-dimension rule.
- *Read deadline:* data is due 0.48 T after capture. That is the checker's
  0.68 T; the runtime check uses 0.7 T.
- *Ladders:* already two to three times the V2.1.10 budgets.
- *Why it must change:* once C1 runs access and recovery back to back,
  `2 × half` no longer describes the cycle, and none of the stacked margins
  was measured against local mismatch.

**Change.**
- **Table form.** For the supported 2:1-mux 6T architecture, each row and
  column class stores two budgets: `access_ps` (capture edge to OUT valid)
  and `cycle_ps` (capture edge to the recovered state, the longer of read
  and write). The schema version rises. 10T entries keep today's form and
  policy, with the V2.2.8 controller.
- **Policy.** `T = ceil_to_50ps(cycle × (1 + m))` and
  `t_acc = ceil_to_50ps(access × (1 + m))` after the capture edge.
  - *One deadline:* `t_acc` serves both the checker and the runtime
    measures, so the 0.02 T reserve between them goes.
  - *Joint dimensions:* explicit joint class budgets cover both dimensions;
    they include each shared controller path once. Single-dimension anchors
    remain in the table, and legacy control keeps the larger budget plus
    smaller-excess rule.
  - *Rounding:* dimensions still round up to the next class anchor, because
    evidence exists only at the anchors.
- **Budgets.** Under C1, internal timing does not depend on the period once
  the period is long enough. A budget is therefore a measured completion
  time, not the result of a search: capture to OUT valid, and capture to
  recovered state, taken from runs at a generous period. Each budget is the
  worst over corners and per-device Monte Carlo samples. The period sweep
  only confirms it.
- **Margin.** `m = 0.10`. Mismatch is already inside the budgets, so `m`
  covers what the screen does not sample: the tail beyond the Monte Carlo
  sample count, model and extraction error, and supply noise beyond the
  extracted mesh.
- **Original planned coverage (superseded by the execution amendments).**
  - *32, 64 and 128 rows and columns:* post-layout nominal worst-PVT
    completion times, multiplied by the Monte Carlo worst-to-nominal ratio
    measured in schematic at the same size and corner. A 32×32 post-layout
    Monte Carlo spot check at SS confirms that the ratio transfers.
  - *256 and 512:* the schematic Monte Carlo worst, multiplied by the
    largest post-layout-to-schematic ratio measured at 32–128. The table
    status marks these classes as extrapolated, because one post-layout
    transient takes about a week.
  - *Non-square shapes that test the joint rule* (for example 512×32,
    32×256, 128×256 and 512×128): schematic, scaled by the same ratio.
- **Two tables.** Both get new versions:
  - `layout_timing_lookup.json` (compact layout), from the
    post-layout-based budgets;
  - `timing_lookup.json` (schematic decks), from schematic Monte Carlo
    alone.

**Code.**
- `sram_compiler/sizing/timing.py`: the table loader, `_half_period_ps`,
  `resolve_timing` and `timing_from_measurements`. The `TimingConfig`
  minimum-period check `2 × max(phases) × (1 + margin)` becomes the cycle
  budget for the self-timed controller; the old check stays for 10T.
- `ACCESS_DEADLINE` in
  `sram_compiler/testbenches/sram_6t_core_MC_testbench.py`, with its
  runtime measures and CLK(min) diagnostic: an absolute `t_acc` for the
  self-timed controller.
- The scorer deadlines.

**Not planned.** An interim trim of the 128-row class on the V2.2.8
controller, which would take it from 21.25 ns to about 19 ns using today's
margins. C1 replaces the contract that the trim would be computed for.

**Original planned confirmation (superseded by the execution amendments).** Every class is re-screened at its new period with the
common gate, including 128×128 post-layout and per-device Monte Carlo at the
derived period. The period sweep must show the first failing period below
`T / (1 + m)`.

## Order and gates

The original rollout below is retained as planning history. The execution
amendments and [V2.2.9 result](ACCESS_SPEEDUP_V2_2_9.md) define the completed
verification scope: a new 32×32 extracted anchor and explicitly conditional
larger-class estimates, as authorized by the user. They replace the original
requirements for new larger-macro physical receipts and transients below.

Each step has the same exit gate:

- **Schematic matrix.** 8×4, 16×16, 32×32, 64×64 and 128×128 with the mux,
  plus the 512-row first read, at SS 0.9 V 125 °C, FS 0.9 V 125 °C,
  FF 1.1 V 125 °C, SF 0.9 V −40 °C and FF 1.1 V −40 °C, with `seq` and
  `mixed` patterns. Everything passes.
- **Monte Carlo.** Per-device runs where the item needs them.
- **Physical verification.** DRC 0 / LVS CORRECT / xRC at 32–512 for any
  netlist change. The controller layout is generated from the netlist, so
  every TIME_CONTROL change rebuilds it.
- **Post-layout.** The g15 flow (`dev/postlayout`) at 32×32 and 64×64,
  10 of 10 cases.
- **Waveform inspection.** The TIME_CONTROL pulses `clk`, `clk_buf`, `cs`,
  `we`, `wl_en`, `rbl`, `rbl_delay`, `s_en`, `w_en`, `PRE` and `sa_iso`, the
  physical far-end crossings, and data retention.
- **Evidence.** A JSON record with solver, model, seed, RC and layout
  revision.

128×128 post-layout transients take 13.5–18 h per case (g14), so they run at
the end of steps 1, 4 and 6 only.

| Step | Content | Exit criterion beyond the common gate |
|---|---|---|
| 0 | Segment-extraction tool (local `dev/`, output into the evidence JSON); B1 step 1 offset Monte Carlo; A3 detector trip study; FF ordering slack per guard from the g15 runs | V_bar per corner recorded; detector candidate chosen; segment table reproduced from the tool |
| 1 | A1 + A2 | Skew chain sized by the 1.5 rule; `iso_release_after_wl_off` passes; 512-row first read passes; 128×128 post-layout 10 of 10 |
| 2 | B1 replica per row class | Both B1 replica criteria met; `sense_lead` positive in every column |
| 3 | A3 | FF −40 °C and FF 125 °C ordering slack ≥ today's minimum; `V_sys` measured post-layout and B1 re-checked with it |
| 4 | C1, with W1/W2 chosen first in schematic and Monte Carlo | Duty sweep and stopped-clock cases pass; write and read completion margins meet the 1.5 rule; 128×128 post-layout 10 of 10 |
| 5 | Completion-time measurements and period sweep: 32/64/128 post-layout nominal, 32×32 post-layout Monte Carlo spot check, schematic Monte Carlo up to 512 rows including the non-square shapes | `access_ps` and `cycle_ps` per class recorded with their evidence |
| 6 | D: new table versions, policy code, deadline, re-screen at the new periods; records, CHANGELOG, READMEs | Common gate at the new periods; sweep confirms the margin |

## Projected effect (segment arithmetic, not measured)

These are estimates at SS 0.9 V 125 °C. They will be replaced by step
measurements.

| Quantity | 32×32 | 64×64 | 128×128 |
|---|---:|---:|---:|
| Read CLK → OUT today | 4.11 | 5.10 | 7.12 |
| after A1 + A2 | ≈ 3.1 | ≈ 4.0 | ≈ 5.6 |
| after B1 | ≈ 2.5 | ≈ 3.2 | ≈ 4.4 |
| after A3 | ≈ 2.3 | ≈ 2.9 | ≈ 3.8 |
| Write CLK → far WL today → after A1 | 2.96 → ≈ 1.8 | 3.69 → ≈ 2.3 | 5.60 → ≈ 3.6 |

With C1, a 32×32 macro needs about 3.7 ns for a read and 4.3 ns for a write
(access plus recovery): a nominal completion of about 4.5 ns at SS. Suppose
mismatch adds 5–10 % and D's 10 % margin goes on top. The 32×32 class would
then land near 5–5.5 ns (9.5 ns today), with a read deadline near 2.7 ns after
capture (4.56 ns today). 64×64 and 128×128 come from step 5.

## Deferred

- **E, driver sizing and physical.** The current selection rule picks the
  smallest class whose SS edge is within 10 % of the class half-period, so a
  long clock licenses slow edges. E would replace it with absolute far-end
  edge targets, shorten the controller's per-chain wiring, and add wide
  metal, both-end drive or repeaters on the PRE and `w_en` bars at 128
  columns and up. Resume if a driver or bar becomes the largest remaining
  segment after A–C.

## Risks and open questions

- **A2 skew bound.** A2 depends on the matched ISO and `s_en` bar geometry.
  If the far-tap skew cannot be bounded with margin, the fallback recovers
  only 0.2–0.3 ns.
- **B1 bar scope.** The bar is only as good as the offset Monte Carlo of the
  compact slot. Weakest-cell analysis at array-level sigma is yield work and
  outside this plan.
- **C1 latch race.** The `access_phase` set/clear race is tightest at FF.
- **C1 timer choice.** W1 adds a column; W2 tracks corners only through its
  delay chain. Monte Carlo decides between them.
- **A3 trip drift.** Detector trip points move with corner, and the residual
  settle must cover that drift.
- **10T.** 10T stays on the V2.2.8 controller until it is screened
  separately.
- **256×256 and 512×512.** They keep schematic evidence only, as today: one
  post-layout transient would take about a week. Their D classes are
  extrapolated from the 32–128 post-layout-to-schematic ratio, and the table
  says so.
- **D margin.** `m = 0.10` is a proposal. It excludes mismatch, which is in
  the budgets. If the Monte Carlo sample counts stay small, the unsampled
  tail argues for a larger `m`.
