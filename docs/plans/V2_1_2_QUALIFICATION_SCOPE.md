# V2.1.2 qualification scope and separate briefs (plan Phase 6)

Written September 14, 2026 as the Phase 6 deliverable of the
[V2.1.1 evaluation plan](V2_1_1_TIMING_FOLLOWUP.md). It defines the inputs a
full timing/driver qualification needs, the PVT and sample matrix the current
tools can support, and two separate briefs (half-select, yield estimators).
It is a scope definition, not evidence: nothing here promotes a record to
`sizing_table.json`, and every figure quoted comes from the referenced records.

## 1. Extracted-metal inputs

The compiler accepts one `interconnect` mapping (`global.yaml`,
`INTERCONNECT_CONFIG` in `main_sram.py`, `--interconnect-config` on the CLI,
or the appended `interconnect=` testbench argument). Its fields are the whole
physical wire model, so extraction must supply every one of them:

| Field | Meaning | Extraction source |
|---|---|---|
| `wl.layer`, `bl.layer` | Technology identifiers, recorded in evidence | Layout metal assignment |
| `wl.pitch_m` | Cell width along a wordline | Bitcell layout |
| `bl.pitch_m` | Cell height along a bitline | Bitcell layout |
| `width_m` | Drawn wire width | Layout |
| `sheet_resistance_ohm` | Sheet resistance at the qualification temperature | PDK, per corner if available |
| `capacitance_f_per_m` | Total capacitance to ground per metre, including neighbour coupling folded to ground | Extraction of the array wire in context |
| `sections_per_half_pitch` | Ladder refinement; total R and C are conserved | Choose 2 or more when a pitch's RC product is large; the Phase 4 refinement case records the sensitivity |
| `cell_pin_rc` | Series pin sections at every cell tap | Contact and via resistance of the access transistors |

Per-pitch resistance is `sheet_resistance_ohm * pitch_m / width_m` and per-pitch
capacitance is `capacitance_f_per_m * pitch_m`. The default (1 ohm / 0.1 fF per
0.6 µm pitch) is illustrative and must not appear in a qualification record.

Assumptions that are fixed in code today and must be replaced by layout
distances before an extracted-metal claim:

- Bitline periphery: three pitches beyond each array port, taps at 0.5, 1.5 and
  2.5 pitches for precharge, write driver and mux/sense (`_add_periphery_wires`).
- Control lines spanning an array dimension (`PRE`, `w_en`, `w_en_bar`, `s_en`,
  `sa_iso`, `wl_en`) at the bitline pitch (`_add_control_wires`).
- Decoder address trunks spanning the row height, true/complement/enable
  ladders partitioned per 3-to-8 block, and the DATA_DFF clock and mux-select
  ladders at the wordline pitch across the column count.
- Local series stubs (`w_rc`, `pi_res`, `pi_cap`) stand in for contact, via and
  storage-node parasitics; extracted values replace them, not add to them.

Not modelled, and to be stated in any record: coupling between neighbouring
bitlines or wordlines as a signal (only its ground-referred capacitance), the
temperature dependence of metal resistance unless the supplied sheet
resistance already includes it, via resistance except through `cell_pin_rc`,
and supply IR drop (VDD and VSS are ideal).

A changed geometry re-runs the functional gates of Phases 2 and 3 at the
frozen lookup clocks and driver classes. Failures are reported, never fitted
away; the clock table changes only through a new evidenced version.

## 2. Supported PVT and sample matrix

Screened so far, all with the illustrative wires and frozen V2.1.0 clocks:

- Nominal devices: TT 1.0 V / 25 °C, SS and SF 0.9 V / 125 °C, FF 1.1 V /
  −40 °C on the V2.1.1 release cases; five corners at 0.8 to 1.1 V and −40 to
  125 °C on the 16x16 write points of the V2.1.2 audit; the Phase 4 matrix of
  this release (widths to 128 columns, heights to 512 rows, wire refinement
  and stress, local stubs on and off, cell-pin RC, 6T and 10T mux, hazards).
- Local mismatch: the Phase 5 pilot, three explicit per-device seeds per case,
  5 % relative sigma on `vth0`, `u0` and `voff` of every MOS (the
  `full-local-v1` policy that a qualification record must declare), expanded
  to ten seeds per case in V2.1.3 together with the 10T class-bound seeds.
- 10T arrays: the V2.1.3 [10T budget](../design/TIMING_10T_BUDGET_V2_1_3.md)
  screens every reachable 10T class bound at SS 0.9 V / 125 °C with and
  without a mux and the 64x16 SF write; the 10T clocks differ from 6T, so
  the array classes below are per cell type.

Proposed qualification matrix per array class and operation:

| Axis | Values | Note |
|---|---|---|
| Corner | TT, SS, FF, SF, FS | Model cards under `tran_models/` |
| VDD | 0.8, 0.9, 1.0, 1.1 V | 0.8 V lies outside the clock derivation; it is screened, not assumed |
| Temperature | −40, 25, 85, 125 °C | |
| First points | SS 0.8/125 read, SS 0.8/−40 read, SF 0.8/125 write, SF 0.8/−40 write, FF 1.1/−40 sequence, FS 0.8/125 sequence, SS 0.9/125 (timing basis), TT 1.0/25 (reference) | Worst known directions first; the full grid is 80 points |
| Array classes | Each lookup anchor boundary: rows 32/64/128/256/512, columns 4 to 512 | The clock is fixed per class, so the class upper bound is the case |
| Mismatch | 10 seeds per point (the V2.1.3 [pilot](../design/TIMING_10T_BUDGET_V2_1_3.md) ran ten seeds on the four Phase 5 cases and three seeds at the 10T class bounds) | Failure-free counts bound the rate only weakly (0 of 100 gives ≤ 3 % at 95 %); statistical yield needs the estimator brief |

Cost bounds the matrix. With full transistor arrays, 4,096 cells take about
an hour per case at eight ranks (V2.1.1 8x512 write 54 min, 64x64 sequence
85 min); 8x512 and 512x4 are the practical full-array boundaries. Larger
classes need the equivalent-cell modes 1 to 4, which are approximations and
need their own accuracy record before they can carry a qualification.

Acceptance rules for every point: runtime release/access/hold/restore checks
and the independent all-column waveform checks both pass; margins are
recorded (PRE90-to-WL50, WL10-to-PRE90 restore, clock-to-Q, capture-to-write);
electrical, numerical (`Time step too small`, `DC Operating Point Failed`) and
incomplete outcomes stay separate; every scheduled seed is accounted for and
no nonfinite sample counts as success; clocks and driver classes stay frozen.

Numerical fallback: the legacy materialized-MPI ensemble path in
`dev/sizing/execution.py` still deletes failed outputs and can spend several
full timeouts. Phase 5 avoided it by running each per-device sample on one
rank through the plain runtime retry. Before per-device runs on arrays that
need MPI, either repair that path (preserve every attempt, bound retries) or
keep one rank per sample and run samples in parallel.

## 3. Half-select brief (separate architecture work)

Today every column of the selected row carries its own write driver and all
drivers apply the same data during a write, so no column is ever half-selected
in a write: the compiler never exercises a cell that sees its wordline high
with precharged bitlines while a neighbouring column is being written. Reads
exercise the read-like condition on every column, which the scorer checks.

Required change, in order:

1. A per-column write enable derived from the column select (the existing mux
   select) and an explicit write mask, so unselected columns keep their write
   drivers off with bitlines precharged. Sizing then follows the number of
   driven columns, not the column count.
2. Checks: half-selected cells of the selected row retain data through the
   write window (Q/QB within 0.1 VDD of their values, as the retention checks
   already require), the selected column writes, and the precharge, restore
   and release checks still hold on every column.
3. Cases: mux ratios 2 and 4, SS and SF 0.9 V / 125 °C, FF 1.1 V / −40 °C,
   address-change hazards, then the per-device seeds of the Phase 5 recipe.
4. Read-modify-write (bit interleaving with write-back) is out of scope.

## 4. Yield-estimator brief (separate numerical work)

`yield_estimation/model_lib/{MC,MNIS,AIS,ACS,HSCS}.py` consume the compiler's
four-value result and treat nonfinite delays as failures (V2.1.0), but they
are not usable as shipped:

- Machine-local paths: `/home/lixy/...` bound files, `sim0`/`sim2` folders and
  model output files; `tool/delete.py` deletes a `/home/lixy/sim` folder at
  import time.
- External dependencies absent from `environment.yml`: `torch`, `gpytorch`,
  `mpmath`, `prettytable`.
- They sample explicit parameter vectors through the `custom` variation mode
  (18 values per 6T cell, scaled by the cell count), not the default per-device
  flow, and their failure criterion is a delay threshold, not the release checks.

Work before any yield number is quoted:

1. Make paths configurable and remove import-time deletion; declare or isolate
   the dependencies.
2. Map an estimator sample to the compiler contract: either a seed on the
   per-device path (one deterministic materialized model per sample) or an
   explicit vector through `get_custom_vars()` with the per-device parameter
   layout; keep the frozen timing and driver baseline through sampling.
3. Define failure from the release checks (access, hold, restore, precharge
   exclusion, finite metrics), and classify numerical non-completion as
   incomplete, never as failure or success.
4. Validate MC against MNIS/AIS/ACS/HSCS on a small array with an inflated
   sigma so the true failure rate is measurable, with confidence intervals,
   before any full-sigma estimate. Budget: a 16x16 sample costs about three
   minutes at four ranks.

## 5. Ordering

Phase 6 follows the reviewed Phase 5 pilot. The extracted-metal inputs gate
timing qualification; the half-select and yield briefs are independent of it
and of each other. None of them changes the V2.0.9 driver classes or the
V2.1.0 clock table without new evidence.
