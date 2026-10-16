# Compact TIME_CONTROL layout — V2.2.8

**Planned controller-to-core interface verified on 2026-10-12.** The complete
32×32 bank with TIME_CONTROL (`planned_02`) passes official Calibre DRC with
zero violations, strict LVS `CORRECT`, and distributed R+C+CC extraction.
Every controller-to-core connection is now an explicit track. No maze search
or negotiation remains at the macro level. The bank shrinks from 5760.7131 to
**4050.6731 µm² (−29.7%)**. No waveform qualification is claimed.

## Review of the previous integration

The previous complete bank (`full_40`) passed DRC and LVS, but its
controller-to-core routing was poorly structured:

- All bank-facing controller pins were M8 pads on the controller's **far
  (left) edge**. Every link first crossed the whole controller away from the
  bank, then came back.
- The 32 data registers sat inside the controller frame. Their 705 nm-pitch
  targets are at the bottom of the columns, so an M3/M4/M5 staircase fanned
  the bus out 16 µm below the bank to M10 landings. M10 then carried it back
  to the controller.
- A 100 nm-grid maze router negotiated 53 nets on M7–M10 only, with
  400–800 nm wires and 800–1600 nm pitch. It needed congestion history,
  periodic rip-up and via-bridge repairs.
- Far-end returns were held on M9 over the controller to keep antenna jumpers.
  They looped around the bank to taps beyond the latch.

| Macro-level routing | `full_40` (maze) | `planned_02` (planned) |
| --- | ---: | ---: |
| Routed nets | 53 | 53 |
| M7–M10 wire | 4062.2 µm | 399.8 µm |
| Via cuts | 853 | 329 |
| Direction/layer changes | 2401 | 0–2 per control/feedback route |
| Complete bank area | 5760.7131 µm² | 4050.6731 µm² |

The `full_40` length counts only maze steps; its fixed M3–M5 data fan-out
is excluded. The planned length counts every macro-level wire from M1 to M10.

## Reference designs

Both reference compilers avoid search routing between control and core. Their
regularity comes from the floorplan, not from the router.

- **OpenRAM** (`b2b069ce`, local checkout):
  - `sram_1bank.place_control` places the control logic beside the bank,
    aligned to the array/port-data boundary. Every output exits on the edge
    facing the bank.
  - The bank exposes one vertical bus line per control signal.
    `connect_vbus` lands each signal with one horizontal wire and one via.
  - Row-address DFFs sit beside the decoder.
  - Data DFFs form a separate array below the port data, connected through
    `channel_route` (left-edge tracks with a vertical constraint graph).
  - Clocks join by rail abutment plus one spine.
- **sram22** (`37eca15`, substrate `60a8a4b`):
  - Blocks are rotated so their pins face their sinks.
  - Channel and gap sizes are computed from track counts before placement.
  - Each top-level route uses explicit track indices and at most a few
    segments. Power fills the remaining tracks last.

## Planned integration

| Block | Placement | Size |
| --- | --- | --- |
| Bank (`dense_bank`, external control) | Origin; decoder on the left | 35.995 × 59.065 µm |
| Control logic | Left of the decoder, 1.0 µm gap; bottom at the periphery floor | 15.245 × 43.4525 µm |
| 32 data registers | 4 × 8 array below the column periphery | 43.010 × 7.500 µm |
| Data channel | Between the array and the bank's DIN escapes | 16 M4 tracks |
| Replica WL driver / output latch | Above the decoder / right of the far-end taps | unchanged cells |

XTIME_CONTROL keeps its source hierarchy, and LVS remains flat. Physically,
the 32 data registers (704 MOS) form their own array. The control logic keeps
686 MOS, the 7 address/control registers and four RC guards. The frozen DFF
master is shared without change.

Every bank-facing controller pin sits on the controller's right edge, at the
height of the bank conductor it drives or observes. The pin plan comes from
the bank manifest; a reused controller is accepted only if its pin plan matches.

| Connection | Route | Layers |
| --- | --- | --- |
| PRE, s_en, sa_iso, w_en | One straight line into the bank's M6 control trunk | M6 |
| A_dff0–4, wl_en | Line to the decoder pin column, then a riser into the pin | M6, M5 |
| pre/sen/iso/wen far-end returns | Bar extended past the supply trunks; one straight line back (PRE: one jog) | M4 tap, M7 |
| Replica bitline | Tap on its column M5; one straight line | M7 |
| Replica WL return | Active replica row at the array's far end, over the decoder | M7, M8, M7 |
| 32-bit data bus | Register risers; left-edge channel; bank escapes | M8, M3/M4 |
| Register clock | Controller pad drop to the array's shared-rail trunk | M10, M5/M6 |
| VDD/VSS | Two straps over the register array, drops and trunk stacks | M9, M10 |

The register array uses OpenRAM-style abutment: shared wells, M1/M2 rails and
a CLK M6 rail per row. Each register Q leaves the array top on its own M8
track; rows below the top cross the registers above. Slots are assigned in the
bank escapes' left-to-right order, so the channel is a river route: one riser,
one M4 track and one riser per bit. A riser under another bit's escape always
takes the lower track. Adjacent via7 stacks alternate by 600 nm to meet via7
spacing.

Returns join the drive net only through the bank conductors, as before.
Observer returns now use at most M8. A layout-level width/spacing check and
metal/poly ownership audits run before export: no cross-net short, and each
net's endpoints form one conductor. Official DRC/LVS remains the acceptance
check.

## Standard-cell implementation

OpenYield generates source-matched INV, NAND2, NAND3, NOR2 and
transmission-gate DFF cells. These are OpenYield templates, not an imported
OpenRAM controller. The source contains 39 DFFs: 32 data, five address and two
control. It also contains 214 inverters, 12 NAND2s, four NAND3s and eight
NOR2s. All 1,390 logical TIME_CONTROL MOS are retained as 2,058 physical
fingers, each preserving model, topology, length and total width.

The planned control logic needs a 15.3 µm frame: the widest frozen driver is
9.74 µm, and it sits beside the 5 µm RC-guard column. Its 662.4 µm² plus the
322.6 µm² register array exceed the 872.2 µm² OpenRAM controller-and-register
reference by 13%. The complete bank is 29.7% smaller. A narrower frame with the
guards above the rows would recover more; it is not implemented.

The standalone controller (`layout time-control`) is unchanged.
Regenerating it reproduces `control_20` exactly: same placements, pins and
source, with zero XOR area. `control_20` remains verified (DRC 0,
LVS `CORRECT`; 838.7843625 µm²).

## Physical verification and RC scope

`outputs/layout/time_control/planned_02_verify/receipt.json` records:

- zero official FreePDK45 Calibre DRC violations;
- LVS `CORRECT`;
- distributed R+C+CC extraction of 3,600 nets.

The source schematic is byte-identical to `full_29` and `full_40`. The bank
retains 9,572 logical MOS and 10,702 physical fingers at 705 nm column and
1365 nm row pitch.

LVS uses `R SHORT SOURCE; C OPEN SOURCE`, so the intentional passive values
are checked separately. A conductance solve of the complete-bank DSPF at the
actual M6 guard terminals gives the following
(`planned_02/guard_resistance.json`):

| Target resistance | Extracted resistance | Error |
| ---: | ---: | ---: |
| 2462.4 Ω | 2473.7000 Ω | +0.459% |
| 4102.4 Ω (two guards) | 4083.6297 Ω | −0.458% |
| 4742.4 Ω | 4724.0990 Ω | −0.386% |

The unchanged passive generator retains its six extracted coupons. Matched
blanks give 1.02011 fF per intended 1 fF plate. This is not the complete
loaded node capacitance.

The [machine-readable evidence](../../data/TIME_CONTROL_PLANNED_INTERFACE_V2_2_8.json)
covers the GDS, source, DSPF, receipts, generator hashes, floorplan, pin plan,
channel assignment and routing comparison. Fifteen local TIME_CONTROL checks
pass, as do 47 tests in seven related layout suites. Python compilation and
`git diff --check` pass.

## Reproduction

Generation is restricted to the source-matched **32×32** plan with last-column
readout. The default `layout bank` remains the verified bank with external
control.

```bash
python main_sram.py layout bank --with-controller --controller-floorplan beside \
  --deck "$DECK" --output-dir outputs/layout/time_control/generated
python main_sram.py layout verify-full --layout-dir outputs/layout/time_control/generated \
  --pdk-dir "$PDK_DIR" --output-dir outputs/layout/time_control/generated_verify
python dev/time_control/guard_resistance.py outputs/layout/time_control/generated
```

`layout controlled-bank` is equivalent. Both build the bank and the edge-pinned
control logic in a sibling `generated_parts/`. Generation takes about two
minutes. `--controller-layout-dir generated_parts/control_logic` reuses control
logic only when its deck hash and pin plan match. `layout time-control` still
generates the standalone controller. All output directories must be new.

## History and preserved artifacts

| Artifact | Status |
| --- | --- |
| `outputs/layout/time_control/planned_02` (+ `_verify`, `_parts`) | Current planned interface; DRC 0, LVS CORRECT, extracted |
| `outputs/layout/time_control/planned_01` | Same plan with supply straps in the channel; DRC 0, LVS CORRECT |
| `outputs/layout/time_control/full_40` | Maze-routed bank; DRC 0, LVS CORRECT; superseded |
| `outputs/layout/time_control/control_20` | Standalone controller; DRC 0, LVS CORRECT |
| `outputs/layout/time_control/full_29` | DRC 0; LVS INCORRECT (latch D shorted to VSS); preserved |

The superseded maze integration is kept locally:
`dev/time_control/macro_control_maze_v228.py`, `macro_negotiation.py` and
`route_search.{py,cpp}`. The [compaction report](TIME_CONTROL_COMPACTION_REPORT_V2_2_8.md),
[compaction evidence](../../data/TIME_CONTROL_COMPACTION_V2_2_8.json) and
[stopped-work record](../../data/TIME_CONTROL_LAYOUT_V2_2_8.json) are unchanged.

No transient/waveform screen, PVT/yield qualification, timing update or RC
default promotion follows from these physical checks. Frozen compiler sizing
and timing remain unchanged. The control logic's internal routing still uses
the existing grid router (M4–M9); only the macro-level interface is planned.
