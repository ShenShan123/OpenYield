# V2.2.7 development: FreePDK45 RC calibration and TIME_CONTROL configuration

Date: 2026-10-09. This is a development record, not a released calibration or
completed waveform screen. The previous release is V2.2.6; its limitations and
the historical V2.2.2 screen remain unchanged.

## Scope and decisions

The requested sequence is physical layout, ground-truth Calibre xRC extraction,
analysis and back-annotation, then replacement of distributed RC defaults.
The user selected FreePDK45 and an 8x8 6T initial reference. The local layout
starting point was derived from OpenYield commit `8bb03ee` (V2.2.6). Its reusable
geometry is now in `sram_compiler/layout/freepdk45/`; it has no dependency on the
ignored contest directory or a machine-specific PDK path.

TIME_CONTROL was originally excluded from that physical reference. Its audit
found 798 MOS, six 45 nm NMOS widths below the installed FreePDK45 Active.1
minimum of 90 nm, twelve off-grid widths, and four explicit RC delay filters
(261–696 ohms, each with 1 fF). The user selected intentional physical passives,
then chose **frozen sizing and deferred full controller layout** when those
geometry conflicts were identified. No transistor legalization, passive
replacement or controller topology change has been performed.

The user also requested that all sizes and other TIME_CONTROL configuration
parameters move into a dedicated YAML file. That refactor is complete and
preserves the baseline circuit.

## Physical reference and extraction

The 20.915 × 10.040 µm macro contains the 64-cell data array, decoder, wordline
drivers, write buffers/drivers, precharge, sense amplifiers, replica column,
replica wordline driver and dummy loads. It excludes TIME_CONTROL, its registers,
startup clamps and stimulus sources. The reference schematic has 876 MOS;
layout fingering and extraction representation can change the physical element
count while preserving LVS equivalence and total device widths.

KLayout Python 0.30.7 builds GDS. Calibre v2025.1_16.10 uses the installed
FreePDK45 `calibreDRC.rul`, `calibreLVS.rul` and `calibrexRC.rul`. The macro and
the independently checked array tile both pass official DRC with zero results
and LVS with `CORRECT`. The macro also passed the retained KLayout DRC/LVS
diagnostic during the initial reproduction.

The initial array tile had seven disconnected VDD stripes because it lacked
the connection normally supplied by column periphery. The new standalone tile
adds an explicit M4 strap and passes its own official DRC/LVS before extraction.
The failed initial run is preserved locally; its extraction is not accepted by
the calibration analysis.

Extraction runs hierarchical LVS for source/layout cross references, then xRC
`-phdb`, `-pdb -rcc` and `-fmt -all`. The coupled DSPF retains source names,
distributed contact/poly/via/metal resistance, coupling capacitance, and MOS
diffusion areas/perimeters. Each receipt records source/GDS/schematic/DSPF,
generator, executable and rule hashes. Analysis checks those hashes and rejects
failed or grounded-only evidence. PDK files, layouts, decks, logs and full
waveforms remain external or under ignored `outputs/`.

## Initial RC reduction

Resistance is solved electrically through the full resistor network and fitted
against actual contact positions. BL contacts share row pairs and span six
pitches; BLB contacts span eight. The previous local script incorrectly used
six for both. A shortest-path sum also cannot represent parallel current paths.

| Array route | Pitch | Effective R per pitch | Quiet-neighbour C per pitch |
|---|---:|---:|---:|
| WL | 1.365 µm | 4.6543644 ohms | 0.78303875 fF |
| BL/BLB mean | 0.400 µm | 1.4285700 ohms | 0.12756753 fF |

The diagnostic YAML maps these to the existing `WireRC` API with 70 nm track
width, effective sheet resistances 0.23868535 and 0.24999975 ohms/square, and
capacitances per metre 5.7365476e-10 and 3.1891883e-10 F/m. Port/pin access
resistance stays in the fit intercept and is excluded from wire R per pitch.
Capacitance is the extractor's incident NET total divided by eight; this is a
quiet-aggressor reduction, not an explicit dynamic-coupling model or a proven
array-size scaling law.

The full macro extraction also shows why one pair of scalar wire classes cannot
calibrate every route:

| Macro net | Incident C | Port-to-terminal effective R range |
|---|---:|---:|
| PRE | 3.854 fF | 155.29–301.83 ohms |
| W_EN | 9.911 fF | 174.99–513.32 ohms |
| S_EN | 5.011 fF | 170.32–274.70 ohms |
| SA_ISO | 4.490 fF | 198.17–355.43 ohms |
| WL_EN | 2.847 fF | 403.07–442.38 ohms |

These R ranges include physical port/access and pin stubs; they are not wire
R per pitch. The coupled macro has 54.822 fF of unique shunt capacitors and
191.430 fF of unique coupling capacitors. Its annotated NET totals sum to
437.682 fF because coupling is incident on two nets. These conventions must
not be mixed or counted twice during reduction.

[`interconnect_freepdk45_8x8.yaml`](../../sram_compiler/config_yaml/interconnect_freepdk45_8x8.yaml)
is a measured **diagnostic override**. Compiler-wide defaults remain illustrative
because the current implementation reuses WL/BL geometry for different control,
decoder, periphery, replica and register-clock routes. The exact DSPF is the
reference for those routes; no all-route default promotion is claimed.

## TIME_CONTROL YAML refactor

[`time_control.yaml`](../../sram_compiler/config_yaml/time_control.yaml) contains
model defaults, every physical transistor width/length and clock stage size,
folding limits, delay stages and dummy loads, guard filter R/C policy, constructor
defaults, and fallback load/effort/scaling rules. Signal topology, gate identities
and connections stay in Python. The 45 nm observers and unusual 910 nm third
clock-stage NMOS remain exactly as supplied.

`SRAM_CONFIG` loads the file beside the global YAML, or from an explicit
`TIME_CONTROL` configuration path. `TimeControlFactory` and building blocks
append `control_config`, preserving positional argument order. Config values
are immutable, finite, and validated before construction. Explicit constructor
arguments and resolved driver-class settings retain precedence. The controller
config participates in the frozen peripheral baseline digest, CLI run identity
and saved run metadata.

Validation used 30 saved TIME_CONTROL blocks spanning 1x1, 8x8, 16x16, 128x64
and 512x256, read/write/sequence operations, and both buffer policies, plus seven
primitive blocks. **All 37 are byte-identical to the pre-refactor netlists.** A
full nominal mode-0 8x8 write deck is also byte-identical. Custom YAML values
reach nested register MOS, clock stages and guard capacitors; invalid values fail
and a changed controller configuration rejects an old frozen driver baseline.

## Validation limits and next steps

The [preserved data](../data/V2_2_7_RC_CALIBRATION.json) records the software and
macro waveform diagnostic outcomes. The comparison uses identical ideal 4 ns
controls for seven write/read operations, probes actual cell MOS terminals, and
checks row selection, storage/retention, sense outputs and WL/BL recovery. It
uses nominal TT models, KLU, 5 ps maximum steps and common DC NODESET hints that
are released for transient analysis. The first coupled post-layout attempt
timed out at 600 seconds and remains a failed attempt; a longer identical-deck
run is recorded separately. A partial transient never counts as a pass.
NODESET values are suggestions; initial data need not match between physical
variants. Storage/retention checks begin after writing the two tested rows.

The identical coupled deck completed with a 2,400-second limit and passed all
360 checks. The pre-layout and reduced-RC cases also passed 360 checks each.
This validates only the specified slow nominal macro sequence; it does not
certify controller timing or RC-model accuracy across operating corners. The
original failed timeout remains recorded separately. The CLI now allows 2,400
seconds per variant by default for this measured workload.

Software validation passed 88 targeted unit/integration checks, the 37 saved
netlist comparisons, full-deck identity, compilation and `git diff --check`.

No TIME_CONTROL waveform screen, 10T/mux physical layout, per-device/PVT yield
screen, large-array scaling qualification or full V2.2.7 screen is complete.
Compiler-wide default promotion requires dedicated route models or explicit
networks, physical controller/passive work when authorized, consistent local
stub/diffusion/coupling accounting, and functional waveform comparisons at the
frozen timing/sizing baseline. The 256x256 operating-point problem remains open.

The [layout guide](../../sram_compiler/layout/README.md) contains reproducible
commands. Python regression scripts remain local and ignored; tracked manifests
and evidence are retained for fresh checkouts.
