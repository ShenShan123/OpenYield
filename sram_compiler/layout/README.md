# FreePDK45 layout and RC calibration

V2.2.7 is in development. The complete **8×8 6T macro** now includes
TIME_CONTROL, input registers, physical RC filters and the output latch, as well
as the array, decoder, replica and periphery. See the
[complete reference](#complete-macro-and-controller-reference) below. The initial
array/macro measurements in the first sections excluded TIME_CONTROL and used
ideal controls; their evidence retains that historical scope. Testbench sources
and startup clamps remain stimulus infrastructure outside both physical layouts.

KLayout's Python library builds actual transistor, contact, via and metal
geometry; Calibre checks the installed FreePDK45 DRC/LVS rules and extracts
distributed **R+C+CC** DSPF with source instance names. FreePDK45 is a predictive
technology, compiled from published information; these are layout-derived
references for that kit, not measured silicon or a manufacturing foundry PDK.
See [NC State's kit description](https://eda.ncsu.edu/freepdk/freepdk45/) and
[Siemens' xRC documentation](https://www.siemens.com/en-gb/products/ic/calibre-design/circuit-verification/xrc/).

The fixed generator preserves the V2.2.6 8×8 reference MOS sizes. It is not a
parameterized physical SRAM compiler. Official LVS must pass against the supplied
compiler deck before extraction evidence is accepted. No equivalent-mode or
per-device-mismatch layout is claimed.

## Reproduce

Run from the repository root, with Python 3.9, PySpice, NumPy, SciPy and PyYAML
from `environment.yml`. Layout generation additionally needs the KLayout Python
package (this run used **0.30.7**); the standalone KLayout application alone does
not provide that Python import. Licensed Calibre xRC and the external FreePDK45
rules are required for extraction. PDK files are not bundled in OpenYield.

```bash
command -v calibre
command -v Xyce
python -c 'import klayout.db, scipy'

python -m sram_compiler.per_device_mc.run --rows 8 --cols 8 \
  --operation write --variation-mode nominal --mc-runs 1 \
  --real-cell-mode 0 --no-waveform \
  --output-dir outputs/rc_calibration/source
```

Use the generated `deck.sp` as `DECK` and set `PDK_DIR` to the directory containing
`ncsu_basekit/techfile/calibre/`. The following variables deliberately refer to
external inputs rather than machine-specific repository defaults:

```bash
export PDK_DIR=/path/to/FreePDK45
DECK=/path/to/generated/deck.sp

python -m sram_compiler.layout.freepdk45.run --deck "$DECK" \
  --pdk-dir "$PDK_DIR" --output-dir outputs/rc_calibration/macro --extract
python -m sram_compiler.layout.freepdk45.run --deck "$DECK" \
  --pdk-dir "$PDK_DIR" --output-dir outputs/rc_calibration/array --array-only --extract
python -m sram_compiler.layout.freepdk45.analyze \
  --macro-dir outputs/rc_calibration/macro \
  --array-dir outputs/rc_calibration/array \
  --output-dir outputs/rc_calibration/analysis
python -m sram_compiler.layout.freepdk45.compare \
  --macro-dir outputs/rc_calibration/macro --pre-layout-deck "$DECK" \
  --models tran_models/models_TT.spice --xyce Xyce \
  --output-dir outputs/rc_calibration/comparison
```

Each output directory must be empty. Receipts record GDS, schematic, source deck,
generator, Calibre executable and DRC/LVS/xRC rule hashes. Failed stages preserve
their logs and record failure. Analysis rejects failed DRC/LVS, grounded-only
extraction, mismatched array/macro sources or rules, and changed GDS/DSPF/schematic
hashes. `--extract` without `--array-only` produces hierarchical and flat macro GDS;
without `--extract` the output is explicitly unverified.

The xRC sequence includes hierarchical LVS to generate the source/layout cross
reference, followed by `-phdb`, `-pdb -rcc`, and `-fmt -all`. The ground node is
Calibre's default `0`; the comparison ties VSS to `0` explicitly. Default kit
reduction settings remain in effect, so retain rule hashes and logs. Coupling
capacitors, contact/poly/via resistances and extracted MOS diffusion areas and
perimeters remain in the exact DSPF reference.

## Reduction and back-annotation

The standalone array needs an explicit M4 VDD strap because its supply stripes
are normally joined by the macro's column periphery. The earlier local tile
omitted that strap. The corrected tile passes Calibre DRC and LVS independently.

The analysis solves each extracted resistor network as a conductance matrix and
fits resistance against actual mirrored contact positions. A shortest path is
not an electrical resistance when there are parallel paths. WL gate contacts
span eight column pitches. Shared BL contacts span six row pitches; BLB contacts
span eight. Applying six to both polarities overestimates BLB wire resistance.

The initial reduced profile is:

| Route | Layer | Cell pitch | R per pitch | C per pitch |
|---|---|---:|---:|---:|
| WL | M2, effective track plus landing environment | 1.365 µm | 4.654 Ω | 0.783 fF |
| BL/BLB mean | M3, effective track plus landing environment | 0.400 µm | 1.429 Ω | 0.128 fF |

For the existing `WireRC` API, `R_pitch = R_sheet * pitch / width` and
`C_pitch = C_per_m * pitch`, with a 70 nm track width. Effective sheet resistance
absorbs the actual landing geometry; it is not a replacement for the kit's metal
sheet-resistance table. The fit intercept retains port/contact/poly pin access
resistance separately and is **not** folded into per-pitch metal resistance.

The C reduction uses Calibre NET incident totals divided by eight. These include
coupling with neighbours held quiet. The exact DSPF preserves dynamic coupling;
NET totals count a coupling capacitor on both participating nets. Summing these
totals is not the sum of unique physical capacitor elements. The reduction is
also a single-size fit and includes endpoint loading; it does not establish a
scaling law for larger arrays.

[`interconnect_freepdk45_8x8.yaml`](../config_yaml/interconnect_freepdk45_8x8.yaml)
is an optional diagnostic override. Generate a second nominal write deck with
`--interconnect-config sram_compiler/config_yaml/interconnect_freepdk45_8x8.yaml`,
then pass it as `--candidate-deck` to the comparison. That runner refuses a
candidate with changed macro MOS models, widths, lengths or counts. It retains
the pre-layout macro's actual R/C elements and replaces the macro with the exact
source-named DSPF in the post-layout case, respecting DSPF port order and probing
physical cell terminals.

The comparison uses identical ideal 4 ns control slots for seven operations:
write two rows, read both, overwrite one, read both again. It checks wordline
selection, stored values and retention, sense outputs, and wordline/bitline
recovery. KLU uses common DC NODESET hints on the 64 cells; these are suggestions,
not forced initial data. Storage/retention checks cover the two rows after they
have been written. The hints are released for transient analysis. It is a slow macro diagnostic, **not** the
TIME_CONTROL waveform screen, full-array PVT/yield qualification, or a default
promotion gate. Results and full CSV waveforms remain under the chosen output.
The initial coupled reference reached a 600-second timeout; the identical deck
completed with a 2,400-second limit and passed all 360 checks. Pre-layout and
reduced-RC cases also passed 360 checks each. The CLI therefore defaults to a
2,400-second limit per variant; solver timing depends on the host.

## Complete macro and controller reference

The subsequent [full macro calibration record](../../docs/design/FULL_RC_CALIBRATION_V2_2_7.md)
records the user-authorized controller legalization and the complete physical
implementation. The original array-only measurements above remain historical
reference evidence. The complete macro now includes TIME_CONTROL, all input
registers, four intentional physical RC filters and the output latch. Official
Calibre DRC reports zero violations and LVS reports `CORRECT`; LVS treats the
physical passives by DC reduction and xRC independently verifies their values.

The new estimated model composes 709 separate per-net R/C files, explicit
cross-net coupling, and independent VDD/VSS power meshes, then back-annotates
junction geometry. The default YAML exposes each network and its own scales. It is calibrated against the
full xRC result, rather than reusing one WL/BL pair for every route. It protects
all MOS/port terminals, preserves DC series resistance, and retains all coupling.
The complete nominal physical waveform comparison is a scoped diagnostic; it
does not certify the full release screen or other array sizes and topologies.

Generate a nominal 8×8 read/write baseline with explicit illustrative wiring:

```bash
python -m sram_compiler.per_device_mc.run --rows 8 --cols 8 \
  --operation 'read&write' --variation-mode nominal \
  --real-cell-mode 0 --corner TT --vdd 1 --temperature 25 --period 9e-9 \
  --interconnect-config sram_compiler/config_yaml/interconnect_illustrative.yaml \
  --output-dir outputs/rc_calibration/native
```

Use the generated `deck.sp` as `DECK` and a FreePDK45 installation as `PDK_DIR`:

```bash
python -m sram_compiler.layout.freepdk45.full_macro --deck "$DECK" \
  --output-dir outputs/rc_calibration/full_layout
python -m sram_compiler.layout.freepdk45.verify_full \
  --layout-dir outputs/rc_calibration/full_layout --pdk-dir "$PDK_DIR" \
  --output-dir outputs/rc_calibration/full_extraction
python -m sram_compiler.layout.freepdk45.reduce_rc \
  --layout-dir outputs/rc_calibration/full_layout \
  --extraction-dir outputs/rc_calibration/full_extraction \
  --maximum-series-ohm 500 --output-dir outputs/rc_calibration/reduced
python -m sram_compiler.layout.freepdk45.compare_full --deck "$DECK" \
  --profile outputs/rc_calibration/reduced/profile.json \
  --extraction-dir outputs/rc_calibration/full_extraction \
  --output-dir outputs/rc_calibration/full_comparison --execute --xyce Xyce
```

Use new output directories. The comparison preserves compiler stimuli and runs
four accesses with final recovery, ending 5 ps before the fifth capture. It uses
common complete logical initial conditions, UIC, 5 ps output, a 20 ps integration
ceiling, and explicit KLU. Completion of Xyce alone does not mark a profile
validated. The independent Python waveform checker remains local and ignored;
[`v227_full_rc_cases.json`](../../tests/spice/v227_full_rc_cases.json) preserves
the protocol in a fresh clone. Raw layouts, DSPF, decks, attempts and waveforms
remain under ignored `outputs/`.

After waveform validation, `split_rc` exports separate net files without changing
any reduced resistor/capacitor value or ordering:

```bash
python -m sram_compiler.layout.freepdk45.split_rc \
  --profile outputs/rc_calibration/reduced/profile.json \
  --output-dir outputs/rc_calibration/per_net
```

Draft exports remain unvalidated. The checked-in schema-2 profile includes
physical/waveform receipts and exact scope; compiler loading checks these,
network hashes, device geometry, controller configuration and model hash.
The default calibrated compiler passes 1,241 complete waveform checks and has
3.40 ps mean / 7.62 ps maximum edge error against full coupled xRC. The known
100 µA power-mesh diagnostic gives 7.36 mV VDD drop and 8.23 mV ground rise.
See the complete record for initialization, passive LVS policy and scope.
