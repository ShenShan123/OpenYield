# FreePDK45 layout and RC calibration

V2.2.7 is in development. The current physical reference is an **8×8 6T macro**:
array, decoder, wordline drivers, write buffers and drivers, precharge, sense
amplifiers, replica column and replica wordline loads. TIME_CONTROL, address/data
registers, startup clamps and testbench sources remain outside the layout.
The user chose frozen sizing and deferred full controller layout. This is not a
full OpenYield layout or an all-route default calibration.

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

## Remaining work before changing compiler-wide defaults

OpenYield currently reuses `wl` for RWL, column controls, mux selects and the
data-register clock, and `bl` for replica bitlines, decoder buses, WL_EN and three
peripheral wire pitches. The full macro extraction inventories those networks;
they have different lengths, contacts, coupling and capacitance. Array WL/BL
numbers alone cannot calibrate all of them. The default
[`interconnect_example.yaml`](../config_yaml/interconnect_example.yaml) therefore
remains illustrative.

TIME_CONTROL also has six 45 nm NMOS widths below FreePDK45 Active.1's 90 nm
minimum, twelve off-grid widths, and four explicit RC filters (261–696 Ω / 1 fF).
The user selected intentional physical passives for those filters, then chose
to freeze sizing and defer full controller layout when these geometry conflicts
were identified. Neither devices nor filters have been silently changed.

Completion requires controller legalization and passive implementation when
authorized, dedicated route classes or explicit networks, local-stub and
diffusion/coupling accounting, size and cell-topology coverage, and waveform
comparison at the frozen timing/sizing baseline. Full controller, register-clock,
10T and mux layouts are absent. No V2.2.7 release, full waveform screen or
all-route default calibration is certified by the current artifacts.

The [development record](../../docs/design/RC_CALIBRATION_V2_2_7.md) and
[preserved data](../../docs/data/V2_2_7_RC_CALIBRATION.json) carry the measured
results and exact scope. Local Python tests remain ignored, as required by
`AGENTS.md`; a fresh clone retains the physical workflow and SPICE case manifest.
