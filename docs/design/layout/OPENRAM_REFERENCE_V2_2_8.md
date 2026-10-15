# OpenRAM 32×32 FreePDK45 reference run

The user-supplied checkout at `/proj/workarea/user5/OpenRAM` was executed at
commit `b2b069ce119d1488cbe6883b2240bceb5c7ce29a`. Its source tree remains unchanged.
The completed reference is **32 words × 32 bits**, one bank, one read/write
port and one word per row, using FreePDK45, TT, 1 V and 25 °C.

The native compiler produced GDS, LEF, source and LVS SPICE, Verilog, a datasheet,
and analytical Liberty output. No SPICE characterization or waveform validation
ran. The primary run enabled perimeter signal pins and retained multiple
must-connect supply pins, matching the supplied FreePDK45 example's
`route_supplies = False` policy. It does not include a completed top-level
power ring. A separate optional power-ring run was interrupted while building
the supply routing graph; it is not accepted evidence.

## Artifacts and geometry

For the closest bank-level placement reference, use the exported bank module:
[bank GDS](../../../outputs/openram_reference/32x32_04/bank_reference/openram_1rw_32x32_freepdk45_bank.gds)
and [bank preview](../../../outputs/openram_reference/32x32_04/bank_reference/reference.png).
It measures **40.070 × 61.330 µm** (**2,457.4931 µm²**), with the decoder and
wordline drivers on the left of the array. This export selects the actual
bank hierarchy from the normalized generated macro; it does not regenerate or
alter the bank's physical shapes. It excludes the macro-level controller,
input registers and perimeter signal escapes.

Artifacts are under `outputs/openram_reference/32x32_04/macro/`:

- [Native GDS](../../../outputs/openram_reference/32x32_04/macro/openram_1rw_32x32_freepdk45.gds)
- [Native layout preview](../../../outputs/openram_reference/32x32_04/macro/reference.png)
- [LEF](../../../outputs/openram_reference/32x32_04/macro/openram_1rw_32x32_freepdk45.lef)
- [Source SPICE](../../../outputs/openram_reference/32x32_04/macro/openram_1rw_32x32_freepdk45.sp)
- [Datasheet](../../../outputs/openram_reference/32x32_04/macro/openram_1rw_32x32_freepdk45.html)

The measured GDS bounding box is **115.605 × 76.220 µm**, or
**8,811.4131 µm²**, including the macro's peripheral geometry and pin escape
routing. The array's actual placement pitch is **0.705 µm per column** and
**1.365 µm per row**. The decoder and wordline-driver array are placed on its
left side. The decoder cell's geometry bounds are 8.865 × 43.9225 µm; this
includes well/rail extensions around the nominal 32-row span.

This is an architectural/layout reference, not an equal-circuit area comparison
with OpenYield. OpenRAM uses different bitcell geometry, sizing and fingering,
and includes its controller and input registers. At the time of this run, OpenYield used a 1.365 µm column / 0.400 µm row
pitch. The subsequent [OpenRAM-pitch bank](OPENRAM_PITCH_V2_2_8.md) adopts the
reference's 0.705 µm column / 1.365 µm row pitch; TIME_CONTROL remains deferred.

## Physical check status

The original GDS is preserved. Calibre rejects one zero-area M1 boundary in the
supplied `sense_amp` library cell at (0, 1.4525) µm. A separate normalized copy
removes that single zero-area object; it removes no physical area. Its audit
retains both GDS hashes and the removed cell/layer/location.

The normalized copy has **zero Calibre DRC violations**, but LVS is
**`INCORRECT`**, reporting different net counts and connectivity discrepancies.
The OpenRAM adapter reports six LVS errors. The check used the installed NCSU
FreePDK45 rules and OpenRAM's own Calibre adapter, including ignored ports and
virtual connections for `vdd`/`gnd` when supply routing is disabled. These
settings and the generated runsets are retained. This reference is therefore
**generated, not physically qualified**; no release or RC-default promotion
follows from it.

The separately exported **bank module** passes **zero DRC violations** and
**LVS `CORRECT`** against the bank subcircuit in the original generated LVS
SPICE file. This is a scoped check using the same OpenRAM adapter policy:
ignored ports and virtual `vdd`/`gnd` connections. It verifies internal bank
connectivity under that supply policy, not a completed physical power network.
The parent macro's LVS failure is preserved separately and is not promoted by
the bank result.

The checkout does not bundle the licensed Calibre rule files at its default
technology paths. Verification supplied the external rule locations and a
resolved layer-map path in the local verification driver, without modifying
OpenRAM's source. [Recorded evidence](../../data/OPENRAM_REFERENCE_V2_2_8.json)
contains configuration, revision and artifact/rule hashes.

## Reproduction

Use a Python environment with the repository requirements, including NumPy,
SciPy and scikit-learn. This run used a local virtual environment with inherited
OpenYield dependencies plus scikit-learn 1.6.1 and coverage. All local setup and
failed attempts remain under ignored `outputs/openram_reference/`.

A configuration equivalent to the completed reference is:

```python
word_size = 32
num_words = 32
words_per_row = 1
num_banks = 1
num_rw_ports = 1
num_r_ports = num_w_ports = 0
tech_name = "freepdk45"
process_corners = ["TT"]
supply_voltages = [1.0]
temperatures = [25]
nominal_corner_only = True
use_nix = False
netlist_only = False
route_supplies = False
perimeter_pins = True
check_lvsdrc = False
analytical_delay = True
use_pex = False
output_name = "openram_1rw_32x32_freepdk45"
output_path = "outputs/openram_reference/new_run/macro"
```

Run from the OpenYield root with the external kit environment loaded:

```bash
export OPENRAM_HOME="$OPENRAM_ROOT/compiler"
export OPENRAM_TECH="$OPENRAM_ROOT/technology"
python "$OPENRAM_ROOT/sram_compiler.py" path/to/config.py
```

`use_nix = False` selects the already-installed tools instead of OpenRAM's
optional Nix bootstrap. The compiler checks were deliberately separate from
artifact generation; a successful compiler exit does not imply DRC/LVS success.
