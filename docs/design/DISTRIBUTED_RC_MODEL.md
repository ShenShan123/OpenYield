# Distributed wordline and bitline RC — V2.0.7

The default remains the star topology. Distributed wiring is opt-in and
requires explicit metal geometry; the compiler supplies no extracted defaults.
The [implementation plan](DISTRIBUTED_RC_PLAN.md) records the work sequence.

Enable it through `global.yaml`'s `interconnect` mapping, the appended
`interconnect=` testbench argument, or the CLI:

```bash
python3 -m sram_compiler.per_device_mc.run --rows 4 --cols 4 \
  --variation-mode nominal --operation 'read&write' \
  --interconnect-config sram_compiler/config_yaml/interconnect_example.yaml
```

This generates a deck. Append `--run-xyce` to simulate. The
[example configuration](../../sram_compiler/config_yaml/interconnect_example.yaml)
uses illustrative 1-ohm / 0.1-fF pitches for diagnostics; replace them with
technology/layout extraction before drawing physical timing or yield conclusions.
For normal MC, omit `--variation-mode nominal`: per-device mismatch remains
the default.

## Configuration and topology

| Field | Meaning |
|---|---|
| `mode` | `star` or `distributed` |
| `cell_pin_rc` | Keep local cell WL/BL/BLB stubs; defaults to true for star and false for distributed |
| `wl`, `bl` | Separate geometry mappings; BL and BLB use the same geometry |
| `layer` | Technology/layer identifier, included in the physical fingerprint |
| `pitch_m`, `width_m` | Cell pitch along the wire and metal width, in metres |
| `sheet_resistance_ohm` | Metal sheet resistance in ohms per square |
| `capacitance_f_per_m` | Total ground capacitance per metre, excluding cell device capacitance |
| `sections_per_half_pitch` | Positive integer subdivision count; increase to check convergence |

For each pitch, `R = sheet_resistance_ohm * pitch_m / width_m` and
`C = capacitance_f_per_m * pitch_m`. All geometry/R/C values must be finite
and positive. Unknown fields are rejected. Wire coupling between adjacent
lines is not included in this first implementation.

WL drivers connect at column zero; bitline precharge, write, mux and sense
circuits connect at row zero. Cells sit at half-pitch centers. Each row has
`num_cols * wl.pitch_m` wire length and each bitline has
`num_rows * bl.pitch_m`. There is a half pitch before the first cell and after
the last cell. Every resistor segment has half its capacitance at each end;
subdivision conserves total R/C.

Inside the array, taps are `WL{row}_tap{col}`, `BL{col}_tap{row}` and
`BLB{col}_tap{row}`. Opposite wire endpoints are `WL{row}_far`,
`BL{col}_far` and `BLB{col}_far`. The public array ports retain their names.

`w_rc` continues to control local storage-node and peripheral RC. Distributed
metal remains present even with `w_rc=False`. With `w_rc=True`, the default
distributed mode removes the generic cell WL/BL/BLB stubs while retaining Q/QB
and peripheral stubs. Set `cell_pin_rc: true` only when those additional
branches represent separately budgeted local parasitics.

## Replica, measurements and equivalent cells

Distributed mode uses the real wordline-driver topology for RWL and requires
matched replica/canonical read operation. The replica column has exactly the
array row count; its K active cells occupy the far rows and the last K RWL
taps. Dummy gates fill the remaining RWL taps. Replica bitlines have the same
wire length as array bitlines, and the replica also carries the corresponding
mux (when enabled), sense-input circuit, precharge and write-driver load.
The TIME precharge guard observes `RWL_far`.

Transient wordline and write-bitline measurements use the selected cell's
local terminal. Read swing uses the actual sense-amplifier input; restoration
uses the far bitline endpoint. Additional waveform probes expose near/far
line endpoints, sense inputs and storage nodes. `cell_probe()` and
`sense_input_probe()` expose the same node names to external scorers.

Equivalent modes 1–4 retain every wire segment and attach each omitted cell's
five-capacitor network at its local taps. Optional local stubs remain separate;
write-power approximations follow local WL voltage. Extraction includes the
configured storage-node RC and uses the effective corner, VDD and temperature,
with the same 27 C nominal model temperature as the main simulator. The cache
includes selected model contents and extraction settings.

Equivalent cells remain approximations: they do not reproduce arbitrary cell
state changes, conductive loading or independent mismatch in omitted devices.
Their accuracy must be compared with full-real arrays for the intended use.
Equivalent extraction requires numeric geometry; SPICE dimension sweeps use
`real_cell_mode=0`.

## Compatibility and qualification

V2.0.7 corrects the testbench's former 10-ohm default to the common
100-ohm / 1-fF local RC default and forwards custom values everywhere.
Compared with V2.0.6 commit `029626b`, all eight checked 4x4 6T/10T read/write
star decks with RC off or explicit 100 ohm are byte-for-byte identical.
The four decks using the old default differ only at ten replica-path resistors,
now 100 ohm. Custom RC settings and equivalent extraction intentionally change.

Frozen driver baselines and qualification identities include the wire model,
geometry, subdivisions and local RC configuration. Changes cannot silently
reuse an earlier baseline. CLI run directories include the new topology and
compiler version. Runtime scoring fingerprints cover compiler sources and
work without the ignored `dev/` directory.

Historical V2.0.5 artifacts and sizing coefficients retain their labels.
Distributed support does not qualify those coefficients for an extracted
array, and no sizing-table record is promoted by this implementation.
See the [V2.0.7 changelog](../CHANGELOG.md) for the validation performed.
