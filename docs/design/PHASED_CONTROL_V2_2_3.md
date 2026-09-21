# V2.2.3: supported envelope, joint-dimension clocks, one access deadline

> **No V2.2.3 screen has been run. There is no V2.2.3 evidence record.**
> This release changes the clock policy, the compiler's runtime data deadline
> and the waveform checker, all of which are hashed sources of the screen, so
> [`docs/data/PHASED_CONTROL_V2_2_2.json`](../data/PHASED_CONTROL_V2_2_2.json)
> certifies the V2.2.2 tree at commit `c6ea3aa` and **does not** certify this
> one. Nothing here may be cited as qualification until
> [the screen below](#running-the-screen) has been run and assembled.

V2.2.3 closes the coverage and contract defects that the V2.2.2 production
review recorded under
[what the screen does not cover](PHASED_CONTROL_V2_2_2.md#what-the-screen-does-not-cover),
except the stopped-clock item, which is deliberately left open. The controller
(`sram_compiler/subcircuits/time_generate.py`), the 6T and 10T cells, the
replica column, the distributed wires and the driver size classes are
unchanged, and no in-envelope array's clock moves.

## The supported envelope

The compiler is built for arrays up to **512 rows by 256 columns**. That is now
a declared, enforced property of the table rather than an implicit one:

```json
"supported_envelope": {"max_rows": 512, "max_cols": 256}
```

`load_timing_lookup` requires the field and requires the last anchor of every
ladder, shared and variant, to equal it. `resolve_timing` rejects a larger
array in every timing mode, including `mode: fixed`, because the array is out
of scope and not only its clock:

```
ValueError: 8x512 is outside the supported array envelope of 512 rows by 256 columns
```

Through V2.2.2 such a size continued the ladder geometrically and carried an
`extrapolated` flag that nothing had to read. No size inside the envelope is
extrapolated any more, so that flag is now always false and the resolver's
variant floor, which only ever mattered under extrapolation, is gone.

The 512-column class is removed from the shared ladder and from both variants,
because 512 columns is outside the envelope. The fifteen 8x512 cases of the
V2.2.2 screen are historical evidence: they remain in
`docs/data/PHASED_CONTROL_V2_2_2.json` and can still be read, but they can no
longer be regenerated from the table, and they are not in the V2.2.3 manifest.

## An array large in both dimensions pays for both

The V2.2.2 policy was `T = ceil_to_50ps(2 * max(row, column) * (1 + margin))`.
A maximum, not a sum: a 256x256 array was handed a 256x4 clock while carrying a
tall bitline and a wide wordline at once. No screened case had both dimensions
large enough to expose it, because every V2.2.2 size satisfied
`rows * cols <= 4096` and 32x32 was the largest array in a high class on both
axes.

V2.2.3 adds the smaller dimension's excess over its own first class:

```text
row_excess = row.half_period_ps    - row_classes[0].half_period_ps
col_excess = column.half_period_ps - column_classes[0].half_period_ps
half       = max(row.half_period_ps, column.half_period_ps)
             + min(row_excess, col_excess)
T          = ceil_to_50ps(2 * half * (1 + margin))
```

A dimension inside its first class has no excess, so the rule collapses to the
V2.2.2 maximum there. **Every array size the V2.2.2 screen covered is in that
position, so every screened clock is bit-identical** (`tests/test_timing_lookup.py`,
`test_an_array_large_in_both_dimensions_pays_for_both`):

| Screened | 2x2 | 8x4 | 16x8 | 32x32 | 64x16 | 128x8 | 256x4 | 512x4 | 8x64 | 8x128 | 8x256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Period (ns) | 9 | 9 | 9 | 9 | 9.5 | 16.5 | 20.25 | 27.75 | 10 | 12 | 14 |

What changes is every size above the first class in both dimensions, which had
no evidence under either rule:

| Joint (6T) | 48x20 | 64x64 | 64x128 | 128x64 | 128x128 | 256x128 | 256x256 | 512x256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Was (ns) | 9.5 | 10 | 12 | 16.5 | 16.5 | 20.25 | 20.25 | 27.75 |
| Now (ns) | 10 | 10.5 | 12.5 | 18.5 | 20.5 | 24.25 | 26.25 | 33.75 |

The envelope corner 512x256 therefore asks for 33.75 ns against the 27.75 ns a
512x4 array gets, and the manifest screens the rule up to 256x256.

## One access deadline

The compiler's generated `.MEASURE` cards checked read data at
`1 ns + (cycle + 0.7) T` while the waveform screen used `k + 0.68 T`. The gap
is 0.02 T: 180 ps at a 9 ns clock and 555 ps at 27.75 ns, and it is exactly the
gap that let a V2.2.1 8x512 read pass the runtime cards while still on the
wrong rail at the checker's deadline. V2.2.2 closed the detection side by
reporting `min_read_output_margin_ps` and left the runtime bar alone.

`ACCESS_DEADLINE = 0.68` now lives in
`sram_compiler/testbenches/sram_6t_core_MC_testbench.py` and the checker
imports it, so `VACCESS_ERROR` and `VHOLD_ERROR` and the screen accept exactly
the same traces. This tightens what `main_sram.py` and the per-device runner
accept: a deck that passed its `.mt0` at 0.7 T and failed at 0.68 T now fails
both.

## What the checker looks at

- **The sense bar scales with the rail.** It was a fixed 0.25 V, which is 28 %
  of a 0.9 V rail but 23 % of a 1.1 V one, so the fast corner was screened to a
  weaker bar than the slow corner. It is now `SENSE_MARGIN_FRACTION = 0.28` of
  VDD, which is the 0.9 V bar, so no point that passed before is relaxed. It
  remains a screening floor: this screen still does not measure the sense
  amplifier's own input offset under local mismatch.
- **More rows are probed on arrays too large to probe whole.** A 512x4 trace
  probed 4 of 512 rows: first, middle, last and target. It now probes the
  quarter points, every addressed row, and **each addressed row's immediate
  neighbours**, which are the rows a wordline or bitline excursion disturbs
  first and were exactly the rows nobody looked at. The rule is one function,
  `probed_rows`, which the runner and the checker both call, because the
  checker fails closed on a missing probe and two copies would drift.
- **`min_storage_polarity_margin_v` is documented for what it is:** a reversal
  bound against a fixed 0.5 VDD boundary, which is not the trip point of every
  cell. The 10T cell is a Schmitt-trigger cell read differentially through the
  same access transistors as the 6T, so it disturbs its storage node further at
  a correspondingly higher trip point; the V2.2.2 record's 0.260 V for 10T
  against 0.291 V for 6T is an artifact of the fixed boundary, not a stability
  deficit. What bounds recovery is the 0.1 VDD rail-error check after the
  access deadline, which is unchanged.

## The screen

`tests/spice/v223_cases.json` is **285 cases**: the 239 V2.2.2 cases inside the
envelope, unchanged, plus 46 new ones. `tests/spice/v223_negative_cases.json`
is **6 negative controls**, now tracked rather than living in an ignored queue.

| Gap from the V2.2.2 review | What the manifest adds |
|---|---|
| No array large in both dimensions | 64x64 (6T, 6T/mux, 10T/mux, per-device, FF), 64x128, 128x64, 128x128 (nominal and per-device), a 128x128 changed-address read, and nominal 256x128 and 256x256. Largest array 65,536 cells against 4,096 before. |
| Corner confounded with voltage and temperature | A second operating point for every skewed corner: FF at 0.9 V/125 C, SS at 1.1 V/-40 C, SF at 0.9 V/-40 C (write-ability, cold), FS at 0.9 V/125 C (read stability, hot), at 8x4, 16x8 (per-device), 32x32 and 128x8. |
| TT screened only at 8x4 | TT 1.0 V/25 C at 32x32, 64x16, 128x8, 256x4, 8x256 and 64x64, plus a 32x32 per-device draw. |
| One mismatch draw at the large arrays | Two more seeds at 256x4 and 512x4, one more at 8x256 and 128x8; 85 per-device cases in all. |
| Complementary column data only at 8 columns | Checkerboard patterns at 64, 128 and 256 columns, and at 64x64, with one per-device 8x256 draw. |
| One negative control, ten of twelve margins never failed | Six controls at 8x4, 32x32, 64x64, 128x8, 8x256 and 512x4, each at a period the lookup would not grant, so different margin families are the ones that break. |

Resulting coverage: SS 144, FF 52, FS 35, SF 35, TT 19; 85 per-device; 17 array
shapes; 29 arbitrary patterns, 5 of them above 8 columns.

### Running the screen

```bash
python3 -m tests.spice.phased_access \
  --output outputs/validation/V2.2.3-phased/main \
  --xyce "$(command -v Xyce)" --workers 4
python3 -m tests.spice.phased_access \
  --cases tests/spice/v223_negative_cases.json \
  --output outputs/validation/V2.2.3-phased/negative \
  --xyce "$(command -v Xyce)" --workers 4   # must exit non-zero
```

Estimated cost, from the V2.2.2 per-case runtimes
(`elapsed ≈ 1.03e-5 h per cell-nanosecond of transient`): **about 560 solver-hours**,
roughly 110 h for the retained cases, 440 h for the new ones and 8 h for the
controls. It is dominated by two cases: 256x256 at about 157 h and 256x128 at
about 73 h, both nominal. The 256x256 deck is 55 MB and its waveform about
4.5 GB, comparable to the 8x512 case the V2.2.2 campaign already handled. A
per-device draw at 256x256 would need a roughly 7 GB model card and is not in
the manifest.

Every case in both manifests was checked to name one directory, resolve a clock
inside the envelope and carry a well-formed pattern
(`tests/test_phased_waveforms.py`, `ScreenManifestTests`), and decks were
generated for 8x4, 32x32, 64x64, 8x64, 128x8, 128x128 and 256x256 to confirm
that the runner and the checker agree on the probe set at the new sizes. No
case has been simulated.

## Still open

- **A stopped or gated clock**, deliberately. PRE is
  `!(clk_bar & wordline_off & enables_off)`, so precharge is a clock-low
  function: a clock stopped high parks the array with floating bitlines and
  open sense pass gates, and a clock stopped low parks it precharged. The
  contract still assumes a free-running 50 % duty cycle and says nothing about
  the clock stopping.
- Extracted metal, half-selected writes and yield estimation
  (`docs/README.md`).
- The sense amplifier's own input offset under local mismatch, which the
  0.28 VDD bar stands in for.
- The sense margin at 512 rows was 0.467 V in V2.2.2 against a 0.25 V bar, the
  end of a steep trend (0.90 / 0.86 / 0.80 / 0.67 / 0.47 V at 32 / 64 / 128 /
  256 / 512 rows). 512 rows being the declared maximum is what bounds that
  trend; it is not headroom, and the V2.2.3 screen should be read for whether
  the joint-dimension clocks hold it.
