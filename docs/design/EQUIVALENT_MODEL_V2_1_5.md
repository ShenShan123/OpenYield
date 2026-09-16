# V2.1.5 equivalent array model: merge into the compiler and measured accuracy

The equivalent array model replaces the cells an addressed access does not
touch with their extracted five-capacitor network and a static-leakage load at
the same local wire tap. V2.1.5 moves the top-level `equivalent_modeling/`
directory into `sram_compiler/equivalent_modeling/` and makes the model a
simulation input with the same shape as the `interconnect` block, then measures
what the approximation costs. The model itself (`SRAMCellParasiticTester`, the
loads and the WL-controlled static-power source) is unchanged and still lives
in `sram_compiler/subcircuits/sram_cell_add_equivalent.py`.

Usage and the model description are in the
[equivalent-model guide](../../sram_compiler/equivalent_modeling/README.md).

## 1. What the merge changed

| Was | Is |
|---|---|
| `equivalent_modeling/README.md` | `sram_compiler/equivalent_modeling/README.md` |
| `equivalent_modeling/main_sram.py` + `run.sh` + `test.ipynb` | `python3 -m sram_compiler.equivalent_modeling.compare` |
| `real_cell_mode` validated inside the core classes only | `resolve_equivalent()` / `EquivalentConfig`, validated once and recorded |
| mode only reachable from Python or `--real-cell-mode` | `equivalent:` block in `global.yaml`, overridden by the option or the argument |

`real_cell_mode` now defaults to `None` in `Sram6TCoreTestbench`,
`Sram6TCoreMcTestbench`, `sram_compiler.per_device_mc.run` and `main_sram.py`,
which resolves to the YAML block exactly as an omitted `interconnect` does. The
tracked default is mode 0, so no existing deck moves: against a detached
`acec3be` worktree, nine reference decks (6T and 10T, nominal and per-device,
with and without a mux) regenerate byte-identically from the merge alone
(`outputs/validation/V2.1.5-10t/deck-comparison/merge_vs_acec3be.json`).

The resolved selection is recorded wherever the run is recorded: `summary.json`
and `*.variation.json` gain an `equivalent` block, and the local validator's
`result.json` carries it too. `full_device_coverage` is still only true for
mode 0 with per-device mismatch, because a per-device sample over an equivalent
array draws local mismatch only for the cells that mode keeps real.

The old experiment driver had stopped working: it called `enable_mc=` and
`Sram6TCoreMcTestbench(...)` positionally in a shape the compiler no longer
accepts, and its notebook read a results directory that no longer exists. The
replacement is a documented module with the compiler's own configuration
loader, argument validation and provenance.

## 2. Measured accuracy and runtime

Mode 0 is the reference. Every other mode runs the same array, wires, local RC,
driver classes, clock, corner, supply, temperature and (nominal) models, so the
difference is the approximation alone. Read and write transients at TT 1.0 V /
25 °C, `w_rc=True` with the illustrative 100 Ω / 1 fF stubs, distributed wires
with the default geometry, target cell at the last row and column,
`variation nominal`, seed 20260915, one sample.

### 16x16 6T array, all five modes

| Mode | Real cells | Read delay (ps) | Read power (µW) | Read static (µW) | Write delay (ps) | Write power (µW) | Runtime (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | full array (reference) | 465.8 | 237.83 | 4.76 | 156.1 | 329.67 | 1544 |
| 1 | cross (target row + column) | 465.3 (-0.09 %) | 237.55 (-0.12 %) | 3.48 (-26.89 %) | 155.7 (-0.28 %) | 330.69 (+0.31 %) | 1298 (-15.95 %) |
| 2 | target row | 465.3 (-0.11 %) | 237.40 (-0.18 %) | -0.34 (-107.17 %) | 156.4 (+0.20 %) | 330.55 (+0.27 %) | 1192 (-22.80 %) |
| 3 | target column | 464.9 (-0.20 %) | 220.98 (-7.08 %) | 4.93 (+3.45 %) | 155.4 (-0.48 %) | 302.01 (-8.39 %) | 1218 (-21.10 %) |
| 4 | target cell | 464.8 (-0.20 %) | 220.88 (-7.13 %) | 4.16 (-12.58 %) | 156.2 (+0.02 %) | 302.92 (-8.12 %) | 1248 (-19.15 %) |

Percentages are relative to mode 0 of the same array. Runtime is wall clock for
the whole point, so for modes 1-4 it includes the Xyce parasitic and
static-current extraction that netlist generation performs.

### 8x8 6T array, where the extraction dominates

| Mode | Real cells | Read delay (ps) | Read power (µW) | Read static (µW) | Write delay (ps) | Write power (µW) | Runtime (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | full array (reference) | 449.2 | 147.99 | 2.35 | 136.5 | 188.28 | 96 |
| 1 | cross (target row + column) | 449.2 (+0.01 %) | 147.66 (-0.23 %) | 0.20 (-91.56 %) | 136.7 (+0.13 %) | 187.79 (-0.26 %) | 126 (+30.75 %) |
| 4 | target cell | 449.1 (-0.02 %) | 143.52 (-3.02 %) | 2.94 (+25.23 %) | 136.6 (+0.06 %) | 178.35 (-5.27 %) | 108 (+12.29 %) |

At 8x8 the equivalent modes are *slower* than the full array: the netlist-time
Xyce extraction costs more than the removed transistors save. Delay accuracy is
unchanged (within 0.02 %), and the static term is again the worst metric
(−92 % in mode 1).

### 32x32 6T array, the cross and target-cell modes

| Mode | Real cells | Read delay (ps) | Read power (µW) | Read static (µW) | Write delay (ps) | Write power (µW) | Runtime (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | full array (reference) | 494.8 | 426.42 | 9.67 | 177.0 | 634.00 | 5056 |
| 1 | cross (target row + column) | 494.1 (-0.14 %) | 425.11 (-0.31 %) | 10.90 (+12.76 %) | 176.1 (-0.51 %) | 635.51 (+0.24 %) | 3199 (-36.73 %) |
| 4 | target cell | 494.1 (-0.14 %) | 372.22 (-12.71 %) | 13.32 (+37.74 %) | 177.6 (+0.37 %) | 564.38 (-10.98 %) | 2497 (-50.62 %) |

### The speed-up grows with the array, the error does not

Runtime of the cross mode (1) against the full array: **+31 % at 8x8, −16 % at
16x16, −37 % at 32x32**; the target-cell mode reaches −51 % at 32x32. The
extraction cost is roughly fixed while the removed transistors grow with the
array, so this is the expected shape — and it is why a small array is the wrong
place to judge the feature. Read-delay error meanwhile stays between −0.20 %
and +0.01 % at every size, and the average-power split between the modes that
keep the whole target row (1 and 2) and the modes that do not (3 and 4) holds:
mode 4's read power is −3.0 % at 8x8, −7.1 % at 16x16 and −12.7 % at 32x32,
growing with the fraction of the accessed row that is replaced.

### What the numbers say

- **Delay tracks the reference.** Every mode reproduces the read delay within
  0.21 % and the write delay within 0.48 %. This is the metric the equivalent
  model was built for and it holds in all four modes.
- **Average power depends on which cells stay real.** Modes 1 and 2 keep the
  whole target row, so the switching current of every accessed cell is still
  transistor-level and average power stays within 0.4 %. Modes 3 and 4 keep only
  the target column (or cell), so the rest of the accessed row contributes only
  an RC and a WL-controlled leakage source: read and write average power fall by
  7 to 8 %. Treat modes 3 and 4 as delay models, not power models.
- **Array static power is not modelled well.** The static term is small
  (4.8 µW against 238 µW average) and the equivalent leakage load misses it by
  −27 % in mode 1 and by −107 % in mode 2, where it changes sign
  (−0.34 µW against +4.76 µW). No mode should be used to compare static or
  standby power.
- **The speed-up is real but not the historical "tens of times".** It grows
  with the array — +31 % (slower) at 8x8, −16 % at 16x16, −37 % at 32x32 for
  the cross mode — because the netlist-time extraction cost is roughly fixed
  while the removed transistors are not. At 16x16 the ordering is not even
  monotonic in the number of removed cells (mode 2 is the fastest, mode 4 is
  not), which is the same effect. The earlier documentation's order-of-magnitude
  figures came from a topology that aggregated the omitted cells' wire loads;
  V2.1.1 keeps every wire segment, so the wires never disappear.

## 3. Limits

- These are three small 6T arrays, one corner, one target cell and one
  operation pair. They bound nothing about waveform correctness, retention,
  sense margin or timing: the waveform scorer and the class-bound evidence runs
  use mode 0 and nothing here changes that.
- The largest array measured is 32x32. The trend is clear but the numbers are
  not an extrapolation to the array sizes the feature is aimed at, and no 10T
  array was measured. Measure your own configuration before relying on the
  model.
- Per-device local mismatch in an equivalent array covers only the retained
  cells, so the equivalent modes cannot carry a yield or mismatch statement.
- Modes 1-4 call Xyce during netlist generation to extract the cell parasitics
  and the static current, so they need Xyce on PATH even to build a deck, and
  that extraction is part of the runtime cost measured above.
- The physical wires are kept in full in every mode, and so are the per-cell
  local loads, so the cost of an equivalent array still grows with the array.
  The speed-up comes from removing transistors, not wires.

## 4. Reproducing

```bash
python3 -m sram_compiler.equivalent_modeling.compare \
    --sizes 16x16 --modes 0,1,2,3,4 --plot
python3 -m sram_compiler.equivalent_modeling.compare \
    --sizes 8x8 --modes 0,1,4
python3 -m sram_compiler.equivalent_modeling.compare \
    --sizes 32x32 --modes 0,1,4 --plot
```

Results, including `settings.json` with every input, are under ignored
`outputs/equivalent_modeling/`.
