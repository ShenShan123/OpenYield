# Equivalent array cells — V2.2.4

Mode 0 keeps every transistor in the SRAM array. Modes 1–4 replace progressively more cells outside the addressed access path with local equivalent loads. Every mode retains all distributed wire segments and their taps. The approximations shorten large-array solves, but only mode 0 gives full-array per-device mismatch coverage.

| Mode | Real cells retained |
|---:|---|
| 0 | Entire array; reference |
| 1 | Target row and target column |
| 2 | Target row |
| 3 | Target column |
| 4 | Target cell |

The default is the `equivalent.mode` value in `sram_compiler/config_yaml/global.yaml` (currently 0). An explicit `--real-cell-mode` overrides it for one run. Generate a reference deck from the repository root:

```bash
python -m sram_compiler.per_device_mc.run \
  --rows 8 --cols 4 --operation read \
  --variation-mode nominal --mc-runs 1 --real-cell-mode 0 \
  --output-dir outputs/equivalent_reference
```

To compare an approximation, repeat with `--real-cell-mode 1` through `4`, keeping array, target, corner, clock, geometry, and wire settings identical. Add `--run-xyce` for simulator measurements. Modes 1–4 need Xyce even when only generating a deck: the compiler extracts cell capacitances and leakage from the active PDK model at the effective VDD and temperature, then caches the result by cell, model-content hash, and physical context. `sram_compiler/subcircuits/sram_cell_add_equivalent.py` implements these loads. A local optional comparison script may be kept under ignored development tools; it is not required by the runtime compiler.

Inspect read/write waveforms, retention, sense margin, and access timing against mode 0 before using an approximate mode. A match in `.MEASURE` output alone does not establish waveform accuracy. Equivalent modes never qualify full-array yield. The default interconnect geometry is illustrative, so use a physical wire configuration when comparing a real design. See the [compiler tutorial](../README.md), [distributed RC design](../../docs/design/DISTRIBUTED_RC_MODEL.md), and [V2.2.4 status](../../docs/design/PHASED_CONTROL_V2_2_4.md).
