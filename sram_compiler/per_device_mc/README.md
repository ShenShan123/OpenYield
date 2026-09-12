# Per-device local mismatch — V2.0.10

This package is part of the SRAM compiler. Independent local mismatch is the
default for both its runner and `Sram6TCoreMcTestbench`: the selected PDK corner
stays fixed while every retained MOS gets its own `vth0`, `u0`, and `voff`
Gaussian expressions. The default relative sigma is 5% for all three parameters.
Array and peripheral MOS devices are included; fingers within one MOS share its
draw. This model does not apply area scaling or calibrated parameter correlations.

`main_sram.py` at the repository root is the script entrance with the same
default: seeded per-device mismatch (`MC_SEED = 20260711`) over the full array,
with the YAML files read in memory. This runner is the command-line form.

V2.0.10 retains every requested measurement-sample index and rejects missing or
unsafe distributed wordline-release measurements. A failed DC operating point
gets one Newton line-search retry with the same sampling seed; the first deck,
log and outputs remain in `dcop_attempt/`. If an unseeded run's actual seed
cannot be recovered, it remains failed. A successful simulator exit is not a
full timing or yield qualification.

Run commands from the repository root:

```bash
# Generate a full-array deck, model cards, and an audit without running Xyce.
python -m sram_compiler.per_device_mc.run \
  --rows 8 --cols 4 --operation read --mc-runs 2 --seed 3 --audit

# Add simulation and waveform plotting.
python -m sram_compiler.per_device_mc.run \
  --rows 8 --cols 4 --operation write --mc-runs 2 --seed 3 --run-xyce
```

Direct script execution is also supported:
`python sram_compiler/per_device_mc/run.py --help`.
In V2.0.6 the former top-level `per_device_mc` package has moved here; update imports to
`sram_compiler.per_device_mc` and CLI commands to the paths above.

The runner defaults to `real_cell_mode=0` (full transistor array), 100 samples,
and `outputs/per_device_mc/`. It loads YAML in memory. Each configuration has a
deterministic output subdirectory; `--output-dir` selects another output root.
The supported operations are `read`, `write`, `read&write`, `hold_snm`, `read_snm`,
and `write_snm`. `--run-xyce` needs Xyce on PATH or `--xyce /path/to/Xyce`.
`--interconnect-config path.yaml` applies a wire mapping in memory (see the
[distributed RC guide](../../docs/design/DISTRIBUTED_RC_MODEL.md)); the run
directory name and `summary.json` include the resolved topology.
After a successful simulation, waveform output is plotted as `waveform.png`.

| Variation mode | Behavior |
|---|---|
| `per-device` (default) | Independent local draws for each retained MOS |
| `nominal` | Fixed corner alone; exactly one run |
| `shared` | Legacy random model cards shared by devices using the same base model |
| `custom` | Process-parameter tables from the cell YAML; sample count matches table rows |

A single per-device sample is random. Use `--variation-mode nominal` for a
deterministic corner run. Equivalent modes 1–4 only vary the retained devices;
replaced cells remain approximations. See the
[equivalent-model guide](../../equivalent_modeling/README.md).
Per-device mismatch requires a separate deck per geometry and cannot be combined
with legacy `.STEP` geometry sweeps.

For in-memory configuration:

```python
from sram_compiler.per_device_mc.run import load_config
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench

config = load_config(8, 4, "TT")
testbench = Sram6TCoreMcTestbench(
    config, corner="TT", real_cell_mode=0, mc_seed=3,
    sim_path="outputs/compiler_example",
)
circuit = testbench.create_testbench("read", 7, 3)
testbench.add_analysis(circuit, "read", 2)  # Enables Xyce stochastic sampling.
```

| Module | Responsibility |
|---|---|
| `run.py` | In-memory YAML loading, CLI, deck export, Xyce execution, and waveform plotting |
| `netlist.py` | Model specialization and hierarchy audits while preserving MOS connectivity, dimensions, and sweep expressions |
| `sampling.py` | Reproducible local Latin hypercube draws materialized into numeric cards for MPI execution |

See the [compiler guide](../README.md) and [sizing guide](../sizing/README.md)
for testbench and qualification details. Generated decks alone do not establish
waveform correctness or timing qualification.
