# Shared utilities — V2.0.6

This package consolidates the former root `utils.py`, reusable `plot_data.py`
functions, and shared optimizer plotting helpers. It is tracked runtime code;
the hardcoded plotting experiment is preserved locally in ignored
`dev/plot_data_demo.py`.

| Module | Public helpers |
|---|---|
| `measurements.py` | `parse_mc_measurements`, `generate_mc_statistics`, `save_mc_results` |
| `waveforms.py` | `read_prn_with_preprocess`, `split_blocks` |
| `plotting.py` | `visualize_results`, `process_simulation_data`, `plot_delay`, `plot_power`, `plot_rc_delay`, `plot_leak_delay`, `plot_merit_history`, `plot_pareto_frontier` |
| `area.py` | `estimate_bitcell_area`, `estimate_array_area`, `estimate_total_area`, `estimate_array_macro_area`, `estimate_total_macro_area`, `estimate_scaled_array_area` |
| `spice.py` | `parse_spice_models`, `write_spice_models`, `remove_comments`, `parse_parameters`, `convert_value` |

Existing imports remain supported:

```python
from utils import estimate_bitcell_area, parse_spice_models, process_simulation_data
```

You can also import directly from a focused module:

```python
from utils.measurements import parse_mc_measurements
from utils.plotting import plot_delay
```

The optimizer's existing `from size_optimization.exp_utils import
plot_merit_history, plot_pareto_frontier` imports remain valid.

## Waveform plots

```python
from utils import process_simulation_data

process_simulation_data(
    "outputs/example/deck.sp.prn", num_mc=2,
    output="outputs/example/waveform.png", selected_columns=["V(BL0)", "V(BLB0)"],
)
```

PRN readers accept indexed or index-free transient/DC output. Sample splitting
preserves DataFrame column labels and rejects inconsistent sample counts.
Specify the expected `num_mc` when processing a waveform.

## SRAM comparison plots

```python
from utils.plotting import plot_delay

path = plot_delay(
    [8, 16], [1e-9, 2e-9], [1e-11, 2e-11],
    [0.5e-9, 0.6e-9], [1e-11, 1e-11],
    "Read", "Write", "access_delay",
)
```

The four comparison helpers retain their original positional parameters and
unit/error-bar scaling. Delay inputs use seconds; power inputs use watts.
Delay, power, and RC comparisons show three-sigma bars; leakage comparisons
retain the original one-sigma bars.

They return the saved path, create missing output directories, and default to
repository-relative `outputs/plots/`. Delay, power, and leakage plots save PDF;
RC plots save PNG. Pass `output_dir=...` to select a different directory or
`show=True` to display a plot after saving. The default is noninteractive;
comparison plot styles are scoped to each call and figures are closed afterward.

Reusable tests are tracked in [`tests/`](../tests/README.md):

```bash
python3 -m unittest tests.test_utils -v
```
