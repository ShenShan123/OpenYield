# Local checks for V2.2.7 and preserved V2.2.4 manifests

Python regression and waveform scripts are kept in local workspaces and ignored by Git. A fresh clone includes this guide and the JSON case manifests under `spice/`, but does not include the Python runner. The manifests preserve the planned V2.2.4 cases; they are not results or qualification evidence.

If your workspace retains the local scripts, run software checks from the repository root:

```bash
python -m pytest -q tests size_optimization/openyield_v2/tests
```

Most compiler checks generate decks or inspect source structure without Xyce. Equivalent-model extraction checks require Xyce even during deck construction. `python -m unittest discover -s dev/tests -v` checks optional ignored development tools when they are present. For a fresh clone, use the compiler's nominal mode 0 smoke command in the [root tutorial](../README.md#validate-changes).

The local compiler checks cover configuration loading, timing and driver lookup, distributed RC connectivity, per-device model specialization, startup conditions, sense and precharge ordering, runner retries, and output parsing. They are useful regression checks, but generated netlists or passing measures do not prove read, write, hold, sense margin, or recovery waveform correctness.

The [SPICE guide](spice/README.md) describes the optional local waveform runner and these tracked manifests:

| Manifest | Planned cases |
|---|---:|
| `spice/v224_cases.json` | 334 main cases, including four dynamic mux reads |
| `spice/v224_mc_cases.json` | 1,444 per-device draws |
| `spice/v224_negative_cases.json` | 6 negative controls |

The V2.2.4 manifests were partially run as an unfinished V2.2.5 screen, and 256x256 still lacks a working operating point. The [V2.2.5 simulation summary](../docs/V2_2_5_SIMULATION_SUMMARY.md) records the stopped campaign and its old model hashes; the [V2.2.2 record](../docs/design/PHASED_CONTROL_V2_2_2.md) describes the last assembled screen, which certifies only that older tree.

[`spice/v227_extended_rc_cases.json`](spice/v227_extended_rc_cases.json)
records the larger-array, PVT, mismatch and reduction-sensitivity diagnostics.
The [extended report](../docs/design/EXTENDED_RC_VALIDATION_V2_2_7.md) separates
functional results from RC accuracy and preserves failed points. Local
`test_rc_pvt.py` and `test_router_clearance.py` check supply-scaled comparison
metrics, same-net/foreign-net via clearance and supply landing protection.
`test_critical_rc_frequency.py` checks passive response against independent
closed-form circuits. These diagnostics do not certify the full V2.2.7 release
or statistical yield.
