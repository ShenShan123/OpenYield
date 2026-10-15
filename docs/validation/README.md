# Simulation validation reports

These reports retain their original models, compiler versions and diagnostic
scope. The V2.2.5 full screen stopped incomplete; V2.2.6 has targeted checks,
and no full V2.2.8 screen is complete. See the [documentation index](../README.md)
for current limits and [design records](../design/README.md) for the circuit changes.

| Report | Scope |
|---|---|
| [V2.2.5 simulation summary](V2_2_5_SIMULATION_SUMMARY.md) | Reconciles the unfinished screen with corrected-model and diagnostic results |
| [Corrected-model corner cases](V2_2_5_CORRECTED_CORNER_CASES.md) | Selected boundary configurations and waveform checks on corrected model hashes |
| [Negative controls](V2_2_5_NEGATIVE_CONTROL_EVALUATION.md) | Short-clock waveform rejections and matched positive references |

Machine-readable evidence remains under [`../data/`](../data/README.md).
Tracked simulation protocols and the optional local runner are described in
the [SPICE guide](../../tests/spice/README.md).
