# OpenYield V2.0.6 documentation

Project plans and release history live here. Usage guides live beside the code
they describe. Commands and code paths in the documents are relative to the
repository root unless stated otherwise.

V2.0.6 integrates the default per-device local mismatch package into the compiler
and establishes this documentation layout. Earlier release entries, design
snapshots, and qualification evidence retain their original version labels.

| Document | Purpose |
|---|---|
| [Driver sizing proposal](DRIVER_SIZING_PROPOSAL.md) | Full working proposal, implementation progress, and current qualification status |
| [Automatic timing proposal](TIMING_AUTOCONFIG.md) | Original timing design and measured basis; implemented timing lives in `sram_compiler/sizing/timing.py` |
| [Changelog](CHANGELOG.md) | Release history and validation results |
| [Development guide](DEVELOPMENT.md) | Tracked regression tests and ignored local experiments/qualification runners |
| [Original V2.0.4 design](design/DRIVER_SIZING_PROPOSAL_V2.0.4.md) | Preserved design snapshot, including its historical source paths |

The supplied evidence CSVs remain at the repository root:
[driver sizing data](../DRIVER_SIZING_data.csv) and
[timing data](../TIMING_AUTOCONFIG_data.csv).

| Directory guide | Contents |
|---|---|
| [SRAM compiler](../sram_compiler/README.md) | Configuration, circuit generation, simulations, and waveform outputs |
| [Shared utilities](../utils/README.md) | Measurement and waveform parsing, plots, SPICE models, and area estimates |
| [Compiler regression tests](../tests/README.md) | Simulator-free checks that run without local development scripts |
| [Per-device mismatch](../sram_compiler/per_device_mc/README.md) | Default local mismatch, CLI, and in-memory configuration |
| [Driver sizing](../sram_compiler/sizing/README.md) | Frozen driver sizes, timing, and qualification workflows |
| [Circuit review](../sram_compiler/CIRCUIT_REVIEW.md) | Circuit findings and verification evidence |
| [Equivalent modeling](../equivalent_modeling/README.md) | Equivalent circuit modes, usage, and accuracy boundaries |
| [Sizing optimization](../size_optimization/README.md) | Circuit-backed algorithms and offline optimizers |
| [Yield estimation](../yield_estimation/README.md) | Monte Carlo and importance-sampling algorithms |
