# Compiler regression tests — V2.0.11

These reusable tests are tracked outside the circuit generator. They cover
driver sizing and frozen loads, MOS connectivity and widths, per-device
mismatch, numeric and swept geometry, sampling, optimizer integration, and
qualification-table validation. They use full transistor arrays and require
neither Xyce nor the ignored local development scripts.

V2.0.10 also checks sense-control RC accounting, the distributed precharge
guard, access-window measurements, write-feedback initialization, preserved
sample indices, and the shared DC retry/seed/evidence contract. V2.0.11 adds
the retry's stale-output contract, the rejected sample's retained provenance,
the precharge windows' clamp to the analysis stop, and the parser's
missing-value and all-missing-frame behaviour. Actual waveform validation is
recorded in the [V2.0.10 review](../docs/design/DISTRIBUTED_RC_V210_REVIEW.md)
and the [V2.0.11 write screen](../docs/design/WRITE_VALIDATION_V211.md).

Keep these tests in Git: they document expected behavior and let every checkout
verify compiler and utility changes. `test_utils.py` also checks measurement
parsing, transient/DC sample splitting, plot files, and existing import paths.
`test_main_entrance.py` checks that `main_sram.py` defaults to seeded per-device
mismatch over the full array and applies its settings in memory without
changing the tracked YAML files.

Run from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

The separate offline optimizer checks are:

```bash
python3 -m unittest discover -s size_optimization/openyield_v2/tests -v
```

Tests of local experiments and qualification runners live under ignored
`dev/tests/`. See the [development guide](../docs/DEVELOPMENT.md) for those
optional tools. Keep runtime testbenches in `sram_compiler/testbenches/`;
they construct the simulator circuits used by the project.

V2.0.7 adds `test_rc_configuration.py` and `test_interconnect.py` for RC-value
propagation, extraction context/cache identity, distributed pi-section R/C
conservation, cell/equivalent taps, matched replica paths and local MC probes.
V2.0.8 adds the `cell_pin_rc` default for directly constructed configurations,
the project-root YAML loader, and the `main_sram.py` interconnect setting.
V2.0.9 adds the `LookupTests` in `test_driver_sizing.py`: fixed size classes,
their rule lower bound, independence from cell/mux/RC/wires, ladder
interpolation and extrapolation, invalid or alternative tables, and the removed
legacy mode; the RC and driver-path tests now exercise `lookup` and `rules_only`.
