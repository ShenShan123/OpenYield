# Compiler regression tests — V2.0.6

These reusable tests are tracked outside the circuit generator. They cover
driver sizing and frozen loads, MOS connectivity and widths, per-device
mismatch, numeric and swept geometry, sampling, optimizer integration, and
qualification-table validation. They use full transistor arrays and require
neither Xyce nor the ignored local development scripts.

Keep these tests in Git: they document expected behavior and let every checkout
verify compiler and utility changes. `test_utils.py` also checks measurement
parsing, transient/DC sample splitting, plot files, and existing import paths.

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
