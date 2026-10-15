# V2.2.7 compiler organization and unified experiments

The compiler stays at **V2.2.7**. `main_sram.py` is the public experiment
entrance, and independent per-device `vth0/u0/voff` mismatch is the ordinary
simulation default. This change adds no waveform or yield qualification.

## Organization

| Component | Responsibility |
|---|---|
| `config_yaml/config.py`, `config_yaml/loader.py` | Configuration types and source-relative, in-memory loading/overrides |
| `variation/` | Variation/sample policy and model-card specialization/auditing |
| `experiments/` | Command dispatch, shared simulation/result flow and equivalent accuracy comparisons |
| `subcircuits/`, `testbenches/` | Existing circuit topology, stimulus and measurement APIs |
| `sizing/`, `equivalent_modeling/`, `layout/` | Existing sizing/timing policy, equivalent modes and physical workflows |

`per_device_mc/` is removed. Its tracked model specializer moves unchanged to
`variation/netlist.py`; configuration loading no longer imports a CLI runner.
The local materialized-sampling helper remains under ignored `dev/`, and local
regression/waveform imports are migrated without adding their Python scripts to
Git. Equivalent comparison is now a tracked experiment so a fresh checkout can
use the unified entrance without an ignored comparison script.

## Entrance and results

The entrance accepts `simulate`, `equivalent`, `layout ACTION`,
`circuit-optimize`, `circuit-experiment`, `optimize`, `compare`, and `experiment`.
No command or direct simulation flags select simulation. The script settings
are defaults, and explicit flags override them. The script's existing
configuration, area, testbench and deck helper API remains available.

Simulation supports both cell architectures, all three transient operations,
and all three SNM operations. Sampled modes use the YAML sample count when no
count is specified; nominal uses one run, and custom mode uses the process-table
row count. Invalid script sample-count types are rejected without argparse
coercion. `--seed random` preserves the Python API's unseeded sampling option.

Settings and CLI runs share deck generation, Xyce retry behavior, result
validation, CSV statistics and summary provenance. Identities include cell
geometry/models, model content, mux, local and distributed RC, controller,
timing, corner, supply, temperature and variation. Repeated attempts retain
separate directories. Failed/undefined ensembles retain their raw results and
status; summaries report mean metrics only for complete, finite ensembles.
SNM retains single-cell coverage metadata even when configured dimensions are
larger. Per-device audits are always saved.

The source relocation changes the compiler scoring identity, so previously
hashed qualification records do not silently carry over. Frozen circuit
dimensions, driver/timing tables, PDK models and physical RC data are preserved.

## Validation scope

Commands run from the repository root with the `environment.yml` Python 3.9 /
PySpice 1.5 environment and Xyce 7.4. Local Python tests remain ignored:

```bash
python -m compileall -q main_sram.py sram_compiler utils size_optimization yield_estimation
python -m unittest discover -s tests -p 'test_*.py' -q
python -m unittest tests.test_unified_entrance -q
python -m pytest -q size_optimization/openyield_v2/tests
git diff --check
```

The offline tests use the available Python/pytest environment. A tracked-only
checkout also passes `--version`, equivalent-comparison help, the optimizer
campaign dry run, and default 2×2 per-device deck generation without `dev/` or
local Python test runners. Command routing/help does not run a full optimizer
campaign or repeat Calibre extraction.

The final focused run passes all 58 checks covering the existing entrance,
configuration review, runtime packaging, local mismatch/sampling, equivalent
inputs, interconnect and failure handling, the unified CLI, qualified-table
context matching, and the five-corner preparation harness. The six offline
optimizer tests also pass. Compilation and the final diff check pass.

Xyce smoke runs use a full mode-0, no-mux 2×2 6T configuration at TT, 1 V,
25 °C; local stubs are 100 ohm / 1 fF and distributed wiring uses the
out-of-scope illustrative fallback. The TT model SHA-256 is
`40fbaa38ec61aa2b3917754dfbc12ff34be47208940359acbf4a55a10ead7cc0`.

| Operation | Variation / seed | Clock | Result |
|---|---|---|---|
| Hold SNM | Nominal / 20260711 | DC | One finite sample; 0.34791 V |
| Write | Nominal / 20260711 | Fixed 30 ns | Runner access and finite-metric checks pass |
| Write | Per-device, two draws / 31 | Fixed 30 ns | Runner access and finite-metric checks pass |

Deck-only checks additionally cover 10T SNM with an unseeded per-device draw,
CLI geometry overrides without YAML writes, and preserved repeat attempts.
Result checks reject missing/undefined SNM samples while retaining provenance.
Two local qualification-table fixtures were corrected to use the configured
interconnect context; the same stale-context mismatch reproduces using the
pre-refactor loader.

The broader 229-test local suite reports 25 assertion failures in seven older
test methods. All 25 failure signatures reproduce on the original `5ac9d58`
tree, exported independently from Git. They cover outdated equality between
implicit illustrative wiring and the calibrated YAML, refined decoder/register
route counts, the old `distributed` mode spelling, and old controller load and
settling-time formulas. Those production inputs and policies are outside this
organization change and remain frozen. This is not an all-green full suite.
The initial run also recorded source-change guards after trailing-newline fixes
and a qualification fixture loaded before its correction; the final focused
run passes those checks against stable sources. Detailed local logs are under
ignored `outputs/v227_*`.

No independent read, write, hold/retention, sense and recovery waveform screen
was run for this organization change. The DC SNM smoke and passing runtime
measures are not waveform correctness or yield evidence. The existing full
V2.2.7 screen and 256×256 operating-point limitations remain.
