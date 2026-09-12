# OpenYield V2.0.11 development tools

The circuit generator lives in `sram_compiler/`. Reusable compiler regression
tests live in the tracked top-level `tests/` directory. Local experiments,
qualification campaign runners, and tests of those tools live under ignored
`dev/`; they remain available in this workspace and are not included in a fresh
checkout. Run commands from the repository root.

## Tracked regression checks

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s size_optimization/openyield_v2/tests -v
python3 -m compileall -q sram_compiler utils tests size_optimization/exp_utils.py
git diff --check
```

The compiler tests require no local development tools or Xyce executable.
`testbenches/` contains the compiler's simulation construction API, which is
runtime code and remains tracked inside `sram_compiler/`.

Keep reusable regression tests in Git alongside the implementation so every
checkout can verify fixes. Only ad hoc experiments, machine-specific scripts,
and tests of ignored local tools belong under `dev/`. Shared utility checks in
`tests/test_utils.py` cover parsing, sample boundaries, headless plots, and
compatibility imports. The former plotting script's hardcoded dataset remains
local-only in `dev/plot_data_demo.py`.

## Local-only tools

The following commands require the ignored `dev/` workspace:

| Local module | Purpose |
|---|---|
| `dev/sizing/campaign.py` | Schedule expensive calibration and qualification campaigns |
| `dev/sizing/local_review.py` | Run bounded diagnostic screens and inspect checkpoints |
| `dev/sizing/qualification.py` | Execute development cases and score their waveforms |
| `dev/sizing/report.py` | Export development evidence and gate table promotion |
| `dev/sizing/offset.py` | Characterize sense-amplifier offset ensembles |
| `dev/sizing/gate_cap.py` | Measure gate-charge reference loads |
| `dev/sizing/gate_fingers.py` | Check wide-gate fingering with DC/transient experiments |
| `dev/sizing/wordline_model.py` | Compare star and distributed wordline models |
| `dev/sizing/execution.py` | Support campaign MPI execution and timeout cleanup |
| `dev/sizing/provenance.py` | Verify local scoring sources against the tracked manifest |
| `dev/summarize_qualification.py` | Aggregate campaign checkpoints with on-disk reruns and `retry_dcop` results (V2.0.9 evidence summary) |
| `dev/tests/` | Tests of these development tools |

```bash
python3 -m unittest discover -s dev/tests -v
```

Keep future ad hoc test scripts and development experiments in `dev/` so they
are ignored automatically. Generated results still belong in `outputs/` or a
temporary directory. Preserve the supplied evidence CSVs and recorded JSON
results with their original version labels.

## Qualification source identity

Runtime table lookup uses the tracked
[`scoring_sources.json`](../sram_compiler/sizing/scoring_sources.json) manifest
plus runtime code hashes. It does not import or read local developer scripts.
The local runners compare every Python source in `dev/sizing/` with the manifest
before issuing a scoring identity. Edited, missing, or extra source files must
not silently reuse the previous identity.

After reviewing a change to the local scoring code, refresh its source manifest:

```python
import hashlib
import json
from pathlib import Path

sources = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(Path("dev/sizing").glob("*.py"))
}
if not sources:
    raise RuntimeError("Local qualification sources are unavailable")
Path("sram_compiler/sizing/scoring_sources.json").write_text(
    json.dumps({"schema": 1, "sources": sources}, indent=2) + "\n"
)
```

A changed manifest invalidates earlier scoring identities. Re-run the affected
qualification before publishing replacement records; do not update old result
hashes or promote partial/failed runs. The `rules_only` rule identity and the
evidence format remain V2.0.5 in V2.0.9; changed physical/scoring hashes require
new evidence. V2.0.9 refreshed the manifest for the `sizing_mode`, `--samples`,
`--seed`, `--cells`, `--mux` and `--no-rc` campaign options.

## Local qualification workflow

The resumable campaign derives a clock from nominal SS read and SS/SF write
phases, freezes it, then runs full-local variation, corner, hazard and RC
checks. A case's completion marker must match its exact deck/model/seed inputs
and simulator binary before simulation output is reused. Cached waveforms are
rescored on each run. Shared and nominal runs cannot qualify the local policy.
If Xyce aborts an individual transient with `Time step too small`, the runner
detects its missing waveform interval even when Xyce returns success. It retries
the same seeded ensemble once with a tighter 5 ps maximum step, preserving the
original attempt and recording the actual timestep. Failed electrical checks or
nonfinite signals do not trigger this retry.
RC calibration uses a longer initial clock because the historical fit excludes
the explicit 100-ohm/1-fF networks; its final clock still comes from measurement.
V2.0.9 adds a second automatic retry: when Xyce reports `DC Operating Point
Failed` for one sampled circuit (KLU direct solver; GMIN and source stepping did
not recover the reproduced 16x16 SF write-box sample), every later sample of the
same native-sampling run starts from a corrupt state and the materialized MPI
path aborts. Newton with line search (`.OPTIONS NONLIN SEARCHMETHOD=2`, a
solver strategy that changes no circuit element) converged every sample of the
reproduced 16x16 SF and 32x32 FF decks that plain Newton failed, whereas GMIN or
source stepping did not and the default linear solver failed other samples of
the same ensembles. On the materialized MPI path the failed sample alone walks
the ladder of `operating_point_fallbacks()` with the same cards and seed: line
search on the same ranks, the default solver on the same ranks, then line search
and the default solver on one rank (the serial rungs are the expensive ones on
large decks); the extra option goes into a
`deck_fallback.sp` copy beside the untouched sample deck and
`execution.operating_point_fallbacks` records the steps per sample. A
materialized sample that stops with `Time step too small` is likewise rerun
alone with a 5 ps maximum step (`deck_tighter_step.sp`,
`execution.timestep_retries`), the per-sample form of the native-path retry. If the whole
case still fails, or for native sampling, the runner reruns the seeded ensemble
once in `retry_dcop/` with the line-search option (`xyce_options` in the result). Results record `linear_solver` (`KLU` or `Xyce default`) and `xyce_options`,
and keep the original attempt with `dc_operating_point_failed`; a retry must
pass every check on its own, and a complete ensemble with an electrical failure
is never retried.

```bash
python3 -m dev.sizing.campaign --dry-run
# V2.0.9 fixed size classes (the compiler default); the tool itself still defaults to rules_only.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m dev.sizing.campaign --sizing-mode lookup --sizes 8x4,16x16,32x32 --screen --samples 3 --seed 82026 --output-dir outputs/qualification/V2.0.9/screen-small --xyce /path/to/Xyce --workers 24
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m dev.sizing.campaign --sizing-mode lookup --sizes 64x64,16x256 --screen --samples 3 --seed 82026 --cells SRAM_6T_CELL --mux off --no-rc --output-dir outputs/qualification/V2.0.9/screen-large --xyce /path/to/Xyce --workers 8 --mpi-ranks 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m dev.sizing.campaign --sizes 8x4 --pilot --xyce /path/to/Xyce --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m dev.sizing.campaign --screen --xyce /path/to/Xyce --workers 24
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m dev.sizing.campaign --xyce /path/to/Xyce --workers 24
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m dev.sizing.offset --xyce /path/to/Xyce
python3 -m dev.sizing.report outputs/qualification/V2.0.5 --table sram_compiler/sizing/sizing_table.json
```

The independent gate-load and wide-gate reference experiments are reproducible:

```bash
python3 -m dev.sizing.gate_cap --xyce /path/to/Xyce
python3 -m dev.sizing.gate_fingers --xyce /path/to/Xyce
python3 -m dev.sizing.wordline_model --xyce /path/to/Xyce
```

The wordline check drives real RC cells with the array's wordline driver and
compares the compiler's per-pin stub model (an ideal row net, a star) with a
distributed line per column pitch; it records first- and last-cell arrival and
slew. The star model is kept for the wordline sizing rule by decision; the
proposal's Stage C outcome records the distributed-line numbers as reference.

The fingering check holds total MOS widths fixed, compares transient edges at
NF=1/16/100, and verifies nearly unchanged DC current. Its reference metadata
includes the model and simulator binary hashes.

Use `--sizes 8x4 --pilot` on the campaign for a small check. `--samples` below
ten, `--pilot`, `--screen` and `--functional` runs are screens, never
qualification; `--cells`, `--mux` and `--no-rc` restrict the scheduled
architectures and `--seed` sets the ensemble seed. `--functional` schedules the
read/write/hold function checks for large or unseen arrays: the three primary
decks, the hot FF/FS 125 C and cold FF -40 C reads and writes with address-change
hazards from 16 rows up (the neighbor cell must retain its data), and the SS
read/write sequence up to 4,096 cells. Hazard rows flip the middle address bit
of the last row as in V2.0.2, falling back to a set bit so non-power-of-two
arrays stay inside the decoded range (`hazard_row()`). Local `Case` objects carry `sizing_mode` (`rules_only` keeps the
V2.0.5 case names; `lookup` cases use fixed classes and reject a
`parasitic_factor`). Full-array Monte Carlo is expensive; the timeout defaults to six hours per sample (60 hours for
a ten-sample deck), and incomplete runs remain failures. Process counts change
scheduling, not circuit tolerances.
`--screen` includes PVT, sequences and RC, postponing the dedicated 100-sample
ensembles. Both pilot and screening runs are ineligible for table promotion.
The full schedule has 112 configurations, 336 calibration decks, 1,682 local
verification decks and 17,156 waveform samples. Review failures before expanding
the schedule. A root lock prevents two campaigns writing the same checkpoint.
The report refuses incomplete campaigns and requires both local sensing and
SS/SF write-tail evidence before promotion. The table remains empty.

Large arrays (at least 1,024 cells) use four MPI ranks by default. Set
`--mpi-ranks` and `--parallel-min-cells` to adjust this; `workers × mpi-ranks`
must fit the available cores. Use the MPI launcher beside the Xyce binary,
with one BLAS thread per rank. Xyce 7.4's native random-expression sampling
crashed on the large MPI test, so these runs sample local Gaussian LHS values
before execution and save numeric cards for each transient. Each parameter's
stream is stable across candidate widths and unrelated device insertions.
Serial small-array checks use native Xyce sampling. Both backends record their
provenance; saved MPI cards support exact reproduction on one or more cores.
The four-core 64x64 execution check passes. The three-sample rule screen finished
on 2026-09-08; its outcome (36 RC wordline-budget failures, timeouts and three
DCOP aborts on materialized MPI samples) is in `docs/DRIVER_SIZING_PROPOSAL.md`.

For a bounded representative-array diagnostic before the full campaign:

```bash
python3 -m dev.sizing.local_review --workers 20 --mpi-ranks 4 --samples 3 --timeout 1800 --xyce /path/to/openyield/bin/Xyce
python3 -m dev.sizing.local_review --summarize
python3 -m dev.sizing.local_review --sizes 16x16 --rc-only --workers 4 --output-dir outputs/qualification/V2.0.5/review-rc-fix --xyce /path/to/openyield/bin/Xyce
```

`--rc-only` keeps only the explicit-RC 16x16 architectures, for reruns after a
change to the RC topology. Use a fresh `--output-dir` to keep earlier evidence.

Half-select qualification remains open. The full working proposal and its
historical evidence remain in [DRIVER_SIZING_PROPOSAL.md](DRIVER_SIZING_PROPOSAL.md).

V2.0.7 adds the [distributed RC model](design/DISTRIBUTED_RC_MODEL.md).
Local `Case` settings accept `interconnect`, `pi_res` (ohms) and `pi_cap`
(farads); scorers read physical endpoints and actual sense inputs.
The release diagnostic runner is `dev/validate_distributed_rc.py`; its V2.0.7
results are under ignored `outputs/validation/V2.0.7/` and the V2.0.8 audit
cases under `outputs/validation/V2.0.8/` (`--output` selects the directory).
V2.0.8 case files additionally accept `w_rc`, `cell_pin_rc`, `replica_k`,
`wire_scale` (multiplies the illustrative per-pitch R and C), `pi_res_ohm`,
`pi_cap_pf` and a per-case Xyce `timeout`. These diagnostics do not promote
a sizing-table record. See the [validation scope](design/DISTRIBUTED_RC_VALIDATION.md).

The [V2.0.9 review](design/DISTRIBUTED_RC_V210_REVIEW.md) retains its local
cases and waveforms under `outputs/validation/V2.0.9-review/`. The diagnostic
now defaults to `lookup`, matching the compiler; historical `fixed` cases
require their historical checkout. Its optional case fields include `cycles`
(4 or 8), `max_step`, `mpi_ranks`, `xyce_options` and `probe_cells: sampled`
(all selected-row cells plus near/middle/far unselected cells). Sampling the
probes does not replace any array transistor. `dev/review_score_v209.py`
independently checks retained traces and records their hashes in `audit.json`.

V2.0.11 extends the same diagnostic to the default star topology with
`"interconnect": "star"` in a case, so an array without distributed wires can
be scored from its trace; the `VWL_PRE_*` measures the compiler emits exist in
distributed mode only. Write cases additionally check that precharge stays off
for the whole write-enable window and that the write driver, not the initial
condition, pulls the bitline down. The [V2.0.11 screen](design/WRITE_VALIDATION_V211.md)
retains its cases under `outputs/validation/V2.0.11-write/`.
