# OpenYield V2.0.6 development tools

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
hashes or promote partial/failed runs. The rule identity and evidence format
remain V2.0.5 in this V2.0.6 release.

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

```bash
python3 -m dev.sizing.campaign --dry-run
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

Use `--sizes 8x4 --pilot` on the campaign for a small check. Full-array Monte
Carlo is expensive; the timeout defaults to six hours per sample (60 hours for
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
