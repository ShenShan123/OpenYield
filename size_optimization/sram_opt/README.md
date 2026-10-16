# SramOpt integration

This package integrates `SramOpt` with the OpenYield
V2.2.6 compiler. Run all commands from the repository root through
`main_opt.py`.

Install the repository environment first, then the optional optimizer packages:

```bash
python -m pip install -r size_optimization/openyield_v2/requirements.txt
```

The yield pipeline also uses `joblib`, which is installed with scikit-learn.

## Design contract

SramOpt keeps the original ordered 26-feature predictor contract: three
architecture fields, six bitcell fields, and seventeen peripheral fields. The
V2.2.6 search has 25 independent coordinates because `num_arrays` is derived
from `rows`, `cols`, and the fixed 262,144-bit capacity. V2.2.6 exposes separate
PD and PG model coordinates; this adapter constrains both to the shared NMOS
model expected by the original SramOpt predictor. Columns are limited to 256.

Check the integration without a dataset, model, or SPICE run:

```bash
python main_opt.py --algorithm SramOpt --check
```

The check reports the compiler version, feature/search dimensions, fixed
capacity, and whether optional model paths exist.

## Train or supply the yield predictor

Training data, fitted bundles, and TabPFN checkpoints are deliberately not
included. To create a compatible bundle from a design-level Monte Carlo CSV:

```bash
python size_optimization/sram_opt/train_yield.py \
  --dataset /path/to/merged_design_targets.csv \
  --checkpoint /path/to/tabpfn-regressor.ckpt \
  --output-root outputs/sram_opt_training \
  --skip-208-sensitivity \
  --device cpu
```

The dataset must contain the ordered 26 design features plus `design_uid`,
`architecture`, `source_server`, `mc_attempts`, `primary_tlim_ns`, and the
selected yield target (default: `target_yield`). `--skip-208-sensitivity` skips
the original campaign's extra validation on source `208`; omit it when that
source has enough designs for the requested folds.
Generated training artifacts remain under the ignored `outputs/` tree.

`joblib` can execute code while loading. Only pass a bundle that you created or
otherwise trust. The runtime rejects reordered features, invalid training
context, a missing checkpoint, and a checkpoint whose SHA-256 differs from the
hash stored in the bundle.

## Run SramOpt

```bash
python main_opt.py \
  --algorithm SramOpt \
  --max-iter 50 \
  --backend SA \
  --bundle /path/to/tabpfn_incontext_training_bundle.joblib \
  --checkpoint /path/to/tabpfn-regressor.ckpt \
  --device cpu
```

`--backend LHS` selects the constrained Latin-hypercube baseline. If
`--yield-target` is omitted, the threshold stored in the bundle is used.
Results and per-candidate simulation files are written below
`outputs/sram_opt/` unless `--output-dir` is supplied.

Each search candidate uses V2.2.6 equivalent-array mode 1 at the nominal TT
corner for circuit screening and queries the external per-device-Monte-Carlo
yield predictor. Mode 1 is a reduced-order approximation; its result is not
full-array yield evidence. Before reporting a selected design, validate it with
mode 0 and V2.2.6 per-device Monte Carlo, including waveform checks for read,
write, retention, sense, and recovery.
