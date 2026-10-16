# Online-CTR

This package contains the online TabPFN coarse-to-refine contribution imported
from `OpenYield_wangc.zip`. It is intentionally registered as **Online-CTR**,
separately from SramOpt. The method builds its surrogate context from real SPICE calls made
during the current run and does not read the archive's old CSV datasets.
It reuses the shared TabPFN utilities in `openyield_v2/surrogate_utils.py`;
this code reuse does not connect Online-CTR to the offline datasets or optimizers.

From the repository root:

```bash
python -m pip install -r size_optimization/openyield_v2/requirements.txt
python main_opt.py --algorithm Online-CTR --check
python main_opt.py --algorithm Online-CTR --max-iter 60 --device cpu
```

Budgets below 60 run context acquisition only. Larger budgets reserve at least
30 real-SPICE calls for final auditing; see `configs/online.yaml`.

The full method needs PyTorch, TabPFN weights, and `pymoo`; the check loads no
weights and performs no SPICE simulation. TabPFN uses its installed package's
weight cache/download mechanism. `max_iter` counts candidate evaluations;
each evaluation can launch several SPICE analyses. The circuit adapter comes
from V2.2.6 `exp_utils.py`, and its results still require waveform review and
mode-0 per-device Monte Carlo before being used as array-yield evidence.
