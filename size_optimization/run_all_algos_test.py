#!/usr/bin/env python3
"""
Quick validation test: run all algorithms with minimal iterations
to verify paths and simulation work correctly.
"""
import os, sys, time, traceback

# Ensure project root on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "size_optimization"))

from experiment import TwoStageOptimizer

ALGORITHMS = ["SA", "PSO", "SMAC", "CBO", "RoSE_Opt", "MOEAD", "MOBO", "NSGA-II"]
MAX_ITER = 1  # minimal iterations just to verify no crash
SEED = 42

results_summary = {}

for algo in ALGORITHMS:
    print("\n" + "=" * 70)
    print(f"  TESTING ALGORITHM: {algo}  (max_iter={MAX_ITER})")
    print("=" * 70)
    t0 = time.time()
    try:
        optimizer = TwoStageOptimizer(seed=SEED, gen_unused_cells=True)
        result = optimizer.run_joint_optimization(max_iter=MAX_ITER, algorithm=algo)
        elapsed = time.time() - t0
        fom = result.get("best_fom") if result else None
        success = result is not None and result.get("best_fom") is not None
        status = "OK" if success else "NO_VALID_SOLUTION"
        results_summary[algo] = {"status": status, "fom": fom, "time": elapsed}
        print(f"\n>>> {algo}: {status}  fom={fom}  time={elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        results_summary[algo] = {"status": "ERROR", "error": str(e), "time": elapsed}
        print(f"\n>>> {algo}: ERROR  ({e})")
        traceback.print_exc()

print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
all_ok = True
for algo in ALGORITHMS:
    r = results_summary[algo]
    status = r["status"]
    t = r.get("time", 0)
    extra = f"fom={r.get('fom')}" if status != "ERROR" else f"error={r.get('error', '')[:80]}"
    marker = "✓" if status != "ERROR" else "✗"
    if status == "ERROR":
        all_ok = False
    print(f"  {marker} {algo:12s}  {status:20s}  {t:7.1f}s  {extra}")

print("\n" + ("ALL ALGORITHMS PASSED ✓" if all_ok else "SOME ALGORITHMS FAILED ✗"))
