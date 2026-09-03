"""Aggregate per-method validation artifacts and apply the formal gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from yield_estimation import EXPERIMENTAL_ALGORITHMS, STABLE_ALGORITHMS


def _verify_manifest(run_root: Path) -> bool:
    manifest = run_root / "MANIFEST.sha256"
    if not manifest.is_file():
        return False
    listed: set[Path] = set()
    try:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            digest, relative_text = line.split("  ", 1)
            relative = Path(relative_text)
            if relative.is_absolute() or ".." in relative.parts or relative in listed:
                return False
            path = run_root / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                return False
            listed.add(relative)
    except (OSError, ValueError):
        return False
    required = {Path(name) for name in ("config.json", "result.json", "summary.csv", "DONE")}
    actual = {
        path.relative_to(run_root)
        for path in run_root.rglob("*")
        if path.is_file() and path != manifest
    }
    return required <= listed and listed == actual


def aggregate(root: Path, reference_probability: float, reference_count: int) -> dict[str, Any]:
    reference_se = math.sqrt(
        reference_probability * (1.0 - reference_probability) / reference_count
    )
    rows: list[dict[str, Any]] = []
    found: set[str] = set()
    for path in sorted(root.glob("*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        algorithm = result.get("algorithm", path.parent.name.upper())
        found.add(algorithm)
        manifest_ok = _verify_manifest(path.parent)
        if algorithm == "BIBD":
            conditions = result.get("conditions", {}).values()
            live_calls = sum(int(item.get("live_calls", 0)) for item in conditions)
            simulator_errors = sum(int(item.get("simulator_errors", 0)) for item in conditions)
            nodes = sorted({
                str(item.get("metadata", {}).get("node"))
                for item in conditions
                if item.get("metadata", {}).get("node")
            })
            passed = bool(
                result["status"] == "ok"
                and result["charged_calls"] == result["budget_target"]
                and simulator_errors == 0
                and manifest_ok
            )
            rows.append({
                "algorithm": algorithm,
                "status": result["status"],
                "budget": result["budget_target"],
                "charged": result["charged_calls"],
                "live_calls": live_calls,
                "simulator_errors": simulator_errors,
                "manifest_ok": manifest_ok,
                "node": ",".join(nodes),
                "strict": False,
                "pass": passed,
                "reason": "multi-condition; single-condition gate not applicable",
                "result_path": str(path),
            })
            continue
        probability = float(result["failure_probability"])
        standard_error = float(result["standard_error"])
        tolerance = max(
            2.0 * math.sqrt(standard_error**2 + reference_se**2),
            0.1 * reference_probability,
        )
        estimate_pass = abs(probability - reference_probability) <= tolerance
        strict = algorithm in STABLE_ALGORITHMS
        passed = (
            result["status"] in {"ok", "ok_zero_failure"}
            and result["charged_calls"] == result["budget_target"]
            and result["simulator_errors"] == 0
            and manifest_ok
            and estimate_pass
        )
        rows.append({
            "algorithm": algorithm,
            "status": result["status"],
            "budget": result["budget_target"],
            "charged": result["charged_calls"],
            "live_calls": result["live_calls"],
            "simulator_errors": result["simulator_errors"],
            "manifest_ok": manifest_ok,
            "failure_probability": probability,
            "relative_error": abs(probability - reference_probability) / reference_probability,
            "standard_error": standard_error,
            "tolerance": tolerance,
            "node": result.get("metadata", {}).get("node"),
            "strict": strict,
            "pass": passed,
            "reason": "ok" if passed else "gate failed",
            "result_path": str(path),
        })

    missing_stable = [name for name in STABLE_ALGORITHMS if name not in found]
    missing_experimental = [name for name in EXPERIMENTAL_ALGORITHMS if name not in found]
    missing = missing_stable + missing_experimental
    strict_pass = not missing_stable and all(row["pass"] for row in rows if row["strict"])
    payload = {
        "reference_probability": reference_probability,
        "reference_count": reference_count,
        "reference_standard_error": reference_se,
        "missing_algorithms": missing,
        "missing_stable_algorithms": missing_stable,
        "missing_experimental_algorithms": missing_experimental,
        "strict_pass": strict_pass,
        "rows": rows,
    }
    json_path = root / "validation_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = root / "validation_summary.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    marker = root / ("ALL_DONE" if strict_pass else "VALIDATION_INCOMPLETE")
    opposite = root / ("VALIDATION_INCOMPLETE" if strict_pass else "ALL_DONE")
    if opposite.exists():
        opposite.unlink()
    marker.write_text("pass\n" if strict_pass else "failed\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-probability", type=float, default=12495 / 99400)
    parser.add_argument("--reference-count", type=int, default=99400)
    args = parser.parse_args()
    payload = aggregate(args.root, args.reference_probability, args.reference_count)
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["strict_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
