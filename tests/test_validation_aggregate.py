from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from yield_estimation.aggregate_validation import aggregate


class ValidationAggregateTests(unittest.TestCase):
    @staticmethod
    def _write_manifest(root: Path) -> None:
        for name in ("config.json", "summary.csv", "DONE"):
            (root / name).write_text(name, encoding="utf-8")
        lines = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "MANIFEST.sha256":
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                lines.append(f"{digest}  {path.relative_to(root)}")
        (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_missing_experimental_methods_do_not_block_stable_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for algorithm in ("MC", "MNIS", "AIS", "ACS", "HSCS", "EFIAL"):
                result_dir = root / algorithm.lower()
                result_dir.mkdir()
                payload = {
                    "algorithm": algorithm,
                    "status": "ok",
                    "budget_target": 5000,
                    "charged_calls": 5000,
                    "live_calls": 5000,
                    "simulator_errors": 0,
                    "failure_probability": 0.125,
                    "standard_error": 0.005,
                    "samples_used": 5000,
                    "metadata": {"node": "test"},
                }
                (result_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
                self._write_manifest(result_dir)

            summary = aggregate(root, 12495 / 99400, 99400)
            self.assertTrue(summary["strict_pass"])
            self.assertEqual(summary["missing_stable_algorithms"], [])
            self.assertEqual(summary["missing_experimental_algorithms"], ["FUSIS", "OPT", "BIBD"])


if __name__ == "__main__":
    unittest.main()
