"""Check the compiler-facing methods without importing legacy ML dependencies."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


class YieldContractTests(unittest.TestCase):
    def test_all_estimators_count_nonfinite_delay_as_failure_and_unpack_four_metrics(self):
        root = Path(__file__).resolve().parents[1] / 'yield_estimation/model_lib'
        calls = 0
        for name in ('AIS', 'ACS', 'MC', 'MNIS', 'HSCS'):
            tree = ast.parse((root / f'{name}.py').read_text())
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
            for method in cls.body:
                if isinstance(method, ast.FunctionDef) and method.name in ('indicator', 'indicator_func'):
                    namespace = {'np': np}
                    exec(compile(ast.Module(body=[method], type_ignores=[]), str(root / f'{name}.py'), 'exec'), namespace)
                    values = np.array([1., 3., -1., np.nan, np.inf, -np.inf])
                    np.testing.assert_array_equal(namespace[method.name](SimpleNamespace(threshold=2.), values),
                                                  [False, True, True, True, True, True])
            for node in ast.walk(cls):
                if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                    continue
                func = node.value.func
                if not isinstance(func, ast.Attribute) or func.attr != 'run_mc_simulation':
                    continue
                # Execute the actual assignment against the compiler's four-value
                # return shape, without running the unrelated estimator workflow.
                sample = np.array([.1, np.nan])
                replacement = ast.Assign(targets=node.targets, value=ast.Name(id='metrics', ctx=ast.Load()))
                module = ast.fix_missing_locations(ast.Module(body=[replacement], type_ignores=[]))
                namespace = {'metrics': (sample, np.ones(2), np.zeros(2), np.ones(2))}
                exec(compile(module, name, 'exec'), namespace)
                self.assertIs(namespace['y'], sample)
                calls += 1
        self.assertEqual(calls, 15)


if __name__ == '__main__':
    unittest.main()
