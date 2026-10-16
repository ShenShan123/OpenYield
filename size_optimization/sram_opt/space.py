"""V2.2.6-compatible mixed design space for the SramOpt method."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from size_optimization.exp_utils import CompositeSRAMParameterSpace

from .contract import FEATURE_COLUMNS, normalize_design


ROW_CHOICES = (16, 32, 64, 128, 256, 512)
COLUMN_CHOICES = (16, 32, 64, 128, 256)
TOTAL_BITS = 262_144


def _category_value(index: int, count: int) -> float:
    return (int(index) + 0.5) / int(count)


class SramOptDesignSpace:
    """Expose 25 search degrees of freedom and a 26-feature decoded design.

    num_arrays is derived from rows, columns and the fixed 32-KiB capacity.
    V2.2.6 has separate PD/PG threshold coordinates; SramOpt collapses them
    to one shared NMOS coordinate to retain its original feature contract.
    """

    feature_dimension = len(FEATURE_COLUMNS)

    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        total_bits: int = TOTAL_BITS,
        row_choices: Sequence[int] = ROW_CHOICES,
        column_choices: Sequence[int] = COLUMN_CHOICES,
    ) -> None:
        if config_path is None:
            config_path = str(Path(__file__).resolve().parents[1] / "config_sram.yaml")
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.total_bits = int(total_bits)
        self.row_choices = tuple(int(value) for value in row_choices)
        self.column_choices = tuple(int(value) for value in column_choices)
        self.base_space = CompositeSRAMParameterSpace(self.config_path)
        if self.base_space.bitcell_dim != 7:
            raise RuntimeError(
                "SramOpt expects the V2.2.6 seven-coordinate bitcell space; "
                f"found {self.base_space.bitcell_dim}"
            )
        if self.base_space.peripheral_dim != 17:
            raise RuntimeError(
                "SramOpt expects 17 varied peripheral coordinates; "
                f"found {self.base_space.peripheral_dim}"
            )
        self.circuit_dimension = 23
        self.search_dimension = 2 + self.circuit_dimension
        self.total_dim = self.search_dimension
        self.bounds = np.vstack(
            [np.zeros(self.search_dimension), np.ones(self.search_dimension)]
        )

    @staticmethod
    def _choice(value: float, choices: Sequence[int]) -> int:
        index = min(int(float(value) * len(choices)), len(choices) - 1)
        return int(choices[max(0, index)])

    @staticmethod
    def _expand_circuit_vector(values: np.ndarray) -> np.ndarray:
        if values.shape != (23,):
            raise ValueError(f"expected 23 circuit coordinates, got {values.shape}")
        shared_nmos = values[4]
        return np.concatenate(
            [values[:4], [shared_nmos, shared_nmos, values[5]], values[6:]]
        )

    def convert_params(self, point: Sequence[float]) -> Dict[str, Any]:
        values = np.asarray(point, dtype=float).reshape(-1)
        if values.shape != (self.search_dimension,):
            raise ValueError(
                f"SramOpt expects {self.search_dimension} search coordinates, "
                f"got {len(values)}"
            )
        if not np.isfinite(values).all():
            raise ValueError("SramOpt coordinates must be finite")
        values = np.clip(values, 0.0, 1.0)
        rows = self._choice(values[0], self.row_choices)
        cols = self._choice(values[1], self.column_choices)
        capacity = rows * cols
        if self.total_bits % capacity:
            raise ValueError("selected architecture does not divide the target capacity")
        params = self.base_space.convert_params(
            self._expand_circuit_vector(values[2:])
        )
        params.update(
            rows=rows,
            cols=cols,
            num_rows=rows,
            num_cols=cols,
            num_arrays=self.total_bits // capacity,
        )
        params["nmos_model_name"] = params["pd_model_name"]
        params["pg_model_name"] = params["pd_model_name"]
        normalize_design(params)
        return params

    def feature_row(self, point: Sequence[float]) -> Dict[str, Any]:
        return normalize_design(self.convert_params(point))

    def get_default_normalized_vector(self) -> np.ndarray:
        base = np.asarray(
            self.base_space.get_default_normalized_vector(), dtype=float
        )
        if base.shape != (24,):
            raise RuntimeError(f"unexpected V2.2.6 default-vector shape: {base.shape}")
        collapsed = np.concatenate([base[:4], [base[4], base[6]], base[7:]])
        return np.concatenate(
            [
                [_category_value(0, len(self.row_choices))],
                [_category_value(0, len(self.column_choices))],
                collapsed,
            ]
        )
