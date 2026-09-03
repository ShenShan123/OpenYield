"""OpenYield-style simulation adapter and strict simulation budget ledger."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union
import csv
import math

import numpy as np


class BudgetExceeded(RuntimeError):
    """Raised before a simulation call would exceed the charged budget."""


@dataclass
class BudgetLedger:
    target: int
    charged_calls: int = 0
    live_calls: int = 0
    cache_calls: int = 0
    retry_calls: int = 0
    simulator_errors: int = 0

    @property
    def remaining(self) -> int:
        return self.target - self.charged_calls

    def charge(self, count: int, *, retry: bool = False, cached: bool = False) -> None:
        if count < 0 or count > self.remaining:
            raise BudgetExceeded(
                f"simulation budget exceeded: requested={count}, remaining={self.remaining}"
            )
        self.charged_calls += count
        if cached:
            self.cache_calls += count
        else:
            self.live_calls += count
        if retry:
            self.retry_calls += count


@dataclass(frozen=True)
class SimulationBatch:
    values: np.ndarray
    statuses: tuple[str, ...]
    errors: tuple[Optional[str], ...]
    charged_calls: int
    live_calls: int
    retry_calls: int
    run_name: str
    artifacts: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        object.__setattr__(self, "values", values)
        if len(self.statuses) != values.size or len(self.errors) != values.size:
            raise ValueError("values/statuses/errors must have equal length")

    @property
    def simulator_errors(self) -> int:
        return sum(status != "ok" for status in self.statuses)

    @property
    def valid_mask(self) -> np.ndarray:
        return np.asarray(self.statuses, dtype=object) == "ok"


class SimulationRunner:
    """Normalize OpenYield simulation backends behind ``run_mc_simulation``.

    The wrapped testbench remains untouched and may continue returning its
    historical tuple. Algorithms consume this runner and always receive a
    :class:`SimulationBatch`.
    """

    def __init__(
        self,
        model: Any,
        simulation_root: Union[str, Path] = "sim/unified",
        *,
        metric: Union[str, int, Callable[[Any], Any]] = 0,
        input_space: str = "physical",
        nominal: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
        max_retries: int = 0,
        quiet: bool = False,
    ) -> None:
        if input_space not in {"physical", "standard_normal"}:
            raise ValueError("input_space must be 'physical' or 'standard_normal'")
        self.model = model
        self.simulation_root = Path(simulation_root)
        self.simulation_root.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.input_space = input_space
        self.nominal = None if nominal is None else np.asarray(nominal, dtype=float).reshape(-1)
        self.sigma = None if sigma is None else np.asarray(sigma, dtype=float).reshape(-1)
        self.max_retries = int(max_retries)
        self.quiet = quiet
        self.ledger: Optional[BudgetLedger] = None
        self._batch_index = 0

    def reset_budget(self, max_num: int) -> BudgetLedger:
        if max_num <= 0:
            raise ValueError("max_num must be positive")
        self.ledger = BudgetLedger(int(max_num))
        return self.ledger

    def _physical_vars(self, values: Optional[np.ndarray], mc_runs: int) -> Optional[np.ndarray]:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[0] != mc_runs:
            raise ValueError(f"vars must have shape ({mc_runs}, D), got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("vars contains NaN or Inf")
        if self.input_space == "physical":
            return array
        if self.nominal is None or self.sigma is None:
            raise ValueError("nominal and sigma are required for standard_normal input")
        if array.shape[1] != self.nominal.size or self.sigma.size != self.nominal.size:
            raise ValueError("normalized vars, nominal and sigma dimensions do not match")
        return self.nominal[None, :] + self.sigma[None, :] * array

    def _set_run_directory(self, run_name: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_name)
        directory = self.simulation_root / f"{self._batch_index:05d}_{safe}"
        directory.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "set_sim_path"):
            self.model.set_sim_path(str(directory))
        elif hasattr(self.model, "sim_path"):
            self.model.sim_path = str(directory)
        return directory

    def _call_model(
        self,
        *,
        operation: str,
        target_row: int,
        target_col: int,
        mc_runs: int,
        temperature: float,
        values: Optional[np.ndarray],
    ) -> Any:
        if hasattr(self.model, "run_mc_simulation"):
            return self.model.run_mc_simulation(
                operation=operation,
                target_row=target_row,
                target_col=target_col,
                mc_runs=mc_runs,
                temperature=temperature,
                vars=values,
            )
        if hasattr(self.model, "sample"):
            if values is None:
                raise ValueError("legacy .sample() backend requires vars")
            return self.model.sample(values, mc_runs)
        if callable(self.model):
            if values is None:
                raise ValueError("callable backend requires vars")
            return self.model(values)
        raise TypeError("model must provide run_mc_simulation(), sample(), or be callable")

    def _read_csv(
        self, path: Union[str, Path]
    ) -> tuple[np.ndarray, tuple[str, ...], tuple[Optional[str], ...]]:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        if not rows:
            return np.empty(0, dtype=float), (), ()
        if isinstance(self.metric, str):
            column = self.metric
        else:
            candidates = ("read_delay_s", "delay", "value", "y")
            column = next((name for name in candidates if name in rows[0]), "")
        if not column:
            raise KeyError("could not determine metric column in CSV result")
        values = np.asarray([float(row[column]) for row in rows], dtype=float)
        statuses = tuple(
            row.get("status", "ok") if np.isfinite(value) else "simulator_error"
            for row, value in zip(rows, values)
        )
        errors = tuple(
            (row.get("error") or None) if status != "ok" else None
            for row, status in zip(rows, statuses)
        )
        return values, statuses, errors

    def _extract_payload(
        self, raw: Any
    ) -> tuple[np.ndarray, tuple[str, ...], tuple[Optional[str], ...]]:
        if isinstance(raw, SimulationBatch):
            return raw.values.copy(), raw.statuses, raw.errors
        if isinstance(raw, (str, Path)):
            return self._read_csv(raw)
        supplied_statuses = None
        supplied_errors = None
        if callable(self.metric):
            raw = self.metric(raw)
        elif isinstance(raw, dict):
            supplied_statuses = raw.get("statuses", raw.get("status"))
            supplied_errors = raw.get("errors", raw.get("error"))
            if isinstance(self.metric, str) and self.metric in raw:
                raw = raw[self.metric]
            else:
                excluded = {"statuses", "status", "errors", "error"}
                raw = next(value for key, value in raw.items() if key not in excluded)
        elif isinstance(raw, tuple):
            index = self.metric if isinstance(self.metric, int) else 0
            raw = raw[index]
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        finite = np.isfinite(values)
        if supplied_statuses is None:
            statuses = tuple("ok" if valid else "simulator_error" for valid in finite)
        elif isinstance(supplied_statuses, str):
            statuses = tuple(supplied_statuses for _ in values)
        else:
            statuses = tuple(str(status) for status in supplied_statuses)
        if supplied_errors is None:
            errors = tuple(None if status == "ok" else "simulation backend reported an error" for status in statuses)
        elif isinstance(supplied_errors, str):
            errors = tuple(supplied_errors or None for _ in values)
        else:
            errors = tuple(None if error in {None, ""} else str(error) for error in supplied_errors)
        statuses = tuple(
            status if valid and status == "ok" else "simulator_error"
            for status, valid in zip(statuses, finite)
        )
        errors = tuple(
            error
            if status == "ok" or error is not None
            else "simulation backend reported an error"
            for status, error in zip(statuses, errors)
        )
        return values, statuses, errors

    def _write_batch_artifact(
        self,
        directory: Path,
        input_vars: Optional[np.ndarray],
        physical_vars: Optional[np.ndarray],
        values: np.ndarray,
        statuses: tuple[str, ...],
        errors: tuple[Optional[str], ...],
    ) -> Path:
        path = directory / "samples.csv"
        input_dimension = 0 if input_vars is None else input_vars.shape[1]
        physical_dimension = 0 if physical_vars is None else physical_vars.shape[1]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["sample_id", "status", "value", "error"]
                + [f"input_{i}" for i in range(input_dimension)]
                + [f"physical_{i}" for i in range(physical_dimension)]
            )
            for index, (value, status, error) in enumerate(zip(values, statuses, errors)):
                row = [index, status, value, error or ""]
                if input_vars is not None:
                    row.extend(input_vars[index].tolist())
                if physical_vars is not None:
                    row.extend(physical_vars[index].tolist())
                writer.writerow(row)
        return path

    @staticmethod
    def _measurement_file(path: Path) -> dict[str, float]:
        values: dict[str, float] = {}
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "=" not in line:
                    continue
                key, raw = line.split("=", 1)
                try:
                    values[key.strip().upper()] = float(raw.strip())
                except ValueError:
                    continue
        except OSError:
            return {}
        return values

    def _native_measurement_fallback(
        self,
        directory: Path,
        operation: str,
        mc_runs: int,
        metric_name: Optional[str] = None,
    ) -> tuple[np.ndarray, tuple[str, ...], tuple[Optional[str], ...]]:
        suffix = "ms" if "snm" in operation else "mt"
        files: dict[int, Path] = {}
        for path in directory.glob(f"**/*.{suffix}*"):
            tail = path.name.rsplit(f".{suffix}", 1)[-1]
            if tail.isdigit():
                files[int(tail)] = path
        output: list[float] = []
        statuses: list[str] = []
        errors: list[Optional[str]] = []
        for index in range(mc_runs):
            measures = self._measurement_file(files[index]) if index in files else {}
            requested = (metric_name or "").upper()
            if requested in {"READ_DELAY_S", "DELAY", "VALUE", "Y"}:
                requested = "TREAD_TOTAL"
            if requested:
                value = measures.get(requested, math.nan)
            elif "snm" in operation:
                value = measures.get(operation.upper(), math.nan)
            else:
                value = measures.get("TREAD_TOTAL", math.nan)
                if not math.isfinite(value) or value <= 0:
                    parts = [measures.get(name, math.nan) for name in ("TSA", "TSWING", "TWLDRV")]
                    value = sum(parts) if all(math.isfinite(item) and item > 0 for item in parts) else math.nan
            output.append(value)
            # A finite non-positive native measure is Xyce's sentinel for a
            # converged circuit that did not complete the measured operation.
            # Keep it as a valid physical outcome; the estimator may classify
            # it as functional failure. A missing/NaN measure is a simulator
            # error and must never be counted as physical failure.
            valid = math.isfinite(value)
            statuses.append("ok" if valid else "simulator_error")
            errors.append(None if valid else "missing or invalid native Xyce measurement")
        return np.asarray(output, dtype=float), tuple(statuses), tuple(errors)

    def run_mc_simulation(
        self,
        operation: str = "read",
        target_row: int = 0,
        target_col: int = 0,
        mc_runs: int = 100,
        temperature: float = 27,
        vars: Optional[np.ndarray] = None,
        *,
        run_name: Optional[str] = None,
    ) -> SimulationBatch:
        if mc_runs <= 0:
            raise ValueError("mc_runs must be positive")
        if self.ledger is None:
            self.reset_budget(mc_runs * (self.max_retries + 1))
        input_vars = None if vars is None else np.asarray(vars, dtype=np.float64)
        if input_vars is not None and input_vars.ndim == 1:
            input_vars = input_vars.reshape(1, -1)
        physical_vars = self._physical_vars(input_vars, mc_runs)
        name = run_name or f"batch_{self._batch_index:05d}"
        directory = self._set_run_directory(name)
        attempts = 0
        last_error: Optional[Exception] = None
        while attempts <= self.max_retries:
            is_retry = attempts > 0
            if mc_runs > self.ledger.remaining:
                if not is_retry:
                    raise BudgetExceeded(
                        f"simulation budget exceeded: requested={mc_runs}, remaining={self.ledger.remaining}"
                    )
                break
            self.ledger.charge(mc_runs, retry=is_retry)
            try:
                backend_log = directory / "backend.log"
                if self.quiet:
                    with backend_log.open("a", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
                        raw = self._call_model(
                            operation=operation,
                            target_row=target_row,
                            target_col=target_col,
                            mc_runs=mc_runs,
                            temperature=temperature,
                            values=physical_vars,
                        )
                else:
                    raw = self._call_model(
                        operation=operation,
                        target_row=target_row,
                        target_col=target_col,
                        mc_runs=mc_runs,
                        temperature=temperature,
                        values=physical_vars,
                    )
                # A string metric names a CSV column or a native Xyce measure.
                # Historical tuple returns do not carry measure names, so read
                # the raw files rather than silently treating tuple item 0 as
                # the requested metric.
                if isinstance(raw, tuple) and isinstance(self.metric, str):
                    values, statuses, errors = self._native_measurement_fallback(
                        directory, operation, mc_runs, self.metric
                    )
                else:
                    values, statuses, errors = self._extract_payload(raw)
                    if values.size != mc_runs:
                        values, statuses, errors = self._native_measurement_fallback(
                            directory,
                            operation,
                            mc_runs,
                            self.metric if isinstance(self.metric, str) else None,
                        )
                if len(statuses) != mc_runs or len(errors) != mc_runs:
                    raise RuntimeError("simulation status/error length does not match values")
                self.ledger.simulator_errors += sum(status != "ok" for status in statuses)
                artifact = self._write_batch_artifact(
                    directory, input_vars, physical_vars, values, statuses, errors
                )
                batch_artifacts = [str(artifact)]
                if backend_log.exists():
                    batch_artifacts.append(str(backend_log))
                batch = SimulationBatch(
                    values=values,
                    statuses=statuses,
                    errors=errors,
                    charged_calls=mc_runs * (attempts + 1),
                    live_calls=mc_runs * (attempts + 1),
                    retry_calls=mc_runs * attempts,
                    run_name=name,
                    artifacts=tuple(batch_artifacts),
                )
                self._batch_index += 1
                return batch
            except BudgetExceeded:
                raise
            except Exception as exc:  # backend failures are data, not physical failures
                last_error = exc
                # Some historical testbenches raise while aggregating CSVs
                # after Xyce has already completed. Recover per-run native
                # measurements when possible; missing points remain errors.
                values, statuses, errors = self._native_measurement_fallback(
                    directory,
                    operation,
                    mc_runs,
                    self.metric if isinstance(self.metric, str) else None,
                )
                if np.isfinite(values).any():
                    self.ledger.simulator_errors += sum(status != "ok" for status in statuses)
                    artifact = self._write_batch_artifact(
                        directory, input_vars, physical_vars, values, statuses, errors
                    )
                    batch_artifacts = [str(artifact)]
                    backend_log = directory / "backend.log"
                    if backend_log.exists():
                        batch_artifacts.append(str(backend_log))
                    self._batch_index += 1
                    return SimulationBatch(
                        values=values,
                        statuses=statuses,
                        errors=errors,
                        charged_calls=mc_runs * (attempts + 1),
                        live_calls=mc_runs * (attempts + 1),
                        retry_calls=mc_runs * attempts,
                        run_name=name,
                        artifacts=tuple(batch_artifacts),
                    )
                self.ledger.simulator_errors += mc_runs
                attempts += 1
                if attempts <= self.max_retries:
                    continue
        message = f"{type(last_error).__name__}: {last_error}"
        values = np.full(mc_runs, np.nan, dtype=float)
        statuses = tuple("simulator_error" for _ in range(mc_runs))
        errors = tuple(message for _ in range(mc_runs))
        artifact = self._write_batch_artifact(
            directory, input_vars, physical_vars, values, statuses, errors
        )
        self._batch_index += 1
        return SimulationBatch(
            values=values,
            statuses=statuses,
            errors=errors,
            charged_calls=mc_runs * attempts,
            live_calls=mc_runs * attempts,
            retry_calls=mc_runs * max(0, attempts - 1),
            run_name=name,
            artifacts=(str(artifact),),
        )


class TargetCellTestbenchAdapter:
    """Expand one-cell process values into an otherwise nominal SRAM array."""

    def __init__(
        self,
        testbench: Any,
        nominal_cell: np.ndarray,
        *,
        num_rows: int,
        num_cols: int,
        target_row: int,
        target_col: int,
    ) -> None:
        self.testbench = testbench
        self.nominal_cell = np.asarray(nominal_cell, dtype=float).reshape(-1)
        self.num_rows = int(num_rows)
        self.num_cols = int(num_cols)
        self.target_row = int(target_row)
        self.target_col = int(target_col)
        self.sim_path = getattr(testbench, "sim_path", "sim")
        if not (0 <= self.target_row < self.num_rows and 0 <= self.target_col < self.num_cols):
            raise ValueError("target cell is outside the SRAM array")

    def set_sim_path(self, path: str) -> None:
        self.sim_path = path
        if hasattr(self.testbench, "set_sim_path"):
            self.testbench.set_sim_path(path)
        else:
            self.testbench.sim_path = path

    def run_mc_simulation(
        self,
        operation: str = "read",
        target_row: int = 0,
        target_col: int = 0,
        mc_runs: int = 100,
        temperature: float = 27,
        vars: Optional[np.ndarray] = None,
    ) -> Any:
        if vars is None:
            raise ValueError("target-cell simulation requires explicit vars")
        cell_values = np.asarray(vars, dtype=float)
        if cell_values.shape != (mc_runs, self.nominal_cell.size):
            raise ValueError(
                f"target-cell vars must have shape {(mc_runs, self.nominal_cell.size)}"
            )
        full = np.tile(self.nominal_cell, (mc_runs, self.num_rows * self.num_cols))
        cell_index = self.target_row * self.num_cols + self.target_col
        start = cell_index * self.nominal_cell.size
        full[:, start:start + self.nominal_cell.size] = cell_values
        return self.testbench.run_mc_simulation(
            operation=operation,
            target_row=target_row,
            target_col=target_col,
            mc_runs=mc_runs,
            temperature=temperature,
            vars=full,
        )
