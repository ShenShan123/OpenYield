#!/usr/bin/env python3
"""Train and audit a TabPFN design-to-yield regressor.

The learning unit is one circuit design. Independent per-device Monte Carlo
draws are integrated into the empirical yield target and are never expanded
into thousands of input columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder


os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")


ROOT = Path(__file__).resolve().parents[2] / "outputs" / "sram_opt_training"
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "TABPFN_CHECKPOINT",
        str(Path.home() / ".cache/tabpfn/tabpfn-v2.6-regressor-v2.6_default.ckpt"),
    )
)

# The five mux fields are fixed in this campaign. The remaining fields are the
# 26 actually varied architecture/circuit parameters described by the paper.
FEATURE_COLUMNS = (
    "rows",
    "cols",
    "num_arrays",
    "pg_width",
    "pd_width",
    "pu_width",
    "length",
    "nmos_model_name",
    "pmos_model_name",
    "sa_p_width",
    "sa_n_width",
    "sa_length",
    "sa_p_model",
    "sa_n_model",
    "prc_p_width",
    "prc_length",
    "prc_p_model",
    "wld_nand_p_width",
    "wld_inv_p_width",
    "wld_nand_n_width",
    "wld_inv_n_width",
    "wld_length",
    "wld_nand_p_model",
    "wld_inv_p_model",
    "wld_nand_n_model",
    "wld_inv_n_model",
)

CATEGORICAL_COLUMNS = (
    "nmos_model_name",
    "pmos_model_name",
    "sa_p_model",
    "sa_n_model",
    "prc_p_model",
    "wld_nand_p_model",
    "wld_inv_p_model",
    "wld_nand_n_model",
    "wld_inv_n_model",
)

NUMERIC_COLUMNS = tuple(column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS)
FEASIBILITY_TARGET = 0.90


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass
class FeatureEncoder:
    ordinal_encoder: OrdinalEncoder
    numeric_medians: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "FeatureEncoder":
        categorical = frame.loc[:, list(CATEGORICAL_COLUMNS)].astype(str)
        ordinal = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float32,
        ).fit(categorical)
        medians = {}
        for column in NUMERIC_COLUMNS:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = float(values.median())
            if not math.isfinite(median):
                raise ValueError(f"numeric training feature is entirely invalid: {column}")
            medians[column] = median
        return cls(ordinal_encoder=ordinal, numeric_medians=medians)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"prediction data is missing features: {missing}")
        result = np.empty((len(frame), len(FEATURE_COLUMNS)), dtype=np.float32)
        categorical = self.ordinal_encoder.transform(
            frame.loc[:, list(CATEGORICAL_COLUMNS)].astype(str)
        )
        categorical_index = {column: index for index, column in enumerate(CATEGORICAL_COLUMNS)}
        for feature_index, column in enumerate(FEATURE_COLUMNS):
            if column in categorical_index:
                result[:, feature_index] = categorical[:, categorical_index[column]]
            else:
                values = pd.to_numeric(frame[column], errors="coerce").fillna(
                    self.numeric_medians[column]
                )
                result[:, feature_index] = values.to_numpy(dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("encoded feature matrix contains nonfinite values")
        return result


def categorical_indices() -> list[int]:
    return [FEATURE_COLUMNS.index(column) for column in CATEGORICAL_COLUMNS]


def forward_target(
    y_raw: np.ndarray,
    attempts: np.ndarray,
    transform: str,
) -> np.ndarray:
    if transform == "raw":
        return y_raw.astype(float)
    if transform == "jeffreys_logit":
        successes = y_raw * attempts
        probability = (successes + 0.5) / (attempts + 1.0)
        return np.log(probability / (1.0 - probability))
    raise ValueError(f"unsupported target transform: {transform}")


def inverse_target(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return np.clip(values, 0.0, 1.0)
    if transform == "jeffreys_logit":
        clipped = np.clip(values, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))
    raise ValueError(f"unsupported target transform: {transform}")


def build_model(
    *,
    checkpoint: Path,
    n_estimators: int,
    random_state: int,
    device: str,
):
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor(
        n_estimators=n_estimators,
        categorical_features_indices=categorical_indices(),
        model_path=checkpoint,
        device=device,
        ignore_pretraining_limits=True,
        fit_mode="fit_preprocessors",
        random_state=random_state,
        n_preprocessing_jobs=1,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else math.nan,
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1])
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0
        else math.nan,
        "max_abs_error": float(np.max(np.abs(y_true - y_pred))),
    }


def wilson_interval(successes: int, attempts: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if attempts <= 0:
        return math.nan, math.nan
    p = successes / attempts
    denominator = 1.0 + z * z / attempts
    center = (p + z * z / (2.0 * attempts)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / attempts + z * z / (4.0 * attempts * attempts)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def decision_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    true_feasible = y_true >= FEASIBILITY_TARGET
    pred_feasible = y_pred >= FEASIBILITY_TARGET
    false_feasible = pred_feasible & ~true_feasible
    false_infeasible = ~pred_feasible & true_feasible
    true_infeasible_n = int((~true_feasible).sum())
    true_feasible_n = int(true_feasible.sum())
    n = len(y_true)
    ff = int(false_feasible.sum())
    fi = int(false_infeasible.sum())
    ff_low, ff_high = wilson_interval(ff, n)
    fi_low, fi_high = wilson_interval(fi, n)
    near = (y_true >= 0.80) & (y_true <= 0.99)
    return {
        "n": n,
        "true_feasible_n": true_feasible_n,
        "true_infeasible_n": true_infeasible_n,
        "predicted_feasible_n": int(pred_feasible.sum()),
        "false_feasible_n": ff,
        "false_feasible_rate_all": ff / n if n else math.nan,
        "false_feasible_rate_conditional": ff / true_infeasible_n
        if true_infeasible_n
        else math.nan,
        "false_feasible_rate_all_wilson95": [ff_low, ff_high],
        "false_infeasible_n": fi,
        "false_infeasible_rate_all": fi / n if n else math.nan,
        "false_infeasible_rate_conditional": fi / true_feasible_n
        if true_feasible_n
        else math.nan,
        "false_infeasible_rate_all_wilson95": [fi_low, fi_high],
        "decision_accuracy": float((pred_feasible == true_feasible).mean()),
        "boundary_80_to_99_n": int(near.sum()),
        "boundary_false_feasible_n": int((false_feasible & near).sum()),
        "boundary_false_infeasible_n": int((false_infeasible & near).sum()),
    }


def grouped_oof(
    frame: pd.DataFrame,
    *,
    target_column: str,
    seeds: Iterable[int],
    folds: int,
    checkpoint: Path,
    n_estimators: int,
    device: str,
    label: str,
    target_transform: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame["design_uid"].nunique() < folds:
        raise ValueError(f"{label}: fewer designs than folds")
    y_all = pd.to_numeric(frame[target_column], errors="raise").to_numpy(dtype=float)
    attempts = pd.to_numeric(frame["mc_attempts"], errors="raise").to_numpy(dtype=float)
    y_model = forward_target(y_all, attempts, target_transform)
    groups = frame["design_uid"].astype(str).to_numpy()
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []

    for repeat_index, seed in enumerate(seeds):
        splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        repeat_pred = np.full(len(frame), np.nan, dtype=float)
        for fold_index, (train_index, test_index) in enumerate(
            splitter.split(frame, y_all, groups), start=1
        ):
            started = time.monotonic()
            encoder = FeatureEncoder.fit(frame.iloc[train_index])
            x_train = encoder.transform(frame.iloc[train_index])
            x_test = encoder.transform(frame.iloc[test_index])
            model = build_model(
                checkpoint=checkpoint,
                n_estimators=n_estimators,
                random_state=seed * 100 + fold_index,
                device=device,
            )
            model.fit(x_train, y_model[train_index])
            prediction = inverse_target(
                np.asarray(model.predict(x_test), dtype=float), target_transform
            )
            repeat_pred[test_index] = prediction
            fold_metric = regression_metrics(y_all[test_index], prediction)
            fold_rows.append(
                {
                    "evaluation": label,
                    "repeat": repeat_index,
                    "seed": seed,
                    "fold": fold_index,
                    "train_n": len(train_index),
                    "test_n": len(test_index),
                    "elapsed_sec": time.monotonic() - started,
                    **fold_metric,
                    **{
                        f"decision_{key}": value
                        for key, value in decision_metrics(
                            y_all[test_index], prediction
                        ).items()
                        if not isinstance(value, list)
                    },
                }
            )
            for local_index, row_index in enumerate(test_index):
                source = frame.iloc[row_index]
                prediction_rows.append(
                    {
                        "evaluation": label,
                        "repeat": repeat_index,
                        "seed": seed,
                        "fold": fold_index,
                        "design_uid": source["design_uid"],
                        "architecture": source["architecture"],
                        "source_server": source["source_server"],
                        "mc_attempts": int(source["mc_attempts"]),
                        "target_yield": y_all[row_index],
                        "predicted_yield": prediction[local_index],
                        "abs_error": abs(prediction[local_index] - y_all[row_index]),
                    }
                )
        if not np.isfinite(repeat_pred).all():
            raise RuntimeError(f"{label}: repeat {repeat_index} has missing OOF predictions")
        repeat_rows.append(
            {
                "evaluation": label,
                "repeat": repeat_index,
                "seed": seed,
                **regression_metrics(y_all, repeat_pred),
                **{
                    f"decision_{key}": value
                    for key, value in decision_metrics(y_all, repeat_pred).items()
                    if not isinstance(value, list)
                },
            }
        )
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(repeat_rows),
    )


def aggregate_oof(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "design_uid",
        "architecture",
        "source_server",
        "mc_attempts",
        "target_yield",
    ]
    aggregate = (
        predictions.groupby(columns, as_index=False)
        .agg(
            predicted_yield=("predicted_yield", "mean"),
            predicted_yield_std=("predicted_yield", "std"),
            prediction_repeats=("predicted_yield", "count"),
        )
        .sort_values(["architecture", "design_uid"])
    )
    aggregate["abs_error"] = (
        aggregate["predicted_yield"] - aggregate["target_yield"]
    ).abs()
    aggregate["true_feasible_90"] = aggregate["target_yield"] >= FEASIBILITY_TARGET
    aggregate["predicted_feasible_90"] = (
        aggregate["predicted_yield"] >= FEASIBILITY_TARGET
    )
    aggregate["false_feasible"] = (
        aggregate["predicted_feasible_90"] & ~aggregate["true_feasible_90"]
    )
    aggregate["false_infeasible"] = (
        ~aggregate["predicted_feasible_90"] & aggregate["true_feasible_90"]
    )
    return aggregate


def grouped_metric_table(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group_value, group in frame.groupby(group_column, sort=True):
        y_true = group["target_yield"].to_numpy(dtype=float)
        y_pred = group["predicted_yield"].to_numpy(dtype=float)
        rows.append(
            {
                group_column: group_value,
                **regression_metrics(y_true, y_pred),
                **{
                    key: value
                    for key, value in decision_metrics(y_true, y_pred).items()
                    if not isinstance(value, list)
                },
            }
        )
    return pd.DataFrame(rows)


def calibration_table(frame: pd.DataFrame) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, 6)
    labels = [f"{bins[index]:.1f}-{bins[index + 1]:.1f}" for index in range(len(bins) - 1)]
    working = frame.copy()
    working["prediction_bin"] = pd.cut(
        working["predicted_yield"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    return (
        working.groupby("prediction_bin", observed=False)
        .agg(
            designs=("design_uid", "count"),
            predicted_mean=("predicted_yield", "mean"),
            observed_mean=("target_yield", "mean"),
            observed_feasible_rate=("true_feasible_90", "mean"),
        )
        .reset_index()
    )


def train_final_bundle(
    frame: pd.DataFrame,
    *,
    target_column: str,
    checkpoint: Path,
    n_estimators: int,
    device: str,
    output_root: Path,
    target_transform: str,
) -> dict[str, Any]:
    encoder = FeatureEncoder.fit(frame)
    x_train = encoder.transform(frame)
    y_train_raw = pd.to_numeric(frame[target_column], errors="raise").to_numpy(dtype=np.float32)
    attempts = pd.to_numeric(frame["mc_attempts"], errors="raise").to_numpy(dtype=np.float32)
    y_train_model = forward_target(y_train_raw, attempts, target_transform).astype(np.float32)
    model = build_model(
        checkpoint=checkpoint,
        n_estimators=n_estimators,
        random_state=20260717,
        device=device,
    )
    started = time.monotonic()
    model.fit(x_train, y_train_model)
    fitted_prediction = inverse_target(
        np.asarray(model.predict(x_train), dtype=float), target_transform
    )
    elapsed = time.monotonic() - started

    bundle = {
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "categorical_feature_indices": categorical_indices(),
        "ordinal_encoder": encoder.ordinal_encoder,
        "numeric_medians": encoder.numeric_medians,
        "x_train": x_train,
        "y_train_model": y_train_model,
        "target_transform": target_transform,
        "design_uid": frame["design_uid"].astype(str).to_numpy(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "n_estimators": n_estimators,
        "target_column": target_column,
        "feasibility_target": FEASIBILITY_TARGET,
        "variation_contract": "independent_per_retained_physical_mos",
        "predictor_contract": "design_features_to_empirical_mc_yield",
    }
    bundle_path = output_root / "models" / "tabpfn_incontext_training_bundle.joblib"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(bundle_path)

    fitted = frame.loc[:, ["design_uid", "architecture", "source_server", target_column]].copy()
    fitted["fitted_yield_in_sample_not_validation"] = fitted_prediction
    atomic_csv(fitted, output_root / "results" / "final_fit_in_sample_predictions.csv")
    return {
        "bundle_path": str(bundle_path),
        "bundle_sha256": sha256_file(bundle_path),
        "fit_elapsed_sec": elapsed,
        "in_sample_metrics_not_validation": regression_metrics(
            y_train_raw, fitted_prediction
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--target", default="target_yield")
    parser.add_argument(
        "--target-transform",
        choices=("raw", "jeffreys_logit"),
        default="raw",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="17,73,211")
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--final-n-estimators", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-208-sensitivity", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
    import torch

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"cached TabPFN checkpoint not found: {args.checkpoint}")
    frame = pd.read_csv(args.dataset)
    required = set(FEATURE_COLUMNS) | {
        "design_uid",
        "architecture",
        "source_server",
        "mc_attempts",
        "primary_tlim_ns",
        args.target,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"training dataset is missing columns: {missing}")
    if frame["design_uid"].duplicated().any():
        raise ValueError("design-level training dataset contains duplicate design_uid")
    if frame.loc[:, list(FEATURE_COLUMNS)].isna().any().any():
        bad = frame.loc[:, list(FEATURE_COLUMNS)].isna().sum()
        raise ValueError(f"training features contain missing values: {bad[bad > 0].to_dict()}")

    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    output_root = args.output_root.resolve()
    results_dir = output_root / "results"
    started = time.monotonic()

    predictions, folds, repeats = grouped_oof(
        frame,
        target_column=args.target,
        seeds=seeds,
        folds=args.folds,
        checkpoint=args.checkpoint,
        n_estimators=args.n_estimators,
        device=args.device,
        label="merged_203_208",
        target_transform=args.target_transform,
    )
    aggregate = aggregate_oof(predictions)
    y_true = aggregate["target_yield"].to_numpy(dtype=float)
    y_pred = aggregate["predicted_yield"].to_numpy(dtype=float)
    primary_metrics = {
        **regression_metrics(y_true, y_pred),
        **decision_metrics(y_true, y_pred),
    }

    atomic_csv(predictions, results_dir / "oof_predictions_all_repeats.csv")
    atomic_csv(aggregate, results_dir / "oof_predictions_by_design.csv")
    atomic_csv(folds, results_dir / "fold_metrics.csv")
    atomic_csv(repeats, results_dir / "repeat_metrics.csv")
    atomic_csv(
        grouped_metric_table(aggregate, "architecture"),
        results_dir / "architecture_metrics.csv",
    )
    atomic_csv(
        grouped_metric_table(aggregate, "source_server"),
        results_dir / "source_metrics.csv",
    )
    atomic_csv(calibration_table(aggregate), results_dir / "yield_calibration_bins.csv")

    sensitivity: dict[str, Any] = {}
    if not args.skip_208_sensitivity:
        only_208 = frame[frame["source_server"].astype(str) == "208"].reset_index(drop=True)
        sensitivity_predictions, sensitivity_folds, sensitivity_repeats = grouped_oof(
            only_208,
            target_column=args.target,
            seeds=seeds[:1],
            folds=args.folds,
            checkpoint=args.checkpoint,
            n_estimators=args.n_estimators,
            device=args.device,
            label="208_only_sensitivity",
            target_transform=args.target_transform,
        )
        sensitivity_aggregate = aggregate_oof(sensitivity_predictions)
        sy_true = sensitivity_aggregate["target_yield"].to_numpy(dtype=float)
        sy_pred = sensitivity_aggregate["predicted_yield"].to_numpy(dtype=float)
        sensitivity = {
            **regression_metrics(sy_true, sy_pred),
            **decision_metrics(sy_true, sy_pred),
        }
        atomic_csv(
            sensitivity_predictions,
            results_dir / "sensitivity_208_only_predictions_all_repeats.csv",
        )
        atomic_csv(
            sensitivity_aggregate,
            results_dir / "sensitivity_208_only_predictions_by_design.csv",
        )
        atomic_csv(sensitivity_folds, results_dir / "sensitivity_208_only_fold_metrics.csv")
        atomic_csv(
            sensitivity_repeats,
            results_dir / "sensitivity_208_only_repeat_metrics.csv",
        )

    # Explicit source holdout: fit only 208 designs and predict the six 203 designs.
    train_208 = frame[frame["source_server"].astype(str) == "208"].reset_index(drop=True)
    test_203 = frame[frame["source_server"].astype(str) == "203"].reset_index(drop=True)
    source_holdout: dict[str, Any] = {}
    if len(train_208) and len(test_203):
        encoder = FeatureEncoder.fit(train_208)
        source_model = build_model(
            checkpoint=args.checkpoint,
            n_estimators=args.n_estimators,
            random_state=203208,
            device=args.device,
        )
        source_train_raw = train_208[args.target].to_numpy(dtype=float)
        source_train_attempts = train_208["mc_attempts"].to_numpy(dtype=float)
        source_model.fit(
            encoder.transform(train_208),
            forward_target(
                source_train_raw, source_train_attempts, args.target_transform
            ),
        )
        source_prediction = inverse_target(
            np.asarray(source_model.predict(encoder.transform(test_203)), dtype=float),
            args.target_transform,
        )
        source_holdout_frame = test_203.loc[
            :, ["design_uid", "architecture", "source_server", "mc_attempts", args.target]
        ].copy()
        source_holdout_frame["predicted_yield"] = source_prediction
        source_holdout_frame["abs_error"] = (
            source_holdout_frame[args.target] - source_holdout_frame["predicted_yield"]
        ).abs()
        atomic_csv(source_holdout_frame, results_dir / "source_holdout_208_train_203_test.csv")
        source_holdout = {
            **regression_metrics(
                test_203[args.target].to_numpy(dtype=float), source_prediction
            ),
            **decision_metrics(
                test_203[args.target].to_numpy(dtype=float), source_prediction
            ),
        }

    final_fit = train_final_bundle(
        frame,
        target_column=args.target,
        checkpoint=args.checkpoint,
        n_estimators=args.final_n_estimators,
        device=args.device,
        output_root=output_root,
        target_transform=args.target_transform,
    )
    elapsed = time.monotonic() - started

    baseline_prediction = np.full(len(frame), frame[args.target].mean(), dtype=float)
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "designs": int(len(frame)),
        "source_design_counts": {
            str(key): int(value)
            for key, value in frame["source_server"].value_counts().sort_index().items()
        },
        "features": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "categorical_features": list(CATEGORICAL_COLUMNS),
        "target": args.target,
        "target_transform": args.target_transform,
        "primary_tlim_ns": float(frame["primary_tlim_ns"].iloc[0]),
        "feasibility_target": FEASIBILITY_TARGET,
        "evaluation": {
            "protocol": "repeated_shuffled_group_kfold_by_design_uid",
            "folds": args.folds,
            "seeds": seeds,
            "n_estimators": args.n_estimators,
            "primary_oof_ensemble_metrics": primary_metrics,
            "repeat_metric_mean": {
                column: float(repeats[column].mean())
                for column in ("mae", "rmse", "r2", "pearson_r")
            },
            "repeat_metric_std": {
                column: float(repeats[column].std(ddof=1))
                for column in ("mae", "rmse", "r2", "pearson_r")
            },
            "constant_mean_baseline": regression_metrics(
                frame[args.target].to_numpy(dtype=float), baseline_prediction
            ),
            "only_208_sensitivity": sensitivity,
            "source_holdout_208_train_203_test": source_holdout,
        },
        "final_fit": final_fit,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "tabpfn_device": args.device,
        "thread_limit": args.threads,
        "elapsed_sec": elapsed,
        "claims_limit": (
            f"Only {len(frame)} designs are available and targets are MC estimates. "
            "OOF results are early evidence, not a high-precision significance claim."
        ),
    }
    atomic_json(report, output_root / "training_audit.json")
    print(json.dumps(report, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
