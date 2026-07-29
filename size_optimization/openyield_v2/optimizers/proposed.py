#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Proposed optimizer: coarse search followed by differentiable refinement."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from pymoo.core.callback import Callback
from pymoo.core.mixed import (
    MixedVariableDuplicateElimination,
    MixedVariableMating,
    MixedVariableSampling,
)
from pymoo.core.problem import Problem
from pymoo.core.variable import Integer, Real
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

Array = np.ndarray


# =============================================================================
# 0. 项目路径与 OpenYield 工具
# =============================================================================

def setup_project_path() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir,
        script_dir.parent,
        script_dir.parents[2],
        Path.cwd(),
        Path.cwd().parent,
    ]
    for path in candidates:
        try:
            path = path.resolve()
        except Exception:
            continue
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return script_dir


SCRIPT_DIR = setup_project_path()

if __package__:
    from ..surrogate_utils import (  # noqa: E402
        seed_set,
        feature_engineering,
        DataLoader,
        TabPFNSurrogate,
        MultiTargetSurrogateManager,
        process_raw_to_system_metrics,
    )
else:
    from surrogate_utils import (  # type: ignore[no-redef]  # noqa: E402
        seed_set,
        feature_engineering,
        DataLoader,
        TabPFNSurrogate,
        MultiTargetSurrogateManager,
        process_raw_to_system_metrics,
    )
if __package__:
    from ..optimization_spec import (  # noqa: E402
        PHYSICAL_POSITIVE_METRICS,
        feasible_mask as shared_feasible_mask,
        load_problem_spec,
    )
    from ..output_schema import (  # noqa: E402
        create_run_directory,
        public_evaluations,
        public_pareto_front,
    )
else:
    from optimization_spec import (  # type: ignore[no-redef]  # noqa: E402
        PHYSICAL_POSITIVE_METRICS,
        feasible_mask as shared_feasible_mask,
        load_problem_spec,
    )
    from output_schema import (  # type: ignore[no-redef]  # noqa: E402
        create_run_directory,
        public_evaluations,
        public_pareto_front,
    )


# =============================================================================
# 1. 默认配置
# =============================================================================

SEED = 33
TOTAL_KB = 32
OUTPUT_COLS = 16

PACKAGE_ROOT = SCRIPT_DIR.parent
DEFAULT_6T_DATASET = str(PACKAGE_ROOT / "datasets" / "train_6t.csv")
DEFAULT_10T_DATASET = str(PACKAGE_ROOT / "datasets" / "train_10t.csv")
DEFAULT_OUTPUT_BASE = str(
    PACKAGE_ROOT / "runs" / "optimization" / "proposed"
)

DEFAULT_TRAIN_RATIO = 0.95
DEFAULT_TEST_RATIO = 0.05
# 0 表示不限制。这里的最大样本数是 6T+10T 合并后的总数。
DEFAULT_MAX_TRAIN_SAMPLES = 500
DEFAULT_DEVICE = "auto"
DEFAULT_BALANCE_TOPOLOGIES = False


# =============================================================================
# 2. pooled_union 输入变量与目标
# =============================================================================

SHARED_CONT_FEATURES = [
    "pu_width", "pd_width", "pg_width", "cell_length",
    "sa_p_width", "sa_n_width", "sa_length",
    "wld_nand_p_width", "wld_inv_p_width",
    "wld_nand_n_width", "wld_inv_n_width", "wld_length",
    "prc_p_width", "prc_length",
]

DERIVED_CONT_FEATURES = ["aspect_ratio", "log_rows", "log_cols"]
ARCH_FEATURES = ["rows", "cols"]

SHARED_CAT_FEATURES = [
    "pu_model", "pd_model", "pg_model",
    "sa_p_model", "sa_n_model",
    "wld_nand_p_model", "wld_inv_p_model",
    "wld_nand_n_model", "wld_inv_n_model",
    "prc_p_model",
]

TOPOLOGY_CAT_FEATURES = ["topology", "fd_model"]
ALL_CAT_FEATURES = SHARED_CAT_FEATURES + TOPOLOGY_CAT_FEATURES

RAW_TARGETS = [
    "hold_snm", "read_snm", "write_snm",
    "raw_read_delay", "raw_write_delay",
    "read_pstc", "read_pdyn", "write_pstc", "write_pdyn",
    "single_array_area",
]

SYSTEM_METRICS = [
    "hold_snm", "read_snm", "write_snm",
    "read_delay", "write_delay",
    "read_power", "write_power", "area",
]

# ============================================================================
# 唯一的多目标配置入口（可任意改成 M 个目标）
#
# 例 1，原始二目标：
# OBJECTIVE_SPECS = [("power_delay_product", "min"), ("min_snm", "max")]
# 例 2，三目标：
# OBJECTIVE_SPECS = [("power_delay_product", "min"), ("min_snm", "max"),
#                    ("area", "min")]
# 例 3，四目标：
# OBJECTIVE_SPECS = [("max_power", "min"), ("max_delay", "min"),
#                    ("area", "min"), ("min_snm", "max")]
# 可选列必须能由 add_optimization_metric_columns 直接得到。
# 优化、Pareto 和 refine 使用 OBJECTIVE_SPECS 中的全部 M 个目标。
# ============================================================================
OBJECTIVE_SPECS: List[Tuple[str, str]] = [
    ("max_power", "min"),
    ("max_delay", "min"),
    ("area", "min"),
    ("min_snm", "max"),
]
OBJECTIVE_NAMES = [name for name, _ in OBJECTIVE_SPECS]
OBJECTIVE_DIRECTIONS = dict(OBJECTIVE_SPECS)
PROBLEM_CONSTRAINTS: List[Dict[str, Any]] = []
ENFORCE_PHYSICAL_VALIDITY = True
OBJECTIVE_LABELS = {
    "power_delay_product": "max_power × max_delay",
    "min_snm": "min_snm",
    "max_delay": "max_delay",
    "max_power": "max_power",
    "area": "area",
}

# 真正连续优化的参数：公共尺寸 + 10T 隐变量 fd_width。
# 6T 时 fd_width 输入会被条件门控为 0。
OPT_CONT_VARS = SHARED_CONT_FEATURES + ["fd_width"]


def configure_problem(problem_config: str | Path) -> None:
    global OBJECTIVE_SPECS
    global OBJECTIVE_NAMES
    global OBJECTIVE_DIRECTIONS
    global PROBLEM_CONSTRAINTS
    global ENFORCE_PHYSICAL_VALIDITY
    objectives, constraints, enforce_physical = load_problem_spec(problem_config)
    OBJECTIVE_SPECS = [
        (str(item["source"]), str(item["direction"])) for item in objectives
    ]
    OBJECTIVE_NAMES = [name for name, _ in OBJECTIVE_SPECS]
    OBJECTIVE_DIRECTIONS = dict(OBJECTIVE_SPECS)
    PROBLEM_CONSTRAINTS = constraints
    ENFORCE_PHYSICAL_VALIDITY = enforce_physical


# =============================================================================
# 3. 通用工具
# =============================================================================

def seed_everything(seed: int) -> None:
    seed_set(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但 torch.cuda.is_available() 为 False。")
    return requested


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def safe_log10(values: Array, eps: float = 1e-30) -> Array:
    values = np.asarray(values, dtype=float)
    return np.log10(np.maximum(values, eps))


def add_optimization_metric_columns(system_df: pd.DataFrame) -> pd.DataFrame:
    """由系统级指标生成联合优化目标。"""
    df = system_df.copy()
    dup_cols = df.columns[df.columns.duplicated()].unique().tolist()
    if dup_cols:
        print(f"[Warning] 发现重复列，保留最后一列: {dup_cols}")
        df = df.loc[:, ~df.columns.duplicated(keep="last")].copy()

    for col in SYSTEM_METRICS + ["min_snm", "max_delay", "max_power", "power_delay_product"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if all(c in df.columns for c in ["hold_snm", "read_snm", "write_snm"]):
        df["min_snm"] = df[["hold_snm", "read_snm", "write_snm"]].min(axis=1)
    if all(c in df.columns for c in ["read_delay", "write_delay"]):
        df["max_delay"] = df[["read_delay", "write_delay"]].max(axis=1)
    if all(c in df.columns for c in ["read_power", "write_power"]):
        df["max_power"] = df[["read_power", "write_power"]].max(axis=1)
    if all(c in df.columns for c in ["max_power", "max_delay"]):
        df["power_delay_product"] = df["max_power"] * df["max_delay"]

    if all(c in df.columns for c in OBJECTIVE_NAMES):
        df["cost_log_pdp"] = safe_log10(df["power_delay_product"].to_numpy(float))
        df["cost_neg_min_snm"] = -df["min_snm"].to_numpy(float)
    return df


def objective_cost_matrix(df: pd.DataFrame, *, for_distance: bool = False) -> Array:
    """按 OBJECTIVE_SPECS 全部转成最小化方向。"""
    df = add_optimization_metric_columns(df)
    columns: List[Array] = []
    for name, direction in OBJECTIVE_SPECS:
        values = df[name].to_numpy(float)
        if for_distance and name == "power_delay_product":
            values = safe_log10(values)
        columns.append(values if direction == "min" else -values)
    return np.column_stack(columns)


def is_pareto_mask(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    valid = np.all(np.isfinite(values), axis=1)
    result = np.zeros(len(values), dtype=bool)
    idx = np.flatnonzero(valid)
    F = values[valid]
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if not keep[i]:
            continue
        dominated = np.any(
            np.all(F <= F[i], axis=1)
            & np.any(F < F[i], axis=1)
        )
        if dominated:
            keep[i] = False
    result[idx[keep]] = True
    return result


def pareto_front(df: pd.DataFrame, *, feasible_only: bool = True) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = add_optimization_metric_columns(df)
    if feasible_only and "is_feasible" in work.columns:
        work = work[_as_bool_series(work["is_feasible"])].copy()
    if work.empty:
        return work.reset_index(drop=True)
    mask = is_pareto_mask(objective_cost_matrix(work))
    out = work.loc[mask].copy()
    return out.sort_values(OBJECTIVE_NAMES[0]).reset_index(drop=True)


def deduplicate_designs(df: pd.DataFrame, decimals: int = 13) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    design_cols = (
        ["topology", "rows", "cols"]
        + SHARED_CONT_FEATURES
        + SHARED_CAT_FEATURES
        + ["fd_width", "fd_model"]
    )
    available = [c for c in design_cols if c in df.columns]
    if not available:
        return df.reset_index(drop=True)
    tmp = df[available].copy()
    for col in SHARED_CONT_FEATURES + ["fd_width"]:
        if col in tmp.columns:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce").round(decimals)
    keep = ~tmp.duplicated(keep="first")
    return df.loc[keep].copy().reset_index(drop=True)


def generate_preference_weights(
    num_preferences: int,
    *,
    edge: float = 0.10,
    seed: int = SEED,
) -> Array:
    """为任意 M 目标生成正且和为 1 的确定性偏好权重。"""
    if num_preferences <= 0:
        raise ValueError("num_preferences 必须为正。")
    m = len(OBJECTIVE_NAMES)
    if m < 2:
        raise ValueError("OBJECTIVE_SPECS 至少需要两个目标。")
    if not 0.0 <= edge < 1.0 / m:
        raise ValueError(f"preference_edge 必须在 [0, {1.0/m:.6g})，当前目标数 M={m}。")
    if num_preferences == 1:
        return np.full((1, m), 1.0 / m, dtype=float)
    if m == 2:
        edge = float(np.clip(edge, 1e-4, 0.4999))
        first = np.linspace(edge, 1.0 - edge, num_preferences)
        return np.column_stack([first, 1.0 - first]).astype(float)

    # 先构造大量 Dirichlet 候选，再用最远点采样选覆盖均匀的 num_preferences 个。
    rng = np.random.default_rng(seed)
    candidates = np.vstack([
        np.eye(m), np.full((1, m), 1.0 / m),
        rng.dirichlet(np.ones(m), size=max(2048, 128 * num_preferences)),
    ])
    candidates = edge + (1.0 - m * edge) * candidates
    chosen = [int(np.argmin(np.linalg.norm(candidates - 1.0 / m, axis=1)))]
    while len(chosen) < min(num_preferences, len(candidates)):
        distances = np.linalg.norm(
            candidates[:, None, :] - candidates[np.asarray(chosen)][None, :, :], axis=2
        ).min(axis=1)
        distances[chosen] = -np.inf
        chosen.append(int(np.argmax(distances)))
    return candidates[np.asarray(chosen)]


def tau_schedule(step: int, steps: int, tau_start: float, tau_end: float) -> float:
    if steps <= 0:
        return float(tau_end)
    ratio = float(step) / float(steps)
    if tau_start <= 0 or tau_end <= 0:
        return float(max(tau_end, 1e-4))
    return float(tau_start * ((tau_end / tau_start) ** ratio))


# =============================================================================
# 4. 数据统一、划分与 pooled_union 代理模型
# =============================================================================

def harmonize_topology_dataframe(raw_df: pd.DataFrame, topology: str) -> pd.DataFrame:
    topology = str(topology).upper()
    if topology not in {"6T", "10T"}:
        raise ValueError(f"不支持的拓扑: {topology}")

    df = raw_df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df["topology"] = topology
    df["source_row"] = np.arange(len(df), dtype=int)

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.strip().str.lower() == "ok"]
    for flag_col in ["valid_snm", "valid_delay", "valid_design"]:
        if flag_col in df.columns:
            df = df[_as_bool_series(df[flag_col])]

    if topology == "6T":
        df["fd_present"] = 0.0
        df["fd_width"] = 0.0
        df["fd_model"] = "NOT_APPLICABLE"
    else:
        df["fd_present"] = 1.0

    required = (
        SHARED_CONT_FEATURES
        + ["fd_present", "fd_width"]
        + ARCH_FEATURES
        + SHARED_CAT_FEATURES
        + TOPOLOGY_CAT_FEATURES
        + RAW_TARGETS
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{topology} 数据集缺少必要列:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    numeric_cols = (
        SHARED_CONT_FEATURES
        + ["fd_present", "fd_width"]
        + ARCH_FEATURES
        + RAW_TARGETS
    )
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ALL_CAT_FEATURES:
        df[col] = df[col].where(df[col].notna(), np.nan)
        df[col] = df[col].astype("string").str.strip()

    df = df.dropna(subset=required).copy()

    positive_cols = (
        SHARED_CONT_FEATURES
        + ARCH_FEATURES
        + [
            "hold_snm", "read_snm", "write_snm",
            "raw_read_delay", "raw_write_delay", "single_array_area",
        ]
    )
    for col in positive_cols:
        df = df[df[col] > 0]
    if topology == "10T":
        df = df[df["fd_width"] > 0]

    df["sample_uid"] = topology + ":" + df["source_row"].astype(str)
    return df.reset_index(drop=True)


def _architecture_stratify_key(df: pd.DataFrame) -> Optional[pd.Series]:
    key = df["rows"].astype(str) + "x" + df["cols"].astype(str)
    counts = key.value_counts()
    if len(counts) > 1 and int(counts.min()) >= 2:
        return key
    return None


def split_one_topology(
    df: pd.DataFrame,
    *,
    train_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按拓扑内部划分训练/测试集，尽量按 rows×cols 分层。"""
    total_ratio = float(train_ratio) + float(test_ratio)
    if total_ratio <= 0:
        raise ValueError("train_ratio + test_ratio 必须大于 0。")
    normalized_test_ratio = float(test_ratio) / total_ratio
    stratify = _architecture_stratify_key(df)
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=normalized_test_ratio,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        train_df, test_df = train_test_split(
            df,
            test_size=normalized_test_ratio,
            random_state=seed,
            shuffle=True,
        )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def limit_rows_stratified(
    df: pd.DataFrame,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    """
    对 pooled 数据做总量限制，同时尽量保持 topology + rows×cols 的组成。

    max_rows <= 0 表示不限制。只要预算不少于分层数量，每个已有分层至少保留1条。
    """
    max_rows = int(max_rows)
    if max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)
    if max_rows < 2:
        return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    work = df.copy()
    work["__stratum__"] = (
        work["topology"].astype(str)
        + "|"
        + work["rows"].astype(int).astype(str)
        + "x"
        + work["cols"].astype(int).astype(str)
    )
    counts = work["__stratum__"].value_counts().sort_index()
    strata = counts.index.tolist()
    rng = np.random.default_rng(seed)

    # 按比例给每层分配额度。
    raw_quota = counts.to_numpy(float) / float(len(work)) * max_rows
    quota = np.floor(raw_quota).astype(int)
    if max_rows >= len(strata):
        quota = np.maximum(quota, 1)
    quota = np.minimum(quota, counts.to_numpy(int))

    # 若超过预算，从配额较多的层逐步减；若不足，则按小数余量和剩余容量补齐。
    while int(quota.sum()) > max_rows:
        candidates = np.flatnonzero(quota > (1 if max_rows >= len(strata) else 0))
        if len(candidates) == 0:
            candidates = np.flatnonzero(quota > 0)
        idx = int(candidates[np.argmax(quota[candidates])])
        quota[idx] -= 1

    fractional = raw_quota - np.floor(raw_quota)
    while int(quota.sum()) < max_rows:
        capacity = counts.to_numpy(int) - quota
        candidates = np.flatnonzero(capacity > 0)
        if len(candidates) == 0:
            break
        best_score = fractional[candidates] + rng.random(len(candidates)) * 1e-9
        idx = int(candidates[np.argmax(best_score)])
        quota[idx] += 1
        fractional[idx] = 0.0

    pieces: List[pd.DataFrame] = []
    for i, stratum in enumerate(strata):
        n_take = int(quota[i])
        if n_take <= 0:
            continue
        group = work[work["__stratum__"] == stratum].drop(columns="__stratum__")
        pieces.append(group.sample(n=n_take, random_state=seed + i))

    result = pd.concat(pieces, ignore_index=True)
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def balance_pooled_training_data(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    counts = df["topology"].value_counts()
    if len(counts) < 2:
        return df.reset_index(drop=True)
    n = int(counts.min())
    pieces = []
    for i, (_, group) in enumerate(df.groupby("topology", sort=True)):
        pieces.append(group.sample(n=n, random_state=seed + i))
    return (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def build_pooled_union_schema(train_csv: str) -> Dict[str, Any]:
    return {
        "cont_features": (
            SHARED_CONT_FEATURES
            + ["fd_present", "fd_width"]
            + DERIVED_CONT_FEATURES
        ),
        "arch_features": ARCH_FEATURES,
        "cat_features": ALL_CAT_FEATURES,
        "targets": RAW_TARGETS,
        "filter_positive_targets": False,
        "filepath": train_csv,
    }


def train_pooled_union_model(
    train_df: pd.DataFrame,
    *,
    output_dir: Path,
    device: str,
    verbose_library_training: bool,
) -> Tuple[DataLoader, MultiTargetSurrogateManager, pd.DataFrame]:
    temp_context = tempfile.TemporaryDirectory(prefix="openyield_train_")
    train_csv = Path(temp_context.name) / "train_context.csv"
    train_df.to_csv(train_csv, index=False)

    dataloader = DataLoader(build_pooled_union_schema(str(train_csv)))
    dataloader.add_feature_engineering(feature_engineering)
    X_train, y_train, df_train_used = dataloader.load_and_preprocess()
    temp_context.cleanup()

    print("\n" + "=" * 108)
    print("[2/8] pooled_union TabPFN 训练")
    print("=" * 108)
    print(f"训练上下文: {X_train.shape[0]} 条 × {X_train.shape[1]} 个编码后特征")

    manager = MultiTargetSurrogateManager(target_names=RAW_TARGETS, device=device)
    if verbose_library_training:
        manager.fit_all(X_train, y_train)
    else:
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                manager.fit_all(X_train, y_train)
        except Exception:
            print(captured.getvalue())
            raise

    if not hasattr(manager, "predict_tensor"):
        raise AttributeError(
            "当前 MultiTargetSurrogateManager 没有 predict_tensor()，"
            "无法进行可微 refine。请检查 surrogate_utils.py 的 predict_tensor() 实现。"
        )

    shared_id = id(manager.shared_regressor)
    child_ids = {
        id(model._regressor)
        for model in manager.models.values()
        if isinstance(model, TabPFNSurrogate)
    }
    if child_ids and child_ids != {shared_id}:
        raise RuntimeError(
            "并非所有目标共享同一个 TabPFNRegressor："
            f"manager={shared_id}, children={child_ids}"
        )
    print(f"代理模型完成；{len(RAW_TARGETS)} 个目标共享 regressor id={shared_id}")
    return dataloader, manager, df_train_used.reset_index(drop=True)


def convert_raw_to_system_metrics(
    raw_df: pd.DataFrame,
    design_df: pd.DataFrame,
) -> pd.DataFrame:
    processed = process_raw_to_system_metrics(
        raw_df_metrics=raw_df[RAW_TARGETS].reset_index(drop=True),
        rows_array=design_df["rows"].to_numpy(),
        cols_array=design_df["cols"].to_numpy(),
        total_KB=TOTAL_KB,
        output_cols=OUTPUT_COLS,
    )
    return add_optimization_metric_columns(processed)


def build_continuous_bounds(
    train_df: pd.DataFrame,
    *,
    lower_q: float,
    upper_q: float,
    min_span: float = 1e-12,
) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    for col in SHARED_CONT_FEATURES:
        values = pd.to_numeric(train_df[col], errors="coerce").dropna().to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            raise ValueError(f"连续变量没有有效数据: {col}")
        low = float(np.quantile(values, lower_q))
        high = float(np.quantile(values, upper_q))
        if high <= low:
            center = float(np.median(values))
            delta = max(abs(center) * 0.05, min_span)
            low, high = center - delta, center + delta
        bounds[col] = (max(low, min_span), max(high, low + min_span))

    fd_values = pd.to_numeric(
        train_df.loc[train_df["topology"] == "10T", "fd_width"],
        errors="coerce",
    ).dropna().to_numpy(float)
    fd_values = fd_values[np.isfinite(fd_values) & (fd_values > 0)]
    if len(fd_values) == 0:
        raise ValueError("10T 训练数据没有有效 fd_width。")
    low = float(np.quantile(fd_values, lower_q))
    high = float(np.quantile(fd_values, upper_q))
    if high <= low:
        center = float(np.median(fd_values))
        delta = max(abs(center) * 0.05, min_span)
        low, high = center - delta, center + delta
    bounds["fd_width"] = (max(low, min_span), max(high, low + min_span))
    return bounds


def build_encoder_categories(dataloader: DataLoader) -> Dict[str, List[str]]:
    encoder = dataloader.preprocessor.named_transformers_["cat"]
    categories: Dict[str, List[str]] = {}
    for cat, values in zip(ALL_CAT_FEATURES, encoder.categories_):
        categories[cat] = [str(v) for v in values]
    return categories


def build_shared_cat_choices(encoder_categories: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {cat: list(encoder_categories[cat]) for cat in SHARED_CAT_FEATURES}


def build_fd_model_choices_10t(encoder_categories: Dict[str, List[str]]) -> List[str]:
    choices = [v for v in encoder_categories["fd_model"] if v != "NOT_APPLICABLE"]
    if not choices:
        raise ValueError("编码器中没有 10T fd_model 候选。")
    return choices


def build_architecture_templates(train_df: pd.DataFrame) -> List[Dict[str, int]]:
    arch_sets = []
    for topo in ["6T", "10T"]:
        part = train_df[train_df["topology"] == topo]
        arch_sets.append(set(zip(part["rows"].astype(int), part["cols"].astype(int))))
    common = sorted(arch_sets[0].intersection(arch_sets[1]))
    if not common:
        raise RuntimeError("6T/10T 没有共同架构，无法公平联合优化。")

    total_bits = TOTAL_KB * 1024 * 8
    result = []
    for rows, cols in common:
        result.append({
            "rows": int(rows),
            "cols": int(cols),
            "num_arrays": int(max(math.ceil(total_bits / (rows * cols)), 1)),
        })
    return result


def find_arch_index(
    architectures: List[Dict[str, int]],
    rows: Any,
    cols: Any,
) -> Optional[int]:
    try:
        rows_i, cols_i = int(rows), int(cols)
    except Exception:
        return None
    for i, arch in enumerate(architectures):
        if int(arch["rows"]) == rows_i and int(arch["cols"]) == cols_i:
            return i
    return None


def default_shared_cat_indices(
    shared_choices: Dict[str, List[str]],
    train_df: pd.DataFrame,
) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for cat, choices in shared_choices.items():
        mode = train_df[cat].astype(str).mode()
        value = str(mode.iloc[0]) if len(mode) else choices[0]
        result[cat] = choices.index(value) if value in choices else 0
    return result


def default_fd_model_index(fd_choices: List[str], train_df: pd.DataFrame) -> int:
    values = train_df.loc[train_df["topology"] == "10T", "fd_model"].astype(str)
    mode = values.mode()
    value = str(mode.iloc[0]) if len(mode) else fd_choices[0]
    return fd_choices.index(value) if value in fd_choices else 0


def default_fd_width_value(bounds: Dict[str, Tuple[float, float]], train_df: pd.DataFrame) -> float:
    values = pd.to_numeric(
        train_df.loc[train_df["topology"] == "10T", "fd_width"],
        errors="coerce",
    ).dropna().to_numpy(float)
    lo, hi = bounds["fd_width"]
    if len(values):
        return float(np.clip(np.median(values), lo, hi))
    return 0.5 * (lo + hi)


def print_design_space(
    bounds: Dict[str, Tuple[float, float]],
    shared_choices: Dict[str, List[str]],
    fd_choices: List[str],
    architectures: List[Dict[str, int]],
) -> None:
    print("\n" + "=" * 108)
    print("[3/8] 联合可微设计空间")
    print("=" * 108)
    print("拓扑变量: topology ∈ {6T, 10T}；refine 默认固定每个代表点的拓扑")
    print(f"共同架构 ({len(architectures)}): {[(a['rows'], a['cols']) for a in architectures]}")
    print(f"连续变量: {len(SHARED_CONT_FEATURES)} 个公共尺寸 + fd_width")
    print(f"公共分类变量: {len(SHARED_CAT_FEATURES)} 个；本版本默认固定")
    print(f"10T fd_model: {fd_choices}")
    print("连续边界:")
    for name, (lo, hi) in bounds.items():
        print(f"  {name:24s}: {lo:.6e} ~ {hi:.6e}")
    print("公共分类候选:")
    for name, values in shared_choices.items():
        print(f"  {name:24s}: {values}")


# =============================================================================
# 6. Coarse Pareto reference points
# =============================================================================

def check_hard_constraints(
    metrics_df: pd.DataFrame,
    *,
    enabled: bool,
    min_snm_limit: float,
    max_delay_limit: float,
    max_power_limit: float,
) -> Array:
    constraints = PROBLEM_CONSTRAINTS if enabled else []
    return shared_feasible_mask(
        metrics_df,
        constraints,
        enforce_physical_validity=ENFORCE_PHYSICAL_VALIDITY,
    )


def standardize_reference_dataframe(
    df: pd.DataFrame,
    *,
    bounds: Dict[str, Tuple[float, float]],
    shared_choices: Dict[str, List[str]],
    fd_choices: List[str],
    architectures: List[Dict[str, int]],
    constraint_enabled: bool,
    min_snm_limit: float,
    max_delay_limit: float,
    max_power_limit: float,
) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")].copy()

    if "topology" not in df.columns and "topology_id" in df.columns:
        topo_id = pd.to_numeric(df["topology_id"], errors="coerce")
        df["topology"] = np.where(topo_id >= 0.5, "10T", "6T")
    if "topology" not in df.columns:
        raise ValueError("参考前沿缺少 topology（或 topology_id）列。")
    df["topology"] = df["topology"].astype(str).str.strip().str.upper()
    df = df[df["topology"].isin(["6T", "10T"])].copy()

    required_design = SHARED_CONT_FEATURES + ARCH_FEATURES + SHARED_CAT_FEATURES
    missing = [c for c in required_design if c not in df.columns]
    if missing:
        raise ValueError(
            "参考前沿缺少联合设计列:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    for col in SHARED_CONT_FEATURES + ARCH_FEATURES + ["fd_width"] + SYSTEM_METRICS + [
        "min_snm", "max_delay", "max_power", "power_delay_product"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_design).copy()
    df["rows"] = pd.to_numeric(df["rows"], errors="coerce")
    df["cols"] = pd.to_numeric(df["cols"], errors="coerce")
    df = df.dropna(subset=["rows", "cols"]).copy()
    df["rows"] = df["rows"].astype(int)
    df["cols"] = df["cols"].astype(int)

    # 只保留第二段代码中定义的 6T/10T 共同架构。
    common_pairs = {(a["rows"], a["cols"]) for a in architectures}
    arch_valid = [(r, c) in common_pairs for r, c in zip(df["rows"], df["cols"])]
    dropped_arch = int(len(df) - np.sum(arch_valid))
    if dropped_arch:
        print(f"[Reference] 丢弃 {dropped_arch} 个不属于共同架构的点。")
    df = df.loc[arch_valid].copy()

    for col in SHARED_CONT_FEATURES:
        lo, hi = bounds[col]
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(lo, hi)

    # 公共类别必须是本次编码器已见类别。
    category_valid = np.ones(len(df), dtype=bool)
    for cat, choices in shared_choices.items():
        values = df[cat].astype(str).str.strip()
        bad = ~values.isin(choices)
        if bad.any():
            print(f"[Reference] {cat} 有 {int(bad.sum())} 行类别未被本次编码器见过，将丢弃。")
        category_valid &= ~bad.to_numpy()
        df[cat] = values
    df = df.loc[category_valid].copy()

    if "fd_width" not in df.columns:
        df["fd_width"] = 0.0
    if "fd_model" not in df.columns:
        df["fd_model"] = "NOT_APPLICABLE"

    is_6t = df["topology"] == "6T"
    df.loc[is_6t, "fd_present"] = 0.0
    df.loc[is_6t, "fd_width"] = 0.0
    df.loc[is_6t, "fd_model"] = "NOT_APPLICABLE"

    is_10t = ~is_6t
    df.loc[is_10t, "fd_present"] = 1.0
    lo_fd, hi_fd = bounds["fd_width"]
    df.loc[is_10t, "fd_width"] = pd.to_numeric(
        df.loc[is_10t, "fd_width"], errors="coerce"
    ).clip(lo_fd, hi_fd)
    df.loc[is_10t, "fd_model"] = df.loc[is_10t, "fd_model"].astype(str)
    fd_valid = (~is_10t) | (
        df["fd_width"].notna()
        & (df["fd_width"] > 0)
        & df["fd_model"].isin(fd_choices)
    )
    if (~fd_valid).any():
        print(f"[Reference] 丢弃 {int((~fd_valid).sum())} 个无效 10T FD 组合。")
    df = df.loc[fd_valid].copy()

    total_bits = TOTAL_KB * 1024 * 8
    df["num_arrays"] = np.maximum(
        np.ceil(total_bits / (df["rows"] * df["cols"])).astype(int),
        1,
    )

    df = add_optimization_metric_columns(df)
    missing_obj = [c for c in OBJECTIVE_NAMES if c not in df.columns]
    if missing_obj:
        raise ValueError(
            "参考前沿必须包含目标列，或包含可推导这些目标的系统指标。"
            f" 缺少: {missing_obj}"
        )
    finite = np.all(np.isfinite(df[OBJECTIVE_NAMES].to_numpy(float)), axis=1)
    if "power_delay_product" in OBJECTIVE_NAMES:
        finite &= df["power_delay_product"].to_numpy(float) > 0
    df = df.loc[finite].copy().reset_index(drop=True)

    df["is_feasible"] = check_hard_constraints(
        df,
        enabled=constraint_enabled,
        min_snm_limit=min_snm_limit,
        max_delay_limit=max_delay_limit,
        max_power_limit=max_power_limit,
    )
    return deduplicate_designs(df)


class CoarseSearchProblem(Problem):
    """The Proposed method's self-contained mixed-variable coarse stage."""

    def __init__(
        self,
        *,
        manager: MultiTargetSurrogateManager,
        dataloader: DataLoader,
        bounds: Dict[str, Tuple[float, float]],
        shared_choices: Dict[str, List[str]],
        fd_choices: List[str],
        architectures: List[Dict[str, int]],
        evaluation_budget: int,
        constraints_enabled: bool,
    ) -> None:
        self.manager = manager
        self.dataloader = dataloader
        self.bounds = bounds
        self.shared_choices = shared_choices
        self.fd_choices = fd_choices
        self.architectures = architectures
        self.evaluation_budget = int(evaluation_budget)
        self.constraints_enabled = bool(constraints_enabled)
        self.eval_counter = 0
        self.records: List[pd.DataFrame] = []

        variables: Dict[str, Any] = {
            "topology_id": Integer(bounds=(0, 1)),
            "arch_id": Integer(bounds=(0, len(architectures) - 1)),
        }
        for name in SHARED_CONT_FEATURES:
            variables[name] = Real(bounds=bounds[name])
        variables["fd_width"] = Real(bounds=bounds["fd_width"])
        for name in SHARED_CAT_FEATURES:
            variables[f"{name}_id"] = Integer(
                bounds=(0, len(shared_choices[name]) - 1)
            )
        variables["fd_model_id"] = Integer(bounds=(0, len(fd_choices) - 1))
        super().__init__(
            vars=variables,
            n_obj=len(OBJECTIVE_NAMES),
            n_ieq_constr=1,
        )

    def candidate_dataframe(self, population: Any) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for individual in population:
            topology = "10T" if int(individual["topology_id"]) == 1 else "6T"
            architecture = self.architectures[int(individual["arch_id"])]
            row: Dict[str, Any] = {
                "topology": topology,
                "rows": int(architecture["rows"]),
                "cols": int(architecture["cols"]),
                "num_arrays": int(architecture["num_arrays"]),
            }
            for name in SHARED_CONT_FEATURES:
                row[name] = float(individual[name])
            for name in SHARED_CAT_FEATURES:
                row[name] = self.shared_choices[name][
                    int(individual[f"{name}_id"])
                ]
            if topology == "10T":
                row["fd_present"] = 1.0
                row["fd_width"] = float(individual["fd_width"])
                row["fd_model"] = self.fd_choices[int(individual["fd_model_id"])]
            else:
                row["fd_present"] = 0.0
                row["fd_width"] = 0.0
                row["fd_model"] = "NOT_APPLICABLE"
            rows.append(row)
        return pd.DataFrame(rows)

    def predict(self, candidates: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        encoded = self.dataloader.transform_features(candidates)
        raw = pd.DataFrame(self.manager.predict(encoded, return_std=False))[RAW_TARGETS]
        metrics = convert_raw_to_system_metrics(raw, candidates)
        return raw.reset_index(drop=True), metrics.reset_index(drop=True)

    def _evaluate(self, population: Any, out: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        candidates = self.candidate_dataframe(population).reset_index(drop=True)
        requested = len(candidates)
        remaining = max(self.evaluation_budget - self.eval_counter, 0)
        count = min(requested, remaining)
        objective_values = np.full(
            (requested, len(OBJECTIVE_NAMES)), 1.0e30, dtype=float
        )
        violations = np.ones((requested, 1), dtype=float)
        if count == 0:
            out["F"] = objective_values
            out["G"] = violations
            return

        evaluated = candidates.iloc[:count].reset_index(drop=True)
        raw, metrics = self.predict(evaluated)
        objective_values[:count] = objective_cost_matrix(metrics)
        feasible = check_hard_constraints(
            metrics,
            enabled=self.constraints_enabled,
            min_snm_limit=0.0,
            max_delay_limit=0.0,
            max_power_limit=0.0,
        )
        violations[:count, 0] = (~feasible).astype(float)
        out["F"] = objective_values
        out["G"] = violations

        first_evaluation = self.eval_counter + 1
        self.eval_counter += count
        record = pd.concat(
            [evaluated, raw.add_prefix("raw_"), metrics], axis=1
        )
        record.insert(
            0,
            "evaluation",
            np.arange(first_evaluation, first_evaluation + count),
        )
        record.insert(0, "stage", "coarse")
        record["is_feasible"] = feasible.astype(bool)
        record["source"] = "coarse"
        self.records.append(record)

    def evaluation_history(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        return pd.concat(self.records, ignore_index=True)


class CoarseBudgetTrace(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.started_at = time.time()
        self.records: List[Dict[str, Any]] = []

    def notify(self, algorithm: Any) -> None:
        self.records.append(
            {
                "generation": int(algorithm.n_gen),
                "n_eval": int(algorithm.evaluator.n_eval),
                "elapsed_seconds": float(time.time() - self.started_at),
            }
        )


def run_coarse_search(
    *,
    manager: MultiTargetSurrogateManager,
    dataloader: DataLoader,
    bounds: Dict[str, Tuple[float, float]],
    shared_choices: Dict[str, List[str]],
    fd_choices: List[str],
    architectures: List[Dict[str, int]],
    max_evals: int,
    pop_size: int,
    seed: int,
    constraints_enabled: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the standalone coarse stage and derive Pareto from all its history."""

    problem = CoarseSearchProblem(
        manager=manager,
        dataloader=dataloader,
        bounds=bounds,
        shared_choices=shared_choices,
        fd_choices=fd_choices,
        architectures=architectures,
        evaluation_budget=max_evals,
        constraints_enabled=constraints_enabled,
    )
    duplicate = MixedVariableDuplicateElimination()
    algorithm = SMSEMOA(
        pop_size=int(pop_size),
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(eliminate_duplicates=duplicate),
        eliminate_duplicates=duplicate,
    )
    callback = CoarseBudgetTrace()
    minimize(
        problem,
        algorithm,
        termination=("n_eval", int(max_evals)),
        seed=int(seed),
        callback=callback,
        verbose=False,
    )
    history = problem.evaluation_history()
    if len(history) != int(max_evals):
        raise RuntimeError(
            f"Coarse stage evaluated {len(history)} points, expected {max_evals}."
        )
    feasible = history[_as_bool_series(history["is_feasible"])].copy()
    if feasible.empty:
        raise RuntimeError(
            "The coarse stage produced no feasible design. Relax the configured "
            "constraints or increase the coarse budget."
        )
    front = deduplicate_designs(pareto_front(feasible, feasible_only=False))
    front["stage"] = "coarse"
    front["source"] = "coarse"
    return front, history, pd.DataFrame(callback.records)


def select_representative_seeds(
    reference_front: pd.DataFrame,
    num_seeds: int,
    seed: int,
) -> pd.DataFrame:
    if num_seeds <= 0:
        raise ValueError("num_seeds 必须为正。")
    if len(reference_front) <= num_seeds:
        selected = reference_front.copy().reset_index(drop=True)
    else:
        costs = objective_cost_matrix(reference_front, for_distance=True)
        finite = np.all(np.isfinite(costs), axis=1)
        work = reference_front.loc[finite].copy().reset_index(drop=True)
        costs = costs[finite]
        cmin = costs.min(axis=0)
        cmax = costs.max(axis=0)
        Z = (costs - cmin) / np.maximum(cmax - cmin, 1e-30)

        chosen: List[int] = [int(np.argmin(np.mean(Z, axis=1)))]
        rng = np.random.default_rng(seed)
        while len(chosen) < min(num_seeds, len(work)):
            distances = np.linalg.norm(
                Z[:, None, :] - Z[np.asarray(chosen)][None, :, :], axis=2
            ).min(axis=1)
            distances[chosen] = -np.inf
            best = np.nanmax(distances)
            candidates = np.flatnonzero(np.isclose(distances, best))
            index = int(rng.choice(candidates)) if len(candidates) else int(np.nanargmax(distances))
            chosen.append(index)
        selected = work.iloc[chosen].copy().reset_index(drop=True)

    selected["seed_index"] = np.arange(len(selected), dtype=int)
    selected["seed_topology"] = selected["topology"].astype(str)
    return selected


def build_seed_points(
    selected: pd.DataFrame,
    *,
    bounds: Dict[str, Tuple[float, float]],
    shared_choices: Dict[str, List[str]],
    fd_choices: List[str],
    architectures: List[Dict[str, int]],
    default_shared_indices: Dict[str, int],
    default_fd_index: int,
    default_fd_width: float,
) -> List[Dict[str, Any]]:
    seeds: List[Dict[str, Any]] = []
    for seed_idx, row in selected.reset_index(drop=True).iterrows():
        topology = str(row["topology"]).upper()
        topology_idx = 1 if topology == "10T" else 0
        arch_idx = find_arch_index(architectures, row["rows"], row["cols"])
        if arch_idx is None:
            continue

        init_cont: List[float] = []
        for var in SHARED_CONT_FEATURES:
            lo, hi = bounds[var]
            value = float(np.clip(float(row[var]), lo, hi))
            init_cont.append(value)
        if topology == "10T":
            lo, hi = bounds["fd_width"]
            fd_value = float(np.clip(float(row["fd_width"]), lo, hi))
        else:
            fd_value = float(default_fd_width)
        init_cont.append(fd_value)

        shared_idx: Dict[str, int] = {}
        for cat, choices in shared_choices.items():
            value = str(row[cat])
            shared_idx[cat] = choices.index(value) if value in choices else default_shared_indices[cat]

        if topology == "10T" and str(row["fd_model"]) in fd_choices:
            fd_idx = fd_choices.index(str(row["fd_model"]))
        else:
            fd_idx = int(default_fd_index)

        seeds.append({
            "seed_index": int(seed_idx),
            "seed_label": f"coarse_representative_{seed_idx}",
            "seed_topology": topology,
            "init_topology_idx": topology_idx,
            "init_arch_idx": int(arch_idx),
            "init_cont": np.asarray(init_cont, dtype=np.float32),
            "init_shared_cat_indices": shared_idx,
            "init_fd_model_idx": int(fd_idx),
            "csv_seed_ref": {name: float(row[name]) for name in OBJECTIVE_NAMES},
            "reference_row": row.to_dict(),
        })
    if not seeds:
        raise RuntimeError("没有成功构建任何 coarse 代表点。")
    return seeds


# =============================================================================
# 7. 可微联合拓扑空间
# =============================================================================

@dataclass
class DifferentiableJointTopologySpace:
    bounds: Dict[str, Tuple[float, float]]
    architectures: List[Dict[str, int]]
    shared_cat_choices: Dict[str, List[str]]
    fd_model_choices: List[str]
    init_cont: np.ndarray
    init_topology_idx: int
    init_arch_idx: int
    init_shared_cat_indices: Dict[str, int]
    init_fd_model_idx: int
    topology_logit_bias: float
    discrete_logit_bias: float
    device: str
    use_gumbel: bool
    fixed_discrete: bool = False

    def __post_init__(self) -> None:
        # 固定离散模式下，所有离散状态都保存为不可变的整数索引。
        # 后续不会再通过 softmax/tau 构造 topology、architecture 或 model 输入。
        self._fixed_topology_idx = int(self.init_topology_idx)
        self._fixed_arch_idx = int(self.init_arch_idx)
        self._fixed_shared_cat_indices = {
            k: int(v) for k, v in self.init_shared_cat_indices.items()
        }
        self._fixed_fd_model_idx = int(self.init_fd_model_idx)

        lows = np.asarray([self.bounds[v][0] for v in OPT_CONT_VARS], dtype=np.float32)
        highs = np.asarray([self.bounds[v][1] for v in OPT_CONT_VARS], dtype=np.float32)
        init = np.asarray(self.init_cont, dtype=np.float32)
        unit = (init - lows) / np.maximum(highs - lows, 1e-30)
        unit = np.clip(unit, 1e-4, 1.0 - 1e-4)
        init_z = np.log(unit / (1.0 - unit)).astype(np.float32)

        self.low_t = torch.tensor(lows, dtype=torch.float32, device=self.device)
        self.high_t = torch.tensor(highs, dtype=torch.float32, device=self.device)
        self.cont_z = torch.nn.Parameter(torch.tensor(init_z, dtype=torch.float32, device=self.device))

        topology_logits = torch.zeros(2, dtype=torch.float32, device=self.device)
        topology_logits[int(self.init_topology_idx)] = float(self.topology_logit_bias)
        self.topology_logits = torch.nn.Parameter(topology_logits)

        arch_logits = torch.zeros(len(self.architectures), dtype=torch.float32, device=self.device)
        arch_logits[int(self.init_arch_idx)] = float(self.discrete_logit_bias)
        self.arch_logits = torch.nn.Parameter(arch_logits)

        self.shared_cat_logits = torch.nn.ParameterDict()
        for cat, choices in self.shared_cat_choices.items():
            logits = torch.zeros(len(choices), dtype=torch.float32, device=self.device)
            logits[int(self.init_shared_cat_indices.get(cat, 0))] = float(self.discrete_logit_bias)
            self.shared_cat_logits[cat] = torch.nn.Parameter(logits)

        fd_logits = torch.zeros(len(self.fd_model_choices), dtype=torch.float32, device=self.device)
        fd_logits[int(self.init_fd_model_idx)] = float(self.discrete_logit_bias)
        self.fd_model_logits = torch.nn.Parameter(fd_logits)

        if self.fixed_discrete:
            # 这一步不是唯一保障；真正的保障是下面的概率函数直接返回严格 one-hot。
            # requires_grad=False 用来防止未来误把这些参数加入优化器。
            self.topology_logits.requires_grad_(False)
            self.arch_logits.requires_grad_(False)
            for parameter in self.shared_cat_logits.parameters():
                parameter.requires_grad_(False)
            self.fd_model_logits.requires_grad_(False)

    def continuous_parameters(self) -> List[torch.nn.Parameter]:
        return [self.cont_z]

    def discrete_parameters(self) -> List[torch.nn.Parameter]:
        if self.fixed_discrete:
            return []
        return [
            self.topology_logits,
            self.arch_logits,
            *list(self.shared_cat_logits.parameters()),
            self.fd_model_logits,
        ]

    def all_parameters(self) -> List[torch.nn.Parameter]:
        return self.continuous_parameters() + self.discrete_parameters()

    def cont_values(self) -> torch.Tensor:
        unit = torch.sigmoid(self.cont_z)
        return self.low_t + unit * (self.high_t - self.low_t)

    def continuous_unit_values(self) -> torch.Tensor:
        return torch.sigmoid(self.cont_z)

    def _strict_one_hot(self, length: int, index: int) -> torch.Tensor:
        result = torch.zeros(length, dtype=torch.float32, device=self.device)
        result[int(index)] = 1.0
        return result

    def _probs(self, logits: torch.Tensor, tau: float) -> torch.Tensor:
        if self.use_gumbel:
            return F.gumbel_softmax(logits, tau=float(tau), hard=False, dim=0)
        return F.softmax(logits / max(float(tau), 1e-6), dim=0)

    def topology_probs(self, tau: float) -> torch.Tensor:
        if self.fixed_discrete:
            return self._strict_one_hot(2, self._fixed_topology_idx)
        return self._probs(self.topology_logits, tau)

    def arch_probs(self, tau: float) -> torch.Tensor:
        if self.fixed_discrete:
            return self._strict_one_hot(len(self.architectures), self._fixed_arch_idx)
        return self._probs(self.arch_logits, tau)

    def shared_cat_probs(self, cat: str, tau: float) -> torch.Tensor:
        if self.fixed_discrete:
            return self._strict_one_hot(
                len(self.shared_cat_choices[cat]),
                self._fixed_shared_cat_indices[cat],
            )
        return self._probs(self.shared_cat_logits[cat], tau)

    def fd_model_probs(self, tau: float) -> torch.Tensor:
        if self.fixed_discrete:
            return self._strict_one_hot(
                len(self.fd_model_choices), self._fixed_fd_model_idx
            )
        return self._probs(self.fd_model_logits, tau)

    def hard_indices(self) -> Tuple[int, int, Dict[str, int], int]:
        if self.fixed_discrete:
            return (
                self._fixed_topology_idx,
                self._fixed_arch_idx,
                dict(self._fixed_shared_cat_indices),
                self._fixed_fd_model_idx,
            )
        topology_idx = int(torch.argmax(self.topology_logits.detach()).cpu().item())
        arch_idx = int(torch.argmax(self.arch_logits.detach()).cpu().item())
        shared_idx = {
            cat: int(torch.argmax(self.shared_cat_logits[cat].detach()).cpu().item())
            for cat in SHARED_CAT_FEATURES
        }
        fd_idx = int(torch.argmax(self.fd_model_logits.detach()).cpu().item())
        return topology_idx, arch_idx, shared_idx, fd_idx

    def hard_row(self) -> Dict[str, Any]:
        topology_idx, arch_idx, shared_idx, fd_idx = self.hard_indices()
        topology = "10T" if topology_idx == 1 else "6T"
        arch = self.architectures[arch_idx]
        cont = self.cont_values().detach().cpu().numpy().astype(float)
        cont_map = {name: float(value) for name, value in zip(OPT_CONT_VARS, cont)}

        row: Dict[str, Any] = {
            "topology_id": topology_idx,
            "topology": topology,
            "arch_id": arch_idx,
            "rows": int(arch["rows"]),
            "cols": int(arch["cols"]),
            "num_arrays": int(arch["num_arrays"]),
        }
        for name in SHARED_CONT_FEATURES:
            row[name] = cont_map[name]
        for cat in SHARED_CAT_FEATURES:
            idx = shared_idx[cat]
            row[f"{cat}_id"] = idx
            row[cat] = self.shared_cat_choices[cat][idx]

        if topology == "10T":
            row["fd_present"] = 1.0
            row["fd_width"] = cont_map["fd_width"]
            row["fd_model_id"] = fd_idx
            row["fd_model"] = self.fd_model_choices[fd_idx]
        else:
            row["fd_present"] = 0.0
            row["fd_width"] = 0.0
            row["fd_model_id"] = -1
            row["fd_model"] = "NOT_APPLICABLE"
        return row


def validate_hard_design_row(
    row: Dict[str, Any],
    *,
    bounds: Dict[str, Tuple[float, float]],
    architectures: List[Dict[str, int]],
    shared_cat_choices: Dict[str, List[str]],
    fd_model_choices: List[str],
) -> None:
    """在送入 TabPFN 前检查硬设计是否合法；发现非法组合立即报错，不保存伪结果。"""
    topology = str(row.get("topology", "")).upper()
    if topology not in {"6T", "10T"}:
        raise ValueError(f"非法硬拓扑: {topology}")

    arch_pairs = {(int(a["rows"]), int(a["cols"])) for a in architectures}
    pair = (int(row["rows"]), int(row["cols"]))
    if pair not in arch_pairs:
        raise ValueError(f"硬设计架构不属于共同架构集合: {pair}")

    for name in SHARED_CONT_FEATURES:
        value = float(row[name])
        lo, hi = bounds[name]
        if not np.isfinite(value) or value < lo - 1e-18 or value > hi + 1e-18:
            raise ValueError(f"硬设计连续变量越界: {name}={value}, bounds=({lo}, {hi})")

    for cat in SHARED_CAT_FEATURES:
        value = str(row[cat])
        if value not in shared_cat_choices[cat]:
            raise ValueError(f"硬设计类别非法: {cat}={value}")

    total_bits = TOTAL_KB * 1024 * 8
    expected_arrays = int(max(math.ceil(total_bits / (int(row["rows"]) * int(row["cols"]))), 1))
    if int(row["num_arrays"]) != expected_arrays:
        raise ValueError(
            f"num_arrays 不一致: 保存值={row['num_arrays']}, 应为={expected_arrays}"
        )

    if topology == "6T":
        if float(row["fd_present"]) != 0.0:
            raise ValueError("6T 硬设计必须 fd_present=0。")
        if abs(float(row["fd_width"])) > 1e-30:
            raise ValueError("6T 硬设计必须 fd_width=0。")
        if str(row["fd_model"]) != "NOT_APPLICABLE":
            raise ValueError("6T 硬设计必须 fd_model=NOT_APPLICABLE。")
    else:
        fd_width = float(row["fd_width"])
        lo, hi = bounds["fd_width"]
        if float(row["fd_present"]) != 1.0:
            raise ValueError("10T 硬设计必须 fd_present=1。")
        if not np.isfinite(fd_width) or fd_width <= 0 or fd_width < lo - 1e-18 or fd_width > hi + 1e-18:
            raise ValueError(f"10T fd_width 非法: {fd_width}, bounds=({lo}, {hi})")
        if str(row["fd_model"]) not in fd_model_choices:
            raise ValueError(f"10T fd_model 非法: {row['fd_model']}")


class DifferentiableJointObjective:
    def __init__(
        self,
        *,
        manager: MultiTargetSurrogateManager,
        dataloader: DataLoader,
        architectures: List[Dict[str, int]],
        encoder_categories: Dict[str, List[str]],
        shared_cat_choices: Dict[str, List[str]],
        fd_model_choices: List[str],
        bounds: Dict[str, Tuple[float, float]],
        device: str,
        constraint_enabled: bool,
        min_snm_limit: float,
        max_delay_limit: float,
        max_power_limit: float,
    ) -> None:
        self.manager = manager
        self.dataloader = dataloader
        self.architectures = architectures
        self.encoder_categories = encoder_categories
        self.shared_cat_choices = shared_cat_choices
        self.fd_model_choices = fd_model_choices
        self.bounds = bounds
        self.device = device
        self.constraint_enabled = constraint_enabled
        self.min_snm_limit = float(min_snm_limit)
        self.max_delay_limit = float(max_delay_limit)
        self.max_power_limit = float(max_power_limit)
        self.feature_names = list(dataloader.feature_names_out)

        self.arch_rows_t = torch.tensor(
            [a["rows"] for a in architectures], dtype=torch.float32, device=device
        )
        self.arch_cols_t = torch.tensor(
            [a["cols"] for a in architectures], dtype=torch.float32, device=device
        )
        self.arch_arrays_t = torch.tensor(
            [a["num_arrays"] for a in architectures], dtype=torch.float32, device=device
        )

    @staticmethod
    def _base_feature_name(feature_name: str) -> str:
        # 兼容 sklearn get_feature_names_out 可能产生的 num__/cat__ 前缀。
        return str(feature_name).split("__", 1)[-1]

    def _build_feature_tensor(
        self,
        space: DifferentiableJointTopologySpace,
        *,
        tau: float,
        hard: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cont = space.cont_values()
        cont_map = {name: cont[i] for i, name in enumerate(OPT_CONT_VARS)}

        if hard:
            topology_idx, arch_idx, shared_idx, fd_idx = space.hard_indices()
            topology_probs = torch.zeros(2, dtype=torch.float32, device=self.device)
            topology_probs[topology_idx] = 1.0
            arch_probs = torch.zeros(len(self.architectures), dtype=torch.float32, device=self.device)
            arch_probs[arch_idx] = 1.0
            shared_probs: Dict[str, torch.Tensor] = {}
            for cat in SHARED_CAT_FEATURES:
                probs = torch.zeros(
                    len(self.shared_cat_choices[cat]), dtype=torch.float32, device=self.device
                )
                probs[shared_idx[cat]] = 1.0
                shared_probs[cat] = probs
            fd_probs_actual = torch.zeros(
                len(self.fd_model_choices), dtype=torch.float32, device=self.device
            )
            fd_probs_actual[fd_idx] = 1.0
        else:
            topology_probs = space.topology_probs(tau)
            arch_probs = space.arch_probs(tau)
            shared_probs = {
                cat: space.shared_cat_probs(cat, tau)
                for cat in SHARED_CAT_FEATURES
            }
            fd_probs_actual = space.fd_model_probs(tau)

        p6 = topology_probs[0]
        p10 = topology_probs[1]
        rows = torch.sum(arch_probs * self.arch_rows_t)
        cols = torch.sum(arch_probs * self.arch_cols_t)
        arrays = torch.sum(arch_probs * self.arch_arrays_t)

        numeric: Dict[str, torch.Tensor] = {
            name: cont_map[name] for name in SHARED_CONT_FEATURES
        }
        numeric["fd_present"] = p10
        numeric["fd_width"] = p10 * cont_map["fd_width"]
        numeric["rows"] = rows
        numeric["cols"] = cols
        numeric["aspect_ratio"] = rows / torch.clamp(cols, min=1.0)
        numeric["log_rows"] = torch.log2(torch.clamp(rows, min=1.0))
        numeric["log_cols"] = torch.log2(torch.clamp(cols, min=1.0))

        # 各分类变量“类别值 -> 概率”的映射。
        cat_value_probs: Dict[str, Dict[str, torch.Tensor]] = {}
        for cat in SHARED_CAT_FEATURES:
            cat_value_probs[cat] = {
                value: shared_probs[cat][i]
                for i, value in enumerate(self.shared_cat_choices[cat])
            }

        cat_value_probs["topology"] = {
            "6T": p6,
            "10T": p10,
        }
        fd_map: Dict[str, torch.Tensor] = {"NOT_APPLICABLE": p6}
        for i, value in enumerate(self.fd_model_choices):
            fd_map[value] = p10 * fd_probs_actual[i]
        cat_value_probs["fd_model"] = fd_map

        values: List[torch.Tensor] = []
        for original_name in self.feature_names:
            name = self._base_feature_name(original_name)
            if name in numeric:
                values.append(numeric[name].reshape(1))
                continue

            matched = False
            # 长名称优先，避免任何潜在前缀歧义。
            for cat in sorted(ALL_CAT_FEATURES, key=len, reverse=True):
                prefix = f"{cat}_"
                if name.startswith(prefix):
                    category = name[len(prefix):]
                    probability = cat_value_probs.get(cat, {}).get(category)
                    if probability is None:
                        probability = torch.zeros((), dtype=torch.float32, device=self.device)
                    values.append(probability.reshape(1))
                    matched = True
                    break
            if not matched:
                raise KeyError(
                    f"无法构造编码后特征 {original_name!r}。"
                    "请检查 DataLoader.feature_names_out 与本脚本输入定义。"
                )

        X = torch.cat(values).reshape(1, -1)
        soft_state = {
            "rows": rows,
            "cols": cols,
            "arrays": arrays,
            "p6": p6,
            "p10": p10,
            "fd_width_candidate": cont_map["fd_width"],
        }
        return X, soft_state

    def _process_raw_tensor(
        self,
        raw: Dict[str, torch.Tensor],
        *,
        rows: torch.Tensor,
        cols: torch.Tensor,
        arrays: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """process_raw_to_system_metrics 的可微版本。"""
        log_arrays = torch.log2(torch.clamp(arrays, min=1.0))
        cs_delay_adder = 4.167213500e-11 * log_arrays
        mux_delay = 1.8e-10
        mux_power = 0.1e-6

        raw_read_delay = raw["raw_read_delay"].reshape(-1)[0]
        raw_write_delay = raw["raw_write_delay"].reshape(-1)[0]
        read_pstc = raw["read_pstc"].reshape(-1)[0]
        read_pdyn = raw["read_pdyn"].reshape(-1)[0]
        write_pstc = raw["write_pstc"].reshape(-1)[0]
        write_pdyn = raw["write_pdyn"].reshape(-1)[0]
        single_array_area = raw["single_array_area"].reshape(-1)[0]

        read_delay = raw_read_delay + cs_delay_adder + log_arrays * mux_delay
        write_delay = raw_write_delay + cs_delay_adder

        factor = torch.ceil(
            torch.tensor(float(OUTPUT_COLS), dtype=torch.float32, device=self.device)
            / torch.clamp(cols, min=1.0)
        )
        factor = torch.clamp(factor, min=1.0)
        multi = (cols < float(OUTPUT_COLS)).float()
        delay_factor = 1.0 + multi * (factor - 1.0)
        read_delay = read_delay * delay_factor
        write_delay = write_delay * delay_factor

        read_power_nom = read_pstc * arrays + read_pdyn
        write_power_nom = write_pstc * arrays + write_pdyn
        read_power_multi = read_pstc * arrays + read_pdyn * factor
        write_power_multi = write_pstc * arrays + write_pdyn * factor
        read_power = (1.0 - multi) * read_power_nom + multi * read_power_multi
        write_power = (1.0 - multi) * write_power_nom + multi * write_power_multi
        read_power = read_power - (arrays - 1.0) * log_arrays * mux_power

        # 与硬约束配合，保留一个极小正值，避免 PDP/对数数值异常。
        read_power = F.relu(read_power) + 1e-15
        write_power = F.relu(write_power) + 1e-15
        area = F.relu(single_array_area * arrays) + 1e-18

        return {
            "hold_snm": raw["hold_snm"].reshape(-1)[0],
            "read_snm": raw["read_snm"].reshape(-1)[0],
            "write_snm": raw["write_snm"].reshape(-1)[0],
            "read_delay": read_delay,
            "write_delay": write_delay,
            "read_power": read_power,
            "write_power": write_power,
            "area": area,
        }

    def predict_soft(
        self,
        space: DifferentiableJointTopologySpace,
        *,
        tau: float,
        hard_discrete: bool,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        X, soft_state = self._build_feature_tensor(
            space, tau=tau, hard=hard_discrete
        )
        raw = self.manager.predict_tensor(X)
        processed = self._process_raw_tensor(
            raw,
            rows=soft_state["rows"],
            cols=soft_state["cols"],
            arrays=soft_state["arrays"],
        )
        objectives = self.objective_values(processed)
        return raw, processed, objectives, soft_state

    @staticmethod
    def objective_values(processed: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        min_snm = torch.min(torch.stack([
            processed["hold_snm"],
            processed["read_snm"],
            processed["write_snm"],
        ]))
        max_delay = torch.max(torch.stack([
            processed["read_delay"],
            processed["write_delay"],
        ]))
        max_power = torch.max(torch.stack([
            processed["read_power"],
            processed["write_power"],
        ]))
        pdp = max_power * max_delay
        return {
            **processed,
            "min_snm": min_snm,
            "max_delay": max_delay,
            "max_power": max_power,
            "power_delay_product": pdp,
        }

    def seed_relative_costs(
        self,
        objectives: Dict[str, torch.Tensor],
        seed_ref: Dict[str, float],
    ) -> torch.Tensor:
        costs: List[torch.Tensor] = []
        for name, direction in OBJECTIVE_SPECS:
            ref = torch.tensor(float(seed_ref[name]), dtype=torch.float32, device=self.device)
            scale = torch.clamp(torch.abs(ref), min=1e-30)
            delta = objectives[name] - ref
            costs.append(delta / scale if direction == "min" else -delta / scale)
        return torch.stack(costs)

    def soft_constraint_violation(
        self,
        processed: Dict[str, torch.Tensor],
        objectives: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        terms: List[torch.Tensor] = []
        available = {**processed, **objectives}
        if ENFORCE_PHYSICAL_VALIDITY:
            for name in PHYSICAL_POSITIVE_METRICS:
                if name in available:
                    scale = torch.clamp(
                        torch.abs(available[name].detach()), min=1e-30
                    )
                    terms.append(F.relu(-available[name]) / scale)

        if self.constraint_enabled:
            for constraint in PROBLEM_CONSTRAINTS:
                name = str(constraint["metric"])
                if name not in available:
                    raise KeyError(
                        f"Constraint metric {name!r} is unavailable to refinement."
                    )
                value = float(constraint["value"])
                scale = max(abs(value), 1e-30)
                if str(constraint["operator"]) == "<=":
                    term = F.relu(available[name] - value) / scale
                else:
                    term = F.relu(value - available[name]) / scale
                terms.append(term)
        if not terms:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        return torch.sum(torch.stack([t.reshape(()) ** 2 for t in terms]))

    def refinement_loss(
        self,
        space: DifferentiableJointTopologySpace,
        *,
        weights: torch.Tensor,
        seed_ref: Dict[str, float],
        tau: float,
        mu: float,
        constraint_penalty: float,
        hard_discrete: bool,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        raw, processed, objectives, soft_state = self.predict_soft(
            space,
            tau=tau,
            hard_discrete=hard_discrete,
        )
        relative_costs = self.seed_relative_costs(objectives, seed_ref)
        weighted = weights * relative_costs
        mu_t = torch.tensor(max(float(mu), 1e-12), dtype=torch.float32, device=self.device)
        # Subtracting mu*log(M) is a constant shift: gradients and optimizer
        # trajectories are unchanged, while the coarse reference now reports
        # an intuitive score of exactly zero. Negative means aggregate
        # improvement and positive means aggregate deterioration.
        smooth_score = mu_t * (
            torch.logsumexp(weighted / mu_t, dim=0)
            - math.log(float(len(OBJECTIVE_NAMES)))
        )
        violation = self.soft_constraint_violation(processed, objectives)
        loss = smooth_score + float(constraint_penalty) * violation
        return (
            loss,
            smooth_score,
            violation,
            raw,
            processed,
            objectives,
            soft_state,
        )

    def evaluate_hard_design(
        self,
        space: DifferentiableJointTopologySpace,
    ) -> pd.DataFrame:
        """
        先将所有离散变量硬化成合法设计，再用该硬设计重新调用 TabPFN。

        返回的 raw/system/objective 指标全部来自硬设计预测，绝不复用 soft forward 的指标。
        """
        row = space.hard_row()
        validate_hard_design_row(
            row,
            bounds=self.bounds,
            architectures=self.architectures,
            shared_cat_choices=self.shared_cat_choices,
            fd_model_choices=self.fd_model_choices,
        )
        design_df = pd.DataFrame([row])
        X_hard = self.dataloader.transform_features(design_df)
        raw_pred = pd.DataFrame(
            self.manager.predict(X_hard, return_std=False)
        )[RAW_TARGETS]
        metrics = convert_raw_to_system_metrics(raw_pred, design_df)
        out = pd.concat([
            design_df.reset_index(drop=True),
            raw_pred.add_prefix("hard_tabpfn_raw_").reset_index(drop=True),
            metrics.reset_index(drop=True),
        ], axis=1)
        out["is_legal_hard_design"] = True
        out["metric_source"] = "hard_design_TabPFN_prediction"
        out["is_feasible"] = check_hard_constraints(
            out,
            enabled=self.constraint_enabled,
            min_snm_limit=self.min_snm_limit,
            max_delay_limit=self.max_delay_limit,
            max_power_limit=self.max_power_limit,
        )
        return out


# =============================================================================
# 8. 单个代表点 / 单个偏好权重的 refine
# =============================================================================

def evaluate_seed_surrogate(
    seed: Dict[str, Any],
    *,
    diff_obj: DifferentiableJointObjective,
    bounds: Dict[str, Tuple[float, float]],
    architectures: List[Dict[str, int]],
    shared_choices: Dict[str, List[str]],
    fd_choices: List[str],
    args: argparse.Namespace,
    device: str,
) -> pd.DataFrame:
    space = DifferentiableJointTopologySpace(
        bounds=bounds,
        architectures=architectures,
        shared_cat_choices=shared_choices,
        fd_model_choices=fd_choices,
        init_cont=seed["init_cont"],
        init_topology_idx=seed["init_topology_idx"],
        init_arch_idx=seed["init_arch_idx"],
        init_shared_cat_indices=seed["init_shared_cat_indices"],
        init_fd_model_idx=seed["init_fd_model_idx"],
        topology_logit_bias=args.topology_init_bias,
        discrete_logit_bias=args.discrete_init_bias,
        device=device,
        use_gumbel=False,
        fixed_discrete=True,
    )
    result = diff_obj.evaluate_hard_design(space)
    result["seed_index"] = int(seed["seed_index"])
    result["seed_topology"] = str(seed["seed_topology"])
    for name in OBJECTIVE_NAMES:
        result[f"csv_ref_{name}"] = float(seed["csv_seed_ref"][name])
    return result


def optimize_one_preference(
    *,
    seed: Dict[str, Any],
    pref_index: int,
    weights_np: Array,
    seed_ref: Dict[str, float],
    diff_obj: DifferentiableJointObjective,
    bounds: Dict[str, Tuple[float, float]],
    architectures: List[Dict[str, int]],
    shared_choices: Dict[str, List[str]],
    fd_choices: List[str],
    args: argparse.Namespace,
    device: str,
) -> pd.DataFrame:
    space = DifferentiableJointTopologySpace(
        bounds=bounds,
        architectures=architectures,
        shared_cat_choices=shared_choices,
        fd_model_choices=fd_choices,
        init_cont=seed["init_cont"],
        init_topology_idx=seed["init_topology_idx"],
        init_arch_idx=seed["init_arch_idx"],
        init_shared_cat_indices=seed["init_shared_cat_indices"],
        init_fd_model_idx=seed["init_fd_model_idx"],
        topology_logit_bias=args.topology_init_bias,
        discrete_logit_bias=args.discrete_init_bias,
        device=device,
        use_gumbel=bool(args.use_gumbel and args.optimize_discrete),
        fixed_discrete=bool(not args.optimize_discrete),
    )

    weights_np = np.asarray(weights_np, dtype=float)
    weights_np = np.clip(weights_np, 1e-8, None)
    weights_np = weights_np / weights_np.sum()
    weights_t = torch.tensor(weights_np, dtype=torch.float32, device=device)

    if args.optimize_discrete:
        optimizer = torch.optim.Adam([
            {"params": [space.cont_z], "lr": args.continuous_lr},
            {"params": [space.topology_logits], "lr": args.topology_lr},
            {"params": [space.arch_logits], "lr": args.discrete_lr},
            {"params": list(space.shared_cat_logits.parameters()), "lr": args.discrete_lr},
            {"params": [space.fd_model_logits], "lr": args.discrete_lr},
        ])
        hard_discrete = False
    else:
        # 严格连续优化：优化器中只有 cont_z。拓扑、架构和所有器件类型
        # 不只是“没有梯度更新”，而且前向输入也始终是种子的严格 one-hot。
        optimizer = torch.optim.Adam([
            {"params": [space.cont_z], "lr": args.continuous_lr},
        ])
        hard_discrete = True

    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(args.refine_steps), 1),
            eta_min=float(args.continuous_lr) * float(args.lr_min_ratio),
        )
    else:
        scheduler = None

    seed_index = int(seed["seed_index"])
    run_index = seed_index * int(args.preferences_per_point) + int(pref_index)
    print("\n" + "=" * 104)
    print(
        f"[Refine] representative={seed_index}, preference={pref_index}, "
        f"topology={seed['seed_topology']}, steps={args.refine_steps}, "
        f"optimize_discrete={args.optimize_discrete}"
    )
    print(
        "权重: "
        + ", ".join(f"{n}={w:.3f}" for n, w in zip(OBJECTIVE_NAMES, weights_np))
    )
    print("Coarse reference: " + ", ".join(f"{name}={seed_ref[name]:.6e}" for name in OBJECTIVE_NAMES))
    print("=" * 104)

    records: List[pd.DataFrame] = []
    best_loss = float("inf")
    best_step = 0
    no_improve_steps = 0
    initial_unit = space.continuous_unit_values().detach().clone()
    last_grad_norm = float("nan")

    for step in range(args.refine_steps + 1):
        if args.reset_torch_seed_each_step:
            current_seed = args.seed + seed_index * 100000 + pref_index * 1000 + step
            torch.manual_seed(current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(current_seed)

        tau = (
            tau_schedule(step, args.refine_steps, args.tau_start, args.tau_end)
            if args.optimize_discrete
            else 1.0
        )
        optimizer.zero_grad(set_to_none=True)

        loss, smooth_score, violation, raw, processed, objectives, soft_state = diff_obj.refinement_loss(
            space,
            weights=weights_t,
            seed_ref=seed_ref,
            tau=tau,
            mu=args.refine_smoothing,
            constraint_penalty=args.constraint_penalty,
            hard_discrete=hard_discrete,
        )
        if not torch.isfinite(loss):
            print(
                f"[Warning] 非有限 loss: seed={seed_index}, pref={pref_index}, step={step}"
            )
            break

        relative = diff_obj.seed_relative_costs(objectives, seed_ref)
        weighted = weights_t * relative
        current_loss = float(loss.detach().cpu().item())
        if current_loss < best_loss - float(args.early_stop_min_delta):
            best_loss = current_loss
            best_step = int(step)
            no_improve_steps = 0
        else:
            no_improve_steps += 1

        # Save every gradient step from the forward pass that was already
            # counted above. With fixed discrete variables (the default),
        # this is a fully valid candidate design and may participate in the
        # final Pareto screening without any additional TabPFN query.
        step_df = pd.DataFrame([space.hard_row()])
        for name, value in raw.items():
            step_df[f"gradient_raw_{name}"] = float(
                value.detach().reshape(-1)[0].cpu().item()
            )
        for name, value in processed.items():
            step_df[name] = float(value.detach().reshape(-1)[0].cpu().item())
        for name, value in objectives.items():
            step_df[name] = float(value.detach().reshape(-1)[0].cpu().item())

        step_values = np.asarray(
            [float(step_df.iloc[0][name]) for name in OBJECTIVE_NAMES]
        )
        reference_values = np.asarray(
            [float(seed_ref[name]) for name in OBJECTIVE_NAMES]
        )
        direction_signs = np.asarray(
            [1.0 if direction == "min" else -1.0 for _, direction in OBJECTIVE_SPECS]
        )
        step_relative = (
            direction_signs
            * (step_values - reference_values)
            / np.maximum(np.abs(reference_values), 1e-30)
        )
        valid_gradient_design = bool(not args.optimize_discrete)
        step_df["stage"] = "refine"
        step_df["source"] = "refine"
        step_df["method"] = "PROPOSED_REFINE"
        step_df["record_kind"] = "gradient_step"
        step_df["metric_source"] = (
            "differentiable_forward_fixed_discrete"
            if valid_gradient_design
            else "soft_discrete_diagnostic"
        )
        step_df["eligible_for_final_pareto"] = valid_gradient_design
        step_df["is_legal_hard_design"] = valid_gradient_design
        step_df["seed_index"] = seed_index
        step_df["seed_label"] = str(seed["seed_label"])
        step_df["seed_topology"] = str(seed["seed_topology"])
        step_df["pref_index"] = int(pref_index)
        step_df["run_index"] = int(run_index)
        step_df["step"] = int(step)
        step_df["evaluations"] = int(step + 1)
        step_df["tau"] = float(tau)
        step_df["loss_total"] = current_loss
        step_df["refine_score"] = float(smooth_score.detach().cpu().item())
        step_df["constraint_violation"] = float(violation.detach().cpu().item())
        step_df["best_loss_so_far"] = best_loss
        step_df["refine_smoothing"] = float(args.refine_smoothing)
        step_df["constraint_penalty"] = float(args.constraint_penalty)
        step_df["fixed_discrete"] = bool(not args.optimize_discrete)
        for i, name in enumerate(OBJECTIVE_NAMES):
            step_df[f"weight_{name}"] = float(weights_np[i])
            step_df[f"seed_ref_{name}"] = float(seed_ref[name])
            step_df[f"relative_cost_{name}"] = float(step_relative[i])
            step_df[f"relative_improve_{name}"] = float(-step_relative[i])
        step_df["hard_score_to_coarse_reference"] = float(
            smooth_score.detach().cpu().item()
        )
        step_df["hard_dominates_selected_reference"] = bool(
            np.all(step_relative <= 0.0) and np.any(step_relative < 0.0)
        )
        step_df["is_feasible"] = (
            check_hard_constraints(
                step_df,
                enabled=diff_obj.constraint_enabled,
                min_snm_limit=diff_obj.min_snm_limit,
                max_delay_limit=diff_obj.max_delay_limit,
                max_power_limit=diff_obj.max_power_limit,
            )
            if valid_gradient_design
            else False
        )
        with torch.no_grad():
            current_unit = space.continuous_unit_values().detach()
            unit_move = torch.abs(current_unit - initial_unit)
            step_df["continuous_unit_move_mean"] = float(
                torch.mean(unit_move).cpu().item()
            )
            step_df["continuous_unit_move_max"] = float(
                torch.max(unit_move).cpu().item()
            )
        step_df["current_continuous_lr"] = float(optimizer.param_groups[0]["lr"])
        step_df["last_gradient_norm"] = float(last_grad_norm)
        step_df["best_soft_loss_step"] = int(best_step)
        step_df["topology_switched_from_seed"] = (
            step_df["topology"].astype(str) != str(seed["seed_topology"])
        )
        records.append(step_df)

        # A hard-design audit is an additional TabPFN query. Printing is
        # deliberately independent, so progress logs never change the budget.
        should_print = (
            step in {0, args.refine_steps}
            or (args.print_every > 0 and step % args.print_every == 0)
        )
        should_hard_audit = (
            step in {0, args.refine_steps}
            or (
                args.hard_audit_every > 0
                and step % args.hard_audit_every == 0
            )
        )
        hard_df: Optional[pd.DataFrame] = None
        if should_hard_audit:
            hard_df = diff_obj.evaluate_hard_design(space)
            hard_df["stage"] = "refine"
            hard_df["source"] = "refine"
            hard_df["method"] = "PROPOSED_REFINE"
            hard_df["record_kind"] = "hard_audit"
            hard_df["eligible_for_final_pareto"] = True
            hard_df["seed_index"] = seed_index
            hard_df["seed_label"] = str(seed["seed_label"])
            hard_df["seed_topology"] = str(seed["seed_topology"])
            hard_df["pref_index"] = int(pref_index)
            hard_df["run_index"] = int(run_index)
            hard_df["step"] = int(step)
            hard_df["evaluations"] = int(step + 1)
            hard_df["tau"] = float(tau)
            hard_df["loss_total_soft"] = current_loss
            hard_df["smooth_score_soft"] = float(smooth_score.detach().cpu().item())
            hard_df["constraint_violation_soft"] = float(violation.detach().cpu().item())
            hard_df["best_loss_so_far"] = best_loss
            hard_df["refine_smoothing"] = float(args.refine_smoothing)
            hard_df["constraint_penalty"] = float(args.constraint_penalty)
            hard_df["fixed_discrete"] = bool(not args.optimize_discrete)

            for i, name in enumerate(OBJECTIVE_NAMES):
                hard_df[f"weight_{name}"] = float(weights_np[i])
                hard_df[f"seed_ref_{name}"] = float(seed_ref[name])
                hard_df[f"soft_obj_{name}"] = float(objectives[name].detach().cpu().item())
                hard_df[f"soft_relative_cost_{name}"] = float(relative[i].detach().cpu().item())
                hard_df[f"soft_relative_improve_{name}"] = float(-relative[i].detach().cpu().item())
                hard_df[f"soft_weighted_cost_{name}"] = float(weighted[i].detach().cpu().item())

            # 硬设计参考点相对代价：用于选择“最佳硬点”，而不是用软目标冒充最终性能。
            hard_values = np.asarray([float(hard_df.iloc[0][name]) for name in OBJECTIVE_NAMES])
            ref_values = np.asarray([float(seed_ref[name]) for name in OBJECTIVE_NAMES])
            signs = np.asarray([1.0 if d == "min" else -1.0 for _, d in OBJECTIVE_SPECS])
            hard_relative = signs * (hard_values - ref_values) / np.maximum(np.abs(ref_values), 1e-30)
            hard_weighted = weights_np * hard_relative
            scaled = hard_weighted / max(float(args.refine_smoothing), 1e-12)
            scaled_max = float(np.max(scaled))
            hard_score = float(args.refine_smoothing) * (
                scaled_max
                + math.log(float(np.exp(scaled - scaled_max).sum()))
                - math.log(float(len(OBJECTIVE_NAMES)))
            )
            for i, name in enumerate(OBJECTIVE_NAMES):
                hard_df[f"hard_relative_cost_{name}"] = hard_relative[i]
                hard_df[f"hard_relative_improve_{name}"] = -hard_relative[i]
            hard_df["hard_score_to_coarse_reference"] = hard_score
            hard_df["hard_dominates_selected_reference"] = bool(
                np.all(hard_relative <= 0.0) and np.any(hard_relative < 0.0)
            )

            bottleneck_idx = int(torch.argmax(weighted.detach()).cpu().item())
            hard_df["soft_bottleneck"] = OBJECTIVE_NAMES[bottleneck_idx]
            with torch.no_grad():
                current_unit = space.continuous_unit_values().detach()
                unit_move = torch.abs(current_unit - initial_unit)
                hard_df["continuous_unit_move_mean"] = float(torch.mean(unit_move).cpu().item())
                hard_df["continuous_unit_move_max"] = float(torch.max(unit_move).cpu().item())
                hard_df["current_continuous_lr"] = float(optimizer.param_groups[0]["lr"])
                hard_df["last_gradient_norm"] = float(last_grad_norm)
                hard_df["best_soft_loss_step"] = int(best_step)
                hard_df["discrete_state_mode"] = (
                    "optimized_soft_then_hardened"
                    if args.optimize_discrete
                    else "strictly_fixed_one_hot"
                )
                hard_df["topology_fixed_indicator_10T"] = float(
                    soft_state["p10"].detach().cpu().item()
                )
                hard_df["soft_rows"] = float(soft_state["rows"].detach().cpu().item())
                hard_df["soft_cols"] = float(soft_state["cols"].detach().cpu().item())
                hard_df["soft_fd_width_candidate"] = float(
                    soft_state["fd_width_candidate"].detach().cpu().item()
                )
                if args.optimize_discrete:
                    topo_probs = space.topology_probs(tau).detach().cpu().numpy()
                    hard_df["soft_topology_prob_6T"] = float(topo_probs[0])
                    hard_df["soft_topology_prob_10T"] = float(topo_probs[1])
                    hard_df["arch_prob_max"] = float(
                        torch.max(space.arch_probs(tau)).detach().cpu().item()
                    )
                    for cat in SHARED_CAT_FEATURES:
                        hard_df[f"{cat}_prob_max"] = float(
                            torch.max(space.shared_cat_probs(cat, tau)).detach().cpu().item()
                        )
                    hard_df["fd_model_prob_max"] = float(
                        torch.max(space.fd_model_probs(tau)).detach().cpu().item()
                    )
            hard_df["topology_switched_from_seed"] = (
                hard_df["topology"].astype(str) != str(seed["seed_topology"])
            )
            records.append(hard_df)

        if should_print:
            display_row = step_df.iloc[0]
            current_unit = space.continuous_unit_values().detach()
            unit_move_max = float(torch.max(torch.abs(current_unit - initial_unit)).cpu().item())
            objective_summary = ", ".join(
                f"{name}={float(display_row[name]):.4e} "
                f"({-float(step_relative[i])*100:+.2f}%)"
                for i, name in enumerate(OBJECTIVE_NAMES)
            )
            topology_text = (
                f"{display_row['topology']} [FIXED]"
                if not args.optimize_discrete
                else f"{display_row['topology']} [SOFT-DIAGNOSTIC]"
            )
            gradient_text = (
                "n/a" if not np.isfinite(last_grad_norm)
                else f"{last_grad_norm:.3e}"
            )
            print(
                f"[seed={seed_index:03d}][pref={pref_index:02d}][step={step:04d}] "
                f"total_loss={current_loss:.5e}, "
                f"refine_score={float(smooth_score.detach().cpu()):+.5e}, "
                f"constraint_violation={float(violation.detach().cpu()):.3e}, "
                f"topology={topology_text}, "
                f"{objective_summary}, "
                f"lr={optimizer.param_groups[0]['lr']:.3e}, "
                f"grad_norm={gradient_text}, move_max={unit_move_max:.3f}, "
                f"feasible={bool(display_row['is_feasible'])}"
            )

        if step == args.refine_steps:
            break

        loss.backward()
        grad_sq = 0.0
        for parameter in space.all_parameters():
            if parameter.grad is not None:
                grad_sq += float(torch.sum(parameter.grad.detach() ** 2).cpu().item())
        last_grad_norm = math.sqrt(max(grad_sq, 0.0))

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                space.all_parameters(), max_norm=float(args.grad_clip_norm)
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if (
            args.early_stop_patience > 0
            and step + 1 >= args.early_stop_min_steps
            and no_improve_steps >= args.early_stop_patience
            and should_hard_audit
        ):
            print(
                f"[EarlyStop] seed={seed_index}, pref={pref_index}, "
                f"step={step + 1}, best_step={best_step}, "
                f"best_soft_loss={best_loss:.6e}"
            )
            break

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


# =============================================================================
# 9. 结果图与统计
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proposed two-stage optimizer: coarse search then gradient refinement."
    )
    parser.add_argument("--data-6t", default=DEFAULT_6T_DATASET)
    parser.add_argument("--data-10t", default=DEFAULT_10T_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEFAULT_DEVICE)
    parser.add_argument(
        "--problem-config",
        default=str(PACKAGE_ROOT / "configs" / "experiment.yaml"),
        help="YAML file containing optimization_problem objectives and constraints.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=1000,
        help=(
            "Exact total TabPFN design-query budget, including coarse search, "
            "differentiable refinement steps and hard-design audits."
        ),
    )
    parser.add_argument(
        "--coarse-evals",
        type=int,
        default=500,
        help="TabPFN query budget used by the coarse stage.",
    )
    parser.add_argument(
        "--coarse-pop-size",
        type=int,
        default=50,
        help="Population size used internally by the coarse stage.",
    )

    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument(
        "--max-train-samples", type=int, default=DEFAULT_MAX_TRAIN_SAMPLES,
        help="6T+10T 合并后的最大训练上下文；0 表示不限制。",
    )
    parser.add_argument(
        "--balance-topologies",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_BALANCE_TOPOLOGIES,
    )
    parser.add_argument(
        "--verbose-library-training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--bounds-lower-q", type=float, default=0.01)
    parser.add_argument("--bounds-upper-q", type=float, default=0.99)

    parser.add_argument("--representative-points", type=int, default=10)
    parser.add_argument("--preferences-per-point", type=int, default=1)
    parser.add_argument("--preference-edge", type=float, default=0.10)
    parser.add_argument(
        "--refine-steps",
        type=int,
        default=47,
        help="Number of optimizer updates in each local gradient trajectory.",
    )
    parser.add_argument(
        "--hard-audit-every",
        "--record-every",
        dest="hard_audit_every",
        type=int,
        default=0,
        help=(
            "Additional hard-design TabPFN audit interval; 0 audits only "
            "start/final. Every gradient step is always saved without an "
            "additional query."
        ),
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print refine progress every N steps; 0 prints only start/final.",
    )

    parser.add_argument("--continuous-lr", type=float, default=10e-2)
    parser.add_argument(
        "--lr-scheduler", choices=["constant", "cosine"], default="cosine",
        help="有限步数下建议 cosine：前期移动快、后期稳定收敛。",
    )
    parser.add_argument(
        "--lr-min-ratio", type=float, default=0.10,
        help="cosine 最终学习率 / 初始学习率。",
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=0,
        help="连续多少步软loss无显著改善后停止；0表示关闭。",
    )
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-6)
    parser.add_argument("--early-stop-min-steps", type=int, default=40)
    parser.add_argument("--topology-lr", type=float, default=8e-2)
    parser.add_argument("--discrete-lr", type=float, default=5e-2)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--refine-smoothing", type=float, default=0.05)
    parser.add_argument("--tau-start", type=float, default=1.2)
    parser.add_argument("--tau-end", type=float, default=0.15)

    # 拓扑偏置必须比旧单拓扑代码小，否则很难发生 6T<->10T 切换。
    parser.add_argument("--topology-init-bias", type=float, default=0.5)
    parser.add_argument("--discrete-init-bias", type=float, default=2.0)
    parser.add_argument(
        "--optimize-discrete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="本版本默认关闭，只优化连续尺寸。未来开启时仍会硬化并重新预测后再保存。",
    )
    parser.add_argument(
        "--use-gumbel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="默认使用确定性 softmax；开启后使用 Gumbel-softmax。",
    )
    parser.add_argument(
        "--reset-torch-seed-each-step",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--seed-ref-source",
        choices=["coarse", "surrogate"],
        default="coarse",
        help="coarse=use coarse-stage values; surrogate=reevaluate representatives.",
    )
    parser.add_argument(
        "--enable-simple-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--constraint-penalty", type=float, default=10.0)
    return parser.parse_args()


def hard_audit_steps(args: argparse.Namespace) -> set[int]:
    """Steps that trigger an additional hardened-design TabPFN prediction."""
    audit_steps = {0, int(args.refine_steps)}
    interval = int(args.hard_audit_every)
    if interval > 0:
        audit_steps.update(range(0, int(args.refine_steps) + 1, interval))
    return audit_steps


def planned_tabpfn_queries(args: argparse.Namespace) -> Dict[str, int]:
    """Return the exact optimizer-only query plan used for fair comparison."""
    gradient_runs = int(args.representative_points) * int(args.preferences_per_point)
    soft_per_run = int(args.refine_steps) + 1
    hard_per_run = len(hard_audit_steps(args))
    seed_reevaluations = (
        int(args.representative_points)
        if args.seed_ref_source == "surrogate"
        else 0
    )
    gradient_queries = gradient_runs * (soft_per_run + hard_per_run)
    total = int(args.coarse_evals) + seed_reevaluations + gradient_queries
    return {
        "coarse": int(args.coarse_evals),
        "seed_reevaluations": seed_reevaluations,
        "gradient_runs": gradient_runs,
        "soft_queries_per_run": soft_per_run,
        "hard_audits_per_run": hard_per_run,
        "gradient_queries": gradient_queries,
        "total": total,
    }


def validate_args(args: argparse.Namespace) -> None:
    for attr in ["data_6t", "data_10t"]:
        path = Path(getattr(args, attr))
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
    if args.max_evals <= 0:
        raise ValueError("max_evals must be positive.")
    if args.coarse_evals <= 0 or args.coarse_pop_size <= 1:
        raise ValueError("Coarse budget must be positive and population size > 1.")
    if args.coarse_evals < args.coarse_pop_size:
        raise ValueError("coarse_evals must be at least coarse_pop_size.")
    if args.coarse_evals % args.coarse_pop_size:
        raise ValueError(
            "coarse_evals must be divisible by coarse_pop_size to avoid a "
            "partially evaluated evolutionary generation."
        )
    if args.train_ratio <= 0 or args.test_ratio <= 0:
        raise ValueError("train_ratio 和 test_ratio 都必须大于 0。")
    if not math.isclose(
        float(args.train_ratio) + float(args.test_ratio), 1.0,
        rel_tol=0.0, abs_tol=1e-8,
    ):
        raise ValueError("train_ratio + test_ratio 必须等于 1。")
    if args.max_train_samples < 0:
        raise ValueError("max_train_samples cannot be negative.")
    if not (0 <= args.bounds_lower_q < args.bounds_upper_q <= 1):
        raise ValueError("边界分位数必须满足 0 <= lower < upper <= 1。")
    if (
        args.representative_points <= 0
        or args.preferences_per_point <= 0
        or args.refine_steps <= 0
    ):
        raise ValueError(
            "representative_points, preferences_per_point and refine_steps must be positive."
        )
    if args.hard_audit_every < 0 or args.print_every < 0:
        raise ValueError("hard_audit_every and print_every cannot be negative.")
    if args.refine_smoothing <= 0:
        raise ValueError("refine_smoothing must be positive.")
    if min(args.continuous_lr, args.topology_lr, args.discrete_lr) <= 0:
        raise ValueError("All learning rates must be positive.")
    if args.grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive.")
    if args.tau_start <= 0 or args.tau_end <= 0:
        raise ValueError("tau_start and tau_end must be positive.")
    if not (0.0 <= args.preference_edge < 1.0):
        raise ValueError("preference_edge must be in [0, 1).")
    if args.constraint_penalty < 0:
        raise ValueError("constraint_penalty cannot be negative.")
    if not (0.0 <= args.lr_min_ratio <= 1.0):
        raise ValueError("lr_min_ratio 必须在 [0, 1] 内。")
    if args.early_stop_patience < 0 or args.early_stop_min_steps < 0:
        raise ValueError("early-stop 参数不能为负。")
    if args.early_stop_patience != 0:
        raise ValueError(
            "Exact budget comparison requires --early-stop-patience 0; "
            "early stopping would make TabPFN calls data-dependent."
        )
    if len(OBJECTIVE_NAMES) < 2 or len(set(OBJECTIVE_NAMES)) != len(OBJECTIVE_NAMES):
        raise ValueError("OBJECTIVE_SPECS 必须包含至少两个、且名称不重复的目标。")
    if any(direction not in {"min", "max"} for direction in OBJECTIVE_DIRECTIONS.values()):
        raise ValueError("OBJECTIVE_SPECS 的方向只能为 min 或 max。")
    plan = planned_tabpfn_queries(args)
    if plan["total"] != int(args.max_evals):
        raise ValueError(
            "Proposed-method TabPFN budget mismatch: "
            f"planned={plan['total']}, --max-evals={args.max_evals}. "
            "Formula: coarse_evals + representative_reevaluations + "
            "representative_points*preferences_per_point*"
            "((refine_steps+1)+hard_audits_per_run). "
            f"Resolved plan={plan}"
        )


# =============================================================================
# 11. Main
# =============================================================================

def main() -> None:
    """Run the reproducible coarse -> refine -> final Proposed workflow."""

    args = parse_args()
    configure_problem(args.problem_config)
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    output_dir, evaluations_dir, fronts_dir = create_run_directory(args.output_dir)

    print("=" * 96)
    print("Proposed optimizer: coarse search -> representative points -> refinement")
    print("=" * 96)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(
        "Objectives: "
        + ", ".join(f"{name} ({direction})" for name, direction in OBJECTIVE_SPECS)
    )
    print(f"Configured constraints: {PROBLEM_CONSTRAINTS}")
    print(
        f"Budget: total={args.max_evals}, coarse={args.coarse_evals}, "
        f"representatives={args.representative_points}, "
        f"preferences/representative={args.preferences_per_point}, "
        f"refine_steps={args.refine_steps}"
    )

    raw_6t = pd.read_csv(args.data_6t)
    raw_10t = pd.read_csv(args.data_10t)
    df_6t = harmonize_topology_dataframe(raw_6t, "6T")
    df_10t = harmonize_topology_dataframe(raw_10t, "10T")
    train_6t, _ = split_one_topology(
        df_6t,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    train_10t, _ = split_one_topology(
        df_10t,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed + 1,
    )
    pooled_train = pd.concat([train_6t, train_10t], ignore_index=True)
    if args.balance_topologies:
        pooled_train = balance_pooled_training_data(pooled_train, args.seed)
    pooled_train = limit_rows_stratified(
        pooled_train, args.max_train_samples, args.seed
    )
    pooled_train = pooled_train.sample(
        frac=1.0, random_state=args.seed
    ).reset_index(drop=True)

    dataloader, manager, _ = train_pooled_union_model(
        pooled_train,
        output_dir=output_dir,
        device=device,
        verbose_library_training=args.verbose_library_training,
    )
    manager.reset_query_count()

    bounds = build_continuous_bounds(
        pooled_train,
        lower_q=args.bounds_lower_q,
        upper_q=args.bounds_upper_q,
    )
    encoder_categories = build_encoder_categories(dataloader)
    shared_choices = build_shared_cat_choices(encoder_categories)
    fd_choices = build_fd_model_choices_10t(encoder_categories)
    architectures = build_architecture_templates(pooled_train)
    print_design_space(bounds, shared_choices, fd_choices, architectures)

    coarse_front, coarse_history, coarse_trace = run_coarse_search(
        manager=manager,
        dataloader=dataloader,
        bounds=bounds,
        shared_choices=shared_choices,
        fd_choices=fd_choices,
        architectures=architectures,
        max_evals=args.coarse_evals,
        pop_size=args.coarse_pop_size,
        seed=args.seed,
        constraints_enabled=args.enable_simple_constraints,
    )

    representatives = select_representative_seeds(
        coarse_front, args.representative_points, args.seed
    )
    if len(representatives) != args.representative_points:
        raise RuntimeError(
            f"The coarse Pareto front contains only {len(representatives)} usable "
            f"points, but {args.representative_points} were requested. Increase "
            "--coarse-evals or reduce --representative-points."
        )

    defaults_shared = default_shared_cat_indices(shared_choices, pooled_train)
    default_fd_idx = default_fd_model_index(fd_choices, pooled_train)
    default_fd_width = default_fd_width_value(bounds, pooled_train)
    seed_points = build_seed_points(
        representatives,
        bounds=bounds,
        shared_choices=shared_choices,
        fd_choices=fd_choices,
        architectures=architectures,
        default_shared_indices=defaults_shared,
        default_fd_index=default_fd_idx,
        default_fd_width=default_fd_width,
    )
    if len(seed_points) != args.representative_points:
        raise RuntimeError(
            "Some representative points could not be mapped back into the "
            "shared design space."
        )

    diff_obj = DifferentiableJointObjective(
        manager=manager,
        dataloader=dataloader,
        architectures=architectures,
        encoder_categories=encoder_categories,
        shared_cat_choices=shared_choices,
        fd_model_choices=fd_choices,
        bounds=bounds,
        device=device,
        constraint_enabled=args.enable_simple_constraints,
        min_snm_limit=0.0,
        max_delay_limit=0.0,
        max_power_limit=0.0,
    )

    reevaluated_reference: Dict[int, Dict[str, float]] = {}
    if args.seed_ref_source == "surrogate":
        reevaluated_rows = [
            evaluate_seed_surrogate(
                seed,
                diff_obj=diff_obj,
                bounds=bounds,
                architectures=architectures,
                shared_choices=shared_choices,
                fd_choices=fd_choices,
                args=args,
                device=device,
            )
            for seed in seed_points
        ]
        reevaluated = pd.concat(reevaluated_rows, ignore_index=True)
        reevaluated_reference = {
            int(row["seed_index"]): {
                name: float(row[name]) for name in OBJECTIVE_NAMES
            }
            for _, row in reevaluated.iterrows()
        }

    weights = generate_preference_weights(
        args.preferences_per_point,
        edge=args.preference_edge,
        seed=args.seed,
    )
    refine_records: List[pd.DataFrame] = []
    started_at = time.time()
    total_runs = len(seed_points) * args.preferences_per_point
    completed = 0
    for seed in seed_points:
        representative_index = int(seed["seed_index"])
        reference = (
            reevaluated_reference[representative_index]
            if args.seed_ref_source == "surrogate"
            else seed["csv_seed_ref"]
        )
        for preference_index, weight in enumerate(weights):
            history = optimize_one_preference(
                seed=seed,
                pref_index=preference_index,
                weights_np=weight,
                seed_ref=reference,
                diff_obj=diff_obj,
                bounds=bounds,
                architectures=architectures,
                shared_choices=shared_choices,
                fd_choices=fd_choices,
                args=args,
                device=device,
            )
            if not history.empty:
                refine_records.append(history)
            completed += 1
            print(f"[Refine] completed {completed}/{total_runs} runs")
    if not refine_records:
        raise RuntimeError("The refinement stage produced no hard-design history.")

    refine_history = add_optimization_metric_columns(
        pd.concat(refine_records, ignore_index=True)
    )
    eligible_refine = refine_history[
        _as_bool_series(refine_history["eligible_for_final_pareto"])
    ].copy()
    feasible_refine = eligible_refine[
        _as_bool_series(eligible_refine["is_feasible"])
    ].copy()
    if feasible_refine.empty:
        raise RuntimeError(
            "No feasible hard design was recorded during refinement."
        )
    refine_front = deduplicate_designs(
        pareto_front(feasible_refine, feasible_only=False)
    )
    refine_front["stage"] = "refine"
    refine_front["source"] = "refine"

    # The final front is selected from every feasible hard point ever queried
    # by either stage, not merely from each stage's final population/front.
    complete_history = pd.concat(
        [coarse_history, refine_history], ignore_index=True, sort=False
    )
    complete_history = add_optimization_metric_columns(complete_history)
    final_candidate_history = pd.concat(
        [coarse_history, eligible_refine], ignore_index=True, sort=False
    )
    final_candidate_history = add_optimization_metric_columns(
        deduplicate_designs(final_candidate_history)
    )
    feasible_complete = final_candidate_history[
        _as_bool_series(final_candidate_history["is_feasible"])
    ].copy()
    final_front = deduplicate_designs(
        pareto_front(feasible_complete, feasible_only=False)
    )
    evaluations = public_evaluations(
        complete_history, "PROPOSED", expected_rows=args.max_evals
    )
    evaluations.to_csv(evaluations_dir / "PROPOSED.csv", index=False)
    final_front = public_pareto_front(
        final_front,
        "PROPOSED",
        objective_columns=OBJECTIVE_NAMES,
    )
    final_front.to_csv(fronts_dir / "PROPOSED.csv", index=False)

    query_plan = planned_tabpfn_queries(args)
    actual_queries = manager.get_query_count()
    if actual_queries != args.max_evals:
        raise RuntimeError(
            f"Proposed used {actual_queries} TabPFN design queries; expected "
            f"{args.max_evals}. Planned breakdown={query_plan}"
        )
    elapsed = time.time() - started_at
    summary: Dict[str, Any] = {
        "algorithm": "PROPOSED",
        "evaluations": int(len(evaluations)),
        "feasible_evaluations": int(evaluations["constraint_valid"].sum()),
        "pareto_front_size": int(len(final_front)),
        "tabpfn_design_queries": int(actual_queries),
        "elapsed_seconds": float(elapsed),
    }
    pd.DataFrame([summary]).to_csv(
        output_dir / "algorithm_summary.csv", index=False
    )

    resolved = vars(args).copy()
    resolved.update(
        {
            "resolved_device": device,
            "objectives": [
                {"source": name, "direction": direction}
                for name, direction in OBJECTIVE_SPECS
            ],
            "constraints": PROBLEM_CONSTRAINTS,
            "continuous_bounds": {
                name: [float(pair[0]), float(pair[1])]
                for name, pair in bounds.items()
            },
            "architectures": architectures,
            "shared_category_choices": shared_choices,
            "fd_model_choices": fd_choices,
            "query_plan": query_plan,
        }
    )
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(resolved, handle, ensure_ascii=False, indent=2)

    print("=" * 96)
    print(f"Evaluations: {evaluations_dir / 'PROPOSED.csv'}")
    print(f"Pareto front: {fronts_dir / 'PROPOSED.csv'}")


if __name__ == "__main__":
    main()
