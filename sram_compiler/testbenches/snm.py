import math
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


def split_xyce_prn_runs(prn_path: str) -> List[pd.DataFrame]:
    """
    读取 Xyce .prn 文件，并按 Monte Carlo / Parameter Sweep 的每组 run 拆分。
    规则：当 Index 重新回到 0 且当前缓存非空时，认为新 run 开始。
    """
    runs = []
    current_rows = []

    with open(prn_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("Index") or s.startswith("End of Xyce"):
                continue

            parts = s.split()
            if len(parts) != 4:
                continue

            try:
                idx = int(parts[0])
                u = float(parts[1])
                v1 = float(parts[2])
                v2 = float(parts[3])
            except ValueError:
                continue

            if idx == 0 and current_rows:
                df = pd.DataFrame(current_rows, columns=["Index", "U", "V1", "V2"])
                runs.append(df)
                current_rows = []

            current_rows.append((idx, u, v1, v2))

    if current_rows:
        df = pd.DataFrame(current_rows, columns=["Index", "U", "V1", "V2"])
        runs.append(df)

    return runs


def find_crossings(u: np.ndarray, d: np.ndarray, merge_tol: float = 1e-4) -> np.ndarray:
    """
    找 d(U)=0 的所有交点，使用线性插值。
    merge_tol: 合并过近交点，避免数值噪声导致重复零点。
    """
    xs = []

    for i in range(len(u) - 1):
        u0, u1 = u[i], u[i + 1]
        d0, d1 = d[i], d[i + 1]

        if d0 == 0.0:
            xs.append(u0)
        elif d0 * d1 < 0.0:
            # 线性插值求零点
            xc = u0 - d0 * (u1 - u0) / (d1 - d0)
            xs.append(xc)

    if d[-1] == 0.0:
        xs.append(u[-1])

    if not xs:
        return np.array([], dtype=float)

    xs = np.array(sorted(xs), dtype=float)

    merged = [xs[0]]
    for x in xs[1:]:
        if abs(x - merged[-1]) <= merge_tol:
            merged[-1] = 0.5 * (merged[-1] + x)
        else:
            merged.append(x)

    return np.array(merged, dtype=float)


def extract_classical_snm_from_run(
    df_run: pd.DataFrame,
    merge_tol: float = 1e-4,
) -> Dict:
    """
    基于定义：
        EV1 = U + sqrt(2)*V(QBD)
        EV2 = -U + sqrt(2)*V(QD)
    则：
        d(U) = V1 - V2
        MAXVD = max_{bounded interval} |d(U)|
        SNM   = MAXVD / sqrt(2)
    关键点：
    1. 先找所有交点 d(U)=0
    2. 相邻交点之间形成“有界区间”
    3. 每个有界区间内取 max |d|
    4. 总 SNM 取所有候选值的最小值

    若交点少于 2 个，则经典 butterfly SNM 不成立，返回 NaN。
    """
    u = df_run["U"].to_numpy(dtype=float)
    v1 = df_run["V1"].to_numpy(dtype=float)
    v2 = df_run["V2"].to_numpy(dtype=float)

    d = v1 - v2
    crossings = find_crossings(u, d, merge_tol=merge_tol)

    if len(crossings) < 2:
        return {
            "snm": float("nan"),
            "maxvd": float("nan"),
            "crossings": crossings.tolist(),
            "status": "undefined_for_classical_butterfly",
            "reason": "Fewer than 2 crossings; no bounded interval.",
            "candidates": [],
        }

    candidates = []
    for left, right in zip(crossings[:-1], crossings[1:]):
        mask = (u >= left) & (u <= right)
        uu = u[mask]
        dd = d[mask]

        if len(uu) == 0:
            continue

        idx = np.argmax(np.abs(dd))
        max_abs_d = float(abs(dd[idx]))
        snm_candidate = max_abs_d / math.sqrt(2.0)

        candidates.append({
            "left": float(left),
            "right": float(right),
            "u_at_max": float(uu[idx]),
            "max_abs_d": max_abs_d,
            "snm_candidate": snm_candidate,
        })

    if not candidates:
        return {
            "snm": float("nan"),
            "maxvd": float("nan"),
            "crossings": crossings.tolist(),
            "status": "undefined_for_classical_butterfly",
            "reason": "No valid bounded interval samples found.",
            "candidates": [],
        }

    # 经典 SNM 取所有内部有界区间候选值中的最小者
    best = min(candidates, key=lambda x: x["snm_candidate"])

    return {
        "snm": float(best["snm_candidate"]),
        "maxvd": float(best["max_abs_d"]),
        "crossings": crossings.tolist(),
        "status": "ok",
        "reason": "",
        "candidates": candidates,
    }


def build_stats_table(df_data: pd.DataFrame, value_columns: List[str]) -> pd.DataFrame:
    """
    生成与你附件风格一致的 stats.csv
    列：
    count, mean, std, min, 1%, 5%, 25%, 50%, 75%, 95%, 99%, max, cv, range, skew, kurtosis
    """
    stats_rows = []

    for col in value_columns:
        s = pd.to_numeric(df_data[col], errors="coerce").dropna()

        if len(s) == 0:
            stats_rows.append({
                "metric": col,
                "count": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "1%": np.nan,
                "5%": np.nan,
                "25%": np.nan,
                "50%": np.nan,
                "75%": np.nan,
                "95%": np.nan,
                "99%": np.nan,
                "max": np.nan,
                "cv": np.nan,
                "range": np.nan,
                "skew": np.nan,
                "kurtosis": np.nan,
            })
            continue

        mean = s.mean()
        std = s.std(ddof=1) if len(s) > 1 else np.nan
        min_v = s.min()
        max_v = s.max()

        stats_rows.append({
            "metric": col,
            "count": float(len(s)),
            "mean": float(mean),
            "std": float(std) if pd.notna(std) else np.nan,
            "min": float(min_v),
            "1%": float(s.quantile(0.01)),
            "5%": float(s.quantile(0.05)),
            "25%": float(s.quantile(0.25)),
            "50%": float(s.quantile(0.50)),
            "75%": float(s.quantile(0.75)),
            "95%": float(s.quantile(0.95)),
            "99%": float(s.quantile(0.99)),
            "max": float(max_v),
            "cv": float(std / mean) if (pd.notna(std) and mean != 0) else np.nan,
            "range": float(max_v - min_v),
            "skew": float(s.skew()) if len(s) >= 3 else np.nan,
            "kurtosis": float(s.kurt()) if len(s) >= 4 else np.nan,
        })

    df_stats = pd.DataFrame(stats_rows).set_index("metric")
    return df_stats

def extract_write_snm_from_run(df_run: pd.DataFrame) -> Dict:
    """

    这和 hold/read 的 classical butterfly bounded-interval SNM 不同。
    write 曲线常常只有 1 个交点，因此这里按网表定义直接取全局最大 gap。
    """
    u = df_run["U"].to_numpy(dtype=float)
    v1 = df_run["V1"].to_numpy(dtype=float)
    v2 = df_run["V2"].to_numpy(dtype=float)

    d_abs = np.abs(v1 - v2)
    idx = int(np.argmax(d_abs))

    maxvd = float(d_abs[idx])
    write_snm = float(maxvd / math.sqrt(2.0))

    return {
        "snm": write_snm,
        "maxvd": maxvd,
        "crossings": [],
        "status": "ok",
        "reason": "write_snm uses global max |V1-V2| over the full sweep.",
        "candidates": [{
            "left": float(u[0]),
            "right": float(u[-1]),
            "u_at_max": float(u[idx]),
            "max_abs_d": maxvd,
            "snm_candidate": write_snm,
        }],
    }

def process_xyce_montecarlo_prn(
    prn_path: str,
    metric_name: str = "SNM",
    operation: str = "hold_snm",
    merge_tol: float = 1e-4,
    out_data_csv: str = None,
    out_stats_csv: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    主入口：
    - 读取 .prn
    - 按 run 拆分
    - 每组计算 SNM / MAXVD
    - 输出 .data.csv / .stats.csv
    """
    prn_path = str(prn_path)
    base = os.path.splitext(prn_path)[0]

    if out_data_csv is None:
        out_data_csv = base + ".data.csv"
    if out_stats_csv is None:
        out_stats_csv = base + ".stats.csv"

    runs = split_xyce_prn_runs(prn_path)

    data_rows = []
    debug_rows = []

    for run_id, df_run in enumerate(runs):
        if operation == "hold_snm" or operation == "read_snm":
            result = extract_classical_snm_from_run(df_run, merge_tol=merge_tol)
        elif operation == "write_snm":
            result = extract_write_snm_from_run(df_run)

        data_rows.append({
            "Run": run_id,
            metric_name: result["snm"],
            "MAXVD": result["maxvd"],
        })

        debug_rows.append({
            "Run": run_id,
            "status": result["status"],
            "reason": result["reason"],
            "crossings": result["crossings"],
            "num_crossings": len(result["crossings"]),
        })

    df_data = pd.DataFrame(data_rows)
    df_stats = build_stats_table(df_data, [metric_name, "MAXVD"])

    df_data.to_csv(out_data_csv, index=False)
    df_stats.to_csv(out_stats_csv)

    # 这个 debug 表可选，用于排查多交点/单交点问题
    debug_csv = base + ".debug.csv"
    pd.DataFrame(debug_rows).to_csv(debug_csv, index=False)

    print(f"Processed runs: {len(runs)}")
    print(f"Saved data : {out_data_csv}")
    print(f"Saved stats: {out_stats_csv}")
    print(f"Saved debug: {debug_csv}")

    return df_data, df_stats


if __name__ == "__main__":
    # 例子1：hold
    process_xyce_montecarlo_prn(
        prn_path="/home/majh/OpenYield/sim/20260318_191048_mc_6t/mc_hold_snm_16x8_rc0_tb.sp.prn",
        metric_name="HOLD_SNM",
    )

    # 例子2：read
    # process_xyce_montecarlo_prn(
    #     prn_path="mc_read_snm_16x8_rc0_tb.sp.prn",
    #     metric_name="READ_SNM",
    # )
