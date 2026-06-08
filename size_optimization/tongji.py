"""
SRAM 单阵列多模式统计脚本（32KB 总容量）

功能：
- 对若干 (rows, cols) 组合，遍历 real_cell_mode 0~4 五种电路模式：
    0 = 全真实（所有 cell，参考基准）
    1 = 等效十字（目标行或目标列真实，其余等效）
    2 = 仅目标行真实
    3 = 仅目标列真实
    4 = 仅目标 cell 真实
- 对每个 (rows, cols, mode) 调用 evaluate_sram_with_config 计算 SNM（hold/read/write 及最小值）、
  读写延迟、读写功耗、面积。
- 面积定义为单元面积 × 阵列比特数（rows × cols）。
- 阵列个数 num_arrays = 262144 / (rows × cols)。
- 结果保存为 CSV，并额外输出各模式相对 mode 0 的相对误差 CSV。

依赖：
- 复用 experiment.evaluate_sram_with_config（已支持 real_cell_mode 显式入参）
- 使用 exp_utils 的参数加载与随机种子工具

使用：
- 在该目录下运行：python tongji.py
"""

import os
import sys
import csv
import time
import resource
from pathlib import Path
from datetime import datetime

from experiment import evaluate_sram_with_config
from exp_utils import get_default_initial_params, get_params_from_yaml, seed_set


TOTAL_BITS = 262144  # 32KB = 262144 bit
# 待遍历的 (rows, cols) 组合（cols 与 ROW/COLUMN_CHOICES 一致，最小 16）。
ROW_COL_PAIRS = [
    (32, 16),
    (64, 32),
    (128, 64),
]


def is_valid_combo(rows: int, cols: int) -> bool:
    """判断 (rows, cols) 是否能整除 32KB 总容量。"""
    array_capacity = rows * cols
    if array_capacity <= 0:
        return False
    return (TOTAL_BITS % array_capacity) == 0


def compute_num_arrays(rows: int, cols: int) -> int:
    """计算阵列个数：32KB 除以单阵列容量。"""
    return TOTAL_BITS // (rows * cols)


def main():
    # 固定随机种子，确保仿真过程可复现（尽量）
    seed_set(42)

    wall_start = time.time()
    cpu_start = time.process_time()

    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    # 将工作目录切换到仓库根，避免相对路径问题（模型与 YAML 加载）
    try:
        repo_root = base_dir.parent
        os.chdir(repo_root)
        print(f"[INFO] 当前工作目录：{repo_root}")
    except Exception as e:
        print(f"[WARN] 设置工作目录失败：{e}，继续使用默认目录。")

    mode_values = list(range(5))  # 0,1,2,3,4
    output_csv = base_dir / "tongji_results_32kB_modes.csv"
    diff_csv = base_dir / "tongji_results_32kB_modes_diff.csv"

    # 优先使用 YAML 中的初始参数
    try:
        params = get_params_from_yaml()
        print(f"[INFO] 已从 YAML 读取初始参数: pu={params['pu_width']}, "
              f"pd={params['pd_width']}, pg={params['pg_width']}, L={params['length']}")
    except Exception as e:
        print(f"[WARN] 从 YAML 读取参数失败，使用默认参数: {e}")
        params = get_default_initial_params()

    headers = [
        "timestamp", "rows", "cols", "num_arrays", "array_capacity", "real_cell_mode",
        "hold_snm", "read_snm", "write_snm", "min_snm",
        "read_delay", "write_delay", "max_delay",
        "read_power", "write_power", "max_power", "area",
        "eval_wall_s", "eval_cpu_s", "valid", "error",
    ]
    diff_columns = [
        "hold_snm", "read_snm", "write_snm", "min_snm",
        "read_delay", "write_delay", "max_delay",
        "read_power", "write_power", "max_power", "area",
    ]
    diff_headers = headers + [f"{col}_diff" for col in diff_columns]

    def to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def relative_diff(value, baseline):
        if value is None or baseline is None:
            return ""
        try:
            if baseline == 0:
                return 0.0 if value == 0 else ""
            return round((value - baseline) / baseline, 6)
        except Exception:
            return ""

    main_loop(output_csv, diff_csv, headers, diff_headers, diff_columns,
              mode_values, params, to_float, relative_diff,
              wall_start, cpu_start)


def main_loop(output_csv, diff_csv, headers, diff_headers, diff_columns,
              mode_values, params, to_float, relative_diff,
              wall_start, cpu_start):
    total_evals = 0
    success_evals = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f, \
         open(diff_csv, "w", newline="", encoding="utf-8") as df:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        diff_writer = csv.DictWriter(df, fieldnames=diff_headers)
        diff_writer.writeheader()

        for rows, cols in ROW_COL_PAIRS:
            if not is_valid_combo(rows, cols):
                print(f"[WARN] 非法组合 rows={rows}, cols={cols}，跳过")
                continue

            combo_results = []
            baseline_row = None
            for real_cell_mode in mode_values:
                total_evals += 1
                ts = datetime.now().isoformat(timespec="seconds")
                print(f"[INFO] 评估组合 rows={rows}, cols={cols}, mode={real_cell_mode}, "
                      f"num_arrays={compute_num_arrays(rows, cols)}")

                try:
                    _rusage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
                    _wall_before = time.time()

                    objectives, constraints, result, success = evaluate_sram_with_config(
                        params.copy(),
                        rows,
                        cols,
                        TOTAL_BITS // (rows * cols),
                        timeout=None,
                        stage_label="stage2",
                        iteration_index=f"{rows}x{cols}_{real_cell_mode}",
                        real_cell_mode=real_cell_mode,
                    )

                    eval_wall = time.time() - _wall_before
                    _rusage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
                    eval_cpu = ((_rusage_after.ru_utime + _rusage_after.ru_stime) -
                                (_rusage_before.ru_utime + _rusage_before.ru_stime))

                    print(f"[INFO] evaluate 返回：success={success}, 有结果={result is not None}, "
                          f"wall={eval_wall:.1f}s, cpu={eval_cpu:.1f}s")

                    row = {
                        "timestamp": ts, "rows": rows, "cols": cols,
                        "num_arrays": compute_num_arrays(rows, cols),
                        "array_capacity": rows * cols, "real_cell_mode": real_cell_mode,
                        "hold_snm": "", "read_snm": "", "write_snm": "", "min_snm": "",
                        "read_delay": "", "write_delay": "", "max_delay": "",
                        "read_power": "", "write_power": "", "max_power": "", "area": "",
                        "eval_wall_s": round(eval_wall, 3), "eval_cpu_s": round(eval_cpu, 3),
                        "valid": False, "error": "",
                    }

                    if success and result is not None:
                        success_evals += 1
                        row.update({
                            "hold_snm": result.get("hold_snm"),
                            "read_snm": result.get("read_snm"),
                            "write_snm": result.get("write_snm"),
                            "min_snm": result.get("min_snm"),
                            "read_delay": result.get("read_delay"),
                            "write_delay": result.get("write_delay"),
                            "max_delay": result.get("max_delay"),
                            "read_power": result.get("read_power"),
                            "write_power": result.get("write_power"),
                            "max_power": result.get("max_power"),
                            "area": result.get("area"),
                            "valid": True,
                        })
                        print(f"[INFO] 已写入成功结果：{rows}x{cols}, mode={real_cell_mode}")
                    else:
                        row["error"] = "evaluation_failed"
                        print(f"[INFO] 写入失败占位：{rows}x{cols}, mode={real_cell_mode}")

                    writer.writerow(row)
                    f.flush()
                    combo_results.append(row)
                    if real_cell_mode == 0:
                        baseline_row = row
                except Exception as e:
                    row = {
                        "timestamp": ts, "rows": rows, "cols": cols,
                        "num_arrays": compute_num_arrays(rows, cols),
                        "array_capacity": rows * cols, "real_cell_mode": real_cell_mode,
                        "hold_snm": "", "read_snm": "", "write_snm": "", "min_snm": "",
                        "read_delay": "", "write_delay": "", "max_delay": "",
                        "read_power": "", "write_power": "", "max_power": "", "area": "",
                        "eval_wall_s": "", "eval_cpu_s": "", "valid": False, "error": str(e),
                    }
                    writer.writerow(row)
                    f.flush()
                    combo_results.append(row)
                    print(f"[WARN] 评估异常：{rows}x{cols}, mode={real_cell_mode}, 错误：{e}")

            # 计算每种模式相对 baseline (mode 0) 的 diff
            for row in combo_results:
                diff_row = row.copy()
                for col in diff_columns:
                    diff_row[f"{col}_diff"] = ""
                if baseline_row is not None and baseline_row.get("valid"):
                    baseline_values = {col: to_float(baseline_row.get(col)) for col in diff_columns}
                    for col in diff_columns:
                        diff_row[f"{col}_diff"] = relative_diff(to_float(row.get(col)), baseline_values[col])
                diff_writer.writerow(diff_row)
                df.flush()

    wall_elapsed = time.time() - wall_start
    cpu_self = time.process_time() - cpu_start
    rusage_ch = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_children = rusage_ch.ru_utime + rusage_ch.ru_stime

    print(f"统计完成：共评估 {total_evals} 个组合，成功 {success_evals} 个。")
    print(f"结果已保存到：{output_csv}")
    print(f"[TIME] 挂钟时间(Wall):   {wall_elapsed:.2f} s")
    print(f"[TIME] 主进程CPU时间:     {cpu_self:.2f} s")
    print(f"[TIME] 子进程CPU时间合计: {cpu_children:.2f} s")


if __name__ == "__main__":
    main()

