# OpenYield 第一次 500 条仿真中的 write 失败案例

本表保留首次 500 条运行的原始失败记录，不表示 V2.1.1 修复后的验证结果。
V2.1.1 的发布验证与适用范围见[分布式 RC 发布报告](../design/DISTRIBUTED_ONLY_V2_1_1.md)。
V2.1.2 的审计结论附在文末（[V2.1.2 审计](#v212-审计)），原始表格与统计保持不变。

- 生成时间：2026-09-12 23:33:02
- 数据来源：`outputs/pvt_metrics_all.csv`
- 范围：第一次 500 条仿真中的 `write` 操作
- 结果：`write` 共 100 条，成功 35 条，失败 65 条

## 共同配置

所有失败案例都通过 `load_config(rows, cols, corner)` 加载同一组基础 YAML：

- `sram_compiler/config_yaml/global.yaml`
- `sram_compiler/config_yaml/sram_6t_cell.yaml`
- `sram_compiler/config_yaml/wordline_driver.yaml`
- `sram_compiler/config_yaml/precharge.yaml`
- `sram_compiler/config_yaml/mux.yaml`
- `sram_compiler/config_yaml/sa.yaml`
- `sram_compiler/config_yaml/write_driver.yaml`
- `sram_compiler/config_yaml/decoder.yaml`

其中数组尺寸来自 `global.yaml` 的 `num_rows/num_cols`，工艺角对应的 PDK 文件来自 `global.yaml` 中的 `pdk_path_<corner>`。VDD 和温度由运行脚本 `collect_pvt_metrics.py` 在运行时设置，不是单独 YAML 文件。

## 失败案例明细

| # | corner | VDD(V) | Temp(C) | Array | PDK model | Log | 错误摘要 |
|---:|---|---:|---:|---|---|---|---|
| 1 | TT | 0.8 | -40 | 16x16 | `tran_models/models_TT.spice` | `outputs/pvt_TT_v0.8_t-40/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 2 | TT | 0.8 | 0 | 16x16 | `tran_models/models_TT.spice` | `outputs/pvt_TT_v0.8_t0/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 3 | TT | 0.8 | 25 | 16x16 | `tran_models/models_TT.spice` | `outputs/pvt_TT_v0.8_t25/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 4 | TT | 0.8 | 85 | 16x16 | `tran_models/models_TT.spice` | `outputs/pvt_TT_v0.8_t85/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 5 | TT | 0.8 | 125 | 16x16 | `tran_models/models_TT.spice` | `outputs/pvt_TT_v0.8_t125/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 6 | TT | 0.9 | -40 | 16x16 | `tran_models/models_TT.spice` | `outputs/pvt_TT_v0.9_t-40/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 7 | TT | 0.9 | 0 | 16x16 | `tran_models/models_TT.spice` | `outputs/pvt_TT_v0.9_t0/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 8 | TT | 0.9 | 125 | 16x16 | `tran_models/models_TT.spice` | `outputs/pvt_TT_v0.9_t125/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 9 | TT | 1.0 | -40 | 16x16 | `tran_models/models_TT.spice` | `outputs/pvt_TT_v1.0_t-40/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 10 | TT | 1.0 | 0 | 16x16 | `tran_models/models_TT.spice` | `outputs/pvt_TT_v1.0_t0/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 11 | TT | 1.0 | 25 | 16x16 | `tran_models/models_TT.spice` | `outputs/pvt_TT_v1.0_t25/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 12 | TT | 1.1 | -40 | 16x16 | `tran_models/models_TT.spice` | `outputs/pvt_TT_v1.1_t-40/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 13 | TT | 1.1 | 0 | 16x16 | `tran_models/models_TT.spice` | `outputs/pvt_TT_v1.1_t0/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 14 | TT | 1.1 | 25 | 16x16 | `tran_models/models_TT.spice` | `outputs/pvt_TT_v1.1_t25/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 15 | TT | 1.1 | 85 | 16x16 | `tran_models/models_TT.spice` | `outputs/pvt_TT_v1.1_t85/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 16 | FF | 0.8 | -40 | 16x16 | `tran_models/models_FF.spice` | `outputs/pvt_FF_v0.8_t-40/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 17 | FF | 0.8 | 0 | 16x16 | `tran_models/models_FF.spice` | `outputs/pvt_FF_v0.8_t0/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 18 | FF | 0.8 | 25 | 16x16 | `tran_models/models_FF.spice` | `outputs/pvt_FF_v0.8_t25/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 19 | FF | 0.9 | -40 | 16x16 | `tran_models/models_FF.spice` | `outputs/pvt_FF_v0.9_t-40/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 20 | FF | 1.0 | -40 | 16x16 | `tran_models/models_FF.spice` | `outputs/pvt_FF_v1.0_t-40/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 21 | FF | 1.0 | 0 | 16x16 | `tran_models/models_FF.spice` | `outputs/pvt_FF_v1.0_t0/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 22 | FF | 1.0 | 85 | 16x16 | `tran_models/models_FF.spice` | `outputs/pvt_FF_v1.0_t85/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 23 | FF | 1.1 | -40 | 16x16 | `tran_models/models_FF.spice` | `outputs/pvt_FF_v1.1_t-40/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 24 | FF | 1.1 | 0 | 16x16 | `tran_models/models_FF.spice` | `outputs/pvt_FF_v1.1_t0/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 25 | FF | 1.1 | 25 | 16x16 | `tran_models/models_FF.spice` | `outputs/pvt_FF_v1.1_t25/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 26 | FF | 1.1 | 85 | 16x16 | `tran_models/models_FF.spice` | `outputs/pvt_FF_v1.1_t85/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 27 | SS | 0.8 | -40 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v0.8_t-40/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 28 | SS | 0.8 | 0 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v0.8_t0/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 29 | SS | 0.8 | 25 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v0.8_t25/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 30 | SS | 0.8 | 85 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v0.8_t85/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 31 | SS | 0.8 | 125 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v0.8_t125/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 32 | SS | 0.9 | -40 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v0.9_t-40/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 33 | SS | 0.9 | 0 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v0.9_t0/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 34 | SS | 0.9 | 25 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v0.9_t25/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 35 | SS | 0.9 | 85 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v0.9_t85/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 36 | SS | 0.9 | 125 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v0.9_t125/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 37 | SS | 1.0 | -40 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v1.0_t-40/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 38 | SS | 1.0 | 85 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v1.0_t85/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 39 | SS | 1.0 | 125 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v1.0_t125/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 40 | SS | 1.1 | -40 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v1.1_t-40/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 41 | SS | 1.1 | 0 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v1.1_t0/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 42 | SS | 1.1 | 25 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v1.1_t25/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 43 | SS | 1.1 | 85 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v1.1_t85/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 44 | SS | 1.1 | 125 | 16x16 | `tran_models/models_SS.spice` | `outputs/pvt_SS_v1.1_t125/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 45 | FS | 0.8 | -40 | 16x16 | `tran_models/models_FS.spice` | `outputs/pvt_FS_v0.8_t-40/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 46 | FS | 0.8 | 25 | 16x16 | `tran_models/models_FS.spice` | `outputs/pvt_FS_v0.8_t25/write/mc_write_16x16_rc1_tb.log` | OpenMPI 资源不足：ORTE Out of resource |
| 47 | FS | 0.8 | 85 | 16x16 | `tran_models/models_FS.spice` | `outputs/pvt_FS_v0.8_t85/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 48 | FS | 0.8 | 125 | 16x16 | `tran_models/models_FS.spice` | `outputs/pvt_FS_v0.8_t125/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 49 | FS | 0.9 | 0 | 16x16 | `tran_models/models_FS.spice` | `outputs/pvt_FS_v0.9_t0/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 50 | FS | 0.9 | 125 | 16x16 | `tran_models/models_FS.spice` | `outputs/pvt_FS_v0.9_t125/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 51 | FS | 1.0 | 0 | 16x16 | `tran_models/models_FS.spice` | `outputs/pvt_FS_v1.0_t0/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 52 | FS | 1.0 | 85 | 16x16 | `tran_models/models_FS.spice` | `outputs/pvt_FS_v1.0_t85/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 53 | FS | 1.1 | -40 | 16x16 | `tran_models/models_FS.spice` | `outputs/pvt_FS_v1.1_t-40/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 54 | FS | 1.1 | 85 | 16x16 | `tran_models/models_FS.spice` | `outputs/pvt_FS_v1.1_t85/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 55 | FS | 1.1 | 125 | 16x16 | `tran_models/models_FS.spice` | `outputs/pvt_FS_v1.1_t125/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 56 | SF | 0.8 | -40 | 16x16 | `tran_models/models_SF.spice` | `outputs/pvt_SF_v0.8_t-40/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 57 | SF | 0.8 | 25 | 16x16 | `tran_models/models_SF.spice` | `outputs/pvt_SF_v0.8_t25/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 58 | SF | 0.8 | 125 | 16x16 | `tran_models/models_SF.spice` | `outputs/pvt_SF_v0.8_t125/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 59 | SF | 0.9 | -40 | 16x16 | `tran_models/models_SF.spice` | `outputs/pvt_SF_v0.9_t-40/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 60 | SF | 0.9 | 0 | 16x16 | `tran_models/models_SF.spice` | `outputs/pvt_SF_v0.9_t0/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 61 | SF | 0.9 | 85 | 16x16 | `tran_models/models_SF.spice` | `outputs/pvt_SF_v0.9_t85/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 62 | SF | 1.0 | -40 | 16x16 | `tran_models/models_SF.spice` | `outputs/pvt_SF_v1.0_t-40/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 63 | SF | 1.0 | 0 | 16x16 | `tran_models/models_SF.spice` | `outputs/pvt_SF_v1.0_t0/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 64 | SF | 1.0 | 85 | 16x16 | `tran_models/models_SF.spice` | `outputs/pvt_SF_v1.0_t85/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |
| 65 | SF | 1.1 | 0 | 16x16 | `tran_models/models_SF.spice` | `outputs/pvt_SF_v1.1_t0/write/mc_write_16x16_rc1_tb.log` | Xyce 收敛失败：Step size reached minimum step size bound |

## 统计

- Corner：FF=11, FS=11, SF=10, SS=18, TT=15
- VDD：0.8=20, 0.9=14, 1.0=14, 1.1=17
- Temperature：-40=17, 0=15, 125=10, 25=10, 85=13

## V2.1.2 审计

审计日期 2026-09-13。上面的表格和统计是提供的原始记录，未作修改。

1. **来源无法追溯。** 本工作区不存在 `outputs/pvt_metrics_all.csv`、65 个日志和 `collect_pvt_metrics.py`。生成时间早于 V2.1.1 提交，源代码版本、变化模式和种子均未知。
2. **错误摘要不是终止原因。** `Step size reached minimum step size bound` 是 Xyce 非线性求解器的警告：本地 49 个包含该行的日志全部正常完成。瞬态终止信息是 `Time step too small`，工作点失败是 `DC Operating Point Failed`。第 1–45 和 47–65 行只能记为“未完成，原因未核实”。
3. **第 46 行来自其他 MPI 环境。** `ORTE` 属于 Open MPI 运行时，而 `openyield` 环境中的 Xyce 链接 MPICH 4.2.3。这是求解器环境或资源问题，不是电路失败。
4. **没有记录到电学写失败。** 表中没有 Q/QB 错误或检查失败，“成功 35 条”的判定口径也未知。失败在 PVT 上没有规律（例如 25 °C 下 TT 0.9 V 通过，而 0.8/1.0/1.1 V 失败），更符合数值或基础设施原因。
5. 温度统计按字符串排序（-40, 0, 125, 25, 85），仅为格式问题。

用 V2.1.1 代码复跑其中 7 个配置：16x16 write、局部 RC、nominal、种子 20260913、4 ns 查表时钟、`openyield` 环境的 Xyce。

| 配置 | 原表 | V2.1.1 复跑 | TWRITE_TOTAL (ps) | VPRE_ACCESS_ERROR (V) |
|---|---|---|---:|---:|
| TT 1.0 V 25 °C | 失败 | 通过 | 155.9 | 0.0051 |
| SS 0.8 V −40 °C | 失败 | 通过 | 147.2 | 0.0032 |
| SS 0.8 V 125 °C | 失败 | 通过 | 442.3 | 0.0042 |
| FF 1.1 V 85 °C | 失败 | 通过 | 187.3 | 0.0064 |
| FS 0.8 V 25 °C | 失败（ORTE） | 通过 | 196.8 | 0.0045 |
| SF 1.1 V 0 °C | 失败 | 通过 | 128.2 | 0.0050 |
| TT 0.9 V 25 °C | 成功（对照） | 通过 | 176.8 | 0.0046 |

所有复跑都没有 Xyce 警告或重试，字线释放、访问、保持、恢复和有限指标检查全部通过，两个 SS 点的波形已目视检查。范围限制：只覆盖 65 条失败中的 6 条（另加 1 个对照点），使用 nominal 而不是原始（未知）的变化模式，也没有运行 `dev/` 中的独立波形评分器。本地产物位于被 git 忽略的 `outputs/reviews/V2.1.1-write-failure-audit/`。

V2.1.2 为后续 PVT 清单提供 CLI `--vdd`/`--temperature`，并在运行元数据中记录 `xyce` 安装路径。
