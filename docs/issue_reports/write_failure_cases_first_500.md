# OpenYield 第一次 500 条仿真中的 write 失败案例

本表保留首次 500 条运行的原始失败记录，不表示 V2.1.1 修复后的验证结果。
V2.1.1 的发布验证与适用范围见[分布式 RC 发布报告](../design/DISTRIBUTED_ONLY_V2_1_1.md)。

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
