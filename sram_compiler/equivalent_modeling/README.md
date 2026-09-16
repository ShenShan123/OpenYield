# SRAM 等效阵列模型使用说明 — V2.1.5

等效电路（Equivalent Circuit）现在是编译器的一个仿真输入项，不再是仓库顶层的独立实验目录。
V2.1.5 把原 `equivalent_modeling/` 合并进 `sram_compiler/equivalent_modeling/`：

| 位置 | 作用 |
|------|------|
| `sram_compiler/equivalent_modeling/__init__.py` | `EquivalentConfig` / `resolve_equivalent`，校验并规范化模式 0–4 |
| `sram_compiler/equivalent_modeling/compare.py` | 等效模型与全真实阵列的精度、运行时间对比入口（替代原 `equivalent_modeling/main_sram.py` 与 `run.sh`） |
| `sram_compiler/subcircuits/sram_cell_add_equivalent.py` | 等效电路的全部实现（寄生提取、RC 建模、静态功耗） |
| `sram_compiler/config_yaml/global.yaml` 的 `equivalent:` 块 | YAML 默认模式 |

V2.1.1 起仅支持并默认使用分布式信号连线；显式 star 配置会报错。等效模式保留全部物理线段，
并在被省略单元的本地抽头连接等效负载。`w_rc` 只控制局部存储/外围串联 RC，与物理连线独立。
参见 [V2.1.1 实现与验证](../../docs/design/DISTRIBUTED_ONLY_V2_1_1.md)。

请从仓库根目录运行文中的命令；代码路径也相对于仓库根目录。
参见 [SRAM 编译器指南](../README.md) 和
[默认逐器件局部失配流程](../per_device_mc/README.md)。

---

## 1. 功能说明

读操作观察目标输出；当前写操作驱动所选行的全部列，没有独立写掩码。全真实模式中整个阵列的所有晶体管都参与 SPICE 求解，大规模 Monte Carlo 优化会重复这一开销。

等效电路用本地等效负载替代非激活单元（WL 低、access 晶体管关断的所有单元），仅保留目标行和目标列的真实晶体管，其余替换为等效模型，形成“十字形”真实区域：

```
         目标列
         ↓
  行 0:  [等效] [等效] [实际] [等效]
  行 1:  [等效] [等效] [实际] [等效]
  目标行: [实际] [实际] [实际] [实际]
  行 3:  [等效] [等效] [实际] [等效]
```

等效模型覆盖以下五类寄生效应：

- **BL/BLB 端**：非目标行的 access 晶体管关断，外部电流完全正比于 dV/dt，建模为等效电容。
- **WL 端**：字线翻转以电容分量为主，采用简化单电容模型。
- **静态漏电**：用等效电阻 `R_static = VDD / I_static` 建模，`I_static` 通过单元级直流仿真自动提取；阵列级近似精度需与全真实模型比较。
- **PI-RC 物理连线**：几何参数定义的 WL/BL/BLB 阶梯线始终保留，等效负载逐单元接到本地抽头；不按省略单元数聚合或缩放线阻/线电容。`w_rc` 独立控制局部串联 RC。
- **WL-BL 交叉耦合电容**：对每一对等效 (row, col) 添加 WL 到 BL/BLB 的耦合电容，由 `SRAMCellParasiticTester` 自动提取，无需手动标定。

等效模式减少真实晶体管数量，但物理线段和本地负载仍随阵列规模增长。此前文档的数十倍加速、0.3% 延迟误差和 2% 功耗误差属于旧拓扑下的说明，不是当前精度保证；当前配置的实测结果见第 2.3 节。

**适用场景**：大阵列单次仿真、优化算法批量评估、Monte Carlo 仿真加速。
注意寄生提取本身要跑 Xyce，其开销与阵列规模基本无关：V2.1.5 实测 8x8 反而变慢
（+12 %~+31 %），16x16 省 16 %~23 %，32x32 省 37 %~51 %（见 2.3 节）。
用之前先在自己的配置上测一次。

**不适用**：等效模式 1–4 是近似，不能作为合格性（qualification）证据；时序类别、驱动尺寸和波形判据的评估必须使用 `mode 0`。逐器件局部失配在等效模式下只作用于保留的真实晶体管。

---

## 2. 作为输入项使用

### 2.1 三个入口，同一个默认值

优先级：显式参数 > `global.yaml` 的 `equivalent:` 块。

```yaml
# sram_compiler/config_yaml/global.yaml
equivalent:
  mode: 0        # 0=全真实（基准）, 1=等效十字, 2=仅目标行, 3=仅目标列, 4=仅目标 cell
```

```bash
# 命令行：不给 --real-cell-mode 时沿用 global.yaml
python3 -m sram_compiler.per_device_mc.run --rows 16 --cols 16 --real-cell-mode 1
```

```python
# Python API：real_cell_mode=None（默认）沿用 global.yaml，显式整数覆盖它
mc_testbench = Sram6TCoreMcTestbench(
    sram_config,
    w_rc=True,              # 开启局部串联 RC；物理分布式连线始终存在
    pi_res=100 @ u_Ohm,     # 局部串联 RC 的示例电阻，非提取金属参数
    pi_cap=0.001 @ u_pF,    # 局部串联 RC 的示例电容，非提取金属参数
    real_cell_mode=1,       # 1=等效十字；0=全真实（参考基准）；None=用 YAML 默认
    ...
)
print(mc_testbench.equivalent.describe())   # "mode 1: cross: the target row and ..."
```

`main_sram.py` 的 `REAL_CELL_MODE` 默认是 `None`，同样沿用 YAML。
运行记录（`*.variation.json`、`summary.json`）新增 `equivalent` 字段，
`full_device_coverage` 仍要求 `mode 0` 且逐器件失配。

等效模式支持 `w_rc=True` 或 `False`，两者都保留物理分布式连线；比较全真实与等效模型时必须使用相同设置。

> 等效模式在**生成网表时**就会调用 Xyce 做单元级寄生提取和静态电流提取，
> 因此 `mode != 0` 需要 Xyce 在 PATH 上，`mode 0` 不需要。

### 2.2 精度与运行时间对比实验

```bash
cd /path/to/OpenYield
python3 -m sram_compiler.equivalent_modeling.compare --sizes 16x16 --modes 0,1,2,3,4 --plot
python3 -m sram_compiler.equivalent_modeling.compare \
    --sizes 16x16,32x32 --modes 0,1 --operations read,write --corner SS --vdd 0.9 --temperature 125
```

结果保存在 `outputs/equivalent_modeling/<时间戳>/`（`outputs/` 已被 git 忽略）：

- `result.csv`：每个 (规模, 模式) 的完整指标、时钟、连线配置与运行时间
- `result_diff.csv`：各模式（1~4）相对 mode 0（全真实）的相对误差
- `result_diff.png`：`--plot` 时的误差柱状图
- `settings.json`：本次运行的全部输入

CSV 主要列含义：

| 列名 | 说明 |
|------|------|
| `real_cell_mode` | 电路模式：0=全真实(基准), 1=等效十字, 2=仅目标行, 3=仅目标列, 4=仅目标cell |
| `r_delay` / `w_delay` | 读 / 写延迟（s） |
| `r_pavg / r_pstc / r_pdyn` | 读平均/静态/动态功耗（W） |
| `w_pavg / w_pstc / w_pdyn` | 写平均/静态/动态功耗（W；模式 3/4 含等效近似） |
| `time_usage_s` | 该 (规模, 模式) 点的构建 + 仿真墙钟时间（s） |

对比实验默认 `--variation nominal`，让各模式共用同一确定性模型，保证误差只来自等效近似；
其余输入（连线、局部 RC、驱动类别、时钟、PVT）由同一份配置固定。

> 另有 `size_optimization/tongji.py`：对若干 (rows, cols) 组合同样遍历 5 种模式，输出单阵列 SNM / 延迟 / 功耗 / 面积统计及相对 mode 0 的误差。

### 2.3 V2.1.5 实测

TT 1.0 V / 25 °C，分布式连线 + 局部 RC，目标单元取末行末列，`nominal`：

| 阵列 | 模式 | 读延迟误差 | 写延迟误差 | 读平均功耗误差 | 静态功耗误差 | 运行时间变化 |
|---|---|---:|---:|---:|---:|---:|
| 16x16 | 1 | −0.09 % | −0.28 % | −0.12 % | −26.9 % | −16 % |
| 16x16 | 2 | −0.11 % | +0.20 % | −0.18 % | −107 %（变号） | −23 % |
| 16x16 | 3 | −0.20 % | −0.48 % | −7.08 % | +3.4 % | −21 % |
| 16x16 | 4 | −0.20 % | +0.02 % | −7.13 % | −12.6 % | −19 % |
| 8x8 | 1 | +0.01 % | +0.13 % | −0.23 % | −91.6 % | **+31 %** |
| 8x8 | 4 | −0.02 % | +0.06 % | −3.02 % | +25.2 % | **+12 %** |
| 32x32 | 1 | −0.14 % | −0.51 % | −0.31 % | +12.8 % | **−37 %** |
| 32x32 | 4 | −0.14 % | +0.37 % | −12.71 % | +37.7 % | **−51 %** |

结论：

1. **延迟**：所有模式、所有规模下误差都 < 0.55 %，这是等效模型真正可用的指标。
2. **平均功耗**：保留整个目标行的模式 1/2 误差 < 0.6 %；只保留目标列/单元的模式 3/4
   偏低，且随被替换的目标行比例增大（8x8 −3.0 %，16x16 −7.1 %，32x32 −12.7 %）。
3. **静态功耗在任何模式下都不可用**（−92 % ~ +38 %，16x16 模式 2 甚至变号）。
4. **加速随阵列规模增长**：模式 1 在 8x8 反而慢 31 %，16x16 省 16 %，32x32 省 37 %；
   模式 4 在 32x32 省 51 %。寄生提取开销基本与规模无关，所以小阵列不要开。

完整记录见 [等效模型精度记录](../../docs/design/EQUIVALENT_MODEL_V2_1_5.md)。

---

## 3. 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `real_cell_mode` | `None`（取 `global.yaml` 的 `equivalent.mode`，即 `0`） | `0` = 全真实晶体管（参考基准）；`1` = 等效十字；`2` = 仅目标行真实；`3` = 仅目标列真实；`4` = 仅目标 cell 真实 |
| `write_power_model` | `False` | `True` = 用 WL 受控行为电流源（按 WL 电压分段拟合静态电流）；`False` = 直流静态电阻。经 testbench 时由 `operation=="write"` 自动置位，无需手动设置 |
| `w_rc` | `False` | `True` = 开启局部存储/外围串联 RC；物理连线不受此开关影响 |
| `pi_res` | 100 Ω | 局部串联 RC 电阻；示例默认值，非提取金属参数 |
| `pi_cap` | 1 fF | 局部串联 RC 电容；示例默认值，非提取金属参数 |
| `target_row` | `0` | 目标单元行（`main_sram.py` / CLI 默认取末行末列） |
| `target_col` | `0` | 目标单元列 |

注意事项：

- 等效模式 1–4 始终保留全部物理连线；局部 RC 和 `cell_pin_rc` 必须与比较的全真实模型一致。
- `target_row` / `target_col` 指定测试激活的目标单元，等效模式下只有目标行/列实例化真实晶体管。如需测试其他位置（如末行末列的最差负载情况），修改这两个参数即可。
- 模式 3/4 支持写仿真，目标 cell 的 Q/QB 翻转仍为晶体管级；被替换 cell 只有 RC 与 WL 受控静态功耗模型，因此其内部写状态和整行动态写功耗应按近似结果使用。
- `pi_res` / `pi_cap` 是局部 RC 参数，不可替代 `interconnect` 中的物理线几何和提取参数。
- 比较精度时，保持 PVT、物理连线、`w_rc`、`pi_res`、`pi_cap`、驱动尺寸和时钟一致；不要通过改变某一侧的物理参数来掩盖误差。

---

## 4. 10T 单元支持

10T SRAM 完全复用相同的等效电路逻辑和 RC 模型，差异仅在寄生参数提取阶段（`cell_type="10T"` 时处理额外的 `fd_nmos_model` / `fd_width` 参数）：

```python
mc_testbench = Sram6TCoreMcTestbench(
    sram_config,
    sram_cell_type="SRAM_10T_CELL",
    real_cell_mode=1,
    w_rc=True,
    ...
)
```

其余参数和用法与 6T 完全相同。`compare.py` 通过 `global.yaml` 的 `sram_cell_type` 选择单元类型。

---

## 5. 实现细节（开发者参考）

### 调用链

```
Sram6TCoreMcTestbench(real_cell_mode=1)   # None → global.yaml 的 equivalent.mode
  └─ resolve_equivalent()  →  self.equivalent (EquivalentConfig)
  └─ create_testbench(operation)
       │   write_power_model = (operation == "write")
       └─ Sram6TCoreFactory(real_cell_mode, write_power_model, ...).create()
            └─ Sram6TCore.build_array()
                 ├─ 实例化目标行/列的真实 Sram6TCell
                 └─ add_equivalent_circuit()  ← 由 sram_cell_add_equivalent.py 注入
                      ├─ SRAMCellParasiticTester.extract_parasitic_caps()
                      ├─ 静态功耗：write_power_model=False → get_static_power_r() 直流静态电阻
                      │           write_power_model=True  → 每行 WL 受控行为电流源 BIWL_POWER_{row}
                      ├─ 保留完整 WL/BL/BLB 物理阶梯线
                      ├─ 在每个省略单元的本地抽头添加等效负载
                      └─ 添加每对等效 (row,col) 的 WL-BL/WLB 交叉耦合电容
```

> `write_power_model` 由 testbench 在写操作（`operation=="write"`）时自动置 `True`，经 `Sram6TCoreFactory` / `Sram10TCoreFactory` 透传到 core，从而在写仿真中启用 WL 受控的动态静态功耗建模；读/SNM 等其他操作下为 `False`，使用直流静态电阻。

### 5-Cap 寄生参数（SRAMCellParasiticTester 自动提取）

| 参数 | 含义 |
|------|------|
| `c_bl` | BL 扩散电容 |
| `c_blb` | BLB 扩散电容 |
| `c_wl` | WL 门电容 |
| `c_wl_bl` | WL-BL 交叉耦合电容 |
| `c_wl_blb` | WL-BLB 交叉耦合电容 |

参数随晶体管尺寸自动重新计算，无需手动输入。
