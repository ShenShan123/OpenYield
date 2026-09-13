# SRAM 等效电路使用说明 — V2.1.2

本文档介绍等效电路（Equivalent Circuit）功能的参数配置和使用方法，代码实现在 `sram_compiler/subcircuits/sram_cell_add_equivalent.py`。

V2.1.1 仅支持并默认使用分布式信号连线；显式 star 配置会报错。等效模式保留全部物理线段，并在被省略单元的本地抽头连接等效负载。`w_rc` 只控制局部存储/外围串联 RC，与物理连线独立。参见 [V2.1.1 实现与验证](../docs/design/DISTRIBUTED_ONLY_V2_1_1.md)。

V2.0.6 将原根目录的等效电路说明迁移到本目录的 `README.md`，历史测量结果保持原版本标记。

请从仓库根目录运行文中的命令；代码路径也相对于仓库根目录。
参见 [SRAM 编译器指南](../sram_compiler/README.md) 和
[默认逐器件局部失配流程](../sram_compiler/per_device_mc/README.md)。

---

## 1. 功能说明

读操作观察目标输出；当前写操作驱动所选行的全部列，没有独立写掩码。全真实模式中整个阵列的所有晶体管都参与 SPICE 求解，大规模 Monte Carlo 优化会重复这一开销。

等效电路用本地等效负载替代非激活单元（WL 低、access 晶体管关断的所有单元），仅保留目标行和目标列的真实晶体管，其余替换为等效模型，形成"十字形"真实区域：

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

等效模式减少真实晶体管数量，但物理线段和本地负载仍随阵列规模增长。此前文档的数十倍加速、0.3% 延迟误差和 2% 功耗误差属于旧拓扑下的说明，不是 V2.1.1 精度保证。当前配置需要与相同 PVT、连线、局部 RC 和固定时钟的全真实阵列比较。

**适用场景**：大阵列（≥ 64×64）单次仿真、优化算法批量评估、Monte Carlo 仿真加速。小阵列（≤ 16×16）仿真时间本身很短，是否开启影响不大。

---

## 2. 快速上手

### 2.1 开启等效电路

`main_sram.py` 默认 `REAL_CELL_MODE = 0`（全真实晶体管，逐器件失配覆盖全部器件）。
要启用等效电路，在 `main_sram.py` 顶部的设置区改为 `REAL_CELL_MODE = 1`；直接构造测试平台时设置：

```python
mc_testbench = Sram6TCoreMcTestbench(
    sram_config,
    w_rc=True,              # 开启局部串联 RC；物理分布式连线始终存在
    pi_res=100 @ u_Ohm,     # 局部串联 RC 的示例电阻，非提取金属参数
    pi_cap=0.001 @ u_pF,    # 局部串联 RC 的示例电容，非提取金属参数
    real_cell_mode=1,       # 1=等效十字模型；0=全真实晶体管（参考基准）
    ...
)
```

等效模式支持 `w_rc=True` 或 `False`，两者都保留物理分布式连线；比较全真实与等效模型时必须使用相同设置。
等效模式下，默认的逐器件局部失配只作用于保留的真实晶体管；被替换的单元是近似模型。

V2.1.1 的 `interconnect.mode: distributed` 是唯一支持的拓扑（见
[分布式 RC 模型指南](../docs/design/DISTRIBUTED_RC_MODEL.md)）。默认几何使用每节距 1 Ω / 0.1 fF 的示例值，不是工艺提取结果。等效模式把每个被省略单元的五电容网络挂在其本地抽头上；旧的 `pi_res/N`、`pi_cap*N` 聚合支路已移除。

### 2.2 精度对比实验

`equivalent_modeling/main_sram.py` 遍历 `real_cell_mode` 0~4 五种模式，对比各等效模式相对全真实电路（mode 0）的精度差异：

```bash
cd /path/to/OpenYield
python equivalent_modeling/main_sram.py
```

结果保存在 `equivalent_modeling/results/<时间戳>/`：

- `result.csv`：每个 (规模, 模式) 的完整指标
- `result_diff.csv`：各模式（1~4）相对 mode 0（全真实）的相对误差

CSV 主要列含义：

| 列名 | 说明 |
|------|------|
| `real_cell_mode` | 电路模式：0=全真实(基准), 1=等效十字, 2=仅目标行, 3=仅目标列, 4=仅目标cell |
| `r_delay` | 读延迟（s） |
| `r_pavg / r_pstc / r_pdyn` | 读平均/静态/动态功耗（W） |
| `w_delay` | 目标 cell 写延迟（s，五种模式均执行写仿真） |
| `w_pavg / w_pstc / w_pdyn` | 写平均/静态/动态功耗（W；模式 3/4 含等效近似） |

> 另有 `size_optimization/tongji.py`：对若干 (rows, cols) 组合同样遍历 5 种模式，输出单阵列 SNM / 延迟 / 功耗 / 面积统计及相对 mode 0 的误差（`tongji_results_32kB_modes.csv` 与 `..._diff.csv`）。

---

## 3. 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `real_cell_mode` | `0` | `0` = 全真实晶体管（参考基准）；`1` = 等效十字；`2` = 仅目标行真实；`3` = 仅目标列真实；`4` = 仅目标 cell 真实 |
| `write_power_model` | `False` | `True` = 用 WL 受控行为电流源（按 WL 电压分段拟合静态电流）；`False` = 直流静态电阻。经 testbench 时由 `operation=="write"` 自动置位，无需手动设置 |
| `w_rc` | `False` | `True` = 开启局部存储/外围串联 RC；物理连线不受此开关影响 |
| `pi_res` | 100 Ω | 局部串联 RC 电阻；示例默认值，非提取金属参数 |
| `pi_cap` | 1 fF | 局部串联 RC 电容；示例默认值，非提取金属参数 |
| `target_row` | `0` | 目标单元行（默认首行） |
| `target_col` | `0` | 目标单元列（默认首列） |

注意事项：

- 等效模式 1–4 始终保留全部物理连线；局部 RC 和 `cell_pin_rc` 必须与比较的全真实模型一致。
- `target_row` / `target_col` 默认取首行首列（`0, 0`）。它们指定测试激活的目标单元，等效模式下只有目标行/列实例化真实晶体管。如需测试其他位置（如末行末列的最差负载情况），修改这两个参数即可。
- 模式 3/4 支持写仿真，目标 cell 的 Q/QB 翻转仍为晶体管级；被替换 cell 只有 RC 与 WL 受控静态功耗模型，因此其内部写状态和整行动态写功耗应按近似结果使用。
- `pi_res` / `pi_cap` 是局部 RC 参数，不可替代 `interconnect` 中的物理线几何和提取参数。

### 在相同物理参数下比较精度

对同一设计点分别跑完整电路和等效电路，保持 PVT、物理连线、`w_rc`、`pi_res`、`pi_cap`、驱动尺寸和时钟一致。比较写入波形、保持、读出与功耗，记录近似误差；不要通过改变某一侧的物理参数来掩盖误差：

```python
# 完整电路（全真实，参考基准）
mc_testbench = Sram6TCoreMcTestbench(..., real_cell_mode=0, w_rc=True, pi_res=100@u_Ohm, pi_cap=0.001@u_pF)

# 等效电路（保持相同物理参数，测量近似误差）
mc_testbench = Sram6TCoreMcTestbench(..., real_cell_mode=1, w_rc=True, pi_res=100@u_Ohm, pi_cap=0.001@u_pF)
```

也可以直接运行 `equivalent_modeling/main_sram.py` 批量对比，结果写入 `result_diff.csv` 方便分析。

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

其余参数和用法与 6T 完全相同。

---

## 5. 实现细节（开发者参考）

### 调用链

```
Sram6TCoreMcTestbench(real_cell_mode=1)   # 1=等效十字
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

### 关键文件

| 文件 | 作用 |
|------|------|
| `sram_compiler/subcircuits/sram_cell_add_equivalent.py` | 等效电路全部实现（寄生提取、RC 建模、静态功耗） |
| `sram_compiler/subcircuits/sram_6t_core.py` | `build_array()` 控制等效/真实 cell 实例化，`real_cell_mode` 参数入口 |
| `sram_compiler/subcircuits/sram_10t_core.py` | 10T 版本，等效逻辑与 6T 共用 |
| `sram_compiler/testbenches/parameter_factor.py` | `Sram6TCoreFactory`，将 `real_cell_mode` 从 testbench 传到 core |
| `equivalent_modeling/main_sram.py` | 等效电路与真实电路准确度对比实验入口 |
