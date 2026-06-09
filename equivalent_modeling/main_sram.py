import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
from sram_compiler.testbenches.sram_6t_core_testbench import Sram6TCoreTestbench  # type: ignore
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench  # type: ignore
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
import numpy as np
from utils import estimate_bitcell_area  # type: ignore
from config import SRAM_CONFIG
from datetime import datetime
import pandas as pd
import time


def time_str_to_float(time_str):
    """处理HH:MM:SS或HH:MM:SS.sss格式"""
    parts = time_str.split(":")

    if len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds


def string_to_float_list(s):
    return np.array([float(item) for item in s.strip("[]").split(",")])


if __name__ == "__main__":
    # equivalent_modeling/results
    # 创建结果文件夹，名字为时间
    output_dir = "equivalent_modeling/results/" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir)
    columns = ["time_usage", "use_mc", "custom_mc", "num_rows", "num_cols", "real_cell_mode"]
    relative_diff = ["r_delay", "r_pavg", "r_pstc", "r_pdyn", "w_delay", "w_pavg", "w_pstc", "w_pdyn"]
    columns.extend(relative_diff)
    result = pd.DataFrame(columns=columns)
    result_diff = pd.DataFrame(columns=columns)

    # real_cell_mode: 0=全真实(基准), 1=等效十字, 2=仅目标行, 3=仅目标列, 4=仅目标cell
    mode_values = list(range(5))

    print("===== 6T SRAM Array Monte Carlo Simulation Debug Session =====")
    for use_mc in [False]:
        for custom_mc in [False]:
            # for sram_6t_cell_yaml in ["sram_compiler/config_yaml/sram_6t_cell_nochange.yaml", "sram_compiler/config_yaml/sram_6t_cell.yaml"]:
            for num_rows, num_cols in [(16, 16), (16, 32), (32, 32)]:
                baseline_index = None
                for real_cell_mode in mode_values:

                    start_time = time.time()

                    # ================== 1. 加载所有配置 ==================
                    sram_config = SRAM_CONFIG()
                    sram_config.load_all_configs(
                        global_file="sram_compiler/config_yaml/global.yaml",
                        circuit_configs={
                            "SRAM_6T_CELL": "sram_compiler/config_yaml/sram_6t_cell.yaml",
                            "WORDLINEDRIVER": "sram_compiler/config_yaml/wordline_driver.yaml",
                            "PRECHARGE": "sram_compiler/config_yaml/precharge.yaml",
                            "COLUMNMUX": "sram_compiler/config_yaml/mux.yaml",
                            "SENSEAMP": "sram_compiler/config_yaml/sa.yaml",
                            "WRITEDRIVER": "sram_compiler/config_yaml/write_driver.yaml",
                            "DECODER": "sram_compiler/config_yaml/decoder.yaml",
                        },
                    )

                    # 2. 生成时间戳子目录
                    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    sim_path = os.path.join(output_dir, f"{time_str}_mc_6t")  # 例如 sim/20250928_153045_mc_6t
                    os.makedirs(sim_path, exist_ok=True)

                    # FreePDK45 default transistor sizes
                    # area = estimate_bitcell_area(w_access=sram_config.sram_6t_cell.nmos_width.value[1], w_pd=sram_config.sram_6t_cell.nmos_width.value[0], w_pu=sram_config.sram_6t_cell.pmos_width.value, l_transistor=sram_config.sram_6t_cell.length.value)  # pg
                    # print(f"Estimated 6T SRAM Cell Area: {area*1e12:.2f} µm²")

                    temperature = sram_config.global_config.temperature
                    # num_rows = sram_config.global_config.num_rows
                    # num_cols = sram_config.global_config.num_cols
                    num_mc = sram_config.global_config.monte_carlo_runs

                    sram_config.global_config.num_rows = num_rows
                    sram_config.global_config.num_cols = num_cols

                    # 默认参数 pd=0.205e-6 、 pg=0.135e-6 、 pu=0.09e-6 、 length=50e-9这组可以，
                    # 类型是VTG

                    mc_testbench = Sram6TCoreMcTestbench(
                        sram_config,
                        w_rc=True,  # Whether add RC to nets
                        pi_res=100 @ u_Ohm,
                        pi_cap=0.001 @ u_pF,
                        vth_std=0.05,  # Process parameter variation is a percentage of its value in model lib
                        enable_mc=use_mc,  # use_mc=False → 关闭 AGAUSS 随机，各模式用同一确定性模型，保证误差只来自等效
                        custom_mc=custom_mc,  # Use your own process params?
                        sweep_cell=False,
                        sweep_precharge=False,
                        sweep_senseamp=False,
                        sweep_wordlinedriver=False,
                        sweep_columnmux=False,
                        sweep_writedriver=False,
                        sweep_decoder=False,
                        corner="TT",  # or FF or SS or FS or SF
                        choose_columnmux=False,
                        q_init_val=0,
                        sim_path=sim_path,
                        # real_cell_mode: 0=全真实(基准), 1=等效十字,
                        #                 2=仅目标行, 3=仅目标列, 4=仅目标cell
                        real_cell_mode=real_cell_mode,
                    )
                    # vars = np.random.rand(num_mc,num_rows*num_cols*18)

                    # For using DC analysis, operation can be 'write_snm' 'hold_snm' 'read_snm'
                    # read_snm = mc_testbench.run_mc_simulation(
                    #     operation='hold_snm', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc,
                    #     vars=None, # Input your data table
                    # )

                    # For using TRAN analysis, operation can be 'write' or 'read'
                    r_delay, r_pavg, r_pstc, r_pdyn = mc_testbench.run_mc_simulation(
                        operation="read",
                        target_row=num_rows - 1,
                        target_col=num_cols - 1,
                        mc_runs=num_mc if use_mc else 1,
                        temperature=temperature,
                        vars=None,  # Input your data table
                    )

                    print(r_delay, r_pavg, r_pstc, r_pdyn)

                    # 模式 3/4（仅目标列 / 仅目标 cell）目标行未完整实例化，跳过写仿真
                    if real_cell_mode in (3, 4):
                        print("[INFO] real_cell_mode 3/4：跳过 write 操作仿真")
                        w_delay, w_pavg, w_pstc, w_pdyn = [0.0], [0.0], [0.0], [0.0]
                    else:
                        w_delay, w_pavg, w_pstc, w_pdyn = mc_testbench.run_mc_simulation(
                            operation="write",
                            target_row=num_rows - 1,
                            target_col=num_cols - 1,
                            mc_runs=num_mc if use_mc else 1,
                            temperature=temperature,
                            vars=None,  # Input your data table
                        )
                    """
                    r_delay, r_pavg, r_pstc, r_pdyn = [1.43, 2], [2, 3], [3, 4], [4, 5]
                    time.sleep((use_mc + custom_mc + real_cell_mode) / 100)"""

                    time_usage = time.time() - start_time
                    # print(time_usage)
                    # 转化时间

                    time_usage = time.strftime("%H:%M:%S", time.gmtime(time_usage))

                    new_line = pd.DataFrame({i: str(eval(i)) for i in columns}, index=[datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")])

                    # mode 0（全真实）作为相对误差基准
                    if real_cell_mode == 0:
                        baseline_index = new_line.index[0]

                    result = pd.concat([result, new_line])
                    result.to_csv(os.path.join(output_dir, "result.csv"))
                    print(new_line.loc[new_line.index[0], :])

                    print("[DEBUG] Monte Carlo simulation completed")

                    # 计算该模式相对基准(mode 0)的误差
                    if baseline_index is not None and new_line.index[0] != baseline_index:
                        result_diff = pd.concat([result_diff, new_line])
                        for col in relative_diff:
                            base_vals = string_to_float_list(result.loc[baseline_index, col])
                            cur_vals = string_to_float_list(result.loc[new_line.index[0], col])
                            with np.errstate(divide="ignore", invalid="ignore"):
                                result_diff.loc[new_line.index[0], col] = np.where(base_vals != 0, (cur_vals - base_vals) / base_vals, 0.0)

                        # time usage
                        col = "time_usage"
                        base_t = time_str_to_float(result.loc[baseline_index, col])
                        cur_t = time_str_to_float(result.loc[new_line.index[0], col])
                        result_diff.loc[new_line.index[0], col] = (cur_t - base_t) / base_t if base_t else 0.0
                        print(result_diff.loc[new_line.index[0], :])

                        result_diff.to_csv(os.path.join(output_dir, "result_diff.csv"))
