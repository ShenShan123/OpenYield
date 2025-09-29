from sram_compiler.testbenches.sram_6t_core_testbench import Sram6TCoreTestbench # type: ignore
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench # type: ignore
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
import numpy as np
from utils import estimate_bitcell_area # type: ignore
from config import SRAM_CONFIG
from datetime import datetime
import os

if __name__ == '__main__':
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
            "DECODER":"sram_compiler/config_yaml/decoder.yaml"
        }
    )

    # 2. 生成时间戳子目录
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    sim_path = os.path.join('sim', f"{time_str}_mc_6t")   # 例如 sim/20250928_153045_mc_6t
    os.makedirs(sim_path, exist_ok=True)

    # FreePDK45 default transistor sizes
    area = estimate_bitcell_area(
        w_access=sram_config.sram_6t_cell.nmos_width.value[1],#pg
        w_pd=sram_config.sram_6t_cell.nmos_width.value[0],
        w_pu=sram_config.sram_6t_cell.pmos_width.value,
        l_transistor=sram_config.sram_6t_cell.length.value
    )
    print(f"Estimated 6T SRAM Cell Area: {area*1e12:.2f} µm²")

    num_rows = sram_config.global_config.num_rows
    num_cols = sram_config.global_config.num_cols
    num_mc = sram_config.global_config.monte_carlo_runs

    print("===== 6T SRAM Array Monte Carlo Simulation Debug Session =====")
    mc_testbench = Sram6TCoreMcTestbench(
        sram_config,
        w_rc=False, # Whether add RC to nets
        pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
        vth_std=0.05, # Process parameter variation is a percentage of its value in model lib
        custom_mc=False, # Use your own process params?
        param_sweep=False,
        sweep_precharge=False,
        sweep_senseamp=False,
        sweep_wordlinedriver=False,
        sweep_columnmux=False,
        sweep_writedriver=False,
        sweep_decoder=False,
        coner='TT',#or FF or SS or FS or SF
        q_init_val=0, sim_path=sim_path,
    )
    # vars = np.random.rand(num_mc,num_rows*num_cols*18)

    # For using DC analysis, operation can be 'write_snm' 'hold_snm' 'read_snm'
    # read_snm = mc_testbench.run_mc_simulation(
    #     operation='hold_snm', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc,
    #     vars=None, # Input your data table
    # )

    # For using TRAN analysis, operation can be 'write' or 'read'
    w_delay, w_pavg = mc_testbench.run_mc_simulation(
        operation='read', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc,
        vars=None, # Input your data table
    )

    print("[DEBUG] Monte Carlo simulation completed")
