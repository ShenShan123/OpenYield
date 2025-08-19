from sram_yield_estimation.model_lib.MC import MC
from sram_yield_estimation.model_lib.MNIS import MNIS
from sram_yield_estimation.model_lib.AIS import AIS
from sram_yield_estimation.model_lib.ACS import ACS
from sram_yield_estimation.model_lib.HSCS import HSCS
from sram_yield_estimation.model_lib.spice import Spice
from sram_yield_estimation.tool.Distribution.normal_v1 import norm_dist
from sram_yield_estimation.tool.Distribution.gmm_v2 import mixture_gaussian
from utils import estimate_bitcell_area
from testbenches.sram_6t_core_testbench2 import Sram6TCoreTestbench # type: ignore
from testbenches.sram_6t_core_MC_testbench2 import Sram6TCoreMcTestbench
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
import numpy as np
from utils import estimate_bitcell_area # type: ignore
from config import SRAM_CONFIG

if __name__ == '__main__':
    # ================== 1. 加载所有配置 ==================
    sram_config = SRAM_CONFIG()
    sram_config.load_all_configs(
        global_file="yaml/global.yaml",
        circuit_configs={
            "SRAM_6T_CELL": "yaml/sram_6t_cell.yaml",
            "WORDLINEDRIVER": "yaml/wordline_driver.yaml",
            "PRECHARGE": "yaml/precharge.yaml",
            "COLUMNMUX": "yaml/mux.yaml",
            "SENSEAMP": "yaml/sa.yaml",
            "WRITEDRIVER": "yaml/write_driver.yaml",
            "DECODER":"yaml/decoder.yaml"
        }
    )

    # FreePDK45 default transistor sizes estimation
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
        param_sweep=True,
        sweep_precharge=True,
        sweep_senseamp=True,
        sweep_wordlinedriver=True,
        sweep_columnmux=True,
        sweep_writedriver=False,
        sweep_decoder=True,
        q_init_val=0, sim_path='sim',
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
