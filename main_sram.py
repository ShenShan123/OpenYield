from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench # type: ignore
#from sram_compiler_beifen.sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench # type: ignore
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
import numpy as np
from utils import estimate_bitcell_area # type: ignore
from sram_compiler.config_yaml.config import SRAM_CONFIG
from datetime import datetime
from sram_compiler.testbenches.yaml_change import summarize_from_csv, update_global_yaml_inplace, update_sram6t_yaml_inplace
import os

if __name__ == '__main__':
       # ================== 0. 更新YAML配置文件 ==================
    # 更新全局配置
    global_config_update = [16, 8, False ]      # num_rows.num_cols ,choose_columnmux   
    update_global_yaml_inplace(
        global_config_update,
        yaml_path="/home/majh/OpenYield/sram_compiler/config_yaml/global.yaml"
    )

    # 更新SRAM 6T单元配置
    sram6t_config_update = [
        9.0e-8, 1.35e-7, 9.0e-8,50.0e-9,     # pd_width pg_width  pu_width length
        "NMOS_VTG", "NMOS_VTG",  "PMOS_VTG"] # pd_model pg_model  pu_model 
    update_sram6t_yaml_inplace(
        sram6t_config_update,
        yaml_path="/home/majh/OpenYield/sram_compiler/config_yaml/sram_6t_cell.yaml"
    )
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

    temperature = sram_config.global_config.temperature
    num_rows = sram_config.global_config.num_rows
    num_cols = sram_config.global_config.num_cols
    num_mc = sram_config.global_config.monte_carlo_runs
    choose_columnmux = sram_config.global_config.choose_columnmux

    print("===== 6T SRAM Array Monte Carlo Simulation Debug Session =====")
    mc_testbench = Sram6TCoreMcTestbench(
        sram_config,
        w_rc=False, # Whether add RC to nets
        pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
        vth_std=0.05, # Process parameter variation is a percentage of its value in model lib
        custom_mc=False, # Use your own process params?
        sweep_cell=False,
        sweep_precharge=False,
        sweep_senseamp=False,
        sweep_wordlinedriver=False,
        sweep_columnmux=False,
        sweep_writedriver=False,
        sweep_decoder=False,
        corner='TT',#or FF or SS or FS or SF
        choose_columnmux=choose_columnmux,# Whether choose column mux or not
        q_init_val=0, sim_path=sim_path,
    )

    operation = 'read&write' #operation can be 'write' or 'read' or 'read&write' or 'hold_snm' or 'write_snm' or 'read_snm'
    if operation == 'write' or operation == 'read' or operation == 'read&write':
        data_csv_path = mc_testbench.run_mc_simulation(
            operation=operation, target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc,temperature=temperature,
            vars=None, # Input your data table
        )
    elif operation == 'hold_snm' or operation == 'write_snm' or operation == 'read_snm':
        read_snm = mc_testbench.run_mc_simulation(
            operation=operation, target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc,temperature=temperature,
            vars=None, # Input your data table
        )
    
    y = summarize_from_csv(os.path.join('/home/majh/OpenYield', data_csv_path),operation);y = np.append(y, area)

    print(f"[INPUT] construct_param: num_rows={global_config_update[0]}, num_cols={global_config_update[1]}, choose_columnmux={global_config_update[2]}")
    print(f"[INPUT] sram6tcell_param: pd_width={sram6t_config_update[0]*1e9:.1f} nm, pg_width={sram6t_config_update[1]*1e9:.1f} nm, pu_width={sram6t_config_update[2]*1e9:.1f} nm, length={sram6t_config_update[3]*1e9:.1f} nm", 
          f"pd_model={sram6t_config_update[4]}, pg_model={sram6t_config_update[5]}, pu_model={sram6t_config_update[6]}")
    print(f"[OUTPUT] y[0]=Delay({y[0]*1e9:.2f} ns), y[1]=Power({y[1]*-1e6:.2f} mW), y[2]=Area({y[2]*1e12:.2f} µm²)")
    print("[DEBUG] Monte Carlo simulation completed")
