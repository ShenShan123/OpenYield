from testbenches.sram_6t_core_testbench import SRAM_6T_Array_Testbench
from testbenches.sram_6t_core_MC_testbench import SRAM_6T_Array_MC_Testbench
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
import numpy as np


if __name__ == '__main__':
    pdk_path = 'model_lib/models.spice'
    nmos_model_name = 'NMOS_VTG'
    pmos_model_name = 'PMOS_VTG'
    pd_width = 0.205e-6
    pu_width = 0.09e-6
    pg_width = 0.135e-6
    length = 50e-9

    # FreePDK45 default transistor sizes
    # area = estimate_bitcell_area(
    #     w_access=pg_width,
    #     w_pd=pd_width,
    #     w_pu=pu_width,
    #     l_transistor=length
    # )
    # print(f"Estimated 6T SRAM Cell Area: {area*1e12:.2f} µm²")

    # assert 0
    num_rows = 32
    num_cols = 2
    # print("===== 6T SRAM Array NgSpice Simulation Debug Session =====")
    # testbench = SRAM_6T_Array_Testbench(
    #     pdk_path, nmos_model_name, pmos_model_name,
    #     pd_width, pu_width, pg_width, length,
    #     num_rows=2, num_cols=2, 
    #     w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.010 @ u_pF)
    
    # testbench.run_simulation(operation='write', target_row=0, target_col=0)
    # print(f"[DEBUG] Simulation completed")
    
    # testbench.run_simulation(operation='write', target_row=1, target_col=1)
    # print("[DEBUG] Simulation of read operation completed")

    print("===== 6T SRAM Array Monte Carlo Simulation Debug Session =====")
    mc_testbench = SRAM_6T_Array_MC_Testbench(
        pdk_path, nmos_model_name, pmos_model_name,
        pd_width, pu_width, pg_width, length,
        num_rows=num_rows, num_cols=num_cols, 
        w_rc=False, # Add RC to nets
        pi_res=10 @ u_Ohm, pi_cap=0.001 @ u_pF,
        custom_mc=True, # Use your process params?
        sim_path='sim',
    )
    num_mc=1500
    vars = np.random.rand(num_mc,num_rows*num_cols*18)
    # Operation can be 'read' 'write' 'write_snm' 'hold_snm' 'read_snm'
    y = mc_testbench.run_mc_simulation(
        operation='write_snm', target_row=0, target_col=0, mc_runs=num_mc, 
        vars=None, # Input your data table
    )
    # print('[DEBUG] y.shape', y.shape)
    # assert 0
    # y = mc_testbench.run_mc_simulation(operation='write', target_row=1, target_col=1, mc_runs=3, vars=vars)

    print("[DEBUG] Monte Carlo simulation completed")