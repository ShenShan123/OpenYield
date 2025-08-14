from testbenches.sram_6t_core_testbench import Sram6TCoreTestbench
from testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
import numpy as np
from utils import estimate_bitcell_area

if __name__ == '__main__':
    vdd = 1.0
    pdk_path = 'model_lib/models.spice'
    nmos_model_name = 'NMOS_VTG'
    pmos_model_name = 'PMOS_VTG'
    pd_width = 0.205e-6
    pu_width = 0.09e-6
    pg_width = 0.135e-6
    length = 50e-9

    # FreePDK45 default transistor sizes
    area = estimate_bitcell_area(
        w_access=pg_width,
        w_pd=pd_width,
        w_pu=pu_width,
        l_transistor=length
    )
    print(f"Estimated 6T SRAM Cell Area: {area*1e12:.2f} µm²")

    num_rows = 64
    num_cols = 4
    num_mc = 1

    print("===== 6T SRAM Array Monte Carlo Simulation Debug Session =====")
    mc_testbench = Sram6TCoreMcTestbench(
        vdd,
        pdk_path, nmos_model_name, pmos_model_name,
        num_rows=num_rows, num_cols=num_cols, 
        pd_width=pd_width, pu_width=pu_width, 
        pg_width=pg_width, length=length,
        w_rc=True, # Whether add RC to nets
        pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
        vth_std=0.05, # Process parameter variation is a percentage of its value in model lib
        custom_mc=False, # Use your own process params?
        q_init_val=0, sim_path='sim',
    )
    # vars = np.random.rand(num_mc,num_rows*num_cols*18)

    # For using DC analysis, operation can be 'write_snm' 'hold_snm' 'read_snm'
    # read_snm = mc_testbench.run_mc_simulation(
    #     operation='write_snm', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, 
    #     vars=None, # Input your data table
    # )

    # For using TRAN analysis, operation can be 'write' or 'read'
    w_delay, w_pavg = mc_testbench.run_mc_simulation(
        operation='read', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, 
        vars=None, # Input your data table
    )

    print("[DEBUG] Monte Carlo simulation completed")