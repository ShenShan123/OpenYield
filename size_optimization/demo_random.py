# 随机搜索算法

from testbenches.sram_6t_core_testbench import Sram6TCoreTestbench
import numpy as np

# generate random parameters
params = {
    'nmos_model_name': 'NMOS_VTG',
    'pmos_model_name': 'PMOS_VTG',
    'pd_width': 0.205e-6,
    'pu_width': 0.09e-6,
    'pg_width': 0.135e-9,
    'length': 50e-9,
}

mc_testbench = Sram6TCoreMcTestbench(
    vdd,
    pdk_path,
    params['nmos_model_name'],
    params['pmos_model_name'],
    num_rows=num_rows,
    num_cols=num_cols,
    pd_width=params['pd_width'],
    pu_width=params['pu_width'],
    pg_width=params['pg_width'],
    length=params['length'],
    w_rc=False,
    pi_res=10 @ u_Ohm,
    pi_cap=0.001 @ u_pF,
    custom_mc=False,
    q_init_val=0,
    sim_path='sim',
)

# generate random parameters

result = mc_testbench.run_mc_simulation()

print(result)