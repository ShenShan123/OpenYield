from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
# from subcircuits.sram_6t_core_wdriver_sensamp import Sram6TCoreWdriverSenseamp
from testbenches.sram_6t_core_wdriver_testbench import Sram6TCoreWdriverTestbench

if __name__ == '__main__':
    vdd = 1.0
    pdk_path = 'model_lib/models.spice'
    nmos_model_name = 'NMOS_VTG'
    pmos_model_name = 'PMOS_VTG'
    num_rows = 2
    num_cols = 2
    pd_width = 0.205e-6
    pu_width = 0.09e-6
    pg_width = 0.135e-6
    length = 50e-9

    # array = Sram6TCoreWdriverSenseamp(
    #     1.0, num_rows, num_cols, 
    #     nmos_model_name, pmos_model_name,
    # )
    # print(array)

    tb = Sram6TCoreWdriverTestbench(
        vdd,
        pdk_path, nmos_model_name, pmos_model_name,
        pd_width, pu_width, pg_width, length,
        num_rows=num_rows, num_cols=num_cols, 
        w_rc=False, # Add RC to nets
        pi_res=10 @ u_Ohm, pi_cap=0.001 @ u_pF,
        custom_mc=False, # Use your process params?
        q_init_val=0, sim_path='sim',
    )

    tb_cir = tb.create_testbench(operation='write', target_row=0, target_col=0)

    print(tb_cir)