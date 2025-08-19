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
    feature_num = num_rows*num_cols*6*3
    mean = np.array([0.4106, 0.045, -0.13, 0.4106, 0.045, -0.13, 0.4106, 0.045, -0.13, -0.3842, 0.02, -0.126, 0.4106, 0.045, -0.13,
                      -0.3842, 0.02, -0.126])
    means = np.tile(mean, num_rows*num_cols)
    spice =  Spice(feature_num ,means)
    warnings.filterwarnings("ignore", category=FutureWarning)
    if RUN_MODEL == "MC":
        p = 0.003
        variances = np.abs(means) * p
        cov_matrix = np.diag(variances)
        f_norm = norm_dist(mu=means, var=cov_matrix)
        mc = MC(f_norm=f_norm, spice=spice, mc_testbench=mc_testbench, feature_num=feature_num, num_cols=num_cols, num_rows=num_rows, 
                means=means, initial_num=1000, sample_num=300, FOM_use_num=100, seed=0,IS_bound_on=True,IS_bound_num=1)
        mc.start_estimate(max_num=1000000000)
        '''
        参数选择：feature_num=18:   p =0.003  
                feature_num=108:   p =0.0095   
                feature_num=1152:   p =0.048
        '''
    elif RUN_MODEL == "MNIS":
        p = 0.005
        variances = np.abs(means) * p
        cov_matrix = np.diag(variances)
        f = norm_dist(mu=means, var=cov_matrix)
        acs = MNIS(spice=spice,f_norm=f,mc_testbench=mc_testbench, feature_num=feature_num, num_rows=num_rows, num_cols =num_cols, IS_bound_num=1, 
                IS_bound_on=True, g_cal_val=0.24, g_sam_val=0.01,initial_fail_num=15, initial_sample_each=200, IS_num=150, FOM_num=13)
        acs.start_estimate(max_num=100000)
        '''
        Parameter：feature_num=18:   p =0.005   initial_fail_num, initial_sample_each,IS_num=15, 200, 150
                feature_num=108:   p =0.0095   initial_fail_num, initial_sample_each,IS_num=15, 100, 100
                feature_num=1152:   p =0.048   initial_fail_num, initial_sample_each,IS_num=10, 100, 80
        '''
    elif RUN_MODEL == "AIS":
        p = 0.004
        variances = np.abs(means) * p
        cov_matrix = np.diag(variances)
        f_norm = norm_dist(mu=means, var=cov_matrix)
        ais = AIS(spice=spice,mc_testbench=mc_testbench,f_norm=f_norm,feature_num=feature_num, num_rows=num_rows, num_cols= num_cols,
                g_cal_num=0.003, initial_failed_data_num=150,
                        num_generate_each_norm=1, sample_num_each_sphere=50, max_gen_times=1000,
                        FOM_num =11,  seed=7072, IS_bound_num=1, IS_bound_on=True)  
        fail_rate, sample_num, fom, used_time = ais.start_estimate(max_num=10000)
        '''
         Parameter：feature_num=18:   p =0.004   g_cal_num=0.003,  
                 feature_num=108: p =0.0095   g_cal_num=0.003,    
                 feature_num=1152: p =0.066  g_cal_num=0.0066
        '''
    elif RUN_MODEL == "ACS":
        p = 0.004
        variances = np.abs(means) * p
        cov_matrix = np.diag(variances)
        f_norm = norm_dist(mu=means, var=cov_matrix)
        acs = ACS(spice=spice, mc_testbench=mc_testbench, f_norm=f_norm, feature_num=feature_num, num_rows=num_rows, num_cols= num_cols, g_cal_val=0.001,
                initial_fail_num=10, initial_sample_each=100, IS_num=100, FOM_num=10,seed=0,IS_bound_num=1, IS_bound_on=True)
        acs.start_estimate(max_num=100000)
        '''
         Parameter：feature_num=18:   p =0.004   g_cal_val=0.001,   
                 feature_num=108: p =0.0098   g_cal_val=0.003,    
                 feature_num=1152:  p =0.064 g_cal_val=0.0064
        '''
    elif RUN_MODEL == "HSCS":
        f_norm = mixture_gaussian(pi=np.array([1]), mu=means.reshape(1,-1), var_num=0.0004)
        hscs = HSCS(spice=spice,f_norm=f_norm, mc_testbench=mc_testbench, means = means,feature_num=feature_num, num_cols = num_cols, num_rows=num_rows,
                    g_var_num=0.001,  bound_num=1, find_MN_sam_num=100,IS_sample_num=100, initial_failed_data_num=12, ratio=0.1,
                                sample_num_each_sphere=100, max_gen_times=100, FOM_num=25, seed=0,IS_bound_num=1, IS_bound_on=True)
        Pfail,sim_num = hscs.start_estimate(max_num=100000)
        '''
         Parameter：feature_num=18:   var_num=0.0004    g_var_num=0.001  FOM_num=25
                 feature_num=108:  var_num=0.0003    g_var_num=0.001  FOM_num=25
              feature_num=1152: var_num=0.001  g_var_num=0.0023   FOM_num=15
        '''
    else:
        print(f"未知模型: {RUN_MODEL}，请检查 RUN_MODEL 设置！")
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
