from yield_estimation.model_lib.MC import MC
from yield_estimation.model_lib.MNIS import MNIS
from yield_estimation.model_lib.AIS import AIS
from yield_estimation.model_lib.ACS import ACS
from yield_estimation.model_lib.HSCS import HSCS
import sys
import numpy as np
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
parent_dir_of_code1 = '/home/lixy/OpenYield-main/yield_estimation'
sys.path.append(parent_dir_of_code1) 
from tool.util import write_data2csv, seed_set
from tool.Distribution.normal_v1 import norm_dist
from tool.Distribution.gmm_v2 import mixture_gaussian
parent_dir_of_code2 = '/home/lixy/OpenYield-main/sram_compiler'
sys.path.append(parent_dir_of_code2) 
from testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
from config import SRAM_CONFIG
from utils import estimate_bitcell_area # type: ignore
RUN_MODEL = "MC"  # Options："MC", "MNIS", "AIS", "ACS", "HSCS"
if __name__ == '__main__':
    sram_config = SRAM_CONFIG()
    sram_config.load_all_configs(
        global_file="/home/lixy/OpenYield-main/sram_compiler/config_yaml/global.yaml",
        circuit_configs={
            "SRAM_6T_CELL": "/home/lixy/OpenYield-main/sram_compiler/config_yaml/sram_6t_cell.yaml",
            "WORDLINEDRIVER": "/home/lixy/OpenYield-main/sram_compiler/config_yaml/wordline_driver.yaml",
            "PRECHARGE": "/home/lixy/OpenYield-main/sram_compiler/config_yaml/precharge.yaml",
            "COLUMNMUX": "/home/lixy/OpenYield-main/sram_compiler/config_yaml/mux.yaml",
            "SENSEAMP": "/home/lixy/OpenYield-main/sram_compiler/config_yaml/sa.yaml",
            "WRITEDRIVER": "/home/lixy/OpenYield-main/sram_compiler/config_yaml/write_driver.yaml",
            "DECODER":"/home/lixy/OpenYield-main/sram_compiler/config_yaml/decoder.yaml"
        }
    )
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
        w_rc=True, # Whether add RC to nets
        pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
        vth_std=0.05, # Process parameter variation is a percentage of its value in model lib
        custom_mc=True, # Use your own process params?
        param_sweep=False,
        sweep_precharge=False,
        sweep_senseamp=False,
        sweep_wordlinedriver=False,
        sweep_columnmux=False,
        sweep_writedriver=False,
        sweep_decoder=False,
        q_init_val=0, sim_path='/home/lixy/OpenYield-main/sim2',
    )
    feature_num = num_rows*num_cols*6*3
    print(feature_num)
    mean = np.array([0.4106, 0.045, -0.13, 0.4106, 0.045, -0.13, 0.4106, 0.045, -0.13, -0.3842, 0.02, -0.126, 0.4106, 0.045, -0.13,
                      -0.3842, 0.02, -0.126])
    means = np.tile(mean, num_rows*num_cols)
    if RUN_MODEL == "MC":
        if feature_num == 18:
            variances = np.abs(means) * 0.003
        elif feature_num ==108:
            variances = np.abs(means) * 0.0095
        elif feature_num ==576:
            variances = np.abs(means) * 0.048
        cov_matrix = np.diag(variances)
        f_norm = norm_dist(mu=means, var=cov_matrix)
        mc = MC(f_norm=f_norm, mc_testbench=mc_testbench, feature_num=feature_num, means=means, initial_num=1000, sample_num=100, FOM_use_num=100, seed=0, IS_bound_on=True,IS_bound_num=1)
        mc.start_estimate(max_num=1000000000)
    elif RUN_MODEL == "MNIS":
        if feature_num ==18:
            variances = np.abs(means) * 0.005 
            initial_fail_num, initial_sample_each,IS_num=15, 200, 150
        elif feature_num ==108:
            variances = np.abs(means) * 0.0095
            initial_fail_num, initial_sample_each,IS_num=15, 100, 100
        elif feature_num ==576:
            variances = np.abs(means) * 0.048
            initial_fail_num, initial_sample_each,IS_num=10, 100, 80
        cov_matrix = np.diag(variances)
        f = norm_dist(mu=means, var=cov_matrix)
        acs = MNIS(f_norm=f,mc_testbench=mc_testbench, feature_num=feature_num, means=means, 
                IS_bound_num=1, IS_bound_on=True, g_cal_val=0.24, g_sam_val=0.01,
                        initial_fail_num=initial_fail_num, initial_sample_each=initial_fail_num, IS_num=IS_num, FOM_num=13)
        acs.start_estimate(max_num=100000)
    elif RUN_MODEL == "AIS":
        if feature_num ==18:
            variances = np.abs(means) * 0.004 
            g_cal_num=0.003
        elif feature_num ==108:
            variances = np.abs(means) * 0.0095
            g_cal_num=0.003
        elif feature_num ==576:
            variances = np.abs(means) * 0.066
            g_cal_num=0.0066
        cov_matrix = np.diag(variances)
        f_norm = norm_dist(mu=means, var=cov_matrix)
        ais = AIS(mc_testbench=mc_testbench,feature_num=feature_num, f_norm=f_norm, g_cal_num=g_cal_num, initial_failed_data_num=150,
                        num_generate_each_norm=1, sample_num_each_sphere=50, max_gen_times=1000,
                        FOM_num =11,  seed=7072, IS_bound_num=1, IS_bound_on=True)  #case4 参数
        fail_rate, sample_num, fom, used_time = ais.start_estimate(max_num=10000)
    elif RUN_MODEL == "ACS":
        if feature_num ==18:
            variances = np.abs(means) * 0.003
            g_cal_val=0.001 
        elif feature_num ==108:
            variances = np.abs(means) * 0.0098
            g_cal_val=0.003
        elif feature_num ==576:
            variances = np.abs(means) * 0.064
            g_cal_val=0.0064
        cov_matrix = np.diag(variances)
        f_norm = norm_dist(mu=means, var=cov_matrix)
        acs = ACS(mc_testbench=mc_testbench, f_norm=f_norm, feature_num=feature_num, g_cal_val=g_cal_val,
                initial_fail_num=10, initial_sample_each=100, IS_num=100, FOM_num=10,seed=0,IS_bound_num=1, IS_bound_on=True)
        acs.start_estimate(max_num=100000)
    elif RUN_MODEL == "HSCS":
        if feature_num == 18:
            var_num=0.0004
            g_var_num=0.001
            FOM_num=25
        elif feature_num ==108:
            var_num=0.0003
            g_var_num=0.001
            FOM_num=25
        elif feature_num ==576:
            var_num=0.001
            g_var_num=0.0023
            FOM_num=15
        f_norm = mixture_gaussian(pi=np.array([1]), mu=means.reshape(1,-1), var_num=var_num)
        hscs = HSCS(f_norm=f_norm, mc_testbench=mc_testbench, means = means,feature_num=feature_num, g_var_num=g_var_num, 
                    bound_num=1, find_MN_sam_num=100,IS_sample_num=100, initial_failed_data_num=12, ratio=0.1,
                                sample_num_each_sphere=100, max_gen_times=100, FOM_num=FOM_num, seed=0,IS_bound_num=1, IS_bound_on=True)
        Pfail,sim_num = hscs.start_estimate(max_num=100000)
    else:
        print(f"未知模型: {RUN_MODEL}，请检查 RUN_MODEL 设置！")
        
