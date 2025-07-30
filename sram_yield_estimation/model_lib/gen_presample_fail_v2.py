import numpy as np
import pandas as pd
import sys
from scipy.stats import norm
import math
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
parent_dir_of_code1 = '/home/lixy/sram_yield_estimation/'
sys.path.append(parent_dir_of_code1) 
from tool.delete import delete_folder_content
from model_lib.spice import Spice
parent_dir_of_code2 = '/home/lixy/OpenYield-main'
# 将父目录添加到模块搜索路径
sys.path.append(parent_dir_of_code2) 
from testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
import warnings
# from statistics import NormalDist
# from uniform import uniform_dist
import torch
np.random.seed(2)

'''
针对不同维度的高维空间，通过自适应调整采样范围，高效采集 "失效样本"（满足特定条件的样本），并将结果保存用于后续分析。
整体逻辑围绕 "从外围到内部逐步缩小采样范围" 来定位失效样本，适用于高维空间中失效事件可能集中在特定区域的场景。
'''
def fang_presampling (dim, spice, mc_testbench, means, num_rows, num_cols):
    # spice = SPICE_case2_new.SPICE_Case2_new()  # 18dim
         # 18dim
    #  初始化
    r = 5
    R = int(0.5*(math.sqrt(dim))+r)
    i = 0
    p = 0.95
    # 确定最外围超球面半径
    isTerminate = False
    while isTerminate == False:
        r = norm.ppf(p)
        print('r:', r)
        i = i+1
        R = R-2*r
        print('R:', R)
        a = np.random.uniform(low=means - 0.001*R, high=means + 0.001*R, size=(100, dim))
        num_mc = a.shape[0]
        print(num_mc)
        y, w_pavg = mc_testbench.run_mc_simulation(operation='write', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, vars=a)
        y_true_1 = y.reshape(num_mc,1)
        print(y_true_1.shape)
        print(y_true_1)
        indic_new = spice.indicator(y_true_1)
        folder_path = '/home/lixy/sim_write0'
        delete_folder_content(folder_path)
        p_fail = indic_new.sum()/indic_new.shape[0]
        if p_fail < 0.1:
            isTerminate = True
            R_min = R
            R_outer = R_min
            r = 1
            print(i)
            print(R)
        p = p-0.05
    # 内推法采集失效样本
    simu = 0
    isTerminate_1 = False
    u = 0
    res = []
    res1 = []
    simu_test = 0
    # while isTerminate_1 == False:
    '''2023ICCAD_lyf
    
    从R_outer开始，逐步缩小R（每次缩小2*r），在新的范围内采样。
    样本量与范围缩小的幅度（R_outer - R）成正比，范围缩小越多，采样越多（因为失效样本可能更集中在小范围内）。
    通过阈值判断样本是否失效（y_true_2[i] > threshold），收集到 100 个失效样本后停止。
    '''
    for i in range(5):  # 6：1093； 5：569 # 3:108
        # p = p+0.03  # 1093
        # p = p + 0.005  # 569
        p = p+0.005
        r = norm.ppf(p)
        print(p)
        print('r:', r)
        print('R_initial:', R)
        R = R-2*r
        b = np.random.uniform(low=means -0.001*R, high=means + 0.001*R, size=(int(50*(R_outer-R)), dim))
        simu += b.shape[0]
        num_mc2 = b.shape[0]
        print(num_mc2)
        y2, w_pavg = mc_testbench.run_mc_simulation(operation='write', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc2, vars=b)
        y_true_2 = y2.reshape(num_mc2,1)
        print(y_true_2.shape)
        indic_new = spice.indicator(y_true_2)
        folder_path = '/home/lixy/sim_write0'
        delete_folder_content(folder_path)
        print(R)
        print('*************************')
        threshold = 8.86271e-10

        for i in range(int(50*(R_outer-R))-1):
            simu_test += 1
            if y_true_2[i] > threshold:
                u = u+1
                res.append(i)
                res1.append(np.hstack([y_true_2[i], b[i]]))
                if u >= 100:
                    # isTerminate_1 = True
                    break
        print(u)
        print(simu)
        print(simu_test)
        # if u = 0:

    return res1, simu_test

if __name__ == "__main__":
    vdd = 1.0
    pdk_path = '/home/lixy/OpenYield-main/model_lib/models.spice'
    nmos_model_name = 'NMOS_VTG'
    pmos_model_name = 'PMOS_VTG'
    pd_width = 0.205e-6
    pu_width = 0.09e-6
    pg_width = 0.135e-6
    length = 50e-9
    num_rows = 1
    num_cols = 1
    feature_num = num_rows*num_cols*6*3
    mean = np.array([0.4106, 0.045, -0.13, 0.4106, 0.045, -0.13, 0.4106, 0.045, -0.13, -0.3842, 0.02, -0.126, 0.4106, 0.045, -0.13,
                      -0.3842, 0.02, -0.126])
    means = np.tile(mean, num_rows*num_cols)
    spice =  Spice(feature_num ,means)
    mc_testbench = Sram6TCoreMcTestbench(
        vdd,
        pdk_path, nmos_model_name, pmos_model_name,
        num_rows=num_rows, num_cols=num_cols, 
        pd_width=pd_width, pu_width=pu_width, 
        pg_width=pg_width, length=length,
        w_rc=True, # Whether add RC to nets
        pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
        vth_std=0.05, # Process parameter variation is a percentage of its value in model lib
        custom_mc=True, # Use your own process params?
        q_init_val=0, sim_path='/home/lixy/sim_write0' )
    warnings.filterwarnings("ignore", category=FutureWarning)
    simu_pre_all = []
    dim = feature_num
    for seed in range(0, 50):
        np.random.seed(seed)
        torch.manual_seed(seed)
        list_data, simu_pre = fang_presampling(dim, spice, mc_testbench, means, num_rows, num_cols)
        data = np.array(list_data)
        # print(data.shape)
        # print(data)
        simu_pre_all.append(simu_pre)
        df = pd.DataFrame(data)
        df.to_csv(f'./pre_fail_{dim}/fail_{dim}dim_{seed}.csv',index= False, header=1)
    df = pd.DataFrame(simu_pre_all)
    df.to_csv(f'./pre_fail_{dim}/fail_{dim}dim_simu.csv', index=False, header=1)
    # indic_new = spice.indicator(y_true)
    # print(indic_new.sum()/indic_new.shape[0])