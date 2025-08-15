import torch
import torch.nn as nn
import pyro.distributions as dist
import pyro.distributions.transforms as T
import numpy as np
import pandas as pd

import sys
from scipy.stats import norm
import math
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
parent_dir_of_code1 = '/home/lixy/sram_yield_estimation/'
sys.path.append(parent_dir_of_code1) 
from tool.delete import delete_folder_content
from tool.util import write_data2csv
from model_lib.spice import Spice
parent_dir_of_code2 = '/home/lixy/OpenYield-main'
# 将父目录添加到模块搜索路径
sys.path.append(parent_dir_of_code2) 
from testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
import warnings
from spice import Spice
from nflows.distributions.normal import StandardNormal
import csv
import os
import time
smoke_test = ('CI' in os.environ)

class Optimal():
    def __init__(self, spice, seed, feature_num, num_rows, num_cols, means, mc_testbench):
        self.spice = spice
        self.seed = seed
        self.feature_num = feature_num
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.means = means
        self.mc_testbench = mc_testbench
    def _calculate_FOM(self, fail_rate_list, FOM_num):
        length = len(fail_rate_list)
        assert length >= 1
        if length == 1 or np.mean(fail_rate_list[-FOM_num:]) == 0:
            return 1
        else:
            return np.std(fail_rate_list[-FOM_num:]) / np.mean(fail_rate_list[-FOM_num:])


    def save_result(self, P_fail, FOM, num, used_time, seed):
        data_info_list = [[P_fail], [FOM], [num], [used_time]]
        write_data2csv(tgt_dir=os.path.join(f"results/Optimal"),  # 保存目的文件
                       tgt_name=f"Optimal_case{self.spice.case}_{seed}.csv",  # 文件名:包含训练数据量, 模型名
                       head_info=('Pfail', 'FOM', 'num', 'used_time'),  # 表头
                       data_info=data_info_list)  # 信息

    '''
    流模型（Normalizing Flow）：通过一系列可逆变换，将简单的基础分布（多元正态）映射到复杂的目标分布（失效样本分布），用于高效采样和概率计算。
    初始数据：使用fang_presampling生成的失效样本作为初始训练数据，帮助流模型快速聚焦于失效区域。
    '''
    def estimate_18dim(self, max_num=1000000):
        # the training data
        df = pd.read_csv(f'/home/lixy/OpenYield-main/sram_yield_estimation/bound_lib/fail_18dim_0.csv')
        data = np.array(df)
        df = pd.read_csv(f'/home/lixy/OpenYield-main/sram_yield_estimation/bound_lib/fail_18dim_simu.csv')
        IS_simulation = np.array(df)
        # set seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # define SPICE
        spice = self.spice
        feature_num = self.feature_num
        num_cols = self.num_cols
        num_rows = self.num_rows
        means = self.means
        mc_testbench = self.mc_testbench
        variances = means * 0.01  # 每个维度的方差
        covariance_matrix = torch.diag(variances)  # 构造对角协方差矩阵

        # the base distribution and transform distribution
        base_dist = dist.MultivariateNormal(loc = means, covariance_matrix=covariance_matrix)
        spline_transform = T.spline_coupling(18, split_dim=18 // 2, hidden_dims=[18 * 4, 18 * 4],
                                             count_bins=10, bound=8)
        flow_dist = torch.distributions.TransformedDistribution(base_dist, [spline_transform])
        w_ex = []
        input_data = data[:, 1:19]  # the training data
        # input_data = data  # the training data
        iteration = []
        p_fail_list = []
        IS_simulation = IS_simulation[self.seed, 0]
        now_time = time.time()
        sample_data_num = 0
        FOM_metric = np.inf
        t = 0
        pre_num = 100
        while (sample_data_num < max_num) and (FOM_metric >= 0.1):
            # for t in range(100):
            t = t + 1
            iteration.append(t)
            print(t, '次更新**********************')
            dataset = torch.tensor(input_data, dtype=torch.float)
            # Training parameters
            steps = 1 if smoke_test else 501
            optimizer = torch.optim.Adam(spline_transform.parameters(), lr=4.5e-5)
            for step in range(steps + 1):
                optimizer.zero_grad()
                loss = -flow_dist.log_prob(dataset).mean()
                loss.backward()
                optimizer.step()
                flow_dist.clear_cache()
                if step % 100 == 0:
                    print('step: {}, loss: {}'.format(step, loss.item()))
            X_flow = flow_dist.sample(torch.Size([100, ])).detach().numpy()  # Sampling from the generated distribution
            IS_simulation += 100
            sample_data_num += 100
            # Calculate the number of failed samples from NF
            num_mc = X_flow.shape[0]
            y, w_pavg = mc_testbench.run_mc_simulation(operation='write', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, vars=X_flow)
            y_flow = y.reshape(num_mc,1)
            u = 0
            for i in range(99):
                if y_flow[i] > 8.86271e-10:
                    u = u + 1
            print('生成失效样本:', u)
            # all_data = X_flow
            all_data = np.vstack([input_data, X_flow])  # all samples
            # Calculate sample weight
            # log_g(x)
            g_prob_log = flow_dist.log_prob(torch.tensor(all_data, dtype=torch.float)).detach().numpy().reshape(-1, 1)
            # f(x)：Multivariate normal distribution
            f_prob_fun = StandardNormal(shape=[18])
            # log_f(x)
            f_prob_log = f_prob_fun.log_prob(torch.tensor(all_data)).detach().numpy().reshape(-1, 1)
            y_value = spice(all_data)
            indic = spice.indicator(y_value)  # I(x)
            w = np.exp(f_prob_log - g_prob_log) * indic  # w= (f/g)*I= exp(log(f/g))*I=exp(log_f-log_g)*I
            # Find the maximum 200 weights and update Training Data
            w_max = (w[np.lexsort(-w.T)])[:pre_num]
            for i in range(pre_num-1):
                if w_max[i] < 1:
                    w_ex.append(w_max[i])
            w = w.tolist()
            pos = []
            for i_d in w_max:
                pos.append(w.index(i_d))
            P = all_data[pos, :]
            input_data = P
            # input_data = all_data
            # Calculate failure rate
            print("************************")
            p_fail = sum(w_ex) / (pre_num * t)
            p_fail = p_fail[0]
            print('fail_rate：', p_fail)
            p_fail_list.append(p_fail)
            FOM_metric = self._calculate_FOM(p_fail_list, 10)

            # save metric & data sampled number
            self.save_result(p_fail, FOM_metric, num=IS_simulation, used_time=time.time() - now_time, seed=self.seed)
            # if FOM_metric < 0.1:
            #     break

    def start_estimate(self, max_num=1000000):
        self.estimate_108dim()
       






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
    for i in range(50):
        try:
            Optimal(spice=spice, seed=i, mc_testbench=mc_testbench, feature_num=feature_num,means= means, num_cols=num_cols, num_rows=num_rows).start_estimate()
        except:
            continue










