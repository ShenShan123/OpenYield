import time
import os
import numpy as np
import sys
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
parent_dir_of_code1 = '/home/lixy/sram_yield_estimation/'
# 将父目录添加到模块搜索路径
sys.path.append(parent_dir_of_code1) 
from tool.util import write_data2csv, seed_set
from tool.delete import delete_folder_content
from tool.Distribution.normal_v1 import norm_dist
parent_dir_of_code = '/home/lixy/Code'
from model_lib.spice import Spice
parent_dir_of_code2 = '/home/lixy/OpenYield-main'
sys.path.append(parent_dir_of_code2) 
from testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
import warnings
class MNIS():
    def __init__(self, spice,mc_testbench,f_norm, feature_num, num_rows, num_cols, g_sam_val, initial_fail_num, initial_sample_each, IS_num, FOM_num, g_cal_val, IS_bound_num, seed=0, IS_bound_on=False):
        self.spice = spice
        self.mc_testbench = mc_testbench
        self.feature_num =feature_num
        self.f_norm, self.g_sam_val, self.initial_fail_num, self.initial_sample_each, self.IS_num, self.FOM_num = f_norm, g_sam_val, initial_fail_num, initial_sample_each, IS_num, FOM_num
        self.seed = seed
        self.IS_bound_num, self.IS_bound_on, self.g_cal_val = IS_bound_num, IS_bound_on, g_cal_val
        seed_set(seed)
        self.num_rows = num_rows
        self.num_cols = num_cols

    def _calculate_val(self, x, y, f_x, g_x, spice):
        log_f_val = f_x.log_pdf(x).reshape([-1])
        log_g_val = g_x.log_pdf(x).reshape([-1])
        I_val = spice.indicator(y).reshape([-1])
        return log_f_val, log_g_val, I_val

    def _calculate_fail_rate_this_round(self, log_f_val, log_g_val, I_val):
        IS_num = log_f_val.shape[0]
        w_val = np.exp(log_f_val - log_g_val)
        #print(IS_num,log_f_val)
        #print(log_g_val)
        w_val[(w_val == np.inf)] = 1e290

        fail_rate_this_round = (w_val * I_val).sum() / IS_num
        return fail_rate_this_round

    def _calculate_fail_rate(self, fail_rate_this_round, fail_rate_list):
        fail_rate = (sum(fail_rate_list) + fail_rate_this_round) / (len(fail_rate_list) + 1)
        return fail_rate

    def _calculate_FOM(self, fail_rate_list, FOM_num):
        length = len(fail_rate_list)
        assert length >= 1
        if length == 1 or np.mean(fail_rate_list[-FOM_num:]) == 0:
            return 1
        else:
            # print("std", np.std(fail_rate_list[-FOM_num:]))
            return np.std(fail_rate_list[-FOM_num:]) / np.mean(fail_rate_list[-FOM_num:])

    def _save_result(self, P_fail, FOM, num, used_time, seed):
        data_info_list = [[P_fail], [FOM], [num], [used_time]]

        write_data2csv(tgt_dir=os.path.join("./results/MNIS"),  # 保存目的文件
                       tgt_name=f"MNIS_read_{self.feature_num}.csv",  # 文件名:包含训练数据量, 模型名
                       head_info=('Pfail', 'FOM', 'num', 'used_time'),  # 表头
                       data_info=data_info_list)  # 信息
        
    def _initial_sampling(self, initial_fail_num, sample_num_each_sphere, spice):
        captured_fail_data_num = 0
        iter_count = 0
        capture_any_fail_data_flag = False
        feat_num = self.feature_num  # feature number of x

        while captured_fail_data_num < initial_fail_num:
            new_x = np.random.uniform(low=spice.low_bounds, high=spice.up_bounds,
                                      size=[sample_num_each_sphere, feat_num])
            num_mc = new_x.shape[0]
            #print(num_mc)
            y, w_pavg = self.mc_testbench.run_mc_simulation(operation='write', target_row=self.num_rows-1, target_col=self.num_cols-1, mc_runs=num_mc, vars=new_x)
            new_y = y.reshape(num_mc,1)
            folder_path = '/home/lixy/sim0'
            delete_folder_content(folder_path)
            # print(new_y)
            y_labels = spice.indicator(new_y).reshape([-1])

            if y_labels.any() == True:
                failed_x = new_x[y_labels]

                if capture_any_fail_data_flag == False:
                    x_samples = failed_x
                else:
                    x_samples = np.vstack([x_samples, failed_x])

                capture_any_fail_data_flag = True
                captured_fail_data_num += failed_x.shape[0]

            iter_count += 1
            print(iter_count, y_labels.any())

        x_samples = x_samples[0:initial_fail_num, :]  # discard excess samples
        sample_total_num = (iter_count + 1) * sample_num_each_sphere
        return x_samples, sample_total_num

    def _IS_bound(self, x, spice, IS_bound_num):
        low_bound_full = np.ones(x.shape) * spice.low_bounds * IS_bound_num
        high_bound_full = np.ones(x.shape) * spice.up_bounds * IS_bound_num
        x[x<low_bound_full] = low_bound_full[x<low_bound_full]
        x[x>high_bound_full] = high_bound_full[x>high_bound_full]
        return x
    
    def start_estimate(self, max_num=None):
        """
            call this function to start the yield estimation process,
            and the numerical results will be saved in "./results/MNIS_case3.csv" automatically.

            :param f_norm: the origin MC distribution, usually a Gaussian distribution
            :param initial_fail_num: the number of initial failed sample
            :param initial_sample_each: the number of samples each time during initial sampling
            :param IS_num: the number of importance sampling (IS) during each importance sampling iteration
            :param g_sam_val: the used variance of importance distribution during sampling采样过程中重要性分布所使用的方差值
            :param FOM_num: the used number of latest fail rates to calculate FOM
        """

        f_norm, g_sam_val, initial_fail_num, initial_sample_each, IS_num, FOM_num = self.f_norm, self.g_sam_val, self.initial_fail_num, self.initial_sample_each, self.IS_num, self.FOM_num
        IS_bound_num, IS_bound_on, g_cal_val = self.IS_bound_num, self.IS_bound_on, self.g_cal_val
        feature_num = self.feature_num
        time1 = time.time()
        num_cols, num_rows = self.num_cols, self.num_rows
        self.x_fail, origin_sample_num = self._initial_sampling(initial_fail_num, initial_sample_each, self.spice)
        num_mc = self.x_fail.shape[0]
        y, w_pavg = self.mc_testbench.run_mc_simulation(operation='write', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, vars=self.x_fail)
        self.y_fail = y.reshape(num_mc,1)
        folder_path = '/home/lixy/sim0'
        delete_folder_content(folder_path)
        #print(self.y_fail) #需要替换
        centered_x_fail = self.x_fail - means
        distances = np.sqrt((centered_x_fail * centered_x_fail).sum(-1))
        min_index = distances.argmin()
        min_norm = self.x_fail[min_index, :]
        #min_norm = self.x_fail[abs(self.x_fail-means).sum(-1).argmin(), :]## L1 使用 L1 范数（欧几里得距离）来找出距离原点最近的失败样本：
        if feature_num ==18:
            variances = np.abs(means) * 0.004
        elif feature_num ==108:
            variances = np.abs(means) * 0.004
        elif feature_num ==1152:
            variances = np.abs(means) * 0.048
        cov_matrix = np.diag(variances)
        g_sam_norm = norm_dist(mu=min_norm, var=cov_matrix)
        g_cal_norm = norm_dist(mu=min_norm, var=cov_matrix)
        fail_rate_list = []
        FOM_list = []
        fail_rate_this_round_list = []
        iter_count = 0
        if max_num == None:
            max_reach_flag = True
        else:
            max_reach_flag = iter_count * IS_num + origin_sample_num < max_num
        FOM = 1

        while ((max_reach_flag) and (FOM>0.05)) or (iter_count<15):
            self.label_fail = None

            # IS samples
            x_IS = g_sam_norm.sample(n=IS_num)

            if IS_bound_on:
                x_IS = self._IS_bound(x_IS, self.spice, IS_bound_num)
            num_mc = x_IS.shape[0]
            #print(num_mc)
            y, w_pavg = self.mc_testbench.run_mc_simulation(operation='write', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, vars=x_IS)
            y_IS = y.reshape(num_mc,1)  #需要替换
            folder_path = '/home/lixy/sim0'
            delete_folder_content(folder_path)
            #print(y_IS)
            # get log f(x), log g(x) and I(x)
            log_f_IS_val, log_g_IS_val, I_IS_val = self._calculate_val(x_IS, y_IS, f_norm, g_cal_norm, self.spice)

            # the fail_rate calculated only using IS samples of this iteration round
            fail_rate_this_round = self._calculate_fail_rate_this_round(log_f_IS_val, log_g_IS_val, I_IS_val)

            # the real overall fail_rate after this iteration
            fail_rate = self._calculate_fail_rate(fail_rate_this_round, fail_rate_this_round_list)

            fail_rate_this_round_list.append(fail_rate_this_round)
            fail_rate_list.append(fail_rate)

            FOM = self._calculate_FOM(fail_rate_list, FOM_num)
            FOM_list.append(FOM)

            iter_count += 1

            self._save_result(fail_rate, FOM, iter_count*IS_num+origin_sample_num, time.time()-time1, self.seed)
            print(f"[MNIS] num:{iter_count * IS_num + origin_sample_num}, pfail:{fail_rate}, FOM:{FOM}")
            max_reach_flag = iter_count * IS_num + origin_sample_num < max_num
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
        q_init_val=0, sim_path='/home/lixy/sim_read0' )
    warnings.filterwarnings("ignore", category=FutureWarning)
    if feature_num ==18:
        variances = np.abs(means) * 0.005 
        initial_fail_num, initial_sample_each,IS_num=15, 200, 150
    elif feature_num ==108:
        variances = np.abs(means) * 0.0095
        initial_fail_num, initial_sample_each,IS_num=15, 100, 100
    elif feature_num ==1152:
        variances = np.abs(means) * 0.048
        initial_fail_num, initial_sample_each,IS_num=10, 100, 80
    cov_matrix = np.diag(variances)
    f = norm_dist(mu=means, var=cov_matrix)
    acs = MNIS(spice=spice,f_norm=f,mc_testbench=mc_testbench, feature_num=feature_num, num_rows=num_rows, num_cols =num_cols,
               IS_bound_num=1, IS_bound_on=True, g_cal_val=0.24, g_sam_val=0.01,
                       initial_fail_num=initial_fail_num, initial_sample_each=initial_fail_num, IS_num=IS_num, FOM_num=13)
    acs.start_estimate(max_num=100000)
'''
    g_cal_val=0.24, g_sam_val=0.01   :无用参数
    f_norm：原始分布
    提议分布的方差：0.004
    原始分布的方差：0.005

    '''
    