import numpy as np
import sys
import time
import os
parent_dir_of_code1 = '/home/lixy/sram_yield_estimation/'
sys.path.append(parent_dir_of_code1) 
from tool.util import write_data2csv, seed_set
from tool.delete import delete_folder_content
from tool.Distribution.normal_v1 import norm_dist
parent_dir_of_code = '/home/lixy/Code'
class MNIS():
    threshold = 3.745662e-09
    _BOUND_FILE_MAP = {
        18: '/home/lixy/OpenYield-main/yield_estimation/bound_lib/model_18_bound.txt',
        108: '/home/lixy/OpenYield-main/yield_estimation/bound_lib/model_108_bound.txt',  
        576: '/home/lixy/OpenYield-main/yield_estimation/bound_lib/model_bound_576.txt',
        2304: '/home/lixy/OpenYield-main/yield_estimation/bound_lib/model_bound_2304.txt'
    }
    def __init__(self, mc_testbench, f_norm, feature_num, means, g_sam_val, initial_fail_num, initial_sample_each, IS_num, FOM_num, g_cal_val, IS_bound_num, seed=0, IS_bound_on=False):
        self.mc_testbench = mc_testbench
        self.feature_num =feature_num
        self.means = means
        self.f_norm, self.g_sam_val, self.initial_fail_num, self.initial_sample_each, self.IS_num, self.FOM_num = f_norm, g_sam_val, initial_fail_num, initial_sample_each, IS_num, FOM_num
        self.seed = seed
        self.IS_bound_num, self.IS_bound_on, self.g_cal_val = IS_bound_num, IS_bound_on, g_cal_val
        seed_set(seed)
        if self.feature_num not in self._BOUND_FILE_MAP:
            raise ValueError(f"不支持的特征数量: {self.feature_num}，仅支持 {list(self._BOUND_FILE_MAP.keys())}")
        bound_file_path = self._BOUND_FILE_MAP[self.feature_num]
        bounds = np.loadtxt(bound_file_path, dtype=np.float64)
        self.up_bounds = bounds[0, :]  
        self.low_bounds = bounds[1, :] 
        self.bound = np.vstack([self.low_bounds, self.up_bounds])
    def save_y_to_txt(self, y, file_path):
        if len(y.shape) != 2 or y.shape[1] != 1:
            raise ValueError("输入的数组不是 n*1 的形状。")
        with open(file_path, 'a') as f:
            np.savetxt(f, y, fmt='%e')
    def indicator(self, y):
        return (y > self.threshold) | (y < 0)
     # Save Monte Carlo simulation results to a CSV file

    def _calculate_val(self, x, y, f_x, g_x):
        log_f_val = f_x.log_pdf(x).reshape([-1])
        log_g_val = g_x.log_pdf(x).reshape([-1])
        I_val = self.indicator(y).reshape([-1])
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

        write_data2csv(tgt_dir=os.path.join("./results/MNIS"), 
                       tgt_name=f"MNIS_read_{self.feature_num}.csv",  
                       head_info=('Pfail', 'FOM', 'num', 'used_time'),  
                       data_info=data_info_list)  
        
    def _initial_sampling(self, initial_fail_num, sample_num_each_sphere):
        captured_fail_data_num = 0
        iter_count = 0
        capture_any_fail_data_flag = False
        feat_num = self.feature_num  # feature number of x

        while captured_fail_data_num < initial_fail_num:
            new_x = np.random.uniform(low=self.low_bounds, high=self.up_bounds,
                                      size=[sample_num_each_sphere, feat_num])
            num_mc = new_x.shape[0]
            #print(num_mc)
            y, w_pavg = self.mc_testbench.run_mc_simulation(operation='read', target_row=1, target_col=1, mc_runs=num_mc, vars=new_x)
            new_y = y.reshape(num_mc,1)
            folder_path = '/home/lixy/sim0'
            delete_folder_content(folder_path)
            # print(new_y)
            y_labels = self.indicator(new_y).reshape([-1])

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

    def _IS_bound(self, x, IS_bound_num):
        low_bound_full = np.ones(x.shape) * self.low_bounds * IS_bound_num
        high_bound_full = np.ones(x.shape) * self.up_bounds * IS_bound_num
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
            :param g_sam_val: the used variance of importance distribution during sampling
            :param FOM_num: the used number of latest fail rates to calculate FOM
        """

        f_norm, g_sam_val, initial_fail_num, initial_sample_each, IS_num, FOM_num = self.f_norm, self.g_sam_val, self.initial_fail_num, self.initial_sample_each, self.IS_num, self.FOM_num
        IS_bound_num, IS_bound_on, g_cal_val = self.IS_bound_num, self.IS_bound_on, self.g_cal_val
        feature_num = self.feature_num
        means = self.means
        time1 = time.time()
        self.x_fail, origin_sample_num = self._initial_sampling(initial_fail_num, initial_sample_each)
        num_mc = self.x_fail.shape[0]
        y, w_pavg = self.mc_testbench.run_mc_simulation(operation='read', target_row=1, target_col=1, mc_runs=num_mc, vars=self.x_fail)
        self.y_fail = y.reshape(num_mc,1)
        folder_path = '/home/lixy/sim0'
        delete_folder_content(folder_path)
        #print(self.y_fail) 
        centered_x_fail = self.x_fail - means
        distances = np.sqrt((centered_x_fail * centered_x_fail).sum(-1))
        min_index = distances.argmin()
        min_norm = self.x_fail[min_index, :]
        #min_norm = self.x_fail[abs(self.x_fail-means).sum(-1).argmin(), :]## L1 uses the L1 norm (Manhattan distance) to identify the failed sample closest to the origin.
        if feature_num ==18:
            variances = np.abs(means) * 0.004
        elif feature_num ==108:
            variances = np.abs(means) * 0.004
        elif feature_num ==576:
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
                x_IS = self._IS_bound(x_IS, IS_bound_num)
            num_mc = x_IS.shape[0]
            #print(num_mc)
            y, w_pavg = self.mc_testbench.run_mc_simulation(operation='read', target_row=1, target_col=1, mc_runs=num_mc, vars=x_IS)
            y_IS = y.reshape(num_mc,1) 
            folder_path = '/home/lixy/sim0'
            delete_folder_content(folder_path)
            #print(y_IS)
            # get log f(x), log g(x) and I(x)
            log_f_IS_val, log_g_IS_val, I_IS_val = self._calculate_val(x_IS, y_IS, f_norm, g_cal_norm)

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
