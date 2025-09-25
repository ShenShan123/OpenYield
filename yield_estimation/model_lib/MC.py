import numpy as np
import time
import os
import torch
import random 
import sys
parent_dir_of_code1 = '/home/lixy/OpenYield-main/yield_estimation'
sys.path.append(parent_dir_of_code1) 
from tool.util import write_data2csv, seed_set
from tool.delete import delete_folder_content
from tool.Distribution.normal_v1 import norm_dist
class MC():
    threshold = 3.745662e-09
    _BOUND_FILE_MAP = {
        18: '/home/lixy/OpenYield-main/yield_estimation/bound_lib/model_18_bound.txt',
        108: '/home/lixy/OpenYield-main/yield_estimation/bound_lib/model_108_bound.txt',  
        576: '/home/lixy/OpenYield-main/yield_estimation/bound_lib/model_bound_576.txt',
        2304: '/home/lixy/OpenYield-main/yield_estimation/bound_lib/model_bound_2304.txt'
    }
    def __init__(self, f_norm, mc_testbench, means, feature_num, IS_bound_num, initial_num=10000, sample_num=10, FOM_use_num=10, seed=0, IS_bound_on=False):
        self.x = None   
        self.means = means
        self.feature_num = feature_num
        self.mc_testbench = mc_testbench
        self.f_norm = f_norm 
        self.initial_num, self.sample_num, self.FOM_use_num = initial_num, sample_num, FOM_use_num
        self.IS_bound_on = IS_bound_on
        self.IS_bound_num = IS_bound_num
        self.seed = seed
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
    def save_result(self, P_fail, FOM, num, used_time):
        data_info_list = [[P_fail], [FOM], [num], [used_time]]

        write_data2csv(tgt_dir=os.path.join("./results/MC"),  
                       tgt_name=f"MC_read_{self.feature_num}.csv",  
                       head_info=('Pfail', 'FOM', 'num', 'used_time'),  
                       data_info=data_info_list)  

     # Private method to extract new samples from the output dataset
    def _get_new_y(self, sample_num, initial_num, i, test_y):   

        if initial_num+(i+1)*sample_num > test_y.shape[0]:
            new_y = test_y[initial_num+i*sample_num:, :]
        else:
            new_y = test_y[initial_num+i*sample_num:initial_num+(i+1)*sample_num, :]
        return new_y  

    # Compute failure probability and Figure of Merit (FOM) based on output data
    def _evaluate(self, y, P_fail_list, FOM_use_num=10): 
        indic = self.indicator(y)
        P_fail = indic.sum() / indic.shape[0]
        P_fail_list = np.hstack([P_fail_list, P_fail])
        # leng = len(P_fail_list)

        if P_fail_list[-FOM_use_num:].mean() == 0:
            FOM = 1
        elif P_fail_list.shape[0] == 1:
            FOM = 1
        else:
            FOM = P_fail_list[-FOM_use_num:].std() / P_fail_list[-FOM_use_num:].mean()
        return P_fail, FOM

  # Extract initial samples from the test_y dataset
    def _get_initial_sample(self, initial_num, test_y):
        return test_y[0:initial_num,:]

    # Randomly shuffle the input test_x data
    def rearrage_x(self, test_x):
        # np.random.shuffle(test_x)
        test_x = np.random.permutation(test_x)
        return test_x
    def _IS_bound(self, x, IS_bound_num):
        low_bound_full = np.ones(x.shape) * self.low_bounds * IS_bound_num
        high_bound_full = np.ones(x.shape) * self.up_bounds * IS_bound_num
        x[x<low_bound_full] = low_bound_full[x<low_bound_full]
        x[x>high_bound_full] = high_bound_full[x>high_bound_full]
        return x
     # Perform Monte Carlo simulation until convergence conditions are met
    def start_estimate(self, max_num=100000):
        means = self.means
        feature_num = self.feature_num
        initial_num, sample_num, FOM_use_num = self.initial_num, self.sample_num, self.FOM_use_num
        IS_bound_num, IS_bound_on = self.IS_bound_num, self.IS_bound_on

        now_time = time.time()
        self.y, P_fail_list, FOM_list, data_num_list = np.empty([0,1]), np.empty([0]), np.empty([0]), np.empty([0])
        
        if feature_num ==18:
            variances = np.abs(means) * 0.003
        elif feature_num ==108:
            variances = np.abs(means) * 0.01 
        elif feature_num == 576:
            variances = np.abs(means) * 0.005
        cov_matrix = np.diag(variances)
        x = np.random.multivariate_normal(mean=means, cov=cov_matrix,
                                          size=initial_num)
        if IS_bound_on:
                x = self._IS_bound(x, IS_bound_num)
        num_mc = x.shape[0]
        y, w_pavg = self.mc_testbench.run_mc_simulation(operation='read', target_row=1, target_col=1, mc_runs=num_mc, vars=x)
        new_y = y.reshape(num_mc,1)
        folder_path = '/home/lixy/OpenYield-main/sim2'
        delete_folder_content(folder_path)
        self.save_y_to_txt(new_y,'/home/lixy/yield_models/model/output_read.txt')
        self.y = np.vstack([self.y, new_y])
        P_fail, FOM = self._evaluate(self.y, P_fail_list, FOM_use_num)
        P_fail_list = np.hstack([P_fail_list, P_fail])
        self.save_result(P_fail=P_fail, FOM=FOM, num=self.y.shape[0], used_time=time.time() - now_time, seed=self.seed)
        print(f" # IS sample: {self.y.shape[0]}, fail_rate: {P_fail}, FOM: {FOM}")
        i=0
        while ((FOM>0.01) and (self.y.shape[0]<max_num)) or (i<40):

            x = self.f_norm.sample(n=sample_num)
            if IS_bound_on:
                x = self._IS_bound(x, IS_bound_num)
            num_mc = x.shape[0]
            print(x.shape())
            y, w_pavg = self.mc_testbench.run_mc_simulation(operation='read', target_row=1, target_col=1, mc_runs=num_mc, vars=x)
            new_y = y.reshape(num_mc,1)
            folder_path = '/home/lixy/OpenYield-main/sim2'
            delete_folder_content(folder_path)
            self.save_y_to_txt(new_y,'/home/lixy/yield_models/model/output_read.txt')
            self.y = np.vstack([self.y, new_y])

            P_fail, FOM = self._evaluate(self.y, P_fail_list, FOM_use_num)
            P_fail_list = np.hstack([P_fail_list, P_fail])

            self.save_result(P_fail=P_fail, FOM=FOM, num=self.y.shape[0], used_time=time.time() - now_time, seed=self.seed)
            print(f"[MC] # IS sample: {self.y.shape[0]}, fail_rate: {P_fail}, FOM: {FOM}")
            i += 1

        return P_fail_list, data_num_list
def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
