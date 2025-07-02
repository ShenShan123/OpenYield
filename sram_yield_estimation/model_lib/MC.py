import numpy as np
import time
import os
import torch
import random 
import sys
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
parent_dir_of_code1 = '/home/lixy/sram_yield_estimation/'
sys.path.append(parent_dir_of_code1) 
from tool.util import write_data2csv, seed_set
from tool.delete import delete_folder_content
from tool.Distribution.normal_v1 import norm_dist
from model_lib.spice import Spice
parent_dir_of_code2 = '/home/lixy/OpenYield-main'
sys.path.append(parent_dir_of_code2) 
from testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
import warnings
class MC():
    def __init__(self, f_norm, mc_testbench,  spice, means, feature_num, num_rows, num_cols, IS_bound_num, initial_num=10000, sample_num=10, FOM_use_num=10, seed=0, IS_bound_on=False):
        self.x = None  
        self.spice = spice  
        self.means = means
        self.feature_num = feature_num
        self.mc_testbench = mc_testbench
        self.f_norm = f_norm 
        self.initial_num, self.sample_num, self.FOM_use_num = initial_num, sample_num, FOM_use_num
        self.IS_bound_on = IS_bound_on
        self.IS_bound_num = IS_bound_num
        self.seed = seed
        seed_set(seed)
        self.num_rows = num_rows
        self.num_cols = num_cols

     # Save Monte Carlo simulation results to a CSV file
    def save_result(self, P_fail, FOM, num, used_time, seed):
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
        indic = self.spice.indicator(y)
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
    def _IS_bound(self, x, spice, IS_bound_num):
        low_bound_full = np.ones(x.shape) * spice.low_bounds * IS_bound_num
        high_bound_full = np.ones(x.shape) * spice.up_bounds * IS_bound_num
        x[x<low_bound_full] = low_bound_full[x<low_bound_full]
        x[x>high_bound_full] = high_bound_full[x>high_bound_full]
        return x
     # Perform Monte Carlo simulation until convergence conditions are met
    def start_estimate(self, max_num=100000):
        means = self.means
        feature_num = self.feature_num
        num_cols,num_rows = self.num_cols, self.num_rows
        initial_num, sample_num, FOM_use_num = self.initial_num, self.sample_num, self.FOM_use_num
        IS_bound_num, IS_bound_on = self.IS_bound_num, self.IS_bound_on

        now_time = time.time()
        self.y, P_fail_list, FOM_list, data_num_list = np.empty([0,1]), np.empty([0]), np.empty([0]), np.empty([0])
        
        if feature_num ==18:
            variances = np.abs(means) * 0.003
        elif feature_num ==108:
            variances = np.abs(means) * 0.01 
        elif feature_num == 1152:
            variances = np.abs(means) * 0.005
        cov_matrix = np.diag(variances)
        x = np.random.multivariate_normal(mean=means, cov=cov_matrix,
                                          size=initial_num)
        if IS_bound_on:
                x = self._IS_bound(x, self.spice, IS_bound_num)
        num_mc = x.shape[0]
        y, w_pavg = self.mc_testbench.run_mc_simulation(operation='write', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, vars=x)
        new_y = y.reshape(num_mc,1)
        folder_path = '/home/lixy/sim_read0'
        delete_folder_content(folder_path)
        self.spice.save_y_to_txt(new_y,'/home/lixy/yield_models/model/output_read.txt')
        self.y = np.vstack([self.y, new_y])
        P_fail, FOM = self._evaluate(self.y, P_fail_list, FOM_use_num)
        P_fail_list = np.hstack([P_fail_list, P_fail])
        self.save_result(P_fail=P_fail, FOM=FOM, num=self.y.shape[0], used_time=time.time() - now_time, seed=self.seed)
        print(f" # IS sample: {self.y.shape[0]}, fail_rate: {P_fail}, FOM: {FOM}")
        i=0
        while ((FOM>0.01) and (self.y.shape[0]<max_num)) or (i<40):

            x = self.f_norm.sample(n=sample_num)
            if IS_bound_on:
                x = self._IS_bound(x, self.spice, IS_bound_num)
            num_mc = x.shape[0]
            y, w_pavg = self.mc_testbench.run_mc_simulation(operation='write', target_row=self.num_rows-1, target_col=self.num_cols-1, mc_runs=num_mc, vars=x)
            new_y = y.reshape(num_mc,1)
            folder_path = '/home/lixy/sim_read0'
            delete_folder_content(folder_path)
            self.spice.save_y_to_txt(new_y,'/home/lixy/yield_models/model/output_read.txt')
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
    if feature_num == 18:
        variances = np.abs(means) * 0.003
    elif feature_num ==108:
        variances = np.abs(means) * 0.0095
    elif feature_num ==1152:
        variances = np.abs(means) * 0.048
    cov_matrix = np.diag(variances)
    f_norm = norm_dist(mu=means, var=cov_matrix)
    mc = MC(f_norm=f_norm, spice=spice,  mc_testbench=mc_testbench, feature_num=feature_num, num_rows=num_rows, num_cols=num_cols, 
            means=means, initial_num=100, sample_num=100, FOM_use_num=100, seed=0,IS_bound_on=True,IS_bound_num=1)
    mc.start_estimate(max_num=1000000000)
   
   
