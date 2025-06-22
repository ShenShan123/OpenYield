import os
import time
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
# 获取当前脚本的父目录
import sys
import numpy as np
parent_dir_of_code1 = '/home/lixy/sram_yield_estimation/tool'
sys.path.append(parent_dir_of_code1) 
from tool.util import write_data2csv, seed_set
from tool.delete import delete_folder_content
from tool.Distribution.normal_v1 import norm_dist
from tool.Distribution.gmm_v2 import mixture_gaussian
from tool.Distribution.guassian_distribution import Guassian_distribution
from tool.Distribution.multi_cone_cluster import cone_cluster
import torch.nn as nn
from scipy.stats.distributions import norm
from model_lib.spice import Spice
parent_dir_of_code2 = '/home/lixy/OpenYield-main'
sys.path.append(parent_dir_of_code2) 
from testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
import mpmath as mp
import warnings
class ACS():
    def __init__(self, spice,mc_testbench, feature_num, num_cols, num_rows, IS_bound_num, IS_bound_on, f_norm, g_cal_val, initial_fail_num, initial_sample_each, IS_num, FOM_num, seed):
        """
        初始化 ACS 类的实例
        :param spice: 一个用于电路仿真的对象，包含电路的参数和方法，用于生成样本对应的输出结果。
        :param f_norm: 原始蒙特卡罗分布的对象，通常为高斯分布，用于生成初始样本。
        :param g_cal_val: 重要性采样分布在计算良率时使用的方差值。
        :param initial_fail_num: 初始失败样本的数量，用于开始阶段获取一定数量的失败样本。
        :param initial_sample_each: 初始采样时每次采样的样本数量。
        :param IS_num: 每次重要性采样迭代轮次中的样本数量。
        :param FOM_num: 用于计算品质因数（FOM）时使用的最新失败率的数量。
        :param seed: 随机数种子，用于保证结果的可重复性。
        """
        self.spice = spice
        self.feature_num = feature_num
        self.seed = seed
        self.mc_testbench = mc_testbench
        self.f_norm, self.g_cal_val, self.initial_fail_num, self.initial_sample_each, self.IS_num, \
        self.FOM_num = f_norm, g_cal_val, initial_fail_num, initial_sample_each, IS_num, FOM_num
        self.IS_bound_num = IS_bound_num
        self.IS_bound_on = IS_bound_on
        self.num_cols = num_cols
        self.num_rows =num_rows
        seed_set(seed)


    #从给定的输入样本 x 和对应的输出结果 y 中筛选出失败的样本
    def _identify_fail(self, x, y, spice):
        """
            return the failed samples from given x,y only
        """
        label_fail = spice.indicator(y).reshape([-1])
        feat_num = spice.feature_num
        if label_fail.any():
            x_fail = x[label_fail, :]
            y_fail = y[label_fail, :].reshape([-1, 1])
        else:
            x_fail = np.empty([0,feat_num])
            y_fail = np.empty([0,1])
        return x_fail, y_fail
    def _IS_bound(self, x, spice, IS_bound_num):
        """
            constrain the area of x
        """
        low_bound_full = np.ones(x.shape) * spice.low_bounds * IS_bound_num
        high_bound_full = np.ones(x.shape) * spice.up_bounds * IS_bound_num
        x[x < low_bound_full] = low_bound_full[x < low_bound_full]
        x[x > high_bound_full] = high_bound_full[x > high_bound_full]
        return x

    #根据失败样本的权重构建一个高斯混合模型（Gaussian Mixture Model, GMM）
    def _construct_mixture_norm(self, weight, x_fail, g_val, labels):
        """
            construct the GMM according to weight of each failed sample x
            Note that: the labels is unused due to the fact that in the paper the ratio induced by the K-means can be divided out...
        """
        # get gmm model
        mix_model = mixture_gaussian(pi=weight, mu=x_fail, var_num=g_val)  #方法来自gmm_v2.py
        return mix_model

    #计算高斯混合模型（GMM）中每个高斯分布的权重
    def _calculate_weight(self, org_norm, x_fail):
        """
            get weights of each normal distribution of the GMM g(x)
        """
        # get gmm ratio (pi)
        mp_exp_broad = np.frompyfunc(mp.exp, 1, 1)#np.frompyfunc 用于将 mpmath 库中的 mp.exp 和 mp.log 函数转换为可以对 numpy 数组进行逐元素操作的函数。
        log_pdf_each = org_norm.log_pdf(x_fail)#调用 org_norm 对象的 log_pdf 方法，计算每个失败样本的对数概率密度。
        pdf_each = mp_exp_broad(log_pdf_each) #使用广播后的 mp_exp_broad 函数，将对数概率密度转换为概率密度。
        pdf_sum = pdf_each.sum()
        weight = (pdf_each / pdf_sum).astype(np.double)
        return weight

    #计算输入样本 x 在原始分布 f(x) 和重要性采样分布 g(x) 下的对数概率密度，以及样本对应的指示函数值。
    def _calculate_val(self, x, y, f_x, g_x, spice):
        """
            calculate log f(x), log g(x) and I(x)
        """
        log_f_val = f_x.log_pdf(x).reshape([-1])
        log_g_val = g_x.log_pdf(x).reshape([-1])
        I_val = spice.indicator(y).reshape([-1])
        return log_f_val, log_g_val, I_val

    #主要功能是根据当前轮次的失败率 fail_rate_this_round 和之前所有轮次的失败率列表 fail_rate_list 来计算截至当前轮次的总体失败率
    def _calculate_fail_rate(self, fail_rate_this_round, fail_rate_list):
        """
            calculate the overall P_f vias all IS samples  通过所有重要性采样（IS）样本计算总体失败率 P_f
        """
        fail_rate = (sum(fail_rate_list) + fail_rate_this_round) / (len(fail_rate_list) + 1)
        return fail_rate

    #计算单个重要性采样（IS）轮次中的失败率 P_f
    def _calculate_fail_rate_this_round(self, log_f_val, log_g_val, I_val):
        """
            calculate P_f in a single IS round
        """
        IS_num = log_f_val.shape[0] #获取 log_f_val 数组的长度，即重要性采样的样本数量。
        w_val = np.exp(log_f_val - log_g_val)

        w_val[(w_val == np.inf)] = 1e290

        fail_rate_this_round = (w_val * I_val).sum() / IS_num
        return fail_rate_this_round

    #其主要功能是计算品质因数。FOM 是一个用于衡量重要性采样过程中失败率估计稳定性的指标，FOM 值越高，说明失败率的估计越稳定。
    def _calculate_FOM(self, fail_rate_list, FOM_num):
        """
            calculate FOM
        """
        length = len(fail_rate_list)
        assert length >= 1
        if length == 1 or np.mean(fail_rate_list[-FOM_num:]) == 0:
            return 1
        else:
            return np.std(fail_rate_list[-FOM_num:]) / np.mean(fail_rate_list[-FOM_num:])
    
    #方法的主要功能是使用均匀分布进行预采样，以获取指定数量的失败样本
    def _initial_sampling(self, initial_fail_num, sample_num_each_sphere, spice):
        """
            pre sampling using uniform distribution
        """
        captured_fail_data_num = 0 #已捕获的失败样本数量，初始化为 0。
        iter_count = 0
        capture_any_fail_data_flag = False
        feat_num = self.feature_num  # feature number of x
        num_rows, num_cols = self.num_rows,self.num_cols
        while captured_fail_data_num < initial_fail_num:#只要已捕获的失败样本数量小于所需的初始失败样本数量，就继续采样。
            new_x = np.random.uniform(low=spice.low_bounds, high=spice.up_bounds, size=[sample_num_each_sphere, feat_num])
            num_mc = new_x.shape[0]
            y, w_pavg = self.mc_testbench.run_mc_simulation(operation='write', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, vars=new_x)
            new_y = y.reshape(num_mc,1)
            self.spice.save_y_to_txt(new_y,'/home/lixy/yield_models/model/output.txt')
            folder_path = '/home/lixy/sim2'
            delete_folder_content(folder_path)
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
        sample_total_num = (iter_count+1) * sample_num_each_sphere

        return x_samples, sample_total_num

    #将重要性采样过程中的关键结果保存到一个 CSV 文件中，方便后续分析和查看
    def _save_result(self, P_fail, FOM, num, used_time, seed):
        data_info_list = [[P_fail], [FOM], [num], [used_time]]

        write_data2csv(tgt_dir=os.path.join("./results/ACS"),  # 保存目的文件
                       tgt_name=f"ACS_write_{self.feature_num}.csv",  # 文件名:包含训练数据量, 模型名
                       head_info=('Pfail', 'FOM', 'num', 'used_time'),  # 表头
                       data_info=data_info_list)  # 信息

    #此方法用于启动良率估计流程，并且会自动将数值结果保存到 "./results/HSCS_case*.csv" 文件中
    def start_estimate(self, max_num=100000):
        """
            call this function to start the yield estimation process,
            and the numerical results will be saved in "./results/HSCS_case*.csv" automatically.
            :param f_norm: the origin MC distribution, usually a Gaussian distribution
            :param initial_sample_each: IS sample number of each Gausian distribution in the GMM g(x)
            :param IS_num: the number of IS samples in each iteration round
            :param initial_fail_num: the number of initial failed sample
            :param initial_sample_each: the number of samples each time during initial sampling
            :param max_gen_times: the max number of IS iterations
            :param g_cal_val: the used variance of importance sampling distribution during yield calculation
            :param FOM_num: the used number of latest fail rates to calculate FOM
        """

        f_norm, g_cal_val, initial_fail_num, initial_sample_each, IS_num, FOM_num = self.f_norm, self.g_cal_val, self.initial_fail_num, self.initial_sample_each, self.IS_num, self.FOM_num

        time1 = time.time()
        num_cols,num_rows = self.num_cols, self.num_rows
        IS_bound_num, IS_bound_on = self.IS_bound_num, self.IS_bound_on
        self.x_fail, origin_sample_num = self._initial_sampling(initial_fail_num, initial_sample_each, self.spice)
        num_mc = self.x_fail.shape[0]
        y, w_pavg = self.mc_testbench.run_mc_simulation(operation='write', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, vars=self.x_fail)
        self.y_fail = y.reshape(num_mc,1)
        self.spice.save_y_to_txt(self.y_fail,'/home/lixy/yield_models/model/output1.txt')
        folder_path = '/home/lixy/sim2'
        delete_folder_content(folder_path)
        k = round(np.sqrt(initial_fail_num))
        d = self.x_fail.shape[-1]
        classifier = cone_cluster(cluster_num=k, dim=self.x_fail.shape[-1]) # K-means classifier

        #初始化列表和计数器
        fail_rate_list = []
        FOM_list = []
        fail_rate_this_round_list = []
        iter_count = 0
        FOM=1

        while ((FOM>=0.05)and(iter_count*IS_num+origin_sample_num<max_num)) or (iter_count<20):
            self.label_fail = None # classifier.cluster(self.x_fail)  # not used

            # weights in the gmm of each normal distribution
            weight_list = self._calculate_weight(f_norm, self.x_fail)

            # the g'(x) used for sampling
            mix_gaussian_val = self._construct_mixture_norm(weight_list, self.x_fail, g_cal_val, self.label_fail)

            # IS sampling
            x_IS = mix_gaussian_val.sample(n=IS_num)
            if IS_bound_on:
                 x_IS= self._IS_bound(x_IS, self.spice, IS_bound_num)
            num_mc = x_IS.shape[0]
            y, w_pavg = self.mc_testbench.run_mc_simulation(operation='write', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, vars=x_IS)
            y_IS = y.reshape(num_mc,1)
            self.spice.save_y_to_txt(y_IS,'/home/lixy/yield_models/model/output1.txt')
            folder_path = '/home/lixy/sim2'
            delete_folder_content(folder_path)
            # collect only the failed samples from the IS samples
            new_x_fail, new_y_fail = self._identify_fail(x=x_IS, y=y_IS, spice=self.spice)
            self.x_fail = np.vstack([self.x_fail, new_x_fail])
            self.y_fail = np.vstack([self.y_fail, new_y_fail])

            # get log f(x), log g(x) and I(x) of IS samples
            log_f_IS_val, log_g_IS_val, I_IS_val = self._calculate_val(x_IS, y_IS, f_norm, mix_gaussian_val, self.spice)

            # the fail_rate calculated only using IS samples of this iteration round
            fail_rate_this_round = self._calculate_fail_rate_this_round(log_f_IS_val, log_g_IS_val, I_IS_val)

            # the real overall fail_rate after this iteration
            fail_rate = self._calculate_fail_rate(fail_rate_this_round, fail_rate_this_round_list)

            fail_rate_this_round_list.append(fail_rate_this_round)
            fail_rate_list.append(fail_rate)

            # calculate FOM
            FOM = self._calculate_FOM(fail_rate_list, FOM_num)
            FOM_list.append(FOM)

            iter_count+=1

            self._save_result(fail_rate, FOM, iter_count*IS_num+origin_sample_num, time.time()-time1, self.seed)

            print(f"num:{iter_count*IS_num+origin_sample_num}, pfail:{fail_rate}, FOM:{FOM}")



if __name__ == "__main__":
    vdd = 1.0
    pdk_path = '/home/lixy/OpenYield-main/model_lib/models.spice'#hfinally adjust
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
        variances = np.abs(means) * 0.003
        g_cal_val=0.001 
    elif feature_num ==108:
        variances = np.abs(means) * 0.0098
        g_cal_val=0.003
    elif feature_num ==1152:
        variances = np.abs(means) * 0.064
        g_cal_val=0.0064
    cov_matrix = np.diag(variances)
    f_norm = norm_dist(mu=means, var=cov_matrix)
    # print(spice.get_yield())
    acs = ACS(spice=spice, mc_testbench=mc_testbench, f_norm=f_norm, feature_num=feature_num, num_rows=num_rows, num_cols= num_cols, g_cal_val=g_cal_val,
              initial_fail_num=10, initial_sample_each=100, IS_num=100, FOM_num=10,seed=0,IS_bound_num=1, IS_bound_on=True)

    acs.start_estimate(max_num=100000)
    '''
     f_norm:原始分布  方差越小，pfail越小
     g_cal_val：提议分布的方差
     FOM_num：计算fom的窗口
    '''