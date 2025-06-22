import numpy as np
import torch
import pandas as pd
import os
import sys
import time
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
parent_dir_of_code1 = '/home/lixy/sram_yield_estimation/'
sys.path.append(parent_dir_of_code1) 
from tool.util import write_data2csv, seed_set
from tool.delete import delete_folder_content
from tool.Distribution.normal_v1 import norm_dist
from tool.Distribution.gmm_v2 import mixture_gaussian
from tool.Distribution.guassian_distribution import Guassian_distribution
import torch.nn as nn
from scipy.stats.distributions import norm
from model_lib.spice import Spice
parent_dir_of_code2 = '/home/lixy/OpenYield-main'
sys.path.append(parent_dir_of_code2) 
from testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
import warnings
class AIS(nn.Module):
    #该方法是类的构造函数，用于初始化类的实例，f_norm：原始的蒙特卡罗分布，通常是一个高斯分布
    def __init__(self, spice, f_norm,mc_testbench,IS_bound_num, IS_bound_on, feature_num, num_rows, num_cols, g_cal_num=1, origin_sam_bound_num=1, initial_failed_data_num=100,
                       num_generate_each_norm=1, sample_num_each_sphere=50, max_gen_times=10,
                       FOM_num=10, spherical_presample_radius=0.1, spherical_presample_step=0.1,seed=0):
        '''
        g_cal_num：GMM（高斯混合模型）中每个高斯分布的IS（重要性采样）样本数量，默认值为1。
        origin_sam_bound_num：初始采样区域边界的缩放因子，默认值为1。
        initial_failed_data_num：初始失效数据的数量，默认值为100。
        num_generate_each_norm：GMM中每个高斯分布生成的IS样本数量，默认值为1。
        sample_num_each_sphere：在超球面上进行预采样的样本数量，默认值为50。
        spherical_presample_radius：球形预采样的初始半径，默认值为0.1。
        spherical_presample_step：球形预采样半径的步长，默认值为0.1。
        '''
        super(AIS, self).__init__()
        self.mc_testbench = mc_testbench
        self.spice = spice
        self.feature_num = feature_num
        self.max_gen_times = max_gen_times
        self.IS_bound_num = IS_bound_num
        self.IS_bound_on = IS_bound_on
        self.low_bounds = spice.low_bounds #保存上下边界
        self.up_bounds = spice.up_bounds
        self.spherical_presample_radius = spherical_presample_radius#保存球形预采样的初始半径和步长
        self.spherical_presample_step = spherical_presample_step
        self.x_samples = None
        self.y_samples = None
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.f_norm, self.g_cal_num, self.origin_sam_bound_num, self.initial_failed_data_num, \
        self.num_generate_each_norm, self.sample_num_each_sphere,  \
        self.FOM_num = f_norm, g_cal_num, origin_sam_bound_num, initial_failed_data_num, \
        num_generate_each_norm, sample_num_each_sphere, FOM_num

        self.seed = seed
        seed_set(seed)

    #保存结果
    def save_result(self, P_fail, FOM, num, used_time, seed):
            data_info_list = [[P_fail], [FOM], [num], [used_time]]
            write_data2csv(tgt_dir=os.path.join("./results/AIS"),  # 保存目的文件
                           tgt_name=f"AIS_read_{self.feature_num}.csv",  # 文件名:包含训练数据量, 模型名
                           head_info=('Pfail', 'FOM', 'num', 'used_time'),  # 表头
                           data_info=data_info_list)  # 信息

    #作用是根据输出的 y 值和 self.spice.threshold 阈值来判断样本是否失效
    def indicator_func(self,y):
        """
            I(X): if the corresponding  y of sample x is failed, return True, otherwise False.
        """
        return (y > self.spice.threshold) | (y < 0)

    def _IS_bound(self, x, spice, IS_bound_num):
        """
            constrain the area of x
        """
        low_bound_full = np.ones(x.shape) * spice.low_bounds * IS_bound_num
        high_bound_full = np.ones(x.shape) * spice.up_bounds * IS_bound_num
        x[x < low_bound_full] = low_bound_full[x < low_bound_full]
        x[x > high_bound_full] = high_bound_full[x > high_bound_full]
        return x


    def pre_sampling(self, initial_failed_data_num, sample_num_each_sphere, radius_interval=0.1, origin_sam_bound_num=1):
        """
            initial sampling
        """
        feat_num = self.feature_num  # feature number of x
        num_cols, num_rows =self.num_cols, self.num_rows
        captured_fail_data_num = 0
        iter_count = 0
        capture_any_fail_data_flag = False

        x_samples = None

        while captured_fail_data_num < initial_failed_data_num:

            # define sphere radius
            max_radius = self.up_bounds.min()
            radius = ((iter_count+1)*radius_interval) - (((iter_count+1)*radius_interval)//max_radius)*max_radius

            # capture failed samples
            # new_x = sample_sphere(num=sample_num_each_sphere,dim=feat_num,radius=radius)
            new_x = np.random.uniform(low=origin_sam_bound_num * self.spice.low_bounds, high=origin_sam_bound_num * self.spice.up_bounds, size=[sample_num_each_sphere,feat_num])
            num_mc = new_x.shape[0]
            y, w_pavg = self.mc_testbench.run_mc_simulation(operation='write', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, vars=new_x)
            new_y = y.reshape(num_mc,1)
            folder_path = '/home/lixy/sim1'
            delete_folder_content(folder_path)
            y_labels = self.indicator_func(new_y).reshape([-1])

            if y_labels.any() == True:
                failed_x = new_x[y_labels]

                if capture_any_fail_data_flag == False:
                    x_samples = failed_x
                else:
                    x_samples = np.vstack([x_samples,failed_x])

                capture_any_fail_data_flag = True
                captured_fail_data_num += failed_x.shape[0]

            iter_count += 1
            print(iter_count, radius, y_labels.any())

        x_samples = x_samples[0:initial_failed_data_num, :]  # discard excessive samples
        initial_sample_total_num = iter_count * sample_num_each_sphere

        return x_samples, initial_sample_total_num
    
    def calculate_weights(self, log_f_val, log_g_val, I_val):
        """
            calculate weights
        """
        weight_list = np.exp(log_f_val - log_g_val) * I_val
        return weight_list

    def _calculate_fail_rate(self, weight_sum_list, N):
        """
            calculate the overall P_f
        """
        fail_rate = sum(weight_sum_list) / N / len(weight_sum_list)
        return fail_rate

    def _calculate_FOM(self, fail_rate_list, FOM_num):
        length = len(fail_rate_list)
        assert length >= 1
        if length == 1 or np.mean(fail_rate_list[-FOM_num:]) == 0:
            return 1
        else:
            return np.std(fail_rate_list[-FOM_num:]) / np.mean(fail_rate_list[-FOM_num:])

    def _calculate_val(self, x, y, f_x, g_x, spice):
        """
            calculate log f(x), log g(x) and I(x)
            输入样本 → 计算原始分布概率 → 计算建议分布概率 → 判断失效状态 → 返回三元组
            
        """
        log_f_val = f_x.log_pdf(x).reshape([-1])
        log_g_val = g_x.log_pdf(x).reshape([-1])
        I_val = spice.indicator(y).reshape([-1])
        return log_f_val, log_g_val, I_val  
    def resampling(self, x, y, weight_list, data_num):
        """
            resamples according to weight_list
        """
        if sum(weight_list) != 0:
            weight_list = weight_list / sum(weight_list)
            weight_list = weight_list.astype(np.float64)

            # 均匀采样
            new_weight_list = np.round(weight_list * data_num).astype(int)
            sample_index = []
            for idx, i in enumerate(new_weight_list):
                for j in range(i):
                    sample_index.append(idx)
            sample_index = np.array(sample_index)
            x_samples = x[sample_index,:]
            y_samples = y[sample_index,:]

        else:
            x_samples = x
            y_samples = y

        return x_samples, y_samples


    def resampling_new(self, x, y, log_f, log_g, I, data_num):
        """
            resamples according to weight_list
        """

        sample_index = np.arange(I.shape[0])[I==1]
        x = x[sample_index]
        y = y[sample_index]
        log_f = log_f[sample_index]
        log_g = log_g[sample_index]

        weight_list = (log_f - log_g)
        sample_index = np.argsort(-weight_list)[:data_num]
        x_samples = x[sample_index,:]
        y_samples = y[sample_index,:]

        return x_samples, y_samples

    def _construct_mixture_norm(self, minnorm_point, betas, spice, g_var_num):
        """
            the labels is unused due to the fact that in the paper the ratio induced by the K-means can be divided out...
        """
        feat_num = spice.feature_num
        mean = minnorm_point
        pi = betas
        mix_model = mixture_gaussian(pi=pi, mu=mean, var_num=g_var_num)
        return mix_model

    #算法流程  （重要性采样 + 自适应 GMM）
    def start_estimate(self, max_num=1000000): 
        """
            call this function to start the yield estimation process,
            and the numerical results will be saved in "./results/ACS_case*.csv" automatically.

            :param f_norm: the origin MC distribution, usually a Gaussian distribution
            :param num_generate_each_norm: IS sample number of each Gausian distribution in the GMM g(x) GMM 由多个高斯分布混合而成，该参数确定每个组成高斯分布的采样数量
            :param origin_sam_bound_num: the zooming scale factor of bound of initial sampling area
            :param initial_failed_data_num: the number of initial failed sample
            :param sample_num_each_sphere: the number of samples each time during initial sampling
            :param FOM_num: the used number of latest fail rates to calculate FOM

            
        """
        f_norm, g_cal_num,  initial_failed_data_num, FOM_num =  self.f_norm, self.g_cal_num,  self.initial_failed_data_num, self.FOM_num
        IS_bound_num, IS_bound_on = self.IS_bound_num, self.IS_bound_on
        self.feature = self.feature_num
        num_cols ,num_rows = self.num_cols, self.num_rows
        sample_num_each_sphere = self.sample_num_each_sphere
        origin_sam_bound_num = self.origin_sam_bound_num
        num_generate_each_norm = self.num_generate_each_norm
        sample_num_each_sphere = self.sample_num_each_sphere
        # get initial failure data
        self.x_samples, initial_sample_total_num = self.pre_sampling(initial_failed_data_num, sample_num_each_sphere, origin_sam_bound_num=origin_sam_bound_num)

        num_mc = self.x_samples.shape[0]
        y, w_pavg = self.mc_testbench.run_mc_simulation(operation='write', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, vars=self.x_samples)
        self.y_samples= y.reshape(num_mc,1)
        folder_path = '/home/lixy/sim1'
        delete_folder_content(folder_path)
        fail_rate_list = []
        FOM_list = []
        data_num_list = []
        weight_sum_list = []
        FOM_metric = np.inf
        gen_time = 0
        gen_data_num = initial_failed_data_num  # number of data to generate, equal to initial failed data num
        now_time = time.time()

        sample_data_num = 0
        # start iteration, till fail_rate converges or simulation times reach the pre-defined maximum
        #norm_mixture = mixture_gaussian(pi=np.ones([self.component]) / self.component, mu=self.x_samples, var_num=3)
        # # 构建初始建议分布（高斯混合模型）
        #gx = Guassian_distribution(n_feature=self.feature, mean=self.mu, var_num=g_cal_num, spice=self.spice)
        #beta = np.full((1, initial_failed_data_num), 1/initial_failed_data_num)[0] #并初始化权重向量 beta，每个元素值为 1/initial_failed_data_num。

        while (sample_data_num<max_num) and (FOM_metric >= 0.05) or (gen_time<35):
            gen_time += 1 # IS time
            x_fail_num = self.x_samples.shape[0]
            # g(x) for IS sampling  混和高斯分布
            norm_mixture = self._construct_mixture_norm(minnorm_point=self.x_samples, betas=np.ones([x_fail_num])/x_fail_num,
                                                         spice=self.spice, g_var_num=g_cal_num)
            # generate new samples from g(x)
            IS_x_samples = norm_mixture.sample(n=num_generate_each_norm * initial_failed_data_num)

            '''
            如果你当前一共有 initial_failed_data_num 个失效点，它们将被当作 GMM 的组件中心（即每个点生成一个高斯子分布）。
            所以整个 GMM 就有 initial_failed_data_num 个高斯分布。
            '''
            if IS_bound_on:
                 IS_x_samples= self._IS_bound(IS_x_samples, self.spice, IS_bound_num)
            num_mc = IS_x_samples.shape[0]
            y, w_pavg = self.mc_testbench.run_mc_simulation(operation='write', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, vars=IS_x_samples)
            IS_y_samples = y.reshape(num_mc,1)
            folder_path = '/home/lixy/sim1'
            delete_folder_content(folder_path)
            # 计算 log f(x), log g(x)和I(x)
            log_f_val, log_g_val, I_val = self._calculate_val(IS_x_samples, IS_y_samples, f_norm, norm_mixture, self.spice)
            # calculate the weight of each x
            weight_list = self.calculate_weights(log_f_val, log_g_val, I_val)
            #print(weight_list)
            weight_sum_list.append(sum(weight_list))

            # calculate the overall P_f
            fail_rate = self._calculate_fail_rate(weight_sum_list, N=initial_failed_data_num) 
             #这里因为我们每次采样的数据量和我们第一次的数据相同，因此我们直接将N定义为我们第一次的失效样本的个数
            fail_rate_list.append(fail_rate)

            # calculate FOM
            FOM_metric = self._calculate_FOM(fail_rate_list, FOM_num)
            FOM_list.append(FOM_metric)
            #Metropolis-Hastings重采样（参数调整）
            #MH_spend_times += gx.AIS_MH_Resample(weight=weight_list, pi_x=self.f_norm)

            log_f_val_origin, log_g_val_origin, I_val_origin = self._calculate_val(self.x_samples, self.y_samples, f_norm, norm_mixture, self.spice)
            origin_weight_list = self.calculate_weights(log_f_val_origin, log_g_val_origin, I_val_origin)


            sample_data_num = initial_sample_total_num + gen_time * initial_failed_data_num  #总的失效个数为预采样的花费和总的重采样花费的总和

            self.x_samples, self.y_samples = self.resampling_new(x=np.vstack([self.x_samples, IS_x_samples]),
                                                                y=np.vstack([self.y_samples, IS_y_samples]),
                                                                log_f=np.concatenate([log_f_val_origin,log_f_val],axis=0),
                                                                log_g=np.concatenate([log_g_val_origin,log_g_val],axis=0),
                                                                I=np.concatenate([I_val_origin,I_val],axis=0),
                                                                data_num=gen_data_num
                                                                 )
            
            data_num_list.append(sample_data_num)

            # save metric & data sampled number
            #################
            used_time = time.time() - now_time
            ##################
            self.save_result(fail_rate, FOM_metric, num=sample_data_num, used_time=time.time() - now_time, seed=self.seed)
            print(f"[AIS] # already sample {sample_data_num} data, fail_rate: {fail_rate}, FOM: {FOM_metric}")

        return fail_rate, sample_data_num, FOM_metric, used_time

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
        variances = np.abs(means) * 0.004 
        g_cal_num=0.003
    elif feature_num ==108:
        variances = np.abs(means) * 0.0095
        g_cal_num=0.003
    elif feature_num ==1152:
        variances = np.abs(means) * 0.066
        g_cal_num=0.0066
    cov_matrix = np.diag(variances)
    f_norm = norm_dist(mu=means, var=cov_matrix)
    ais = AIS(spice=spice,mc_testbench=mc_testbench,feature_num=feature_num, num_rows=num_rows, num_cols= num_cols, 
              f_norm=f_norm, g_cal_num=g_cal_num, initial_failed_data_num=150,
                       num_generate_each_norm=1, sample_num_each_sphere=50, max_gen_times=1000,
                       FOM_num =11,  seed=7072, IS_bound_num=1, IS_bound_on=True)  #case4 参数
    fail_rate, sample_num, fom, used_time = ais.start_estimate(max_num=10000)
    '''
    f_norm :原始分布
    g_cal_num：混合高斯分布的方差
    initial_failed_data_num：初始失败样本的数量
    num_generate_each_norm：“每个 GMM 组件（即每个高斯分布）生成的样本数量
    sample_num_each_sphere:初始采样每次的样本数(均匀采样)
    FOM_num：FOM计算窗口
    不断更新最优的失效样本
    '''