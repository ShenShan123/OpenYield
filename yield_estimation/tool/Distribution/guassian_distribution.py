import numpy as np
import torch
import gpytorch

# 均值向量和协方差矩阵
from mpmath import mp

mean1 = torch.zeros(50,18)
covar1 = torch.eye(18).unsqueeze(0).expand(50, -1, -1)

mvn1 = gpytorch.distributions.MultivariateNormal(mean1, covar1)#创建了50个独立的18维正态分布
class Guassian_distribution(torch.nn.Module):
    def __init__(self, n_feature, mean, var_num, spice):
        super(Guassian_distribution, self).__init__()
        self.feature,self.var_num,self.mean,self.spice = n_feature,var_num,torch.from_numpy(mean),spice
        self.components = self.mean.shape[0]
        self.var = torch.eye(n_feature).unsqueeze(0).expand(self.components, -1, -1) * self.var_num
        #取前五个权重进行重新采样
        self.weight_num = 5
        self.repeat_times = [int(self.components*0.6), int(self.components*0.2),int(self.components*0.1), int(self.components*0.05), 0]
        # 计算最后一个元素的值
        self.repeat_times[4] = self.components - np.sum(self.repeat_times[:-1])
        # var和var_step2是一样的！
        self.var_step2 = var_num * torch.eye(n_feature).unsqueeze(0).expand(self.components, -1, -1)
        #建立高斯分布
        self.mvn = gpytorch.distributions.MultivariateNormal(self.mean, self.var)

    #将列表中小于某个阈值（1e-290）的元素替换为一个指定的小数值（1e-250）
    def replace_zeros(self, lst, small_num=1e-250):
        return [small_num if x < 1e-290 else x for x in lst]

    #根据给定的权重 beta 计算每个样本点 x 的对数概率值。
    def Ourmodel_estimate_log_prob(self, x, beta):
        """
            根据权重来计算每个样本点的概率值
        """
        beta_ACS = self.replace_zeros(beta)
        beta_ACS_log = np.log(beta_ACS)
        self.mvn = gpytorch.distributions.MultivariateNormal(self.mean, self.var)
        mp_exp_broad = np.frompyfunc(mp.exp, 1, 1)
        mp_log_broad = np.frompyfunc(mp.log, 1, 1)
        x = torch.tensor(x, dtype=torch.float64)
        x_sample = x.reshape(-1, 1, self.feature)
        x_pro_log = self.mvn.log_prob(x_sample)
        x_pro_log = x_pro_log + beta_ACS_log
        x_pro_log = x_pro_log.numpy()
        prob_constant = mp_exp_broad(x_pro_log)  # mp type
        prob = prob_constant.sum(axis=-1).reshape([-1,1])
        log_prob_ACS = mp_log_broad(prob).astype(float)
        return log_prob_ACS

    #根据给定的权重 beta 计算每个样本点 x 的对数概率值。
    def ACS_estimate_log_prob(self, x, beta_ACS):
        beta_ACS = self.replace_zeros(beta_ACS)
        beta_ACS_log = np.log(beta_ACS)
        self.mvn = gpytorch.distributions.MultivariateNormal(self.mean, self.var)
        mp_exp_broad = np.frompyfunc(mp.exp, 1, 1)
        mp_log_broad = np.frompyfunc(mp.log, 1, 1)
        x = torch.tensor(x, dtype=torch.float64)
        x_sample = x.reshape(-1, 1, self.feature)
        x_pro_log = self.mvn.log_prob(x_sample)
        x_pro_log = x_pro_log + beta_ACS_log
        x_pro_log = x_pro_log.numpy()
        prob_constant = mp_exp_broad(x_pro_log)  # mp type
        prob = prob_constant.sum(axis=-1).reshape([-1,1])
        log_prob_ACS = mp_log_broad(prob).astype(float)
        return log_prob_ACS

    #给定的权重 beta_ACS 计算输入样本 x 的对数概率值。
    def ACS_estimate_log_prob_for1093(self, x, beta_ACS):
        beta_ACS = self.replace_zeros(beta_ACS)
        beta_ACS_log = np.log(beta_ACS)
        self.mvn = torch.distributions.MultivariateNormal(loc=self.mean, covariance_matrix=self.var)
        mp_exp_broad = np.frompyfunc(mp.exp, 1, 1)
        mp_log_broad = np.frompyfunc(mp.log, 1, 1)
        #x = torch.tensor(x, dtype=torch.float64)
        #x_sample = x.reshape(-1, 1, self.feature)
        #print(x_sample.type())

        x = torch.from_numpy(x)
        if x.type() != self.mean.type():
            x = x.to(self.mean.type())

        x = x.reshape(-1, 1, self.feature)
        #print(x.shape)
        #x = x.to(torch.double)
        x_pro_log = self.mvn.log_prob(x)
        x_pro_log = x_pro_log + beta_ACS_log
        x_pro_log = x_pro_log.numpy()
        prob_constant = mp_exp_broad(x_pro_log)  # mp type
        prob = prob_constant.sum(axis=-1).reshape([-1,1])
        log_prob_ACS = mp_log_broad(prob).astype(float)
        return log_prob_ACS

    def HSCS_estimate_log_prob(self, x, beta_HSCS):
        beta_ACS = self.replace_zeros(beta_HSCS)
        beta_ACS_log = np.log(beta_ACS)
        self.mvn = gpytorch.distributions.MultivariateNormal(self.mean, self.var)
        mp_exp_broad = np.frompyfunc(mp.exp, 1, 1)
        mp_log_broad = np.frompyfunc(mp.log, 1, 1)
        x = torch.tensor(x, dtype=torch.float64)
        x_sample = x.reshape(-1, 1, self.feature)
        x_pro_log = self.mvn.log_prob(x_sample)
        x_pro_log = x_pro_log + beta_ACS_log
        x_pro_log = x_pro_log.numpy()
        prob_constant = mp_exp_broad(x_pro_log)  # mp type
        prob = prob_constant.sum(axis=-1).reshape([-1,1])
        log_prob_HSCS = mp_log_broad(prob).astype(float)
        return log_prob_HSCS

    def AIS_estimate_log_prob(self, x):
        """
            AIS预测函数是将样本点带入到每一个高斯函数里面去，并给他们相同的权重1/N，
            AIS传输进来的weight只需要是（1/个数）的beta就行了。
        """
        AIS_weight_average = torch.full((1, self.components), 1/self.components)
        AIS_weight_average_log = torch.log(AIS_weight_average)
        self.mvn = gpytorch.distributions.MultivariateNormal(self.mean, self.var)
        mp_exp_broad = np.frompyfunc(mp.exp, 1, 1)
        mp_log_broad = np.frompyfunc(mp.log, 1, 1)
        a,b = x.shape[0], x.shape[1]
        x = torch.Tensor(x)
        x = x.reshape(a, 1, b)
        x_pro_log = self.mvn.log_prob(x) + AIS_weight_average_log
        x_pro_log = x_pro_log.numpy()
        prob_constant = mp_exp_broad(x_pro_log)  # mp type
        prob = prob_constant.sum(axis=-1).reshape([-1,1])
        log_prob = mp_log_broad(prob).astype(float)
        return log_prob

    def AIS_estimate_log_prob_v2(self, x, weight):
        """
            老的这个AIS预测函数是将样本点带入到每一个高斯函数里面去，并给他们相同的权重1/N，
            但是效果并不好，我们给了AIS预测一个新的预测方式，就是根据他们的权重(归一完之后)，
            去进行计算概率。
            AIS传输进来的weight只需要是（1/个数）的beta就行了，当然也可以改。
        """
        weight /= np.sum(weight)
        weight_AIS = self.replace_zeros(weight)
        AIS_weight_average_log = np.log(weight_AIS)
        self.mvn = gpytorch.distributions.MultivariateNormal(self.mean, self.var)
        mp_exp_broad = np.frompyfunc(mp.exp, 1, 1)
        mp_log_broad = np.frompyfunc(mp.log, 1, 1)
        a,b = x.shape[0], x.shape[1]
        x = torch.Tensor(x)
        x = x.reshape(a, 1, b)
        x_pro_log = self.mvn.log_prob(x) + AIS_weight_average_log
        x_pro_log = x_pro_log.numpy()
        prob_constant = mp_exp_broad(x_pro_log)  # mp type
        prob = prob_constant.sum(axis=-1).reshape([-1,1])
        log_prob = mp_log_broad(prob).astype(float)
        return log_prob

    def AIS_estimate_log_prob_for1093(self, x, weight):
        weight /= np.sum(weight)
        weight = self.replace_zeros(weight)
        weight_log = np.log(weight)
        self.mvn = torch.distributions.MultivariateNormal(loc=self.mean, covariance_matrix=self.var)
        mp_exp_broad = np.frompyfunc(mp.exp, 1, 1)
        mp_log_broad = np.frompyfunc(mp.log, 1, 1)

        # print(x.type())
        x_ = torch.from_numpy(x)
        if x_.type() != self.mean.type():
            x_ = x_.to(self.mean.type())

        x_ = x_.reshape(-1, 1, self.feature)

        x_pro_log = self.mvn.log_prob(x_)
        x_pro_log = x_pro_log + weight_log
        x_pro_log = x_pro_log.numpy()
        prob_constant = mp_exp_broad(x_pro_log)  # mp type
        prob = prob_constant.sum(axis=-1).reshape([-1,1])
        log_prob_AIS = mp_log_broad(prob).astype(float)
        return log_prob_AIS

    #方法用于根据给定的权重 beta 对高斯分布进行重新采样，以生成指定数量的样本
    def Ourmodel_Sample_propagation(self, Sample_num, beta):
        """
            用于ACS的重新采样,这个采样方式是根据权重去取，如果这个高斯的权重越大，越有可能从高斯中进行采样
        """

        beta = torch.tensor(beta)
        counts = torch.distributions.multinomial.Multinomial(total_count=Sample_num,
                                                             probs=beta.squeeze()).sample()
        x = torch.empty(0, device=counts.device)
        # 这里注意 components要和高斯函数的个数对应
        for k in np.arange(self.mean.shape[0])[counts > 0]:
            d_k = gpytorch.distributions.MultivariateNormal(self.mean[k], self.var[k])
            x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])
            x = torch.cat((x, x_k), dim=0)
        return x.numpy()

    #该方法用于根据给定的权重 beta 对高斯分布进行重新采样，以生成指定数量的样本
    def ACS_Sample_propagation(self, Sample_num, beta, mean_ACS):
        '''
        Sample_num：需要生成的样本总数。
        beta：一个表示权重的数组，用于多项分布抽样，决定从每个高斯分布中采样的概率。
        mean_ACS：一个数组，包含每个高斯分布的均值。
        '''
        beta = torch.tensor(beta, dtype=torch.float64)
        mean_ACS = torch.tensor(mean_ACS)
        counts = torch.distributions.multinomial.Multinomial(total_count=Sample_num,
                                                             probs=beta.squeeze()).sample()
        x = torch.empty(0, device=counts.device)
        # 这里注意 components要和高斯函数的个数对应
        for k in np.arange(mean_ACS.shape[0])[counts > 0]:
            d_k = gpytorch.distributions.MultivariateNormal(mean_ACS[k], self.var[k])
            x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])
            x = torch.cat((x, x_k), dim=0)
        return x.numpy()

    #检查输入的数组 array 中的所有元素是否都为零。如果是，则将数组中的所有元素替换为 1；如果不是，则直接返回原数组。
    def check_allzero(self,array):
        if all(elem == 0 for elem in array):
            array = [1] * len(array)
        return array



    def ACS_Sample_propagation_for1093(self, Sample_num, beta, mean_ACS):
        beta = torch.tensor(beta)
        mean_ACS = torch.tensor(mean_ACS)
        if len(beta) == 1:
            x = torch.empty(0)
            counts = [Sample_num]
            sample_inter_flag = False
            while not sample_inter_flag:
                d_k = gpytorch.distributions.MultivariateNormal(mean_ACS, self.var[0])
                x_k = torch.stack([d_k.sample() for _ in range(int(counts[0]))])
                x_k_nan_mask = torch.isnan(x_k)
                # 输出检测结果
                if torch.any(x_k_nan_mask):
                    pass
                else:
                    sample_inter_flag = True
            x = torch.cat((x, x_k), dim=0)
        else:

            counts = torch.distributions.multinomial.Multinomial(total_count=Sample_num,
                                                             probs=beta.squeeze()).sample()
            x = torch.empty(0)
            # 这里注意 components要和高斯函数的个数对应
            for k in np.arange(mean_ACS.shape[0])[counts > 0]:
                sample_inter_flag = False
                while not sample_inter_flag:
                    d_k = gpytorch.distributions.MultivariateNormal(mean_ACS[k], self.var[k])
                    x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])
                    x_k_nan_mask = torch.isnan(x_k)
                    # 输出检测结果
                    if torch.any(x_k_nan_mask):
                        pass
                    else:
                        sample_inter_flag = True
                x = torch.cat((x, x_k), dim=0)
        return x.numpy()


    def ACS_Sample_propagation_new(self, index_dict, mean_ACS):
        """
        输入：
            index_dict：记录了每个聚类点对应的索引
            mean_ACS：记录了ACS的高斯分布的均值
        注意：
            我们的预采样个数和重新采样个数是相同的，我们这里做的只是更新了每个聚类点的一类中的数据
        """
        x = torch.empty(0)
        mean_ACS = torch.tensor(mean_ACS)
        # 最外层循环的次数和聚类点的个数相同
        for i in range(len(index_dict)):
            d_k = gpytorch.distributions.MultivariateNormal(mean_ACS[i], self.var[i])
            if len(index_dict[i]) != 0:
                x_k = torch.stack([d_k.sample() for _ in range(len(index_dict[i]))])
                x = torch.cat((x, x_k), dim=0)
            else:
                pass
        return x.numpy()

    def HSCS_Sample_propagation(self, Sample_num, beta, mean_HSCS):
        beta = torch.tensor(beta)
        mean_HSCS = torch.tensor(mean_HSCS)
        counts = torch.distributions.multinomial.Multinomial(total_count=Sample_num,
                                                             probs=beta.squeeze()).sample()
        x = torch.empty(0, device=counts.device)
        # Only iterate over components with non-zero counts
        for k in np.arange(mean_HSCS.shape[0])[counts > 0]:
            d_k = gpytorch.distributions.MultivariateNormal(mean_HSCS[k], self.var[k])
            x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])
            x = torch.cat((x, x_k), dim=0)
        return x.numpy()

    def AIS_Sample_propagation(self):
        """
            这个函数的均值为多少,就能够产生多少个样本。
            一次性产生mean.shape[0]个样本，每个样本服从对应的均值和方差组成的正态分布。
        """
        sample_inter_flag = False
        while not sample_inter_flag:
            self.mvn = gpytorch.distributions.MultivariateNormal(self.mean, self.var)
            x = self.mvn.rsample(sample_shape=torch.Size([1])).reshape(self.components, self.feature)
            x_nan_mask = torch.isnan(x)
        # 输出检测结果
            if torch.any(x_nan_mask):
                pass
            else:
                sample_inter_flag = True
        return x.numpy()






    def AIS_Sample_propagation_v2(self, weight, IS_num):
        """
            这个函数的均值为多少,就能够产生多少个样本。
            原本的函数是给所有的高斯函数都是一个1/N的概率
            新的函数设置了，根据权重来产生样本，越重越大，越有可能从此高斯函数里面去进行采样。
            因为设置为1/N的效果不如人意，因此设置为根据权重进行采样进行测试。QaQ
        """
        weight = torch.tensor(weight)
        counts = torch.distributions.multinomial.Multinomial(total_count=IS_num, probs=weight.squeeze()).sample()
        x = torch.empty(0, device=counts.device)
        # j 是 counts的索引，sample是对应的索引里面的元素
        # y记录从1到50每个时间出现的次数
        y = torch.cat([torch.full([int(sample)], j, device=counts.device) for j, sample in enumerate(counts)])
        # Only iterate over components with non-zero counts
        for k in np.arange(self.mean.shape[0])[counts > 0]:
            d_k = gpytorch.distributions.MultivariateNormal(self.mean[k], self.var[k])
            x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])
            x = torch.cat((x, x_k), dim=0)
        return x.numpy()

    def AIS_Sample_propagation_for1093(self, Sample_num, beta, mean_ACS):
        beta = torch.tensor(beta)
        mean_ACS = torch.tensor(mean_ACS)
        if len(beta) == 1:
            x = torch.empty(0)
            counts = [Sample_num]
            sample_inter_flag = False
            while not sample_inter_flag:
                d_k = gpytorch.distributions.MultivariateNormal(mean_ACS, self.var[0])
                x_k = torch.stack([d_k.sample() for _ in range(int(counts[0]))])
                x_k_nan_mask = torch.isnan(x_k)
                # 输出检测结果
                if torch.any(x_k_nan_mask):
                    pass
                else:
                    sample_inter_flag = True
            x = torch.cat((x, x_k), dim=0)
        else:
            counts = torch.distributions.multinomial.Multinomial(total_count=Sample_num,
                                                             probs=beta.squeeze()).sample()
            x = torch.empty(0)
            # 这里注意 components要和高斯函数的个数对应
            for k in np.arange(mean_ACS.shape[0])[counts > 0]:
                sample_inter_flag = False
                while not sample_inter_flag:
                    d_k = gpytorch.distributions.MultivariateNormal(mean_ACS[k], self.var[k])
                    x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])
                    x_k_nan_mask = torch.isnan(x_k)
                    # 输出检测结果
                    if torch.any(x_k_nan_mask):
                        pass
                    else:
                        sample_inter_flag = True
                x = torch.cat((x, x_k), dim=0)
        return x.numpy()




    def AIS_Resample(self, weight, IS_num):
        """"
            本函数以输入的权重为概率，凭借概率值进行生成对应的样本，概率越大 ，更有有可能取到对应的高斯分布进行采样
            weight: 每个高斯分布的所占权重
            IS_num: 要通过这个总的高斯分布产生的样本数
            pi_x: 失效样本服从的真实分布pi(x)
        """
        #n为50，抽取50个样本

        weight = torch.tensor(self.replace_zeros(weight), dtype=torch.float64)

        counts = torch.distributions.multinomial.Multinomial(total_count=IS_num, probs=weight.squeeze()).sample()
        x = torch.empty(0, device=counts.device)
        # j 是 counts的索引，sample是对应的索引里面的元素
        # y记录从1到50每个时间出现的次数
        y = torch.cat([torch.full([int(sample)], j, device=counts.device) for j, sample in enumerate(counts)])
        # Only iterate over components with non-zero counts
        for k in np.arange(self.mean.shape[0])[counts > 0]:
            sample_inter_flag = False
            while not sample_inter_flag:
                d_k = gpytorch.distributions.MultivariateNormal(self.mean[k], self.var_step2[k])
                x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])
                x_k_nan_mask = torch.isnan(x_k)
                # 输出检测结果
                if torch.any(x_k_nan_mask):
                    pass
                else:
                    sample_inter_flag = True
            x = torch.cat((x, x_k), dim=0)
        return x.numpy()


    def check_inf(self, array):
        """
            此函数 用于检测出inf的数字，并用e290代替掉这个数字
        """
        # 检测inf
        is_inf = np.isinf(array)
        # 将inf替换为特定的数
        replacement_value = 1e290
        b = np.where(is_inf, replacement_value, array)
        return b
    
    #实现了基于 Metropolis-Hastings（MH）算法的重采样过程，用于从高斯分布中采样并更新均值
    def AIS_MH_Resample(self, weight, pi_x):
        """
        输入值：
            AIS_mean: 要通过这个总的高斯分布的均值
            weight: 每个高斯分布的所占权重
            pi_x: 失效样本服从的真实分布pi(x)
        放回值：
            spend_times: 花费的查找次数
        """
        AIS_mean = self.mean
        spend_times = 0
        #
        import copy
        # 代替掉0，防止后面的产生1/0产生警告。
        weight_copy = self.replace_zeros(weight)

        # 随机抽取一个mu, mu的可能性和 1/weight成正比
        weight_ = torch.tensor(weight_copy, dtype=torch.float64)
        weight_ = 1 / weight_
        # 以weight为probs，随机取一个事件，[0 1 0 0 0...0 0]
        weight_ =  torch.tensor(self.check_inf(weight_), dtype=torch.float64)
        weight_norm = weight_/torch.sum(weight_)
        counts = torch.distributions.multinomial.Multinomial(total_count=1, probs=weight_norm.squeeze()).sample()
        # 选中被抽到的数值为1的索引
        index = np.where(counts == 1)
        # 产生一个新的mu，但是这个mu必须是错误样本。
        new_mu_flag = False
        while(new_mu_flag == False):
            new_mu = self.AIS_Resample(weight=weight_copy, IS_num=1)
            new_y = self.spice.run_simulation(new_mu)
            new_mu_flag = self.spice.indicator(new_y)
            spend_times += 1
        # 拿到新的mu，并计算接受率, 注意这里得到的都是log值
        pi_mu_new,pi_mu_last = pi_x.log_pdf(new_mu), pi_x.log_pdf(AIS_mean[index])
        gx_mu_new,gx_mu_last = self.AIS_estimate_log_prob(new_mu), self.AIS_estimate_log_prob(AIS_mean[index])
        accept_ratio = pi_mu_new+gx_mu_last-pi_mu_last-gx_mu_new
        if accept_ratio > 0: #此处等效于接受率大于1，则必有a>r
            AIS_mean[int(index[0][0])] = torch.tensor(new_mu)
        else:
            r = np.random.rand(1)
            ratio = np.exp(accept_ratio)
            if ratio > r:
                AIS_mean[int(index[0][0])] = torch.tensor(new_mu)
            else:
                pass
        self.mean = AIS_mean
        return spend_times

    def AIS_MH_Resample_for1093(self, weight, pi_x):
        """
        输入值：
            AIS_mean: 要通过这个总的高斯分布的均值
            weight: 每个高斯分布的所占权重
            pi_x: 失效样本服从的真实分布pi(x)
        放回值：
            spend_times: 花费的查找次数
        """
        AIS_mean = self.mean
        spend_times = 0
        #
        import copy
        wight_copy = copy.deepcopy(weight)
        # 代替掉0，防止后面的产生1/0产生警告。
        weight = self.replace_zeros(weight)
        # 随机抽取一个mu, mu的可能性和 1/weight成正比
        weight = torch.tensor(weight)
        weight = 1 / weight
        # 以weight为probs，随机取一个事件，[0 1 0 0 0...0 0]
        counts = torch.distributions.multinomial.Multinomial(total_count=1, probs=weight.squeeze()).sample()
        # 选中被抽到的数值为1的索引
        index = np.where(counts == 1)
        # 产生一个新的mu，但是这个mu必须是错误样本。
        new_mu_flag = False
        while(new_mu_flag == False):
            new_mu = self.AIS_Resample(weight=wight_copy, IS_num=1)
            new_y = self.spice(new_mu)
            new_mu_flag = self.spice.indicator(new_y)
            spend_times += 1
        # 拿到新的mu，并计算接受率, 注意这里得到的都是log值
        pi_mu_new,pi_mu_last = pi_x.log_pdf(new_mu), pi_x.log_pdf(AIS_mean[index])
        gx_mu_new,gx_mu_last = self.AIS_estimate_log_prob_for1093(new_mu, wight_copy), self.AIS_estimate_log_prob_for1093(AIS_mean[index].numpy(), wight_copy)
        accept_ratio = pi_mu_new+gx_mu_last-pi_mu_last-gx_mu_new
        if accept_ratio > 0: #此处等效于接受率大于1，则必有a>r
            AIS_mean[int(index[0][0])] = torch.tensor(new_mu)
            #print("fuck")
        else:
            r = np.random.rand(1)
            ratio = np.exp(accept_ratio)
            if ratio > r:
                AIS_mean[int(index[0][0])] = torch.tensor(new_mu)
                #print("fuck")
            else:
                pass
        self.mean = AIS_mean
        return spend_times


    #方法用于从高斯分布中进行采样
    def sample_step2_new(self, weight):
        #n为50，抽取50个样本
        sample_index = np.argsort(-weight)[:self.weight_num].reshape(-1)
        x = torch.empty(0)
        for index, num in enumerate(sample_index):
            d_k = gpytorch.distributions.MultivariateNormal(self.mean[num, :], self.var_step2[num, :])
            #x_k = torch.stack([d_k.sample(sample_shape=torch.Size([1])) for _ in range(int(repeat_times[index]))])
            x_k = d_k.sample(sample_shape=torch.Size([self.repeat_times[index]]))
            x = torch.cat((x, x_k), dim=0)
        return x.numpy()

    def sample_step2_new2(self, weight, gen_num):
        #n为50，抽取50个样本   抽样次数是根据传入的参数 gen_num 动态计算得出的。
        sample_index = np.argsort(-weight)[:self.weight_num].reshape(-1)
        x = torch.empty(0)
        self.repeat_times = [int(gen_num * 0.6), int(gen_num * 0.2), int(gen_num * 0.1),
                             int(gen_num * 0.05), 0]
        # 计算最后一个元素的值
        self.repeat_times[4] = gen_num - np.sum(self.repeat_times[:-1])
        for index, num in enumerate(sample_index):
            d_k = gpytorch.distributions.MultivariateNormal(self.mean[num, :], self.var_step2[num, :])
            #x_k = torch.stack([d_k.sample(sample_shape=torch.Size([1])) for _ in range(int(repeat_times[index]))])
            x_k = d_k.sample(sample_shape=torch.Size([self.repeat_times[index]]))
            x = torch.cat((x, x_k), dim=0)
        return x.numpy()

    def sample_step2_new3(self, weight):
        """
            抽取n个样本，选取权重最大的那个样本，复制t次，权重越大，t越大
        """
        #n为50，抽取50个样本
        weight = torch.tensor(weight)
        counts = torch.distributions.multinomial.Multinomial(total_count=self.components, probs=weight.squeeze()).sample()
        x = torch.tensor([])
        # j 是 counts的索引，sample是对应的索引里面的元素
        # y记录从1到50每个时间出现的次数
        count_list = [aa.tolist() for aa in counts]
        y = torch.cat([torch.full([int(sample)], j, device=counts.device) for j, sample in enumerate(counts)])
        # Only iterate over components with non-zero counts
        for k in np.arange(self.components)[counts > 0]:
            d_k = gpytorch.distributions.MultivariateNormal(self.mean[k], self.var_step2[k])
            x_k = d_k.sample().reshape(1, -1)
            for i in range(int(count_list[k])):
                x = torch.cat((x, x_k), dim=0)
        return x.numpy()


    def sample_step2_new4(self, weight):
        """
            抽取n个样本，选取权重最大的那个样本，复制t次，权重越大，t越大
        """
        #n为50，抽取50个样本
        weight = torch.tensor(weight)
        counts = torch.distributions.multinomial.Multinomial(total_count=self.components, probs=weight.squeeze()).sample()
        x = torch.tensor([])
        # j 是 counts的索引，sample是对应的索引里面的元素
        # y记录从1到50每个时间出现的次数
        count_list = [aa.tolist() for aa in counts]
        y = torch.cat([torch.full([int(sample)], j, device=counts.device) for j, sample in enumerate(counts)])
        # Only iterate over components with non-zero counts
        for k in np.arange(self.components)[counts > 0]:
            x_k = self.mean[k].reshape(1, -1)
            for i in range(int(count_list[k])):
                x = torch.cat((x, x_k), dim=0)
        return x.numpy()


if __name__ == "__main__":
    import sys
    # 获取当前脚本的父目录
    parent_dir_of_code = "D:/desktop/open_spice/Code/"
    # 将父目录添加到模块搜索路径
    sys.path.append(parent_dir_of_code)
    from Code.Exp.Data.SPICE_case2.SPICE_case2 import SPICE_Case2
    mean1 = torch.zeros(50, 18)
    covar1 = torch.eye(18).unsqueeze(0).expand(50, -1, -1)
    weight = torch.ones(1, 50)
    spice = SPICE_Case2()
    mvn = Guassian_distribution(n_feature=18, mean=mean1.numpy(), var_num=covar1,spice=spice)
    test_sample = torch.randn(50, 18)
    test_resample = mvn.sample_step2_new(weight) ####报错
    pro = mvn.estimate_log_prob(test_sample)
    print(test_resample)

