from size_optimization.borl_model.actor_net import ActorPolicyNet
from size_optimization.borl_model.critic_net import CriticNet
import torch.nn as nn
import torch
from torch.distributions import Categorical
from size_optimization.borl_model.buffer import Memory
import logging

# 设置日志记录器
logger = logging.getLogger("BORL.actor_critic")

# 强制使用CPU设备
device = torch.device("cpu")


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64, memory=Memory()):
        super(ActorCritic, self).__init__()

        # 记录模型创建信息
        logger.info(f"创建ActorCritic模型, 输入维度: {input_dim}, 动作维度: {action_dim}, 隐藏层维度: {hidden_dim}")

        # actor
        self.action_layer = ActorPolicyNet(input_dim, action_dim, hidden_dim=64).to(device)

        # critic
        self.value_layer = CriticNet(input_dim, hidden_dim=64).to(device)

        self.memory = memory

    def act(self, state_spec):
        try:
            # 确保输入在正确设备上
            state_spec = state_spec.to(device)

            # 记录输入维度
            logger.debug(f"状态维度: {state_spec.shape}")

            # 获取动作概率
            action_probs = self.action_layer(state_spec)
            logger.debug(f"动作概率维度: {action_probs.shape}")

            # 调整维度
            if action_probs.dim() == 3:  # [batch, action_dim, input_dim]
                action_probs = action_probs.squeeze(0).permute(1, 0)  # [input_dim, action_dim]

            logger.debug(f"调整后的动作概率维度: {action_probs.shape}")

            # 创建分布并采样
            dist = torch.distributions.Categorical(probs=action_probs)
            sampled = dist.sample()  # [input_dim]

            # 计算动作和日志概率
            action = sampled.unsqueeze(0)  # [1, input_dim]
            log_prob = dist.log_prob(sampled)
            log_prob_sum = torch.sum(log_prob, dim=0)

            # 存储到记忆中
            self.memory.states_spec.append(state_spec)
            self.memory.actions.append(action)
            self.memory.logprobs.append(log_prob_sum)

            logger.debug(f"采样的动作维度: {action.shape}")
            return action.detach()

        except Exception as e:
            logger.error(f"act()方法出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            # 返回零动作作为后备
            return torch.zeros_like(state_spec).to(device)

    def evaluate(self, state_spec, action):
        try:
            # 确保输入在正确设备上
            state_spec = state_spec.to(device)
            action = action.to(device)

            # 获取动作概率
            action_probs = self.action_layer(state_spec)

            # 转换维度
            probs = action_probs
            if action_probs.dim() == 3:  # [batch, action_dim, input_dim]
                probs = action_probs.permute(0, 2, 1)  # → [batch, input_dim, action_dim]

            # 创建分布
            dists = Categorical(probs=probs)

            # 计算日志概率
            action_logprobs = dists.log_prob(action)
            log_prob_sum = action_logprobs.sum(dim=1)

            # 计算熵
            dist_entropy = dists.entropy()
            dist_entropy_sum = dist_entropy.sum(dim=1)

            # 获取状态值
            state_value = self.value_layer(state_spec)

            return log_prob_sum, torch.squeeze(state_value), dist_entropy_sum

        except Exception as e:
            logger.error(f"evaluate()方法出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            # 返回默认值
            return (torch.zeros(state_spec.shape[0]).to(device),
                    torch.zeros(state_spec.shape[0]).to(device),
                    torch.zeros(state_spec.shape[0]).to(device))
