import torch.nn as nn
import torch
from sram_optimization.borl_model.actor_critic import ActorCritic
import logging

# 设置日志记录器
logger = logging.getLogger("BORL.ppo")

# 强制使用CPU设备
device = torch.device("cpu")


class PPO:
    def __init__(self, state_spec_dim, action_dim, hidden_dim, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        logger.info(f"初始化PPO, 状态维度: {state_spec_dim}, 动作维度: {action_dim}, 学习率: {lr}")

        # 创建策略网络
        self.policy = ActorCritic(state_spec_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        # 创建旧策略网络
        self.policy_old = ActorCritic(state_spec_dim, action_dim, hidden_dim).to(device)

        # 复制参数
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 确保所有参数都在CPU上
        for name, param in self.policy.named_parameters():
            param.data = param.data.to(device)
        for name, param in self.policy_old.named_parameters():
            param.data = param.data.to(device)

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        """更新策略网络"""
        try:
            # 检查记忆是否为空
            if len(memory.rewards) == 0:
                logger.warning("记忆为空，跳过更新")
                return

            # 蒙特卡洛估计状态奖励
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            # 标准化奖励
            rewards = torch.tensor(rewards, dtype=torch.float64).to(device)

            # 检查rewards维度
            if rewards.dim() == 0:
                rewards = rewards.unsqueeze(0)

            # 仅在有多个样本时标准化
            if rewards.size()[0] > 1:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

            logger.debug(f"奖励维度: {rewards.shape}, 均值: {rewards.mean()}, 标准差: {rewards.std()}")

            # 将列表转换为张量
            old_spec_states = torch.stack(memory.states_spec).to(device).detach()
            old_actions = torch.stack(memory.actions).to(device).detach()
            old_logprobs = torch.stack(memory.logprobs).to(device).detach()

            # 调整维度
            if old_spec_states.dim() > 2:
                old_spec_states = old_spec_states.squeeze(1)
            if old_actions.dim() > 2:
                old_actions = old_actions.squeeze(1)

            logger.debug(
                f"状态维度: {old_spec_states.shape}, 动作维度: {old_actions.shape}, 日志概率维度: {old_logprobs.shape}")

            # 优化策略K个回合
            for _ in range(self.K_epochs):
                # 评估旧动作和值
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_spec_states, old_actions)

                # 找到比率 (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # 找到替代损失
                advantages = rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

                # 梯度步骤
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

            # 将新权重复制到旧策略
            self.policy_old.load_state_dict(self.policy.state_dict())

            # 清除记忆
            memory.clear_memory()

            logger.debug("PPO更新成功")

        except Exception as e:
            logger.error(f"PPO更新出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            # 清除记忆以避免使用有问题的数据
            memory.clear_memory()
