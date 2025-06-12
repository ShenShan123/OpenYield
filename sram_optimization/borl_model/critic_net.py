import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# 设置日志记录器
logger = logging.getLogger("BORL.critic_net")

# 强制使用CPU设备
device = torch.device("cpu")


class CriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(CriticNet, self).__init__()

        # 记录创建信息
        logger.info(f"创建CriticNet, 输入维度: {input_dim}, 隐藏层维度: {hidden_dim}")

        # 定义网络层
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.fc3 = nn.Linear(hidden_dim, 1).to(device)  # 输出是单个值（状态值估计）

    def forward(self, x):
        try:
            # 确保输入在CPU上
            x = x.to(device)

            logger.debug(f"CriticNet 输入维度: {x.shape}")

            # 前向传播
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)  # 不应用激活函数

            logger.debug(f"CriticNet 输出维度: {x.shape}")

            return x  # 输出是标量值（预期奖励）

        except Exception as e:
            logger.error(f"CriticNet前向传播出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            # 返回零张量作为备选
            if x.dim() == 2:
                return torch.zeros((x.shape[0], 1)).to(device)
            else:
                return torch.zeros(1).to(device)


# 测试代码
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.DEBUG)

    # 示例用法:
    input_dim = 10  # 用户定义的电路维度
    critic = CriticNet(input_dim)
    sample_input = torch.randn(1, input_dim)  # 带有一个样本的示例批次
    value_estimate = critic(sample_input)

    print("评论家输出 (值估计):", value_estimate.item())  # 应该是单个标量
