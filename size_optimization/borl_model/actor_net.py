import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# 设置日志记录器
logger = logging.getLogger("BORL.actor_net")

# 强制使用CPU设备
device = torch.device("cpu")


class ActorPolicyNet(nn.Module):
    def __init__(self, input_dim, action_dim=3, hidden_dim=64):
        super(ActorPolicyNet, self).__init__()

        # 记录创建信息
        logger.info(f"创建ActorPolicyNet, 输入维度: {input_dim}, 动作维度: {action_dim}, 隐藏层维度: {hidden_dim}")

        self.action_dim = action_dim
        self.input_dim = input_dim

        # 定义网络层
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.fc3 = nn.Linear(hidden_dim, action_dim * input_dim).to(device)  # 输出 action_dim * input_dim

    def forward(self, x):
        try:
            # 确保输入在CPU上
            x = x.to(device)

            logger.debug(f"ActorPolicyNet 输入维度: {x.shape}")

            # 前向传播
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)  # 不应用激活函数以允许策略灵活性

            # 检查输出维度
            logger.debug(f"FC3 输出维度: {x.shape}")

            # 计算预期的reshape形状
            batch_size = x.shape[0]
            expected_shape = (batch_size, self.action_dim, self.input_dim)
            logger.debug(f"期望的reshape维度: {expected_shape}")

            # 检查reshape是否可行
            if x.numel() != torch.prod(torch.tensor(expected_shape)):
                logger.warning(f"警告: 无法reshape为期望形状 {expected_shape}, 实际形状 {x.shape}")
                # 适应性调整
                actual_input_dim = x.numel() // (batch_size * self.action_dim)
                logger.warning(f"实际输入维度似乎是: {actual_input_dim}")
                x = x.view(batch_size, self.action_dim, actual_input_dim)
            else:
                # 正常reshape
                x = x.view(batch_size, self.action_dim, self.input_dim)

            logger.debug(f"Reshape后维度: {x.shape}")

            # 应用softmax获取动作概率
            x = F.softmax(x, dim=1)

            return x

        except Exception as e:
            logger.error(f"ActorPolicyNet前向传播出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            # 返回零张量作为备选
            return torch.zeros((x.shape[0], self.action_dim, self.input_dim)).to(device)


# 测试代码
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.DEBUG)

    # 示例用法:
    input_dim = 10  # 替换为您的电路维度
    model = ActorPolicyNet(input_dim)
    sample_input = torch.randn(1, input_dim)  # 带有一个样本的示例批次
    output = model(sample_input)

    print("输出形状:", output.shape)  # 预期: (1, 3, input_dim)
