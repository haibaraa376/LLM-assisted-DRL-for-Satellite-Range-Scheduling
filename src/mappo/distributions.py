"""实现映射到[0,1]且保留精确变换密度的高斯分布。"""

import torch


class SquashedNormal01:
    """将独立Normal变量通过 ``(tanh(u)+1)/2`` 映射到开区间(0,1)。

    直接clip会让边界概率质量与PPO使用的log probability不一致。本类保存
    变换前raw action，并在密度中扣除Tanh及缩放的Jacobian。
    """

    def __init__(self, mean, log_std, epsilon=1.0e-6):
        """创建分布；mean与log_std须同形且有限，epsilon用于数值稳定。"""
        if mean.shape != log_std.shape:
            raise ValueError("有界高斯的mean与log_std形状必须一致")
        if not torch.isfinite(mean).all() or not torch.isfinite(log_std).all():
            raise ValueError("有界高斯参数包含NaN或Inf")
        if epsilon <= 0.0:
            raise ValueError("Jacobian稳定项epsilon必须为正数")
        self.mean = mean
        self.log_std = log_std
        self.epsilon = epsilon
        self.normal = torch.distributions.Normal(mean, log_std.exp())

    @staticmethod
    def _squash(raw_action):
        """将任意实数张量平滑映射到(0,1)，不做直接裁剪。"""
        return (torch.tanh(raw_action) + 1.0) * 0.5

    def sample(self):
        """返回可重参数化的raw action和对应[0,1]有界动作。"""
        raw_action = self.normal.rsample()
        return raw_action, self._squash(raw_action)

    def deterministic(self):
        """使用Normal均值返回确定性raw action和有界动作。"""
        return self.mean, self._squash(self.mean)

    def log_prob(self, raw_action):
        """返回包含Tanh与二分之一缩放Jacobian的逐元素精确log密度。"""
        if raw_action.shape != self.mean.shape:
            raise ValueError("raw action形状与分布参数不一致")
        tanh_value = torch.tanh(raw_action)
        log_jacobian = torch.log(
            (1.0 - tanh_value.square()) * 0.5 + self.epsilon
        )
        result = self.normal.log_prob(raw_action) - log_jacobian
        if not torch.isfinite(result).all():
            raise RuntimeError("有界高斯log probability出现NaN或Inf")
        return result

    def base_entropy(self):
        """返回原始Normal熵，作为稳定近似而非变换后分布的解析熵。"""
        result = self.normal.entropy()
        if not torch.isfinite(result).all():
            raise RuntimeError("高斯熵出现NaN或Inf")
        return result
