"""提供不修改环境的及时性增量诊断奖励。"""

import math


class TimelinessDeltaReward:
    """使用相邻环境状态的及时性增量构造共享诊断奖励。

    该标量由15颗卫星共享，只用于验证MAPPO训练链路，不是PPO、LLM-PPO或
    RAPPO正式实验奖励，也不混入负载均衡、完成、过期或拒绝项。
    """

    def __init__(self, scale=10.0, tolerance=1.0e-9):
        """设置正缩放因子和允许的浮点负误差容差。"""
        if scale <= 0.0 or tolerance < 0.0:
            raise ValueError("诊断奖励缩放必须为正，容差不能为负")
        self.scale = float(scale)
        self.tolerance = float(tolerance)
        self.previous_timeliness = None

    def reset(self, initial_info):
        """记录episode初始累计及时性，不返回奖励。"""
        value = float(initial_info["timeliness_raw"])
        if not math.isfinite(value):
            raise ValueError("初始及时性必须是有限数")
        self.previous_timeliness = value

    def compute(self, current_info):
        """返回 ``(T_t-T_{t-1})/scale``，并更新内部上一时刻值。"""
        if self.previous_timeliness is None:
            raise RuntimeError("计算诊断奖励前必须先调用reset")
        current = float(current_info["timeliness_raw"])
        if not math.isfinite(current):
            raise ValueError("当前及时性必须是有限数")
        delta = current - self.previous_timeliness
        if delta < -self.tolerance:
            raise RuntimeError("累计及时性出现明显下降，环境指标可能损坏")
        if delta < 0.0:
            delta = 0.0
        self.previous_timeliness = current
        return delta / self.scale
