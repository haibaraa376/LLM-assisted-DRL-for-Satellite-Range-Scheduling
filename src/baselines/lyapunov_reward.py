"""实现多智能体PPO-Lya的Lyapunov势函数奖励塑形基线。"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from srs_env.models import TaskStatus
from mappo.reward_metadata import RewardLogMetadata


@dataclass(frozen=True)
class LyapunovFeatures:
    """描述积压、过期风险和卫星利用率不均衡。"""

    backlog: float
    expiration_risk: float
    utilization_imbalance: float

    def __post_init__(self):
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Lyapunov特征包含NaN或Inf")


@dataclass(frozen=True)
class LyapunovBreakdown:
    """记录相邻状态势函数、塑形项和基础人工奖励。"""

    previous_features: LyapunovFeatures
    current_features: LyapunovFeatures
    previous_potential: float
    current_potential: float
    shaping_reward: float
    manual_reward: float
    total_reward: float

    def component_values(self):
        """返回统一训练日志要求的标量分量。"""
        return {
            "manual_reward": self.manual_reward,
            "lyapunov_shaping": self.shaping_reward,
            "backlog": self.current_features.backlog,
            "expiration_risk": self.current_features.expiration_risk,
            "utilization_imbalance": (
                self.current_features.utilization_imbalance
            ),
            "previous_potential": self.previous_potential,
            "current_potential": self.current_potential,
            "total_reward": self.total_reward,
        }


def extract_lyapunov_features(environment, config):
    """只读环境状态，计算三项归一化Lyapunov特征。"""
    active = [
        state
        for state in environment.tasks.values()
        if state.status == TaskStatus.ACTIVE
    ]
    backlog_terms = []
    risk_terms = []
    for state in active:
        task = state.definition
        remaining = float(
            np.clip(
                (task.data_size_mbit - state.delivered_to_ground_mbit)
                / task.data_size_mbit,
                0.0,
                1.0,
            )
        )
        priority = float(np.clip(task.priority / 10.0, 0.0, 1.0))
        slack = max(task.expiration_time_s - environment.current_time_s, 0.0)
        urgency = 1.0 - float(
            np.clip(slack / task.survival_time_s, 0.0, 1.0)
        )
        backlog_terms.append(priority * remaining)
        risk_terms.append(
            priority
            * remaining
            * urgency ** float(config["urgency_power"])
        )
    backlog = float(np.mean(backlog_terms)) if backlog_terms else 0.0
    expiration_risk = float(np.mean(risk_terms)) if risk_terms else 0.0

    arrived_rows = [
        environment.task_index[task_id]
        for task_id, state in environment.tasks.items()
        if state.definition.arrival_time_s <= environment.current_time_s
        and state.status != TaskStatus.COMPLETED
    ]
    imbalances = []
    for row in arrived_rows:
        utilization = np.divide(
            environment.outgoing_seconds[row],
            environment.total_window_seconds,
            out=np.zeros_like(environment.outgoing_seconds[row]),
            where=environment.total_window_seconds > 0.0,
        )
        imbalances.append(float(np.std(utilization)))
    utilization_imbalance = (
        float(np.mean(imbalances)) if imbalances else 0.0
    )
    return LyapunovFeatures(backlog, expiration_risk, utilization_imbalance)


def lyapunov_potential(features, weights):
    """按配置权重组合Lyapunov势函数。"""
    value = sum(
        float(weights[name]) * float(getattr(features, name))
        for name in ("backlog", "expiration_risk", "utilization_imbalance")
    )
    if not math.isfinite(value):
        raise ValueError("Lyapunov势函数包含NaN或Inf")
    return float(value)


class PpoLyaReward:
    """在现有人工奖励上加入势函数差分塑形。"""

    environment_aware = True

    def __init__(self, manual_reward, config):
        self.manual_reward_model = manual_reward
        self.config = config
        self.previous_features = None
        self.episode_initial_potential = None
        self.warning_count = 0
        self.last_breakdown = None

    def reset(self, environment):
        """同步重置人工奖励事件快照和Lyapunov前态。"""
        self.manual_reward_model.reset(environment)
        self.previous_features = extract_lyapunov_features(
            environment,
            self.config,
        )
        self.episode_initial_potential = lyapunov_potential(
            self.previous_features,
            self.config["feature_weights"],
        )
        self.warning_count = 0
        self.last_breakdown = None

    def compute(self, environment, info):
        """使用真实下一状态势函数计算塑形，不对终止状态强制置零。"""
        if self.previous_features is None:
            raise RuntimeError("PPO-Lya奖励必须先reset")
        manual = self.manual_reward_model.compute(environment, info)
        current_features = extract_lyapunov_features(environment, self.config)
        weights = self.config["feature_weights"]
        previous_potential = lyapunov_potential(
            self.previous_features,
            weights,
        )
        current_potential = lyapunov_potential(current_features, weights)
        shaping = float(self.config["shaping_coefficient"]) * (
            previous_potential
            - float(self.config["gamma"]) * current_potential
        )
        if not math.isfinite(shaping):
            raise ValueError("Lyapunov塑形奖励包含NaN或Inf")
        if abs(shaping) > float(self.config["hard_failure_abs_shaping"]):
            raise RuntimeError("Lyapunov塑形奖励超过硬失败阈值")
        total = manual.total_reward + shaping
        if not math.isfinite(total):
            raise ValueError("PPO-Lya总奖励包含NaN或Inf")
        breakdown = LyapunovBreakdown(
            previous_features=self.previous_features,
            current_features=current_features,
            previous_potential=previous_potential,
            current_potential=current_potential,
            shaping_reward=shaping,
            manual_reward=manual.total_reward,
            total_reward=total,
        )
        self.previous_features = current_features
        self.warning_count = self.manual_reward_model.warning_count
        self.last_breakdown = breakdown
        return breakdown

    @property
    def episode_current_potential(self):
        """返回最近状态势函数，用于Episode日志。"""
        if self.previous_features is None:
            return None
        return lyapunov_potential(
            self.previous_features,
            self.config["feature_weights"],
        )

    @property
    def log_metadata(self):
        """声明训练奖励由人工基础奖励和Lyapunov塑形组成。"""
        return RewardLogMetadata(
            reward_method="manual_plus_lyapunov",
            base_reward_name="manual_reward",
            shaping_reward_name="lyapunov_shaping",
        )
