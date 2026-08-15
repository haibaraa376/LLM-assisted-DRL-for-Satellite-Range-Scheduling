"""ICLR 2025 ORSO论文Algorithm 2的D3RB项目级Episode分配实现。"""

from dataclasses import asdict, dataclass
import math
from typing import Optional


@dataclass
class D3RBLearnerState:
    """保存一个奖励候选的D3RB统计量和最近验证结果。"""

    candidate_id: str
    episodes_trained: int = 0
    n: int = 0
    u_hat: float = 0.0
    d_hat: float = 0.0
    phi: float = 0.0
    latest_task_reward: Optional[float] = None
    best_task_reward: Optional[float] = None
    latest_validation: Optional[dict] = None
    best_validation: Optional[dict] = None
    misspecification_count: int = 0

    @property
    def mean_task_reward(self):
        """返回经验任务效用均值；没有观测时不做除零计算。"""
        return self.u_hat / self.n if self.n else 0.0

    def to_dict(self):
        """导出JSON友好的最终状态。"""
        payload = asdict(self)
        payload["mean_task_reward"] = self.mean_task_reward
        return payload


class D3RBSelector:
    """D3RB选择器：只按照论文Algorithm 2的phi最小规则分配预算。"""

    def __init__(self, candidate_ids, d_min, delta, confidence_constant):
        if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("D3RB候选ID必须非空且互不重复")
        self.candidate_ids = tuple(sorted(candidate_ids))
        self.d_min = self._positive_finite(d_min, "d_min")
        self.delta = self._positive_finite(delta, "delta")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta必须位于(0,1)")
        self.confidence_constant = self._positive_finite(
            confidence_constant,
            "confidence_constant",
        )
        self.states = {
            candidate_id: D3RBLearnerState(
                candidate_id=candidate_id,
                d_hat=self.d_min,
                phi=self.d_min,
            )
            for candidate_id in self.candidate_ids
        }

    @staticmethod
    def _positive_finite(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("{0}必须是数值".format(name))
        if not math.isfinite(float(value)) or value <= 0.0:
            raise ValueError("{0}必须为正有限数".format(name))
        return float(value)

    def confidence(self, state):
        """实现论文式(8)/(9)中的 c*sqrt(n log(K) log(n/delta))/n。"""
        if state.n <= 0:
            raise ValueError("n=0时不得计算D3RB confidence")
        log_k = math.log(len(self.candidate_ids))
        log_ratio = math.log(state.n / self.delta)
        radicand = state.n * log_k * log_ratio
        if radicand < 0.0 or not math.isfinite(radicand):
            raise ValueError("D3RB confidence的sqrt参数非法")
        value = self.confidence_constant * math.sqrt(radicand) / state.n
        if not math.isfinite(value):
            raise ValueError("D3RB confidence不是有限数")
        return value

    def select(self, maximum_episodes):
        """选择eligible learner中phi最小者；并使用确定性并列规则。"""
        maximum = int(maximum_episodes)
        eligible = [
            state
            for state in self.states.values()
            if state.episodes_trained < maximum
        ]
        if not eligible:
            raise RuntimeError("所有ORSO候选均已达到Episode上限")
        for state in eligible:
            self._assert_finite_state(state)
        return min(
            eligible,
            key=lambda state: (
                state.phi,
                state.episodes_trained,
                state.candidate_id,
            ),
        ).candidate_id

    def update(self, candidate_id, task_reward, validation):
        """执行论文Algorithm 2的观测、misspecification test与phi更新。"""
        if candidate_id not in self.states:
            raise KeyError("未知D3RB候选：{0}".format(candidate_id))
        task_reward = float(task_reward)
        if not math.isfinite(task_reward):
            raise ValueError("D3RB task reward必须是有限数")
        if not isinstance(validation, dict):
            raise ValueError("D3RB必须接收完整validation结果")
        state = self.states[candidate_id]
        d_hat_before = state.d_hat
        phi_before = state.phi
        state.n += 1
        state.u_hat += task_reward
        state.episodes_trained += 1
        state.latest_task_reward = task_reward
        state.latest_validation = dict(validation)
        if state.best_task_reward is None or task_reward > state.best_task_reward:
            state.best_task_reward = task_reward
            state.best_validation = dict(validation)

        # ORSO论文Algorithm 2 / D3RB式(9)：使用更新前的d_hat进行失配检验。
        confidence_term = self.confidence(state)
        lhs = (
            state.mean_task_reward
            + d_hat_before * math.sqrt(state.n) / state.n
            + confidence_term
        )
        observed = [item for item in self.states.values() if item.n > 0]
        rhs = max(item.mean_task_reward - self.confidence(item) for item in observed)
        if not math.isfinite(lhs) or not math.isfinite(rhs):
            raise ValueError("D3RB misspecification test出现非有限值")
        triggered = lhs < rhs
        if triggered:
            state.d_hat = 2.0 * d_hat_before
            state.misspecification_count += 1
        # 论文Algorithm 2第19行：phi = d_hat * sqrt(n)。
        state.phi = state.d_hat * math.sqrt(state.n)
        self._assert_finite_state(state)
        return {
            "n_before": state.n - 1,
            "n_after": state.n,
            "u_hat_before": state.u_hat - task_reward,
            "u_hat_after": state.u_hat,
            "mean_task_reward": state.mean_task_reward,
            "d_hat_before": d_hat_before,
            "d_hat_after": state.d_hat,
            "phi_before": phi_before,
            "phi_after": state.phi,
            "confidence_term": confidence_term,
            "misspecification_triggered": triggered,
        }

    @staticmethod
    def _assert_finite_state(state):
        values = (state.d_hat, state.phi, state.u_hat, state.mean_task_reward)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("D3RB状态包含NaN或Inf：{0}".format(state.candidate_id))
