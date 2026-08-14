"""使用冻结的LLM七项权重直接组合既有RewardFeatures。"""

from dataclasses import dataclass
import hashlib
import json
import math

from mappo.manual_reward import RewardBreakdown, combine_manual_reward
from mappo.reward_metadata import RewardLogMetadata


def raw_reward_spec_weights(reward_spec):
    """把严格Schema字段映射为八项内部权重，冲突项始终为零。"""
    positive = reward_spec.positive_weights
    penalty = reward_spec.penalty_weights
    if penalty["coordination_conflict_rate"] != 0.0:
        raise ValueError("coordination_conflict_rate必须永久固定为0")
    return {
        "sgl_progress": float(positive["sgl_progress"]),
        "relay_progress": float(positive["relay_progress"]),
        "completion": float(positive["completion_score"]),
        "balance": float(positive["balance_score"]),
        "expiration": float(penalty["expiration_loss"]),
        "invalid_action": float(penalty["invalid_action_rate"]),
        "coordination_conflict": 0.0,
        "relay_cost": float(penalty["relay_cost"]),
    }


def normalized_reward_spec_weights(reward_spec, manual_weights=None):
    """只对七项可调权重做L1归一化；保留旧参数仅为调用兼容。"""
    del manual_weights
    raw = raw_reward_spec_weights(reward_spec)
    l1 = sum(abs(value) for value in raw.values())
    if not math.isfinite(l1) or l1 <= 0.0:
        raise ValueError("七项LLM奖励权重的L1范数必须为正有限数")
    effective = {name: value / l1 for name, value in raw.items()}
    if effective["coordination_conflict"] != 0.0:
        raise RuntimeError("冲突权重归一化后必须仍为0")
    digest = hashlib.sha256(
        json.dumps(effective, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return effective, {
        "normalization_mode": "l1_unit_direct_llm_7d",
        "normalization_factor": 1.0 / l1,
        "raw_weight_l1": l1,
        "effective_weight_l1": sum(abs(value) for value in effective.values()),
        "raw_weights": raw,
        "effective_weights": effective,
        "effective_weights_sha256": digest,
        "conflict_fixed_zero": True,
    }


def reward_spec_weights(reward_spec, manual_weights=None):
    """返回训练实际生效的单位L1权重。"""
    return normalized_reward_spec_weights(reward_spec, manual_weights)[0]


@dataclass(frozen=True)
class LlmRewardBreakdown:
    """为直接LLM奖励补充冻结Spec ID，不引入基础奖励或塑形项。"""

    direct: RewardBreakdown
    reward_spec_id: str

    @property
    def total_reward(self):
        return self.direct.total_reward

    @property
    def features(self):
        return self.direct.features

    def component_values(self):
        return self.direct.component_values()


class LlmWeightReward:
    """直接使用 R_total = R_llm，不含R_base或alpha。"""

    environment_aware = True

    def __init__(self, feature_extractor, reward_spec, composition=None):
        if composition not in (None, {}):
            raise ValueError("直接LLM奖励不接受reward_composition")
        self.feature_extractor = feature_extractor
        self.reward_spec = reward_spec
        self.reward_spec_id = reward_spec.spec_id
        self.effective_weights, self.weight_metadata = normalized_reward_spec_weights(
            reward_spec
        )
        self.weight_metadata["reward_spec_id"] = self.reward_spec_id
        self.warning_count = 0
        self.last_breakdown = None

    def reset(self, environment):
        self.feature_extractor.reset(environment)
        self.warning_count = 0
        self.last_breakdown = None

    def compute(self, environment, info):
        """计算固定符号的七项直接LLM奖励。"""
        extracted = self.feature_extractor.compute(environment, info)
        direct = combine_manual_reward(
            extracted.features,
            self.effective_weights,
            self.feature_extractor.config["numerical"],
        )
        breakdown = LlmRewardBreakdown(direct, self.reward_spec_id)
        if abs(breakdown.total_reward) > self.feature_extractor.config["numerical"]["warning_abs_reward"]:
            self.warning_count += 1
        self.last_breakdown = breakdown
        return breakdown

    @property
    def log_metadata(self):
        return RewardLogMetadata(
            reward_method="llm_weight_reward",
            base_reward_name="llm_weight_reward",
            shaping_reward_name=None,
        )
