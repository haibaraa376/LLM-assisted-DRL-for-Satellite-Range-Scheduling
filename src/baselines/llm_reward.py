"""使用冻结LLM权重组合现有八项奖励特征。"""

from dataclasses import dataclass
import hashlib
import json

from mappo.manual_reward import RewardBreakdown, combine_manual_reward
from mappo.reward_metadata import RewardLogMetadata


def raw_reward_spec_weights(reward_spec):
    """把严格Schema字段映射为未经尺度校正的八项原始权重。"""
    positive = reward_spec.positive_weights
    penalty = reward_spec.penalty_weights
    return {
        "sgl_progress": positive["sgl_progress"],
        "relay_progress": positive["relay_progress"],
        "completion": positive["completion_score"],
        "balance": positive["balance_score"],
        "expiration": penalty["expiration_loss"],
        "invalid_action": penalty["invalid_action_rate"],
        "coordination_conflict": penalty["coordination_conflict_rate"],
        "relay_cost": penalty["relay_cost"],
    }


def normalized_reward_spec_weights(reward_spec, manual_weights):
    """按L1范数把LLM权重缩放到人工奖励的总权重尺度。"""
    raw = raw_reward_spec_weights(reward_spec)
    raw_sum = float(sum(abs(value) for value in raw.values()))
    manual_sum = float(sum(abs(value) for value in manual_weights.values()))
    if raw_sum <= 0.0 or manual_sum <= 0.0:
        raise ValueError("奖励权重L1范数必须大于0")
    factor = manual_sum / raw_sum
    effective = {name: float(value) * factor for name, value in raw.items()}
    digest = hashlib.sha256(
        json.dumps(
            effective,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return effective, {
        "normalization_mode": "l1_match_manual",
        "normalization_factor": factor,
        "raw_weight_l1": raw_sum,
        "effective_weight_l1": manual_sum,
        "effective_weights_sha256": digest,
        "raw_weights": raw,
        "effective_weights": effective,
    }


def reward_spec_weights(reward_spec, manual_weights=None):
    """兼容读取权重；提供人工权重时返回正式有效权重。"""
    if manual_weights is None:
        return raw_reward_spec_weights(reward_spec)
    return normalized_reward_spec_weights(reward_spec, manual_weights)[0]


@dataclass(frozen=True)
class LlmRewardBreakdown:
    """在标准奖励分解外附带冻结Reward Spec标识。"""

    base: RewardBreakdown
    reward_spec_id: str

    @property
    def total_reward(self):
        return self.base.total_reward

    @property
    def features(self):
        return self.base.features

    def component_values(self):
        return self.base.component_values()


class LlmWeightReward:
    """复用ManualReward特征提取器，只替换本地固定组合权重。"""

    environment_aware = True

    def __init__(self, feature_extractor, reward_spec):
        self.feature_extractor = feature_extractor
        self.reward_spec = reward_spec
        self.reward_spec_id = reward_spec.spec_id
        self.effective_weights, normalization = normalized_reward_spec_weights(
            reward_spec,
            feature_extractor.config["weights"],
        )
        self.weight_metadata = {
            "reward_spec_id": reward_spec.spec_id,
            **normalization,
        }
        self.warning_count = 0
        self.last_breakdown = None

    def reset(self, environment):
        self.feature_extractor.reset(environment)
        self.warning_count = 0
        self.last_breakdown = None

    def compute(self, environment, info):
        """提取完全相同的特征，并应用经Schema验证的八项权重。"""
        extracted = self.feature_extractor.compute(environment, info)
        base = combine_manual_reward(
            extracted.features,
            self.effective_weights,
            self.feature_extractor.config["numerical"],
        )
        breakdown = LlmRewardBreakdown(base, self.reward_spec_id)
        if abs(base.total_reward) > self.feature_extractor.config["numerical"][
            "warning_abs_reward"
        ]:
            self.warning_count += 1
        self.last_breakdown = breakdown
        return breakdown

    @property
    def log_metadata(self):
        """声明冻结权重奖励本身是基础训练奖励，不含塑形。"""
        return RewardLogMetadata(
            reward_method="llm_weight_reward",
            base_reward_name="llm_weight_reward",
            shaping_reward_name=None,
        )
