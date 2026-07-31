"""使用冻结LLM权重组合现有八项奖励特征。"""

from dataclasses import dataclass

from mappo.manual_reward import RewardBreakdown, combine_manual_reward
from mappo.reward_metadata import RewardLogMetadata


def reward_spec_weights(reward_spec):
    """把严格Schema字段映射到人工奖励组合函数的固定键。"""
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
            reward_spec_weights(self.reward_spec),
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
