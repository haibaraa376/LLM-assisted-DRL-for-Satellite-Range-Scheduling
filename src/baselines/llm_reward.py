"""使用冻结LLM权重组合现有八项奖励特征。"""

from dataclasses import dataclass
import hashlib
import json
import math

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
    llm_reward: float
    alpha: float
    llm_components: dict

    @property
    def total_reward(self):
        return self.base.total_reward + self.alpha * self.llm_reward

    @property
    def features(self):
        return self.base.features

    def component_values(self):
        return {
            **self.base.component_values(),
            **{
                "llm_" + name: value
                for name, value in self.llm_components.items()
            },
            "base_reward": self.base.total_reward,
            "llm_reward": self.llm_reward,
            "llm_shaping_reward": self.alpha * self.llm_reward,
            "total_reward": self.total_reward,
        }


class LlmWeightReward:
    """固定核心任务底座，并把LLM八项权重作为有界附加塑形。"""

    environment_aware = True

    def __init__(self, feature_extractor, reward_spec, composition=None):
        self.feature_extractor = feature_extractor
        self.reward_spec = reward_spec
        self.reward_spec_id = reward_spec.spec_id
        composition = composition or {}
        self.alpha = float(composition.get("alpha", 1.0))
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("LLM塑形系数alpha必须为非负有限数")
        raw = raw_reward_spec_weights(reward_spec)
        raw_sum = sum(raw.values())
        if not math.isfinite(raw_sum) or raw_sum <= 0.0:
            raise ValueError("LLM八项原始权重之和必须为正有限数")
        self.effective_weights = {
            name: value / raw_sum for name, value in raw.items()
        }
        scales = composition.get("feature_scales") or {
            "sgl_progress": 1.0,
            "relay_progress": 1.0,
            "completion_score": 1.0,
            "balance_score": 1.0,
            "expiration_loss": 1.0,
            "invalid_action_rate": 1.0,
            "coordination_conflict_rate": 1.0,
            "relay_cost": 1.0,
        }
        if set(scales) != {
            "sgl_progress", "relay_progress", "completion_score", "balance_score",
            "expiration_loss", "invalid_action_rate", "coordination_conflict_rate", "relay_cost",
        }:
            raise ValueError("LLM特征缩放必须完整包含八项RewardFeatures")
        self.feature_scales = {name: float(value) for name, value in scales.items()}
        if not all(math.isfinite(value) and value > 0.0 for value in self.feature_scales.values()):
            raise ValueError("LLM特征缩放必须为正有限数")
        manual = feature_extractor.config["weights"]
        self.base_weights = {
            "sgl_progress": manual["sgl_progress"], "relay_progress": 0.0,
            "completion": manual["completion"], "balance": 0.0,
            "expiration": manual["expiration"], "invalid_action": 0.0,
            "coordination_conflict": 0.0, "relay_cost": 0.0,
        }
        normalization = {
            "normalization_mode": "l1_unit_base_plus_llm",
            "normalization_factor": 1.0 / raw_sum,
            "raw_weight_l1": raw_sum,
            "effective_weight_l1": 1.0,
            "raw_weights": raw,
            "effective_weights": self.effective_weights,
            "alpha": self.alpha,
            "base_features": ["sgl_progress", "completion_score", "expiration_loss"],
        }
        normalization["effective_weights_sha256"] = hashlib.sha256(
            json.dumps(self.effective_weights, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
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
        """计算 R_total = R_base + alpha * R_llm，二者职责相互独立。"""
        extracted = self.feature_extractor.compute(environment, info)
        base = combine_manual_reward(
            extracted.features,
            self.base_weights,
            self.feature_extractor.config["numerical"],
        )
        feature_values = extracted.features.__dict__
        normalized = {
            name: max(0.0, min(1.0, feature_values[name] / self.feature_scales[name]))
            for name in self.feature_scales
            if name != "balance_score"
        }
        normalized["balance_score"] = max(
            -1.0,
            min(1.0, feature_values["balance_score"] / self.feature_scales["balance_score"]),
        )
        # 分量单独保存，既能复核固定正负方向，也能计算候选的实际塑形贡献。
        llm_components = {
            "sgl_progress": (
                self.effective_weights["sgl_progress"]
                * normalized["sgl_progress"]
            ),
            "relay_progress": (
                self.effective_weights["relay_progress"]
                * normalized["relay_progress"]
            ),
            "completion": (
                self.effective_weights["completion"]
                * normalized["completion_score"]
            ),
            "balance": (
                self.effective_weights["balance"]
                * normalized["balance_score"]
            ),
            "expiration": (
                -self.effective_weights["expiration"]
                * normalized["expiration_loss"]
            ),
            "invalid_action": (
                -self.effective_weights["invalid_action"]
                * normalized["invalid_action_rate"]
            ),
            "coordination_conflict": (
                -self.effective_weights["coordination_conflict"]
                * normalized["coordination_conflict_rate"]
            ),
            "relay_cost": (
                -self.effective_weights["relay_cost"]
                * normalized["relay_cost"]
            ),
        }
        llm_reward = sum(llm_components.values())
        if not math.isfinite(llm_reward) or not -1.0 <= llm_reward <= 1.0:
            raise RuntimeError("归一化LLM塑形奖励必须位于[-1,1]")
        breakdown = LlmRewardBreakdown(
            base,
            self.reward_spec_id,
            llm_reward,
            self.alpha,
            llm_components,
        )
        if abs(breakdown.total_reward) > self.feature_extractor.config["numerical"][
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
