"""定义LLM只能输出的严格奖励权重JSON Schema。"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Optional


POSITIVE_WEIGHT_NAMES = (
    "sgl_progress",
    "relay_progress",
    "completion_score",
    "balance_score",
)
PENALTY_WEIGHT_NAMES = (
    "expiration_loss",
    "invalid_action_rate",
    "coordination_conflict_rate",
    "relay_cost",
)
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "reward_name",
    "positive_weights",
    "penalty_weights",
    "rationale",
    "expected_behavior",
    "risk_notes",
    "parent_candidate_id",
}
_UNSAFE_TEXT = re.compile(
    r"```|\bimport\b|\bfrom\s+\w+\s+import\b|\bdef\s+\w+|"
    r"[A-Za-z]:[\\/]|(?:^|\s)/(?:home|tmp|usr|var|Users)/",
    re.IGNORECASE,
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


@dataclass(frozen=True)
class LlmRewardSpec:
    """经过安全校验、可冻结且不可执行的八项奖励权重。"""

    schema_version: str
    reward_name: str
    positive_weights: dict
    penalty_weights: dict
    rationale: str
    expected_behavior: list
    risk_notes: list
    parent_candidate_id: Optional[str]

    @classmethod
    def from_json(cls, content, minimum=0.0, maximum=3.0):
        """解析严格JSON；拒绝NaN、重复/未知字段和尾随文本。"""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM奖励响应为空")

        def reject_constant(value):
            raise ValueError("LLM奖励JSON不得包含{0}".format(value))

        try:
            data = json.loads(content, parse_constant=reject_constant)
        except json.JSONDecodeError as error:
            raise ValueError("LLM奖励响应不是完整合法JSON") from error
        return cls.from_dict(data, minimum, maximum)

    @classmethod
    def from_dict(cls, data, minimum=0.0, maximum=3.0):
        """从字典构造并执行全部结构、数值和文本安全检查。"""
        if not isinstance(data, dict) or set(data) != _TOP_LEVEL_FIELDS:
            raise ValueError("LLM奖励JSON字段缺失或包含未知字段")
        if data["schema_version"] != "2.0":
            raise ValueError("schema_version必须为2.0")
        if not isinstance(data["reward_name"], str) or not _SAFE_NAME.fullmatch(
            data["reward_name"]
        ):
            raise ValueError("reward_name必须是安全的非空标识符")
        cls._validate_weights(
            data["positive_weights"],
            POSITIVE_WEIGHT_NAMES,
            minimum,
            maximum,
        )
        cls._validate_weights(
            data["penalty_weights"],
            PENALTY_WEIGHT_NAMES,
            minimum,
            maximum,
        )
        if data["penalty_weights"]["coordination_conflict_rate"] != 0.0:
            raise ValueError("coordination_conflict_rate必须永久固定为0")
        adjustable = list(data["positive_weights"].values()) + [
            value
            for name, value in data["penalty_weights"].items()
            if name != "coordination_conflict_rate"
        ]
        if sum(float(value) for value in adjustable) <= 0.0:
            raise ValueError("七项可调权重不能全部为0")
        cls._validate_text(data["rationale"], "rationale", required=True)
        cls._validate_text_list(
            data["expected_behavior"],
            "expected_behavior",
            required=True,
        )
        cls._validate_text_list(data["risk_notes"], "risk_notes", required=False)
        parent = data["parent_candidate_id"]
        if parent is not None and (
            not isinstance(parent, str) or not _SAFE_NAME.fullmatch(parent)
        ):
            raise ValueError("parent_candidate_id格式不安全")
        return cls(**data)

    @staticmethod
    def _validate_weights(values, expected_names, minimum, maximum):
        if not isinstance(values, dict) or set(values) != set(expected_names):
            raise ValueError("奖励权重字段必须完整且不得扩展")
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("奖励权重{0}必须是数值".format(name))
            if not math.isfinite(float(value)):
                raise ValueError("奖励权重{0}不是有限数".format(name))
            if not minimum <= float(value) <= maximum:
                raise ValueError("奖励权重{0}越界".format(name))

    @staticmethod
    def _validate_text(value, name, required):
        if not isinstance(value, str) or (required and not value.strip()):
            raise ValueError("{0}必须是非空文本".format(name))
        if _UNSAFE_TEXT.search(value):
            raise ValueError("{0}包含代码或路径".format(name))

    @classmethod
    def _validate_text_list(cls, value, name, required):
        if not isinstance(value, list) or (required and not value):
            raise ValueError("{0}必须是非空文本列表".format(name))
        for item in value:
            cls._validate_text(item, name, required=True)

    def to_dict(self):
        """返回不含运行时对象的JSON字典。"""
        return asdict(self)

    @property
    def spec_id(self):
        """用规范JSON计算稳定候选标识。"""
        canonical = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, path):
        """冻结为UTF-8 JSON，不包含密钥或可执行内容。"""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path, minimum=0.0, maximum=3.0):
        """读取并重新执行严格校验，防止冻结文件被手工污染。"""
        content = Path(path).read_text(encoding="utf-8")
        return cls.from_json(content, minimum, maximum)


def default_mock_specs():
    """返回两个方向合理且权重不同的Mock候选。"""
    base = {
        "schema_version": "2.0",
        "reward_name": "deadline_delivery_mock_v1",
        "positive_weights": {
            "sgl_progress": 1.20,
            "relay_progress": 0.10,
            "completion_score": 0.80,
            "balance_score": 0.03,
        },
        "penalty_weights": {
            "expiration_loss": 0.90,
            "invalid_action_rate": 0.08,
            "coordination_conflict_rate": 0.0,
            "relay_cost": 0.04,
        },
        "rationale": "奖励最终下传与任务完成，抑制过期、冲突和循环中继。",
        "expected_behavior": ["提高SGL送达", "降低任务过期", "减少重复竞争"],
        "risk_notes": ["过高过期惩罚可能使早期策略偏保守"],
        "parent_candidate_id": None,
    }
    second = json.loads(json.dumps(base, ensure_ascii=False))
    second["reward_name"] = "deadline_delivery_mock_v2"
    second["positive_weights"]["completion_score"] = 1.0
    return [LlmRewardSpec.from_dict(base), LlmRewardSpec.from_dict(second)]
