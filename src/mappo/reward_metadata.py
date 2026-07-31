"""定义训练Runner读取的统一奖励日志元数据。"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RewardLogMetadata:
    """描述训练奖励的基础项、塑形项和稳定方法名称。"""

    reward_method: str
    base_reward_name: str
    shaping_reward_name: Optional[str]
