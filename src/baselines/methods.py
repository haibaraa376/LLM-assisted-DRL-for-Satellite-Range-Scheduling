"""集中定义基线方法ID、显示名称和奖励规范需求。"""

from enum import Enum


class BaselineMethod(str, Enum):
    """统一训练框架支持的三种多智能体基线。"""

    MANUAL_MAPPO = "manual_mappo"
    PPO_LYA = "ppo_lya"
    LLM_PPO = "llm_ppo"


DEFAULT_METHOD_ORDER = (
    BaselineMethod.MANUAL_MAPPO,
    BaselineMethod.PPO_LYA,
    BaselineMethod.LLM_PPO,
)

_DISPLAY_NAMES = {
    BaselineMethod.MANUAL_MAPPO: "Manual-MAPPO",
    BaselineMethod.PPO_LYA: "PPO-Lya",
    BaselineMethod.LLM_PPO: "LLM-PPO",
}


def parse_baseline_method(value):
    """把CLI字符串严格解析为方法枚举。"""
    try:
        return BaselineMethod(value)
    except ValueError as error:
        allowed = ", ".join(item.value for item in DEFAULT_METHOD_ORDER)
        raise ValueError("未知基线方法；允许值：{0}".format(allowed)) from error


def parse_baseline_methods(values):
    """保留用户顺序解析单个、多个或all，并拒绝重复。"""
    values = list(values)
    if not values:
        raise ValueError("至少选择一种基线方法")
    if "all" in values:
        if values != ["all"]:
            raise ValueError("all不能与具体方法同时使用")
        return list(DEFAULT_METHOD_ORDER)
    methods = [parse_baseline_method(value) for value in values]
    if len(methods) != len(set(methods)):
        raise ValueError("基线方法不得重复")
    return methods


def display_name(method):
    return _DISPLAY_NAMES[BaselineMethod(method)]


def requires_reward_spec(method):
    return BaselineMethod(method) == BaselineMethod.LLM_PPO


def may_require_live_api(method):
    """只有LLM-PPO准备新奖励时可能触发真实API。"""
    return BaselineMethod(method) == BaselineMethod.LLM_PPO
