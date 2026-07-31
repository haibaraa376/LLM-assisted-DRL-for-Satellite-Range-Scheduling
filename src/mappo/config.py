"""加载并严格验证第三天MAPPO配置。"""

import math
from pathlib import Path

import torch
import yaml


def load_mappo_config(path=Path("configs/mappo.yaml")):
    """读取UTF-8 YAML并返回 ``mappo`` 字典；不修改配置文件。"""
    with Path(path).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or "mappo" not in document:
        raise ValueError("MAPPO配置必须包含mappo根节点")
    config = document["mappo"]
    validate_mappo_config(config)
    return config


def validate_mappo_config(config):
    """验证固定维度、算法范围及CTDE设置，不静默修正错误值。"""
    encoding = config["encoding"]
    algorithm = config["algorithm"]
    required = {
        "agents.count": (config["agents"]["count"], 15),
        "encoding.candidate_task_count": (
            encoding["candidate_task_count"],
            4,
        ),
        "encoding.target_slot_count": (encoding["target_slot_count"], 19),
        "encoding.target_choice_count_with_idle": (
            encoding["target_choice_count_with_idle"],
            20,
        ),
        "encoding.expected_actor_observation_dim": (
            encoding["expected_actor_observation_dim"],
            213,
        ),
        "encoding.expected_critic_state_dim": (
            encoding["expected_critic_state_dim"],
            3200,
        ),
    }
    for name, (actual, expected) in required.items():
        if actual != expected:
            raise ValueError("配置项{0}必须等于{1}".format(name, expected))
    if config["agents"]["shared_actor"] is not True:
        raise ValueError("第三天必须使用15颗卫星共享的Actor")
    if config["agents"]["centralized_critic"] is not True:
        raise ValueError("第三天必须使用集中式Critic")
    if not 0.0 < algorithm["gamma"] <= 1.0:
        raise ValueError("gamma必须位于(0,1]")
    if not 0.0 <= algorithm["gae_lambda"] <= 1.0:
        raise ValueError("gae_lambda必须位于[0,1]")
    for name in ("clip_ratio", "actor_learning_rate", "critic_learning_rate"):
        if algorithm[name] <= 0.0:
            raise ValueError("算法配置{0}必须为正数".format(name))
    if config["rollout"]["steps_per_update"] <= 0:
        raise ValueError("每次更新的rollout步数必须为正数")
    if config["smoke_training"]["update_count"] <= 0:
        raise ValueError("冒烟训练更新次数必须为正数")
    _validate_manual_reward_config(config["manual_reward"])
    _validate_manual_training_config(config["manual_training"])


def _validate_manual_reward_config(config):
    """验证人工奖励权重、归一化和违反代码分类。"""
    if not isinstance(config["enabled"], bool):
        raise ValueError("manual_reward.enabled必须是布尔值")
    if config["shared_global_reward"] is not True:
        raise ValueError("人工奖励必须是15颗卫星共享的团队奖励")
    expected_weights = {
        "sgl_progress",
        "relay_progress",
        "completion",
        "balance",
        "expiration",
        "invalid_action",
        "coordination_conflict",
        "relay_cost",
    }
    if set(config["weights"]) != expected_weights:
        raise ValueError("人工奖励权重字段不完整")
    for name, value in config["weights"].items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("人工奖励权重{0}必须是非负有限数".format(name))
    for name, value in config["normalization"].items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("人工奖励归一化参数{0}必须是正有限数".format(name))
    numerical = config["numerical"]
    if not 0 < numerical["warning_abs_reward"] < numerical["hard_failure_abs_reward"]:
        raise ValueError("奖励硬失败阈值必须大于正的警告阈值")
    if numerical["tolerance"] <= 0 or not math.isfinite(numerical["tolerance"]):
        raise ValueError("人工奖励数值容差必须是正有限数")
    invalid = set(config["invalid_violation_codes"])
    coordination = set(config["coordination_violation_codes"])
    if not invalid or not coordination:
        raise ValueError("无效动作和协调冲突代码集合不能为空")
    if invalid & coordination:
        raise ValueError("无效动作与协调冲突代码集合不能重叠")


def _validate_manual_training_config(config):
    """验证train/validation隔离、任务规模、Episode和Checkpoint路径。"""
    if config["split"] != "train":
        raise ValueError("人工MAPPO训练只能使用train划分")
    validation = config["validation"]
    if validation["split"] != "validation":
        raise ValueError("模型选择只能使用validation划分，禁止使用test")
    if "test" in (config["split"], validation["split"]):
        raise ValueError("第三天训练和验证不得访问test划分")
    if not 0 < config["task_count"] <= 700:
        raise ValueError("train任务数必须位于1到700")
    if not 0 < validation["task_count"] <= 150:
        raise ValueError("validation任务数必须位于1到150")
    if config["episode_count"] <= 0:
        raise ValueError("正式训练Episode数量必须为正数")
    if config["rollout_steps_per_update"] <= 0:
        raise ValueError("正式训练Rollout步数必须为正数")
    seeds = validation["seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("validation seeds必须非空且互不重复")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError("validation seed必须是非负整数")
    checkpoint = config["checkpoint"]
    if Path(checkpoint["best_path"]) == Path(checkpoint["last_path"]):
        raise ValueError("best和last Checkpoint路径必须不同")
    if checkpoint["save_at_episode_boundary_only"] is not True:
        raise ValueError("正式Checkpoint只能在Episode边界保存")


def validate_runtime_compatibility(config, environment):
    """核对MAPPO固定形状与当前第二天环境，仅读取环境元数据。"""
    if len(environment.dataset.satellite_ids) != 15:
        raise ValueError("MAPPO运行数据必须包含15颗卫星")
    if len(environment.dataset.ground_station_ids) != 4:
        raise ValueError("MAPPO运行数据必须包含4个地面站")
    environment_limit = environment.config["action"]["candidate_task_count"]
    if environment_limit != config["encoding"]["candidate_task_count"]:
        raise ValueError("MAPPO候选任务数与环境配置不一致")


def resolve_device(device_name):
    """将auto解析为CUDA或CPU，并拒绝当前机器不可用的显式设备。"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("配置要求CUDA，但当前PyTorch无法使用CUDA")
    return device
