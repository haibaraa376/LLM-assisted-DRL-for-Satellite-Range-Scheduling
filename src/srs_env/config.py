"""加载并验证第二天环境与任务配置。"""

from pathlib import Path

import yaml


def _load(path):
    """读取一个UTF-8 YAML文件并返回字典。"""
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def validate_environment_config(config):
    """检查30秒、24小时、15智能体及数值容差等固定环境约束。"""
    environment = config["environment"]
    if environment["horizon_seconds"] != 86400 or environment["decision_step_seconds"] != 30:
        raise ValueError("环境必须使用24小时和30秒决策步")
    if environment["agent_count"] != 15:
        raise ValueError("环境必须包含15颗卫星智能体")
    if environment["metrics"]["load_balance"]["b_max"] <= 0:
        raise ValueError("b_max 必须为正数")
    if environment["numerical"]["data_tolerance_mbit"] <= 0:
        raise ValueError("数据量容差必须为正数")
    for key in ("time_tolerance_seconds", "ratio_tolerance"):
        if environment["numerical"][key] <= 0:
            raise ValueError("配置项 numerical.{0} 必须为正数".format(key))
    action = environment["action"]
    concurrency = environment["concurrency"]
    resources = environment["resources"]
    expected = {
        "action.candidate_task_count": (action["candidate_task_count"], 4),
        "action.max_subactions_per_satellite": (
            action["max_subactions_per_satellite"],
            4,
        ),
        "concurrency.max_outgoing_inter_satellite_links_per_satellite_per_slot": (
            concurrency["max_outgoing_inter_satellite_links_per_satellite_per_slot"],
            3,
        ),
        "concurrency.max_sgl_links_per_satellite_per_slot": (
            concurrency["max_sgl_links_per_satellite_per_slot"],
            1,
        ),
        "concurrency.max_transmissions_per_task_per_slot": (
            concurrency["max_transmissions_per_task_per_slot"],
            1,
        ),
        "concurrency.max_tasks_per_physical_inter_satellite_link_per_slot": (
            concurrency[
                "max_tasks_per_physical_inter_satellite_link_per_slot"
            ],
            1,
        ),
        "resources.inter_satellite_interface_count": (
            resources["inter_satellite_interface_count"],
            3,
        ),
        "resources.sgl_interface_count": (resources["sgl_interface_count"], 1),
        "resources.ground_station_antenna_count": (
            resources["ground_station_antenna_count"],
            1,
        ),
    }
    for path, (actual, required) in expected.items():
        if actual != required:
            raise ValueError("配置项 {0} 必须等于 {1}".format(path, required))
    if concurrency["same_slot_forwarding"] is not False:
        raise ValueError("配置项 concurrency.same_slot_forwarding 必须为false")


def validate_task_config(config):
    """检查任务源域、寿命分段、数据量范围和训练划分比例。"""
    database = config["task_database"]
    lifetime = database["survival_time_seconds"]
    values = [lifetime["priority_1_to_3"], lifetime["priority_4_to_6"], lifetime["priority_7_to_9"], lifetime["priority_10"]]
    if database["database_size"] <= 0 or database["source_domain_id"] != "D2":
        raise ValueError("任务数量必须为正，且任务源域必须是D2")
    if not values[0] >= values[1] >= values[2] >= values[3] > 0:
        raise ValueError("任务生存时间必须满足 T1 >= T2 >= T3 >= T4 > 0")
    size = database["data_size_mbit"]
    if size["minimum"] <= 0 or size["maximum"] < size["minimum"]:
        raise ValueError("任务数据量范围不合法")
    if abs(sum(database["split"].values()) - 1.0) > 1e-9:
        raise ValueError("任务划分比例之和必须为1")
    train_size = int(database["database_size"] * database["split"]["train"])
    if config["episode"]["task_count"] > train_size:
        raise ValueError("episode任务数不能超过默认train划分数量")


def load_environment_config(path=Path("configs/environment.yaml")):
    """加载并验证环境配置。"""
    config = _load(path)
    validate_environment_config(config)
    return config


def load_task_config(path=Path("configs/tasks.yaml")):
    """加载并验证任务数据库配置。"""
    config = _load(path)
    validate_task_config(config)
    return config
