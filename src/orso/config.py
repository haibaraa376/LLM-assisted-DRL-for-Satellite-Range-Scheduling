"""读取并验证ORSO特有配置，不复制既有基线或MAPPO配置。"""

import math
from pathlib import Path

import yaml


def load_orso_config(path=Path("configs/orso.yaml")):
    """读取UTF-8 YAML，返回 ``orso`` 节点并立即严格校验。"""
    with Path(path).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or "orso" not in document:
        raise ValueError("ORSO配置必须包含orso根节点")
    config = document["orso"]
    validate_orso_config(config)
    return config


def _finite(value, name, *, positive=False, nonnegative=False):
    """拒绝布尔值、NaN、Inf及不满足边界的数值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{0}必须是数值".format(name))
    if not math.isfinite(float(value)):
        raise ValueError("{0}必须为有限数".format(name))
    if positive and value <= 0:
        raise ValueError("{0}必须为正数".format(name))
    if nonnegative and value < 0:
        raise ValueError("{0}必须为非负数".format(name))


def validate_orso_config(config, allow_smoke_override=False):
    """校验固定候选集、D3RB边界和项目级预算适配。"""
    if config.get("method_name") != "orso":
        raise ValueError("ORSO方法名必须为orso")
    generation = config["generation"]
    training = config["training"]
    utility = config["task_utility"]
    d3rb = config["d3rb"]

    _finite(generation["rounds"], "generation.rounds", positive=True)
    _finite(generation["candidates"], "generation.candidates", positive=True)
    if int(generation["rounds"]) != 1:
        raise ValueError("ORSO只允许一次生成固定候选集")

    for name in (
        "warmup_episodes_per_candidate",
        "allocation_quantum_episodes",
        "total_candidate_episode_budget",
        "max_episodes_per_candidate",
    ):
        _finite(training[name], "training.{0}".format(name), positive=True)
    if int(training["allocation_quantum_episodes"]) != 1:
        raise ValueError("ORSO仅支持每次分配一个Episode")
    candidate_count = int(generation["candidates"])
    warmup = int(training["warmup_episodes_per_candidate"])
    total_budget = int(training["total_candidate_episode_budget"])
    maximum = int(training["max_episodes_per_candidate"])
    if total_budget < candidate_count * warmup:
        raise ValueError("ORSO总预算不足以完成所有候选的warmup")
    if maximum < warmup:
        raise ValueError("每候选Episode上限不得小于warmup")
    if candidate_count * maximum < total_budget:
        raise ValueError("所有候选Episode上限之和不足以覆盖总预算")
    if training["evaluation_protocol"] != "reward_search":
        raise ValueError("ORSO搜索只能使用reward_search验证协议")

    if utility["primary_metric"] != "completion_rate_mean":
        raise ValueError("ORSO D3RB任务效用必须为completion_rate_mean")
    _finite(utility["valid_min"], "task_utility.valid_min", nonnegative=True)
    _finite(utility["valid_max"], "task_utility.valid_max", positive=True)
    if float(utility["valid_min"]) >= float(utility["valid_max"]):
        raise ValueError("任务效用上下界顺序错误")

    _finite(d3rb["d_min"], "d3rb.d_min", positive=True)
    _finite(d3rb["confidence_constant"], "d3rb.confidence_constant", positive=True)
    _finite(d3rb["delta"], "d3rb.delta", positive=True)
    if not 0.0 < float(d3rb["delta"]) < 1.0:
        raise ValueError("d3rb.delta必须位于(0,1)")

    expected_metrics = (
        ("completion_rate_mean", "max"),
        ("delivered_data_mbit_mean", "max"),
        ("load_balance_mean_per_task_mean", "max"),
    )
    metrics = tuple(
        (item.get("metric"), item.get("direction"))
        for item in config["final_selection"]["metrics"]
    )
    if metrics != expected_metrics:
        raise ValueError("ORSO最终选择必须为Completion > Delivered Data > Load Balance")
    _finite(
        config["final_selection"]["tail_episodes"],
        "final_selection.tail_episodes",
        positive=True,
    )
    expected_tail = 1 if allow_smoke_override else 3
    if int(config["final_selection"]["tail_episodes"]) != expected_tail:
        raise ValueError(
            "ORSO最终选择必须使用最后{0}个Episode的validation均值".format(
                expected_tail
            )
        )
    if int(config["final_selection"]["tail_episodes"]) > warmup:
        raise ValueError("ORSO最终选择窗口不得大于每候选warmup Episode数")
    artifacts = config["artifacts"]
    for name, value in artifacts.items():
        if not isinstance(value, bool):
            raise ValueError("artifacts.{0}必须是布尔值".format(name))
    if artifacts["save_episode_checkpoints"] is not False:
        raise ValueError("ORSO不得保存逐Episode checkpoint")
    for name in (
        "save_candidate_learning_curve_json",
        "save_candidate_learning_curve_csv",
        "save_candidate_learning_curve_png",
        "save_search_trace",
    ):
        if artifacts[name] is not True:
            raise ValueError("ORSO必须保留{0}".format(name))
    if not str(config["output"]["root"]).strip():
        raise ValueError("ORSO输出根目录不能为空")
