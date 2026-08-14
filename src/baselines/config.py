"""加载并严格校验统一基线配置。"""

import math
from pathlib import Path
from urllib.parse import urlparse

import yaml


def load_baseline_config(path=Path("configs/baselines.yaml")):
    """读取UTF-8 YAML并返回baselines根节点。"""
    with Path(path).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or "baselines" not in document:
        raise ValueError("基线配置必须包含baselines根节点")
    config = document["baselines"]
    validate_baseline_config(config)
    return config


def _finite_number(value, name, positive=False, nonnegative=False):
    """验证普通有限数，明确拒绝布尔值和NaN/Inf。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{0}必须是数值".format(name))
    if not math.isfinite(float(value)):
        raise ValueError("{0}必须是有限数".format(name))
    if positive and value <= 0:
        raise ValueError("{0}必须为正数".format(name))
    if nonnegative and value < 0:
        raise ValueError("{0}必须为非负数".format(name))


def validate_baseline_config(config, mappo_config=None):
    """校验划分、公平性、Lyapunov、Provider和预算边界。"""
    training = config["training"]
    if training["split"] != "train":
        raise ValueError("基线训练只允许train划分")
    if training["validation"]["protocol"] != "checkpoint_selection":
        raise ValueError("基线模型选择必须使用checkpoint_selection协议")
    if "split" in training["validation"]:
        raise ValueError("validation划分已由隔离协议取代，不得直接配置split")
    if training["split"] == "test":
        raise ValueError("基线训练不得使用test划分")
    for name in ("task_count", "episode_count", "rollout_steps"):
        _finite_number(training[name], "training.{0}".format(name), positive=True)
    _finite_number(
        training["training_seed"],
        "training.training_seed",
        nonnegative=True,
    )
    protocols = config["evaluation_protocols"]
    _finite_number(protocols["split_seed"], "协议划分seed", nonnegative=True)
    # checkpoint_selection的任务数由training.task_count在运行时推导，
    # 不在YAML中重复保存，避免二者改动后发生冲突。
    if "task_count" in protocols["checkpoint_selection"]:
        raise ValueError(
            "checkpoint_selection.task_count由training.task_count自动推导，"
            "不得重复配置"
        )
    for protocol_name in ("reward_search", "test"):
        protocol = protocols[protocol_name]
        _finite_number(
            protocol["task_count"],
            "协议任务数.{0}".format(protocol_name),
            positive=True,
        )
    for protocol_name in ("reward_search", "checkpoint_selection", "test"):
        protocol = protocols[protocol_name]
        seeds = protocol["seeds"]
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("协议seeds必须非空且互不重复")

    lyapunov = config["methods"]["ppo_lya"]["lyapunov"]
    for name in (
        "gamma",
        "shaping_coefficient",
        "urgency_power",
        "numerical_tolerance",
        "hard_failure_abs_shaping",
    ):
        _finite_number(lyapunov[name], "lyapunov.{0}".format(name), positive=True)
    weights = lyapunov["feature_weights"]
    if set(weights) != {
        "backlog",
        "expiration_risk",
        "expired_undelivered",
        "utilization_imbalance",
    }:
        raise ValueError("Lyapunov特征权重必须恰好包含四项")
    for name, value in weights.items():
        _finite_number(value, "lyapunov权重.{0}".format(name), nonnegative=True)
    if sum(weights.values()) <= 0:
        raise ValueError("Lyapunov权重总和必须大于0")
    if mappo_config is not None and not math.isclose(
        float(lyapunov["gamma"]),
        float(mappo_config["algorithm"]["gamma"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Lyapunov gamma必须与MAPPO gamma一致")

    llm = config["methods"]["llm_ppo"]
    provider = llm["provider"]
    if provider["default"] != "mock":
        raise ValueError("默认Provider必须为mock")
    if provider["live_provider"] != "deepseek":
        raise ValueError("真实Provider必须为deepseek")
    parsed = urlparse(provider["base_url"])
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("DeepSeek API地址必须是有效HTTPS URL")
    variable_name = provider["api_key_env"]
    if variable_name != "DEEPSEEK_API_KEY":
        raise ValueError("API配置只能保存DEEPSEEK_API_KEY变量名")
    if any(token in variable_name.lower() for token in ("sk-", "bearer ")):
        raise ValueError("api_key_env不得包含真实密钥")
    if llm["output"]["allow_arbitrary_python"] is not False:
        raise ValueError("LLM不得生成或执行任意Python")
    limits = llm["weight_limits"]
    _finite_number(limits["minimum"], "weight minimum", nonnegative=True)
    _finite_number(limits["maximum"], "weight maximum", positive=True)
    if limits["minimum"] >= limits["maximum"]:
        raise ValueError("奖励权重上下界顺序错误")
    for name in (
        "max_tokens",
        "timeout_seconds",
        "max_retries",
    ):
        _finite_number(
            provider[name],
            "provider.{0}".format(name),
            nonnegative=name == "max_retries",
            positive=name != "max_retries",
        )
    search = llm["search"]
    for name in (
        "rounds",
        "candidates_per_round",
        "candidate_training_episodes",
        "candidate_task_count",
        "maximum_api_calls",
        "maximum_total_input_tokens",
        "maximum_total_output_tokens",
    ):
        _finite_number(search[name], "search.{0}".format(name), positive=True)
    if search["evaluation_protocol"] != "reward_search":
        raise ValueError("LLM候选搜索必须使用reward_search协议")
    selection = config["candidate_selection"]
    _finite_number(selection["tail_episodes"], "候选尾部Episode数", positive=True)
    if int(selection["tail_episodes"]) > int(search["candidate_training_episodes"]):
        raise ValueError("候选尾部Episode数不得超过候选训练Episode数")

    from mappo.model_selection import normalize_best_model_rule

    normalize_best_model_rule(config["best_model_rule"])
    artifacts = config.get("artifact_management", {})
    for name in (
        "save_episode_checkpoints",
        "write_learning_curves",
        "compact_completed_runs",
    ):
        if not isinstance(artifacts.get(name), bool):
            raise ValueError("artifact_management.{0}必须是布尔值".format(name))
    eligibility = config["candidate_eligibility"]
    _finite_number(
        eligibility["minimum_accepted_sgl_mean"],
        "候选最低SGL均值",
        nonnegative=True,
    )
    dominance = eligibility["maximum_single_component_dominance"]
    _finite_number(dominance, "候选奖励支配阈值", positive=True)
    if dominance > 1.0:
        raise ValueError("候选奖励支配阈值不得超过1")
    llm_ratio = eligibility["maximum_llm_contribution_ratio"]
    _finite_number(llm_ratio, "候选LLM贡献阈值", nonnegative=True)
    if llm_ratio > 1.0:
        raise ValueError("候选LLM贡献阈值不得超过1")
    composition = config["reward_composition"]
    if composition["mode"] != "base_plus_llm":
        raise ValueError("LLM奖励组合模式必须是base_plus_llm")
    _finite_number(composition["alpha"], "LLM塑形alpha", nonnegative=True)
    if composition["base"] != {
        "source": "manual_reward",
        "features": ["sgl_progress", "completion_score", "expiration_loss"],
    }:
        raise ValueError("基础奖励必须固定为人工SGL、完成和过期三项")
    for name, value in composition["feature_scales"].items():
        _finite_number(value, "LLM特征缩放.{0}".format(name), positive=True)
    if len(composition["feature_scales"]) != 8:
        raise ValueError("LLM特征缩放必须恰好包含八项")
    staged = config["staged_search"]
    if staged["enabled"] is not True:
        raise ValueError("LLM候选筛选必须启用固定分阶段流程")
    for name in (
        "initial_episodes", "keep_after_stage_1", "extra_episodes_stage_2",
        "keep_after_stage_2", "extra_episodes_stage_3", "keep_after_stage_3",
        "confirmation_episodes", "max_episodes_per_candidate",
    ):
        _finite_number(staged[name], "staged_search.{0}".format(name), positive=True)
    if int(staged["initial_episodes"]) < 2:
        raise ValueError("第一阶段至少需要两个完整Episode")
    expected_total = (
        int(staged["initial_episodes"])
        + int(staged["extra_episodes_stage_2"])
        + int(staged["extra_episodes_stage_3"])
        + int(staged["confirmation_episodes"])
    )
    if expected_total != int(staged["max_episodes_per_candidate"]):
        raise ValueError("分阶段Episode总数必须等于候选最大Episode数")
    if not (
        int(search["candidates_per_round"])
        >= int(staged["keep_after_stage_1"])
        >= int(staged["keep_after_stage_2"])
        >= int(staged["keep_after_stage_3"])
    ):
        raise ValueError("分阶段保留数量必须单调且不超过每轮候选数")
