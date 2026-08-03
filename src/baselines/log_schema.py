"""构造奖励语义明确的2.0日志，并兼容读取历史Episode记录。"""

from copy import deepcopy
import math


LOG_SCHEMA_VERSION = "2.1"


def resolve_reward_log_semantics(
    reward_model,
    total_training_reward,
    reward_component_sums,
):
    """根据奖励模型只读元数据计算基础奖励与塑形奖励分解。"""
    metadata = reward_model.log_metadata
    total = float(total_training_reward)
    if metadata.shaping_reward_name is None:
        base_sum = total
        shaping_sum = 0.0
    else:
        try:
            base_sum = float(reward_component_sums[metadata.base_reward_name])
            shaping_sum = float(
                reward_component_sums[metadata.shaping_reward_name]
            )
        except KeyError as error:
            raise ValueError("奖励分量缺少基础奖励或塑形奖励") from error
    values = (total, base_sum, shaping_sum)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("奖励日志包含NaN或Inf")
    if not math.isclose(
        total,
        base_sum + shaping_sum,
        rel_tol=1.0e-7,
        abs_tol=1.0e-8,
    ):
        raise ValueError("训练总奖励与基础奖励、塑形奖励之和不一致")
    fields = {
        "log_schema_version": LOG_SCHEMA_VERSION,
        "reward_method": metadata.reward_method,
        "base_reward_name": metadata.base_reward_name,
        "base_reward_sum": base_sum,
        "shaping_reward_name": metadata.shaping_reward_name,
        "shaping_reward_sum": shaping_sum,
    }
    reward_spec_id = getattr(reward_model, "reward_spec_id", None)
    if reward_spec_id is not None:
        fields["reward_spec_id"] = reward_spec_id
    weight_metadata = getattr(reward_model, "weight_metadata", None)
    if weight_metadata is not None:
        fields["llm_reward_weight_metadata"] = deepcopy(weight_metadata)
    return fields


def build_update_log_record(
    record,
    reward_model,
    total_training_reward,
    reward_component_sums,
):
    """为单个Rollout更新记录增加统一奖励Schema。"""
    output = dict(record)
    output.pop("total_manual_reward", None)
    output["total_training_reward"] = float(total_training_reward)
    output.update(
        resolve_reward_log_semantics(
            reward_model,
            total_training_reward,
            reward_component_sums,
        )
    )
    return output


def build_episode_log_record(
    record,
    reward_model,
    total_training_reward,
    reward_component_sums,
):
    """构造不含错误total_manual_reward字段的新Episode记录。"""
    output = dict(record)
    output.pop("total_manual_reward", None)
    output["total_training_reward"] = float(total_training_reward)
    output.update(
        resolve_reward_log_semantics(
            reward_model,
            total_training_reward,
            reward_component_sums,
        )
    )
    return output


def normalize_episode_log_record(record):
    """只读归一化新旧日志，不改写原字典或源文件。"""
    source = deepcopy(record)
    version = source.get("log_schema_version")
    if version in {LOG_SCHEMA_VERSION, "2.0"}:
        _validate_normalized_record(source)
        source["source_log_schema_version"] = version
        source["log_schema_version"] = LOG_SCHEMA_VERSION
        source["normalized_log_schema_version"] = LOG_SCHEMA_VERSION
        return source

    if "total_training_reward" in source:
        total = float(source["total_training_reward"])
    elif "total_manual_reward" in source:
        total = float(source["total_manual_reward"])
    else:
        raise ValueError("历史Episode日志缺少训练总奖励")
    method = source.get("method", "manual_mappo")
    components = source.get("reward_component_sums", {})
    if method == "ppo_lya":
        base = float(
            source.get(
                "manual_reward_sum",
                components.get("manual_reward", total),
            )
        )
        shaping = float(
            source.get(
                "lyapunov_shaping_sum",
                components.get("lyapunov_shaping", total - base),
            )
        )
        semantics = {
            "reward_method": "manual_plus_lyapunov",
            "base_reward_name": "manual_reward",
            "base_reward_sum": base,
            "shaping_reward_name": "lyapunov_shaping",
            "shaping_reward_sum": shaping,
        }
    elif method == "llm_ppo":
        semantics = {
            "reward_method": "llm_weight_reward",
            "base_reward_name": "llm_weight_reward",
            "base_reward_sum": total,
            "shaping_reward_name": None,
            "shaping_reward_sum": 0.0,
        }
    else:
        semantics = {
            "reward_method": "manual_reward",
            "base_reward_name": "manual_reward",
            "base_reward_sum": total,
            "shaping_reward_name": None,
            "shaping_reward_sum": 0.0,
        }
    source.pop("total_manual_reward", None)
    source.update(semantics)
    source["total_training_reward"] = total
    source["log_schema_version"] = LOG_SCHEMA_VERSION
    source["source_log_schema_version"] = version or "legacy"
    source["normalized_log_schema_version"] = LOG_SCHEMA_VERSION
    _validate_normalized_record(source)
    return source


def _validate_normalized_record(record):
    """验证统一记录的奖励分解一致且全部有限。"""
    values = (
        float(record["total_training_reward"]),
        float(record["base_reward_sum"]),
        float(record["shaping_reward_sum"]),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("归一化Episode日志包含NaN或Inf")
    if not math.isclose(
        values[0],
        values[1] + values[2],
        rel_tol=1.0e-7,
        abs_tol=1.0e-8,
    ):
        raise ValueError("归一化Episode奖励分解不一致")
