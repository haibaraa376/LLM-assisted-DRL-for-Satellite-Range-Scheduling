"""把各方法Summary汇总为不依赖训练累计奖励的统一比较。"""

from pathlib import Path

from mappo.model_selection import (
    DEFAULT_SELECTION_RULE,
    normalize_selection_rule,
    rank_validation_results,
)

from .run_management import atomic_write_json, load_json


def build_comparison(
    run_directory,
    methods,
    method_states=None,
    selection_rule=None,
    protocol_name="checkpoint_selection",
    output_name="comparison.json",
):
    """读取独立方法Summary并生成validation主导的比较记录。"""
    root = Path(run_directory)
    records = []
    states = method_states or {}
    for method in methods:
        method_id = getattr(method, "value", method)
        summary_path = root / method_id / "summary.json"
        state = states.get(method_id, {})
        if not summary_path.exists():
            records.append(
                {
                    "method": method_id,
                    "status": state.get("status", "not_completed"),
                }
            )
            continue
        summary = load_json(summary_path)
        validation = summary.get("best_validation_result") or {}
        records.append(
            {
                "method": method_id,
                "status": state.get("status", "completed"),
                "target_episode_count": summary["target_episode_count"],
                "completed_episode_count": state.get(
                    "completed_episode_count",
                    summary["target_episode_count"],
                ),
                "total_environment_steps": state.get(
                    "total_environment_steps",
                    summary["environment_steps_this_run"],
                ),
                "best_checkpoint": state.get("best_checkpoint"),
                "last_checkpoint": state.get("last_checkpoint"),
                "best_validation_episode": summary.get("best_episode_index"),
                "timeliness_raw_mean": validation.get("timeliness_raw_mean"),
                "delivered_timeliness_raw_mean": validation.get(
                    "delivered_timeliness_raw_mean"
                ),
                "load_balance_mean_per_task_mean": validation.get(
                    "load_balance_mean_per_task_mean"
                ),
                "completed_task_count_mean": validation.get(
                    "completed_task_count_mean"
                ),
                "expired_task_count_mean": validation.get(
                    "expired_task_count_mean"
                ),
                "delivered_data_mbit_mean": validation.get(
                    "delivered_data_mbit_mean"
                ),
                "completion_rate_mean": validation.get("completion_rate_mean"),
                "expiration_rate_mean": validation.get("expiration_rate_mean"),
                "rejected_subaction_rate_mean": validation.get(
                    "rejected_subaction_rate_mean"
                ),
                "sgl_action_fraction_mean": validation.get(
                    "sgl_action_fraction_mean"
                ),
                "training_duration_seconds": state.get(
                    "training_duration_seconds"
                ),
                "reward_method": state.get("reward_method"),
                "reward_spec_id": state.get("reward_spec_id"),
            }
        )
    completed = [
        item
        for item in records
        if item.get("delivered_timeliness_raw_mean") is not None
    ]
    rule = normalize_selection_rule(selection_rule or DEFAULT_SELECTION_RULE)
    ranking = rank_validation_results(
        completed,
        rule=rule,
    )
    payload = {
        "schema_version": "2.0",
        "selection_rule": rule,
        "evaluation_protocol": protocol_name,
        "methods": records,
        "validation_ranking": [item["method"] for item in ranking],
    }
    atomic_write_json(root / output_name, payload)
    return payload
