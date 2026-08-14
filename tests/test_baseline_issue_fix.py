"""验证基线问题修复的纯函数、协议和奖励语义，不启动训练。"""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from baselines.baseline_runner import build_training_config
from baselines.config import load_baseline_config
from baselines.llm_reward import normalized_reward_spec_weights
from baselines.llm_schema import default_mock_specs
from baselines.lyapunov_reward import PpoLyaReward, extract_lyapunov_features
from mappo.config import load_mappo_config
from mappo.evaluation_protocol import build_evaluation_protocol
from mappo.model_selection import (
    compare_validation_results,
    rank_validation_results,
)
from mappo.training_runner import BaselineTrainingRunner
from srs_env.models import TaskDefinition, TaskState, TaskStatus
from srs_env.tasks import load_task_splits
from srs_env.tasks import load_task_database, sample_episode_tasks


def _metrics(**overrides):
    values = {
        "delivered_timeliness_raw_mean": 10.0,
        "completion_rate_mean": 0.5,
        "expiration_rate_mean": 0.4,
        "delivered_data_mbit_mean": 100.0,
        "load_balance_mean_per_task_mean": 0.2,
        "rejected_subaction_rate_mean": 0.3,
    }
    values.update(overrides)
    return values


def _dummy_environment(current_time=99.0):
    task = TaskDefinition(
        task_id="task_protocol_test",
        source_satellite_id="os01",
        target_ground_station_id="gs01",
        priority=10,
        data_size_mbit=100.0,
        survival_time_s=100.0,
        arrival_time_s=0.0,
        expiration_time_s=100.0,
        nominal_sgl_duration_s=10.0,
    )
    state = TaskState(task, TaskStatus.ACTIVE, np.array([100.0] + [0.0] * 14))
    return SimpleNamespace(
        tasks={task.task_id: state},
        task_index={task.task_id: 0},
        current_time_s=current_time,
        outgoing_seconds=np.zeros((1, 15)),
        total_window_seconds=np.ones(15) * 100.0,
    )


class _ZeroManualReward:
    warning_count = 0

    def reset(self, environment):
        del environment

    def compute(self, environment, info):
        del environment, info
        return SimpleNamespace(
            total_reward=0.0,
            component_values=lambda: {},
        )


def test_model_selection_uses_tolerance_lexicographic_order():
    config = load_baseline_config()
    rule = config["best_model_rule"]["metrics"]
    within_primary_tolerance = _metrics(
        delivered_timeliness_raw_mean=10.0 - 5.0e-7,
        completion_rate_mean=0.6,
    )
    assert compare_validation_results(within_primary_tolerance, _metrics(), rule) == 1
    clearly_worse_primary = _metrics(
        delivered_timeliness_raw_mean=10.0 - 2.0e-6,
        completion_rate_mean=1.0,
    )
    assert compare_validation_results(clearly_worse_primary, _metrics(), rule) == -1


def test_model_selection_stable_and_rejects_nonfinite():
    config = load_baseline_config()
    rule = config["best_model_rule"]["metrics"]
    records = [{"id": "a", **_metrics()}, {"id": "b", **_metrics()}]
    assert [item["id"] for item in rank_validation_results(records, rule=rule)] == [
        "a",
        "b",
    ]
    with pytest.raises(ValueError, match="NaN或Inf"):
        compare_validation_results(_metrics(completion_rate_mean=np.nan), _metrics(), rule)
    missing_late_metric = _metrics(delivered_timeliness_raw_mean=20.0)
    missing_late_metric.pop("rejected_subaction_rate_mean")
    with pytest.raises(ValueError, match="缺少"):
        compare_validation_results(missing_late_metric, _metrics(), rule)


def test_checkpoint_protocol_uses_full_validation_pool_and_stays_off_train_test(
    tmp_path,
):
    baseline = load_baseline_config()
    config = baseline["evaluation_protocols"]
    splits = load_task_splits()
    search = build_evaluation_protocol("reward_search", config, splits)
    training = build_training_config(baseline, tmp_path)
    checkpoint = build_evaluation_protocol(
        "checkpoint_selection",
        training["evaluation_protocols"],
        splits,
    )
    repeated = build_evaluation_protocol("reward_search", config, splits)
    pools = [set(search["pool_task_ids"]), set(checkpoint["pool_task_ids"]), set(build_evaluation_protocol("test", config, splits)["pool_task_ids"])]
    assert all(pools[left].isdisjoint(pools[right]) for left in range(3) for right in range(left + 1, 3))
    assert set(splits["train"]).isdisjoint(set().union(*pools))
    assert search["pool_sha256"] == repeated["pool_sha256"]
    assert search["pool_size"] == 150
    assert checkpoint["pool_size"] == 150

    baseline["training"]["task_count"] = 150
    training = build_training_config(baseline, tmp_path)
    assert (
        training["evaluation_protocols"]["checkpoint_selection"]["task_count"]
        == 150
    )

def test_llm_weights_match_manual_l1_without_mutating_raw_spec():
    mappo = load_mappo_config()
    spec = default_mock_specs()[0]
    original = deepcopy(spec.to_dict())
    effective, metadata = normalized_reward_spec_weights(
        spec,
        mappo["manual_reward"]["weights"],
    )
    assert sum(abs(value) for value in effective.values()) == pytest.approx(2.32)
    assert effective["coordination_conflict"] == 0.0
    assert metadata["target_weight_l1"] == pytest.approx(2.32)
    assert metadata["normalization_mode"] == "l1_manual_scale_direct_llm_7d"
    assert spec.to_dict() == original


def test_manual_and_llm_use_the_same_fixed_training_task_ids():
    """两种正式方法共用Runner的training_seed和train池，因此任务完全相同。"""
    config = load_baseline_config()
    database, splits = load_task_database(), load_task_splits()
    manual_ids = [
        task.task_id for task in sample_episode_tasks(
            database,
            splits["train"],
            config["training"]["task_count"],
            config["training"]["training_seed"],
        )
    ]
    llm_ids = [
        task.task_id for task in sample_episode_tasks(
            database,
            splits["train"],
            config["training"]["task_count"],
            config["training"]["training_seed"],
        )
    ]
    assert len(manual_ids) == 150
    assert manual_ids == llm_ids


def test_reward_dominance_uses_only_eight_weighted_components():
    record = {
        "reward_component_abs_sums": {
            "weighted_sgl_progress": 9.0,
            "weighted_relay_progress": 0.0,
            "weighted_completion": 1.0,
            "weighted_balance": 0.0,
            "weighted_expiration": 0.0,
            "weighted_invalid_action": 0.0,
            "weighted_coordination_conflict": 0.0,
            "weighted_relay_cost": 0.0,
            "total_reward": 1000.0,
            "backlog": 1000.0,
        }
    }
    diagnostics = BaselineTrainingRunner._reward_diagnostics([record])
    assert diagnostics["weighted_component_abs_total"] == pytest.approx(10.0)
    assert diagnostics["maximum_single_component_dominance"] == pytest.approx(0.9)


def test_expired_debt_persists_and_completion_shaping_is_larger():
    config = load_baseline_config()["methods"]["ppo_lya"]["lyapunov"]
    completion_env = _dummy_environment()
    completion_reward = PpoLyaReward(_ZeroManualReward(), config)
    completion_reward.reset(completion_env)
    completion_env.tasks["task_protocol_test"].status = TaskStatus.COMPLETED
    completion = completion_reward.compute(completion_env, {})

    expiration_env = _dummy_environment()
    expiration_reward = PpoLyaReward(_ZeroManualReward(), config)
    expiration_reward.reset(expiration_env)
    expiration_env.tasks["task_protocol_test"].status = TaskStatus.EXPIRED
    expiration = expiration_reward.compute(expiration_env, {})
    expiration_env.current_time_s = 200.0
    persisted = extract_lyapunov_features(expiration_env, config)

    assert completion.shaping_reward > expiration.shaping_reward
    assert expiration.current_features.expired_undelivered == pytest.approx(1.0)
    assert persisted.expired_undelivered == pytest.approx(1.0)
