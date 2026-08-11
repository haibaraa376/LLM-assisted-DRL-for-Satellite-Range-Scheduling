"""验证固定任务人工奖励诊断的精简输出转换。"""

import csv

import pytest

from scripts.train_manual_diagnostics import (
    REWARD_PROFILES,
    write_diagnostic_csvs,
)


def _component_abs_sums(conflict):
    """构造八项完整的绝对奖励贡献，避免测试依赖训练环境。"""
    return {
        "weighted_sgl_progress": 2.0,
        "weighted_relay_progress": 1.0,
        "weighted_completion": 3.0,
        "weighted_balance": 1.0,
        "weighted_expiration": 4.0,
        "weighted_invalid_action": 1.0,
        "weighted_coordination_conflict": conflict,
        "weighted_relay_cost": 2.0,
    }


def test_diagnostic_profiles_match_requested_weights():
    assert REWARD_PROFILES["no_conflict"]["coordination_conflict"] == 0.0
    assert REWARD_PROFILES["no_conflict"]["completion"] == 0.5
    assert REWARD_PROFILES["balanced"]["coordination_conflict"] == 0.004
    assert REWARD_PROFILES["balanced"]["completion"] == 1.0


def test_diagnostic_csvs_keep_only_requested_metrics_and_absolute_ratios(tmp_path):
    episodes = [
        {
            "episode_index": 0,
            "completed_task_count": 6,
            "expired_task_count": 4,
            "delivered_timeliness_raw": 12.5,
            "delivered_data_mbit": 321.0,
            "total_training_reward": -3.0,
            "reward_component_abs_sums": _component_abs_sums(0.0),
        }
    ]
    updates = [
        {
            "episode_index": 0,
            "update_index": 1,
            "approximate_kl": 0.02,
            "clip_fraction": 0.1,
            "critic_loss": 0.4,
            "entropy": 1.2,
        }
    ]

    write_diagnostic_csvs(tmp_path, episodes, updates, task_count=20)

    with (tmp_path / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    assert list(metrics[0]) == [
        "episode",
        "completion_rate",
        "expiration_rate",
        "delivered_timeliness_raw",
        "delivered_data_mbit",
        "episode_reward",
    ]
    assert float(metrics[0]["completion_rate"]) == pytest.approx(0.3)
    assert float(metrics[0]["expiration_rate"]) == pytest.approx(0.2)

    with (tmp_path / "ppo_diagnostics.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        diagnostics = list(csv.DictReader(stream))
    assert float(diagnostics[0]["approximate_kl"]) == pytest.approx(0.02)
    assert float(diagnostics[0]["critic_loss"]) == pytest.approx(0.4)

    with (tmp_path / "reward_contributions.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        contributions = list(csv.DictReader(stream))
    ratios = [
        float(contributions[0][name])
        for name in contributions[0]
        if name != "episode"
    ]
    assert sum(ratios) == pytest.approx(1.0)
    assert float(contributions[0]["coordination_conflict"]) == 0.0
