"""验证 HERON-inspired Tail-3 聚合与层级筛选的关键边界。"""

import json
import math

import pytest

from baselines.candidate_search import aggregate_tail_validation, build_staged_plan
from baselines.config import load_baseline_config
from mappo.model_selection import compare_heron_results, rank_validation_results


HIERARCHY = [
    {
        "metric": "completion_rate",
        "direction": "max",
        "absolute_margin": 0.0,
        "relative_margin": 0.0,
        "std_scale": 1.0,
    },
    {
        "metric": "expiration_rate",
        "direction": "min",
        "absolute_margin": 0.0,
        "relative_margin": 0.0,
        "std_scale": 1.0,
    },
    {
        "metric": "delivered_timeliness_raw",
        "direction": "max",
        "absolute_margin": 0.0,
        "relative_margin": 0.0,
        "std_scale": 1.0,
    },
]


def metrics(
    completion=0.7,
    expiration=0.2,
    delivery=10.0,
    accepted_sgl=2.0,
    std=0.01,
    count=3,
):
    return {
        "completion_rate_mean": completion,
        "completion_rate_std": std,
        "completion_rate_sample_count": count,
        "expiration_rate_mean": expiration,
        "expiration_rate_std": std,
        "expiration_rate_sample_count": count,
        "delivered_timeliness_raw_mean": delivery,
        "delivered_timeliness_raw_std": std,
        "delivered_timeliness_raw_sample_count": count,
        "accepted_sgl_count_mean": accepted_sgl,
        "accepted_sgl_count_std": std,
        "accepted_sgl_count_sample_count": count,
    }


def curve_point(index, completion, accepted_sgl=2.0):
    """构造已完成、已守恒且已写入 validation 的曲线点。"""
    values = metrics(completion=completion, accepted_sgl=accepted_sgl)
    return {
        "episode_index": index,
        "full_episode": True,
        "data_conservation_passed": True,
        "validation": values,
        **values,
    }


def test_default_search_budget_and_tail_window():
    config = load_baseline_config()
    search = config["methods"]["llm_ppo"]["search"]
    assert search["rounds"] == 5
    assert search["candidates_per_round"] == 8
    assert search["candidate_training_episodes"] == 5
    assert config["candidate_selection"]["tail_episodes"] == 3


def test_fixed_staged_budget_and_smoke_scaling():
    config = load_baseline_config()["staged_search"]
    formal = build_staged_plan(8, config)
    assert [(item["candidates"], item["episodes"]) for item in formal] == [
        (8, 2), (4, 3), (2, 4), (1, 5),
    ]
    assert 8 * 2 + 4 + 2 + 1 == 23
    smoke = build_staged_plan(4, config)
    assert [(item["candidates"], item["episodes"]) for item in smoke] == [
        (4, 2), (2, 3), (1, 4), (1, 5),
    ]


def test_tail_aggregation_uses_only_episodes_three_to_five(tmp_path):
    # 前两项故意极端，若错误使用“最佳单Episode”将得到完全不同的结论。
    path = tmp_path / "learning_curve.json"
    points = [
        curve_point(0, 1.0),
        curve_point(1, 0.0),
        curve_point(2, 0.4, accepted_sgl=3.0),
        curve_point(3, 0.5, accepted_sgl=6.0),
        curve_point(4, 0.6, accepted_sgl=9.0),
    ]
    path.write_text(json.dumps(points), encoding="utf-8")
    result = aggregate_tail_validation(path, HIERARCHY, tail_episodes=3)
    assert result["selection_window_start_episode"] == 3
    assert result["selection_window_end_episode"] == 5
    assert result["selection_window_size"] == 3
    assert result["aggregated_validation"]["completion_rate_mean"] == pytest.approx(0.5)
    assert result["aggregated_validation"]["completion_rate_std"] == pytest.approx(0.1)
    assert result["aggregated_validation"]["completion_rate_sample_count"] == 3
    assert result["aggregated_validation"]["accepted_sgl_count_mean"] == pytest.approx(6.0)
    assert result["aggregated_validation"]["accepted_sgl_count_sample_count"] == 3


def test_tail_diagnostics_average_llm_contribution_from_episodes_three_to_five(tmp_path):
    path = tmp_path / "learning_curve.json"
    points = [curve_point(index, 0.5) for index in range(5)]
    names = (
        "weighted_sgl_progress", "weighted_relay_progress", "weighted_completion",
        "weighted_balance", "weighted_expiration", "weighted_invalid_action",
        "weighted_coordination_conflict", "weighted_relay_cost", "llm_sgl_progress",
        "llm_relay_progress", "llm_completion", "llm_balance", "llm_expiration",
        "llm_invalid_action", "llm_coordination_conflict", "llm_relay_cost",
    )
    for index, point in enumerate(points):
        point["reward_component_abs_sums"] = {name: 0.0 for name in names}
        point["reward_component_abs_sums"]["weighted_sgl_progress"] = 10.0
        point["reward_component_abs_sums"]["llm_sgl_progress"] = float(index + 1)
    path.write_text(json.dumps(points), encoding="utf-8")
    result = aggregate_tail_validation(path, HIERARCHY, 3, candidate_id="candidate_tail")
    diagnostics = result["aggregated_diagnostics"]
    assert diagnostics["llm_abs_sum"] == pytest.approx(4.0)
    assert diagnostics["llm_contribution_ratio"] == pytest.approx(4.0 / 14.0)


def test_stable_tail_candidate_beats_early_peak_candidate(tmp_path):
    path_a, path_b = tmp_path / "a.json", tmp_path / "b.json"
    path_a.write_text(json.dumps([curve_point(i, value) for i, value in enumerate([1.0, 0.1, 0.2, 0.2, 0.2])]), encoding="utf-8")
    path_b.write_text(json.dumps([curve_point(i, value) for i, value in enumerate([0.4, 0.4, 0.6, 0.6, 0.6])]), encoding="utf-8")
    a = aggregate_tail_validation(path_a, HIERARCHY, 3)["aggregated_validation"]
    b = aggregate_tail_validation(path_b, HIERARCHY, 3)["aggregated_validation"]
    rule = {"mode": "heron_hierarchical", "hierarchy": HIERARCHY}
    ranked = rank_validation_results(
        [
            {"candidate_id": "a", "reward_spec_id": "a", "validation": a},
            {"candidate_id": "b", "reward_spec_id": "b", "validation": b},
        ],
        rule=rule,
    )
    assert ranked[0]["candidate_id"] == "b"


def test_completion_priority_and_standard_error_trace():
    left = metrics(completion=1.0, expiration=0.9, delivery=1.0, std=0.3)
    right = metrics(completion=0.5, expiration=0.0, delivery=100.0, std=0.3)
    result, trace = compare_heron_results(left, right, HIERARCHY)
    assert result == 1
    assert trace[-1]["metric"] == "completion_rate"
    assert trace[-1]["pooled_standard_error"] == pytest.approx(
        math.sqrt(0.3**2 / 3 + 0.3**2 / 3)
    )
    assert "pooled_uncertainty" not in trace[-1]


def test_margin_can_continue_to_min_direction_and_tie_breaker_is_stable():
    result, trace = compare_heron_results(
        metrics(completion=0.72, expiration=0.1, std=0.1),
        metrics(completion=0.70, expiration=0.3, std=0.1),
        HIERARCHY,
    )
    assert result == 1
    assert trace[0]["decision"] == "equivalent"
    assert trace[-1]["metric"] == "expiration_rate"
    equal = metrics()
    result, trace = compare_heron_results(
        equal,
        equal,
        HIERARCHY,
        "b",
        "a",
        "same",
        "same",
    )
    assert result == -1
    assert trace[-1]["reason"] == "reward_spec_id_then_candidate_id"


def test_missing_or_insufficient_tail_episode_fails_closed(tmp_path):
    path = tmp_path / "learning_curve.json"
    path.write_text(json.dumps([curve_point(0, 0.5), curve_point(1, 0.5)]), encoding="utf-8")
    result = aggregate_tail_validation(path, HIERARCHY, tail_episodes=3)
    assert result["eligible"] is False
    assert result["reason"] == "insufficient_complete_tail_episodes"


def test_missing_eligibility_validation_field_names_candidate(tmp_path):
    path = tmp_path / "learning_curve.json"
    points = [curve_point(index, 0.5) for index in range(3)]
    for point in points:
        point.pop("accepted_sgl_count_mean")
    path.write_text(json.dumps(points), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate_x.*accepted_sgl_count_mean"):
        aggregate_tail_validation(path, HIERARCHY, 3, candidate_id="candidate_x")
