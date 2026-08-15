"""验证ORSO配置、D3RB公式和与既有直接LLM奖励的语义边界。"""

from copy import deepcopy
import math

import pytest

from baselines.llm_reward import reward_spec_weights
from baselines.llm_schema import default_mock_specs
from orso.config import load_orso_config, validate_orso_config
from orso.d3rb import D3RBSelector
from orso.search import (
    _rank_tail_validations,
    _task_utility,
    build_warmup_schedule,
)


def _validation(completion):
    return {
        "completion_rate_mean": completion,
        "delivered_data_mbit_mean": 1.0,
        "load_balance_mean_per_task_mean": 0.0,
    }


def test_d3rb_initialization_uses_d_min_for_every_candidate():
    selector = D3RBSelector(("candidate_01", "candidate_02"), 0.1, 0.05, 0.1)
    for state in selector.states.values():
        assert state.n == 0
        assert state.u_hat == 0.0
        assert state.d_hat == pytest.approx(0.1)
        assert state.phi == pytest.approx(0.1)


def test_d3rb_selects_minimum_phi_then_episode_count_then_candidate_id():
    selector = D3RBSelector(("candidate_02", "candidate_01"), 0.1, 0.05, 0.1)
    assert selector.select(3) == "candidate_01"
    selector.states["candidate_01"].episodes_trained = 1
    assert selector.select(3) == "candidate_02"
    selector.states["candidate_02"].phi = 0.2
    assert selector.select(3) == "candidate_01"
    selector.states["candidate_01"].episodes_trained = 3
    assert selector.select(3) == "candidate_02"


def test_d3rb_update_matches_paper_confidence_formula_without_trigger():
    selector = D3RBSelector(("candidate_01", "candidate_02"), 0.1, 0.5, 0.1)
    result = selector.update("candidate_01", 0.9, _validation(0.9))
    expected_confidence = 0.1 * math.sqrt(math.log(2) * math.log(1 / 0.5))
    state = selector.states["candidate_01"]
    assert result["confidence_term"] == pytest.approx(expected_confidence)
    assert result["misspecification_triggered"] is False
    assert state.n == 1
    assert state.u_hat == pytest.approx(0.9)
    assert state.mean_task_reward == pytest.approx(0.9)
    assert state.d_hat == pytest.approx(0.1)
    assert state.phi == pytest.approx(0.1)


def test_d3rb_doubles_regret_coefficient_when_misspecification_triggers():
    selector = D3RBSelector(("candidate_01", "candidate_02"), 0.1, 0.5, 0.1)
    selector.update("candidate_02", 1.0, _validation(1.0))
    result = selector.update("candidate_01", 0.0, _validation(0.0))
    state = selector.states["candidate_01"]
    assert result["misspecification_triggered"] is True
    assert state.d_hat == pytest.approx(0.2)
    assert state.phi == pytest.approx(0.2)
    assert state.misspecification_count == 1


def test_first_warmup_round_defers_misspecification_until_all_candidates_observed():
    candidate_ids = tuple(
        "candidate_{0:02d}".format(index)
        for index in range(1, 9)
    )
    selector = D3RBSelector(candidate_ids, 0.1, 0.5, 0.1)
    for index, candidate_id in enumerate(candidate_ids):
        result = selector.update(
            candidate_id,
            0.9 if candidate_id == "candidate_02" else 0.0,
            _validation(0.9 if candidate_id == "candidate_02" else 0.0),
            enable_misspecification=False,
        )
        assert result["misspecification_triggered"] is False
        assert result["confidence_term"] is None
        assert selector.states[candidate_id].d_hat == pytest.approx(0.1)
    assert all(state.n == 1 for state in selector.states.values())

    second_round = selector.update(
        "candidate_01",
        0.0,
        _validation(0.0),
        enable_misspecification=True,
    )
    assert second_round["misspecification_triggered"] is True
    assert selector.states["candidate_01"].d_hat == pytest.approx(0.2)


@pytest.mark.parametrize(
    "path,value",
    [
        (("generation", "candidates"), 0),
        (("training", "warmup_episodes_per_candidate"), 0),
        (("training", "allocation_quantum_episodes"), 2),
        (("training", "total_candidate_episode_budget"), 1),
        (("training", "max_episodes_per_candidate"), 2),
        (("training", "max_episodes_per_candidate"), 9),
        (("final_selection", "tail_episodes"), 2),
        (("d3rb", "d_min"), 0.0),
        (("d3rb", "delta"), 0.0),
        (("d3rb", "delta"), 1.0),
        (("d3rb", "confidence_constant"), 0.0),
    ],
)
def test_orso_invalid_project_config_fails(path, value):
    config = deepcopy(load_orso_config())
    config[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        validate_orso_config(config)


def test_warmup_counts_against_budget_and_uses_round_robin_order():
    candidates = ("candidate_01", "candidate_02", "candidate_03")
    schedule = build_warmup_schedule(candidates, 2)
    assert schedule == [
        "candidate_01", "candidate_02", "candidate_03",
        "candidate_01", "candidate_02", "candidate_03",
    ]
    assert len(schedule) == len(candidates) * 2


def test_task_utility_uses_completion_not_training_reward():
    utility = {"primary_metric": "completion_rate_mean", "valid_min": 0.0, "valid_max": 1.0}
    validation = _validation(0.75)
    validation["mean_step_reward"] = -999.0
    assert _task_utility("candidate_01", validation, utility) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="completion_rate_mean"):
        _task_utility("candidate_01", {"mean_step_reward": 999.0}, utility)


def test_orso_search_protocol_cannot_use_checkpoint_selection_or_test():
    config = deepcopy(load_orso_config())
    config["training"]["evaluation_protocol"] = "checkpoint_selection"
    with pytest.raises(ValueError, match="reward_search"):
        validate_orso_config(config)


def test_final_selection_uses_tail_three_average_not_last_episode():
    ranked, selection = _rank_tail_validations(
        {
            # candidate_01的最后一个Episode更好，但其tail-3均值明显更差。
            "candidate_01": [
                _validation(0.1),
                _validation(0.1),
                _validation(0.1),
                _validation(0.99),
            ],
            "candidate_02": [
                _validation(0.6),
                _validation(0.6),
                _validation(0.6),
                _validation(0.8),
            ],
        },
        3,
    )
    assert ranked[0]["candidate_id"] == "candidate_02"
    assert selection["candidate_01"]["tail3_validation"][
        "completion_rate_mean"
    ] == pytest.approx((0.1 + 0.1 + 0.99) / 3)
    assert selection["candidate_02"]["final_selection_window"] == {
        "start_episode": 2,
        "end_episode": 4,
        "size": 3,
    }


def test_orso_uses_existing_direct_llm_reward_semantics():
    spec = default_mock_specs()[0]
    weights = reward_spec_weights(spec, target_l1=2.32)
    assert weights["coordination_conflict"] == 0.0
    assert sum(abs(value) for value in weights.values()) == pytest.approx(2.32)
