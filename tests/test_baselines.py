"""验证PPO-Lya、LLM奖励安全层、缓存预算和公平训练链路。"""

from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from baselines.baseline_runner import build_baseline_components, make_runner
from baselines.candidate_search import (
    CachedRewardGenerator,
    GenerationBudget,
    candidate_cache_key,
    diagnose_reward_spec,
)
from baselines.config import load_baseline_config, validate_baseline_config
from baselines.live_api_confirmation import LiveApiApproval
from baselines.llm_prompt import (
    build_feedback_reward_prompt,
    build_initial_reward_prompt,
)
from baselines.llm_provider import (
    DeepSeekRewardGenerationProvider,
    FatalProviderError,
    MockRewardGenerationProvider,
    RetryableProviderError,
)
from baselines.llm_reward import LlmWeightReward
from baselines.llm_schema import LlmRewardSpec, default_mock_specs
from baselines.lyapunov_reward import (
    PpoLyaReward,
    extract_lyapunov_features,
    lyapunov_potential,
)
from mappo.config import load_mappo_config
from mappo.manual_reward import RewardFeatures, combine_manual_reward
from mappo.trainer import parameter_vector
from srs_env.models import TaskDefinition, TaskState, TaskStatus


@pytest.fixture(scope="module")
def configs():
    return load_baseline_config(), load_mappo_config()


@pytest.fixture
def valid_spec():
    return default_mock_specs()[0]


def _spec_dict():
    return default_mock_specs()[0].to_dict()


def _dummy_environment():
    task = TaskDefinition(
        task_id="task_test",
        source_satellite_id="os01",
        target_ground_station_id="gs01",
        priority=10,
        data_size_mbit=100.0,
        survival_time_s=100.0,
        arrival_time_s=0.0,
        expiration_time_s=100.0,
        nominal_sgl_duration_s=10.0,
    )
    state = TaskState(
        task,
        TaskStatus.ACTIVE,
        np.array([100.0] + [0.0] * 14),
    )
    return SimpleNamespace(
        tasks={task.task_id: state},
        task_index={task.task_id: 0},
        current_time_s=0.0,
        outgoing_seconds=np.zeros((1, 15)),
        total_window_seconds=np.ones(15) * 100.0,
    )


class _FixedManualReward:
    def __init__(self, value=0.25, features=None, numerical=None):
        self.value = value
        self.features = features or RewardFeatures(*(0.0 for _ in range(8)))
        self.config = {
            "weights": {
                "sgl_progress": 1.0,
                "relay_progress": 0.15,
                "completion": 0.5,
                "balance": 0.05,
                "expiration": 0.5,
                "invalid_action": 0.1,
                "coordination_conflict": 0.03,
                "relay_cost": 0.02,
            },
            "numerical": numerical
            or {"warning_abs_reward": 2.0, "hard_failure_abs_reward": 10.0},
        }
        self.warning_count = 0

    def reset(self, environment):
        del environment

    def compute(self, environment, info):
        del environment, info
        result = combine_manual_reward(
            self.features,
            self.config["weights"],
            self.config["numerical"],
        )
        if self.value == result.total_reward:
            return result
        return SimpleNamespace(total_reward=self.value, features=self.features)


def _generator(tmp_path, responses, configs, max_calls=4, max_input=100000):
    baseline_config, _ = configs
    llm = deepcopy(baseline_config["methods"]["llm_ppo"])
    llm["cache"]["directory"] = str(tmp_path / "cache")
    provider_config = deepcopy(llm["provider"])
    provider_config["max_retries"] = 2
    provider_config["retry_delays_seconds"] = [0, 0]
    provider = MockRewardGenerationProvider(responses)
    budget = GenerationBudget(max_calls, max_input, 20000)
    generator = CachedRewardGenerator(
        provider,
        provider_config,
        llm,
        budget,
        sleep=lambda _: None,
    )
    return generator, provider, budget


def test_01_config_is_valid(configs):
    validate_baseline_config(*configs)


@pytest.mark.parametrize(
    "path,value,match",
    [
        (("training", "split"), "test", "train划分"),
        (("training", "validation", "split"), "test", "validation划分"),
        (("methods", "llm_ppo", "provider", "default"), "deepseek", "mock"),
        (("methods", "llm_ppo", "provider", "base_url"), "http://x", "HTTPS"),
        (("methods", "llm_ppo", "provider", "api_key_env"), "sk-secret", "变量名"),
    ],
)
def test_02_to_06_invalid_config_rejected(configs, path, value, match):
    config = deepcopy(configs[0])
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        validate_baseline_config(config, configs[1])


def test_07_lyapunov_backlog(configs):
    environment = _dummy_environment()
    features = extract_lyapunov_features(
        environment,
        configs[0]["methods"]["ppo_lya"]["lyapunov"],
    )
    assert features.backlog == pytest.approx(1.0)


def test_08_lyapunov_expiration_risk(configs):
    environment = _dummy_environment()
    environment.current_time_s = 50.0
    features = extract_lyapunov_features(
        environment,
        configs[0]["methods"]["ppo_lya"]["lyapunov"],
    )
    assert features.expiration_risk == pytest.approx(0.25)


def test_09_lyapunov_imbalance(configs):
    environment = _dummy_environment()
    environment.outgoing_seconds[0, 0] = 50.0
    features = extract_lyapunov_features(
        environment,
        configs[0]["methods"]["ppo_lya"]["lyapunov"],
    )
    expected = np.std(np.array([0.5] + [0.0] * 14))
    assert features.utilization_imbalance == pytest.approx(expected)


def test_10_no_active_task_has_zero_backlog_and_risk(configs):
    environment = _dummy_environment()
    environment.tasks["task_test"].status = TaskStatus.COMPLETED
    features = extract_lyapunov_features(
        environment,
        configs[0]["methods"]["ppo_lya"]["lyapunov"],
    )
    assert features.backlog == 0.0
    assert features.expiration_risk == 0.0


def test_11_potential_formula(configs):
    environment = _dummy_environment()
    config = configs[0]["methods"]["ppo_lya"]["lyapunov"]
    features = extract_lyapunov_features(environment, config)
    assert lyapunov_potential(features, config["feature_weights"]) == pytest.approx(0.45)


def test_12_state_improvement_shaping_positive(configs):
    environment = _dummy_environment()
    reward = PpoLyaReward(
        _FixedManualReward(),
        configs[0]["methods"]["ppo_lya"]["lyapunov"],
    )
    reward.reset(environment)
    environment.tasks["task_test"].delivered_to_ground_mbit = 50.0
    result = reward.compute(environment, {})
    assert result.shaping_reward > 0.0


def test_13_state_worsening_shaping_negative(configs):
    environment = _dummy_environment()
    reward = PpoLyaReward(
        _FixedManualReward(),
        configs[0]["methods"]["ppo_lya"]["lyapunov"],
    )
    reward.reset(environment)
    environment.current_time_s = 99.0
    result = reward.compute(environment, {})
    assert result.shaping_reward < 0.0


def test_14_ppo_lya_total_is_manual_plus_shaping(configs):
    environment = _dummy_environment()
    reward = PpoLyaReward(
        _FixedManualReward(0.25),
        configs[0]["methods"]["ppo_lya"]["lyapunov"],
    )
    reward.reset(environment)
    result = reward.compute(environment, {})
    assert result.total_reward == pytest.approx(0.25 + result.shaping_reward)


def test_15_ppo_lya_reset_replaces_previous_state(configs):
    environment = _dummy_environment()
    reward = PpoLyaReward(
        _FixedManualReward(),
        configs[0]["methods"]["ppo_lya"]["lyapunov"],
    )
    reward.reset(environment)
    first = reward.episode_initial_potential
    environment.tasks["task_test"].delivered_to_ground_mbit = 50.0
    reward.reset(environment)
    assert reward.episode_initial_potential < first


def test_16_terminal_uses_real_next_potential(configs):
    environment = _dummy_environment()
    reward = PpoLyaReward(
        _FixedManualReward(0.0),
        configs[0]["methods"]["ppo_lya"]["lyapunov"],
    )
    reward.reset(environment)
    environment.tasks["task_test"].status = TaskStatus.COMPLETED
    result = reward.compute(environment, {"terminated": True})
    assert result.current_potential == 0.0
    assert result.shaping_reward == pytest.approx(
        0.2 * result.previous_potential
    )


def test_17_lyapunov_values_are_finite(configs):
    environment = _dummy_environment()
    features = extract_lyapunov_features(
        environment,
        configs[0]["methods"]["ppo_lya"]["lyapunov"],
    )
    assert all(math.isfinite(value) for value in features.__dict__.values())


def test_18_valid_schema_parses(valid_spec):
    parsed = LlmRewardSpec.from_json(json.dumps(valid_spec.to_dict(), ensure_ascii=False))
    assert parsed.spec_id == valid_spec.spec_id


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda data: data.pop("rationale"), "字段"),
        (lambda data: data.update({"unknown": 1}), "字段"),
        (lambda data: data.update({"schema_version": "2.0"}), "2.1"),
        (lambda data: data["positive_weights"].update({"sgl_progress": 4.0}), "越界"),
        (lambda data: data["positive_weights"].update({"sgl_progress": float("nan")}), "有限"),
        (lambda data: data.update({"rationale": "```python"}), "代码"),
        (lambda data: data.update({"rationale": "import os"}), "代码"),
        (lambda data: data.update({"rationale": "def reward(): pass"}), "代码"),
        (lambda data: data.update({"rationale": "C:\\secret\\key.txt"}), "路径"),
        (lambda data: data["positive_weights"].pop("balance_score"), "完整"),
    ],
)
def test_19_to_28_invalid_schema_rejected(mutation, match):
    data = _spec_dict()
    mutation(data)
    with pytest.raises(ValueError, match=match):
        LlmRewardSpec.from_dict(data)


def test_29_llm_reward_reuses_features(valid_spec):
    features = RewardFeatures(0.1, 0.2, 0.3, 0.0, 0.1, 0.0, 0.0, 0.2)
    extractor = _FixedManualReward(features=features)
    reward = LlmWeightReward(extractor, valid_spec)
    reward.reset(object())
    result = reward.compute(object(), {})
    assert result.features is features
    assert result.reward_spec_id == valid_spec.spec_id


@pytest.mark.parametrize(
    "feature_index,sign",
    [(0, 1), (2, 1), (4, -1), (5, -1), (7, -1)],
)
def test_30_to_35_llm_reward_directions(valid_spec, feature_index, sign):
    values = [0.0] * 8
    values[feature_index] = 0.1
    extractor = _FixedManualReward(features=RewardFeatures(*values))
    reward = LlmWeightReward(extractor, valid_spec)
    reward.reset(object())
    assert math.copysign(1.0, reward.compute(object(), {}).total_reward) == sign


def test_36_candidate_diagnostics_accept(valid_spec, configs):
    result = diagnose_reward_spec(
        valid_spec,
        configs[1]["manual_reward"]["numerical"],
        configs[0]["methods"]["llm_ppo"]["l1_target_scale"],
    )
    assert result["status"] == "accepted_for_training"


def test_37_candidate_diagnostics_accepts_fixed_zero_conflict(configs):
    data = _spec_dict()
    data["penalty_weights"]["coordination_conflict_rate"] = 0.0
    result = diagnose_reward_spec(
        LlmRewardSpec.from_dict(data),
        configs[1]["manual_reward"]["numerical"],
        configs[0]["methods"]["llm_ppo"]["l1_target_scale"],
    )
    assert result["status"] == "accepted_for_training"


def test_38_mock_valid_response(tmp_path, configs):
    generator, provider, budget = _generator(tmp_path, [_spec_dict()], configs)
    spec, audit = generator.generate("输出JSON候选A")
    assert spec.reward_name
    assert provider.call_count == budget.api_calls == 1
    assert audit["cache_hit"] is False


def test_39_mock_empty_response_retries(tmp_path, configs):
    generator, provider, _ = _generator(
        tmp_path,
        ["", _spec_dict()],
        configs,
    )
    _, audit = generator.generate("输出JSON候选B")
    assert provider.call_count == 2
    assert audit["retry_count"] == 1


def test_40_mock_timeout_has_finite_retries(tmp_path, configs):
    generator, provider, _ = _generator(
        tmp_path,
        [RetryableProviderError("timeout")] * 3,
        configs,
    )
    with pytest.raises(RuntimeError, match="有限重试"):
        generator.generate("输出JSON候选C")
    assert provider.call_count == 3


def test_41_cache_hit_does_not_call_provider(tmp_path, configs):
    generator, provider, budget = _generator(tmp_path, [_spec_dict()], configs)
    generator.generate("输出JSON候选D")
    _, audit = generator.generate("输出JSON候选D")
    assert audit["cache_hit"] is True
    assert provider.call_count == budget.api_calls == 1


def test_42_prompt_change_changes_cache_key():
    first = candidate_cache_key("mock", "m", {}, "JSON A")
    second = candidate_cache_key("mock", "m", {}, "JSON B")
    assert first != second


def test_42a_candidate_identity_prevents_round_cache_sharing(tmp_path, configs):
    """同一父候选下，2轮×3候选仍必须产生六次独立Mock生成。"""
    generator, provider, _ = _generator(
        tmp_path,
        [_spec_dict() for _ in range(6)],
        configs,
        max_calls=6,
    )
    llm = configs[0]["methods"]["llm_ppo"]
    prompts, audits = [], []
    for round_index in (1, 2):
        for candidate_index in range(1, 4):
            candidate_id = "round_{0:02d}_candidate_{1:02d}".format(
                round_index,
                candidate_index,
            )
            args = (
                configs[0]["training"]["task_count"],
                llm["l1_target_scale"],
                round_index,
                candidate_index,
                3,
                candidate_id,
            )
            prompt = (
                build_initial_reward_prompt(*args)
                if round_index == 1
                else build_feedback_reward_prompt(
                    *args,
                    "round_01_candidate_01",
                    {"last2_validation": {"completion_rate_mean": 0.5}},
                )
            )
            _, audit = generator.generate(
                prompt,
                {
                    "task_count": 150,
                    "round_index": round_index,
                    "candidate_index": candidate_index,
                    "candidate_id": candidate_id,
                },
            )
            prompts.append(prompt)
            audits.append(audit)
    assert len(set(prompts)) == 6
    assert len({audit["call_id"] for audit in audits}) == 6
    assert all(audit["cache_hit"] is False for audit in audits)
    assert provider.call_count == 6
    _, repeated = generator.generate(
        prompts[-1],
        {"task_count": 150, "round_index": 2, "candidate_index": 3, "candidate_id": "round_02_candidate_03"},
    )
    assert repeated["cache_hit"] is True
    assert provider.call_count == 6


def test_43_call_budget_stops_generation(tmp_path, configs):
    generator, _, _ = _generator(tmp_path, [_spec_dict()], configs, max_calls=0)
    with pytest.raises(RuntimeError, match="调用预算"):
        generator.generate("输出JSON候选E")


def test_44_input_budget_stops_generation(tmp_path, configs):
    generator, _, _ = _generator(tmp_path, [_spec_dict()], configs, max_input=1)
    with pytest.raises(RuntimeError, match="输入Token预算"):
        generator.generate("这是一个明显超过四个字符的JSON候选Prompt")


def test_45_live_provider_requires_approval(configs):
    provider_config = configs[0]["methods"]["llm_ppo"]["provider"]
    with pytest.raises(FatalProviderError, match="批准"):
        DeepSeekRewardGenerationProvider(provider_config, approval=None)


def test_46_live_provider_requires_key(configs, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    provider_config = configs[0]["methods"]["llm_ppo"]["provider"]
    with pytest.raises(FatalProviderError, match="未设置"):
        DeepSeekRewardGenerationProvider(
            provider_config,
            approval=LiveApiApproval(
                approved=True,
                approved_at="unit-test",
            ),
        )


def test_47_prompt_contains_json_and_all_features(configs):
    prompt = build_initial_reward_prompt(
        configs[0]["training"]["task_count"],
        configs[0]["methods"]["llm_ppo"]["l1_target_scale"],
        1,
        1,
        8,
        "round_01_candidate_01",
    )
    assert "JSON" in prompt
    for name in RewardFeatures.__dataclass_fields__:
        assert name in prompt


def test_48_feedback_prompt_has_parent_without_sensitive_path(configs):
    prompt = build_feedback_reward_prompt(
        configs[0]["training"]["task_count"],
        configs[0]["methods"]["llm_ppo"]["l1_target_scale"],
        2,
        1,
        8,
        "round_02_candidate_01",
        "round_01_candidate_01",
        {"timeliness_raw_mean": 2.0},
    )
    assert "round_01_candidate_01" in prompt
    assert "DEEPSEEK_API_KEY" not in prompt
    assert "D:\\" not in prompt


def test_49_candidates_have_identical_initial_networks(configs, valid_spec):
    baseline_config, mappo = configs
    first = build_baseline_components(
        "llm_ppo", baseline_config, mappo, valid_spec
    )
    second = build_baseline_components(
        "llm_ppo", baseline_config, mappo, valid_spec
    )
    assert torch.equal(parameter_vector(first[2]), parameter_vector(second[2]))
    assert torch.equal(parameter_vector(first[3]), parameter_vector(second[3]))


@pytest.mark.parametrize("method", ["ppo_lya", "llm_ppo"])
def test_50_and_51_short_training_changes_parameters_and_conserves(
    method,
    configs,
    valid_spec,
    tmp_path,
):
    baseline_config = deepcopy(configs[0])
    baseline_config["device"] = "cpu"
    spec = valid_spec if method == "llm_ppo" else None
    config, encoder, actor, _, trainer, evaluator = build_baseline_components(
        method,
        baseline_config,
        configs[1],
        spec,
    )
    before = parameter_vector(actor)
    runner = make_runner(
        method,
        trainer,
        evaluator,
        config,
        encoder,
        baseline_config,
        tmp_path / method,
    )
    summary = runner.run(
        target_episode_count=1,
        skip_validation=True,
        max_steps_per_episode=1,
    )
    assert not torch.equal(before, parameter_vector(actor))
    assert summary["data_conservation_passed"]
    assert Path(runner.training["checkpoint"]["last_path"]).exists()


def test_52_deepseek_request_uses_json_thinking_without_sampling(
    configs,
    monkeypatch,
):
    """真实Provider参数遵循Chat Completions约定且不读取推理内容。"""
    captured = {}

    class Completions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(
                content=json.dumps(_spec_dict(), ensure_ascii=False),
                reasoning_content="不得读取或保存的内部推理",
            )
            return SimpleNamespace(
                id="response-test",
                choices=[SimpleNamespace(message=message)],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=34),
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-only-key")
    provider_config = configs[0]["methods"]["llm_ppo"]["provider"]
    provider = DeepSeekRewardGenerationProvider(
        provider_config,
        approval=LiveApiApproval(
            approved=True,
            approved_at="unit-test",
        ),
        client=client,
    )
    result = provider.generate_reward_spec(
        "输出JSON",
        {"system_prompt": "只输出JSON"},
    )
    assert result.content
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["extra_body"] == {"thinking": {"type": "enabled"}}
    assert captured["reasoning_effort"] == "high"
    assert "temperature" not in captured
    assert "top_p" not in captured
    assert not hasattr(result, "reasoning_content")


def test_53_audit_does_not_store_key_or_reasoning(tmp_path, configs, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-secret-value")
    generator, _, _ = _generator(tmp_path, [_spec_dict()], configs)
    generator.generate("输出安全JSON候选")
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert "unit-test-secret-value" not in text
    assert "reasoning_content" not in text


def test_54_invalid_json_stops_after_finite_retries(tmp_path, configs):
    generator, provider, _ = _generator(
        tmp_path,
        ["{", "not-json", "[]"],
        configs,
    )
    with pytest.raises(RuntimeError, match="有限重试"):
        generator.generate("输出JSON候选F")
    assert provider.call_count == 3
