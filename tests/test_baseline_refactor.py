"""验证基线语义化重构、日志兼容、注册与统一编排边界。"""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import pytest

from baselines.live_api_confirmation import (
    LiveApiPlan,
    confirm_live_api_call,
)
from baselines.llm_schema import default_mock_specs
from baselines.log_schema import (
    build_episode_log_record,
    build_update_log_record,
    normalize_episode_log_record,
)
from baselines.methods import (
    DEFAULT_METHOD_ORDER,
    BaselineMethod,
    parse_baseline_methods,
)
from baselines.orchestrator import BaselineOrchestrator
from baselines.reward_spec_registry import register_reward_spec
from mappo.manual_reward import ManualReward


class _InteractiveInput(io.StringIO):
    """允许单元测试模拟可交互终端，不接触真实标准输入。"""

    def isatty(self):
        return True


class _NonInteractiveInput(io.StringIO):
    def isatty(self):
        return False


def _api_plan(tmp_path):
    return LiveApiPlan(
        model="deepseek-reasoner",
        rounds=1,
        candidates_per_round=2,
        candidate_training_episodes=1,
        maximum_api_calls=2,
        maximum_input_tokens=100,
        maximum_output_tokens=100,
        output_directory=str(tmp_path),
    )


def _manual_reward():
    return ManualReward(
        {
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
            "normalization": {
                "balance_epsilon": 1.0e-8,
            },
            "coordination_violation_codes": [],
            "numerical": {
                "warning_abs_reward": 2.0,
                "hard_failure_abs_reward": 10.0,
            },
        }
    )


def _write_fake_summary(output_directory, episodes):
    output = Path(output_directory)
    summary = {
        "target_episode_count": episodes,
        "environment_steps_this_run": episodes * 3,
        "best_episode_index": 0,
        "best_validation_result": {
            "timeliness_raw_mean": 1.0,
            "load_balance_mean_per_task_mean": 0.5,
            "completed_task_count_mean": 1.0,
            "expired_task_count_mean": 0.0,
            "delivered_data_mbit_mean": 10.0,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "last_checkpoint.pt").write_bytes(b"mock checkpoint")


def test_log_schema_has_explicit_training_reward_and_rejects_invalid_values():
    reward = _manual_reward()
    episode = build_episode_log_record(
        {"total_manual_reward": 99.0},
        reward,
        1.25,
        {},
    )
    update = build_update_log_record({}, reward, 1.25, {})
    assert "total_manual_reward" not in episode
    assert episode["total_training_reward"] == pytest.approx(1.25)
    assert update["total_training_reward"] == pytest.approx(1.25)
    assert episode["log_schema_version"] == "2.1"
    with pytest.raises(ValueError, match="NaN|Inf"):
        build_episode_log_record({}, reward, float("nan"), {})


@pytest.mark.parametrize(
    "legacy,expected_method,expected_base,expected_shaping",
    [
        (
            {"method": "manual_mappo", "total_manual_reward": 2.0},
            "manual_reward",
            2.0,
            0.0,
        ),
        (
            {
                "method": "ppo_lya",
                "total_manual_reward": 3.0,
                "manual_reward_sum": 2.0,
                "lyapunov_shaping_sum": 1.0,
            },
            "manual_plus_lyapunov",
            2.0,
            1.0,
        ),
        (
            {"method": "llm_ppo", "total_manual_reward": 4.0},
            "llm_weight_reward",
            4.0,
            0.0,
        ),
    ],
)
def test_legacy_log_normalization_is_read_only(
    legacy,
    expected_method,
    expected_base,
    expected_shaping,
):
    original = deepcopy(legacy)
    normalized = normalize_episode_log_record(legacy)
    assert legacy == original
    assert normalized["reward_method"] == expected_method
    assert normalized["base_reward_sum"] == pytest.approx(expected_base)
    assert normalized["shaping_reward_sum"] == pytest.approx(expected_shaping)
    assert normalized["log_schema_version"] == "2.1"


def test_legacy_log_rejects_inconsistent_reward_decomposition():
    with pytest.raises(ValueError, match="不一致"):
        normalize_episode_log_record(
            {
                "method": "ppo_lya",
                "total_manual_reward": 4.0,
                "manual_reward_sum": 2.0,
                "lyapunov_shaping_sum": 1.0,
            }
        )


def test_log_schema_20_is_read_as_21_without_fabricated_diagnostics():
    record = {
        "log_schema_version": "2.0",
        "total_training_reward": 1.0,
        "base_reward_sum": 1.0,
        "shaping_reward_sum": 0.0,
    }
    normalized = normalize_episode_log_record(record)
    assert normalized["source_log_schema_version"] == "2.0"
    assert normalized["normalized_log_schema_version"] == "2.1"
    assert "reward_component_abs_sums" not in normalized


def test_reward_spec_registration_is_atomic_and_does_not_change_source(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / "registry" / "selected_reward_spec.json"
    source.write_text(
        json.dumps(default_mock_specs()[0].to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    before = source.read_bytes()
    metadata = register_reward_spec(source, destination)
    assert source.read_bytes() == before
    assert metadata["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert destination.is_file()
    with pytest.raises(FileExistsError):
        register_reward_spec(source, destination)
    register_reward_spec(source, destination, force=True)


def test_invalid_reward_spec_is_rejected_without_output(tmp_path):
    source = tmp_path / "invalid.json"
    destination = tmp_path / "selected.json"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        register_reward_spec(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("answer", ["yes\n", "Y\n", "\n"])
def test_live_api_confirmation_rejects_everything_except_exact_yes(
    answer,
    tmp_path,
):
    output = io.StringIO()
    with pytest.raises(RuntimeError, match="已取消"):
        confirm_live_api_call(
            _api_plan(tmp_path),
            input_stream=_InteractiveInput(answer),
            output_stream=output,
        )
    assert "已取消真实API调用" in output.getvalue()


def test_live_api_confirmation_accepts_exact_yes_and_rejects_pipe(tmp_path):
    approval = confirm_live_api_call(
        _api_plan(tmp_path),
        input_stream=_InteractiveInput("YES\n"),
        output_stream=io.StringIO(),
    )
    assert approval.approved
    with pytest.raises(RuntimeError, match="非交互"):
        confirm_live_api_call(
            _api_plan(tmp_path),
            input_stream=_NonInteractiveInput("YES\n"),
            output_stream=io.StringIO(),
        )


def test_method_parser_preserves_order_and_rejects_duplicates():
    assert parse_baseline_methods(["ppo_lya", "manual_mappo"]) == [
        BaselineMethod.PPO_LYA,
        BaselineMethod.MANUAL_MAPPO,
    ]
    assert parse_baseline_methods(["all"]) == list(DEFAULT_METHOD_ORDER)
    with pytest.raises(ValueError, match="重复"):
        parse_baseline_methods(["ppo_lya", "ppo_lya"])


def test_orchestrator_updates_manifest_and_creates_comparison(tmp_path):
    calls = []

    def executor(**kwargs):
        calls.append(kwargs)
        _write_fake_summary(kwargs["output_directory"], kwargs["episodes"])
        return {
            "total_environment_steps": kwargs["episodes"] * 3,
            "reward_method": kwargs["method"].value,
            "reward_spec_id": None,
        }

    run_directory = tmp_path / "run"
    orchestrator = BaselineOrchestrator(
        method_executor=executor,
        device="cpu",
        seed=123,
    )
    methods = [BaselineMethod.PPO_LYA, BaselineMethod.MANUAL_MAPPO]
    result_directory, summary, comparison = orchestrator.run(
        methods,
        episodes=2,
        skip_validation=True,
        explicit_run_directory=run_directory,
    )
    assert result_directory == run_directory.resolve()
    assert [item["method"] for item in comparison["methods"]] == [
        "ppo_lya",
        "manual_mappo",
    ]
    assert [call["method"] for call in calls] == methods
    assert all(call["episodes"] == 2 for call in calls)
    assert summary["status"] == "completed"
    manifest = json.loads(
        (run_directory / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["seed"] == 123
    assert all(
        state["status"] == "completed"
        for state in manifest["method_states"].values()
    )
    with pytest.raises(FileExistsError):
        orchestrator.run(
            [BaselineMethod.MANUAL_MAPPO],
            episodes=1,
            explicit_run_directory=run_directory,
        )


def test_orchestrator_failure_policy_and_resume_skip(tmp_path):
    calls = []

    def executor(**kwargs):
        calls.append(kwargs["method"])
        if kwargs["method"] == BaselineMethod.PPO_LYA:
            raise RuntimeError("mock failure")
        _write_fake_summary(kwargs["output_directory"], kwargs["episodes"])
        return {
            "total_environment_steps": 1,
            "reward_method": kwargs["method"].value,
            "reward_spec_id": None,
        }

    orchestrator = BaselineOrchestrator(
        method_executor=executor,
        device="cpu",
    )
    run_directory = tmp_path / "continue"
    _, summary, _ = orchestrator.run(
        [BaselineMethod.PPO_LYA, BaselineMethod.MANUAL_MAPPO],
        episodes=1,
        continue_on_error=True,
        explicit_run_directory=run_directory,
    )
    assert calls == [BaselineMethod.PPO_LYA, BaselineMethod.MANUAL_MAPPO]
    assert summary["status"] == "completed_with_errors"

    completed_run = tmp_path / "resume"
    skip_calls = []
    resume_paths = []

    def successful_executor(**kwargs):
        skip_calls.append(kwargs["method"])
        resume_paths.append(kwargs["resume_checkpoint"])
        _write_fake_summary(kwargs["output_directory"], kwargs["episodes"])
        return {
            "total_environment_steps": 1,
            "reward_method": kwargs["method"].value,
            "reward_spec_id": None,
        }

    successful = BaselineOrchestrator(
        method_executor=successful_executor,
        device="cpu",
    )
    successful.run(
        [BaselineMethod.MANUAL_MAPPO],
        episodes=1,
        explicit_run_directory=completed_run,
    )
    skip_calls.clear()
    successful.run(
        [BaselineMethod.MANUAL_MAPPO],
        episodes=1,
        resume_run=completed_run,
    )
    assert skip_calls == []

    manifest_path = completed_run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = manifest["method_states"]["manual_mappo"]
    state["status"] = "running"
    state["completed_episode_count"] = 0
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    successful.run(
        [BaselineMethod.MANUAL_MAPPO],
        episodes=1,
        resume_run=completed_run,
    )
    assert skip_calls == [BaselineMethod.MANUAL_MAPPO]
    assert resume_paths[-1] == completed_run / "manual_mappo" / "last_checkpoint.pt"


def test_orchestrator_stops_by_default_and_rejects_changed_config(tmp_path):
    calls = []

    def failing_executor(**kwargs):
        calls.append(kwargs["method"])
        raise RuntimeError("stop now")

    orchestrator = BaselineOrchestrator(
        method_executor=failing_executor,
        device="cpu",
    )
    run_directory = tmp_path / "stop"
    with pytest.raises(RuntimeError, match="stop now"):
        orchestrator.run(
            [BaselineMethod.PPO_LYA, BaselineMethod.MANUAL_MAPPO],
            episodes=1,
            explicit_run_directory=run_directory,
        )
    assert calls == [BaselineMethod.PPO_LYA]

    manifest_path = run_directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_sha256"] = "changed"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="配置已变化"):
        orchestrator.run(
            [BaselineMethod.PPO_LYA, BaselineMethod.MANUAL_MAPPO],
            episodes=1,
            resume_run=run_directory,
        )


def test_orchestrator_rejects_changed_reward_spec_on_resume(tmp_path):
    first_spec = tmp_path / "first.json"
    second_spec = tmp_path / "second.json"
    first_spec.write_text(
        json.dumps(default_mock_specs()[0].to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    second_spec.write_text(
        json.dumps(default_mock_specs()[1].to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )

    def executor(**kwargs):
        _write_fake_summary(kwargs["output_directory"], kwargs["episodes"])
        return {
            "total_environment_steps": 1,
            "reward_method": "llm_weight_reward",
            "reward_spec_id": kwargs["reward_spec"].spec_id,
        }

    run_directory = tmp_path / "llm_run"
    orchestrator = BaselineOrchestrator(
        method_executor=executor,
        device="cpu",
    )
    orchestrator.run(
        [BaselineMethod.LLM_PPO],
        episodes=1,
        reward_spec_path=first_spec,
        explicit_run_directory=run_directory,
    )
    with pytest.raises(ValueError, match="奖励规范已变化"):
        orchestrator.run(
            [BaselineMethod.LLM_PPO],
            episodes=2,
            reward_spec_path=second_spec,
            resume_run=run_directory,
        )


def test_cli_has_no_silent_api_confirmation_bypass():
    scripts = (
        "train_baselines.py",
        "generate_llm_reward_candidates.py",
        "run_llm_ppo_search.py",
    )
    for name in scripts:
        source = (Path("scripts") / name).read_text(encoding="utf-8")
        assert "--yes" not in source


def test_active_code_has_no_legacy_stage_names_or_paths():
    """动态构造旧标识，避免验收测试自身成为扫描命中项。"""
    legacy_lower = "day" + str(4)
    legacy_title = legacy_lower.title()
    legacy_chinese = "".join(chr(code) for code in (31532, 22235, 22825))
    assert not (Path("src") / legacy_lower).exists()
    assert not (Path("configs") / (legacy_lower + "_baselines.yaml")).exists()
    assert not (
        Path("scripts") / ("train_" + legacy_lower + "_baseline.py")
    ).exists()
    assert not (
        Path("scripts") / ("evaluate_" + legacy_lower + "_baseline.py")
    ).exists()

    paths = []
    for root in ("configs", "src", "scripts", "tests"):
        paths.extend(
            path
            for path in Path(root).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".yaml", ".yml", ".toml", ".md"}
        )
    paths.extend((Path("README.md"), Path("pyproject.toml")))
    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert legacy_lower not in content, path
        assert legacy_title not in content, path
        assert legacy_chinese not in content, path
