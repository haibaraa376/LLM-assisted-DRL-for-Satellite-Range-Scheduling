"""验证人工奖励、正式Episode、validation、Best与Resume语义。"""

from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from mappo.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from mappo.config import load_mappo_config, validate_mappo_config
from mappo.encoding import MappoObservationEncoder
from mappo.evaluator import MappoEvaluator, is_better_validation_result
from mappo.manual_reward import (
    ManualReward,
    RewardFeatures,
    combine_manual_reward,
)
from mappo.training_runner import BaselineTrainingRunner
from mappo.networks import CentralizedCritic, SharedActor
from mappo.trainer import MappoTrainer, parameter_vector
from mappo.utils import set_global_seed
from srs_env.config import load_environment_config
from srs_env.data import load_skyfield_dataset
from srs_env.environment import CrossDomainSatelliteRangeSchedulingEnv
from srs_env.models import TaskStatus, TransmissionRecord
from srs_env.tasks import load_task_database, load_task_splits


@pytest.fixture(scope="module")
def resources():
    """共享冻结数据和配置，测试环境状态仍逐项创建。"""
    return (
        load_mappo_config(),
        load_skyfield_dataset(),
        load_task_database(),
        load_task_splits(),
    )


def make_environment(resources, task_ids=None):
    """创建可按划分重置的环境，可选立即载入固定任务ID。"""
    _, dataset, database, splits = resources
    environment = CrossDomainSatelliteRangeSchedulingEnv(
        dataset,
        load_environment_config(),
        task_database=database,
        task_splits=splits,
    )
    if task_ids is not None:
        environment.reset(task_ids=task_ids)
    return environment


def make_record(
    task,
    source=None,
    target=None,
    link_type="SGL",
    ratio=0.1,
    accepted=True,
    codes=(),
):
    """构造奖励公式单元测试所需的传输记录。"""
    source = source or task.source_satellite_id
    target = target or task.target_ground_station_id
    amount = task.data_size_mbit * ratio if accepted else 0.0
    return TransmissionRecord(
        source_satellite_id=source,
        subaction_index=0,
        composite_source_id=source,
        target_id=target,
        task_id=task.task_id,
        link_type=link_type,
        requested_ratio=ratio,
        requested_start_s=task.arrival_time_s,
        accepted=accepted,
        transmitted_data_mbit=amount,
        actual_start_s=task.arrival_time_s if accepted else None,
        actual_end_s=task.arrival_time_s + 1.0 if accepted else None,
        rate_mbps=60.0 if link_type == "SGL" else 80.0,
        violation_codes=tuple(codes),
        projected=False,
    )


def make_info(records):
    """构造ManualReward读取的最小info。"""
    return {
        "transmission_records": list(records),
        "submitted_subaction_count": len(records),
    }


def single_task_setup(resources):
    """选择validation最高优先级任务并返回环境、状态和奖励模型。"""
    config, _, database, splits = resources
    task_id = max(
        splits["validation"],
        key=lambda item: (database[item].priority, item),
    )
    environment = make_environment(resources, [task_id])
    state = environment.tasks[task_id]
    state.status = TaskStatus.ACTIVE
    reward = ManualReward(config["manual_reward"])
    reward.reset(environment)
    return config, environment, state, reward


def test_01_manual_reward_configuration_validation(resources):
    """拒绝负权重、非正归一化、代码重叠、test划分和相同路径。"""
    config = resources[0]
    for mutation, message in (
        (lambda item: item["manual_reward"]["weights"].update(sgl_progress=-1), "非负"),
        (
            lambda item: item["manual_reward"]["normalization"].update(
                balance_epsilon=0
            ),
            "正有限",
        ),
        (
            lambda item: item["manual_reward"]["coordination_violation_codes"].append(
                "INVALID_TASK"
            ),
            "不能重叠",
        ),
        (lambda item: item["manual_training"].update(split="test"), "train划分"),
        (
            lambda item: item["manual_training"]["validation"].update(split="test"),
            "validation",
        ),
        (
            lambda item: item["manual_training"]["checkpoint"].update(
                best_path="same.pt", last_path="same.pt"
            ),
            "路径必须不同",
        ),
    ):
        broken = deepcopy(config)
        mutation(broken)
        with pytest.raises(ValueError, match=message):
            validate_mappo_config(broken)


def test_02_sgl_progress_matches_manual_formula(resources):
    """成功SGL的q×p×s除以4与手工计算一致。"""
    config, environment, state, reward = single_task_setup(resources)
    task = state.definition
    record = make_record(task, ratio=0.2)
    breakdown = reward.compute(environment, make_info([record]))
    expected = (
        0.2
        * task.priority
        / config["manual_reward"]["normalization"]["max_priority"]
        * 1.0
        / config["manual_reward"]["normalization"]["sgl_parallel_reference"]
    )
    assert np.isclose(breakdown.features.sgl_progress, expected)
    assert breakdown.total_reward > 0


def test_03_relay_progress_cost_and_net_are_weaker_than_sgl(resources):
    """同等任务比例下中继有进展和成本，净贡献小于SGL。"""
    _, environment, state, reward = single_task_setup(resources)
    task = state.definition
    relay = reward.compute(
        environment,
        make_info([make_record(task, target="cs01", link_type="IDL")]),
    )
    reward.reset(environment)
    sgl = reward.compute(environment, make_info([make_record(task)]))
    assert relay.features.relay_progress > 0
    assert relay.features.relay_cost > 0
    assert 0 < relay.total_reward < sgl.total_reward


def test_04_completion_reward_occurs_once(resources):
    """ACTIVE到COMPLETED只在状态变化后的第一步产生完成分量。"""
    _, environment, state, reward = single_task_setup(resources)
    state.delivered_to_ground_mbit = state.definition.data_size_mbit
    state.status = TaskStatus.COMPLETED
    first = reward.compute(environment, make_info([]))
    second = reward.compute(environment, make_info([]))
    assert first.features.completion_score > 0
    assert second.features.completion_score == 0


def test_05_expiration_penalty_occurs_once(resources):
    """ACTIVE到EXPIRED只惩罚一次，并按未送达比例缩小。"""
    _, environment, state, reward = single_task_setup(resources)
    state.delivered_to_ground_mbit = state.definition.data_size_mbit * 0.25
    state.status = TaskStatus.EXPIRED
    first = reward.compute(environment, make_info([]))
    second = reward.compute(environment, make_info([]))
    assert first.features.expiration_loss > 0
    assert first.total_reward < 0
    assert second.features.expiration_loss == 0


def test_06_idle_slot_has_zero_balance_and_reward(resources):
    """无传输且无状态事件时均衡项和总奖励都为0。"""
    _, environment, _, reward = single_task_setup(resources)
    breakdown = reward.compute(environment, make_info([]))
    assert breakdown.features.balance_score == 0
    assert breakdown.total_reward == 0


def test_07_zero_utilization_does_not_create_fake_balance_reward(resources):
    """执行前所有利用率为0时，第一条成功传输的均衡项仍为0。"""
    _, environment, state, reward = single_task_setup(resources)
    breakdown = reward.compute(
        environment,
        make_info([make_record(state.definition)]),
    )
    assert breakdown.features.balance_score == 0


def test_08_underloaded_and_overloaded_sources_have_opposite_balance(resources):
    """相对欠载源产生正均衡值，过载源产生负均衡值。"""
    _, environment, state, reward = single_task_setup(resources)
    task = state.definition
    task_row = environment.task_index[task.task_id]
    source = environment.dataset.satellite_index[task.source_satellite_id]
    other = (source + 1) % len(environment.dataset.satellite_ids)
    environment.outgoing_seconds[task_row, other] = 100.0
    reward.reset(environment)
    underloaded = reward.compute(environment, make_info([make_record(task)]))
    environment.outgoing_seconds[task_row].fill(0.0)
    environment.outgoing_seconds[task_row, source] = 100.0
    reward.reset(environment)
    overloaded = reward.compute(environment, make_info([make_record(task)]))
    assert underloaded.features.balance_score > 0
    assert overloaded.features.balance_score < 0


def test_09_invalid_action_rate_is_separate(resources):
    """明显无效拒绝只进入invalid_action_rate。"""
    _, environment, state, reward = single_task_setup(resources)
    record = make_record(
        state.definition,
        accepted=False,
        codes=("INVALID_TARGET",),
    )
    features = reward.compute(environment, make_info([record])).features
    assert features.invalid_action_rate == 1.0
    assert features.coordination_conflict_rate == 0.0


def test_10_coordination_conflict_rate_is_separate(resources):
    """全局任务竞争只进入coordination_conflict_rate。"""
    _, environment, state, reward = single_task_setup(resources)
    record = make_record(
        state.definition,
        accepted=False,
        codes=("TASK_ALREADY_SCHEDULED_THIS_SLOT",),
    )
    features = reward.compute(environment, make_info([record])).features
    assert features.invalid_action_rate == 0.0
    assert features.coordination_conflict_rate == 1.0


def test_11_unknown_violation_code_fails(resources):
    """未知违反代码不能被静默漏计。"""
    _, environment, state, reward = single_task_setup(resources)
    record = make_record(
        state.definition,
        accepted=False,
        codes=("UNKNOWN_CODE",),
    )
    with pytest.raises(RuntimeError, match="未知违反代码"):
        reward.compute(environment, make_info([record]))


def test_12_reward_is_finite_and_hard_limit_is_enforced(resources):
    """合理随机特征均产生有限奖励，异常大值触发硬阈值。"""
    config = resources[0]["manual_reward"]
    rng = np.random.default_rng(2025)
    for _ in range(100):
        features = RewardFeatures(*rng.uniform(0.0, 1.0, size=8))
        breakdown = combine_manual_reward(
            features,
            config["weights"],
            config["numerical"],
        )
        assert math.isfinite(breakdown.total_reward)
    huge = RewardFeatures(100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="硬失败阈值"):
        combine_manual_reward(huge, config["weights"], config["numerical"])


def test_13_full_idle_episode_has_nonpositive_reward(resources):
    """固定小任务集空闲运行完整2880步，累计奖励只能为非正。"""
    config, _, _, splits = resources
    environment = make_environment(resources)
    observations, _ = environment.reset(task_ids=splits["validation"][:3])
    reward = ManualReward(config["manual_reward"])
    reward.reset(environment)
    total = 0.0
    while not environment.terminated:
        observations, _, _, _, info = environment.step({})
        total += reward.compute(environment, info).total_reward
    assert total <= 0.0
    environment.check_data_conservation()


def test_14_train_episode_sampling_is_reproducible_and_isolated(resources):
    """同seed的train任务一致，不同seed不同，且均属于train集合。"""
    environment = make_environment(resources)
    _, first = environment.reset(seed=2025, split="train", task_count=20)
    _, second = environment.reset(seed=2025, split="train", task_count=20)
    _, third = environment.reset(seed=2026, split="train", task_count=20)
    train_ids = set(resources[3]["train"])
    assert first["selected_task_ids"] == second["selected_task_ids"]
    assert first["selected_task_ids"] != third["selected_task_ids"]
    assert set(first["selected_task_ids"]) <= train_ids


def test_15_fixed_validation_is_reproducible(resources):
    """相同Actor与validation seed的短评估指标完全一致。"""
    config = resources[0]
    set_global_seed(config["seed"])
    environment = make_environment(resources)
    environment.reset(seed=3025, split="validation", task_count=10)
    encoder = MappoObservationEncoder(environment, config)
    actor = SharedActor(config).cpu()
    evaluator = MappoEvaluator(environment, encoder, actor, "cpu")
    first = evaluator.evaluate([3025], 10, max_steps=16)
    second = evaluator.evaluate([3025], 10, max_steps=16)
    assert first == second


def test_16_test_split_is_rejected_by_training_configuration(resources):
    """训练和验证配置均不能访问最终test划分。"""
    for location in ("training", "validation"):
        config = deepcopy(resources[0])
        if location == "training":
            config["manual_training"]["split"] = "test"
        else:
            config["manual_training"]["validation"]["split"] = "test"
        with pytest.raises(ValueError):
            validate_mappo_config(config)


def test_17_best_model_rule_uses_primary_then_secondary():
    """Best规则覆盖空best、主指标、容差内次指标和NaN拒绝。"""
    high = {
        "timeliness_raw_mean": 10.0,
        "load_balance_mean_per_task_mean": 0.8,
    }
    low = {
        "timeliness_raw_mean": 9.0,
        "load_balance_mean_per_task_mean": 0.9,
    }
    close_better = {
        "timeliness_raw_mean": 10.0 + 1.0e-7,
        "load_balance_mean_per_task_mean": 0.9,
    }
    assert is_better_validation_result(high, None, 1.0e-6)
    assert is_better_validation_result(high, low, 1.0e-6)
    assert not is_better_validation_result(low, high, 1.0e-6)
    assert is_better_validation_result(close_better, high, 1.0e-6)
    broken = dict(high, timeliness_raw_mean=float("nan"))
    with pytest.raises(ValueError, match="NaN"):
        is_better_validation_result(broken, high, 1.0e-6)


def _formal_components(resources, config=None, custom_database=None, custom_splits=None):
    """构造正式训练测试使用的环境、Trainer和Evaluator。"""
    base_config, dataset, database, splits = resources
    config = deepcopy(config or base_config)
    database = custom_database or database
    splits = custom_splits or splits
    set_global_seed(config["seed"])
    environment = CrossDomainSatelliteRangeSchedulingEnv(
        dataset,
        load_environment_config(),
        task_database=database,
        task_splits=splits,
    )
    encoder = MappoObservationEncoder(environment, config)
    actor = SharedActor(config)
    critic = CentralizedCritic(config)
    trainer = MappoTrainer(
        environment,
        encoder,
        actor,
        critic,
        config,
        reward_model=ManualReward(config["manual_reward"]),
        auto_reset_on_done=False,
    )
    evaluator = MappoEvaluator(environment, encoder, actor, trainer.device)
    return environment, encoder, actor, critic, trainer, evaluator


def test_18_terminal_rollout_stops_without_auto_reset(resources):
    """Episode终点Buffer提前结束、bootstrap为0且Trainer不自动重置。"""
    config = deepcopy(resources[0])
    config["device"] = "cpu"
    environment, _, _, _, trainer, _ = _formal_components(resources, config)
    trainer.reset_episode(2025, "train", 5)
    environment.step_index = environment.dataset.step_count - 1
    environment.current_time_s = environment.step_index * environment.step_seconds
    environment.terminated = False
    environment._update_task_statuses()
    trainer.set_observations(environment._observations())
    trainer.reward_model.reset(environment)
    buffer = trainer.collect_rollout(64)
    assert buffer.size == 1
    assert buffer.bootstrap_value == 0.0
    assert trainer.last_rollout_statistics["episode_done"]
    assert environment.terminated


def test_19_formal_checkpoint_resume_restores_state_and_rng(resources, tmp_path):
    """正式Checkpoint恢复网络、优化器、训练索引和Trainer RNG。"""
    config = deepcopy(resources[0])
    config["device"] = "cpu"
    _, encoder, actor, critic, trainer, _ = _formal_components(resources, config)
    training_state = {
        "episode_index": 0,
        "next_episode_index": 1,
        "update_index": 3,
        "environment_steps": 64,
        "best_validation_result": None,
        "best_episode_index": None,
    }
    path = tmp_path / "last.pt"
    rng_state = capture_rng_state(trainer.rng)
    save_checkpoint(
        path,
        actor,
        critic,
        trainer.actor_optimizer,
        trainer.critic_optimizer,
        config,
        3,
        encoder.metadata(),
        training_state=training_state,
        rng_state=rng_state,
    )
    set_global_seed(config["seed"])
    restored_actor = SharedActor(config)
    restored_critic = CentralizedCritic(config)
    restored_actor_optimizer = torch.optim.Adam(restored_actor.parameters())
    restored_critic_optimizer = torch.optim.Adam(restored_critic.parameters())
    checkpoint = load_checkpoint(
        path,
        restored_actor,
        restored_critic,
        restored_actor_optimizer,
        restored_critic_optimizer,
        encoder.metadata(),
    )
    if torch.cuda.is_available():
        # 模拟以CUDA map_location加载后，RNG状态张量被迁移到GPU的情况。
        checkpoint["rng_state"]["torch_cuda_rng_states"] = [
            state.cuda()
            for state in checkpoint["rng_state"]["torch_cuda_rng_states"]
        ]
    restored_rng = np.random.default_rng(1)
    restore_rng_state(checkpoint["rng_state"], restored_rng)
    assert torch.equal(parameter_vector(actor), parameter_vector(restored_actor))
    assert torch.equal(parameter_vector(critic), parameter_vector(restored_critic))
    assert checkpoint["training_state"]["next_episode_index"] == 1
    expected_rng = np.random.default_rng()
    expected_rng.bit_generator.state = rng_state["trainer_generator_state"]
    assert restored_rng.random() == expected_rng.random()


def test_20_short_formal_training_updates_and_logs_components(resources, tmp_path):
    """到达时间为0的小场景完成真实更新、奖励日志和Last保存。"""
    config, _, database, splits = resources
    config = deepcopy(config)
    config["device"] = "cpu"
    config["algorithm"]["actor_epochs"] = 1
    config["algorithm"]["critic_epochs"] = 1
    config["algorithm"]["actor_minibatch_size"] = 60
    config["algorithm"]["critic_minibatch_size"] = 4
    selected_ids = splits["train"][:5]
    custom_database = {}
    for task_id in selected_ids:
        task = database[task_id]
        custom_database[task_id] = replace(
            task,
            arrival_time_s=0.0,
            expiration_time_s=task.survival_time_s,
        )
    custom_splits = {
        "train": selected_ids,
        "validation": selected_ids,
        "test": [],
    }
    training = config["manual_training"]
    training["task_count"] = 5
    training["validation"]["task_count"] = 5
    training["checkpoint"]["best_path"] = str(tmp_path / "best.pt")
    training["checkpoint"]["last_path"] = str(tmp_path / "last.pt")
    training["logging"]["update_log_path"] = str(tmp_path / "updates.jsonl")
    training["logging"]["episode_log_path"] = str(tmp_path / "episodes.jsonl")
    training["logging"]["validation_log_path"] = str(tmp_path / "validation.jsonl")
    training["logging"]["summary_path"] = str(tmp_path / "summary.json")
    environment, encoder, actor, critic, trainer, evaluator = _formal_components(
        resources,
        config,
        custom_database,
        custom_splits,
    )
    actor_before = parameter_vector(actor)
    critic_before = parameter_vector(critic)
    summary = BaselineTrainingRunner(
        trainer,
        evaluator,
        config,
        encoder,
    ).run(
        target_episode_count=1,
        skip_validation=True,
        max_steps_per_episode=4,
    )
    assert summary["data_conservation_passed"]
    assert not torch.equal(actor_before, parameter_vector(actor))
    assert not torch.equal(critic_before, parameter_vector(critic))
    update_log = (tmp_path / "updates.jsonl").read_text(encoding="utf-8")
    assert "reward_component_sums" in update_log
    assert trainer.last_rollout_statistics["invalid_masked_action_count"] == 0
    assert (tmp_path / "last.pt").exists()
    environment.check_data_conservation()
