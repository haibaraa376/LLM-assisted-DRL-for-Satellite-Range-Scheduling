"""验证第三天固定编码、共享Actor、集中式Critic和基础MAPPO链路。"""

from copy import deepcopy

import numpy as np
import pytest
import torch

from mappo.buffer import RolloutBuffer
from mappo.checkpoint import load_checkpoint, save_checkpoint
from mappo.config import load_mappo_config
from mappo.distributions import SquashedNormal01
from mappo.encoding import (
    EncodedAgentObservation,
    MappoObservationEncoder,
    decode_composite_action,
)
from mappo.networks import CentralizedCritic, SharedActor
from mappo.trainer import MappoTrainer, parameter_vector
from mappo.utils import set_global_seed
from srs_env.config import load_environment_config
from srs_env.data import load_skyfield_dataset
from srs_env.environment import CrossDomainSatelliteRangeSchedulingEnv
from srs_env.models import SatelliteCompositeAction, TaskStatus
from srs_env.tasks import load_task_database, load_task_splits, sample_episode_tasks


@pytest.fixture(scope="module")
def shared_inputs():
    """只加载一次冻结数据；每个测试仍创建独立环境状态。"""
    config = load_mappo_config()
    dataset = load_skyfield_dataset()
    database = load_task_database()
    tasks = sample_episode_tasks(
        database,
        load_task_splits()["validation"],
        20,
        config["seed"],
    )
    return config, dataset, tasks


@pytest.fixture
def setup(shared_inputs):
    """创建重置后的真实20任务环境及其编码器。"""
    config, dataset, tasks = shared_inputs
    environment = CrossDomainSatelliteRangeSchedulingEnv(
        dataset,
        load_environment_config(),
        tasks,
    )
    observations, _ = environment.reset(seed=config["seed"])
    encoder = MappoObservationEncoder(environment, config)
    return config, environment, encoder, observations


def advance_until_candidate(environment, observations):
    """用空动作推进到首个任务可被某颗卫星编码。"""
    while not any(item["candidate_tasks"] for item in observations.values()):
        observations, _, terminated, _, _ = environment.step({})
        if terminated:
            raise RuntimeError("测试episode中没有候选任务")
    return observations


def test_01_encoding_dimensions_and_finite_values(setup):
    """Actor固定213维、Critic固定3200维，所有值为float32有限数。"""
    _, environment, encoder, observations = setup
    encoded = encoder.encode_all_agents(observations)
    assert len(encoded) == 15
    assert all(item.observation.shape == (213,) for item in encoded.values())
    assert all(item.observation.dtype == np.float32 for item in encoded.values())
    assert all(np.all(np.isfinite(item.observation)) for item in encoded.values())
    critic_state = encoder.encode_critic_state(
        encoded,
        environment.get_global_state(),
    )
    assert critic_state.shape == (3200,)
    assert critic_state.dtype == np.float32


def test_02_task_sorting_top_four_and_zero_padding(setup):
    """候选任务按到期、优先级、持有量、ID排序，空槽全零。"""
    _, environment, encoder, observations = setup
    satellite_id = "os01"
    items = []
    for index, state in enumerate(list(environment.tasks.values())[:5]):
        state.status = TaskStatus.ACTIVE
        held = float(100 + index)
        items.append(
            {
                "task_id": state.definition.task_id,
                "priority": state.definition.priority,
                "held_data_mbit": held,
                "expiration_time_s": state.definition.expiration_time_s,
            }
        )
    structured = dict(observations[satellite_id])
    structured["candidate_tasks"] = list(reversed(items))
    encoded = encoder.encode_agent(satellite_id, structured)
    expected = sorted(
        items,
        key=lambda item: (
            item["expiration_time_s"],
            -item["priority"],
            -item["held_data_mbit"],
            item["task_id"],
        ),
    )[:4]
    assert encoded.task_ids == tuple(item["task_id"] for item in expected)

    empty = encoder.encode_agent("cs01", observations["cs01"])
    assert empty.task_ids == (None, None, None, None)
    assert np.all(empty.observation[10 : 10 + 32] == 0.0)
    assert np.all(empty.base_target_mask[:, 0])
    assert not np.any(empty.base_target_mask[:, 1:])


def test_03_target_order_is_stable(setup):
    """19个目标始终为数据集15星顺序后接4个地面站。"""
    _, environment, encoder, observations = setup
    first = encoder.encode_agent("os01", observations["os01"]).target_ids
    observations, _ = environment.reset(seed=999)
    second = encoder.encode_agent("os01", observations["os01"]).target_ids
    expected = tuple(environment.dataset.satellite_ids) + tuple(
        environment.dataset.ground_station_ids
    )
    assert first == expected
    assert second == expected


def test_04_base_mask_matches_local_environment_candidates(setup):
    """基础Mask保留IDLE，屏蔽自身、错误地面站、无窗口与空槽。"""
    _, environment, encoder, observations = setup
    observations = advance_until_candidate(environment, observations)
    satellite_id = next(
        item for item in encoder.satellite_ids if observations[item]["candidate_tasks"]
    )
    encoded = encoder.encode_agent(satellite_id, observations[satellite_id])
    candidates = set(environment.get_action_candidates(satellite_id))
    task_id = encoded.task_ids[0]
    assert encoded.base_target_mask[0, 0]
    own_choice = encoder.target_ids.index(satellite_id) + 1
    assert not encoded.base_target_mask[0, own_choice]
    for choice, target_id in enumerate(encoded.target_ids, start=1):
        expected = (task_id, target_id) in candidates
        if target_id in environment.dataset.ground_station_index:
            expected &= (
                target_id
                == environment.tasks[task_id].definition.target_ground_station_id
            )
        assert encoded.base_target_mask[0, choice] == expected
    for slot in range(1, 4):
        if encoded.task_ids[slot] is None:
            assert encoded.base_target_mask[slot].sum() == 1


def test_05_autoregressive_mask_limits_counts_and_duplicates(shared_inputs):
    """动态Mask最多选择3颗不同卫星和1个地面站，并始终保留IDLE。"""
    config = shared_inputs[0]
    set_global_seed(config["seed"])
    actor = SharedActor(config)
    with torch.no_grad():
        actor.target_head.weight.zero_()
        bias = actor.target_head.bias.view(4, 20)
        bias.zero_()
        bias[0, 1] = 10.0
        bias[1, 2] = 10.0
        bias[2, 3] = 10.0
        bias[3, 4] = 20.0
        bias[3, 16] = 10.0
    observations = torch.zeros(2, 213)
    masks = torch.ones(2, 4, 20, dtype=torch.bool)
    actions = actor.sample_actions(observations, masks, deterministic=True)
    assert actions.target_choices[0].tolist() == [1, 2, 3, 16]
    for row in actions.target_choices.tolist():
        satellites = [choice for choice in row if 1 <= choice <= 15]
        grounds = [choice for choice in row if choice >= 16]
        assert len(satellites) <= 3
        assert len(satellites) == len(set(satellites))
        assert len(grounds) <= 1


def test_06_action_decoding_preserves_slots_and_bounds():
    """IDLE槽不生成子动作，其余槽保持任务、目标及连续动作。"""
    observation = np.zeros(213, dtype=np.float32)
    mask = np.ones((4, 20), dtype=bool)
    encoded = EncodedAgentObservation(
        observation,
        mask,
        ("a", "b", "c", "d"),
        tuple("target_{0}".format(index) for index in range(19)),
    )
    bounded = np.array(
        [[0.0, 0.0], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7]],
        dtype=np.float32,
    )
    action = decode_composite_action(encoded, np.array([0, 2, 0, 4]), bounded)
    assert isinstance(action, SatelliteCompositeAction)
    assert [item.task_id for item in action.transmissions] == ["b", "d"]
    assert [item.target_id for item in action.transmissions] == ["target_1", "target_3"]
    assert all(0.0 <= item.transmission_ratio <= 1.0 for item in action.transmissions)


def test_07_squashed_normal_probability_is_finite_and_exact():
    """Tanh有界高斯动作在(0,1)内且包含有限Jacobian密度。"""
    mean = torch.zeros(128, 4, 2)
    log_std = torch.full_like(mean, -0.5)
    distribution = SquashedNormal01(mean, log_std)
    raw, bounded = distribution.sample()
    assert torch.all((bounded > 0.0) & (bounded < 1.0))
    assert torch.isfinite(raw).all()
    assert torch.isfinite(distribution.log_prob(raw)).all()
    expected = (torch.tanh(raw) + 1.0) * 0.5
    assert torch.equal(bounded, expected)


def test_08_masked_categorical_never_samples_invalid_high_logit(shared_inputs):
    """被Mask目标即使logit极高也不能被采样。"""
    set_global_seed(shared_inputs[0]["seed"])
    actor = SharedActor(shared_inputs[0])
    with torch.no_grad():
        actor.target_head.weight.zero_()
        actor.target_head.bias.zero_()
        actor.target_head.bias.view(4, 20)[:, 5] = 1.0e6
    masks = torch.zeros(128, 4, 20, dtype=torch.bool)
    masks[:, :, 0] = True
    actions = actor.sample_actions(torch.zeros(128, 213), masks)
    assert torch.all(actions.target_choices == 0)


def test_09_actor_batch_shapes_and_log_probability_roundtrip(shared_inputs):
    """共享Actor批量形状正确，sample与evaluate的log probability一致。"""
    set_global_seed(shared_inputs[0]["seed"])
    actor = SharedActor(shared_inputs[0])
    observations = torch.randn(15, 213)
    masks = torch.ones(15, 4, 20, dtype=torch.bool)
    sampled = actor.sample_actions(observations, masks)
    evaluated, entropy = actor.evaluate_actions(
        observations,
        masks,
        sampled.target_choices,
        sampled.raw_continuous_actions,
    )
    assert sampled.target_choices.shape == (15, 4)
    assert sampled.raw_continuous_actions.shape == (15, 4, 2)
    assert sampled.bounded_continuous_actions.shape == (15, 4, 2)
    assert sampled.log_prob.shape == (15,)
    assert sampled.entropy.shape == (15,)
    assert torch.allclose(sampled.log_prob, evaluated, atol=1.0e-5)
    assert torch.isfinite(entropy).all()


def test_10_centralized_critic_output_shape(shared_inputs):
    """Critic对7个全局状态仅输出7个共享价值。"""
    set_global_seed(shared_inputs[0]["seed"])
    critic = CentralizedCritic(shared_inputs[0])
    values = critic(torch.randn(7, 3200))
    assert values.shape == (7,)
    assert torch.isfinite(values).all()


def test_11_gae_matches_manual_terminal_example():
    """手工序列验证GAE递推，终止状态不会使用bootstrap。"""
    buffer = RolloutBuffer(2)
    zeros = np.zeros((15, 213), dtype=np.float32)
    critic = np.zeros(3200, dtype=np.float32)
    masks = np.zeros((15, 4, 20), dtype=bool)
    masks[:, :, 0] = True
    choices = np.zeros((15, 4), dtype=np.int64)
    raw = np.zeros((15, 4, 2), dtype=np.float32)
    log_probs = np.zeros(15, dtype=np.float32)
    buffer.add(zeros, critic, masks, choices, raw, log_probs, 1.0, False, 0.5)
    buffer.add(zeros, critic, masks, choices, raw, log_probs, 2.0, True, 0.25)
    buffer.compute_gae(bootstrap_value=100.0, gamma=1.0, gae_lambda=1.0)
    assert np.allclose(buffer.advantages, [2.5, 1.75])
    assert np.allclose(buffer.returns, [3.0, 2.0])


def test_12_rollout_buffer_shapes_and_dtypes():
    """Buffer保存Actor T×N数据与Critic T数据，类型符合接口。"""
    buffer = RolloutBuffer(3)
    for _ in range(2):
        masks = np.zeros((15, 4, 20), dtype=bool)
        masks[:, :, 0] = True
        buffer.add(
            np.zeros((15, 213), dtype=np.float32),
            np.zeros(3200, dtype=np.float32),
            masks,
            np.zeros((15, 4), dtype=np.int64),
            np.zeros((15, 4, 2), dtype=np.float32),
            np.zeros(15, dtype=np.float32),
            0.0,
            False,
            0.0,
        )
    buffer.compute_gae(0.0, 0.99, 0.95)
    assert buffer.observations.shape == (3, 15, 213)
    assert buffer.critic_states.shape == (3, 3200)
    assert buffer.base_target_masks.dtype == bool
    assert buffer.target_actions.dtype == np.int64
    actor_batch = next(buffer.iter_actor_minibatches(30, shuffle=False))
    critic_batch = next(buffer.iter_critic_minibatches(2, shuffle=False))
    assert actor_batch["observations"].shape[0] == 30
    assert critic_batch["critic_states"].shape[0] == 2


def test_13_ppo_update_is_finite_and_changes_both_networks(setup):
    """小型真实Rollout执行一次更新后，两套网络参数均变化且统计有限。"""
    config, environment, encoder, _ = setup
    small_config = deepcopy(config)
    small_config["algorithm"]["actor_epochs"] = 1
    small_config["algorithm"]["critic_epochs"] = 1
    small_config["algorithm"]["actor_minibatch_size"] = 60
    small_config["algorithm"]["critic_minibatch_size"] = 4
    set_global_seed(small_config["seed"])
    actor = SharedActor(small_config)
    critic = CentralizedCritic(small_config)
    trainer = MappoTrainer(environment, encoder, actor, critic, small_config)
    trainer.observations = advance_until_candidate(
        environment,
        trainer.observations,
    )
    trainer.reward_model.reset({"timeliness_raw": environment.timeliness_raw})
    actor_before = parameter_vector(actor)
    critic_before = parameter_vector(critic)
    statistics = trainer.update(trainer.collect_rollout(4))
    assert all(np.isfinite(value) for value in statistics.__dict__.values())
    assert not torch.equal(actor_before, parameter_vector(actor))
    assert not torch.equal(critic_before, parameter_vector(critic))


def test_14_checkpoint_roundtrip_preserves_metadata_and_actions(setup, tmp_path):
    """Checkpoint往返后参数、顺序元数据和确定性动作完全一致。"""
    config, _, encoder, _ = setup
    set_global_seed(config["seed"])
    actor = SharedActor(config)
    critic = CentralizedCritic(config)
    actor_optimizer = torch.optim.Adam(actor.parameters())
    critic_optimizer = torch.optim.Adam(critic.parameters())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        config,
        1,
        encoder.metadata(),
    )
    set_global_seed(config["seed"])
    restored_actor = SharedActor(config)
    restored_critic = CentralizedCritic(config)
    load_checkpoint(
        path,
        restored_actor,
        restored_critic,
        expected_encoder_metadata=encoder.metadata(),
    )
    assert torch.equal(parameter_vector(actor), parameter_vector(restored_actor))
    observations = torch.zeros(15, 213)
    masks = torch.zeros(15, 4, 20, dtype=torch.bool)
    masks[:, :, 0] = True
    original = actor.sample_actions(observations, masks, deterministic=True)
    restored = restored_actor.sample_actions(observations, masks, deterministic=True)
    assert torch.equal(original.target_choices, restored.target_choices)
    assert torch.equal(
        original.bounded_continuous_actions,
        restored.bounded_continuous_actions,
    )


def test_15_real_environment_sixteen_step_integration(setup):
    """真实环境运行16步，动作均可解码、奖励有限、Mask合法且数据守恒。"""
    config, environment, encoder, _ = setup
    integration_config = deepcopy(config)
    integration_config["device"] = "cpu"
    set_global_seed(integration_config["seed"])
    trainer = MappoTrainer(
        environment,
        encoder,
        SharedActor(integration_config),
        CentralizedCritic(integration_config),
        integration_config,
    )
    buffer = trainer.collect_rollout(16)
    assert buffer.size == 16
    assert np.all(np.isfinite(buffer.shared_rewards[: buffer.size]))
    assert trainer.last_rollout_statistics["invalid_masked_action_count"] == 0
    environment._check_data_conservation()


def test_16_same_seed_produces_identical_actor_parameters(shared_inputs):
    """相同全局种子必须让两套Actor的初始参数逐项完全相同。"""
    config = shared_inputs[0]
    with pytest.raises(ValueError, match="非负整数"):
        set_global_seed(-1)
    with pytest.raises(ValueError, match="非负整数"):
        set_global_seed(2025.0)
    set_global_seed(2025)
    first = SharedActor(config)
    set_global_seed(2025)
    second = SharedActor(config)
    assert all(
        torch.equal(first_parameter, second_parameter)
        for first_parameter, second_parameter in zip(
            first.parameters(),
            second.parameters(),
        )
    )


def test_17_same_seed_produces_identical_critic_parameters(shared_inputs):
    """相同全局种子必须让两套Critic的初始参数逐项完全相同。"""
    config = shared_inputs[0]
    set_global_seed(2025)
    first = CentralizedCritic(config)
    set_global_seed(2025)
    second = CentralizedCritic(config)
    assert all(
        torch.equal(first_parameter, second_parameter)
        for first_parameter, second_parameter in zip(
            first.parameters(),
            second.parameters(),
        )
    )


def test_18_different_seeds_change_actor_parameters(shared_inputs):
    """不同种子至少改变Actor中的一个参数张量。"""
    config = shared_inputs[0]
    set_global_seed(2025)
    first = SharedActor(config)
    set_global_seed(2026)
    second = SharedActor(config)
    assert any(
        not torch.equal(first_parameter, second_parameter)
        for first_parameter, second_parameter in zip(
            first.parameters(),
            second.parameters(),
        )
    )


def test_19_same_seed_actors_produce_identical_deterministic_actions(shared_inputs):
    """相同初始化的Actor对同一观测和Mask产生完全一致的确定性动作。"""
    config = shared_inputs[0]
    set_global_seed(2025)
    first = SharedActor(config)
    set_global_seed(2025)
    second = SharedActor(config)
    observations = torch.linspace(-1.0, 1.0, 15 * 213).reshape(15, 213)
    masks = torch.ones(15, 4, 20, dtype=torch.bool)
    first_actions = first.sample_actions(observations, masks, deterministic=True)
    second_actions = second.sample_actions(observations, masks, deterministic=True)
    assert torch.equal(first_actions.target_choices, second_actions.target_choices)
    assert torch.equal(
        first_actions.bounded_continuous_actions,
        second_actions.bounded_continuous_actions,
    )


def test_20_cpu_short_training_is_reproducible(shared_inputs):
    """同种子CPU短训练的loss、KL、熵和参数变化范数可重复。"""
    base_config, dataset, tasks = shared_inputs

    def run_once():
        config = deepcopy(base_config)
        config["device"] = "cpu"
        config["algorithm"]["actor_epochs"] = 1
        config["algorithm"]["critic_epochs"] = 1
        config["algorithm"]["actor_minibatch_size"] = 60
        config["algorithm"]["critic_minibatch_size"] = 4
        set_global_seed(config["seed"])
        environment = CrossDomainSatelliteRangeSchedulingEnv(
            dataset,
            load_environment_config(),
            tasks,
        )
        environment.reset(seed=config["seed"])
        encoder = MappoObservationEncoder(environment, config)
        actor = SharedActor(config)
        critic = CentralizedCritic(config)
        trainer = MappoTrainer(environment, encoder, actor, critic, config)
        trainer.observations = advance_until_candidate(
            environment,
            trainer.observations,
        )
        trainer.reward_model.reset({"timeliness_raw": environment.timeliness_raw})
        actor_before = parameter_vector(actor)
        critic_before = parameter_vector(critic)
        statistics = trainer.update(trainer.collect_rollout(4))
        return np.array(
            [
                statistics.actor_loss,
                statistics.critic_loss,
                statistics.entropy,
                statistics.approximate_kl,
                float(
                    torch.linalg.vector_norm(
                        parameter_vector(actor) - actor_before
                    )
                ),
                float(
                    torch.linalg.vector_norm(
                        parameter_vector(critic) - critic_before
                    )
                ),
            ]
        )

    first = run_once()
    second = run_once()
    assert np.all(np.isfinite(first))
    assert np.allclose(first, second, rtol=1.0e-7, atol=1.0e-9)
